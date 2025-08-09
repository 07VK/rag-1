import os
import gradio as gr
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        # Check for GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # State-of-the-art embedding model with GPU support
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': device}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            cache_folder="/tmp/hf_cache"
        )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in Colab Secrets.")
        genai.configure(api_key=api_key)
        self.generative_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            google_api_key=api_key
        )
        self.vector_store = None
        self.bm25 = None
        self.documents = None
        self.retrieval_chain = None
        print("RAG Pipeline Initialized.")

    def create_vector_store_from_pdf(self, pdf_file):
        if not pdf_file:
            return "Please upload a PDF file first."

        print(f"Processing PDF: {pdf_file.name}")
        try:
            # Extract text with PyMuPDF, preserving structure
            doc = fitz.open(pdf_file.name)
            text = ""
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()  # Text is in index 4
                    if block_text:
                        text += f"[Page {page_num + 1}]\n{block_text}\n\n"
            doc.close()

            # Split text with RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            self.documents = [Document(page_content=chunk, metadata={"page": chunk.split('\n')[0]}) for chunk in chunks]
            print(f"Split document into {len(self.documents)} chunks.")

            # Create FAISS vector store with GPU support
            print("Creating vector store with FAISS...")
            self.vector_store = FAISS.from_documents(self.documents, self.embedding_model)

            # Create BM25 index for keyword search (CPU-based)
            tokenized_docs = [doc.page_content.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print("BM25 index created successfully.")

            self.setup_retrieval_chain()
            return f"PDF '{pdf_file.name}' processed successfully. You can now ask questions."
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            return "An error occurred. Check the console for details."

    def setup_retrieval_chain(self):
        prompt_template = """
        You are a medical assistant answering questions based on a provided patient document. Use ONLY the information in the context to provide accurate, detailed answers.
        - Synthesize information from all relevant sections (e.g., history, diagnostics, procedures, labs, medications) to ensure completeness.
        - Prioritize specific details like numerical values (e.g., lab results, ejection fraction), dates, and procedural outcomes.
        - For risk-related questions, infer likelihood based on medical history, labs, and clinical findings if explicit data is unavailable.
        - If the answer is not in the context, state: "The answer is not available in the provided document."
        - Format answers clearly, using bullet points for clarity when listing multiple points, and provide enough verbose, detailed, and well-structured responses that fully address the question.
        - Structure answers with clear sections or bullet points to enhance readability, avoiding vague or repetitive language also EXPLAIN THE MEDICAL TERMS TO NORMAL READABLE LANGUAGE.
        - Provide a brief explanation of medical terms or procedures to ensure clarity for non-expert readers.
        - If the answer is not available in the context, clearly state: "The answer is not available in the provided document."
        <context>
        {context}
        </context>

        Question: {input}
        Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        question_answer_chain = create_stuff_documents_chain(self.generative_model, prompt)
        self.retrieval_chain = create_retrieval_chain(
            self.vector_store.as_retriever(search_kwargs={"k": 10}),
            question_answer_chain
        )
        print("Retrieval chain is set up.")

    def answer_question(self, question: str):
        if not self.retrieval_chain or not self.vector_store or not self.bm25:
            return "Error: Please process a PDF first."
        if not question:
            return "Error: Please provide a question."

        print(f"Received question: {question}")
        try:
            # Hybrid search: Combine FAISS and BM25
            # FAISS vector search
            faiss_results = self.vector_store.similarity_search_with_score(question, k=10)
            faiss_docs = [doc for doc, _ in faiss_results]
            faiss_scores = {doc.page_content: score for doc, score in faiss_results}

            # BM25 keyword search
            tokenized_query = question.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_docs = sorted(
                [(doc, score) for doc, score in zip(self.documents, bm25_scores) if score > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Combine results using page_content for deduplication
            content_to_doc = {doc.page_content: doc for doc in self.documents}
            combined_contents = list(set(
                [doc.page_content for doc, _ in faiss_results] +
                [doc.page_content for doc, _ in bm25_top_docs]
            ))
            combined_docs = [content_to_doc[content] for content in combined_contents]

            # Combine scores (weighted: 0.7 vector, 0.3 keyword)
            combined_scores = {}
            for doc in combined_docs:
                faiss_score = faiss_scores.get(doc.page_content, 0)
                bm25_score = next((score for d, score in bm25_top_docs if d.page_content == doc.page_content), 0)
                combined_scores[doc.page_content] = 0.7 * (1 - faiss_score) + 0.3 * bm25_score

            # Select top 5 documents based on combined scores
            top_docs = sorted(
                combined_docs,
                key=lambda doc: combined_scores[doc.page_content],
                reverse=True
            )[:5]

            # Prepare context for LLM
            context = "\n".join([doc.page_content for doc in top_docs])
            response = self.retrieval_chain.invoke({"input": question, "context": context})
            return response['answer']
        except Exception as e:
            print(f"Error answering question: {e}")
            return "An error occurred. Check the console for details."

# Gradio UI
rag_pipeline = RAGPipeline()

def process_pdf_interface(file_obj):
    if file_obj is None:
        return "Please upload a file.", None, gr.update(visible=False)
    result = rag_pipeline.create_vector_store_from_pdf(file_obj)
    chat_visible = "successfully" in result.lower()
    return result, None, gr.update(visible=chat_visible)

def chat_interface(user_input, history):
    history = history or []
    history.append([user_input, None])
    answer = rag_pipeline.answer_question(user_input)
    history[-1][1] = answer
    return history, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        1. Upload a PDF document.
        2. Click "Process PDF".
        3. Ask questions in the chat box below.
        """
    )
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_button = gr.Button("Process PDF", variant="primary")
    status_display = gr.Textbox(label="Status", interactive=False)
    with gr.Column(visible=False) as chat_area:
        chatbot = gr.Chatbot(label="Ask Questions About Your Document")
        question_box = gr.Textbox(label="Your Question", placeholder="Type your question here...")
        clear_button = gr.Button("Clear Chat")

    process_button.click(fn=process_pdf_interface, inputs=[pdf_upload], outputs=[status_display, chatbot, chat_area])
    question_box.submit(fn=chat_interface, inputs=[question_box, chatbot], outputs=[chatbot, question_box])
    clear_button.click(lambda: (None, ""), outputs=[chatbot, question_box])

demo.launch(server_name="0.0.0.0", server_port=7860)
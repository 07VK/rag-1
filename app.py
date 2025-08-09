# app.py
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import nest_asyncio

# --- LangChain & AI Model Imports ---
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
    """
    Encapsulates the entire RAG process from document ingestion to question answering.
    """
    def __init__(self):
        print("Initializing RAG Pipeline...")
        # Automatically select the device (GPU if available, otherwise CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Use a lightweight, fast sentence-transformer model for embeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': device}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            cache_folder="/tmp/hf_cache"  # Cache models for faster startups
        )

        # Configure the Gemini API key from environment variables
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")
        genai.configure(api_key=api_key)

        # Initialize the generative model (Gemini Flash)
        self.generative_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            google_api_key=api_key
        )

        # Initialize state variables
        self.vector_store = None
        self.bm25 = None
        self.documents = None
        self.retrieval_chain = None
        print("RAG Pipeline Initialized Successfully.")

    def create_vector_store_from_pdf(self, pdf_bytes: bytes, filename: str):
        """
        Processes a PDF file from bytes, creates embeddings, and sets up the retrieval chain.
        """
        print(f"Processing PDF: {filename}")
        try:
            # Open PDF from memory
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            # Extract text from each page, preserving some structure
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()  # Text is in index 4 of the block tuple
                    if block_text:
                        text += f"[Page {page_num + 1}]\n{block_text}\n\n"
            doc.close()

            # Split the extracted text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            self.documents = [Document(page_content=chunk, metadata={"page": chunk.split('\n')[0]}) for chunk in chunks]
            print(f"Split document into {len(self.documents)} chunks.")

            # Create FAISS vector store for semantic search
            print("Creating FAISS vector store...")
            self.vector_store = FAISS.from_documents(self.documents, self.embedding_model)

            # Create BM25 index for keyword search
            tokenized_docs = [doc.page_content.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print("BM25 index created successfully.")

            # Set up the final retrieval chain
            self.setup_retrieval_chain()
            return f"PDF '{filename}' processed successfully. You can now ask questions."
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

    def setup_retrieval_chain(self):
        """
        Defines the prompt and creates the LangChain retrieval chain.
        """
        prompt_template = """
        You are an AI assistant answering questions based on a provided document. Use ONLY the information in the context to provide accurate, detailed answers.
        - Synthesize information from all relevant sections to ensure completeness.
        - Prioritize specific details like numerical values, dates, and procedural outcomes.
        - If the answer is not in the context, state: "The answer is not available in the provided document."
        - Format answers clearly, using bullet points for lists.
        - Explain complex terms in simple language for a non-expert.
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
        """
        Performs hybrid search and generates an answer for a given question.
        """
        if not all([self.retrieval_chain, self.vector_store, self.bm25]):
            raise HTTPException(status_code=400, detail="Error: Please process a PDF first.")
        if not question:
            raise HTTPException(status_code=400, detail="Error: Please provide a question.")

        print(f"Received question: {question}")
        try:
            # Perform hybrid search by combining FAISS (semantic) and BM25 (keyword) results
            faiss_results = self.vector_store.similarity_search_with_score(question, k=10)
            tokenized_query = question.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_docs_with_scores = sorted(
                [(doc, score) for doc, score in zip(self.documents, bm25_scores) if score > 0],
                key=lambda x: x[1], reverse=True
            )[:10]

            # Combine and deduplicate results
            combined_docs = {doc.page_content: doc for doc, _ in faiss_results}
            combined_docs.update({doc.page_content: doc for doc, _ in bm25_top_docs_with_scores})

            # Simple re-ranking can be added here if needed. For now, we use the combined context.
            top_docs = list(combined_docs.values())[:5] # Limit context to top 5 combined results

            context = "\n\n".join([doc.page_content for doc in top_docs])
            response = self.retrieval_chain.invoke({"input": question, "context": context})
            return response['answer']
        except Exception as e:
            print(f"Error answering question: {e}")
            raise HTTPException(status_code=500, detail=f"Error answering question: {e}")


# --- FastAPI App Definition ---
app = FastAPI()

# Instantiate the pipeline on startup
rag_pipeline = RAGPipeline()

# Mount directories for static files (CSS, JS) and templates (HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API Endpoints ---
@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """Handles PDF file upload and processing."""
    pdf_bytes = await file.read()
    try:
        result_message = rag_pipeline.create_vector_store_from_pdf(pdf_bytes, file.filename)
        return {"status": "success", "message": result_message}
    except HTTPException as e:
        return {"status": "error", "message": e.detail}

@app.post("/ask")
async def ask_question_endpoint(request_data: dict):
    """Handles user questions and returns the model's answer."""
    question = request_data.get("question")
    try:
        answer = rag_pipeline.answer_question(question)
        return {"answer": answer}
    except HTTPException as e:
        return {"answer": e.detail}

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serves the main index.html page."""
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# --- Main execution block for running with uvicorn ---
if __name__ == "__main__":
    # This allows running the script directly with `python app.py`
    # The Dockerfile uses a more direct `uvicorn` command.
    uvicorn.run(app, host="0.0.0.0", port=8000)

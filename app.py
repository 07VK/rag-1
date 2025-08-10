import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates # Correct way to serve templates
import nest_asyncio

# --- LangChain & AI Model Imports ---
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz # PyMuPDF
from rank_bm25 import BM25Okapi
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- RAG Pipeline Class (No Changes Needed) ---
class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': device}
        # Using a temporary cache folder suitable for Docker environments
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            cache_folder="/tmp/hf_cache" 
        )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")
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
        print("RAG Pipeline Initialized Successfully.")

    def create_vector_store_from_pdf(self, pdf_bytes: bytes, filename: str):
        print(f"Processing PDF: {filename}")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()
                    if block_text:
                        # Add page metadata directly into the text for context
                        text += f"[Page {page_num + 1}]\n{block_text}\n\n"
            doc.close()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            # Create LangChain Document objects
            self.documents = [Document(page_content=chunk) for chunk in chunks]
            
            print(f"Split document into {len(self.documents)} chunks.")
            
            print("Creating FAISS vector store...")
            self.vector_store = FAISS.from_documents(self.documents, self.embedding_model)
            
            tokenized_docs = [doc.page_content.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print("BM25 index created successfully.")
            
            self.setup_retrieval_chain()
            return f"PDF '{filename}' processed successfully. You can now ask questions."
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    def setup_retrieval_chain(self):
        prompt_template = """You are an AI assistant for ClearChartAI. Your task is to answer questions based on the medical document provided.
        Instructions:
        - Use ONLY the information present in the provided context to answer the question.
        - Synthesize information from all relevant sections to form a complete and coherent answer.
        - If the answer is not found within the context, you MUST state: "The answer is not available in the provided document." Do not make up information.
        - Format your answers clearly. Use bullet points for lists if appropriate.

        <context>
        {context}
        </context>

        Question: {input}
        Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        Youtube_chain = create_stuff_documents_chain(self.generative_model, prompt)
        
        # Using the vector store's retriever
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        self.retrieval_chain = create_retrieval_chain(retriever, Youtube_chain)
        print("Retrieval chain is set up.")

    def answer_question(self, question: str):
        if not self.retrieval_chain or not self.vector_store:
            raise HTTPException(status_code=400, detail="Error: A document has not been processed yet. Please upload a PDF first.")
        if not question:
            raise HTTPException(status_code=400, detail="Error: Please provide a question.")
            
        print(f"Received question: {question}")
        try:
            # The retrieval chain now handles fetching context and generating the answer
            response = self.retrieval_chain.invoke({"input": question})
            return response.get('answer', "Could not generate an answer.")
        except Exception as e:
            print(f"Error answering question: {e}")
            raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

# --- FastAPI App Setup ---
app = FastAPI()

# ADDED: Setup for Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Instantiate the pipeline on startup
rag_pipeline = RAGPipeline()

# --- API Endpoints ---
@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    pdf_bytes = await file.read()
    try:
        # Re-initialize the pipeline for the new document
        # This ensures each upload is a fresh session
        global rag_pipeline
        rag_pipeline = RAGPipeline()
        result_message = rag_pipeline.create_vector_store_from_pdf(pdf_bytes, file.filename)
        return {"status": "success", "message": result_message}
    except Exception as e:
        # Catch potential exceptions from pipeline creation as well
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question_endpoint(request_data: dict):
    question = request_data.get("question")
    try:
        answer = rag_pipeline.answer_question(question)
        return {"answer": answer}
    except HTTPException as e:
        # Re-raise the HTTP exception from the pipeline
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# MODIFIED: Serve the main "About" page
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ADDED: Serve the "Demo" page
@app.get("/demo", response_class=HTMLResponse)
async def get_demo_page(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

# Allow nest_asyncio for environments like Google Colab if needed
nest_asyncio.apply()

# --- Main execution block ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import fitz
import numpy as np
import google.generativeai as genai
from pinecone import Pinecone
import uuid
import time
from pydantic import BaseModel

# --- 1. Configuration and Initialization ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST]):
    raise ValueError("Missing one or more required environment variables")

genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# --- 2. FastAPI App Setup ---
app = FastAPI()

class AskRequest(BaseModel):
    question: str
    session_id: str

# --- 3. Core RAG Functions ---

def get_gemini_embeddings(texts, model="models/text-embedding-004"):
    """Generates embeddings for a list of texts using the Gemini API."""
    try:
        return genai.embed_content(model=model, content=texts)["embedding"]
    except Exception as e:
        print(f"Error embedding content: {e}. Retrying...")
        time.sleep(1)
        return genai.embed_content(model=model, content=texts)["embedding"]

def layout_aware_chunking(pdf_bytes: bytes):
    """Extracts text from PDF bytes respecting paragraphs and layout blocks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page in doc:
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            block_text = block[4].strip()
            if len(block_text) > 50:  # Filter out very short, likely irrelevant blocks
                chunks.append(block_text)
    doc.close()
    print(f"Document split into {len(chunks)} layout-aware chunks.")
    return chunks

# --- 4. API Endpoints ---

@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """
    Handles PDF upload, processing, and embedding.
    Returns a unique session_id for the chat.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    session_id = str(uuid.uuid4())
    pdf_bytes = await file.read()
    
    try:
        chunks = layout_aware_chunking(pdf_bytes)
        ttl_in_seconds = 3600  # 1 hr
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = [f"{session_id}-{i+j}" for j in range(len(batch_chunks))]
            
            embeddings = get_gemini_embeddings(batch_chunks)
            
            vectors_to_upsert = []
            for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                vectors_to_upsert.append({
                    "id": batch_ids[j],
                    "values": embedding,
                    "metadata": {"text": chunk_text}
                })
            
            index.upsert(vectors=vectors_to_upsert, ttl=ttl_in_seconds)
            print(f"Upserted batch {i//batch_size + 1} with a 1-hour TTL.")

        return {"status": "success", "message": "PDF processed successfully.", "session_id": session_id}

    except Exception as e:
        print(f"Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question_endpoint(request_data: AskRequest):
    """
    Receives a question and session_id, retrieves context, and generates an answer.
    """
    question = request_data.question
    session_id = request_data.session_id

    if not question or not session_id:
        raise HTTPException(status_code=400, detail="Question and session_id are required.")

    try:
        question_embedding = get_gemini_embeddings([question])[0]
        query_results = index.query(vector=question_embedding, top_k=4, include_metadata=True)
        
        context_chunks = [
            match['metadata']['text'] 
            for match in query_results['matches'] 
            if match['id'].startswith(session_id)
        ]
        
        if not context_chunks:
            return {"answer": "I could not find relevant information in the uploaded document for this session."}
            
        context = "\n---\n".join(context_chunks)

        prompt = f"""
        **Your Role and Instructions:**
        You are a helpful AI assistant designed to explain information from medical documents in a simple and clear way. Your tone should always be gentle and reassuring.

        **Core Rules:**
        1.  **Strictly Grounded:** Base your entire answer ONLY on the information found in the "Context" provided below. Do not add any outside information.
        2.  **No Hallucinations:** If the answer is not in the context, you MUST state: "I could not find information about that in this document."
        3.  **NO DIAGNOSIS OR ADVICE:** You MUST NOT provide any form of diagnosis, medical advice, or treatment suggestions. Your role is to explain the information present, not to interpret it for the user's personal health.

        **Response Formatting Rules:**
        1.  **Simple Language:** When you encounter a medical term, you must explain it in very simple terms, as if you were talking to a 5-year-old.
        2.  **Provide Context:** After explaining a term or result, you must add context by explaining what a "regular condition would be for an average person."
        3.  **Word Count:** Keep the total response length under 100 words.

        ---
        **Context from the Document:**
        {context}
        ---

        **Question:**
        {question}

        **Answer:**
        """

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return {"answer": response.text}

    except Exception as e:
        print(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- 5. Static Files and Catch-All Route ---

# This serves files like CSS, JS, and images from the 'static' directory,
# which is where your Dockerfile places the built React app.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{catch_all:path}", response_class=FileResponse)
async def serve_react_app(catch_all: str):
    """
    This catch-all route serves the React app's index.html file for any path
    that is not an API route. This allows React Router to handle the navigation.
    """
    return FileResponse("static/index.html")

# --- 6. Main execution block (for local testing) ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

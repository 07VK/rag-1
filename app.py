import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
import numpy as np
import google.generativeai as genai
from pinecone import Pinecone
import uuid
import time
from pydantic import BaseModel

# --- 1. Configuration and Initialization ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_HOST = os.environ.get("PINECONE_HOST")

    if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST]):
        raise ValueError("CRITICAL: Missing one or more required environment variables")

    print("INFO: All environment variables found. Initializing services...")
    
    genai.configure(api_key=GEMINI_API_KEY)
    print("INFO: Gemini configured successfully.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("INFO: Pinecone client initialized.")
    
    index = pc.Index(host=PINECONE_HOST)
    print("SUCCESS: Connected to Pinecone index.")

except Exception as e:
    print(f"FATAL: Application failed to start. Error: {e}")
    raise

# --- 2. FastAPI App Setup ---
app = FastAPI()

class ChatMessage(BaseModel):
    sender: str
    text: str

class AskRequest(BaseModel):
    question: str
    session_id: str
    history: list[ChatMessage]

# --- 3. Core RAG Functions ---

def get_gemini_embeddings(texts, model="models/text-embedding-004"):
    try:
        return genai.embed_content(model=model, content=texts)["embedding"]
    except Exception as e:
        print(f"Error embedding content: {e}. Retrying...")
        time.sleep(1)
        return genai.embed_content(model=model, content=texts)["embedding"]

def layout_aware_chunking(pdf_bytes: bytes, min_chunk_chars=250, max_chunk_chars=1000):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    current_chunk = ""
    for page in doc:
        blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
        for block in blocks:
            block_text = block[4].strip()
            if not block_text:
                continue
            if len(current_chunk) > min_chunk_chars and block_text.startswith((" ", "\n")):
                chunks.append(current_chunk)
                current_chunk = ""
            if len(current_chunk) + len(block_text) + 1 > max_chunk_chars:
                chunks.append(current_chunk)
                current_chunk = block_text
            else:
                current_chunk += " " + block_text
    if current_chunk:
        chunks.append(current_chunk)
    final_chunks = [chunk for chunk in chunks if len(chunk) > min_chunk_chars]
    print(f"Document split into {len(final_chunks)} semantically grouped chunks.")
    return final_chunks


# --- 4. API Endpoints ---

@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    session_id = str(uuid.uuid4())
    pdf_bytes = await file.read()
    try:
        chunks = layout_aware_chunking(pdf_bytes)
        if not chunks:
             raise HTTPException(status_code=400, detail="Could not extract any meaningful content from the PDF.")
        ttl_in_seconds = 3600
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
    question = request_data.question.strip().lower()
    session_id = request_data.session_id
    history = request_data.history

    if not question or not session_id:
        raise HTTPException(status_code=400, detail="Question and session_id are required.")

    greetings = ["hello", "hi", "hey", "yo", "greetings"]
    gratitude = ["thanks", "thank you", "thx", "appreciate it", "ok", "sounds good"]
    if question in greetings:
        return {"answer": "Hello! I'm ready to help. What would you like to know about this document?"}
    if question in gratitude:
        return {"answer": "You're welcome! Let me know if you have any other questions."}

    try:
        question_embedding = get_gemini_embeddings([question])[0]
        # --- IMPROVEMENT: Retrieve slightly more context ---
        query_results = index.query(vector=question_embedding, top_k=5, include_metadata=True)
        context_chunks = [match['metadata']['text'] for match in query_results['matches'] if match['id'].startswith(session_id)]
        if not context_chunks:
            return {"answer": "My apologies, I couldn't get you any relevant information according to the query from the uploaded document."}
        context = "\n---\n".join(context_chunks)

        formatted_history = "\n".join([f"{msg.sender}: {msg.text}" for msg in history])

        # --- NEW, SMARTER PROMPT ---
        prompt = f"""
        **Your Persona:**
        You are Clari, a friendly, empathetic, and highly intelligent AI health assistant. Your goal is to make medical records easy to understand.

        **Core Directives:**
        1.  **Be Conversational:** Do not re-introduce yourself. Use the "Previous Conversation History" to understand the flow of the chat and provide natural, contextual follow-up answers.
        2.  **Strictly Adhere to the Document:** Your answers must come ONLY from the "Context from the Document" provided. Never invent information.
        3.  **Explain, Don't Advise (The Smart Way):**
            * You **MUST NOT** give medical advice, your personal opinion, or make a new diagnosis.
            * **Crucially:** If the document already contains a diagnosis (like "NSTEMI") or a medication (like "Aspirin"), your primary job is to **explain it clearly**. If the user asks "Did I have a heart attack?" and the document says "Diagnosis: NSTEMI," you should explain, "The document states a diagnosis of NSTEMI, which is a type of heart attack. It is described as..." This is explaining, not diagnosing.
        4.  **Handle Missing Information Gracefully:** If the answer is not in the context, state it clearly and politely: "My apologies, I couldn't get you any relevant information according to the query from the uploaded document. It might be best to discuss this with your doctor."
        5.  **Format for Clarity:**
            * Use **bold text** for medical terms and key concepts.
            * Use bullet points to break down complex lists (like medications).
            * Keep responses concise and directly answer the user's question, but provide enough detail to be truly helpful.

        ---
        **Previous Conversation History:**
        {formatted_history}
        ---

        **Context from the Document:**
        {context}
        ---

        **Current Question from User:**
        {question}

        **Your Answer:**
        """

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return {"answer": response.text}

    except Exception as e:
        print(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- 5. Static Files and Catch-All Route ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{catch_all:path}", response_class=FileResponse)
async def serve_react_app(catch_all: str):
    return FileResponse("static/index.html")

# --- 6. Main execution block ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
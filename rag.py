# rag.py
from pathlib import Path
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import requests
import hashlib
import os

# --- Config ---
DOCUMENT_FOLDER = "data"  # Folder containing PDFs
CHROMA_DIR = "rag_chroma"
COLLECTION_NAME = "documents"
MODEL_NAME = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Helper: Hash a string to detect duplicates ---
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# --- Helper: Extract text from a PDF file ---
def extract_pdf_text(path: Path) -> str:
    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)

# --- Load embedding model and DB ---
print("[1] Initializing embedder and ChromaDB...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# --- Load existing IDs to avoid duplicates ---
existing_ids = set(collection.get()["ids"])
print(f"  → Existing chunks in DB: {len(existing_ids)}")

# --- Process all PDFs in folder ---
print(f"[2] Scanning folder: {DOCUMENT_FOLDER}")
pdf_files = list(Path(DOCUMENT_FOLDER).glob("*.pdf"))

if not pdf_files:
    print("  ⚠️ No PDFs found.")
    exit()

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
new_chunks = 0

for pdf_path in pdf_files:
    print(f"\n→ Processing: {pdf_path.name}")
    text = extract_pdf_text(pdf_path)
    chunks = splitter.split_text(text)

    for chunk in chunks:
        chunk_id = hash_text(chunk + pdf_path.name)  # include filename to namespace IDs
        if chunk_id in existing_ids:
            continue  # skip duplicate

        embedding = embedder.encode(chunk)
        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{"source": pdf_path.name}]
        )
        existing_ids.add(chunk_id)
        new_chunks += 1

print(f"\n✅ Ingestion complete. New chunks added: {new_chunks}")

# --- Ask a question ---
print("\n[3] Ready for questions.")
while True:
    query = input("\nAsk a question (or 'exit'): ").strip()
    if query.lower() in ("exit", "quit"):
        break

    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=["documents", "metadatas"]
    )
    retrieved_chunks = results['documents'][0]
    sources = results.get("metadatas", [[]])[0] or [{} for _ in retrieved_chunks]
    context = "\n\n".join(f"[{sources[i]['source']}]: {retrieved_chunks[i]}" for i in range(len(retrieved_chunks)))

    prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    print("\n→ Asking Mistral via Ollama...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    print("\n" + response.json()["response"])

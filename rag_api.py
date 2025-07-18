from flask import Flask, request, jsonify
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import hashlib
import os

# --- Configuration ---
CHROMA_DIR = "rag_chroma"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434"

# --- Initialize components ---
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# --- Flask setup ---
app = Flask(__name__)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.get_json()

    # Extract the last user message from the OpenAI-style input
    messages = data.get("messages", [])
    user_query = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_query = m["content"]
            break

    if not user_query:
        return jsonify({"error": "No user message found."}), 400

    # Embed the query
    query_embedding = embedder.encode([user_query])[0]

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=["documents", "metadatas"]
    )

    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    context = "\n\n".join(f"[{sources[i]['source']}]: {chunks[i]}" for i in range(len(chunks)))

    # Build prompt
    prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"""

    # Call Ollama
    ollama_response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response_text = ollama_response.json().get("response", "")

    # Return OpenAI-compatible response
    return jsonify({
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 0,
        "model": OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai  

INDEX_DIR = "faiss_index"

# ---------------- Retrieval ----------------
def load_faiss_index(model_name="all-MiniLM-L6-v2", device="cpu"):
    index_file = os.path.join(INDEX_DIR, "faiss_index.bin")
    meta_file = os.path.join(INDEX_DIR, "metadata.json")

    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        raise FileNotFoundError("❌ FAISS index or metadata not found. Run Segment 4 first.")

    index = faiss.read_index(index_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(model_name, device=device)
    return index, metadata, model

def retrieve(query, top_k=3):
    index, metadata, model = load_faiss_index()

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        results.append(metadata[idx]["text"])
    return results

# ---------------- Gemini Chatbot ----------------
def ask_gemini(query):
    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file")

    # Initialize client
    client = genai.Client(api_key=api_key)

    # Retrieve context
    context_chunks = retrieve(query, top_k=3)
    context = "\n".join(context_chunks)

    # Build prompt
    prompt = f"""
    You are a helpful assistant. Use the following context to answer:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Call Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # ✅ updated model name
        contents=prompt
    )

    return response.text

# ---------------- Main Loop ----------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk me something (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        try:
            answer = ask_gemini(query)
            print("\n🤖", answer)
        except Exception as e:
            print("⚠️ Error:", e)

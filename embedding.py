import os
import json
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = "chunks"
INDEX_DIR = "faiss_index"

# ---------------- Step 1: Load latest chunk file ----------------
def load_latest_chunks(chunks_dir=CHUNKS_DIR):
    files = glob.glob(os.path.join(chunks_dir, "chunks_*.json"))
    if not files:
        raise FileNotFoundError("❌ No chunk files found in 'chunks/' folder. Run chunking first.")
    
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks from {latest_file}")
    return chunks, latest_file

# ---------------- Step 2: Generate embeddings ----------------
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", device="cpu"):
    model = SentenceTransformer(model_name, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"✅ Generated embeddings with shape {embeddings.shape}")
    return embeddings

# ---------------- Step 3: Save FAISS index ----------------
def save_faiss_index(embeddings, chunks, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    index_file = os.path.join(index_dir, "faiss_index.bin")
    meta_file = os.path.join(index_dir, "metadata.json")

    # Save FAISS index
    faiss.write_index(index, index_file)

    # Save metadata (to map vectors back to chunks)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved FAISS index to {index_file}")
    print(f"✅ Saved metadata to {meta_file}")

# ---------------- Run the pipeline ----------------
if __name__ == "__main__":
    # Pick GPU if available (GTX 1650 works with CUDA if installed)
    device = "cuda" if SentenceTransformer("all-MiniLM-L6-v2")._target_device.type == "cuda" else "cpu"
    
    chunks, chunk_file = load_latest_chunks()
    embeddings = embed_chunks(chunks, device=device)
    save_faiss_index(embeddings, chunks)

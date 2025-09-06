import os
import json
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

RESULTS_DIR = "results"
CHUNKS_DIR = "chunks"

# ---------------- Step 1: Load parsed docs ----------------
def load_documents(results_dir=RESULTS_DIR):
    docs = []
    for file in os.listdir(results_dir):
        if file.endswith("-output.json"):
            with open(os.path.join(results_dir, file), "r", encoding="utf-8") as f:
                elements = json.load(f)
                for e in elements:
                    text = e.get("text")
                    if text and text.strip():
                        docs.append({"text": text.strip(), "source": file})
    print(f"✅ Loaded {len(docs)} text elements from {results_dir}")
    return docs

# ---------------- Step 2: Chunking ----------------
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({"text": chunk, "source": d["source"]})
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ---------------- Step 3: Save Chunks ----------------
def save_chunks(chunks, chunks_dir=CHUNKS_DIR):
    os.makedirs(chunks_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(chunks_dir, f"chunks_{timestamp}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Saved {len(chunks)} chunks to {output_file}")
    return output_file

# ---------------- Test the pipeline ----------------
if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    saved_file = save_chunks(chunks)

    print("\n🔎 Sample loaded document:")
    print(docs[0])

    print("\n🔎 First 2 chunks:")
    for c in chunks[:2]:
        print(c)

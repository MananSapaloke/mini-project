import os
import json
import time
import glob
import faiss
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
from dotenv import load_dotenv
from unstructured.partition.html import partition_html
from unstructured.staging.base import elements_to_json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import streamlit as st
from google import genai


# ================================
# CONFIG
# ================================
RESULTS_DIR = "results"
CHUNKS_DIR = "chunks"
INDEX_DIR = "faiss_index"

# Create folders if missing
for folder in [RESULTS_DIR, CHUNKS_DIR, INDEX_DIR]:
    os.makedirs(folder, exist_ok=True)


# ================================
# HELPER FUNCTIONS
# ================================
def get_resource_type(url):
    if url.endswith('.pdf'): return 'pdf'
    if url.endswith('.docx'): return 'docx'
    if url.endswith('.doc'): return 'doc'
    return 'html'

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and \
           parsed.scheme in ['http', 'https'] and not parsed.fragment


# ================================
# STEP 1: Crawl Website
# ================================
def crawl_website(start_url, max_depth):
    urls_to_visit = [(start_url, 0)]
    visited_urls = set()
    crawled_resources = []
    
    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)
        if current_url in visited_urls or current_depth > max_depth:
            continue
        try:
            visited_urls.add(current_url)
            headers = {'User-Agent': 'MyProjectCrawler/1.0'}
            response = requests.get(current_url, timeout=10, headers=headers)
            response.raise_for_status()

            resource_info = {
                'source_url': current_url,
                'type': get_resource_type(current_url),
                'depth': current_depth,
            }
            crawled_resources.append(resource_info)

            if resource_info['type'] == 'html' and current_depth < max_depth:
                soup = BeautifulSoup(response.content, 'html.parser')
                for link_tag in soup.find_all('a', href=True):
                    href = link_tag['href']
                    absolute_url = urljoin(current_url, href)
                    if is_valid_url(absolute_url):
                        if urlparse(absolute_url).netloc == urlparse(start_url).netloc:
                            if absolute_url not in visited_urls:
                                urls_to_visit.append((absolute_url, current_depth + 1))

        except Exception:
            continue
    return crawled_resources


# ================================
# STEP 2: Process with Unstructured
# ================================
def process_with_unstructured(resources, results_dir=RESULTS_DIR):
    for i, res in enumerate(resources, 1):
        if res['type'] != 'html':
            continue
        url = res['source_url']
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html_content = response.text
            tmp_file = os.path.join(results_dir, f"page_{i}.html")

            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            elements = partition_html(filename=tmp_file)
            output_file = os.path.join(results_dir, f"page_{i}-output.json")
            elements_to_json(elements=elements, filename=output_file)
            os.remove(tmp_file)
        except Exception:
            continue


# ================================
# STEP 3: Load & Chunk Documents
# ================================
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
    return docs

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
    return chunks

def save_chunks(chunks, chunks_dir=CHUNKS_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(chunks_dir, f"chunks_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    return output_file


# ================================
# STEP 4: Embedding + FAISS Index
# ================================
def load_latest_chunks(chunks_dir=CHUNKS_DIR):
    files = glob.glob(os.path.join(chunks_dir, "chunks_*.json"))
    if not files:
        return None, None
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks, latest_file

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", device="cpu"):
    model = SentenceTransformer(model_name, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def save_faiss_index(embeddings, chunks, index_dir=INDEX_DIR):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    index_file = os.path.join(index_dir, "faiss_index.bin")
    meta_file = os.path.join(index_dir, "metadata.json")
    faiss.write_index(index, index_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

def load_faiss_index(model_name="all-MiniLM-L6-v2", device="cpu"):
    index_file = os.path.join(INDEX_DIR, "faiss_index.bin")
    meta_file = os.path.join(INDEX_DIR, "metadata.json")
    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        return None, None, None
    index = faiss.read_index(index_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(model_name, device=device)
    return index, metadata, model


# ================================
# STEP 5: Retrieval + Gemini
# ================================
def retrieve(query, top_k=3):
    index, metadata, model = load_faiss_index()
    if not index:
        return []
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        results.append(metadata[idx]["text"])
    return results

def ask_gemini(query):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ GEMINI_API_KEY missing in .env"
    client = genai.Client(api_key=api_key)
    context_chunks = retrieve(query, top_k=3)
    context = "\n".join(context_chunks)
    prompt = f"""
    Use the following context to answer:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="Web Crawler + RAG Bot", layout="wide")
st.title("🌐 Web Crawler + RAG Chatbot (Gemini + FAISS)")

# Sidebar
st.sidebar.header("⚙️ Settings")
url_input = st.sidebar.text_input("Website URL")
depth_input = st.sidebar.number_input("Crawl Depth", min_value=0, max_value=5, value=1, step=1)

if st.sidebar.button("🚀 Run Pipeline"):
    if url_input:
        with st.spinner("Crawling website..."):
            resources = crawl_website(url_input, depth_input)
            with open("crawled_resources.json", "w", encoding="utf-8") as f:
                json.dump(resources, f, indent=4, ensure_ascii=False)
        with st.spinner("Processing with Unstructured..."):
            process_with_unstructured(resources)
        with st.spinner("Loading + chunking docs..."):
            docs = load_documents()
            chunks = chunk_documents(docs)
            save_chunks(chunks)
        with st.spinner("Embedding + Building FAISS index..."):
            chunks, _ = load_latest_chunks()
            if chunks:
                device = "cuda" if SentenceTransformer("all-MiniLM-L6-v2")._target_device.type == "cuda" else "cpu"
                embeddings = embed_chunks(chunks, device=device)
                save_faiss_index(embeddings, chunks)
        st.success("✅ Pipeline completed! You can now ask questions.")


# Chatbot section
st.subheader("🤖 Ask Questions")
user_query = st.text_input("Type your question here:")
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            answer = ask_gemini(user_query)
        st.markdown(f"**Answer:** {answer}")

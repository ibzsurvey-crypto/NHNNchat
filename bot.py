import os
import json
import base64
from typing import List

import fitz
import pymupdf4llm
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nest_asyncio

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from openai import AsyncOpenAI

nest_asyncio.apply()

# ===========================
# 1. CONFIGURATION
# ===========================
PDF_FOLDER = "./pdfs"
IMAGE_FOLDER = "./page_images"
INDEX_PATH = "./node_index.json"

MODEL_NAME = "all-MiniLM-L6-v2"   # Embeddings
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TOP_K_NODES = 3

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ===========================
# 2. INIT OCR & EMBEDDINGS
# ===========================
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=True)
print("OCR ready.")

embed_model = SentenceTransformer(MODEL_NAME)
print("Embedding model ready.")

# ===========================
# 3. INIT VLM CLIENT
# ===========================
# Provide your GROQ API key here or via environment
client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ===========================
# 4. UTILS
# ===========================
def run_ocr(image_path):
    try:
        result = ocr.ocr(image_path)
        lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                lines.append(text)
        return "\n".join(lines)
    except Exception as e:
        print("OCR failed:", e)
        return ""

def render_page(pdf_path, page_num, scale=0.5, output_dir=IMAGE_FOLDER):
    os.makedirs(output_dir, exist_ok=True)
    page_id = f"{os.path.basename(pdf_path)}_p{page_num}"
    image_path = os.path.join(output_dir, f"{page_id}.jpg")
    if not os.path.exists(image_path):
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        pix.save(image_path)
        doc.close()
    return image_path

# ===========================
# 5. INDEX BUILD / LOAD
# ===========================
def build_index():
    node_index = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Processing PDF: {pdf_file}")

        try:
            md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        except:
            md_pages = []

        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            page_num = i + 1
            page_id = f"{pdf_file}_p{page_num}"
            image_path = render_page(pdf_path, page_num)

            md_text = md_pages[i]["text"] if i < len(md_pages) and isinstance(md_pages[i], dict) else ""
            ocr_text = run_ocr(image_path) if len(md_text.strip()) < 30 else ""
            combined_text = md_text + "\n" + ocr_text

            node_index.append({
                "doc_id": pdf_file.replace(".pdf", ""),
                "page_num": page_num,
                "page_id": page_id,
                "image_path": image_path,
                "text": combined_text
            })
        doc.close()

    with open(INDEX_PATH, "w") as f:
        json.dump(node_index, f, indent=2)
    print(f"Node index saved: {INDEX_PATH}")
    return node_index

def load_index():
    if os.path.exists(INDEX_PATH):
        print("Loading cached node index...")
        with open(INDEX_PATH,"r") as f:
            return json.load(f)
    return build_index()

nodes = load_index()

# ===========================
# 6. FAISS INDEX
# ===========================
print("Creating node embeddings...")
node_texts = [n["text"] for n in nodes]
node_embeddings = embed_model.encode(node_texts, convert_to_numpy=True)

dim = node_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(node_embeddings)
print(f"FAISS index ready with {len(nodes)} nodes.")

def semantic_search(query, top_k=TOP_K_NODES):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(q_emb, top_k)
    return [nodes[i] for i in I[0]]

# ===========================
# 7. VLM QUERY
# ===========================
async def ask_vlm(query, retrieved_nodes):
    content = [{"type":"text","text":f"""
Answer ONLY using the provided pages.
Question:
{query}

Rules:
- Cite DOC ID and PAGE NUMBER
- Use only visible information
- Be concise
"""}]

    for node in retrieved_nodes:
        content.append({"type":"text","text":f"[SOURCE: {node['doc_id']} PAGE {node['page_num']}]"})
        with open(node["image_path"], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.append({"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}})

    response = await client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role":"user","content":content}],
        temperature=0,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ===========================
# 8. FASTAPI APP
# ===========================
app = FastAPI(title="Hierarchical PDF Chatbot")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Hierarchical PDF Chatbot API is running."}

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    retrieved_nodes = semantic_search(req.query)
    print("Top nodes:")
    for n in retrieved_nodes:
        print(f"  {n['doc_id']} page {n['page_num']}")
    answer = await ask_vlm(req.query, retrieved_nodes)
    return {"answer": answer}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = os.path.join(PDF_FOLDER, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    # Rebuild index after upload
    global nodes, faiss_index
    nodes = build_index()
    node_texts = [n["text"] for n in nodes]
    node_embeddings = embed_model.encode(node_texts, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(node_embeddings.shape[1])
    faiss_index.add(node_embeddings)
    return {"message": f"{file.filename} uploaded and index rebuilt."}

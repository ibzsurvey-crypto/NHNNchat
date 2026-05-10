# fastapi_pdf_chatbot_light.py

import os
import json
import base64
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

# ===========================
# CONFIG
# ===========================
INDEX_PATH = "./node_index.json"
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TOP_K_NODES = 3

# ===========================
# LOAD NODE INDEX
# ===========================
with open(INDEX_PATH, "r") as f:
    nodes = json.load(f)

print(f"Loaded {len(nodes)} nodes from node_index.json")

# ===========================
# BUILD FAISS INDEX
# ===========================
# node['embedding'] is a list -> convert to numpy float32
node_embeddings = np.array([n["embedding"] for n in nodes], dtype=np.float32)
dim = node_embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(node_embeddings)
print(f"FAISS index ready with {len(nodes)} nodes.")

# ===========================
# SEMANTIC SEARCH
# ===========================
def semantic_search(query_embedding, top_k=TOP_K_NODES):
    D, I = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return [nodes[i] for i in I[0]]

# ===========================
# VLM CLIENT
# ===========================
client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

async def ask_vlm(query, retrieved_nodes):
    # Build content for VLM
    content = [{"type": "text", "text": f"""
Answer ONLY using the provided pages.
Question:
{query}

Rules:
- Cite DOC ID, PAGE NUMBER, CHUNK NUMBER
- Use only visible information
- Be concise
"""}]

    for node in retrieved_nodes:
        content.append({"type": "text", "text": f"[SOURCE: {node['doc_id']} PAGE {node['page_num']} CHUNK {node['chunk_num']}]"})
        # Pre-rendered images
        if os.path.exists(node["image_path"]):
            with open(node["image_path"], "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

    response = await client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ===========================
# FASTAPI APP
# ===========================
app = FastAPI(title="Lightweight PDF Chatbot")

class QueryRequest(BaseModel):
    query: str
    embedding: list  # precomputed embedding of the query

@app.get("/")
async def root():
    return {"message": "Lightweight PDF Chatbot API is running."}

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    # Use precomputed embedding sent by client
    retrieved_nodes = semantic_search(req.embedding)
    answer = await ask_vlm(req.query, retrieved_nodes)
    return {"answer": answer}

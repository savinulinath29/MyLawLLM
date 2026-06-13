import os
import pickle
import logging
from typing import Any, List, Optional, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from rank_bm25 import BM25Okapi
try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    try:
        from langchain_qdrant import Qdrant as QdrantVectorStore
    except ImportError:
        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MyLawLLM")

# Configuration
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_docs")

EMBED_MODEL = "all-MiniLM-L6-v2"
GPT_MODEL = "gpt-4o"

DENSE_TOP_K = 15
BM25_TOP_K = 15
FINAL_TOP_K = 5

BM25_PATH = "bm25_index.pkl"
TEXTS_PATH = "all_texts.pkl"

# Environment Validation
missing_vars = []
if not QDRANT_API_KEY: missing_vars.append("QDRANT_API_KEY")
if not QDRANT_URL: missing_vars.append("QDRANT_URL")
if not GITHUB_TOKEN: missing_vars.append("GITHUB_TOKEN")

if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}. Backend may fail during RAG.")

# Initialize Components
logger.info("Initializing MyLawLLM components...")

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    if QDRANT_URL:
        # Check if it's a cloud URL to set the correct port
        is_cloud = "qdrant.io" in QDRANT_URL
        qdrant_client = QdrantClient(
            url=QDRANT_URL, 
            port=443 if is_cloud else 6333,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False, # Use REST for better compatibility on free tiers
        )
    else:
        qdrant_client = None
    
    if qdrant_client:
        try:
            db = QdrantVectorStore(
                client=qdrant_client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
            )
        except TypeError:
            db = QdrantVectorStore(
                client=qdrant_client,
                collection_name=QDRANT_COLLECTION,
                embeddings=embeddings,
            )
    else:
        db = None
except Exception as e:
    logger.error(f"Failed to initialize Vector Store: {e}")
    db = None

# Load BM25 Index
bm25 = None
all_texts = []
all_metadata = []

if os.path.exists(BM25_PATH) and os.path.exists(TEXTS_PATH):
    logger.info("Loading BM25 index from disk...")
    try:
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(TEXTS_PATH, "rb") as f:
            all_texts, all_metadata = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading BM25 index: {e}")
else:
    logger.warning("BM25 index not found. Hybrid search will be restricted.")

openai_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
) if GITHUB_TOKEN else None

def search(query: str) -> List[Dict[str, Any]]:
    if not db or not bm25:
        logger.warning("Search components not fully initialized")
        return []

    try:
        # Dense Search
        dense = db.similarity_search_with_score(query, k=DENSE_TOP_K)

        # Sparse Search (BM25)
        bm25_scores = bm25.get_scores(query.lower().split())
        sparse_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:BM25_TOP_K]

        pool: Dict[str, Dict[str, Any]] = {}

        for doc, score in dense:
            content = doc.page_content
            pool[content] = {
                "content": content,
                "source": doc.metadata.get("source", "Unknown Document"),
                "dense_score": float(score),
                "bm25_score": 0.0,
            }

        for i in sparse_idx:
            content = all_texts[i]
            metadata = all_metadata[i] if i < len(all_metadata) else {"source": "Unknown Document"}
            source = metadata.get("source", "Unknown Document") if isinstance(metadata, dict) else "Unknown Document"

            if content in pool:
                pool[content]["bm25_score"] = float(bm25_scores[i])
            else:
                pool[content] = {
                    "content": content,
                    "source": source,
                    "dense_score": 0.0,
                    "bm25_score": float(bm25_scores[i]),
                }

        chunks = list(pool.values())
        if not chunks:
            return []

        # Reciprocal Rank Fusion / Scoring
        max_dense = max((c["dense_score"] for c in chunks), default=1.0) or 1.0
        max_bm25 = max((c["bm25_score"] for c in chunks), default=1.0) or 1.0

        for c in chunks:
            dense_comp = 1 - (c["dense_score"] / max_dense) if max_dense else 0.0
            bm25_comp = c["bm25_score"] / max_bm25 if max_bm25 else 0.0
            c["score"] = 0.5 * dense_comp + 0.5 * bm25_comp

        return sorted(chunks, key=lambda c: c["score"], reverse=True)[:FINAL_TOP_K]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

SYSTEM_PROMPT = """You are MyLawLLM, a premium legal assistant specializing in Sri Lankan Law.
Your goal is to provide authoritative, accurate, and helpful legal information based ONLY on the provided context.

Maintain a professional, trustworthy, and empathetic tone. 

STRUCTURE YOUR RESPONSE:
1. **Plain-English Answer**: A clear, direct explanation of the legal situation in simple terms.
2. **Legal Basis**: A detailed list of specific Acts, Ordinances, or Codes with Section numbers that support your answer.

RULES:
- Use only the provided legal text. 
- If the answer isn't in the text, state: "I cannot find a specific legal basis for this in my current database."
- Always cite the Act name and Section clearly.
- Be concise but thorough.
"""

app = FastAPI(title="MyLawLLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    history: List[Message] = Field(default_factory=list)

@app.post("/ask")
async def ask(req: Query):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not configured.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = search(question)
    if not chunks:
        return {
            "answer": "I'm sorry, I couldn't find any relevant legal documents in my database to answer your question accurately.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[{i + 1}. Source: {c['source']}]\n{c['content']}" for i, c in enumerate(chunks)
    )

    history = [{"role": m.role, "content": m.content} for m in req.history[-6:]]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
    ]

    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.1,
        )
        answer = response.choices[0].message.content or "No response generated."
        
        return {
            "answer": answer,
            "sources": [
                {"source": c["source"], "excerpt": c["content"][:400]} 
                for c in chunks
            ],
        }
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response from legal assistant.")

@app.get("/health")
def health():
    return {"status": "healthy", "components": {
        "vector_db": db is not None,
        "bm25": bm25 is not None,
        "llm": openai_client is not None
    }}

# Static File Serving
if os.path.isdir("frontend"):
    # Mount frontend for explicit /assets access if needed
    app.mount("/assets", StaticFiles(directory="frontend", html=False), name="assets")

@app.get("/")
async def serve_index():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"message": "Frontend not found"})

# Catch-all to serve static files from frontend root OR index.html (for SPA)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Ignore API routes
    if full_path.startswith("ask") or full_path.startswith("health") or full_path.startswith("assets"):
        raise HTTPException(status_code=404)
        
    # Check if the file exists in the frontend directory
    file_path = os.path.join("frontend", full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fallback to index.html for SPA behavior
    return await serve_index()
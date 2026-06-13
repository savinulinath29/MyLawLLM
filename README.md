# MyLawLLM 🏛️

> A legal assistant for Sri Lankan law that combines hybrid retrieval with LLM reasoning to deliver accurate, grounded, and understandable answers.

---

## Overview

MyLawLLM bridges the gap between complex legal language and everyday understanding. Instead of relying purely on a language model's training data, the system retrieves relevant legal text from a curated knowledge base and uses it as grounding context for every response — ensuring answers are both interpretable and traceable to real sources.

---

## How It Works
```text
User Query
    │
    ▼
Hybrid Retrieval
    ├── Dense Search   (Embeddings / Semantic Similarity)
    └── Sparse Search  (BM25 / Keyword Relevance)
    │
    ▼
Merge & Rank Results
    │
    ▼
Top Chunks → LLM Context
    │
    ▼
Structured Response
    ├── Plain-English Explanation
    └── Legal Basis (Acts & Sections cited)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| LLM API | OpenAI-compatible endpoint (GitHub Models / Azure) |
| Vector Database | Qdrant (Cloud-hosted) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Search | Hybrid Retrieval (Dense + BM25 via `rank-bm25`) |
| Frontend | HTML, CSS, JavaScript |

---

## Key Idea

Instead of asking an LLM to *know* the law, **MyLawLLM forces the model to *read* the law first, then answer.**

This makes responses:
- ✅ **Reliable** — grounded in actual legal text, not model memory
- ✅ **Transparent** — every answer cites the relevant Act and section
- ✅ **Practical** — written in plain English anyone can understand

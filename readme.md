# High-Performance Asynchronous RAG API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI and Python, engineered for end-to-end speed — from concurrent chunked downloads and intelligent caching, to fast unified indexing and async answer generation.

---

## Architecture & Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Dense semantic search (FAISS) + sparse lexical matching (BM25 Okapi) |
| **Lightweight Ranking** | Reciprocal Rank Fusion (RRF) + Maximal Marginal Relevance (MMR) |
| **Async Downloads** | Concurrent, chunked byte-range requests via `aiohttp` |
| **Local Caching** | Hashes and caches PDFs, chunks, embeddings, and vector indices |
| **Query Refinement** | LLM pre-pass strips noise and extracts keywords before vector search |

> **Note:** The heavy neural cross-encoder reranking stage has been intentionally removed due to GPU constraints. RRF and MMR efficiently bridge this gap without the massive compute overhead.

---

## Setup

### 1. Prerequisites

Python 3.9+ is required.

### 2. Install Dependencies

```bash
pip install fastapi uvicorn aiohttp openai pymupdf python-docx rank_bm25 faiss-cpu numpy pydantic uvloop
```

### 3. Set Environment Variable

```bash
# macOS / Linux
export OPENAI_API_KEY="your-api-key-here"

# Windows
set OPENAI_API_KEY="your-api-key-here"
```

> Required for `text-embedding-3-small` and `gpt-4o-mini`.

---

## Running the Server

```bash
python rag_server.py
```

The server starts at `http://0.0.0.0:8000` ( `http://localhost:8000` ).

---

## Using Custom Documents

> **Important:** Standard cloud sharing links (e.g. Google Drive) return an HTML viewer page — not the raw file. The API requires a direct binary stream.

To use your own `.pdf` or `.docx` files:

1. Upload the file to a **public GitHub repository**.
2. Open the file on GitHub and click the **"Raw"** button.
3. Copy the URL — it must begin with `https://raw.githubusercontent.com/...`
4. Use that raw URL in your API requests.

---

## Authentication

All operational endpoints require a Bearer token in the `Authorization` header.

```
Authorization: Bearer 0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176
```

---

## Sample Documents

Use any of these for testing:

- `https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc1.pdf`
- `https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc2.pdf`
- `https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc3.pdf`
- `https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc4.pdf`
- `https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc5.pdf`

---

## Testing the API

The API supports batching — pass multiple document URLs and multiple questions in a single request; they are processed concurrently.

### Option A — Swagger UI (Recommended)

1. Go to `http://localhost:8000/docs`
2. Click the **Authorize** 🔒 button and paste the Bearer token.
3. Open `POST /api/v1/hackrx/run` → click **Try it out**.
4. Paste the request body below and click **Execute**.

```json
{
  "documents": [
    "https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc1.pdf",
    "https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc2.pdf"
  ],
  "questions": [
    "What is the limit of indemnity payable for cleaning out the engine?",
    "According to the Definitions section, what is the exact age limit defined for a 'Dependent Child'?"
  ]
}
```

### Option B — cURL

```bash
curl -X POST 'http://localhost:8000/api/v1/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer 0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      "https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc1.pdf",
      "https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc2.pdf"
    ],
    "questions": [
      "What is the limit of indemnity payable for cleaning out the engine?",
      "According to the Definitions section, what is the exact age limit defined for a '\''Dependent Child'\''?"
    ]
  }'
```
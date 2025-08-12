# HackRxRAG Application

Async retrieval-augmented generation (RAG) system for answering questions over policy and legal PDF/DOCX documents highly optimised for rapid indexing to answer generation.
Hybrid search (BM25 + FAISS) combined with GPT-4o-mini answer generation. Built for low-latency throughput using parallel downloads, batched embeddings, and multi-level caching.

## How it works

1. Downloads documents in parallel (range-request HTTP, cached to disk)
2. Chunks text respecting paragraph boundaries
3. Embeds chunks via `text-embedding-3-small`, indexes with BM25 + FAISS
4. Refines each query with GPT (extracts keywords, cleans phrasing)
5. Hybrid search: 55% BM25 lexical + 45% FAISS semantic, fused via RRF, reranked via MMR
6. Generates a concise answer with GPT-4o-mini from the top passages

## Prerequisites

```
pip install openai aiohttp uvicorn fastapi pydantic numpy PyMuPDF python-docx rank-bm25 faiss-cpu
```

Set your OpenAI key:

```
export OPENAI_API_KEY="sk-..."   # Linux/macOS
$env:OPENAI_API_KEY="sk-..."    # Windows PowerShell
```

## Run as a server

```
python rag_server.py
```

Starts on `http://0.0.0.0:8000`.

**Health check:**
```
GET /health
```

**Query endpoint** (Bearer token required):
```
POST /hackrx/run
Authorization: Bearer 0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176
Content-Type: application/json

{
  "documents": "https://example.com/policy.pdf",
  "questions": ["What is the age limit for a dependent child?"]
}
```

`documents` and `questions` each accept a single string or a list of strings.

**Response:**
```json
{ "answers": ["The dependent child age limit is 25 years."] }
```

## Run standalone (demo)

```
python rag_app.py
```

Runs against a hardcoded sample PDF and prints answers to stdout. Edit `main()` at the bottom of `rag_app.py` to change the URLs and questions.

## Caching

Processed documents (chunks, embeddings, BM25/FAISS indices) are cached under:
- Windows: `%LOCALAPPDATA%\Temp\rag_cache\`
- Raw PDF bytes: `%LOCALAPPDATA%\Temp\pdf_cache\`

Cache is keyed by URL. Subsequent requests for the same document skip all processing.

## Configuration

All tuneable constants are at the top of `rag_app.py`:

| Constant | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 600 | Words per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K_CANDIDATES` | 50 | Candidates from hybrid search |
| `FINAL_TOP_K` | 12 | Passages passed to the LLM |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `COMPLETION_MODEL` | `gpt-4o-mini` | OpenAI completion model |

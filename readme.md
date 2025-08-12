# High-Performance Asynchronous RAG API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI and Python. 

**Engineered for Maximum Speed:** This system is fiercely optimized for end-to-end velocity. From concurrent, chunked byte-range downloading and intelligent local caching, to lightning-fast unified indexing and asynchronous answer generation, every component is designed to minimize latency without sacrificing accuracy.

## Core Architecture & Features

* Hybrid Retrieval: Combines dense semantic search (FAISS) with sparse lexical matching (BM25 Okapi) for context awareness and exact-keyword precision.
* Lightweight Ranking (RRF & MMR): Fuses search strategies using Reciprocal Rank Fusion, then applies Maximal Marginal Relevance to maximize context diversity and reduce redundancy. 
  > Note: The heavy neural cross-encoder reranking stage has been intentionally removed from this pipeline due to GPU constraints. RRF and MMR efficiently bridge this gap, ensuring high relevance without the massive compute overhead.
* Async Downloads: Employs concurrent, chunked byte-range requests via aiohttp to download large files at maximum network throughput.
* Local Caching: Hashes and caches PDF binaries, text chunks, embeddings, and vector indices locally to bypass processing entirely on repeat queries.
* Query Refinement: Uses a fast LLM pass to strip noise from user questions and extract specific keywords before executing vector search.

## Setup Instructions

1. Prerequisites: Ensure you have Python 3.9+ installed.
2. Install Dependencies:
   pip install fastapi uvicorn aiohttp openai pymupdf python-docx rank_bm25 faiss-cpu numpy pydantic uvloop

3. Environment Variables:
   Set your OpenAI API key (required for text-embedding-3-small and gpt-4o-mini).
   Windows: set OPENAI_API_KEY="your-api-key-here"
   Mac/Linux: export OPENAI_API_KEY="your-api-key-here"

## Running the Server

Start the FastAPI server by running the script directly:
python rag_server.py

The server will initialize on http://0.0.0.0:8000

## Using Custom PDFs

This API is strictly designed to process raw binary file streams. Standard cloud storage sharing links (like Google Drive) will fail because they return a web-viewer HTML page, not the file itself.

To use your own documents:
1. Upload your .pdf or .docx file to a public GitHub repository.
2. Open the file in GitHub and click the "Raw" button in the top right.
3. Copy the URL from your browser (it must start with https://raw.githubusercontent.com/...).
4. Use this raw URL in your API requests.

## Testing the API

Note: All operational endpoints are secured. You must use the following Bearer token:
0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176

**Example Files for Testing:**
You can use any of these sample documents to test the system:
- https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc1.pdf
- https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc2.pdf
- https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc3.pdf
- https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc4.pdf
- https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc5.pdf

**Batch Processing:**
The API natively supports batching for maximum throughput. You can pass multiple document URLs and multiple questions in a single request, and the system will process them concurrently.

Method 1: Interactive Swagger UI (Recommended)
1. Navigate to: http://localhost:8000/docs
2. Click the "Authorize" button (padlock icon).
3. Enter the token above into the Value field, click Authorize, then Close.
4. Open the POST /api/v1/hackrx/run endpoint and click "Try it out".
5. Use the following JSON body to test multiple files and questions:
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
6. Click Execute.

Method 2: Standard POST Request (cURL)
curl -X 'POST' \
  'http://localhost:8000/api/v1/hackrx/run' \
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
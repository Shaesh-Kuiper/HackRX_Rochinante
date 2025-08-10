# OpenRAG - High-Performance RAG System

A lightning-fast Retrieval-Augmented Generation (RAG) system designed to process documents and answer questions in under 20 seconds. OpenRAG combines advanced document processing, hybrid search, and GPU-accelerated reranking to deliver accurate answers from various document formats.

## 🚀 Features

- **Sub-20 Second Performance**: Optimized pipeline with aggressive caching and parallel processing
- **Multi-Format Support**: PDF, DOCX, and email documents
- **Hybrid Search**: Combines BM25 (sparse) and dense vector search with FAISS
- **Advanced Query Processing**: Automatic query refinement with lexical expansion
- **GPU-Accelerated Reranking**: Uses BAAI/bge-reranker-v2-m3 model on A100 GPU
- **Intelligent Caching**: Document and embedding caching for repeated queries
- **Parallel Processing**: Concurrent downloads, embeddings, and completions
- **RESTful API**: FastAPI-based server with comprehensive endpoints

## 🏗️ Architecture

### Core Components

1. **`rag_app.py`** - Main RAG pipeline with:
   - Document download and parsing
   - Text chunking (500 tokens, 60 overlap)
   - Parallel embedding generation
   - Hybrid search (BM25 + dense)
   - MMR (Maximal Marginal Relevance) for diversity
   - Cross-encoder reranking

2. **`rag_server.py`** - FastAPI server providing:
   - `/hackrx/run` - Main RAG endpoint
   - `/api/v1/hackrx/run` - API v1 endpoint
   - `/health` - Health check endpoint
   - Authentication and CORS support

3. **`Colab_reranker_instance.ipynb`** - GPU reranker service:
   - BAAI/bge-reranker-v2-m3 model
   - Batch processing optimization
   - A100 GPU acceleration
   - Ngrok tunnel for external access

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Main programming language
- **FastAPI** - Web framework for API server
- **OpenAI API** - LLM for embeddings and completions
- **PyMuPDF (fitz)** - PDF processing
- **python-docx** - DOCX document processing
- **FAISS** - Vector similarity search
- **asyncio/aiohttp** - Asynchronous processing

### Machine Learning
- **OpenAI text-embedding-3-small** - Document embeddings
- **OpenAI gpt-4.1-nano** - Answer generation
- **BAAI/bge-reranker-v2-m3** - Cross-encoder reranking
- **FlagEmbedding** - Reranking framework
- **transformers** - Model loading and inference

### Search & Retrieval
- **rank_bm25** - BM25 sparse retrieval
- **numpy** - Numerical computations
- **scikit-learn** - Additional ML utilities

## 📋 Workflow

The RAG pipeline follows this optimized workflow:

### 1. Parallel Document Processing
```
Document URL → Download (with caching) → Parse → Chunk (500/60) → Cache
```

### 2. Query Refinement
For each input query, the system generates:
- **Cleaned**: Grammar and spelling corrections
- **Refined**: Contextually relevant reformulation
- **Lexical 1-3**: Synonymous terms for better retrieval

Example:
```json
{
  "query": "Msed my train, cnacelled, riot, will I get covered?",
  "cleaned": "my train was cancelled due to a riot will the policy cover for it?",
  "refined": "coverage for train cancellation due to riots",
  "lexical_1": ["train", "rail", "carriage", "railcar", "wagon"],
  "lexical_2": ["cancelled", "postponed", "delayed"],
  "lexical_3": ["riot", "mob", "uproar", "unrest", "protest"]
}
```

### 3. Hybrid Search Pipeline
```
Query → Embedding → Dense Search (FAISS) + Sparse Search (BM25) → 
RRF Fusion → MMR Deduplication → Top 30 Candidates
```

### 4. GPU Reranking
```
Top 30 Candidates → Cross-Encoder Reranking → Top 3 Results → Answer Generation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Google Colab account (for GPU reranker)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Shaesh-Kuiper/HackRX_Rochinante.git
cd HackRX_Rochinante
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### GPU Reranker Setup (Google Colab)

1. **Open the Colab notebook**
   - Upload `Colab_reranker_instance.ipynb` to Google Colab
   - Ensure A100 GPU runtime is selected

2. **Run the setup cells**
   - Install dependencies
   - Initialize the reranker model
   - Start the FastAPI server with ngrok tunnel

3. **Copy the ngrok URL**
   - Update `RERANKER_URL` in `rag_app.py` and `rag_server.py`

## 🔑 API Key Setup

### OpenAI API Key

1. **Get your API key**
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key

2. **Set environment variable**

**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. **Or update the code directly**
```python
# In rag_app.py
OPENAI_API_KEY = "your-api-key-here"
```

### Authentication Token

The API uses bearer token authentication. Update the token in `rag_server.py`:
```python
EXPECTED_AUTH_TOKEN = "your-secure-token-here"
```

## 🏃‍♂️ Usage

### Starting the Server

1. **Start the main RAG server**
```bash
python rag_server.py
```
Server will be available at `http://localhost:8000`

2. **Start the GPU reranker** (in Google Colab)
```python
# Run all cells in Colab_reranker_instance.ipynb
```

### API Endpoints

#### Main RAG Endpoint
```http
POST /hackrx/run
Authorization: Bearer your-token-here
Content-Type: application/json

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?"
  ]
}
```

#### Health Check
```http
GET /health
```

#### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Example Usage

```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer your-token-here",
    "Content-Type": "application/json"
}

data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
        "What does the policy cover for train cancellations?",
        "Are riots covered under this insurance policy?"
    ]
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

for i, answer in enumerate(result["answers"]):
    print(f"Q{i+1}: {data['questions'][i]}")
    print(f"A{i+1}: {answer}\n")
```

## ⚡ Performance Optimizations

### Caching Strategy
- **Document Cache**: Stores parsed documents and embeddings
- **PDF Cache**: Caches downloaded PDFs to avoid re-downloading
- **Embedding Cache**: Reuses embeddings for repeated documents

### Parallel Processing
- **Concurrent Downloads**: Up to 50 parallel range requests
- **Batch Embeddings**: Process up to 256 texts simultaneously  
- **Async Operations**: Non-blocking I/O throughout the pipeline

### Memory Management
- **Lazy Imports**: Load heavy libraries only when needed
- **Efficient Data Structures**: NumPy arrays for embeddings
- **Garbage Collection**: Explicit cleanup of large objects

## 🔧 Configuration

Key configuration parameters in `rag_app.py`:

```python
CHUNK_SIZE = 500                    # Token size per chunk
CHUNK_OVERLAP = 60                  # Overlap between chunks
TOP_K_CANDIDATES = 50               # Initial retrieval candidates
FINAL_TOP_K = 30                    # After MMR filtering
RERANK_TOP_K = 3                    # Final reranked results
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4.1-nano"
MAX_CONCURRENT_EMBEDDINGS = 100     # Parallel embedding requests
MAX_CONCURRENT_COMPLETIONS = 10     # Parallel completion requests
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Rate Limits**
   - Reduce `MAX_CONCURRENT_EMBEDDINGS` and `MAX_CONCURRENT_COMPLETIONS`
   - Implement exponential backoff

2. **Memory Issues**
   - Reduce `BATCH_SIZE` in reranker
   - Clear cache periodically

3. **Slow Performance**
   - Ensure GPU reranker is running
   - Check network connectivity to ngrok tunnel
   - Verify caching is working

4. **Document Parsing Errors**
   - Ensure document URLs are accessible
   - Check file format support (PDF, DOCX, email)

### Logs and Monitoring

The system provides detailed logging:
- Request/response timing
- Cache hit/miss rates
- Processing pipeline stages
- Error tracking with stack traces

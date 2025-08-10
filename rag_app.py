# rag_app.py - Main RAG Application with sub-20s performance
import asyncio
import aiohttp
import time
import sys
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
from collections import defaultdict
import hashlib
import re

# ADD once at startup (top of file or inside main() before anything else that touches asyncio)
try:
    if sys.platform != "win32":          # only use uvloop on non-Windows (Linux/macOS/Colab)
        import uvloop                    # no install/usage on Windows
        uvloop.install()
except Exception:
    pass

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
RERANKER_URL = "https://c27ef0ec42c9.ngrok-free.app/rerank"  # Update with your ngrok URL
RERANKER_PAIRS_URL = "https://c27ef0ec42c9.ngrok-free.app/rerank_pairs"  # Update with your ngrok URL
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
TOP_K_CANDIDATES = 50
FINAL_TOP_K = 30
RERANK_TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4.1-nano"  # Fast model for speed
MAX_CONCURRENT_EMBEDDINGS = 100
MAX_CONCURRENT_COMPLETIONS = 10

# Lazy imports - will be initialized when needed
_client = None
_encoding = None

def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client

def get_tiktoken_encoding():
    """Lazy initialization of tiktoken encoding"""
    global _encoding
    if _encoding is None:
        import tiktoken
        _encoding = tiktoken.encoding_for_model("gpt-4.1-nano")
    return _encoding

# ---- tiny disk cache helpers (optional but very effective) ----
def _pdf_cache_path(url: str) -> str:
    # include SAS query so you don't reuse stale blobs by accident
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"pdf_cache_{h}.bin")

def load_pdf_from_cache(url: str) -> Optional[bytes]:
    p = _pdf_cache_path(url)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None

def save_pdf_to_cache(url: str, data: bytes) -> None:
    try:
        with open(_pdf_cache_path(url), "wb") as f:
            f.write(data)
    except Exception:
        pass

# ---- fast downloader ----
async def download_pdf_fast(
    url: str,
    session: aiohttp.ClientSession,
    max_workers: int = 50,  # Increase from 12 to 50 for faster parallel downloads
    chunk_bytes: int = 2 * 1024 * 1024,  # Reduce to 2MB chunks for more parallelism
) -> bytes:
    """
    Ultra-fast parallel Range GET with aggressive optimization.
    """
    # Check cache first
    cached = load_pdf_from_cache(url)
    if cached is not None and len(cached) > 0:
        return cached

    # HEAD request to get size and check file type
    try:
        async with session.head(url, allow_redirects=True) as resp:
            resp.raise_for_status()
            
            # Check content type for unsupported files
            content_type = resp.headers.get("Content-Type", "").lower()
            if any(x in content_type for x in ["application/zip", "application/x-zip", 
                                                "application/x-compressed", "application/x-7z",
                                                "application/x-rar", "application/x-tar"]):
                raise ValueError("File type not supported: Archive/Binary file")
            
            size = int(resp.headers.get("Content-Length", "0"))
            accepts_range = "bytes" in resp.headers.get("Accept-Ranges", "").lower()
    except Exception as e:
        if "File type not supported" in str(e):
            raise e
        size = 0
        accepts_range = False

    # For small files or no range support, single GET
    if (not accepts_range) or size == 0 or size <= 5 * 1024 * 1024:  # 5MB threshold
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            save_pdf_to_cache(url, data)
            return data

    # Aggressive parallel downloading for large files
    ranges: List[Tuple[int, int]] = []
    for start in range(0, size, chunk_bytes):
        end = min(start + chunk_bytes - 1, size - 1)
        ranges.append((start, end))

    # Pre-allocate the entire buffer
    result_buffer = bytearray(size)
    
    # Use semaphore for concurrency control
    sem = asyncio.Semaphore(max_workers)
    
    async def _fetch_range(start: int, end: int) -> None:
        headers = {"Range": f"bytes={start}-{end}"}
        async with sem:
            for retry in range(3):  # Add retry logic for reliability
                try:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status not in (200, 206):
                            resp.raise_for_status()
                        data = await resp.read()
                        result_buffer[start:start + len(data)] = data
                        return
                except asyncio.TimeoutError:
                    if retry == 2:
                        raise
                    await asyncio.sleep(0.1 * (retry + 1))

    # Execute all range requests in parallel
    tasks = [asyncio.create_task(_fetch_range(s, e)) for (s, e) in ranges]
    await asyncio.gather(*tasks)

    data = bytes(result_buffer)
    save_pdf_to_cache(url, data)
    return data

@dataclass
class Chunk:
    text: str
    index: int
    embedding: Optional[np.ndarray] = None
    
@dataclass
class RefinedQuery:
    cleaned: str
    refined: str
    lexical_1: Optional[List[str]]
    lexical_2: Optional[List[str]]
    lexical_3: Optional[List[str]]

class Timer:
    """Context manager for timing operations"""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting: {self.name}")
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"✅ Completed: {self.name} in {elapsed:.3f}s")

class DocumentCache:
    """Cache manager for document chunks and embeddings"""
    
    @staticmethod
    def _cache_dir():
        cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    @staticmethod
    def _get_cache_key(url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()
    
    @staticmethod
    def _get_cache_paths(url: str) -> Dict[str, str]:
        """Get all cache file paths for a document"""
        cache_key = DocumentCache._get_cache_key(url)
        cache_dir = DocumentCache._cache_dir()
        return {
            "chunks": os.path.join(cache_dir, f"{cache_key}_chunks.pkl"),
            "embeddings": os.path.join(cache_dir, f"{cache_key}_embeddings.npy"),
            "metadata": os.path.join(cache_dir, f"{cache_key}_metadata.json"),
            "bm25": os.path.join(cache_dir, f"{cache_key}_bm25.pkl"),
            "faiss": os.path.join(cache_dir, f"{cache_key}_faiss.pkl")
        }
    
    @staticmethod
    def save_document_cache(url: str, chunks: List[Chunk], bm25_index=None, faiss_index=None):
        """Save chunks and embeddings to cache"""
        try:
            paths = DocumentCache._get_cache_paths(url)
            
            # Save chunks (without embeddings to save space)
            chunks_data = [(c.text, c.index) for c in chunks]
            with open(paths["chunks"], "wb") as f:
                pickle.dump(chunks_data, f)
            
            # Save embeddings separately as numpy array
            if chunks and chunks[0].embedding is not None:
                embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
                np.save(paths["embeddings"], embeddings)
            
            # Save metadata
            metadata = {
                "url": url,
                "num_chunks": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "timestamp": time.time(),
                "has_embeddings": chunks[0].embedding is not None if chunks else False
            }
            with open(paths["metadata"], "w") as f:
                json.dump(metadata, f)
            
            # Save BM25 index if provided
            if bm25_index is not None:
                with open(paths["bm25"], "wb") as f:
                    pickle.dump(bm25_index, f)
            
            # Save FAISS index if provided
            if faiss_index is not None:
                import faiss
                faiss.write_index(faiss_index, paths["faiss"])
            
            print(f"✅ Cached document: {len(chunks)} chunks with embeddings")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to cache document: {e}")
            return False
    
    @staticmethod
    def load_document_cache(url: str) -> Optional[Tuple[List[Chunk], any, any]]:
        """Load chunks and embeddings from cache"""
        try:
            paths = DocumentCache._get_cache_paths(url)
            
            # Check if all required files exist
            if not os.path.exists(paths["chunks"]) or not os.path.exists(paths["metadata"]):
                return None
            
            # Load metadata and check validity
            with open(paths["metadata"], "r") as f:
                metadata = json.load(f)
            
            # Check if cache settings match current settings
            if (metadata.get("chunk_size") != CHUNK_SIZE or 
                metadata.get("chunk_overlap") != CHUNK_OVERLAP):
                print(f"⚠️ Cache settings mismatch, regenerating...")
                return None
            
            # Load chunks
            with open(paths["chunks"], "rb") as f:
                chunks_data = pickle.load(f)
            
            chunks = [Chunk(text=text, index=idx) for text, idx in chunks_data]
            
            # Load embeddings if available
            if os.path.exists(paths["embeddings"]):
                embeddings = np.load(paths["embeddings"])
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            
            # Load BM25 index if available
            bm25_index = None
            if os.path.exists(paths["bm25"]):
                with open(paths["bm25"], "rb") as f:
                    bm25_index = pickle.load(f)
            
            # Load FAISS index if available
            faiss_index = None
            if os.path.exists(paths["faiss"]):
                import faiss
                faiss_index = faiss.read_index(paths["faiss"])
            
            print(f"✅ Loaded from cache: {len(chunks)} chunks with embeddings")
            return chunks, bm25_index, faiss_index
            
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
            return None
    
    @staticmethod
    def clear_cache(url: str = None):
        """Clear cache for specific URL or all cache"""
        try:
            if url:
                paths = DocumentCache._get_cache_paths(url)
                for path in paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                print(f"✅ Cleared cache for URL: {url[:50]}...")
            else:
                cache_dir = DocumentCache._cache_dir()
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)
                print("✅ Cleared all cache")
        except Exception as e:
            print(f"⚠️ Failed to clear cache: {e}")

async def download_pdf(url: str, session: aiohttp.ClientSession) -> bytes:
    """Download PDF from URL asynchronously"""
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.read()

def parse_pdf(pdf_bytes: bytes) -> str:
    """Parse PDF and extract text"""
    import fitz  # PyMuPDF - lazy import
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def detect_file_type(content: bytes, url: str = "") -> str:
    """Detect file type from content magic bytes or URL"""
    # Check magic bytes
    if content[:4] == b'%PDF':
        return 'pdf'
    elif content[:4] == b'PK\x03\x04':
        if b'word/' in content[:4096]:
            return 'docx'
        elif b'[Content_Types].xml' in content[:4096]:
            return 'docx'
        else:
            return 'zip'  # Generic zip file
    elif content[:2] == b'PK':
        return 'zip'
    elif content[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':  # Old .doc format
        return 'docx'  # Try to parse as docx
    elif b'From:' in content[:1000] or b'Subject:' in content[:1000] or b'Date:' in content[:1000]:
        return 'email'
    elif content[:4] in [b'Rar!', b'\x52\x61\x72\x21']:
        return 'rar'
    elif content[:3] == b'\x1f\x8b\x08':
        return 'gzip'
    
    # Fallback to URL extension
    url_lower = url.lower()
    if '.pdf' in url_lower:
        return 'pdf'
    elif '.docx' in url_lower:
        return 'docx'
    elif '.doc' in url_lower:
        return 'docx'
    elif '.eml' in url_lower or '.msg' in url_lower:
        return 'email'
    
    return 'unknown'

def parse_docx(docx_bytes: bytes) -> str:
    """Parse DOCX and extract text"""
    import io
    from docx import Document  # pip install python-docx
    
    try:
        doc = Document(io.BytesIO(docx_bytes))
        text_parts = []
        
        # Extract paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)
        
        return '\n'.join(text_parts).strip()
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        raise ValueError(f"Failed to parse DOCX: {str(e)}")

def parse_email(email_bytes: bytes) -> str:
    """Parse email and extract text content"""
    import email
    from email.policy import default
    
    try:
        # Parse email
        msg = email.message_from_bytes(email_bytes, policy=default)
        
        text_parts = []
        
        # Add headers
        text_parts.append(f"From: {msg.get('From', 'Unknown')}")
        text_parts.append(f"To: {msg.get('To', 'Unknown')}")
        text_parts.append(f"Subject: {msg.get('Subject', 'No Subject')}")
        text_parts.append(f"Date: {msg.get('Date', 'Unknown')}")
        text_parts.append("---")
        
        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        text_parts.append(payload.decode('utf-8', errors='ignore'))
                elif part.get_content_type() == "text/html":
                    # Only use HTML if no plain text available
                    if not any("text/plain" in p.get_content_type() for p in msg.walk()):
                        payload = part.get_payload(decode=True)
                        if payload:
                            # Simple HTML stripping
                            import re
                            html_text = payload.decode('utf-8', errors='ignore')
                            text = re.sub('<[^<]+?>', '', html_text)
                            text_parts.append(text)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                text_parts.append(payload.decode('utf-8', errors='ignore'))
        
        return '\n'.join(text_parts).strip()
    except Exception as e:
        print(f"Error parsing email: {e}")
        raise ValueError(f"Failed to parse email: {str(e)}")

def parse_document(content: bytes, url: str = "") -> str:
    """Universal document parser that detects and parses different formats"""
    file_type = detect_file_type(content, url)
    
    if file_type == 'pdf':
        return parse_pdf(content)
    elif file_type == 'docx':
        return parse_docx(content)
    elif file_type == 'email':
        return parse_email(content)
    elif file_type in ['zip', 'rar', 'gzip', 'unknown']:
        raise ValueError(f"File type not supported or file too large to process: {file_type}")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Create overlapping chunks from text"""
    chunks = []
    words = text.split()
    
    if len(words) == 0:
        return []
    
    # Handle edge case where text is smaller than chunk size
    if len(words) <= chunk_size:
        return [Chunk(text=text, index=0)]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(Chunk(text=chunk_text, index=len(chunks)))
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

async def generate_embedding_batch(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings for a batch of texts"""
    try:
        client = get_openai_client()
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [np.array(e.embedding) for e in response.data]
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return [np.zeros(1536) for _ in texts]  # Return zero vectors on error

async def embed_texts_batched(texts: List[str], batch_size: int = 256) -> List[np.ndarray]:
    """Massively parallel embedding (batched + gathered)."""
    # Split into batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    client = get_openai_client()
    async def _embed(batch):
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    tasks = [asyncio.create_task(_embed(b)) for b in batches]
    results = await asyncio.gather(*tasks)
    # flatten
    out = []
    for r in results:
        out.extend(r)
    return out

async def generate_embeddings_parallel(chunks: List[Chunk]) -> None:
    # big fast batches; use gather to run requests concurrently
    texts = [c.text for c in chunks]
    # Tune batch_size (128–512). Start with 256.
    embeddings = await embed_texts_batched(texts, batch_size=256)
    for c, e in zip(chunks, embeddings):
        # normalize and store (so everything downstream is cosine/IP-consistent)
        norm = np.linalg.norm(e) + 1e-12
        c.embedding = (e / norm).astype(np.float32)

async def refine_query(query: str) -> RefinedQuery:
    """Refine and expand query with LLM"""
    prompt = f"""Given this query about an insurance policy document, create a refined version and extract lexical terms.

Query: {query}

Return a JSON object with:
- "cleaned": Fix spelling and grammar errors
- "refined": A concise search query focusing on key concepts
- "lexical_1": List of primary search terms and synonyms (most important)
- "lexical_2": List of secondary search terms and synonyms (or null if not needed)
- "lexical_3": List of tertiary search terms and synonyms (use only if ABSOLUTELY NECESSARY or null if not needed)

Each lexical list should contain only contextually relevant synonyms.
Use null for lexical_2 or lexical_3 if they're not necessary.

few shot:
example 1 a person asking insurqaance policy:

query: Msed my train, cnacelled, riot , will I get covered?

output :

	[
		"cleaned": "my train was cancelled due to an riot will the policy cover for it?" ,
		"refined": "coverage (or compensation) for train cancellation, riots". 
		"lexical 1": "train, rail, carriage, railcar , wagon."
		"lexical 2": "cancelled, postponed, delayed." 
		"lexical 3": "riot,mob, uproar, unrest, protest, rebellion, civil."
	]

look how it took contextually relevant synonyms in lexical 1,2,and 3

example 2 (Mathematical Principles of Natural Philosophy) : 
query : reson bokk translated to english

output : 
	[
		"cleaned": "reason book translated to English",
		"refined": "translation of the book into English",
		"lexical 1": "translated, rendered, converted, interpreted",
		"lexical 2": none  
		"lexical 3": none
	]

NOTE dont use square brackets , only use proper json

here only the translation has importance no other like boo or anything has any important relevance so we only use one lexica 1 and leave the rest as none


Example output:
{{"cleaned": "...", "refined": "...", "lexical_1": ["term1", "term2"], "lexical_2": ["term3"], "lexical_3": null}}"""

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        return RefinedQuery(
            cleaned=result.get("cleaned", query),
            refined=result.get("refined", query),
            lexical_1=result.get("lexical_1"),
            lexical_2=result.get("lexical_2"),
            lexical_3=result.get("lexical_3")
        )
    except Exception as e:
        print(f"❌ Error refining query: {e}")
        return RefinedQuery(cleaned=query, refined=query, lexical_1=[query], lexical_2=None, lexical_3=None)

async def refine_queries_parallel(queries: List[str]) -> List[RefinedQuery]:
    """Refine multiple queries in parallel"""
    tasks = [refine_query(query) for query in queries]
    return await asyncio.gather(*tasks)

def create_bm25_index(chunks: List[Chunk]):
    """Create BM25 index from chunks"""
    from rank_bm25 import BM25Okapi  # lazy import
    tokenized_chunks = [chunk.text.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / norms).astype(np.float32)

def create_faiss_index(chunks: List[Chunk]):
    """Create FAISS index from chunk embeddings"""
    import faiss  # lazy import
    if not chunks or chunks[0].embedding is None:
        return None
    embs = np.array([c.embedding for c in chunks], dtype=np.float32)
    embs = _l2_normalize_rows(embs)  # cosine-ready
    index = faiss.IndexFlatIP(embs.shape[1])  # inner product
    index.add(embs)
    return index

def hybrid_search(
    query_embedding: np.ndarray,
    lexical_terms: List[str],
    chunks: List[Chunk],
    bm25_index,  # BM25Okapi - removed type hint for lazy import
    faiss_index,  # faiss.IndexFlatIP - removed type hint for lazy import
    top_k: int = TOP_K_CANDIDATES
) -> List[Tuple[int, float]]:
    """Perform hybrid BM25 + dense search"""
    bm25_query = " ".join(lexical_terms).lower().split()
    bm25_scores = bm25_index.get_scores(bm25_query)  # shape [N]

    if faiss_index is not None:
        # query_embedding must be normalized for IP
        qe = query_embedding.astype(np.float32)
        qe = qe / (np.linalg.norm(qe) + 1e-12)
        sims, idxs = faiss_index.search(qe.reshape(1, -1), min(top_k, len(chunks)))
        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        for i, s in zip(idxs[0], sims[0]):
            dense_scores[i] = s  # IP ~ cosine

        # normalize BM25 to [0,1] (cheap & stable)
        bmin, bmax = bm25_scores.min(), bm25_scores.max()
        bm25_norm = (bm25_scores - bmin) / (bmax - bmin + 1e-10)

        combined = 0.3 * bm25_norm + 0.7 * dense_scores  # weights as before
        top_idx = np.argpartition(-combined, range(min(top_k, len(chunks))))[:top_k]
        top_idx = top_idx[np.argsort(-combined[top_idx])]
        return [(int(i), float(combined[i])) for i in top_idx]

    # fallback: BM25 only
    top_idx = np.argpartition(-bm25_scores, range(min(top_k, len(chunks))))[:top_k]
    top_idx = top_idx[np.argsort(-bm25_scores[top_idx])]
    return [(int(i), float(bm25_scores[i])) for i in top_idx]

def reciprocal_rank_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[int]:
    """Fuse multiple rankings using RRF"""
    scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, (idx, score) in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    
    # Sort by RRF score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_items]

def mmr_fast(
    chunks: List[Chunk],
    query_embedding: np.ndarray,
    candidate_indices: List[int],
    top_k: int = FINAL_TOP_K,
    lambda_param: float = 0.5
) -> List[int]:
    if not candidate_indices:
        return []
    E = np.stack([chunks[i].embedding for i in candidate_indices])  # (C, D) normalized
    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    rel = E @ q  # (C,)
    selected = []
    mask = np.ones(len(candidate_indices), dtype=bool)

    # Precompute candidate-candidate sims (C, C) once
    S = E @ E.T

    for _ in range(min(top_k, len(candidate_indices))):
        if not selected:
            j = np.argmax(rel * mask)
        else:
            # maximum similarity to any selected
            max_sim = S[:, selected].max(axis=1)
            mmr = lambda_param * rel - (1 - lambda_param) * max_sim
            mmr[~mask] = -1e9
            j = np.argmax(mmr)
        selected.append(j)
        mask[j] = False

    return [candidate_indices[j] for j in selected]

async def rerank_with_cross_encoder(
    query: str,
    passages: List[str],
    session: aiohttp.ClientSession,
    reranker_url: str = RERANKER_URL
) -> List[Dict]:
    """Rerank passages using external GPU reranker"""
    try:
        payload = {"query": query, "passages": passages, "top_k": RERANK_TOP_K}
        async with session.post(reranker_url, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["top_passages"]
            else:
                print(f"❌ Reranker error: {resp.status}")
                return []
    except Exception as e:
        print(f"❌ Error calling reranker: {e}")
        return [{"passage": p, "score": 1.0, "original_index": i} for i, p in enumerate(passages[:RERANK_TOP_K])]

async def generate_answer(query: str, context: str) -> str:
    """Generate answer using LLM"""
    prompt = f"""Answer the following question based on the provided context from an insurance policy document.

NOTE:
1) Answer extremely shortly , keeping only the necessary details.
2) do not ignore the reason , if necessary provide a extremely but informative reason
3) use numbers instead of words 
4) do not include irrelevant details
5) no need to quote the section number 
6) if there is a follow up action a person need to take , mention it too
7) if Agreed on a condition put that CONDITION TOO, also mention exceptions if necessary. 
8) If SOMETHING IS COVERED MENTION THE TIME PERIOD AND CONDITIONS IN SHORT TOO
9) FOLLOWING THE ABOVE POINTS SHOULD NOT MEAN YOU SHOULD MENTION ALL OF THE THINGS U SEE IN THE PROVIDED CONTEXT! ONLY IF NECESSARY
10) DO NOT BE TOO VERBOSE

Context:
{context}

Question: {query}

Please provide a clear, concise answer based only on the information in the context. If the answer is not in the context, say so."""

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return "Error generating answer."

async def process_single_question(
    query: str,
    refined: RefinedQuery,
    query_embedding: np.ndarray,
    chunks: List[Chunk],
    bm25_index,  # BM25Okapi - removed type hint for lazy import
    faiss_index,  # faiss.IndexFlatIP - removed type hint for lazy import
    session: aiohttp.ClientSession
) -> Dict:
    """Process a single question through the entire pipeline"""
    question_start = time.time()
    
    # NEW helper inside process_single_question:
    async def _hybrid_for_terms(terms):
        return await asyncio.to_thread(
            hybrid_search, query_embedding, terms, chunks, bm25_index, faiss_index, TOP_K_CANDIDATES
        )
    
    # 3. Perform hybrid search for each lexical group (NOW IN PARALLEL)
    all_rankings = []
    lex_groups = [refined.lexical_1, refined.lexical_2, refined.lexical_3]
    tasks = [ _hybrid_for_terms(terms) for terms in lex_groups if terms ]
    if tasks:
        with Timer("Hybrid search (all lexical groups in parallel)"):
            all_rankings = await asyncio.gather(*tasks)
    else:
        all_rankings = []
    
    # 4. RRF fusion
    with Timer("RRF fusion"):
        fused_indices = reciprocal_rank_fusion(all_rankings)
    
    # 5. MMR and deduplication
    with Timer("MMR and deduplication"):
        final_indices = mmr_fast(
            chunks,
            query_embedding,
            fused_indices[:min(len(fused_indices), 100)],  # Consider top 100 from RRF
            FINAL_TOP_K
        )
    
    # 6. Rerank with cross-encoder
    with Timer("Cross-encoder reranking"):
        passages_to_rerank = [chunks[idx].text for idx in final_indices]
        reranked = await rerank_with_cross_encoder(refined.refined, passages_to_rerank, session)
    
    # 7. Generate answer
    with Timer("Answer generation"):
        if reranked:
            context = "\n\n".join([r["passage"] for r in reranked])
            answer = await generate_answer(query, context)
        else:
            answer = "No relevant information found."
    
    question_time = time.time() - question_start
    
    return {
        "question": query,
        "answer": answer,
        "time": question_time,
        "refined_query": refined.refined,
        "passages_used": len(reranked)
    }

async def retrieve_candidates_only(
    q: str,
    rq: RefinedQuery,
    qe: np.ndarray,
    chunks: List[Chunk],
    bm25_index,  # BM25Okapi - removed type hint for lazy import
    faiss_index,  # faiss.IndexFlatIP - removed type hint for lazy import
) -> Dict:
    # same as process_single_question() up through MMR
    async def _hybrid_for_terms(terms):
        return await asyncio.to_thread(
            hybrid_search, qe, terms, chunks, bm25_index, faiss_index, TOP_K_CANDIDATES
        )

    lex_groups = [rq.lexical_1, rq.lexical_2, rq.lexical_3]
    tasks = [_hybrid_for_terms(t) for t in lex_groups if t]
    all_rankings = await asyncio.gather(*tasks) if tasks else []

    fused_indices = reciprocal_rank_fusion(all_rankings)
    final_indices = mmr_fast(
        chunks, qe, fused_indices[:min(len(fused_indices), 100)], FINAL_TOP_K
    )

    passages = [chunks[i].text for i in final_indices]
    return {
        "question": q,
        "refined": rq.refined,
        "indices": final_indices,
        "passages": passages,
    }

async def process_rag_pipeline(
    document_url: str,
    questions: List[str],
    reranker_url: str = RERANKER_URL
) -> List[Dict]:
    """
    Process RAG pipeline for given document and questions.
    Returns list of results with question, answer, and metadata.
    """
    total_start = time.time()
    
    # Update global reranker URL if provided
    global RERANKER_URL
    if reranker_url:
        RERANKER_URL = reranker_url
    
    connector = aiohttp.TCPConnector(
        limit=200,
        limit_per_host=200,
        ttl_dns_cache=300,
        force_close=False,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(
        total=30,
        sock_connect=5,
        sock_read=10
    )

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
    ) as session:
        
        # Try to load from cache first
        cache_result = DocumentCache.load_document_cache(document_url)
        processed_from_scratch = False

        if cache_result is not None:
            # Document is cached
            chunks, bm25_index, faiss_index = cache_result
            print(f"📦 Using cached document with {len(chunks)} chunks")
            
            # Only process queries
            with Timer("PHASE 1: Query refinement and embedding"):
                refined_queries = await refine_queries_parallel(questions)
                refined_texts = [rq.refined for rq in refined_queries]
                query_embeddings = await embed_texts_batched(refined_texts, batch_size=512)
            
            # Recreate indices if not cached
            if bm25_index is None:
                with Timer("Recreating BM25 index from cache"):
                    bm25_index = create_bm25_index(chunks)
            if faiss_index is None:
                with Timer("Recreating FAISS index from cache"):
                    faiss_index = create_faiss_index(chunks)
        else:
            # Document not cached, process normally
            print("📄 Document not in cache, processing from scratch...")
            processed_from_scratch = True
            
            # Phase 1: Parallel PDF processing and query refinement
            with Timer("PHASE 1: Parallel PDF download + Query refinement"):
                pdf_task = download_pdf_fast(document_url, session)
                query_task = refine_queries_parallel(questions)
                pdf_bytes, refined_queries = await asyncio.gather(pdf_task, query_task)
            
            # Phase 2: Document parsing and chunking
            with Timer("PHASE 2: Document parsing and chunking"):
                text = parse_document(pdf_bytes, document_url)
                print(f"📊 Extracted {len(text)} characters from document")
                
                chunks = create_chunks(text)
                print(f"📦 Created {len(chunks)} chunks")
            
            # Phase 3: Generate embeddings
            with Timer("PHASE 3: Embeddings (queries + chunks) in parallel"):
                refined_texts = [rq.refined for rq in refined_queries]
                query_embed_task = asyncio.create_task(embed_texts_batched(refined_texts, batch_size=512))
                chunk_embed_task = asyncio.create_task(generate_embeddings_parallel(chunks))
                query_embeddings, _ = await asyncio.gather(query_embed_task, chunk_embed_task)
            
            # Phase 4: Create indices
            with Timer("PHASE 4: Index creation (BM25 + FAISS)"):
                bm25_index = create_bm25_index(chunks)
                faiss_index = create_faiss_index(chunks)
        
        # Phase 5: Process all questions in parallel
        with Timer("PHASE 5: Retrieval for all questions (no rerank)"):
            retrieval_tasks = [
                retrieve_candidates_only(q, rq, qe, chunks, bm25_index, faiss_index)
                for q, rq, qe in zip(questions, refined_queries, query_embeddings)
            ]
            retrievals = await asyncio.gather(*retrieval_tasks)

        # Build one big pair list for the reranker
        pairs_payload = []
        qid_order = []
        for qi, r in enumerate(retrievals):
            qid = f"q{qi}"
            qid_order.append(qid)
            for j, (idx, passage) in enumerate(zip(r["indices"], r["passages"])):
                pairs_payload.append({
                    "qid": qid,
                    "query": r["refined"],
                    "passage": passage,
                    "original_index": j
                })

        with Timer("PHASE 5b: ONE cross-encoder rerank for all questions"):
            async with session.post(RERANKER_PAIRS_URL, json={"pairs": pairs_payload, "top_k_per_q": RERANK_TOP_K}) as resp:
                resp.raise_for_status()
                batch = await resp.json()

        qid_to_top = batch["results"]

        # Answer generation in parallel
        with Timer("PHASE 5c: Answer generation (parallel)"):
            ans_tasks = []
            for qi, r in enumerate(retrievals):
                qid = f"q{qi}"
                top_passages = qid_to_top[qid]
                context = "\n\n".join([p["passage"] for p in top_passages])
                ans_tasks.append(generate_answer(r["question"], context))
            answers = await asyncio.gather(*ans_tasks)

        # Prepare results
        results = []
        for qi, r in enumerate(retrievals):
            qid = f"q{qi}"
            results.append({
                "question": r["question"],
                "answer": answers[qi],
                "refined_query": r["refined"],
                "passages_used": len(qid_to_top[qid])
            })
        
        # Save to cache if processed from scratch
        if processed_from_scratch:
            with Timer("CACHING: Saving document to cache"):
                DocumentCache.save_document_cache(document_url, chunks, bm25_index, faiss_index)
        
        total_time = time.time() - total_start
        print(f"\n✅ Pipeline completed in {total_time:.2f}s for {len(questions)} questions")
        
        return results

async def main():
    """Main RAG pipeline"""
    total_start = time.time()
    
    # Sample inputs
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]

    connector = aiohttp.TCPConnector(
        limit=200,  # Increase from 100 to 200
        limit_per_host=200,  # Increase for same host
        ttl_dns_cache=300,
        force_close=False,  # Keep connections alive
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(
        total=30,  # Reduce total timeout for faster failures
        sock_connect=5,
        sock_read=10
    )

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        # gzip won't help PDFs, but it's fine for JSON calls
    ) as session:
    
        print("="*80)
        print("🚀 HIGH-PERFORMANCE RAG SYSTEM")
        print("="*80)
        print(f"📄 PDF URL: {pdf_url[:50]}...")
        print(f"❓ Processing {len(questions)} questions")
        print("="*80)
        
        # Try to load from cache first
        cache_result = DocumentCache.load_document_cache(pdf_url)
        processed_from_scratch = False  # Track if we processed from scratch

        if cache_result is not None:
            # Document is cached, just refine queries and embed them
            chunks, bm25_index, faiss_index = cache_result
            print(f"📦 Using cached document with {len(chunks)} chunks")
            
            # Only process queries (parallel with cache validation)
            with Timer("PHASE 1: Query refinement and embedding"):
                refined_queries = await refine_queries_parallel(questions)
                refined_texts = [rq.refined for rq in refined_queries]
                query_embeddings = await embed_texts_batched(refined_texts, batch_size=512)
            
            # Recreate indices if not cached
            if bm25_index is None:
                with Timer("Recreating BM25 index from cache"):
                    bm25_index = create_bm25_index(chunks)
            if faiss_index is None:
                with Timer("Recreating FAISS index from cache"):
                    faiss_index = create_faiss_index(chunks)
        else:
            # Document not cached, process normally
            print("📄 Document not in cache, processing from scratch...")
            processed_from_scratch = True  # Mark that we processed from scratch
            
            # Phase 1: Parallel PDF processing and query refinement
            with Timer("PHASE 1: Parallel PDF download + Query refinement"):
                pdf_task = download_pdf_fast(pdf_url, session)
                query_task = refine_queries_parallel(questions)
                pdf_bytes, refined_queries = await asyncio.gather(pdf_task, query_task)
            
            # Phase 2: Document parsing and chunking
            with Timer("PHASE 2: Document parsing and chunking"):
                try:
                    text = parse_document(pdf_bytes, pdf_url)
                    print(f"📊 Extracted {len(text)} characters from document")
                except ValueError as e:
                    if "not supported" in str(e) or "too large" in str(e):
                        print(f"❌ {e}")
                        return
                    raise
                
                chunks = create_chunks(text)
                print(f"📦 Created {len(chunks)} chunks")
            
            # Phase 3: Generate embeddings (chunks only, queries in parallel)
            with Timer("PHASE 3: Embeddings (queries + chunks) in parallel"):
                refined_texts = [rq.refined for rq in refined_queries]
                query_embed_task = asyncio.create_task(embed_texts_batched(refined_texts, batch_size=512))
                chunk_embed_task = asyncio.create_task(generate_embeddings_parallel(chunks))
                query_embeddings, _ = await asyncio.gather(query_embed_task, chunk_embed_task)
            
            # Phase 4: Create indices
            with Timer("PHASE 4: Index creation (BM25 + FAISS)"):
                bm25_index = create_bm25_index(chunks)
                faiss_index = create_faiss_index(chunks)
    
        # Phase 5: Process all questions in parallel
        with Timer("PHASE 5: Retrieval for all questions (no rerank)"):
            retrieval_tasks = [
                retrieve_candidates_only(q, rq, qe, chunks, bm25_index, faiss_index)
                for q, rq, qe in zip(questions, refined_queries, query_embeddings)
            ]
            retrievals = await asyncio.gather(*retrieval_tasks)

        # Build one big pair list for the reranker
        pairs_payload = []
        qid_order = []
        for qi, r in enumerate(retrievals):
            qid = f"q{qi}"
            qid_order.append(qid)
            for j, (idx, passage) in enumerate(zip(r["indices"], r["passages"])):
                pairs_payload.append({
                    "qid": qid,
                    "query": r["refined"],
                    "passage": passage,
                    "original_index": j
                })

        with Timer("PHASE 5b: ONE cross-encoder rerank for all questions"):
            async with session.post(RERANKER_PAIRS_URL, json={"pairs": pairs_payload, "top_k_per_q": RERANK_TOP_K}) as resp:
                resp.raise_for_status()
                batch = await resp.json()

        qid_to_top = batch["results"]  # qid -> [{passage, score, original_index}...]

        # Now answer generation in parallel
        with Timer("PHASE 5c: Answer generation (parallel)"):
            ans_tasks = []
            for qi, r in enumerate(retrievals):
                qid = f"q{qi}"
                top_passages = qid_to_top[qid]
                context = "\n\n".join([p["passage"] for p in top_passages])
                ans_tasks.append(generate_answer(r["question"], context))
            answers = await asyncio.gather(*ans_tasks)

        results = []
        for qi, r in enumerate(retrievals):
            qid = f"q{qi}"
            results.append({
                "question": r["question"],
                "answer": answers[qi],
                "time": None,  # you can compute per-Q if you want
                "refined_query": r["refined"],
                "passages_used": len(qid_to_top[qid])
            })
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nQuestion {i}: {result['question']}")
        print(f"Answer: {result['answer']}")
        time_str = f"{result['time']:.3f}s" if result['time'] is not None else "N/A"
        print(f"Time: {time_str}")
        print(f"Refined: {result['refined_query']}")
        print(f"Passages used: {result['passages_used']}")
        print("-"*40)
    
    total_time = time.time() - total_start
    avg_time = total_time / len(questions) if questions else 0
    
    print("\n" + "="*80)
    print("⏱️  PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per question: {avg_time:.2f}s")
    print(f"Target achieved: {'✅ YES' if total_time < 20 else '❌ NO'} (< 20s)")
    print("="*80)
    
    # Save to cache after all results are displayed (only if processed from scratch)
    if processed_from_scratch:
        with Timer("CACHING: Saving document to cache"):
            DocumentCache.save_document_cache(pdf_url, chunks, bm25_index, faiss_index)

if __name__ == "__main__":
    # Ensure event loop policy for Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main pipeline
    asyncio.run(main())

# Export for use as module
__all__ = ['process_rag_pipeline', 'DocumentCache']

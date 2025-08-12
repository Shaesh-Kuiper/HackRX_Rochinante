import asyncio
import aiohttp
import time
import sys
import os
import json
import pickle
import numpy as np
import re
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    if sys.platform != "win32":
        import uvloop
        uvloop.install()
except ImportError:
    pass

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
CHUNK_SIZE = 600       
CHUNK_OVERLAP = 150
TOP_K_CANDIDATES = 50
FINAL_TOP_K = 12       
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4o-mini" 

_client = None

def get_openai_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client

def _pdf_cache_path(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "pdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"pdf_{h}.bin")

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

async def download_pdf_fast(
    url: str,
    session: aiohttp.ClientSession,
    max_workers: int = 30, 
    chunk_bytes: int = 2 * 1024 * 1024,
) -> bytes:
    cached = load_pdf_from_cache(url)
    if cached is not None and len(cached) > 0:
        return cached

    try:
        async with session.head(url, allow_redirects=True) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            if any(x in content_type for x in ["zip", "compressed", "rar", "tar", "7z"]):
                raise ValueError("File type not supported: Archive/Binary file")
            size = int(resp.headers.get("Content-Length", "0"))
            accepts_range = "bytes" in resp.headers.get("Accept-Ranges", "").lower()
    except Exception as e:
        if "File type not supported" in str(e): raise e
        size = 0
        accepts_range = False

    if (not accepts_range) or size == 0 or size <= 5 * 1024 * 1024:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            save_pdf_to_cache(url, data)
            return data

    ranges: List[Tuple[int, int]] = []
    for start in range(0, size, chunk_bytes):
        end = min(start + chunk_bytes - 1, size - 1)
        ranges.append((start, end))

    result_buffer = bytearray(size)
    sem = asyncio.Semaphore(max_workers)
    
    async def _fetch_range(start: int, end: int) -> None:
        headers = {"Range": f"bytes={start}-{end}"}
        async with sem:
            for retry in range(3):
                try:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status not in (200, 206): resp.raise_for_status()
                        data = await resp.read()
                        result_buffer[start:start + len(data)] = data
                        return
                except asyncio.TimeoutError:
                    if retry == 2: raise
                    await asyncio.sleep(0.1 * (retry + 1))

    tasks = [asyncio.create_task(_fetch_range(s, e)) for (s, e) in ranges]
    await asyncio.gather(*tasks)

    data = bytes(result_buffer)
    save_pdf_to_cache(url, data)
    return data

@dataclass
class Chunk:
    __slots__ = ['text', 'index', 'embedding']
    text: str
    index: int
    embedding: Optional[np.ndarray]
    
@dataclass
class RefinedQuery:
    __slots__ = ['refined', 'keywords']
    refined: str
    keywords: List[str]

class Timer:
    def __init__(self, name: str):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        print(f"⏱️  Starting: {self.name}")
        return self
        
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        print(f"Completed: {self.name} in {elapsed:.3f}s")

class DocumentCache:
    @staticmethod
    def _cache_dir():
        cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    @staticmethod
    def _get_cache_key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()
    
    @staticmethod
    def _get_cache_paths(url: str) -> Dict[str, str]:
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
        try:
            paths = DocumentCache._get_cache_paths(url)
            chunks_data = [(c.text, c.index) for c in chunks]
            with open(paths["chunks"], "wb") as f:
                pickle.dump(chunks_data, f)
            
            if chunks and chunks[0].embedding is not None:
                embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
                np.save(paths["embeddings"], embeddings)
            
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
            
            if bm25_index is not None:
                with open(paths["bm25"], "wb") as f:
                    pickle.dump(bm25_index, f)
            
            if faiss_index is not None:
                import faiss
                faiss.write_index(faiss_index, paths["faiss"])
                
            print(f"Cached document ({url[-20:]}): {len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"Failed to cache document: {e}")
            return False
    
    @staticmethod
    def load_document_cache(url: str) -> Optional[Tuple[List[Chunk], any, any]]:
        try:
            paths = DocumentCache._get_cache_paths(url)
            if not os.path.exists(paths["chunks"]) or not os.path.exists(paths["metadata"]):
                return None
            
            with open(paths["metadata"], "r") as f:
                metadata = json.load(f)
            
            if metadata.get("chunk_size") != CHUNK_SIZE or metadata.get("chunk_overlap") != CHUNK_OVERLAP:
                return None
            
            with open(paths["chunks"], "rb") as f:
                chunks_data = pickle.load(f)
            chunks = [Chunk(text=text, index=idx, embedding=None) for text, idx in chunks_data]
            
            if os.path.exists(paths["embeddings"]):
                embeddings = np.load(paths["embeddings"])
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            
            bm25_index = None
            if os.path.exists(paths["bm25"]):
                with open(paths["bm25"], "rb") as f:
                    bm25_index = pickle.load(f)
            
            faiss_index = None
            if os.path.exists(paths["faiss"]):
                import faiss
                faiss_index = faiss.read_index(paths["faiss"])
                
            print(f"Loaded from cache ({url[-20:]}): {len(chunks)} chunks")
            return chunks, bm25_index, faiss_index
        except Exception:
            return None

def detect_file_type(content: bytes, url: str = "") -> str:
    if content[:4] == b'%PDF': return 'pdf'
    elif content[:4] == b'PK\x03\x04':
        if b'word/' in content[:4096] or b'[Content_Types].xml' in content[:4096]: return 'docx'
        return 'zip'
    elif content[:2] == b'PK': return 'zip'
    elif b'From:' in content[:1000] and b'Subject:' in content[:1000]: return 'email'
    
    url_lower = url.lower()
    if '.pdf' in url_lower: return 'pdf'
    elif '.docx' in url_lower or '.doc' in url_lower: return 'docx'
    
    return 'unknown'

def parse_pdf(pdf_bytes: bytes) -> str:
    import fitz 
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    doc.close()
    return text.strip()

def parse_docx(docx_bytes: bytes) -> str:
    import io
    from docx import Document
    doc = Document(io.BytesIO(docx_bytes))
    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()]).strip()

def parse_document(content: bytes, url: str = "") -> str:
    file_type = detect_file_type(content, url)
    if file_type == 'pdf': return parse_pdf(content)
    elif file_type == 'docx': return parse_docx(content)
    else: raise ValueError(f"Unsupported or unknown file type: {file_type}")

# Policy-aware chunking: Respects paragraph boundaries first
def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
            
        if current_length + len(words) <= chunk_size:
            current_chunk.extend(words)
            current_length += len(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
            if len(words) > chunk_size:
                for i in range(0, len(words), chunk_size - overlap):
                    chunks.append(" ".join(words[i:i + chunk_size]))
                current_chunk = []
                current_length = 0
            else:
                current_chunk = words
                current_length = len(words)
                
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return [Chunk(text=c, index=i, embedding=None) for i, c in enumerate(chunks)]

async def embed_texts_batched(texts: List[str], batch_size: int = 512) -> List[np.ndarray]:
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    client = get_openai_client()
    async def _embed(batch):
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    
    tasks = [asyncio.create_task(_embed(b)) for b in batches]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

async def generate_embeddings_parallel(chunks: List[Chunk]) -> None:
    texts = [c.text for c in chunks]
    embeddings = await embed_texts_batched(texts)
    for c, e in zip(chunks, embeddings):
        norm = np.linalg.norm(e) + 1e-12
        c.embedding = (e / norm).astype(np.float32)

async def refine_query(query: str) -> RefinedQuery:
    prompt = f"""Refine this search query strictly for a legal/policy document. 
Return a JSON object with:
- "refined": A clean, concise version of the core question focusing on specific legal or numerical entities.
- "keywords": An array of 3-6 highly specific terms (e.g., exact limits, specific clause names, unique nouns).

Query: {query}"""

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=250 
        )
        raw_content = response.choices[0].message.content
        
        try:
            result = json.loads(raw_content)
            return RefinedQuery(
                refined=result.get("refined", query),
                keywords=result.get("keywords", [query])
            )
        except json.JSONDecodeError:
            # Fallback
            refined_match = re.search(r'"refined"\s*:\s*"([^"]+)"', raw_content)
            keywords_match = re.findall(r'"([^"]+)"', raw_content.split('"keywords"')[-1]) if '"keywords"' in raw_content else []
            
            refined_val = refined_match.group(1) if refined_match else query
            keywords_val = [k for k in keywords_match if k not in ["keywords", "refined"]]
            
            if not keywords_val:
                keywords_val = [query]
                
            print(f"Recovered structurally malformed query refinement via Regex fallback.")
            return RefinedQuery(refined=refined_val, keywords=keywords_val)

    except Exception as e:
        print(f"Error refining query completely failed: {e}")
        return RefinedQuery(refined=query, keywords=[query])

def create_bm25_index(chunks: List[Chunk]):
    from rank_bm25 import BM25Okapi
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / norms).astype(np.float32)

def create_faiss_index(chunks: List[Chunk]):
    import faiss
    if not chunks or chunks[0].embedding is None: return None
    embs = np.array([c.embedding for c in chunks], dtype=np.float32)
    embs = _l2_normalize_rows(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def hybrid_search(
    query_embedding: np.ndarray,
    lexical_terms: List[str],
    chunks: List[Chunk],
    bm25_index,
    faiss_index,
    top_k: int = TOP_K_CANDIDATES
) -> List[Tuple[int, float]]:
    
    bm25_query = " ".join(lexical_terms).lower().split()
    bm25_scores = bm25_index.get_scores(bm25_query)

    if faiss_index is not None:
        qe = query_embedding.astype(np.float32)
        qe = qe / (np.linalg.norm(qe) + 1e-12)
        sims, idxs = faiss_index.search(qe.reshape(1, -1), min(top_k, len(chunks)))
        
        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        for i, s in zip(idxs[0], sims[0]):
            dense_scores[i] = s

        bmin, bmax = bm25_scores.min(), bm25_scores.max()
        bm25_norm = (bm25_scores - bmin) / (bmax - bmin + 1e-10) if bmax > bmin else np.zeros_like(bm25_scores)

        # Policy weight tune: 55% Lexical (exact matches matter), 45% Dense
        combined = 0.55 * bm25_norm + 0.45 * dense_scores
        top_idx = np.argpartition(-combined, range(min(top_k, len(chunks))))[:top_k]
        top_idx = top_idx[np.argsort(-combined[top_idx])]
        return [(int(i), float(combined[i])) for i in top_idx]

    top_idx = np.argpartition(-bm25_scores, range(min(top_k, len(chunks))))[:top_k]
    top_idx = top_idx[np.argsort(-bm25_scores[top_idx])]
    return [(int(i), float(bm25_scores[i])) for i in top_idx]

def reciprocal_rank_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[int]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def mmr_fast(
    chunks: List[Chunk],
    query_embedding: np.ndarray,
    candidate_indices: List[int],
    top_k: int = FINAL_TOP_K,
    lambda_param: float = 0.5
) -> List[int]:
    if not candidate_indices: return []
    
    E = np.stack([chunks[i].embedding for i in candidate_indices])
    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    rel = E @ q
    
    selected = []
    mask = np.ones(len(candidate_indices), dtype=bool)
    S = E @ E.T

    for _ in range(min(top_k, len(candidate_indices))):
        if not selected:
            j = np.argmax(rel * mask)
        else:
            max_sim = S[:, selected].max(axis=1)
            mmr = lambda_param * rel - (1 - lambda_param) * max_sim
            mmr[~mask] = -1e9
            j = np.argmax(mmr)
        selected.append(j)
        mask[j] = False

    return [candidate_indices[j] for j in selected]

async def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are an expert policy/legal analyst AI. Answer the question strictly using ONLY the provided Context. 

Context:
{context}

Question: {query}

Guidelines:
1) Be exceptionally concise. Give the exact numbers, ages, or conditions immediately.
2) Do not hallucinate. Do not infer outside the text.
3) IMPORTANT: Always state exclusions or strict conditions if the context mentions them (e.g., "Subject to section 4").
4) If the answer is not present, reply exactly: "Not enough information in the provided document." """

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."

async def retrieve_candidates_only(
    q: str, rq: RefinedQuery, qe: np.ndarray,
    chunks: List[Chunk], bm25_index, faiss_index
) -> Dict:
    
    search_queries = [rq.keywords, q.split()]
    
    async def _hybrid_for_terms(terms):
        return await asyncio.to_thread(
            hybrid_search, qe, terms, chunks, bm25_index, faiss_index, TOP_K_CANDIDATES
        )

    tasks = [_hybrid_for_terms(t) for t in search_queries if t]
    all_rankings = await asyncio.gather(*tasks) if tasks else []

    fused_indices = reciprocal_rank_fusion(all_rankings)
    final_indices = mmr_fast(chunks, qe, fused_indices[:100], FINAL_TOP_K)

    passages = [chunks[i].text for i in final_indices]
    return {
        "question": q,
        "refined": rq.refined,
        "indices": final_indices,
        "passages": passages,
    }

async def _process_one_question(
    q: str,
    all_chunks: List[Chunk],
    bm25_index,
    faiss_index,
) -> Dict:
    rq = await refine_query(q)
    qe = (await embed_texts_batched([rq.refined]))[0]
    r = await retrieve_candidates_only(q, rq, qe, all_chunks, bm25_index, faiss_index)
    context = "\n\n".join(r["passages"])
    answer = await generate_answer(q, context)
    return {
        "question": q,
        "answer": answer,
        "refined_query": rq.refined,
        "passages_used": len(r["passages"]),
    }

async def process_single_document(url: str, session: aiohttp.ClientSession) -> Tuple[List[Chunk], any, any]:
    cache_result = await asyncio.to_thread(DocumentCache.load_document_cache, url)
    if cache_result is not None:
        return cache_result

    pdf_bytes = await download_pdf_fast(url, session)
    text = await asyncio.to_thread(parse_document, pdf_bytes, url)
    chunks = await asyncio.to_thread(create_chunks, text)
    
    await generate_embeddings_parallel(chunks)
    
    bm25_index = await asyncio.to_thread(create_bm25_index, chunks)
    faiss_index = await asyncio.to_thread(create_faiss_index, chunks)

    await asyncio.to_thread(DocumentCache.save_document_cache, url, chunks, bm25_index, faiss_index)
    return chunks, bm25_index, faiss_index

async def process_rag_pipeline(document_urls: List[str], questions: List[str]) -> List[Dict]:
    total_start = time.perf_counter()
    
    document_urls = list(dict.fromkeys(document_urls))

    connector = aiohttp.TCPConnector(limit=200, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with Timer("Concurrent Document Processing"):
            doc_tasks = [process_single_document(url, session) for url in document_urls]
            doc_results = await asyncio.gather(*doc_tasks)

        all_chunks = []
        for chunks, _, _ in doc_results:
            all_chunks.extend(chunks)
            
        for i, chunk in enumerate(all_chunks):
            chunk.index = i 

        with Timer("Unified Index Creation"):
            if len(document_urls) == 1:
                _, bm25_index, faiss_index = doc_results[0]
            else:
                bm25_index = await asyncio.to_thread(create_bm25_index, all_chunks)
                faiss_index = await asyncio.to_thread(create_faiss_index, all_chunks)

        with Timer("Per-Question Pipeline"):
            results = list(await asyncio.gather(*[
                _process_one_question(q, all_chunks, bm25_index, faiss_index)
                for q in questions
            ]))
        
        print(f"\nPipeline completed in {time.perf_counter() - total_start:.2f}s")
        return results

async def main():
    pdf_urls = [
        "https://raw.githubusercontent.com/Shaesh-Kuiper/HackRX_Rochinante/main/doc2.pdf",
    ]
    questions = [
        "According to the Definitions section, what is the exact age limit defined for a 'Dependent Child'?",
        "What is the limit of indemnity payable for cleaning out the engine?"
    ]

    print("🚀 STARTING HIGH-PERFORMANCE MULTI-DOC RAG")
    results = await process_rag_pipeline(pdf_urls, questions)
    
    for i, result in enumerate(results, 1):
        print(f"\nQ{i}: {result['question']}")
        print(f"A: {result['answer']}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

__all__ = ['process_rag_pipeline', 'DocumentCache']
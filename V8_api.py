import fitz  
import re
import os
import time
import pickle
import hashlib
import math
import asyncio
import aiohttp
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import faiss
import tiktoken
from rank_bm25 import BM25Okapi
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import threading



EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4.1-mini")         
REFINE_MODEL = os.getenv("REFINE_MODEL", "gpt-4.1-mini")   
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4.1-mini")   


RATE_LIMITS = {
    "default": {"TPM": 2_000_000, "RPM": 5000},
    REFINE_MODEL: {"TPM": 2_000_000, "RPM": 5000},
    RERANK_MODEL: {"TPM": 2_000_000, "RPM": 5000},
    GEN_MODEL: {"TPM": 2_000_000, "RPM": 5000},
    EMBED_MODEL: {"TPM": 2_000_000, "RPM": 5000},
}


CHUNK_TOKENS = 450
CHUNK_OVERLAP = 60

RETR_TOPK = 25  
FINAL_K = 3      
EXPAND_CHARS = 100  


EMBED_BATCH_SIZE = 256      
EMBED_CONCURRENCY = 8     
FAISS_THREADS = min(8, os.cpu_count() or 8)
faiss.omp_set_num_threads(FAISS_THREADS)


QUERY_CONCURRENCY = int(os.getenv("QUERY_CONCURRENCY", "10"))



@dataclass
class Chunk:
    text: str
    page_num: int
    chunk_type: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    clean_text: Optional[str] = None

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def encode(self, text: str):
        return self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens)

tok = Tokenizer()



class RateLimiter:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.window_start = defaultdict(lambda: 0.0)
        self.tokens_used = defaultdict(lambda: 0)
        self.requests_used = defaultdict(lambda: 0)

    async def reserve(self, model: str, est_tokens: int, requests: int = 1):
        limits = RATE_LIMITS.get(model, RATE_LIMITS["default"])
        now = time.time()
        async with self.lock:
            start = self.window_start[model]
            if now - start >= 60.0:
                self.window_start[model] = now
                self.tokens_used[model] = 0
                self.requests_used[model] = 0

            
            def need_wait():
                return (self.tokens_used[model] + est_tokens > limits["TPM"] or
                        self.requests_used[model] + requests > limits["RPM"])

            while need_wait():
                await asyncio.sleep(0.05)
                now = time.time()
                if now - self.window_start[model] >= 60.0:
                    self.window_start[model] = now
                    self.tokens_used[model] = 0
                    self.requests_used[model] = 0

            self.tokens_used[model] += est_tokens
            self.requests_used[model] += requests

rate_limiter = RateLimiter()



class InsuranceDomainAnalyzer:
    def __init__(self):
        self.domain_patterns = {
            'coverage_terms': [
                'coverage','benefits','insured','policy','protection',
                'compensation','claim','payout','sum assured'
            ],
            'exclusions': [
                'excluded','not covered','limitation','restriction',
                'except','provided that','subject to'
            ],
            'medical_conditions': [
                'cataract','diabetes','hypertension','obesity','bariatric',
                'pre-existing','disease','illness','surgery','treatment'
            ],
            'financial_terms': [
                'premium','deductible','co-pay','room rent','sub-limit',
                'waiting period','grace period','sum insured'
            ],
            'procedures': [
                'bone marrow','transplant','opd','day-care','hospitalization',
                'surgery','treatment','procedure'
            ]
        }

    def analyze_chunk(self, text: str) -> Dict[str, any]:
        tl = text.lower()
        scores = {}
        for cat, terms in self.domain_patterns.items():
            scores[cat] = sum(1 for t in terms if t in tl)
        money_matches = len(re.findall(r'₹[\d,]+|rs\.?\s*[\d,]+|\d+%', tl))
        time_matches = len(re.findall(r'\d+\s*(?:days?|months?|years?)', tl))
        return {
            'domain_scores': scores,
            'has_numbers': money_matches + time_matches > 0,
            'relevance_score': sum(scores.values()) + (money_matches + time_matches) * 0.5
        }



class TextProcessor:
    def __init__(self):
        self.enc = tok

    def clean_text_once(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
        text = re.sub(r'\s*[;:,]\s*', lambda m: m.group(0).strip() + ' ', text)
        return text.strip()

    def chunk_by_tokens(self, text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP) -> List[str]:
        tokens = self.enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            chunk_text = self.enc.decode(tokens[start:end])
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = end - overlap_tokens
        return chunks



class DeduplicationEngine:
    def __init__(self):
        self.seen_hashes = set()

    def deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        start = time.time()
        unique, exact_dupes, low_value = [], 0, 0
        for c in chunks:
            tn = re.sub(r'\s+', ' ', (c.clean_text or c.text).lower().strip())
            h = hashlib.md5(tn.encode()).hexdigest()
            if h in self.seen_hashes:
                exact_dupes += 1
                continue
            words = tn.split()
            if len(words) < 40 and not re.search(r'\d', tn):
                low_value += 1
                continue
            if len(words) > 0 and len(set(words)) / len(words) < 0.2:
                low_value += 1
                continue
            self.seen_hashes.add(h)
            unique.append(c)
        print(f"🔍 Deduplication: {len(chunks)} → {len(unique)} | exact={exact_dupes} low={low_value} | {time.time()-start:.3f}s")
        return unique



class OptimizedEmbedder:
    def __init__(self, api_key: str, model: str = EMBED_MODEL):
        self.api_key = api_key
        self.model = model
        self.sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        self.timeout = aiohttp.ClientTimeout(total=60, connect=10)
        
        self.connector = aiohttp.TCPConnector(limit=128, limit_per_host=64, ttl_dns_cache=300)
        self._session: Optional[aiohttp.ClientSession] = None
        self._sess_lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._sess_lock:
            if self._session is None or self._session.closed:
              
                self._session = aiohttp.ClientSession(
                    timeout=self.timeout,
                    connector=self.connector
                )
            return self._session

    async def aclose(self):
        async with self._sess_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            
            if not self.connector.closed:
                self.connector.close()

    async def _safe_post(self, url: str, headers: Dict, json_payload: Dict) -> Dict:
       
        for attempt in range(3):
            session = await self._ensure_session()
            try:
                async with self.sem:
                    async with session.post(url, headers=headers, json=json_payload) as r:
                        r.raise_for_status()
                        return await r.json()
            except Exception as e:
                
                if attempt < 2:
                    await asyncio.sleep(0.3 * (2 ** attempt))
                    
                    async with self._sess_lock:
                        if self._session and not self._session.closed:
                            await self._session.close()
                        self._session = None
                    continue
                print(f"❌ Embedding error after retries: {e}")
                raise

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        data = {"model": self.model, "input": texts}
        est_toks = sum(tok.count(t) for t in texts) + 100
        await rate_limiter.reserve(self.model, est_toks, requests=1)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            js = await self._safe_post("https://api.openai.com/v1/embeddings", headers, data)
            return [e["embedding"] for e in js["data"]]
        except Exception:
            
            dim = 3072 if "large" in self.model else 1536
            return [np.zeros(dim).tolist() for _ in texts]

    async def embed_chunks_async(self, chunks: List[Chunk]) -> List[Chunk]:
        start = time.time()
        to_embed = [(i, (c.clean_text or c.text)[:8000]) for i, c in enumerate(chunks)]
        if not to_embed:
            print("✅ No chunks to embed.")
            return chunks

    
        batches = []
        for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
            b = to_embed[i:i + EMBED_BATCH_SIZE]
            batch_idx = [idx for idx, _ in b]
            batch_txt = [txt for _, txt in b]
            batches.append((batch_idx, batch_txt))

        
        results: List[Optional[List[float]]] = [None] * len(to_embed)

        async def run_one(batch_idx, batch_txt):
            embs = await self.embed_batch_async(batch_txt)  
            for i, e in zip(batch_idx, embs):
                results[i] = e

      
        await asyncio.gather(*(run_one(bi, bt) for bi, bt in batches))

        
        for i, emb in enumerate(results):
            if emb is None:
                
                dim = 3072 if "large" in self.model else 1536
                emb = [0.0] * dim
            chunks[i].embedding = np.array(emb, dtype=np.float32)

        print(f"⚡ Embedded {len(to_embed)} chunks in {time.time()-start:.3f}s")
        return chunks

    async def embed_query_async(self, query: str) -> np.ndarray:
        embs = await self.embed_batch_async([query])
        return np.array(embs[0], dtype=np.float32)



class OpenAIChatHTTP:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.connector = aiohttp.TCPConnector(limit=64, limit_per_host=16, ttl_dns_cache=300)
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=self.timeout, connector=self.connector)
            return self._session

    async def aclose(self):
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
            if not self.connector.closed:
                self.connector.close()

    async def chat(self, model: str, messages: List[Dict], temperature: float = 0,
                   max_tokens: Optional[int] = None, response_format: Optional[Dict] = None) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

       
        est_tokens = sum(tok.count(m.get("content", "")) for m in messages) + (max_tokens or 128)
        await rate_limiter.reserve(model, est_tokens, requests=1)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for attempt in range(3):
            sess = await self._ensure_session()
            try:
                async with sess.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=payload) as r:
                    r.raise_for_status()
                    js = await r.json()
                    return js["choices"][0]["message"]["content"]
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(0.25 * (2 ** attempt))
                    
                    async with self._lock:
                        if self._session and not self._session.closed:
                            await self._session.close()
                        self._session = None
                    continue
                raise



class HybridVectorStore:
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []
        self.bm25 = None
        self.tokenized_chunks = []
        self._lock = threading.Lock()

    def add_chunks(self, chunks: List[Chunk]):
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            return
        with self._lock:
            embs = np.vstack([c.embedding for c in valid]).astype('float32')
            faiss.normalize_L2(embs)
            self.index.add(embs)
            self.chunks.extend(valid)
            self.tokenized_chunks = [re.findall(r"[a-zA-Z0-9]+", (c.clean_text or c.text).lower()) for c in valid]
            self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"🗂️ Built hybrid index over {len(valid)} chunks")

    def hybrid_search(self, query_embedding: np.ndarray, query_text: str, k: int = RETR_TOPK) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        q = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(q)
        
        mult = 2
        vv_scores, vv_idx = self.index.search(q, min(k*mult, len(self.chunks)))

        q_tokens = re.findall(r"[a-zA-Z0-9]+", query_text.lower())
        bm25_scores = self.bm25.get_scores(q_tokens)

        combined = {}
        for idx, score in zip(vv_idx[0], vv_scores[0]):
            if idx < len(self.chunks):
                combined[idx] = {'v': float(score), 'b': 0.0}
        for idx, score in enumerate(bm25_scores):
            if idx not in combined:
                combined[idx] = {'v': 0.0, 'b': float(score)}
            else:
                combined[idx]['b'] = float(score)

        if not combined:
            return []
        max_v = max(s['v'] for s in combined.values()) or 1.0
        max_b = max(s['b'] for s in combined.values()) or 1.0

        fused = []
        for idx, s in combined.items():
            nv = s['v'] / max_v
            nb = s['b'] / max_b
            score = 0.6 * nv + 0.4 * nb
            fused.append((self.chunks[idx], score))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]


class QueryRefiner:
    def __init__(self, chat_http: Optional[OpenAIChatHTTP], model: str = REFINE_MODEL):
        self.chat_http = chat_http
        self.model = model

    async def refine(self, original_q: str) -> str:
        if not self.chat_http:
            return original_q.lower().strip()
        
        sys = "You rewrite insurance questions into compact retrieval queries. Output only the rewritten query."
        usr = f"User: {original_q}\nRefined:"
        txt = await self.chat_http.chat(self.model,
                                        [{"role":"system","content":sys},{"role":"user","content":usr}],
                                        temperature=0, max_tokens=48)
        return re.sub(r'\s+', ' ', txt).strip()



class LLMReranker:
    def __init__(self, chat_http: Optional[OpenAIChatHTTP], model: str = RERANK_MODEL, max_snippet_tokens: int = 120):
        self.chat_http = chat_http
        self.model = model
        self.max_snippet_tokens = max_snippet_tokens

    def _trim_to_tokens(self, text: str, limit: int) -> str:
        ids = tok.encode(text)
        if len(ids) <= limit: return text
        return tok.decode(ids[:limit])

    async def rerank(self, question: str, candidates: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        if not self.chat_http or not candidates:
            return candidates[:FINAL_K]

        payload_items = []
        for i, (chunk, base) in enumerate(candidates):
            snippet = self._trim_to_tokens(chunk.clean_text or chunk.text, self.max_snippet_tokens)
            payload_items.append({"id": i, "base": round(float(base), 4), "page": chunk.page_num, "text": snippet})

       
        sys = "Respond ONLY with JSON array of objects {\"id\":int,\"score\":0-100}."
        usr = json.dumps({"q": question, "snippets": payload_items}, ensure_ascii=False)

        txt = await self.chat_http.chat(self.model,
                                        [{"role":"system","content":sys},{"role":"user","content":usr}],
                                        temperature=0,
                                        max_tokens=256,
                                        response_format={"type":"json_object"})
        try:
            
            data = json.loads(txt)
            if isinstance(data, dict) and "scores" in data: data = data["scores"]
            score_map = {int(it["id"]): float(it["score"]) for it in data}
        except Exception:
            
            score_map = {i: base for i, (_, base) in enumerate(candidates)}

        reranked = []
        for i, (chunk, base) in enumerate(candidates):
            reranked.append((chunk, 0.85*base + 0.15*score_map.get(i, base)))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:FINAL_K]



class OptimizedRAGParser:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.chat_http = OpenAIChatHTTP(self.api_key) if self.api_key else None
        self.text_processor = TextProcessor()
        self.domain_analyzer = InsuranceDomainAnalyzer()
        self.embedder = OptimizedEmbedder(self.api_key) if self.api_key else None
        self.dedup_engine = DeduplicationEngine()
        self.timing_logs: Dict[str, float] = {}

    def extract_pages_fast(self, pdf_path: str) -> Dict[int, str]:
        start = time.time()
        pages = {}
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                pages[i] = page.get_text("text")
        self.timing_logs['pdf_extract'] = time.time() - start
        print(f"📄 Extracted {len(pages)} pages in {self.timing_logs['pdf_extract']:.3f}s")
        return pages

    def strip_repeated_lines(self, pages: Dict[int, str], min_len: int = 6, freq_thresh: float = 0.3) -> Dict[int, str]:
        line_counts = Counter()
        per_page = {}
        for p, txt in pages.items():
            lines = [l.strip() for l in txt.splitlines() if len(l.strip()) >= min_len]
            per_page[p] = lines
            line_counts.update(set(lines))
        cutoff = int(len(pages) * freq_thresh)
        boiler = {l for l, c in line_counts.items() if c >= cutoff}
        cleaned = {}
        for p, lines in per_page.items():
            kept = [l for l in lines if l not in boiler]
            cleaned[p] = "\n".join(kept)
        print(f"🧹 Removed {len(boiler)} boilerplate patterns")
        return cleaned

    def _page_to_chunks(self, args):
        page_num, text, pdf_path = args
        if not text.strip():
            return []
        tp = self.text_processor
        clean_text = tp.clean_text_once(text)
        chunk_texts = tp.chunk_by_tokens(clean_text)
        out = []
        for chunk_text in chunk_texts:
            analysis = self.domain_analyzer.analyze_chunk(chunk_text)
            out.append(Chunk(
                text=chunk_text,
                clean_text=chunk_text,
                page_num=page_num + 1,
                chunk_type="insurance_content",
                metadata={
                    'word_count': len(chunk_text.split()),
                    'char_length': len(chunk_text),
                    'domain_analysis': analysis,
                    'page_source': pdf_path
                }
            ))
        return out

    def parse_pdf_optimized(self, pdf_path: str, max_workers: int = max(4, (os.cpu_count() or 8) // 2)) -> Tuple[List[Chunk], Dict[int, str]]:
        total = time.time()
        pages = self.extract_pages_fast(pdf_path)
        pages = self.strip_repeated_lines(pages)
        chunk_start = time.time()

        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            all_chunks_lists = list(ex.map(self._page_to_chunks, [(p, txt, pdf_path) for p, txt in pages.items()]))

        chunks = [c for sub in all_chunks_lists for c in sub]
        self.timing_logs['chunking'] = time.time() - chunk_start

        dedup_start = time.time()
        chunks = self.dedup_engine.deduplicate_chunks(chunks)
        self.timing_logs['deduplication'] = time.time() - dedup_start

        self.timing_logs['total_parsing'] = time.time() - total
        print(f"📊 Final result: {len(chunks)} unique chunks | parse {self.timing_logs['total_parsing']:.3f}s")
        return chunks, pages  

class OptimizedRetriever:
    def __init__(self, parser: OptimizedRAGParser):
        self.parser = parser
        self.vector_store = HybridVectorStore()
        self.chunks_by_page: Dict[Tuple[str, int], List[Chunk]] = defaultdict(list)
        self.pages_fulltext: Dict[Tuple[str, int], str] = {}

    async def index_documents(self, pdf_paths: List[str]) -> List[Chunk]:
        all_chunks = []
        pages_map = {}

        for path in pdf_paths:
            print(f"📄 Processing: {path}")
            chunks, pages = self.parser.parse_pdf_optimized(path)
            all_chunks.extend(chunks)
            for pnum, ptxt in pages.items():
                self.pages_fulltext[(path, pnum + 1)] = ptxt

        
        for c in all_chunks:
            key = (c.metadata.get('page_source', ''), c.page_num)
            self.chunks_by_page[key].append(c)

        if self.parser.embedder:
            embed_start = time.time()
            all_chunks = await self.parser.embedder.embed_chunks_async(all_chunks)
            self.parser.timing_logs['embedding'] = time.time() - embed_start

        
        idx_start = time.time()
        self.vector_store.add_chunks(all_chunks)
        self.parser.timing_logs['indexing'] = time.time() - idx_start
        return all_chunks

    def expand_by_chars(self, chunk: Chunk, pad: int = EXPAND_CHARS) -> str:
        src = chunk.metadata.get('page_source', '')
        key = (src, chunk.page_num)
        page_text = self.pages_fulltext.get(key)
        if not page_text:
            return chunk.clean_text or chunk.text

        
        txt = chunk.clean_text or chunk.text
        idx = page_text.find(txt)
        if idx == -1:
            return txt
        start = max(0, idx - pad)
        end = min(len(page_text), idx + len(txt) + pad)
        return page_text[start:end]

    async def retrieve(self, refined_query: str, original_query: str, k: int = RETR_TOPK) -> List[Tuple[Chunk, float]]:
        if not self.parser.embedder:
            
            return []
        qemb = await self.parser.embedder.embed_query_async(refined_query)
        res = self.vector_store.hybrid_search(qemb, refined_query, k=k)
        return res


class Answerer:
    def __init__(self, chat_http: Optional[OpenAIChatHTTP], model: str = GEN_MODEL):
        self.chat_http = chat_http
        self.model = model

    def _pack_contexts(self, contexts: List[str], max_total_tokens: int = 900) -> List[str]:
        packed, total = [], 0
        for ctx in contexts:
            ids = tok.encode(ctx)
            if total + len(ids) > max_total_tokens:
                
                remain = max_total_tokens - total
                if remain <= 0: break
                packed.append(tok.decode(ids[:remain]))
                break
            packed.append(ctx); total += len(ids)
        return packed

    async def generate(self, question: str, contexts: List[str]) -> str:
        if not self.chat_http:
            return "No API key provided for generation."
        contexts = self._pack_contexts(contexts, max_total_tokens=900)  
        ctx_text = "\n\n---\n\n".join([f"{c}" for c in contexts])

        sys = "You are an expert insurance policy analyst. Answer ONLY from context. If not present, say 'Not specified in the provided documents'."
        usr = f"Context:\n{ctx_text}\n\nQuestion: {question}\n\nAnswer concisely with exact amounts/durations/conditions."

        return await self.chat_http.chat(self.model,
                                         [{"role":"system","content":sys},{"role":"user","content":usr}],
                                         temperature=0,
                                         max_tokens=220)



async def _process_one_query(original_q: str, refiner, retriever, reranker, answerer):
    refined_q = await refiner.refine(original_q)
    candidates = await retriever.retrieve(refined_q, original_q, k=RETR_TOPK)
    reranked = await reranker.rerank(original_q, candidates)
    expanded = [retriever.expand_by_chars(ch, EXPAND_CHARS) for ch, _ in reranked]
    answer = await answerer.generate(original_q, expanded)
    return {
        "query": original_q,
        "refined_query": refined_q,
        "answer": answer,
        "contexts": [(c[:200] + "...") if len(c) > 200 else c for c in expanded]
    }

async def run_rag_pipeline_v8(pdf_paths: List[str], queries: List[str], api_key: Optional[str] = None):
    """
    Full pipeline:
     1) Parse + index (parallel)
     2) Process all queries concurrently with semaphore gate
    """
    print("🚀 OPTIMIZED RAG PIPELINE v8.2")
    print("="*80)
    t0 = time.time()

    parser = OptimizedRAGParser(api_key)
    retriever = OptimizedRetriever(parser)
    refiner = QueryRefiner(parser.chat_http, REFINE_MODEL)
    reranker = LLMReranker(parser.chat_http, RERANK_MODEL)
    answerer = Answerer(parser.chat_http, GEN_MODEL)

   
    print(f"📚 Indexing {len(pdf_paths)} document(s)...")
    _ = await retriever.index_documents(pdf_paths)

  
    sem = asyncio.Semaphore(QUERY_CONCURRENCY)
    async def _guard(q):
        async with sem:
            return await _process_one_query(q, refiner, retriever, reranker, answerer)

    print(f"\n� Processing {len(queries)} queries concurrently (limit={QUERY_CONCURRENCY})...")
    tasks = [asyncio.create_task(_guard(q)) for q in queries]
    results = await asyncio.gather(*tasks)

    total = time.time() - t0
    print(f"\n⏱️ TIMING SUMMARY:")
    print("="*50)
    for stage, timing in parser.timing_logs.items():
        print(f"{stage:<20}: {timing:.3f}s")
    print(f"{'total_pipeline':<20}: {total:.3f}s")
    
    
    if parser.embedder:
        await parser.embedder.aclose()
    if parser.chat_http:
        await parser.chat_http.aclose()
    
    print(f"✅ Done in {total:.2f}s")
    return results



if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    pdf_paths = ["NP.pdf"]  
    queries = [
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
    if not api_key:
        print("⚠️ Set OPENAI_API_KEY environment variable")
    try:
        results = asyncio.run(run_rag_pipeline_v8(pdf_paths, queries, api_key))
        with open(f"rag_results_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n✅ Pipeline completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()

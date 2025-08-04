import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
import os
from collections import deque, Counter
import hashlib
import faiss
import pickle
from pathlib import Path
import pathlib
import asyncio
import aiohttp
import json
import math
import email
import threading
# Enable multi-threading for FAISS
faiss.omp_set_num_threads(min(8, os.cpu_count()))
import html2text
import threading
import traceback
from docx import Document
from utils_io import download_blob, async_download_blob
import net_utils

@dataclass
class Chunk:
    text: str
    page_num: int
    chunk_type: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    clean_text: Optional[str] = None
    term_frequencies: Optional[Dict[str, int]] = None  # New: term frequencies for IDF

class DocumentTermAnalyzer:
    """Analyzes document-wide term frequencies for better relevance scoring"""
    
    def __init__(self):
        self.term_frequencies: Dict[str, int] = {}
        self.document_frequency: Dict[str, int] = {}  # How many chunks contain each term
        self.total_chunks = 0
        self.entity_patterns = {
            'transport_types': {
                'train': ['train', 'railway', 'railroad', 'locomotive', 'rail transport'],
                'flight': ['flight', 'airplane', 'aircraft', 'aviation', 'air travel', 'plane'],
                'bus': ['bus', 'coach', 'motor coach'],
                'ship': ['ship', 'vessel', 'cruise', 'ferry', 'boat'],
                'car': ['car', 'automobile', 'vehicle', 'auto']
            },
            'delay_contexts': {
                'mechanical': ['mechanical', 'breakdown', 'malfunction', 'technical'],
                'weather': ['weather', 'storm', 'fog', 'snow', 'rain'],
                'operational': ['schedule', 'timetable', 'delay', 'late', 'postponed']
            }
        }
    
    def analyze_document(self, chunks: List[Chunk]):
        """Analyze document to build term importance maps - PARALLELIZED"""
        self.total_chunks = len(chunks)
        
        # Process chunks in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        def process_chunk(chunk):
            text = (chunk.clean_text or chunk.text).lower()
            terms = self._extract_terms(text)
            chunk_terms = Counter(terms)
            chunk.term_frequencies = dict(chunk_terms)
            return chunk_terms, set(terms)
        
        # Process all chunks in parallel
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() * 2)) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        # Aggregate results
        for chunk_terms, unique_terms in results:
            for term, freq in chunk_terms.items():
                self.term_frequencies[term] = self.term_frequencies.get(term, 0) + freq
            for term in unique_terms:
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
        
        print(f"📊 Analyzed {len(self.term_frequencies)} unique terms across {self.total_chunks} chunks")
        
        # Print some rare terms for debugging
        rare_terms = {term: freq for term, freq in self.term_frequencies.items() 
                     if freq <= 5 and len(term) > 3}
        if rare_terms:
            print(f"🔍 Sample rare terms: {list(rare_terms.keys())[:10]}")
    
    def _extract_terms(self, text: str) -> List[str]:
        """Enhanced term extraction preserving important phrases"""
        terms = []
        
        # Extract single words (filtered)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        words = [w.lower() for w in words if len(w) > 2]
        terms.extend(words)
        
        # Extract important phrases
        important_phrases = [
            'air travel', 'rail transport', 'motor coach', 'cruise ship',
            'flight delay', 'train delay', 'mechanical failure',
            'weather delay', 'schedule change', 'trip cancellation'
        ]
        
        for phrase in important_phrases:
            if phrase in text:
                # Add both the phrase and its components
                terms.append(phrase.replace(' ', '_'))
        
        return terms
    
    def get_term_importance(self, term: str) -> float:
        """Calculate term importance using TF-IDF concepts"""
        if term not in self.document_frequency:
            return 0.0
        
        # Calculate IDF (inverse document frequency)
        df = self.document_frequency[term]
        idf = math.log(self.total_chunks / df) if df > 0 else 0
        
        # Boost very rare terms (appearing in <5% of chunks)
        rarity_boost = 1.0
        if df / self.total_chunks < 0.05:
            rarity_boost = 2.0
        if df / self.total_chunks < 0.02:  # Very rare
            rarity_boost = 3.0
        
        return idf * rarity_boost
    
    def identify_query_entities(self, query: str) -> Dict[str, List[str]]:
        """Identify specific entities in the query"""
        query_lower = query.lower()
        found_entities = {}
        
        for category, entity_types in self.entity_patterns.items():
            found_entities[category] = []
            for entity_type, terms in entity_types.items():
                for term in terms:
                    if term in query_lower:
                        found_entities[category].append(entity_type)
                        break
        
        return found_entities

class AnswerGenerator:
    """Generate final answers using GPT-4.1 nano with few-shot examples"""
    
    def __init__(self, api_key: str, max_concurrent: int = 4500):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.sem = asyncio.Semaphore(max_concurrent)
        
        # Few-shot examples for better answer generation
        self.few_shot_examples = [
            {
                "question": "What is the grace period for premium payment under the SBI student loan Policy?",
                "context": "The policy ...(lots of text)...available.",
                "answer": "A grace period of ten (10) days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.[pg 32, sec3.2.1, SBI_policy.pdf]"
            },
            {
                "question": "What is the waiting period for Organ Failure (OrgF) to be covered?",
                "context": "Pre-existing...(Lots of text)...date.",
                "answer": "There is a waiting period of 12 (12) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.[pg 49, Clause 3.2, disease_cov.DOCX]"
            }
        ]
    
    async def generate_answer_async(self, question: str, retrieved_chunks: List[Tuple[Chunk, float]], 
                                   session: aiohttp.ClientSession) -> str:
        """Generate answer using GPT-4.1 nano with retrieved context"""
        if not self.client:
            return "Unable to generate answer - no API key provided"
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks[:3]):  # Use top 3 chunks
            context_parts.append(f"Context {i+1} (Relevance: {score:.3f}):\n{chunk.clean_text or chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Build few-shot prompt
        few_shot_text = ""
        for example in self.few_shot_examples:
            few_shot_text += f"""
Question: {example['question']}
Context: {example['context']}
Answer: {example['answer']}

"""
        
        answer_prompt = f"""You are an expert insurance policy analyst. Answer questions accurately based on the provided context from policy documents.

Here are some examples of how to answer:(note you're final answer must be short and precise, yes or no , the reason , the source in few words)

{few_shot_text}

Now answer this question:

Question: {question}

Context: {context}

Instructions:
1. answer like a real human expert // (don't Answer : ..)
2. Answer based ONLY on the provided context
3. Be specific and include relevant details (amounts, time periods, conditions)
4. Include specific policy terms and conditions when relevant
5. Keep the answer precise, short ,concise but complete and dont use abbreviation
Answer:"""

        try:
            headers = {
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4.1-mini",  # Using mini for cost-effectiveness
                "messages": [{"role": "user", "content": answer_prompt}],
                "max_tokens": 400,
                "temperature": 0  # Low temperature for factual accuracy
            }
            
            async with self.sem:
                async with session.post("https://api.openai.com/v1/chat/completions", 
                                      headers=headers, json=data) as response:
                    result = await response.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    return answer
                
        except Exception as e:
            print(f"Answer generation error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def generate_answer(self, question: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> str:
        """Synchronous wrapper for answer generation"""
        if not self.client:
            return "Unable to generate answer - no API key provided"
            
        async def _generate():
            return await self.generate_answer_async(question, retrieved_chunks, net_utils.shared_session)
        
        return asyncio.run(_generate())

class UltraFastVectorStore:
    """In-memory vector store using FAISS - faster than Qdrant for small-medium datasets"""
    
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        # Using IndexFlatIP for maximum speed (inner product)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []
        self.is_normalized = False
        self._lock = threading.Lock()
        
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks with embeddings to the store"""
        # Filter chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            return
        
        with self._lock:
            # Stack embeddings
            embeddings = np.vstack([c.embedding for c in valid_chunks]).astype('float32')
            
            # Normalize for cosine similarity (FAISS uses inner product)
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            self.chunks.extend(valid_chunks)
            self.is_normalized = True
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Ultra-fast similarity search"""
        if len(self.chunks) == 0:
            return []
        
        # Prepare query
        query_vec = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        # Search
        scores, indices = self.index.search(query_vec, min(k, len(self.chunks)))
        
        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save index and chunks"""
        save_data = {
            'chunks': self.chunks,
            'dimension': self.dimension
        }
        # Save chunks
        with open(f"{path}_chunks.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_index.faiss")
    
    def load(self, path: str):
        """Load index and chunks"""
        # Load chunks
        with open(f"{path}_chunks.pkl", 'rb') as f:
            save_data = pickle.load(f)
            self.chunks = save_data['chunks']
            self.dimension = save_data['dimension']
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")

class AsyncEmbedder:
    """Async embeddings for 2-3x faster processing"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", max_concurrent: int = 4500):
        # Optimal concurrency for OpenAI rate limits
        # 100 concurrent requests is more realistic than 4800
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.sem = asyncio.Semaphore(max_concurrent)
        
    async def embed_batch_async(self, texts: List[str], session: aiohttp.ClientSession) -> List[List[float]]:
        """Embed a batch of texts asynchronously"""
        data = {
            "model": self.model,
            "input": texts
        }
        
        try:
            async with self.sem:
                async with session.post(self.base_url, headers=self.headers, json=data) as response:
                    result = await response.json()
                    return [e["embedding"] for e in result["data"]]
        except Exception as e:
            print(f"Async embedding error: {e}")
            dimension = 3072 if "large" in self.model else 1536
            return [np.zeros(dimension).tolist() for _ in texts]
    
    async def embed_all_chunks_async(self, chunks: List[Chunk], session: aiohttp.ClientSession = None) -> List[Chunk]:
        """Embed all chunks using async requests - much faster"""
        start_time = time.time()
        
        # Use clean_text for embedding (better quality)
        texts = [chunk.clean_text[:8000] if chunk.clean_text else chunk.text[:8000] for chunk in chunks]
        
        # Optimized batch size for large embeddings
        batch_size = 100
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Use provided session or create new one
        if session:
            tasks = []
            for batch in batches:
                task = self.embed_batch_async(batch, session)
                tasks.append(task)
            
            # Execute all tasks concurrently
            all_embeddings = await asyncio.gather(*tasks)
        else:
            # Use shared session when none provided
            session = net_utils.shared_session
            tasks = []
            for batch in batches:
                task = self.embed_batch_async(batch, session)
                tasks.append(task)
            
            # Execute all tasks concurrently
            all_embeddings = await asyncio.gather(*tasks)
        
        # Flatten results
        embeddings = []
        for batch_embeddings in all_embeddings:
            embeddings.extend(batch_embeddings)
        
        # Assign embeddings to chunks
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                chunk.embedding = np.array(embeddings[i])
        
        embed_time = time.time() - start_time
        print(f"🚀 Async embedded {len(chunks)} chunks in {embed_time:.3f}s ({len(chunks)/embed_time:.1f} chunks/sec)")
        
        return chunks

class EnhancedTextProcessor:
    """Enhanced text processing for better retrieval quality"""
    
    def __init__(self):
        # Enhanced legal term patterns
        self.legal_terms = {
            'obligations': ['shall', 'must', 'required', 'obligation', 'duty', 'responsible'],
            'prohibitions': ['prohibited', 'not permitted', 'forbidden', 'restricted'],
            'conditions': ['if', 'when', 'unless', 'provided that', 'subject to', 'in case'],
            'consequences': ['penalty', 'fine', 'termination', 'breach', 'default', 'violation'],
            'coverage': ['covered', 'insured', 'benefits', 'coverage', 'protection', 'compensation'],
            'exclusions': ['excluded', 'not covered', 'except', 'limitation', 'restriction']
        }
        
        # Targeted synonym mapping - more conservative
        self.targeted_synonyms = {
            'death': ['deceased', 'demise', 'passing', 'fatality'],
            'accident': ['incident', 'mishap', 'occurrence', 'event'],
            'cover': ['coverage', 'protection', 'benefit'],
            'insured': ['policyholder', 'beneficiary', 'covered person'],
            # Removed 'transport' synonyms to avoid confusion
        }
        
        # Transport-specific terms for better entity recognition
        self.transport_terms = {
            'train': ['train', 'railway', 'railroad', 'rail', 'locomotive'],
            'flight': ['flight', 'airplane', 'aircraft', 'plane', 'aviation', 'air'],
            'bus': ['bus', 'coach', 'motor coach'],
            'ship': ['ship', 'vessel', 'cruise', 'ferry', 'boat'],
            'car': ['car', 'automobile', 'vehicle', 'auto']
        }
        
        # Stopwords to remove for better matching
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Pre-compile frequently used patterns
        self._word_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')
        self._camelcase_pattern = re.compile(r'([a-z])([A-Z])')
        self._number_letter_pattern = re.compile(r'(\d+)([A-Za-z])')
        self._letter_number_pattern = re.compile(r'([A-Za-z])(\d+)')
        self._transport_pattern = re.compile(
            r'\b(?:train|flight|bus|ship|car|vehicle|transport)\b', 
            re.IGNORECASE
        )
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning for better embedding quality"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = self._camelcase_pattern.sub(r'\1 \2', text)  # Split camelCase
        text = self._number_letter_pattern.sub(r'\1 \2', text)  # Split number+letter
        text = self._letter_number_pattern.sub(r'\1 \2', text)  # Split letter+number
        
        # Normalize legal punctuation
        text = re.sub(r'\s*;\s*', '; ', text)
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_semantic_keywords(self, text: str) -> List[str]:
        """Extract semantically important keywords"""
        text_lower = text.lower()
        keywords = []
        
        # Extract legal concepts
        for category, terms in self.legal_terms.items():
            for term in terms:
                if term in text_lower:
                    keywords.append(f"{category}_{term}")
        
        # Extract transport-specific terms
        for transport_type, terms in self.transport_terms.items():
            for term in terms:
                if term in text_lower:
                    keywords.append(f"transport_{transport_type}")
                    break  # Only add once per transport type
        
        # Extract named entities (improved)
        words = text.split()
        named_entities = []
        for i, word in enumerate(words):
            # Skip if it's the first word or follows a period
            if i == 0 or (i > 0 and words[i-1].endswith('.')):
                continue
            # Check for capitalized words
            if re.match(r'^[A-Z][a-z]+$', word):
                # Check for multi-word entities
                entity = word
                j = i + 1
                while j < len(words) and re.match(r'^[A-Z][a-z]+$', words[j]):
                    entity += ' ' + words[j]
                    j += 1
                named_entities.append(entity)
        keywords.extend(named_entities[:5])  # Limit to avoid noise
        
        # Extract important phrases
        important_phrases = re.findall(r'\b(?:in case of|subject to|provided that|except for|including but not limited to)\b[^.]*', text, re.IGNORECASE)
        keywords.extend([phrase.strip() for phrase in important_phrases[:3]])
        
        return list(set(keywords))
    
    def enhanced_query_analysis(self, query: str) -> Dict[str, any]:
        """Enhanced query analysis for better matching"""
        query_lower = query.lower()
        
        analysis = {
            'key_entities': [],
            'intent_type': 'general',
            'critical_terms': [],
            'transport_type': None,
            'context_terms': []
        }
        
        # Identify transport type
        for transport_type, terms in self.transport_terms.items():
            if any(term in query_lower for term in terms):
                analysis['transport_type'] = transport_type
                analysis['critical_terms'].extend([t for t in terms if t in query_lower])
                break
        
        # Identify intent
        if any(word in query_lower for word in ['cover', 'coverage', 'insured', 'benefit']):
            analysis['intent_type'] = 'coverage_inquiry'
        elif any(word in query_lower for word in ['exclude', 'not covered', 'limitation']):
            analysis['intent_type'] = 'exclusion_inquiry'
        elif any(word in query_lower for word in ['how', 'what', 'when', 'where']):
            analysis['intent_type'] = 'information_request'
        
        # Extract critical terms (non-stopwords, length > 2)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        analysis['critical_terms'].extend([w for w in words if w not in self.stopwords])
        
        return analysis
    
    def expand_query_conservatively(self, query: str, query_analysis: Dict) -> str:
        """Conservative query expansion to avoid noise"""
        words = query.lower().split()
        expanded_words = list(words)  # Start with original words
        
        # Only add synonyms for non-transport terms to avoid confusion
        for word in words:
            if word in self.targeted_synonyms and query_analysis['transport_type'] is None:
                # Only add one synonym to avoid noise
                expanded_words.append(self.targeted_synonyms[word][0])
        
        return ' '.join(expanded_words)

class UltraFastRAGParser:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, api_key: str = None):
        self.chunk_size = chunk_size
        self.overlap = overlap  # Increased overlap for better context
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        embedding_model = "text-embedding-3-large"
        self.async_embedder = AsyncEmbedder(self.api_key, embedding_model) if self.api_key else None
        self.text_processor = EnhancedTextProcessor()
        self.term_analyzer = DocumentTermAnalyzer()
        
        # Enhanced patterns for better chunking
        self.section_pattern = re.compile(
            r'^(?:(?:SECTION|Section|SEC\.|ARTICLE|Article|ART\.|CHAPTER|Chapter|PART|Part|§)\s*[\d\.]+|'
            r'\d+\.\s+[A-Z][A-Z\s]{2,}|'
            r'[IVXLCDM]+\.\s+[A-Z])',
            re.MULTILINE
        )
        self.subsection_pattern = re.compile(r'^(?:\d+\.\d+|\([a-zA-Z0-9]+\))\s+', re.MULTILINE)
        self.legal_markers = re.compile(
            r'\b(?:shall|must|may|prohibited|required|pursuant|notwithstanding|whereas|hereby|'
            r'provided that|subject to|in accordance with|obligations?|liabilit(?:y|ies)|'
            r'warrant(?:y|ies)|indemnif(?:y|ication)|terminat(?:e|ion)|breach|coverage|insured|'
            r'benefits?|compensation|excluded?|limitation|train|flight|bus|ship|car|vehicle)\b',
            re.IGNORECASE
        )
        
    def _batch_clean_texts(self, texts: List[str]) -> List[str]:
        """Clean multiple texts in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() * 2)) as executor:
            return list(executor.map(self.text_processor.clean_text, texts))
        
    def extract_pages_parallel(self, pdf_path: str) -> Dict[int, str]:
        """Extract all pages in parallel - optimized"""
        # Process pages in larger batches for speed
        batch_size = 50
        all_pages = {}
        
        # More aggressive parallelism
        max_workers = min(16, os.cpu_count() * 2)
        
        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count
            
            def _page_range(start: int, end: int) -> Dict[int, str]:
                """Extract a range of pages from already opened document"""
                pages = {}
                for page_num in range(start, end):
                    page = doc[page_num]
                    text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                    pages[page_num] = text
                return pages
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for start in range(0, page_count, batch_size):
                    end = min(start + batch_size, page_count)
                    future = executor.submit(_page_range, start, end)
                    futures.append(future)
                
                for future in as_completed(futures):
                    pages = future.result()
                    all_pages.update(pages)
        
        return all_pages
    
    def parse_pdf(self, pdf_path: str) -> List[Chunk]:
        """Enhanced parsing with better chunking"""
        start_time = time.time()
        
        # Extract all pages in parallel
        pages = self.extract_pages_parallel(pdf_path)
        
        # Batch clean all pages at once
        page_texts = list(pages.values())
        cleaned_texts = self._batch_clean_texts(page_texts)
        pages = {k: cleaned_texts[i] for i, k in enumerate(pages.keys())}
        
        # Enhanced smart chunking with better boundaries
        chunks = self.enhanced_smart_chunk_v3(pages)
        
        # Analyze document terms after chunking
        self.term_analyzer.analyze_document(chunks)
        
        extraction_time = time.time() - start_time
        print(f"📄 Extracted & chunked {len(pages)} pages in {extraction_time:.3f}s ({len(pages)/extraction_time:.1f} pages/sec)")
        
        return chunks
    
    def enhanced_smart_chunk_v3(self, pages: Dict[int, str]) -> List[Chunk]:
        """Enhanced chunking with better semantic preservation and boundary detection"""
        chunks = []
        
        # Combine all text with page tracking
        full_text = ""
        page_boundaries = []
        current_pos = 0
        
        for page_num in sorted(pages.keys()):
            page_text = self.text_processor.clean_text(pages[page_num])
            full_text += page_text + "\n\n"
            page_boundaries.append((current_pos, current_pos + len(page_text), page_num + 1))
            current_pos += len(page_text) + 2
        
        # Find all section headers
        section_matches = list(self.section_pattern.finditer(full_text))
        
        if section_matches:
            # Process sections in parallel
            from concurrent.futures import ThreadPoolExecutor
            
            def process_section(i, match):
                start = match.start()
                if i + 1 < len(section_matches):
                    end = section_matches[i + 1].start()
                else:
                    end = len(full_text)
                
                section_text = full_text[start:end].strip()
                page_num = self._find_page_number(start, page_boundaries)
                section_header = match.group(0).strip()
                
                if len(section_text) > self.chunk_size:
                    return self._smart_split_section_v2(section_text, page_boundaries, start, section_header)
                else:
                    return [self.create_enhanced_chunk(section_text, page_num, section_header)]
            
            # Process all sections in parallel
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() * 2)) as executor:
                futures = [executor.submit(process_section, i, match) 
                          for i, match in enumerate(section_matches)]
                
                for future in as_completed(futures):
                    chunks.extend(future.result())
        else:
            # Fallback: semantic paragraph chunking
            chunks = self._semantic_paragraph_chunking_v2(full_text, page_boundaries)
        
        return chunks
    
    def parse_document(self, path: pathlib.Path) -> List[Chunk]:
        """Generic entry point for parsing different document types"""
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(str(path))
        elif ext in {".docx", ".doc"}:
            return self.parse_docx(path)
        elif ext in {".eml", ".msg"}:
            return self.parse_email(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def parse_docx(self, path: pathlib.Path) -> List[Chunk]:
        """Parse DOCX files treating paragraphs like pages"""
        doc = Document(path)
        # Treat paragraphs like pages ⇒ enumerate for page_num.
        pages = {i: p.text for i, p in enumerate(doc.paragraphs) if p.text.strip()}
        return self.enhanced_smart_chunk_v3(pages)
    
    def parse_email(self, path: pathlib.Path) -> List[Chunk]:
        """Parse email files (.eml, .msg)"""
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f)
        parts = []
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="ignore"))
            elif ctype == "text/html":
                html = part.get_payload(decode=True).decode(errors="ignore")
                parts.append(html2text.html2text(html))
        text = "\n\n".join(parts)
        pages = {0: text}
        return self.enhanced_smart_chunk_v3(pages)
    
    def parse_blobs(self, urls: List[str]) -> List[Chunk]:
        """Parse multiple blob URLs and return all chunks"""
        all_chunks = []
        for url in urls:
            print(f"📥 Downloading: {url}")
            file_path = download_blob(url)
            try:
                chunks = self.parse_document(file_path)
                all_chunks.extend(chunks)
                print(f"✅ Processed {len(chunks)} chunks from {file_path.suffix} file")
            finally:
                file_path.unlink(missing_ok=True)  # clean temp
        return all_chunks
    
    async def parse_blobs_async(self, urls: List[str]) -> List[Chunk]:
        """Async parse multiple blob URLs and return all chunks"""
        async with aiohttp.ClientSession() as session:
            # Download all files concurrently
            print(f"📥 Downloading {len(urls)} documents concurrently...")
            download_start = time.time()
            tasks = [async_download_blob(url, session) for url in urls]
            file_paths = await asyncio.gather(*tasks)
            download_time = time.time() - download_start
            print(f"⚡ Downloaded {len(file_paths)} files in {download_time:.3f}s ({len(file_paths)/download_time:.1f} files/sec)")
            
            # Parse documents using ThreadPoolExecutor to avoid blocking
            parse_start = time.time()
            all_chunks = []
            
            def parse_single_document(file_path):
                try:
                    chunks = self.parse_document(file_path)
                    print(f"✅ Processed {len(chunks)} chunks from {file_path.suffix} file")
                    return chunks
                finally:
                    file_path.unlink(missing_ok=True)  # clean temp
            
            # Use ThreadPoolExecutor for CPU-heavy PDF parsing
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count()*2)) as executor:
                # Submit all parsing tasks
                parse_tasks = [executor.submit(parse_single_document, file_path) for file_path in file_paths]
                
                # Collect results
                for future in as_completed(parse_tasks):
                    chunks = future.result()
                    all_chunks.extend(chunks)
            
            parse_time = time.time() - parse_start
            print(f"⚡ Parsed {len(all_chunks)} total chunks in {parse_time:.3f}s")
            
            return all_chunks
    
    def _smart_split_section_v2(self, text: str, page_boundaries: List, global_start: int, section_header: str) -> List[Chunk]:
        """Enhanced section splitting with transport-aware boundaries"""
        chunks = []
        
        # Try to split by subsections first
        subsection_matches = list(self.subsection_pattern.finditer(text))
        
        if subsection_matches:
            # Split by subsections with enhanced overlap
            for i, match in enumerate(subsection_matches):
                start = match.start()
                if i + 1 < len(subsection_matches):
                    end = subsection_matches[i + 1].start()
                else:
                    end = len(text)
                
                sub_text = text[start:end].strip()
                
                # Enhanced context preservation - look for transport-related content
                if i > 0 and self.overlap > 0:
                    prev_start = max(0, subsection_matches[i-1].start())
                    # Look for transport terms in previous section
                    prev_text = text[prev_start:start]
                    transport_matches = re.findall(r'\b(?:train|flight|bus|ship|car|vehicle|transport)\b', prev_text, re.IGNORECASE)
                    
                    if transport_matches:
                        # Include more context if transport terms are present
                        overlap_size = min(self.overlap * 2, len(prev_text))
                        prev_end = min(start, prev_start + overlap_size)
                        context = text[prev_start:prev_end].strip()
                        if context:
                            sub_text = context + "\n\n" + sub_text
                    else:
                        # Normal overlap
                        prev_end = min(start, prev_start + self.overlap)
                        context = text[prev_start:prev_end].strip()
                        if context:
                            sub_text = context + "\n\n" + sub_text
                
                if sub_text:
                    page_num = self._find_page_number(global_start + start, page_boundaries)
                    chunk = self.create_enhanced_chunk(sub_text, page_num, section_header)
                    chunks.append(chunk)
        else:
            # Semantic paragraph splitting with transport awareness
            chunks = self._transport_aware_paragraph_split(text, page_boundaries, global_start, section_header)
        
        return chunks
    
    def _transport_aware_paragraph_split(self, text: str, page_boundaries: List, global_start: int, section_header: str = "") -> List[Chunk]:
        """Split text by paragraphs with transport-term awareness"""
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        current_chunk = []
        current_size = 0
        
        for i, para in enumerate(paragraphs):
            # Check for transport terms in current paragraph
            has_transport_terms = bool(re.search(r'\b(?:train|flight|bus|ship|car|vehicle|transport)\b', para, re.IGNORECASE))
            
            # Check if adding this paragraph would exceed chunk size
            if current_size + len(para) > self.chunk_size and current_chunk:
                # Create chunk with current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                page_num = self._find_page_number(global_start, page_boundaries)
                chunk = self.create_enhanced_chunk(chunk_text, page_num, section_header)
                chunks.append(chunk)
                
                # Enhanced overlap for transport terms
                overlap_paras = []
                overlap_size = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    para_text = current_chunk[j]
                    has_transport_in_para = bool(re.search(r'\b(?:train|flight|bus|ship|car|vehicle|transport)\b', para_text, re.IGNORECASE))
                    
                    # Include more context if transport terms are present
                    max_overlap = self.overlap * 2 if has_transport_in_para else self.overlap
                    
                    if overlap_size + len(para_text) <= max_overlap:
                        overlap_paras.insert(0, para_text)
                        overlap_size += len(para_text)
                    else:
                        break
                
                current_chunk = overlap_paras
                current_size = overlap_size
            
            current_chunk.append(para)
            current_size += len(para)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            page_num = self._find_page_number(global_start, page_boundaries)
            chunk = self.create_enhanced_chunk(chunk_text, page_num, section_header)
            chunks.append(chunk)
        
        return chunks
    
    def _semantic_paragraph_chunking_v2(self, full_text: str, page_boundaries: List) -> List[Chunk]:
        """Enhanced fallback semantic chunking with transport awareness"""
        # Split by paragraphs and create semantic chunks
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        current_pos = 0
        
        for para in paragraphs:
            # Check for transport terms
            has_transport_terms = bool(re.search(r'\b(?:train|flight|bus|ship|car|vehicle|transport)\b', para, re.IGNORECASE))
            
            if current_size + len(para) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = '\n\n'.join(current_chunk)
                page_num = self._find_page_number(current_pos, page_boundaries)
                chunk = self.create_enhanced_chunk(chunk_text, page_num, "")
                chunks.append(chunk)
                
                # Enhanced overlap for transport terms
                if self.overlap > 0 and len(current_chunk) > 0:
                    # Keep more context if transport terms present
                    if has_transport_terms:
                        # Keep last 2 paragraphs if they fit in overlap
                        keep_paras = []
                        keep_size = 0
                        for j in range(len(current_chunk) - 1, -1, -1):
                            if keep_size + len(current_chunk[j]) <= self.overlap * 2:
                                keep_paras.insert(0, current_chunk[j])
                                keep_size += len(current_chunk[j])
                            else:
                                break
                        current_chunk = keep_paras
                        current_size = keep_size
                    else:
                        # Normal overlap
                        current_chunk = current_chunk[-1:] if current_chunk else []
                        current_size = len(current_chunk[0]) if current_chunk else 0
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(para)
            current_size += len(para)
            current_pos += len(para)
        
        # Final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            page_num = self._find_page_number(current_pos, page_boundaries)
            chunk = self.create_enhanced_chunk(chunk_text, page_num, "")
            chunks.append(chunk)
        
        return chunks
    
    def _find_page_number(self, position: int, page_boundaries: List[Tuple[int, int, int]]) -> int:
        """Find page number for a given text position"""
        for start, end, page_num in page_boundaries:
            if start <= position < end:
                return page_num
        return 1
    
    def create_enhanced_chunk(self, text: str, page_num: int, section_header: str) -> Chunk:
        """Create chunk with enhanced metadata and preprocessing"""
        # Clean text for better embeddings
        clean_text = self.text_processor.clean_text(text)
        
        # Enhanced chunk type detection
        chunk_type = "text"
        if self.section_pattern.match(text):
            chunk_type = "section_header"
        elif self.legal_markers.search(text):
            legal_density = len(self.legal_markers.findall(text)) / max(len(text.split()), 1)
            if legal_density > 0.03:
                chunk_type = "legal_provision"
            elif legal_density > 0.01:
                chunk_type = "legal_text"
        
        # Detect transport-specific chunks
        transport_terms = re.findall(r'\b(?:train|flight|bus|ship|car|vehicle|transport)\b', text, re.IGNORECASE)
        if transport_terms:
            chunk_type = f"transport_{chunk_type}"
        
        # Extract enhanced features
        semantic_keywords = self.text_processor.extract_semantic_keywords(text)
        
        # Generate unique ID
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        return Chunk(
            text=text,
            clean_text=clean_text,
            page_num=page_num,
            chunk_type=chunk_type,
            metadata={
                'chunk_id': chunk_id,
                'section': section_header,
                'word_count': len(text.split()),
                'has_legal_terms': bool(self.legal_markers.search(text)),
                'semantic_keywords': semantic_keywords,
                'legal_density': len(self.legal_markers.findall(text)) / max(len(text.split()), 1),
                'char_length': len(text),
                'transport_terms': list(set(transport_terms)),
                'transport_count': len(transport_terms)
            }
        )
    
    async def embed_chunks_async(self, chunks: List[Chunk]) -> List[Chunk]:
        """Use async embedding for speed"""
        if not self.async_embedder:
            print("⚠️ No API key, skipping embeddings")
            return chunks
        
        return await self.async_embedder.embed_all_chunks_async(chunks)
    
    def embed_chunks_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Wrapper for async embedding - handles event loop detection"""
        if not self.api_key:
            print("⚠️ No API key, creating random embeddings for testing")
            for chunk in chunks:
                chunk.embedding = np.random.randn(3072).astype(np.float32)
            return chunks
        
        # Try to detect if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we're in an async context, we can't use asyncio.run()
            # The caller should use embed_chunks_async directly
            raise RuntimeError("embed_chunks_batch cannot be called from async context. Use embed_chunks_async instead.")
        except RuntimeError:
            # No running loop, we can use asyncio.run()
            return asyncio.run(self.embed_chunks_async(chunks))
    
    async def embed_chunks_async(self, chunks: List[Chunk]) -> List[Chunk]:
        """Async wrapper for embedding with session reuse"""
        if not self.async_embedder:
            print("⚠️ No API key, creating random embeddings for testing")
            for chunk in chunks:
                chunk.embedding = np.random.randn(3072).astype(np.float32)
            return chunks
        
        return await self.async_embedder.embed_all_chunks_async(chunks)
    
    def embed_query(self, query: str, query_analysis: Dict = None) -> np.ndarray:
        """Embed a single query with conservative enhancement"""
        if not self.client:
            return np.random.randn(3072).astype(np.float32)
        
        # Conservative query enhancement
        if query_analysis:
            enhanced_query = self.text_processor.expand_query_conservatively(query, query_analysis)
        else:
            enhanced_query = query
        
        try:
            model = "text-embedding-3-large"
            response = self.client.embeddings.create(
                model=model,
                input=enhanced_query
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Query embedding error: {e}")
            return np.zeros(3072)

class UltraFastRAGRetriever:
    """Enhanced RAG retrieval system with term-importance aware scoring"""
    
    def __init__(self, parser: UltraFastRAGParser):
        self.parser = parser
        self.vector_store = UltraFastVectorStore()
        self.text_processor = EnhancedTextProcessor()
        self.term_analyzer = parser.term_analyzer
        # Initialize LLM components
        self.answer_generator = AnswerGenerator(parser.api_key)
        
    def index_document(self, pdf_path: str) -> List[Chunk]:
        """Index a document"""
        # Parse and embed
        chunks = self.parser.parse_pdf(pdf_path)
        chunks_with_embeddings = self.parser.embed_chunks_batch(chunks)
        
        # Index
        index_start = time.time()
        self.vector_store.add_chunks(chunks_with_embeddings)
        index_time = time.time() - index_start
        print(f"🗂️ Indexed {len(chunks)} chunks in {index_time:.3f}s")
        
        return chunks_with_embeddings
    
    def index_documents(self, blob_urls: List[str]) -> List[Chunk]:
        """Index multiple documents from blob URLs"""
        # Parse all documents
        chunks = self.parser.parse_blobs(blob_urls)
        chunks_with_embeddings = self.parser.embed_chunks_batch(chunks)
        
        # Index
        index_start = time.time()
        self.vector_store.add_chunks(chunks_with_embeddings)
        index_time = time.time() - index_start
        print(f"🗂️ Indexed {len(chunks)} chunks in {index_time:.3f}s")
        
        return chunks_with_embeddings
    
    async def index_documents_async(self, blob_urls: List[str]) -> List[Chunk]:
        """Async index multiple documents from blob URLs"""
        # Parse all documents asynchronously
        chunks = await self.parser.parse_blobs_async(blob_urls)
        
        # Embed chunks asynchronously
        chunks_with_embeddings = await self.parser.embed_chunks_async(chunks)
        
        # Index
        index_start = time.time()
        self.vector_store.add_chunks(chunks_with_embeddings)
        index_time = time.time() - index_start
        print(f"🗂️ Indexed {len(chunks)} chunks in {index_time:.3f}s")
        
        return chunks_with_embeddings
    
    async def retrieve_and_answer_async(self, query: str, k: int = 5,
                                        session: aiohttp.ClientSession = None) -> Tuple[List[Tuple[Chunk, float]], str]:
        """Enhanced retrieval with query refinement and answer generation"""
        start_time = time.time()
        
        # Step 1: Analyze original query
        query_analysis = self.text_processor.enhanced_query_analysis(query)
        
        # Embed original query directly
        query_embedding = self.parser.embed_query(query, query_analysis)
        
        # Step 4: Search with more candidates for better reranking
        search_start = time.time()
        results = self.vector_store.search(query_embedding, k * 4)  # Get 4x more for better reranking
        search_time = time.time() - search_start
        
        # Step 5: Term-importance aware reranking
        reranked = self._term_importance_rerank(results, query, query_analysis, k)
        
        # Step 6: Generate answer using GPT-4o-mini (async)
        answer = await self.answer_generator.generate_answer_async(query, reranked, session)
        
        total_time = time.time() - start_time
        print(f"🔍 Total retrieval + answer time: {total_time:.3f}s (search: {search_time:.3f}s)")
        
        return reranked, answer
    
    def retrieve_and_answer(self, query: str, k: int = 5) -> Tuple[List[Tuple[Chunk, float]], str]:
        """Synchronous wrapper for retrieve and answer"""
        return asyncio.run(self.retrieve_and_answer_async(query, k))
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Enhanced retrieval with term-importance aware scoring"""
        # Analyze query first
        query_analysis = self.text_processor.enhanced_query_analysis(query)
        print(f"🎯 Query analysis: {query_analysis}")
        
        # Embed query
        query_start = time.time()
        query_embedding = self.parser.embed_query(query, query_analysis)
        query_time = time.time() - query_start
        
        # Search with more candidates for better reranking
        search_start = time.time()
        results = self.vector_store.search(query_embedding, k * 4)  # Get 4x more for better reranking
        search_time = time.time() - search_start
        
        # Term-importance aware reranking
        reranked = self._term_importance_rerank(results, query, query_analysis, k)
        
        print(f"🔍 Query embedded in {query_time:.3f}s, retrieved in {search_time:.3f}s")
        
        return reranked
    
    def _term_importance_rerank(self, results: List[Tuple[Chunk, float]], query: str, query_analysis: Dict, top_k: int) -> List[Tuple[Chunk, float]]:
        """Enhanced reranking with term importance and entity awareness"""
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b[a-zA-Z]{2,}\b', query_lower))
        critical_terms = set(query_analysis.get('critical_terms', []))
        transport_type = query_analysis.get('transport_type')
        
        scored_results = []
        
        for chunk, base_score in results:
            chunk_lower = (chunk.clean_text or chunk.text).lower()
            
            # 1. Term Importance Score (IDF-based)
            term_importance_score = 0
            exact_matches = 0
            critical_matches = 0
            
            for term in query_terms:
                if len(term) <= 2:  # Skip very short terms
                    continue
                    
                # Check if term exists in chunk
                if re.search(rf'\b{re.escape(term)}\b', chunk_lower):
                    exact_matches += 1
                    
                    # Get term importance from document analysis
                    importance = self.term_analyzer.get_term_importance(term)
                    term_importance_score += importance
                    
                    # Extra boost for critical terms
                    if term in critical_terms:
                        critical_matches += 1
                        term_importance_score += importance * 2
            
            # 2. Transport Type Matching (CRITICAL for train vs flight)
            transport_match_score = 0
            if transport_type:
                chunk_transport_terms = chunk.metadata.get('transport_terms', [])
                chunk_transport_lower = [t.lower() for t in chunk_transport_terms]
                
                # Strong penalty for wrong transport type
                wrong_transport_penalty = 0
                for t_type, terms in self.text_processor.transport_terms.items():
                    if t_type != transport_type:
                        for term in terms:
                            if term in chunk_lower:
                                wrong_transport_penalty -= 2.0  # Heavy penalty
                
                # Strong bonus for correct transport type
                correct_transport_terms = self.text_processor.transport_terms.get(transport_type, [])
                for term in correct_transport_terms:
                    if term in chunk_lower:
                        transport_match_score += 3.0  # Strong bonus
                        
                        # Extra bonus if it's the exact term from query
                        if term in query_lower:
                            transport_match_score += 2.0
                
                transport_match_score += wrong_transport_penalty
            
            # 3. Exact phrase matching
            phrase_score = 0
            if len(query_terms) > 1:
                # Create meaningful phrases from query
                query_words = query_lower.split()
                for i in range(len(query_words) - 1):
                    bigram = f"{query_words[i]} {query_words[i+1]}"
                    if bigram in chunk_lower:
                        phrase_score += 1.0
                
                # Full query match
                if query_lower.strip() in chunk_lower:
                    phrase_score += 2.0
            
            # 4. Context relevance (for coverage inquiries)
            context_score = 0
            intent_type = query_analysis.get('intent_type', 'general')
            if intent_type == 'coverage_inquiry':
                coverage_terms = ['covered', 'coverage', 'benefits', 'compensation', 'insured', 'policy']
                context_matches = sum(1 for term in coverage_terms if term in chunk_lower)
                context_score = min(context_matches * 0.3, 1.0)
            
            # 5. Section relevance
            section_score = 0
            if chunk.metadata.get('section'):
                section_lower = chunk.metadata['section'].lower()
                section_matches = sum(1 for term in critical_terms if term in section_lower)
                if section_matches > 0:
                    section_score = (section_matches / len(critical_terms)) * 0.5
            
            # 6. Chunk type bonus
            type_bonus = 0
            chunk_type = chunk.chunk_type
            if transport_type and chunk_type.startswith('transport_'):
                type_bonus = 0.3
            elif chunk_type in ['legal_provision', 'legal_text'] and intent_type == 'coverage_inquiry':
                type_bonus = 0.2
            
            # 7. Position and length factors
            position_bonus = 0
            if exact_matches > 0:
                # Find first occurrence of any query term
                first_pos = len(chunk_lower)
                for term in critical_terms:
                    pos = chunk_lower.find(term)
                    if pos != -1:
                        first_pos = min(first_pos, pos)
                
                # Bonus for terms appearing early
                if first_pos < len(chunk_lower) * 0.3:
                    position_bonus = 0.2
            
            # Length penalty for very long chunks without many matches
            length_penalty = 0
            word_count = chunk.metadata.get('word_count', 0)
            if word_count > 500 and exact_matches <= 1:
                length_penalty = -0.3
            
            # FINAL SCORE with optimized weights emphasizing term importance
            final_score = (
                0.25 * base_score +                    # Vector similarity
                0.30 * min(term_importance_score, 3.0) +  # Term importance (KEY)
                0.25 * transport_match_score +         # Transport matching (KEY for train/flight)
                0.08 * phrase_score +                  # Phrase matching
                0.05 * context_score +                 # Context relevance
                0.03 * section_score +                 # Section relevance
                0.02 * type_bonus +                    # Type bonus
                0.02 * position_bonus +                # Position bonus
                length_penalty                         # Length penalty
            )
            
            scored_results.append((chunk, final_score, {
                'base_score': base_score,
                'term_importance': term_importance_score,
                'transport_match': transport_match_score,
                'exact_matches': exact_matches,
                'critical_matches': critical_matches,
                'phrase_score': phrase_score,
                'context_score': context_score
            }))
        
        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [(chunk, score) for chunk, score, debug in scored_results[:top_k]]

def print_retrieval_results(results: List[Tuple[Chunk, float]], query: str):
    """Print retrieval results with enhanced formatting and debug info"""
    print(f"\n🔍 QUERY: '{query}'")
    print(f"📊 Top {len(results)} results:")
    print("="*100)
    
    # Prepare query terms for highlighting
    query_terms = set(re.findall(r'\b[a-zA-Z]{2,}\b', query.lower()))
    important_terms = [term for term in query_terms if len(term) > 2]
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n🏆 RESULT {i} | Relevance: {score:.4f}")
        print(f"📄 Page: {chunk.page_num} | Type: {chunk.chunk_type}")
        print(f"📌 Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"🚗 Transport terms: {chunk.metadata.get('transport_terms', [])}")
        print(f"🔑 Semantic keywords: {', '.join(chunk.metadata.get('semantic_keywords', [])[:5])}")
        print(f"⚖️ Legal density: {chunk.metadata.get('legal_density', 0):.2%}")
        
        print(f"\n📝 Content:")
        print("-"*100)
        
        # Use clean_text if available for better display
        text = chunk.clean_text if chunk.clean_text else chunk.text
        
        # Smart text preview with context
        if len(text) > 800:
            # Find most relevant part containing query terms
            sentences = text.split('. ')
            best_sentences = []
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                score = sum(1 for term in important_terms if term in sentence_lower)
                if score > 0:
                    # Include this sentence and some context
                    start = max(0, i - 1)
                    end = min(len(sentences), i + 2)
                    best_sentences.extend(sentences[start:end])
                    break
            
            if best_sentences:
                text = '. '.join(best_sentences) + '.'
            else:
                # Fallback to first part
                text = '. '.join(sentences[:4]) + '.'
            
            if len(sentences) > 4:
                text = text + " ..."
        
        # Highlight query terms
        for term in important_terms:
            # Use regex for word boundaries
            pattern = re.compile(f'\\b{re.escape(term)}\\b', re.IGNORECASE)
            text = pattern.sub(f'**{term.upper()}**', text)
        
        print(text)
        print("-"*100)


async def run_complete_rag_pipeline_async(document_urls: List[str], queries: List[str], api_key: str = None):
    """Run the complete enhanced RAG pipeline with async processing and parallel query handling"""
    print(f"🚀 ENHANCED ULTRA-FAST RAG PIPELINE v6.0 (Async + Parallel)")
    print(f"📄 Documents: {len(document_urls)} URLs")
    for i, url in enumerate(document_urls, 1):
        print(f"   {i}. {url[:80]}...")
    total_start = time.time()
    
    # Initialize components with enhanced settings
    parser = UltraFastRAGParser(chunk_size=1000, overlap=200, api_key=api_key)
    retriever = UltraFastRAGRetriever(parser)
    
    # Index documents asynchronously (with caching)
    cache_path = f"cache_multi_docs_v6_llm_enhanced"
    index_start = time.time()
    chunks = await retriever.index_documents_async(document_urls)
    index_time = time.time() - index_start
    
    print(f"\n⚡ Async indexing complete in {index_time:.3f}s")
    print(f"📊 Total chunks: {len(chunks)}")
    
    # Process queries in parallel for maximum speed
    query_start = time.time()
    print(f"\n🔍 Processing {len(queries)} queries in parallel...")
    
    # Launch all queries concurrently using shared session
    tasks = [retriever.retrieve_and_answer_async(query, k=5, session=net_utils.shared_session) for query in queries]
    all_results = await asyncio.gather(*tasks)
    
    query_time = time.time() - query_start
    print(f"⚡ All queries processed in {query_time:.3f}s ({len(queries)/query_time:.1f} queries/sec)")
    
    # Process and display results
    all_qa_pairs = []
    
    for i, (query, (results, answer)) in enumerate(zip(queries, all_results), 1):
        print(f"\n{'='*100}")
        print(f"📋 QUERY {i}/{len(queries)}: {query}")
        
        # Print results with debug info
        print_retrieval_results(results, query)
        
        # Print the LLM-generated answer
        print(f"\n🤖 AI-GENERATED ANSWER:")
        print("="*100)
        print(answer)
        print("="*100)
        
        # Store QA pair
        all_qa_pairs.append({
            "question": query,
            "answer": answer,
            "retrieval_results": [(chunk.metadata.get('chunk_id', ''), score) for chunk, score in results]
        })
    
    # Total time
    total_time = time.time() - total_start
    print(f"\n🎯 TOTAL ASYNC PIPELINE TIME: {total_time:.3f}s")
    print(f"📈 Speed improvement: Processing {len(queries)} queries + {len(document_urls)} documents")
    
    # Print summary of all Q&A pairs
    print(f"\n📋 SUMMARY - All Questions & Answers:")
    print("="*120)
    for i, qa in enumerate(all_qa_pairs, 1):
        print(f"\nQ{i}: {qa['question']}")
        print(f"A{i}: {qa['answer']}")
        print("-"*80)
    
    return retriever, all_qa_pairs

def run_complete_rag_pipeline(document_urls: List[str], queries: List[str], api_key: str = None):
    """Run the complete enhanced RAG pipeline with LLM-powered answering"""
    print(f"🚀 ENHANCED ULTRA-FAST RAG PIPELINE v5.0 (LLM-Powered Answers)")
    print(f"📄 Documents: {len(document_urls)} URLs")
    for i, url in enumerate(document_urls, 1):
        print(f"   {i}. {url[:80]}...")
    total_start = time.time()
    
    # Initialize components with enhanced settings
    parser = UltraFastRAGParser(chunk_size=1000, overlap=200, api_key=api_key)
    retriever = UltraFastRAGRetriever(parser)
    
    # Index documents (with caching)
    cache_path = f"cache_multi_docs_v6_llm_enhanced"
    index_start = time.time()
    chunks = retriever.index_documents(document_urls, cache_path)
    index_time = time.time() - index_start
    
    print(f"\n⚡ Indexing complete in {index_time:.3f}s")
    print(f"📊 Total chunks: {len(chunks)}")
    
    # Process queries with LLM enhancement
    all_qa_pairs = []
    
    for query in queries:
        print(f"\n{'='*100}")
        query_start = time.time()
        
        # Retrieve with enhanced query refinement and answer generation
        results, answer = retriever.retrieve_and_answer(query, k=5)
        
        query_time = time.time() - query_start
        
        # Print results with debug info
        print_retrieval_results(results, query)
        
        # Print the LLM-generated answer
        print(f"\n🤖 AI-GENERATED ANSWER:")
        print("="*100)
        print(answer)
        print("="*100)
        
        print(f"\n⏱️ Query processed in {query_time:.3f}s")
        
        # Store QA pair
        all_qa_pairs.append({
            "question": query,
            "answer": answer,
            "retrieval_results": [(chunk.metadata.get('chunk_id', ''), score) for chunk, score in results]
        })
    
    # Total time
    total_time = time.time() - total_start
    print(f"\n🎯 TOTAL PIPELINE TIME: {total_time:.3f}s")
    
    # Print summary of all Q&A pairs
    print(f"\n📋 SUMMARY - All Questions & Answers:")
    print("="*120)
    for i, qa in enumerate(all_qa_pairs, 1):
        print(f"\nQ{i}: {qa['question']}")
        print(f"A{i}: {qa['answer']}")
        print("-"*80)
    
    return retriever, all_qa_pairs

if __name__ == "__main__":
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key
    
    # Example URLs - replace with your actual blob URLs
    urls = [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        # "https://example.com/notice_of_loss.eml?...",
    ]
    
    # For local testing, you can use file:// URLs or just provide the local path
    #urls = ["file://doc2.pdf"]  # This will work with your existing doc2.pdf
    
    # Example queries for testing accuracy (with various insurance policy questions)
    queries = [
         "have Cataract, how much the policy covers?",
         "am obese, does the policy cover me ?",
         "under NPMPP , do you cover Bone marrow surgery?"
    ]
    
    if not api_key:
        print("⚠️ Warning: OPENAI_API_KEY not set. Using random embeddings and no answer generation.")
    
    try:
        # Choose between sync and async pipeline
        use_async = True  # Set to False for traditional sync pipeline
        
        if use_async:
            print("🚀 Running ASYNC pipeline for maximum speed...")
            # Run the async pipeline
            retriever, qa_pairs = asyncio.run(run_complete_rag_pipeline_async(urls, queries, api_key))
        else:
            print("🐌 Running SYNC pipeline (traditional)...")
            # Run the enhanced pipeline with LLM integration (sync)
            retriever, qa_pairs = run_complete_rag_pipeline(urls, queries, api_key)
        
        # Optional: Save Q&A pairs to JSON
        output_file = f"qa_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2, default=str)
        print(f"\n💾 Q&A results saved to {output_file}")
        
    except FileNotFoundError:
        print("❌ File not found. Please update the URLs.")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
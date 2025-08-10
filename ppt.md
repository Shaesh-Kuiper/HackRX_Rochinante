# RAG Approach Comparison

| Feature / Aspect                     | **Our System** (LLM-first Lexical Refinement)                                       | **RAPTOR** (Tree Summaries)                                      | **LongRAG** (Long-context Retrieval)                             | **MacRAG / MaC-RAG** (Memory-Augmented Conversational RAG)       |
|---------------------------------------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| **Initial Step**                      | LLM **refines the query first**, generating multi-level lexical synonyms tailored to match headings & semantic meaning | Builds **hierarchical summaries** of the corpus before retrieval | Uses very large context windows to stuff in as much source text as possible | Stores & retrieves **conversation history** alongside new context |
| **Retrieval Core**                    | Hybrid **BM25 + FAISS** → RRF fusion → MMR diversification                           | Graph/tree traversal over summary hierarchy                      | Vector search or BM25 → feed many chunks directly into LLM        | Vector search + memory store retrieval                           |
| **Reranking**                         | GPU **cross-encoder** rerank (batched across all Qs)                                 | Usually not GPU-rerank focused                                    | Often no rerank or lightweight rerank                             | Often no rerank or lightweight rerank                             |
| **Latency Strategy**                  | Heavy **parallelism + caching**; one-pass rerank for all Qs                           | Slower due to summary creation & multi-hop reasoning              | Can be slow/expensive due to huge context token count             | Moderate; depends on memory store size                           |
| **Explainability**                    | Returns **exact clause snippets** used for the answer                                | Can reference summary nodes, but less direct to original text     | Direct, but often from large context without fine-grained targeting | May show snippets from memory or retrieved docs                   |
| **Adaptability to Noisy Queries**     | High — query refinement handles typos, vague/incomplete phrasing                      | Moderate — depends on how well summaries capture rare phrasing    | Moderate — large context may help, but retrieval may still miss   | Moderate — conversation context may help, but not for one-off Qs |
| **Preprocessing Needs**               | Low — works directly on raw docs (PDF/DOCX/email)                                     | High — needs to pre-build and maintain summary trees              | Low–moderate — needs embeddings & vector store                    | Moderate — needs embeddings & memory management                  |

---

**Summary of Strengths**

Our system **wins** in scenarios where:
- Queries are **vague, incomplete, or noisy** — the LLM-first refinement intelligently shapes them into high-precision lexical and semantic probes.
- **Speed and scale** matter — aggressive async, batching, and one-pass GPU rerank keep latency low even for many queries at once.
- **Explainability is critical** — clause-level, document-grounded citations come built-in.
- **No heavy preprocessing** is feasible — works directly from source documents without building and maintaining large summary graphs or long-memory stores.

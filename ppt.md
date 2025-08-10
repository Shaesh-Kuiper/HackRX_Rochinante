# Comparison with differnet SOTA RAG Approaches 

| Aspect (Top-5)            | **Our System (LLM-first Lexical Refinement)**                               | **RAPTOR** (Tree Summaries)              | **LongRAG** (Long Context)                   | **MacRAG / MaC-RAG** (Conv. Memory)        | **Why this is better** |
|---------------------------|---------------------------------------------------------------------|------------------------------------------|----------------------------------------------|--------------------------------------------|------------------------|
| 1) Vague Query Handling   | **LLM refines into 3 lexical tiers** → higher hit rate              | Depends on summary quality               | Big context may still miss exact targets     | Helps across turns; weaker for one-offs    | Turns messy asks into **precise probes** without corpus prep → **higher recall** on ambiguous inputs. |
| 2) Retrieval Precision    | **BM25 + FAISS → RRF → MMR** + **GPU cross-encoder rerank**         | Summary traversal, less fine-grained     | Broad recall, lower precision                | Memory + basic retrieval                   | Blends lexical + semantic, **de-dupes**, then **reranks with a cross-encoder** → **fewer misses & shorter context**. |
| 3) Evidence & Explainability | **Clause-level snippets** returned                              | Points to summaries, not clauses         | Evidence buried in long context              | Snippets from memory/docs                  | Direct **clause citations** → **audit-ready**, traceable, easy to validate. |
| 4) Latency & Cost         | **Async + batching; one-pass `/rerank_pairs`** (token-lean)         | Prep & multi-hop add latency/cost        | Large token windows = slower, pricier        | Grows with memory size/state               | **Throughput↑, cost↓**; scales to many Qs under tight SLOs with **stable latency**. |
| 5) Setup & Maintenance    | **Low** — raw PDFs/DOCX/emails + caching                            | **High** — build/maintain trees          | Low–mod; heavy context mgmt                  | Mod — memory policies/state to manage      | **Fast onboarding**, minimal plumbing, resilient to document updates. |

---

## Why we win 
We **ask smarter first**. LLM-guided query refinement + hybrid search + GPU rerank yields **exact clauses fast**, with **low prep and cost**—perfect for policies/contracts/emails where questions are messy but answers must be **accurate, explainable, and quick**.

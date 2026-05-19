# Insurance Pipeline — Architecture Reference

> **Purpose:** Quick-reference for understanding the repo before demos or interviews.

---

## The Core Problem with the Old Pipeline

The original pipeline was **single-shot**: one query → one vector search → one answer. If the embedding missed the intent, the query was vague, or the document used different terminology — you got a bad answer with no recovery mechanism.

---

## Pipeline Overview (2026 SOTA)

```
Query → Intelligence Layer → Strategy Router → Retrieval Loop → Context Engine → Generate → Self-Critique → Done
         ├─ classify          ├─ pick the       ├─ retry if       ├─ dedup          ├─ enriched    └─ verify
         ├─ extract entities  └─ right tool     └─ insufficient   ├─ compress           prompt          quality
         └─ expand / HyDE                                         └─ hierarchical
```

---

## Key Components

### 1. `query_intelligence.py` — Query Understanding Layer

**Problem:** `"What's claim CLM-2024-001 about?"` and `"Compare all denied water damage claims and explain the patterns"` both got a single vector search. Simple queries worked; complex ones failed.

| Feature | What It Does |
|---|---|
| **Entity Extraction** | Regex-based (zero-latency) — pulls claim numbers, names, damage types, amounts → used as metadata filters |
| **Query Routing** | Classifies into `SIMPLE / MODERATE / COMPLEX / STRUCTURED` → each gets a different retrieval strategy. Rule-based first (free), LLM fallback for ambiguous cases |
| **HyDE** | For moderate queries: generates a hypothetical answer document and embeds *that* instead of the raw question — bridges the semantic gap between questions and documents (10–30% retrieval improvement) |
| **Multi-Query Expansion** | For complex queries: generates 4 reformulations, retrieves for each, merges with **Reciprocal Rank Fusion (RRF)** — same algorithm used by Elasticsearch, Pinecone, Weaviate |

> **Cost:** ~200–500 tokens per query for routing + expansion. Pays for itself by avoiding bad retrievals that waste generation tokens.

---

### 2. `knowledge_graph.py` — Relational Intelligence

**Problem:** Everything was stored as flat vectors. `"Which claims from policyholder Mueller were denied?"` requires traversing relationships — vectors find *similar text*, not *connected entities*.

| Feature | What It Does |
|---|---|
| **Entity Extraction** | During Stage 2, extracts claimants, claims, policies, damage types, invoices as graph **nodes** |
| **Relationship Building** | Creates edges: `filed_claim`, `has_policy`, `covers`, `documented_by`, `invoiced_by` |
| **Graph Traversal** | At query time, if the query mentions "Mueller" → traverses the graph → finds all connected claims and documents → injects as structured facts into the prompt |
| **Graph-Guided Retrieval** | Uses related filenames from the graph to do targeted ChromaDB searches, catching documents that vector search would miss |

> **Implementation choice:** NetworkX (in-memory) over Neo4j. The graph is small (~100s of nodes) — in-memory is 10× faster with zero infrastructure overhead.

---

### 3. `context_engine.py` — Context Engineering

**Problem:** The old `build_context()` just concatenated chunks in retrieval-score order. With multi-query expansion producing 10+ chunks, many are duplicates, some are low-relevance, and the LLM loses document structure.

| Feature | What It Does |
|---|---|
| **Deduplication** | Removes chunks with >80% text overlap. Keeps the highest-scoring version |
| **Hierarchical Organization** | Groups chunks by source document, sorts by `chunk_index`. LLM sees "Document A → Chunk 1, Chunk 2" instead of random interleaving |
| **Relevance-Weighted Compression** | High-confidence chunks → full text. Low-confidence chunks → 1–2 sentence summary. Fits 2× more information in the same token budget |
| **Structured Context Layers** | (1) Graph facts (deterministic, most reliable) → (2) Primary context (full text, high relevance) → (3) Supplementary context (compressed, lower relevance) |
| **Sufficiency Check (CRAG)** | Before generating: *"Is this context enough to answer?"* — if not, triggers a retry with expanded queries. This is the "Corrective" in **Corrective RAG** |

---

### 4. `evaluation_enhanced.py` — RAGAS-Style Evaluation

**Problem:** The old evaluation used a single GPT-4o score (1–5). A score of 3 could mean *wrong documents retrieved* or *right documents but hallucinated answer* — you need separate metrics for each failure mode.

| Metric | How It Works |
|---|---|
| **Faithfulness (0–1)** | Decomposes the answer into atomic statements, checks each against the context — catches hallucination at the statement level |
| **Answer Relevancy (0–1)** | Generates reverse questions from the answer, checks if they match the original query — catches off-topic answers |
| **Context Precision (0–1)** | Checks if each retrieved chunk is relevant, weighted by rank position — irrelevant chunks at rank 1 hurt more than at rank 5 |
| **Hallucination Detection** | Looks for fabricated names, numbers, dates, claim IDs that appear in the answer but NOT in the context |
| **Composite Score** | Weighted combination that pinpoints exactly what to fix |

---

### 5. `agentic_rag.py` — The Orchestrator

**Problem:** Ties everything together into one intelligent pipeline that *decides* how to answer each query, rather than applying the same approach to everything.

**Full execution flow:**

1. Safety check (prompt injection detection)
2. Query Intelligence → produces a `QueryPlan`
3. Strategy execution — one of:
   - Single-shot
   - HyDE
   - Multi-query + RRF
   - Graph-enhanced
   - Structured lookup
4. Context Engineering (dedup → organize → compress → enrich)
5. Sufficiency check → retry if needed (up to 3 iterations)
6. Confidence gating — refuse if all chunks are irrelevant
7. Enhanced answer generation (graph facts + structured info injected)
8. Self-critique (LLM verifies its own answer quality)
9. Citation verification
10. Comprehensive logging (strategy used, iteration count, token cost, timing)

---

### 6. Integration Changes

| File | Change |
|---|---|
| `stage2_extraction.py` | Now auto-builds the knowledge graph after extraction |
| `query.py` | Completely rewritten to use the agentic pipeline (with fallback to legacy) |
| `pipeline.py` | Added `evaluation_enhanced` as a runnable stage |
| `config.py` | Added all new configuration parameters (all env-var overridable) |
| `requirements.txt` | Added `networkx` for graph operations |

---

## Architecture Comparison

| Dimension | Before | After |
|---|---|---|
| Query understanding | None — raw query → vector search | Entity extraction + complexity classification + routing |
| Retrieval strategy | Always single-shot dense + post-hoc BM25 | Adaptive: single-shot / HyDE / multi-query+RRF / graph-guided |
| Recovery mechanism | None — bad retrieval = bad answer | Sufficiency check → retry with expanded queries (up to 3×) |
| Relationship queries | Impossible — flat vectors only | Knowledge graph traversal for multi-hop queries |
| Context quality | Raw chunk concatenation | Dedup → hierarchical → compress → structured layers |
| Answer verification | Citation regex check only | Self-critique + citation verification |
| Evaluation | Single LLM-as-judge score (1–5) | 4 RAGAS metrics + hallucination detection + composite score |
| Structured queries | Separate payout path only | Auto-detected and routed to deterministic lookup |
| Cost control | Fixed cost per query | Proportional: simple queries cheap, complex queries invest more |

---

## Running the Pipeline

```bash
# Run the agentic pipeline (replaces old stage5)
python3 agentic_rag.py

# Build/rebuild knowledge graph manually
python3 knowledge_graph.py

# Run enhanced evaluation
python3 evaluation_enhanced.py

# Via API — automatically uses agentic pipeline
# POST /api/query {"query": "..."}

# Disable agentic mode (fallback to legacy)
export AGENTIC_RAG_ENABLED=false
```

> All new features are controlled by environment variables in `config.py` — toggle without code changes.
> The legacy pipeline (`stage5_retrieval.py`) is untouched and remains the fallback.

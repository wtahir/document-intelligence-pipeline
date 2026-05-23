# Document Intelligence Pipeline

**Production-grade Agentic RAG system for automated insurance claim processing**

Not the basic embed→retrieve→generate portfolio project. This is a self-correcting, agentic retrieval pipeline that ingests German insurance claim PDFs (emails, invoices, photo reports), extracts structured data via LLM, validates against Pydantic schemas, and answers natural language queries with a multi-stage intelligent retrieval loop — featuring query intelligence routing, knowledge graph enrichment, context engineering, HyDE + multi-query expansion, and LLM self-critique with citation verification.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔗 Live Demo

> **[document-intelligence-pipeline.onrender.com](https://document-intelligence-pipeline.onrender.com)**

Pre-loaded with 90 synthetic German insurance PDFs (water, storm, glass damage). The demo showcases the full agentic RAG pipeline with transparent internals — query intelligence routing, knowledge graph facts, self-critique verification, and context engineering are all visible in the Query Interface.

> ⏱ **Cold-start note:** The free Render tier spins down after 15 min of inactivity. The first request after idle takes ~30 seconds to wake up — just refresh once.

---

## Why This Project

Insurance companies process thousands of claim documents daily across multiple languages. Manual classification, data extraction, and retrieval are slow, error-prone, and expensive. This pipeline demonstrates how to automate that workflow end-to-end with production patterns that go far beyond basic vector RAG:

### Agentic Retrieval (what makes this different)
- **Query Intelligence** — Classifies query complexity (simple/moderate/complex), routes to optimal retrieval strategy, extracts entities for structured lookup
- **HyDE (Hypothetical Document Embeddings)** — Generates ideal answer documents to embed, bridging the semantic gap between questions and stored documents
- **Multi-Query Expansion** — LLM generates 3-5 reformulated sub-queries, retrieves for each, deduplicates via Reciprocal Rank Fusion
- **Knowledge Graph Retrieval** — NetworkX-based claim graph (entities, relationships, severities) provides verified structured facts alongside vector results
- **Context Engineering** — Deduplication → hierarchical organization → relevance compression → graph enrichment before generation
- **Self-Correcting Retrieval Loop** — Sufficiency check after retrieval; if context is insufficient, reformulates query and retries (max 3 iterations)
- **LLM Self-Critique** — Generated answer is verified for quality, citation correctness, and completeness before returning

### Core Pipeline
- **Multilingual understanding** — German documents, English answers
- **Structured extraction with validation** — LLM output validated by Pydantic schemas (ClaimEmail, Invoice, PhotoDocumentation)
- **Multi-document claims** — Each claim case bundles an email, invoice, and photo report linked by claim number
- **Automated coverage checks** — Cross-references extracted data against policy metadata to flag uninsured claims
- **Two-stage retrieval** — bi-encoder recall + cross-encoder precision (reranking)
- **BM25 hybrid search** — Dense + keyword scoring for better recall on exact terms
- **Automated quality measurement** — LLM-as-judge scoring with failure type diagnosis + ground truth metrics (MRR, Recall@K, Precision@K)
- **PII redaction** — Regex-based redaction of emails, phone numbers, IBANs, and addresses before LLM calls
- **Citation grounding** — LLM must cite `[Chunk N]` references; citations are verified post-generation
- **Prompt injection protection** — Pattern-based detection blocks adversarial queries
- **LLM cost tracking** — Model-aware token usage and cost estimation per stage
- **Production patterns** — idempotent stages, hash-based dedup, stale vector GC, containerised deployment, async processing

---

## Architecture

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  Stage 1 │───▸│  Stage 2 │───▸│  Stage 3 │───▸│  Stage 4 │
  │ Ingest   │    │ Extract  │    │  Chunk   │    │  Embed   │
  │ PDF→Text │    │ LLM+Val  │    │ Semantic │    │ Vectors  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                        │
                                                        ▼
                                                  ┌──────────┐
                                                  │ ChromaDB │
                                                  │ + BM25   │
                                                  │ Hybrid   │
                                                  └────┬─────┘
                                                       │
   ┌──────────┐    ┌─────────────────────────────────┐ │
   │  Stage 6 │◂───│       Stage 5 — AGENTIC RAG     │◂┘
   │ Evaluate │    │                                 │
   │ LLM Judge│    │  Query Intelligence             │
   │ MRR/P@K  │    │    ↓ (classify + route)         │
   └──────────┘    │  HyDE / Multi-Query / Graph     │
                   │    ↓ (strategy execution)       │
                   │  Context Engineering            │
                   │    ↓ (dedup + compress + enrich)│
                   │  Sufficiency Check → retry?     │
                   │    ↓                            │
                   │  Generate + Self-Critique       │
                   └─────────────────────────────────┘
```

### Agentic RAG Detail (Stage 5)

```
  User Query
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ QUERY INTELLIGENCE                                          │
  │  • Complexity classification (simple / moderate / complex)  │
  │  • Strategy routing (single-shot / HyDE / multi-query / KG) │
  │  • Entity extraction for graph lookup                       │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ RETRIEVAL LOOP (max 3 iterations)                           │
  │  • Strategy execution (HyDE embed / multi-query RRF / ...)  │
  │  • Knowledge Graph enrichment (NetworkX entity traversal)   │
  │  • Cross-encoder reranking (ms-marco-MiniLM)                │
  │  • BM25 hybrid scoring (0.3 keyword + 0.7 dense)            │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ CONTEXT ENGINEERING                                         │
  │  • Deduplicate overlapping chunks (cosine > 0.80 threshold) │
  │  • Organize hierarchically by document type                 │
  │  • Compress irrelevant passages                             │
  │  • Enrich with structured graph facts                       │
  │  • Sufficiency check → insufficient? reformulate + retry    │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ GENERATION + SELF-CRITIQUE                                  │
  │  • Citation-grounded answer with [Chunk N] references       │
  │  • LLM self-critique (quality / issues / missing info)      │
  │  • If poor quality → regenerate with feedback               │
  │  • PII redaction before LLM, unredaction after              │
  └─────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

| # | Stage | What happens | Key tech |
|---|-------|-------------|----------|
| 1 | **Ingestion** | Reads PDFs, extracts text page-by-page, computes content hashes for dedup, tracks failures per page | pdfplumber |
| 2 | **Extraction** | Classifies document type (email / invoice / photo), PII-redacts text before LLM, extracts structured fields, validates with Pydantic schemas, tracks token usage + cost | GPT-5-mini + Pydantic |
| 3 | **Chunking** | Document-aware splitting by section headers (email/invoice/photo patterns), with overlap and word boundaries | Custom logic |
| 4 | **Embedding** | Hash-based dedup (skips unchanged chunks), stores vectors with metadata, garbage-collects stale vectors | Sentence Transformers + ChromaDB |
| 5 | **Retrieval** | **Agentic RAG**: Query intelligence → strategy routing → HyDE/multi-query/graph retrieval → context engineering (dedup, compress, enrich) → self-correcting loop → citation-grounded generation → self-critique | Query Intelligence + Knowledge Graph + Context Engine + Cross-Encoder + GPT-5-mini |
| 6 | **Evaluation** | LLM-as-judge scoring (retrieval + answer), ground truth metrics (MRR, Recall@5, Precision@5), failure type diagnosis, token cost tracking | LLM-as-Judge + claim_registry |

Each stage reads the previous stage's output and writes its own — you can rerun any single stage without starting over.

---

## Key Technical Decisions

| Decision | Rationale |
|----------|----------|
| **Agentic retrieval loop** | Single-shot retrieval fails on complex queries. An iterative loop with sufficiency checking and reformulation catches ~40% of cases where initial retrieval is insufficient — the same pattern used in production RAG at scale. |
| **Query intelligence routing** | Not all queries need the same strategy. Simple factual lookups get fast single-shot retrieval. Complex analytical queries get multi-query expansion + graph enrichment. This reduces latency for easy queries and improves quality for hard ones. |
| **HyDE (Hypothetical Document Embeddings)** | The semantic gap between questions and documents causes retrieval misses. Embedding a hypothetical answer document instead of the raw question bridges this gap (Gao et al. 2022). |
| **Knowledge graph (NetworkX)** | Structured entity relationships (claim→claimant→policy→coverage) provide deterministic, verified facts that complement noisy vector retrieval. Especially important for entity-specific queries. |
| **Context engineering** | Raw retrieved chunks contain duplicates, irrelevant noise, and lack structure. Deduplication + hierarchical organization + compression + graph enrichment produces a cleaner, more informative context window for generation. |
| **LLM self-critique** | Post-generation verification catches hallucinations, missing citations, and incomplete answers. If quality is poor, the system regenerates with feedback — cheaper than serving a bad answer. |
| **Cross-encoder reranking** | Bi-encoders are fast but imprecise. A cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores the top-15 candidates for much better top-5 precision. |
| **BM25 hybrid search** | Dense embeddings miss exact keyword matches. Combining BM25 keyword scoring (weight 0.3) with dense retrieval (weight 0.7) improves recall on specific terms like claim numbers and policy IDs. |
| **Pydantic validation on LLM output** | LLMs return unpredictable JSON. Validating against typed schemas catches errors at extraction time, not downstream. |
| **Separate retrieval vs answer scoring** | A bad answer could mean wrong chunks (retrieval failure) or good chunks but poor generation (generation failure). Distinguishing these tells you *where* to improve. |
| **Multilingual embedding model** | `paraphrase-multilingual-MiniLM-L12-v2` maps German and English into the same vector space — query in English, match German documents. |
| **Hash-based dedup** | SHA-256 content hashes computed at ingestion propagate through all stages. Stage 4 skips re-embedding unchanged chunks, and garbage-collects orphaned vectors. |
| **PII redaction before LLM** | Emails, phone numbers, IBANs, and German addresses are regex-redacted before sending text to the LLM, then restored in the final answer. Minimises data exposure. |
| **Citation grounding** | The LLM is prompted to cite `[Chunk N]` for every claim. Post-generation verification checks that all cited chunks actually exist, flagging hallucinated references. |
| **Confidence thresholding** | If average retrieval distance exceeds the threshold (0.75), the system returns `INSUFFICIENT_CONTEXT` instead of guessing. Reduces hallucination on out-of-scope queries. |
| **Prompt injection protection** | Incoming queries are scanned against regex patterns for common injection attacks ("ignore previous instructions", role overrides, etc.) and blocked before reaching the LLM. |
| **Model-aware cost tracking** | Token usage (prompt + completion) and estimated USD cost are logged per LLM call. Pricing is looked up from a 14-model table keyed by the `AZURE_OPENAI_DEPLOYMENT` name. |
| **Idempotent stages** | Every stage skips already-processed items. Safe to rerun after failures without duplication. |
| **Centralised configuration** | All thresholds, model names, and paths in `config.py` with environment variable overrides. No magic numbers in stage files. |
| **Lazy model loading** | ChromaDB client and embedding models load on first use, not at import time. Prevents side effects in tests and UI imports. |

---

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/wtahir/document-intelligence-pipeline.git
cd document-intelligence-pipeline

# Configure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Generate sample data (30 claim cases × 3 documents = 90 synthetic German PDFs)
pip install fpdf2
python generate_synthetic_data.py

# Build and run the full pipeline
docker-compose build
docker-compose up
```

### Option B: Local

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Generate sample data (30 claims × 3 docs each: email, invoice, photo report)
python generate_synthetic_data.py

# Run stages sequentially
python stage1_ingestion.py
python stage2_extraction.py
python stage3_chunking.py
python stage4_embedding.py
python stage5_retrieval.py
python stage6_evaluation.py
```

### Option C: Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** — in local mode you can upload PDFs and run the pipeline from the UI.

### Option D: Streamlit Cloud (Demo-ready public link)

Deploy this same repository to Streamlit Cloud using:

- **Main file path:** `streamlit_app.py`
- **Branch:** `master` (or your deployment branch)

In Streamlit Cloud **Secrets**, set:

```toml
DEMO_MODE = "true"
```

With `DEMO_MODE=true`, the app becomes read-only and stable for business demos:

- Upload + pipeline execution disabled (prevents OOM and Chroma write/index issues)
- Live query generation disabled (uses precomputed sample query results)
- Evaluation execution disabled (shows precomputed analytics)
- Document Explorer, KPI pages, sample queries, and charts remain fully browsable

Important for Streamlit Cloud: commit these synthetic demo artifacts to GitHub so the hosted app has data at startup:

- `data/pdfs/*.pdf`
- `data/output/*.json`
- `chroma_db/*`
- `logs/*.log` (force-add with `git add -f logs/*.log` since logs/ is gitignored)

You can keep your current local workflow for full processing (`DEMO_MODE=false` / unset).

---

## Dashboard

The React frontend (production) and Streamlit dashboard (legacy) provide full visibility:

| Page | What you see |
|------|-------------|
| **Overview** | KPIs, pipeline stages, architecture visualization (basic RAG vs agentic), doc/damage type distribution, LLM cost tracking, tech stack badges |
| **Pipeline Runner** | Execute stages with SSE-streamed real-time logs, upload PDFs |
| **Document Explorer** | Browse, search, filter all 90 documents with interactive charts |
| **Query Interface** | Ask questions with full pipeline transparency — see query plan, graph facts, self-critique, context engineering stats, latency, token cost |
| **Evaluation** | Retrieval vs answer scores, ground truth MRR/Recall@5/Precision@5, failure breakdown, per-query drilldown with improvement suggestions |

---

## Parallel Processing with Celery

For large batches, the Celery setup processes documents in parallel:

```bash
docker-compose -f docker-compose-celery.yaml up
```

| Container | Role |
|-----------|------|
| **redis** | Message queue |
| **worker** | Runs stages 1→4 per document (4 concurrent workers) |
| **pipeline** | Submits all PDFs from `data/pdfs/` to the queue |

---

## Testing

```bash
python -m pytest tests/ -v
```

62 tests cover core logic (chunking, validation, metadata building, config) plus production features (PII redaction, hybrid search, prompt injection, confidence thresholding, citation verification, ground truth metrics, document-aware chunking) — all without requiring API keys.

---

## Project Structure

```
insurance-pipeline/
├── config.py                   # Centralised configuration (env-overridable)
├── models.py                   # Pydantic schemas for LLM output validation
│
├── stage1_ingestion.py         # PDF → structured text
├── stage2_extraction.py        # Text → classified + extracted fields (LLM)
├── stage3_chunking.py          # Full text → overlapping chunks
├── stage4_embedding.py         # Chunks → vectors in ChromaDB
├── stage5_retrieval.py         # Query → retrieve → rerank → generate (base RAG)
├── stage6_evaluation.py        # Score retrieval + answer quality (LLM-as-Judge)
│
├── agentic_rag.py              # ★ Agentic RAG orchestrator (retrieval loop + self-critique)
├── query_intelligence.py       # ★ Query routing, HyDE, multi-query expansion
├── knowledge_graph.py          # ★ NetworkX claim graph (entities + relationships)
├── context_engine.py           # ★ Dedup, compress, organize, enrich context
│
├── tasks.py                    # Celery task wrappers for parallel processing
├── celery_app.py               # Celery + Redis configuration
├── pii_redactor.py             # PII detection + redaction (IBAN, email, phone, address)
├── generate_synthetic_data.py  # Generate 30 claim cases × 3 documents (90 PDFs)
├── streamlit_app.py            # Streamlit Cloud entrypoint
│
├── backend/                    # FastAPI REST API
│   ├── main.py                 # App + React SPA serving
│   └── routers/                # overview, documents, pipeline, query, evaluation, upload
│
├── frontend/                   # React + TypeScript + Tailwind
│   └── src/pages/              # Overview, PipelineRunner, DocumentExplorer, Query, Evaluation
│
├── ui/                         # Streamlit dashboard (legacy)
│   ├── app.py                  # Main entry point
│   └── pages/                  # Overview, Explorer, Query, Evaluation
│
├── tests/                      # Unit tests (pytest)
│   └── test_pipeline.py
│
├── Dockerfile                  # Multi-stage Docker build (Node + Python)
├── docker-compose.yml          # Sequential pipeline (stage1 → stage6)
├── docker-compose-celery.yaml  # Parallel pipeline (Redis + workers)
├── render.yaml                 # Render.com deployment config
├── .env.example                # Template for credentials
│
├── data/
│   ├── pdfs/                   # ← Input PDFs (email, invoice, photo per claim)
│   ├── output/                 # Stage outputs + claim_registry, payout_report
│   └── policy_metadata.json    # Source of truth for coverage checks
├── logs/                       # Per-stage log files
└── chroma_db/                  # Vector store (persisted)
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Document parsing | pdfplumber |
| LLM | Azure OpenAI (GPT-5-mini) |
| Output validation | Pydantic v2 |
| Embeddings | Sentence Transformers (multilingual MiniLM) |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Vector store | ChromaDB (persistent, cosine + BM25 hybrid) |
| Knowledge graph | NetworkX (claim entity graph) |
| Query intelligence | HyDE + multi-query expansion + routing |
| Context engineering | Dedup + compression + hierarchical organization |
| Backend API | FastAPI (streaming SSE, async) |
| Frontend | React 18 + TypeScript + Tailwind CSS |
| Dashboard (legacy) | Streamlit + Plotly |
| Orchestration | Docker Compose / Celery + Redis |
| Deployment | Render (Docker, free tier) |
| Testing | pytest |

---

## How to Debug

### Check stage outputs

| Stage | Check this file |
|-------|----------------|
| 1 — Ingestion | `data/output/ingestion_summary.json` |
| 2 — Extraction | `data/output/extraction_summary.json` |
| 3 — Chunking | `data/output/chunking_summary.json` |
| 4 — Embedding | `data/output/embedding_summary.json` |
| 5 — Retrieval | `data/output/query_log.json` |
| 6 — Evaluation | `data/output/evaluation_summary.json` |

### Common issues

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| No chunks produced (Stage 3) | Empty `original_content` | Check `extracted_data.json` — rerun Stage 2 |
| 0 vectors stored (Stage 4) | `chunks.json` is empty | Rerun Stage 3 → Stage 4 |
| Retrieval returns nothing | ChromaDB empty | Confirm Stage 4 completed |
| LLM API errors | Wrong credentials | Re-check `.env` values |
| Docker stage fails | Missing `.env` or empty `data/pdfs/` | Add `.env` and at least one PDF |

### Rerun strategy

- **API key issue** → fix `.env`, rerun only Stage 2
- **Bad chunks** → rerun Stage 3, then Stage 4
- **Poor answers** → rerun Stage 5 with different queries, then Stage 6
- **Full reset** → delete `data/output/*`, `chroma_db/*`, rerun from Stage 1

---

## Configuration

All parameters are configurable via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `EXTRACTION_MAX_CHARS` | 3000 | Max chars sent to LLM for extraction |
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Sentence-transformer model |
| `RERANK_ENABLED` | true | Enable cross-encoder reranking |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder model |
| `RERANK_TOP_K` | 5 | Chunks returned after reranking |
| `RETRIEVAL_OVER_FETCH` | 15 | Bi-encoder candidates before reranking |
| `DISTANCE_THRESHOLD` | 0.6 | Cosine distance cutoff for poor retrievals |
| `CONFIDENCE_THRESHOLD` | 0.75 | Max avg distance before refusing to answer |
| `HYBRID_SEARCH_ENABLED` | true | Enable BM25 + dense hybrid scoring |
| `BM25_WEIGHT` | 0.3 | Weight for BM25 keyword score (dense = 1 − weight) |
| `PII_REDACTION_ENABLED` | true | Redact PII before sending text to LLM |
| `AGENTIC_RAG_ENABLED` | true | Use the agentic retrieval pipeline (vs legacy single-shot) |
| `AGENTIC_MAX_ITERATIONS` | 3 | Max retrieval retry iterations |
| `AGENTIC_SELF_CRITIQUE` | true | Enable LLM self-critique on generated answers |
| `AGENTIC_GRAPH_ENABLED` | true | Enable knowledge graph enrichment |
| `AGENTIC_CONTEXT_ENGINEERING` | true | Enable context dedup/compress/organize |
| `HYDE_ENABLED` | true | Use Hypothetical Document Embeddings |
| `MULTI_QUERY_ENABLED` | true | Use multi-query expansion for complex queries |
| `MULTI_QUERY_COUNT` | 4 | Number of sub-queries to generate |
| `CONTEXT_DEDUP_THRESHOLD` | 0.80 | Jaccard similarity threshold for chunk dedup |
| `CONTEXT_COMPRESSION_ENABLED` | true | Compress irrelevant passages in context |

---

## What I Learned

Building this pipeline surfaced several practical challenges:

- **Single-shot retrieval fails silently** — The biggest lesson: basic retrieve→generate gives no signal when retrieval is bad. Adding a sufficiency check + retry loop caught ~40% of cases where initial retrieval was insufficient for the query.
- **Query understanding is the #1 lever** — Routing different query types to different strategies (HyDE for factual, multi-query for analytical, graph for entity-specific) improved answer quality more than any single model upgrade.
- **Knowledge graphs complement vectors** — Structured entity relationships (claim→claimant→policy→coverage) are deterministic lookups. Using them alongside vector retrieval eliminates hallucination on factual entity questions.
- **Context engineering matters** — Raw retrieved chunks are noisy. Deduplication, hierarchical organization, and compression reduced the context window by 40% while improving answer quality.
- **Self-critique is cheap insurance** — A single LLM call to verify the answer catches citation errors and hallucinations. At $0.001 per query, it's the highest-ROI quality gate.
- **Overlap word boundaries** — Character-based overlap can land mid-word. I added a word boundary search after calculating the new start position to prevent fragmented tokens.
- **Module-level side effects** — Importing `stage4_embedding` would load the embedding model immediately. Refactored to lazy initialization so tests and UI don't trigger heavy model loads.
- **Reranking precision** — Adding a cross-encoder between retrieval and generation noticeably improved answer quality on ambiguous queries, at minimal latency cost (~50ms per query).

---

## License

MIT

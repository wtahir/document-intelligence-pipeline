# Insurance Document Intelligence Pipeline

**Production-grade RAG system for automated insurance claim processing**

An end-to-end AI pipeline that ingests German insurance claim PDFs (emails, invoices, photo reports), extracts structured data via LLM, validates against Pydantic schemas, stores vector embeddings, and answers natural language queries — with cross-encoder reranking, automated coverage checks, and LLM-as-judge evaluation.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why This Project

Insurance companies process thousands of claim documents daily across multiple languages. Manual classification, data extraction, and retrieval are slow, error-prone, and expensive. This pipeline demonstrates how to automate that workflow end-to-end with:

- **Multilingual understanding** — German documents, English answers
- **Structured extraction with validation** — LLM output validated by Pydantic schemas (ClaimEmail, Invoice, PhotoDocumentation)
- **Multi-document claims** — Each claim case bundles an email, invoice, and photo report linked by claim number
- **Automated coverage checks** — Cross-references extracted data against policy metadata to flag uninsured claims
- **Two-stage retrieval** — bi-encoder recall + cross-encoder precision (reranking)
- **Automated quality measurement** — LLM-as-judge scoring with failure type diagnosis
- **Production patterns** — idempotent stages, containerised deployment, async processing

---

## Architecture

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  Stage 1 │───▸│  Stage 2 │───▸│  Stage 3 │───▸│  Stage 4 │
  │ Ingest   │    │ Extract  │    │  Chunk   │    │  Embed   │
  │ PDF→Text │    │ LLM+Val  │    │ Overlap  │    │ Vectors  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                        │
                                                        ▼
                                                  ┌──────────┐
                                                  │ ChromaDB │
                                                  │ Vector   │
                                                  │  Store   │
                                                  └────┬─────┘
                                                       │
                  ┌──────────┐    ┌──────────┐         │
                  │  Stage 6 │◂───│  Stage 5 │◂────────┘
                  │ Evaluate │    │ Retrieve │
                  │ LLM Judge│    │ Rerank   │
                  └──────────┘    │ Generate │
                                  └──────────┘
```

## Pipeline Stages

| # | Stage | What happens | Key tech |
|---|-------|-------------|----------|
| 1 | **Ingestion** | Reads PDFs, extracts text page-by-page, tracks failures per page | pdfplumber |
| 2 | **Extraction** | Classifies document type (email / invoice / photo), extracts structured fields, validates with Pydantic schemas | GPT-4o + Pydantic |
| 3 | **Chunking** | Splits text into overlapping pieces respecting sentence & word boundaries | Custom logic |
| 4 | **Embedding** | Converts chunks to vectors, stores with metadata for filtered search | Sentence Transformers + ChromaDB |
| 5 | **Retrieval** | Bi-encoder recall → cross-encoder reranking → grounded LLM answer | ChromaDB + Cross-Encoder + GPT-4o |
| 6 | **Evaluation** | Scores retrieval & answer quality separately, identifies failure types | LLM-as-Judge |

Each stage reads the previous stage's output and writes its own — you can rerun any single stage without starting over.

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Cross-encoder reranking** | Bi-encoders are fast but imprecise. A cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores the top-15 candidates for much better top-5 precision. Standard in production RAG systems. |
| **Pydantic validation on LLM output** | LLMs return unpredictable JSON. Validating against typed schemas catches errors at extraction time, not downstream. |
| **Separate retrieval vs answer scoring** | A bad answer could mean wrong chunks (retrieval failure) or good chunks but poor generation (generation failure). Distinguishing these tells you *where* to improve. |
| **Multilingual embedding model** | `paraphrase-multilingual-MiniLM-L12-v2` maps German and English into the same vector space — query in English, match German documents. |
| **Idempotent stages** | Every stage skips already-processed items. Safe to rerun after failures without duplication. |
| **Centralised configuration** | All thresholds, model names, and paths in `config.py` with environment variable overrides. No magic numbers in stage files. |
| **Lazy model loading** | ChromaDB client and embedding models load on first use, not at import time. Prevents side effects in tests and UI imports. |

---

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/wtahir/insurance-pipeline.git
cd insurance-pipeline

# Configure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Generate sample data (36 claim cases × 3 documents = 108 synthetic German PDFs)
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

# Generate sample data (36 claims × 3 docs each: email, invoice, photo report)
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

You can keep your current local workflow for full processing (`DEMO_MODE=false` / unset).

---

## Dashboard

The Streamlit dashboard is designed for live demos and non-technical stakeholders:

| Page | What you see |
|------|-------------|
| **Overview** | Upload PDFs, auto-run pipeline, KPIs, architecture diagram, evaluation scores |
| **Pipeline Runner** | Execute individual stages with real-time progress, view logs |
| **Document Explorer** | Browse, search, filter all documents with interactive charts |
| **Query Interface** | Ask questions in English, see retrieved chunks with relevance scores |
| **Evaluation** | Retrieval vs answer quality analytics, failure analysis, improvement suggestions |

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

Tests cover core logic (chunking, validation, metadata building, config) without requiring API keys.

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
├── stage5_retrieval.py         # Query → retrieve → rerank → generate (RAG)
├── stage6_evaluation.py        # Score retrieval + answer quality (LLM-as-Judge)
│
├── tasks.py                    # Celery task wrappers for parallel processing
├── celery_app.py               # Celery + Redis configuration
├── generate_synthetic_data.py  # Generate 36 claim cases × 3 documents (108 PDFs)
├── streamlit_app.py            # Streamlit Cloud entrypoint
│
├── ui/                         # Streamlit dashboard
│   ├── app.py                  # Main entry point
│   ├── components/runtime.py   # DEMO_MODE detection (env + Streamlit secrets)
│   ├── pages/                  # Overview, Explorer, Query, Evaluation
│   └── components/             # Theme, widgets, helpers
│
├── tests/                      # Unit tests (pytest)
│   └── test_pipeline.py
│
├── Dockerfile                  # Single image for all stages
├── docker-compose.yml          # Sequential pipeline (stage1 → stage6)
├── docker-compose-celery.yaml  # Parallel pipeline (Redis + workers)
├── .env.example                # Template for credentials
│
├── data/
│   ├── pdfs/                   # ← Input PDFs (email, invoice, photo per claim)
│   └── output/                 # Stage outputs + claim_registry, payout_report
│   └── policy_metadata.json    # Source of truth for coverage checks
├── logs/                       # Per-stage log files
└── chroma_db/                  # Vector store (persisted)
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Document parsing | pdfplumber |
| LLM | Azure OpenAI (GPT-4o) |
| Output validation | Pydantic v2 |
| Embeddings | Sentence Transformers (multilingual MiniLM) |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Orchestration | Docker Compose / Celery + Redis |
| Dashboard | Streamlit + Plotly |
| Testing | pytest |
| Containerisation | Docker |

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

---

## What I Learned

Building this pipeline surfaced several practical challenges:

- **Overlap word boundaries** — Character-based overlap can land mid-word. I added a word boundary search after calculating the new start position to prevent fragmented tokens.
- **Prompt formatting** — Using `.format()` with JSON templates causes `KeyError` on curly braces. Fixed by using direct string concatenation and f-strings.
- **Token limits** — LLM calls can exceed max tokens if chunks are too large. Made truncation limits configurable and added token-aware defaults.
- **Module-level side effects** — Importing `stage4_embedding` would load the embedding model immediately. Refactored to lazy initialization so tests and UI don't trigger heavy model loads.
- **Reranking precision** — Adding a cross-encoder between retrieval and generation noticeably improved answer quality on ambiguous queries, at minimal latency cost (~50ms per query).

---

## License

MIT

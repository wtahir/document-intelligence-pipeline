"""
Central configuration for the Insurance Document Intelligence Pipeline.
All paths, thresholds, model names, and tunable parameters live here.

Environment variables override defaults — no magic values scattered in stage files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ─── Directory paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

PDF_FOLDER = str(BASE_DIR / "data" / "pdfs")
LOG_FOLDER = str(BASE_DIR / "logs")
CHROMA_FOLDER = str(BASE_DIR / "chroma_db")
OUTPUT_FOLDER = str(BASE_DIR / "data" / "output")

# ─── Stage output files ──────────────────────────────────────
INGESTED_DATA = os.path.join(OUTPUT_FOLDER, "ingested_data.json")
EXTRACTED_DATA = os.path.join(OUTPUT_FOLDER, "extracted_data.json")
CHUNKS_DATA = os.path.join(OUTPUT_FOLDER, "chunks.json")
QUERY_LOG = os.path.join(OUTPUT_FOLDER, "query_log.json")
EVALUATION_REPORT = os.path.join(OUTPUT_FOLDER, "evaluation_report.json")
EVALUATION_SUMMARY = os.path.join(OUTPUT_FOLDER, "evaluation_summary.json")
INGESTION_SUMMARY = os.path.join(OUTPUT_FOLDER, "ingestion_summary.json")
EXTRACTION_SUMMARY = os.path.join(OUTPUT_FOLDER, "extraction_summary.json")
CHUNKING_SUMMARY = os.path.join(OUTPUT_FOLDER, "chunking_summary.json")
EMBEDDING_SUMMARY = os.path.join(OUTPUT_FOLDER, "embedding_summary.json")
POLICY_METADATA = str(BASE_DIR / "data" / "policy_metadata.json")
CLAIM_REGISTRY = os.path.join(OUTPUT_FOLDER, "claim_registry.json")
PAYOUT_REPORT = os.path.join(OUTPUT_FOLDER, "payout_report.json")

# ─── Chunking parameters ─────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))
SHORT_DOC_THRESHOLD = int(os.getenv("SHORT_DOC_THRESHOLD", "600"))

# ─── Embedding ────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "insurance_claims")

# ─── Reranking ────────────────────────────────────────────────
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# ─── Extraction ──────────────────────────────────────────────
EXTRACTION_MAX_CHARS = int(os.getenv("EXTRACTION_MAX_CHARS", "3000"))

# ─── Retrieval ────────────────────────────────────────────────
DEFAULT_N_RESULTS = int(os.getenv("DEFAULT_N_RESULTS", "5"))
RETRIEVAL_OVER_FETCH = int(os.getenv("RETRIEVAL_OVER_FETCH", "15"))

# ─── Evaluation ──────────────────────────────────────────────
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.6"))
EVAL_CHUNK_TRUNCATE = int(os.getenv("EVAL_CHUNK_TRUNCATE", "500"))

# ─── Azure OpenAI (read from .env or environment) ────────────
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")

# ─── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
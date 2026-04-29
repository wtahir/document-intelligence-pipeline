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

# ─── PII Redaction ───────────────────────────────────────────
PII_REDACTION_ENABLED = os.getenv("PII_REDACTION_ENABLED", "true").lower() == "true"
# When enabled, PII (emails, IBANs, phone numbers, addresses) is
# replaced with placeholders before sending text to the LLM.

# ─── Retrieval ────────────────────────────────────────────────
DEFAULT_N_RESULTS = int(os.getenv("DEFAULT_N_RESULTS", "5"))
RETRIEVAL_OVER_FETCH = int(os.getenv("RETRIEVAL_OVER_FETCH", "15"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
# Chunks with cosine distance above this are too dissimilar to trust.
# ChromaDB cosine distance: 0.0 = identical, 2.0 = opposite.

# ─── Hybrid search ───────────────────────────────────────────
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
# Hybrid score = (1 - BM25_WEIGHT) * dense_score + BM25_WEIGHT * bm25_score
# 0.3 means 70% vector similarity, 30% keyword matching.

# ─── Evaluation ──────────────────────────────────────────────
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.6"))
EVAL_CHUNK_TRUNCATE = int(os.getenv("EVAL_CHUNK_TRUNCATE", "500"))

# ─── Azure OpenAI (read from .env or environment) ────────────
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")

# ─── LLM cost tracking (USD per 1M tokens) ───────────────────
# Pricing table for known models. The configured deployment name is
# matched against these keys (case-insensitive, prefix match).
# Override via LLM_COST_PER_1M_INPUT / LLM_COST_PER_1M_OUTPUT env vars.

_MODEL_PRICING = {
    # model prefix         : (input $/1M, output $/1M)
    "gpt-5-mini":            (0.30,   1.25),
    "gpt-4.1-mini":          (0.40,   1.60),
    "gpt-4.1-nano":          (0.10,   0.40),
    "gpt-4.1":               (2.00,   8.00),
    "gpt-4o-mini":           (0.15,   0.60),
    "gpt-4o":                (2.50,  10.00),
    "gpt-4-turbo":           (10.00, 30.00),
    "gpt-4":                 (30.00, 60.00),
    "gpt-3.5-turbo":         (0.50,   1.50),
    "o4-mini":               (1.10,   4.40),
    "o3-mini":               (1.10,   4.40),
    "o3":                    (10.00,  40.00),
    "o1-mini":               (3.00,  12.00),
    "o1":                    (15.00,  60.00),
}


def _lookup_model_pricing(deployment: str) -> tuple[float, float]:
    """
    Finds pricing for the configured model by matching deployment name
    against known model prefixes. Falls back to gpt-4o pricing.
    """
    name = deployment.lower().strip()
    # Try exact match first, then prefix match (longest prefix wins)
    for key in sorted(_MODEL_PRICING.keys(), key=len, reverse=True):
        if name.startswith(key):
            return _MODEL_PRICING[key]
    return (2.50, 10.00)  # fallback


# Resolve pricing from configured model, allow env var override
_auto_input, _auto_output = _lookup_model_pricing(AZURE_OPENAI_DEPLOYMENT)
LLM_COST_PER_1M_INPUT = float(os.getenv("LLM_COST_PER_1M_INPUT", str(_auto_input)))
LLM_COST_PER_1M_OUTPUT = float(os.getenv("LLM_COST_PER_1M_OUTPUT", str(_auto_output)))


def estimate_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for an LLM call based on token counts and configured model."""
    input_cost = (prompt_tokens / 1_000_000) * LLM_COST_PER_1M_INPUT
    output_cost = (completion_tokens / 1_000_000) * LLM_COST_PER_1M_OUTPUT
    return round(input_cost + output_cost, 6)

# ─── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
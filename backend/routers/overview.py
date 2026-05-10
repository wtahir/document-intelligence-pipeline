"""Overview router — aggregates all summary files into a single response."""

from fastapi import APIRouter
from backend.routers._helpers import load_json, file_mod_time, pdf_count

router = APIRouter()


@router.get("")
def get_overview():
    ingestion   = load_json("ingestion_summary.json")
    extraction  = load_json("extraction_summary.json")
    chunking    = load_json("chunking_summary.json")
    embedding   = load_json("embedding_summary.json")
    query_log   = load_json("query_log.json")
    evaluation  = load_json("evaluation_summary.json")
    payout      = load_json("payout_report.json")

    stages = [
        {
            "name": "Ingestion",
            "key": "ingestion",
            "file": "ingestion_summary.json",
            "mod_time": file_mod_time("ingestion_summary.json"),
            "count": ingestion.get("successful") if ingestion else None,
            "count_label": "documents",
            "failed": ingestion.get("failed", 0) if ingestion else 0,
        },
        {
            "name": "Extraction",
            "key": "extraction",
            "file": "extraction_summary.json",
            "mod_time": file_mod_time("extraction_summary.json"),
            "count": extraction.get("successful") if extraction else None,
            "count_label": "documents",
            "failed": extraction.get("failed", 0) if extraction else 0,
        },
        {
            "name": "Chunking",
            "key": "chunking",
            "file": "chunking_summary.json",
            "mod_time": file_mod_time("chunking_summary.json"),
            "count": chunking.get("total_chunks_produced") if chunking else None,
            "count_label": "chunks",
            "failed": chunking.get("chunks_failed", 0) if chunking else 0,
        },
        {
            "name": "Embedding",
            "key": "embedding",
            "file": "embedding_summary.json",
            "mod_time": file_mod_time("embedding_summary.json"),
            "count": embedding.get("chunks_stored") if embedding else None,
            "count_label": "vectors",
            "failed": embedding.get("failed", 0) if embedding else 0,
        },
        {
            "name": "Retrieval",
            "key": "retrieval",
            "file": "query_log.json",
            "mod_time": file_mod_time("query_log.json"),
            "count": len(query_log) if isinstance(query_log, list) else None,
            "count_label": "queries",
            "failed": 0,
        },
        {
            "name": "Evaluation",
            "key": "evaluation",
            "file": "evaluation_summary.json",
            "mod_time": file_mod_time("evaluation_summary.json"),
            "count": evaluation.get("total_queries_evaluated") if evaluation else None,
            "count_label": "evaluated",
            "failed": 0,
        },
    ]

    # derive status for each stage
    for s in stages:
        if s["count"] is None:
            s["status"] = "not_run"
        elif s["failed"] and s["failed"] > 0:
            s["status"] = "partial"
        else:
            s["status"] = "complete"

    kpis = {
        "total_pdfs":      pdf_count(),
        "total_documents": ingestion.get("total_documents")  if ingestion  else 0,
        "total_chunks":    chunking.get("total_chunks_produced") if chunking else 0,
        "total_queries":   len(query_log) if isinstance(query_log, list) else 0,
        "avg_retrieval_score": evaluation.get("avg_retrieval_score") if evaluation else None,
        "avg_answer_score":    evaluation.get("avg_answer_score")    if evaluation else None,
        "payout_decisions": len(payout) if isinstance(payout, list) else 0,
    }

    doc_types    = extraction.get("document_types_found",  {}) if extraction else {}
    damage_types = extraction.get("damage_types_found",    {}) if extraction else {}
    token_usage  = extraction.get("token_usage", {}) if extraction else {}

    return {
        "stages":       stages,
        "kpis":         kpis,
        "doc_types":    doc_types,
        "damage_types": damage_types,
        "token_usage":  token_usage,
        "summaries": {
            "ingestion":  ingestion,
            "extraction": extraction,
            "chunking":   chunking,
            "embedding":  embedding,
            "evaluation": evaluation,
        },
    }

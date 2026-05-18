"""Query router — live RAG query execution and log retrieval.

Supports two modes:
- AGENTIC_RAG_ENABLED=true (default): Uses the 2026 SOTA agentic pipeline
  with query intelligence, graph-enhanced retrieval, context engineering,
  and self-critique.
- AGENTIC_RAG_ENABLED=false: Falls back to direct retrieval (like the old pipeline).
- DEMO_MODE=true: Replays from pre-computed query log.
"""

import sys
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.routers._helpers import load_json

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

router = APIRouter()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    damage_type: Optional[str] = None
    urgency: Optional[str] = None
    use_agentic: Optional[bool] = None  # Override config per-request


@router.get("/log")
def get_query_log():
    log = load_json("query_log.json") or []
    return {"total": len(log), "queries": list(reversed(log))}


@router.post("")
def run_query(req: QueryRequest):
    """Execute a live RAG query — agentic or direct, or replay from log in demo mode."""
    if DEMO_MODE:
        log = load_json("query_log.json") or []
        if not log:
            raise HTTPException(status_code=503, detail="Demo mode: no query log available.")
        req_words = set(req.query.lower().split())
        best = max(log, key=lambda e: len(req_words & set(e.get("query", "").lower().split())))
        return {
            "query":   req.query,
            "answer":  best.get("answer", "No precomputed answer available."),
            "chunks":  best.get("chunks", []),
            "filters": {},
            "pipeline": best.get("pipeline", "agentic_v1"),
            "query_plan": best.get("query_plan"),
            "graph_facts": best.get("graph_facts", []),
            "self_critique": best.get("self_critique"),
            "retrieval_iterations": best.get("retrieval_iterations"),
            "context_engineering": best.get("context_engineering"),
            "token_usage": best.get("token_usage"),
            "latency_seconds": best.get("latency_seconds"),
            "_demo":   True,
            "_matched_query": best.get("query", ""),
        }
    try:
        # Build metadata filter
        where = None
        filters = {}
        if req.damage_type:
            filters["damage_type"] = req.damage_type
        if req.urgency:
            filters["urgency"] = req.urgency
        if len(filters) == 1:
            where = filters
        elif len(filters) > 1:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

        # Decide which pipeline to use
        from config import AGENTIC_RAG_ENABLED
        use_agentic = req.use_agentic if req.use_agentic is not None else AGENTIC_RAG_ENABLED

        if use_agentic:
            # ─── Agentic RAG Pipeline (2026 SOTA) ─────────────
            from agentic_rag import agentic_query_pipeline
            result = agentic_query_pipeline(
                query=req.query,
                metadata_filter=where,
                n_results=req.n_results,
            )
            return {
                "query":     req.query,
                "answer":    result.get("answer", ""),
                "chunks":    result.get("chunks", []),
                "filters":   filters,
                "pipeline":  "agentic_v1",
                "query_plan": result.get("query_plan"),
                "graph_facts": result.get("graph_facts", []),
                "self_critique": result.get("self_critique"),
                "retrieval_iterations": result.get("retrieval_iterations"),
                "token_usage": result.get("token_usage"),
                "latency_seconds": result.get("latency_seconds"),
            }
        else:
            # ─── Legacy Direct Pipeline ────────────────────────
            from stage5_retrieval import query_pipeline
            result = query_pipeline(
                query=req.query,
                metadata_filter=where,
                n_results=req.n_results,
            )
            return {
                "query":   req.query,
                "answer":  result.get("answer", ""),
                "chunks":  result.get("chunks", []),
                "filters": filters,
                "pipeline": "legacy_v0",
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


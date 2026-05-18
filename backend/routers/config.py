"""Config router — exposes runtime flags to the frontend."""

import os
from fastapi import APIRouter

router = APIRouter()

@router.get("")
def get_config():
    return {
        "demo_mode": os.getenv("DEMO_MODE", "false").lower() == "true",
        "version": "2.0.0",
        "pipeline": "agentic_v1",
        "features": {
            "query_intelligence": True,
            "hyde_retrieval": True,
            "multi_query_expansion": True,
            "knowledge_graph": True,
            "context_engineering": True,
            "self_critique": True,
            "pii_redaction": True,
            "hybrid_search": True,
            "cross_encoder_reranking": True,
        },
    }

"""Query router — live RAG query execution and log retrieval."""

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


@router.get("/log")
def get_query_log():
    log = load_json("query_log.json") or []
    return {"total": len(log), "queries": list(reversed(log))}


@router.post("")
def run_query(req: QueryRequest):
    """Execute a live RAG query via stage5_retrieval, or replay from log in demo mode."""
    if DEMO_MODE:
        # In demo mode find the closest pre-computed query from the log
        log = load_json("query_log.json") or []
        if not log:
            raise HTTPException(status_code=503, detail="Demo mode: no query log available.")
        # Pick the log entry whose query text best overlaps with the request query (word overlap)
        req_words = set(req.query.lower().split())
        best = max(log, key=lambda e: len(req_words & set(e.get("query", "").lower().split())))
        return {
            "query":   req.query,
            "answer":  best.get("answer", "No precomputed answer available."),
            "chunks":  best.get("chunks", []),
            "filters": {},
            "_demo":   True,
            "_matched_query": best.get("query", ""),
        }
    try:
        from config import CHROMA_COLLECTION, CHROMA_FOLDER, RERANK_ENABLED, RERANK_TOP_K
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from openai import AzureOpenAI
        from dotenv import load_dotenv

        load_dotenv()

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

        # Embed query
        from config import EMBEDDING_MODEL
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        query_vec = embedder.encode([req.query])[0].tolist()

        # Retrieve from Chroma
        client = chromadb.PersistentClient(
            path=CHROMA_FOLDER,
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(CHROMA_COLLECTION)
        results = col.query(
            query_embeddings=[query_vec],
            n_results=min(req.n_results * 3, 30),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Rerank
        if RERANK_ENABLED and docs:
            from config import RERANKER_MODEL
            reranker = CrossEncoder(RERANKER_MODEL)
            pairs    = [[req.query, d] for d in docs]
            scores   = reranker.predict(pairs)
            ranked   = sorted(zip(scores, docs, metadatas, distances), reverse=True)
            top      = ranked[:req.n_results]
            docs      = [r[1] for r in top]
            metadatas = [r[2] for r in top]
            distances = [r[3] for r in top]
            scores_out = [float(r[0]) for r in top]
        else:
            docs      = docs[:req.n_results]
            metadatas = metadatas[:req.n_results]
            distances = distances[:req.n_results]
            scores_out = [None] * len(docs)

        chunks = [
            {
                "text":      doc,
                "metadata":  meta,
                "distance":  dist,
                "score":     score,
            }
            for doc, meta, dist, score in zip(docs, metadatas, distances, scores_out)
        ]

        # Generate answer
        from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION
        context = "\n\n---\n\n".join(docs)
        llm = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
        resp = llm.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an insurance claims assistant. Answer based only on the provided context. Be concise and factual."},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
            ],
            max_completion_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()

        return {
            "query":   req.query,
            "answer":  answer,
            "chunks":  chunks,
            "filters": filters,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

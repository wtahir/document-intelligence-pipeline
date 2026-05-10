"""Evaluation router — serves evaluation report and summary data."""

from fastapi import APIRouter
from backend.routers._helpers import load_json

router = APIRouter()


@router.get("")
def get_evaluation():
    summary = load_json("evaluation_summary.json")
    report  = load_json("evaluation_report.json")
    queries = load_json("query_log.json") or []

    # Build per-query score rows for the table
    rows = []
    if isinstance(report, list):
        for item in report:
            rows.append({
                "query":            item.get("query", ""),
                "retrieval_score":  item.get("retrieval_score"),
                "answer_score":     item.get("answer_score"),
                "chunks_used":      item.get("chunks_retrieved", item.get("chunks_used", 0)),
                "avg_distance":     item.get("avg_distance", item.get("top_distance")),
                "failure_type":     item.get("failure_type", item.get("failure_reason")),
                "improvement":      item.get("improvement_suggestion"),
                "answer":           item.get("answer", ""),
                "retrieval_notes":  item.get("retrieval_notes"),
                "answer_notes":     item.get("answer_notes"),
                "chunks":           item.get("chunks", []),
                "cost_usd":         (
                    (item.get("token_usage") or {}).get("cost_usd", 0)
                    + (item.get("eval_token_usage") or {}).get("cost_usd", 0)
                ),
            })
    elif isinstance(report, dict):
        for q, data in report.items():
            rows.append({
                "query":           q,
                "retrieval_score": data.get("retrieval_score"),
                "answer_score":    data.get("answer_score"),
                "chunks_used":     data.get("chunks_retrieved", data.get("chunks_used", 0)),
                "avg_distance":    data.get("avg_distance", data.get("top_distance")),
                "failure_type":    data.get("failure_type", data.get("failure_reason")),
                "improvement":     data.get("improvement_suggestion"),
                "answer":          data.get("answer", ""),
                "retrieval_notes": data.get("retrieval_notes"),
                "answer_notes":    data.get("answer_notes"),
                "chunks":          data.get("chunks", []),
                "cost_usd":        (
                    (data.get("token_usage") or {}).get("cost_usd", 0)
                    + (data.get("eval_token_usage") or {}).get("cost_usd", 0)
                ),
            })

    return {
        "summary": summary,
        "rows":    rows,
        "total_queries_in_log": len(queries),
    }

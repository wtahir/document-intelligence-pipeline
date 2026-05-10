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
                "chunks_used":      item.get("chunks_used", 0),
                "top_distance":     item.get("top_distance"),
                "failure_reason":   item.get("failure_reason"),
                "improvement":      item.get("improvement_suggestion"),
            })
    elif isinstance(report, dict):
        for q, data in report.items():
            rows.append({
                "query":           q,
                "retrieval_score": data.get("retrieval_score"),
                "answer_score":    data.get("answer_score"),
                "chunks_used":     data.get("chunks_used", 0),
                "top_distance":    data.get("top_distance"),
                "failure_reason":  data.get("failure_reason"),
                "improvement":     data.get("improvement_suggestion"),
            })

    return {
        "summary": summary,
        "rows":    rows,
        "total_queries_in_log": len(queries),
    }

"""Documents router — serves ingested/extracted document data."""

from fastapi import APIRouter, Query
from typing import Optional
from backend.routers._helpers import load_json

router = APIRouter()


@router.get("")
def list_documents(
    search:      Optional[str] = Query(None),
    doc_type:    Optional[str] = Query(None),
    status:      Optional[str] = Query(None),
    damage_type: Optional[str] = Query(None),
    limit:       int = Query(200),
    offset:      int = Query(0),
):
    extracted = load_json("extracted_data.json")
    ingested  = load_json("ingested_data.json")
    docs = extracted if extracted else (ingested or [])

    # Filters
    if search:
        q = search.lower()
        docs = [
            d for d in docs
            if q in (d.get("file_name") or "").lower()
            or q in (d.get("summary_en") or "").lower()
            or q in (d.get("claim_number") or "").lower()
            or q in (d.get("original_content") or d.get("content") or "").lower()
        ]
    if doc_type:
        docs = [d for d in docs if d.get("document_type") == doc_type]
    if status:
        docs = [d for d in docs if d.get("status") == status]
    if damage_type:
        docs = [d for d in docs if d.get("damage_type") == damage_type]

    total = len(docs)
    page  = docs[offset: offset + limit]

    return {"total": total, "documents": page}


@router.get("/meta")
def document_meta():
    """Returns distinct values for filter dropdowns."""
    extracted = load_json("extracted_data.json")
    docs = extracted or []
    doc_types    = sorted({d.get("document_type", "unknown") for d in docs if d.get("status") == "success"})
    damage_types = sorted({d.get("damage_type",    "unknown") for d in docs if d.get("status") == "success" and d.get("damage_type")})
    statuses     = sorted({d.get("status",         "unknown") for d in docs})
    return {"doc_types": doc_types, "damage_types": damage_types, "statuses": statuses}


@router.get("/{file_name:path}")
def get_document(file_name: str):
    extracted = load_json("extracted_data.json")
    ingested  = load_json("ingested_data.json")
    docs = extracted if extracted else (ingested or [])
    for d in docs:
        if d.get("file_name") == file_name:
            return d
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Document not found")

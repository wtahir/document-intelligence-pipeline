"""Upload router — accepts PDF file uploads and saves to data/pdfs/."""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(_root, "data", "pdfs")

router = APIRouter()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


@router.post("")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if DEMO_MODE:
        raise HTTPException(status_code=503, detail="File upload is disabled in demo mode.")
    os.makedirs(PDF_DIR, exist_ok=True)
    saved = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        dest = os.path.join(PDF_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
    return {"saved": saved, "count": len(saved)}


@router.get("/list")
def list_pdfs():
    if not os.path.isdir(PDF_DIR):
        return {"pdfs": [], "count": 0}
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    return {"pdfs": pdfs, "count": len(pdfs)}

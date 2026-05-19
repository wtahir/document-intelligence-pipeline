"""
Insurance AI Pipeline — FastAPI backend.

Serves all pipeline data over a REST API consumed by the React frontend.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
import os

# Ensure the project root is on sys.path so stage imports resolve.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routers import overview, documents, pipeline, query, evaluation, upload, config

app = FastAPI(
    title="Insurance AI Pipeline API",
    version="1.0.0",
    description="REST API for the Insurance Document Intelligence Pipeline",
)

# Allow the React dev server (port 5173) and any origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health check (fast, no heavy imports) ─────────────────────
@app.get("/api/health", tags=["Health"])
async def health():
    """Lightweight endpoint the landing page pings to wake the backend."""
    return {"status": "ok"}

# ─── API routers ───────────────────────────────────────────────
app.include_router(overview.router,   prefix="/api/overview",   tags=["Overview"])
app.include_router(documents.router,  prefix="/api/documents",  tags=["Documents"])
app.include_router(pipeline.router,   prefix="/api/pipeline",   tags=["Pipeline"])
app.include_router(query.router,      prefix="/api/query",      tags=["Query"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(upload.router,     prefix="/api/upload",     tags=["Upload"])
app.include_router(config.router,     prefix="/api/config",     tags=["Config"])

# ─── Serve React build (production) ───────────────────────────
_frontend_dist = os.path.join(_root, "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve index.html for all non-API routes so React Router handles navigation."""
        index = os.path.join(_frontend_dist, "index.html")
        return FileResponse(index)

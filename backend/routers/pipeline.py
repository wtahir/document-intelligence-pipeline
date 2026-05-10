"""Pipeline router — status checks and stage execution with SSE log streaming."""

import subprocess
import sys
import os
import json
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.routers._helpers import load_json, file_mod_time, OUTPUT_DIR

router = APIRouter()

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE_MAP = {
    "ingestion":  "stage1_ingestion.py",
    "extraction": "stage2_extraction.py",
    "chunking":   "stage3_chunking.py",
    "embedding":  "stage4_embedding.py",
    "retrieval":  "stage5_retrieval.py",
    "evaluation": "stage6_evaluation.py",
}


@router.get("/status")
def pipeline_status():
    """Returns run status and key metric for every stage."""
    stages = []
    for key, script in STAGE_MAP.items():
        summary_file = {
            "ingestion":  "ingestion_summary.json",
            "extraction": "extraction_summary.json",
            "chunking":   "chunking_summary.json",
            "embedding":  "embedding_summary.json",
            "retrieval":  "query_log.json",
            "evaluation": "evaluation_summary.json",
        }[key]
        data = load_json(summary_file)
        count_key = {
            "ingestion":  "successful",
            "extraction": "successful",
            "chunking":   "total_chunks_produced",
            "embedding":  "chunks_stored",
            "retrieval":  None,
            "evaluation": "total_queries_evaluated",
        }[key]

        if data is None:
            status = "not_run"
            count  = None
        elif count_key is None:
            count  = len(data) if isinstance(data, list) else None
            status = "complete" if count else "not_run"
        else:
            count  = data.get(count_key)
            failed = data.get("failed", data.get("chunks_failed", 0))
            status = "partial" if (failed and failed > 0) else "complete"

        stages.append({
            "key":      key,
            "name":     key.capitalize(),
            "script":   script,
            "status":   status,
            "count":    count,
            "mod_time": file_mod_time(summary_file),
        })

    return {"stages": stages}


@router.post("/run/{stage_key}")
def run_stage(stage_key: str):
    if DEMO_MODE:
        raise HTTPException(status_code=503, detail="Pipeline execution is disabled in demo mode.")
    """
    Launch a pipeline stage as a subprocess and stream its stdout+stderr
    back to the client as SSE (text/event-stream), line by line.
    """
    if stage_key not in STAGE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {stage_key}")

    script = os.path.join(_root, STAGE_MAP[stage_key])

    async def event_stream() -> AsyncIterator[str]:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=_root,
        )
        for line in process.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
        process.wait()
        yield f"data: {json.dumps({'done': True, 'returncode': process.returncode})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/run-all")
def run_all_stages():
    if DEMO_MODE:
        raise HTTPException(status_code=503, detail="Pipeline execution is disabled in demo mode.")
    """Run all 6 stages sequentially, streaming combined output."""
    async def event_stream() -> AsyncIterator[str]:
        for key, script_name in STAGE_MAP.items():
            script = os.path.join(_root, script_name)
            yield f"data: {json.dumps({'stage': key, 'started': True})}\n\n"
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=_root,
            )
            for line in process.stdout:
                yield f"data: {json.dumps({'stage': key, 'line': line.rstrip()})}\n\n"
            process.wait()
            yield f"data: {json.dumps({'stage': key, 'done': True, 'returncode': process.returncode})}\n\n"
            if process.returncode != 0:
                yield f"data: {json.dumps({'abort': True, 'stage': key})}\n\n"
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")

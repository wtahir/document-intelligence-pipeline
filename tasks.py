# tasks.py
# Each pipeline stage becomes a Celery task.
# A task is just a function decorated with @app.task.
# Celery handles the queuing, retrying, and result tracking.

import json
import logging
import os
from celery import chain
from celery_app import app

# Import the existing stage functions directly.
# The key insight — no need to rewrite the pipeline.
# Just wrap it. The stage functions stay exactly as they are.
from stage1_ingestion import extract_text_from_pdf
from stage2_extraction import extract_document, client as azure_client
from stage3_chunking import chunk_document
from stage4_embedding import _get_collection, build_metadata

from config import OUTPUT_FOLDER, LOG_FOLDER, LOG_FORMAT
from datetime import datetime

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "celery_pipeline.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)


# --- Tasks ---
# bind=True gives the task access to itself via 'self'
# This is needed for self.retry() on failures
# max_retries=3 means Celery retries failed tasks 3 times before giving up
# rate_limit controls how many tasks run per minute across all workers

@app.task(bind=True, max_retries=3, rate_limit="30/m")
def ingest_task(self, pdf_path: str) -> dict:
    """
    Stage 1 as a Celery task.
    Takes a file path, returns ingested document dict.
    """
    try:
        logging.info(f"Ingesting: {pdf_path}")
        result = extract_text_from_pdf(pdf_path)

        if result["status"] == "failed":
            raise ValueError(f"Ingestion failed: {result.get('error')}")

        return result

    except Exception as e:
        logging.error(f"Ingest task failed for {pdf_path}: {e}")
        # self.retry re-queues this task with a 60 second delay
        # exc=e preserves the original exception for logging
        raise self.retry(exc=e, countdown=60)


@app.task(bind=True, max_retries=3, rate_limit="10/m")
def extract_task(self, ingested_doc: dict) -> dict:
    """
    Stage 2 as a Celery task.
    Receives ingested document from ingest_task via chain.
    Rate limited to 10/min to respect Azure OpenAI limits.
    """
    try:
        file_name = ingested_doc.get("file_name", "unknown")
        logging.info(f"Extracting: {file_name}")

        result = extract_document(ingested_doc)

        if result.get("status") == "failed":
            raise ValueError(f"Extraction failed: {result.get('reason')}")

        return result

    except Exception as e:
        logging.error(f"Extract task failed: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(bind=True, max_retries=3)
def chunk_task(self, extracted_doc: dict) -> list[dict]:
    """
    Stage 3 as a Celery task.
    Receives extracted document, returns list of chunk dicts.
    """
    try:
        file_name = extracted_doc.get("file_name", "unknown")
        logging.info(f"Chunking: {file_name}")

        chunks = chunk_document(extracted_doc)

        if not chunks:
            raise ValueError(f"No chunks produced for {file_name}")

        return chunks

    except Exception as e:
        logging.error(f"Chunk task failed: {e}")
        raise self.retry(exc=e, countdown=30)


@app.task(bind=True, max_retries=3)
def embed_task(self, chunks: list[dict]) -> dict:
    """
    Stage 4 as a Celery task.
    Receives chunks, embeds and stores in ChromaDB.
    Returns a summary dict.
    """
    try:
        if not chunks:
            return {"status": "skipped", "reason": "no chunks"}

        file_name = chunks[0].get("file_name", "unknown")
        logging.info(f"Embedding {len(chunks)} chunks for: {file_name}")

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [build_metadata(c) for c in chunks]

        _get_collection().upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        logging.info(f"Embedded {len(chunks)} chunks for {file_name}")

        return {
            "status": "success",
            "file_name": file_name,
            "chunks_embedded": len(chunks),
            "completed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Embed task failed: {e}")
        raise self.retry(exc=e, countdown=30)


def process_document(pdf_path: str):
    """
    Creates a chain for a single document.
    This is the key Celery concept — chain links tasks so each
    task's output becomes the next task's input, automatically,
     asynchronously, with retry handling at each step.

    The | operator is Celery's chain syntax.
    .s() creates a "signature" — a lazy task definition
    that doesn't execute yet, just describes what to run.
    """
    pipeline_chain = chain(
        ingest_task.s(pdf_path),   # output → input of next
        extract_task.s(),           # receives ingest output
        chunk_task.s(),             # receives extract output
        embed_task.s()              # receives chunk output
    )

    # .delay() submits the chain to Redis asynchronously
    # It returns immediately — the work happens in the background
    result = pipeline_chain.delay()
    logging.info(f"Submitted chain for {pdf_path}, task_id={result.id}")
    return result.id


def process_all_documents(pdf_folder: str):
    """
    Submits all PDFs in a folder as independent chains.
    Each document gets its own chain running concurrently.
    This function returns immediately after submitting all tasks —
    the actual processing happens in worker processes.
    """
    pdf_files = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.endswith(".pdf")
    ]

    print(f"Submitting {len(pdf_files)} documents to Celery...")
    task_ids = []

    for pdf_path in pdf_files:
        task_id = process_document(pdf_path)
        task_ids.append({"file": pdf_path, "task_id": task_id})
        print(f"  Submitted: {os.path.basename(pdf_path)}")

    # Save task IDs so you can check status later
    with open(f"{OUTPUT_FOLDER}/celery_tasks.json", "w") as f:
        json.dump(task_ids, f, indent=2)

    print(f"\nAll tasks submitted. Check logs/celery_pipeline.log for progress.")
    print(f"Task IDs saved to {OUTPUT_FOLDER}/celery_tasks.json")
    return task_ids
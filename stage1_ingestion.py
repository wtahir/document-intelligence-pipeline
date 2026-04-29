# This stage is responsible for ingesting the raw data into the pipeline. It reads the data from the source, 
# performs any necessary preprocessing, and saves it in a format suitable for the next stages of the pipeline.

import os
import json
import hashlib
import logging
from datetime import datetime
from config import PDF_FOLDER, LOG_FOLDER, OUTPUT_FOLDER, INGESTED_DATA, INGESTION_SUMMARY, LOG_FORMAT
from tqdm import tqdm
import pdfplumber

# Create folders if they don't exist
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, 'ingestion.log'),
    level=logging.INFO,
    format=LOG_FORMAT,
)

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extracts text from a single PDF file.
    Returns a dict with text content, metadata, and a content hash
    for deduplication. The hash lets downstream stages detect whether
    a file has changed since the last run.
    Returns None if the file cannot be processed.
    
    Why a separate function? So the main loop can call this
    per file and handle failures without stopping the whole batch.
    """
    try:
        pages_text = []
        failed_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()  # Can return None for scanned pages

                if text is None or text.strip() == "":
                    # Don't crash — log it and continue to next page
                    failed_pages.append(page_num + 1)
                    logging.warning(f"Page {page_num + 1} of {pdf_path} returned no text (possibly scanned image)")
                else:
                    pages_text.append(text)

        full_text = "\n".join(pages_text)

        # Content hash for dedup — if the file content hasn't changed,
        # downstream stages can skip re-processing.
        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        return {
            "file_name": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "content_hash": content_hash,
            "total_pages": total_pages,
            "failed_pages": failed_pages,
            "pages_extracted": total_pages - len(failed_pages),
            "content": full_text,
            "ingested_at": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        # This file failed entirely — log it and return a failure record
        # We return a dict instead of None so the main loop always gets a consistent type
        logging.error(f"Failed to process {pdf_path}: {e}")
        return {
            "file_name": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "status": "failed",
            "error": str(e),
            "ingested_at": datetime.now().isoformat()
        }


def ingest_data():
    """
    Processes all PDFs in PDF_FOLDER.
    Saves successful results and a summary report separately.
    """
    if not os.path.exists(PDF_FOLDER):
        logging.error(f"PDF folder not found: {PDF_FOLDER}")
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

    if not pdf_files:
        logging.warning("No PDF files found in folder.")
        return

    logging.info(f"Starting ingestion of {len(pdf_files)} files.")

    successful = []
    failed = []
    start_time = datetime.now()

    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        result = extract_text_from_pdf(pdf_path)  # Never crashes — always returns a dict

        if result["status"] == "success":
            successful.append(result)
            logging.info(f"OK: {pdf_file} — {result['pages_extracted']}/{result['total_pages']} pages extracted")
        else:
            failed.append(result)

    # Save successful results for Stage 2
    with open(INGESTED_DATA, "w") as f:
        json.dump(successful, f, indent=2)

    # Save a summary report so you can see pipeline health at a glance
    duration = (datetime.now() - start_time).total_seconds()
    summary = {
        "run_at": datetime.now().isoformat(),
        "total_files": len(pdf_files),
        "successful": len(successful),
        "failed": len(failed),
        "failed_files": [f["file_name"] for f in failed],
        "duration_seconds": round(duration, 2),
    }

    with open(INGESTION_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Ingestion complete. {len(successful)}/{len(pdf_files)} files succeeded.")
    print(f"Done. {len(successful)} succeeded, {len(failed)} failed. Check logs for details.")


if __name__ == "__main__":
    ingest_data()

# stage2_extraction.py
# Classifies each document and extracts structured fields.
#
# The pipeline processes three document types from insurance claims:
#   1. claim_email — claimant reporting damage (water/storm/glass)
#   2. invoice — repair cost with line items and total amount
#   3. photo_documentation — damage evidence with assessment
#
# For each document this stage:
#   1. Classifies the document type and damage category
#   2. Extracts structured fields (claimant, damage type, damaged object, amounts)
#   3. Validates output against Pydantic schemas
#   4. Saves structured records for downstream stages
#
# Input: data/output/ingested_data.json (from Stage 1)
# Output: data/output/extracted_data.json
#         data/output/extraction_summary.json

import os
import json
import logging
import time
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, ValidationError
from openai import AzureOpenAI, RateLimitError
from dotenv import load_dotenv
from config import (
    INGESTED_DATA, OUTPUT_FOLDER, EXTRACTED_DATA, EXTRACTION_SUMMARY,
    EXTRACTION_MAX_CHARS, PII_REDACTION_ENABLED,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION, LOG_FOLDER, LOG_FORMAT,
    estimate_llm_cost,
)
from models import ClaimEmail, InvoiceDocument, PhotoDocumentation, UnknownDocument
from pii_redactor import redact

load_dotenv()

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "extraction.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

# --- Azure OpenAI client ---
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

DEPLOYMENT_NAME = AZURE_OPENAI_DEPLOYMENT


# --- Extraction Prompt ---
# This prompt is designed to extract the specific business fields needed
# for the claims processing workflow:
#   1. Classify damage type (water / storm / glass)
#   2. Identify the damaged object
#   3. Extract claimed amount from invoices
#   4. Extract claimant and policy information for policy lookup
#
# IMPORTANT: We do NOT use RAG for classification or structured extraction.
# This is a deliberate design decision to avoid the "RAG rabbit hole" —
# where semantic search retrieves tangentially related chunks that mislead
# the model. Classification and field extraction are deterministic tasks
# that work better with direct LLM prompting on the full document text.

EXTRACTION_PROMPT = """You are an insurance document analyst processing claim documents.

Your tasks:
1. Classify the DOCUMENT TYPE. Choose from: claim_email, invoice, photo_documentation, unknown
2. Classify the DAMAGE TYPE. Choose from: water, storm, glass, unknown
3. Extract structured fields based on the document type
4. Write a concise English summary (2-3 sentences)
5. Estimate your confidence from 0.0 to 1.0

IMPORTANT: Respond ONLY with valid JSON. No explanation, no markdown, no code fences.
If a field cannot be found, use null. Never omit a field entirely.

For claim_email (damage notification from claimant), return:
{
  "document_type": "claim_email",
  "language": "<detected language code>",
  "claim_number": "<claim number or null>",
  "date": "<email date YYYY-MM-DD or null>",
  "sender": "<sender email or null>",
  "recipient": "<recipient or null>",
  "claimant_name": "<full name of the person making the claim>",
  "policy_number": "<policy number or null>",
  "damage_type": "<water|storm|glass|unknown>",
  "damaged_object": "<specific object that was damaged, e.g. kitchen ceiling, roof tiles, window pane>",
  "damage_date": "<when damage occurred YYYY-MM-DD or null>",
  "damage_description": "<brief description of what happened>",
  "summary_en": "<english summary>",
  "urgency": "<low|normal|high>",
  "confidence": <0.0 to 1.0>
}

For invoice (repair cost document), return:
{
  "document_type": "invoice",
  "language": "<detected language code>",
  "invoice_number": "<invoice number or null>",
  "claim_number": "<claim number or null>",
  "claimant_name": "<name of person billed or null>",
  "policy_number": "<policy number or null>",
  "damage_type": "<water|storm|glass|unknown>",
  "damaged_object": "<what was repaired or null>",
  "vendor": "<repair company name or null>",
  "invoice_date": "<YYYY-MM-DD or null>",
  "subtotal_eur": <subtotal as number or null>,
  "tax_eur": <tax amount as number or null>,
  "total_amount_eur": <total amount as number or null>,
  "line_items": ["<service 1>", "<service 2>"],
  "summary_en": "<english summary>",
  "confidence": <0.0 to 1.0>
}

For photo_documentation (damage evidence report), return:
{
  "document_type": "photo_documentation",
  "language": "<detected language code>",
  "claim_number": "<claim number or null>",
  "claimant_name": "<name or null>",
  "damage_type": "<water|storm|glass|unknown>",
  "damaged_object": "<what is shown damaged or null>",
  "photo_date": "<YYYY-MM-DD or null>",
  "damage_severity": "<minor|moderate|severe|total_loss>",
  "repair_recommendation": "<replacement or repair recommendation>",
  "summary_en": "<english summary>",
  "confidence": <0.0 to 1.0>
}

For unknown documents, return:
{
  "document_type": "unknown",
  "language": "<detected language>",
  "summary_en": "<english summary>",
  "confidence": <0.0 to 1.0>
}

Document text:
"""


def truncate_text(text, max_chars=EXTRACTION_MAX_CHARS):
    """
    Takes only the first max_chars characters for classification.
    The claim number, damage type, and claimant details are always
    in the first section of our structured documents.
    """
    return text[:max_chars] if len(text) > max_chars else text


def validate_extraction(raw):
    """
    Routes the raw LLM output to the correct Pydantic model.
    Returns a validated model instance and the document type string.
    """
    doc_type = raw.get("document_type", "unknown")

    if doc_type == "claim_email":
        return ClaimEmail(**raw), doc_type
    elif doc_type == "invoice":
        return InvoiceDocument(**raw), doc_type
    elif doc_type == "photo_documentation":
        return PhotoDocumentation(**raw), doc_type
    else:
        return UnknownDocument(
            document_type="unknown",
            language=raw.get("language", "unknown"),
            summary_en=raw.get("summary_en", "Could not summarize."),
            confidence=raw.get("confidence", 0.0)
        ), "unknown"


def extract_document(document):
    """
    Processes a single document through the LLM.
    Returns a result dict with status, extracted fields, and metadata.
    Never crashes — failures are captured and returned as failed records.
    """
    file_name = document.get("file_name", "unknown")
    content = document.get("content", "")

    if not content.strip():
        logging.warning(f"Empty content for {file_name}, skipping LLM call.")
        return {
            "file_name": file_name,
            "status": "skipped",
            "reason": "empty content",
            "extracted_at": datetime.now().isoformat()
        }

    truncated = truncate_text(content, max_chars=EXTRACTION_MAX_CHARS)

    # PII redaction — replace sensitive data with placeholders before LLM call
    redacted_text, pii_mapping = redact(truncated, redact_pii=PII_REDACTION_ENABLED)
    prompt = EXTRACTION_PROMPT + "\n" + redacted_text

    try:
        # Retry loop for Azure OpenAI rate limiting (429 errors)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1000
                )
                break  # Success — exit retry loop
            except RateLimitError as e:
                wait = min(2 ** attempt * 2, 60)  # 2, 4, 8, 16, 32 sec
                logging.warning(f"Rate limited on {file_name} (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                print(f"[throttled, retry in {wait}s]", end=" ", flush=True)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise  # Give up after max retries

        raw_text = response.choices[0].message.content.strip()

        # Track token usage for cost monitoring
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "cost_usd": estimate_llm_cost(
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
        }

        # Parse JSON — strip markdown fences if LLM added them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        raw_dict = json.loads(raw_text)

        validated, doc_type = validate_extraction(raw_dict)

        logging.info(f"OK: {file_name} -> {doc_type} (confidence: {validated.confidence}) | tokens: {token_usage['total_tokens']}")

        return {
            "file_name": file_name,
            "file_path": document.get("file_path"),
            "original_content": content,
            "content_hash": document.get("content_hash"),
            "total_pages": document.get("total_pages"),
            "failed_pages": document.get("failed_pages", []),
            "status": "success",
            "extracted_at": datetime.now().isoformat(),
            "token_usage": token_usage,
            **validated.model_dump()
        }

    except json.JSONDecodeError as e:
        logging.error(f"JSON parse failed for {file_name}: {e} | Raw: {raw_text[:200]}")
        return {"file_name": file_name, "status": "failed", "reason": f"json_parse_error: {e}"}

    except ValidationError as e:
        logging.error(f"Validation failed for {file_name}: {e}")
        return {"file_name": file_name, "status": "failed", "reason": f"validation_error: {str(e)}"}

    except Exception as e:
        logging.error(f"Unexpected error for {file_name}: {e}")
        return {"file_name": file_name, "status": "failed", "reason": str(e)}


def extract_all():
    input_path = INGESTED_DATA

    if not os.path.exists(input_path):
        raise FileNotFoundError("ingested_data.json not found. Run Stage 1 first.")

    with open(input_path, "r") as f:
        documents = json.load(f)

    logging.info(f"Starting extraction for {len(documents)} documents.")
    print(f"Processing {len(documents)} documents...")

    # Load already successful results to avoid reprocessing
    existing_results = {}
    output_path = EXTRACTED_DATA
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for r in json.load(f):
                if r.get("status") == "success":
                    existing_results[r["file_name"]] = r

    # Separate already-done from pending
    pending = []
    results_map = {}  # file_name -> result, preserves order later
    for doc in documents:
        file_name = doc.get("file_name")
        if file_name in existing_results:
            results_map[file_name] = existing_results[file_name]
            print(f"  {file_name} -- cached")
        else:
            pending.append(doc)

    cached = len(results_map)
    if cached:
        print(f"  ({cached} already done, {len(pending)} to process)")

    # ─── Concurrent extraction ────────────────────────────
    # Azure OpenAI quotas are typically 6-30 RPM for gpt-4o class models.
    # We use 4 workers to get ~4x speedup while staying under rate limits.
    # The per-request retry in extract_document() handles 429s if we hit the cap.
    MAX_WORKERS = 4
    successful, failed, skipped = cached, 0, 0
    completed = 0

    if pending:
        print(f"  Extracting {len(pending)} docs with {MAX_WORKERS} concurrent workers...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_doc = {
                executor.submit(extract_document, doc): doc
                for doc in pending
            }

            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                file_name = doc.get("file_name")
                completed += 1

                try:
                    result = future.result()
                except Exception as e:
                    result = {"file_name": file_name, "status": "failed", "reason": str(e)}

                results_map[file_name] = result

                status = result.get("status")
                if status == "success":
                    successful += 1
                    doc_type = result.get("document_type")
                    damage = result.get("damage_type", "?")
                    print(f"  [{cached + completed}/{len(documents)}] {file_name} "
                          f"-> {doc_type} [{damage}] (conf: {result.get('confidence')})")
                elif status == "skipped":
                    skipped += 1
                    print(f"  [{cached + completed}/{len(documents)}] {file_name} -- skipped")
                else:
                    failed += 1
                    print(f"  [{cached + completed}/{len(documents)}] {file_name} "
                          f"X failed: {result.get('reason', '')[:60]}")

                # Incremental save every 5 completions
                if completed % 5 == 0 or completed == len(pending):
                    ordered = [results_map[d.get("file_name")] for d in documents
                               if d.get("file_name") in results_map]
                    with open(EXTRACTED_DATA, "w") as f:
                        json.dump(ordered, f, indent=2, ensure_ascii=False)

    # Final ordered save
    results = [results_map[d.get("file_name")] for d in documents
               if d.get("file_name") in results_map]
    with open(EXTRACTED_DATA, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Build summary with damage type and document type breakdowns
    doc_type_counts = {}
    damage_type_counts = {}
    total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in results:
        if r.get("status") == "success":
            t = r.get("document_type", "unknown")
            doc_type_counts[t] = doc_type_counts.get(t, 0) + 1
            d = r.get("damage_type", "unknown")
            if d:
                damage_type_counts[d] = damage_type_counts.get(d, 0) + 1
            # Aggregate token usage
            tu = r.get("token_usage", {})
            total_tokens_used["prompt_tokens"] += tu.get("prompt_tokens", 0)
            total_tokens_used["completion_tokens"] += tu.get("completion_tokens", 0)
            total_tokens_used["total_tokens"] += tu.get("total_tokens", 0)
            total_tokens_used["cost_usd"] = total_tokens_used.get("cost_usd", 0) + tu.get("cost_usd", 0)

    summary = {
        "run_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "document_types_found": doc_type_counts,
        "damage_types_found": damage_type_counts,
        "token_usage": total_tokens_used,
    }

    with open(EXTRACTION_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Extraction complete. {successful} success, {failed} failed, {skipped} skipped.")
    print(f"\nDone. {successful} succeeded, {failed} failed, {skipped} skipped.")
    print(f"Document types: {doc_type_counts}")
    print(f"Damage types: {damage_type_counts}")

    # --- Build Knowledge Graph ---
    # The graph captures entity relationships (claimant → claim → policy)
    # that flat vector search cannot represent. Built automatically after
    # extraction so it's always up-to-date with the latest data.
    try:
        from knowledge_graph import build_and_save_graph
        print("\nBuilding knowledge graph...")
        graph = build_and_save_graph()
    except Exception as e:
        logging.warning(f"Knowledge graph build failed (non-fatal): {e}")
        print(f"  Knowledge graph build failed (non-fatal): {e}")


if __name__ == "__main__":
    extract_all()

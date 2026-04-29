# stage3_chunking.py
# Input: data/output/extracted_data.json (from Stage 2)
# Output: data/output/chunks.json (list of chunks ready for embedding)
#
# Each chunk is a self-contained unit with text + all metadata needed
# to reconstruct context when it comes back as a search result in Stage 5.
#
# Document-aware chunking:
#   - Invoices: preserve line-item tables as a single chunk
#   - Emails: keep headers + body together
#   - Photos: keep photo descriptions grouped
# Falls back to character-based splitting for long documents.

import os
import re
import json
import logging
from datetime import datetime
from config import (
    EXTRACTED_DATA, OUTPUT_FOLDER, CHUNKS_DATA, CHUNKING_SUMMARY,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, SHORT_DOC_THRESHOLD,
    LOG_FOLDER, LOG_FORMAT,
)

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "chunking.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

# --- Chunking configuration ---
# All parameters are now centralised in config.py and can be overridden
# via environment variables (CHUNK_SIZE, CHUNK_OVERLAP, etc.).
# chunk_size: how many characters per chunk
# chunk_overlap: how many characters repeated between consecutive chunks
#   Why overlap? If a claim number appears at the end of chunk 3 and the
#   relevant context is at the start of chunk 4, overlap ensures both
#   chunks contain enough context to be useful independently.
# min_chunk_size: chunks smaller than this get discarded — too small to be useful
# short_doc_threshold: documents with fewer characters than this
#   get stored as a single chunk regardless of size


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into overlapping chunks, respecting word boundaries.
    
    Why not split on tokens? Character splitting is simpler and predictable.
    Token counts vary by model — characters are universal.
    The difference in practice for retrieval quality is minimal.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to end at a sentence boundary first
            boundary = text.rfind(".", start, end)
            if boundary != -1 and boundary > start + (chunk_size // 2):
                end = boundary + 1
            else:
                # No sentence boundary — at least end at a word boundary
                word_boundary = text.rfind(" ", start, end)
                if word_boundary != -1:
                    end = word_boundary + 1

        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)

        # Move back by overlap, then forward to next word boundary
        # so we never start a chunk mid-word
        new_start = end - overlap
        if new_start > start and new_start < len(text):
            # Find the next space after new_start to land on a word boundary
            next_space = text.find(" ", new_start)
            if next_space != -1 and next_space < end:
                new_start = next_space + 1

        start = new_start

    return chunks


# ─── Document-aware section splitting ────────────────────────────────
#
# Instead of blindly splitting on character count, we first try to
# identify structural sections in each document type. This preserves
# semantic boundaries — an invoice's line-item table stays in one chunk,
# email headers stay with the body, etc.
#
# If a section is still too large, it falls back to chunk_text().

# Section boundary patterns for each document type
_EMAIL_SECTIONS = re.compile(
    r'^(CLAIMANT DETAILS:|CLAIM DETAILS:|DAMAGE DESCRIPTION:|Dear |Best regards)',
    re.MULTILINE
)
_INVOICE_SECTIONS = re.compile(
    r'^(BILL TO:|REFERENCE:|SERVICES RENDERED:|PAYMENT DETAILS:)',
    re.MULTILINE
)
_PHOTO_SECTIONS = re.compile(
    r'^(PHOTO EVIDENCE:|DAMAGE ASSESSMENT NOTES:|Photo \d+:)',
    re.MULTILINE
)


def _split_by_sections(text: str, pattern: re.Pattern) -> list[str]:
    """
    Splits text at structural section boundaries identified by a regex.
    Returns a list of sections. Each section includes its header line.
    """
    positions = [m.start() for m in pattern.finditer(text)]

    if not positions:
        return [text]  # No sections found — return whole text

    sections = []
    # Include any preamble before the first section
    if positions[0] > 0:
        preamble = text[:positions[0]].strip()
        if preamble:
            sections.append(preamble)

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[pos:end].strip()
        if section:
            sections.append(section)

    return sections


def _chunk_sections(sections: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Takes pre-split sections and produces final chunks.
    - Small sections that fit together are merged (up to chunk_size).
    - Large sections are sub-split using chunk_text().
    This preserves structure while respecting size limits.
    """
    chunks = []
    current = ""

    for section in sections:
        # If this section alone exceeds chunk_size, sub-split it
        if len(section) > chunk_size:
            # Flush anything accumulated
            if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current.strip())
            current = ""
            # Sub-split the oversized section
            sub_chunks = chunk_text(section, chunk_size, overlap)
            chunks.extend(sub_chunks)
        elif len(current) + len(section) + 2 <= chunk_size:
            # Merge small sections together
            current = current + "\n\n" + section if current else section
        else:
            # Current buffer is full — flush it
            if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current.strip())
            current = section

    # Don't forget the last accumulated section
    if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current.strip())

    return chunks


def chunk_text_document_aware(text: str, document_type: str,
                               chunk_size: int, overlap: int) -> list[str]:
    """
    Document-aware chunking entry point.
    Picks the right section pattern based on document type,
    splits into semantic sections, then produces final chunks.
    Falls back to plain chunk_text() if no pattern matches.
    """
    pattern_map = {
        "claim_email": _EMAIL_SECTIONS,
        "invoice": _INVOICE_SECTIONS,
        "photo_documentation": _PHOTO_SECTIONS,
    }
    pattern = pattern_map.get(document_type)

    if pattern:
        sections = _split_by_sections(text, pattern)
        return _chunk_sections(sections, chunk_size, overlap)
    else:
        return chunk_text(text, chunk_size, overlap)


def build_chunk_record(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    document: dict
) -> dict:
    """
    Wraps a text chunk with all metadata needed for retrieval in Stage 5.
    
    Think of this as the 'label' on the chunk. When Stage 5 retrieves
    this chunk as relevant to a query, everything needed to display
    a useful answer is already attached — no need to look up the
    original document again.
    """
    return {
        # Unique ID for this chunk — used as the ChromaDB document ID in Stage 4
        "chunk_id": f"{document['file_name']}_chunk_{chunk_index}",

        # The actual text that gets embedded
        "text": chunk_text,

        # Position metadata — useful for reassembling document context
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "is_single_chunk": total_chunks == 1,

        # Document-level metadata — attached to every chunk from this document
        "file_name": document.get("file_name"),
        "file_path": document.get("file_path"),
        "document_type": document.get("document_type", "unknown"),
        "language": document.get("language", "unknown"),
        "claim_number": document.get("claim_number"),
        "claimant_name": document.get("claimant_name"),
        "policy_number": document.get("policy_number"),
        "damage_type": document.get("damage_type"),
        "damaged_object": document.get("damaged_object"),
        "date": document.get("date"),
        "sender": document.get("sender"),
        "summary_en": document.get("summary_en"),
        "urgency": document.get("urgency", "normal"),
        "confidence": document.get("confidence", 0.0),
        "failed_pages": document.get("failed_pages", []),

        # Invoice-specific fields (null for non-invoices)
        "total_amount_eur": document.get("total_amount_eur"),
        "vendor": document.get("vendor"),

        # Photo-specific fields (null for non-photos)
        "damage_severity": document.get("damage_severity"),

        # Lineage — content hash from Stage 1 for dedup in Stage 4
        "content_hash": document.get("content_hash"),

        # Pipeline metadata
        "chunked_at": datetime.now().isoformat()
    }


def chunk_document(document: dict) -> list[dict]:
    """
    Processes a single document into a list of chunk records.
    Handles the short document case explicitly.
    """
    file_name = document.get("file_name", "unknown")
    content = document.get("original_content", "").strip()

    if not content:
        logging.warning(f"No content to chunk for {file_name}")
        return []

    # Short document — store as single chunk, no splitting needed
    if len(content) <= SHORT_DOC_THRESHOLD:
        logging.info(f"{file_name} is short ({len(content)} chars) — storing as single chunk")
        return [build_chunk_record(content, 0, 1, document)]

    # Document-aware chunking — preserves structural sections
    doc_type = document.get("document_type", "unknown")
    text_chunks = chunk_text_document_aware(content, doc_type, CHUNK_SIZE, CHUNK_OVERLAP)

    if not text_chunks:
        logging.warning(f"Chunking produced no results for {file_name}")
        return []

    chunk_records = []
    for i, chunk in enumerate(text_chunks):
        record = build_chunk_record(chunk, i, len(text_chunks), document)
        chunk_records.append(record)

    logging.info(f"{file_name} → {len(chunk_records)} chunks (doc length: {len(content)} chars)")
    return chunk_records


def chunk_all():
    input_path = EXTRACTED_DATA

    if not os.path.exists(input_path):
        raise FileNotFoundError("extracted_data.json not found. Run Stage 2 first.")

    with open(input_path, "r") as f:
        documents = json.load(f)

    # Only process successfully extracted documents
    successful_docs = [d for d in documents if d.get("status") == "success"]
    logging.info(f"Starting chunking for {len(successful_docs)} documents.")
    print(f"Chunking {len(successful_docs)} documents...")

    all_chunks = []
    total_docs_processed = 0
    total_single_chunk_docs = 0
    start_time = datetime.now()

    for doc in successful_docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        total_docs_processed += 1
        if chunks and chunks[0].get("is_single_chunk"):
            total_single_chunk_docs += 1

    # Save all chunks for Stage 4
    with open(CHUNKS_DATA, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    duration = (datetime.now() - start_time).total_seconds()

    summary = {
        "run_at": datetime.now().isoformat(),
        "documents_processed": total_docs_processed,
        "total_chunks_produced": len(all_chunks),
        "single_chunk_documents": total_single_chunk_docs,
        "multi_chunk_documents": total_docs_processed - total_single_chunk_docs,
        "avg_chunks_per_document": round(len(all_chunks) / total_docs_processed, 2) if total_docs_processed else 0,
        "duration_seconds": round(duration, 2),
        "chunk_config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE,
            "short_doc_threshold": SHORT_DOC_THRESHOLD
        }
    }

    with open(CHUNKING_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Chunking complete. {len(all_chunks)} chunks from {total_docs_processed} documents.")
    print(f"Done. {len(all_chunks)} total chunks from {total_docs_processed} documents.")
    print(f"Single-chunk docs: {total_single_chunk_docs}, Multi-chunk docs: {total_docs_processed - total_single_chunk_docs}")


if __name__ == "__main__":
    chunk_all()
# stage4_embedding.py
# Input: data/output/chunks.json (from Stage 3)
# Output: ChromaDB collection stored in chroma_db/ folder
#         data/output/embedding_summary.json
#
# This stage converts text chunks into vector embeddings and stores them
# in ChromaDB with metadata. After this stage the documents are
# semantically searchable — similar meaning finds similar chunks
# even without exact keyword matches.

import os
import json
import logging
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from config import (
    CHUNKS_DATA, OUTPUT_FOLDER, EMBEDDING_SUMMARY, CHROMA_FOLDER,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHROMA_COLLECTION,
    LOG_FOLDER, LOG_FORMAT,
)
from dotenv import load_dotenv

load_dotenv()

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "embedding.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

# --- Embedding model ---
# We use sentence-transformers running locally.
# "paraphrase-multilingual-MiniLM-L12-v2" is specifically designed for
# multilingual text — it understands German and English in the same
# vector space. This means a German chunk and an English query can
# still match if they mean the same thing.
# It runs on CPU, no API cost, no internet needed after first download.


# --- Lazy initialisation ---
# ChromaDB client and collection are created on first use, not at import time.
# This prevents side effects when importing this module in tests or UI code.

_chroma_client = None
_collection = None
_embedding_fn = None


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=_get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

# Batch size for ChromaDB upserts — configurable via EMBEDDING_BATCH_SIZE.
BATCH_SIZE = EMBEDDING_BATCH_SIZE


def build_metadata(chunk: dict) -> dict:
    """
    Extracts only the filterable/useful metadata fields for ChromaDB.
    ChromaDB metadata values must be str, int, float, or bool — not None.
    We convert None to empty string to avoid storage errors.

    Why not store everything? ChromaDB metadata is not compressed —
    storing large fields like original_content in metadata bloats the index.
    The full chunk text is stored separately as the document field.
    """
    def safe(value) -> str:
        # ChromaDB rejects None values in metadata — convert to empty string
        return str(value) if value is not None else ""

    return {
        "file_name": safe(chunk.get("file_name")),
        "document_type": safe(chunk.get("document_type")),
        "claim_number": safe(chunk.get("claim_number")),
        "claimant_name": safe(chunk.get("claimant_name")),
        "policy_number": safe(chunk.get("policy_number")),
        "damage_type": safe(chunk.get("damage_type")),
        "damaged_object": safe(chunk.get("damaged_object")),
        "date": safe(chunk.get("date")),
        "sender": safe(chunk.get("sender")),
        "urgency": safe(chunk.get("urgency")),
        "language": safe(chunk.get("language")),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "total_chunks": int(chunk.get("total_chunks", 1)),
        "is_single_chunk": bool(chunk.get("is_single_chunk", False)),
        "summary_en": safe(chunk.get("summary_en")),
        "total_amount_eur": safe(chunk.get("total_amount_eur")),
        "vendor": safe(chunk.get("vendor")),
        "damage_severity": safe(chunk.get("damage_severity")),
        "content_hash": safe(chunk.get("content_hash")),
    }


def embed_and_store(chunks: list[dict]):
    """
    Embeds chunks in batches and upserts into ChromaDB.
    Uses upsert instead of add — if a chunk_id already exists
    it gets updated rather than causing a duplicate error.
    This makes the stage safely rerunnable without clearing the DB first.
    """
    collection = _get_collection()
    total = len(chunks)
    stored = 0
    failed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")

        try:
            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [build_metadata(chunk) for chunk in batch]

            # upsert embeds the documents using embedding_fn automatically
            # then stores vector + metadata + id together
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            stored += len(batch)
            print(f"✓")
            logging.info(f"Batch {batch_num}/{total_batches} stored ({len(batch)} chunks)")

        except Exception as e:
            failed += len(batch)
            print(f"✗ failed: {e}")
            logging.error(f"Batch {batch_num} failed: {e}")

    return stored, failed


def embed_all():
    input_path = CHUNKS_DATA

    if not os.path.exists(input_path):
        raise FileNotFoundError("chunks.json not found. Run Stage 3 first.")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks into ChromaDB...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Collection: {CHROMA_COLLECTION}\n")

    # Check how many chunks are already in the collection
    collection = _get_collection()
    existing_count = collection.count()

    # --- Hash-based dedup ---
    # If chunks already exist in the collection, compare content hashes
    # to skip re-embedding unchanged documents. This makes the stage
    # idempotent — re-running after adding new PDFs only embeds the new ones.
    skipped = 0
    if existing_count > 0:
        chunk_ids = [c["chunk_id"] for c in chunks]
        # Fetch existing IDs and their hashes in batches
        existing_hashes = {}
        for batch_start in range(0, len(chunk_ids), BATCH_SIZE):
            batch_ids = chunk_ids[batch_start:batch_start + BATCH_SIZE]
            try:
                existing = collection.get(ids=batch_ids, include=["metadatas"])
                for eid, emeta in zip(existing["ids"], existing["metadatas"]):
                    existing_hashes[eid] = emeta.get("content_hash", "")
            except Exception:
                pass  # IDs not found — they're new chunks

        new_chunks = []
        for chunk in chunks:
            cid = chunk["chunk_id"]
            new_hash = chunk.get("content_hash", "")
            if cid in existing_hashes and existing_hashes[cid] == new_hash and new_hash:
                skipped += 1
            else:
                new_chunks.append(chunk)

        if skipped:
            print(f"Skipping {skipped} unchanged chunks (hash match)")
        if not new_chunks:
            print("All chunks are up-to-date — nothing to embed.")
            summary = {
                "run_at": datetime.now().isoformat(),
                "embedding_model": EMBEDDING_MODEL,
                "chunks_processed": len(chunks),
                "chunks_skipped_dedup": skipped,
                "chunks_stored": 0,
                "chunks_failed": 0,
                "total_vectors_in_collection": existing_count,
                "duration_seconds": 0,
                "chunks_per_second": 0,
            }
            with open(EMBEDDING_SUMMARY, "w") as f:
                json.dump(summary, f, indent=2)
            return

        chunks = new_chunks
        print(f"{len(chunks)} new/changed chunks to embed\n")

    start_time = datetime.now()
    stored, failed = embed_and_store(chunks)
    duration = (datetime.now() - start_time).total_seconds()

    # --- Stale vector garbage collection ---
    # Remove vectors from ChromaDB that no longer exist in chunks.json.
    # This happens when PDFs are deleted or re-processed with different
    # chunking parameters, leaving orphan vectors in the DB.
    stale_removed = 0
    collection = _get_collection()

    try:
        # Get all chunk IDs that SHOULD exist (from the full input file)
        with open(input_path, "r", encoding="utf-8") as f:
            all_current_chunks = json.load(f)
        valid_ids = set(c["chunk_id"] for c in all_current_chunks)

        # Get all IDs currently in ChromaDB
        all_stored = collection.get(include=[])
        stored_ids = set(all_stored["ids"]) if all_stored["ids"] else set()

        # Find stale IDs — in ChromaDB but not in current chunks
        stale_ids = stored_ids - valid_ids
        if stale_ids:
            # ChromaDB delete requires a list
            stale_list = list(stale_ids)
            for batch_start in range(0, len(stale_list), BATCH_SIZE):
                batch = stale_list[batch_start:batch_start + BATCH_SIZE]
                collection.delete(ids=batch)
            stale_removed = len(stale_ids)
            print(f"Garbage collection: removed {stale_removed} stale vectors")
            logging.info(f"GC: removed {stale_removed} stale vectors: {list(stale_ids)[:10]}...")
    except Exception as e:
        logging.warning(f"Stale vector cleanup failed (non-fatal): {e}")

    final_count = _get_collection().count()

    summary = {
        "run_at": datetime.now().isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "chunks_processed": stored + failed + skipped,
        "chunks_skipped_dedup": skipped,
        "chunks_stored": stored,
        "chunks_failed": failed,
        "stale_vectors_removed": stale_removed,
        "total_vectors_in_collection": final_count,
        "duration_seconds": round(duration, 2),
        "chunks_per_second": round(len(chunks) / duration, 1) if duration > 0 else 0
    }

    with open(EMBEDDING_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Embedding complete. {stored} stored, {failed} failed. Duration: {duration:.1f}s")
    print(f"\nDone. {stored} chunks stored, {failed} failed.")
    print(f"Total vectors in collection: {final_count}")
    print(f"Duration: {duration:.1f}s")


if __name__ == "__main__":
    embed_all()
# context_engine.py
# ═══════════════════════════════════════════════════════════════════
# CONTEXT ENGINEERING — Smart context building for the LLM
# ═══════════════════════════════════════════════════════════════════
#
# Why this matters:
#   Your old pipeline shoved all retrieved chunks into a flat string
#   and hoped the LLM would figure it out. This works for 3-5 chunks,
#   but falls apart when you have:
#   - Multi-source results (some from vector search, some from graph)
#   - Hierarchical documents (parent-child chunks)
#   - Redundant chunks (overlap from multi-query expansion)
#   - Mixed relevance (high-confidence + low-confidence chunks)
#
#   Context Engineering is the practice of CURATING what goes into
#   the LLM prompt. In 2026, this is considered more important than
#   the retrieval itself — a mediocre retriever with great context
#   engineering beats a great retriever with naive context building.
#
# This module does four things:
#   1. DEDUPLICATION — Removes redundant chunks from multi-query results
#   2. HIERARCHICAL ORDERING — Groups chunks by document → section → chunk
#   3. RELEVANCE-WEIGHTED COMPRESSION — Summarizes low-relevance chunks,
#      keeps high-relevance chunks in full
#   4. STRUCTURED CONTEXT INJECTION — Adds graph facts and metadata
#      as structured headers before the unstructured chunks

import json
import logging
import os
from typing import Optional
from config import (
    LOG_FOLDER, LOG_FORMAT,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION,
    estimate_llm_cost,
)
from openai import AzureOpenAI

os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "context_engine.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

_azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT


# ─── 1. Chunk Deduplication ───────────────────────────────────────
#
# Multi-query expansion and graph-enhanced retrieval both produce
# overlapping results. We need to deduplicate while keeping the
# highest-confidence version of each chunk.
#
# Three strategies:
# a) Exact dedup: same chunk_id → keep the one with best score
# b) Near-dedup: >80% text overlap → merge into one, keep best metadata
# c) Subsumption: chunk A is entirely contained in chunk B → keep B only

def _text_similarity(a: str, b: str) -> float:
    """
    Fast jaccard similarity on word sets.
    Not as precise as cosine similarity, but runs in microseconds.
    Good enough for deduplication.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def deduplicate_chunks(chunks: list[dict], similarity_threshold: float = 0.80) -> list[dict]:
    """
    Removes duplicate and near-duplicate chunks.

    Strategy:
    1. Sort by relevance (best first)
    2. For each chunk, check against all already-kept chunks
    3. If >80% similar to any kept chunk, skip it
    4. Otherwise, keep it

    This is O(n²) but n is typically <30, so it runs in microseconds.
    """
    if not chunks:
        return chunks

    # Sort: lower distance = more relevant, higher rerank_score = more relevant
    def _sort_key(c):
        rerank = c.get("rerank_score", 0)
        rrf = c.get("rrf_score", 0)
        distance = c.get("distance", 1.0)
        return -(rerank + rrf) + distance  # Lower is better

    sorted_chunks = sorted(chunks, key=_sort_key)

    kept = []
    for chunk in sorted_chunks:
        text = chunk.get("text", "")
        is_duplicate = False

        for kept_chunk in kept:
            if _text_similarity(text, kept_chunk.get("text", "")) > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(chunk)

    if len(kept) < len(chunks):
        logging.info(f"Dedup: {len(chunks)} -> {len(kept)} chunks ({len(chunks) - len(kept)} duplicates removed)")

    return kept


# ─── 2. Hierarchical Context Organization ────────────────────────
#
# Instead of dumping chunks in retrieval-score order, we organize them
# by document structure:
#   Document A:
#     - Chunk 1 (claim email header)
#     - Chunk 2 (claim email body)
#   Document B:
#     - Chunk 1 (invoice details)
#
# This helps the LLM understand relationships between chunks from
# the same document, which is critical for multi-document queries.

def organize_hierarchically(chunks: list[dict]) -> list[dict]:
    """
    Groups chunks by source document and sorts by chunk_index within each group.
    Adds hierarchy metadata for context building.

    Returns chunks in order: grouped by file, sorted by chunk_index within each file.
    Cross-document ordering is by best relevance score within each group.
    """
    if not chunks:
        return chunks

    # Group by file_name
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        file_name = chunk.get("metadata", {}).get("file_name", "unknown")
        if file_name not in groups:
            groups[file_name] = []
        groups[file_name].append(chunk)

    # Sort each group by chunk_index
    for file_name in groups:
        groups[file_name].sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

    # Order groups by best chunk relevance within each group
    def _group_score(file_group: list[dict]) -> float:
        scores = []
        for c in file_group:
            # Combine available scores
            score = c.get("rerank_score", 0) + c.get("rrf_score", 0)
            if score == 0:
                score = 1.0 - c.get("distance", 1.0)
            scores.append(score)
        return max(scores) if scores else 0.0

    sorted_groups = sorted(groups.values(), key=_group_score, reverse=True)

    # Flatten back to list with hierarchy markers
    result = []
    for group in sorted_groups:
        for i, chunk in enumerate(group):
            chunk["_hierarchy"] = {
                "is_first_in_group": i == 0,
                "group_size": len(group),
                "position_in_group": i,
            }
            result.append(chunk)

    return result


# ─── 3. Relevance-Weighted Compression ───────────────────────────
#
# Not all chunks deserve the same amount of context window real estate.
# High-relevance chunks (rerank_score > 0.5 or low distance) get their
# full text. Low-relevance chunks are compressed to 1-2 sentences.
#
# This fits more information into the same token budget.

_COMPRESS_PROMPT = """Compress this insurance document chunk into 1-2 sentences.
Keep: claim numbers, damage types, amounts, dates, names.
Drop: filler text, formatting, repetitive details.

Chunk:
{text}

Compressed (1-2 sentences):"""


def compress_chunk(text: str) -> tuple[str, dict]:
    """Compresses a low-relevance chunk to save context window space."""
    try:
        response = _azure_client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _COMPRESS_PROMPT.format(text=text)}],
            max_completion_tokens=100,
            temperature=0.0,
        )
        compressed = response.choices[0].message.content.strip()
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cost_usd": estimate_llm_cost(
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
        }
        return compressed, token_usage
    except Exception as e:
        logging.error(f"Compression failed: {e}")
        # Fallback: truncate to first 200 chars
        return text[:200] + "..." if len(text) > 200 else text, {}


def apply_relevance_compression(
    chunks: list[dict],
    high_relevance_threshold: float = 0.3,
    max_compressed: int = 3,
) -> tuple[list[dict], dict]:
    """
    Compresses low-relevance chunks while keeping high-relevance ones in full.

    Args:
        chunks: List of chunk dicts with "text", "distance", optionally "rerank_score"
        high_relevance_threshold: Distance below this = high relevance (keep full)
        max_compressed: Maximum number of chunks to compress (to control LLM costs)

    Returns:
        (modified_chunks, total_token_usage)

    Why not compress everything?
    - High-relevance chunks contain the answer — compression might lose key details
    - Compression costs LLM tokens — we only compress when the savings justify it
    - max_compressed prevents runaway costs on queries with many low-relevance chunks
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    compressed_count = 0

    for chunk in chunks:
        distance = chunk.get("distance", 1.0)
        rerank_score = chunk.get("rerank_score")

        # Determine if this is a high-relevance chunk
        is_high_relevance = distance < high_relevance_threshold
        if rerank_score is not None:
            is_high_relevance = is_high_relevance or rerank_score > 0.5

        if not is_high_relevance and compressed_count < max_compressed:
            original_len = len(chunk.get("text", ""))
            compressed_text, usage = compress_chunk(chunk["text"])
            chunk["text"] = compressed_text
            chunk["_compressed"] = True
            chunk["_original_length"] = original_len
            compressed_count += 1

            for k in total_usage:
                total_usage[k] = round(total_usage[k] + usage.get(k, 0), 6)

    if compressed_count > 0:
        logging.info(f"Compressed {compressed_count} low-relevance chunks")

    return chunks, total_usage


# ─── 4. Structured Context Building ──────────────────────────────
#
# The final context sent to the LLM is structured in layers:
#
# Layer 1: GRAPH FACTS (structured, deterministic)
#   "Claim CLM-2024-001: filed by Mueller, water damage, EUR 2,500"
#
# Layer 2: DOCUMENT CONTEXT (hierarchical, high-relevance full text)
#   [Document: email_001.pdf — Claim Email]
#     Chunk 1: ...full text...
#     Chunk 2: ...full text...
#
# Layer 3: SUPPLEMENTARY (compressed low-relevance chunks)
#   [Supporting context — compressed]
#     Chunk 5: one-sentence summary
#
# This gives the LLM structured facts first (most reliable),
# then full unstructured context, then compressed supplementary info.

def build_enriched_context(
    chunks: list[dict],
    graph_facts: Optional[list[str]] = None,
    query_plan_info: Optional[dict] = None,
) -> str:
    """
    Builds the final context string for the LLM prompt.
    Organized in three layers for optimal comprehension.

    Args:
        chunks: Deduplicated, hierarchically organized, relevance-compressed chunks
        graph_facts: Structured facts from knowledge graph traversal
        query_plan_info: Info about how the query was processed (for transparency)

    Returns:
        Formatted context string ready for the LLM prompt.
    """
    parts = []

    # Layer 1: Graph facts (if available)
    if graph_facts:
        parts.append("=== STRUCTURED FACTS (from knowledge graph) ===")
        for i, fact in enumerate(graph_facts, 1):
            parts.append(f"  Fact {i}: {fact}")
        parts.append("")

    # Layer 2 & 3: Document chunks (organized hierarchically)
    high_relevance_parts = []
    supplementary_parts = []
    current_file = None
    chunk_counter = 0

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        hierarchy = chunk.get("_hierarchy", {})
        is_compressed = chunk.get("_compressed", False)
        chunk_counter += 1

        # Document header when starting a new file group
        if hierarchy.get("is_first_in_group", False):
            file_name = meta.get("file_name", "unknown")
            doc_type = meta.get("document_type", "unknown")
            claim_num = meta.get("claim_number", "?")
            current_file = file_name
            header = f"\n--- Document: {file_name} | Type: {doc_type} | Claim: {claim_num} ---"

            if is_compressed:
                supplementary_parts.append(header)
            else:
                high_relevance_parts.append(header)

        # Chunk content
        rerank_info = f" | Relevance: {chunk['rerank_score']:.3f}" if "rerank_score" in chunk else ""
        rrf_info = f" | RRF: {chunk['rrf_score']:.4f}" if "rrf_score" in chunk else ""
        compressed_tag = " (compressed)" if is_compressed else ""

        chunk_header = (
            f"[Chunk {chunk_counter} | Distance: {chunk.get('distance', '?')}"
            f"{rerank_info}{rrf_info}{compressed_tag}]"
        )

        chunk_text = f"{chunk_header}\n{chunk['text']}"

        if is_compressed:
            supplementary_parts.append(chunk_text)
        else:
            high_relevance_parts.append(chunk_text)

    # Assemble final context
    if high_relevance_parts:
        parts.append("=== PRIMARY CONTEXT (high relevance) ===")
        parts.extend(high_relevance_parts)

    if supplementary_parts:
        parts.append("\n=== SUPPLEMENTARY CONTEXT (lower relevance, compressed) ===")
        parts.extend(supplementary_parts)

    return "\n".join(parts)


# ─── Self-Critique Context Check ─────────────────────────────────
#
# Before sending context to the generation LLM, we do a quick check:
# Is this context actually sufficient to answer the query?
# If not, we can request additional retrieval passes.
#
# This is the "C" in CRAG (Corrective RAG).

_SUFFICIENCY_PROMPT = """You are checking if retrieved context is sufficient to answer a question about insurance claims.

Question: {query}

Context summary:
- {n_chunks} chunks retrieved from {n_docs} documents
- Document types: {doc_types}
- Claim numbers mentioned: {claims}
- Graph facts available: {n_facts}

Based on the question, is this context likely sufficient?

Respond with ONLY valid JSON:
{{"sufficient": true/false, "missing": "<what information is missing, if any>", "suggestion": "<retrieval action to take>"}}"""


def check_context_sufficiency(
    query: str,
    chunks: list[dict],
    graph_facts: list[str],
) -> tuple[bool, str, dict]:
    """
    Quick LLM check: is the retrieved context sufficient to answer the query?

    Returns:
        (is_sufficient, missing_info, token_usage)

    This is lightweight — only sends metadata, not the full chunks.
    Uses ~200 tokens total.
    """
    doc_types = list(set(c.get("metadata", {}).get("document_type", "?") for c in chunks))
    claims = list(set(c.get("metadata", {}).get("claim_number", "?") for c in chunks if c.get("metadata", {}).get("claim_number")))
    n_docs = len(set(c.get("metadata", {}).get("file_name", "") for c in chunks))

    prompt = _SUFFICIENCY_PROMPT.format(
        query=query,
        n_chunks=len(chunks),
        n_docs=n_docs,
        doc_types=", ".join(doc_types),
        claims=", ".join(claims) if claims else "none",
        n_facts=len(graph_facts),
    )

    try:
        response = _azure_client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cost_usd": estimate_llm_cost(
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
        }

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        is_sufficient = result.get("sufficient", True)
        missing = result.get("missing", "")

        if not is_sufficient:
            logging.info(f"Context insufficiency detected: {missing}")

        return is_sufficient, missing, token_usage

    except Exception as e:
        logging.warning(f"Sufficiency check failed: {e}, assuming sufficient")
        return True, "", {}

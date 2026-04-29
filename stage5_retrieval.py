# stage5_retrieval.py
# Input: ChromaDB collection (from Stage 4) + user query
# Output: data/output/query_log.json — every query and answer logged
#         data/output/payout_report.json — policy check + payout decisions
#
# This stage has five responsibilities:
# 1. Retrieve relevant chunks from ChromaDB based on a query
# 2. Rerank results with a cross-encoder for precision (if enabled)
# 3. Generate an answer using GPT-4o with retrieved chunks as context
# 4. Check claimant coverage against policy metadata (deterministic, not RAG)
# 5. Produce payout confirmations with coverage validation
#
# DESIGN DECISION — Avoiding the RAG rabbit hole:
# Policy checks and payout decisions use DETERMINISTIC metadata lookups,
# not semantic search. RAG is reserved for answering unstructured questions
# about claim content. This prevents the system from hallucinating coverage
# status or payout amounts based on semantically similar but wrong documents.

import os
import json
import logging
from datetime import datetime
from openai import AzureOpenAI
import chromadb
from chromadb.utils import embedding_functions
from config import (
    QUERY_LOG, OUTPUT_FOLDER, CHROMA_FOLDER, CHROMA_COLLECTION,
    EMBEDDING_MODEL, RERANKER_MODEL, RERANK_ENABLED, RERANK_TOP_K,
    DEFAULT_N_RESULTS, RETRIEVAL_OVER_FETCH,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION,
    LOG_FOLDER, LOG_FORMAT,
    POLICY_METADATA, EXTRACTED_DATA, PAYOUT_REPORT,
)
from dotenv import load_dotenv

load_dotenv()

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "retrieval.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

# --- Clients ---
azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
DEPLOYMENT_NAME = AZURE_OPENAI_DEPLOYMENT

# --- Lazy-loaded resources ---
_chroma_collection = None
_reranker = None


def _get_collection():
    """Lazy-load ChromaDB collection to avoid import-time side effects."""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_FOLDER)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn,
        )
    return _chroma_collection


def _get_reranker():
    """Lazy-load cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
        logging.info(f"Loaded reranker model: {RERANKER_MODEL}")
    return _reranker


# --- System Prompt ---
SYSTEM_PROMPT = """You are an insurance claims processing assistant.
You answer questions based strictly on the provided context chunks from insurance claim documents.
The documents contain claim emails, repair invoices, and damage photo documentation.

You help with:
- Classifying damage types (water, storm, glass)
- Identifying damaged objects
- Extracting claimed amounts from invoices
- Summarising claim status

If the context does not contain enough information to answer, say so clearly.
Never make up information not present in the context.
When referencing specific claims, always mention the claim number."""

QUERY_LOG_PATH = QUERY_LOG


def load_query_log():
    if os.path.exists(QUERY_LOG_PATH):
        with open(QUERY_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_query_log(log):
    with open(QUERY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def retrieve_chunks(search_text, metadata_filter=None, n_results=DEFAULT_N_RESULTS):
    """Retrieves semantically similar chunks from ChromaDB."""
    collection = _get_collection()
    fetch_count = RETRIEVAL_OVER_FETCH if RERANK_ENABLED else n_results

    query_params = {
        "query_texts": [search_text],
        "n_results": fetch_count,
        "include": ["documents", "metadatas", "distances"],
    }

    if metadata_filter:
        query_params["where"] = metadata_filter

    results = collection.query(**query_params)

    chunks = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": metadata,
            "distance": round(distance, 4),
        })

    return chunks


def rerank_chunks(query, chunks, top_k=RERANK_TOP_K):
    """Re-scores chunks using a cross-encoder for higher precision."""
    if not chunks:
        return chunks

    reranker = _get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

    logging.info(
        f"Reranked {len(chunks)} -> {len(reranked)} chunks. "
        f"Score range: {reranked[-1]['rerank_score']:.3f} - {reranked[0]['rerank_score']:.3f}"
    )

    return reranked


def build_context(chunks):
    """Formats retrieved chunks into a context block for the GPT-4o prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        rerank_info = f" | Relevance: {chunk['rerank_score']:.3f}" if "rerank_score" in chunk else ""
        context_parts.append(
            f"[Chunk {i+1} | File: {meta.get('file_name')} | "
            f"Claim: {meta.get('claim_number')} | "
            f"Type: {meta.get('document_type')} | "
            f"Damage: {meta.get('damage_type')} | "
            f"Distance: {chunk['distance']}{rerank_info}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query, context):
    """Sends query + context to GPT-4o and returns the answer."""
    user_message = f"""Context from insurance claim documents:

{context}

Question: {query}

Answer based only on the context above:"""

    response = azure_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=1000,
    )

    return response.choices[0].message.content.strip()


def query_pipeline(query, metadata_filter=None, n_results=DEFAULT_N_RESULTS):
    """
    Full RAG pipeline for a single query:
    retrieve -> rerank (optional) -> build context -> generate answer -> log
    """
    logging.info(f"Query: {query} | Filter: {metadata_filter} | n_results: {n_results}")

    chunks = retrieve_chunks(query, metadata_filter, n_results)

    if RERANK_ENABLED and chunks:
        chunks = rerank_chunks(query, chunks, top_k=n_results)

    if not chunks:
        answer = "No relevant documents found for this query."
        logging.warning(f"No chunks retrieved for query: {query}")
    else:
        context = build_context(chunks)
        answer = generate_answer(query, context)

    result = {
        "query_id": f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "query": query,
        "metadata_filter": metadata_filter,
        "n_results_requested": n_results,
        "chunks_retrieved": len(chunks),
        "reranking_enabled": RERANK_ENABLED,
        "chunks": chunks,
        "answer": answer,
        "queried_at": datetime.now().isoformat(),
        "evaluated": False,
        "score": None,
        "evaluation_notes": None,
    }

    log = load_query_log()
    log.append(result)
    save_query_log(log)

    logging.info(f"Query answered. chunks_retrieved={len(chunks)}, query_id={result['query_id']}")

    return result


# ═══════════════════════════════════════════════════════════════════
# POLICY CHECK & PAYOUT — Deterministic, NOT RAG
# ═══════════════════════════════════════════════════════════════════
#
# This section deliberately avoids using RAG for policy lookups.
# Policy coverage is a factual yes/no question that MUST be answered
# from structured metadata, not from semantic similarity search.
#
# If we used RAG here, the system could:
# - Retrieve a chunk from a DIFFERENT claimant with similar damage
# - Hallucinate that the claimant is covered based on similar names
# - Return the wrong policy limits because a semantically close
#   document had different numbers
#
# This is the "RAG rabbit hole" — once the model starts answering
# structured questions with unstructured retrieval, every tangentially
# related document becomes a potential source of wrong answers.

def load_policy_metadata():
    """Loads policy metadata from the structured JSON file."""
    if not os.path.exists(POLICY_METADATA):
        logging.warning("policy_metadata.json not found")
        return []
    with open(POLICY_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extracted_data():
    """Loads extracted document data for claim information."""
    if not os.path.exists(EXTRACTED_DATA):
        logging.warning("extracted_data.json not found")
        return []
    with open(EXTRACTED_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def find_policy_for_claimant(claimant_name, policies):
    """
    Finds a policy by exact name match against the policy metadata.
    This is a DETERMINISTIC lookup — no semantic search, no LLM.
    Returns the policy dict or None if not found.
    """
    if not claimant_name:
        return None
    name_lower = claimant_name.strip().lower()
    for policy in policies:
        if policy.get("policyholder_name", "").strip().lower() == name_lower:
            return policy
    return None


def check_coverage(policy, damage_type, damaged_object=None):
    """
    Checks if the policy covers the specific damage type AND damaged item.
    Returns a dict with coverage status and details.

    Two-level check:
    1. Is the damage TYPE covered? (water_damage, storm_damage, glass_damage)
    2. Is the specific damaged OBJECT listed in covered_items?

    If the damage type is covered but the item isn't, the claim is denied
    with a clear reason — this is a realistic insurance scenario.
    """
    if not policy:
        return {
            "is_covered": False,
            "reason": "No policy found for this claimant",
            "policy_number": None,
            "coverage_types": [],
            "covered_items": [],
            "coverage_limit_eur": 0,
            "deductible_eur": 0,
        }

    coverage_key = f"{damage_type}_damage" if damage_type else None
    coverage_types = policy.get("coverage_types", [])
    type_covered = coverage_key in coverage_types if coverage_key else False

    # Level 2: Check if the specific damaged object is in the covered items list
    covered_items_map = policy.get("covered_items", {})
    covered_items_for_type = covered_items_map.get(coverage_key, [])
    item_covered = True  # Default to True if no item specified or no item list exists
    if damaged_object and covered_items_for_type:
        item_covered = damaged_object.strip().lower() in [
            item.strip().lower() for item in covered_items_for_type
        ]

    is_covered = type_covered and item_covered

    if is_covered:
        reason = f"Policy covers {coverage_key} and item '{damaged_object}' is listed"
    elif type_covered and not item_covered:
        reason = (f"Policy covers {coverage_key} but '{damaged_object}' is NOT in the covered items list. "
                  f"Covered items: {', '.join(covered_items_for_type)}")
    else:
        reason = f"Policy does NOT cover {coverage_key}. Covered: {', '.join(coverage_types)}"

    return {
        "is_covered": is_covered,
        "reason": reason,
        "policy_number": policy.get("policy_number"),
        "coverage_types": coverage_types,
        "covered_items": covered_items_for_type,
        "coverage_limit_eur": policy.get("coverage_limit_eur", 0),
        "deductible_eur": policy.get("deductible_eur", 0),
    }


def calculate_payout(coverage_result, claimed_amount):
    """
    Calculates the payout amount based on coverage and claimed amount.
    Deterministic business logic — no LLM involved.
    """
    if not coverage_result["is_covered"]:
        return {
            "approved": False,
            "payout_amount_eur": 0,
            "reason": coverage_result["reason"],
        }

    if not claimed_amount or claimed_amount <= 0:
        return {
            "approved": False,
            "payout_amount_eur": 0,
            "reason": "No valid claimed amount found in invoice",
        }

    deductible = coverage_result["deductible_eur"]
    limit = coverage_result["coverage_limit_eur"]

    # Payout = claimed amount minus deductible, capped at coverage limit
    payout = max(0, claimed_amount - deductible)
    payout = min(payout, limit)

    return {
        "approved": True,
        "claimed_amount_eur": round(claimed_amount, 2),
        "deductible_eur": deductible,
        "payout_amount_eur": round(payout, 2),
        "coverage_limit_eur": limit,
        "reason": f"Claim approved. EUR {round(payout, 2)} after EUR {deductible} deductible (limit: EUR {limit})",
    }


def generate_payout_report():
    """
    Generates the full payout report by:
    1. Loading all extracted claims
    2. Grouping documents by claim number
    3. Looking up policy coverage (deterministic)
    4. Calculating payout amounts (deterministic)

    This function demonstrates why policy checks should NOT use RAG:
    every field is looked up from structured metadata, not searched.
    """
    policies = load_policy_metadata()
    extracted = load_extracted_data()

    if not policies:
        print("No policy metadata found. Run generate_synthetic_data.py first.")
        return []
    if not extracted:
        print("No extracted data found. Run Stage 2 first.")
        return []

    # Group documents by claim number
    claims = {}
    for doc in extracted:
        if doc.get("status") != "success":
            continue
        claim_num = doc.get("claim_number")
        if not claim_num:
            continue
        if claim_num not in claims:
            claims[claim_num] = {"documents": [], "damage_type": None,
                                  "damaged_object": None, "claimant_name": None,
                                  "total_amount_eur": None, "policy_number": None}

        claims[claim_num]["documents"].append(doc)

        # Merge info from different document types
        if doc.get("damage_type"):
            claims[claim_num]["damage_type"] = doc["damage_type"]
        if doc.get("damaged_object"):
            claims[claim_num]["damaged_object"] = doc["damaged_object"]
        if doc.get("claimant_name"):
            claims[claim_num]["claimant_name"] = doc["claimant_name"]
        if doc.get("total_amount_eur"):
            claims[claim_num]["total_amount_eur"] = doc["total_amount_eur"]
        if doc.get("policy_number"):
            claims[claim_num]["policy_number"] = doc["policy_number"]

    print(f"Processing {len(claims)} unique claims for payout decisions...\n")

    payout_decisions = []
    approved_count = 0
    denied_count = 0
    total_payout = 0

    for claim_number, claim_info in claims.items():
        claimant = claim_info["claimant_name"]
        damage_type = claim_info["damage_type"]
        damaged_object = claim_info["damaged_object"]
        claimed_amount = claim_info["total_amount_eur"]
        doc_types = [d.get("document_type") for d in claim_info["documents"]]

        # Step 1: Find policy (deterministic lookup)
        policy = find_policy_for_claimant(claimant, policies)

        # Step 2: Check coverage — both damage type AND damaged object (deterministic)
        coverage = check_coverage(policy, damage_type, damaged_object)

        # Step 3: Calculate payout (deterministic)
        payout = calculate_payout(coverage, claimed_amount)

        decision = {
            "claim_number": claim_number,
            "claimant_name": claimant,
            "damage_type": damage_type,
            "damaged_object": damaged_object,
            "claimed_amount_eur": claimed_amount,
            "document_types": doc_types,
            "policy_number": coverage["policy_number"],
            "coverage_check": coverage,
            "payout_decision": payout,
            "processed_at": datetime.now().isoformat(),
        }
        payout_decisions.append(decision)

        if payout["approved"]:
            approved_count += 1
            total_payout += payout["payout_amount_eur"]
            status = f"APPROVED - EUR {payout['payout_amount_eur']:.2f}"
        else:
            denied_count += 1
            status = f"DENIED - {payout['reason'][:60]}"

        print(f"  {claim_number} | {claimant or 'Unknown'} | {damage_type or '?'} | {status}")

    # Save payout report
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_claims": len(claims),
        "approved": approved_count,
        "denied": denied_count,
        "total_payout_eur": round(total_payout, 2),
        "decisions": payout_decisions,
    }

    with open(PAYOUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPayout Report Summary:")
    print(f"  Total claims: {len(claims)}")
    print(f"  Approved: {approved_count}")
    print(f"  Denied: {denied_count}")
    print(f"  Total payout: EUR {total_payout:,.2f}")
    print(f"  Report saved to: {PAYOUT_REPORT}")

    return payout_decisions


def run_test_queries():
    """
    Runs test queries that exercise the business workflow:
    damage classification, object identification, amount extraction.
    """
    test_queries = [
        {
            "query": "What water damage claims have been filed? List the damaged objects.",
            "filter": {"damage_type": "water"},
        },
        {
            "query": "Show me all storm damage claims and what was damaged.",
            "filter": {"damage_type": "storm"},
        },
        {
            "query": "What glass damage claims exist? What objects were broken?",
            "filter": {"damage_type": "glass"},
        },
        {
            "query": "What are the total amounts claimed in the invoices?",
            "filter": {"document_type": "invoice"},
        },
        {
            "query": "Which claims have severe damage according to the photo documentation?",
            "filter": {"document_type": "photo_documentation"},
        },
    ]

    print(f"Running test queries (reranking: {'ON' if RERANK_ENABLED else 'OFF'})...\n")

    for i, test in enumerate(test_queries):
        print(f"Query {i+1}: {test['query']}")
        if test["filter"]:
            print(f"Filter: {test['filter']}")

        result = query_pipeline(
            query=test["query"],
            metadata_filter=test["filter"],
            n_results=DEFAULT_N_RESULTS,
        )

        print(f"Answer: {result['answer'][:400]}...")
        print(f"Chunks retrieved: {result['chunks_retrieved']}")
        avg_dist = (
            round(sum(c["distance"] for c in result["chunks"]) / len(result["chunks"]), 3)
            if result["chunks"] else "N/A"
        )
        print(f"Avg distance: {avg_dist}")
        print(f"Query ID: {result['query_id']}")
        print("-" * 60 + "\n")

    # Also generate the payout report
    print("\n" + "=" * 60)
    print("GENERATING PAYOUT REPORT (Deterministic Policy Check)")
    print("=" * 60 + "\n")
    generate_payout_report()

    print(f"\nDone. {len(test_queries)} queries logged to {QUERY_LOG_PATH}")


if __name__ == "__main__":
    run_test_queries()

# query_intelligence.py
# ═══════════════════════════════════════════════════════════════════
# QUERY INTELLIGENCE LAYER — The #1 missing piece in naive RAG
# ═══════════════════════════════════════════════════════════════════
#
# Why this matters:
#   In your old pipeline, the raw user query goes straight to ChromaDB.
#   "What happened with the Mueller claim?" embeds as a generic question.
#   The dense retriever finds chunks about "claims" and "happening" — not
#   necessarily about Mueller.
#
#   This module fixes retrieval BEFORE it happens, using three techniques:
#
#   1. QUERY ROUTING — Classifies query complexity and picks a strategy.
#      Simple factual lookups get fast single-shot retrieval.
#      Complex analytical queries get multi-step agentic retrieval.
#
#   2. HyDE (Hypothetical Document Embeddings) — Instead of embedding
#      the question, we ask the LLM to imagine what a perfect answer
#      document would look like, then embed THAT. This bridges the
#      semantic gap between questions and documents.
#      Paper: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval
#      without Relevance Labels"
#
#   3. MULTI-QUERY EXPANSION — For complex queries, the LLM generates
#      3-5 reformulated sub-queries. We retrieve for each, deduplicate,
#      and merge results. This catches documents that any single query
#      phrasing would miss.
#
# These three together fix ~60% of retrieval failures in production RAG.

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION,
    estimate_llm_cost, LOG_FOLDER, LOG_FORMAT,
)
import os

os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "query_intelligence.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

_azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT


# ─── Query Classification ────────────────────────────────────────

class QueryComplexity(str, Enum):
    """
    Determines which retrieval strategy to use.

    SIMPLE — Single-fact questions. One retrieval pass is enough.
      Examples: "What is claim CLM-2024-001 about?"
                "Who filed the water damage claim?"

    MODERATE — Needs targeted retrieval with better query formulation.
      HyDE or metadata-filtered retrieval.
      Examples: "What water damage claims have been filed?"
                "Show me invoices over EUR 1000"

    COMPLEX — Multi-hop reasoning, comparisons, aggregations.
      Needs multi-query expansion + agentic multi-step retrieval.
      Examples: "Compare all denied claims and explain the denial reasons"
                "Which claimant has the most expensive total claims?"

    STRUCTURED — Can be answered from structured data alone, no RAG needed.
      Routes to deterministic lookup (policy checks, payout calculations).
      Examples: "Is Mueller covered for water damage?"
                "What's the payout for claim CLM-2024-003?"
    """
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    STRUCTURED = "structured"


@dataclass
class QueryPlan:
    """The output of query intelligence — tells the retrieval layer what to do."""
    original_query: str
    complexity: QueryComplexity
    strategy: str  # "single_shot", "hyde", "multi_query", "structured_lookup", "agentic"
    hyde_document: Optional[str] = None  # hypothetical document for HyDE
    expanded_queries: list[str] = field(default_factory=list)  # multi-query expansions
    metadata_filters: Optional[dict] = None  # extracted filters from query
    extracted_entities: dict = field(default_factory=dict)  # claim numbers, names, etc.
    reasoning: str = ""  # why this classification was chosen
    token_usage: dict = field(default_factory=dict)


# ─── Entity Extraction from Query ────────────────────────────────

# Pre-compiled patterns for fast entity extraction from queries
_CLAIM_NUMBER_RE = re.compile(r'CLM[-–]\d{4}[-–]\d{3,}', re.IGNORECASE)
_DAMAGE_TYPE_RE = re.compile(r'\b(water|storm|glass)\s*(damage)?\b', re.IGNORECASE)
_AMOUNT_RE = re.compile(r'EUR\s*[\d,.]+|\d+[\d,.]*\s*EUR', re.IGNORECASE)
_NAME_INDICATORS = re.compile(
    r'\b(?:claimant|policyholder|customer|herr|frau|mr|mrs|ms)\s+([A-ZÄÖÜa-zäöüß]+(?:\s+[A-ZÄÖÜa-zäöüß]+)*)',
    re.IGNORECASE,
)
_DOC_TYPE_RE = re.compile(
    r'\b(invoice|email|claim.?email|photo.?documentation|photo)\b', re.IGNORECASE
)


def extract_entities_from_query(query: str) -> dict:
    """
    Fast regex-based entity extraction from user queries.
    No LLM call needed — this runs in microseconds.

    Why regex instead of NER?
    Insurance domain has highly structured identifiers (CLM-XXXX-XXX,
    EUR amounts, specific damage types). Regex is 100% precise for these,
    whereas NER models add latency and can miss domain-specific patterns.
    """
    entities = {}

    claim_matches = _CLAIM_NUMBER_RE.findall(query)
    if claim_matches:
        entities["claim_numbers"] = claim_matches

    damage_matches = _DAMAGE_TYPE_RE.findall(query)
    if damage_matches:
        entities["damage_types"] = list(set(m[0].lower() for m in damage_matches))

    amount_matches = _AMOUNT_RE.findall(query)
    if amount_matches:
        entities["amounts"] = amount_matches

    name_matches = _NAME_INDICATORS.findall(query)
    if name_matches:
        entities["names"] = name_matches

    doc_type_matches = _DOC_TYPE_RE.findall(query)
    if doc_type_matches:
        # Normalize to pipeline document types
        type_map = {
            "invoice": "invoice",
            "email": "claim_email",
            "claim_email": "claim_email",
            "claimemail": "claim_email",
            "photo_documentation": "photo_documentation",
            "photodocumentation": "photo_documentation",
            "photo": "photo_documentation",
        }
        entities["document_types"] = list(set(
            type_map.get(m.lower().replace(" ", "").replace("-", ""), m.lower())
            for m in doc_type_matches
        ))

    return entities


def _build_metadata_filter(entities: dict) -> Optional[dict]:
    """
    Converts extracted entities into a ChromaDB metadata filter.
    Only creates a filter if we can be precise — vague filters hurt more than help.
    """
    conditions = []

    if "damage_types" in entities and len(entities["damage_types"]) == 1:
        conditions.append({"damage_type": entities["damage_types"][0]})

    if "document_types" in entities and len(entities["document_types"]) == 1:
        conditions.append({"document_type": entities["document_types"][0]})

    if "claim_numbers" in entities and len(entities["claim_numbers"]) == 1:
        conditions.append({"claim_number": entities["claim_numbers"][0]})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ─── Query Classification ────────────────────────────────────────

# Rule-based classification first (fast, no LLM call).
# Falls back to LLM classification for ambiguous queries.

_STRUCTURED_PATTERNS = [
    re.compile(r'\b(covered|coverage|policy|payout|deductible|limit)\b', re.I),
    re.compile(r'\b(is .* covered|check coverage|policy lookup)\b', re.I),
    re.compile(r'\b(calculate|compute) .* payout\b', re.I),
]

_COMPLEX_INDICATORS = [
    re.compile(r'\b(compare|contrast|difference|versus|vs)\b', re.I),
    re.compile(r'\b(all|every|each|summary of all|total across)\b', re.I),
    re.compile(r'\b(why.*denied|explain.*reason|analyze)\b', re.I),
    re.compile(r'\b(most expensive|highest|lowest|ranking|trend)\b', re.I),
    re.compile(r'\b(how many|count|total number)\b', re.I),
    re.compile(r'\b(relationship|connected|linked|related to)\b', re.I),
]


def _classify_rule_based(query: str, entities: dict) -> Optional[QueryComplexity]:
    """
    Fast rule-based classification. Returns None if uncertain → falls back to LLM.

    Why rules first?
    - No latency for simple queries (saves 500ms–1s per query)
    - Deterministic — same query always gets same classification
    - LLM classification is reserved for genuinely ambiguous queries
    """
    q_lower = query.lower().strip()

    # Structured: policy/coverage questions with specific identifiers
    if entities.get("names") or entities.get("claim_numbers"):
        for p in _STRUCTURED_PATTERNS:
            if p.search(q_lower):
                return QueryComplexity.STRUCTURED

    # Complex: aggregation, comparison, multi-hop reasoning
    complex_hits = sum(1 for p in _COMPLEX_INDICATORS if p.search(q_lower))
    if complex_hits >= 2:
        return QueryComplexity.COMPLEX

    # Simple: specific entity mentioned + short query
    if entities.get("claim_numbers") and len(query.split()) < 15:
        return QueryComplexity.SIMPLE

    # Moderate: has a damage type or doc type filter but isn't complex
    if entities.get("damage_types") or entities.get("document_types"):
        return QueryComplexity.MODERATE

    # Short queries are usually simple
    if len(query.split()) < 8:
        return QueryComplexity.SIMPLE

    return None  # Uncertain → LLM classification


_CLASSIFY_PROMPT = """Classify this insurance claims query into exactly one category.

Categories:
- "simple": Single-fact lookup. One retrieval pass answers it.
- "moderate": Needs targeted search with filters or better query formulation.
- "complex": Multi-hop reasoning, comparisons, or aggregations across multiple documents.
- "structured": Answerable from structured policy/payout data alone, no document search needed.

Query: {query}

Respond with ONLY a JSON object:
{{"complexity": "<simple|moderate|complex|structured>", "reasoning": "<one sentence>"}}"""


def _classify_with_llm(query: str) -> tuple[QueryComplexity, str, dict]:
    """LLM-based classification for queries where rules are insufficient."""
    prompt = _CLASSIFY_PROMPT.format(query=query)
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

        # Parse response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        complexity = QueryComplexity(result.get("complexity", "moderate"))
        reasoning = result.get("reasoning", "")
        return complexity, reasoning, token_usage

    except Exception as e:
        logging.warning(f"LLM classification failed: {e}, defaulting to moderate")
        return QueryComplexity.MODERATE, f"LLM fallback: {e}", {}


# ─── HyDE: Hypothetical Document Embeddings ─────────────────────
#
# Core idea: Instead of searching for "What water damage claims exist?",
# we generate a hypothetical ANSWER document:
#   "Water damage claim CLM-2024-001 was filed by Hans Mueller on
#    2024-03-15 for damage to kitchen ceiling caused by pipe burst..."
#
# We embed this hypothetical document and search for SIMILAR documents.
# This works because the hypothetical answer is semantically closer to
# actual answer documents than the question is.
#
# The hypothetical document doesn't need to be factually correct —
# it just needs to be in the right "semantic neighborhood."

_HYDE_PROMPT = """You are generating a hypothetical insurance document that would answer this question.
The document should look like a real insurance claim document (email, invoice, or photo report).
Include realistic details: claim numbers (CLM-XXXX-XXX format), damage types, amounts in EUR, German names.
Do NOT answer the question — generate a DOCUMENT that would contain the answer.

Question: {query}

Generate a realistic 100-150 word insurance document:"""


def generate_hyde_document(query: str) -> tuple[str, dict]:
    """
    Generates a hypothetical document for HyDE retrieval.

    Returns:
        (hypothetical_document, token_usage)
    """
    try:
        response = _azure_client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            max_completion_tokens=250,
            temperature=0.7,  # Some creativity for diverse documents
        )
        hyde_doc = response.choices[0].message.content.strip()
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cost_usd": estimate_llm_cost(
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
        }
        logging.info(f"HyDE document generated ({len(hyde_doc)} chars)")
        return hyde_doc, token_usage

    except Exception as e:
        logging.error(f"HyDE generation failed: {e}")
        return query, {}  # Fallback to original query


# ─── Multi-Query Expansion ───────────────────────────────────────
#
# For complex queries, a single search often misses relevant documents
# because one phrasing can't cover all aspects. Multi-query generates
# 3-5 reformulations that each capture a different facet:
#
# Original: "Compare denied water damage claims and explain why they were denied"
# Expanded:
#   1. "water damage claims that were denied"
#   2. "denial reasons for insurance claims water"
#   3. "coverage check water damage policy"
#   4. "claims not covered water damage"
#
# Each expansion retrieves independently, results are deduplicated and
# merged using Reciprocal Rank Fusion (RRF).

_MULTI_QUERY_PROMPT = """Generate {n} different search queries to find insurance documents that would help answer this question.
Each query should approach the question from a different angle.
Queries should be specific, like real document search queries.

Original question: {query}

Respond with ONLY a JSON array of strings, no explanation:
["{n} different search queries"]"""


def expand_query(query: str, n: int = 4) -> tuple[list[str], dict]:
    """
    Generates multiple reformulations of a query for broader retrieval.

    Returns:
        (list_of_expanded_queries, token_usage)

    The original query is always included as the first item.
    """
    try:
        response = _azure_client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _MULTI_QUERY_PROMPT.format(query=query, n=n)}],
            max_completion_tokens=300,
            temperature=0.5,
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

        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = [query]

        # Always include original query first
        all_queries = [query] + [q for q in queries if q != query]
        logging.info(f"Multi-query: expanded to {len(all_queries)} queries")
        return all_queries, token_usage

    except Exception as e:
        logging.error(f"Multi-query expansion failed: {e}")
        return [query], {}


# ─── Reciprocal Rank Fusion (RRF) ────────────────────────────────
#
# When we retrieve from multiple queries, we need to merge the result
# lists. RRF is the standard method (used by Elasticsearch, Pinecone,
# Weaviate, and most production RAG systems in 2026).
#
# For each document, its RRF score is:
#   score = Σ 1 / (k + rank_in_list_i)
#
# Where k=60 is a constant that prevents top-ranked docs from dominating.
# Documents that appear in multiple result lists get boosted.

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 5,
) -> list[dict]:
    """
    Merges multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of retrieval results from different queries.
                      Each item is a list of chunk dicts with "text" and "metadata".
        k: RRF constant (default 60, standard in literature)
        top_n: Number of final results to return

    Returns:
        Merged and re-ranked list of unique chunks.
    """
    # Score accumulator keyed by chunk_id (or text hash as fallback)
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            # Use chunk_id from metadata if available, else hash the text
            chunk_id = chunk.get("metadata", {}).get("chunk_id") or str(hash(chunk.get("text", "")))

            rrf_score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

            # Keep the chunk with the best individual score
            if chunk_id not in chunk_map or chunk.get("distance", 2.0) < chunk_map[chunk_id].get("distance", 2.0):
                chunk_map[chunk_id] = chunk

    # Sort by RRF score and take top_n
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_n]

    results = []
    for cid in ranked_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(scores[cid], 6)
        results.append(chunk)

    logging.info(
        f"RRF: merged {len(result_lists)} result lists "
        f"({sum(len(r) for r in result_lists)} total chunks) "
        f"-> {len(results)} unique results"
    )

    return results


# ─── Main Entry Point ────────────────────────────────────────────

def analyze_query(query: str) -> QueryPlan:
    """
    The main entry point for query intelligence.
    Analyzes the query and produces a QueryPlan that tells the
    retrieval layer exactly what to do.

    Flow:
    1. Extract entities (regex, fast)
    2. Build metadata filter from entities
    3. Classify complexity (rules first, LLM fallback)
    4. Pick strategy based on complexity
    5. Generate HyDE document or expand queries as needed

    Returns a QueryPlan dataclass.
    """
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def _accumulate_tokens(usage: dict):
        for k in total_tokens:
            total_tokens[k] = round(total_tokens[k] + usage.get(k, 0), 6)

    # Step 1: Entity extraction (fast, no LLM)
    entities = extract_entities_from_query(query)
    metadata_filter = _build_metadata_filter(entities)

    # Step 2: Classify complexity
    complexity = _classify_rule_based(query, entities)
    reasoning = ""

    if complexity is None:
        complexity, reasoning, classify_tokens = _classify_with_llm(query)
        _accumulate_tokens(classify_tokens)
    else:
        reasoning = f"Rule-based: classified as {complexity.value}"

    # Step 3: Pick strategy and execute preparatory steps
    plan = QueryPlan(
        original_query=query,
        complexity=complexity,
        strategy="single_shot",  # default
        metadata_filters=metadata_filter,
        extracted_entities=entities,
        reasoning=reasoning,
    )

    if complexity == QueryComplexity.STRUCTURED:
        plan.strategy = "structured_lookup"
        # No retrieval needed — deterministic policy/payout lookup

    elif complexity == QueryComplexity.SIMPLE:
        plan.strategy = "single_shot"
        # Plain vector search is fine for simple queries

    elif complexity == QueryComplexity.MODERATE:
        # Use HyDE for better retrieval on moderate queries
        plan.strategy = "hyde"
        hyde_doc, hyde_tokens = generate_hyde_document(query)
        plan.hyde_document = hyde_doc
        _accumulate_tokens(hyde_tokens)

    elif complexity == QueryComplexity.COMPLEX:
        # Full multi-query expansion + agentic retrieval for complex queries
        plan.strategy = "agentic"
        expanded, expand_tokens = expand_query(query, n=4)
        plan.expanded_queries = expanded
        _accumulate_tokens(expand_tokens)

        # Also generate HyDE for the original query
        hyde_doc, hyde_tokens = generate_hyde_document(query)
        plan.hyde_document = hyde_doc
        _accumulate_tokens(hyde_tokens)

    plan.token_usage = total_tokens

    logging.info(
        f"Query plan: complexity={complexity.value} strategy={plan.strategy} "
        f"entities={len(entities)} filters={'yes' if metadata_filter else 'no'} "
        f"expanded_queries={len(plan.expanded_queries)}"
    )

    return plan

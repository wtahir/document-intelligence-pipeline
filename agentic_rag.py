# agentic_rag.py
# ═══════════════════════════════════════════════════════════════════
# AGENTIC RAG ORCHESTRATOR — The 2026 SOTA retrieval brain
# ═══════════════════════════════════════════════════════════════════
#
# Why this is the biggest upgrade:
#   Your old query_pipeline() does: retrieve → rerank → generate.
#   One shot, one chance. If retrieval fails, the answer is bad.
#
#   Agentic RAG adds INTELLIGENCE to the retrieval process:
#   - The LLM DECIDES which retrieval strategy to use
#   - If first retrieval is bad, it RETRIES with a different approach
#   - It can USE TOOLS (graph lookup, policy check, structured query)
#   - It CRITIQUES its own answer before returning it
#
# Architecture:
#   ┌──────────────────────────────────────────────────────────┐
#   │  User Query                                              │
#   │    ↓                                                     │
#   │  Query Intelligence (classify + route + expand)          │
#   │    ↓                                                     │
#   │  ┌──────────────────────────────────────────────┐        │
#   │  │  RETRIEVAL LOOP (max 3 iterations)           │        │
#   │  │    ↓                                         │        │
#   │  │  Strategy Execution:                         │        │
#   │  │    - Single-shot (simple queries)            │        │
#   │  │    - HyDE retrieval (moderate queries)       │        │
#   │  │    - Multi-query + RRF (complex queries)     │        │
#   │  │    - Graph traversal (relationship queries)  │        │
#   │  │    - Structured lookup (policy queries)      │        │
#   │  │    ↓                                         │        │
#   │  │  Context Engineering:                        │        │
#   │  │    - Dedup → Organize → Compress → Enrich    │        │
#   │  │    ↓                                         │        │
#   │  │  Sufficiency Check:                          │        │
#   │  │    - Is context enough? → Yes: proceed       │        │
#   │  │    - No: reformulate and retry               │        │
#   │  └──────────────────────────────────────────────┘        │
#   │    ↓                                                     │
#   │  Answer Generation (with enriched context)               │
#   │    ↓                                                     │
#   │  Self-Critique (verify answer quality)                   │
#   │    ↓                                                     │
#   │  Final Answer (with citations, confidence, metadata)     │
#   └──────────────────────────────────────────────────────────┘
#
# This replaces the old single-shot pipeline with an intelligent,
# self-correcting retrieval system. The key insight: retrieval is
# not a single step — it's a LOOP that should iterate until the
# context is sufficient or a retry budget is exhausted.

import os
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import (
    QUERY_LOG, OUTPUT_FOLDER, CHROMA_FOLDER, CHROMA_COLLECTION,
    EMBEDDING_MODEL, RERANKER_MODEL, RERANK_ENABLED, RERANK_TOP_K,
    DEFAULT_N_RESULTS, RETRIEVAL_OVER_FETCH, CONFIDENCE_THRESHOLD,
    HYBRID_SEARCH_ENABLED, BM25_WEIGHT, PII_REDACTION_ENABLED,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION,
    LOG_FOLDER, LOG_FORMAT, POLICY_METADATA, EXTRACTED_DATA,
    estimate_llm_cost,
)
from pii_redactor import redact_chunks, unredact

from query_intelligence import (
    analyze_query, QueryComplexity, QueryPlan,
    reciprocal_rank_fusion, extract_entities_from_query,
)
from knowledge_graph import load_graph, get_graph_context
from context_engine import (
    deduplicate_chunks, organize_hierarchically,
    apply_relevance_compression, build_enriched_context,
    check_context_sufficiency,
)

# Import existing stage5 functions we still need
from stage5_retrieval import (
    retrieve_chunks, rerank_chunks, hybrid_rerank,
    _check_prompt_injection, _check_confidence, _verify_citations,
    generate_answer, build_context,
    load_query_log, save_query_log,
    find_policy_for_claimant, check_coverage, calculate_payout,
    load_policy_metadata, load_extracted_data,
    SYSTEM_PROMPT,
)

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "agentic_rag.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)


# ─── Enhanced System Prompt ───────────────────────────────────────
# The system prompt now acknowledges structured facts from the graph
# and the layered context structure.

ENHANCED_SYSTEM_PROMPT = """You are an insurance claims processing assistant with access to both
structured knowledge (from a knowledge graph) and unstructured documents (claim emails, invoices, photo reports).

RESPONSE RULES:
1. Prioritize STRUCTURED FACTS when available — they are verified and deterministic.
2. Use DOCUMENT CONTEXT for details not captured in structured facts.
3. Cite sources using [Chunk N] notation for every factual claim from documents.
4. If structured facts and document context conflict, note the discrepancy.
5. If context is insufficient, say "INSUFFICIENT_CONTEXT" followed by what's missing.
6. Never fabricate information. Never make up claim numbers, amounts, or names.
7. When referencing specific claims, always include the claim number.
8. For coverage/payout questions, note that these are policy-based decisions, not interpretations.

DOCUMENT TYPES YOU MAY ENCOUNTER:
- Claim emails: Damage reports from policyholders
- Invoices: Repair cost documentation with line items and totals
- Photo documentation: Damage severity assessments with repair recommendations"""


# ─── Self-Critique Prompt ─────────────────────────────────────────

_SELF_CRITIQUE_PROMPT = """Review this answer for quality. Check for:
1. Does it answer the question completely?
2. Are all factual claims supported by [Chunk N] citations?
3. Are there any statements that seem fabricated or unsupported?
4. Is anything important missing?

Question: {query}
Answer: {answer}

Respond with ONLY valid JSON:
{{
  "quality": "good"|"needs_improvement"|"poor",
  "issues": ["list of specific issues, empty if good"],
  "missing_info": "what's missing, or 'nothing'",
  "should_retry": true/false
}}"""


@dataclass
class AgenticResult:
    """Complete result from the agentic RAG pipeline."""
    query_id: str
    query: str
    query_plan: dict
    iterations: int
    strategy_used: str
    chunks_retrieved: int
    graph_facts_used: int
    context_sufficient: bool
    answer: str
    citation_info: dict = field(default_factory=dict)
    self_critique: dict = field(default_factory=dict)
    confidence_pass: bool = True
    confidence_reason: str = "OK"
    token_usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    chunks: list = field(default_factory=list)
    metadata_filter: Optional[dict] = None
    reranking_enabled: bool = RERANK_ENABLED
    hybrid_search_enabled: bool = HYBRID_SEARCH_ENABLED
    pii_redaction_enabled: bool = PII_REDACTION_ENABLED
    queried_at: str = ""
    evaluated: bool = False
    score: Optional[float] = None
    evaluation_notes: Optional[str] = None


# ─── Retrieval Strategies ─────────────────────────────────────────

def _execute_single_shot(query: str, metadata_filter: Optional[dict], n_results: int) -> list[dict]:
    """Standard single-shot retrieval with reranking."""
    chunks = retrieve_chunks(query, metadata_filter, n_results)
    if RERANK_ENABLED and chunks:
        chunks = rerank_chunks(query, chunks, top_k=n_results)
    elif HYBRID_SEARCH_ENABLED and chunks:
        chunks = hybrid_rerank(query, chunks, n_results)
    return chunks


def _execute_hyde(query: str, hyde_doc: str, metadata_filter: Optional[dict], n_results: int) -> list[dict]:
    """
    HyDE retrieval: search using the hypothetical document embedding
    instead of the question embedding. Also search with original query
    and merge results using RRF.
    """
    # Retrieve using hypothetical document
    hyde_chunks = retrieve_chunks(hyde_doc, metadata_filter, n_results)

    # Also retrieve using original query
    original_chunks = retrieve_chunks(query, metadata_filter, n_results)

    # Merge with RRF
    merged = reciprocal_rank_fusion([hyde_chunks, original_chunks], top_n=n_results)

    # Rerank the merged results
    if RERANK_ENABLED and merged:
        merged = rerank_chunks(query, merged, top_k=n_results)

    return merged


def _execute_multi_query(
    original_query: str,
    expanded_queries: list[str],
    hyde_doc: Optional[str],
    metadata_filter: Optional[dict],
    n_results: int,
) -> list[dict]:
    """
    Multi-query retrieval: retrieve for each expanded query,
    merge with RRF, then rerank.
    """
    result_lists = []

    # Retrieve for each expanded query
    for q in expanded_queries:
        chunks = retrieve_chunks(q, metadata_filter, n_results)
        if chunks:
            result_lists.append(chunks)

    # Also retrieve with HyDE document if available
    if hyde_doc:
        hyde_chunks = retrieve_chunks(hyde_doc, metadata_filter, n_results)
        if hyde_chunks:
            result_lists.append(hyde_chunks)

    if not result_lists:
        return []

    # Merge all results with RRF
    merged = reciprocal_rank_fusion(result_lists, top_n=n_results * 2)

    # Rerank the top merged results
    if RERANK_ENABLED and merged:
        merged = rerank_chunks(original_query, merged, top_k=n_results)

    return merged


def _execute_graph_enhanced(
    query: str,
    query_plan: QueryPlan,
    metadata_filter: Optional[dict],
    n_results: int,
) -> tuple[list[dict], list[str]]:
    """
    Graph-enhanced retrieval: use knowledge graph to find related
    entities and their source files, then retrieve from those files
    in addition to standard vector search.

    Returns:
        (chunks, graph_facts)
    """
    graph = load_graph()
    graph_facts = []

    if graph:
        graph_context = get_graph_context(query_plan.extracted_entities, graph)
        graph_facts = graph_context.get("graph_facts", [])
        related_files = graph_context.get("related_files", [])

        # Retrieve from related files (graph-guided retrieval)
        graph_chunks = []
        for file_name in related_files[:5]:  # Limit to prevent over-retrieval
            file_chunks = retrieve_chunks(
                query,
                metadata_filter={"file_name": file_name},
                n_results=3,
            )
            graph_chunks.extend(file_chunks)

        # Also do standard vector search
        standard_chunks = retrieve_chunks(query, metadata_filter, n_results)

        # Merge graph and standard results
        all_result_lists = [standard_chunks]
        if graph_chunks:
            all_result_lists.append(graph_chunks)

        merged = reciprocal_rank_fusion(all_result_lists, top_n=n_results)

        if RERANK_ENABLED and merged:
            merged = rerank_chunks(query, merged, top_k=n_results)

        return merged, graph_facts

    # Fallback to standard retrieval if no graph
    chunks = _execute_single_shot(query, metadata_filter, n_results)
    return chunks, graph_facts


def _execute_structured_lookup(query: str, entities: dict) -> Optional[dict]:
    """
    Handles structured/policy queries deterministically.
    Returns a structured answer dict, or None if not applicable.
    """
    policies = load_policy_metadata()
    if not policies:
        return None

    # Try to find the claimant
    claimant_name = None
    if entities.get("names"):
        claimant_name = entities["names"][0]
    elif entities.get("claim_numbers"):
        # Look up claimant by claim number from extracted data
        extracted = load_extracted_data()
        for doc in extracted:
            if doc.get("claim_number") in entities["claim_numbers"]:
                claimant_name = doc.get("claimant_name")
                break

    if not claimant_name:
        return None

    policy = find_policy_for_claimant(claimant_name, policies)

    damage_types = entities.get("damage_types", [])
    damage_type = damage_types[0] if damage_types else None

    coverage = check_coverage(policy, damage_type)

    return {
        "claimant": claimant_name,
        "policy": policy,
        "coverage": coverage,
        "query_type": "structured_lookup",
    }


# ─── Self-Critique ────────────────────────────────────────────────

def _self_critique(query: str, answer: str) -> tuple[dict, dict]:
    """
    LLM self-critique of the generated answer.
    Returns (critique_result, token_usage).
    """
    from openai import AzureOpenAI as _AzureOpenAI
    client = _AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": _SELF_CRITIQUE_PROMPT.format(
                query=query, answer=answer,
            )}],
            max_completion_tokens=300,
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
        critique = json.loads(raw)
        return critique, token_usage

    except Exception as e:
        logging.error(f"Self-critique failed: {e}")
        return {"quality": "unknown", "issues": [str(e)], "should_retry": False}, {}


# ─── Enhanced Answer Generation ───────────────────────────────────

def _generate_answer_enhanced(query: str, context: str, structured_info: Optional[dict] = None) -> tuple[str, dict]:
    """
    Enhanced answer generation with structured facts integration.
    """
    from openai import AzureOpenAI as _AzureOpenAI
    client = _AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    # Build user message
    parts = [f"Context from insurance claim documents:\n\n{context}"]

    if structured_info:
        parts.append(f"\nStructured policy information:\n{json.dumps(structured_info, indent=2, default=str)}")

    parts.append(f"\nQuestion: {query}")
    parts.append("\nAnswer based on the context. Cite sources using [Chunk N] notation:")

    user_message = "\n".join(parts)

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=1000,
    )

    answer = response.choices[0].message.content.strip()
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

    return answer, token_usage


# ─── Main Agentic Pipeline ───────────────────────────────────────

def agentic_query_pipeline(
    query: str,
    metadata_filter: Optional[dict] = None,
    n_results: int = DEFAULT_N_RESULTS,
    max_iterations: int = 3,
    enable_self_critique: bool = True,
    enable_graph: bool = True,
    enable_context_engineering: bool = True,
) -> dict:
    """
    The 2026 SOTA agentic RAG pipeline.

    Replaces the old query_pipeline() with an intelligent, multi-step,
    self-correcting retrieval system.

    Args:
        query: User's question
        metadata_filter: Optional ChromaDB filter (can be auto-detected)
        n_results: Number of chunks to retrieve per pass
        max_iterations: Maximum retrieval attempts before giving up
        enable_self_critique: Whether to run self-critique on the answer
        enable_graph: Whether to use knowledge graph enhancement
        enable_context_engineering: Whether to apply context engineering

    Returns:
        dict with answer, chunks, metadata, timing, costs, etc.
    """
    _start = time.monotonic()
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}

    def _acc(usage: dict):
        for k in total_tokens:
            total_tokens[k] = round(total_tokens[k] + usage.get(k, 0), 6)

    logging.info(f"Agentic query: {query[:100]} | filter={metadata_filter}")

    # --- Safety check ---
    is_safe, injection_reason = _check_prompt_injection(query)
    if not is_safe:
        logging.warning(f"Prompt injection blocked: {query[:100]}")
        result = {
            "query_id": f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "query": query,
            "blocked": True,
            "blocked_reason": injection_reason,
            "answer": "This query was blocked by the safety filter. Please rephrase your question.",
            "pipeline_version": "agentic_v1",
            "latency_seconds": round(time.monotonic() - _start, 3),
            "queried_at": datetime.now().isoformat(),
            "evaluated": False,
        }
        log = load_query_log()
        log.append(result)
        save_query_log(log)
        return result

    # --- Step 1: Query Intelligence ---
    query_plan = analyze_query(query)
    _acc(query_plan.token_usage)

    # Use auto-detected filter if none provided
    if metadata_filter is None and query_plan.metadata_filters:
        metadata_filter = query_plan.metadata_filters

    # --- Step 2: Structured lookup (if applicable) ---
    structured_info = None
    if query_plan.strategy == "structured_lookup":
        structured_info = _execute_structured_lookup(query, query_plan.extracted_entities)
        if structured_info:
            # For pure structured queries, we still do a light retrieval
            # for context but the answer is primarily from structured data
            pass

    # --- Step 3: Retrieval Loop ---
    chunks = []
    graph_facts = []
    iteration = 0
    context_sufficient = False

    while iteration < max_iterations and not context_sufficient:
        iteration += 1
        logging.info(f"Retrieval iteration {iteration}/{max_iterations}")

        if query_plan.strategy == "single_shot" or query_plan.strategy == "structured_lookup":
            chunks = _execute_single_shot(query, metadata_filter, n_results)

        elif query_plan.strategy == "hyde":
            chunks = _execute_hyde(
                query, query_plan.hyde_document or query,
                metadata_filter, n_results,
            )

        elif query_plan.strategy == "agentic":
            if enable_graph:
                chunks, graph_facts = _execute_graph_enhanced(
                    query, query_plan, metadata_filter, n_results,
                )
            else:
                chunks = _execute_multi_query(
                    query, query_plan.expanded_queries,
                    query_plan.hyde_document, metadata_filter, n_results,
                )

        # --- Step 4: Context Engineering ---
        if enable_context_engineering and chunks:
            chunks = deduplicate_chunks(chunks)
            chunks = organize_hierarchically(chunks)

            # Sufficiency check (only on first iteration, to save costs)
            if iteration == 1:
                is_sufficient, missing, suff_tokens = check_context_sufficiency(
                    query, chunks, graph_facts,
                )
                _acc(suff_tokens)

                if is_sufficient:
                    context_sufficient = True
                else:
                    logging.info(f"Context insufficient: {missing}. Will retry.")
                    # Expand query for retry
                    if not query_plan.expanded_queries:
                        from query_intelligence import expand_query
                        expanded, expand_tokens = expand_query(query, n=3)
                        query_plan.expanded_queries = expanded
                        query_plan.strategy = "agentic"
                        _acc(expand_tokens)
                    continue
            else:
                context_sufficient = True
        else:
            context_sufficient = True

    # --- Step 5: Confidence Check ---
    is_confident = True
    confidence_reason = "OK"

    if not chunks:
        answer = "No relevant documents found for this query."
        token_usage_gen = {}
    else:
        is_confident, avg_distance, confidence_reason = _check_confidence(chunks)

        if not is_confident:
            answer = (
                f"INSUFFICIENT_CONTEXT: {confidence_reason} "
                f"The retrieved documents are not relevant enough to provide a reliable answer. "
                f"Please try rephrasing your query or adding metadata filters."
            )
            token_usage_gen = {}
        else:
            # --- Step 6: Build Context & Generate ---
            if PII_REDACTION_ENABLED:
                redacted, pii_mapping = redact_chunks(chunks)
                if enable_context_engineering:
                    context = build_enriched_context(redacted, graph_facts)
                else:
                    context = build_context(redacted)
            else:
                pii_mapping = {}
                if enable_context_engineering:
                    context = build_enriched_context(chunks, graph_facts)
                else:
                    context = build_context(chunks)

            answer, token_usage_gen = _generate_answer_enhanced(query, context, structured_info)
            _acc(token_usage_gen)

            # Restore PII
            if pii_mapping:
                answer = unredact(answer, pii_mapping)

    # --- Step 7: Self-Critique ---
    critique = {}
    if enable_self_critique and is_confident and chunks:
        critique, critique_tokens = _self_critique(query, answer)
        _acc(critique_tokens)

        # If critique says answer needs improvement and we have budget, retry generation
        if critique.get("should_retry") and iteration < max_iterations:
            logging.info(f"Self-critique triggered retry: {critique.get('issues', [])}")
            # Re-generate with critique feedback
            feedback_context = context + f"\n\n[Self-critique feedback: {', '.join(critique.get('issues', []))}]"
            answer, retry_tokens = _generate_answer_enhanced(query, feedback_context, structured_info)
            _acc(retry_tokens)
            if pii_mapping:
                answer = unredact(answer, pii_mapping)

    # --- Step 8: Citation Verification ---
    citation_info = _verify_citations(answer, len(chunks)) if chunks else {}

    # --- Step 9: Build Result ---
    _duration = round(time.monotonic() - _start, 3)

    result = {
        "query_id": f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "query": query,
        "pipeline_version": "agentic_v1",
        "query_plan": {
            "complexity": query_plan.complexity.value,
            "strategy": query_plan.strategy,
            "entities": query_plan.extracted_entities,
            "expanded_queries": query_plan.expanded_queries,
            "hyde_used": query_plan.hyde_document is not None,
            "reasoning": query_plan.reasoning,
        },
        "metadata_filter": metadata_filter,
        "n_results_requested": n_results,
        "retrieval_iterations": iteration,
        "chunks_retrieved": len(chunks),
        "graph_facts_used": len(graph_facts),
        "graph_facts": graph_facts,
        "context_sufficient": context_sufficient,
        "reranking_enabled": RERANK_ENABLED,
        "hybrid_search_enabled": HYBRID_SEARCH_ENABLED,
        "pii_redaction_enabled": PII_REDACTION_ENABLED,
        "confidence_pass": is_confident,
        "confidence_reason": confidence_reason,
        "citation_info": citation_info,
        "self_critique": critique,
        "structured_info": structured_info if structured_info else None,
        "token_usage": total_tokens,
        "latency_seconds": _duration,
        "chunks": chunks,
        "answer": answer,
        "queried_at": datetime.now().isoformat(),
        "evaluated": False,
        "score": None,
        "evaluation_notes": None,
    }

    # Save to query log
    log = load_query_log()
    log.append(result)
    save_query_log(log)

    logging.info(
        f"Agentic query complete. strategy={query_plan.strategy} "
        f"iterations={iteration} chunks={len(chunks)} "
        f"graph_facts={len(graph_facts)} critique={critique.get('quality', '?')} "
        f"latency={_duration}s cost=${total_tokens.get('cost_usd', 0):.4f}"
    )

    return result


# ─── Test Queries ─────────────────────────────────────────────────

def run_agentic_test_queries():
    """
    Runs test queries through the agentic pipeline.
    Tests all four complexity levels.
    """
    test_queries = [
        # SIMPLE — single-shot retrieval
        {
            "query": "What water damage claims have been filed? List the damaged objects.",
            "filter": {"damage_type": "water"},
            "expected_strategy": "single_shot or hyde",
        },
        # MODERATE — HyDE retrieval
        {
            "query": "Show me all storm damage claims and what was damaged.",
            "filter": {"damage_type": "storm"},
            "expected_strategy": "hyde",
        },
        # COMPLEX — multi-query + graph
        {
            "query": "Compare all damage claims and identify which types have the highest claimed amounts.",
            "filter": None,
            "expected_strategy": "agentic",
        },
        # STRUCTURED — deterministic lookup
        {
            "query": "What are the total amounts claimed in the invoices?",
            "filter": {"document_type": "invoice"},
            "expected_strategy": "single_shot",
        },
        # COMPLEX — multi-hop reasoning
        {
            "query": "Which claims have severe damage according to the photo documentation and what were the repair costs?",
            "filter": None,
            "expected_strategy": "agentic",
        },
    ]

    print("=" * 70)
    print("AGENTIC RAG PIPELINE — Test Queries")
    print("=" * 70)
    print(f"Reranking: {'ON' if RERANK_ENABLED else 'OFF'}")
    print(f"Hybrid search: {'ON' if HYBRID_SEARCH_ENABLED else 'OFF'}")
    print()

    for i, test in enumerate(test_queries):
        print(f"\n{'─'*70}")
        print(f"Query {i+1}: {test['query']}")
        if test["filter"]:
            print(f"Filter: {test['filter']}")
        print(f"Expected: {test['expected_strategy']}")
        print()

        result = agentic_query_pipeline(
            query=test["query"],
            metadata_filter=test["filter"],
            n_results=DEFAULT_N_RESULTS,
        )

        plan = result.get("query_plan", {})
        print(f"  Strategy: {plan.get('strategy')}")
        print(f"  Complexity: {plan.get('complexity')}")
        print(f"  Entities: {plan.get('entities')}")
        print(f"  Iterations: {result.get('retrieval_iterations')}")
        print(f"  Chunks: {result['chunks_retrieved']}")
        print(f"  Graph facts: {result.get('graph_facts_used', 0)}")
        print(f"  Latency: {result['latency_seconds']}s")
        print(f"  Cost: ${result['token_usage'].get('cost_usd', 0):.4f}")

        critique = result.get("self_critique", {})
        if critique:
            print(f"  Self-critique: {critique.get('quality', '?')}")

        print(f"  Answer: {result['answer'][:300]}...")
        print()

    print(f"\nDone. {len(test_queries)} queries logged to {QUERY_LOG}")


if __name__ == "__main__":
    run_agentic_test_queries()

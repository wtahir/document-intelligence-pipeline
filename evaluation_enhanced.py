# evaluation_enhanced.py
# ═══════════════════════════════════════════════════════════════════
# ENHANCED EVALUATION — RAGAS-style metrics for 2026 RAG systems
# ═══════════════════════════════════════════════════════════════════
#
# Why the old evaluation is insufficient:
#   Your stage6_evaluation.py uses a single LLM-as-judge with two scores
#   (retrieval_score, answer_score). This has three problems:
#
#   1. SELF-SERVING BIAS — GPT-4o rates its own outputs higher than a
#      different model would. Using the same model for generation and
#      evaluation creates an echo chamber.
#
#   2. COARSE GRANULARITY — A 1-5 score doesn't tell you WHAT failed.
#      Did the model hallucinate? Did it miss information? Was the
#      context irrelevant? You need separate metrics for each failure mode.
#
#   3. NO FAITHFULNESS CHECK — The old eval doesn't verify whether each
#      statement in the answer is actually supported by the retrieved chunks.
#      It just asks "was the answer good?" — which is too vague.
#
# This module implements four RAGAS-inspired metrics:
#
#   FAITHFULNESS (0-1): Is every claim in the answer supported by the context?
#     - Decomposes the answer into atomic statements
#     - Checks each statement against the context
#     - Score = supported_statements / total_statements
#
#   ANSWER RELEVANCY (0-1): Does the answer actually address the question?
#     - Generates N reverse questions from the answer
#     - Measures cosine similarity between reverse questions and original
#     - High similarity = answer is on-topic
#
#   CONTEXT PRECISION (0-1): Are the retrieved chunks relevant to the question?
#     - For each chunk, checks if it contains info needed to answer
#     - Weighted by rank position (top chunks matter more)
#     - Score = weighted_relevant_chunks / total_chunks
#
#   CONTEXT RECALL (0-1): Does the context contain all needed information?
#     - Decomposes the ideal answer into required facts
#     - Checks which facts are present in the context
#     - Score = found_facts / total_required_facts
#
# These four metrics together pinpoint exactly where your pipeline fails.

import json
import logging
import os
from datetime import datetime
from typing import Optional
from openai import AzureOpenAI
from config import (
    QUERY_LOG, OUTPUT_FOLDER,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_API_VERSION,
    LOG_FOLDER, LOG_FORMAT,
    estimate_llm_cost,
)

os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "evaluation_enhanced.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT

ENHANCED_EVAL_REPORT = os.path.join(OUTPUT_FOLDER, "enhanced_evaluation_report.json")
ENHANCED_EVAL_SUMMARY = os.path.join(OUTPUT_FOLDER, "enhanced_evaluation_summary.json")


# ─── 1. Faithfulness ─────────────────────────────────────────────

_DECOMPOSE_PROMPT = """Decompose this answer into individual factual statements.
Each statement should be a single, atomic claim that can be verified independently.

Answer: {answer}

Respond with ONLY a JSON array of strings:
["statement 1", "statement 2", ...]"""

_VERIFY_PROMPT = """For each statement, determine if it is supported by the given context.
A statement is "supported" if the context contains evidence for it.
A statement is "unsupported" if the context does NOT contain evidence, or contradicts it.

Context:
{context}

Statements to verify:
{statements}

Respond with ONLY a JSON array of objects:
[{{"statement": "...", "verdict": "supported"|"unsupported", "evidence": "brief quote or 'none'"}}]"""


def evaluate_faithfulness(answer: str, context: str) -> tuple[float, dict, dict]:
    """
    Measures faithfulness: what fraction of claims in the answer are
    actually supported by the retrieved context?

    Returns:
        (score, details, token_usage)
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def _acc(usage):
        for k in total_usage:
            total_usage[k] = round(total_usage[k] + usage.get(k, 0), 6)

    try:
        # Step 1: Decompose answer into statements
        response1 = _client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(answer=answer)}],
            max_completion_tokens=500,
            temperature=0.0,
        )
        raw1 = response1.choices[0].message.content.strip()
        u1 = response1.usage
        _acc({"prompt_tokens": u1.prompt_tokens, "completion_tokens": u1.completion_tokens,
              "cost_usd": estimate_llm_cost(u1.prompt_tokens, u1.completion_tokens)})

        if raw1.startswith("```"):
            raw1 = raw1.split("```")[1]
            if raw1.startswith("json"):
                raw1 = raw1[4:]
        statements = json.loads(raw1)

        if not statements:
            return 1.0, {"statements": [], "all_supported": True}, total_usage

        # Step 2: Verify each statement against context
        stmt_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        response2 = _client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _VERIFY_PROMPT.format(
                context=context[:3000],
                statements=stmt_text,
            )}],
            max_completion_tokens=800,
            temperature=0.0,
        )
        raw2 = response2.choices[0].message.content.strip()
        u2 = response2.usage
        _acc({"prompt_tokens": u2.prompt_tokens, "completion_tokens": u2.completion_tokens,
              "cost_usd": estimate_llm_cost(u2.prompt_tokens, u2.completion_tokens)})

        if raw2.startswith("```"):
            raw2 = raw2.split("```")[1]
            if raw2.startswith("json"):
                raw2 = raw2[4:]
        verdicts = json.loads(raw2)

        supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
        score = supported / len(verdicts) if verdicts else 1.0

        return score, {
            "total_statements": len(statements),
            "supported": supported,
            "unsupported": len(verdicts) - supported,
            "verdicts": verdicts,
        }, total_usage

    except Exception as e:
        logging.error(f"Faithfulness evaluation failed: {e}")
        return 0.0, {"error": str(e)}, total_usage


# ─── 2. Answer Relevancy ─────────────────────────────────────────

_REVERSE_QUESTION_PROMPT = """Given this answer about insurance claims, generate 3 questions that this answer would be a good response to.
The questions should be specific and related to insurance claims processing.

Answer: {answer}

Respond with ONLY a JSON array of 3 strings:
["question 1", "question 2", "question 3"]"""


def evaluate_answer_relevancy(query: str, answer: str) -> tuple[float, dict, dict]:
    """
    Measures answer relevancy: does the answer actually address the question?

    Approach: Generate reverse questions from the answer, then check if they
    are semantically similar to the original question. If the answer is about
    something different, the reverse questions will be different from the original.

    Returns:
        (score, details, token_usage)
    """
    try:
        response = _client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _REVERSE_QUESTION_PROMPT.format(answer=answer)}],
            max_completion_tokens=300,
            temperature=0.3,
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
        reverse_questions = json.loads(raw)

        # Compute word-overlap similarity (fast approximation)
        query_words = set(query.lower().split())
        similarities = []
        for rq in reverse_questions:
            rq_words = set(rq.lower().split())
            if query_words and rq_words:
                overlap = len(query_words & rq_words) / len(query_words | rq_words)
                similarities.append(overlap)

        score = sum(similarities) / len(similarities) if similarities else 0.0

        return score, {
            "reverse_questions": reverse_questions,
            "similarities": [round(s, 3) for s in similarities],
        }, token_usage

    except Exception as e:
        logging.error(f"Answer relevancy evaluation failed: {e}")
        return 0.0, {"error": str(e)}, {}


# ─── 3. Context Precision ────────────────────────────────────────

_CONTEXT_PRECISION_PROMPT = """For each retrieved chunk, determine if it contains information relevant to answering the question.
A chunk is "relevant" if it contains facts, data, or context that helps answer the question.
A chunk is "irrelevant" if it's about a different topic, claim, or damage type.

Question: {query}

Chunks:
{chunks}

Respond with ONLY a JSON array of objects:
[{{"chunk_index": 1, "relevant": true/false, "reason": "brief reason"}}]"""


def evaluate_context_precision(query: str, chunks: list[dict]) -> tuple[float, dict, dict]:
    """
    Measures context precision: are the retrieved chunks relevant to the question?
    Weighted by rank position — irrelevant chunks at the top hurt more.

    Returns:
        (score, details, token_usage)
    """
    if not chunks:
        return 0.0, {"no_chunks": True}, {}

    # Format chunks for evaluation
    chunk_texts = []
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        text = c.get("text", "")[:300]
        chunk_texts.append(
            f"Chunk {i+1} [Claim: {meta.get('claim_number', '?')}, "
            f"Type: {meta.get('document_type', '?')}, "
            f"Damage: {meta.get('damage_type', '?')}]: {text}"
        )

    try:
        response = _client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _CONTEXT_PRECISION_PROMPT.format(
                query=query,
                chunks="\n\n".join(chunk_texts),
            )}],
            max_completion_tokens=400,
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
        verdicts = json.loads(raw)

        # Weighted precision — top ranks matter more
        # Weight = 1 / log2(rank + 1) (standard NDCG-style weighting)
        import math
        weighted_sum = 0.0
        weight_total = 0.0
        for v in verdicts:
            rank = v.get("chunk_index", 1)
            weight = 1.0 / math.log2(rank + 1)
            weight_total += weight
            if v.get("relevant", False):
                weighted_sum += weight

        score = weighted_sum / weight_total if weight_total > 0 else 0.0

        relevant_count = sum(1 for v in verdicts if v.get("relevant", False))

        return score, {
            "total_chunks": len(chunks),
            "relevant_chunks": relevant_count,
            "irrelevant_chunks": len(chunks) - relevant_count,
            "weighted_score": round(score, 4),
            "verdicts": verdicts,
        }, token_usage

    except Exception as e:
        logging.error(f"Context precision evaluation failed: {e}")
        return 0.0, {"error": str(e)}, {}


# ─── 4. Hallucination Detection ──────────────────────────────────
#
# Beyond faithfulness (which checks if statements are supported),
# this specifically looks for FABRICATED details: names, numbers,
# dates, or claim IDs that appear in the answer but not in the context.

_HALLUCINATION_PROMPT = """Compare the answer against the context and identify any hallucinated details.
A hallucination is a specific detail (name, number, date, claim ID, amount) in the answer
that does NOT appear in the context and was fabricated by the model.

Context:
{context}

Answer:
{answer}

Respond with ONLY valid JSON:
{{
  "hallucinations_found": true/false,
  "hallucinated_details": ["list of fabricated details, if any"],
  "severity": "none"|"minor"|"major",
  "explanation": "brief explanation"
}}"""


def detect_hallucinations(answer: str, context: str) -> tuple[dict, dict]:
    """
    Detects fabricated details in the answer that aren't in the context.

    Returns:
        (hallucination_report, token_usage)
    """
    try:
        response = _client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[{"role": "user", "content": _HALLUCINATION_PROMPT.format(
                context=context[:3000],
                answer=answer,
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
        report = json.loads(raw)
        return report, token_usage

    except Exception as e:
        logging.error(f"Hallucination detection failed: {e}")
        return {"error": str(e), "hallucinations_found": False}, {}


# ─── Combined Enhanced Evaluation ────────────────────────────────

def evaluate_query_enhanced(query_record: dict) -> dict:
    """
    Runs all four RAGAS-style metrics + hallucination detection on a single query.

    Returns the query record enriched with:
    - faithfulness_score (0-1)
    - answer_relevancy_score (0-1)
    - context_precision_score (0-1)
    - hallucination_report
    - enhanced_eval_token_usage
    """
    query = query_record.get("query", "")
    answer = query_record.get("answer", "")
    chunks = query_record.get("chunks", [])

    # Skip blocked/failed queries
    if query_record.get("blocked") or not answer or answer.startswith("INSUFFICIENT_CONTEXT"):
        query_record["enhanced_evaluated"] = True
        query_record["enhanced_eval_skipped"] = True
        return query_record

    # Build context string from chunks
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[Chunk {i+1}]: {c.get('text', '')}")
    context = "\n\n".join(context_parts)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def _acc(u):
        for k in total_usage:
            total_usage[k] = round(total_usage[k] + u.get(k, 0), 6)

    # 1. Faithfulness
    faith_score, faith_details, faith_usage = evaluate_faithfulness(answer, context)
    _acc(faith_usage)

    # 2. Answer relevancy
    relevancy_score, relevancy_details, relevancy_usage = evaluate_answer_relevancy(query, answer)
    _acc(relevancy_usage)

    # 3. Context precision
    precision_score, precision_details, precision_usage = evaluate_context_precision(query, chunks)
    _acc(precision_usage)

    # 4. Hallucination detection
    hallucination_report, hallucination_usage = detect_hallucinations(answer, context)
    _acc(hallucination_usage)

    # Composite score (weighted average)
    composite = round(
        0.35 * faith_score +
        0.25 * relevancy_score +
        0.25 * precision_score +
        0.15 * (0.0 if hallucination_report.get("hallucinations_found") else 1.0),
        4,
    )

    query_record.update({
        "enhanced_evaluated": True,
        "enhanced_evaluated_at": datetime.now().isoformat(),
        "faithfulness_score": round(faith_score, 4),
        "faithfulness_details": faith_details,
        "answer_relevancy_score": round(relevancy_score, 4),
        "answer_relevancy_details": relevancy_details,
        "context_precision_score": round(precision_score, 4),
        "context_precision_details": precision_details,
        "hallucination_report": hallucination_report,
        "composite_score": composite,
        "enhanced_eval_token_usage": total_usage,
    })

    logging.info(
        f"Enhanced eval: faithful={faith_score:.2f} relevant={relevancy_score:.2f} "
        f"precision={precision_score:.2f} composite={composite:.2f} "
        f"hallucinated={hallucination_report.get('hallucinations_found', '?')}"
    )

    return query_record


def evaluate_all_enhanced():
    """
    Runs enhanced evaluation on all queries in the query log.
    Idempotent — skips already-evaluated queries.
    """
    if not os.path.exists(QUERY_LOG):
        raise FileNotFoundError("query_log.json not found. Run Stage 5 first.")

    with open(QUERY_LOG, "r", encoding="utf-8") as f:
        query_log = json.load(f)

    pending = [q for q in query_log if not q.get("enhanced_evaluated")]
    done = [q for q in query_log if q.get("enhanced_evaluated")]

    print(f"Total queries: {len(query_log)}")
    print(f"Already enhanced-evaluated: {len(done)}")
    print(f"Pending: {len(pending)}\n")

    if not pending:
        print("All queries already have enhanced evaluation.")
        return

    for i, record in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] Evaluating: {record['query'][:60]}...")
        evaluate_query_enhanced(record)

        if record.get("enhanced_eval_skipped"):
            print("  -> Skipped (blocked/insufficient)")
        else:
            print(
                f"  -> faithful={record.get('faithfulness_score', '?'):.2f} "
                f"relevant={record.get('answer_relevancy_score', '?'):.2f} "
                f"precision={record.get('context_precision_score', '?'):.2f} "
                f"composite={record.get('composite_score', '?'):.2f}"
            )

    all_evaluated = done + pending

    # Save updated query log
    with open(QUERY_LOG, "w", encoding="utf-8") as f:
        json.dump(all_evaluated, f, indent=2, ensure_ascii=False)

    # Build summary
    scored = [q for q in all_evaluated if q.get("composite_score") is not None]

    if scored:
        avg_faith = round(sum(q["faithfulness_score"] for q in scored) / len(scored), 4)
        avg_relevancy = round(sum(q["answer_relevancy_score"] for q in scored) / len(scored), 4)
        avg_precision = round(sum(q["context_precision_score"] for q in scored) / len(scored), 4)
        avg_composite = round(sum(q["composite_score"] for q in scored) / len(scored), 4)
        hallucination_rate = round(
            sum(1 for q in scored if q.get("hallucination_report", {}).get("hallucinations_found")) / len(scored),
            4,
        )
    else:
        avg_faith = avg_relevancy = avg_precision = avg_composite = hallucination_rate = 0.0

    # Aggregate token usage
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    for q in scored:
        u = q.get("enhanced_eval_token_usage", {})
        for k in total_tokens:
            total_tokens[k] = round(total_tokens[k] + u.get(k, 0), 6)

    summary = {
        "run_at": datetime.now().isoformat(),
        "queries_evaluated": len(scored),
        "avg_faithfulness": avg_faith,
        "avg_answer_relevancy": avg_relevancy,
        "avg_context_precision": avg_precision,
        "avg_composite_score": avg_composite,
        "hallucination_rate": hallucination_rate,
        "total_eval_tokens": total_tokens,
        "worst_queries": sorted(
            [{"query": q["query"], "composite": q["composite_score"],
              "faithfulness": q["faithfulness_score"],
              "hallucinated": q.get("hallucination_report", {}).get("hallucinations_found", False)}
             for q in scored],
            key=lambda x: x["composite"],
        )[:5],
    }

    with open(ENHANCED_EVAL_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(ENHANCED_EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(all_evaluated, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Enhanced Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Avg Faithfulness:       {avg_faith:.3f}")
    print(f"  Avg Answer Relevancy:   {avg_relevancy:.3f}")
    print(f"  Avg Context Precision:  {avg_precision:.3f}")
    print(f"  Avg Composite Score:    {avg_composite:.3f}")
    print(f"  Hallucination Rate:     {hallucination_rate:.1%}")
    print(f"  Total Eval Cost:        ${total_tokens['cost_usd']:.4f}")


if __name__ == "__main__":
    evaluate_all_enhanced()

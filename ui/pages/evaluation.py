"""
Evaluation Dashboard — Visual analytics for RAG pipeline quality.
Shows retrieval vs answer scores, failure analysis, distance distributions,
and actionable improvement suggestions.
"""

import streamlit as st
import pandas as pd
import json
import os

from ui.components.theme import inject_css
from ui.components.runtime import is_demo_mode
from ui.components.widgets import (
    render_hero,
    render_kpi_row,
    render_score_gauge,
    load_json,
)


def render():
    inject_css()
    demo_mode = is_demo_mode()

    render_hero(
        title="Evaluation Dashboard",
        subtitle="Analyse RAG pipeline quality for water, storm, and glass damage claim queries. "
                 "Review retrieval and answer scores, failure breakdowns, distance distributions, "
                 "and actionable improvement suggestions from GPT-4o-as-judge.",
        badge="Quality Analytics",
    )

    # ─── Load data ─────────────────────────────────────────
    evaluation = load_json("evaluation_summary.json")
    query_log = load_json("query_log.json")
    eval_report = load_json("evaluation_report.json")

    if not evaluation and not query_log:
        st.warning("⚠️ No evaluation data available. Run Stage 5 (Queries) and Stage 6 (Evaluation) first.")

        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Run Stage 5** — Execute test queries to build the query log
        2. **Run Stage 6** — Evaluate all queries with GPT-4o-as-judge
        3. **Return here** — View the evaluation analytics
        """)
        return

    # ─── Top-level scores ─────────────────────────────────
    st.markdown("### Overall Scores")

    if evaluation:
        score_cols = st.columns(4)

        with score_cols[0]:
            avg_ret = evaluation.get("avg_retrieval_score", 0)
            color = _score_color(avg_ret)
            render_score_gauge(avg_ret, 5, "Retrieval Quality", color)

        with score_cols[1]:
            avg_ans = evaluation.get("avg_answer_score", 0)
            color = _score_color(avg_ans)
            render_score_gauge(avg_ans, 5, "Answer Quality", color)

        with score_cols[2]:
            total_q = evaluation.get("total_queries_evaluated", 0)
            st.markdown(f"""
            <div class="score-display">
                <div class="score-value" style="color:#06B6D4;">{total_q}</div>
                <div class="score-label">Queries Evaluated</div>
            </div>
            """, unsafe_allow_html=True)

        with score_cols[3]:
            attention = evaluation.get("queries_needing_attention", [])
            attn_count = len(attention)
            color = "#10B981" if attn_count == 0 else "#F59E0B" if attn_count < 3 else "#EF4444"
            st.markdown(f"""
            <div class="score-display">
                <div class="score-value" style="color:{color};">{attn_count}</div>
                <div class="score-label">Need Attention</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Failure Analysis ─────────────────────────────────
    if evaluation:
        st.markdown("### Failure Analysis")

        fail_cols = st.columns(2)

        with fail_cols[0]:
            st.markdown("**Failure Type Breakdown**")
            failures = evaluation.get("failure_type_breakdown", {})

            if failures:
                import plotly.express as px

                df_fail = pd.DataFrame([
                    {"Type": k.title(), "Count": v}
                    for k, v in failures.items()
                ])

                colors_map = {
                    "None": "#10B981",
                    "Retrieval": "#F59E0B",
                    "Generation": "#06B6D4",
                    "Both": "#EF4444",
                }

                fig = px.pie(
                    df_fail, names="Type", values="Count",
                    color="Type",
                    color_discrete_map=colors_map,
                    hole=0.45,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC", height=320,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, width='stretch')

                # Failure interpretation
                for ftype, count in failures.items():
                    interpretation = _failure_interpretation(ftype)
                    st.markdown(f"""
                    <div style="background:#1E293B; border:1px solid #334155; border-radius:8px;
                                padding:12px; margin-bottom:6px;">
                        <span style="font-weight:600; color:#F8FAFC;">{ftype.title()}</span>
                        <span style="color:#94A3B8;"> ({count} queries) — </span>
                        <span style="color:#94A3B8; font-size:0.85rem;">{interpretation}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No failure data available.")

        with fail_cols[1]:
            st.markdown("**Score Distribution**")
            if query_log:
                scored = [q for q in query_log if q.get("retrieval_score") is not None]
                if scored:
                    import plotly.graph_objects as go

                    ret_scores = [q.get("retrieval_score", 0) for q in scored]
                    ans_scores = [q.get("answer_score", 0) for q in scored]

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=ret_scores, name="Retrieval",
                        marker_color="#6366F1", opacity=0.7,
                        xbins=dict(start=0.5, end=5.5, size=1),
                    ))
                    fig.add_trace(go.Histogram(
                        x=ans_scores, name="Answer",
                        marker_color="#06B6D4", opacity=0.7,
                        xbins=dict(start=0.5, end=5.5, size=1),
                    ))
                    fig.update_layout(
                        barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC", height=320,
                        margin=dict(l=40, r=20, t=20, b=40),
                        xaxis=dict(title="Score (1-5)", gridcolor="#334155", dtick=1),
                        yaxis=dict(title="Count", gridcolor="#334155"),
                        legend=dict(font=dict(color="#94A3B8")),
                    )
                    st.plotly_chart(fig, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Distance Analysis ────────────────────────────────
    if query_log:
        st.markdown("### Retrieval Distance Analysis")

        all_distances = []
        for q in query_log:
            for chunk in q.get("chunks", []):
                all_distances.append({
                    "distance": chunk.get("distance", 1.0),
                    "query": q.get("query", "")[:50],
                    "file": chunk.get("metadata", {}).get("file_name", "—"),
                })

        if all_distances:
            import plotly.express as px

            dist_cols = st.columns(2)

            with dist_cols[0]:
                st.markdown("**Distance Distribution**")
                df_dist = pd.DataFrame(all_distances)
                fig = px.histogram(
                    df_dist, x="distance", nbins=20,
                    color_discrete_sequence=["#6366F1"],
                )
                fig.add_vline(x=0.6, line_dash="dash", line_color="#EF4444",
                              annotation_text="Threshold (0.6)", annotation_font_color="#EF4444")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC", height=300,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(title="Cosine Distance", gridcolor="#334155"),
                    yaxis=dict(title="Count", gridcolor="#334155"),
                )
                st.plotly_chart(fig, width='stretch')

            with dist_cols[1]:
                st.markdown("**Distance by Query**")
                query_distances = {}
                for q in query_log:
                    qtext = q.get("query", "")[:40]
                    dists = [c.get("distance", 1.0) for c in q.get("chunks", [])]
                    if dists:
                        query_distances[qtext] = sum(dists) / len(dists)

                if query_distances:
                    df_qd = pd.DataFrame([
                        {"Query": k, "Avg Distance": v}
                        for k, v in query_distances.items()
                    ]).sort_values("Avg Distance")

                    fig = px.bar(
                        df_qd, x="Avg Distance", y="Query", orientation="h",
                        color="Avg Distance",
                        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                        range_color=[0, 1],
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC", height=300,
                        margin=dict(l=20, r=20, t=10, b=40),
                        xaxis=dict(gridcolor="#334155"),
                        yaxis=dict(gridcolor="#334155"),
                        coloraxis_colorbar=dict(title="Distance"),
                    )
                    st.plotly_chart(fig, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Detailed Query Evaluations ───────────────────────
    if query_log:
        st.markdown("### Detailed Query Evaluations")

        scored_queries = [q for q in query_log if q.get("evaluated")]
        if scored_queries:
            # Summary table
            table_data = []
            for q in scored_queries:
                ret_score = q.get("retrieval_score")
                ans_score = q.get("answer_score")
                table_data.append({
                    "Query": q.get("query", "")[:60] + "..." if len(q.get("query", "")) > 60 else q.get("query", ""),
                    "Retrieval": str(ret_score) if ret_score is not None else "—",
                    "Answer": str(ans_score) if ans_score is not None else "—",
                    "Failure": q.get("failure_type", "—"),
                    "Avg Dist": f"{q.get('avg_distance', 0):.3f}" if q.get("avg_distance") else "—",
                    "Chunks": q.get("chunks_retrieved", 0),
                })

            df = pd.DataFrame(table_data)

            def color_score(val):
                try:
                    v = int(val)
                    if v >= 4:
                        return "color: #10B981; font-weight: 600"
                    elif v >= 3:
                        return "color: #F59E0B; font-weight: 600"
                    else:
                        return "color: #EF4444; font-weight: 600"
                except (ValueError, TypeError):
                    return ""

            def color_failure(val):
                colors = {"none": "#10B981", "retrieval": "#F59E0B", "generation": "#06B6D4", "both": "#EF4444"}
                return f"color: {colors.get(str(val).lower(), '#94A3B8')}; font-weight: 600"

            styled = df.style.map(color_score, subset=["Retrieval", "Answer"])
            styled = styled.map(color_failure, subset=["Failure"])

            st.dataframe(styled, width='stretch', hide_index=True, height=300)

            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed cards
            st.markdown("### Improvement Suggestions")
            attention = [q for q in scored_queries if (q.get("answer_score") or 5) < 4]

            if attention:
                for q in attention:
                    ret_color = _score_color(q.get("retrieval_score", 0))
                    ans_color = _score_color(q.get("answer_score", 0))

                    st.markdown(f"""
                    <div style="background:#1E293B; border:1px solid #334155; border-radius:12px;
                                padding:20px; margin-bottom:12px;">
                        <div style="color:#F8FAFC; font-weight:600; margin-bottom:10px;">
                            ❓ {q.get('query', '')}
                        </div>
                        <div style="display:flex; gap:20px; margin-bottom:12px; flex-wrap:wrap;">
                            <div>
                                <span style="color:#94A3B8; font-size:0.8rem;">Retrieval: </span>
                                <span style="color:{ret_color}; font-weight:600;">{q.get('retrieval_score', '—')}/5</span>
                            </div>
                            <div>
                                <span style="color:#94A3B8; font-size:0.8rem;">Answer: </span>
                                <span style="color:{ans_color}; font-weight:600;">{q.get('answer_score', '—')}/5</span>
                            </div>
                            <div>
                                <span style="color:#94A3B8; font-size:0.8rem;">Avg Distance: </span>
                                <span style="color:#F8FAFC;">{q.get('avg_distance', '—')}</span>
                            </div>
                        </div>
                        <div style="background:#0F172A; border-radius:8px; padding:14px; margin-bottom:8px;">
                            <div style="color:#94A3B8; font-size:0.75rem; margin-bottom:4px;">
                                📝 Retrieval Notes</div>
                            <div style="color:#CBD5E1; font-size:0.85rem;">
                                {q.get('retrieval_notes', '—')}</div>
                        </div>
                        <div style="background:#0F172A; border-radius:8px; padding:14px; margin-bottom:8px;">
                            <div style="color:#94A3B8; font-size:0.75rem; margin-bottom:4px;">
                                📝 Answer Notes</div>
                            <div style="color:#CBD5E1; font-size:0.85rem;">
                                {q.get('answer_notes', '—')}</div>
                        </div>
                        <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3);
                                    border-radius:8px; padding:14px;">
                            <div style="color:#818CF8; font-size:0.75rem; margin-bottom:4px;">
                                💡 Improvement Suggestion</div>
                            <div style="color:#F8FAFC; font-size:0.9rem; font-weight:500;">
                                {q.get('improvement_suggestion', '—')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 All evaluated queries scored 4/5 or above. Nice work!")
        else:
            st.info("No evaluated queries found. Run Stage 6 (Evaluation) first.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Run Evaluation ───────────────────────────────────
    st.markdown("### Run Evaluation")
    eval_cols = st.columns([2, 1])

    with eval_cols[0]:
        if demo_mode:
            st.markdown("""
            Demo mode is active — evaluation execution is disabled on hosted deployments.
            The charts above show precomputed evaluation results for water/storm/glass damage queries.
            """)
        else:
            st.markdown("""
            Run GPT-4o-as-judge to evaluate all pending queries from the query log.
            This scores both retrieval quality and answer quality on a 1–5 scale.

            **Cost:** ~0.01–0.03 USD per query (GPT-4o input + output tokens)
            """)

    with eval_cols[1]:
        if st.button("Run Evaluation", type="primary", disabled=demo_mode):
            with st.spinner("🔍 Evaluating queries with GPT-4o..."):
                try:
                    from stage6_evaluation import evaluate_all
                    import io
                    from contextlib import redirect_stdout
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        evaluate_all()
                    output = buf.getvalue()
                    st.success("✅ Evaluation complete!")
                    with st.expander("📋 Output"):
                        st.code(output, language="text")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")


# ─── Helpers ──────────────────────────────────────────────

def _score_color(score) -> str:
    """Return a colour based on score value."""
    try:
        s = float(score)
        if s >= 4:
            return "#10B981"
        elif s >= 3:
            return "#F59E0B"
        else:
            return "#EF4444"
    except (ValueError, TypeError):
        return "#94A3B8"


def _failure_interpretation(ftype: str) -> str:
    """Return a human-readable interpretation of a failure type."""
    interpretations = {
        "none": "Pipeline working well — both retrieval and generation are producing quality results.",
        "retrieval": "Wrong or irrelevant chunks retrieved. Consider: better embeddings, more overlap, reranking.",
        "generation": "Good chunks found but answer quality is poor. Consider: prompt tuning, more context.",
        "both": "Both retrieval and generation failing. Likely a fundamental data or pipeline issue.",
    }
    return interpretations.get(ftype.lower(), "Unknown failure type.")

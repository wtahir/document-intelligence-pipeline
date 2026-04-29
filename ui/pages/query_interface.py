"""
Query Interface page — Interactive RAG Q&A with the insurance documents.
Supports natural language queries, metadata filters, and shows retrieved
chunks with relevance scores.
"""

import streamlit as st
import sys
import os
import json
import html
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        title="Query Interface",
        subtitle="Ask questions about water, storm, and glass damage claims. "
                 "The RAG pipeline retrieves relevant chunks and generates grounded answers. "
                 "For policy coverage and payouts, see the deterministic Payout Report.",
        badge="AI Assistant",
    )

    query_log = load_json("query_log.json") or []

    # ─── Query form / Demo selector ──────────────────────
    st.markdown("### Ask a Question")

    if demo_mode:
        st.info(
            "Demo mode is active — live query execution is disabled. "
            "Select a precomputed query below to explore retrieved chunks and answers "
            "for water, storm, and glass damage claims."
        )
        if query_log:
            recent = list(reversed(query_log))
            selected = st.selectbox(
                "Sample query",
                options=range(len(recent)),
                format_func=lambda i: f"{i+1}. {recent[i].get('query', 'No query')[:100]}",
            )
            _render_query_result(recent[selected], 0)
        else:
            st.warning("No sample query log found. Add `data/output/query_log.json` to demo this page.")
    else:
        query_col, options_col = st.columns([3, 1])

        with query_col:
            user_query = st.text_area(
                "Your question",
                placeholder="e.g., What objects were damaged in water damage claims?\n"
                            "Show me all storm damage invoices and their total amounts.\n"
                            "Which glass damage claims have severe damage assessments?",
                height=120,
                label_visibility="collapsed",
            )

        with options_col:
            st.markdown("**Options**")
            n_results = st.slider("Chunks to retrieve", 1, 15, 5)

            use_filter = st.toggle("Use metadata filter", value=False)

            urgency_filter = None
            claim_filter = None
            if use_filter:
                urgency_filter = st.selectbox("Urgency filter", ["Any", "high", "normal", "low"])
                claim_filter = st.text_input("Claim number filter", placeholder="e.g., 2025 1033831")

        # ─── Submit ───────────────────────────────────────────
        if st.button("Search & Generate Answer", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a question.")
                return

            # Build metadata filter
            metadata_filter = None
            if use_filter:
                filters = []
                if urgency_filter and urgency_filter != "Any":
                    filters.append({"urgency": urgency_filter})
                if claim_filter and claim_filter.strip():
                    filters.append({"claim_number": claim_filter.strip()})

                if len(filters) == 1:
                    metadata_filter = filters[0]
                elif len(filters) > 1:
                    metadata_filter = {"$and": filters}

            # Execute query
            with st.spinner("🔍 Retrieving relevant chunks and generating answer..."):
                try:
                    from stage5_retrieval import query_pipeline
                    result = query_pipeline(
                        query=user_query,
                        metadata_filter=metadata_filter,
                        n_results=n_results,
                    )

                    # Store result in session state for display
                    if "query_results" not in st.session_state:
                        st.session_state.query_results = []
                    st.session_state.query_results.insert(0, result)

                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
                    st.info("Make sure Stage 4 (Embedding) has been run and ChromaDB has vectors stored.")
                    return

    # ─── Display results ──────────────────────────────────
    if "query_results" in st.session_state and st.session_state.query_results:
        for idx, result in enumerate(st.session_state.query_results[:5]):  # Show last 5
            _render_query_result(result, idx)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Query History ────────────────────────────────────
    st.markdown("### Query History")

    if query_log:
        st.markdown(f"**{len(query_log)} queries logged**")

        for i, entry in enumerate(reversed(query_log[-10:])):  # Last 10
            with st.expander(
                f"🔹 {entry.get('query', 'No query')[:80]}  •  "
                f"{entry.get('chunks_retrieved', 0)} chunks  •  "
                f"{entry.get('queried_at', '')[:10]}",
                expanded=False,
            ):
                ans_col, meta_col = st.columns([3, 1])
                with ans_col:
                    st.markdown("**Answer:**")
                    st.markdown(entry.get("answer", "No answer"))
                with meta_col:
                    st.markdown(f"**Query ID:** `{entry.get('query_id', '—')}`")
                    st.markdown(f"**Chunks:** {entry.get('chunks_retrieved', 0)}")
                    if entry.get("evaluated"):
                        st.markdown(f"**Retrieval Score:** {entry.get('retrieval_score', '—')}/5")
                        st.markdown(f"**Answer Score:** {entry.get('answer_score', '—')}/5")
                        ft = entry.get("failure_type", "—")
                        color = "#10B981" if ft == "none" else "#EF4444"
                        st.markdown(f"**Failure Type:** <span style='color:{color}'>{ft}</span>",
                                    unsafe_allow_html=True)
    else:
        st.info("No query history yet. Ask a question above to get started!")


def _render_query_result(result: dict, idx: int):
    """Render a single query result with answer and chunks."""
    st.markdown("<br>", unsafe_allow_html=True)

    is_latest = idx == 0
    border_color = "#6366F1" if is_latest else "#334155"

    st.markdown(f"""
    <div style="background:#1E293B; border:1px solid {border_color}; border-radius:12px;
                padding:24px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <div style="color:#94A3B8; font-size:0.8rem;">
                {"🟢 Latest Query" if is_latest else "🔹 Previous Query"}  •
                {result.get('queried_at', '')[:19]}  •
                {result.get('chunks_retrieved', 0)} chunks retrieved
            </div>
            <div style="color:#475569; font-size:0.75rem;">
                {result.get('query_id', '')}
            </div>
        </div>
        <div style="color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:12px;">
            ❓ {result.get('query', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Answer box
    answer = result.get("answer", "No answer generated.")
    reranking_label = "Reranked" if result.get("reranking_enabled") else "Bi-encoder only"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #1E293B, #0F172A);
                border:1px solid #334155; border-left:4px solid #6366F1;
                border-radius:8px; padding:20px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase;
                        letter-spacing:0.05em;">AI Generated Answer</div>
            <div style="color:#818CF8; font-size:0.7rem; font-weight:600;
                        background:rgba(99,102,241,0.15); padding:2px 8px; border-radius:10px;">
                {reranking_label}</div>
        </div>
        <div style="color:#F8FAFC; line-height:1.7; font-size:0.95rem;">
            {answer}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Retrieved chunks
    chunks = result.get("chunks", [])
    if chunks:
        with st.expander(f"📚 View {len(chunks)} Retrieved Chunks", expanded=is_latest):
            for ci, chunk in enumerate(chunks):
                distance = chunk.get("distance", 1.0)
                meta = chunk.get("metadata", {})

                # Color-code by distance quality
                if distance < 0.3:
                    quality_color = "#10B981"
                    quality_label = "Excellent match"
                elif distance < 0.5:
                    quality_color = "#06B6D4"
                    quality_label = "Good match"
                elif distance < 0.7:
                    quality_color = "#F59E0B"
                    quality_label = "Fair match"
                else:
                    quality_color = "#EF4444"
                    quality_label = "Weak match"

                rerank_score = chunk.get("rerank_score")
                rerank_html = (
                    f'<span style="color:#818CF8; margin-left:8px; font-size:0.8rem;">'
                    f'Rerank: {rerank_score:.3f}</span>'
                    if rerank_score is not None else ""
                )

                chunk_text = chunk.get('text', '')
                chunk_text_display = html.escape(chunk_text[:500]) + ('...' if len(chunk_text) > 500 else '')
                st.html(f"""
                <div style="background:#0F172A; border:1px solid #334155; border-radius:8px;
                            padding:16px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:10px;">
                        <div>
                            <span style="color:#F8FAFC; font-weight:600;">Chunk {ci+1}</span>
                            <span style="color:#94A3B8; margin-left:8px; font-size:0.8rem;">
                                {html.escape(meta.get('file_name', '—'))}
                            </span>
                        </div>
                        <div>
                            <span style="color:{quality_color}; font-size:0.8rem; font-weight:600;">
                                &#9679; {html.escape(quality_label)}
                            </span>
                            <span style="color:#94A3B8; margin-left:8px; font-size:0.8rem;">
                                Distance: {distance:.4f}
                            </span>
                            {rerank_html}
                        </div>
                    </div>
                    <div style="display:flex; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
                        <span style="color:#94A3B8; font-size:0.75rem;">
                            🏷️ {html.escape(meta.get('document_type', '—'))}</span>
                        <span style="color:#94A3B8; font-size:0.75rem;">
                            📋 Claim: {html.escape(meta.get('claim_number', '—'))}</span>
                        <span style="color:#94A3B8; font-size:0.75rem;">
                            � Damage: {html.escape(meta.get('damage_type', '—'))}</span>
                        <span style="color:#94A3B8; font-size:0.75rem;">
                            🔧 Object: {html.escape(meta.get('damaged_object', '—'))}</span>
                    </div>
                    <div style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;
                                background:#1E293B; border-radius:6px; padding:12px; white-space:pre-wrap;">
                        {chunk_text_display}
                    </div>
                </div>
                """)

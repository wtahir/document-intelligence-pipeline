"""
Overview page — Upload documents and view pipeline status.
The upload area is the primary entry point. After uploading,
the pipeline runs automatically through Stages 1-4.
"""

import streamlit as st
import os
from ui.components.theme import inject_css
from ui.components.runtime import is_demo_mode
from ui.components.widgets import (
    render_hero,
    render_pipeline_flow,
    render_tech_stack,
    render_kpi_row,
    render_stage_card,
    render_score_gauge,
    load_json,
    get_pdf_count,
    get_file_mod_time,
    auto_run_pipeline,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")


def _get_upload_fingerprint(files):
    """Create a fingerprint from uploaded file names + sizes to detect new uploads."""
    return frozenset((f.name, f.size) for f in files)


def render():
    inject_css()
    demo_mode = is_demo_mode()

    # ─── Hero ─────────────────────────────────────────────
    render_hero(
        title="Insurance Claim Intelligence",
        subtitle="Upload water, storm, or glass damage claim PDFs. The pipeline automatically "
                 "ingests, classifies damage type, extracts amounts, checks policy coverage, "
                 "and calculates payout decisions.",
        badge="AI-Powered Pipeline",
    )

    # Initialise session state
    if "pipeline_completed" not in st.session_state:
        st.session_state.pipeline_completed = False
    if "last_upload_fingerprint" not in st.session_state:
        st.session_state.last_upload_fingerprint = None

    # ─── Upload Section ──────────────────────────────────
    st.markdown("### Upload Documents")

    if demo_mode:
        st.info(
            "Demo mode is active — upload and pipeline execution are disabled for stability. "
            "Explore the preloaded water/storm/glass damage claims via **Document Explorer**, "
            "**Query Interface**, and **Evaluation**. Policy coverage checks and payout "
            "decisions are computed deterministically (not via RAG) from policy metadata."
        )
    else:
        uploaded_files = st.file_uploader(
            "Drag and drop PDF files here, or click to browse",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

        if uploaded_files:
            fingerprint = _get_upload_fingerprint(uploaded_files)

            # Only run pipeline if this is a NEW upload (not a re-render)
            if fingerprint != st.session_state.last_upload_fingerprint:
                # Save files to data/pdfs/
                os.makedirs(PDF_DIR, exist_ok=True)
                saved_names = []
                for uf in uploaded_files:
                    dest = os.path.join(PDF_DIR, uf.name)
                    with open(dest, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_names.append(uf.name)

                st.success(f"{len(saved_names)} file(s) saved: {', '.join(saved_names)}")

                # Auto-run pipeline (Stages 1-4)
                st.markdown("---")
                st.markdown("**Running pipeline automatically...**")

                success = auto_run_pipeline()
                st.session_state.last_upload_fingerprint = fingerprint
                st.session_state.pipeline_completed = success

                if success:
                    st.info(
                        "Pipeline complete. Use the sidebar to navigate to "
                        "**Query Interface** or **Document Explorer**."
                    )
            else:
                # Already processed these files — just show status
                if st.session_state.pipeline_completed:
                    st.info("These documents have already been processed. Use the sidebar to navigate.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Pipeline status (only if data exists) ───────────
    ingestion = load_json("ingestion_summary.json")
    extraction = load_json("extraction_summary.json")
    chunking = load_json("chunking_summary.json")
    embedding = load_json("embedding_summary.json")
    evaluation = load_json("evaluation_summary.json")

    pdf_count = get_pdf_count()

    has_data = any([ingestion, extraction, chunking, embedding, evaluation])

    if has_data:
        st.markdown("### Pipeline Status")

        ingested = ingestion.get("successful", 0) if ingestion else 0
        extracted = extraction.get("successful", 0) if extraction else 0
        chunks = chunking.get("total_chunks_produced", 0) if chunking else 0
        vectors = embedding.get("total_vectors_in_collection", 0) if embedding else 0

        render_kpi_row([
            {"label": "Source PDFs", "value": pdf_count, "icon": ""},
            {"label": "Ingested", "value": ingested, "icon": "",
             "delta": f"{ingestion.get('failed', 0)} failed" if ingestion and ingestion.get("failed") else None},
            {"label": "Extracted", "value": extracted, "icon": "",
             "delta": f"{extraction.get('failed', 0)} failed" if extraction and extraction.get("failed") else None},
            {"label": "Chunks Created", "value": chunks, "icon": ""},
            {"label": "Vectors Stored", "value": vectors, "icon": ""},
        ])

        st.markdown("<br>", unsafe_allow_html=True)

    # ─── Pipeline Architecture ───────────────────────────
    st.markdown("### Pipeline Architecture")
    render_pipeline_flow()

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Stage Details ───────────────────────────────────
    st.markdown("### Pipeline Stages")
    cols = st.columns(6)

    stages = [
        (1, "Ingestion", "PDF parsing with pdfplumber, text extraction, metadata capture", ""),
        (2, "Extraction", "GPT-4o classification & structured field extraction with Pydantic validation", ""),
        (3, "Chunking", "Sentence-aware splitting with overlap, word-boundary respect", ""),
        (4, "Embedding", "Multilingual sentence-transformers → ChromaDB vector store", ""),
        (5, "Retrieval", "Semantic search + GPT-4o RAG answers + deterministic policy check & payout", ""),
        (6, "Evaluation", "GPT-4o-as-judge scoring retrieval & answer quality", ""),
    ]

    for col, (num, title, desc, icon) in zip(cols, stages):
        with col:
            render_stage_card(num, title, desc, icon)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Evaluation + Doc Types (only when data exists) ──
    if has_data:
        left, right = st.columns(2)

        with left:
            st.markdown("### Evaluation Scores")
            if evaluation:
                score_cols = st.columns(2)
                with score_cols[0]:
                    avg_ret = evaluation.get("avg_retrieval_score", 0)
                    color = "#10B981" if avg_ret >= 4 else "#F59E0B" if avg_ret >= 3 else "#EF4444"
                    render_score_gauge(avg_ret, 5, "Avg Retrieval Score", color)
                with score_cols[1]:
                    avg_ans = evaluation.get("avg_answer_score", 0)
                    color = "#10B981" if avg_ans >= 4 else "#F59E0B" if avg_ans >= 3 else "#EF4444"
                    render_score_gauge(avg_ans, 5, "Avg Answer Score", color)
            else:
                st.info("No evaluation data yet. Use the Evaluation page to run scoring.")

        with right:
            st.markdown("### Damage Types")
            if extraction:
                damage_types = extraction.get("damage_types_found", {})
                if damage_types:
                    import plotly.express as px
                    import pandas as pd

                    df = pd.DataFrame([
                        {"Type": k.title(), "Count": v}
                        for k, v in damage_types.items()
                    ])
                    fig = px.pie(
                        df, names="Type", values="Count",
                        color="Type",
                        color_discrete_map={"Water": "#06B6D4", "Storm": "#F59E0B", "Glass": "#818CF8"},
                        hole=0.4,
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC",
                        legend=dict(font=dict(color="#94A3B8")),
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=300,
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No damage types detected yet.")
            else:
                st.info("No extraction data yet.")

        st.markdown("<br>", unsafe_allow_html=True)

    # ─── Technology Stack ────────────────────────────────
    st.markdown("### Technology Stack")
    render_tech_stack()

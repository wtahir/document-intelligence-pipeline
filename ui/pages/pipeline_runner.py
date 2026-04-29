"""
Pipeline Runner page — Execute individual stages or the full pipeline
with real-time progress indicators and live log display.
"""

import streamlit as st
import sys
import os
import json
import time
import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.components.theme import inject_css
from ui.components.runtime import is_demo_mode
from ui.components.widgets import (
    render_hero,
    render_kpi_row,
    render_status_badge,
    load_json,
    get_pdf_count,
    get_file_mod_time,
    OUTPUT_DIR,
    BASE_DIR,
)


def render():
    inject_css()
    demo_mode = is_demo_mode()

    render_hero(
        title="Pipeline Runner",
        subtitle="Execute individual processing stages for water, storm, and glass damage claims. "
                 "Stages 1-4 process documents; Stage 5 runs queries and generates the payout report; "
                 "Stage 6 evaluates answer quality.",
        badge="Operations",
    )

    if demo_mode:
        st.warning(
            "Demo mode is active — pipeline execution is disabled on hosted deployments. "
            "Explore the preloaded water/storm/glass damage dataset via Document Explorer, Query Interface, and Evaluation."
        )

    # ─── Current pipeline status ──────────────────────────
    st.markdown("### Current Pipeline Status")

    stages_info = [
        {
            "name": "Ingestion",
            "file": "ingestion_summary.json",
            "icon": "",
            "check_key": "successful",
        },
        {
            "name": "Extraction",
            "file": "extraction_summary.json",
            "icon": "",
            "check_key": "successful",
        },
        {
            "name": "Chunking",
            "file": "chunking_summary.json",
            "icon": "",
            "check_key": "total_chunks_produced",
        },
        {
            "name": "Embedding",
            "file": "embedding_summary.json",
            "icon": "",
            "check_key": "chunks_stored",
        },
        {
            "name": "Retrieval",
            "file": "query_log.json",
            "icon": "",
            "check_key": None,
        },
        {
            "name": "Evaluation",
            "file": "evaluation_summary.json",
            "icon": "",
            "check_key": "total_queries_evaluated",
        },
    ]

    cols = st.columns(6)
    for col, info in zip(cols, stages_info):
        with col:
            data = load_json(info["file"])
            mod_time = get_file_mod_time(info["file"])

            if data is None:
                status = "Not Run"
                status_color = "#475569"
                count = "—"
            elif info["check_key"] is None:
                # query_log is a list
                status = "Complete"
                status_color = "#10B981"
                count = f"{len(data)} queries" if isinstance(data, list) else "—"
            else:
                val = data.get(info["check_key"], 0)
                failed = data.get("failed", data.get("chunks_failed", 0))
                if failed and failed > 0:
                    status = "Partial"
                    status_color = "#F59E0B"
                else:
                    status = "Complete"
                    status_color = "#10B981"
                count = str(val)

            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:12px;
                        padding:18px; text-align:center;">
                <div style="font-size:1.2rem; margin-bottom:6px; font-weight:600; color:#6366F1;">{info['name'][:3].upper()}</div>
                <div style="color:#F8FAFC; font-weight:600; font-size:0.9rem;">{info['name']}</div>
                <div style="color:{status_color}; font-size:0.8rem; font-weight:600;
                            margin:6px 0;">{status}</div>
                <div style="color:#94A3B8; font-size:1.3rem; font-weight:700;">{count}</div>
                <div style="color:#475569; font-size:0.7rem; margin-top:6px;">{mod_time}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Stage Execution ──────────────────────────────────
    st.markdown("### Run Pipeline Stages")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Full Pipeline",
        "Stage 1: Ingestion",
        "Stage 2: Extraction",
        "Stage 3–4: Chunk & Embed",
    ])

    with tab1:
        st.markdown("""
        Run the complete pipeline from ingestion through embedding.
        This will process all PDFs in the `data/pdfs/` folder sequentially through all stages.

        **Note:** Stage 2 (Extraction) calls Azure OpenAI and may take several minutes depending
        on the number of documents.
        """)

        pdf_count = get_pdf_count()
        st.info(f"**{pdf_count}** PDF files found in `data/pdfs/`")

        if st.button("Run Full Pipeline", key="run_full", type="primary", disabled=demo_mode):
            _run_full_pipeline()

    with tab2:
        st.markdown("""
        **Stage 1: Ingestion** reads all PDFs from `data/pdfs/`, extracts text using pdfplumber,
        and saves structured records to `ingested_data.json`.

        This stage runs locally — no API calls needed.
        """)

        if st.button("Run Ingestion", key="run_ingest", disabled=demo_mode):
            _run_stage("ingestion", _execute_ingestion)

    with tab3:
        st.markdown("""
        **Stage 2: Extraction** sends each document to Azure OpenAI GPT-4o for classification
        and structured field extraction. Results are validated with Pydantic models.

        ⚠️ This stage makes API calls and will incur costs. Rate limited to avoid throttling.
        """)

        ingestion_data = load_json("ingestion_summary.json")
        if ingestion_data:
            st.success(f"✅ {ingestion_data.get('successful', 0)} documents available from ingestion")
        else:
            st.warning("⚠️ Run Stage 1 (Ingestion) first")

        if st.button("Run Extraction", key="run_extract", disabled=demo_mode):
            _run_stage("extraction", _execute_extraction)

    with tab4:
        st.markdown("""
        **Stage 3: Chunking** splits extracted documents into overlapping text chunks.
        **Stage 4: Embedding** converts chunks into vectors and stores them in ChromaDB.

        These stages run locally using the multilingual sentence-transformer model.
        """)

        extraction_data = load_json("extraction_summary.json")
        if extraction_data:
            st.success(f"✅ {extraction_data.get('successful', 0)} documents available from extraction")
        else:
            st.warning("⚠️ Run Stage 2 (Extraction) first")

        if st.button("Run Chunking + Embedding", key="run_chunk_embed", disabled=demo_mode):
            _run_stage("chunking & embedding", _execute_chunk_and_embed)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Log Viewer ───────────────────────────────────────
    st.markdown("### Recent Logs")
    log_file = st.selectbox(
        "Select log file",
        ["ingestion.log", "extraction.log", "chunking.log", "embedding.log", "retrieval.log", "evaluation.log"],
        label_visibility="collapsed",
    )
    log_path = os.path.join(BASE_DIR, "logs", log_file)
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
        recent = lines[-50:] if len(lines) > 50 else lines
        st.code("".join(recent), language="log")
    else:
        st.info(f"No log file found: {log_file}")


# ─── Execution helpers ──────────────────────────────────────

def _run_stage(stage_name: str, execute_fn):
    """Run a single pipeline stage with progress display."""
    progress = st.progress(0, text=f"Starting {stage_name}...")
    log_area = st.empty()

    try:
        progress.progress(10, text=f"Running {stage_name}...")
        output = execute_fn()
        progress.progress(100, text=f"✅ {stage_name.title()} complete!")
        if output:
            with st.expander("📋 Execution Output", expanded=True):
                st.code(output, language="text")
        st.success(f"✅ {stage_name.title()} completed successfully!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        progress.progress(100, text=f"❌ {stage_name.title()} failed")
        st.error(f"❌ Error: {str(e)}")


def _run_full_pipeline():
    """Run all stages sequentially."""
    stages = [
        ("Stage 1: Ingestion", _execute_ingestion),
        ("Stage 2: Extraction", _execute_extraction),
        ("Stage 3–4: Chunking & Embedding", _execute_chunk_and_embed),
    ]

    overall = st.progress(0, text="Starting full pipeline...")
    status_container = st.container()

    for i, (name, fn) in enumerate(stages):
        pct = int((i / len(stages)) * 100)
        overall.progress(pct, text=f"Running {name}...")

        with status_container:
            with st.spinner(f"⏳ {name}..."):
                try:
                    output = fn()
                    st.success(f"✅ {name} — done")
                except Exception as e:
                    st.error(f"❌ {name} failed: {e}")
                    overall.progress(100, text="Pipeline stopped due to error")
                    return

    overall.progress(100, text="✅ Full pipeline complete!")
    st.balloons()
    time.sleep(2)
    st.rerun()


def _capture_output(fn):
    """Run a function and capture its stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _execute_ingestion() -> str:
    from stage1_ingestion import ingest_data
    return _capture_output(ingest_data)


def _execute_extraction() -> str:
    from stage2_extraction import extract_all
    return _capture_output(extract_all)


def _execute_chunk_and_embed() -> str:
    from stage3_chunking import chunk_all
    from stage4_embedding import embed_all
    out1 = _capture_output(chunk_all)
    out2 = _capture_output(embed_all)
    return out1 + "\n" + out2

"""Streamlit UI for the AI Document Intelligence System."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import (
    UPLOAD_DIR,
    ConfigError,
    ensure_data_dirs,
    get_settings,
    require_api_key,
)
from src.comparator import compare_documents
from src.embeddings import build_embeddings, build_llm
from src.indexer import index_pdf
from src.output_models import (
    DocumentComparison,
    DocumentSummary,
    QuestionAnswer,
    RiskAnalysis,
    SourceCitation,
)
from src.rag_chain import answer_question
from src.risk_analyzer import analyze_risks
from src.summarizer import collect_document_chunks, summarize_documents
from src.utils import AppError
from src.vector_store import listed_documents, load_vector_store


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------

def _init_state() -> None:
    ensure_data_dirs()

    st.session_state.setdefault("store", None)
    st.session_state.setdefault("chat_history", [])


# ---------------------------------------------------------
# CACHED MODELS
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _embeddings():
    """Create and cache the embedding model."""
    print("\n[APP] Initializing embeddings...")
    embeddings = build_embeddings()
    print(f"[APP] Embeddings ready: {type(embeddings).__name__}")
    return embeddings


@st.cache_resource(show_spinner=False)
def _llm():
    """Create and cache the LLM."""
    print("\n[APP] Initializing LLM...")
    llm = build_llm()
    print(f"[APP] LLM ready: {type(llm).__name__}")
    return llm


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

def _configured() -> bool:
    """
    Check whether the application is configured.

    Ollama does NOT require an OpenAI API key.
    """

    try:
        settings = get_settings()

        print("\n[APP] Configuration check")
        print(f"[APP] Provider: {settings.provider}")
        print(f"[APP] LLM: {settings.llm_model}")
        print(f"[APP] Embedding: {settings.embedding_model}")

        # Ollama works locally without an API key.
        if settings.provider == "ollama":
            return True

        # OpenAI still requires an API key if explicitly selected.
        require_api_key(settings)
        return True

    except ConfigError as exc:
        print(f"[APP] Configuration error: {exc}")
        return False


# ---------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------

def _get_store():
    """Load the vector store when needed."""

    if st.session_state.store is None and _configured():
        try:
            print("\n[APP] Loading vector store...")

            st.session_state.store = load_vector_store(
                _embeddings()
            )

            print("[APP] Vector store loaded successfully.")

        except AppError as exc:
            print(f"[APP] Vector store error: {exc}")
            st.sidebar.error(str(exc))

        except Exception as exc:
            print(f"[APP] Unexpected vector store error: {exc}")
            st.sidebar.error(f"Could not load vector store: {exc}")

    return st.session_state.store


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

def _save_upload(uploaded) -> Path:
    destination = UPLOAD_DIR / uploaded.name
    destination.write_bytes(uploaded.getbuffer())
    return destination


# ---------------------------------------------------------
# SOURCES
# ---------------------------------------------------------

def _render_sources(sources: list[SourceCitation]) -> None:
    if not sources:
        st.caption("No source citations available.")
        return

    with st.expander("Sources", expanded=True):
        for source in sources:
            page = f" · page {source.page}" if source.page else ""

            st.markdown(
                f"**{source.filename}**{page}"
            )

            if source.excerpt:
                st.caption(source.excerpt)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def _sidebar() -> tuple[str | None, list[str]]:
    st.sidebar.header("Documents")

    settings = get_settings()

    # -----------------------------------------------------
    # Provider information
    # -----------------------------------------------------

    st.sidebar.caption(
        f"Provider: `{settings.provider}`"
    )

    st.sidebar.caption(
        f"Model: `{settings.llm_model}` · top_k={settings.top_k}"
    )

    st.sidebar.caption(
        f"Embedding: `{settings.embedding_model}`"
    )

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------

    uploads = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploads and st.sidebar.button(
        "Index documents",
        type="primary",
        use_container_width=True,
    ):

        print("\n")
        print("=" * 60)
        print("[APP] INDEXING STARTED")
        print("=" * 60)

        if not _configured():
            st.sidebar.error(
                "Application configuration is invalid."
            )
        else:

            with st.sidebar.status(
                "Indexing…",
                expanded=True,
            ) as status:

                try:
                    embeddings = _embeddings()

                    store = _get_store()

                    for uploaded in uploads:

                        print(
                            f"\n[APP] Processing: {uploaded.name}"
                        )

                        path = _save_upload(uploaded)

                        st.write(
                            f"Processing {uploaded.name}"
                        )

                        store, info = index_pdf(
                            path,
                            embeddings,
                            store,
                        )

                        if info.get("skipped"):

                            st.write(
                                f"Skipped duplicate: "
                                f"{info['filename']}"
                            )

                            print(
                                f"[APP] Skipped duplicate: "
                                f"{info['filename']}"
                            )

                        else:

                            message = (
                                f"Indexed {info['filename']} "
                                f"({info.get('pages')} pages, "
                                f"{info.get('chunk_count')} chunks)"
                            )

                            st.write(message)

                            print(
                                f"[APP] {message}"
                            )

                    st.session_state.store = store

                    status.update(
                        label="Indexing complete",
                        state="complete",
                    )

                    print("\n[APP] INDEXING COMPLETE")
                    print("=" * 60)

                    st.rerun()

                except (AppError, ConfigError) as exc:

                    print(
                        f"\n[APP] INDEXING ERROR: {exc}"
                    )

                    status.update(
                        label="Indexing failed",
                        state="error",
                    )

                    st.error(str(exc))

                except Exception as exc:

                    print(
                        f"\n[APP] UNEXPECTED INDEXING ERROR: "
                        f"{repr(exc)}"
                    )

                    status.update(
                        label="Indexing failed",
                        state="error",
                    )

                    st.error(
                        f"Unexpected indexing error: {exc}"
                    )

    # -----------------------------------------------------
    # Indexed documents
    # -----------------------------------------------------

    records = listed_documents()

    names = [
        item["filename"]
        for item in records
    ]

    st.sidebar.subheader("Indexed documents")

    if not records:

        st.sidebar.info(
            "No documents indexed yet."
        )

        selected = None

    else:

        for item in records:

            st.sidebar.markdown(
                f"- {item['filename']}  \n"
                f"`{item.get('chunk_count', '?')} chunks`"
            )

        selected = st.sidebar.selectbox(
            "Document for operations",
            names,
        )

    return selected, names


# ---------------------------------------------------------
# CHAT / Q&A
# ---------------------------------------------------------

def _tab_chat(
    selected: str | None,
    names: list[str],
) -> None:

    st.subheader("Chat with documents")

    scope = st.radio(
        "Search in",
        options=[
            "Selected document",
            "All indexed documents",
        ],
        horizontal=True,
        disabled=not names,
    )

    # Previous messages
    for role, content in st.session_state.chat_history:

        with st.chat_message(role):
            st.write(content)

    question = st.chat_input(
        "Ask a question about the uploaded documents"
    )

    if not question:
        return

    if not names:
        st.error(
            "Upload and index at least one PDF first."
        )
        return

    filenames = (
        [selected]
        if scope == "Selected document" and selected
        else None
    )

    st.session_state.chat_history.append(
        ("user", question)
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving relevant context…"
        ):

            try:

                print("\n[APP] Q&A started")
                print(f"[APP] Question: {question}")

                result: QuestionAnswer = answer_question(
                    question,
                    _get_store(),
                    llm=_llm(),
                    filenames=filenames,
                )

                print("[APP] Q&A completed")

            except (AppError, ConfigError) as exc:

                print(
                    f"[APP] Q&A ERROR: {exc}"
                )

                st.error(str(exc))
                return

            except Exception as exc:

                print(
                    f"[APP] Q&A UNEXPECTED ERROR: "
                    f"{repr(exc)}"
                )

                st.error(
                    f"Unexpected Q&A error: {exc}"
                )
                return

        st.write(result.answer)

        if result.found_in_documents:

            st.caption(
                f"Confidence: {result.confidence}"
            )

            _render_sources(
                result.sources
            )

        st.session_state.chat_history.append(
            ("assistant", result.answer)
        )


# ---------------------------------------------------------
# SUMMARIZE
# ---------------------------------------------------------

def _tab_summarize(
    selected: str | None,
) -> None:

    st.subheader("Summarize")

    if not selected:

        st.info(
            "Index a document and select it in the sidebar."
        )

        return

    if st.button(
        "Generate summary",
        type="primary",
        key="generate_summary_button",
    ):

        print("\n")
        print("=" * 60)
        print("[APP] SUMMARY BUTTON PRESSED")
        print("=" * 60)
        print(f"[APP] Selected document: {selected}")

        with st.spinner(
            "Summarizing document with Ollama…"
        ):

            try:

                print(
                    "[APP] Getting vector store..."
                )

                store = _get_store()

                if store is None:
                    raise AppError(
                        "Vector store is not available. "
                        "Please index the document first."
                    )

                print(
                    "[APP] Collecting document chunks..."
                )

                chunks = collect_document_chunks(
                    store,
                    selected,
                )

                print(
                    f"[APP] Retrieved chunks: {len(chunks)}"
                )

                if not chunks:
                    raise AppError(
                        "No indexed chunks were found "
                        f"for {selected}."
                    )

                print(
                    "[APP] Creating/loading Ollama LLM..."
                )

                llm = _llm()

                print(
                    f"[APP] LLM object: {type(llm).__name__}"
                )

                print(
                    "[APP] Calling summarize_documents..."
                )

                summary: DocumentSummary = (
                    summarize_documents(
                        selected,
                        store,
                        all_chunks=chunks,
                        llm=llm,
                        use_cache=False,
                    )
                )

                print(
                    "[APP] SUMMARY COMPLETED SUCCESSFULLY"
                )

            except (AppError, ConfigError) as exc:

                print(
                    "\n[APP] SUMMARY ERROR:"
                )
                print(repr(exc))

                st.error(str(exc))
                return

            except Exception as exc:

                print(
                    "\n[APP] SUMMARY UNEXPECTED ERROR:"
                )
                print(repr(exc))

                st.error(
                    f"Unexpected summarization error: {exc}"
                )

                return

        # -------------------------------------------------
        # Display summary
        # -------------------------------------------------

        st.markdown(
            "#### Executive summary"
        )

        st.write(
            summary.executive_summary
        )

        st.markdown(
            "#### Key points"
        )

        if summary.key_points:

            for point in summary.key_points:
                st.markdown(
                    f"- {point}"
                )

        else:

            st.caption(
                "No key points found."
            )

        st.markdown(
            "#### Important dates"
        )

        if summary.important_dates:

            for item in summary.important_dates:
                st.markdown(
                    f"- {item}"
                )

        else:

            st.caption(
                "No dates found in the document."
            )

        st.markdown(
            "#### Potential risks"
        )

        if summary.potential_risks:

            for item in summary.potential_risks:
                st.markdown(
                    f"- {item}"
                )

        else:

            st.caption(
                "No potential risks identified."
            )

        _render_sources(
            summary.sources
        )


# ---------------------------------------------------------
# KEY POINTS
# ---------------------------------------------------------

def _tab_key_points(
    selected: str | None,
) -> None:

    st.subheader("Key points")

    if not selected:

        st.info(
            "Index a document and select it in the sidebar."
        )

        return

    if st.button(
        "Extract key points",
        type="primary",
        key="extract_key_points_button",
    ):

        print("\n")
        print("=" * 60)
        print("[APP] KEY POINTS BUTTON PRESSED")
        print("=" * 60)
        print(
            f"[APP] Selected document: {selected}"
        )

        with st.spinner(
            "Extracting key points with Ollama…"
        ):

            try:

                print(
                    "[APP] Getting vector store..."
                )

                store = _get_store()

                if store is None:
                    raise AppError(
                        "Vector store is not available. "
                        "Please index the document first."
                    )

                print(
                    "[APP] Collecting document chunks..."
                )

                chunks = collect_document_chunks(
                    store,
                    selected,
                )

                print(
                    f"[APP] Retrieved chunks: {len(chunks)}"
                )

                if not chunks:
                    raise AppError(
                        "No indexed chunks were found "
                        f"for {selected}."
                    )

                print(
                    "[APP] Creating/loading Ollama LLM..."
                )

                llm = _llm()

                print(
                    "[APP] Calling summarize_documents "
                    "for key-point extraction..."
                )

                summary = summarize_documents(
                    selected,
                    store,
                    all_chunks=chunks,
                    llm=llm,
                    use_cache=False,
                )

                print(
                    "[APP] KEY POINTS COMPLETED"
                )

            except (AppError, ConfigError) as exc:

                print(
                    "\n[APP] KEY POINTS ERROR:"
                )
                print(repr(exc))

                st.error(str(exc))
                return

            except Exception as exc:

                print(
                    "\n[APP] KEY POINTS UNEXPECTED ERROR:"
                )
                print(repr(exc))

                st.error(
                    f"Unexpected key-points error: {exc}"
                )

                return

        if not summary.key_points:

            st.warning(
                "No key points were found."
            )

            return

        for point in summary.key_points:

            st.markdown(
                f"- {point}"
            )

        _render_sources(
            summary.sources
        )


# ---------------------------------------------------------
# RISK ANALYSIS
# ---------------------------------------------------------

def _tab_risks(
    selected: str | None,
) -> None:

    st.subheader("Risk analysis")

    st.warning(
        "Potential risk analysis only — "
        "this is not legal advice."
    )

    if not selected:

        st.info(
            "Index a document and select it in the sidebar."
        )

        return

    if st.button(
        "Analyze risks",
        type="primary",
        key="analyze_risks_button",
    ):

        with st.spinner(
            "Analyzing potential risks…"
        ):

            try:

                analysis: RiskAnalysis = (
                    analyze_risks(
                        selected,
                        _get_store(),
                        llm=_llm(),
                    )
                )

            except (AppError, ConfigError) as exc:

                st.error(str(exc))
                return

            except Exception as exc:

                print(
                    f"[APP] RISK ERROR: {repr(exc)}"
                )

                st.error(
                    f"Unexpected risk-analysis error: {exc}"
                )

                return

        st.caption(
            analysis.disclaimer
        )

        st.markdown(
            f"**Overall severity:** "
            f"{analysis.overall_severity}"
        )

        if not analysis.risks:

            st.success(
                "No potential issues were "
                "identified in the retrieved content."
            )

            return

        for risk in analysis.risks:

            st.markdown(
                f"### {risk.title} · {risk.severity}"
            )

            st.caption(
                risk.category
            )

            st.write(
                risk.explanation
            )

            _render_sources(
                risk.sources
            )


# ---------------------------------------------------------
# COMPARE DOCUMENTS
# ---------------------------------------------------------

def _tab_compare(
    names: list[str],
) -> None:

    st.subheader(
        "Compare documents"
    )

    if len(names) < 2:

        st.info(
            "Index at least two PDFs to compare them."
        )

        return

    col_a, col_b = st.columns(2)

    with col_a:

        doc_a = st.selectbox(
            "Document A",
            names,
            key="compare_a",
        )

    with col_b:

        remaining = [
            name
            for name in names
            if name != doc_a
        ] or names

        doc_b = st.selectbox(
            "Document B",
            remaining,
            key="compare_b",
        )

    if st.button(
        "Compare",
        type="primary",
        key="compare_documents_button",
    ):

        with st.spinner(
            "Comparing retrieved clauses…"
        ):

            try:

                result: DocumentComparison = (
                    compare_documents(
                        doc_a,
                        doc_b,
                        _get_store(),
                        llm=_llm(),
                    )
                )

            except (AppError, ConfigError) as exc:

                st.error(str(exc))
                return

            except Exception as exc:

                print(
                    f"[APP] COMPARE ERROR: "
                    f"{repr(exc)}"
                )

                st.error(
                    f"Unexpected comparison error: {exc}"
                )

                return

        st.markdown(
            "#### Comparison summary"
        )

        st.write(
            result.summary
        )

        if not result.differences:

            st.success(
                "No meaningful differences were "
                "detected in the retrieved content."
            )

            return

        for diff in result.differences:

            st.markdown(
                f"### {diff.category} · "
                f"{diff.change_type}"
            )

            left, right = st.columns(2)

            with left:

                st.markdown(
                    f"**{result.document_a}**"
                )

                st.write(
                    diff.document_a_value
                )

            with right:

                st.markdown(
                    f"**{result.document_b}**"
                )

                st.write(
                    diff.document_b_value
                )

            st.markdown(
                f"**Impact:** {diff.impact}"
            )

            _render_sources(
                diff.sources
            )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    _init_state()

    st.title(
        "AI Document Intelligence"
    )

    st.caption(
        "RAG Q&A, summaries, risk analysis, "
        "and comparison with source citations."
    )

    selected, names = _sidebar()

    chat, summarize, points, risks, compare = st.tabs(
        [
            "Chat",
            "Summarize",
            "Key Points",
            "Risk Analysis",
            "Compare Documents",
        ]
    )

    with chat:

        _tab_chat(
            selected,
            names,
        )

    with summarize:

        _tab_summarize(
            selected
        )

    with points:

        _tab_key_points(
            selected
        )

    with risks:

        _tab_risks(
            selected
        )

    with compare:

        _tab_compare(
            names
        )


if __name__ == "__main__":
    main()
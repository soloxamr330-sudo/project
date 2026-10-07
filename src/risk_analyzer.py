"""Potential risk analysis from retrieved document context. Not legal advice."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.embeddings import build_llm
from src.output_models import RiskAnalysis
from src.retriever import retrieve
from src.summarizer import collect_document_chunks
from src.utils import (
    AppError,
    citations_from_documents,
    format_context,
    invoke_structured,
    read_json_cache,
    write_json_cache,
)
from src.vector_store import LocalVectorStore

RISK_QUERY = (
    "termination penalty payment penalty automatic renewal unusual obligations "
    "deadline restrictive clause liability indemnification late fee auto-renew"
)

SYSTEM_PROMPT = """You analyze documents for POTENTIAL risks.
This is not legal advice and you must say so.
Only report issues supported by the provided text.
Look for: termination penalties, payment penalties, automatic renewal,
unusual obligations, deadlines, and restrictive clauses.
Severity:
- HIGH: significant financial penalty, automatic lock-in, or severe restriction
- MEDIUM: notable obligation, deadline, or one-sided term
- LOW: minor or standard caution
If no risks are found, return an empty risks list and overall_severity LOW.
Never invent clauses or page numbers.
"""


def analyze_risks(
    filename: str,
    store: LocalVectorStore | None,
    llm: BaseChatModel | None = None,
    use_cache: bool = True,
) -> RiskAnalysis:
    if not filename:
        raise AppError("Select a document for risk analysis.")

    docs = retrieve(store, RISK_QUERY, k=10, filenames=[filename])
    extra = []
    try:
        extra = collect_document_chunks(store, filename, max_chunks=20)
    except AppError:
        extra = []
    merged = _dedupe_docs(docs + extra)[:16]
    if not merged:
        raise AppError(f"No indexed content found for {filename}.")

    file_hash = str(merged[0].metadata.get("file_hash", filename))
    cache_key = f"risks_{file_hash}.json"
    if use_cache:
        cached = read_json_cache(cache_key)
        if cached:
            return RiskAnalysis.model_validate(cached)

    llm = llm or build_llm()
    user_prompt = (
        f"Document: {filename}\n\nContext:\n{format_context(merged)}\n\n"
        "Return a structured potential risk analysis based only on this context."
    )
    try:
        result = invoke_structured(llm, RiskAnalysis, SYSTEM_PROMPT, user_prompt)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("Risk analysis failed. Check your API key and try again.") from exc

    if "legal advice" not in result.disclaimer.lower():
        result.disclaimer = (
            "This is potential risk analysis based on uploaded documents, not legal advice."
        )
    for risk in result.risks:
        if not risk.sources:
            risk.sources = citations_from_documents(merged, limit=3)

    if use_cache:
        write_json_cache(cache_key, result.model_dump())
    return result


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    seen: set[tuple[str, int | None, str]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = (
            str(doc.metadata.get("source")),
            doc.metadata.get("page"),
            doc.page_content[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique

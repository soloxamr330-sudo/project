"""Compare two indexed documents and return structured differences."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from src.embeddings import build_llm
from src.output_models import DocumentComparison
from src.retriever import retrieve
from src.summarizer import collect_document_chunks
from src.utils import (
    AppError,
    format_context,
    invoke_structured,
    read_json_cache,
    write_json_cache,
)
from src.vector_store import LocalVectorStore

COMPARE_QUERY = (
    "payment amount date obligation termination responsibility fee penalty "
    "renewal liability deadline clause party"
)

SYSTEM_PROMPT = """You compare two documents.
Identify meaningful differences only:
added/removed clauses, changed amounts, changed dates, changed obligations,
payment changes, termination changes, and responsibility changes.
For each difference include both document values and the likely impact.
Use only the provided text. Do not invent missing clauses.
If documents are substantially the same, say so and return few or no differences.
This is informational analysis, not legal advice.
"""


def compare_documents(
    filename_a: str,
    filename_b: str,
    store: LocalVectorStore | None,
    llm: BaseChatModel | None = None,
    use_cache: bool = True,
) -> DocumentComparison:
    if not filename_a or not filename_b:
        raise AppError("Select two documents to compare.")
    if filename_a == filename_b:
        raise AppError("Choose two different documents.")

    docs_a = _context_for(store, filename_a)
    docs_b = _context_for(store, filename_b)
    hash_a = str(docs_a[0].metadata.get("file_hash", filename_a))
    hash_b = str(docs_b[0].metadata.get("file_hash", filename_b))
    cache_key = f"compare_{hash_a}_{hash_b}.json"
    if use_cache:
        cached = read_json_cache(cache_key)
        if cached:
            return DocumentComparison.model_validate(cached)

    llm = llm or build_llm()
    user_prompt = (
        f"Document A: {filename_a}\n{format_context(docs_a)}\n\n"
        f"Document B: {filename_b}\n{format_context(docs_b)}\n\n"
        "Return a structured comparison of meaningful differences."
    )
    try:
        result = invoke_structured(llm, DocumentComparison, SYSTEM_PROMPT, user_prompt)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("Document comparison failed. Check your API key and try again.") from exc

    result.document_a = filename_a
    result.document_b = filename_b
    if use_cache:
        write_json_cache(cache_key, result.model_dump())
    return result


def _context_for(store: LocalVectorStore | None, filename: str):
    retrieved = retrieve(store, COMPARE_QUERY, k=8, filenames=[filename])
    extras = collect_document_chunks(store, filename, max_chunks=16)
    merged = []
    seen = set()
    for doc in retrieved + extras:
        key = (doc.metadata.get("page"), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    if not merged:
        raise AppError(f"No indexed content found for {filename}.")
    return merged[:14]

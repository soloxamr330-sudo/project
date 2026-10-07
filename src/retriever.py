"""Retrieval of relevant chunks from the persistent vector store."""

from __future__ import annotations

from langchain_core.documents import Document
from src.vector_store import LocalVectorStore

from src.config import get_settings
from src.utils import AppError


def retrieve(
    store: LocalVectorStore | None,
    query: str,
    k: int | None = None,
    filenames: list[str] | None = None,
) -> list[Document]:
    if store is None:
        raise AppError("No documents are indexed yet. Upload a PDF first.")
    query = (query or "").strip()
    if not query:
        raise AppError("Enter a question or topic to search.")

    top_k = k or get_settings().top_k
    try:
        candidates = store.similarity_search(query, k=top_k, filenames=filenames)
    except Exception as exc:
        raise AppError("Vector search failed. The index may be corrupted.") from exc

    if filenames:
        allowed = set(filenames)
        candidates = [doc for doc in candidates if doc.metadata.get("source") in allowed]

    return candidates[:top_k]


def retrieve_for_document(
    store: LocalVectorStore | None,
    query: str,
    filename: str,
    k: int | None = None,
) -> list[Document]:
    return retrieve(store, query, k=k, filenames=[filename])

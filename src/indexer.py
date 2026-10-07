"""Indexing pipeline: PDF -> chunks -> embeddings -> persistent vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.utils import AppError, file_sha256
from src.vector_store import (
    LocalVectorStore,
    add_documents,
    is_indexed,
    load_vector_store,
    register_document,
    save_vector_store,
)


def index_pdf(
    path: str | Path,
    embeddings: Embeddings,
    store: LocalVectorStore | None = None,
) -> tuple[LocalVectorStore, dict]:
    pdf_path = Path(path)
    file_hash = file_sha256(pdf_path)
    if is_indexed(file_hash):
        store = store or load_vector_store(embeddings)
        if store is None:
            raise AppError("Document is registered but the vector index is missing. Re-upload to rebuild.")
        return store, {
            "filename": pdf_path.name,
            "file_hash": file_hash,
            "skipped": True,
            "reason": "Already indexed (identical file hash).",
        }

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    if not chunks:
        raise AppError(f"{pdf_path.name} produced no usable text chunks.")

    store = store or load_vector_store(embeddings)
    store = add_documents(store, chunks, embeddings)
    record = register_document(pdf_path, len(chunks))
    return store, {
        "filename": pdf_path.name,
        "file_hash": file_hash,
        "skipped": False,
        "chunk_count": len(chunks),
        "pages": len(documents),
        **record,
    }


def persist_store(store: LocalVectorStore) -> None:
    save_vector_store(store)

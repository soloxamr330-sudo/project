"""Persistent FAISS vector store with a document registry to skip duplicate indexing."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import REGISTRY_PATH, VECTOR_DIR, ensure_data_dirs
from src.utils import AppError, file_sha256

def _index_file() -> Path:
    return VECTOR_DIR / "index.faiss"


def _docs_file() -> Path:
    return VECTOR_DIR / "documents.pkl"


class LocalVectorStore:
    """Cosine-similarity FAISS index with in-memory document metadata."""

    def __init__(self, embeddings: Embeddings, index: faiss.Index, documents: list[Document]):
        self.embeddings = embeddings
        self.index = index
        self.documents = documents

    def add_documents(self, chunks: list[Document]) -> None:
        vectors = _embed_matrix(self.embeddings, [chunk.page_content for chunk in chunks])
        if self.index.ntotal == 0 and vectors.shape[1] != self.index.d:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents.extend(chunks)

    def similarity_search(
        self,
        query: str,
        k: int,
        filenames: list[str] | None = None,
    ) -> list[Document]:
        if self.index.ntotal == 0:
            return []
        query_vec = _embed_matrix(self.embeddings, [query])
        if filenames:
            allowed = set(filenames)
            eligible = [
                i
                for i, doc in enumerate(self.documents)
                if doc.metadata.get("source") in allowed
            ]
            if not eligible:
                return []
            k = min(k, len(eligible))
            subset = _reconstruct_vectors(self.index, eligible)
            scores = subset @ query_vec[0]
            ranked = np.argsort(-scores)[:k]
            return [self.documents[eligible[int(i)]] for i in ranked]

        k = min(k, self.index.ntotal)
        _scores, indices = self.index.search(query_vec, k)
        results: list[Document] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(self.documents[int(idx)])
        return results

    def documents_for(self, filename: str, max_chunks: int | None = None) -> list[Document]:
        matched = [doc for doc in self.documents if doc.metadata.get("source") == filename]
        if max_chunks is None:
            return matched
        return matched[:max_chunks]


def _reconstruct_vectors(index: faiss.Index, ids: list[int]) -> np.ndarray:
    id_array = np.asarray(ids, dtype=np.int64)
    try:
        vectors = index.reconstruct_batch(id_array)
    except Exception:
        vectors = np.vstack([index.reconstruct(int(i)) for i in ids])
    return np.asarray(vectors, dtype="float32")


def _embed_matrix(embeddings: Embeddings, texts: list[str]) -> np.ndarray:
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    if vectors.ndim != 2 or vectors.size == 0:
        raise AppError("Embeddings returned an empty vector.")
    faiss.normalize_L2(vectors)
    return vectors


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"documents": {}}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"documents": {}}
    data.setdefault("documents", {})
    return data


def _save_registry(registry: dict[str, Any]) -> None:
    ensure_data_dirs()
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def listed_documents() -> list[dict[str, Any]]:
    docs = _load_registry()["documents"]
    return sorted(docs.values(), key=lambda item: item.get("filename", ""))


def is_indexed(file_hash: str) -> bool:
    return file_hash in _load_registry()["documents"]


def load_vector_store(embeddings: Embeddings) -> LocalVectorStore | None:
    ensure_data_dirs()
    if not _index_file().exists() or not _docs_file().exists():
        return None
    try:
        index = faiss.read_index(str(_index_file()))
        documents = pickle.loads(_docs_file().read_bytes())
        if not isinstance(documents, list):
            raise ValueError("Invalid document store")
        return LocalVectorStore(embeddings, index, documents)
    except Exception as exc:
        raise AppError("Failed to load the vector database. Try re-indexing documents.") from exc


def save_vector_store(store: LocalVectorStore) -> None:
    ensure_data_dirs()
    try:
        faiss.write_index(store.index, str(_index_file()))
        _docs_file().write_bytes(pickle.dumps(store.documents))
    except Exception as exc:
        raise AppError("Failed to save the vector database.") from exc


def add_documents(
    store: LocalVectorStore | None,
    chunks: list[Document],
    embeddings: Embeddings,
) -> LocalVectorStore:
    if not chunks:
        raise AppError("No text chunks were produced for indexing.")
    try:
        if store is None:
            vectors = _embed_matrix(embeddings, [chunk.page_content for chunk in chunks])
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            store = LocalVectorStore(embeddings, index, list(chunks))
        else:
            store.add_documents(chunks)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("Failed to index documents into the vector database.") from exc
    save_vector_store(store)
    return store


def register_document(path: Path, chunk_count: int) -> dict[str, Any]:
    file_hash = file_sha256(path)
    registry = _load_registry()
    record = {
        "file_hash": file_hash,
        "filename": path.name,
        "path": str(path.resolve()),
        "chunk_count": chunk_count,
    }
    registry["documents"][file_hash] = record
    _save_registry(registry)
    return record


def document_filenames() -> list[str]:
    return [item["filename"] for item in listed_documents()]

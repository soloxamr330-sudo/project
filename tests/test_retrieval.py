from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retriever import retrieve
from src.text_splitter import split_documents
from src.utils import AppError
from src.vector_store import add_documents


class TinyEmbeddings(Embeddings):
    """Deterministic embeddings for offline retrieval tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("payment")),
            float(lowered.count("termination")),
            float(lowered.count("penalty")),
            float(lowered.count("renewal")),
        ]


def test_retrieve_relevant_chunks_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.vector_store.VECTOR_DIR", tmp_path)
    docs = [
        Document(
            page_content="The payment amount is 5000 USD due on the first of each month.",
            metadata={"source": "a.pdf", "page": 1, "file_hash": "h1"},
        ),
        Document(
            page_content="Termination requires 30 days notice and a 10 percent penalty.",
            metadata={"source": "a.pdf", "page": 2, "file_hash": "h1"},
        ),
        Document(
            page_content="The office plants should be watered weekly.",
            metadata={"source": "a.pdf", "page": 3, "file_hash": "h1"},
        ),
    ]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=0)
    store = add_documents(None, chunks, TinyEmbeddings())
    results = retrieve(store, "What is the termination penalty?", k=1)
    assert results
    assert "termination" in results[0].page_content.lower()
    assert results[0].metadata["page"] == 2


def test_retrieve_without_store_raises() -> None:
    try:
        retrieve(None, "hello")
        assert False, "expected AppError"
    except AppError:
        pass


def test_retrieve_filters_to_selected_document(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.vector_store.VECTOR_DIR", tmp_path)
    docs = []
    for i in range(20):
        docs.append(
            Document(
                page_content=f"Termination penalty in vendor B contract section {i}.",
                metadata={"source": "b.pdf", "page": i + 1, "file_hash": "hb"},
            )
        )
    docs.append(
        Document(
            page_content="Termination penalty in client A is 5 percent.",
            metadata={"source": "a.pdf", "page": 1, "file_hash": "ha"},
        )
    )
    store = add_documents(None, docs, TinyEmbeddings())
    results = retrieve(store, "termination penalty", k=2, filenames=["a.pdf"])
    assert results
    assert all(doc.metadata["source"] == "a.pdf" for doc in results)
    assert "client A" in results[0].page_content
    other = retrieve(store, "termination penalty", k=3, filenames=["b.pdf"])
    assert other
    assert all(doc.metadata["source"] == "b.pdf" for doc in other)

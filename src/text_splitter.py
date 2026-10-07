"""Text chunking that preserves PDF page metadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    cleaned: list[Document] = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue
        metadata = dict(chunk.metadata)
        metadata.setdefault("source", "unknown")
        if "page" in metadata and metadata["page"] not in (None, ""):
            metadata["page"] = int(metadata["page"])
        chunk.page_content = text
        chunk.metadata = metadata
        cleaned.append(chunk)
    return cleaned

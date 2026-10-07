"""PDF loading with page-level metadata."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.utils import AppError, file_sha256

MIN_EXTRACTED_CHARS = 20
MAX_PDF_BYTES = 40 * 1024 * 1024


def load_pdf(path: str | Path) -> list[Document]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise AppError(f"File not found: {pdf_path.name}")
    if pdf_path.suffix.lower() != ".pdf":
        raise AppError(f"{pdf_path.name} is not a PDF.")
    if pdf_path.stat().st_size == 0:
        raise AppError(f"{pdf_path.name} is empty.")
    if pdf_path.stat().st_size > MAX_PDF_BYTES:
        raise AppError(f"{pdf_path.name} exceeds the 40 MB upload limit.")

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError, ValueError) as exc:
        raise AppError(f"{pdf_path.name} is not a valid PDF.") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise AppError(f"{pdf_path.name} is encrypted and cannot be read.") from exc

    file_hash = file_sha256(pdf_path)
    documents: list[Document] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": index + 1,
                    "file_path": str(pdf_path.resolve()),
                    "file_hash": file_hash,
                },
            )
        )

    extracted = "\n".join(doc.page_content for doc in documents).strip()
    if len(extracted) < MIN_EXTRACTED_CHARS:
        raise AppError(
            f"{pdf_path.name} has little or no extractable text. "
            "It may be scanned, image-only, or corrupted. OCR is not supported."
        )
    return documents


def load_pdfs(paths: list[str | Path]) -> tuple[list[Document], list[str]]:
    docs: list[Document] = []
    errors: list[str] = []
    for path in paths:
        try:
            docs.extend(load_pdf(path))
        except AppError as exc:
            errors.append(str(exc))
    if not docs and errors:
        raise AppError("\n".join(errors))
    return docs, errors

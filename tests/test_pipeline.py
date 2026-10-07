from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.document_loader import load_pdf
from src.output_models import (
    NOT_FOUND_ANSWER,
    DocumentComparison,
    DocumentSummary,
    QuestionAnswer,
    RiskAnalysis,
    RiskItem,
    SourceCitation,
)
from src.text_splitter import split_documents
from src.utils import AppError, citations_from_documents, file_sha256, format_context, text_sha256
from src.vector_store import is_indexed, listed_documents
from tests.pdf_factory import write_pdf


def test_pdf_load_preserves_page_metadata(tmp_path: Path) -> None:
    pdf = write_pdf(
        tmp_path / "contract.pdf",
        ["Payment is due in 30 days.", "Termination penalty is 10 percent."],
    )
    docs = load_pdf(pdf)
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "contract.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert "Payment" in docs[0].page_content
    assert "Termination" in docs[1].page_content
    assert docs[0].metadata["file_hash"] == file_sha256(pdf)


def test_empty_pdf_raises(tmp_path: Path) -> None:
    empty = tmp_path / "blank.pdf"
    empty.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntrailer\n<<>>\n%%EOF\n")
    with pytest.raises(AppError):
        load_pdf(empty)


def test_invalid_pdf_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not.pdf"
    bogus.write_bytes(b"this is not a pdf")
    with pytest.raises(AppError):
        load_pdf(bogus)


def test_scanned_like_pdf_raises(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scan.pdf", [" ", "  "])
    with pytest.raises(AppError, match="extractable text"):
        load_pdf(pdf)


def test_chunking_keeps_metadata(tmp_path: Path) -> None:
    pdf = write_pdf(
        tmp_path / "report.pdf",
        ["Alpha " * 80, "Beta clause with a deadline of 2026-01-01. " * 20],
    )
    docs = load_pdf(pdf)
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=20)
    assert chunks
    assert all(chunk.metadata.get("source") == "report.pdf" for chunk in chunks)
    assert all("page" in chunk.metadata for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_pydantic_models_validate() -> None:
    answer = QuestionAnswer(
        answer=NOT_FOUND_ANSWER,
        found_in_documents=False,
        sources=[],
        confidence="low",
    )
    assert answer.found_in_documents is False

    summary = DocumentSummary(
        executive_summary="A short summary.",
        key_points=["Point one"],
        important_dates=["2026-01-01"],
        potential_risks=["Auto-renewal"],
        sources=[SourceCitation(filename="a.pdf", page=1)],
    )
    assert summary.key_points[0] == "Point one"

    risks = RiskAnalysis(
        overall_severity="HIGH",
        risks=[
            RiskItem(
                title="Termination penalty",
                category="termination penalties",
                severity="HIGH",
                explanation="10% fee on early termination.",
                sources=[SourceCitation(filename="a.pdf", page=2)],
            )
        ],
    )
    assert risks.risks[0].severity == "HIGH"

    comparison = DocumentComparison(
        document_a="a.pdf",
        document_b="b.pdf",
        summary="Payment terms changed.",
        differences=[],
    )
    assert comparison.document_a == "a.pdf"

    with pytest.raises(ValidationError):
        QuestionAnswer(answer="x", found_in_documents=True, confidence="extreme")


def test_citations_and_context_helpers(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "src.pdf", ["Clause A lives on page one."])
    docs = load_pdf(pdf)
    citations = citations_from_documents(docs)
    assert citations[0].filename == "src.pdf"
    assert citations[0].page == 1
    context = format_context(docs)
    assert "src.pdf" in context
    assert "page 1" in context


def test_hashing_is_stable(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "same.pdf", ["Stable content"])
    assert file_sha256(pdf) == file_sha256(pdf)
    assert text_sha256("abc") == text_sha256("abc")
    assert text_sha256("abc") != text_sha256("abd")


def test_registry_empty_without_index(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.vector_store.REGISTRY_PATH", tmp_path / "registry.json")
    assert listed_documents() == []
    assert is_indexed("missing") is False

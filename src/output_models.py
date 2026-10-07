"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    filename: str = Field(description="Source PDF filename")
    page: int | None = Field(default=None, description="1-based page number when known")
    excerpt: str | None = Field(default=None, description="Short supporting excerpt")


class QuestionAnswer(BaseModel):
    answer: str
    found_in_documents: bool
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class DocumentSummary(BaseModel):
    executive_summary: str
    key_points: list[str] = Field(default_factory=list)
    important_dates: list[str] = Field(default_factory=list)
    potential_risks: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)


class RiskItem(BaseModel):
    title: str
    category: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    explanation: str
    sources: list[SourceCitation] = Field(default_factory=list)


class RiskAnalysis(BaseModel):
    disclaimer: str = (
        "This is potential risk analysis based on uploaded documents, not legal advice."
    )
    overall_severity: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    risks: list[RiskItem] = Field(default_factory=list)


class DifferenceItem(BaseModel):
    category: str
    change_type: Literal["added", "removed", "modified"]
    document_a_value: str
    document_b_value: str
    impact: str
    sources: list[SourceCitation] = Field(default_factory=list)


class DocumentComparison(BaseModel):
    document_a: str
    document_b: str
    summary: str
    differences: list[DifferenceItem] = Field(default_factory=list)


NOT_FOUND_ANSWER = "I could not find this information in the uploaded documents."

"""Shared helpers for hashing, citations, caching, and LLM structured calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from src.config import CACHE_DIR, ensure_data_dirs
from src.output_models import SourceCitation

T = TypeVar("T", bound=BaseModel)


class AppError(RuntimeError):
    """User-facing application error."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_path(name: str) -> Path:
    ensure_data_dirs()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return CACHE_DIR / safe


def read_json_cache(name: str) -> dict[str, Any] | None:
    path = cache_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json_cache(name: str, payload: dict[str, Any]) -> None:
    path = cache_path(name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def citations_from_documents(docs: list[Document], limit: int = 8) -> list[SourceCitation]:
    seen: set[tuple[str, int | None, str]] = set()
    citations: list[SourceCitation] = []
    for doc in docs:
        filename = str(doc.metadata.get("source") or "unknown")
        page_raw = doc.metadata.get("page")
        page = int(page_raw) if page_raw not in (None, "") else None
        excerpt = " ".join(doc.page_content.split())[:280]
        key = (filename, page, excerpt[:80])
        if key in seen:
            continue
        seen.add(key)
        citations.append(SourceCitation(filename=filename, page=page, excerpt=excerpt))
        if len(citations) >= limit:
            break
    return citations


def format_context(docs: list[Document]) -> str:
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Chunk {idx} | {source} | page {page}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
) -> T:
    """Prefer native structured output; fall back to JSON parsing."""
    messages = [
        ("system", system_prompt),
        ("human", user_prompt),
    ]
    try:
        structured = llm.with_structured_output(schema)
        result = structured.invoke(messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except Exception:
        pass

    json_system = (
        system_prompt
        + "\nReturn ONLY valid JSON matching this schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    raw = llm.invoke(
        [
            ("system", json_system),
            ("human", user_prompt),
        ]
    )
    content = getattr(raw, "content", raw)
    if not isinstance(content, str):
        content = str(content)
    parsed = _extract_json(content)
    try:
        return schema.model_validate(parsed)
    except ValidationError as exc:
        raise AppError("The model returned an invalid structured response.") from exc


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise AppError("Could not parse JSON from the model response.")

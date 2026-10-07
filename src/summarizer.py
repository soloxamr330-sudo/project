"""Document summarization and key-point extraction using the configured LLM."""

from __future__ import annotations

import re
from typing import Any

from src.config import require_api_key
from src.embeddings import build_llm
from src.output_models import DocumentSummary
from src.utils import (
    AppError,
    citations_from_documents,
    format_context,
)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI document analysis assistant.

Your job is to analyze ONLY the document text provided by the user.

IMPORTANT RULES:
1. Never invent information.
2. Never use outside knowledge.
3. Preserve exact names, dates, amounts, percentages,
   obligations, deadlines and contractual conditions.
4. Every answer must be supported by the supplied document.
5. Do not return JSON.
6. Do not explain your reasoning.
7. Do not add introductory text before the required sections.

You MUST return exactly these four sections:

EXECUTIVE_SUMMARY:
Write one clear paragraph summarizing the document.

KEY_POINTS:
- Important fact
- Important fact
- Important fact
- Important fact
- Important fact

IMPORTANT_DATES:
- Date: what it represents
- Date: what it represents

POTENTIAL_RISKS:
- Concrete risk supported by the document
- Concrete risk supported by the document

If there are no important dates, write:
- None identified

If there are no concrete risks, write:
- None identified
"""


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _clean_text(value: Any) -> str:
    """Convert an LLM response content value to clean text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    # Some LangChain responses may contain structured blocks.
    if isinstance(value, list):
        parts: list[str] = []

        for item in value:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(value).strip()


def _remove_markdown_prefix(text: str) -> str:
    """Remove common Markdown/list prefixes."""

    text = text.strip()

    # Bullet
    text = re.sub(r"^[-*•]\s+", "", text)

    # Numbered list
    text = re.sub(r"^\d+[\.\)]\s+", "", text)

    return text.strip()


def _is_empty_marker(text: str) -> bool:
    normalized = text.strip().lower()

    return normalized in {
        "",
        "none",
        "none identified",
        "none identified.",
        "n/a",
        "n/a.",
        "not applicable",
        "not applicable.",
        "no risks",
        "no risks identified",
        "no risks identified.",
        "no important dates",
        "no important dates identified",
    }


# ============================================================
# SECTION PARSER
# ============================================================

def _detect_section(line: str) -> str | None:
    """
    Detect one of the four supported sections.

    Handles:
      EXECUTIVE_SUMMARY:
      EXECUTIVE SUMMARY:
      ## EXECUTIVE_SUMMARY
      **EXECUTIVE_SUMMARY:**
      KEY POINTS:
      etc.
    """

    cleaned = line.strip()

    # Remove Markdown decoration.
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.lstrip("#").strip()

    normalized = cleaned.upper()

    # Remove trailing colon.
    normalized = normalized.rstrip(":").strip()

    # Normalize spaces / underscores.
    normalized = re.sub(r"[\s_]+", " ", normalized)

    mapping = {
        "EXECUTIVE SUMMARY": "EXECUTIVE_SUMMARY",
        "SUMMARY": "EXECUTIVE_SUMMARY",
        "KEY POINTS": "KEY_POINTS",
        "KEY POINT": "KEY_POINTS",
        "IMPORTANT DATES": "IMPORTANT_DATES",
        "IMPORTANT DATE": "IMPORTANT_DATES",
        "POTENTIAL RISKS": "POTENTIAL_RISKS",
        "POTENTIAL RISK": "POTENTIAL_RISKS",
        "RISKS": "POTENTIAL_RISKS",
        "RISK ANALYSIS": "POTENTIAL_RISKS",
    }

    return mapping.get(normalized)


def _parse_sections(text: str) -> DocumentSummary:
    """
    Parse the raw Ollama response into DocumentSummary.

    The parser intentionally supports several formatting styles because
    local models do not always reproduce prompts exactly.
    """

    text = _clean_text(text)

    print("\n")
    print("=" * 60)
    print("========== RAW OLLAMA SUMMARY RESPONSE ==========")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()

    if not text:
        raise AppError(
            "Ollama returned an empty response."
        )

    sections = {
        "EXECUTIVE_SUMMARY": [],
        "KEY_POINTS": [],
        "IMPORTANT_DATES": [],
        "POTENTIAL_RISKS": [],
    }

    current: str | None = None

    lines = text.splitlines()

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        detected = _detect_section(line)

        if detected:
            current = detected
            continue

        if current is None:
            # Ignore random introductory text before the first section.
            continue

        cleaned = _remove_markdown_prefix(line)

        # Ignore empty markers.
        if _is_empty_marker(cleaned):
            continue

        sections[current].append(cleaned)

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    executive_parts = sections["EXECUTIVE_SUMMARY"]

    executive_summary = " ".join(
        part.strip()
        for part in executive_parts
        if part.strip()
    ).strip()

    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    def clean_list(values: list[str]) -> list[str]:

        result: list[str] = []

        for value in values:

            value = _remove_markdown_prefix(value)

            if not value:
                continue

            if _is_empty_marker(value):
                continue

            # Avoid duplicated identical entries.
            if value not in result:
                result.append(value)

        return result

    key_points = clean_list(
        sections["KEY_POINTS"]
    )

    important_dates = clean_list(
        sections["IMPORTANT_DATES"]
    )

    potential_risks = clean_list(
        sections["POTENTIAL_RISKS"]
    )

    # --------------------------------------------------------
    # Debug parsed result
    # --------------------------------------------------------

    print("=" * 60)
    print("========== PARSED SUMMARY ==========")
    print("=" * 60)

    print(
        "Executive summary:",
        repr(executive_summary),
    )

    print(
        "Key points:",
        len(key_points),
    )

    for item in key_points:
        print(" -", item)

    print(
        "Important dates:",
        len(important_dates),
    )

    for item in important_dates:
        print(" -", item)

    print(
        "Potential risks:",
        len(potential_risks),
    )

    for item in potential_risks:
        print(" -", item)

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not executive_summary:

        # Sometimes a model returns the summary without correctly
        # writing the section heading. Try to recover useful prose.
        fallback_lines = []

        for raw_line in lines:

            cleaned = _remove_markdown_prefix(
                raw_line
            )

            if not cleaned:
                continue

            if _detect_section(raw_line):
                continue

            if _is_empty_marker(cleaned):
                continue

            fallback_lines.append(cleaned)

        if fallback_lines:

            executive_summary = " ".join(
                fallback_lines[:4]
            ).strip()

    if not executive_summary:

        raise AppError(
            "Ollama returned a response, but no usable "
            "executive summary could be extracted. "
            "See the Terminal for the raw Ollama response."
        )

    return DocumentSummary(
        executive_summary=executive_summary,
        key_points=key_points,
        important_dates=important_dates,
        potential_risks=potential_risks,
    )


# ============================================================
# LLM CALL
# ============================================================

def _call_llm(
    llm: Any,
    context: str,
) -> DocumentSummary:
    """Send document context to Ollama and parse the result."""

    prompt = f"""
Analyze the following document.

DOCUMENT
============================================================
{context}
============================================================

Return ONLY these four sections:

EXECUTIVE_SUMMARY:
Write one meaningful paragraph describing the document.

KEY_POINTS:
- Extract at least five important concrete facts when possible.
- Include names, amounts, services, obligations,
  responsibilities, clauses and conditions.

IMPORTANT_DATES:
- Extract every important date.
- Explain what each date represents.
- Do not invent dates.

POTENTIAL_RISKS:
- Extract concrete contractual or business risks
  explicitly supported by the document.
- If there are no supported risks, write:
  - None identified

Use ONLY the document above.

Do not return JSON.
Do not explain your instructions.
Do not use outside knowledge.
"""

    print()
    print("=" * 60)
    print("========== CALLING OLLAMA ==========")
    print("=" * 60)
    print(
        "Context length:",
        len(context),
    )
    print()

    try:

        response = llm.invoke(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                (
                    "human",
                    prompt,
                ),
            ]
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("========== OLLAMA CALL ERROR ==========")
        print("=" * 60)
        print(repr(exc))
        print("=" * 60)
        print()

        raise AppError(
            "Summarization failed while calling Ollama. "
            f"Error: {exc}"
        ) from exc

    content = getattr(
        response,
        "content",
        response,
    )

    content = _clean_text(content)

    print()
    print("=" * 60)
    print("========== OLLAMA RESPONSE RECEIVED ==========")
    print("=" * 60)
    print(
        "Response length:",
        len(content),
    )
    print("=" * 60)
    print()

    if not content:

        raise AppError(
            "Ollama returned an empty response."
        )

    return _parse_sections(content)


# ============================================================
# MAIN SUMMARY FUNCTION
# ============================================================

def summarize_documents(
    filename,
    store,
    all_chunks=None,
    llm=None,
    use_cache=False,
):
    """
    Generate a structured summary from all indexed chunks
    belonging to one document.
    """

    print()
    print("=" * 60)
    print("========== SUMMARY FUNCTION STARTED ==========")
    print("=" * 60)
    print(
        "Filename:",
        filename,
    )

    if not filename:

        raise AppError(
            "Select a document to summarize."
        )

    # --------------------------------------------------------
    # Get chunks
    # --------------------------------------------------------

    if all_chunks is not None:

        docs = list(all_chunks)

    else:

        docs = collect_document_chunks(
            store,
            filename,
        )

    print(
        "Document chunks:",
        len(docs),
    )

    if not docs:

        raise AppError(
            f"No indexed content found for {filename}."
        )

    # --------------------------------------------------------
    # Debug chunks
    # --------------------------------------------------------

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        print()
        print(
            f"--- CHUNK {index} ---"
        )

        print(
            doc.page_content[:1000]
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = format_context(
        docs
    )

    print()
    print("=" * 60)
    print(
        "CONTEXT LENGTH:",
        len(context),
    )
    print("=" * 60)

    if not context.strip():

        raise AppError(
            "The indexed document chunks contain no text."
        )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    if llm is None:

        print(
            "No LLM supplied. Building LLM..."
        )

        llm = build_llm()

    print(
        "LLM CREATED:",
        type(llm).__name__,
    )

    # --------------------------------------------------------
    # Call Ollama
    # --------------------------------------------------------

    result = _call_llm(
        llm,
        context,
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    result.sources = citations_from_documents(
        docs
    )

    print()
    print("=" * 60)
    print("========== SUMMARY SUCCESS ==========")
    print("=" * 60)

    print(
        "Executive summary length:",
        len(result.executive_summary),
    )

    print(
        "Key points:",
        len(result.key_points),
    )

    print(
        "Important dates:",
        len(result.important_dates),
    )

    print(
        "Potential risks:",
        len(result.potential_risks),
    )

    print("=" * 60)
    print()

    return result


# ============================================================
# KEY POINTS HELPER
# ============================================================

def key_points_from_summary(
    summary: DocumentSummary,
) -> list[str]:

    return list(
        summary.key_points
    )


# ============================================================
# DOCUMENT CHUNK COLLECTION
# ============================================================

def collect_document_chunks(
    store,
    filename,
    max_chunks=None,
):
    """Return all indexed chunks belonging to a document."""

    if store is None:

        raise AppError(
            "No documents are indexed yet."
        )

    if not filename:

        raise AppError(
            "No document was selected."
        )

    try:

        docs = store.documents_for(
            filename,
            max_chunks=max_chunks,
        )

    except Exception as exc:

        raise AppError(
            "Could not read chunks from the vector store."
        ) from exc

    if docs is None:

        return []

    return list(docs)
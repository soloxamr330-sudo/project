"""Question answering over retrieved context with structured output."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.embeddings import build_llm
from src.output_models import NOT_FOUND_ANSWER, QuestionAnswer, SourceCitation
from src.retriever import retrieve
from src.utils import AppError, citations_from_documents, format_context, invoke_structured
from src.vector_store import LocalVectorStore

SYSTEM_PROMPT = """You are a document Q&A assistant.
Answer ONLY from the retrieved context.
If the context does not contain the answer, set found_in_documents=false and set answer exactly to:
"I could not find this information in the uploaded documents."
Never invent facts, dates, clauses, prices, names, or page numbers.
Only cite sources that appear in the context metadata.
Use the provided filename and page numbers; do not guess missing pages.
"""


def answer_question(
    question: str,
    store: LocalVectorStore | None,
    llm: BaseChatModel | None = None,
    filenames: list[str] | None = None,
    k: int | None = None,
) -> QuestionAnswer:
    question = (question or "").strip()
    if not question:
        raise AppError("Please enter a question.")

    docs = retrieve(store, question, k=k, filenames=filenames)
    if not docs:
        return QuestionAnswer(
            answer=NOT_FOUND_ANSWER,
            found_in_documents=False,
            sources=[],
            confidence="low",
        )

    context = format_context(docs)
    user_prompt = (
        f"Question:\n{question}\n\nRetrieved context:\n{context}\n\n"
        "Respond with a structured answer grounded only in this context."
    )
    llm = llm or build_llm()
    try:
        result = invoke_structured(llm, QuestionAnswer, SYSTEM_PROMPT, user_prompt)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("The language model request failed. Check your API key and try again.") from exc

    if not result.found_in_documents:
        result.answer = NOT_FOUND_ANSWER
        result.sources = []
        result.confidence = "low"
        return result

    result.sources = _ground_sources(result.sources, docs)
    return result


def _ground_sources(sources: list[SourceCitation] | None, docs: list[Document]) -> list[SourceCitation]:
    allowed_files = {str(doc.metadata.get("source") or "unknown") for doc in docs}
    allowed_pages = {
        (str(doc.metadata.get("source") or "unknown"), doc.metadata.get("page"))
        for doc in docs
    }
    grounded = []
    for source in sources or []:
        if source.filename not in allowed_files:
            continue
        if source.page is not None and (source.filename, source.page) not in allowed_pages:
            source.page = None
        grounded.append(source)
    return grounded or citations_from_documents(docs)

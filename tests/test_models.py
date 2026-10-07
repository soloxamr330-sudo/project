from langchain_core.documents import Document

from src.output_models import NOT_FOUND_ANSWER, QuestionAnswer, SourceCitation
from src.rag_chain import _ground_sources


def test_not_found_constant() -> None:
    result = QuestionAnswer(
        answer=NOT_FOUND_ANSWER,
        found_in_documents=False,
        confidence="low",
    )
    assert result.answer == "I could not find this information in the uploaded documents."


def test_ground_sources_drops_foreign_citations() -> None:
    docs = [
        Document(page_content="Payment is 5000.", metadata={"source": "a.pdf", "page": 1}),
    ]
    sources = [
        SourceCitation(filename="other.pdf", page=9, excerpt="invented"),
        SourceCitation(filename="a.pdf", page=1, excerpt="Payment is 5000."),
    ]
    grounded = _ground_sources(sources, docs)
    assert all(item.filename == "a.pdf" for item in grounded)
    assert grounded[0].page == 1

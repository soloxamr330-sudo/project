# AI Document Intelligence System

RAG application for PDF Q&A, summarization, risk analysis, and document comparison with structured outputs and source citations.

## Overview

Upload one or more PDFs. The app extracts page-aware text, chunks it, embeds the chunks, and stores them in a local FAISS index. Questions retrieve only the relevant chunks. The language model answers from that context and returns Pydantic-structured results.

## Features

- Multi-PDF upload and persistent indexing
- Skip re-indexing identical files (SHA-256)
- Retrieval-augmented Q&A with filename and page citations
- Hierarchical summarization for long documents
- Key-point extraction
- Potential risk analysis (not legal advice)
- Two-document comparison
- Structured outputs via Pydantic
- Local vector storage (no cloud database)

## Architecture

PDF → Loader (text + page metadata) → Chunking → Embeddings → Persistent FAISS store → Retriever (`TOP_K`) → LLM → Pydantic structured output → Streamlit UI

Normal questions never send the full PDF. Only retrieved chunks are passed to the model. Long summaries use map-reduce over chunk groups. Summaries, risks, and comparisons are cached by document hash.

## Tech stack

- Python 3.11+
- Streamlit
- LangChain (`langchain-openai`, `langchain-text-splitters`, `langchain-core`)
- FAISS (local persistent index)
- PyPDF
- Pydantic
- python-dotenv
- OpenAI-compatible LLM and embeddings

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## `.env` setup

Copy `.env.example` to `.env` and fill in values:

```
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=4
```

Optional:

```
OPENAI_BASE_URL=https://api.openai.com/v1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

Never commit `.env`. Secrets are not hardcoded.

## Running the app

```powershell
streamlit run app.py
```

1. Upload PDFs in the sidebar and click **Index documents**.
2. Select a document for summarize / key points / risk analysis.
3. Use **Chat** for Q&A (selected document or all documents).
4. Use **Compare Documents** after two files are indexed.

## RAG explanation

1. Each PDF page is loaded as text with `source` and `page` metadata.
2. Text is split into overlapping chunks.
3. Chunks are embedded and stored on disk under `data/faiss`.
4. A question is embedded and the closest `TOP_K` chunks are retrieved.
5. The LLM must answer only from those chunks.
6. If the answer is not present, the app returns: `I could not find this information in the uploaded documents.`
7. Citations show the real filename and page from chunk metadata.

## Example questions

- What is the termination notice period?
- When is payment due and what late fees apply?
- Does this agreement auto-renew?
- Who is responsible for insurance?
- What dates or deadlines are listed?

## Troubleshooting

| Issue | What to do |
| --- | --- |
| Missing API key | Copy `.env.example` to `.env` and set `OPENAI_API_KEY` |
| API errors | Check key, model name, and optional `OPENAI_BASE_URL` |
| Invalid PDF | Re-export the file as a standard PDF |
| Empty / scanned PDF | Text extraction failed; OCR is not included |
| Duplicate file skipped | Same SHA-256 is already in the index |
| No relevant answer | The retrieved chunks did not contain the fact |
| Vector DB error | Delete `data/faiss` and `data/registry.json`, then re-index |
| Large document | Summaries run hierarchically; indexing still embeds each chunk once |

## Limitations

- No OCR for image-only or scanned PDFs
- Comparison uses retrieved and sampled chunks, not a guaranteed full legal redline
- Risk analysis is heuristic and is **not legal advice**
- Quality depends on the configured OpenAI-compatible models
- FAISS metadata filtering is applied after similarity search
- LLM/embedding calls require a valid API key and were not exercised in tests if credentials are absent

## Tests

```powershell
python -m compileall .
pytest
```

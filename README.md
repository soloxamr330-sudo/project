# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

# 📄 AI Document Intelligence System

A Streamlit application that lets you upload PDF documents and then ask questions, summarize, analyze potential risks, and compare two documents using a Retrieval-Augmented Generation (RAG) pipeline. Answers include filename and page citations. The language model can run locally through **Ollama** (default) or through an **OpenAI-compatible** API.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Islam Amr Hussien                    |
| Project Name     | AI Document Intelligence System      |
| GitHub Username  | soloxamr330-sudo                     |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en) |

---

# 📖 Project Overview

Reading long contracts and reports to find a single clause, a deadline, or a change between two versions is slow and error-prone. This project provides a document assistant that works on your own PDFs.

**How it works**

1. PDFs are loaded page by page with `pypdf`, keeping the filename and page number as metadata.
2. The text is split into overlapping chunks and embedded with the configured embedding model.
3. Embeddings are stored in a persistent local **FAISS** index, with a JSON registry of indexed files.
4. For a question, the most similar chunks (`TOP_K`) are retrieved and sent to the LLM together with a prompt that tells it to answer only from that context.
5. The result is returned as a validated **Pydantic** object and shown in the Streamlit UI with source citations.

**What users can do**

- Upload and index one or more PDFs.
- Chat with a selected document or with all indexed documents.
- Generate a summary and key points for a document.
- Run a potential-risk analysis (not legal advice).
- Compare two indexed documents and list the differences.

The system is oriented toward contract-style documents (the prompts look for payment terms, termination, renewal, liability, deadlines, and similar clauses), but it works on any text-based PDF.

---

# ✨ Features

- **Multi-PDF upload and persistent indexing** – the FAISS index and document registry are saved under `data/` and reloaded on the next run.
- **Duplicate detection** – files are hashed with SHA-256; an identical file that is already indexed is skipped.
- **Page-aware PDF loading** – each page keeps `source` (filename), `page`, and `file_hash` metadata; files over 40 MB, empty files, invalid PDFs, and PDFs with little or no extractable text are rejected with a clear error. Encrypted PDFs are opened only if they accept an empty password.
- **Chunking** – `RecursiveCharacterTextSplitter` with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP` (defaults 1000 / 200).
- **Semantic retrieval** – cosine-similarity search over a FAISS `IndexFlatIP` with L2-normalized vectors; results can be limited to one selected document.
- **Question answering with citations** – answers are grounded in retrieved chunks; citations that do not match retrieved content are dropped, and if nothing relevant is found the app returns: *"I could not find this information in the uploaded documents."* A confidence level (`low` / `medium` / `high`) is shown.
- **Summarization and key points** – produces an executive summary, key points, important dates, and potential risks for a selected document.
- **Risk analysis** – structured list of potential risks with category, severity (`LOW` / `MEDIUM` / `HIGH`), explanation, and sources, plus an overall severity and a "not legal advice" disclaimer. Results are cached per document hash.
- **Two-document comparison** – structured differences (`added` / `removed` / `modified`) with each document's value and the likely impact. Results are cached per document pair.
- **Structured outputs** – Q&A, risk analysis, and comparison use Pydantic schemas via LangChain's `with_structured_output`, with a JSON-parsing fallback.
- **Switchable LLM provider** – Ollama or OpenAI, selected with `LLM_PROVIDER`.
- **Streamlit interface** – sidebar for upload and document selection, with tabs for Chat, Summarize, Key Points, Risk Analysis, and Compare Documents.
- **Automated tests** – `pytest` tests for loading, chunking, retrieval, models, and citation grounding (using a fake embedding class, so no model is required).

---

# 🛠️ Technologies Used

| Area | Technology |
| ---- | ---------- |
| Language | Python |
| UI | [Streamlit](https://streamlit.io/) |
| LLM orchestration | LangChain (`langchain`, `langchain-core`, `langchain-text-splitters`) |
| LLM / embedding providers | Ollama via `langchain-ollama` (default), OpenAI-compatible via `langchain-openai` |
| Default models | `llama3.2:latest` (LLM), `nomic-embed-text:latest` (embeddings) |
| Vector search | FAISS (`faiss-cpu`) with NumPy |
| PDF parsing | `pypdf` |
| Data validation | Pydantic v2 |
| Configuration | `python-dotenv` (`.env` file) |
| Testing | `pytest` |

Dependency versions are the minimum versions listed in `requirements.txt`.

---

# ⚙️ Installation

### 1. Clone the repository and enter the project folder

```bash
git clone https://github.com/soloxamr330-sudo/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install langchain-ollama
```

> `langchain-ollama` is imported by `src/embeddings.py` when `LLM_PROVIDER=ollama` (the default) but is not listed in `requirements.txt`, so it is installed separately.

### 4. Choose a provider

**Option A – Ollama (default, runs locally, no API key)**

Install [Ollama](https://ollama.com/), then pull the default models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Option B – OpenAI-compatible API**

Set `LLM_PROVIDER=openai` and provide an API key (see below).

### 5. Configure environment variables

Copy the example file and edit it:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LLM_MODEL` | `llama3.2:latest` | Chat model name |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `OPENAI_API_KEY` | empty | Required only when `LLM_PROVIDER=openai` |
| `OPENAI_BASE_URL` | empty | Optional custom OpenAI-compatible endpoint |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

Example `.env` for the default local setup:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

Example `.env` for OpenAI (the model names must be set explicitly, because the built-in defaults are Ollama model names):

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_API_KEY
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

> 🔒 Never commit your `.env` file. It is already listed in `.gitignore`.

---

# 🚀 Usage

### Start the app

Make sure Ollama is running (if you use it), then:

```bash
streamlit run app.py
```

Streamlit prints the local URL in the terminal (by default `http://localhost:8501`).

### Workflow

1. **Upload and index** – in the sidebar, upload one or more PDFs and click **Index documents**. The app reports the number of pages and chunks for each file, or notes that a duplicate was skipped.
2. **Select a document** – pick a file under **Document for operations** in the sidebar.
3. **Use the tabs**

| Tab | What it does | Output |
| --- | ------------ | ------ |
| **Chat** | Ask a question about the selected document or all indexed documents | Answer, confidence, and expandable sources (filename, page, excerpt) |
| **Summarize** | Click **Generate summary** | Executive summary, key points, important dates, potential risks, sources |
| **Key Points** | Click **Extract key points** | Bullet list of key points and sources |
| **Risk Analysis** | Click **Analyze risks** | Overall severity and a list of risks with category, severity, explanation, sources |
| **Compare Documents** | Choose Document A and B (at least two indexed PDFs), click **Compare** | Summary and per-category differences side by side, with impact and sources |

### Example questions

- What is the termination notice period?
- When is payment due and what late fees apply?
- Does this agreement auto-renew?
- Who is responsible for insurance?
- What dates or deadlines are listed?

### Run the tests

```bash
pytest
```

### Project structure

```text
.
├── app.py                 # Streamlit UI
├── requirements.txt
├── pytest.ini
├── .env.example
├── src/
│   ├── config.py          # Settings and environment variables
│   ├── embeddings.py      # LLM and embedding factories (Ollama / OpenAI)
│   ├── document_loader.py # PDF loading with page metadata and validation
│   ├── text_splitter.py   # Chunking
│   ├── indexer.py         # PDF -> chunks -> embeddings -> store
│   ├── vector_store.py    # FAISS store, document registry, persistence
│   ├── retriever.py       # Similarity retrieval with optional file filter
│   ├── rag_chain.py       # Question answering and citation grounding
│   ├── summarizer.py      # Summary and key-point extraction
│   ├── risk_analyzer.py   # Potential risk analysis
│   ├── comparator.py      # Two-document comparison
│   ├── output_models.py   # Pydantic schemas
│   └── utils.py           # Hashing, caching, citations, structured LLM calls
├── tests/                 # pytest tests
└── data/                  # Generated at runtime: uploads, faiss, cache, registry.json
```

---

# 📸 Demo

No screenshots, GIFs, or demo videos are included in the project files yet.

To add them later, place the images in a folder such as `docs/screenshots/` and reference them here, for example:

```markdown
![Chat tab](docs/screenshots/chat.png)
```

---

# 📈 Results

The project does not include an evaluation script or formal metrics (accuracy, retrieval scores, or latency), so no numerical results are reported.

**Demonstrated functionality**

- The repository includes two sample PDFs (`AI_Document_Test_Contract_Original.pdf` and `AI_Document_Test_Contract_Amended.pdf`) that were indexed, with 6 chunks each recorded in `data/registry.json`.
- A saved comparison of these two files (in `data/cache/`) identified changed dates: the UAT target date moved from 20 November 2026 to 27 November 2026, and the production launch date from 1 December 2026 to 15 December 2026. The same saved output also contains a questionable "Payment" entry, which shows that LLM output should be reviewed against the source documents.
- The `tests/` folder checks PDF loading and page metadata, rejection of empty, invalid, and scanned-like PDFs, chunk metadata, retrieval relevance and per-document filtering, Pydantic model validation, and citation grounding.

---

# ⚠️ Limitations

- No OCR: scanned or image-only PDFs are rejected.
- Summarization sends all chunks of the selected document to the model in a single call, so very long documents may exceed the model's context window.
- Comparison and risk analysis use a subset of each document's chunks (retrieved plus the first few), not the full text.
- Risk analysis and comparison are informational and are **not legal advice**.
- Output quality depends on the configured model; small local models may produce imperfect results.
- The FAISS metadata is stored with `pickle`; only load index files you created yourself.

---

# 🔮 Future Improvements

These are ideas for future work and are **not** implemented yet.

* Add `langchain-ollama` to `requirements.txt`.
* Add map-reduce (hierarchical) summarization so long documents fit the model context.
* Add OCR support for scanned PDFs.
* Support more file types such as DOCX and TXT.
* Add an evaluation script with a labeled question set to measure retrieval and answer quality.
* Run risk analysis and comparison over the full document instead of a sampled subset.
* Add page-level highlighting or a document viewer next to the cited sources.
* Add a Dockerfile for easier deployment.

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.

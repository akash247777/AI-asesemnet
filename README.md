# AI Pipeline: Weather + PDF RAG (LangChain, LangGraph, Qdrant, Streamlit)

An end-to-end, agentic mini-app that:

- Fetches real-time weather from OpenWeatherMap
- Answers questions about your PDFs using RAG (Retrieval-Augmented Generation)
- Routes between Weather vs RAG using a LangGraph state machine
- Uses Qdrant as a vector database for embeddings and retrieval
- Supports multiple LLM/embeddings providers with sensible fallbacks
- Offers a Streamlit chat UI and CLI utilities

# Key Technologies

- **LangChain** (chains, prompts, vectorstores)
- **LangGraph** (router graph and node composition)
- **Qdrant** (vector database; local via Docker or Cloud)
- **Hugging Face Inference / Google GenAI** (LLM + embeddings providers)
- **Streamlit** (UI)
- **LangSmith** (optional tracing/eval)

# Project Structure

```

├── src
│   ├── app.py                 # Streamlit UI (upload PDF, ask questions)
│   ├── config.py              # Settings from environment (.env)
│   ├── embeddings.py          # Embeddings factory with HF/Google/local fallbacks
│   ├── graph.py               # LangGraph router → weather or rag nodes
│   ├── llm.py                 # LLM factory: Google (Gemini) or HF Inference (nscale) or local
│   ├── rag.py                 # PDF ingest, split, retrieve, answer
│   ├── vectorstore.py         # Qdrant client, collection management, dimension checks
│   └── weather.py             # OpenWeather fetch + LLM summary (with non-LLM fallback)
├── scripts
│   ├── ingest_pdf.py          # CLI: ingest a PDF into Qdrant
│   └── evaluate_langsmith.py  # CLI: quick evaluation runner (logs to LangSmith)
├── tests                      # Minimal tests (some require API keys)
│   ├── test_graph.py
│   ├── test_rag.py
│   └── test_weather.py
├── data
│   ├── README.md              # Put your PDFs here (e.g., sample.pdf)
│   └── _uploaded.pdf          # Temporary file created by UI uploads
├── requirements.txt
└── README.md
```

# Prerequisites

- Python 3.10+
- Windows PowerShell or a POSIX shell
- Either:
  - Docker Desktop (to run Qdrant locally), or
  - A Qdrant Cloud account (recommended)
- API keys you plan to use:
  - OpenWeatherMap (`OPENWEATHER_API_KEY`)
  - Either Hugging Face or Google GenAI for LLM/embeddings
  - LangSmith (optional)

# Setup

# 1) Create a virtual environment and install dependencies

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

# 2 Configure environment variables

Create a `.env` file in the project root with the variables you intend to use. Example:

```env
# --- LLM (choose provider) ---
# Use Google Gemini (recommended when GOOGLE_API_KEY is available)
LLM_PROVIDER=google
GOOGLE_API_KEY=YOUR_GOOGLE_KEY
GOOGLE_LLM_MODEL=gemini-1.5-flash

# Or use Hugging Face Inference (nscale provider)
# HF_TOKEN=YOUR_HF_TOKEN
# HUGGINGFACE_API_KEY=YOUR_HF_TOKEN
HF_LLM_MODEL=Qwen/Qwen3-4B-Thinking-2507
HF_PROVIDER=nscale

# Force a local CPU model instead (fallback path)
# LLM_BACKEND=local
# LOCAL_LLM_MODEL=google/flan-t5-small

# --- Embeddings ---
# Preferred: HF Inference embeddings via HUGGINGFACE_API_KEY
# Optional: Google embeddings when EMBEDDINGS_PROVIDER=google
# EMBEDDINGS_PROVIDER=google
# GOOGLE_EMBEDDINGS_MODEL=text-embedding-004
# Fallback local embeddings (sentence-transformers)
# EMBEDDINGS_BACKEND=local
# EMBEDDINGS_DEVICE=cpu

# --- Weather ---
OPENWEATHER_API_KEY=YOUR_OPENWEATHER_KEY

# --- Qdrant ---
# For local Docker: QDRANT_URL=http://localhost:6333
# For Cloud: set both URL and API key
QDRANT_URL=YOUR_QDRANT_URL
QDRANT_API_KEY=YOUR_QDRANT_API_KEY
QDRANT_COLLECTION=pdf_documents
# If dimensions mismatch, auto drop & recreate the collection
# QDRANT_AUTO_RECREATE=true

# --- UI ---
# Show retrieval sources in Streamlit when using RAG
# SHOW_SOURCES=true

# --- LangSmith / LangChain tracing (optional) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGSMITH_KEY
LANGCHAIN_PROJECT=ai-pipeline-assignment
# Legacy aliases also supported by config
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=YOUR_LANGSMITH_KEY
LANGSMITH_PROJECT=ai-pipeline-assignment
```

Important:

- Do not commit real secrets. Rotate any keys that may have been committed.
- If you prefer not to use a `.env` file, export variables directly in your shell.

# 3 Run Qdrant

Option A — Local via Docker:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Option B — Qdrant Cloud: set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`.

# Running the App

Start the Streamlit UI from the project root:

```powershell
streamlit run src/app.py
```

In the sidebar:

- Upload a PDF to index. The app writes a temp file to `data/_uploaded.pdf` and ingests it.
- Click “Reset graph” to rebuild the LangGraph router.

In the main pane, type questions like:

- "What's the weather in Paris?"
- "Summarize section 2 of the PDF."

Enable `SHOW_SOURCES=true` to display RAG source metadata under answers.

# CLI Utilities

# Ingest a PDF into Qdrant

```powershell
python scripts/ingest_pdf.py --pdf .\data\your.pdf --collection pdf_documents
```

# Quick evaluation with LangSmith logging (optional)

```powershell
python scripts/evaluate_langsmith.py
```

# Tests

Run all tests:

```powershell
pytest -q
```

Notes:

- Some tests are skipped unless required keys are set. For example, weather tests require `OPENWEATHER_API_KEY`. Graph/RAG tests require an LLM key (`HUGGINGFACE_API_KEY` or `GOOGLE_API_KEY`) and a reachable Qdrant.
- `tests/test_rag.py` expects a `data/sample.pdf`. Provide one or skip the test.

## Architecture and Behavior

### Router and Nodes (`src/graph.py`)

- Classifies the input via simple keyword heuristics into `weather` or `rag`.
- Weather node:
  - Extracts a city from the question (naive heuristic; defaults to "London").
  - Calls `fetch_weather` → `summarize_weather`.
  - Stores the summary text in Qdrant with metadata `{type: "weather", city}`.
- RAG node:
  - Retrieves top-k chunks from Qdrant and generates an answer using the configured LLM.

### LLMs (`src/llm.py`)

- Providers:
  - Google Gemini via `ChatGoogleGenerativeAI` when `LLM_PROVIDER=google` or a `GOOGLE_API_KEY` is present.
  - Hugging Face Inference via custom `HFNScaleChat` (provider `nscale`) using `HF_TOKEN`/`HUGGINGFACE_API_KEY`.
  - Local CPU fallback via a small `flan-t5` pipeline when `LLM_BACKEND=local`.
- Prompts are built with `build_answer_prompt`. Outputs are normalized with `format_output`.

### Embeddings (`src/embeddings.py`)

- Preference order:
  1) Google embeddings when `EMBEDDINGS_PROVIDER=google`.
  2) Hugging Face Inference embeddings (no local Torch) when `HUGGINGFACE_API_KEY` is available.
  3) Local sentence-transformers (`HuggingFaceEmbeddings`) when remote is unavailable.
- Default embedding model is `BAAI/bge-small-en-v1.5` (dimension 384).

### Vector Store (`src/vectorstore.py`)

- Creates or verifies Qdrant collections. Probes embedding dimension and checks for mismatches.
- Set `QDRANT_AUTO_RECREATE=true` to drop & recreate collections automatically on dimension mismatch.

### Weather (`src/weather.py`)

- Fetches from OpenWeatherMap; retries with a simplified city token on 404s.
- Summarizes with the active LLM when possible; otherwise returns a deterministic summary.

### RAG (`src/rag.py`)

- Loads PDFs via `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter`.
- Uses `get_retriever` to perform similarity search; answers with the active LLM and includes source metadata.

### Streamlit UI (`src/app.py`)

- Ensures proper event loop handling when running in Streamlit.
- Optional sources display controlled by `SHOW_SOURCES`.

## Troubleshooting

- Qdrant connection errors: verify `QDRANT_URL`/`QDRANT_API_KEY` and that the service is reachable. For local Docker, visit `http://localhost:6333/collections`.
- Collection dimension mismatch: either switch `QDRANT_COLLECTION` or set `QDRANT_AUTO_RECREATE=true`.
- Missing keys: set `OPENWEATHER_API_KEY` and an LLM provider key (`GOOGLE_API_KEY` or `HF_TOKEN`/`HUGGINGFACE_API_KEY`).
- Event loop errors in Streamlit: the code creates a loop when needed and falls back to async `.ainvoke()`/`.aadd_*()` where applicable.
- PDF ingestion: large PDFs can be slow; try a small sample first. Ensure your file is in `data/`.

## License

Add your preferred license here.

---

If anything fails to run, double-check environment variables and that Qdrant is reachable.


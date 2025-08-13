## AI Pipeline Assignment (LangChain + LangGraph + LangSmith + Qdrant + Streamlit)

This project demonstrates a simple agentic pipeline that can:

- Fetch real-time weather using the OpenWeatherMap API
- Answer questions from a PDF using RAG (Retrieval-Augmented Generation)
- Decide (via LangGraph) whether to call weather or RAG
- Process responses with an LLM (Qwen/Qwen3-4B-Thinking-2507 via Hugging Face Inference)
- Create embeddings and store them in Qdrant (vector DB)
- Retrieve and summarize info using RAG
- Log and evaluate runs with LangSmith
- Provide a Streamlit chat UI

### 1) Prerequisites

- Python 3.10+
- Docker Desktop (for local Qdrant) or a Qdrant Cloud account
- Accounts/keys:
  - Hugging Face Inference API key
  - OpenWeatherMap API key
  - LangSmith API key (optional but recommended)

### 2) Quick Start

From PowerShell in the project root:

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Create a `.env` from the template and fill in your keys:

```powershell
copy .env.example .env
# Edit .env and fill the placeholders
```

Run Qdrant locally (Docker) or set `QDRANT_URL`/`QDRANT_API_KEY` for Qdrant Cloud:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Option A: Ingest a PDF from CLI:

```powershell
python scripts/ingest_pdf.py --pdf .\data\your.pdf --collection pdf_documents
```

Option B: Upload a PDF directly from the Streamlit UI (it will ingest automatically).

Run the Streamlit app (ensure PowerShell's working directory is the project root so `src` is importable):

```powershell
streamlit run src/app.py
```

Run tests:

```powershell
pytest -q
```

### 3) What to Try in the UI

- “What’s the weather in Paris?” → Router chooses weather API → LLM summarizes → Stored in Qdrant
- “Summarize section 2 of the PDF” → Router chooses RAG → Retriever returns chunks → LLM answers with citations

### 4) Project Structure

```
.
├── src
│   ├── app.py                 # Streamlit UI
│   ├── config.py              # Env + settings
│   ├── embeddings.py          # HF embeddings factory
│   ├── graph.py               # LangGraph: route → weather/RAG → respond
│   ├── llm.py                 # Qwen HF endpoint wrapper
│   ├── rag.py                 # PDF ingest + RAG query
│   ├── vectorstore.py         # Qdrant helpers
│   └── weather.py             # Weather API + summarization
├── scripts
│   └── ingest_pdf.py          # CLI PDF ingestion
├── tests
│   ├── test_graph.py          # Router & end-to-end sanity
│   ├── test_rag.py            # Retrieval logic (integration)
│   └── test_weather.py        # API handling
├── data
│   └── README.md              # Put your PDFs here
├── .env.example
├── requirements.txt
└── README.md
```

### 5) Underlying Concepts (Short)

- **Embeddings**: Map text to numeric vectors. Similar meaning → similar vectors. We use `BAAI/bge-small-en-v1.5` via Hugging Face Inference API (no local torch needed) to embed both PDF chunks and weather summaries.
- **Vector DB (Qdrant)**: Stores vectors with payloads (metadata). We upsert embeddings and later perform similarity search (cosine distance) to retrieve relevant chunks.
- **RAG**: Retrieve-Augment-Generate. Retrieve top-k relevant chunks from Qdrant → feed to the LLM → generate grounded answers with citations.
- **LangGraph**: Declarative control flow around LLM calls. We build a small state machine: route → weather or rag → respond.
- **LangChain**: Abstractions for LLMs, prompts, tools, retrievers, and chains.
- **LangSmith**: Observability and evaluation. With env vars set, all runs are traced. You can also set up datasets and evaluators for grading outputs.

### 6) Configuration

Copy `.env.example` to `.env` and fill values. Important ones:

- `HUGGINGFACE_API_KEY`: Required for Qwen endpoint and embeddings
- `OPENWEATHER_API_KEY`: Required for weather API
- `QDRANT_URL`, `QDRANT_API_KEY`: Required for Qdrant Cloud, or use local Docker at `http://localhost:6333`
- `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`: Optional for tracing/eval

### 7) Evaluating with LangSmith (Optional)

With LangSmith env vars set (`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`), runs are auto-logged. You can create a small dataset in LangSmith (e.g., a few weather and PDF questions) and run ad-hoc evaluations there. A simple CLI runner is included:

```powershell
python scripts/evaluate_langsmith.py
```

### 8) Notes on Qwen/Qwen3-4B-Thinking-2507

- Accessed via Hugging Face Inference endpoint through `HuggingFaceEndpoint` + `ChatHuggingFace` in LangChain
- We set modest `max_new_tokens` and `temperature` for concise, grounded outputs
- Prompts ask the model not to disclose hidden chain-of-thought and to give concise rationales

---

If anything fails to run, double-check environment variables and that Qdrant is reachable (`http://localhost:6333/collections`).



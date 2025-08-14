# AI Pipeline Assignment

**Tech Stack:** LangChain + LangGraph + LangSmith + Qdrant + Streamlit

This project demonstrates a simple agentic pipeline capable of:

- Fetching real-time weather data using the OpenWeatherMap API
- Answering questions from PDFs using Retrieval-Augmented Generation (RAG)
- Dynamically deciding (via LangGraph) whether to use the weather API or RAG
- Processing responses with an LLM (Qwen/Qwen3-4B-Thinking-2507 via Hugging Face Inference)
- Creating and storing embeddings in Qdrant (vector database)
- Retrieving and summarizing information using RAG
- Logging and evaluating runs with LangSmith
- Providing a user-friendly Streamlit chat interface

---

## 1. Prerequisites

- **Python 3.10+**
- **Docker Desktop** (for local Qdrant) or a **Qdrant Cloud** account
- API keys for:
  - Hugging Face Inference
  - OpenWeatherMap
  - LangSmith (optional, but recommended for observability)

---

## 2. Quick Start

### Setup

From PowerShell, in the project root:

```powershell
# Create and activate a virtual environment
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
Environment Configuration
Copy the example environment file and fill in your API keys:
 Copycopy .env.example .env
# Edit .env and replace the placeholders with your actual keys
Run Qdrant
Start Qdrant locally using Docker:
 Copydocker run -p 6333:6333 -p 6334:6334 qdrant/qdrant\:latest
Or, set QDRANT_URL and QDRANT_API_KEY in your .env file for Qdrant Cloud.
Ingest a PDF
Option A: Ingest a PDF from the command line:
 Copypython scripts/ingest_pdf.py --pdf .\data\your.pdf --collection pdf_documents
Option B: Upload a PDF directly via the Streamlit UI (automatic ingestion).
Start the Streamlit App
Ensure your working directory is the project root so src is importable:
 Copystreamlit run src/app.py
Run Tests
 Copypytest -q

3. Example UI Interactions

"What's the weather in Paris?"
The router selects the weather API, the LLM summarizes the response, and the result is stored in Qdrant.
"Summarize section 2 of the PDF"
The router selects RAG, retrieves relevant chunks, and the LLM answers with citations.


4. Project Structure
 Copy.
├── src
│   ├── app.py                 # Streamlit UI
│   ├── config.py              # Environment and settings
│   ├── embeddings.py          # Hugging Face embeddings factory
│   ├── graph.py               # LangGraph: routing, weather/RAG, response
│   ├── llm.py                 # Qwen Hugging Face endpoint wrapper
│   ├── rag.py                 # PDF ingestion and RAG query logic
│   ├── vectorstore.py         # Qdrant utilities
│   └── weather.py             # Weather API and summarization
├── scripts
│   └── ingest_pdf.py          # CLI PDF ingestion script
├── tests
│   ├── test_graph.py          # Router and end-to-end tests
│   ├── test_rag.py            # Retrieval logic (integration tests)
│   └── test_weather.py        # Weather API handling tests
├── data
│   └── README.md              # Place your PDFs here
├── .env.example
├── requirements.txt
└── README.md

5. Key Concepts


Embeddings:
Text is mapped to numeric vectors. Similar text produces similar vectors. This project uses BAAI/bge-small-en-v1.5 via Hugging Face Inference API (no local PyTorch required) to embed both PDF chunks and weather summaries.


Vector Database (Qdrant):
Stores vectors with metadata. Embeddings are upserted and later retrieved using similarity search (cosine distance).


Retrieval-Augmented Generation (RAG):
Relevant chunks are retrieved from Qdrant and fed to the LLM to generate grounded answers with citations.


LangGraph:
Provides declarative control flow around LLM calls. The pipeline routes queries to either the weather API or RAG, then responds.


LangChain:
Offers abstractions for LLMs, prompts, tools, retrievers, and chains.


LangSmith:
Enables observability and evaluation. With environment variables set, all runs are traced. You can create datasets and evaluators for output grading.



6. Configuration
Copy .env.example to .env and set the following required variables:

HUGGINGFACE_API_KEY (for Qwen endpoint and embeddings)
OPENWEATHER_API_KEY (for weather API)
QDRANT_URL, QDRANT_API_KEY (for Qdrant Cloud; use http://localhost:6333 for local Docker)

Optional variables for LangSmith:

LANGSMITH_API_KEY
LANGSMITH_PROJECT
LANGSMITH_TRACING


7. Evaluating with LangSmith (Optional)
With LangSmith environment variables set (LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT), all runs are automatically logged. You can create a dataset in LangSmith and run evaluations using:
 Copypython scripts/evaluate_langsmith.py

8. Notes on Qwen/Qwen3-4B-Thinking-2507

Accessed via Hugging Face Inference endpoint using HuggingFaceEndpoint and ChatHuggingFace in LangChain.
Configured with modest max_new_tokens and temperature for concise, grounded outputs.
Prompts are designed to prevent disclosure of hidden chain-of-thought and encourage concise rationales.


Troubleshooting
If you encounter issues, verify:

All environment variables are set correctly.
Qdrant is reachable at http://localhost:6333/collections.

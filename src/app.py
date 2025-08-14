import os
import sys
import io
from typing import Dict

import streamlit as st

# Ensure project root is on sys.path so `src` is importable when running `streamlit run src/app.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph import build_graph
from src.rag import ingest_pdf_into_qdrant


st.set_page_config(page_title="AI Pipeline: Weather + RAG", page_icon="⛅")
st.title("AI Pipeline: Weather + RAG (LangGraph + Qdrant + Qwen)")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()


with st.sidebar:
    st.header("PDF Ingestion")
    if st.button("Reset graph"):
        if "graph" in st.session_state:
            del st.session_state["graph"]
        st.session_state.graph = build_graph()
        st.success("Graph reset.")
    uploaded = st.file_uploader("Upload a PDF to index", type=["pdf"]) 
    if uploaded is not None:
        bytes_data = uploaded.read()
        tmp_path = "./data/_uploaded.pdf"
        with open(tmp_path, "wb") as f:
            f.write(bytes_data)
        try:
            res = ingest_pdf_into_qdrant(tmp_path)
            st.success(f"Ingested {res['num_chunks']} chunks into '{res['collection']}'")
        except Exception as exc:
            st.error(
                "Failed to ingest PDF into Qdrant Cloud. Verify QDRANT_URL and QDRANT_API_KEY in your environment.\n\n"
                f"Details: {exc}"
            )

st.write("Ask about weather (e.g., 'What's the weather in Paris?') or your PDF (e.g., 'Summarize section 2').")

user_input = st.text_input("Your question")
if st.button("Ask") and user_input.strip():
    def _invoke_graph_safely(graph, payload: Dict[str, str]):
        try:
            return graph.invoke(payload)
        except RuntimeError as exc:
            # Fallback for environments without a running event loop (e.g., Streamlit ScriptRunner)
            if "There is no current event loop" in str(exc):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(graph.ainvoke(payload))
            raise

    result: Dict | None = _invoke_graph_safely(st.session_state.graph, {"question": user_input.strip()})
    if not isinstance(result, dict):
        st.error("Unexpected empty result from graph. Please try again.")
    else:
        st.subheader("Answer")
        st.write(result.get("answer", ""))
        show_sources = os.getenv("SHOW_SOURCES", "").strip().lower() in ("1", "true", "yes", "on")
        if show_sources and result.get("route") == "rag" and result.get("sources"):
            st.caption("Sources (metadata):")
            st.json(result["sources"])



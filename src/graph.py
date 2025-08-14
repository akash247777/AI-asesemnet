from typing import Literal, Dict, Any, TypedDict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda

try:
    from src.rag import rag_answer
    from src.weather import fetch_weather, summarize_weather
    from src.vectorstore import get_vectorstore
    from src.embeddings import build_embeddings
except Exception:  # fallback when running as a script from src/
    from rag import rag_answer
    from weather import fetch_weather, summarize_weather
    from vectorstore import get_vectorstore
    from embeddings import build_embeddings


class RouterState(TypedDict, total=False):
    question: str
    answer: str
    route: str
    sources: List[Dict[str, Any]]


def _extract_question(state: RouterState) -> str:
    # Robustly handle different state shapes that may occur with LangGraph
    if isinstance(state, dict):
        if "question" in state and isinstance(state["question"], str):
            return state["question"]
        # Sometimes input may be nested
        input_val = state.get("input")
        if isinstance(input_val, dict) and isinstance(input_val.get("question"), str):
            return input_val["question"]
        if isinstance(input_val, str):
            return input_val
    # Fallback: stringify whole state
    return str(state)


def classify_route(state: RouterState) -> Literal["weather", "rag"]:
    user_input: str = _extract_question(state).lower()
    # Simple heuristic routing; could be replaced by LLM classifier
    keywords = ["weather", "temperature", "forecast", "rain", "wind", "humidity"]
    route = "weather" if any(k in user_input for k in keywords) else "rag"
    try:
        print(f"[router] input='{user_input[:60]}' -> route='{route}'")
    except Exception:
        pass
    return route


def weather_node(state: RouterState) -> RouterState:
    try:
        # Ensure an event loop exists for any async vectorstore/LLM paths in Streamlit's thread
        try:
            import asyncio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except Exception:
            pass

        question = _extract_question(state)
        # Try to extract a city, naive approach: last token or after 'in'
        lower = question.lower()
        city = ""
        if " in " in lower:
            city = question.split(" in ")[-1].strip(" ?!.,")
        if not city:
            city = "London"  # default fallback

        raw = fetch_weather(city)
        summary = summarize_weather(raw, city)

        # persist weather summary into vector db (demonstrates embeddings storage)
        embeddings = build_embeddings()
        vs = get_vectorstore(embeddings)
        try:
            vs.add_texts([summary], metadatas=[{"type": "weather", "city": city}])
        except RuntimeError as exc:
            msg = str(exc)
            if "There is no current event loop" in msg or "no running event loop" in msg:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(vs.aadd_texts([summary], metadatas=[{"type": "weather", "city": city}]))
            else:
                raise
        try:
            print(f"[weather_node] city='{city}', summary_len={len(summary)}")
        except Exception:
            pass

        return {
            "answer": summary,
            "route": "weather",
        }
    except Exception as exc:
        return {
            "answer": f"Weather lookup unavailable right now. Details: {exc}",
            "route": "weather",
        }


def rag_node(state: RouterState) -> RouterState:
    try:
        question = _extract_question(state)
        res = rag_answer(question)
        try:
            print(f"[rag_node] answer_len={len(res.get('answer',''))}, sources={len(res.get('sources',[]))}")
        except Exception:
            pass
        return {
            "answer": res["answer"],
            "sources": res.get("sources", []),
            "route": "rag",
        }
    except Exception as exc:
        return {
            "answer": f"RAG is unavailable right now. Details: {exc}",
            "sources": [],
            "route": "rag",
        }


def build_graph():
    g = StateGraph(RouterState)
    g.add_node("weather", weather_node)
    g.add_node("rag", rag_node)

    g.add_conditional_edges(START, classify_route, {"weather": "weather", "rag": "rag"})
    g.add_edge("weather", END)
    g.add_edge("rag", END)
    return g.compile()



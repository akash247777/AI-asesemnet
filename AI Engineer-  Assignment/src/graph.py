from typing import Literal, Dict, Any

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


class RouterState(dict):
    pass


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
    if any(k in user_input for k in keywords):
        return "weather"
    return "rag"


def weather_node(state: RouterState) -> RouterState:
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
    vs.add_texts([summary], metadatas=[{"type": "weather", "city": city}])

    state["answer"] = summary
    state["route"] = "weather"
    return state


def rag_node(state: RouterState) -> RouterState:
    question = _extract_question(state)
    res = rag_answer(question)
    state["answer"] = res["answer"]
    state["sources"] = res["sources"]
    state["route"] = "rag"
    return state


def build_graph():
    g = StateGraph(RouterState)
    g.add_node("weather", RunnableLambda(weather_node))
    g.add_node("rag", RunnableLambda(rag_node))

    g.add_conditional_edges(START, classify_route, {"weather": "weather", "rag": "rag"})
    g.add_edge("weather", END)
    g.add_edge("rag", END)
    return g.compile()



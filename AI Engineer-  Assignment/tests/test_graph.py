import os
import pytest

from src.graph import build_graph


@pytest.mark.skipif(
    not os.getenv("HUGGINGFACE_API_KEY") or not os.getenv("OPENWEATHER_API_KEY"),
    reason="Requires HUGGINGFACE_API_KEY and OPENWEATHER_API_KEY",
)
def test_router_weather():
    graph = build_graph()
    result = graph.invoke({"question": "What's the weather in London?"})
    assert result["route"] == "weather"
    assert isinstance(result["answer"], str)


@pytest.mark.skipif(
    not os.getenv("HUGGINGFACE_API_KEY"), reason="Requires HUGGINGFACE_API_KEY"
)
def test_router_rag():
    graph = build_graph()
    result = graph.invoke({"question": "Summarize section 1 of the PDF."})
    assert result["route"] == "rag"


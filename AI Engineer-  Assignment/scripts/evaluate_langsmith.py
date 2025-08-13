"""
Ad-hoc evaluation driver. With LangSmith tracing enabled (env vars),
all runs will be logged under the configured project.

Env needed:
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY=...
- LANGCHAIN_PROJECT=ai-pipeline-assignment (or custom)
"""

from typing import List, Dict

from src.graph import build_graph


EXAMPLES: List[Dict[str, str]] = [
    {"question": "What's the weather in Tokyo?"},
    {"question": "Summarize the introduction of the uploaded PDF."},
]


def main():
    graph = build_graph()
    for i, ex in enumerate(EXAMPLES, 1):
        result = graph.invoke(ex)
        print(f"Case {i} | route={result.get('route')}\n{result.get('answer')}\n")


if __name__ == "__main__":
    main()



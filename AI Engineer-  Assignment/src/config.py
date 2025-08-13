import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    hf_llm_model: str = os.getenv("HF_LLM_MODEL", "Qwen/Qwen3-4B-Thinking-2507")

    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "pdf_documents")

    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "ai-pipeline-assignment")


_CACHED_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is None:
        _CACHED_SETTINGS = Settings()
    return _CACHED_SETTINGS



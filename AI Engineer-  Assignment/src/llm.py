from typing import Any, Dict

from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

try:
    from src.config import get_settings
except Exception:
    from config import get_settings


def build_llm() -> BaseChatModel:
    settings = get_settings()
    if not settings.huggingface_api_key:
        raise ValueError("Missing HUGGINGFACE_API_KEY in environment")

    hf_llm = HuggingFaceEndpoint(
        repo_id=settings.hf_llm_model,
        temperature=0.3,
        max_new_tokens=512,
        huggingfacehub_api_token=settings.huggingface_api_key,
        timeout=120,
    )
    return ChatHuggingFace(llm=hf_llm)


def build_answer_prompt(system_instructions: str = "") -> ChatPromptTemplate:
    system = (
        system_instructions
        or "You are a helpful assistant. Answer concisely, cite sources when provided, and do not reveal chain-of-thought."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nRespond with a short, well-structured answer.",
            ),
        ]
    )


def format_output(generated: Any) -> str:
    # ChatHuggingFace returns AIMessage; get content as string
    try:
        return generated.content if hasattr(generated, "content") else str(generated)
    except Exception:
        return str(generated)



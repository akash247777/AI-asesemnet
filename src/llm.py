from typing import Any, Dict, List
import os

from langchain_huggingface import ChatHuggingFace
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration
from huggingface_hub import InferenceClient

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

try:
    from src.config import get_settings
except Exception:
    from config import get_settings


class HFNScaleChat(BaseChatModel):
    """LangChain ChatModel wrapper around Hugging Face InferenceClient chat.completions.

    Uses provider from HF_PROVIDER (default "nscale") and token from HF_TOKEN (preferred)
    or HUGGINGFACEHUB_API_TOKEN or HUGGING_FACE_HUB_TOKEN or HUGGINGFACE_API_KEY.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        provider: str = "nscale",
    ) -> None:
        super().__init__()
        self._model = model
        self._temperature = float(temperature)
        self._max_new_tokens = int(max_new_tokens)
        self._provider = provider
        self._client = InferenceClient(provider=provider, api_key=api_key)

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "hf_nscale"

    @property
    def _identifying_params(self) -> dict:  # type: ignore[override]
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_new_tokens": self._max_new_tokens,
            "provider": self._provider,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        converted: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                role = "user"
            content = getattr(m, "content", str(m))
            converted.append({"role": role, "content": content})
        return converted

    def _generate(
        self, messages: List[BaseMessage], stop: None = None, run_manager: None = None, **kwargs: Any
    ) -> ChatResult:
        hf_messages = self._convert_messages(messages)
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=hf_messages,
            temperature=self._temperature,
            max_tokens=self._max_new_tokens,
        )
        choice = completion.choices[0]
        msg_obj = getattr(choice, "message", choice)
        content = getattr(msg_obj, "content", str(msg_obj))
        ai_msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def build_llm() -> BaseChatModel:
    """Return a chat model that uses HF InferenceClient with provider nscale when remote.

    Remote is enforced when LLM_BACKEND=remote. If no token is available, raise a clear error.
    If LLM_BACKEND=local, build a small local model.
    """
    settings = get_settings()
    backend_override = os.getenv("LLM_BACKEND", "").strip().lower()
    provider_override = os.getenv("LLM_PROVIDER", "").strip().lower()

    if backend_override == "local":
        return _build_local_chat_llm()

    # Prefer Google if explicitly selected or GOOGLE_API_KEY is present
    google_key_env = os.getenv("GOOGLE_API_KEY") or getattr(settings, "google_api_key", "")
    if provider_override == "google" or (not provider_override and google_key_env):
        google_key = google_key_env
        if not google_key:
            raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai is not installed. Add it to requirements.txt and pip install.")
        google_model = os.getenv("GOOGLE_LLM_MODEL") or getattr(settings, "google_llm_model", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=google_model, api_key=google_key, temperature=0.3, max_output_tokens=512)

    # Remote path: prefer common HF token env vars, then fallback to HUGGINGFACE_API_KEY
    api_key = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or getattr(settings, "huggingface_api_key", "")
    )
    if not api_key:
        raise RuntimeError(
            "LLM_BACKEND=remote requires a provider API key. Set GOOGLE_API_KEY (preferred) or HF_TOKEN/HUGGINGFACE_API_KEY in .env"
        )

    model_id = getattr(settings, "hf_llm_model", "Qwen/Qwen3-4B-Thinking-2507")
    provider = os.getenv("HF_PROVIDER", "nscale").strip() or "nscale"
    try:
        return HFNScaleChat(
            model=model_id,
            api_key=api_key,
            temperature=0.3,
            max_new_tokens=512,
            provider=provider,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize HF InferenceClient (provider=nscale). "
            "Verify HF_TOKEN (or HUGGINGFACE_API_KEY) and HF_LLM_MODEL. Details: " + str(exc)
        )


def _build_local_chat_llm() -> BaseChatModel:
    """Build a small local text2text model for CPU inference.

    Uses google/flan-t5-small by default to keep downloads light.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_community.llms import HuggingFacePipeline

    local_model = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model)
    text2text = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=256,
        temperature=0.3,
    )
    hf_local = HuggingFacePipeline(pipeline=text2text)
    return ChatHuggingFace(llm=hf_local)


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



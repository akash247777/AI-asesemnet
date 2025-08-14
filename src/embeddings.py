import os

# Avoid importing TensorFlow/Keras paths inside transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from langchain_huggingface import HuggingFaceEndpointEmbeddings

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    GoogleGenerativeAIEmbeddings = None  # type: ignore


def build_embeddings(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Build an embeddings instance with robust fallback.

    Preference order:
    1) Hugging Face Inference API via `HUGGINGFACE_API_KEY` (fast, no local Torch)
    2) Local sentence-transformers model via `langchain_community` if API token is missing/invalid
    """
    try:
        from src.config import get_settings
    except Exception:
        from config import get_settings

    settings = get_settings()

    # Prefer Google embeddings when requested or when GOOGLE_API_KEY is present
    embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "").strip().lower()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if embeddings_provider == "google" or (google_api_key and embeddings_provider != "local"):
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError(
                "langchain-google-genai is not installed. Add it to requirements.txt and pip install."
            )
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google embeddings")
        google_emb_model = os.getenv("GOOGLE_EMBEDDINGS_MODEL", "text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=google_emb_model, google_api_key=google_api_key)

    # Allow forcing local backend via env var to avoid HF auth entirely
    backend_override = os.getenv("EMBEDDINGS_BACKEND", "").strip().lower()
    if backend_override == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        device = os.getenv("EMBEDDINGS_DEVICE", "cpu")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Try remote Inference API first when a token is present
    if getattr(settings, "huggingface_api_key", None):
        try:
            hf = HuggingFaceEndpointEmbeddings(
                repo_id=model_name,
                huggingfacehub_api_token=settings.huggingface_api_key,
            )
            # Validate token early; fall back on auth failure
            try:
                _ = hf.embed_query("ping")
                return hf
            except Exception:
                pass
        except Exception:
            # Fall back to local model if remote API auth fails or is unavailable
            pass

    # Fallback: local embeddings using sentence-transformers (force CPU to avoid meta tensor issues)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as exc:
        # Provide a clearer error if community embeddings are unavailable
        raise RuntimeError(
            "Hugging Face Inference API unavailable and local embeddings backend is missing. "
            "Ensure `langchain-community` and `sentence-transformers` are installed."
        ) from exc

    device = os.getenv("EMBEDDINGS_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )



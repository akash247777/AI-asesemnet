from langchain_huggingface import HuggingFaceEndpointEmbeddings


def build_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEndpointEmbeddings:
    """Use Hugging Face Inference Endpoint for embeddings to avoid local Torch issues."""
    try:
        from src.config import get_settings
    except Exception:
        from config import get_settings

    settings = get_settings()
    if not settings.huggingface_api_key:
        raise ValueError("Missing HUGGINGFACE_API_KEY for embeddings")

    return HuggingFaceEndpointEmbeddings(
        repo_id=model_name,
        huggingfacehub_api_token=settings.huggingface_api_key,
    )



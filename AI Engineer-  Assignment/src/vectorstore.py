from typing import Optional

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .config import get_settings


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_api_key:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def ensure_qdrant_ready(client: QdrantClient) -> None:
    """Checks connectivity to Qdrant and raises a helpful error if unreachable."""
    try:
        client.get_collections()
    except Exception as exc:
        settings = get_settings()
        raise RuntimeError(
            "Qdrant is not reachable. Verify QDRANT_URL and QDRANT_API_KEY (Qdrant Cloud) and network connectivity.\n"
            f"Tried: {settings.qdrant_url}"
        ) from exc


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    exists = False
    try:
        info = client.get_collection(collection_name)
        if info.status:  # collection exists
            exists = True
    except Exception:
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_vectorstore(embeddings, collection_name: Optional[str] = None) -> Qdrant:
    settings = get_settings()
    collection = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    ensure_qdrant_ready(client)
    # Try to ensure collection exists. vector_size will be set automatically by LangChain on first upsert.
    # For first-time creation we need size; we can delay creation and let add_texts handle it via upsert.
    return Qdrant(client=client, collection_name=collection, embeddings=embeddings)



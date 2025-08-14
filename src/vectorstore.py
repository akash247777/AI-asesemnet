from typing import Optional
import os
import re

from langchain_qdrant import Qdrant
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


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int | None = None) -> None:
    exists = False
    try:
        info = client.get_collection(collection_name)
        if getattr(info, "status", None):  # collection exists
            exists = True
    except Exception:
        exists = False

    if not exists and vector_size is not None:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _get_existing_vector_size(client: QdrantClient, collection_name: str) -> int | None:
    try:
        info = client.get_collection(collection_name)
        cfg = getattr(info, "config", None)
        params = getattr(cfg, "params", None) if cfg is not None else None
        vectors = getattr(params, "vectors", None) if params is not None else None
        if vectors is None:
            return None
        size = getattr(vectors, "size", None)
        if isinstance(size, int):
            return size
        # Some client versions return dict-like vectors
        if isinstance(vectors, dict):
            return vectors.get("size")
    except Exception:
        return None
    return None


def _detect_embedding_dimension(embeddings) -> int:
    """Best-effort detection of embedding vector dimension by probing a single query."""
    try:
        vec = embeddings.embed_query("dimension probe")
        if isinstance(vec, (list, tuple)) and vec:
            return len(vec)
    except Exception:
        pass
    # Default to common BGE-small dimension when detection fails
    return 384


def get_vectorstore(embeddings, collection_name: Optional[str] = None) -> Qdrant:
    settings = get_settings()
    collection = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    ensure_qdrant_ready(client)
    # Ensure collection exists with the correct dimension for the active embeddings
    try:
        dim = _detect_embedding_dimension(embeddings)
        ensure_collection(client, collection, vector_size=dim)
        existing = _get_existing_vector_size(client, collection)
        if isinstance(existing, int) and existing != dim:
            # Optionally auto-recreate on mismatch to avoid 400 errors
            if os.getenv("QDRANT_AUTO_RECREATE", "").strip().lower() == "true":
                try:
                    client.delete_collection(collection)
                except Exception:
                    pass
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    f"Qdrant collection '{collection}' has dimension {existing}, but embeddings produce {dim}. "
                    "Set QDRANT_AUTO_RECREATE=true in .env to drop & recreate the collection automatically, "
                    "or change QDRANT_COLLECTION to a new name."
                )
    except Exception:
        pass
    # Ensure event loop exists for async paths used by vectorstore
    try:
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except Exception:
        pass
    return Qdrant(client=client, collection_name=collection, embeddings=embeddings)



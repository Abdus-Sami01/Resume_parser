"""Dense embedding backends.

"hash" is a deterministic, dependency-free bag-of-tokens hashing embedder —
not semantically meaningful, but stable and fast, so the rest of the pipeline
(retrieval, scoring, tests) is fully exercisable offline. Swap to "openai"
(text-embedding-3-small) or a local bge-large-en-v1.5 for real semantic
recall in production.
"""
import hashlib
import math
from typing import Protocol

from app.config import get_settings
# Shared with the sparse index so the dense and sparse halves never tokenize differently.
from app.services.search.vector_store import tokenize


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingClient:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()
        from openai import OpenAI

        client_kwargs: dict = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = OpenAI(**client_kwargs)
        self._model = settings.embedding_model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding


def get_embedding_client() -> EmbeddingClient:
    settings = get_settings()
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingClient()
    return HashEmbeddingClient(dim=settings.embedding_dim)

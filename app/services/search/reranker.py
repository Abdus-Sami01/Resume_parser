"""Stage-2 reranking: exact contextual fit between one JD and one candidate.

"lexical" is a dependency-free token-overlap scorer (Jaccard-ish) — a stand-in
that keeps the two-stage pipeline shape correct offline. "cross_encoder" is
the production backend: a real cross-encoder (bge-reranker-large or the
Cohere Rerank API) that jointly encodes (query, document) pairs for a much
sharper relevance score than retrieval similarity alone.
"""
from typing import Protocol

from app.config import get_settings
from app.services.search.vector_store import tokenize


class Reranker(Protocol):
    def score(self, query: str, document: str) -> float: ...


class LexicalOverlapReranker:
    """Jaccard overlap between query and document tokens, in [0, 1]."""

    def score(self, query: str, document: str) -> float:
        query_tokens = set(tokenize(query))
        doc_tokens = set(tokenize(document))
        if not query_tokens or not doc_tokens:
            return 0.0
        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        return len(intersection) / len(union)


class CrossEncoderReranker:
    """Production backend: sentence-transformers CrossEncoder (e.g. bge-reranker-large)."""

    def __init__(self) -> None:
        settings = get_settings()
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.reranker_model)

    def score(self, query: str, document: str) -> float:
        return float(self._model.predict([(query, document)])[0])


def get_reranker() -> Reranker:
    settings = get_settings()
    if settings.reranker_backend == "cross_encoder":
        return CrossEncoderReranker()
    return LexicalOverlapReranker()

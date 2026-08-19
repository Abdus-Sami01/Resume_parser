"""Stage-2 reranking: exact contextual fit between one JD and a batch of candidates.

The interface is deliberately batch-first. Cross-encoders jointly encode each
(query, document) pair, so scoring N candidates one call at a time costs N
sequential forward passes; handing the model the whole batch lets it fill the
GPU in a few passes instead. A per-pair interface would push that cost onto
every caller and get harder to undo as callers multiply.

"lexical" is a dependency-free token-overlap scorer that keeps the two-stage
pipeline shape correct offline. "cross_encoder" is the production backend
(bge-reranker-large, or the Cohere Rerank API).
"""
from typing import Protocol

from app.config import get_settings
from app.services.search.vector_store import tokenize


class Reranker(Protocol):
    def score_batch(self, query: str, documents: list[str]) -> list[float]: ...


class LexicalOverlapReranker:
    """Jaccard overlap between query and document tokens, in [0, 1]."""

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [0.0] * len(documents)

        scores: list[float] = []
        for document in documents:
            doc_tokens = set(tokenize(document))
            if not doc_tokens:
                scores.append(0.0)
                continue
            scores.append(len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens))
        return scores


class CrossEncoderReranker:
    """Production backend: sentence-transformers CrossEncoder (e.g. bge-reranker-large)."""

    def __init__(self) -> None:
        settings = get_settings()
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.reranker_model)

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        return [float(score) for score in self._model.predict(pairs)]


def get_reranker() -> Reranker:
    settings = get_settings()
    if settings.reranker_backend == "cross_encoder":
        return CrossEncoderReranker()
    return LexicalOverlapReranker()

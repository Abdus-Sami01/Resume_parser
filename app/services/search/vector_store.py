"""Hybrid (dense + sparse) vector store.

"memory" is an in-process implementation (cosine similarity for dense,
term-overlap BM25-ish scoring for sparse) — no server needed, good for tests
and small deployments. "qdrant" is the production backend: a single Qdrant
collection with both a dense vector and a sparse (BM25-style) vector per
point, combined server-side.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


@dataclass
class IndexedDocument:
    id: str
    dense_vector: list[float]
    sparse_terms: Counter
    payload: dict


class VectorStore(Protocol):
    def upsert(self, doc_id: str, dense_vector: list[float], text: str, payload: dict) -> None: ...

    def search(self, query_dense: list[float], query_text: str, top_k: int) -> list[SearchHit]: ...


class InMemoryHybridVectorStore:
    """Dependency-free hybrid store: cosine (dense) + BM25 (sparse), combined by weighted sum."""

    def __init__(self, dense_weight: float = 0.6, sparse_weight: float = 0.4) -> None:
        self._docs: dict[str, IndexedDocument] = {}
        self._doc_freq: Counter = Counter()
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def upsert(self, doc_id: str, dense_vector: list[float], text: str, payload: dict) -> None:
        terms = Counter(tokenize(text))
        if doc_id in self._docs:
            self._unindex_terms(self._docs[doc_id].sparse_terms)
        self._docs[doc_id] = IndexedDocument(doc_id, dense_vector, terms, payload)
        self._doc_freq.update(terms.keys())

    def _unindex_terms(self, terms: Counter) -> None:
        for term in terms:
            self._doc_freq[term] -= 1
            if self._doc_freq[term] <= 0:
                del self._doc_freq[term]

    def search(self, query_dense: list[float], query_text: str, top_k: int) -> list[SearchHit]:
        query_terms = tokenize(query_text)
        hits: list[SearchHit] = []

        for doc in self._docs.values():
            dense_score = self._cosine(query_dense, doc.dense_vector)
            sparse_score = self._bm25_like(query_terms, doc.sparse_terms)
            combined = self._dense_weight * dense_score + self._sparse_weight * sparse_score
            hits.append(SearchHit(id=doc.id, score=combined, payload=doc.payload))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _bm25_like(self, query_terms: list[str], doc_terms: Counter) -> float:
        if not query_terms or not doc_terms:
            return 0.0
        n_docs = max(len(self._docs), 1)
        doc_len = sum(doc_terms.values()) or 1
        score = 0.0
        for term in set(query_terms):
            tf = doc_terms.get(term, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(term, 1)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf / (tf + 1.2))
        return score / doc_len**0.5

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class QdrantHybridVectorStore:
    """Production backend: Qdrant collection with a dense + sparse (BM25) vector per point."""

    def __init__(self) -> None:
        settings = get_settings()
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        self._settings = settings
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self._collection = settings.qdrant_collection

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={"dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

    def upsert(self, doc_id: str, dense_vector: list[float], text: str, payload: dict) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        terms = Counter(tokenize(text))
        indices = [hash(term) % (2**31) for term in terms]
        values = [float(count) for count in terms.values()]

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=doc_id,
                    vector={"dense": dense_vector, "sparse": SparseVector(indices=indices, values=values)},
                    payload=payload,
                )
            ],
        )

    def search(self, query_dense: list[float], query_text: str, top_k: int) -> list[SearchHit]:
        results = self._client.query_points(
            collection_name=self._collection,
            using="dense",
            query=query_dense,
            limit=top_k,
            with_payload=True,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in results.points]


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "qdrant":
        return QdrantHybridVectorStore()
    return InMemoryHybridVectorStore()

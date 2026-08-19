"""Hybrid (dense + sparse) vector store.

"memory" is an in-process implementation (cosine similarity for dense,
term-overlap BM25-ish scoring for sparse) — no server needed, good for tests
and small deployments. "qdrant" is the production backend: a single Qdrant
collection with both a dense vector and a sparse (BM25-style) vector per
point, combined server-side.
"""
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Qdrant sparse indices must be stable across processes. Python's builtin hash()
# is randomized per interpreter (PYTHONHASHSEED), so a term indexed by a Celery
# worker would land on a different index than the same term at query time.
_SPARSE_INDEX_SPACE = 2**31


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sparse_index(term: str) -> int:
    """Deterministic, process-stable index for a sparse-vector term."""
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SPARSE_INDEX_SPACE


def payload_matches_filters(payload: dict, filters: dict | None) -> bool:
    """Scalar values compare by equality; list values require every item to be present."""
    if not filters:
        return True

    for key, expected in filters.items():
        actual = payload.get(key)
        if isinstance(expected, list):
            if not isinstance(actual, list) or any(item not in actual for item in expected):
                return False
        elif actual != expected:
            return False

    return True


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

    def search(
        self,
        query_dense: list[float],
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]: ...


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

    def search(
        self,
        query_dense: list[float],
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        query_terms = tokenize(query_text)
        hits: list[SearchHit] = []

        for doc in self._docs.values():
            if not payload_matches_filters(doc.payload, filters):
                continue
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
        # ":memory:" runs qdrant-client's embedded local mode, which exercises this
        # backend in tests without a server. Anything else is treated as a URL.
        if settings.qdrant_url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
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
        indices = [sparse_index(term) for term in terms]
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

    def search(
        self,
        query_dense: list[float],
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        """Runs dense and sparse branches server-side and fuses them with Reciprocal Rank Fusion."""
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        terms = Counter(tokenize(query_text))
        query_sparse = SparseVector(
            indices=[sparse_index(term) for term in terms],
            values=[float(count) for count in terms.values()],
        )
        query_filter = self._build_filter(filters)

        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=query_dense, using="dense", limit=top_k, filter=query_filter),
                Prefetch(query=query_sparse, using="sparse", limit=top_k, filter=query_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in results.points]

    @staticmethod
    def _build_filter(filters: dict | None):
        if not filters:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        for key, expected in filters.items():
            # Qdrant matches a MatchValue against any element of an array payload field,
            # so a list expands to one condition per required item.
            for value in expected if isinstance(expected, list) else [expected]:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "qdrant":
        return QdrantHybridVectorStore()
    return InMemoryHybridVectorStore()

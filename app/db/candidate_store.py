"""In-process store for parsed candidate profiles, keyed by candidate id.

Stands in for the structured-profile database (e.g. Postgres) in the real
architecture — the vector store only holds embeddings + a thin payload, this
holds the full CandidateProfile the API returns.
"""
from dataclasses import dataclass
from functools import lru_cache

from app.schemas.candidate import CandidateProfile


@dataclass
class CandidateRecord:
    candidate_id: str
    profile: CandidateProfile
    raw_text: str


class CandidateStore:
    def __init__(self) -> None:
        self._records: dict[str, CandidateRecord] = {}

    def save(self, record: CandidateRecord) -> None:
        self._records[record.candidate_id] = record

    def get(self, candidate_id: str) -> CandidateRecord | None:
        return self._records.get(candidate_id)

    def all(self) -> list[CandidateRecord]:
        return list(self._records.values())


@lru_cache
def get_candidate_store() -> CandidateStore:
    return CandidateStore()

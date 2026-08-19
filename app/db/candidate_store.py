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
    fingerprint: str = ""


class CandidateStore:
    def __init__(self) -> None:
        self._records: dict[str, CandidateRecord] = {}
        self._by_fingerprint: dict[str, str] = {}

    def save(self, record: CandidateRecord) -> None:
        previous = self._records.get(record.candidate_id)
        if previous and previous.fingerprint and previous.fingerprint != record.fingerprint:
            self._by_fingerprint.pop(previous.fingerprint, None)

        self._records[record.candidate_id] = record
        if record.fingerprint:
            self._by_fingerprint[record.fingerprint] = record.candidate_id

    def get(self, candidate_id: str) -> CandidateRecord | None:
        return self._records.get(candidate_id)

    def find_by_fingerprint(self, fingerprint: str) -> str | None:
        return self._by_fingerprint.get(fingerprint)

    def delete(self, candidate_id: str) -> bool:
        record = self._records.pop(candidate_id, None)
        if record is None:
            return False
        if record.fingerprint:
            self._by_fingerprint.pop(record.fingerprint, None)
        return True

    def all(self) -> list[CandidateRecord]:
        return list(self._records.values())

    def page(self, offset: int = 0, limit: int = 50) -> tuple[list[CandidateRecord], int]:
        """Returns one page plus the unpaged total, so clients can render "x of y"."""
        records = self.all()
        return records[offset : offset + limit], len(records)


@lru_cache
def get_candidate_store() -> CandidateStore:
    return CandidateStore()

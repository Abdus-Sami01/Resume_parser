"""Candidate profile storage, in two interchangeable backends.

The vector store holds embeddings and a thin payload; this holds the full
CandidateProfile the API returns. "memory" keeps everything in-process (fast,
zero setup, lost on restart); "sqlite" persists to a file so records survive a
restart and are shared between the API and its Celery workers.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Protocol

from app.config import get_settings
from app.schemas.candidate import CandidateProfile


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CandidateRecord:
    candidate_id: str
    profile: CandidateProfile
    raw_text: str
    fingerprint: str = ""
    created_at: datetime = field(default_factory=_utcnow)


class CandidateStoreProtocol(Protocol):
    def save(self, record: CandidateRecord) -> None: ...

    def get(self, candidate_id: str) -> CandidateRecord | None: ...

    def find_by_fingerprint(self, fingerprint: str) -> str | None: ...

    def delete(self, candidate_id: str) -> bool: ...

    def all(self) -> list[CandidateRecord]: ...

    def page(self, offset: int = 0, limit: int = 50) -> tuple[list[CandidateRecord], int]: ...


class CandidateStore:
    """In-process store. Everything is lost when the process exits."""

    def __init__(self) -> None:
        self._records: dict[str, CandidateRecord] = {}
        self._by_fingerprint: dict[str, str] = {}

    def save(self, record: CandidateRecord) -> None:
        previous = self._records.get(record.candidate_id)
        if previous and previous.fingerprint and previous.fingerprint != record.fingerprint:
            self._by_fingerprint.pop(previous.fingerprint, None)
        if previous:
            # An update keeps its original creation time rather than resetting it.
            record.created_at = previous.created_at

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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    fingerprint  TEXT,
    raw_text     TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_fingerprint ON candidates (fingerprint);
CREATE INDEX IF NOT EXISTS idx_candidates_created_at  ON candidates (created_at);
"""


class SqliteCandidateStore:
    """File-backed store. Records outlive the process and are visible to workers."""

    def __init__(self, path: str) -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        # A :memory: database only exists for as long as its connection, so that
        # mode holds one shared connection instead of opening per operation.
        self._lock = threading.Lock()
        self._shared = sqlite3.connect(path, check_same_thread=False) if path == ":memory:" else None

        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._shared or sqlite3.connect(self._path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            finally:
                if self._shared is None:
                    connection.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=row["candidate_id"],
            profile=CandidateProfile.model_validate_json(row["profile_json"]),
            raw_text=row["raw_text"],
            fingerprint=row["fingerprint"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def save(self, record: CandidateRecord) -> None:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM candidates WHERE candidate_id = ?", (record.candidate_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else record.created_at.isoformat()

            connection.execute(
                "REPLACE INTO candidates "
                "(candidate_id, fingerprint, raw_text, profile_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.candidate_id,
                    record.fingerprint or None,
                    record.raw_text,
                    record.profile.model_dump_json(),
                    created_at,
                ),
            )

    def get(self, candidate_id: str) -> CandidateRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def find_by_fingerprint(self, fingerprint: str) -> str | None:
        if not fingerprint:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT candidate_id FROM candidates WHERE fingerprint = ? LIMIT 1", (fingerprint,)
            ).fetchone()
        return row["candidate_id"] if row else None

    def delete(self, candidate_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM candidates WHERE candidate_id = ?", (candidate_id,)
            )
        return cursor.rowcount > 0

    def all(self) -> list[CandidateRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM candidates ORDER BY created_at").fetchall()
        return [self._to_record(row) for row in rows]

    def page(self, offset: int = 0, limit: int = 50) -> tuple[list[CandidateRecord], int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
            rows = connection.execute(
                "SELECT * FROM candidates ORDER BY created_at LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._to_record(row) for row in rows], total


@lru_cache
def get_candidate_store() -> CandidateStoreProtocol:
    settings = get_settings()
    if settings.store_backend == "sqlite":
        return SqliteCandidateStore(settings.sqlite_path)
    return CandidateStore()

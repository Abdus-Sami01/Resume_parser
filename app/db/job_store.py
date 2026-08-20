"""Job posting storage, in two interchangeable backends.

Postings were previously parsed and discarded, which made every match a one-shot
call: a saved posting can be re-run as the candidate pool grows, and is what the
reverse direction (candidate -> jobs) ranks against.
"""
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Protocol

from app.config import get_settings
from app.schemas.job import JobProfile


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    profile: JobProfile
    created_at: datetime = field(default_factory=_utcnow)


class JobStoreProtocol(Protocol):
    def save(self, profile: JobProfile, job_id: str | None = None) -> JobRecord: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def delete(self, job_id: str) -> bool: ...

    def all(self) -> list[JobRecord]: ...

    def page(self, offset: int = 0, limit: int = 50) -> tuple[list[JobRecord], int]: ...


class JobStore:
    """In-process store. Everything is lost when the process exits."""

    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}

    def save(self, profile: JobProfile, job_id: str | None = None) -> JobRecord:
        job_id = job_id or str(uuid.uuid4())
        existing = self._records.get(job_id)

        record = JobRecord(
            job_id=job_id,
            profile=profile,
            # An update keeps its original creation time rather than resetting it.
            created_at=existing.created_at if existing else _utcnow(),
        )
        self._records[job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self._records.get(job_id)

    def delete(self, job_id: str) -> bool:
        return self._records.pop(job_id, None) is not None

    def all(self) -> list[JobRecord]:
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def page(self, offset: int = 0, limit: int = 50) -> tuple[list[JobRecord], int]:
        records = self.all()
        return records[offset : offset + limit], len(records)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at);
"""


class SqliteJobStore:
    """File-backed store. Postings outlive the process."""

    def __init__(self, path: str) -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

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
    def _to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            profile=JobProfile.model_validate_json(row["profile_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def save(self, profile: JobProfile, job_id: str | None = None) -> JobRecord:
        job_id = job_id or str(uuid.uuid4())

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else _utcnow().isoformat()

            connection.execute(
                "REPLACE INTO jobs (job_id, profile_json, created_at) VALUES (?, ?, ?)",
                (job_id, profile.model_dump_json(), created_at),
            )

        return JobRecord(job_id=job_id, profile=profile, created_at=datetime.fromisoformat(created_at))

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._to_record(row) if row else None

    def delete(self, job_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        return cursor.rowcount > 0

    def all(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._to_record(row) for row in rows]

    def page(self, offset: int = 0, limit: int = 50) -> tuple[list[JobRecord], int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._to_record(row) for row in rows], total


@lru_cache
def get_job_store() -> JobStoreProtocol:
    settings = get_settings()
    if settings.store_backend == "sqlite":
        return SqliteJobStore(settings.sqlite_path)
    return JobStore()

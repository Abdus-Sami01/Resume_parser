"""Pipeline storage, in the same two backends as candidates and jobs."""
import sqlite3
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Protocol

from app.config import get_settings
from app.schemas.pipeline import MatchSnapshot, PipelineEntry, Stage, StageEvent


class PipelineStoreProtocol(Protocol):
    def upsert(self, entry: PipelineEntry) -> PipelineEntry: ...

    def get(self, job_id: str, candidate_id: str) -> PipelineEntry | None: ...

    def delete(self, job_id: str, candidate_id: str) -> bool: ...

    def for_job(self, job_id: str) -> list[PipelineEntry]: ...

    def for_candidate(self, candidate_id: str) -> list[PipelineEntry]: ...

    def delete_for_job(self, job_id: str) -> int: ...

    def delete_for_candidate(self, candidate_id: str) -> int: ...


class PipelineStore:
    """In-process store, keyed by the (job, candidate) pair."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], PipelineEntry] = {}

    def upsert(self, entry: PipelineEntry) -> PipelineEntry:
        self._entries[(entry.job_id, entry.candidate_id)] = entry
        return entry

    def get(self, job_id: str, candidate_id: str) -> PipelineEntry | None:
        return self._entries.get((job_id, candidate_id))

    def delete(self, job_id: str, candidate_id: str) -> bool:
        return self._entries.pop((job_id, candidate_id), None) is not None

    def for_job(self, job_id: str) -> list[PipelineEntry]:
        return [e for e in self._entries.values() if e.job_id == job_id]

    def for_candidate(self, candidate_id: str) -> list[PipelineEntry]:
        return [e for e in self._entries.values() if e.candidate_id == candidate_id]

    def delete_for_job(self, job_id: str) -> int:
        doomed = [key for key in self._entries if key[0] == job_id]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def delete_for_candidate(self, candidate_id: str) -> int:
        doomed = [key for key in self._entries if key[1] == candidate_id]
        for key in doomed:
            del self._entries[key]
        return len(doomed)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline (
    job_id       TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    stage        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    history_json TEXT NOT NULL,
    snapshot_json TEXT,
    PRIMARY KEY (job_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_pipeline_job       ON pipeline (job_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_candidate ON pipeline (candidate_id);
"""


class SqlitePipelineStore:
    """File-backed store, so stage history outlives the process."""

    def __init__(self, path: str) -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._shared = sqlite3.connect(path, check_same_thread=False) if path == ":memory:" else None

        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Adds columns introduced after a database was first created.

        `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new
        column never appears on a database that predates it — every read then
        fails on a file that looks perfectly valid.
        """
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(pipeline)")}
        if "snapshot_json" not in existing:
            connection.execute("ALTER TABLE pipeline ADD COLUMN snapshot_json TEXT")

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
    def _to_entry(row: sqlite3.Row) -> PipelineEntry:
        import json

        raw_snapshot = row["snapshot_json"] if "snapshot_json" in row.keys() else None

        return PipelineEntry(
            job_id=row["job_id"],
            candidate_id=row["candidate_id"],
            stage=Stage(row["stage"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            history=[StageEvent.model_validate(event) for event in json.loads(row["history_json"])],
            match_snapshot=MatchSnapshot.model_validate_json(raw_snapshot) if raw_snapshot else None,
        )

    def upsert(self, entry: PipelineEntry) -> PipelineEntry:
        import json

        with self._connect() as connection:
            connection.execute(
                "REPLACE INTO pipeline "
                "(job_id, candidate_id, stage, created_at, updated_at, history_json, snapshot_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.job_id,
                    entry.candidate_id,
                    entry.stage.value,
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    json.dumps([event.model_dump(mode="json") for event in entry.history]),
                    entry.match_snapshot.model_dump_json() if entry.match_snapshot else None,
                ),
            )
        return entry

    def get(self, job_id: str, candidate_id: str) -> PipelineEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pipeline WHERE job_id = ? AND candidate_id = ?",
                (job_id, candidate_id),
            ).fetchone()
        return self._to_entry(row) if row else None

    def delete(self, job_id: str, candidate_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM pipeline WHERE job_id = ? AND candidate_id = ?", (job_id, candidate_id)
            )
        return cursor.rowcount > 0

    def for_job(self, job_id: str) -> list[PipelineEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pipeline WHERE job_id = ? ORDER BY updated_at DESC", (job_id,)
            ).fetchall()
        return [self._to_entry(row) for row in rows]

    def for_candidate(self, candidate_id: str) -> list[PipelineEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pipeline WHERE candidate_id = ? ORDER BY updated_at DESC",
                (candidate_id,),
            ).fetchall()
        return [self._to_entry(row) for row in rows]

    def delete_for_job(self, job_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM pipeline WHERE job_id = ?", (job_id,))
        return cursor.rowcount

    def delete_for_candidate(self, candidate_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM pipeline WHERE candidate_id = ?", (candidate_id,)
            )
        return cursor.rowcount


@lru_cache
def get_pipeline_store() -> PipelineStoreProtocol:
    settings = get_settings()
    if settings.store_backend == "sqlite":
        return SqlitePipelineStore(settings.sqlite_path)
    return PipelineStore()

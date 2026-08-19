"""In-process store for parsed job postings.

Postings were previously parsed and discarded, which made every match a one-shot
call: a saved posting can be re-run as the candidate pool grows, and is what the
reverse direction (candidate -> jobs) ranks against.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

from app.schemas.job import JobProfile


@dataclass
class JobRecord:
    job_id: str
    profile: JobProfile
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class JobStore:
    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}

    def save(self, profile: JobProfile, job_id: str | None = None) -> JobRecord:
        job_id = job_id or str(uuid.uuid4())
        existing = self._records.get(job_id)

        record = JobRecord(
            job_id=job_id,
            profile=profile,
            # An update keeps its original creation time rather than resetting it.
            created_at=existing.created_at if existing else datetime.now(timezone.utc),
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


@lru_cache
def get_job_store() -> JobStore:
    return JobStore()

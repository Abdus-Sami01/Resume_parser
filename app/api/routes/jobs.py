"""Job posting endpoints: stateless parsing, plus a persisted catalogue."""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.job_store import get_job_store
from app.schemas.job import JobProfile, JobWeights
from app.schemas.match import MatchResult
from app.services.extraction.job_extractor import get_job_extractor
from app.services.search.matcher import match

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobParseRequest(BaseModel):
    title: str
    description: str
    weights: JobWeights | None = None


class StoredJob(BaseModel):
    job_id: str
    profile: JobProfile
    created_at: str


class JobPage(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[StoredJob]


def _to_stored(record) -> StoredJob:
    return StoredJob(
        job_id=record.job_id, profile=record.profile, created_at=record.created_at.isoformat()
    )


@router.post("/parse", response_model=JobProfile)
async def parse_job(request: JobParseRequest) -> JobProfile:
    """Parses without persisting — useful for previewing how a posting is interpreted."""
    return get_job_extractor().extract(request.title, request.description, request.weights)


@router.post("", response_model=StoredJob, status_code=201)
async def create_job(request: JobParseRequest) -> StoredJob:
    profile = get_job_extractor().extract(request.title, request.description, request.weights)
    return _to_stored(get_job_store().save(profile))


@router.get("", response_model=JobPage)
async def list_jobs(
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
) -> JobPage:
    records, total = get_job_store().page(offset=offset, limit=limit)
    return JobPage(
        total=total, offset=offset, limit=limit, items=[_to_stored(r) for r in records]
    )


@router.get("/{job_id}", response_model=StoredJob)
async def read_job(job_id: str) -> StoredJob:
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_stored(record)


@router.put("/{job_id}", response_model=StoredJob)
async def replace_job(job_id: str, request: JobParseRequest) -> StoredJob:
    store = get_job_store()
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")

    profile = get_job_extractor().extract(request.title, request.description, request.weights)
    return _to_stored(store.save(profile, job_id=job_id))


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    if not get_job_store().delete(job_id):
        raise HTTPException(status_code=404, detail="job not found")


@router.post("/{job_id}/match", response_model=list[MatchResult])
async def match_stored_job(
    job_id: str,
    top_k: int | None = Query(None, ge=1, le=1000),
    top_n: int | None = Query(None, ge=1, le=100),
) -> list[MatchResult]:
    """Re-runs a saved posting against the current candidate pool."""
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return match(record.profile, top_k=top_k, top_n=top_n)

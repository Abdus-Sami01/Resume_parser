"""Job posting endpoints: stateless parsing, plus a persisted catalogue."""
import csv
import io

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db.job_store import get_job_store
from app.schemas.job import JobProfile, JobStatus, JobWeights
from app.schemas.match import MatchResult
from app.services.extraction.job_extractor import get_job_extractor
from app.services.privacy import pseudonym_for, redact_profile
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
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: JobStatus | None = Query(None, description="Filter to one status"),
) -> JobPage:
    if status is not None:
        matching = [r for r in get_job_store().all() if r.profile.status == status]
        return JobPage(
            total=len(matching),
            offset=offset,
            limit=limit,
            items=[_to_stored(r) for r in matching[offset : offset + limit]],
        )

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


class JobStatusRequest(BaseModel):
    status: JobStatus


@router.patch("/{job_id}/status", response_model=StoredJob)
async def set_job_status(job_id: str, request: JobStatusRequest) -> StoredJob:
    """Changes a req's status without re-parsing it.

    Closing a role keeps it and its pipeline readable — the hiring history is the
    point — while taking it out of reverse matching and sourcing analytics.
    """
    store = get_job_store()
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")

    updated = record.profile.model_copy(update={"status": request.status})
    return _to_stored(store.save(updated, job_id=job_id))


@router.get("/{job_id}/status")
async def read_job_status(job_id: str) -> dict:
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "status": record.profile.status, "is_active": record.profile.is_active}


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    """Removes the posting and its pipeline, which is meaningless without it."""
    from app.db.pipeline_store import get_pipeline_store

    if not get_job_store().delete(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    get_pipeline_store().delete_for_job(job_id)


@router.post("/{job_id}/match", response_model=list[MatchResult])
async def match_stored_job(
    job_id: str,
    top_k: int | None = Query(None, ge=1, le=1000),
    top_n: int | None = Query(None, ge=1, le=100),
    blind: bool = Query(False, description="Redact identifying fields for bias-reduced review"),
) -> list[MatchResult]:
    """Re-runs a saved posting against the current candidate pool."""
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")

    results = match(record.profile, top_k=top_k, top_n=top_n)
    if not blind:
        return results

    return [
        result.model_copy(
            update={"candidate": redact_profile(result.candidate, result.candidate_id)}
        )
        for result in results
    ]


_EXPORT_COLUMNS = [
    "rank",
    "candidate_id",
    "name",
    "email",
    "score",
    "skills_score",
    "experience_score",
    "education_score",
    "certification_score",
    "years_experience",
    "required_years",
    "degree_matched",
    "matched_required_skills",
    "missing_required_skills",
    "matched_certifications",
    "missing_certifications",
]


@router.get("/{job_id}/match/export")
async def export_matches_csv(
    job_id: str,
    top_n: int | None = Query(None, ge=1, le=500),
    blind: bool = Query(False, description="Redact identifying columns"),
) -> StreamingResponse:
    """Match results as CSV, for handing a shortlist to someone who lives in a spreadsheet.

    Missing required skills travel with each row: a shortlist that shows only
    scores forces the reader back into the API to learn why anyone ranked where
    they did.
    """
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_EXPORT_COLUMNS)

    for rank, result in enumerate(match(record.profile, top_n=top_n), start=1):
        writer.writerow(
            [
                rank,
                result.candidate_id,
                pseudonym_for(result.candidate_id) if blind else result.candidate.name,
                "" if blind else (result.candidate.email or ""),
                f"{result.breakdown.weighted_total:.4f}",
                f"{result.breakdown.skills:.4f}",
                f"{result.breakdown.experience:.4f}",
                f"{result.breakdown.education:.4f}",
                f"{result.breakdown.certifications:.4f}",
                result.evidence.experience.candidate_years,
                result.evidence.experience.required_years,
                result.evidence.education.matched_degree or "",
                "; ".join(result.evidence.skills.matched_required),
                "; ".join(result.evidence.skills.missing_required),
                "; ".join(result.evidence.certifications.matched),
                "; ".join(result.evidence.certifications.missing),
            ]
        )

    buffer.seek(0)
    filename = f"matches-{job_id}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

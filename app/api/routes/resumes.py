"""Resume endpoints: upload (inline, async, bulk), retrieval, listing, and erasure."""
from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.candidate_store import get_candidate_store
from app.schemas.candidate import CandidateProfile
from app.schemas.match import JobMatchResult
from app.services.extraction.document_parser import get_document_parser
from app.services.extraction.resume_extractor import get_resume_extractor
from app.services.dedupe import find_all_duplicates, find_duplicates_for, merge_candidates
from app.services.search.matcher import (
    delete_candidate,
    find_similar_candidates,
    index_candidate,
    match_jobs_for_candidate,
)
from app.workers.tasks import get_task_state, submit_resume_parse

router = APIRouter(prefix="/resumes", tags=["resumes"])


class ResumeUploadResponse(BaseModel):
    candidate_id: str
    profile: CandidateProfile


class ResumeTaskResponse(BaseModel):
    task_id: str
    state: str
    candidate_id: str | None = None
    error: str | None = None


class BulkUploadItem(BaseModel):
    filename: str
    candidate_id: str | None = None
    error: str | None = None


class BulkUploadResponse(BaseModel):
    succeeded: int
    failed: int
    items: list[BulkUploadItem]


class CandidateSummary(BaseModel):
    candidate_id: str
    name: str
    email: str | None = None
    skills: list[str]
    total_years_experience: float


class SimilarCandidate(BaseModel):
    candidate_id: str
    score: float
    profile: CandidateProfile


class DuplicatePair(BaseModel):
    candidate_id: str
    other_candidate_id: str
    confidence: str
    reasons: list[str]


class MergeRequest(BaseModel):
    absorb_candidate_id: str = Field(..., description="The record folded into this one and deleted")


class CandidatePage(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[CandidateSummary]


def _summarize(record) -> CandidateSummary:
    return CandidateSummary(
        candidate_id=record.candidate_id,
        name=record.profile.name,
        email=record.profile.email,
        skills=record.profile.skills,
        total_years_experience=record.profile.total_years_experience,
    )


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    """Reads in chunks and stops at the cap, so an oversized upload is never fully buffered."""
    limit = get_settings().max_upload_bytes
    chunks: list[bytes] = []
    total = 0

    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"file exceeds {limit} bytes")
        chunks.append(chunk)

    if not total:
        raise HTTPException(status_code=400, detail="empty file")
    return b"".join(chunks), file.filename or "resume.txt"


def _parse_to_profile(file_bytes: bytes, filename: str) -> tuple[CandidateProfile, str]:
    try:
        raw_text = get_document_parser().parse(file_bytes, filename)
        return get_resume_extractor().extract(raw_text), raw_text
    except Exception as exc:
        # A corrupt upload is the client's problem to fix, not a server fault.
        raise HTTPException(status_code=422, detail=f"could not parse {filename}: {exc}") from exc


@router.post("", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile) -> ResumeUploadResponse:
    """Parse inline. Fine for plain text; use /resumes/async for heavy PDF or LLM extraction."""
    file_bytes, filename = await _read_upload(file)
    profile, raw_text = _parse_to_profile(file_bytes, filename)
    return ResumeUploadResponse(
        candidate_id=index_candidate(profile, raw_text), profile=profile
    )


@router.post("/bulk", response_model=BulkUploadResponse)
async def upload_resumes_bulk(files: list[UploadFile]) -> BulkUploadResponse:
    """Ingests a batch, reporting per-file outcomes.

    One unreadable file in a batch of two hundred must not discard the other
    hundred and ninety-nine, so failures are collected rather than raised.
    """
    items: list[BulkUploadItem] = []

    for file in files:
        filename = file.filename or "resume.txt"
        try:
            file_bytes, filename = await _read_upload(file)
            profile, raw_text = _parse_to_profile(file_bytes, filename)
            items.append(
                BulkUploadItem(filename=filename, candidate_id=index_candidate(profile, raw_text))
            )
        except HTTPException as exc:
            items.append(BulkUploadItem(filename=filename, error=str(exc.detail)))

    succeeded = sum(1 for item in items if item.candidate_id)
    return BulkUploadResponse(
        succeeded=succeeded, failed=len(items) - succeeded, items=items
    )


@router.post("/async", response_model=ResumeTaskResponse, status_code=202)
async def upload_resume_async(file: UploadFile) -> ResumeTaskResponse:
    """Dispatch parsing to a worker so slow extraction cannot time out the request."""
    file_bytes, filename = await _read_upload(file)
    return ResumeTaskResponse(**get_task_state(submit_resume_parse(file_bytes, filename)))


@router.get("", response_model=CandidatePage)
async def list_resumes(
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
) -> CandidatePage:
    records, total = get_candidate_store().page(offset=offset, limit=limit)
    return CandidatePage(
        total=total, offset=offset, limit=limit, items=[_summarize(r) for r in records]
    )


@router.get("/tasks/{task_id}", response_model=ResumeTaskResponse)
async def read_task(task_id: str) -> ResumeTaskResponse:
    return ResumeTaskResponse(**get_task_state(task_id))


@router.get("/duplicates", response_model=list[DuplicatePair])
async def list_duplicates() -> list[DuplicatePair]:
    """Likely same-person records across the whole pool, highest confidence first.

    Surfaced rather than merged automatically: collapsing two records discards a
    version of someone's history, which is a judgement call, not a cleanup task.
    """
    return [DuplicatePair(**vars(pair)) for pair in find_all_duplicates()]


# Declared above "/{candidate_id}" deliberately: FastAPI matches in declaration
# order, so a literal path registered afterwards is captured by the parameter.
@router.get("/{candidate_id}", response_model=CandidateProfile)
async def read_resume(candidate_id: str) -> CandidateProfile:
    record = get_candidate_store().get(candidate_id)
    if record is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return record.profile


@router.delete("/{candidate_id}", status_code=204)
async def erase_resume(candidate_id: str) -> None:
    """Erases the profile and its index entry — resumes are personal data."""
    if not delete_candidate(candidate_id):
        raise HTTPException(status_code=404, detail="candidate not found")


@router.get("/{candidate_id}/jobs", response_model=list[JobMatchResult])
async def match_jobs(
    candidate_id: str, top_n: int | None = Query(None, ge=1, le=100)
) -> list[JobMatchResult]:
    """Reverse match: rank stored postings for this candidate."""
    if get_candidate_store().get(candidate_id) is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return match_jobs_for_candidate(candidate_id, top_n=top_n)


@router.get("/{candidate_id}/similar", response_model=list[SimilarCandidate])
async def similar_candidates(
    candidate_id: str, top_n: int | None = Query(None, ge=1, le=100)
) -> list[SimilarCandidate]:
    """"More people like this one" — the usual follow-up to a good match or a good hire."""
    if get_candidate_store().get(candidate_id) is None:
        raise HTTPException(status_code=404, detail="candidate not found")

    return [
        SimilarCandidate(candidate_id=record.candidate_id, score=score, profile=record.profile)
        for record, score in find_similar_candidates(candidate_id, top_n=top_n)
    ]



@router.get("/{candidate_id}/duplicates", response_model=list[DuplicatePair])
async def candidate_duplicates(candidate_id: str) -> list[DuplicatePair]:
    if get_candidate_store().get(candidate_id) is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return [DuplicatePair(**vars(pair)) for pair in find_duplicates_for(candidate_id)]


@router.post("/{candidate_id}/merge", response_model=CandidateProfile)
async def merge_candidate(candidate_id: str, request: MergeRequest) -> CandidateProfile:
    """Folds another record into this one, carrying its pipeline history across."""
    try:
        merged = merge_candidates(candidate_id, request.absorb_candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return merged.profile

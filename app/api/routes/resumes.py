from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.db.candidate_store import get_candidate_store
from app.schemas.candidate import CandidateProfile
from app.services.extraction.document_parser import get_document_parser
from app.services.extraction.resume_extractor import get_resume_extractor
from app.services.search.matcher import index_candidate
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


@router.post("", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile) -> ResumeUploadResponse:
    """Parse inline. Fine for plain text; use /resumes/async for heavy PDF or LLM extraction."""
    file_bytes, filename = await _read_upload(file)

    try:
        raw_text = get_document_parser().parse(file_bytes, filename)
        profile = get_resume_extractor().extract(raw_text)
    except HTTPException:
        raise
    except Exception as exc:
        # A corrupt upload is the client's problem to fix, not a server fault.
        raise HTTPException(status_code=422, detail=f"could not parse {filename}: {exc}") from exc

    candidate_id = index_candidate(profile, raw_text)
    return ResumeUploadResponse(candidate_id=candidate_id, profile=profile)


@router.post("/async", response_model=ResumeTaskResponse, status_code=202)
async def upload_resume_async(file: UploadFile) -> ResumeTaskResponse:
    """Dispatch parsing to a worker so slow extraction cannot time out the request."""
    file_bytes, filename = await _read_upload(file)
    task_id = submit_resume_parse(file_bytes, filename)
    return ResumeTaskResponse(**get_task_state(task_id))


@router.get("/tasks/{task_id}", response_model=ResumeTaskResponse)
async def read_task(task_id: str) -> ResumeTaskResponse:
    return ResumeTaskResponse(**get_task_state(task_id))


@router.get("/{candidate_id}", response_model=CandidateProfile)
async def read_resume(candidate_id: str) -> CandidateProfile:
    record = get_candidate_store().get(candidate_id)
    if record is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return record.profile

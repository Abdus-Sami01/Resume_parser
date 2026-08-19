from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.schemas.candidate import CandidateProfile
from app.services.extraction.document_parser import get_document_parser
from app.services.extraction.resume_extractor import get_resume_extractor
from app.services.search.matcher import index_candidate

router = APIRouter(prefix="/resumes", tags=["resumes"])


class ResumeUploadResponse(BaseModel):
    candidate_id: str
    profile: CandidateProfile


@router.post("", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile) -> ResumeUploadResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    raw_text = get_document_parser().parse(file_bytes, file.filename or "resume.txt")
    profile = get_resume_extractor().extract(raw_text)
    candidate_id = index_candidate(profile, raw_text)

    return ResumeUploadResponse(candidate_id=candidate_id, profile=profile)

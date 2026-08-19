from fastapi import APIRouter
from pydantic import BaseModel

from app.schemas.job import JobProfile, JobWeights
from app.services.extraction.job_extractor import get_job_extractor

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobParseRequest(BaseModel):
    title: str
    description: str
    weights: JobWeights | None = None


@router.post("/parse", response_model=JobProfile)
async def parse_job(request: JobParseRequest) -> JobProfile:
    return get_job_extractor().extract(request.title, request.description, request.weights)

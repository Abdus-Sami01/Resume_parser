from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas.job import JobProfile
from app.schemas.match import MatchResult
from app.services.search.matcher import match

router = APIRouter(prefix="/search", tags=["search"])


class MatchRequest(BaseModel):
    job: JobProfile
    top_k: int | None = Field(default=None, ge=1, le=1000, description="Stage-1 retrieval depth")
    top_n: int | None = Field(default=None, ge=1, le=100, description="Candidates returned after reranking")
    filters: dict = Field(
        default_factory=dict,
        description='Hard metadata constraints, e.g. {"skills": ["python"], "location": "Remote"}',
    )


@router.post("/match", response_model=list[MatchResult])
async def match_candidates(request: MatchRequest) -> list[MatchResult]:
    return match(request.job, top_k=request.top_k, top_n=request.top_n, filters=request.filters or None)

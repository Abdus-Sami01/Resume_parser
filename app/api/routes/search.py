from fastapi import APIRouter

from app.schemas.job import JobProfile
from app.schemas.match import MatchResult
from app.services.search.matcher import match

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/match", response_model=list[MatchResult])
async def match_candidates(job: JobProfile, top_k: int | None = None, top_n: int | None = None) -> list[MatchResult]:
    return match(job, top_k=top_k, top_n=top_n)

"""Search endpoints: job-driven matching, and pool search with no posting involved."""
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile
from app.schemas.match import MatchResult
from app.services.search.matcher import match, search_candidates

router = APIRouter(prefix="/search", tags=["search"])


class MatchRequest(BaseModel):
    job: JobProfile
    top_k: int | None = Field(default=None, ge=1, le=1000, description="Stage-1 retrieval depth")
    top_n: int | None = Field(default=None, ge=1, le=100, description="Candidates returned after reranking")
    filters: dict = Field(
        default_factory=dict,
        description='Hard metadata constraints, e.g. {"skills": ["python"], "location": "Remote"}',
    )


class CandidateSearchRequest(BaseModel):
    """Criteria a recruiter has in mind, rather than a written posting."""

    query: str = Field("", description="Free text, matched semantically and by keyword")
    skills: list[str] = Field(default_factory=list, description="Every listed skill must be present")
    location: str | None = None
    min_years_experience: float | None = Field(default=None, ge=0)
    max_years_experience: float | None = Field(default=None, ge=0)
    certifications: list[str] = Field(default_factory=list)
    top_n: int | None = Field(default=None, ge=1, le=200)

    def to_filters(self) -> dict:
        filters: dict = {}
        if self.skills:
            filters["skills"] = [skill.strip().lower() for skill in self.skills]
        if self.certifications:
            filters["certifications"] = self.certifications
        if self.location:
            filters["location"] = self.location

        years: dict[str, float] = {}
        if self.min_years_experience is not None:
            years["gte"] = self.min_years_experience
        if self.max_years_experience is not None:
            years["lte"] = self.max_years_experience
        if years:
            filters["total_years_experience"] = years

        return filters


class CandidateSearchHit(BaseModel):
    candidate_id: str
    score: float
    profile: CandidateProfile


@router.post("/match", response_model=list[MatchResult])
async def match_candidates(request: MatchRequest) -> list[MatchResult]:
    return match(request.job, top_k=request.top_k, top_n=request.top_n, filters=request.filters or None)


@router.post("/candidates", response_model=list[CandidateSearchHit])
async def search_pool(request: CandidateSearchRequest) -> list[CandidateSearchHit]:
    """Search the talent pool directly — "who do we already have" rather than "who fits this role"."""
    found = search_candidates(
        query=request.query, filters=request.to_filters() or None, top_n=request.top_n
    )
    return [
        CandidateSearchHit(candidate_id=record.candidate_id, score=score, profile=record.profile)
        for record, score in found
    ]

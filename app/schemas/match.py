"""Match result schema — the explainable score breakdown returned by the API."""
from pydantic import BaseModel

from app.schemas.candidate import CandidateProfile


class ScoreBreakdown(BaseModel):
    skills: float
    experience: float
    education: float
    weighted_total: float
    retrieval_score: float
    rerank_score: float


class MatchResult(BaseModel):
    candidate_id: str
    candidate: CandidateProfile
    breakdown: ScoreBreakdown

    @property
    def final_score(self) -> float:
        return self.breakdown.weighted_total

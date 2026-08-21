"""Match result schemas — the explainable breakdown returned by the API.

A bare score tells a recruiter a candidate ranked 0.62 but not why, and not what
would change it. Every component therefore ships the evidence it was computed
from: which required skills were present, which were missing, how the candidate's
tenure compared to the bar, and whether the education requirement was met.
"""
from pydantic import BaseModel, Field

from app.schemas.candidate import CandidateProfile


class SkillEvidence(BaseModel):
    matched_required: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    matched_preferred: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    extra: list[str] = Field(
        default_factory=list, description="Candidate skills the posting never asked for"
    )

    @property
    def meets_all_required(self) -> bool:
        return not self.missing_required


class ExperienceEvidence(BaseModel):
    candidate_years: float = Field(0.0, description="Total tenure across every role")
    relevant_years: float = Field(0.0, description="Tenure weighted by role relevance")
    required_years: float = 0.0
    meets_requirement: bool = True
    relevant_roles: list[str] = Field(default_factory=list)
    unrelated_roles: list[str] = Field(default_factory=list)
    stale_roles: list[str] = Field(
        default_factory=list, description="Relevant work that ended long enough ago to be discounted"
    )
    most_recent_relevant_year: float | None = None


class EducationEvidence(BaseModel):
    required: str = ""
    required_degree_level: str = ""
    matched_degree: str | None = None
    matched_field: str | None = None
    meets_requirement: bool = True


class CertificationEvidence(BaseModel):
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    meets_all_required: bool = True


class MatchEvidence(BaseModel):
    skills: SkillEvidence
    experience: ExperienceEvidence
    education: EducationEvidence
    certifications: CertificationEvidence = Field(default_factory=CertificationEvidence)


class ScoreBreakdown(BaseModel):
    skills: float
    experience: float
    education: float
    certifications: float = 1.0
    weighted_total: float
    retrieval_score: float
    rerank_score: float


class MatchResult(BaseModel):
    candidate_id: str
    candidate: CandidateProfile
    breakdown: ScoreBreakdown
    evidence: MatchEvidence

    @property
    def final_score(self) -> float:
        return self.breakdown.weighted_total


class JobMatchResult(BaseModel):
    """The reverse direction: one candidate scored against a stored posting."""

    job_id: str
    job_title: str
    breakdown: ScoreBreakdown
    evidence: MatchEvidence

    @property
    def final_score(self) -> float:
        return self.breakdown.weighted_total

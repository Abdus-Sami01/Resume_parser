"""Job description schema — separates required vs. preferred and carries section weights."""
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    OPEN = "open"
    ON_HOLD = "on_hold"
    FILLED = "filled"
    CLOSED = "closed"


# Only an open role is actively recruiting. The rest stay readable — closing a
# req must not erase its pipeline history — but they stop pulling candidates
# toward work nobody is hiring for.
ACTIVE_JOB_STATUSES: set[JobStatus] = {JobStatus.OPEN}


class JobWeights(BaseModel):
    experience: float = 0.5
    skills: float = 0.4
    education: float = 0.1
    # Defaults to zero so every existing weighting still sums to 1.0. The extractor
    # raises it only for postings that actually name a required certification.
    certifications: float = 0.0

    @model_validator(mode="after")
    def _sum_to_one(self) -> "JobWeights":
        total = self.experience + self.skills + self.education + self.certifications
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"job weights must sum to 1.0, got {total}")
        return self


class JobProfile(BaseModel):
    title: str
    status: JobStatus = JobStatus.OPEN
    company: str = ""
    location: str = ""
    remote: bool = False
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = 0
    required_education: str = Field(
        "", description="Field of study or degree level the posting requires"
    )
    required_degree_level: str = Field("", description="e.g. bachelor, master, phd")
    required_certifications: list[str] = Field(default_factory=list)
    description: str = ""
    weights: JobWeights = Field(default_factory=JobWeights)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_JOB_STATUSES

    @property
    def all_skills(self) -> list[str]:
        return [*self.required_skills, *self.preferred_skills]

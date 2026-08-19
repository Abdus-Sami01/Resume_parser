"""Job description schema — separates required vs. preferred and carries section weights."""
from pydantic import BaseModel, Field, model_validator


class JobWeights(BaseModel):
    experience: float = 0.5
    skills: float = 0.4
    education: float = 0.1

    @model_validator(mode="after")
    def _sum_to_one(self) -> "JobWeights":
        total = self.experience + self.skills + self.education
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"job weights must sum to 1.0, got {total}")
        return self


class JobProfile(BaseModel):
    title: str
    company: str = ""
    location: str = ""
    remote: bool = False
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = 0
    required_education: str = ""
    description: str = ""
    weights: JobWeights = Field(default_factory=JobWeights)

    @property
    def all_skills(self) -> list[str]:
        return [*self.required_skills, *self.preferred_skills]

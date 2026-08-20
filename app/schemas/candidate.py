"""Structured candidate profile schema — the strict target for resume extraction."""
from pydantic import BaseModel, EmailStr, Field, computed_field


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str = ""
    graduation_year: int | None = None


class Experience(BaseModel):
    company: str
    role: str
    years: float = Field(ge=0, description="Duration of this role, in years")
    achievements: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    name: str
    email: EmailStr | None = None
    phone: str | None = None
    location: str | None = None
    skills: list[str] = Field(default_factory=list, description="Standardized professional skills")
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    summary: str = ""

    @computed_field  # serialized, so clients need not re-sum the experience array
    @property
    def total_years_experience(self) -> float:
        return round(sum(exp.years for exp in self.experience), 2)

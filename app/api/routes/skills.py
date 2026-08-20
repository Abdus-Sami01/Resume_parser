"""Skill taxonomy management.

The taxonomy decides whether "K8s" on a resume matches "Kubernetes" in a posting,
so a deployment that cannot extend it silently mis-scores its own niche skills.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.taxonomy.skill_standardizer import get_skill_standardizer

router = APIRouter(prefix="/skills", tags=["skills"])


class SkillEntry(BaseModel):
    skill: str = Field(..., min_length=1)
    aliases: list[str] = Field(default_factory=list)


class TaxonomyOut(BaseModel):
    total: int
    skills: dict[str, list[str]]


class StandardizeRequest(BaseModel):
    values: list[str]


@router.get("", response_model=TaxonomyOut)
async def list_skills() -> TaxonomyOut:
    known = get_skill_standardizer().known_skills()
    return TaxonomyOut(total=len(known), skills=known)


@router.post("", response_model=SkillEntry, status_code=201)
async def add_skill(entry: SkillEntry) -> SkillEntry:
    """Adds a skill, or merges aliases into an existing one. Persists to the overlay file."""
    try:
        result = get_skill_standardizer().add_skill(entry.skill, entry.aliases)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SkillEntry(**result)


@router.delete("/{skill}", status_code=204)
async def remove_skill(skill: str) -> None:
    if not get_skill_standardizer().remove_skill(skill):
        raise HTTPException(status_code=404, detail="skill not in taxonomy")


@router.post("/standardize", response_model=dict[str, str | None])
async def standardize(request: StandardizeRequest) -> dict[str, str | None]:
    """Previews how raw strings resolve — useful for debugging why a match scored low."""
    standardizer = get_skill_standardizer()
    return {value: standardizer.standardize(value) for value in request.values}

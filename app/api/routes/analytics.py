"""Aggregate reporting over the talent pool and the open postings."""
from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.search.analytics import build_overview, build_parse_coverage

router = APIRouter(prefix="/analytics", tags=["analytics"])


class SkillCountOut(BaseModel):
    skill: str
    candidates: int
    share: float


class SkillGapOut(BaseModel):
    skill: str
    required_by_jobs: int
    candidates_with_skill: int
    coverage: float


class PoolOverviewOut(BaseModel):
    total_candidates: int
    total_jobs: int
    distinct_skills: int
    median_years_experience: float
    experience_distribution: dict[str, int]
    top_skills: list[SkillCountOut]
    skill_gaps: list[SkillGapOut]


@router.get("/overview", response_model=PoolOverviewOut)
async def pool_overview(
    top_skills: int = Query(10, ge=1, le=100),
    gaps: int = Query(10, ge=1, le=100),
) -> PoolOverviewOut:
    """What the pool contains, and which posted requirements it cannot cover.

    `skill_gaps` is the actionable half: requirements ranked by worst coverage,
    naming what to source for rather than restating what the pool already has.
    """
    overview = build_overview(top_skills_limit=top_skills, gap_limit=gaps)
    return PoolOverviewOut(
        total_candidates=overview.total_candidates,
        total_jobs=overview.total_jobs,
        distinct_skills=overview.distinct_skills,
        median_years_experience=overview.median_years_experience,
        experience_distribution=overview.experience_distribution,
        top_skills=[SkillCountOut(**vars(s)) for s in overview.top_skills],
        skill_gaps=[SkillGapOut(**vars(g)) for g in overview.skill_gaps],
    )


class FieldCoverageOut(BaseModel):
    field: str
    present: int
    missing: int
    coverage: float
    scorer_assumption: str


class ThinProfileOut(BaseModel):
    candidate_id: str
    name: str
    missing_fields: list[str]


class ParseCoverageOut(BaseModel):
    total_candidates: int
    total_roles: int
    fields: list[FieldCoverageOut]
    needs_review: list[ThinProfileOut]


@router.get("/parse-coverage", response_model=ParseCoverageOut)
async def parse_coverage(review: int = Query(10, ge=1, le=100)) -> ParseCoverageOut:
    """How much of each resume the extractor actually recovered.

    The matcher credits what it cannot read rather than penalising the candidate
    for it, which is correct per person and hides a systemic problem: a pool with
    thin extraction ranks on assumptions. `scorer_assumption` names what each
    missing field buys, and `needs_review` is a re-extraction queue ordered by how
    little was recovered.
    """
    coverage = build_parse_coverage(review_limit=review)
    return ParseCoverageOut(
        total_candidates=coverage.total_candidates,
        total_roles=coverage.total_roles,
        fields=[FieldCoverageOut(**vars(f)) for f in coverage.fields],
        needs_review=[ThinProfileOut(**vars(t)) for t in coverage.needs_review],
    )

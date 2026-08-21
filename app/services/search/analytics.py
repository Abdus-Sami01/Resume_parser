"""Talent-pool analytics: what the stored candidates and postings say in aggregate.

Individual matches answer "is this person a fit". These answer the questions a
hiring manager asks after seeing a few of them — what is actually in the pool,
and which requirements the pool cannot currently satisfy.
"""
from collections import Counter
from dataclasses import dataclass, field

from app.db.candidate_store import get_candidate_store
from app.db.job_store import get_job_store

# Buckets chosen to read like seniority bands rather than raw numbers.
_EXPERIENCE_BANDS: list[tuple[str, float, float]] = [
    ("0-2", 0.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10+", 10.0, float("inf")),
]


@dataclass
class SkillCount:
    skill: str
    candidates: int
    share: float


@dataclass
class SkillGap:
    skill: str
    required_by_jobs: int
    candidates_with_skill: int
    coverage: float


@dataclass
class PoolOverview:
    total_candidates: int
    total_jobs: int  # open roles only
    distinct_skills: int
    median_years_experience: float
    experience_distribution: dict[str, int] = field(default_factory=dict)
    top_skills: list[SkillCount] = field(default_factory=list)
    skill_gaps: list[SkillGap] = field(default_factory=list)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[midpoint], 2)
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 2)


def _band_for(years: float) -> str:
    for label, lower, upper in _EXPERIENCE_BANDS:
        if lower <= years < upper:
            return label
    return _EXPERIENCE_BANDS[-1][0]


def build_overview(top_skills_limit: int = 10, gap_limit: int = 10) -> PoolOverview:
    candidates = get_candidate_store().all()
    jobs = [r for r in get_job_store().all() if r.profile.is_active]

    skill_counts: Counter = Counter()
    experience_bands = {label: 0 for label, _, _ in _EXPERIENCE_BANDS}
    years: list[float] = []

    for record in candidates:
        # A skill listed twice on one resume is still one candidate who has it.
        for skill in set(record.profile.skills):
            skill_counts[skill] += 1
        candidate_years = record.profile.total_years_experience
        years.append(candidate_years)
        experience_bands[_band_for(candidate_years)] += 1

    total = len(candidates)
    top_skills = [
        SkillCount(skill=skill, candidates=count, share=round(count / total, 4) if total else 0.0)
        for skill, count in skill_counts.most_common(top_skills_limit)
    ]

    return PoolOverview(
        total_candidates=total,
        total_jobs=len(jobs),
        distinct_skills=len(skill_counts),
        median_years_experience=_median(years),
        experience_distribution=experience_bands,
        top_skills=top_skills,
        skill_gaps=_skill_gaps(skill_counts, total, gap_limit),
    )


def _skill_gaps(skill_counts: Counter, total_candidates: int, limit: int) -> list[SkillGap]:
    """Requirements across open postings ranked by how poorly the pool covers them.

    This is the actionable half of the report: it names what to source for, rather
    than restating what the pool already has.
    """
    demand: Counter = Counter()
    for record in get_job_store().all():
        # Counting closed reqs would send sourcing after roles nobody is hiring for.
        if not record.profile.is_active:
            continue
        for skill in set(record.profile.required_skills):
            demand[skill] += 1

    gaps = [
        SkillGap(
            skill=skill,
            required_by_jobs=job_count,
            candidates_with_skill=skill_counts.get(skill, 0),
            coverage=round(skill_counts.get(skill, 0) / total_candidates, 4) if total_candidates else 0.0,
        )
        for skill, job_count in demand.items()
    ]

    # Worst coverage first, breaking ties by how many postings demand the skill.
    gaps.sort(key=lambda gap: (gap.coverage, -gap.required_by_jobs))
    return gaps[:limit]

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
class FieldCoverage:
    field: str
    present: int
    missing: int
    coverage: float
    scorer_assumption: str


@dataclass
class ThinProfile:
    candidate_id: str
    name: str
    missing_fields: list[str]


@dataclass
class ParseCoverage:
    total_candidates: int
    total_roles: int
    fields: list[FieldCoverage] = field(default_factory=list)
    needs_review: list[ThinProfile] = field(default_factory=list)


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


# Each entry pairs a field with what the matcher does when it is missing. The
# second half is the point: the scorer credits parse gaps in full
# (`FULL_CREDIT_ON_PARSE_GAP`), so a pool with poor coverage produces scores that
# look confident while resting on assumptions. This report is how that stays
# visible instead of being silently absorbed into the rankings.
_ROLE_FIELDS: list[tuple[str, str]] = [
    ("experience.role", "tenure counts as fully relevant to any posting"),
    ("experience.end_year", "the role is treated as current, so no recency discount applies"),
    ("experience.achievements", "skills in that role are never dated, so none are marked stale"),
]
_PROFILE_FIELDS: list[tuple[str, str]] = [
    ("email", "the candidate cannot be contacted from the record"),
    ("experience", "no years requirement can be evaluated against them"),
    ("education", "an education requirement scores zero rather than being assumed"),
    ("skills", "every required skill reads as missing"),
]


def build_parse_coverage(review_limit: int = 10) -> ParseCoverage:
    """How much of the pool the extractor actually recovered, field by field.

    Matching quality is bounded by extraction quality, and the bound is invisible
    from the match results themselves: a role whose title we could not read scores
    as fully relevant, which is the right call per candidate and a misleading one
    in aggregate. Reading coverage next to the rankings is what separates "these
    candidates fit" from "we could not read enough to say otherwise".
    """
    records = get_candidate_store().all()

    role_present = {name: 0 for name, _ in _ROLE_FIELDS}
    profile_present = {name: 0 for name, _ in _PROFILE_FIELDS}
    total_roles = 0
    thin: list[ThinProfile] = []

    for record in records:
        profile = record.profile
        missing: list[str] = []

        for name, _ in _PROFILE_FIELDS:
            value = getattr(profile, name, None)
            if value:
                profile_present[name] += 1
            else:
                missing.append(name)

        for entry in profile.experience:
            total_roles += 1
            readable_role = entry.role and entry.role.strip().lower() not in {"", "unknown"}
            if readable_role:
                role_present["experience.role"] += 1
            # A current role has no end date to recover, so counting it as missing
            # would report the extractor failing on something that is not there.
            if entry.is_current or entry.end_year is not None:
                role_present["experience.end_year"] += 1
            if entry.achievements:
                role_present["experience.achievements"] += 1
            else:
                # Naming the role is what makes the queue actionable; when that is
                # the field we failed to read, the employer is the next best handle.
                label = entry.role if readable_role else (entry.company or "an unnamed role")
                missing.append(f"achievements for {label}")

        if missing:
            thin.append(
                ThinProfile(
                    candidate_id=record.candidate_id, name=profile.name, missing_fields=missing
                )
            )

    fields = [
        _coverage(name, profile_present[name], len(records), assumption)
        for name, assumption in _PROFILE_FIELDS
    ] + [
        _coverage(name, role_present[name], total_roles, assumption)
        for name, assumption in _ROLE_FIELDS
    ]

    # Worst-parsed first: this list is a re-extraction queue, not a directory.
    thin.sort(key=lambda profile: len(profile.missing_fields), reverse=True)

    return ParseCoverage(
        total_candidates=len(records),
        total_roles=total_roles,
        fields=fields,
        needs_review=thin[:review_limit],
    )


def _coverage(name: str, present: int, total: int, assumption: str) -> FieldCoverage:
    return FieldCoverage(
        field=name,
        present=present,
        missing=total - present,
        # An empty pool has nothing missing; reporting 0% coverage would read as
        # a broken extractor rather than as no data.
        coverage=round(present / total, 4) if total else 1.0,
        scorer_assumption=assumption,
    )

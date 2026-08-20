"""Orchestrates the full matching pipeline: index candidates, then two-stage match a job against them.

Stage 1 (retrieval): hybrid dense+sparse search pulls the top-K candidates.
Stage 2 (reranking): each of the top-K is scored by a cross-encoder-style
reranker against the exact job description, then blended with the
section-weighted structured score (skills/experience/education) for the
final, explainable ranking.
"""
import hashlib
import uuid

from app.config import get_settings
from app.db.candidate_store import CandidateRecord, get_candidate_store
from app.db.job_store import get_job_store
from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile
from app.schemas.match import (
    EducationEvidence,
    ExperienceEvidence,
    JobMatchResult,
    MatchEvidence,
    MatchResult,
    ScoreBreakdown,
    SkillEvidence,
)
from app.services.search.embeddings import get_embedding_client
from app.services.search.reranker import get_reranker
from app.services.search.vector_store import get_vector_store


def content_fingerprint(raw_text: str) -> str:
    """Stable id for a resume's text, so re-uploading one file does not clone the candidate."""
    normalized = " ".join(raw_text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def index_candidate(profile: CandidateProfile, raw_text: str, candidate_id: str | None = None) -> str:
    embedder = get_embedding_client()
    vector_store = get_vector_store()
    candidate_store = get_candidate_store()

    fingerprint = content_fingerprint(raw_text)
    # Re-uploading the same resume must update that candidate, not add a second copy
    # of the same person to every result list.
    candidate_id = candidate_id or candidate_store.find_by_fingerprint(fingerprint) or str(uuid.uuid4())

    searchable_text = _candidate_searchable_text(profile, raw_text)
    dense_vector = embedder.embed(searchable_text)

    # The record is saved first: a failure after this point leaves a candidate that is
    # merely unindexed, whereas the reverse order leaves a phantom vector that wins a
    # retrieval slot and then resolves to nothing.
    candidate_store.save(
        CandidateRecord(
            candidate_id=candidate_id, profile=profile, raw_text=raw_text, fingerprint=fingerprint
        )
    )
    vector_store.upsert(
        doc_id=candidate_id,
        dense_vector=dense_vector,
        text=searchable_text,
        payload={
            "candidate_id": candidate_id,
            "skills": profile.skills,
            "location": profile.location or "",
            "total_years_experience": profile.total_years_experience,
            "certifications": profile.certifications,
        },
    )

    return candidate_id


def match(
    job: JobProfile,
    top_k: int | None = None,
    top_n: int | None = None,
    filters: dict | None = None,
) -> list[MatchResult]:
    """Two-stage match. `filters` are hard metadata constraints applied during retrieval."""
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    embedder = get_embedding_client()
    vector_store = get_vector_store()
    reranker = get_reranker()
    candidate_store = get_candidate_store()

    query_text = _job_searchable_text(job)
    query_dense = embedder.embed(query_text)

    stage1_hits = vector_store.search(
        query_dense=query_dense, query_text=query_text, top_k=top_k, filters=filters
    )

    # Resolve every retrieved hit first so the reranker sees one batch rather than
    # one forward pass per candidate.
    retrieved = [(hit, candidate_store.get(hit.id)) for hit in stage1_hits]
    retrieved = [(hit, record) for hit, record in retrieved if record is not None]
    if not retrieved:
        return []

    rerank_scores = reranker.score_batch(query_text, [record.raw_text for _, record in retrieved])

    results: list[MatchResult] = []
    for (hit, record), rerank_score in zip(retrieved, rerank_scores):
        breakdown, evidence = _score_breakdown(
            job, record.profile, retrieval_score=hit.score, rerank_score=rerank_score
        )
        results.append(
            MatchResult(
                candidate_id=record.candidate_id,
                candidate=record.profile,
                breakdown=breakdown,
                evidence=evidence,
            )
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_n]


def _score_breakdown(
    job: JobProfile, candidate: CandidateProfile, *, retrieval_score: float, rerank_score: float
) -> tuple[ScoreBreakdown, MatchEvidence]:
    """Scores every component and returns the evidence each score was derived from."""
    skills_score, skills_evidence = _skills_score(job, candidate)
    experience_score, experience_evidence = _experience_score(job, candidate)
    education_score, education_evidence = _education_score(job, candidate)

    structured_score = (
        job.weights.skills * skills_score
        + job.weights.experience * experience_score
        + job.weights.education * education_score
    )
    weighted_total = 0.5 * rerank_score + 0.5 * structured_score

    breakdown = ScoreBreakdown(
        skills=skills_score,
        experience=experience_score,
        education=education_score,
        weighted_total=weighted_total,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
    )
    evidence = MatchEvidence(
        skills=skills_evidence, experience=experience_evidence, education=education_evidence
    )
    return breakdown, evidence


def _skills_score(job: JobProfile, candidate: CandidateProfile) -> tuple[float, SkillEvidence]:
    candidate_skills = set(candidate.skills)
    required = set(job.required_skills)
    preferred = set(job.preferred_skills)

    evidence = SkillEvidence(
        matched_required=sorted(required & candidate_skills),
        missing_required=sorted(required - candidate_skills),
        matched_preferred=sorted(preferred & candidate_skills),
        missing_preferred=sorted(preferred - candidate_skills),
        extra=sorted(candidate_skills - required - preferred),
    )

    if not required and not preferred:
        return 1.0, evidence

    required_hit = len(evidence.matched_required) / len(required) if required else 1.0
    preferred_hit = len(evidence.matched_preferred) / len(preferred) if preferred else 1.0

    # Required coverage dominates; preferred skills nudge the score up.
    return round(0.8 * required_hit + 0.2 * preferred_hit, 4), evidence


def _experience_score(
    job: JobProfile, candidate: CandidateProfile
) -> tuple[float, ExperienceEvidence]:
    candidate_years = candidate.total_years_experience
    required_years = job.min_years_experience

    if required_years <= 0:
        return 1.0, ExperienceEvidence(
            candidate_years=candidate_years, required_years=0.0, meets_requirement=True
        )

    score = round(min(candidate_years / required_years, 1.0), 4)
    return score, ExperienceEvidence(
        candidate_years=candidate_years,
        required_years=required_years,
        meets_requirement=candidate_years >= required_years,
    )


def _education_score(
    job: JobProfile, candidate: CandidateProfile
) -> tuple[float, EducationEvidence]:
    if not job.required_education:
        return 1.0, EducationEvidence(required="", matched_degree=None, meets_requirement=True)

    required = job.required_education.lower()
    for education in candidate.education:
        if required in education.degree.lower() or required in education.field_of_study.lower():
            return 1.0, EducationEvidence(
                required=job.required_education,
                matched_degree=education.degree,
                meets_requirement=True,
            )

    # A degree that does not match still beats no listed education at all.
    partial = 0.5 if candidate.education else 0.0
    return partial, EducationEvidence(
        required=job.required_education, matched_degree=None, meets_requirement=False
    )


def _candidate_searchable_text(profile: CandidateProfile, raw_text: str) -> str:
    skills = ", ".join(profile.skills)
    return f"{profile.summary}\nSkills: {skills}\n{raw_text}"


def _job_searchable_text(job: JobProfile) -> str:
    skills = ", ".join(job.all_skills)
    return f"{job.title}\nRequired skills: {skills}\n{job.description}"


def delete_candidate(candidate_id: str) -> bool:
    """Removes a candidate from both the profile store and the index.

    Resumes are personal data, so erasure has to reach the vector index and the
    pipeline history too — a profile deleted from one store and left in another is
    still discoverable, and stage notes carry the candidate id alongside free text
    a reviewer wrote about them.
    """
    from app.db.pipeline_store import get_pipeline_store

    candidate_store = get_candidate_store()
    vector_store = get_vector_store()

    existed = candidate_store.delete(candidate_id)
    vector_store.delete(candidate_id)
    get_pipeline_store().delete_for_candidate(candidate_id)
    return existed


def match_jobs_for_candidate(
    candidate_id: str, top_n: int | None = None
) -> list[JobMatchResult]:
    """The reverse direction: rank stored postings for one candidate.

    Postings are scored by a linear scan rather than a second vector index.
    A deployment holds orders of magnitude fewer open roles than resumes, so the
    scan stays cheap; if that stops being true this wants its own index.
    """
    settings = get_settings()
    top_n = top_n or settings.rerank_top_n

    candidate_store = get_candidate_store()
    job_store = get_job_store()
    reranker = get_reranker()

    record = candidate_store.get(candidate_id)
    if record is None:
        return []

    job_records = job_store.all()
    if not job_records:
        return []

    # One batched rerank call across every posting, same as the forward direction.
    rerank_scores = reranker.score_batch(
        record.raw_text, [_job_searchable_text(job.profile) for job in job_records]
    )

    results: list[JobMatchResult] = []
    for job_record, rerank_score in zip(job_records, rerank_scores):
        breakdown, evidence = _score_breakdown(
            job_record.profile, record.profile, retrieval_score=0.0, rerank_score=rerank_score
        )
        results.append(
            JobMatchResult(
                job_id=job_record.job_id,
                job_title=job_record.profile.title,
                breakdown=breakdown,
                evidence=evidence,
            )
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_n]


def reindex_all() -> int:
    """Rebuilds the vector index from persisted candidate records.

    Durable records plus an in-process index is an inconsistent pair after a
    restart: the candidate is listed by the API but matches nothing, which reads
    as "no results" rather than as a broken index. Rebuilding re-embeds each
    stored resume, so it costs one embedding call per candidate.
    """
    candidate_store = get_candidate_store()
    embedder = get_embedding_client()
    vector_store = get_vector_store()

    indexed = 0
    for record in candidate_store.all():
        searchable_text = _candidate_searchable_text(record.profile, record.raw_text)
        vector_store.upsert(
            doc_id=record.candidate_id,
            dense_vector=embedder.embed(searchable_text),
            text=searchable_text,
            payload={
                "candidate_id": record.candidate_id,
                "skills": record.profile.skills,
                "location": record.profile.location or "",
                "total_years_experience": record.profile.total_years_experience,
                "certifications": record.profile.certifications,
            },
        )
        indexed += 1

    return indexed


def reindex_if_index_is_empty() -> int:
    """Rebuilds only when records exist but the index does not.

    Checking the index rather than the configured backend keeps this correct for
    every combination: Qdrant persists its own vectors, so it reports a non-zero
    count and is left alone.
    """
    if get_vector_store().count() > 0:
        return 0
    if not get_candidate_store().all():
        return 0
    return reindex_all()


def search_candidates(
    query: str = "",
    filters: dict | None = None,
    top_n: int | None = None,
) -> list[tuple[CandidateRecord, float]]:
    """Free-text + structured search over the talent pool, with no job posting involved.

    The job-driven path answers "who fits this posting". This answers the other
    half of the workflow — "who do we already have" — where a recruiter has
    criteria in mind rather than a written role: Python, five years, remote.

    With no query text the dense half has nothing to rank on, so results fall back
    to the pool order and the structured filters do all the work.
    """
    settings = get_settings()
    top_n = top_n or settings.rerank_top_n

    embedder = get_embedding_client()
    vector_store = get_vector_store()
    candidate_store = get_candidate_store()

    hits = vector_store.search(
        query_dense=embedder.embed(query) if query else [],
        query_text=query,
        top_k=max(top_n, settings.retrieval_top_k),
        filters=filters,
    )

    found: list[tuple[CandidateRecord, float]] = []
    for hit in hits:
        record = candidate_store.get(hit.id)
        if record is not None:
            found.append((record, hit.score))

    return found[:top_n]


def find_similar_candidates(
    candidate_id: str, top_n: int | None = None
) -> list[tuple[CandidateRecord, float]]:
    """"More people like this one" — the standard follow-up to a good hire or a good match.

    The reference candidate is excluded from its own results, which it would
    otherwise top by a wide margin.
    """
    settings = get_settings()
    top_n = top_n or settings.rerank_top_n

    candidate_store = get_candidate_store()
    reference = candidate_store.get(candidate_id)
    if reference is None:
        return []

    embedder = get_embedding_client()
    vector_store = get_vector_store()

    reference_text = _candidate_searchable_text(reference.profile, reference.raw_text)
    hits = vector_store.search(
        query_dense=embedder.embed(reference_text),
        query_text=reference_text,
        # One extra slot, since the reference itself is expected back and dropped.
        top_k=top_n + 1,
    )

    similar: list[tuple[CandidateRecord, float]] = []
    for hit in hits:
        if hit.id == candidate_id:
            continue
        record = candidate_store.get(hit.id)
        if record is not None:
            similar.append((record, hit.score))

    return similar[:top_n]

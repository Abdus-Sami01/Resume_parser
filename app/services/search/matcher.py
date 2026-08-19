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
from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile
from app.schemas.match import MatchResult, ScoreBreakdown
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

    results = [
        MatchResult(
            candidate_id=record.candidate_id,
            candidate=record.profile,
            breakdown=_score_breakdown(
                job, record.profile, retrieval_score=hit.score, rerank_score=rerank_score
            ),
        )
        for (hit, record), rerank_score in zip(retrieved, rerank_scores)
    ]

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_n]


def _score_breakdown(
    job: JobProfile, candidate: CandidateProfile, *, retrieval_score: float, rerank_score: float
) -> ScoreBreakdown:
    skills_score = _skills_score(job, candidate)
    experience_score = _experience_score(job, candidate)
    education_score = _education_score(job, candidate)

    structured_score = (
        job.weights.skills * skills_score
        + job.weights.experience * experience_score
        + job.weights.education * education_score
    )
    weighted_total = 0.5 * rerank_score + 0.5 * structured_score

    return ScoreBreakdown(
        skills=skills_score,
        experience=experience_score,
        education=education_score,
        weighted_total=weighted_total,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
    )


def _skills_score(job: JobProfile, candidate: CandidateProfile) -> float:
    candidate_skills = set(candidate.skills)
    required = set(job.required_skills)
    preferred = set(job.preferred_skills)

    if not required and not preferred:
        return 1.0

    required_hit = len(required & candidate_skills) / len(required) if required else 1.0
    preferred_hit = len(preferred & candidate_skills) / len(preferred) if preferred else 1.0

    # Required coverage dominates; preferred skills nudge the score up.
    return round(0.8 * required_hit + 0.2 * preferred_hit, 4)


def _experience_score(job: JobProfile, candidate: CandidateProfile) -> float:
    if job.min_years_experience <= 0:
        return 1.0
    return round(min(candidate.total_years_experience / job.min_years_experience, 1.0), 4)


def _education_score(job: JobProfile, candidate: CandidateProfile) -> float:
    if not job.required_education:
        return 1.0
    required = job.required_education.lower()
    for edu in candidate.education:
        if required in edu.degree.lower() or required in edu.field_of_study.lower():
            return 1.0
    return 0.5 if candidate.education else 0.0


def _candidate_searchable_text(profile: CandidateProfile, raw_text: str) -> str:
    skills = ", ".join(profile.skills)
    return f"{profile.summary}\nSkills: {skills}\n{raw_text}"


def _job_searchable_text(job: JobProfile) -> str:
    skills = ", ".join(job.all_skills)
    return f"{job.title}\nRequired skills: {skills}\n{job.description}"

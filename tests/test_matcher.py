from app.schemas.candidate import CandidateProfile, Experience
from app.schemas.job import JobProfile, JobWeights
from app.services.search.matcher import index_candidate, match

STRONG_CANDIDATE = CandidateProfile(
    name="Strong Match",
    skills=["python", "fastapi", "postgresql", "aws", "docker"],
    experience=[Experience(company="Acme", role="Backend Engineer", years=6)],
)
STRONG_TEXT = "Senior backend engineer with 6 years building Python FastAPI services on AWS."

WEAK_CANDIDATE = CandidateProfile(
    name="Weak Match",
    skills=["photoshop", "illustrator"],
    experience=[Experience(company="Studio", role="Graphic Designer", years=6)],
)
WEAK_TEXT = "Graphic designer with 6 years of experience in Photoshop and Illustrator."

JOB = JobProfile(
    title="Backend Engineer",
    required_skills=["python", "fastapi", "postgresql"],
    preferred_skills=["aws"],
    min_years_experience=3,
    description="Looking for a backend engineer skilled in Python, FastAPI and PostgreSQL, AWS a plus.",
    weights=JobWeights(experience=0.5, skills=0.4, education=0.1),
)


def test_matcher_ranks_relevant_candidate_above_irrelevant_one():
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")
    index_candidate(WEAK_CANDIDATE, WEAK_TEXT, candidate_id="weak")

    results = match(JOB)

    assert [r.candidate_id for r in results][0] == "strong"
    strong_result = next(r for r in results if r.candidate_id == "strong")
    weak_result = next(r for r in results if r.candidate_id == "weak")
    assert strong_result.final_score > weak_result.final_score


def test_matcher_score_breakdown_is_bounded_and_explainable():
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")

    results = match(JOB)
    breakdown = results[0].breakdown

    assert 0.0 <= breakdown.skills <= 1.0
    assert 0.0 <= breakdown.experience <= 1.0
    assert 0.0 <= breakdown.education <= 1.0
    assert breakdown.skills > 0.9  # all required skills present


def test_matcher_respects_top_n():
    for i in range(5):
        index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id=f"c{i}")

    results = match(JOB, top_n=2)
    assert len(results) == 2

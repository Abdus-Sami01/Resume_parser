import subprocess
import sys

from app.schemas.candidate import CandidateProfile, Experience
from app.schemas.job import JobProfile, JobWeights
from app.services.search.matcher import index_candidate, match
from app.services.search.vector_store import payload_matches_filters, sparse_index

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


def test_metadata_filter_drops_candidates_lacking_a_required_skill():
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")
    index_candidate(WEAK_CANDIDATE, WEAK_TEXT, candidate_id="weak")

    results = match(JOB, filters={"skills": ["python"]})

    assert [r.candidate_id for r in results] == ["strong"]


def test_metadata_filter_combines_constraints_conjunctively():
    remote_candidate = STRONG_CANDIDATE.model_copy(update={"location": "Remote"})
    onsite_candidate = STRONG_CANDIDATE.model_copy(update={"location": "Berlin"})
    index_candidate(remote_candidate, STRONG_TEXT, candidate_id="remote")
    index_candidate(onsite_candidate, STRONG_TEXT, candidate_id="onsite")

    results = match(JOB, filters={"skills": ["python"], "location": "Remote"})

    assert [r.candidate_id for r in results] == ["remote"]


def test_no_filters_leaves_retrieval_untouched():
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")
    index_candidate(WEAK_CANDIDATE, WEAK_TEXT, candidate_id="weak")

    assert len(match(JOB, filters=None)) == 2
    assert len(match(JOB, filters={})) == 2


def test_sparse_index_is_stable_across_interpreter_processes():
    """Guards the Qdrant sparse vectors: a worker and the API must agree on term indices."""
    script = (
        "from app.services.search.vector_store import sparse_index;"
        "print(sparse_index('python'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }

    assert len(runs) == 1
    assert runs.pop() == str(sparse_index("python"))


def test_qdrant_backend_fuses_dense_and_sparse_and_applies_filters(monkeypatch):
    """Runs the production backend against qdrant-client's embedded mode."""
    import uuid

    from app.config import get_settings
    from app.services.search.vector_store import QdrantHybridVectorStore

    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()

    store = QdrantHybridVectorStore()
    backend_id, design_id = str(uuid.uuid4()), str(uuid.uuid4())
    store.upsert(backend_id, [1.0, 0, 0, 0], "python fastapi backend engineer", {"skills": ["python"]})
    store.upsert(design_id, [0, 1.0, 0, 0], "photoshop illustrator designer", {"skills": ["photoshop"]})

    hits = store.search([1.0, 0, 0, 0], "python backend", top_k=10)
    assert [h.id for h in hits][0] == backend_id

    filtered = store.search([1.0, 0, 0, 0], "python backend", top_k=10, filters={"skills": ["photoshop"]})
    assert [h.id for h in filtered] == [design_id]

    get_settings.cache_clear()


def test_payload_filter_requires_every_item_of_a_list_constraint():
    payload = {"skills": ["python", "aws"]}

    assert payload_matches_filters(payload, {"skills": ["python", "aws"]})
    assert not payload_matches_filters(payload, {"skills": ["python", "kafka"]})
    assert not payload_matches_filters({"skills": "python"}, {"skills": ["python"]})

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


def test_reranker_is_called_once_for_the_whole_candidate_batch(monkeypatch):
    """A per-pair reranker call costs one cross-encoder forward pass per candidate."""
    from app.services.search import matcher as matcher_module

    calls: list[list[str]] = []

    class CountingReranker:
        def score_batch(self, query: str, documents: list[str]) -> list[float]:
            calls.append(documents)
            return [0.5] * len(documents)

    monkeypatch.setattr(matcher_module, "get_reranker", lambda: CountingReranker())

    for i in range(3):
        index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id=f"c{i}")
    match(JOB)

    assert len(calls) == 1
    assert len(calls[0]) == 3


def test_rerank_scores_stay_aligned_with_their_candidates(monkeypatch):
    """Guards the zip between the retrieved batch and the reranker's score list."""
    from app.services.search import matcher as matcher_module

    class PerDocumentReranker:
        def score_batch(self, query: str, documents: list[str]) -> list[float]:
            # Score purely off the document so the expected pairing is unambiguous.
            return [1.0 if "photoshop" in doc.lower() else 0.0 for doc in documents]

    monkeypatch.setattr(matcher_module, "get_reranker", lambda: PerDocumentReranker())

    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")
    index_candidate(WEAK_CANDIDATE, WEAK_TEXT, candidate_id="weak")

    by_id = {r.candidate_id: r.breakdown.rerank_score for r in match(JOB)}

    assert by_id["weak"] == 1.0
    assert by_id["strong"] == 0.0


def test_reuploading_the_same_resume_updates_rather_than_clones():
    """Two records for one person would put them in every result list twice."""
    first = index_candidate(STRONG_CANDIDATE, STRONG_TEXT)
    second = index_candidate(STRONG_CANDIDATE, STRONG_TEXT)

    assert first == second
    assert [r.candidate_id for r in match(JOB)] == [first]


def test_whitespace_only_differences_are_treated_as_the_same_resume():
    first = index_candidate(STRONG_CANDIDATE, STRONG_TEXT)
    second = index_candidate(STRONG_CANDIDATE, f"  {STRONG_TEXT.replace(' ', '  ')}  \n")

    assert first == second


def test_a_genuinely_different_resume_gets_its_own_candidate():
    first = index_candidate(STRONG_CANDIDATE, STRONG_TEXT)
    second = index_candidate(WEAK_CANDIDATE, WEAK_TEXT)

    assert first != second
    assert len(match(JOB)) == 2


def test_non_ascii_text_survives_tokenization():
    """`[a-z0-9]+` turned 'José García' into ['jos', 'garc', 'a'] and dropped CJK entirely."""
    from app.services.search.vector_store import tokenize

    assert tokenize("José García") == ["jose", "garcia"]
    assert tokenize("Müller Schröder") == ["muller", "schroder"]
    assert "python" in tokenize("北京大学 Python")
    assert tokenize("Python FastAPI") == ["python", "fastapi"]  # ASCII unchanged


def test_accented_and_unaccented_spellings_match_each_other():
    from app.services.search.vector_store import tokenize

    assert tokenize("Jose") == tokenize("José")


def test_a_candidate_with_an_accented_name_is_retrievable():
    profile = STRONG_CANDIDATE.model_copy(update={"name": "José García"})
    index_candidate(profile, "José García. Senior Python engineer with FastAPI and AWS.")

    assert match(JOB)[0].candidate.name == "José García"


def test_dense_and_sparse_halves_share_one_tokenizer():
    """Divergent tokenizers would index and query the same text differently."""
    from app.services.search import embeddings, vector_store

    assert embeddings.tokenize is vector_store.tokenize


# --- Match evidence -------------------------------------------------------


def test_evidence_names_the_skills_behind_the_score():
    """A bare 0.62 tells a recruiter nothing about what is missing."""
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")

    evidence = match(JOB)[0].evidence.skills

    assert set(evidence.matched_required) == {"python", "fastapi", "postgresql"}
    assert evidence.missing_required == []
    assert evidence.matched_preferred == ["aws"]
    assert "docker" in evidence.extra  # held but never asked for
    assert evidence.meets_all_required is True


def test_evidence_lists_what_a_weak_candidate_is_missing():
    index_candidate(WEAK_CANDIDATE, WEAK_TEXT, candidate_id="weak")

    evidence = match(JOB)[0].evidence.skills

    assert evidence.matched_required == []
    assert set(evidence.missing_required) == {"python", "fastapi", "postgresql"}
    assert evidence.meets_all_required is False


def test_experience_evidence_reports_both_sides_of_the_comparison():
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")

    evidence = match(JOB)[0].evidence.experience

    assert evidence.candidate_years == 6.0
    assert evidence.required_years == 3.0
    assert evidence.meets_requirement is True


def test_education_evidence_records_the_degree_that_satisfied_the_requirement():
    from app.schemas.candidate import Education

    graduate = STRONG_CANDIDATE.model_copy(
        update={"education": [Education(institution="Stanford", degree="B.S.", field_of_study="Computer Science")]}
    )
    job = JOB.model_copy(update={"required_education": "Computer Science"})
    index_candidate(graduate, STRONG_TEXT, candidate_id="grad")

    evidence = match(job)[0].evidence.education

    assert evidence.meets_requirement is True
    assert evidence.matched_degree == "B.S."
    assert evidence.matched_field == "Computer Science"


def test_education_evidence_flags_an_unmet_requirement():
    job = JOB.model_copy(update={"required_education": "Computer Science"})
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="no-degree")

    evidence = match(job)[0].evidence.education

    assert evidence.meets_requirement is False
    assert evidence.matched_degree is None


# --- Erasure and reverse matching -----------------------------------------


def test_delete_candidate_clears_both_stores():
    from app.db.candidate_store import get_candidate_store
    from app.services.search.matcher import delete_candidate
    from app.services.search.vector_store import get_vector_store

    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="doomed")
    assert get_vector_store().count() == 1

    assert delete_candidate("doomed") is True
    assert get_candidate_store().get("doomed") is None
    assert get_vector_store().count() == 0
    assert match(JOB) == []


def test_deleting_an_absent_candidate_reports_false():
    from app.services.search.matcher import delete_candidate

    assert delete_candidate("never-existed") is False


def test_deleting_a_document_restores_term_document_frequencies():
    """Stale document frequencies would skew every later BM25 score."""
    from app.services.search.vector_store import InMemoryHybridVectorStore

    store = InMemoryHybridVectorStore()
    store.upsert("a", [1.0], "python engineer", {})
    store.upsert("b", [1.0], "python designer", {})
    assert store._doc_freq["python"] == 2

    store.delete("a")
    assert store._doc_freq["python"] == 1
    assert store.count() == 1


def test_reverse_match_ranks_stored_jobs_for_a_candidate():
    from app.db.job_store import get_job_store
    from app.services.search.matcher import match_jobs_for_candidate

    job_store = get_job_store()
    backend = job_store.save(JOB)
    design = job_store.save(
        JobProfile(title="Designer", required_skills=["photoshop"], description="Photoshop work")
    )
    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")

    results = match_jobs_for_candidate("strong")

    assert [r.job_id for r in results] == [backend.job_id, design.job_id]
    assert results[0].evidence.skills.missing_required == []


def test_reverse_match_returns_nothing_without_stored_jobs():
    from app.services.search.matcher import match_jobs_for_candidate

    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")
    assert match_jobs_for_candidate("strong") == []


def test_updating_a_stored_job_keeps_its_creation_time():
    from app.db.job_store import get_job_store

    store = get_job_store()
    original = store.save(JOB)
    updated = store.save(JOB.model_copy(update={"title": "Staff Engineer"}), job_id=original.job_id)

    assert updated.created_at == original.created_at
    assert len(store.all()) == 1


# --- Index rebuild --------------------------------------------------------


def test_reindex_restores_search_after_the_index_is_lost():
    """Durable records with an in-process index are inconsistent after a restart.

    The candidate is still listed by the API but matches nothing, which reads as
    "no results" rather than as a broken index.
    """
    from app.services.search.matcher import reindex_all
    from app.services.search.vector_store import get_vector_store

    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")

    # Simulate a restart: records survive, the vector index does not.
    get_vector_store().delete("strong")
    assert match(JOB) == []

    assert reindex_all() == 1
    assert [r.candidate_id for r in match(JOB)] == ["strong"]


def test_rebuild_is_skipped_when_the_index_is_already_populated():
    """Qdrant persists its own vectors, so a populated index must not be re-embedded."""
    from app.services.search.matcher import reindex_if_index_is_empty

    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")

    assert reindex_if_index_is_empty() == 0


def test_rebuild_is_skipped_when_there_are_no_records():
    from app.services.search.matcher import reindex_if_index_is_empty

    assert reindex_if_index_is_empty() == 0


def test_rebuild_runs_when_records_exist_without_an_index():
    from app.services.search.matcher import reindex_if_index_is_empty
    from app.services.search.vector_store import get_vector_store

    index_candidate(STRONG_CANDIDATE, STRONG_TEXT, candidate_id="strong")
    index_candidate(WEAK_CANDIDATE, WEAK_TEXT, candidate_id="weak")
    get_vector_store().delete("strong")
    get_vector_store().delete("weak")

    assert reindex_if_index_is_empty() == 2
    assert len(match(JOB)) == 2


# --- Education and certification scoring ----------------------------------


def _graduate(degree: str, field: str) -> CandidateProfile:
    from app.schemas.candidate import Education

    return STRONG_CANDIDATE.model_copy(
        update={"education": [Education(institution="Uni", degree=degree, field_of_study=field)]}
    )


CS_JOB = JOB.model_copy(
    update={"required_education": "computer science", "required_degree_level": "bachelor"}
)


def test_an_exact_degree_and_field_scores_full_marks():
    from app.services.search.matcher import _education_score

    score, evidence = _education_score(CS_JOB, _graduate("B.S.", "Computer Science"))

    assert score == 1.0
    assert evidence.meets_requirement is True


def test_a_higher_degree_satisfies_a_lower_requirement():
    """Comparing degree strings for equality would reject someone over-qualified."""
    from app.services.search.matcher import _education_score

    score, _ = _education_score(CS_JOB, _graduate("M.S.", "Computer Science"))
    assert score == 1.0

    score, _ = _education_score(CS_JOB, _graduate("PhD", "Computer Science"))
    assert score == 1.0


def test_the_right_level_in_the_wrong_field_scores_partially():
    from app.services.search.matcher import _education_score

    score, evidence = _education_score(CS_JOB, _graduate("B.A.", "History"))

    assert 0 < score < 1
    assert evidence.meets_requirement is False
    assert evidence.matched_field is None


def test_no_education_at_all_scores_zero_and_says_so():
    """The score and the evidence must agree — a 0.0 alongside "meets_requirement" is worse than either."""
    from app.services.search.matcher import _education_score

    score, evidence = _education_score(CS_JOB, STRONG_CANDIDATE)

    assert score == 0.0
    assert evidence.meets_requirement is False


def test_a_job_with_no_education_requirement_scores_everyone_equally():
    from app.services.search.matcher import _education_score

    assert _education_score(JOB, STRONG_CANDIDATE)[0] == 1.0
    assert _education_score(JOB, _graduate("PhD", "Computer Science"))[0] == 1.0


def test_certifications_match_despite_differing_wording():
    """A posting says "...Solutions Architect"; a resume says "...Solutions Architect - Associate"."""
    from app.services.search.matcher import _certifications_score

    job = JOB.model_copy(update={"required_certifications": ["AWS Certified Solutions Architect"]})
    holder = STRONG_CANDIDATE.model_copy(
        update={"certifications": ["AWS Certified Solutions Architect - Associate"]}
    )

    score, evidence = _certifications_score(job, holder)

    assert score == 1.0
    assert evidence.meets_all_required is True
    assert evidence.missing == []


def test_missing_certifications_are_named():
    from app.services.search.matcher import _certifications_score

    job = JOB.model_copy(update={"required_certifications": ["AWS Certified Solutions Architect"]})

    score, evidence = _certifications_score(job, STRONG_CANDIDATE)

    assert score == 0.0
    assert evidence.missing == ["AWS Certified Solutions Architect"]
    assert evidence.meets_all_required is False


def test_partial_certification_coverage_scores_proportionally():
    from app.services.search.matcher import _certifications_score

    job = JOB.model_copy(update={"required_certifications": ["AWS Certified", "CISSP Certified"]})
    holder = STRONG_CANDIDATE.model_copy(update={"certifications": ["AWS Certified"]})

    assert _certifications_score(job, holder)[0] == 0.5


def test_a_job_requiring_no_certifications_scores_everyone_equally():
    from app.services.search.matcher import _certifications_score

    assert _certifications_score(JOB, STRONG_CANDIDATE)[0] == 1.0

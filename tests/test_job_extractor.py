from app.services.extraction.job_extractor import HeuristicJobExtractor
import pytest

SAMPLE_JD = """\
We are hiring a Backend Engineer, fully remote.

Required:
Python, PostgreSQL, 3+ years experience

Preferred:
AWS, Kubernetes
"""


def test_job_extractor_splits_required_and_preferred_skills():
    job = HeuristicJobExtractor().extract("Backend Engineer", SAMPLE_JD)
    assert "python" in job.required_skills
    assert "postgresql" in job.required_skills
    assert "aws" in job.preferred_skills
    assert "aws" not in job.required_skills


def test_job_extractor_detects_remote_and_min_years():
    job = HeuristicJobExtractor().extract("Backend Engineer", SAMPLE_JD)
    assert job.remote is True
    assert job.min_years_experience == 3.0


def test_job_extractor_default_weights_come_from_settings():
    job = HeuristicJobExtractor().extract("Backend Engineer", SAMPLE_JD)
    assert abs(job.weights.experience + job.weights.skills + job.weights.education - 1.0) < 1e-6


def test_min_years_ignores_prose_outside_the_requirements():
    """Company-history prose must not set the experience bar."""
    jd = """Backend Engineer at a company with 20 years of history.

Required:
Python, PostgreSQL, 3+ years experience

Preferred:
AWS, 8 years of distributed systems work
"""
    assert HeuristicJobExtractor().extract("Backend Engineer", jd).min_years_experience == 3.0


def test_min_years_ignores_a_preferred_nice_to_have():
    jd = "Required:\n3+ years experience\n\nPreferred:\n10 years of Kubernetes\n"
    assert HeuristicJobExtractor().extract("Backend Engineer", jd).min_years_experience == 3.0


def test_unsectioned_posting_only_counts_years_tied_to_experience():
    jd = "Join our team, founded 15 years ago. We want 3+ years of experience in Python."
    assert HeuristicJobExtractor().extract("Backend Engineer", jd).min_years_experience == 3.0


def test_posting_with_no_experience_requirement_sets_no_bar():
    assert HeuristicJobExtractor().extract("Eng", "Our office opened 12 years ago.").min_years_experience == 0.0


def test_a_qualified_candidate_is_not_penalised_by_unrelated_year_mentions():
    from app.schemas.candidate import CandidateProfile, Experience
    from app.services.search.matcher import _experience_score

    jd = "Backend role at a firm with 20 years of history.\n\nRequired:\n3+ years experience\n"
    job = HeuristicJobExtractor().extract("Backend Engineer", jd)
    # A real role title: experience is now relevance-weighted, so a placeholder
    # would score at the unrelated floor and obscure what this test is about.
    candidate = CandidateProfile(
        name="X", experience=[Experience(company="A", role="Backend Engineer", years=4)]
    )

    score, evidence = _experience_score(job, candidate)
    assert score == 1.0
    assert evidence.required_years == 3.0
    assert evidence.meets_requirement is True


# --- Education and certification requirements ----------------------------


def test_degree_level_and_field_are_extracted():
    """Both fed a scoring weight that was inert because nothing ever set them."""
    jd = "Required:\nPython, 5+ years experience\nBachelor's degree in Computer Science\n"
    job = HeuristicJobExtractor().extract("Engineer", jd)

    assert job.required_degree_level == "bachelor"
    assert job.required_education == "computer science"


@pytest.mark.parametrize(
    "line,level,field",
    [
        ("M.S. in Electrical Engineering", "master", "electrical engineering"),
        ("PhD in Machine Learning, or equivalent", "phd", "machine learning"),
        ("Associate's degree in Nursing", "associate", "nursing"),
    ],
)
def test_degree_variants_are_recognised(line, level, field):
    job = HeuristicJobExtractor().extract("Role", f"Required:\n{line}\n")
    assert job.required_degree_level == level
    assert job.required_education == field


def test_a_posting_without_education_sets_no_requirement():
    job = HeuristicJobExtractor().extract("Role", "Required:\nPython, PostgreSQL\n")
    assert job.required_degree_level == ""
    assert job.required_education == ""


def test_field_extraction_ignores_unrelated_in_phrases():
    """"5+ years in Python" must not be read as a field of study."""
    jd = "Required:\n5+ years in Python\nBachelor's degree in Computer Science\n"
    assert HeuristicJobExtractor().extract("Role", jd).required_education == "computer science"


def test_required_certifications_are_extracted():
    jd = "Required:\nPython\nAWS Certified Solutions Architect\n"
    job = HeuristicJobExtractor().extract("Role", jd)
    assert job.required_certifications == ["AWS Certified Solutions Architect"]


def test_certification_weight_is_only_taken_when_one_is_required():
    """A component that is always zero never fires; one that always fires is worse."""
    without = HeuristicJobExtractor().extract("Role", "Required:\nPython\n")
    assert without.weights.certifications == 0.0
    assert without.weights.experience == 0.5

    with_cert = HeuristicJobExtractor().extract(
        "Role", "Required:\nPython\nAWS Certified Solutions Architect\n"
    )
    assert with_cert.weights.certifications > 0
    assert with_cert.weights.experience < 0.5  # rebalanced, not added on top


def test_auto_weighting_still_sums_to_one():
    job = HeuristicJobExtractor().extract(
        "Role", "Required:\nPython\nAWS Certified Solutions Architect\n"
    )
    weights = job.weights
    total = weights.experience + weights.skills + weights.education + weights.certifications
    assert abs(total - 1.0) < 1e-6


def test_explicit_weights_are_never_overridden():
    from app.schemas.job import JobWeights

    explicit = JobWeights(experience=0.7, skills=0.2, education=0.1)
    job = HeuristicJobExtractor().extract(
        "Role", "Required:\nAWS Certified Solutions Architect\n", explicit
    )
    assert job.weights.experience == 0.7
    assert job.weights.certifications == 0.0

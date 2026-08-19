from app.services.extraction.job_extractor import HeuristicJobExtractor

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
    candidate = CandidateProfile(name="X", experience=[Experience(company="A", role="B", years=4)])

    assert _experience_score(job, candidate) == 1.0

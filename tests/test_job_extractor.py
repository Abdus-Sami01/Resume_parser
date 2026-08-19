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

from app.services.extraction.resume_extractor import HeuristicResumeExtractor

SAMPLE_RESUME = """\
Jane Doe
Senior Backend Engineer, jane.doe@example.com, +1 415-555-0100
5+ years of experience building services in Python and Go.
Skills: Python, FastAPI, PostgreSQL, Docker, AWS, Kubernetes
"""


def test_heuristic_extractor_pulls_contact_info():
    profile = HeuristicResumeExtractor().extract(SAMPLE_RESUME)
    assert profile.name == "Jane Doe"
    assert profile.email == "jane.doe@example.com"
    assert "415-555-0100" in (profile.phone or "")


def test_heuristic_extractor_standardizes_skills():
    profile = HeuristicResumeExtractor().extract(SAMPLE_RESUME)
    assert "python" in profile.skills
    assert "fastapi" in profile.skills
    assert "aws" in profile.skills


def test_heuristic_extractor_estimates_years_of_experience():
    profile = HeuristicResumeExtractor().extract(SAMPLE_RESUME)
    assert profile.total_years_experience == 5.0

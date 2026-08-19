from app.schemas.candidate import CandidateProfile
from app.services.extraction.resume_extractor import (
    HeuristicResumeExtractor,
    standardize_profile_skills,
)

SAMPLE_RESUME = """\
Jane Doe
Senior Backend Engineer, jane.doe@example.com, +1 415-555-0100
5+ years of experience building services in Python and Go.
Skills: Python, FastAPI, PostgreSQL, Docker, AWS, Kubernetes
B.S. in Computer Science, Stanford University, 2018
Certifications:
AWS Certified Solutions Architect
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


def test_heuristic_extractor_pulls_education():
    profile = HeuristicResumeExtractor().extract(SAMPLE_RESUME)
    assert len(profile.education) == 1

    education = profile.education[0]
    assert education.institution == "Stanford University"
    assert education.graduation_year == 2018
    assert "Computer Science" in education.field_of_study


def test_heuristic_extractor_pulls_certifications():
    profile = HeuristicResumeExtractor().extract(SAMPLE_RESUME)
    assert "AWS Certified Solutions Architect" in profile.certifications


def test_standardize_profile_skills_maps_raw_llm_output_onto_the_taxonomy():
    profile = CandidateProfile(name="Test", skills=["React.js", "ReactJS", "Amazon Web Services"])
    standardized = standardize_profile_skills(profile)

    assert standardized.skills.count("react") == 1
    assert "aws" in standardized.skills


def test_standardize_profile_skills_keeps_unknown_skills_rather_than_dropping_them():
    profile = CandidateProfile(name="Test", skills=["Python", "Obscure Internal Tool"])
    standardized = standardize_profile_skills(profile)

    assert "python" in standardized.skills
    assert "obscure internal tool" in standardized.skills

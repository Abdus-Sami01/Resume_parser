import pytest
from pydantic import ValidationError

from app.schemas.candidate import CandidateProfile, Experience
from app.schemas.job import JobProfile, JobWeights


def test_candidate_total_years_experience_sums_roles():
    profile = CandidateProfile(
        name="Ada Lovelace",
        experience=[
            Experience(company="A", role="Engineer", years=2),
            Experience(company="B", role="Senior Engineer", years=3.5),
        ],
    )
    assert profile.total_years_experience == 5.5


def test_experience_years_must_be_non_negative():
    with pytest.raises(ValidationError):
        Experience(company="A", role="Engineer", years=-1)


def test_job_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        JobWeights(experience=0.5, skills=0.5, education=0.5)


def test_job_profile_all_skills_combines_required_and_preferred():
    job = JobProfile(
        title="Backend Engineer",
        required_skills=["python", "sql"],
        preferred_skills=["aws"],
    )
    assert job.all_skills == ["python", "sql", "aws"]

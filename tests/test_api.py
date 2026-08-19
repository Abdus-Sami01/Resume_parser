from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_RESUME = b"""Jane Doe
jane.doe@example.com
5+ years of experience in Python, FastAPI, PostgreSQL, AWS.
"""

SAMPLE_JD = {
    "title": "Backend Engineer",
    "description": (
        "Required:\nPython, PostgreSQL, 3+ years experience\n\nPreferred:\nAWS"
    ),
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_resume_returns_structured_profile():
    response = client.post(
        "/resumes", files={"file": ("resume.txt", SAMPLE_RESUME, "text/plain")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["name"] == "Jane Doe"
    assert "python" in body["profile"]["skills"]
    assert body["candidate_id"]


def test_parse_job_splits_required_and_preferred():
    response = client.post("/jobs/parse", json=SAMPLE_JD)
    assert response.status_code == 200
    body = response.json()
    assert "python" in body["required_skills"]
    assert "aws" in body["preferred_skills"]


def test_end_to_end_upload_then_match():
    client.post("/resumes", files={"file": ("resume.txt", SAMPLE_RESUME, "text/plain")})
    job = client.post("/jobs/parse", json=SAMPLE_JD).json()

    response = client.post("/search/match", json={"job": job})
    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert results[0]["candidate"]["name"] == "Jane Doe"


def test_match_filters_exclude_candidates_missing_a_required_skill():
    client.post("/resumes", files={"file": ("resume.txt", SAMPLE_RESUME, "text/plain")})
    job = client.post("/jobs/parse", json=SAMPLE_JD).json()

    matched = client.post("/search/match", json={"job": job, "filters": {"skills": ["python"]}})
    assert len(matched.json()) == 1

    excluded = client.post("/search/match", json={"job": job, "filters": {"skills": ["kafka"]}})
    assert excluded.json() == []

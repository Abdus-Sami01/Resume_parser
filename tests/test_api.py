import pytest

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


def _build_pdf(lines: list[str]) -> bytes:
    """Renders a real PDF so the pypdf extraction branch is exercised, not just text uploads."""
    pytest.importorskip("reportlab")
    from io import BytesIO

    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for index, line in enumerate(lines):
        pdf.drawString(72, 750 - index * 20, line)
    pdf.save()
    return buffer.getvalue()


def test_real_pdf_flows_through_the_whole_pipeline():
    pdf_bytes = _build_pdf(
        [
            "Jane Doe",
            "jane.doe@example.com",
            "Senior Backend Engineer, Acme Corp, Jan 2019 - Dec 2022",
            "Skills: Python, FastAPI, PostgreSQL, AWS",
            "B.S. in Computer Science, Stanford University, 2018",
        ]
    )

    response = client.post("/resumes", files={"file": ("jane.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 200

    profile = response.json()["profile"]
    assert profile["name"] == "Jane Doe"
    assert profile["email"] == "jane.doe@example.com"
    assert "python" in profile["skills"]
    assert profile["education"][0]["institution"] == "Stanford University"
    assert sum(e["years"] for e in profile["experience"]) > 3


def test_corrupt_upload_is_a_client_error_not_a_server_crash():
    response = client.post(
        "/resumes", files={"file": ("broken.pdf", b"this is not a pdf at all", "application/pdf")}
    )
    assert response.status_code == 422
    assert "broken.pdf" in response.json()["detail"]


def test_empty_upload_is_rejected():
    response = client.post("/resumes", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_async_upload_dispatches_and_reports_completion():
    response = client.post("/resumes/async", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")})
    assert response.status_code == 202

    body = response.json()
    assert body["state"] == "SUCCESS"  # eager backend runs inline
    assert body["candidate_id"]


def test_task_status_is_retrievable_and_not_shadowed_by_the_candidate_route():
    """`/resumes/tasks/{id}` sits next to `/resumes/{candidate_id}` and must win."""
    task_id = client.post(
        "/resumes/async", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")}
    ).json()["task_id"]

    response = client.get(f"/resumes/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json()["task_id"] == task_id


def test_parsed_profile_can_be_fetched_by_candidate_id():
    candidate_id = client.post(
        "/resumes", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")}
    ).json()["candidate_id"]

    assert client.get(f"/resumes/{candidate_id}").json()["name"] == "Jane Doe"
    assert client.get("/resumes/does-not-exist").status_code == 404


@pytest.mark.parametrize("params", [{"top_k": 0}, {"top_k": -5}, {"top_n": 0}, {"top_k": 10**6}])
def test_out_of_range_retrieval_params_are_rejected(params):
    job = client.post("/jobs/parse", json=SAMPLE_JD).json()
    assert client.post("/search/match", json={"job": job, **params}).status_code == 422

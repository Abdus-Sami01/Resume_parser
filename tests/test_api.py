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


def test_eager_task_results_do_not_grow_without_bound():
    """The API process is long-lived, so nothing else would ever evict these."""
    from app.workers import tasks

    monkey_limit = 5
    original_limit, tasks._EAGER_RESULT_LIMIT = tasks._EAGER_RESULT_LIMIT, monkey_limit
    tasks._EAGER_RESULTS.clear()
    try:
        for _ in range(monkey_limit + 3):
            client.post("/resumes/async", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")})
        assert len(tasks._EAGER_RESULTS) == monkey_limit
    finally:
        tasks._EAGER_RESULT_LIMIT = original_limit
        tasks._EAGER_RESULTS.clear()


def test_oversized_upload_is_rejected_without_being_buffered():
    from app.config import get_settings

    settings = get_settings()
    original, settings.max_upload_bytes = settings.max_upload_bytes, 1024
    try:
        response = client.post(
            "/resumes", files={"file": ("big.txt", b"x" * 5000, "text/plain")}
        )
        assert response.status_code == 413
    finally:
        settings.max_upload_bytes = original


def test_uploads_within_the_cap_still_succeed():
    assert client.post(
        "/resumes", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")}
    ).status_code == 200


# --- Bulk ingestion -------------------------------------------------------

BOB_RESUME = b"Bob Ray\nbob@x.com\nGraphic Designer, Studio Ltd, 2015 - 2020\nSkills: Photoshop\n"


def test_bulk_upload_reports_per_file_outcomes():
    response = client.post(
        "/resumes/bulk",
        files=[
            ("files", ("jane.txt", SAMPLE_RESUME, "text/plain")),
            ("files", ("bob.txt", BOB_RESUME, "text/plain")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 2
    assert body["failed"] == 0
    assert all(item["candidate_id"] for item in body["items"])


def test_one_bad_file_does_not_discard_the_rest_of_the_batch():
    """A single unreadable resume must not cost the other files in the upload."""
    response = client.post(
        "/resumes/bulk",
        files=[
            ("files", ("jane.txt", SAMPLE_RESUME, "text/plain")),
            ("files", ("broken.pdf", b"not a pdf", "application/pdf")),
            ("files", ("bob.txt", BOB_RESUME, "text/plain")),
        ],
    )

    body = response.json()
    assert body["succeeded"] == 2
    assert body["failed"] == 1

    failed = next(item for item in body["items"] if item["error"])
    assert failed["filename"] == "broken.pdf"
    assert failed["candidate_id"] is None


# --- Listing and pagination -----------------------------------------------


def test_candidates_can_be_listed_with_pagination():
    for index in range(3):
        client.post(
            "/resumes",
            files={"file": (f"r{index}.txt", f"Person {index}\np{index}@x.com\n2 years of Python.\n".encode(), "text/plain")},
        )

    page = client.get("/resumes?offset=0&limit=2").json()

    assert page["total"] == 3
    assert page["limit"] == 2
    assert len(page["items"]) == 2

    second = client.get("/resumes?offset=2&limit=2").json()
    assert len(second["items"]) == 1


@pytest.mark.parametrize("query", ["?limit=0", "?limit=500", "?offset=-1"])
def test_listing_rejects_out_of_range_paging(query):
    assert client.get(f"/resumes{query}").status_code == 422


# --- Erasure --------------------------------------------------------------


def test_deleting_a_candidate_removes_them_from_search_too():
    """Erasure that leaves the vector behind still exposes the person in results."""
    candidate_id = client.post(
        "/resumes", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")}
    ).json()["candidate_id"]
    job = client.post("/jobs", json=SAMPLE_JD).json()

    assert client.post(f"/jobs/{job['job_id']}/match").json()

    assert client.delete(f"/resumes/{candidate_id}").status_code == 204
    assert client.get(f"/resumes/{candidate_id}").status_code == 404
    assert client.post(f"/jobs/{job['job_id']}/match").json() == []


def test_deleting_an_unknown_candidate_is_a_404():
    assert client.delete("/resumes/does-not-exist").status_code == 404


# --- Job catalogue --------------------------------------------------------


def test_job_can_be_created_read_and_listed():
    created = client.post("/jobs", json=SAMPLE_JD)
    assert created.status_code == 201

    job_id = created.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["profile"]["title"] == "Backend Engineer"
    assert client.get("/jobs").json()["total"] == 1


def test_job_can_be_replaced_in_place():
    job_id = client.post("/jobs", json=SAMPLE_JD).json()["job_id"]

    updated = client.put(
        f"/jobs/{job_id}",
        json={"title": "Staff Engineer", "description": "Required:\nGolang\n"},
    )

    assert updated.status_code == 200
    assert updated.json()["job_id"] == job_id  # same id, not a new posting
    assert updated.json()["profile"]["title"] == "Staff Engineer"
    assert client.get("/jobs").json()["total"] == 1


def test_job_can_be_deleted():
    job_id = client.post("/jobs", json=SAMPLE_JD).json()["job_id"]

    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404


@pytest.mark.parametrize(
    "method,path",
    [("get", "/jobs/nope"), ("delete", "/jobs/nope"), ("post", "/jobs/nope/match")],
)
def test_unknown_job_is_a_404(method, path):
    assert getattr(client, method)(path).status_code == 404


def test_stored_job_can_be_rerun_against_the_current_pool():
    job_id = client.post("/jobs", json=SAMPLE_JD).json()["job_id"]
    assert client.post(f"/jobs/{job_id}/match").json() == []

    client.post("/resumes", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")})

    assert len(client.post(f"/jobs/{job_id}/match").json()) == 1


# --- Reverse matching -----------------------------------------------------


def test_candidate_can_be_matched_against_stored_jobs():
    candidate_id = client.post(
        "/resumes", files={"file": ("r.txt", SAMPLE_RESUME, "text/plain")}
    ).json()["candidate_id"]
    client.post("/jobs", json=SAMPLE_JD)
    client.post("/jobs", json={"title": "Designer", "description": "Required:\nPhotoshop\n"})

    results = client.get(f"/resumes/{candidate_id}/jobs").json()

    assert len(results) == 2
    assert results[0]["job_title"] == "Backend Engineer"  # the Python role outranks Designer
    assert results[0]["breakdown"]["weighted_total"] > results[1]["breakdown"]["weighted_total"]


def test_reverse_match_for_unknown_candidate_is_a_404():
    assert client.get("/resumes/nope/jobs").status_code == 404

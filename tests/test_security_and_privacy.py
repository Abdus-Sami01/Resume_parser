"""API key auth, blind screening, and request timing."""
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)

RESUME = (
    b"Jane Doe\njane.doe@example.com\n+1 415-555-0100\n"
    b"Senior Backend Engineer, Acme Corp, Jan 2015 - Dec 2022\n"
    b"Skills: Python, FastAPI, PostgreSQL, AWS\n"
    b"B.S. in Computer Science, Stanford University, 2018\n"
)
JD = {"title": "Backend Engineer", "description": "Required:\nPython, PostgreSQL, 3+ years experience\n"}


@pytest.fixture
def authenticated(monkeypatch):
    monkeypatch.setenv("API_KEYS", "key-one,key-two")
    get_settings.cache_clear()
    yield {"X-API-Key": "key-one"}
    get_settings.cache_clear()


# --- Authentication -------------------------------------------------------


def test_endpoints_are_open_when_no_keys_are_configured():
    """The zero-config local run must keep working; the warning covers the risk."""
    assert client.get("/resumes").status_code == 200


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/resumes"),
        ("get", "/jobs"),
        ("get", "/analytics/overview"),
        ("get", "/skills"),
        ("post", "/search/candidates"),
    ],
)
def test_every_router_rejects_an_unauthenticated_request(authenticated, method, path):
    """Auth is attached per-router so a later route cannot quietly skip it."""
    assert getattr(client, method)(path).status_code == 401


def test_a_wrong_key_is_rejected(authenticated):
    assert client.get("/resumes", headers={"X-API-Key": "wrong"}).status_code == 401


def test_any_configured_key_is_accepted(authenticated):
    assert client.get("/resumes", headers={"X-API-Key": "key-one"}).status_code == 200
    assert client.get("/resumes", headers={"X-API-Key": "key-two"}).status_code == 200


def test_health_stays_open_for_probes(authenticated):
    """Load balancers and container probes cannot present a key."""
    assert client.get("/health").status_code == 200


def test_rejection_advertises_the_scheme(authenticated):
    response = client.get("/resumes")
    assert response.headers.get("WWW-Authenticate") == "API-Key"


# --- Blind screening ------------------------------------------------------


@pytest.fixture
def matched_job():
    client.post("/resumes", files={"file": ("r.txt", RESUME, "text/plain")})
    return client.post("/jobs", json=JD).json()["job_id"]


def test_blind_mode_removes_identifying_fields(matched_job):
    candidate = client.post(f"/jobs/{matched_job}/match?blind=true").json()[0]["candidate"]

    assert candidate["name"].startswith("Candidate ")
    assert candidate["email"] is None
    assert candidate["phone"] is None
    assert candidate["summary"] == ""
    assert candidate["education"][0]["institution"] == ""  # prestige signal


def test_blind_mode_keeps_everything_a_decision_needs(matched_job):
    candidate = client.post(f"/jobs/{matched_job}/match?blind=true").json()[0]["candidate"]

    assert "python" in candidate["skills"]
    assert candidate["total_years_experience"] > 7
    assert candidate["education"][0]["degree"]  # degree kept, school dropped


def test_blinding_does_not_change_the_ranking(matched_job):
    """Scoring never reads the redacted fields, so the shortlist must be identical."""
    normal = client.post(f"/jobs/{matched_job}/match").json()
    blind = client.post(f"/jobs/{matched_job}/match?blind=true").json()

    assert [r["candidate_id"] for r in normal] == [r["candidate_id"] for r in blind]
    assert [r["breakdown"]["weighted_total"] for r in normal] == [
        r["breakdown"]["weighted_total"] for r in blind
    ]


def test_pseudonym_is_stable_for_the_same_candidate(matched_job):
    first = client.post(f"/jobs/{matched_job}/match?blind=true").json()[0]["candidate"]["name"]
    second = client.post(f"/jobs/{matched_job}/match?blind=true").json()[0]["candidate"]["name"]

    assert first == second  # reviewers can refer to "Candidate ab12cd34" across calls


def test_ad_hoc_match_supports_blind_mode():
    client.post("/resumes", files={"file": ("r.txt", RESUME, "text/plain")})
    job = client.post("/jobs/parse", json=JD).json()

    blind = client.post("/search/match", json={"job": job, "blind": True}).json()
    assert blind[0]["candidate"]["email"] is None


def test_csv_export_can_be_blinded(matched_job):
    import csv
    import io

    rows = list(
        csv.DictReader(io.StringIO(client.get(f"/jobs/{matched_job}/match/export?blind=true").text))
    )

    assert rows[0]["name"].startswith("Candidate ")
    assert rows[0]["email"] == ""
    assert "python" in rows[0]["matched_required_skills"]  # evidence still present


def test_redaction_never_mutates_the_stored_profile(matched_job):
    client.post(f"/jobs/{matched_job}/match?blind=true")

    stored = client.get("/resumes").json()["items"][0]
    assert stored["name"] == "Jane Doe"  # the record itself is untouched


# --- Request timing -------------------------------------------------------


def test_every_response_carries_its_duration():
    """Latency is dominated by model calls a caller cannot otherwise see."""
    response = client.get("/health")

    assert "X-Process-Time-Ms" in response.headers
    assert float(response.headers["X-Process-Time-Ms"]) >= 0

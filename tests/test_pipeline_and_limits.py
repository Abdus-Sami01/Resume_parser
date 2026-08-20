"""Hiring pipeline stage tracking, and per-caller rate limiting."""
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services.rate_limit import (
    InMemorySlidingWindowLimiter,
    caller_identity,
    get_rate_limiter,
)

client = TestClient(app)

JANE = b"Jane Doe\njane@x.com\nSenior Backend Engineer, Acme Corp, Jan 2015 - Dec 2022\nSkills: Python, PostgreSQL\n"
RAJ = b"Raj Patel\nraj@x.com\nBackend Engineer, Beta Systems, Jan 2021 - Dec 2023\nSkills: Python\n"
JD = {"title": "Backend Engineer", "description": "Required:\nPython, PostgreSQL\n"}


@pytest.fixture
def board():
    client.post("/resumes/bulk", files=[
        ("files", ("jane.txt", JANE, "text/plain")),
        ("files", ("raj.txt", RAJ, "text/plain")),
    ])
    ids = {i["name"]: i["candidate_id"] for i in client.get("/resumes").json()["items"]}
    job_id = client.post("/jobs", json=JD).json()["job_id"]
    return {"job_id": job_id, "ids": ids}


# --- Pipeline membership --------------------------------------------------


def test_candidate_can_be_added_to_a_pipeline(board):
    response = client.post(
        f"/jobs/{board['job_id']}/pipeline",
        json={"candidate_id": board["ids"]["Jane Doe"], "actor": "recruiter@co"},
    )

    assert response.status_code == 201
    assert response.json()["stage"] == "applied"
    assert len(response.json()["history"]) == 1


def test_adding_the_same_candidate_twice_is_a_conflict(board):
    payload = {"candidate_id": board["ids"]["Jane Doe"]}
    client.post(f"/jobs/{board['job_id']}/pipeline", json=payload)

    duplicate = client.post(f"/jobs/{board['job_id']}/pipeline", json=payload)

    assert duplicate.status_code == 409
    assert "already in this pipeline" in duplicate.json()["detail"]


def test_pipeline_rejects_unknown_job_or_candidate(board):
    assert client.post(
        "/jobs/nope/pipeline", json={"candidate_id": board["ids"]["Jane Doe"]}
    ).status_code == 404
    assert client.post(
        f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": "nope"}
    ).status_code == 404


def test_candidate_can_be_removed_from_a_pipeline(board):
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    assert client.delete(f"/jobs/{board['job_id']}/pipeline/{jane}").status_code == 204
    assert client.delete(f"/jobs/{board['job_id']}/pipeline/{jane}").status_code == 404


# --- Stage transitions ----------------------------------------------------


def test_moving_a_candidate_records_the_transition(board):
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    moved = client.patch(
        f"/jobs/{board['job_id']}/pipeline/{jane}",
        json={"stage": "screening", "note": "strong python", "actor": "hm@co"},
    ).json()

    assert moved["stage"] == "screening"
    latest = moved["history"][-1]
    assert latest["from_stage"] == "applied"
    assert latest["to_stage"] == "screening"
    assert latest["note"] == "strong python"
    assert latest["actor"] == "hm@co"


def test_full_progression_accumulates_one_event_per_move(board):
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    for stage in ("screening", "interview", "offer", "hired"):
        client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": stage})

    entry = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"]
    assert entry["stage"] == "hired"
    assert [e["to_stage"] for e in entry["history"]] == [
        "applied", "screening", "interview", "offer", "hired",
    ]


def test_moving_to_the_current_stage_does_not_pad_the_audit_trail(board):
    """A no-op must not look like a decision someone made."""
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "applied"})

    entry = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"]
    assert len(entry["history"]) == 1


def test_a_candidate_can_be_moved_backwards(board):
    """Real processes revisit stages; a rigid state machine only gets worked around."""
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "interview"})

    back = client.patch(
        f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "screening", "note": "redo"}
    )

    assert back.status_code == 200
    assert back.json()["stage"] == "screening"
    assert back.json()["history"][-1]["from_stage"] == "interview"


def test_moving_someone_not_in_the_pipeline_is_a_404(board):
    assert client.patch(
        f"/jobs/{board['job_id']}/pipeline/{board['ids']['Jane Doe']}", json={"stage": "offer"}
    ).status_code == 404


def test_an_invalid_stage_is_rejected(board):
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    assert client.patch(
        f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "wizard"}
    ).status_code == 422


# --- Board views ----------------------------------------------------------


def test_board_reports_counts_per_stage(board):
    jane, raj = board["ids"]["Jane Doe"], board["ids"]["Raj Patel"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": raj})
    client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "offer"})

    listing = client.get(f"/jobs/{board['job_id']}/pipeline").json()

    assert listing["total"] == 2
    assert listing["stage_counts"] == {"applied": 1, "offer": 1}


def test_board_can_be_filtered_to_one_stage(board):
    jane, raj = board["ids"]["Jane Doe"], board["ids"]["Raj Patel"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": raj})
    client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "offer"})

    offers = client.get(f"/jobs/{board['job_id']}/pipeline?stage=offer").json()

    assert offers["total"] == 1
    assert offers["items"][0]["candidate_name"] == "Jane Doe"
    # Counts stay unfiltered, so the board still shows the whole shape.
    assert offers["stage_counts"] == {"applied": 1, "offer": 1}


def test_board_supports_blind_review(board):
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": board["ids"]["Jane Doe"]})

    blind = client.get(f"/jobs/{board['job_id']}/pipeline?blind=true").json()

    assert blind["items"][0]["candidate_name"].startswith("Candidate ")


def test_one_candidate_can_sit_at_different_stages_across_roles(board):
    """Stage belongs to the (candidate, job) pair — a global status could not express this."""
    jane = board["ids"]["Jane Doe"]
    other_job = client.post(
        "/jobs", json={"title": "Platform", "description": "Required:\nKubernetes\n"}
    ).json()["job_id"]

    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "offer"})
    client.post(f"/jobs/{other_job}/pipeline", json={"candidate_id": jane, "stage": "rejected"})

    applications = client.get(f"/resumes/{jane}/applications").json()
    stages = {app["job_id"]: app["stage"] for app in applications}

    assert stages[board["job_id"]] == "offer"
    assert stages[other_job] == "rejected"


# --- Funnel ---------------------------------------------------------------


def test_funnel_counts_stages_ever_reached_not_just_current(board):
    """Counting only current stages would report a funnel that never converts."""
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    for stage in ("screening", "interview", "offer"):
        client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": stage})

    steps = {s["stage"]: s for s in client.get(f"/jobs/{board['job_id']}/pipeline/funnel").json()["steps"]}

    assert steps["applied"]["ever_reached"] == 1
    assert steps["applied"]["currently_here"] == 0  # moved on
    assert steps["offer"]["ever_reached"] == 1
    assert steps["offer"]["currently_here"] == 1


def test_funnel_reports_conversion_between_steps(board):
    jane, raj = board["ids"]["Jane Doe"], board["ids"]["Raj Patel"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": raj})
    client.patch(f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "screening"})

    funnel = client.get(f"/jobs/{board['job_id']}/pipeline/funnel").json()
    steps = {s["stage"]: s for s in funnel["steps"]}

    assert steps["applied"]["conversion_from_previous"] is None  # nothing precedes it
    assert steps["screening"]["conversion_from_previous"] == 0.5  # 1 of 2 advanced


def test_funnel_separates_exits_from_active(board):
    jane, raj = board["ids"]["Jane Doe"], board["ids"]["Raj Patel"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": raj})
    client.patch(f"/jobs/{board['job_id']}/pipeline/{raj}", json={"stage": "rejected"})

    funnel = client.get(f"/jobs/{board['job_id']}/pipeline/funnel").json()

    assert funnel["total_candidates"] == 2
    assert funnel["active"] == 1
    assert funnel["exits"]["rejected"] == 1


# --- Cascade deletes ------------------------------------------------------


def test_erasing_a_candidate_clears_their_pipeline_history(board):
    """Stage notes carry the candidate id plus free text a reviewer wrote about them."""
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    client.patch(
        f"/jobs/{board['job_id']}/pipeline/{jane}", json={"stage": "rejected", "note": "personal note"}
    )

    client.delete(f"/resumes/{jane}")

    listing = client.get(f"/jobs/{board['job_id']}/pipeline").json()
    assert listing["total"] == 0


def test_deleting_a_job_clears_its_pipeline(board):
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    client.delete(f"/jobs/{board['job_id']}")

    assert client.get(f"/resumes/{jane}/applications").json() == []


# --- Rate limiting --------------------------------------------------------


def test_sliding_window_admits_then_blocks():
    limiter = InMemorySlidingWindowLimiter(limit=3)

    decisions = [limiter.check("key:abc") for _ in range(5)]

    assert [d.allowed for d in decisions] == [True, True, True, False, False]
    assert [d.remaining for d in decisions[:3]] == [2, 1, 0]
    assert decisions[3].retry_after_seconds > 0


def test_each_caller_gets_an_independent_quota():
    limiter = InMemorySlidingWindowLimiter(limit=2)

    limiter.check("key:one")
    limiter.check("key:one")

    assert limiter.check("key:one").allowed is False
    assert limiter.check("key:two").allowed is True


def test_identity_prefers_the_api_key_over_the_ip():
    """IP alone would put a whole corporate NAT into one bucket."""
    assert caller_identity("abc", "10.0.0.1") == "key:abc"
    assert caller_identity(None, "10.0.0.1") == "ip:10.0.0.1"
    assert caller_identity(None, None) == "ip:unknown"


@pytest.fixture
def throttled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    get_settings.cache_clear()
    get_rate_limiter.cache_clear()
    yield
    get_settings.cache_clear()
    get_rate_limiter.cache_clear()


def test_requests_over_the_limit_get_429(throttled):
    codes = [client.get("/jobs").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_throttled_response_tells_the_caller_when_to_retry(throttled):
    for _ in range(3):
        client.get("/jobs")

    blocked = client.get("/jobs")

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


def test_allowed_responses_advertise_the_remaining_quota(throttled):
    assert client.get("/jobs").headers["X-RateLimit-Remaining"] == "2"
    assert client.get("/jobs").headers["X-RateLimit-Remaining"] == "1"


def test_health_is_never_throttled(throttled):
    """Throttling a health check would take the service out of its load balancer."""
    for _ in range(10):
        assert client.get("/health").status_code == 200


def test_limiting_is_off_by_default():
    assert [client.get("/jobs").status_code for _ in range(6)] == [200] * 6

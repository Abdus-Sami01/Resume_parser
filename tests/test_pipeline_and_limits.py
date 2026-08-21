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


# --- Shortlisting: match straight into the pipeline -----------------------


def test_shortlist_adds_the_top_matches_in_one_call(board):
    """Acting on a top-ten used to mean eleven requests and a client-side loop."""
    response = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist",
        json={"top_n": 2, "stage": "screening", "actor": "recruiter@co"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["added"] == 2
    assert all(entry["stage"] == "screening" for entry in body["entries"])
    assert client.get(f"/jobs/{board['job_id']}/pipeline").json()["total"] == 2


def test_shortlist_preserves_match_order(board):
    ranked = [
        result["candidate_id"] for result in client.post(f"/jobs/{board['job_id']}/match").json()
    ]

    entries = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 2}
    ).json()["entries"]

    assert [entry["candidate_id"] for entry in entries] == ranked[:2]


def test_reshortlisting_reports_the_overlap_instead_of_failing(board):
    """Re-running after new resumes arrive is the normal use, and must not error."""
    client.post(f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 2})

    second = client.post(f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 2}).json()

    assert second["added"] == 0
    assert second["skipped"] == 2
    assert {d["reason"] for d in second["skipped_details"]} == {"already in pipeline"}


def test_shortlist_respects_a_minimum_score(board):
    everyone = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 10}
    ).json()["added"]

    # Clear and retry with a floor no candidate can clear.
    for entry in client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"]:
        client.delete(f"/jobs/{board['job_id']}/pipeline/{entry['entry']['candidate_id']}")

    none_qualify = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 10, "min_score": 0.99}
    ).json()

    assert everyone > 0
    assert none_qualify["added"] == 0


def test_shortlisting_an_unknown_job_is_a_404():
    assert client.post("/jobs/nope/pipeline/shortlist", json={"top_n": 5}).status_code == 404


def test_shortlist_route_is_not_shadowed_by_the_candidate_route(board):
    """`/pipeline/shortlist` sits beside `/pipeline/{candidate_id}`."""
    assert client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 1}
    ).status_code == 201


# --- Bulk stage moves -----------------------------------------------------


def test_bulk_move_advances_many_candidates_at_once(board):
    entries = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 2}
    ).json()["entries"]
    ids = [entry["candidate_id"] for entry in entries]

    response = client.patch(
        f"/jobs/{board['job_id']}/pipeline",
        json={"candidate_ids": ids, "stage": "rejected", "note": "below bar"},
    )

    assert response.status_code == 200
    assert response.json()["moved"] == 2
    assert client.get(f"/jobs/{board['job_id']}/pipeline").json()["stage_counts"] == {"rejected": 2}


def test_one_bad_id_does_not_discard_the_rest_of_the_batch(board):
    entries = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 2}
    ).json()["entries"]
    ids = [entry["candidate_id"] for entry in entries]

    body = client.patch(
        f"/jobs/{board['job_id']}/pipeline",
        json={"candidate_ids": ids + ["ghost-id"], "stage": "rejected"},
    ).json()

    assert body["moved"] == 2
    assert body["failed"] == 1
    assert body["failed_details"][0]["candidate_id"] == "ghost-id"


def test_bulk_move_records_history_for_every_candidate(board):
    entries = client.post(
        f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 2}
    ).json()["entries"]

    client.patch(
        f"/jobs/{board['job_id']}/pipeline",
        json={
            "candidate_ids": [e["candidate_id"] for e in entries],
            "stage": "rejected",
            "note": "below bar",
            "actor": "hm@co",
        },
    )

    for item in client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"]:
        latest = item["entry"]["history"][-1]
        assert latest["to_stage"] == "rejected"
        assert latest["note"] == "below bar"
        assert latest["actor"] == "hm@co"


def test_bulk_move_requires_at_least_one_candidate(board):
    assert client.patch(
        f"/jobs/{board['job_id']}/pipeline", json={"candidate_ids": [], "stage": "rejected"}
    ).status_code == 422


# --- Reranker blend -------------------------------------------------------


def test_blend_shifts_authority_between_reranker_and_structured_scoring(monkeypatch):
    """The lexical fallback rewards terse resumes; the blend is how that gets dialled down."""
    from app.services.search.matcher import _score_breakdown
    from app.schemas.job import JobProfile
    from app.schemas.candidate import CandidateProfile, Experience

    job = JobProfile(title="Backend Engineer", required_skills=["python"], min_years_experience=5)
    candidate = CandidateProfile(
        name="X",
        skills=["python"],
        experience=[Experience(company="A", role="Backend Engineer", years=10)],
    )

    monkeypatch.setenv("RERANK_BLEND", "0.0")
    get_settings.cache_clear()
    structured_only, _ = _score_breakdown(job, candidate, retrieval_score=0.0, rerank_score=0.0)

    monkeypatch.setenv("RERANK_BLEND", "1.0")
    get_settings.cache_clear()
    rerank_only, _ = _score_breakdown(job, candidate, retrieval_score=0.0, rerank_score=0.0)

    assert structured_only.weighted_total > 0.9  # perfect structured fit
    assert rerank_only.weighted_total == 0.0  # rerank score of zero dominates
    get_settings.cache_clear()


@pytest.mark.parametrize("bad", ["-0.1", "1.5"])
def test_an_out_of_range_blend_is_rejected(monkeypatch, bad):
    monkeypatch.setenv("RERANK_BLEND", bad)
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()
    get_settings.cache_clear()


# --- Match snapshots -------------------------------------------------------


def test_shortlisting_records_why_the_candidate_was_added(board):
    """Re-running the match later answers a different question; the pool has moved."""
    client.post(f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 1})

    snapshot = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"][
        "match_snapshot"
    ]

    assert snapshot is not None
    assert snapshot["score"] > 0
    assert "skills" in snapshot["evidence"]
    assert snapshot["captured_at"]


def test_the_snapshot_matches_the_score_that_selected_the_candidate(board):
    """Recomputing instead of reusing could disagree with the ranking itself."""
    ranked = client.post(f"/jobs/{board['job_id']}/match").json()[0]

    client.post(f"/jobs/{board['job_id']}/pipeline/shortlist", json={"top_n": 1})
    entry = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"]

    assert entry["match_snapshot"]["score"] == pytest.approx(
        ranked["breakdown"]["weighted_total"], abs=1e-6
    )


def test_manually_added_candidates_also_get_a_snapshot(board):
    client.post(
        f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": board["ids"]["Jane Doe"]}
    )

    entry = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"]
    assert entry["match_snapshot"] is not None


def test_rescoring_reports_drift_after_the_posting_changes(board):
    """A candidate scoring 0.61 then and 0.44 now means something moved."""
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})

    client.put(
        f"/jobs/{board['job_id']}",
        json={
            "title": "Backend Engineer",
            "description": "Required:\nPython, PostgreSQL, Kafka, Spark, 10+ years experience\n",
        },
    )

    drift = client.get(f"/jobs/{board['job_id']}/pipeline/{jane}/rescore").json()

    assert drift["original"]["score"] > drift["current"]["score"]
    assert drift["score_delta"] < 0
    assert "kafka" in drift["current"]["evidence"]["skills"]["missing_required"]


def test_rescoring_never_overwrites_the_original(board):
    """The stored snapshot is the record of the decision; refreshing it destroys that."""
    jane = board["ids"]["Jane Doe"]
    client.post(f"/jobs/{board['job_id']}/pipeline", json={"candidate_id": jane})
    original = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"][
        "match_snapshot"
    ]["score"]

    client.get(f"/jobs/{board['job_id']}/pipeline/{jane}/rescore")

    still = client.get(f"/jobs/{board['job_id']}/pipeline").json()["items"][0]["entry"][
        "match_snapshot"
    ]["score"]
    assert still == original


def test_rescoring_someone_not_in_the_pipeline_is_a_404(board):
    assert client.get(
        f"/jobs/{board['job_id']}/pipeline/{board['ids']['Jane Doe']}/rescore"
    ).status_code == 404


def test_snapshots_survive_a_restart(tmp_path, monkeypatch):
    from app.db.pipeline_store import SqlitePipelineStore
    from app.schemas.match import MatchEvidence, ScoreBreakdown
    from app.schemas.pipeline import MatchSnapshot, PipelineEntry
    from app.schemas.match import (
        CertificationEvidence,
        EducationEvidence,
        ExperienceEvidence,
        SkillEvidence,
    )

    path = str(tmp_path / "pipeline.db")
    snapshot = MatchSnapshot(
        score=0.61,
        breakdown=ScoreBreakdown(
            skills=0.8, experience=1.0, education=1.0, certifications=1.0,
            weighted_total=0.61, retrieval_score=0.0, rerank_score=0.4,
        ),
        evidence=MatchEvidence(
            skills=SkillEvidence(missing_required=["kafka"]),
            experience=ExperienceEvidence(),
            education=EducationEvidence(),
            certifications=CertificationEvidence(),
        ),
    )
    SqlitePipelineStore(path).upsert(
        PipelineEntry(job_id="j", candidate_id="c", match_snapshot=snapshot)
    )

    reopened = SqlitePipelineStore(path).get("j", "c")

    assert reopened.match_snapshot.score == 0.61
    assert reopened.match_snapshot.evidence.skills.missing_required == ["kafka"]


def test_a_database_predating_the_snapshot_column_still_opens(tmp_path):
    """CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so the column
    would never appear on a file that predates it."""
    import json
    import sqlite3

    from app.db.pipeline_store import SqlitePipelineStore

    path = str(tmp_path / "old.db")
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE pipeline (job_id TEXT NOT NULL, candidate_id TEXT NOT NULL,"
        " stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " history_json TEXT NOT NULL, PRIMARY KEY (job_id, candidate_id));"
    )
    legacy.execute(
        "INSERT INTO pipeline VALUES ('j','c','interview','2026-01-01T00:00:00+00:00',"
        "'2026-01-01T00:00:00+00:00', ?)",
        (json.dumps([{"to_stage": "applied", "at": "2026-01-01T00:00:00+00:00"}]),),
    )
    legacy.commit()
    legacy.close()

    entry = SqlitePipelineStore(path).get("j", "c")

    assert entry.stage.value == "interview"
    assert entry.match_snapshot is None  # nothing to migrate, but the row still reads

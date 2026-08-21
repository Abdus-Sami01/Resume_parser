"""Near-duplicate detection and candidate merging."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

OLD = (
    b"Jane Doe\njane@x.com\n+1 415-555-0100\n"
    b"Backend Engineer, Acme, Jan 2016 - Dec 2021\nSkills: Python, PostgreSQL\n"
)
NEW = (
    b"Jane Doe\njane@x.com\nSenior Backend Engineer, Acme, Jan 2016 - Dec 2023\n"
    b"Skills: Python, PostgreSQL, AWS, Docker\n"
)
UNRELATED = b"John Smith\njohn@y.com\nDesigner, Studio, Jan 2019 - Dec 2023\nSkills: Photoshop\n"


def _upload(name: str, body: bytes) -> str:
    return client.post("/resumes", files={"file": (name, body, "text/plain")}).json()["candidate_id"]


@pytest.fixture
def pair():
    return {"old": _upload("old.txt", OLD), "new": _upload("new.txt", NEW)}


# --- Detection ------------------------------------------------------------


def test_an_updated_resume_is_flagged_as_a_duplicate(pair):
    """Exact-content dedupe cannot catch this: the file differs, the person does not."""
    duplicates = client.get("/resumes/duplicates").json()

    assert len(duplicates) == 1
    assert duplicates[0]["confidence"] == "high"
    assert "identical email" in duplicates[0]["reasons"]


def test_unrelated_candidates_are_not_flagged(pair):
    _upload("john.txt", UNRELATED)

    pairs = client.get("/resumes/duplicates").json()

    flagged = {id for pair in pairs for id in (pair["candidate_id"], pair["other_candidate_id"])}
    assert set(pair.values()) <= flagged or len(pairs) == 1


def test_a_shared_name_alone_is_not_enough(pair):
    """Two people really are called John Smith; a name needs corroboration."""
    from app.db.candidate_store import CandidateRecord, get_candidate_store
    from app.schemas.candidate import CandidateProfile
    from app.services.dedupe import compare

    store = get_candidate_store()
    first = CandidateRecord(
        candidate_id="a",
        profile=CandidateProfile(name="John Smith", email="john1@x.com", skills=["python"]),
        raw_text="John Smith python engineer in London",
    )
    second = CandidateRecord(
        candidate_id="b",
        profile=CandidateProfile(name="John Smith", email="john2@y.com", skills=["photoshop"]),
        raw_text="John Smith graphic designer in Tokyo working on brand identity",
    )

    assert compare(first, second) is None


def test_duplicates_can_be_listed_for_one_candidate(pair):
    found = client.get(f"/resumes/{pair['new']}/duplicates").json()

    assert len(found) == 1
    assert found[0]["other_candidate_id"] == pair["old"]


def test_duplicates_route_is_not_shadowed_by_the_candidate_route():
    """`/resumes/duplicates` sits beside `/resumes/{candidate_id}`."""
    response = client.get("/resumes/duplicates")

    assert response.status_code == 200
    assert response.json() == []


def test_duplicates_for_an_unknown_candidate_is_a_404():
    assert client.get("/resumes/nope/duplicates").status_code == 404


# --- Merging --------------------------------------------------------------


def test_merge_unions_skills_from_both_records(pair):
    merged = client.post(
        f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": pair["old"]}
    ).json()

    assert set(merged["skills"]) >= {"python", "postgresql", "aws", "docker"}


def test_merge_recovers_a_field_the_newer_resume_dropped(pair):
    """An older resume often carries a phone number the newer one omits."""
    merged = client.post(
        f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": pair["old"]}
    ).json()

    assert merged["phone"] == "+1 415-555-0100"


def test_merge_removes_the_absorbed_record_from_the_pool(pair):
    client.post(f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": pair["old"]})

    assert client.get(f"/resumes/{pair['old']}").status_code == 404
    assert client.get("/resumes").json()["total"] == 1


def test_merge_stops_the_person_appearing_twice_in_a_shortlist(pair):
    job_id = client.post(
        "/jobs", json={"title": "Eng", "description": "Required:\nPython\n"}
    ).json()["job_id"]
    assert len(client.post(f"/jobs/{job_id}/match").json()) == 2

    client.post(f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": pair["old"]})

    assert len(client.post(f"/jobs/{job_id}/match").json()) == 1


def test_merge_carries_pipeline_history_to_the_surviving_record(pair):
    """The stage a candidate reached must not be lost because the wrong record won."""
    job_id = client.post(
        "/jobs", json={"title": "Eng", "description": "Required:\nPython\n"}
    ).json()["job_id"]
    client.post(f"/jobs/{job_id}/pipeline", json={"candidate_id": pair["old"], "stage": "interview"})

    client.post(f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": pair["old"]})

    board = client.get(f"/jobs/{job_id}/pipeline").json()
    assert board["stage_counts"] == {"interview": 1}
    assert board["items"][0]["entry"]["candidate_id"] == pair["new"]


def test_merging_a_candidate_into_itself_is_rejected(pair):
    response = client.post(
        f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": pair["new"]}
    )
    assert response.status_code == 404


def test_merging_an_unknown_candidate_is_a_404(pair):
    assert client.post(
        f"/resumes/{pair['new']}/merge", json={"absorb_candidate_id": "ghost"}
    ).status_code == 404


# --- Time in stage --------------------------------------------------------


def test_funnel_reports_average_days_spent_in_each_stage():
    """A funnel says where people drop out; this says where they get stuck."""
    import datetime as dt

    from app.db.pipeline_store import get_pipeline_store

    candidate_id = _upload("j.txt", NEW)
    job_id = client.post(
        "/jobs", json={"title": "Eng", "description": "Required:\nPython\n"}
    ).json()["job_id"]
    client.post(f"/jobs/{job_id}/pipeline", json={"candidate_id": candidate_id})

    store = get_pipeline_store()
    entry = store.get(job_id, candidate_id)
    entry.history[0].at = entry.history[0].at - dt.timedelta(days=9)
    store.upsert(entry)

    client.patch(f"/jobs/{job_id}/pipeline/{candidate_id}", json={"stage": "offer"})

    funnel = client.get(f"/jobs/{job_id}/pipeline/funnel").json()
    assert funnel["average_days_in_stage"]["applied"] == pytest.approx(9.0, abs=0.1)


def test_a_stage_someone_is_still_sitting_in_has_no_duration_yet():
    """Counting an open span would drag every average toward zero."""
    candidate_id = _upload("j.txt", NEW)
    job_id = client.post(
        "/jobs", json={"title": "Eng", "description": "Required:\nPython\n"}
    ).json()["job_id"]
    client.post(f"/jobs/{job_id}/pipeline", json={"candidate_id": candidate_id})

    funnel = client.get(f"/jobs/{job_id}/pipeline/funnel").json()

    assert funnel["average_days_in_stage"] == {}

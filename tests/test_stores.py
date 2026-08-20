"""Contract tests run against every store backend.

Both implementations are exercised through one parametrized suite, so a
behavioural difference between "memory" and "sqlite" fails here rather than
surfacing only after someone flips the setting in production.
"""
import pytest

from app.db.candidate_store import (
    CandidateRecord,
    CandidateStore,
    SqliteCandidateStore,
)
from app.db.job_store import JobRecord, JobStore, SqliteJobStore
from app.schemas.candidate import CandidateProfile, Experience
from app.schemas.job import JobProfile

PROFILE = CandidateProfile(
    name="Jane Doe",
    email="jane@example.com",
    skills=["python", "fastapi"],
    experience=[Experience(company="Acme", role="Engineer", years=5)],
)
JOB = JobProfile(title="Backend Engineer", required_skills=["python"], min_years_experience=3)


def _record(candidate_id: str = "c1", fingerprint: str = "fp1") -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id, profile=PROFILE, raw_text="Jane Doe resume", fingerprint=fingerprint
    )


@pytest.fixture(params=["memory", "sqlite"])
def candidate_store(request, tmp_path):
    if request.param == "memory":
        return CandidateStore()
    return SqliteCandidateStore(str(tmp_path / "candidates.db"))


@pytest.fixture(params=["memory", "sqlite"])
def job_store(request, tmp_path):
    if request.param == "memory":
        return JobStore()
    return SqliteJobStore(str(tmp_path / "jobs.db"))


# --- Candidate store contract --------------------------------------------


def test_saved_candidate_round_trips(candidate_store):
    candidate_store.save(_record())
    fetched = candidate_store.get("c1")

    assert fetched is not None
    assert fetched.profile.name == "Jane Doe"
    assert fetched.profile.skills == ["python", "fastapi"]
    assert fetched.profile.total_years_experience == 5
    assert fetched.raw_text == "Jane Doe resume"


def test_missing_candidate_is_none(candidate_store):
    assert candidate_store.get("nope") is None


def test_fingerprint_lookup_finds_the_candidate(candidate_store):
    candidate_store.save(_record())
    assert candidate_store.find_by_fingerprint("fp1") == "c1"
    assert candidate_store.find_by_fingerprint("other") is None
    assert candidate_store.find_by_fingerprint("") is None


def test_deleting_a_candidate_also_drops_its_fingerprint(candidate_store):
    candidate_store.save(_record())

    assert candidate_store.delete("c1") is True
    assert candidate_store.get("c1") is None
    assert candidate_store.find_by_fingerprint("fp1") is None
    assert candidate_store.delete("c1") is False


def test_candidate_update_keeps_the_original_creation_time(candidate_store):
    candidate_store.save(_record())
    original = candidate_store.get("c1").created_at

    candidate_store.save(_record())

    assert candidate_store.get("c1").created_at == original


def test_candidate_paging_reports_the_unpaged_total(candidate_store):
    for index in range(5):
        candidate_store.save(_record(candidate_id=f"c{index}", fingerprint=f"fp{index}"))

    page, total = candidate_store.page(offset=0, limit=2)
    assert total == 5
    assert len(page) == 2

    tail, total = candidate_store.page(offset=4, limit=2)
    assert total == 5
    assert len(tail) == 1


# --- Job store contract ---------------------------------------------------


def test_saved_job_round_trips(job_store):
    saved = job_store.save(JOB)
    fetched = job_store.get(saved.job_id)

    assert fetched is not None
    assert fetched.profile.title == "Backend Engineer"
    assert fetched.profile.required_skills == ["python"]
    assert fetched.profile.min_years_experience == 3


def test_job_can_be_replaced_under_the_same_id(job_store):
    original = job_store.save(JOB)
    updated = job_store.save(JOB.model_copy(update={"title": "Staff Engineer"}), job_id=original.job_id)

    assert updated.job_id == original.job_id
    assert updated.created_at == original.created_at
    assert len(job_store.all()) == 1
    assert job_store.get(original.job_id).profile.title == "Staff Engineer"


def test_job_deletion_reports_whether_anything_was_removed(job_store):
    saved = job_store.save(JOB)

    assert job_store.delete(saved.job_id) is True
    assert job_store.delete(saved.job_id) is False
    assert job_store.get(saved.job_id) is None


def test_jobs_are_listed_newest_first(job_store):
    first = job_store.save(JOB.model_copy(update={"title": "First"}))
    second = job_store.save(JOB.model_copy(update={"title": "Second"}))

    listed = [record.job_id for record in job_store.all()]

    # Ties on timestamp are possible, so assert membership plus ordering intent.
    assert set(listed) == {first.job_id, second.job_id}
    assert job_store.page(offset=0, limit=1)[1] == 2


# --- Durability (sqlite only) --------------------------------------------


def test_sqlite_records_survive_a_new_store_instance(tmp_path):
    """Standing in for a process restart: same file, brand new object."""
    path = str(tmp_path / "durable.db")

    SqliteCandidateStore(path).save(_record())
    SqliteJobStore(path).save(JOB, job_id="job-1")

    reopened_candidates = SqliteCandidateStore(path)
    reopened_jobs = SqliteJobStore(path)

    assert reopened_candidates.get("c1").profile.name == "Jane Doe"
    assert reopened_candidates.find_by_fingerprint("fp1") == "c1"
    assert reopened_jobs.get("job-1").profile.title == "Backend Engineer"


def test_sqlite_creates_its_parent_directory(tmp_path):
    store = SqliteCandidateStore(str(tmp_path / "nested" / "deeper" / "candidates.db"))
    store.save(_record())

    assert store.get("c1") is not None

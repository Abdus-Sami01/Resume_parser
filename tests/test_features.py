"""Pool search, find-similar, analytics, taxonomy management, and CSV export."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

POOL = [
    ("jane.txt", b"Jane Doe\njane@x.com\nSenior Backend Engineer, Acme Corp, Jan 2015 - Dec 2022\nSkills: Python, FastAPI, PostgreSQL, AWS\n"),
    ("raj.txt", b"Raj Patel\nraj@x.com\nBackend Engineer, Beta Systems, Jan 2021 - Dec 2023\nSkills: Python, Django, PostgreSQL\n"),
    ("mei.txt", b"Mei Chen\nmei@x.com\nPlatform Engineer, Nova Labs, Jan 2017 - Dec 2023\nSkills: Golang, Kubernetes, AWS, Terraform\n"),
]


@pytest.fixture
def pool():
    client.post("/resumes/bulk", files=[("files", (n, b, "text/plain")) for n, b in POOL])
    return {
        item["name"]: item["candidate_id"] for item in client.get("/resumes").json()["items"]
    }


# --- Pool search ----------------------------------------------------------


def test_pool_search_filters_by_skill(pool):
    hits = client.post("/search/candidates", json={"skills": ["python"]}).json()
    assert {hit["profile"]["name"] for hit in hits} == {"Jane Doe", "Raj Patel"}


def test_pool_search_applies_a_years_floor(pool):
    """A "5+ years" filter must exclude during retrieval, not consume a slot then get dropped."""
    hits = client.post(
        "/search/candidates", json={"skills": ["python"], "min_years_experience": 5}
    ).json()

    names = {hit["profile"]["name"] for hit in hits}
    assert "Jane Doe" in names
    assert "Raj Patel" not in names  # has python, but under the floor


def test_pool_search_applies_a_years_ceiling(pool):
    hits = client.post("/search/candidates", json={"max_years_experience": 4}).json()
    assert {hit["profile"]["name"] for hit in hits} == {"Raj Patel"}


def test_pool_search_with_no_criteria_returns_everyone(pool):
    assert len(client.post("/search/candidates", json={}).json()) == 3


def test_pool_search_ranks_by_free_text_relevance(pool):
    hits = client.post("/search/candidates", json={"query": "kubernetes terraform platform"}).json()
    assert hits[0]["profile"]["name"] == "Mei Chen"


# --- Find similar ---------------------------------------------------------


def test_similar_excludes_the_reference_candidate(pool):
    similar = client.get(f"/resumes/{pool['Jane Doe']}/similar").json()

    assert pool["Jane Doe"] not in {hit["candidate_id"] for hit in similar}
    assert len(similar) == 2


def test_similar_ranks_the_closest_profile_first(pool):
    similar = client.get(f"/resumes/{pool['Jane Doe']}/similar").json()
    # Raj shares python/postgresql with Jane; Mei shares neither.
    assert similar[0]["profile"]["name"] == "Raj Patel"


def test_similar_for_unknown_candidate_is_a_404():
    assert client.get("/resumes/nope/similar").status_code == 404


# --- Analytics ------------------------------------------------------------


def test_overview_summarizes_the_pool(pool):
    overview = client.get("/analytics/overview").json()

    assert overview["total_candidates"] == 3
    assert overview["distinct_skills"] > 0
    assert sum(overview["experience_distribution"].values()) == 3
    assert overview["median_years_experience"] > 0


def test_top_skills_count_candidates_not_mentions(pool):
    overview = client.get("/analytics/overview").json()
    by_skill = {entry["skill"]: entry for entry in overview["top_skills"]}

    assert by_skill["python"]["candidates"] == 2
    assert by_skill["python"]["share"] == pytest.approx(2 / 3, abs=0.01)


def test_skill_gaps_rank_uncovered_requirements_first(pool):
    """The actionable half: what the postings need that the pool cannot supply."""
    client.post(
        "/jobs",
        json={"title": "Backend", "description": "Required:\nPython, Kafka\n"},
    )
    client.post("/jobs", json={"title": "Data", "description": "Required:\nKafka, Spark\n"})

    gaps = client.get("/analytics/overview").json()["skill_gaps"]

    assert gaps[0]["skill"] in {"kafka", "spark"}
    assert gaps[0]["candidates_with_skill"] == 0
    assert gaps[0]["coverage"] == 0.0

    kafka = next(gap for gap in gaps if gap["skill"] == "kafka")
    assert kafka["required_by_jobs"] == 2  # demanded twice, held by nobody


def test_overview_is_safe_on_an_empty_pool():
    overview = client.get("/analytics/overview").json()

    assert overview["total_candidates"] == 0
    assert overview["median_years_experience"] == 0.0
    assert overview["top_skills"] == []


# --- Taxonomy management --------------------------------------------------


def test_new_skill_becomes_matchable_immediately(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.services.taxonomy.skill_standardizer import get_skill_standardizer

    monkeypatch.setenv("CUSTOM_SKILLS_PATH", str(tmp_path / "custom.json"))
    get_settings.cache_clear()
    get_skill_standardizer.cache_clear()
    try:
        before = client.post("/skills/standardize", json={"values": ["od-cli"]}).json()
        assert before["od-cli"] is None

        created = client.post("/skills", json={"skill": "ourdeploy", "aliases": ["od-cli"]})
        assert created.status_code == 201

        after = client.post("/skills/standardize", json={"values": ["od-cli", "OurDeploy"]}).json()
        assert after == {"od-cli": "ourdeploy", "OurDeploy": "ourdeploy"}
    finally:
        get_skill_standardizer.cache_clear()
        get_settings.cache_clear()


def test_custom_skills_persist_to_the_overlay_not_the_bundled_file(tmp_path, monkeypatch):
    """The shipped taxonomy must stay updatable without clobbering local additions."""
    import json

    from app.config import get_settings
    from app.services.taxonomy.skill_standardizer import _TAXONOMY_PATH, get_skill_standardizer

    overlay = tmp_path / "custom.json"
    monkeypatch.setenv("CUSTOM_SKILLS_PATH", str(overlay))
    get_settings.cache_clear()
    get_skill_standardizer.cache_clear()
    try:
        bundled_before = _TAXONOMY_PATH.read_text()
        client.post("/skills", json={"skill": "ourdeploy", "aliases": ["od-cli"]})

        assert "ourdeploy" in json.loads(overlay.read_text())
        assert _TAXONOMY_PATH.read_text() == bundled_before  # untouched
    finally:
        get_skill_standardizer.cache_clear()
        get_settings.cache_clear()


def test_taxonomy_can_be_listed_and_pruned(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.services.taxonomy.skill_standardizer import get_skill_standardizer

    monkeypatch.setenv("CUSTOM_SKILLS_PATH", str(tmp_path / "custom.json"))
    get_settings.cache_clear()
    get_skill_standardizer.cache_clear()
    try:
        assert client.get("/skills").json()["total"] > 10

        client.post("/skills", json={"skill": "throwaway", "aliases": []})
        assert client.delete("/skills/throwaway").status_code == 204
        assert client.delete("/skills/throwaway").status_code == 404
    finally:
        get_skill_standardizer.cache_clear()
        get_settings.cache_clear()


# --- CSV export -----------------------------------------------------------


def test_csv_export_carries_the_evidence_not_just_scores(pool):
    """A shortlist of bare numbers sends the reader back to the API to learn why."""
    import csv
    import io

    job_id = client.post(
        "/jobs", json={"title": "Backend", "description": "Required:\nPython, PostgreSQL, Kafka\n"}
    ).json()["job_id"]

    response = client.get(f"/jobs/{job_id}/match/export")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert f"matches-{job_id}.csv" in response.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert [row["rank"] for row in rows] == ["1", "2", "3"]

    jane = next(row for row in rows if row["name"] == "Jane Doe")
    assert jane["email"] == "jane@x.com"
    assert "kafka" in jane["missing_required_skills"]  # nobody in the pool has it
    assert "python" in jane["matched_required_skills"]
    assert float(jane["years_experience"]) > 7

    mei = next(row for row in rows if row["name"] == "Mei Chen")
    assert mei["matched_required_skills"] == ""  # shares none of the required skills


def test_csv_export_for_unknown_job_is_a_404():
    assert client.get("/jobs/nope/match/export").status_code == 404


# --- Serialized computed field -------------------------------------------


def test_total_years_experience_is_returned_by_the_api(pool):
    profile = client.get(f"/resumes/{pool['Jane Doe']}").json()
    assert profile["total_years_experience"] > 7


# --- Job status ------------------------------------------------------------


def test_job_status_defaults_to_open():
    created = client.post("/jobs", json={"title": "Eng", "description": "Required:\nPython\n"}).json()
    assert created["profile"]["status"] == "open"


def test_status_can_be_changed_without_reparsing():
    job_id = client.post(
        "/jobs", json={"title": "Eng", "description": "Required:\nPython, PostgreSQL\n"}
    ).json()["job_id"]

    updated = client.patch(f"/jobs/{job_id}/status", json={"status": "filled"})

    assert updated.status_code == 200
    assert updated.json()["profile"]["status"] == "filled"
    # The parsed requirements survive the status change.
    assert "python" in updated.json()["profile"]["required_skills"]


def test_closed_roles_drop_out_of_skill_gap_analysis(pool):
    """Sourcing against a req nobody is hiring for is wasted effort."""
    job_id = client.post(
        "/jobs", json={"title": "Data", "description": "Required:\nKafka\n"}
    ).json()["job_id"]
    assert any(g["skill"] == "kafka" for g in client.get("/analytics/overview").json()["skill_gaps"])

    client.patch(f"/jobs/{job_id}/status", json={"status": "closed"})

    assert not any(
        g["skill"] == "kafka" for g in client.get("/analytics/overview").json()["skill_gaps"]
    )


def test_jobs_can_be_filtered_by_status():
    open_id = client.post("/jobs", json={"title": "Open", "description": "Required:\nPython\n"}).json()["job_id"]
    filled_id = client.post("/jobs", json={"title": "Filled", "description": "Required:\nPython\n"}).json()["job_id"]
    client.patch(f"/jobs/{filled_id}/status", json={"status": "filled"})

    assert [i["profile"]["title"] for i in client.get("/jobs?status=open").json()["items"]] == ["Open"]
    assert [i["profile"]["title"] for i in client.get("/jobs?status=filled").json()["items"]] == ["Filled"]
    assert client.get("/jobs").json()["total"] == 2  # unfiltered still shows everything


def test_status_endpoints_404_on_an_unknown_job():
    assert client.get("/jobs/nope/status").status_code == 404
    assert client.patch("/jobs/nope/status", json={"status": "closed"}).status_code == 404


def test_an_invalid_status_is_rejected():
    job_id = client.post("/jobs", json={"title": "Eng", "description": "Required:\nPython\n"}).json()["job_id"]
    assert client.patch(f"/jobs/{job_id}/status", json={"status": "banana"}).status_code == 422


# --- Parse coverage -------------------------------------------------------


def _coverage(review: int = 10) -> dict:
    response = client.get("/analytics/parse-coverage", params={"review": review})
    assert response.status_code == 200
    return {row["field"]: row for row in response.json()["fields"]}


def test_parse_coverage_reports_every_scored_field(pool):
    fields = _coverage()

    assert set(fields) == {
        "email",
        "experience",
        "education",
        "skills",
        "experience.role",
        "experience.end_year",
        "experience.achievements",
    }


def test_parse_coverage_names_what_the_scorer_assumes_for_each_gap(pool):
    """A coverage number without its consequence is trivia; the pairing is the report."""
    assert "recency" in _coverage()["experience.end_year"]["scorer_assumption"]


def test_a_pool_with_no_bullets_reports_zero_achievement_coverage(pool):
    """Every fixture resume is a title and a skills line, so no role carries bullets."""
    achievements = _coverage()["experience.achievements"]

    assert achievements["present"] == 0
    assert achievements["coverage"] == 0.0


def test_bullets_lift_achievement_coverage():
    client.post(
        "/resumes",
        files={
            "file": (
                "ann.txt",
                b"Ann Lee\nann@x.com\nBackend Engineer, Acme, Jan 2020 - Dec 2023\n"
                b"- Built Python services\nSkills: Python\n",
                "text/plain",
            )
        },
    )

    assert _coverage()["experience.achievements"]["coverage"] == 1.0


def test_a_current_role_is_not_counted_as_a_missing_end_date():
    """Nothing failed to parse: the resume says the role is ongoing."""
    client.post(
        "/resumes",
        files={
            "file": (
                "bo.txt",
                b"Bo Ng\nbo@x.com\nBackend Engineer, Acme, Jan 2020 - Present\nSkills: Python\n",
                "text/plain",
            )
        },
    )

    assert _coverage()["experience.end_year"]["missing"] == 0


def test_needs_review_is_ordered_worst_first(pool):
    review = client.get("/analytics/parse-coverage").json()["needs_review"]

    assert review
    counts = [len(entry["missing_fields"]) for entry in review]
    assert counts == sorted(counts, reverse=True)


def test_parse_coverage_on_an_empty_pool_reports_full_coverage_not_zero():
    """With nothing stored, 0% would read as a broken extractor rather than no data."""
    body = client.get("/analytics/parse-coverage").json()

    assert body["total_candidates"] == 0
    assert body["needs_review"] == []
    assert all(row["coverage"] == 1.0 for row in body["fields"])

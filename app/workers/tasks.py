"""Background tasks: heavy parsing/extraction/indexing runs off the request path.

Marker-class PDF parsing plus an LLM extraction call can take tens of seconds,
which is longer than a typical HTTP timeout, so uploads are dispatched here
rather than handled inline.
"""
from typing import Any

from app.config import get_settings
from app.services.extraction.document_parser import get_document_parser
from app.services.extraction.resume_extractor import get_resume_extractor
from app.services.search.matcher import index_candidate
from app.workers.celery_app import celery_app

# Eager mode has no result backend to query later, so completed runs are kept here
# to give the status endpoint identical behaviour in both modes.
_EAGER_RESULTS: dict[str, dict[str, Any]] = {}


@celery_app.task(name="app.workers.tasks.parse_and_index_resume")
def parse_and_index_resume(file_bytes: bytes, filename: str) -> str:
    raw_text = get_document_parser().parse(file_bytes, filename)
    profile = get_resume_extractor().extract(raw_text)
    return index_candidate(profile, raw_text)


def submit_resume_parse(file_bytes: bytes, filename: str) -> str:
    """Dispatch a parse job and return its task id."""
    async_result = parse_and_index_resume.delay(file_bytes, filename)

    if get_settings().task_backend == "eager":
        _EAGER_RESULTS[async_result.id] = _describe(async_result)

    return async_result.id


def get_task_state(task_id: str) -> dict[str, Any]:
    if task_id in _EAGER_RESULTS:
        return _EAGER_RESULTS[task_id]
    return _describe(celery_app.AsyncResult(task_id))


def _describe(async_result: Any) -> dict[str, Any]:
    state = async_result.state
    described: dict[str, Any] = {"task_id": async_result.id, "state": state}

    if state == "SUCCESS":
        described["candidate_id"] = async_result.result
    elif state == "FAILURE":
        described["error"] = str(async_result.result)

    return described

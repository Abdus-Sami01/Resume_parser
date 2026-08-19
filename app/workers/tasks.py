"""Background tasks: heavy parsing/extraction/indexing runs off the request path."""
from app.services.extraction.document_parser import get_document_parser
from app.services.extraction.resume_extractor import get_resume_extractor
from app.services.search.matcher import index_candidate
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks.parse_and_index_resume")
def parse_and_index_resume(file_bytes: bytes, filename: str) -> str:
    raw_text = get_document_parser().parse(file_bytes, filename)
    profile = get_resume_extractor().extract(raw_text)
    return index_candidate(profile, raw_text)

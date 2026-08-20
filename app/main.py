import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import analytics, jobs, resumes, search, skills
from app.config import get_settings
from app.services.search.matcher import reindex_if_index_is_empty

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.reindex_on_startup:
        restored = reindex_if_index_is_empty()
        if restored:
            logger.info("rebuilt vector index for %d persisted candidate(s)", restored)
    yield


app = FastAPI(
    title="Resume Parser & Semantic Matching",
    description="Structured resume/JD extraction with hybrid retrieval + reranking.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(resumes.router)
app.include_router(jobs.router)
app.include_router(search.router)
app.include_router(analytics.router)
app.include_router(skills.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

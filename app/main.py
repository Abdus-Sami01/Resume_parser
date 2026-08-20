import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from app.api.routes import analytics, jobs, resumes, search, skills
from app.api.security import require_api_key, warn_if_unauthenticated
from app.config import get_settings
from app.services.search.matcher import reindex_if_index_is_empty

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    warn_if_unauthenticated()

    if settings.reindex_on_startup:
        restored = reindex_if_index_is_empty()
        if restored:
            logger.info("rebuilt vector index for %d persisted candidate(s)", restored)
    yield


app = FastAPI(
    title="Resume Parser & Semantic Matching",
    description="Structured resume/JD extraction with hybrid retrieval + reranking.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_request_timing(request: Request, call_next):
    """Times every request and returns it as a header.

    Latency here is dominated by model calls whose cost is invisible from the
    outside — an embedding round trip, a cross-encoder pass over fifty
    candidates — so a caller seeing a slow match has no way to tell a slow model
    from a slow network without this.
    """
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s -> %s in %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# Auth is applied per-router rather than per-endpoint so a new route cannot be
# added later that quietly skips it.
_protected = [Depends(require_api_key)]

app.include_router(resumes.router, dependencies=_protected)
app.include_router(jobs.router, dependencies=_protected)
app.include_router(search.router, dependencies=_protected)
app.include_router(analytics.router, dependencies=_protected)
app.include_router(skills.router, dependencies=_protected)


@app.get("/health")
async def health() -> dict:
    """Unauthenticated on purpose: load balancers and probes cannot carry a key."""
    return {"status": "ok"}

import pytest

from app.config import get_settings
from app.db.candidate_store import get_candidate_store
from app.db.job_store import get_job_store
from app.db.pipeline_store import get_pipeline_store
from app.services.rate_limit import get_rate_limiter
from app.services.search.vector_store import get_vector_store


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Each test gets fresh in-memory stores."""
    get_settings.cache_clear()
    get_vector_store.cache_clear()
    get_candidate_store.cache_clear()
    get_job_store.cache_clear()
    get_pipeline_store.cache_clear()
    get_rate_limiter.cache_clear()
    yield

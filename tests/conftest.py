import pytest

from app.config import get_settings
from app.db.candidate_store import get_candidate_store
from app.services.search.vector_store import get_vector_store


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Each test gets a fresh in-memory vector store / candidate store."""
    get_settings.cache_clear()
    get_vector_store.cache_clear()
    get_candidate_store.cache_clear()
    yield

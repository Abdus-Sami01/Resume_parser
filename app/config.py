"""Environment-driven settings for every pluggable backend in the pipeline."""
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Extraction pipeline
    extraction_backend: Literal["heuristic", "llm"] = "heuristic"
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "gpt-4o-mini"

    # Vector search pipeline
    vector_store_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "resumes"

    # Embeddings
    embedding_backend: Literal["hash", "openai"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256

    # Reranking
    reranker_backend: Literal["lexical", "cross_encoder"] = "lexical"
    reranker_model: str = "BAAI/bge-reranker-large"

    # Scoring weights
    weight_experience: float = 0.5
    weight_skills: float = 0.4
    weight_education: float = 0.1

    # Record storage
    # "memory" is in-process and lost on restart; "sqlite" persists to a file and
    # is visible to Celery workers as well as the API.
    store_backend: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "data/resume_parser.db"
    # Re-embeds every stored resume when the index is empty but records exist.
    reindex_on_startup: bool = True

    # Skill taxonomy overlay: runtime additions land here, leaving the bundled
    # skills.json free to be updated by a future release.
    custom_skills_path: str = "data/custom_skills.json"

    # Security. Comma-separated API keys; empty disables auth (development only).
    api_keys: str = ""

    # Rate limiting. 0 disables it. "memory" is per-process and undercounts across
    # workers; "redis" shares one window cluster-wide.
    rate_limit_per_minute: int = 0
    rate_limit_backend: Literal["memory", "redis"] = "memory"

    # Uploads
    max_upload_bytes: int = 10 * 1024 * 1024

    # Retrieval
    retrieval_top_k: int = 50
    rerank_top_n: int = 10

    # Celery / Redis
    # "eager" runs tasks inline in the calling process (no broker needed);
    # "celery" dispatches to a worker over Redis.
    task_backend: Literal["eager", "celery"] = "eager"
    redis_url: str = "redis://localhost:6379/0"

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Settings":
        total = self.weight_experience + self.weight_skills + self.weight_education
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

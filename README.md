# Resume Parser & Semantic Matching Platform

A production-shaped implementation of a two-pipeline resume intelligence system:

1. **Ingestion / Parsing Pipeline** — turns messy resume files (PDF/DOCX) and job
   descriptions into strict, validated JSON profiles.
2. **Semantic Matching / Scoring Pipeline** — hybrid (dense + sparse) retrieval
   over a vector database, followed by cross-encoder reranking, to produce
   ranked candidates with an explainable score breakdown.

```
[Resume (PDF/Docx)] -> [Layout-aware Text Extraction] -> [Structured Extraction] -+
                                                                                   |
[Job Description]   -> [Structured Extraction & Weighting] --------------------->-+
                                                                                   v
                                                                    [Structured JSON Profiles]
                                                                                   |
                                                                                   v
                          [Vector DB] <- Hybrid Search (Dense + BM25/Sparse) <----+
                                 |
                                 v
                [Cross-Encoder Reranker] -> [Ranked Candidates + Score Breakdown]
```

## Tech stack (and where it lives in this repo)

| Layer | Technology | Module |
|---|---|---|
| Backend API | FastAPI + Pydantic v2 | `app/main.py`, `app/api/` |
| Document processing | Marker / Textract (pluggable) | `app/services/extraction/document_parser.py` |
| Extraction engine | Instructor + LLM (pluggable, heuristic fallback) | `app/services/extraction/resume_extractor.py`, `job_extractor.py` |
| Skill standardization | Taxonomy + fuzzy matching | `app/services/taxonomy/skill_standardizer.py` |
| Embeddings | OpenAI / BGE (pluggable) | `app/services/search/embeddings.py` |
| Vector DB | Qdrant (pluggable, in-memory fallback) | `app/services/search/vector_store.py` |
| Reranking | Cross-encoder (pluggable, lexical fallback) | `app/services/search/reranker.py` |
| Matching orchestration | Two-stage retrieval + weighted scoring | `app/services/search/matcher.py` |
| Async tasks | Celery + Redis | `app/workers/` |

Every AI/infra dependency (LLM client, embedding model, vector DB, reranker) is
defined behind a small `Protocol` interface with a **local, dependency-free
fallback** implementation. This means:

- The app boots and the full pipeline is exercisable with **zero external
  services or API keys** (heuristic extraction, in-memory cosine search,
  lexical-overlap reranking).
- Swapping in the real production backend (OpenAI/Groq + Instructor, Qdrant,
  a `bge-reranker` cross-encoder) is a one-line change in `app/config.py`
  (`EXTRACTION_BACKEND`, `VECTOR_STORE_BACKEND`, `RERANKER_BACKEND`), because
  the orchestration code only depends on the Protocol, never the concrete
  class.

## Project layout

```
app/
  config.py                 Settings (env-driven, pydantic-settings)
  main.py                   FastAPI app + routers
  api/routes/                resumes.py, jobs.py, search.py
  schemas/                   CandidateProfile, JobProfile, MatchResult, ...
  services/extraction/       document parsing + structured extraction
  services/taxonomy/         skill standardization against a taxonomy
  services/search/           embeddings, vector store, reranker, matcher
  workers/                   Celery app + background tasks
  db/                        Qdrant client factory
tests/                       pytest suite (runs fully offline via fallbacks)
scripts/seed_taxonomy.py     loads the bundled skill taxonomy
docker-compose.yml           Qdrant + Redis for local production-like runs
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline / heuristic mode (no external services needed)
uvicorn app.main:app --reload

# Production-like mode
docker compose up -d          # starts Qdrant + Redis
export EXTRACTION_BACKEND=llm
export OPENAI_API_KEY=sk-...
export VECTOR_STORE_BACKEND=qdrant
export RERANKER_BACKEND=cross_encoder
uvicorn app.main:app --reload
celery -A app.workers.celery_app worker --loglevel=info
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite exercises schema validation, skill standardization, the
two-stage matcher, and the API endpoints entirely against the fallback
(in-process) backends, so it needs no network access, API keys, or running
services.

## Score breakdown

`MatchResult` reports not just a single number but a weighted breakdown
(`skills`, `experience`, `education`) plus the raw retrieval score and the
reranker score, so the API response is explainable rather than a black box.

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

## Matching API

`POST /search/match` takes the job profile plus optional retrieval controls:

```json
{
  "job": { "title": "Backend Engineer", "required_skills": ["python"], "...": "..." },
  "top_k": 50,
  "top_n": 10,
  "filters": { "skills": ["python"], "location": "Remote" }
}
```

`filters` are **hard constraints** applied during stage-1 retrieval, not score
nudges — a candidate missing a filtered skill is never retrieved, so it cannot
be rescued by a high semantic score. Scalar values compare by equality; list
values require every item to be present. Both backends implement identical
semantics (in-memory predicate, Qdrant `must` conditions).

## Score breakdown

`MatchResult` reports not just a single number but a weighted breakdown
(`skills`, `experience`, `education`) plus the raw retrieval score and the
reranker score, so the API response is explainable rather than a black box.

`experience` carries the largest weight (0.5 by default), so how tenure is
counted matters more than anything else in the rubric. The heuristic extractor
parses each dated role separately and **sums** the tenures — reading the longest
single "N years" mention understates anyone who lists roles individually, and
misses resumes written purely as date ranges (`Jan 2019 - Dec 2022`), which is
most of them. A self-reported "N years" phrase is used only as a fallback when
there are no dates to work from.

## Notes on the search backends

- **Hybrid means hybrid.** The Qdrant backend runs the dense and sparse
  branches as separate prefetches and fuses them with Reciprocal Rank Fusion
  server-side. Querying only the dense vector while still writing a sparse one
  silently throws away the keyword half of the index.
- **Sparse term indices must be process-stable.** They are derived with
  `blake2b`, not Python's builtin `hash()`, which is randomized per interpreter
  via `PYTHONHASHSEED` — under `hash()` a term indexed by a Celery worker lands
  on a different index than the same term at query time, making every sparse
  vector unmatchable across processes. `test_sparse_index_is_stable_across_interpreter_processes`
  guards this.
- **Set `QDRANT_URL=:memory:`** to run the Qdrant backend in qdrant-client's
  embedded mode — no server required, which is how the production backend gets
  real test coverage instead of only ever running in production.
- **Reranking is batch-first by interface.** `Reranker.score_batch(query, documents)`
  takes the whole candidate set, because a cross-encoder jointly encodes each
  pair — a per-pair interface silently costs one forward pass per candidate
  (50 sequential passes at the default `RETRIEVAL_TOP_K`) and gets harder to
  change as callers accumulate.

## Skill vocabulary

Both the resume side and the job side run their skills through the same
taxonomy before matching, since `_skills_score` is a set intersection —
if one side emits `"React.js"` and the other `"react"`, the intersection is
empty and the score silently collapses to zero. Skills the taxonomy does not
recognise are kept lowercased rather than dropped, so niche skills still match
by exact string.

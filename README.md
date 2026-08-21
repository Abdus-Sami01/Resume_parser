# Resume Parser & Semantic Matching Platform

[![CI](https://github.com/Abdus-Sami01/Resume_parser/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdus-Sami01/Resume_parser/actions/workflows/ci.yml)

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
| Record storage | SQLite (pluggable, in-memory fallback) | `app/db/` |
| Authentication | API key header | `app/api/security.py` |
| Blind screening | Field redaction | `app/services/privacy.py` |
| Rate limiting | Sliding window (memory or Redis) | `app/services/rate_limit.py` |
| Hiring pipeline | Stage tracking + audit trail | `app/services/pipeline.py`, `app/db/pipeline_store.py` |

Every AI/infra dependency (LLM client, embedding model, vector DB, reranker) is
defined behind a small `Protocol` interface with a **local, dependency-free
fallback** implementation. This means:

- The app boots and the full pipeline is exercisable with **zero external
  services or API keys** (heuristic extraction, in-memory cosine search,
  lexical-overlap reranking).
- Swapping in the real production backend (OpenAI/Groq + Instructor, Qdrant,
  a `bge-reranker` cross-encoder) is a one-line change in `app/config.py`
  (`EXTRACTION_BACKEND`, `VECTOR_STORE_BACKEND`, `RERANKER_BACKEND`, `TASK_BACKEND`), because
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
  db/                        candidate + job record stores
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
export TASK_BACKEND=celery
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

## Authentication

The service stores resumes — personal data — so an open deployment exposes names,
emails, and phone numbers to anyone who can reach the port.

Set `API_KEYS` to a comma-separated list and every endpoint requires a matching
`X-API-Key` header. `/health` stays open, because load balancers and container
probes cannot present a key.

```bash
export API_KEYS=key-one,key-two
curl -H "X-API-Key: key-one" localhost:8000/resumes
```

With `API_KEYS` empty, auth is **off** and everything is reachable. That keeps
the zero-config local run working, and it is a development default rather than a
safe production one — so startup logs a warning whenever it applies, instead of
letting an open deployment pass silently. Keys are compared with
`hmac.compare_digest`, so a caller cannot probe one character at a time, and the
dependency is attached per-router rather than per-endpoint so a route added later
cannot quietly skip it.

## Blind screening

`?blind=true` on the match endpoints (and `blind: true` on `POST /search/match`)
redacts name, email, phone, free-text summary, and institution names:

```
normal: name='Jane Doe'            email='jane.doe@example.com'  school='Stanford University'
blind : name='Candidate 22230152'  email=None                    school=''
        skills=['python','postgresql','aws','fastapi']  years=7.92  degree='B.S. in Computer Science'
```

Skills, tenure, and the degree itself stay — everything the decision actually
rests on. The summary goes because free text routinely reintroduces a name or
pronoun that the structured fields just removed, and institution goes because
prestige skews a first pass.

**Ranking is identical either way.** Scoring never reads the redacted fields, so
a blind shortlist is the same shortlist in the same order; the only difference is
what the reviewer sees. A test asserts this rather than leaving it to trust, and
another asserts the stored record is never mutated by redaction.

## Observability

Every response carries `X-Process-Time-Ms` and every request is logged with
method, path, status, and duration. Latency here is dominated by model calls that
are invisible from outside — an embedding round trip, a cross-encoder pass over
fifty candidates — so without this a caller cannot distinguish a slow model from
a slow network.

## API surface

**Candidates**

| Endpoint | Purpose |
|---|---|
| `POST /resumes` | Parse inline and return the profile. Fine for text; can outlast an HTTP timeout under Marker + LLM extraction. |
| `POST /resumes/bulk` | Ingest a batch. Reports per-file outcomes — one unreadable file never discards the rest. |
| `POST /resumes/async` | Dispatch parsing to a worker, returns `202` with a `task_id`. |
| `GET /resumes` | Paginated candidate list (`offset`, `limit`) with an unpaged `total`. |
| `GET /resumes/tasks/{task_id}` | Task state, plus `candidate_id` once it succeeds. |
| `GET /resumes/{candidate_id}` | Fetch a parsed profile. |
| `DELETE /resumes/{candidate_id}` | Erase the profile **and** its index entry. |
| `GET /resumes/{candidate_id}/jobs` | Reverse match: rank stored postings for this candidate. |

**Jobs**

| Endpoint | Purpose |
|---|---|
| `POST /jobs/parse` | Parse without persisting — preview how a posting is interpreted. |
| `POST /jobs` | Parse and persist, returns a `job_id`. |
| `GET /jobs` | Paginated posting catalogue, newest first. |
| `GET`/`PUT`/`DELETE /jobs/{job_id}` | Read, replace in place (keeping the id), or remove. |
| `POST /jobs/{job_id}/match` | Re-run a saved posting against the current candidate pool. |

**Search**

| Endpoint | Purpose |
|---|---|
| `POST /search/match` | Match an ad-hoc job profile without persisting it. |
| `POST /search/candidates` | Search the pool directly — free text plus skills, location, and experience range. |
| `GET /resumes/{candidate_id}/similar` | "More people like this one." |
| `GET /resumes/duplicates` | Likely same-person records across the pool. |
| `GET /resumes/{candidate_id}/duplicates` | Likely duplicates of one candidate. |
| `POST /resumes/{candidate_id}/merge` | Fold another record into this one. |

**Hiring pipeline**

| Endpoint | Purpose |
|---|---|
| `POST /jobs/{job_id}/pipeline` | Add a candidate to a role's pipeline. |
| `POST /jobs/{job_id}/pipeline/shortlist` | Run the match and add the top results in one call. |
| `PATCH /jobs/{job_id}/pipeline` | Move several candidates at once. |
| `GET /jobs/{job_id}/pipeline` | The board, with per-stage counts. Supports `?stage=` and `?blind=true`. |
| `PATCH /jobs/{job_id}/pipeline/{candidate_id}` | Move a stage, appending to the audit trail. |
| `DELETE /jobs/{job_id}/pipeline/{candidate_id}` | Remove from the pipeline. |
| `GET /jobs/{job_id}/pipeline/funnel` | Stage counts and conversion between steps. |
| `GET /resumes/{candidate_id}/applications` | Every role this candidate is in play for. |

**Analytics and taxonomy**

| Endpoint | Purpose |
|---|---|
| `GET /analytics/overview` | Pool composition, experience bands, top skills, and skill gaps. |
| `GET /jobs/{job_id}/match/export` | Shortlist as CSV, evidence columns included. |
| `GET`/`POST /skills` | List the skill taxonomy, or add a skill and its aliases. |
| `DELETE /skills/{skill}` | Remove a skill from the taxonomy. |
| `POST /skills/standardize` | Preview how raw strings resolve — for debugging a low score. |

Deletion reaches both stores deliberately. Resumes are personal data, and a
profile erased from the record store but left in the vector index is still
discoverable through search — so `DELETE` is not complete until the vector is
gone too.

`TASK_BACKEND` picks how `/resumes/async` runs. `eager` (default) executes the
task inline in the API process, so the endpoint works with no broker running;
`celery` dispatches to a real worker over Redis. Both report state through the
same task endpoint, so client code does not change between them.

A malformed upload returns `422` with the failing filename rather than a `500`;
one over `MAX_UPLOAD_BYTES` returns `413`. The body is read in chunks and abandoned
at the cap, so an oversized file is never fully buffered into memory.

Re-uploading the same resume updates that candidate instead of creating a second
one — identity is a SHA-256 of the whitespace-normalized text, so the same content
under a different filename resolves to the same candidate. Merging by email (an
updated resume replacing an older one) is deliberately **not** done here: that is a
product decision about overwriting history, not an obvious correctness fix.

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

## Searching the pool without a posting

The job-driven path answers "who fits this role". `POST /search/candidates`
answers the other half of the workflow — "who do we already have" — where the
criteria are in a recruiter's head rather than in a written posting:

```json
{ "query": "backend python services", "skills": ["python"], "min_years_experience": 5 }
```

Experience bounds are **retrieval filters, not post-filters**. A candidate under
the floor is never retrieved, so they cannot occupy one of the top-k slots and
push a qualified person out of the results. The filter language has three shapes,
chosen by value type: scalar means equality, a list means every item must be
present, and a dict means a numeric range (`{"gte": 5}`).

## Talent-pool analytics

`GET /analytics/overview` reports composition — headcount, distinct skills,
median experience, seniority bands, and the most common skills counted by
*candidate* rather than by mention.

`skill_gaps` is the actionable half. It cross-references what open postings
require against what the pool actually holds, worst coverage first:

```
kafka        wanted by 2 job(s); 0 candidate(s) have it; coverage 0%
kubernetes   wanted by 1 job(s); 1 candidate(s) have it; coverage 25%
python       wanted by 1 job(s); 2 candidate(s) have it; coverage 50%
```

That names what to source for, rather than restating what you already have.

## Extending the skill taxonomy

The taxonomy decides whether "K8s" on a resume matches "Kubernetes" in a posting,
so a deployment that cannot extend it silently mis-scores its own niche skills —
internal tooling, new frameworks, regional certifications. `POST /skills` adds a
skill or merges aliases into an existing one, and it takes effect immediately.

Additions are written to an overlay file (`CUSTOM_SKILLS_PATH`), never to the
bundled `skills.json`, so a future release can ship an updated taxonomy without
clobbering whatever a deployment added locally.

`POST /skills/standardize` previews how raw strings resolve, which is usually the
fastest way to explain why a match scored lower than expected.

## Hiring pipeline

Stage belongs to a **(candidate, job) pair**, not to a candidate. The same person
can be at `offer` for one role and `rejected` for another, and a single global
status could not express that — `GET /resumes/{id}/applications` returns all of
them.

Stages are `applied → screening → interview → offer → hired`, plus `rejected` and
`withdrawn` as exits. Transitions are deliberately **not** constrained by a state
machine: real processes send people back a stage, revive a rejection, or skip a
step, and a rigid graph only teaches users to work around it. What makes that safe
is the audit trail — every move records the previous stage, the new one, who made
it, and why:

```
None    -> applied    by recruiter@co  ''
applied -> screening  by hm@co         'strong python'
screening -> interview by hm@co        'panel booked'
```

Moving to the stage someone is already in is a no-op, so a mis-click never lands
in the history as a decision that was made.

### Shortlisting

Matching produced a ranked list and the pipeline accepted one candidate per call,
so acting on a top-ten meant eleven requests plus a client-side loop that had to
re-derive the ranking to know what to send. `POST /jobs/{id}/pipeline/shortlist`
closes that loop:

```json
{ "top_n": 3, "min_score": 0.55, "stage": "screening", "actor": "recruiter@co" }
→ { "added": 3, "skipped": 0, "entries": [...] }
```

Re-running it after new resumes arrive is the normal way to use it, so candidates
already in the pipeline are **reported as skipped rather than raised** — the
overlap must not fail the call. `min_score` stops at the first result below the
floor, since the list is already ranked.

`PATCH /jobs/{id}/pipeline` moves a batch: rejecting the tail of a shortlist is
one decision, not twenty. One stale id does not discard the other nineteen moves —
failures come back per candidate, and every successful move still lands in the
audit trail with its note and actor.

### Funnel

`GET /jobs/{id}/pipeline/funnel` counts stages **ever reached**, not just current
ones. Counting current stages would report a funnel that never converts — someone
now at `offer` has already passed through screening and interview, so counting
only where they stand now shows those steps as empty:

```
applied    here=0  ever_reached=3  conv=  -
screening  here=1  ever_reached=2  conv=67%
interview  here=0  ever_reached=1  conv=50%
offer      here=1  ever_reached=1  conv=100%
```

Erasing a candidate clears their pipeline entries too. Stage notes carry the
candidate id alongside free text a reviewer wrote about them, so leaving those
behind would defeat the erasure.

## Rate limiting

`RATE_LIMIT_PER_MINUTE` throttles per caller — by API key when one is present, by
client IP otherwise. Keying on IP alone would put an entire corporate NAT into a
single bucket; keying on the API key gives each tenant its own quota.

```
req 1: 200  X-RateLimit-Remaining=2
req 3: 200  X-RateLimit-Remaining=0
req 4: 429  Retry-After=60
```

The check runs **before** any work: parsing an upload runs a document pipeline and,
in production, an LLM call, so the request has to be rejected before that cost is
incurred rather than after. `/health` is exempt, because throttling a health check
would take the service out of its own load balancer.

`RATE_LIMIT_BACKEND=memory` (default) is a per-process sliding window — correct
for one worker and **honestly wrong for several**, since four uvicorn workers each
admit the full quota and the effective limit is four times what was configured.
Use `redis` whenever more than one process serves traffic; it shares one window
across every worker and machine. It is a true sliding window rather than a fixed
bucket, so a caller cannot send a full quota at 0:59 and another at 1:00.

## Score breakdown and match evidence

`MatchResult` reports a weighted `breakdown` (`skills`, `experience`,
`education`) plus the raw retrieval and reranker scores — and an `evidence`
object naming what each score was computed from. A bare `0.62` tells a recruiter
a candidate ranked mid-pack but not why, and not what would change it:

```json
{
  "breakdown": { "skills": 0.84, "experience": 1.0, "weighted_total": 0.59, "...": "..." },
  "evidence": {
    "skills": {
      "matched_required": ["fastapi", "postgresql", "python"],
      "missing_required": [],
      "matched_preferred": ["aws"],
      "extra": ["docker"]
    },
    "experience": { "candidate_years": 3.92, "required_years": 3.0, "meets_requirement": true },
    "education":  { "required": "Computer Science", "matched_degree": "B.S.", "meets_requirement": true }
  }
}
```

Evidence is produced by the same functions that compute the scores, returned as
`(score, evidence)` pairs, so the explanation cannot drift out of sync with the
number it explains.

`experience` carries the largest weight (0.5 by default), so how tenure is
counted matters more than anything else in the rubric. The heuristic extractor
parses each dated role separately and **sums** the tenures — reading the longest
single "N years" mention understates anyone who lists roles individually, and
misses resumes written purely as date ranges (`Jan 2019 - Dec 2022`), which is
most of them. A self-reported "N years" phrase is used only as a fallback when
there are no dates to work from.

The job side reads its experience bar from the requirements block alone. Scanning
a whole posting lets unrelated prose set it — "a company with 20 years of history"
turns a 3-year requirement into a 20-year one, and a "preferred" nice-to-have
outranks the real minimum. Either one quietly fails qualified candidates.

## Notes on document parsing

- **DOCX tables are walked, not skipped.** `Document.paragraphs` omits table
  cells, so a resume laid out in a table yields nothing but the name — every
  skill and role dropped with no error raised. The parser walks the body in
  document order and descends into tables (and nested tables) instead.
- **`cffi` is pinned** because `pypdf` reaches `cryptography`, and a partial
  install raises PyO3's `PanicException`, which derives from `BaseException` —
  so `except Exception` around the parser will not catch it.
- The bundled parsers handle single-column documents. Multi-column PDFs still
  need a layout-aware backend (Marker/Textract) to preserve reading order.

## Known limitations

- **SQLite is single-writer.** `STORE_BACKEND=sqlite` is durable and fine for a
  single API process plus workers, but it serializes writes. A high-write
  deployment wants Postgres behind the same store interfaces.
- **Reverse matching scans linearly.** `GET /resumes/{id}/jobs` scores every
  stored posting rather than querying a second index. A deployment holds far
  fewer open roles than resumes, so this stays cheap — but it is a scan, and it
  wants its own index if that assumption stops holding.
- **Deduplication is by exact content.** Re-uploading a byte-identical resume
  updates one candidate; an edited resume from the same person creates a second.
  Merging by email is a product decision about overwriting history, so it is
  left open deliberately.
- **Institution names are keyword-matched.** The parser recognises "Stanford
  University" but not bare acronyms like "MIT" or "Caltech", so those land with an
  empty institution. The degree and field still extract correctly.
- **Duplicate scanning is pairwise.** Fine at recruiting scale; a pool in the
  hundreds of thousands needs blocking on a cheap key before comparison.
- **CJK is not word-segmented.** Tokenization keeps CJK runs intact rather than
  dropping them, but does not split them into words; that needs a segmenter.

## Persistence

`STORE_BACKEND` selects where candidate and job records live. `memory` (default)
keeps everything in-process — fast, zero setup, gone on restart. `sqlite`
persists to `SQLITE_PATH`, so records outlive the process and are visible to
Celery workers as well as the API. Both backends are exercised by one
parametrized contract suite in `tests/test_stores.py`, so a behavioural
difference between them fails in CI rather than after someone flips the setting.

Durable records and an in-process vector index are an inconsistent pair. After a
restart the candidate is still listed by the API but matches nothing, which
reads as "no results" rather than as a broken index:

```
process 1: uploaded 2 candidates, 1 job
process 2: candidates: 2 | jobs: 1 | MATCH after restart: 0 results   <- before
process 2: candidates: 2 | jobs: 1 | MATCH after restart: 2 results   <- after
```

So a startup hook rebuilds the index when records exist but the index is empty.
It checks the **index**, not the configured backend, which keeps it correct for
every combination — Qdrant persists its own vectors, reports a non-zero count,
and is left alone. Rebuilding re-embeds each stored resume, so it costs one
embedding call per candidate; set `REINDEX_ON_STARTUP=false` to skip it.

Fully durable production setup is `STORE_BACKEND=sqlite` (records) plus
`VECTOR_STORE_BACKEND=qdrant` (vectors), where nothing needs rebuilding at all.

## Education and certification requirements

Both were live in the schema and dead in practice: `required_education` fed a 0.1
scoring weight that the extractor never populated, so a posting demanding a CS
degree scored identically for someone with none, and `JobProfile` had nowhere to
require a certification at all.

The extractor now reads both out of the requirements block:

```
degree level : bachelor
field        : computer science
certs        : ['AWS Certified Solutions Architect']
weights      : {experience: 0.45, skills: 0.36, education: 0.09, certifications: 0.1}
```

Two comparisons are deliberately loose:

- **Degree level is ranked, not matched.** A Master's satisfies a Bachelor's
  requirement — string equality would reject an over-qualified candidate. The
  right level in the wrong field, or the right field at the wrong level, scores
  partially rather than failing outright.
- **Certifications are compared on token containment.** A posting says "AWS
  Certified Solutions Architect" while the resume says "AWS Certified Solutions
  Architect - Associate"; exact equality would call that a miss.

The certification weight starts at **0.0** and the extractor raises it to 0.1
(rebalancing the others proportionally) only for postings that actually name one.
A component that is always zero never fires; one that always fires distorts every
posting that never mentioned certifications. Explicit `weights` always win over
this.

With skills and tenure held identical, the components now separate candidates
that were previously indistinguishable:

```
Ann Lee  total=0.6857  edu=1.00 cert=1.00   M.S. Computer Science + AWS cert
Ben Ray  total=0.4673  edu=0.60 cert=0.00   B.A. History, no cert
Cara Yu  total=0.4403  edu=0.00 cert=0.00   no degree, no cert
```

## Duplicate candidates

Content-hash deduplication catches the same file uploaded twice. It cannot catch
the far more common case — the same person applying again with an **updated**
resume, which produces a second record and puts them in every shortlist twice:

```
pool:      Jane Doe ['python','postgresql']            <- last year's resume
           Jane Doe ['python','postgresql','aws',...]  <- this year's
shortlist: ['Jane Doe', 'Jane Doe', 'John Smith']
```

`GET /resumes/duplicates` weighs several independent signals and says which fired:

```
high  b993dfd9 ~ 1a16fade  reasons: ['identical email', 'identical name']
```

Email is strong but often missing from a parsed resume, and a matching name alone
is not evidence — two people really are called John Smith. So the weaker signals
require corroboration (skill overlap, phone, or resume-text overlap) before a pair
is reported at all.

**Detection and merging are deliberately separate.** Collapsing two records
discards a version of someone's history, which is a judgement with consequences,
so nothing merges automatically. `POST /resumes/{id}/merge` performs it when a
human decides:

```
skills unioned : ['python','postgresql','aws','docker']
phone recovered from the older resume: '+1 415-555-0100'
pipeline history carried across: {'interview': 1}   <- re-keyed to the survivor
shortlist now  : ['Jane Doe', 'John Smith']
```

Lists are unioned and scalars keep the surviving record's value unless it is
empty, so an older resume fills gaps the newer one dropped. Pipeline entries move
with the person — losing the stage someone reached because the wrong record won a
merge would be the more damaging failure. When both records sat in the same
pipeline, the further-along stage survives.

## Time in stage

`average_days_in_stage` on the funnel reports how long candidates sat at each
stage before moving on:

```
average_days_in_stage: {'applied': 9.0}
```

A funnel says where people drop out; this says where they get **stuck**, which is
the actionable half — a stage nobody leaves for nine days is a scheduling problem,
not a candidate-quality one. Only completed spans are averaged: someone still
sitting in a stage has no duration yet, and counting the open span would drag
every average toward zero.

## The reranker blend

`RERANK_BLEND` (default 0.5) sets how much of the final score comes from stage-2
reranking, with the rest from the structured components. It used to be a hardcoded
`0.5` inside the scorer, which is the wrong shape for a value whose right setting
depends on **which reranker is running**.

A cross-encoder that judges (job, resume) jointly earns a large share. The lexical
fallback is token overlap, which rewards a document for being short. That is not
hypothetical — with both candidates fully covering the requirements and no
years/education/certification bar to separate them, the structured halves tie
exactly and the terser resume wins outright on density:

```
rerank_blend=0.5   Ann 0.6429   Ben 0.3893
rerank_blend=0.25  Ann 0.8214   Ben 0.4840
```

Lower it when running `RERANKER_BACKEND=lexical`; leave it at 0.5 or raise it once
a real cross-encoder is in place.

## Notes on the search backends

- **Tokenization is Unicode-aware.** `[a-z0-9]+` turned "José García" into
  `['jos', 'garc', 'a']` and dropped CJK entirely, quietly wrecking retrieval for
  every non-English resume. Tokens are matched with `\w+` and combining marks are
  folded, so "Jose" and "José" also match each other. ASCII is unaffected.
- **The dense and sparse halves share one tokenizer.** `embeddings.py` imports
  `tokenize` from the vector store rather than keeping its own copy — two
  definitions would drift and index text differently from how they query it.

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

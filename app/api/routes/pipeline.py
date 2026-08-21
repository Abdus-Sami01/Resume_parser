"""Hiring pipeline endpoints: who is in play for a role, and where they stand."""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.candidate_store import get_candidate_store
from app.db.job_store import get_job_store
from app.schemas.pipeline import PipelineEntry, Stage
from app.services import pipeline as pipeline_service
from app.services.pipeline import PipelineError
from app.services.privacy import pseudonym_for

router = APIRouter(tags=["pipeline"])


class AddToPipelineRequest(BaseModel):
    candidate_id: str
    stage: Stage = Stage.APPLIED
    note: str = ""
    actor: str = Field("", description="Who made the change, for the audit trail")


class MoveStageRequest(BaseModel):
    stage: Stage
    note: str = ""
    actor: str = ""


class ShortlistRequest(BaseModel):
    """Match and shortlist in one step, rather than one request per candidate."""

    top_n: int = Field(10, ge=1, le=200)
    min_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Skip anyone scoring below this"
    )
    stage: Stage = Stage.APPLIED
    note: str = ""
    actor: str = ""


class ShortlistResponse(BaseModel):
    added: int
    skipped: int
    entries: list[PipelineEntry]
    skipped_details: list[dict]


class BulkMoveRequest(BaseModel):
    candidate_ids: list[str] = Field(..., min_length=1)
    stage: Stage
    note: str = ""
    actor: str = ""


class BulkMoveResponse(BaseModel):
    moved: int
    failed: int
    entries: list[PipelineEntry]
    failed_details: list[dict]


class PipelineItem(BaseModel):
    entry: PipelineEntry
    candidate_name: str


class PipelineList(BaseModel):
    job_id: str
    total: int
    stage_counts: dict[str, int]
    items: list[PipelineItem]


def _require_job(job_id: str) -> None:
    if get_job_store().get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")


def _require_candidate(candidate_id: str) -> None:
    if get_candidate_store().get(candidate_id) is None:
        raise HTTPException(status_code=404, detail="candidate not found")


def _decorate(entry: PipelineEntry, blind: bool) -> PipelineItem:
    record = get_candidate_store().get(entry.candidate_id)
    if blind or record is None:
        name = pseudonym_for(entry.candidate_id)
    else:
        name = record.profile.name
    return PipelineItem(entry=entry, candidate_name=name)


@router.post("/jobs/{job_id}/pipeline", response_model=PipelineEntry, status_code=201)
async def add_to_pipeline(job_id: str, request: AddToPipelineRequest) -> PipelineEntry:
    _require_job(job_id)
    _require_candidate(request.candidate_id)

    try:
        return pipeline_service.add_candidate(
            job_id, request.candidate_id, request.stage, request.note, request.actor
        )
    except PipelineError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/pipeline/shortlist", response_model=ShortlistResponse, status_code=201)
async def shortlist(job_id: str, request: ShortlistRequest) -> ShortlistResponse:
    """Runs the match and puts the top results straight into the pipeline."""
    _require_job(job_id)

    try:
        result = pipeline_service.shortlist_from_match(
            job_id,
            top_n=request.top_n,
            min_score=request.min_score,
            stage=request.stage,
            note=request.note,
            actor=request.actor,
        )
    except PipelineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ShortlistResponse(**result)


@router.patch("/jobs/{job_id}/pipeline", response_model=BulkMoveResponse)
async def bulk_move(job_id: str, request: BulkMoveRequest) -> BulkMoveResponse:
    """Moves several candidates at once — rejecting a tail is one decision, not twenty."""
    _require_job(job_id)

    return BulkMoveResponse(
        **pipeline_service.move_many(
            job_id, request.candidate_ids, request.stage, request.note, request.actor
        )
    )


@router.get("/jobs/{job_id}/pipeline", response_model=PipelineList)
async def list_pipeline(
    job_id: str,
    stage: Stage | None = Query(None, description="Filter to a single stage"),
    blind: bool = Query(False, description="Replace names with stable pseudonyms"),
) -> PipelineList:
    _require_job(job_id)

    entries = pipeline_service.entries_for_job(job_id, stage=stage)
    all_entries = pipeline_service.entries_for_job(job_id)

    counts: dict[str, int] = {}
    for entry in all_entries:
        counts[entry.stage.value] = counts.get(entry.stage.value, 0) + 1

    return PipelineList(
        job_id=job_id,
        total=len(entries),
        stage_counts=counts,
        items=[_decorate(entry, blind) for entry in entries],
    )


@router.patch("/jobs/{job_id}/pipeline/{candidate_id}", response_model=PipelineEntry)
async def move_stage(job_id: str, candidate_id: str, request: MoveStageRequest) -> PipelineEntry:
    """Moves a candidate to a new stage, appending to the audit trail."""
    _require_job(job_id)

    try:
        return pipeline_service.move_candidate(
            job_id, candidate_id, request.stage, request.note, request.actor
        )
    except PipelineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/jobs/{job_id}/pipeline/{candidate_id}", status_code=204)
async def remove_from_pipeline(job_id: str, candidate_id: str) -> None:
    if not pipeline_service.remove_candidate(job_id, candidate_id):
        raise HTTPException(status_code=404, detail="candidate is not in this pipeline")


@router.get("/jobs/{job_id}/pipeline/funnel")
async def pipeline_funnel(job_id: str) -> dict:
    """Stage counts and conversion, computed from history rather than current stage."""
    _require_job(job_id)
    return pipeline_service.funnel_for_job(job_id)


@router.get("/resumes/{candidate_id}/applications", response_model=list[PipelineEntry])
async def candidate_applications(candidate_id: str) -> list[PipelineEntry]:
    """Every pipeline this candidate is in — the same person can be at different stages per role."""
    _require_candidate(candidate_id)
    return pipeline_service.entries_for_candidate(candidate_id)

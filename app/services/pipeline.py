"""Pipeline transitions and funnel reporting."""
from collections import Counter
from datetime import datetime, timezone

from app.db.pipeline_store import get_pipeline_store
from app.schemas.pipeline import EXIT_STAGES, FUNNEL_ORDER, PipelineEntry, Stage, StageEvent


class PipelineError(ValueError):
    """Raised for a transition the caller should be told about rather than shown a 500."""


def add_candidate(
    job_id: str, candidate_id: str, stage: Stage = Stage.APPLIED, note: str = "", actor: str = ""
) -> PipelineEntry:
    store = get_pipeline_store()

    existing = store.get(job_id, candidate_id)
    if existing is not None:
        raise PipelineError(
            f"candidate is already in this pipeline at stage '{existing.stage.value}'"
        )

    entry = PipelineEntry(
        job_id=job_id,
        candidate_id=candidate_id,
        stage=stage,
        history=[StageEvent(from_stage=None, to_stage=stage, note=note, actor=actor)],
    )
    return store.upsert(entry)


def move_candidate(
    job_id: str, candidate_id: str, stage: Stage, note: str = "", actor: str = ""
) -> PipelineEntry:
    """Moves a candidate and records the transition.

    Transitions are deliberately unrestricted rather than driven by a fixed state
    machine — real processes send people back a stage, revive a rejection, or skip
    a step, and a rigid graph only teaches users to work around it. The history is
    what makes that safe: every move is recorded with who made it and why.
    """
    store = get_pipeline_store()

    entry = store.get(job_id, candidate_id)
    if entry is None:
        raise PipelineError("candidate is not in this pipeline")

    if entry.stage == stage:
        # A no-op must not pad the audit trail with a move that never happened.
        return entry

    entry.history.append(
        StageEvent(from_stage=entry.stage, to_stage=stage, note=note, actor=actor)
    )
    entry.stage = stage
    entry.updated_at = datetime.now(timezone.utc)

    return store.upsert(entry)


def remove_candidate(job_id: str, candidate_id: str) -> bool:
    return get_pipeline_store().delete(job_id, candidate_id)


def entries_for_job(job_id: str, stage: Stage | None = None) -> list[PipelineEntry]:
    entries = get_pipeline_store().for_job(job_id)
    if stage is not None:
        entries = [entry for entry in entries if entry.stage == stage]
    return sorted(entries, key=lambda entry: entry.updated_at, reverse=True)


def entries_for_candidate(candidate_id: str) -> list[PipelineEntry]:
    return sorted(
        get_pipeline_store().for_candidate(candidate_id),
        key=lambda entry: entry.updated_at,
        reverse=True,
    )


def funnel_for_job(job_id: str) -> dict:
    """Stage counts plus conversion between consecutive steps.

    Counting only current stages would report a funnel that never converts: a
    candidate now at "offer" has passed through screening and interview, so the
    history is what makes each step's total meaningful.
    """
    entries = get_pipeline_store().for_job(job_id)

    current: Counter = Counter(entry.stage for entry in entries)
    reached: Counter = Counter()
    for entry in entries:
        for stage in {event.to_stage for event in entry.history} | {entry.stage}:
            reached[stage] += 1

    steps = []
    for index, stage in enumerate(FUNNEL_ORDER):
        previous_total = reached[FUNNEL_ORDER[index - 1]] if index else None
        steps.append(
            {
                "stage": stage.value,
                "currently_here": current.get(stage, 0),
                "ever_reached": reached.get(stage, 0),
                "conversion_from_previous": (
                    round(reached[stage] / previous_total, 4)
                    if previous_total
                    else None
                ),
            }
        )

    return {
        "job_id": job_id,
        "total_candidates": len(entries),
        "active": sum(1 for entry in entries if entry.is_active),
        "steps": steps,
        "exits": {stage.value: current.get(stage, 0) for stage in EXIT_STAGES},
    }


def shortlist_from_match(
    job_id: str,
    top_n: int = 10,
    min_score: float = 0.0,
    stage: Stage = Stage.APPLIED,
    note: str = "",
    actor: str = "",
) -> dict:
    """Runs the match and adds the result to the pipeline in one step.

    Shortlisting was the one thing the system made hard: matching produced a
    ranked list and the pipeline accepted one candidate per call, so acting on a
    top-ten meant eleven requests and a client-side loop that had to re-derive the
    ranking to know what to send.

    Candidates already in the pipeline are reported as skipped rather than raised,
    because re-shortlisting after new resumes arrive is the normal way to use this
    and must not fail on the overlap.
    """
    from app.db.job_store import get_job_store
    from app.services.search.matcher import match

    record = get_job_store().get(job_id)
    if record is None:
        raise PipelineError("job not found")

    added: list[PipelineEntry] = []
    skipped: list[dict] = []

    for result in match(record.profile, top_n=top_n):
        if result.final_score < min_score:
            # Ranked order means everything after this is below the floor too.
            break

        try:
            added.append(
                add_candidate(job_id, result.candidate_id, stage, note=note, actor=actor)
            )
        except PipelineError:
            skipped.append({"candidate_id": result.candidate_id, "reason": "already in pipeline"})

    return {"added": len(added), "skipped": len(skipped), "entries": added, "skipped_details": skipped}


def move_many(
    job_id: str, candidate_ids: list[str], stage: Stage, note: str = "", actor: str = ""
) -> dict:
    """Moves a batch, reporting per-candidate outcomes.

    Rejecting the tail of a shortlist is one decision, not twenty, and one id that
    is no longer in the pipeline must not discard the other nineteen moves.
    """
    moved: list[PipelineEntry] = []
    failed: list[dict] = []

    for candidate_id in candidate_ids:
        try:
            moved.append(move_candidate(job_id, candidate_id, stage, note=note, actor=actor))
        except PipelineError as exc:
            failed.append({"candidate_id": candidate_id, "reason": str(exc)})

    return {"moved": len(moved), "failed": len(failed), "entries": moved, "failed_details": failed}

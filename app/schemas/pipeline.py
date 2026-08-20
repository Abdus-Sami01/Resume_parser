"""Hiring pipeline schemas.

A stage belongs to a (candidate, job) pair, not to a candidate: the same person
can be in interviews for one role and rejected for another, and a single global
status could not express that.
"""
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Stage(str, Enum):
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


# Ordered for funnel reporting. Rejected and withdrawn are exits rather than
# steps, so they are counted separately instead of being ranked among the rest.
FUNNEL_ORDER: list[Stage] = [
    Stage.APPLIED,
    Stage.SCREENING,
    Stage.INTERVIEW,
    Stage.OFFER,
    Stage.HIRED,
]
EXIT_STAGES: set[Stage] = {Stage.REJECTED, Stage.WITHDRAWN}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StageEvent(BaseModel):
    """One transition. The history exists because "why was this person rejected in
    March" is a question every hiring team eventually has to answer."""

    from_stage: Stage | None = None
    to_stage: Stage
    at: datetime = Field(default_factory=_utcnow)
    note: str = ""
    actor: str = ""


class PipelineEntry(BaseModel):
    job_id: str
    candidate_id: str
    stage: Stage = Stage.APPLIED
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    history: list[StageEvent] = Field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.stage not in EXIT_STAGES

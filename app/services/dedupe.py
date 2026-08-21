"""Near-duplicate candidate detection and merging.

Exact-content deduplication catches the same file uploaded twice. It cannot catch
the far more common case: the same person applying again with an updated resume,
which produces a second record and puts them in every shortlist twice.

Detection and merging are kept separate on purpose. Deciding that two records are
one person is a judgement with real consequences — merging discards a version of
someone's history — so this surfaces candidates and lets a human act, rather than
silently collapsing records behind the recruiter's back.
"""
from dataclasses import dataclass, field

from app.db.candidate_store import CandidateRecord, get_candidate_store
from app.schemas.candidate import CandidateProfile
from app.services.search.vector_store import tokenize

# Enough overlap that two profiles are describing one career rather than two.
_SKILL_OVERLAP_THRESHOLD = 0.6
_CONTENT_OVERLAP_THRESHOLD = 0.7


@dataclass
class DuplicateCandidate:
    candidate_id: str
    other_candidate_id: str
    confidence: str
    reasons: list[str] = field(default_factory=list)


def _normalized_name(profile: CandidateProfile) -> str:
    return " ".join(tokenize(profile.name))


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def compare(first: CandidateRecord, second: CandidateRecord) -> DuplicateCandidate | None:
    """Weighs several independent signals rather than trusting any one of them.

    Email alone is strong but frequently absent from a parsed resume; a matching
    name alone is weak, since common names collide. Requiring corroboration for
    the weaker signals keeps namesakes from being flagged as one person.
    """
    reasons: list[str] = []
    confidence = ""

    left, right = first.profile, second.profile

    emails_match = bool(left.email and right.email and left.email.lower() == right.email.lower())
    if emails_match:
        reasons.append("identical email")
        confidence = "high"

    names_match = bool(_normalized_name(left) and _normalized_name(left) == _normalized_name(right))
    if names_match:
        reasons.append("identical name")

    skill_overlap = _jaccard(set(left.skills), set(right.skills))
    if skill_overlap >= _SKILL_OVERLAP_THRESHOLD:
        reasons.append(f"skills overlap {skill_overlap:.0%}")

    phones_match = bool(left.phone and right.phone and left.phone == right.phone)
    if phones_match:
        reasons.append("identical phone")

    content_overlap = _jaccard(set(tokenize(first.raw_text)), set(tokenize(second.raw_text)))
    if content_overlap >= _CONTENT_OVERLAP_THRESHOLD:
        reasons.append(f"resume text overlap {content_overlap:.0%}")

    if not confidence:
        # A name on its own is not evidence — two people really are called John Smith.
        corroborated = names_match and (
            skill_overlap >= _SKILL_OVERLAP_THRESHOLD
            or phones_match
            or content_overlap >= _CONTENT_OVERLAP_THRESHOLD
        )
        if corroborated or content_overlap >= _CONTENT_OVERLAP_THRESHOLD:
            confidence = "medium"

    if not confidence:
        return None

    return DuplicateCandidate(
        candidate_id=first.candidate_id,
        other_candidate_id=second.candidate_id,
        confidence=confidence,
        reasons=reasons,
    )


def find_duplicates_for(candidate_id: str) -> list[DuplicateCandidate]:
    store = get_candidate_store()
    record = store.get(candidate_id)
    if record is None:
        return []

    found = []
    for other in store.all():
        if other.candidate_id == candidate_id:
            continue
        match = compare(record, other)
        if match is not None:
            found.append(match)

    return sorted(found, key=lambda d: (d.confidence != "high", d.other_candidate_id))


def find_all_duplicates() -> list[DuplicateCandidate]:
    """Pairwise scan over the pool. Fine at recruiting scale; wants blocking beyond it."""
    records = get_candidate_store().all()

    found = []
    for index, record in enumerate(records):
        for other in records[index + 1 :]:
            match = compare(record, other)
            if match is not None:
                found.append(match)

    return sorted(found, key=lambda d: d.confidence != "high")


def merge_profiles(keep: CandidateProfile, absorb: CandidateProfile) -> CandidateProfile:
    """Combines two profiles without losing anything the other one knew.

    Lists are unioned. Scalars keep the surviving record's value unless it is
    empty, in which case the other record fills the gap — an older resume often
    carries a phone number the newer one dropped.
    """
    merged_skills = list(keep.skills)
    merged_skills += [skill for skill in absorb.skills if skill not in merged_skills]

    merged_certs = list(keep.certifications)
    merged_certs += [cert for cert in absorb.certifications if cert not in merged_certs]

    # The longer history is the more complete one; picking by count avoids
    # arbitrating entry by entry between two versions of the same career.
    experience = keep.experience if len(keep.experience) >= len(absorb.experience) else absorb.experience
    education = keep.education if len(keep.education) >= len(absorb.education) else absorb.education

    return keep.model_copy(
        update={
            "email": keep.email or absorb.email,
            "phone": keep.phone or absorb.phone,
            "location": keep.location or absorb.location,
            "summary": keep.summary or absorb.summary,
            "skills": merged_skills,
            "certifications": merged_certs,
            "experience": experience,
            "education": education,
        }
    )


def merge_candidates(keep_id: str, absorb_id: str) -> CandidateRecord:
    """Folds one candidate into another, moving pipeline history with them."""
    from app.db.pipeline_store import get_pipeline_store
    from app.services.search.matcher import delete_candidate, index_candidate

    store = get_candidate_store()
    keep = store.get(keep_id)
    absorb = store.get(absorb_id)

    if keep is None or absorb is None:
        raise ValueError("both candidates must exist")
    if keep_id == absorb_id:
        raise ValueError("cannot merge a candidate into itself")

    merged_profile = merge_profiles(keep.profile, absorb.profile)
    # The longer raw text keeps the most searchable material for re-indexing.
    raw_text = keep.raw_text if len(keep.raw_text) >= len(absorb.raw_text) else absorb.raw_text

    pipeline_store = get_pipeline_store()
    for entry in pipeline_store.for_candidate(absorb_id):
        existing = pipeline_store.get(entry.job_id, keep_id)
        if existing is None:
            # Re-key the entry onto the surviving candidate, history intact.
            pipeline_store.upsert(entry.model_copy(update={"candidate_id": keep_id}))
        else:
            # Both records were in the same pipeline; the further-along stage wins,
            # since discarding progress is the more damaging mistake.
            existing.history.extend(entry.history)
            pipeline_store.upsert(existing)

    delete_candidate(absorb_id)
    index_candidate(merged_profile, raw_text, candidate_id=keep_id)

    return store.get(keep_id)

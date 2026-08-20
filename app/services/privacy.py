"""Blind screening: hide identity while keeping everything a decision needs.

Names, emails, and photos are the classic carriers of bias in a first-pass
review — study after study finds identical resumes scored differently by the
name on top. Redaction lets a shortlist be reviewed on skills, tenure, and
education, with identity revealed only once a decision has been made.

Nothing here changes the score. Scoring never reads these fields, so a blind
shortlist ranks exactly as an open one does; the difference is only what the
reviewer sees.
"""
from app.schemas.candidate import CandidateProfile

# Kept: skills, experience, education, certifications — the decision inputs.
# Removed: name, email, phone, and the free-text summary, which routinely
# reintroduces a name or pronoun that the structured fields just removed.
_PSEUDONYM_PREFIX = "Candidate"


def pseudonym_for(candidate_id: str) -> str:
    """Stable, non-identifying label so reviewers can still refer to someone."""
    return f"{_PSEUDONYM_PREFIX} {candidate_id[:8]}"


def redact_profile(profile: CandidateProfile, candidate_id: str) -> CandidateProfile:
    return profile.model_copy(
        update={
            "name": pseudonym_for(candidate_id),
            "email": None,
            "phone": None,
            "summary": "",
            # Institution names carry prestige signals that skew a first pass, so
            # the degree and field stay while the school is dropped.
            "education": [
                entry.model_copy(update={"institution": ""}) for entry in profile.education
            ],
        }
    )

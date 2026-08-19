"""Job description -> JobProfile, separating required vs. preferred skills and weighting sections."""
import re
from typing import Protocol

from app.config import get_settings
from app.schemas.job import JobProfile, JobWeights
from app.services.taxonomy.skill_standardizer import get_skill_standardizer

_REQUIRED_HEADERS = re.compile(r"(required|must[- ]have|minimum qualifications)", re.IGNORECASE)
_PREFERRED_HEADERS = re.compile(r"(preferred|nice[- ]to[- ]have|bonus)", re.IGNORECASE)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE)
# The gap before the experience keyword must not cross a digit or a sentence
# boundary, or "founded 15 years ago. We want 3+ years of experience" matches 15.
_EXPERIENCE_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?(?:\s+(?:of|in|with))?[^\d.]{0,25}?"
    r"\b(?:experience|exp|background|hands[- ]on)\b",
    re.IGNORECASE,
)
_REMOTE_RE = re.compile(r"\bremote\b", re.IGNORECASE)


class JobExtractor(Protocol):
    def extract(self, title: str, raw_text: str, weights: JobWeights | None = None) -> JobProfile: ...


class HeuristicJobExtractor:
    """Splits a JD into required/preferred skill sections via header keywords."""

    def __init__(self) -> None:
        self._standardizer = get_skill_standardizer()

    def extract(self, title: str, raw_text: str, weights: JobWeights | None = None) -> JobProfile:
        required_section, preferred_section = self._split_sections(raw_text)

        required_skills = self._standardizer.extract_and_standardize(required_section or raw_text)
        preferred_skills = self._standardizer.extract_and_standardize(preferred_section)
        preferred_skills = [s for s in preferred_skills if s not in required_skills]

        min_years = self._extract_min_years(required_section, raw_text)

        settings = get_settings()
        return JobProfile(
            title=title,
            remote=bool(_REMOTE_RE.search(raw_text)),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_years_experience=min_years,
            description=raw_text,
            weights=weights
            or JobWeights(
                experience=settings.weight_experience,
                skills=settings.weight_skills,
                education=settings.weight_education,
            ),
        )

    @staticmethod
    def _extract_min_years(required_section: str, raw_text: str) -> float:
        """Reads the experience bar from the requirements only.

        Scanning the whole posting lets unrelated prose set the bar — "a company
        with 20 years of history" turns a 3-year requirement into a 20-year one,
        and a "preferred" nice-to-have outranks the actual minimum. Both silently
        penalise qualified candidates on the heaviest-weighted score component.
        """
        if required_section.strip():
            # Everything here is a stated requirement, so the highest bar governs.
            return max((float(y) for y in _YEARS_RE.findall(required_section)), default=0.0)

        # No requirements block to scope to, so only count years tied to experience wording.
        return max((float(y) for y in _EXPERIENCE_YEARS_RE.findall(raw_text)), default=0.0)

    @staticmethod
    def _split_sections(raw_text: str) -> tuple[str, str]:
        """Best-effort split into (required block, preferred block) by header keywords."""
        lines = raw_text.splitlines()
        required: list[str] = []
        preferred: list[str] = []
        current: list[str] | None = None

        for line in lines:
            if _REQUIRED_HEADERS.search(line):
                current = required
                continue
            if _PREFERRED_HEADERS.search(line):
                current = preferred
                continue
            if current is not None:
                current.append(line)

        return "\n".join(required), "\n".join(preferred)


def get_job_extractor() -> JobExtractor:
    return HeuristicJobExtractor()

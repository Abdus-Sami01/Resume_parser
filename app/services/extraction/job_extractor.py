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

_DEGREE_LEVEL_RE = re.compile(
    r"\b(ph\.?d|doctorate|master(?:'?s)?|m\.?sc?|m\.?s|bachelor(?:'?s)?|b\.?sc?|b\.?s|b\.?a|associate(?:'?s)?)\b",
    re.IGNORECASE,
)
_DEGREE_LEVEL_CANONICAL = {
    "phd": "phd", "ph.d": "phd", "doctorate": "phd",
    "master": "master", "masters": "master", "master's": "master", "msc": "master", "ms": "master", "m.s": "master", "m.sc": "master",
    "bachelor": "bachelor", "bachelors": "bachelor", "bachelor's": "bachelor", "bsc": "bachelor", "bs": "bachelor", "b.s": "bachelor", "b.sc": "bachelor", "ba": "bachelor", "b.a": "bachelor",
    "associate": "associate", "associates": "associate", "associate's": "associate",
}
# "Bachelor's degree in Computer Science" -> the field is what a resume can be matched on.
# MULTILINE so the field can end at a line break — a requirements block puts each
# item on its own line, and without it "$" only ever matches the end of the block.
_FIELD_OF_STUDY_RE = re.compile(
    r"(?:degree\s+)?in\s+([A-Za-z][A-Za-z\s&/]{2,45}?)(?=\s*[,;.()]|\s+or\b|\s+with\b|$)",
    re.IGNORECASE | re.MULTILINE,
)
_CERTIFICATION_RE = re.compile(
    r"^[\s\-•*]*([A-Za-z0-9][A-Za-z0-9 .+/#-]{3,60}?\b(?:certified|certification|certificate)\b[A-Za-z0-9 .+/#-]{0,40})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


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
        scope = required_section or raw_text
        degree_level, field_of_study = self._extract_education(scope)
        certifications = self._extract_certifications(scope)

        settings = get_settings()
        return JobProfile(
            title=title,
            remote=bool(_REMOTE_RE.search(raw_text)),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_years_experience=min_years,
            required_education=field_of_study or degree_level,
            required_degree_level=degree_level,
            required_certifications=certifications,
            description=raw_text,
            weights=weights or self._default_weights(settings, bool(certifications)),
        )

    @staticmethod
    def _default_weights(settings, has_certifications: bool) -> JobWeights:
        """Gives certifications real weight only when a posting actually requires one.

        A component that is always zero is a component that never fires, so the
        weight is taken from the others when — and only when — there is something
        for it to score. An explicit `weights` argument always wins over this.
        """
        experience = settings.weight_experience
        skills = settings.weight_skills
        education = settings.weight_education

        if not has_certifications:
            return JobWeights(experience=experience, skills=skills, education=education)

        share = 0.1
        scale = 1.0 - share
        return JobWeights(
            experience=round(experience * scale, 6),
            skills=round(skills * scale, 6),
            education=round(education * scale, 6),
            certifications=round(1.0 - round(experience * scale, 6) - round(skills * scale, 6) - round(education * scale, 6), 6),
        )

    @staticmethod
    def _extract_education(text: str) -> tuple[str, str]:
        """Returns (degree level, field of study), either of which may be empty."""
        level_match = _DEGREE_LEVEL_RE.search(text)
        degree_level = ""
        if level_match:
            key = level_match.group(1).lower().rstrip(".")
            degree_level = _DEGREE_LEVEL_CANONICAL.get(key, key)

        field = ""
        if level_match:
            # Only look for the field after the degree word, so "5+ years in Python"
            # elsewhere in the block cannot be mistaken for a field of study.
            field_match = _FIELD_OF_STUDY_RE.search(text[level_match.end() :])
            if field_match:
                field = field_match.group(1).strip().lower()

        return degree_level, field

    @staticmethod
    def _extract_certifications(text: str) -> list[str]:
        found: list[str] = []
        for match in _CERTIFICATION_RE.finditer(text):
            cleaned = " ".join(match.group(1).split())
            if cleaned and cleaned not in found:
                found.append(cleaned)
        return found

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

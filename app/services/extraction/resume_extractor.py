"""Structured extraction: raw resume text -> validated CandidateProfile.

Two backends, selected by `settings.extraction_backend`:

- "llm": Instructor + an OpenAI-compatible client forces the model to return
  JSON matching CandidateProfile exactly, no manual parsing/repair needed.
- "heuristic": regex/keyword based extraction. No API key or network
  required; degrades gracefully instead of throwing on unfamiliar formats.
  This is what makes the app runnable and testable fully offline.
"""
import re
from typing import Protocol

from app.config import get_settings
from app.schemas.candidate import CandidateProfile, Education, Experience
from app.services.taxonomy.skill_standardizer import get_skill_standardizer

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s().]{7,}\d)")
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE)

_DEGREE_RE = re.compile(
    r"\b(ph\.?d|doctorate|m\.?b\.?a|m\.?sc?|b\.?sc?|b\.?a|b\.?tech|m\.?tech|"
    r"master(?:'?s)?|bachelor(?:'?s)?|associate(?:'?s)?)\b",
    re.IGNORECASE,
)
_INSTITUTION_RE = re.compile(r"\b(university|college|institute|school|academy)\b", re.IGNORECASE)
_GRAD_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FIELD_RE = re.compile(r"\b(?:in|of)\s+([A-Za-z][A-Za-z\s&]{2,40}?)(?=\s*[,;|]|\s+at\b|$)", re.IGNORECASE)
_CERT_HEADER_RE = re.compile(r"^\s*certification[s]?\s*:?\s*$", re.IGNORECASE)
_CERT_INLINE_RE = re.compile(r"^\s*certification[s]?\s*:\s*(.+)$", re.IGNORECASE)
_CERT_PHRASE_RE = re.compile(r"\bcertified\b|\bcertificate\b", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r"^\s*[A-Z][A-Za-z ]{2,30}\s*:?\s*$")


class ResumeExtractor(Protocol):
    def extract(self, raw_text: str) -> CandidateProfile: ...


def standardize_profile_skills(profile: CandidateProfile) -> CandidateProfile:
    """Map a profile's raw skills onto the taxonomy so they share the job side's vocabulary.

    Skills the taxonomy does not recognise are kept lowercased rather than dropped:
    losing a niche skill entirely is worse than carrying it in a raw form that can
    still match a job posting by exact string.
    """
    standardizer = get_skill_standardizer()
    canonical: list[str] = []

    for raw_skill in profile.skills:
        resolved = standardizer.standardize(raw_skill) or raw_skill.strip().lower()
        if resolved and resolved not in canonical:
            canonical.append(resolved)

    return profile.model_copy(update={"skills": canonical})


class HeuristicResumeExtractor:
    """Lightweight, dependency-free structured extraction from resume text."""

    def __init__(self) -> None:
        self._standardizer = get_skill_standardizer()

    def extract(self, raw_text: str) -> CandidateProfile:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        name = lines[0] if lines else "Unknown"

        email_match = _EMAIL_RE.search(raw_text)
        phone_match = _PHONE_RE.search(raw_text)

        skills = self._standardizer.extract_and_standardize(raw_text)
        years = self._estimate_total_years(raw_text)
        experience = (
            [Experience(company="Unknown", role="Unknown", years=years, achievements=[])]
            if years
            else []
        )

        return CandidateProfile(
            name=name,
            email=email_match.group(0) if email_match else None,
            phone=phone_match.group(0).strip() if phone_match else None,
            skills=skills,
            experience=experience,
            education=self._extract_education(lines),
            certifications=self._extract_certifications(lines),
            summary=" ".join(lines[1:4]),
        )

    @staticmethod
    def _estimate_total_years(raw_text: str) -> float:
        matches = _YEARS_RE.findall(raw_text)
        return max((float(m) for m in matches), default=0.0)

    @staticmethod
    def _extract_education(lines: list[str]) -> list[Education]:
        """Any line naming a degree becomes one Education entry."""
        entries: list[Education] = []

        for line in lines:
            degree_match = _DEGREE_RE.search(line)
            if not degree_match:
                continue

            segments = [segment.strip() for segment in re.split(r"[,;|]", line) if segment.strip()]
            institution = next((s for s in segments if _INSTITUTION_RE.search(s)), "")
            degree_segment = next((s for s in segments if _DEGREE_RE.search(s)), degree_match.group(0))

            field_match = _FIELD_RE.search(degree_segment)
            year_match = _GRAD_YEAR_RE.search(line)

            entries.append(
                Education(
                    institution=institution,
                    degree=degree_segment,
                    field_of_study=field_match.group(1).strip() if field_match else "",
                    graduation_year=int(year_match.group(0)) if year_match else None,
                )
            )

        return entries

    @staticmethod
    def _extract_certifications(lines: list[str]) -> list[str]:
        """Collects both a `Certifications:` section and standalone 'certified' mentions."""
        found: list[str] = []
        in_section = False

        for line in lines:
            inline_match = _CERT_INLINE_RE.match(line)
            if inline_match:
                found.extend(part.strip() for part in inline_match.group(1).split(",") if part.strip())
                continue

            if _CERT_HEADER_RE.match(line):
                in_section = True
                continue

            if in_section:
                # A new section header ends the certifications block.
                if _SECTION_HEADER_RE.match(line) and not _CERT_PHRASE_RE.search(line):
                    in_section = False
                else:
                    found.append(line.lstrip("-•* ").strip())
                    continue

            if _CERT_PHRASE_RE.search(line) and len(line) < 120:
                found.append(line.lstrip("-•* ").strip())

        deduped: list[str] = []
        for cert in found:
            if cert and cert not in deduped:
                deduped.append(cert)
        return deduped


class LLMResumeExtractor:
    """Instructor-backed structured extraction against a real LLM."""

    def __init__(self) -> None:
        settings = get_settings()
        import instructor
        from openai import OpenAI

        client_kwargs: dict = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = instructor.from_openai(OpenAI(**client_kwargs))
        self._model = settings.llm_model

    def extract(self, raw_text: str) -> CandidateProfile:
        profile = self._client.chat.completions.create(
            model=self._model,
            response_model=CandidateProfile,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract a structured candidate profile from this resume:\n\n{raw_text}",
                }
            ],
        )
        return standardize_profile_skills(profile)


def get_resume_extractor() -> ResumeExtractor:
    settings = get_settings()
    if settings.extraction_backend == "llm":
        return LLMResumeExtractor()
    return HeuristicResumeExtractor()

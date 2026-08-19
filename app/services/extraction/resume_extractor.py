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
from app.schemas.candidate import CandidateProfile, Experience
from app.services.taxonomy.skill_standardizer import get_skill_standardizer

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s().]{7,}\d)")
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE)


class ResumeExtractor(Protocol):
    def extract(self, raw_text: str) -> CandidateProfile: ...


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
            summary=" ".join(lines[1:4]),
        )

    @staticmethod
    def _estimate_total_years(raw_text: str) -> float:
        matches = _YEARS_RE.findall(raw_text)
        return max((float(m) for m in matches), default=0.0)


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
        return self._client.chat.completions.create(
            model=self._model,
            response_model=CandidateProfile,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract a structured candidate profile from this resume:\n\n{raw_text}",
                }
            ],
        )


def get_resume_extractor() -> ResumeExtractor:
    settings = get_settings()
    if settings.extraction_backend == "llm":
        return LLMResumeExtractor()
    return HeuristicResumeExtractor()

"""Maps raw skill mentions ("React.js", "ReactJS", "React") onto one canonical taxonomy entry.

Production systems typically link against ESCO/O*NET; the bundled `skills.json`
plays that role here — a small canonical-name -> alias-list taxonomy. Swap the
JSON file (or point `TAXONOMY_PATH` at an ESCO export) without touching the
matching code.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

_TAXONOMY_PATH = Path(__file__).parent / "skills.json"
_FUZZY_MATCH_THRESHOLD = 90


class SkillStandardizer:
    def __init__(self, taxonomy: dict[str, list[str]]) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        for canonical, aliases in taxonomy.items():
            for alias in [canonical, *aliases]:
                self._alias_to_canonical[alias.lower()] = canonical

    def standardize(self, raw_skill: str) -> str | None:
        """Resolve one raw skill string to its canonical taxonomy name, or None."""
        key = raw_skill.strip().lower()
        if not key:
            return None
        if key in self._alias_to_canonical:
            return self._alias_to_canonical[key]
        return self._fuzzy_lookup(key)

    def extract_and_standardize(self, text: str) -> list[str]:
        """Scan free text for any taxonomy alias and return the deduped canonical skills found."""
        found: list[str] = []
        lowered = text.lower()
        for alias, canonical in self._alias_to_canonical.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
                if canonical not in found:
                    found.append(canonical)
        return found

    def _fuzzy_lookup(self, key: str) -> str | None:
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return None

        match = process.extractOne(
            key, self._alias_to_canonical.keys(), scorer=fuzz.ratio, score_cutoff=_FUZZY_MATCH_THRESHOLD
        )
        return self._alias_to_canonical[match[0]] if match else None


@lru_cache
def get_skill_standardizer() -> SkillStandardizer:
    taxonomy = json.loads(_TAXONOMY_PATH.read_text())
    return SkillStandardizer(taxonomy)

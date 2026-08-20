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

from app.config import get_settings

_TAXONOMY_PATH = Path(__file__).parent / "skills.json"
_FUZZY_MATCH_THRESHOLD = 90


class SkillStandardizer:
    def __init__(self, taxonomy: dict[str, list[str]], custom_path: Path | None = None) -> None:
        self._taxonomy: dict[str, list[str]] = {k: list(v) for k, v in taxonomy.items()}
        self._custom_path = custom_path
        self._alias_to_canonical: dict[str, str] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._alias_to_canonical = {}
        for canonical, aliases in self._taxonomy.items():
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


    # --- Runtime extension -------------------------------------------------

    def add_skill(self, canonical: str, aliases: list[str] | None = None) -> dict:
        """Registers a skill, or merges new aliases into one that already exists.

        Every deployment meets skills the bundled taxonomy has never heard of —
        internal tooling, new frameworks, regional certifications. Without this
        they stay unnormalized forever, and a resume saying "K8s" never matches a
        posting asking for "Kubernetes".
        """
        canonical = canonical.strip().lower()
        if not canonical:
            raise ValueError("canonical skill name cannot be empty")

        merged = self._taxonomy.setdefault(canonical, [])
        for alias in aliases or []:
            cleaned = alias.strip().lower()
            if cleaned and cleaned != canonical and cleaned not in merged:
                merged.append(cleaned)

        self._rebuild_index()
        self._persist_custom()
        return {"skill": canonical, "aliases": merged}

    def remove_skill(self, canonical: str) -> bool:
        removed = self._taxonomy.pop(canonical.strip().lower(), None) is not None
        if removed:
            self._rebuild_index()
            self._persist_custom()
        return removed

    def known_skills(self) -> dict[str, list[str]]:
        return {canonical: list(aliases) for canonical, aliases in sorted(self._taxonomy.items())}

    def _persist_custom(self) -> None:
        """Writes additions to an overlay file, never to the bundled taxonomy.

        Keeping them separate means the shipped `skills.json` can be updated by a
        future release without clobbering whatever a deployment added locally.
        """
        if self._custom_path is None:
            return

        bundled = json.loads(_TAXONOMY_PATH.read_text())
        overlay = {
            canonical: aliases
            for canonical, aliases in self._taxonomy.items()
            if canonical not in bundled or sorted(aliases) != sorted(bundled[canonical])
        }

        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        self._custom_path.write_text(json.dumps(overlay, indent=2, ensure_ascii=False))


def _load_taxonomy(custom_path: Path | None) -> dict[str, list[str]]:
    taxonomy: dict[str, list[str]] = json.loads(_TAXONOMY_PATH.read_text())

    if custom_path and custom_path.exists():
        for canonical, aliases in json.loads(custom_path.read_text()).items():
            existing = taxonomy.setdefault(canonical, [])
            existing.extend(alias for alias in aliases if alias not in existing)

    return taxonomy


@lru_cache
def get_skill_standardizer() -> SkillStandardizer:
    custom_path = Path(get_settings().custom_skills_path)
    return SkillStandardizer(_load_taxonomy(custom_path), custom_path=custom_path)

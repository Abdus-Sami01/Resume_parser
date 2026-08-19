"""Prints the loaded skill taxonomy, as a sanity check after editing skills.json."""
from app.services.taxonomy.skill_standardizer import get_skill_standardizer

if __name__ == "__main__":
    standardizer = get_skill_standardizer()
    print(f"{len(standardizer._alias_to_canonical)} aliases loaded")
    for alias, canonical in sorted(standardizer._alias_to_canonical.items()):
        print(f"  {alias!r} -> {canonical}")

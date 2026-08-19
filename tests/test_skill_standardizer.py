from app.services.taxonomy.skill_standardizer import get_skill_standardizer


def test_aliases_resolve_to_the_same_canonical_skill():
    standardizer = get_skill_standardizer()
    assert standardizer.standardize("React.js") == "react"
    assert standardizer.standardize("ReactJS") == "react"
    assert standardizer.standardize("react") == "react"


def test_unknown_skill_returns_none():
    standardizer = get_skill_standardizer()
    assert standardizer.standardize("underwater basket weaving") is None


def test_extract_and_standardize_finds_multiple_skills_in_free_text():
    standardizer = get_skill_standardizer()
    text = "Built REST APIs in Python and deployed them with Docker on AWS."
    skills = standardizer.extract_and_standardize(text)
    assert set(skills) == {"rest api", "python", "docker", "aws"}


def test_extract_and_standardize_does_not_match_substrings_of_other_words():
    standardizer = get_skill_standardizer()
    # "go" should not match inside "good" or "golden"
    skills = standardizer.extract_and_standardize("We had a good, golden opportunity.")
    assert "golang" not in skills

"""Narrative style templates and class → theme mappings."""
from __future__ import annotations

STYLE_TEMPLATES: dict[str, str] = {
    "poetry": "a short lyric poem",
}

# Maps (dataset, style) -> {class_name: narrative_theme}
# Themes evoke class content without naming it — edit freely to improve generation quality.
LABEL_MAPS: dict[tuple[str, str], dict[str, str]] = {
    ("pubmed", "poetry"): {
        "Diabetes_Mellitus_Experimental": "fire and trial",
        "Diabetes_Mellitus_Type_1": "the unbroken storm",
        "Diabetes_Mellitus_Type_2": "slow tide",
    },
    ("cora", "poetry"): {
        "Case_Based": "keeper of stories",
        "Genetic_Algorithms": "seeds and seasons",
        "Neural_Networks": "the weaving mind",
        "Probabilistic_Methods": "fog and chance",
        "Reinforcement_Learning": "the returning traveler",
        "Rule_Learning": "law and stone",
        "Theory": "the open horizon",
    },
}


def map_proportions_to_themes(
    proportions: dict[str, float], dataset: str, style: str
) -> list[tuple[str, float]]:
    """Map class proportions to (theme, pct) pairs, sorted descending by proportion."""
    key = (dataset, style)
    if key not in LABEL_MAPS:
        raise ValueError(f"No label map for {key!r}. Add it to LABEL_MAPS in narratives.py")
    label_map = LABEL_MAPS[key]
    themed = [(label_map[cls], pct) for cls, pct in proportions.items() if cls in label_map]
    return sorted(themed, key=lambda x: x[1], reverse=True)

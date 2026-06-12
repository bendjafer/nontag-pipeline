"""Narrative style templates and class → topic mappings.

Two INDEPENDENT axes, kept separate so style comparisons are controlled by construction:
  - TOPIC_MAPS:     dataset -> {class_name: topic}. The semantic assignment. FIXED across styles.
  - STYLE_TEMPLATES: style  -> rendering instruction. ONLY this changes between conditions.

A node's topics come from TOPIC_MAPS[dataset] regardless of style, so any accuracy difference
across styles is attributable to STYLE alone — not to a different topic assignment. Topics are
arbitrary-but-fixed: only consistency (same class -> same topic) and mutual separability matter,
because the predictor learns the association from train/val text. Output must EVOKE the topic,
never name it (see build_generation_prompt), so distinct topics avoid trivial keyword matching.
"""
from __future__ import annotations

# style -> how to render the topics. Add a key here to add a new style condition.
STYLE_TEMPLATES: dict[str, str] = {
    "poetry": "a short lyric poem",
    "news": "a short news report",
    "story": "a brief short story",
}

# dataset -> {class_name: topic}. Topics must be mutually distinct; same map used for every style.
TOPIC_MAPS: dict[str, dict[str, str]] = {
    "pubmed": {
        "Diabetes_Mellitus_Experimental": "war",
        "Diabetes_Mellitus_Type_1": "love",
        "Diabetes_Mellitus_Type_2": "nature",
    },
    "cora": {
        "Case_Based": "war",
        "Genetic_Algorithms": "love",
        "Neural_Networks": "nature",
        "Probabilistic_Methods": "the city",
        "Reinforcement_Learning": "music",
        "Rule_Learning": "memory",
        "Theory": "the cosmos",
    },
}


def map_proportions_to_themes(
    proportions: dict[str, float], dataset: str
) -> list[tuple[str, float]]:
    """Map class proportions to (topic, pct) pairs, sorted descending by proportion."""
    if dataset not in TOPIC_MAPS:
        raise ValueError(f"No topic map for {dataset!r}. Add it to TOPIC_MAPS in narratives.py")
    topic_map = TOPIC_MAPS[dataset]
    themed = [(topic_map[cls], pct) for cls, pct in proportions.items() if cls in topic_map]
    return sorted(themed, key=lambda x: x[1], reverse=True)

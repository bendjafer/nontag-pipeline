"""Narrative style templates and class → topic mappings.

Two INDEPENDENT axes, kept separate so style comparisons are controlled by construction:
  - TOPIC_MAPS:     dataset -> {class_name: topic}. The semantic assignment. FIXED across styles.
  - STYLE_TEMPLATES: style  -> rendering instruction. ONLY this changes between conditions.

A node's topics come from TOPIC_MAPS[dataset] regardless of style, so any accuracy difference
across styles is attributable to STYLE alone — not to a different topic assignment. Topics are
broad literary themes — not tied to the literal class meaning — so the predictor must learn
from style and structure, not from semantic leakage. Output must EVOKE the topic, never name
it (see build_generation_prompt).
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
        "Diabetes_Mellitus_Experimental": "nature",
        "Diabetes_Mellitus_Type_1": "war",
        "Diabetes_Mellitus_Type_2": "love",
    },
    "cora": {
        # Case_Based: reasoning by retrieving and adapting stored past examples
        "Case_Based": "the archive",
        # Genetic_Algorithms: natural selection, mutation, survival of the fittest solutions
        "Genetic_Algorithms": "evolution",
        # Neural_Networks: brain-inspired layered computation, learning from patterns
        "Neural_Networks": "the mind",
        # Probabilistic_Methods: uncertainty, degrees of belief, navigating the unknown
        "Probabilistic_Methods": "the fog",
        # Reinforcement_Learning: agents acting in environments, reward and trial
        "Reinforcement_Learning": "the arena",
        # Rule_Learning: explicit symbolic rules, deterministic structured inference
        "Rule_Learning": "the ledger",
        # Theory: abstract proofs, mathematical foundations, universal principles
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
    unknown = [cls for cls in proportions if cls not in topic_map]
    if unknown:
        raise ValueError(
            f"Classes {unknown} have no topic mapping for dataset {dataset!r}. "
            "Add them to TOPIC_MAPS in narratives.py"
        )
    themed = [(topic_map[cls], pct) for cls, pct in proportions.items()]
    # Secondary sort by topic name ensures stable ordering on tied proportions.
    return sorted(themed, key=lambda x: (-x[1], x[0]))

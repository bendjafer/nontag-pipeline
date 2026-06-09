"""Prompt construction and per-node text generation."""
from __future__ import annotations
import networkx as nx

from nontag_pipeline import llm
from nontag_pipeline.narratives import map_proportions_to_themes
from nontag_pipeline.select import neighbor_label_proportions, ppr_selection
from nontag_pipeline import config

_SYSTEM = (
    "You generate a single synthetic text in a specified literary style. Never mention graphs, "
    "nodes, neighbors, labels, categories, or percentages. Output only the text."
)


def build_generation_prompt(
    themes_with_pct: list[tuple[str, float]], style: str, target_len: int
) -> tuple[str, str]:
    """Return (system, user) prompt pair for the given themes and style."""
    theme_lines = "\n".join(
        f"- {theme}: {round(pct * 100)}%" for theme, pct in themes_with_pct
    )
    user = (
        "This item most likely shares the themes of the important related items around it.\n"
        f"Themes and their proportions:\n{theme_lines}\n"
        f"Write one {style} of about {target_len} words whose content embodies these themes "
        "in those proportions: the dominant theme sets the overall mood, the minor themes are "
        "woven in.\n"
        "Do NOT name the themes or state any numbers — evoke them only through imagery and content."
    )
    return _SYSTEM, user


def generate_node_text(
    G: nx.Graph,
    v: int,
    visible_labels: dict[int, int],
    class_names: list[str],
    dataset: str,
    style: str,
    target_len: int,
    k: int = config.PPR_K,
    m: int = config.PPR_M,
) -> dict:
    """Select neighbors, compute proportions, generate styled text for node v."""
    selected = ppr_selection(G, v, k=k, m=m)
    proportions = neighbor_label_proportions(selected, visible_labels, class_names)

    if not proportions:
        return {"node": v, "status": "no_signal", "text": None}

    themes = map_proportions_to_themes(proportions, dataset, style)
    system, user = build_generation_prompt(themes, style, target_len)
    text = llm.complete(user, system=system)
    return {"node": v, "status": "ok", "text": text, "themes": themes}

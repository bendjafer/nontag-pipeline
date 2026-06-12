"""Prompt construction and per-node text generation."""
from __future__ import annotations
import networkx as nx

from nontag_pipeline import llm
from nontag_pipeline.narratives import STYLE_TEMPLATES, map_proportions_to_themes
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
    if style not in STYLE_TEMPLATES:
        raise ValueError(f"No style template for {style!r}. Add it to STYLE_TEMPLATES in narratives.py")
    style_desc = STYLE_TEMPLATES[style]
    theme_lines = "\n".join(
        f"- {theme}: {round(pct * 100)}%" for theme, pct in themes_with_pct
    )
    user = (
        "This item most likely shares the themes of the important related items around it.\n"
        f"Themes and their proportions:\n{theme_lines}\n"
        f"Write {style_desc} of about {target_len} words whose content embodies these themes "
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
    alpha: float = config.PPR_ALPHA,
) -> dict:
    """Select neighbors, compute proportions, generate styled text for node v."""
    selected = ppr_selection(G, v, k=k, m=m, alpha=alpha)
    # LEAKAGE RULE: v's own label must never influence its text. visible_labels
    # excludes test labels by construction; this guards the remaining path.
    assert v not in selected, f"Leakage: node {v} selected as its own neighbor"
    proportions = neighbor_label_proportions(selected, visible_labels, class_names)

    # Evidence strength: proportions are renormalized over visible neighbors only,
    # so record how many of the selected neighbors actually contributed.
    n_selected = len(selected)
    n_visible = sum(1 for u in selected if u in visible_labels)

    if not proportions:
        return {
            "node": v, "status": "no_signal", "text": None,
            "n_selected": n_selected, "n_visible": n_visible,
        }

    themes = map_proportions_to_themes(proportions, dataset)
    system, user = build_generation_prompt(themes, style, target_len)
    text = llm.complete(user, system=system)
    return {
        "node": v, "status": "ok", "text": text, "themes": themes,
        "n_selected": n_selected, "n_visible": n_visible,
    }

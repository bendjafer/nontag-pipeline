"""Neighbor Selection: Personalized PageRank (PPR) based selection."""
from __future__ import annotations
import networkx as nx


def ppr_selection(
    G: nx.Graph, v: int, k: int = 2, m: int = 10,
    alpha: float = 0.85, restrict_khop: bool = True
) -> list[int]:
    """Top-m nodes most relevant to v by Personalized PageRank."""
    ppr = nx.pagerank(G, alpha=alpha, personalization={v: 1.0})

    if restrict_khop:
        candidates = set(nx.single_source_shortest_path_length(G, v, cutoff=k))
    else:
        candidates = set(G.nodes())
    candidates.discard(v)

    ranked = sorted(candidates, key=lambda u: ppr.get(u, 0.0), reverse=True)
    return ranked[:m]


def neighbor_label_proportions(
    selected: list[int],
    visible_labels: dict[int, int],
    class_names: list[str],
) -> dict[str, float]:
    """Proportion of each class among selected neighbors with visible labels."""
    counts: dict[str, int] = {}
    for u in selected:
        if u in visible_labels:
            name = class_names[visible_labels[u]]
            counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}

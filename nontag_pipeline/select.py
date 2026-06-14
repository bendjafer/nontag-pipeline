"""Neighbor Selection: Personalized PageRank (PPR) based selection."""
from __future__ import annotations
import networkx as nx


def ppr_selection(
    G: nx.Graph, v: int, k: int = 2, m: int = 10,
    alpha: float = 0.85, restrict_khop: bool = True
) -> list[int]:
    """Top-m nodes most relevant to v by Personalized PageRank."""
    if restrict_khop:
        # PPR on the k-hop subgraph: candidates are restricted to it anyway,
        # and this avoids a full-graph pagerank per node (hours on PubMed).
        khop = set(nx.single_source_shortest_path_length(G, v, cutoff=k))
        H = G.subgraph(khop)
    else:
        H = G
    # nx.pagerank converges when total error < N*tol; tighten tol so rankings
    # among close-scoring neighbors are not noise on larger subgraphs. Small N
    # shrinks the budget, needing >100 iterations — hence the raised max_iter.
    try:
        ppr = nx.pagerank(H, alpha=alpha, personalization={v: 1.0}, tol=1e-8, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        # Fall back to uniform scores so the node still gets neighbors selected;
        # quality is slightly lower but the run continues uninterrupted.
        ppr = {u: 1.0 / len(H) for u in H.nodes()}

    candidates = set(H.nodes())
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

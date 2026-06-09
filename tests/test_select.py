import networkx as nx
import pytest
from nontag_pipeline.select import ppr_selection, neighbor_label_proportions


def _triangle() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return G


def test_ppr_excludes_self():
    G = _triangle()
    assert 0 not in ppr_selection(G, 0, k=2, m=10)


def test_ppr_returns_at_most_m():
    G = _triangle()
    assert len(ppr_selection(G, 0, k=2, m=2)) <= 2


def test_ppr_restrict_khop_limits_candidates():
    # star graph: centre=0, leaves=1..10
    G = nx.star_graph(10)
    # from leaf 1 with k=1: only direct neighbors of node 1 are candidates: just node 0
    result = ppr_selection(G, 1, k=1, m=5, restrict_khop=True)
    # all results must be within 1 hop of node 1
    for u in result:
        assert nx.shortest_path_length(G, 1, u) <= 1
    assert result[0] == 0  # star center should have highest PPR from any leaf


def test_proportions_all_visible():
    visible = {1: 0, 2: 1}
    classes = ["ClassA", "ClassB"]
    props = neighbor_label_proportions([1, 2], visible, classes)
    assert abs(props["ClassA"] - 0.5) < 1e-9
    assert abs(props["ClassB"] - 0.5) < 1e-9


def test_proportions_skips_invisible_neighbors():
    visible = {1: 0}          # node 2 is test — not in visible
    classes = ["ClassA", "ClassB"]
    props = neighbor_label_proportions([1, 2], visible, classes)
    assert props == {"ClassA": 1.0}


def test_proportions_no_visible_returns_empty():
    props = neighbor_label_proportions([1, 2], {}, ["ClassA", "ClassB"])
    assert props == {}


def test_proportions_sum_to_one():
    visible = {1: 0, 2: 1, 3: 2}
    classes = ["A", "B", "C"]
    props = neighbor_label_proportions([1, 2, 3], visible, classes)
    assert abs(sum(props.values()) - 1.0) < 1e-9


def test_ppr_no_restrict_uses_full_graph():
    # path graph: 0-1-2-3-4
    # node 0 with k=1 restrict would only see node 1
    # with restrict_khop=False, nodes beyond 1-hop are candidates
    G = nx.path_graph(5)
    result = ppr_selection(G, 0, k=1, m=10, restrict_khop=False)
    # nodes 2, 3, 4 should be in candidates (beyond k=1 hop)
    assert any(u in result for u in [2, 3, 4])


def test_ppr_restrict_khop_star_center_first():
    # star graph: centre=0, leaves=1..10
    # from leaf 1 with k=1, only node 0 is a candidate
    G = nx.star_graph(10)
    result = ppr_selection(G, 1, k=1, m=5, restrict_khop=True)
    assert result[0] == 0  # star center should have highest PPR from any leaf

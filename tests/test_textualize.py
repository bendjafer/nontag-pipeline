import networkx as nx
import torch
import numpy as np
from unittest.mock import patch
from nontag_pipeline.textualize import build_generation_prompt, generate_node_text


def test_prompt_system_forbids_graph_terms():
    system, _ = build_generation_prompt([("the unbroken storm", 0.7)], "poetry", 80)
    forbidden = ["graph", "node", "neighbor", "label", "categor", "percentage"]
    lowered = system.lower()
    for word in forbidden:
        assert word in lowered, f"System prompt must mention forbidding '{word}'"


def test_prompt_user_contains_all_themes():
    themes = [("the unbroken storm", 0.7), ("fire and trial", 0.3)]
    _, user = build_generation_prompt(themes, "poetry", 80)
    assert "the unbroken storm" in user
    assert "fire and trial" in user


def test_prompt_user_contains_target_len():
    _, user = build_generation_prompt([("slow tide", 1.0)], "poetry", 120)
    assert "120" in user


def test_prompt_user_contains_style():
    _, user = build_generation_prompt([("fog and chance", 1.0)], "poetry", 80)
    assert "poetry" in user


def test_prompt_percentages_shown_as_integers():
    _, user = build_generation_prompt([("slow tide", 0.666)], "poetry", 80)
    assert "67%" in user


def _make_small_graph():
    """3-node graph: node 0 (test), nodes 1+2 (train)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    y = np.array([1, 0, 1])                  # labels
    train_mask = torch.tensor([False, True, True])
    val_mask   = torch.tensor([False, False, False])
    test_mask  = torch.tensor([True,  False, False])
    class_names = ["Diabetes_Mellitus_Experimental", "Diabetes_Mellitus_Type_1"]
    return G, y, train_mask, val_mask, test_mask, class_names


def test_generate_returns_ok_when_neighbors_visible():
    G, y, tr, va, te, cls = _make_small_graph()
    visible = {n: int(y[n]) for n in range(len(y)) if tr[n] or va[n]}

    with patch("nontag_pipeline.llm.complete", return_value="a poem"):
        result = generate_node_text(
            G, 0, visible, cls, "pubmed", "poetry", 80
        )
    assert result["status"] == "ok"
    assert result["text"] == "a poem"
    assert result["node"] == 0


def test_generate_returns_no_signal_when_no_visible_neighbors():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    y = np.array([1, 0, 1])
    class_names = ["ClassA", "ClassB"]
    visible = {}   # no train/val labels at all

    result = generate_node_text(G, 0, visible, class_names, "pubmed", "poetry", 80)
    assert result["status"] == "no_signal"
    assert result["text"] is None

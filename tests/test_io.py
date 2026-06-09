# tests/test_io.py
import json
import numpy as np
import networkx as nx
import torch
import pytest
from pathlib import Path
from nontag_pipeline.io import save_pseudo_tag


def _minimal_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    y = np.array([0, 1, 2])
    train_mask = torch.tensor([True,  False, False])
    val_mask   = torch.tensor([False, True,  False])
    test_mask  = torch.tensor([False, False, True])
    class_names = ["ClassA", "ClassB", "ClassC"]
    records = [
        {"node": 0, "status": "ok",        "text": "poem for 0"},
        {"node": 1, "status": "ok",        "text": "poem for 1"},
        {"node": 2, "status": "no_signal", "text": None},
    ]
    return G, y, train_mask, val_mask, test_mask, class_names, records


def test_save_creates_pt_file(tmp_path):
    args = _minimal_graph()
    pt_path, _ = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    assert pt_path.exists()


def test_save_creates_json_file(tmp_path):
    args = _minimal_graph()
    _, json_path = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    assert json_path.exists()


def test_json_contains_all_nodes(tmp_path):
    args = _minimal_graph()
    _, json_path = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    data = json.loads(json_path.read_text())
    assert len(data["nodes"]) == 3


def test_json_splits_are_correct(tmp_path):
    args = _minimal_graph()
    _, json_path = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    data = json.loads(json_path.read_text())
    splits = {n["node_id"]: n["split"] for n in data["nodes"]}
    assert splits[0] == "train"
    assert splits[1] == "val"
    assert splits[2] == "test"


def test_pt_has_raw_texts(tmp_path):
    args = _minimal_graph()
    pt_path, _ = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    saved = torch.load(pt_path, weights_only=False)
    assert saved.raw_texts[0] == "poem for 0"
    assert saved.raw_texts[2] is None


def test_pt_edge_index_shape(tmp_path):
    args = _minimal_graph()
    pt_path, _ = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    saved = torch.load(pt_path, weights_only=False)
    assert saved.edge_index.shape[0] == 2

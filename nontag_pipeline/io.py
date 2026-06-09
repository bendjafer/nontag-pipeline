# nontag_pipeline/io.py
"""Save pseudo-TAG graph as PyG .pt and human-readable .json."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def save_pseudo_tag(
    G: nx.Graph,
    y: np.ndarray,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    class_names: list[str],
    records: list[dict],
    output_dir: str | Path,
    dataset: str,
    style: str,
) -> tuple[Path, Path]:
    """Write pseudo-TAG to <output_dir>/pseudo_tag_<dataset>_<style>.{pt,json}."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"pseudo_tag_{dataset}_{style}"

    raw_texts: list[str | None] = [None] * G.number_of_nodes()
    for rec in records:
        raw_texts[rec["node"]] = rec.get("text")

    edges = list(G.edges())
    if edges:
        src = [u for u, _ in edges] + [v for _, v in edges]
        dst = [v for _, v in edges] + [u for u, _ in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    pyg_data = Data(
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    pyg_data.raw_texts = raw_texts
    pyg_data.class_names = class_names

    pt_path = out / f"{stem}.pt"
    torch.save(pyg_data, pt_path)

    def _split(n: int) -> str:
        if train_mask[n].item():
            return "train"
        if val_mask[n].item():
            return "val"
        return "test"

    json_obj = {
        "dataset": dataset,
        "style": style,
        "class_names": class_names,
        "nodes": [
            {
                "node_id": int(n),
                "text": raw_texts[n],
                "label": int(y[n]),
                "label_name": class_names[int(y[n])],
                "split": _split(n),
            }
            for n in G.nodes()
        ],
        "edges": [[int(u), int(v)] for u, v in G.edges()],
    }
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(json_obj, indent=2))

    return pt_path, json_path

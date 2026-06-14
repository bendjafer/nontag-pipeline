# nontag_pipeline/io.py
"""Save pseudo-TAG graph as PyG .pt and human-readable .json."""
from __future__ import annotations
import json
import os
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
    tag: str = "",
) -> tuple[Path, Path]:
    """Write pseudo-TAG to <output_dir>/pseudo_tag_<dataset>_<style>[_<tag>].{pt,json}."""
    n = G.number_of_nodes()
    if set(G.nodes()) != set(range(n)):
        raise ValueError("save_pseudo_tag requires node ids 0..n-1 (raw_texts is index-aligned)")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"pseudo_tag_{dataset}_{style}" + (f"_{tag}" if tag else "")

    raw_texts: list[str | None] = [None] * G.number_of_nodes()
    rec_by_node: dict[int, dict] = {}
    for rec in records:
        v = rec["node"]
        if not (0 <= v < n):
            raise ValueError(f"Record node_id {v} is out of range [0, {n})")
        raw_texts[v] = rec.get("text")
        rec_by_node[v] = rec

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
    pt_tmp  = pt_path.with_suffix(".pt.tmp")
    torch.save(pyg_data, pt_tmp)
    os.replace(pt_tmp, pt_path)

    def _split(v: int) -> str:
        if train_mask[v].item(): return "train"
        if val_mask[v].item():   return "val"
        if test_mask[v].item():  return "test"
        return "not_sampled"

    json_obj = {
        "dataset": dataset,
        "style": style,
        "class_names": class_names,
        "warning": (
            "Contains true labels for ALL textualized nodes, including test. For human "
            "inspection and scoring only — never feed this JSON to a predictor; "
            "use the .pt file."
        ),
        # Only emit nodes that were textualized; non-sampled nodes have no text and
        # would appear with misleading split labels in a subset run.
        "nodes": [
            {
                "node_id": v,
                "text": rec["text"],
                "label": int(y[v]),
                "label_name": class_names[int(y[v])],
                "split": _split(v),
                "n_selected": rec.get("n_selected"),
                "n_visible": rec.get("n_visible"),
            }
            for v, rec in rec_by_node.items()
        ],
        "edges": [[int(u), int(v)] for u, v in G.edges()],
    }
    json_path = out / f"{stem}.json"
    json_tmp  = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(json_obj, indent=2))
    os.replace(json_tmp, json_path)

    return pt_path, json_path

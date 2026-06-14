"""Phase 2 — call LLM using pre-computed proportions and save the pseudo-TAG.

Reads precomputed_<dataset>.json, maps class proportions to narrative topics,
generates one text per node via the LLM, and saves the result.

Changing narrative topics or style only requires rerunning this step.

Usage:
  python pipeline/textualize.py
  python pipeline/textualize.py --style news
  python pipeline/textualize.py --style story --dataset cora

Output:
  outputs/pseudo_tag_<dataset>_<style>.pt
  outputs/pseudo_tag_<dataset>_<style>.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.narratives import STYLE_TEMPLATES, map_proportions_to_themes
from nontag_pipeline.textualize import build_generation_prompt
from nontag_pipeline import llm
from nontag_pipeline.io import save_pseudo_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--style",   default=config.STYLE,   choices=list(STYLE_TEMPLATES.keys()))
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    precomputed_path = Path(config.OUTPUT_DIR) / f"precomputed_{args.dataset}.json"
    if not precomputed_path.exists():
        raise FileNotFoundError(
            f"{precomputed_path} not found — run pipeline/precompute.py first."
        )

    precomputed = json.loads(precomputed_path.read_text())
    if precomputed.get("dataset") != args.dataset:
        raise ValueError(
            f"Precomputed file is for dataset={precomputed.get('dataset')!r}, "
            f"but --dataset={args.dataset!r}."
        )
    node_records = precomputed["nodes"]
    print(f"Loaded {len(node_records)} precomputed nodes from {precomputed_path.name}")

    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        args.dataset, seed=config.SEED, root=config.DATA_ROOT
    )
    if precomputed.get("class_names") != class_names:
        raise ValueError(
            f"class_names in precomputed file {precomputed.get('class_names')} "
            f"do not match dataset {class_names}. Delete and rerun pipeline/precompute.py."
        )

    all_node_ids = set(int(n) for n in G.nodes())
    stored_ids   = set(r["node_id"] for r in node_records)
    if stored_ids != all_node_ids:
        n_nodes = G.number_of_nodes()
        pilot_train = torch.zeros(n_nodes, dtype=torch.bool)
        pilot_val   = torch.zeros(n_nodes, dtype=torch.bool)
        pilot_test  = torch.zeros(n_nodes, dtype=torch.bool)
        for r in node_records:
            v = r["node_id"]
            if r["split"] == "train":  pilot_train[v] = True
            elif r["split"] == "val":  pilot_val[v]   = True
            elif r["split"] == "test": pilot_test[v]  = True
        train_mask, val_mask, test_mask = pilot_train, pilot_val, pilot_test

    records: list[dict] = []
    total = len(node_records)
    n_no_signal = 0

    for i, node_data in enumerate(node_records):
        v = node_data["node_id"]

        if node_data["status"] == "no_signal":
            records.append({"node": v, "status": "no_signal", "text": None,
                            "n_selected": node_data["n_selected"], "n_visible": 0})
            n_no_signal += 1
        else:
            themes       = map_proportions_to_themes(node_data["proportions"], args.dataset)
            system, user = build_generation_prompt(themes, args.style, config.TARGET_LEN)
            text         = llm.complete(user, system=system)
            records.append({"node": v, "status": "ok", "text": text,
                            "n_selected": node_data["n_selected"],
                            "n_visible":  node_data["n_visible"]})

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total}", flush=True)

    pt_path, json_path = save_pseudo_tag(
        G, y, train_mask, val_mask, test_mask, class_names, records,
        output_dir=config.OUTPUT_DIR, dataset=args.dataset, style=args.style, tag="",
    )

    print(f"\nDone. {total - n_no_signal} textualized, {n_no_signal} no_signal.")
    print(f"Saved:\n  {pt_path}\n  {json_path}")


if __name__ == "__main__":
    main()

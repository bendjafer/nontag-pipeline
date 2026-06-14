"""Step 2 — call LLM using pre-computed proportions and save the pseudo-TAG.

Reads the precomputed JSON produced by run_precompute.py, maps class proportions
to narrative topics (from config), generates one text per node via the LLM, and
saves the result as a PyG .pt file (with raw_texts) and a human-readable .json.

Changing narrative topics in narratives.py only requires rerunning this step —
run_precompute.py does not need to be rerun.

Usage:
  python run_textualize.py                   # reads precomputed_<dataset>.json
  python run_textualize.py --n-train 1000    # reads precomputed_<dataset>_1000.json
  python run_textualize.py --style news      # generates news-style text

Output:
  outputs/pseudo_tag_<dataset>_<style>.pt / .json          (all nodes)
  outputs/pseudo_tag_<dataset>_<style>_<N>.pt / .json      (subset)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.narratives import STYLE_TEMPLATES, map_proportions_to_themes
from nontag_pipeline.textualize import build_generation_prompt
from nontag_pipeline import llm
from nontag_pipeline.io import save_pseudo_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=0,
                   help="Must match the value used in run_precompute.py (0 = all)")
    p.add_argument("--style",   default=config.STYLE,   choices=list(STYLE_TEMPLATES.keys()))
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = str(args.n_train) if args.n_train > 0 else ""

    precomputed_path = (
        Path(config.OUTPUT_DIR)
        / f"precomputed_{args.dataset}{('_' + tag) if tag else ''}.json"
    )
    if not precomputed_path.exists():
        raise FileNotFoundError(
            f"{precomputed_path} not found — run run_precompute.py first."
        )

    precomputed = json.loads(precomputed_path.read_text())
    if precomputed.get("dataset") != args.dataset:
        raise ValueError(
            f"Precomputed file is for dataset={precomputed.get('dataset')!r}, "
            f"but --dataset={args.dataset!r}."
        )
    node_records = precomputed["nodes"]
    print(f"Loaded {len(node_records)} precomputed nodes from {precomputed_path.name}")

    # Load graph for edge_index, y, and masks (needed for .pt and .json output)
    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        args.dataset, seed=config.SEED, root=config.DATA_ROOT
    )
    if precomputed.get("class_names") != class_names:
        raise ValueError(
            f"class_names in precomputed file {precomputed.get('class_names')} "
            f"do not match dataset {class_names}. Delete and rerun run_precompute.py."
        )

    # Build pilot masks if a subset was selected
    if args.n_train > 0:
        n_nodes = G.number_of_nodes()
        pilot_train = torch.zeros(n_nodes, dtype=torch.bool)
        pilot_val   = torch.zeros(n_nodes, dtype=torch.bool)
        pilot_test  = torch.zeros(n_nodes, dtype=torch.bool)
        for r in node_records:
            v = r["node_id"]
            if r["split"] == "train":   pilot_train[v] = True
            elif r["split"] == "val":   pilot_val[v]   = True
            elif r["split"] == "test":  pilot_test[v]  = True
            else:
                raise ValueError(f"node {v} has unknown split {r['split']!r}")
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
            # Map class proportions → narrative topics here, not in precompute.
            # This means changing TOPIC_MAPS only requires rerunning this step.
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
        output_dir=config.OUTPUT_DIR, dataset=args.dataset, style=args.style, tag=tag,
    )

    print(f"\nDone. {total - n_no_signal} textualized, {n_no_signal} no_signal.")
    print(f"Saved:\n  {pt_path}\n  {json_path}")


if __name__ == "__main__":
    main()

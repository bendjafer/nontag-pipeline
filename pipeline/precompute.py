"""Phase 1 — pre-compute PPR selections and class proportions for every node.

No LLM calls and no narrative dependency. Run this before pipeline/textualize.py.
Proportions are stored as raw class-name ratios; topic mapping is applied in
Phase 2, so you can change narrative topics without rerunning this step.

Usage:
  python pipeline/precompute.py                   # all nodes
  python pipeline/precompute.py --n-train 1000    # 1000 train + proportional val/test

Output:
  outputs/precomputed_<dataset>.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.select import ppr_selection, neighbor_label_proportions


def _checkpoint(out_path: Path, args: argparse.Namespace, class_names: list, records: list) -> None:
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "dataset": args.dataset,
        "n_train": args.n_train, "class_names": class_names,
        "ppr_k": config.PPR_K, "ppr_m": config.PPR_M, "ppr_alpha": config.PPR_ALPHA,
        "nodes": records,
    }, indent=2))
    os.replace(tmp, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=0,
                   help="Training nodes to include (0 = all)")
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        args.dataset, seed=config.SEED, root=config.DATA_ROOT
    )

    visible_labels = {
        int(n): int(y[n])
        for n in G.nodes()
        if train_mask[n].item() or val_mask[n].item()
    }
    assert not any(test_mask[int(n)].item() for n in visible_labels), \
        "Leakage: test node found in visible_labels"

    train_nodes = [int(n) for n in G.nodes() if train_mask[n].item()]
    val_nodes   = [int(n) for n in G.nodes() if val_mask[n].item()]
    test_nodes  = [int(n) for n in G.nodes() if test_mask[n].item()]

    if args.n_train > 0:
        ratio   = args.n_train / len(train_nodes)
        n_val   = max(1, round(len(val_nodes)  * ratio))
        n_test  = max(1, round(len(test_nodes) * ratio))
        rng     = random.Random(config.SEED)
        s_train = rng.sample(train_nodes, min(args.n_train, len(train_nodes)))
        s_val   = rng.sample(val_nodes,   min(n_val,        len(val_nodes)))
        s_test  = rng.sample(test_nodes,  min(n_test,       len(test_nodes)))
        selected_nodes = set(s_train + s_val + s_test)
        print(f"Subset: {len(s_train)} train / {len(s_val)} val / {len(s_test)} test")
    else:
        selected_nodes = set(G.nodes())
        print(f"All nodes: {G.number_of_nodes()}")

    def split_of(v: int) -> str:
        if train_mask[v].item(): return "train"
        if val_mask[v].item():   return "val"
        return "test"

    out_dir  = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"precomputed_{args.dataset}.json"

    already_done: dict[int, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        stored  = (existing.get("ppr_k"), existing.get("ppr_m"), existing.get("ppr_alpha"))
        current = (config.PPR_K, config.PPR_M, config.PPR_ALPHA)
        if stored != current:
            raise ValueError(
                f"PPR params in {out_path.name} {stored} differ from config {current}. "
                "Delete the file or restore the original params before resuming."
            )
        already_done = {
            r["node_id"]: r
            for r in existing.get("nodes", [])
            if r["node_id"] in selected_nodes
        }
        print(f"Resuming — {len(already_done)} nodes already computed")

    records: list[dict] = list(already_done.values())
    remaining = selected_nodes - already_done.keys()
    total = len(selected_nodes)
    done  = len(already_done)

    for node_id in G.nodes():
        v = int(node_id)
        if v not in remaining:
            continue

        selected    = ppr_selection(G, v, k=config.PPR_K, m=config.PPR_M, alpha=config.PPR_ALPHA)
        proportions = neighbor_label_proportions(selected, visible_labels, class_names)
        n_visible   = sum(1 for u in selected if u in visible_labels)

        records.append({
            "node_id": v, "split": split_of(v),
            "label": int(y[v]), "label_name": class_names[int(y[v])],
            "status": "no_signal" if not proportions else "ok",
            "n_selected": len(selected), "n_visible": n_visible,
            "proportions": proportions,
        })

        done += 1
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)
            _checkpoint(out_path, args, class_names, records)

    if not remaining:
        _checkpoint(out_path, args, class_names, records)

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\nDone — {total} nodes, {n_ok} with signal, {total - n_ok} no_signal")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

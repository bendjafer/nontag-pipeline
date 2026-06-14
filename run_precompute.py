"""Step 1 — pre-compute PPR selections, proportions, and themes for every node.

No LLM calls. Run this before run_textualize.py.

Usage:
  python run_precompute.py                   # all nodes
  python run_precompute.py --n-train 1000    # 1000 train + proportional val/test

Output:
  outputs/precomputed_<dataset>_<style>.json          (all nodes)
  outputs/precomputed_<dataset>_<style>_<N>.json      (subset)
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.select import ppr_selection, neighbor_label_proportions
from nontag_pipeline.narratives import map_proportions_to_themes, STYLE_TEMPLATES


def _checkpoint(out_path: Path, args: argparse.Namespace, class_names: list, records: list) -> None:
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "dataset": args.dataset, "style": args.style,
        "n_train": args.n_train, "class_names": class_names,
        "ppr_k": config.PPR_K, "ppr_m": config.PPR_M, "ppr_alpha": config.PPR_ALPHA,
        "nodes": records,
    }, indent=2))
    os.replace(tmp, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=0,
                   help="Training nodes to include (0 = all)")
    p.add_argument("--style",   default=config.STYLE,   choices=list(STYLE_TEMPLATES.keys()))
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        args.dataset, seed=config.SEED, root=config.DATA_ROOT
    )

    # Only train+val labels are visible during generation (leakage rule)
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
        ratio  = args.n_train / len(train_nodes)
        n_val  = max(1, round(len(val_nodes)  * ratio))
        n_test = max(1, round(len(test_nodes) * ratio))
        rng    = random.Random(config.SEED)
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
    tag      = f"_{args.n_train}" if args.n_train > 0 else ""
    out_path = out_dir / f"precomputed_{args.dataset}_{args.style}{tag}.json"

    # Resume: load already-computed nodes so we skip them
    already_done: dict[int, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        stored = (existing.get("ppr_k"), existing.get("ppr_m"), existing.get("ppr_alpha"))
        current = (config.PPR_K, config.PPR_M, config.PPR_ALPHA)
        if stored != current:
            raise ValueError(
                f"PPR params in {out_path.name} {stored} differ from config {current}. "
                "Delete the file or restore the original params before resuming."
            )
        # Only keep nodes that are still in the current selected set
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

        if not proportions:
            records.append({
                "node_id": v, "split": split_of(v),
                "label": int(y[v]), "label_name": class_names[int(y[v])],
                "status": "no_signal",
                "n_selected": len(selected), "n_visible": 0,
                "proportions": {}, "themes": [],
            })
        else:
            themes    = map_proportions_to_themes(proportions, args.dataset)
            n_visible = sum(1 for u in selected if u in visible_labels)
            records.append({
                "node_id": v, "split": split_of(v),
                "label": int(y[v]), "label_name": class_names[int(y[v])],
                "status": "ok",
                "n_selected": len(selected), "n_visible": n_visible,
                "proportions": proportions,
                "themes": [[theme, round(pct, 4)] for theme, pct in themes],
            })

        done += 1
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)
            _checkpoint(out_path, args, class_names, records)

    # If all nodes were already done (nothing to compute), checkpoint was never
    # triggered — write once to normalise any filtered-out nodes from the old file.
    if not remaining:
        _checkpoint(out_path, args, class_names, records)

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\nDone — {total} nodes, {n_ok} with signal, {total - n_ok} no_signal")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

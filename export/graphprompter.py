"""Export pseudo-TAG to GraphPrompter's expected directory layout.

Run AFTER run_embed.py --embedder sbert so that data.x exists.

Usage:
    python export/graphprompter.py
    python export/graphprompter.py --style news --dataset pubmed

Produces:
    dataset/tape_<dataset>/processed/data.pt
    dataset/tape_<dataset>/processed/text.csv
    dataset/tape_<dataset>/split/train_indices.txt
    dataset/tape_<dataset>/split/val_indices.txt
    dataset/tape_<dataset>/split/test_indices.txt

Copy the entire dataset/ folder into the cloned graphprompter repo.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nontag_pipeline import config
from nontag_pipeline.narratives import STYLE_TEMPLATES
from nontag_pipeline.embed import sbert as sbert_emb

EMBEDDER = sbert_emb  # GraphPrompter uses SBERT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    p.add_argument("--style",   default=config.STYLE,   choices=list(STYLE_TEMPLATES.keys()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = EMBEDDER.DEFAULT_MODEL
    stem    = f"pseudo_tag_{args.dataset}_{args.style}_{model_name}"
    in_path = Path(config.OUTPUT_DIR) / f"{stem}.pt"

    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found — run: python run_embed.py --embedder sbert"
        )

    print(f"Loading {in_path.name} ...")
    data = torch.load(in_path, weights_only=False)

    if not hasattr(data, "x") or data.x is None:
        raise RuntimeError("data.x is missing — run run_embed.py first.")

    class_names: list[str] = data.class_names
    raw_texts: list[str | None] = data.raw_texts

    # Extract induced subgraph of textualized nodes only — avoids flooding
    # GNN message passing with zero vectors for untextualized nodes.
    textualized = torch.tensor(
        [i for i, t in enumerate(raw_texts) if t is not None], dtype=torch.long
    )
    old_to_new = {int(old): new for new, old in enumerate(textualized.tolist())}

    src, dst = data.edge_index
    keep = [int(s.item()) in old_to_new and int(d.item()) in old_to_new
            for s, d in zip(src, dst)]
    new_src = torch.tensor([old_to_new[int(s)] for s, k in zip(src, keep) if k])
    new_dst = torch.tensor([old_to_new[int(d)] for d, k in zip(dst, keep) if k])
    new_edge_index = torch.stack([new_src, new_dst], dim=0)

    sub_x     = data.x[textualized]
    sub_y     = data.y[textualized]
    sub_texts = [raw_texts[i] for i in textualized.tolist()]
    n         = len(textualized)

    def remap_mask(mask_tensor):
        return [old_to_new[i] for i in mask_tensor.nonzero(as_tuple=True)[0].tolist()
                if i in old_to_new]

    splits = {
        "train": remap_mask(data.train_mask),
        "val":   remap_mask(data.val_mask),
        "test":  remap_mask(data.test_mask),
    }

    print(f"Induced subgraph: {n} nodes, {new_edge_index.shape[1]} edges "
          f"(was {data.y.shape[0]} nodes, {data.edge_index.shape[1]} edges)")

    out_root = Path("dataset") / f"tape_{args.dataset}"
    proc_dir = out_root / "processed"
    splt_dir = out_root / "split"
    proc_dir.mkdir(parents=True, exist_ok=True)
    splt_dir.mkdir(parents=True, exist_ok=True)

    from torch_geometric.data import Data
    graph = Data(x=sub_x, edge_index=new_edge_index, y=sub_y, num_nodes=n)
    pt_path = proc_dir / "data.pt"
    torch.save(graph, pt_path.with_suffix(".pt.tmp"))
    os.replace(pt_path.with_suffix(".pt.tmp"), pt_path)
    print(f"Saved data.pt  ({n} nodes, x={sub_x.shape})")

    csv_path = proc_dir / "text.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "label", "title", "abstract"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "node_id":  i,
                "label":    class_names[int(sub_y[i])],
                "title":    "",
                "abstract": sub_texts[i],
            })
    print(f"Saved text.csv ({n} rows)")

    for split, indices in splits.items():
        path = splt_dir / f"{split}_indices.txt"
        path.write_text("\n".join(map(str, indices)))
        print(f"Saved {split}_indices.txt ({len(indices)} nodes)")

    print()
    print(f"Output dir : {out_root.resolve()}")
    print(f"x dim      : {sub_x.shape[1]}  ← patch num_features in graphprompter/src/dataset/pubmed.py")
    print(f"Classes    : {class_names}")


if __name__ == "__main__":
    main()

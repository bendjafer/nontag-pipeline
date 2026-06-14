"""Step 3 — embed node texts into feature vectors and save updated pseudo-TAG.

Reads a pseudo-TAG .pt file produced by run_textualize.py, embeds raw_texts
into a node feature matrix x using the chosen embedding model, and saves a new
.pt file with data.x populated. This x is the node feature input for GNN-based
predictors.

Embedders:
  sbert     — sentence-transformers/all-mpnet-base-v2 (768-dim)  → GraphPrompter
  graphgpt  — bert-base-uncased mean-pool (768-dim)              → GraphGPT

Usage:
  python run_embed.py --embedder sbert
  python run_embed.py --embedder graphgpt
  python run_embed.py --embedder sbert --n-train 1000 --style news

Output:
  outputs/pseudo_tag_<dataset>_<style>[_<N>]_<embedder>.pt
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch

from nontag_pipeline import config
from nontag_pipeline.narratives import STYLE_TEMPLATES
from nontag_pipeline.embed import sbert as sbert_emb
from nontag_pipeline.embed import graphgpt as graphgpt_emb

EMBEDDERS = {
    "sbert":    sbert_emb,
    "graphgpt": graphgpt_emb,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", required=True, choices=list(EMBEDDERS.keys()),
                   help="Embedding model to use")
    p.add_argument("--n-train", type=int, default=0,
                   help="Must match the value used in run_textualize.py (0 = all)")
    p.add_argument("--style",   default=config.STYLE,   choices=list(STYLE_TEMPLATES.keys()))
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    p.add_argument("--device",  default="cpu",
                   help="Device for embedding (cpu / cuda / mps)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag  = f"_{args.n_train}" if args.n_train > 0 else ""
    stem = f"pseudo_tag_{args.dataset}_{args.style}{tag}"

    in_path = Path(config.OUTPUT_DIR) / f"{stem}.pt"
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found — run run_textualize.py first."
        )

    print(f"Loading {in_path.name} ...")
    data = torch.load(in_path, weights_only=False)

    raw_texts: list[str | None] = data.raw_texts
    n_total     = len(raw_texts)
    n_with_text = sum(1 for t in raw_texts if t is not None)
    print(f"Nodes: {n_total} total, {n_with_text} with text, "
          f"{n_total - n_with_text} None (will be zero vectors)")

    print(f"Embedding with {args.embedder} on {args.device} ...")
    embedder = EMBEDDERS[args.embedder]
    x = embedder.embed(raw_texts, device=args.device)

    assert x.shape[0] == n_total, \
        f"Embedder returned {x.shape[0]} rows but expected {n_total}"

    data.x = x
    print(f"x shape: {x.shape}")

    out_path = Path(config.OUTPUT_DIR) / f"{stem}_{args.embedder}.pt"
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

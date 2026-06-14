"""Phase 3 — embed node texts into feature vectors and save updated pseudo-TAG.

Reads pseudo_tag_<dataset>_<style>.pt produced by pipeline/textualize.py, embeds
raw_texts into a node feature matrix x, and saves a new .pt file with data.x.

Embedders:
  sbert     — sentence-transformers/all-mpnet-base-v2 (768-dim)  → GraphPrompter
  graphgpt  — bert-base-uncased mean-pool (768-dim)              → GraphGPT

Usage:
  python pipeline/embed.py --embedder sbert
  python pipeline/embed.py --embedder graphgpt --style news

Output:
  outputs/pseudo_tag_<dataset>_<style>_<model-name>.pt
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    p.add_argument("--embedder", required=True, choices=list(EMBEDDERS.keys()))
    p.add_argument("--style",   default=config.STYLE,   choices=list(STYLE_TEMPLATES.keys()))
    p.add_argument("--dataset", default=config.DATASET, choices=["pubmed", "cora"])
    p.add_argument("--device",  default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stem = f"pseudo_tag_{args.dataset}_{args.style}"

    in_path = Path(config.OUTPUT_DIR) / f"{stem}.pt"
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found — run pipeline/textualize.py first."
        )

    print(f"Loading {in_path.name} ...")
    data = torch.load(in_path, weights_only=False)

    raw_texts: list[str | None] = data.raw_texts
    n_total     = len(raw_texts)
    n_with_text = sum(1 for t in raw_texts if t is not None)
    print(f"Nodes: {n_total} total, {n_with_text} with text, "
          f"{n_total - n_with_text} None (will be zero vectors)")

    print(f"Embedding with {args.embedder} on {args.device} ...")
    x = EMBEDDERS[args.embedder].embed(raw_texts, device=args.device)

    assert x.shape[0] == n_total
    data.x = x
    print(f"x shape: {x.shape}")

    model_name = EMBEDDERS[args.embedder].DEFAULT_MODEL
    out_path = Path(config.OUTPUT_DIR) / f"{stem}_{model_name}.pt"
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

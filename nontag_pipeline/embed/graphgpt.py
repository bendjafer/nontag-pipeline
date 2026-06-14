"""Transformer embedder for GraphGPT.

GraphGPT (Tang et al., 2023) feeds node texts through a language model and uses
the resulting representations as initial node features for its graph encoder (GNN).
Mean pooling over non-padding token embeddings from the last hidden state is the
standard feature extraction approach used in their preprocessing pipeline.

Reference model: bert-base-uncased (768-dim), compatible with GraphGPT's GNN input.
For higher-quality embeddings use intfloat/e5-large-v2 (1024-dim) — change DEFAULT_MODEL.
Nodes with no text (None) receive a zero vector.
"""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "bert-base-uncased"


def embed(
    texts: list[str | None],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    """Mean-pool last hidden states from a transformer. Returns (N, dim) float tensor.

    Nodes whose text is None get a zero vector.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dim = model.config.hidden_size
    valid_idx   = [i for i, t in enumerate(texts) if t is not None]
    valid_texts = [texts[i] for i in valid_idx]

    x = torch.zeros(len(texts), dim, dtype=torch.float32)
    if not valid_texts:
        return x

    with torch.no_grad():
        for start in range(0, len(valid_texts), batch_size):
            batch   = valid_texts[start : start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)

            output = model(**encoded)
            token_emb = output.last_hidden_state           # (B, T, D)
            mask      = encoded["attention_mask"]          # (B, T)

            # Mean pool over non-padding tokens
            mask_exp   = mask.unsqueeze(-1).float()        # (B, T, 1)
            pooled     = (token_emb * mask_exp).sum(1)    # (B, D)
            counts     = mask_exp.sum(1).clamp(min=1e-9)  # (B, 1)
            embeddings = (pooled / counts).cpu().float()  # (B, D)

            for j, idx in enumerate(valid_idx[start : start + batch_size]):
                x[idx] = embeddings[j]

            processed = min(start + batch_size, len(valid_texts))
            print(f"  {processed}/{len(valid_texts)}", flush=True)

    return x

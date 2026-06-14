"""SBERT embedder for GraphPrompter.

GraphPrompter (Chen et al., 2023) embeds node texts with a Sentence-BERT model
and uses the resulting embeddings as node features x fed into its GNN component.

Reference model: sentence-transformers/all-mpnet-base-v2 (768-dim).
Nodes with no text (None) receive a zero vector.
"""
from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-mpnet-base-v2"


def embed(
    texts: list[str | None],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """Embed texts with SBERT. Returns float tensor of shape (N, dim).

    Nodes whose text is None get a zero vector; they can be identified via
    raw_texts[i] is None in the .pt file.
    """
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()

    valid_idx   = [i for i, t in enumerate(texts) if t is not None]
    valid_texts = [texts[i] for i in valid_idx]

    x = torch.zeros(len(texts), dim, dtype=torch.float32)
    if not valid_texts:
        return x

    embeddings = model.encode(
        valid_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )
    for i, idx in enumerate(valid_idx):
        x[idx] = embeddings[i].cpu().float()

    return x

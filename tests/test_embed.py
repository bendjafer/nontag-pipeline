"""Unit tests for nontag_pipeline.embed — mocked so no model downloads needed."""
from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch


# ── helpers ────────────────────────────────────────────────────────────────

class _FakeBatch:
    """Minimal tokenizer-output stand-in with real tensors.
    Implements keys() so **encoded unpacking works in the embedder."""
    def __init__(self, B: int, T: int = 8):
        self._d = {
            "input_ids":      torch.zeros(B, T, dtype=torch.long),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
    def __getitem__(self, k): return self._d[k]
    def keys(self):           return self._d.keys()
    def to(self, device):     return self


class _FakeOutput:
    """Minimal model-output stand-in."""
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


def _make_tok_model(n_texts: int, dim: int = 768, T: int = 8):
    """Return (tokenizer_mock, model_mock) that produce real tensors."""
    tok = MagicMock()
    tok.side_effect = lambda texts, **kw: _FakeBatch(len(texts), T)

    mod = MagicMock()
    mod.config.hidden_size = dim
    mod.eval.return_value  = None
    mod.to.return_value    = mod
    mod.side_effect = lambda **kw: _FakeOutput(
        torch.ones(kw["input_ids"].shape[0], T, dim) * 0.5
    )
    return tok, mod


# ── SBERT ──────────────────────────────────────────────────────────────────

def _mock_sbert(dim: int = 768):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim
    model.encode.side_effect = lambda texts, **kw: (
        torch.ones(len(texts), dim) * 0.1
        if kw.get("convert_to_tensor")
        else (torch.ones(len(texts), dim) * 0.1).numpy()
    )
    return model


def test_sbert_output_shape():
    texts = ["hello", "world", None, "foo"]
    with patch("nontag_pipeline.embed.sbert.SentenceTransformer", return_value=_mock_sbert()):
        from nontag_pipeline.embed import sbert
        x = sbert.embed(texts)
    assert x.shape == (4, 768)


def test_sbert_none_is_zero_vector():
    texts = ["hello", None, "world"]
    with patch("nontag_pipeline.embed.sbert.SentenceTransformer", return_value=_mock_sbert()):
        from nontag_pipeline.embed import sbert
        x = sbert.embed(texts)
    assert x[1].sum().item() == 0.0


def test_sbert_non_none_is_nonzero():
    texts = ["hello", None]
    with patch("nontag_pipeline.embed.sbert.SentenceTransformer", return_value=_mock_sbert()):
        from nontag_pipeline.embed import sbert
        x = sbert.embed(texts)
    assert x[0].sum().item() != 0.0


def test_sbert_all_none_returns_zeros():
    texts = [None, None, None]
    with patch("nontag_pipeline.embed.sbert.SentenceTransformer", return_value=_mock_sbert()):
        from nontag_pipeline.embed import sbert
        x = sbert.embed(texts)
    assert x.shape == (3, 768)
    assert (x == 0).all()


def test_sbert_returns_float32():
    texts = ["hello"]
    with patch("nontag_pipeline.embed.sbert.SentenceTransformer", return_value=_mock_sbert()):
        from nontag_pipeline.embed import sbert
        x = sbert.embed(texts)
    assert x.dtype == torch.float32


# ── GraphGPT ───────────────────────────────────────────────────────────────

def test_graphgpt_output_shape():
    texts = ["alpha", "beta"]
    tok, mod = _make_tok_model(n_texts=2)
    with patch("nontag_pipeline.embed.graphgpt.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("nontag_pipeline.embed.graphgpt.AutoModel.from_pretrained",    return_value=mod):
        from nontag_pipeline.embed import graphgpt
        x = graphgpt.embed(texts, batch_size=2)
    assert x.shape == (2, 768)


def test_graphgpt_none_is_zero():
    texts = [None, "hello"]
    tok, mod = _make_tok_model(n_texts=1)
    with patch("nontag_pipeline.embed.graphgpt.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("nontag_pipeline.embed.graphgpt.AutoModel.from_pretrained",    return_value=mod):
        from nontag_pipeline.embed import graphgpt
        x = graphgpt.embed(texts, batch_size=4)
    assert x[0].sum().item() == 0.0   # None → zero
    assert x[1].sum().item() != 0.0   # text → nonzero


def test_graphgpt_all_none_returns_zeros():
    texts = [None, None]
    tok, mod = _make_tok_model(n_texts=0)
    with patch("nontag_pipeline.embed.graphgpt.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("nontag_pipeline.embed.graphgpt.AutoModel.from_pretrained",    return_value=mod):
        from nontag_pipeline.embed import graphgpt
        x = graphgpt.embed(texts)
    assert x.shape == (2, 768)
    assert (x == 0).all()
    mod.assert_not_called()


def test_graphgpt_returns_float32():
    texts = [None]
    tok, mod = _make_tok_model(n_texts=0)
    with patch("nontag_pipeline.embed.graphgpt.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("nontag_pipeline.embed.graphgpt.AutoModel.from_pretrained",    return_value=mod):
        from nontag_pipeline.embed import graphgpt
        x = graphgpt.embed(texts)
    assert x.dtype == torch.float32


def test_graphgpt_non_none_is_nonzero():
    texts = ["only text"]
    tok, mod = _make_tok_model(n_texts=1)
    with patch("nontag_pipeline.embed.graphgpt.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("nontag_pipeline.embed.graphgpt.AutoModel.from_pretrained",    return_value=mod):
        from nontag_pipeline.embed import graphgpt
        x = graphgpt.embed(texts, batch_size=4)
    assert x[0].sum().item() != 0.0

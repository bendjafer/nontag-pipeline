import torch
from nontag_pipeline.data import make_split_masks


def test_masks_are_disjoint():
    tr, va, te = make_split_masks(100, seed=42)
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()


def test_masks_cover_all_nodes():
    tr, va, te = make_split_masks(100, seed=42)
    assert (tr | va | te).all()


def test_split_sizes_are_60_20_20():
    tr, va, te = make_split_masks(1000, seed=42)
    assert tr.sum().item() == 600
    assert va.sum().item() == 200
    assert te.sum().item() == 200


def test_split_is_deterministic_for_seed():
    a = make_split_masks(500, seed=42)
    b = make_split_masks(500, seed=42)
    for ma, mb in zip(a, b):
        assert torch.equal(ma, mb)


def test_split_differs_across_seeds():
    a = make_split_masks(500, seed=42)
    b = make_split_masks(500, seed=7)
    assert any(not torch.equal(ma, mb) for ma, mb in zip(a, b))

"""Tests for data_loader utilities."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from picochem.data_loader import split_dataset


def _make_pairs(n=100):
    rng = np.random.default_rng(0)
    return [
        (rng.integers(0, 10, size=rng.integers(3, 8)).astype(np.int32),
         rng.integers(0, 10, size=rng.integers(3, 8)).astype(np.int32))
        for _ in range(n)
    ]


def test_split_sizes():
    pairs = _make_pairs(100)
    train, val = split_dataset(pairs, val_fraction=0.1, seed=0)
    assert len(train) + len(val) == len(pairs)
    assert len(val) == 10


def test_split_disjoint():
    pairs = _make_pairs(100)
    train, val = split_dataset(pairs, val_fraction=0.1, seed=0)
    # Use id() to confirm no object appears in both sets
    train_ids = {id(p) for p in train}
    val_ids   = {id(p) for p in val}
    assert train_ids.isdisjoint(val_ids)


def test_split_reproducible():
    pairs = _make_pairs(100)
    train1, val1 = split_dataset(pairs, val_fraction=0.1, seed=42)
    train2, val2 = split_dataset(pairs, val_fraction=0.1, seed=42)
    assert [id(p) for p in val1] == [id(p) for p in val2]
    assert [id(p) for p in train1] == [id(p) for p in train2]


def test_split_different_seeds():
    pairs = _make_pairs(100)
    _, val1 = split_dataset(pairs, val_fraction=0.1, seed=0)
    _, val2 = split_dataset(pairs, val_fraction=0.1, seed=1)
    # Different seeds should (almost certainly) give different splits
    assert [id(p) for p in val1] != [id(p) for p in val2]


def test_split_min_val_one():
    pairs = _make_pairs(5)
    train, val = split_dataset(pairs, val_fraction=0.01, seed=0)
    assert len(val) >= 1
    assert len(train) + len(val) == 5

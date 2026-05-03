"""Tests for LR scheduler functions."""
import sys
sys.path.insert(0, ".")

import pytest
from picochem.scheduler import (
    linear_warmup_cosine_decay,
    linear_warmup_linear_decay,
)

WARMUP = 100
TOTAL  = 1000
PEAK   = 1e-3
MIN    = 1e-5


# ── cosine schedule ──────────────────────────────────────────────────────────

def test_cosine_step_zero():
    assert linear_warmup_cosine_decay(0, WARMUP, TOTAL, PEAK, MIN) == 0.0


def test_cosine_at_warmup_end():
    lr = linear_warmup_cosine_decay(WARMUP, WARMUP, TOTAL, PEAK, MIN)
    assert abs(lr - PEAK) < 1e-12


def test_cosine_at_total_steps():
    lr = linear_warmup_cosine_decay(TOTAL, WARMUP, TOTAL, PEAK, MIN)
    assert abs(lr - MIN) < 1e-12


def test_cosine_beyond_total_steps():
    lr = linear_warmup_cosine_decay(TOTAL + 500, WARMUP, TOTAL, PEAK, MIN)
    assert lr == MIN


def test_cosine_warmup_monotone():
    lrs = [linear_warmup_cosine_decay(s, WARMUP, TOTAL, PEAK, MIN)
           for s in range(0, WARMUP + 1)]
    for a, b in zip(lrs, lrs[1:]):
        assert b >= a - 1e-14, f"warmup not monotone: {a} -> {b}"


def test_cosine_decay_monotone():
    lrs = [linear_warmup_cosine_decay(s, WARMUP, TOTAL, PEAK, MIN)
           for s in range(WARMUP, TOTAL + 1)]
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-14, f"cosine decay not monotone: {a} -> {b}"


# ── linear schedule ───────────────────────────────────────────────────────────

def test_linear_step_zero():
    assert linear_warmup_linear_decay(0, WARMUP, TOTAL, PEAK, MIN) == 0.0


def test_linear_at_warmup_end():
    lr = linear_warmup_linear_decay(WARMUP, WARMUP, TOTAL, PEAK, MIN)
    assert abs(lr - PEAK) < 1e-12


def test_linear_at_total_steps():
    lr = linear_warmup_linear_decay(TOTAL, WARMUP, TOTAL, PEAK, MIN)
    assert abs(lr - MIN) < 1e-12


def test_linear_beyond_total_steps():
    lr = linear_warmup_linear_decay(TOTAL + 500, WARMUP, TOTAL, PEAK, MIN)
    assert lr == MIN


def test_linear_warmup_monotone():
    lrs = [linear_warmup_linear_decay(s, WARMUP, TOTAL, PEAK, MIN)
           for s in range(0, WARMUP + 1)]
    for a, b in zip(lrs, lrs[1:]):
        assert b >= a - 1e-14, f"warmup not monotone: {a} -> {b}"


def test_linear_decay_monotone():
    lrs = [linear_warmup_linear_decay(s, WARMUP, TOTAL, PEAK, MIN)
           for s in range(WARMUP, TOTAL + 1)]
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-14, f"linear decay not monotone: {a} -> {b}"

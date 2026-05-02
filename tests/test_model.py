"""Forward-pass shape and stability tests."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from picochem.model import init_params, model_forward


@pytest.fixture
def small_config():
    return {
        'src_vocab': 50, 'tgt_vocab': 100,
        'd_model': 32, 'n_heads': 2, 'd_ff': 128,
        'n_enc_layers': 2, 'n_dec_layers': 2,
        'max_src_len': 50, 'max_tgt_len': 80,
    }


def test_forward_shape(small_config):
    rng = np.random.default_rng(0)
    params = init_params(small_config, rng)

    B, S, T = 3, 10, 15
    src_ids = rng.integers(0, small_config['src_vocab'], size=(B, S))
    tgt_ids = rng.integers(0, small_config['tgt_vocab'], size=(B, T))
    src_mask = np.ones((B, S), dtype=np.float64)
    tgt_mask = np.ones((B, T), dtype=np.float64)

    logits, _ = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, small_config)

    assert logits.shape == (B, T, small_config['tgt_vocab'])
    assert not np.isnan(logits).any()
    assert not np.isinf(logits).any()


def test_forward_loss_at_init(small_config):
    rng = np.random.default_rng(1)
    params = init_params(small_config, rng)

    B, S, T = 4, 12, 18
    src_ids = rng.integers(0, small_config['src_vocab'], size=(B, S))
    tgt_ids = rng.integers(0, small_config['tgt_vocab'], size=(B, T))
    src_mask = np.ones((B, S), dtype=np.float64)
    tgt_mask = np.ones((B, T), dtype=np.float64)

    logits, _ = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, small_config)

    log_z = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) + logits.max(-1, keepdims=True)
    log_probs = logits - log_z
    target_lp = log_probs.reshape(B*T, -1)[np.arange(B*T), tgt_ids.reshape(-1)]
    loss = -target_lp.mean()

    expected = np.log(small_config['tgt_vocab'])
    assert abs(loss - expected) < 1.5


def test_padding_doesnt_break(small_config):
    rng = np.random.default_rng(2)
    params = init_params(small_config, rng)

    B, S, T = 2, 10, 15
    src_ids = rng.integers(0, small_config['src_vocab'], size=(B, S))
    tgt_ids = rng.integers(0, small_config['tgt_vocab'], size=(B, T))

    src_mask = np.zeros((B, S), dtype=np.float64)
    src_mask[0, :5] = 1.0
    src_mask[1, :8] = 1.0

    tgt_mask = np.zeros((B, T), dtype=np.float64)
    tgt_mask[0, :8] = 1.0
    tgt_mask[1, :12] = 1.0

    logits, _ = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, small_config)

    assert logits.shape == (B, T, small_config['tgt_vocab'])
    assert not np.isnan(logits).any()
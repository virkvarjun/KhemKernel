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

import numpy as np
from picochem.model import init_params, model_forward, model_backward, compute_loss, loss_backward


def test_model_backward_via_gradient_check():
    """End-to-end gradient check: numerical vs. analytical for a few parameters."""
    rng = np.random.default_rng(0)
    config = {
        'src_vocab': 20, 'tgt_vocab': 30,
        'd_model': 16, 'n_heads': 2, 'd_ff': 32,
        'n_enc_layers': 1, 'n_dec_layers': 1,
        'max_src_len': 10, 'max_tgt_len': 12,
    }
    params = init_params(config, rng)

    B, S, T = 2, 5, 6
    src_ids = rng.integers(0, config['src_vocab'], size=(B, S))
    tgt_ids = rng.integers(0, config['tgt_vocab'], size=(B, T))
    targets = rng.integers(0, config['tgt_vocab'], size=(B, T))
    src_mask = np.ones((B, S), dtype=np.float64)
    tgt_mask = np.ones((B, T), dtype=np.float64)

    def compute_loss_for_params(p):
        logits, _ = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, p, config)
        loss, _ = compute_loss(logits, targets)
        return loss

    # Run forward and backward to get analytical grads
    logits, fwd_cache = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, config)
    loss, loss_cache = compute_loss(logits, targets)
    grad_logits = loss_backward(1.0, loss_cache)
    grads = model_backward(grad_logits, fwd_cache, params, config)

    # Spot-check a few parameters with finite differences
    eps = 1e-5

    # Check final_ln_gamma
    for idx in [(0,), (5,), (15,)]:
        original = params['final_ln_gamma'][idx]
        params['final_ln_gamma'][idx] = original + eps
        loss_plus = compute_loss_for_params(params)
        params['final_ln_gamma'][idx] = original - eps
        loss_minus = compute_loss_for_params(params)
        params['final_ln_gamma'][idx] = original
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical = grads['final_ln_gamma'][idx]
        assert abs(num_grad - analytical) < 1e-4, \
            f"final_ln_gamma{idx}: num={num_grad:.6f} analytical={analytical:.6f}"

    # Check tgt_token_embed (tests both contributions)
    for idx in [(0, 0), (5, 3), (10, 7)]:
        original = params['tgt_token_embed'][idx]
        params['tgt_token_embed'][idx] = original + eps
        loss_plus = compute_loss_for_params(params)
        params['tgt_token_embed'][idx] = original - eps
        loss_minus = compute_loss_for_params(params)
        params['tgt_token_embed'][idx] = original
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical = grads['tgt_token_embed'][idx]
        assert abs(num_grad - analytical) < 1e-4, \
            f"tgt_token_embed{idx}: num={num_grad:.6f} analytical={analytical:.6f}"

    # Check src_pos_embed (tests batch-summing)
    for idx in [(0, 0), (3, 5)]:
        original = params['src_pos_embed'][idx]
        params['src_pos_embed'][idx] = original + eps
        loss_plus = compute_loss_for_params(params)
        params['src_pos_embed'][idx] = original - eps
        loss_minus = compute_loss_for_params(params)
        params['src_pos_embed'][idx] = original
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical = grads['src_pos_embed'][idx]
        assert abs(num_grad - analytical) < 1e-4, \
            f"src_pos_embed{idx}: num={num_grad:.6f} analytical={analytical:.6f}"

    # Check a weight inside an encoder block
    for idx in [(0, 0), (5, 8)]:
        original = params['encoder_blocks'][0]['attn_Wq'][idx]
        params['encoder_blocks'][0]['attn_Wq'][idx] = original + eps
        loss_plus = compute_loss_for_params(params)
        params['encoder_blocks'][0]['attn_Wq'][idx] = original - eps
        loss_minus = compute_loss_for_params(params)
        params['encoder_blocks'][0]['attn_Wq'][idx] = original
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical = grads['encoder_blocks'][0]['attn_Wq'][idx]
        assert abs(num_grad - analytical) < 1e-4, \
            f"enc_block_0.attn_Wq{idx}: num={num_grad:.6f} analytical={analytical:.6f}"

    # Check a weight inside a decoder block (tests cross-attention path)
    for idx in [(0, 0), (8, 5)]:
        original = params['decoder_blocks'][0]['cross_Wv'][idx]
        params['decoder_blocks'][0]['cross_Wv'][idx] = original + eps
        loss_plus = compute_loss_for_params(params)
        params['decoder_blocks'][0]['cross_Wv'][idx] = original - eps
        loss_minus = compute_loss_for_params(params)
        params['decoder_blocks'][0]['cross_Wv'][idx] = original
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical = grads['decoder_blocks'][0]['cross_Wv'][idx]
        assert abs(num_grad - analytical) < 1e-4, \
            f"dec_block_0.cross_Wv{idx}: num={num_grad:.6f} analytical={analytical:.6f}"
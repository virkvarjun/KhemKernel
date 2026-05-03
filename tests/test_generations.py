"""Test greedy decoding."""

import sys
sys.path.insert(0, ".")

import numpy as np

from picochem.model import init_params, greedy_decode


def test_greedy_decode_runs():
    rng = np.random.default_rng(0)
    config = {
        'src_vocab': 30, 'tgt_vocab': 50,
        'd_model': 16, 'n_heads': 2, 'd_ff': 32,
        'n_enc_layers': 1, 'n_dec_layers': 1,
        'max_src_len': 20, 'max_tgt_len': 30,
    }
    params = init_params(config, rng)

    src_ids = rng.integers(2, 30, size=(1, 10)).astype(np.int32)
    src_mask = np.ones((1, 10), dtype=np.float64)

    tokens = greedy_decode(
        src_ids, src_mask, params, config,
        start_token=0, end_token=1, pad_token=2,
        max_length=20,
    )

    assert tokens[0] == 0  # starts with <start>
    assert len(tokens) <= 20
    # Either ended with <end> or hit max length
    assert tokens[-1] == 1 or len(tokens) == 20
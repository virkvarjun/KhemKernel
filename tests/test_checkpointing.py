"""Round-trip test for save_checkpoint / load_checkpoint."""
import os
import tempfile

import numpy as np
import pytest

from picochem.checkpointing import save_checkpoint, load_checkpoint


@pytest.fixture
def small_state():
    rng = np.random.default_rng(0)

    params = {
        "embed": rng.standard_normal((3, 4)),
        "blocks": [
            {
                "W": rng.standard_normal((4, 4)),
                "b": rng.standard_normal((4,)),
            },
            {
                "W": rng.standard_normal((4, 4)),
                "b": rng.standard_normal((4,)),
            },
        ],
        "final": rng.standard_normal((4,)),
    }

    def _zeros_like(p):
        if isinstance(p, np.ndarray):
            return {"m": np.zeros_like(p), "v": np.zeros_like(p)}
        if isinstance(p, dict):
            return {k: _zeros_like(v) for k, v in p.items()}
        if isinstance(p, list):
            return [_zeros_like(v) for v in p]

    optimizer_state = _zeros_like(params)
    # Give the optimizer state some non-zero values
    optimizer_state["embed"]["m"][:] = 0.1
    optimizer_state["blocks"][1]["W"]["v"][:] = 0.5

    config = {"d_model": 4, "n_heads": 2, "src_vocab": 50, "tgt_vocab": 100}
    step = 137

    return params, optimizer_state, step, config


def test_roundtrip(small_state):
    params, state, step, config = small_state

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(path, params, state, step, config)
        p2, s2, step2, cfg2 = load_checkpoint(path)

        assert step2 == step
        assert cfg2 == config

        # Check params (top-level array and nested list items)
        np.testing.assert_array_equal(p2["embed"],           params["embed"])
        np.testing.assert_array_equal(p2["final"],           params["final"])
        np.testing.assert_array_equal(p2["blocks"][0]["W"],  params["blocks"][0]["W"])
        np.testing.assert_array_equal(p2["blocks"][1]["b"],  params["blocks"][1]["b"])

        # Check optimizer state (leaf dicts with m/v)
        np.testing.assert_array_equal(s2["embed"]["m"],          state["embed"]["m"])
        np.testing.assert_array_equal(s2["blocks"][1]["W"]["v"], state["blocks"][1]["W"]["v"])

    finally:
        os.unlink(path)


def test_structure_preserved(small_state):
    params, state, step, config = small_state

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(path, params, state, step, config)
        p2, s2, _, _ = load_checkpoint(path)

        assert isinstance(p2["blocks"], list)
        assert len(p2["blocks"]) == 2
        assert isinstance(p2["blocks"][0], dict)

        assert isinstance(s2["blocks"], list)
        assert isinstance(s2["blocks"][0]["W"], dict)
        assert set(s2["blocks"][0]["W"].keys()) == {"m", "v"}

    finally:
        os.unlink(path)

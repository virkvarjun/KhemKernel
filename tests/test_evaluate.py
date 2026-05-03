"""Tests for picochem.evaluate."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from picochem.evaluate import parse_trace, name_to_smiles, evaluate_model, OPSIN_AVAILABLE


# ── parse_trace ───────────────────────────────────────────────────────────────

def test_parse_trace_extracts_name():
    trace = (
        "<parent>benzene</parent>"
        "<groups>none</groups>"
        "<atoms>6</atoms>"
        "<rings>1</rings>"
        "<name>benzene</name>"
    )
    assert parse_trace(trace) == "benzene"


def test_parse_trace_multiword_name():
    trace = "<parent>chain_C2</parent><groups>carboxylic_acid</groups><atoms>2</atoms><rings>0</rings><name>acetic acid</name>"
    assert parse_trace(trace) == "acetic acid"


def test_parse_trace_no_tag():
    assert parse_trace("no name tag here") is None


def test_parse_trace_empty_tag():
    assert parse_trace("<name></name>") is None


def test_parse_trace_whitespace_only():
    assert parse_trace("<name>   </name>") is None


def test_parse_trace_none_input():
    assert parse_trace(None) is None


def test_parse_trace_empty_string():
    assert parse_trace("") is None


# ── name_to_smiles ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN requires Java")
def test_name_to_smiles_benzene():
    smi = name_to_smiles("benzene")
    assert smi is not None
    # Canonical SMILES for benzene
    assert "c1ccccc1" in smi or "C1=CC=CC=C1" in smi or smi == "c1ccccc1"


@pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN requires Java")
def test_name_to_smiles_acetic_acid():
    smi = name_to_smiles("acetic acid")
    assert smi is not None


def test_name_to_smiles_nonsense():
    # Must return None whether OPSIN is available or not
    result = name_to_smiles("xyzfoobar99999nonsense")
    assert result is None


def test_name_to_smiles_none_input():
    assert name_to_smiles(None) is None


def test_name_to_smiles_empty():
    assert name_to_smiles("") is None


# ── evaluate_model ────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_setup():
    """Minimal model config + random params + synthetic val pairs."""
    from picochem.model import init_params
    from picochem.data import encode_smiles, encode_iupac

    config = {
        "src_vocab": 30, "tgt_vocab": 50,
        "d_model": 16, "n_heads": 2, "d_ff": 32,
        "n_enc_layers": 1, "n_dec_layers": 1,
        "max_src_len": 20, "max_tgt_len": 30,
    }
    rng = np.random.default_rng(7)
    params = init_params(config, rng)

    # Build tiny itos dicts
    smiles_itos = {i: str(i) for i in range(30)}
    iupac_itos  = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
    for i in range(4, 50):
        iupac_itos[i] = f"tok{i}"

    # Create synthetic pairs: random int arrays
    val_pairs = [
        (
            rng.integers(4, 30, size=5, dtype=np.int32),
            np.array([1, 5, 6, 7, 2], dtype=np.int32),   # <start>…<end>
        )
        for _ in range(10)
    ]

    return params, config, val_pairs, smiles_itos, iupac_itos


def test_evaluate_model_runs(tiny_setup):
    params, config, val_pairs, smiles_itos, iupac_itos = tiny_setup
    result = evaluate_model(
        params, config, val_pairs, smiles_itos, iupac_itos,
        n_samples=5, max_length=20,
    )
    assert "n_evaluated" in result
    assert result["n_evaluated"] == 5
    assert 0.0 <= result["trace_validity_rate"]  <= 1.0
    assert 0.0 <= result["opsin_parse_rate"]     <= 1.0
    assert 0.0 <= result["structure_match_rate"] <= 1.0
    assert len(result["samples"]) == 5


def test_evaluate_model_sample_keys(tiny_setup):
    params, config, val_pairs, smiles_itos, iupac_itos = tiny_setup
    result = evaluate_model(
        params, config, val_pairs, smiles_itos, iupac_itos,
        n_samples=3, max_length=15,
    )
    for sample in result["samples"]:
        assert "predicted_name"   in sample
        assert "predicted_smiles" in sample
        assert "target_smiles"    in sample
        assert "match"            in sample

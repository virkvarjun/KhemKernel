"""Evaluation pipeline: greedy decoding → trace parsing → OPSIN → structure match."""
import re
import warnings

import numpy as np

from picochem.data import decode_smiles, decode_iupac
from picochem.model import model_forward, greedy_decode

# ---------------------------------------------------------------------------
# OPSIN integration
# ---------------------------------------------------------------------------

OPSIN_AVAILABLE = False
_OPSIN_BACKEND  = "none"

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import py2opsin as _py2opsin

    def _opsin_convert(name):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _py2opsin.py2opsin(name)
            return result if result and result.strip() else None
        except Exception:
            return None

    # Probe: if Java is absent py2opsin will still import but fail at call time.
    _probe = _opsin_convert("benzene")
    if _probe is not None:
        OPSIN_AVAILABLE = True
        _OPSIN_BACKEND  = "py2opsin"

except Exception:
    pass

if not OPSIN_AVAILABLE:
    import os as _os
    import subprocess as _subprocess
    _OPSIN_JAR = _os.environ.get("OPSIN_JAR", "opsin.jar")
    if _os.path.exists(_OPSIN_JAR):
        def _opsin_convert(name):
            try:
                r = _subprocess.run(
                    ["java", "-jar", _OPSIN_JAR, "-osmi", name],
                    capture_output=True, text=True, timeout=10,
                )
                smi = r.stdout.strip()
                return smi if smi else None
            except Exception:
                return None
        OPSIN_AVAILABLE = True
        _OPSIN_BACKEND  = "opsin-jar"
    else:
        def _opsin_convert(name):
            return None

# ---------------------------------------------------------------------------
# RDKit canonicalization
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem as _Chem
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog("rdApp.*")

    def _canonicalize(smi):
        if not smi:
            return None
        mol = _Chem.MolFromSmiles(smi)
        return _Chem.MolToSmiles(mol) if mol else None
except ImportError:
    def _canonicalize(smi):
        return smi if smi else None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)


def parse_trace(trace_text):
    """Extract the IUPAC name from inside the <name>…</name> tag.

    Returns
    -------
    str or None — stripped name string, or None if tag absent or empty.
    """
    if not trace_text:
        return None
    m = _NAME_RE.search(trace_text)
    if not m:
        return None
    name = m.group(1).strip()
    return name if name else None


def name_to_smiles(name):
    """Convert an IUPAC name to canonical SMILES via OPSIN + RDKit.

    Returns None if OPSIN is unavailable, the name fails to parse,
    or RDKit cannot canonicalize the result.
    """
    if not name:
        return None
    raw = _opsin_convert(name)
    if raw is None:
        return None
    return _canonicalize(raw)


def evaluate_model(params, config, val_pairs, smiles_itos, iupac_itos,
                   n_samples=500, max_length=None):
    """Generate traces for n_samples pairs and compute evaluation metrics.

    Parameters
    ----------
    params : dict
    config : dict
    val_pairs : list of (src_ids, tgt_ids)
    smiles_itos : dict[int, str]
    iupac_itos  : dict[int, str]
    n_samples : int
    max_length : int or None
        Max generation length; defaults to config['max_tgt_len'].

    Returns
    -------
    dict with keys:
        n_evaluated, trace_validity_rate, opsin_parse_rate,
        structure_match_rate, samples
    """
    if max_length is None:
        max_length = config["max_tgt_len"]

    # Build reverse vocab for start/end tokens
    iupac_stoi = {v: k for k, v in iupac_itos.items()}
    start_id   = iupac_stoi.get("<start>", 1)
    end_id     = iupac_stoi.get("<end>",   2)
    pad_id     = iupac_stoi.get("<pad>",   0)

    rng = np.random.default_rng(0)
    n   = min(n_samples, len(val_pairs))
    idx = rng.choice(len(val_pairs), size=n, replace=False)

    n_valid = 0
    n_opsin = 0
    n_match = 0
    samples = []

    for i in idx:
        src_ids, _ = val_pairs[i]
        src = src_ids[np.newaxis, :]
        src_mask = np.ones((1, len(src_ids)), dtype=np.float64)

        gen_ids = greedy_decode(
            src, src_mask, params, config,
            start_token=start_id, end_token=end_id, pad_token=pad_id,
            max_length=max_length,
        )
        gen_str = decode_iupac(np.array(gen_ids), iupac_itos)

        pred_name = parse_trace(gen_str)
        if pred_name is not None:
            n_valid += 1

        # Ground-truth SMILES: canonicalize source tokens
        ref_smiles_raw = decode_smiles(src_ids, smiles_itos)
        target_smiles  = _canonicalize(ref_smiles_raw)

        pred_smiles = None
        matched     = False
        if pred_name is not None:
            pred_smiles = name_to_smiles(pred_name)
            if pred_smiles is not None:
                n_opsin += 1
                if target_smiles is not None and pred_smiles == target_smiles:
                    n_match += 1
                    matched = True

        samples.append({
            "predicted_name":   pred_name,
            "predicted_smiles": pred_smiles,
            "target_smiles":    target_smiles,
            "match":            matched,
        })

    return {
        "n_evaluated":          n,
        "trace_validity_rate":  n_valid / n if n else 0.0,
        "opsin_parse_rate":     n_opsin / n if n else 0.0,
        "structure_match_rate": n_match / n if n else 0.0,
        "samples":              samples,
    }

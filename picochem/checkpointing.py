"""Save and load model checkpoints as compressed .npz files."""
import json
import numpy as np


# Separator used to encode nested key paths inside npz file entries.
# Must not appear in any parameter name or dict key.
_SEP = "__"


def _flatten(obj, prefix=""):
    """Recursively flatten a nested params/state structure to {dotted_key: ndarray}."""
    arrays = {}
    if isinstance(obj, np.ndarray):
        arrays[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{prefix}{_SEP}{k}" if prefix else str(k)
            arrays.update(_flatten(v, child))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child = f"{prefix}{_SEP}{i}" if prefix else str(i)
            arrays.update(_flatten(v, child))
    return arrays


def _unflatten(flat):
    """Reconstruct a nested structure from {sep-delimited_key: ndarray}.

    Dict keys that are all digits are converted to lists.
    """
    root = {}
    for key, val in flat.items():
        parts = key.split(_SEP)
        node = root
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = val

    def _convert(obj):
        if not isinstance(obj, dict):
            return obj
        converted = {k: _convert(v) for k, v in obj.items()}
        if converted and all(k.isdigit() for k in converted):
            return [converted[str(i)] for i in range(len(converted))]
        return converted

    return _convert(root)


def save_checkpoint(path, params, optimizer_state, step, config):
    """Save params, optimizer state, step, and config to a single .npz file.

    Arrays are stored with keys like
    ``params__encoder_blocks__0__attn_Wq`` and
    ``state__encoder_blocks__0__attn_Wq__m``.
    """
    flat_params = _flatten(params, "params")
    flat_state = _flatten(optimizer_state, "state")

    arrays = {}
    arrays.update(flat_params)
    arrays.update(flat_state)
    arrays["_meta_step"] = np.array(step, dtype=np.int64)
    arrays["_meta_config"] = np.frombuffer(
        json.dumps(config).encode("utf-8"), dtype=np.uint8
    )

    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns
    -------
    params : dict
    optimizer_state : dict
    step : int
    config : dict
    """
    npz = np.load(path, allow_pickle=False)

    step = int(npz["_meta_step"])
    config = json.loads(bytes(npz["_meta_config"].tolist()).decode("utf-8"))

    flat_params = {
        k[len("params" + _SEP):]: npz[k]
        for k in npz.files
        if k.startswith("params" + _SEP)
    }
    flat_state = {
        k[len("state" + _SEP):]: npz[k]
        for k in npz.files
        if k.startswith("state" + _SEP)
    }

    params = _unflatten(flat_params)
    optimizer_state = _unflatten(flat_state)

    return params, optimizer_state, step, config

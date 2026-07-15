"""Input generation for the precision-matching harness (simple version).

Three input groups:
  1. uniform float  -- default, identical behaviour to the current harness
  2. uniform int    -- token ids etc.; activated by declaring an integer dtype
  3. real images    -- detection models; see real_images.py

plus a "rect_mask" kind for inpainting hole masks (LaMa).

Spec format is the existing model_config.yaml one. Each input is either the
positional list

    [[shape], min, max, dtype, layout]                  # e.g. [[1,3,224,224], 0, 1, "float32", "BCHW"]
    [[shape], min, max, dtype, layout, kind]            # kind: "uniform" (default) | "rect_mask"

or an equivalent dict {"shape": ..., "min": ..., "max": ..., "dtype": ...,
"layout": ..., "kind": ...}.

Integer ranges are half-open [min, max): declare token ids as
[0, vocab_size).  A constant input (e.g. an all-ones attention_mask) is the
degenerate range [1, 2).

Reference and candidate must be fed the *same* arrays, so generation is fully
deterministic given (specs, seed): one np.random.default_rng(seed) is consumed
in spec order.
"""

from __future__ import annotations

import numpy as np

_KINDS = ("uniform", "rect_mask")


def _parse_spec(spec):
    if isinstance(spec, dict):
        shape = spec["shape"]
        lo, hi = spec["min"], spec["max"]
        dtype = spec["dtype"]
        kind = spec.get("kind", "uniform")
    else:
        if len(spec) not in (5, 6):
            raise ValueError(f"input spec must have 5 or 6 fields, got {spec!r}")
        shape, lo, hi, dtype = spec[0], spec[1], spec[2], spec[3]
        kind = spec[5] if len(spec) == 6 else "uniform"
    if kind not in _KINDS:
        raise ValueError(f"unknown input kind {kind!r}, expected one of {_KINDS}")
    shape = tuple(int(s) for s in shape)
    return shape, lo, hi, np.dtype(dtype), kind


def _rect_mask(shape, dtype, rng):
    """Binary hole mask for inpainting models: 1-3 random rectangles set to 1,
    each covering 10-40% of the corresponding spatial side. The last two axes
    are treated as (H, W)."""
    if len(shape) < 2:
        raise ValueError(f"rect_mask needs at least 2 dims (H, W), got {shape}")
    h, w = shape[-2], shape[-1]
    mask = np.zeros(shape, dtype=dtype)
    for _ in range(int(rng.integers(1, 4))):
        rh = max(1, int(h * rng.uniform(0.1, 0.4)))
        rw = max(1, int(w * rng.uniform(0.1, 0.4)))
        y0 = int(rng.integers(0, max(1, h - rh + 1)))
        x0 = int(rng.integers(0, max(1, w - rw + 1)))
        mask[..., y0:y0 + rh, x0:x0 + rw] = 1
    return mask


def generate_input(spec, rng):
    """Generate one input array from a spec, consuming the given rng."""
    shape, lo, hi, dtype, kind = _parse_spec(spec)
    if kind == "rect_mask":
        return _rect_mask(shape, dtype, rng)
    if np.issubdtype(dtype, np.integer) or dtype == np.dtype(bool):
        lo_i, hi_i = int(lo), int(hi)
        if hi_i <= lo_i:
            raise ValueError(f"integer range [{lo_i}, {hi_i}) is empty")
        if dtype == np.dtype(bool):  # [0, 2) -> random False/True
            return rng.integers(lo_i, hi_i, size=shape).astype(bool)
        return rng.integers(lo_i, hi_i, size=shape, dtype=dtype)
    return rng.uniform(float(lo), float(hi), size=shape).astype(dtype)


def generate_inputs(specs, seed):
    """Generate the full input list for one trial.

    Deterministic: same (specs, seed) always yields identical arrays, so the
    reference and the candidate can be run in separate processes.
    """
    rng = np.random.default_rng(seed)
    return [generate_input(spec, rng) for spec in specs]

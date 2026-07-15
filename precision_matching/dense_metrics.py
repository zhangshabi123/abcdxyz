"""The four dense metrics, exactly as defined in the precision-matching doc.

For a reference output `a` and converted output `b` of the same shape:

    max_atol        max |a - b|
    max_rtol        max( |a - b| / (|a| + 1e-12) )
    nrmse           ||a - b||_2 / max(||a||_2, atol * sqrt(N))
    mismatch_ratio  mean( |a - b| > atol + rtol * |a| )

Record-only diagnostics; no pass/fail. All arithmetic in float64.
Non-finite values (NaN/Inf) surface loudly: they propagate into max_atol /
max_rtol, count as mismatches in mismatch_ratio, and the *_nonfinite counts
say on which side they appeared.
"""

from __future__ import annotations

import math

import numpy as np

METRIC_KEYS = ("max_atol", "max_rtol", "nrmse", "mismatch_ratio")


def dense_metrics(ref, cand, atol, rtol):
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(cand, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: ref {a.shape} vs cand {b.shape}")
    if a.size == 0:
        raise ValueError("cannot compare empty tensors")
    diff = np.abs(a - b)
    n = a.size
    diff_norm = float(np.linalg.norm((a - b).ravel()))
    denom = max(float(np.linalg.norm(a.ravel())), float(atol) * math.sqrt(n))
    if denom > 0:
        nrmse = diff_norm / denom
    else:  # atol=0 with an all-zero reference: identical -> 0, else loud
        nrmse = 0.0 if diff_norm == 0.0 else float("inf")
    # NaN > x is always False, so non-finite diffs must be counted explicitly.
    violation = (diff > atol + rtol * np.abs(a)) | ~np.isfinite(diff)
    return {
        "max_atol": float(np.max(diff)),
        "max_rtol": float(np.max(diff / (np.abs(a) + 1e-12))),
        "nrmse": float(nrmse),
        "mismatch_ratio": float(np.mean(violation)),
        "ref_nonfinite": int(n - np.isfinite(a).sum()),
        "cand_nonfinite": int(n - np.isfinite(b).sum()),
    }


def aggregate_worst(per_tensor_metrics):
    """Worst case across output tensors (the doc's aggregation convention)."""
    per_tensor_metrics = list(per_tensor_metrics)
    if not per_tensor_metrics:
        raise ValueError("nothing to aggregate")
    # np.max propagates NaN regardless of order; builtin max() would silently
    # drop a NaN that is not in the first position.
    out = {k: float(np.max([m[k] for m in per_tensor_metrics])) for k in METRIC_KEYS}
    out["ref_nonfinite"] = sum(m["ref_nonfinite"] for m in per_tensor_metrics)
    out["cand_nonfinite"] = sum(m["cand_nonfinite"] for m in per_tensor_metrics)
    return out

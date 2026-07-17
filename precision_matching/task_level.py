"""Task-level verification glue: "did the answer change?"

Value-level metrics say how far the numbers drifted; these record whether the
model's discrete answers changed. All record-only. Report task-level columns
SEPARATELY from value-level ones -- do not aggregate across the two.
"""

from __future__ import annotations

import numpy as np

from .decision_agreement import top1_agreement
from .detection_matching import match_detections


def detection_task_metrics(postprocess_fn, ref_args, cand_args,
                           pp_kwargs=None, match_kwargs=None):
    """Apply the SAME harness-side postprocess to both sides' raw outputs,
    then record matching metrics. ref_args / cand_args are tuples of the raw
    arrays for postprocess_fn (e.g. (logits, boxes) for the query-detector
    lens, (raw,) for the yolo lens); pp_kwargs are shared by construction --
    the only variable is the raw tensors.
    """
    pp_kwargs = pp_kwargs or {}
    ref_det = postprocess_fn(*ref_args, **pp_kwargs)
    cand_det = postprocess_fn(*cand_args, **pp_kwargs)
    out = match_detections(ref_det, cand_det, **(match_kwargs or {}))
    return out


def lm_task_metrics(ref_logits, cand_logits):
    """Next-token answer agreement on a real text, teacher-forced logits
    [B, L, V]: fraction of positions where both sides predict the same
    next token."""
    return {"token_top1_agreement": top1_agreement(ref_logits, cand_logits, axis=-1)}


def embedding_cosine(ref_hidden, cand_hidden, token_axis=1):
    """Mean-pooled cosine for encoder-only models (longformer/bigbird/deberta).
    NOT a task metric -- encoders have no discrete answer; this is just a
    readable second value-level number on real-distribution input.
    Hidden states [B, L, H]; pooled over token_axis. Both-zero vectors -> 1.0
    (identical), one-zero -> 0.0."""
    a = np.asarray(ref_hidden, dtype=np.float64)
    b = np.asarray(cand_hidden, dtype=np.float64)
    # Check BEFORE pooling: pooling erases the token axis, which would let a
    # length-mismatched (e.g. padded) candidate slip through silently.
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a.mean(axis=token_axis).ravel()
    b = b.mean(axis=token_axis).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

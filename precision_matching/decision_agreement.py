"""Optional per-family "fifth number": did the model's discrete decision change?

The four dense metrics measure how far the numbers drifted; these record
whether the *answer* changed. Read together: ugly numbers + identical answers
usually means benign discrete jitter; pretty numbers + flipped answers is
exactly the case a human should look at.

    classification  [B, C]        top1_agreement(ref, cand)            # axis=-1
    semantic seg    [B, C, H, W]  top1_agreement(ref, cand, axis=1)
    LM logits       [B, L, V]     top1_agreement(ref, cand)            # axis=-1
    binary masks    (birefnet/clipseg/sam2/mobilesam)  binary_mask_iou
"""

from __future__ import annotations

import numpy as np


def top1_agreement(ref_logits, cand_logits, axis=-1):
    """Fraction of positions whose argmax along `axis` is identical."""
    a = np.asarray(ref_logits)
    b = np.asarray(cand_logits)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.argmax(a, axis=axis) == np.argmax(b, axis=axis)))


def binary_mask_iou(ref, cand, threshold=0.5):
    """IoU of the two masks after thresholding. Both empty -> 1.0 (identical)."""
    a = np.asarray(ref) > threshold
    b = np.asarray(cand) > threshold
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)

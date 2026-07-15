"""Matching-based comparison for variable-length detection outputs.

Mask R-CNN / Keypoint R-CNN return list[dict] with a data-dependent number of
detections -- element-wise metrics cannot even align "51 boxes vs 48 boxes".
Here we first MATCH detections between reference and candidate, then compare
the matched pairs. Everything is recorded, nothing is judged, mirroring the
measure-don't-judge design of the dense metrics.

Matching rule (deterministic): reference boxes in descending score order; each
is matched to the not-yet-matched candidate box with the same label and the
highest IoU >= iou_threshold.

Input format per side: dict with
    boxes  [N, 4] xyxy   (required)
    scores [N]           (required)
    labels [N]           (required)
    masks  [N, 1, H, W] or [N, H, W], soft 0-1   (optional)
    keypoints [N, K, 3] (x, y, vis)              (optional)
i.e. exactly one element of torchvision's detection output, as numpy.
"""

from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU of two xyxy box sets: [N, 4] x [M, 4] -> [N, M]."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    # Degenerate (zero-area) boxes: identical coordinates count as IoU 1 so
    # that a bit-identical candidate still matches its reference.
    degenerate = union <= 0
    if degenerate.any():
        same = (np.abs(a[:, None, :] - b[None, :, :]) <= 1e-9).all(axis=-1)
        iou = np.where(degenerate & same, 1.0, iou)
    return iou


def _normalize(det):
    boxes = np.asarray(det["boxes"], dtype=np.float64).reshape(-1, 4)
    n = boxes.shape[0]
    scores = np.asarray(det["scores"], dtype=np.float64).reshape(-1)
    labels = np.asarray(det["labels"]).reshape(-1)
    if scores.shape[0] != n or labels.shape[0] != n:
        raise ValueError(f"inconsistent detection fields: {n} boxes, "
                         f"{scores.shape[0]} scores, {labels.shape[0]} labels")
    masks = det.get("masks")
    if masks is not None:
        masks = np.asarray(masks, dtype=np.float64)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
    kpts = det.get("keypoints")
    if kpts is not None:
        kpts = np.asarray(kpts, dtype=np.float64)
    return boxes, scores, labels, masks, kpts


def _stat(values, fn):
    values = list(values)
    return float(fn(values)) if values else None


def match_detections(ref, cand, iou_threshold=0.5, mask_threshold=0.5):
    """Match candidate detections to reference detections and record
    per-image diagnostics (all JSON-safe; None where undefined)."""
    r_boxes, r_scores, r_labels, r_masks, r_kpts = _normalize(ref)
    c_boxes, c_scores, c_labels, c_masks, c_kpts = _normalize(cand)
    n_ref, n_cand = r_boxes.shape[0], c_boxes.shape[0]

    iou = iou_matrix(r_boxes, c_boxes)
    cand_taken = np.zeros(n_cand, dtype=bool)
    pairs = []  # (ref_idx, cand_idx, iou)
    for i in np.argsort(-r_scores, kind="stable"):
        eligible = (~cand_taken) & (c_labels == r_labels[i]) & (iou[i] >= iou_threshold)
        if not eligible.any():
            continue
        j = int(np.argmax(np.where(eligible, iou[i], -1.0)))
        cand_taken[j] = True
        pairs.append((int(i), j, float(iou[i, j])))

    matched_ref = {i for i, _, _ in pairs}
    unmatched_ref_scores = [r_scores[i] for i in range(n_ref) if i not in matched_ref]
    unmatched_cand_scores = list(c_scores[~cand_taken])
    ious = [v for _, _, v in pairs]
    score_diffs = [abs(r_scores[i] - c_scores[j]) for i, j, _ in pairs]

    out = {
        "n_ref": n_ref,
        "n_cand": n_cand,
        "n_matched": len(pairs),
        "ref_match_rate": len(pairs) / n_ref if n_ref else None,
        "cand_match_rate": len(pairs) / n_cand if n_cand else None,
        "matched_iou_mean": _stat(ious, np.mean),
        "matched_iou_min": _stat(ious, np.min),
        "matched_score_absdiff_mean": _stat(score_diffs, np.mean),
        "matched_score_absdiff_max": _stat(score_diffs, np.max),
        # High unmatched score = a real disagreement; low = detection-threshold edge.
        "unmatched_ref_max_score": _stat(unmatched_ref_scores, np.max),
        "unmatched_cand_max_score": _stat(unmatched_cand_scores, np.max),
    }

    if r_masks is not None and c_masks is not None:
        mask_ious = []
        for i, j, _ in pairs:
            rm = r_masks[i] > mask_threshold
            cm = c_masks[j] > mask_threshold
            union = np.logical_or(rm, cm).sum()
            mask_ious.append(
                1.0 if union == 0 else np.logical_and(rm, cm).sum() / union
            )
        out["matched_mask_iou_mean"] = _stat(mask_ious, np.mean)
        out["matched_mask_iou_min"] = _stat(mask_ious, np.min)

    if r_kpts is not None and c_kpts is not None:
        l2s, vis_agree = [], []
        for i, j, _ in pairs:
            d = np.linalg.norm(r_kpts[i, :, :2] - c_kpts[j, :, :2], axis=-1)
            l2s.append(float(d.mean()))
            vis_agree.append(float(np.mean((r_kpts[i, :, 2] > 0) == (c_kpts[j, :, 2] > 0))))
        out["matched_kpt_l2_mean"] = _stat(l2s, np.mean)
        out["matched_kpt_l2_max"] = _stat(l2s, np.max)
        out["matched_kpt_vis_agreement"] = _stat(vis_agree, np.mean)

    return out


def _npz_path(path):
    # np.savez appends ".npz" to suffix-less paths but np.load does not;
    # normalize on both sides so save/load accept the same path string.
    path = str(path)
    return path if path.endswith(".npz") else path + ".npz"


def save_detections(path, det):
    """Save one detection dict as .npz (np.save cannot store dicts)."""
    np.savez(_npz_path(path), **{k: np.asarray(v) for k, v in det.items() if v is not None})


def load_detections(path):
    with np.load(_npz_path(path)) as z:
        return {k: z[k] for k in z.files}

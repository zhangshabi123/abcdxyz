"""Harness-side post-processing: raw detector tensors -> detection dicts.

Task-level verification for the non-R-CNN detectors compares POST-PROCESSED
detections. The one rule that makes this valid: the SAME function with the
SAME parameters runs on both the reference and the candidate raw outputs, so
the only variable is the raw tensors themselves. These lenses do not need to
bit-match each library's own postprocess -- they define the harness's view.

All pure numpy; everything returns {"boxes": [N,4] xyxy, "scores": [N],
"labels": [N]} ready for detection_matching.match_detections.
"""

from __future__ import annotations

import numpy as np

from .detection_matching import iou_matrix


def stable_sigmoid(x):
    """Overflow-safe sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def cxcywh_to_xyxy(boxes):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nms_numpy(boxes, scores, iou_threshold=0.5):
    """Indices kept by classic greedy NMS, in descending score order."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=int)


def postprocess_query_detector(class_logits, pred_boxes, image_size=(1.0, 1.0),
                               score_threshold=0.3, top_k=None):
    """Generic lens for DETR-family raw outputs (rtdetr / owlvit /
    groundingdino): per-query score = max sigmoid logit over the class/token
    axis, label = its argmax; boxes are cxcywh normalized to [0,1] (the
    DETR-family convention) and get scaled to image_size = (H, W).

    class_logits [B,Q,C] or [Q,C]; pred_boxes [B,Q,4] or [Q,4].
    """
    logits = np.asarray(class_logits, dtype=np.float64)
    boxes = np.asarray(pred_boxes, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[0]
    if boxes.ndim == 3:
        boxes = boxes[0]
    if logits.shape[0] != boxes.shape[0]:
        raise ValueError(f"query count mismatch: logits {logits.shape} vs boxes {boxes.shape}")

    probs = stable_sigmoid(logits)
    scores = probs.max(axis=-1)
    labels = probs.argmax(axis=-1)
    keep = np.flatnonzero(scores >= score_threshold)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    if top_k is not None:
        keep = keep[:top_k]
    h, w = image_size
    xyxy = cxcywh_to_xyxy(boxes[keep]) * np.array([w, h, w, h], dtype=np.float64)
    return {"boxes": xyxy, "scores": scores[keep], "labels": labels[keep].astype(np.int64)}


def postprocess_yolo_head(raw, score_threshold=0.25, iou_threshold=0.45):
    """Lens for the ultralytics-style raw head [B, 4+K, N] (or [4+K, N]):
    rows 0-3 are decoded cxcywh boxes in absolute pixels, rows 4.. are the K
    class scores (ALREADY sigmoided by the ultralytics inference head -- no
    sigmoid applied here). Per-anchor best class, threshold, then per-class NMS.
    """
    r = np.asarray(raw, dtype=np.float64)
    if r.ndim == 3:
        r = r[0]
    if r.shape[0] < 5:
        raise ValueError(f"expected [4+K, N] head, got shape {r.shape}")
    boxes = cxcywh_to_xyxy(r[:4].T)
    cls = r[4:].T  # [N, K]
    scores = cls.max(axis=1)
    labels = cls.argmax(axis=1)
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    final = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        final.extend(idx[nms_numpy(boxes[idx], scores[idx], iou_threshold)])
    final = np.asarray(final, dtype=int)
    final = final[np.argsort(-scores[final], kind="stable")]
    return {"boxes": boxes[final], "scores": scores[final], "labels": labels[final].astype(np.int64)}


def unpack_padded_detections(arr, score_threshold=0.0):
    """Lens for already-postprocessed fixed-row outputs like EfficientDet's
    DetBenchPredict [B, 100, 6] = (xmin, ymin, xmax, ymax, score, class),
    zero-padded: strip padding / sub-threshold rows into a detection dict.

    Note: effdet's class column is the 1-based COCO category_id (background=0,
    same value as padding rows -- hence score>0 to tell them apart). Fine for
    ref-vs-candidate comparison under this same lens, but do NOT index the
    0-based COCO_CLASS_NAMES with it."""
    r = np.asarray(arr, dtype=np.float64)
    if r.ndim == 3:
        r = r[0]
    if r.ndim != 2 or r.shape[1] != 6:
        raise ValueError(f"expected [N, 6] detections, got shape {r.shape}")
    keep = r[:, 4] > max(0.0, score_threshold)
    r = r[keep]
    return {"boxes": r[:, :4], "scores": r[:, 4], "labels": r[:, 5].astype(np.int64)}

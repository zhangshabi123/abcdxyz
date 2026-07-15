import os
import tempfile

import numpy as np

from precision_matching.detection_matching import (
    iou_matrix,
    load_detections,
    match_detections,
    save_detections,
)


def det(boxes, scores, labels, masks=None, keypoints=None):
    d = {
        "boxes": np.asarray(boxes, dtype=np.float64),
        "scores": np.asarray(scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    if masks is not None:
        d["masks"] = np.asarray(masks, dtype=np.float64)
    if keypoints is not None:
        d["keypoints"] = np.asarray(keypoints, dtype=np.float64)
    return d


THREE = det(
    boxes=[[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 70]],
    scores=[0.9, 0.8, 0.3],
    labels=[1, 2, 1],
)


def test_iou_matrix_basics():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    iou = iou_matrix(a, b)
    assert abs(iou[0, 0] - 1.0) < 1e-12
    assert abs(iou[0, 1] - (50 / 150)) < 1e-12
    assert iou[0, 2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
    # degenerate zero-area boxes: no div-by-zero; identical coords -> IoU 1
    z = np.array([[5, 5, 5, 5]], dtype=float)
    assert iou_matrix(z, z)[0, 0] == 1.0
    z2 = np.array([[7, 7, 7, 7]], dtype=float)
    assert iou_matrix(z, z2)[0, 0] == 0.0


def test_identical_full_match():
    m = match_detections(THREE, THREE)
    assert m["n_ref"] == m["n_cand"] == m["n_matched"] == 3
    assert m["ref_match_rate"] == 1.0 and m["cand_match_rate"] == 1.0
    assert m["matched_iou_min"] == 1.0
    assert m["matched_score_absdiff_max"] == 0.0
    assert m["unmatched_ref_max_score"] is None


def test_permuted_candidate_order_is_irrelevant():
    perm = [2, 0, 1]
    cand = det(
        boxes=np.asarray(THREE["boxes"])[perm],
        scores=np.asarray(THREE["scores"])[perm],
        labels=np.asarray(THREE["labels"])[perm],
    )
    m = match_detections(THREE, cand)
    assert m["n_matched"] == 3 and m["matched_iou_min"] == 1.0


def test_shifted_box_and_score_drift_recorded():
    cand = det(
        boxes=[[1, 0, 11, 10], [20, 20, 40, 40], [50, 50, 60, 70]],
        scores=[0.85, 0.8, 0.3],
        labels=[1, 2, 1],
    )
    m = match_detections(THREE, cand)
    assert m["n_matched"] == 3
    assert m["matched_iou_min"] < 1.0
    assert abs(m["matched_score_absdiff_max"] - 0.05) < 1e-12


def test_label_flip_makes_pair_unmatched():
    cand = det(THREE["boxes"], THREE["scores"], [1, 1, 1])  # label 2 -> 1
    m = match_detections(THREE, cand)
    assert m["n_matched"] == 2
    assert abs(m["unmatched_ref_max_score"] - 0.8) < 1e-12
    assert abs(m["unmatched_cand_max_score"] - 0.8) < 1e-12


def test_empty_cases():
    empty = det(np.zeros((0, 4)), [], [])
    m = match_detections(empty, empty)
    assert m["n_matched"] == 0
    assert m["ref_match_rate"] is None and m["cand_match_rate"] is None
    m = match_detections(empty, THREE)
    assert m["ref_match_rate"] is None and m["cand_match_rate"] == 0.0
    assert abs(m["unmatched_cand_max_score"] - 0.9) < 1e-12
    # empty boxes saved as shape (0,) still normalize
    m = match_detections(det([], [], []), THREE)
    assert m["n_ref"] == 0


def test_extra_candidate_box():
    cand = det(
        boxes=[[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 70], [80, 80, 90, 90]],
        scores=[0.9, 0.8, 0.3, 0.1],
        labels=[1, 2, 1, 1],
    )
    m = match_detections(THREE, cand)
    assert m["ref_match_rate"] == 1.0
    assert abs(m["cand_match_rate"] - 0.75) < 1e-12
    assert abs(m["unmatched_cand_max_score"] - 0.1) < 1e-12  # low -> threshold edge


def test_duplicate_candidates_greedy_takes_best_then_next():
    # two near-identical cand boxes compete for one ref box
    ref = det([[0, 0, 10, 10]], [0.9], [1])
    cand = det([[0, 0, 10, 10], [1, 0, 11, 10]], [0.9, 0.5], [1, 1])
    m = match_detections(ref, cand)
    assert m["n_matched"] == 1 and m["matched_iou_min"] == 1.0
    assert abs(m["unmatched_cand_max_score"] - 0.5) < 1e-12


def test_masks_metrics():
    mask_a = np.zeros((1, 1, 16, 16))
    mask_a[..., :8, :8] = 1.0
    ref = det([[0, 0, 8, 8]], [0.9], [1], masks=mask_a)
    m = match_detections(ref, ref)
    assert m["matched_mask_iou_min"] == 1.0
    mask_b = np.zeros((1, 1, 16, 16))
    mask_b[..., 8:, 8:] = 1.0  # disjoint
    cand = det([[0, 0, 8, 8]], [0.9], [1], masks=mask_b)
    m = match_detections(ref, cand)
    assert m["matched_mask_iou_min"] == 0.0


def test_keypoints_metrics():
    k = np.zeros((1, 17, 3))
    k[0, :, 0] = np.arange(17)
    k[0, :, 2] = 1
    ref = det([[0, 0, 20, 20]], [0.9], [1], keypoints=k)
    shifted = k.copy()
    shifted[0, :, 0] += 3.0
    shifted[0, :, 1] += 4.0
    shifted[0, 0, 2] = 0  # one visibility flip
    cand = det([[0, 0, 20, 20]], [0.9], [1], keypoints=shifted)
    m = match_detections(ref, cand)
    assert abs(m["matched_kpt_l2_mean"] - 5.0) < 1e-9  # 3-4-5 triangle
    assert abs(m["matched_kpt_vis_agreement"] - 16 / 17) < 1e-9


def test_identical_degenerate_box_still_matches():
    deg = det([[5, 5, 5, 5]], [0.9], [1])
    m = match_detections(deg, deg)
    assert m["n_matched"] == 1 and m["ref_match_rate"] == 1.0
    other = det([[7, 7, 7, 7]], [0.9], [1])  # different degenerate box
    assert match_detections(deg, other)["n_matched"] == 0


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "det.npz")
        save_detections(path, THREE)
        loaded = load_detections(path)
        m = match_detections(THREE, loaded)
        assert m["n_matched"] == 3 and m["matched_iou_min"] == 1.0


def test_save_load_without_npz_suffix():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "det_trial_0")  # np.savez appends .npz itself
        save_detections(path, THREE)
        loaded = load_detections(path)
        assert match_detections(THREE, loaded)["n_matched"] == 3


def test_inconsistent_fields_raise():
    try:
        match_detections(det([[0, 0, 1, 1]], [0.5, 0.4], [1]), THREE)
    except ValueError:
        pass
    else:
        raise AssertionError("should raise")

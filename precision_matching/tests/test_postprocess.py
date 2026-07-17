import numpy as np

from precision_matching.postprocess import (
    cxcywh_to_xyxy,
    nms_numpy,
    postprocess_query_detector,
    postprocess_yolo_head,
    stable_sigmoid,
    unpack_padded_detections,
)


def test_stable_sigmoid_no_overflow_and_correct():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    with np.errstate(over="raise"):  # overflow must not occur
        y = stable_sigmoid(x)
    assert y[0] == 0.0 and y[4] == 1.0 and abs(y[2] - 0.5) < 1e-15
    assert abs(y[3] - (1 / (1 + np.exp(-1)))) < 1e-12
    assert abs(y[1] + y[3] - 1.0) < 1e-12  # sigmoid(-x) = 1 - sigmoid(x)


def test_cxcywh_to_xyxy():
    out = cxcywh_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]))
    assert np.allclose(out, [[0.4, 0.3, 0.6, 0.7]])


def test_nms_suppresses_overlaps_keeps_distant():
    # box 1 vs box 0: inter 8x10=80, union 120 -> IoU = 2/3
    boxes = np.array([[0, 0, 10, 10], [2, 0, 12, 10], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms_numpy(boxes, scores, iou_threshold=0.5)
    assert list(keep) == [0, 2]  # IoU 0.667 > 0.5, suppressed
    keep = nms_numpy(boxes, scores, iou_threshold=0.7)
    assert list(keep) == [0, 1, 2]  # IoU 0.667 <= 0.7, kept
    assert nms_numpy(np.zeros((0, 4)), np.zeros(0)).size == 0


def test_query_detector_lens():
    # 4 queries, 3 classes: q0 confident class1, q1 background, q2/q3 mid
    logits = np.full((4, 3), -5.0)
    logits[0, 1] = 3.0
    logits[2, 2] = 2.0
    logits[3, 0] = 1.0
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]] * 4)
    det = postprocess_query_detector(logits, boxes, image_size=(100, 200),
                                     score_threshold=0.5)
    assert list(det["labels"]) == [1, 2, 0]  # score-descending order
    assert det["scores"][0] > det["scores"][1] > det["scores"][2] >= 0.5
    # cxcywh (0.5,0.5,0.2,0.2) on (H=100, W=200) -> (80, 40, 120, 60)
    assert np.allclose(det["boxes"][0], [80, 40, 120, 60])
    # top_k truncates after sorting
    det = postprocess_query_detector(logits, boxes, score_threshold=0.5, top_k=1)
    assert list(det["labels"]) == [1]
    # batch dim accepted; all-background yields empty
    det = postprocess_query_detector(np.full((1, 4, 3), -9.0), boxes[None],
                                     score_threshold=0.5)
    assert det["boxes"].shape == (0, 4) and det["labels"].size == 0


def test_query_detector_identical_inputs_match_perfectly():
    from precision_matching.task_level import detection_task_metrics

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(10, 5))
    boxes = np.abs(rng.normal(0.4, 0.1, size=(10, 4)))
    m = detection_task_metrics(
        postprocess_query_detector, (logits, boxes), (logits.copy(), boxes.copy()),
        pp_kwargs={"image_size": (640, 640), "score_threshold": 0.3},
    )
    if m["n_ref"]:
        assert m["ref_match_rate"] == 1.0 and m["matched_iou_min"] == 1.0


def test_yolo_head_lens():
    # [4+2 classes, 3 anchors], boxes cxcywh absolute, cls already sigmoided
    raw = np.array([
        [50.0, 51.0, 200.0],   # cx
        [50.0, 50.0, 200.0],   # cy
        [20.0, 20.0, 10.0],    # w
        [20.0, 20.0, 10.0],    # h
        [0.90, 0.85, 0.10],    # class 0 score
        [0.10, 0.20, 0.60],    # class 1 score
    ])
    det = postprocess_yolo_head(raw, score_threshold=0.25, iou_threshold=0.45)
    # anchors 0/1 same class, IoU ~0.90 -> anchor 1 suppressed; anchor 2 kept
    assert det["boxes"].shape == (2, 4)
    assert list(det["labels"]) == [0, 1]
    assert np.allclose(det["boxes"][0], [40, 40, 60, 60])
    # batch dim accepted; empty when nothing passes threshold
    det = postprocess_yolo_head(raw[None], score_threshold=0.95)
    assert det["boxes"].shape == (0, 4)


def test_unpack_padded_detections():
    arr = np.zeros((1, 100, 6))
    arr[0, 0] = [10, 10, 50, 50, 0.9, 3]
    arr[0, 1] = [20, 20, 60, 60, 0.4, 7]
    det = unpack_padded_detections(arr)
    assert det["boxes"].shape == (2, 4)
    assert list(det["labels"]) == [3, 7]
    det = unpack_padded_detections(arr, score_threshold=0.5)
    assert det["boxes"].shape == (1, 4)
    try:
        unpack_padded_detections(np.zeros((100, 5)))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong last dim should raise")

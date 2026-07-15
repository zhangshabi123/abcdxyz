import numpy as np

from precision_matching.decision_agreement import binary_mask_iou, top1_agreement


def test_identical_logits_agree():
    x = np.random.default_rng(0).normal(size=(1, 1000))
    assert top1_agreement(x, x.copy()) == 1.0


def test_constructed_flip_fraction():
    # 4 positions, flip the argmax at exactly 2
    ref = np.zeros((4, 3))
    ref[:, 0] = 1.0
    cand = ref.copy()
    cand[0], cand[1] = [0, 2, 0], [0, 0, 2]
    assert top1_agreement(ref, cand) == 0.5


def test_semantic_seg_layout_axis1():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(1, 150, 8, 8))
    assert top1_agreement(ref, ref.copy(), axis=1) == 1.0
    cand = ref + rng.normal(size=ref.shape) * 1e-8  # tiny noise, ties unlikely
    assert top1_agreement(ref, cand, axis=1) > 0.9


def test_lm_logits_layout():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(1, 64, 320))
    assert top1_agreement(ref, ref.copy()) == 1.0


def test_shape_mismatch_raises():
    try:
        top1_agreement(np.ones((1, 3)), np.ones((1, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("should raise")


def test_binary_mask_iou():
    a = np.zeros((16, 16))
    a[:8, :8] = 1.0
    assert binary_mask_iou(a, a.copy()) == 1.0
    b = np.zeros((16, 16))
    b[8:, 8:] = 1.0
    assert binary_mask_iou(a, b) == 0.0
    half = a.copy()
    half[:8, :4] = 0.0
    assert abs(binary_mask_iou(a, half) - 0.5) < 1e-12
    z = np.zeros((4, 4))
    assert binary_mask_iou(z, z) == 1.0  # both empty = identical

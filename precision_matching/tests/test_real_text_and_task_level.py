import numpy as np

from precision_matching.real_text import (
    COCO_CLASS_NAMES,
    REAL_TEXT,
    groundingdino_prompt,
    openvocab_queries,
    token_ids_for,
)
from precision_matching.task_level import (
    embedding_cosine,
    lm_task_metrics,
)


class FakeWordTokenizer:
    """HF-style callable: one id per whitespace word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(w) % 30000 for w in text.split()]}


def test_real_text_is_long_natural_ascii():
    words = REAL_TEXT.split()
    assert len(words) >= 400  # long enough to tile to 1024 tokens quickly
    assert REAL_TEXT.isascii()
    assert not REAL_TEXT.startswith("\n") and not REAL_TEXT.endswith("\n")


def test_coco_class_names():
    assert len(COCO_CLASS_NAMES) == 80
    assert len(set(COCO_CLASS_NAMES)) == 80
    assert COCO_CLASS_NAMES[0] == "person" and COCO_CLASS_NAMES[-1] == "toothbrush"
    assert all(n == n.lower() for n in COCO_CLASS_NAMES)


def test_openvocab_prompt_forms():
    q = openvocab_queries()
    assert q == COCO_CLASS_NAMES and q is not COCO_CLASS_NAMES  # copy, not alias
    p = groundingdino_prompt()
    assert p.startswith("person. bicycle.") and p.endswith("toothbrush.")


def test_token_ids_exact_length_tiling_and_truncation():
    tok = FakeWordTokenizer()
    n_words = len(REAL_TEXT.split())
    ids = token_ids_for(tok, 1024)
    assert len(ids) == 1024
    assert ids[:n_words] == token_ids_for(tok, n_words)  # tiling repeats prefix
    assert len(token_ids_for(tok, 32)) == 32
    # determinism
    assert token_ids_for(tok, 1024) == ids
    try:
        token_ids_for(lambda t, add_special_tokens: {"input_ids": []}, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("empty tokenization should raise")
    for bad_len in (0, -5):
        try:
            token_ids_for(tok, bad_len)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive target_len should raise")


def test_lm_task_metrics():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(1, 64, 320))
    m = lm_task_metrics(logits, logits.copy())
    assert m["token_top1_agreement"] == 1.0
    flipped = logits.copy()
    flipped[0, :32] = -flipped[0, :32]  # change the argmax on half the positions
    m = lm_task_metrics(logits, flipped)
    assert m["token_top1_agreement"] < 1.0


def test_embedding_cosine():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(1, 128, 64))
    assert abs(embedding_cosine(h, h.copy()) - 1.0) < 1e-12
    assert abs(embedding_cosine(h, -h) + 1.0) < 1e-12
    z = np.zeros((1, 128, 64))
    assert embedding_cosine(z, z) == 1.0  # both zero = identical
    assert embedding_cosine(h, z) == 0.0
    # orthogonal pooled vectors -> ~0
    a = np.zeros((1, 2, 2))
    b = np.zeros((1, 2, 2))
    a[0, :, 0] = 1.0
    b[0, :, 1] = 1.0
    assert abs(embedding_cosine(a, b)) < 1e-12
    # token-length mismatch must raise, not silently pool it away
    try:
        embedding_cosine(np.ones((1, 100, 64)), np.ones((1, 128, 64)))
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch should raise")

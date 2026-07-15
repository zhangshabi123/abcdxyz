import numpy as np

from precision_matching.input_gen import generate_input, generate_inputs

FLOAT_SPEC = [[1, 3, 8, 8], 0, 1, "float32", "BCHW"]
INT_SPEC = [[1, 512], 0, 100, "int64", "BL"]
ONES_SPEC = [[1, 512], 1, 2, "int64", "BL"]
MASK_SPEC = [[1, 1, 64, 64], 0, 1, "float32", "BCHW", "rect_mask"]


def test_float_dtype_and_range():
    x = generate_inputs([FLOAT_SPEC], seed=0)[0]
    assert x.dtype == np.float32 and x.shape == (1, 3, 8, 8)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_int_dtype_and_halfopen_range():
    x = generate_input(INT_SPEC, np.random.default_rng(0))
    assert x.dtype == np.int64 and x.shape == (1, 512)
    assert x.min() >= 0 and x.max() <= 99  # high is exclusive
    big = generate_input([[1, 100000], 0, 100, "int64", "BL"], np.random.default_rng(0))
    assert big.max() == 99 and big.min() == 0  # full range actually reached


def test_constant_ones_attention_mask():
    x = generate_input(ONES_SPEC, np.random.default_rng(0))
    assert (x == 1).all()


def test_deterministic_across_processes():
    a = generate_inputs([FLOAT_SPEC, INT_SPEC], seed=7)
    b = generate_inputs([FLOAT_SPEC, INT_SPEC], seed=7)
    c = generate_inputs([FLOAT_SPEC, INT_SPEC], seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_rect_mask_is_binary_with_both_values():
    m = generate_input(MASK_SPEC, np.random.default_rng(3))
    assert m.dtype == np.float32 and m.shape == (1, 1, 64, 64)
    assert set(np.unique(m)) == {0.0, 1.0}


def test_bool_dtype_gives_random_binary():
    x = generate_input([[1000], 0, 2, "bool", "B"], np.random.default_rng(0))
    assert x.dtype == np.bool_
    frac = float(x.mean())
    assert 0.4 < frac < 0.6  # random 0/1, not the all-True float pathology


def test_dict_spec_and_bad_kind():
    d = {"shape": [2, 4], "min": 0, "max": 5, "dtype": "int32", "layout": "BL"}
    x = generate_input(d, np.random.default_rng(0))
    assert x.dtype == np.int32 and x.shape == (2, 4)
    try:
        generate_input([[1], 0, 1, "float32", "B", "nope"], np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("bad kind should raise")


def test_empty_int_range_raises():
    try:
        generate_input([[1], 5, 5, "int64", "B"], np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("empty range should raise")

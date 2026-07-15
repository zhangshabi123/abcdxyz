import numpy as np

from precision_matching.dense_metrics import aggregate_worst, dense_metrics

ATOL, RTOL = 1e-5, 1e-3


def test_identical_is_all_zero():
    a = np.random.default_rng(0).normal(size=(4, 100))
    m = dense_metrics(a, a.copy(), ATOL, RTOL)
    assert m["max_atol"] == 0 and m["max_rtol"] == 0
    assert m["nrmse"] == 0 and m["mismatch_ratio"] == 0
    assert m["ref_nonfinite"] == 0 and m["cand_nonfinite"] == 0


def test_uniform_perturbation_known_values():
    a = np.ones((10, 10))
    m = dense_metrics(a, a + 0.01, ATOL, RTOL)
    assert abs(m["max_atol"] - 0.01) < 1e-12
    assert abs(m["max_rtol"] - 0.01) < 1e-9
    # every element exceeds atol + rtol*|a| = 0.00101
    assert m["mismatch_ratio"] == 1.0
    # ||diff|| / ||a|| = 0.01*sqrt(N) / sqrt(N)
    assert abs(m["nrmse"] - 0.01) < 1e-12


def test_near_zero_reference_rtol_explodes_but_nrmse_floored():
    a = np.zeros(1000)
    b = a + 1e-6
    m = dense_metrics(a, b, ATOL, RTOL)
    assert m["max_rtol"] > 1e3          # the known max_rtol pathology
    # floor: ||diff||/(atol*sqrt(N)) = 1e-6/1e-5 = 0.1, not a blow-up
    assert abs(m["nrmse"] - 0.1) < 1e-9


def test_single_bad_element_shows_in_max_not_nrmse():
    a = np.ones(100000)
    b = a.copy()
    b[0] += 5.0
    m = dense_metrics(a, b, ATOL, RTOL)
    assert m["max_atol"] == 5.0
    assert m["mismatch_ratio"] == 1.0 / 100000
    assert m["nrmse"] < 0.02


def test_nonfinite_counted_and_propagates():
    a = np.ones(10)
    b = a.copy()
    b[3] = np.nan
    m = dense_metrics(a, b, ATOL, RTOL)
    assert m["cand_nonfinite"] == 1 and m["ref_nonfinite"] == 0
    assert np.isnan(m["max_atol"])  # NaN metric IS the broken-output signal


def test_shape_mismatch_and_empty_raise():
    for args in [(np.ones(3), np.ones(4)), (np.ones(0), np.ones(0))]:
        try:
            dense_metrics(args[0], args[1], ATOL, RTOL)
        except ValueError:
            pass
        else:
            raise AssertionError("should raise")


def test_nonfinite_counts_as_mismatch():
    a = np.ones(100)
    b = a.copy()
    b[:10] = np.nan
    m = dense_metrics(a, b, ATOL, RTOL)
    assert abs(m["mismatch_ratio"] - 0.1) < 1e-12
    all_nan = dense_metrics(a, np.full(100, np.nan), ATOL, RTOL)
    assert all_nan["mismatch_ratio"] == 1.0


def test_aggregate_worst_nan_order_independent():
    a = np.ones(10)
    good = dense_metrics(a, a + 0.01, ATOL, RTOL)
    broken = dense_metrics(a, np.full(10, np.nan), ATOL, RTOL)
    for order in ([good, broken], [broken, good]):
        agg = aggregate_worst(order)
        assert np.isnan(agg["max_atol"])  # NaN must survive aggregation


def test_zero_atol_all_zero_reference():
    z = np.zeros(16)
    m = dense_metrics(z, z.copy(), atol=0.0, rtol=0.0)
    assert m["nrmse"] == 0.0  # identical outputs must not report NaN
    m = dense_metrics(z, z + 1e-9, atol=0.0, rtol=0.0)
    assert m["nrmse"] == float("inf")


def test_aggregate_worst_takes_max_per_key():
    a, b = np.ones(10), np.ones(10)
    m1 = dense_metrics(a, b + 0.01, ATOL, RTOL)
    m2 = dense_metrics(a, b + 0.02, ATOL, RTOL)
    agg = aggregate_worst([m1, m2])
    assert agg["max_atol"] == m2["max_atol"]
    assert agg["nrmse"] == m2["nrmse"]
    assert agg["cand_nonfinite"] == 0

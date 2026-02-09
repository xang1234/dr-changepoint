import threading
import time

import numpy as np
import pytest

import cpd


def test_pelt_fit_predict_roundtrip_l2() -> None:
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    result = cpd.Pelt(model="l2").fit(values).predict(pen=1.0)
    assert result.breakpoints == [5, 10]
    assert result.change_points == [5]


def test_pelt_large_array_smoke() -> None:
    n = 100_000
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 2.0

    result = (
        cpd.Pelt(model="l2", min_segment_len=50, jump=50, max_change_points=8)
        .fit(values)
        .predict(pen=1.0)
    )

    assert result.breakpoints[-1] == n
    assert result.breakpoints == sorted(set(result.breakpoints))


def test_pelt_rejects_invalid_model() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        cpd.Pelt(model="does-not-exist")


def test_pelt_rejects_invalid_data() -> None:
    with pytest.raises(TypeError, match="expected float32 or float64"):
        cpd.Pelt(model="l2").fit(np.array([1, 2, 3], dtype=np.int64))


def test_pelt_predict_requires_exactly_one_stopping_arg() -> None:
    detector = cpd.Pelt(model="l2").fit(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="exactly one"):
        detector.predict()

    with pytest.raises(ValueError, match="exactly one"):
        detector.predict(pen=1.0, n_bkps=1)


def test_pelt_predict_before_fit_is_clear_error() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\.\\.\\.\\) must be called before predict"):
        cpd.Pelt(model="l2").predict(pen=1.0)


def test_pelt_releases_gil_during_predict() -> None:
    n = 100_000
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 2.0

    state = {"running": True, "in_predict": False, "during_predict_ticks": 0}

    def worker() -> None:
        while state["running"]:
            if state["in_predict"]:
                state["during_predict_ticks"] += 1
            time.sleep(0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.01)

    try:
        state["in_predict"] = True
        _ = (
            cpd.Pelt(model="l2", min_segment_len=50, jump=50, max_change_points=8)
            .fit(values)
            .predict(pen=1.0)
        )
    finally:
        state["in_predict"] = False
        state["running"] = False
        thread.join(timeout=2.0)

    assert state["during_predict_ticks"] > 0

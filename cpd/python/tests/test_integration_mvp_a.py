import numpy as np
import pytest
import threading
import time

import cpd


def _three_regime_signal() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(40, dtype=np.float64),
            np.full(40, 8.0, dtype=np.float64),
            np.full(40, -4.0, dtype=np.float64),
        ]
    )


def test_import_surface_exposes_mvp_a_api() -> None:
    assert hasattr(cpd, "Pelt")
    assert hasattr(cpd, "Binseg")
    assert hasattr(cpd, "Fpop")
    assert hasattr(cpd, "detect_offline")


def test_pelt_binseg_and_fpop_detect_known_breakpoints() -> None:
    x = _three_regime_signal()

    pelt = cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
    binseg = cpd.Binseg(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
    fpop = cpd.Fpop(min_segment_len=2).fit(x).predict(n_bkps=2)

    assert pelt.breakpoints == [40, 80, 120]
    assert binseg.breakpoints == [40, 80, 120]
    assert fpop.breakpoints == [40, 80, 120]


def test_detect_offline_matches_class_api_for_pelt_binseg_and_fpop() -> None:
    x = _three_regime_signal()

    pelt_class = cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
    pelt_low = cpd.detect_offline(
        x,
        detector="pelt",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
        repro_mode="balanced",
    )

    binseg_class = cpd.Binseg(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
    binseg_low = cpd.detect_offline(
        x,
        detector="binseg",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
        repro_mode="balanced",
    )

    fpop_class = cpd.Fpop(min_segment_len=2).fit(x).predict(n_bkps=2)
    fpop_low = cpd.detect_offline(
        x,
        detector="fpop",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
        repro_mode="balanced",
    )

    assert pelt_low.breakpoints == pelt_class.breakpoints
    assert binseg_low.breakpoints == binseg_class.breakpoints
    assert fpop_low.breakpoints == fpop_class.breakpoints


def test_detect_offline_accepts_pipeline_spec() -> None:
    x = _three_regime_signal()
    pipeline = {
        "detector": {"kind": "pelt", "params_per_segment": 2},
        "cost": "l2",
        "constraints": {"min_segment_len": 2},
        "stopping": {"n_bkps": 2},
    }

    pipelined = cpd.detect_offline(x, pipeline=pipeline)
    explicit = cpd.detect_offline(
        x,
        detector="pelt",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
    )

    assert pipelined.breakpoints == explicit.breakpoints


def test_detect_offline_accepts_rust_serde_pipeline_shape() -> None:
    x = _three_regime_signal()
    pipeline = {
        "detector": {
            "Offline": {
                "Pelt": {
                    "stopping": {"KnownK": 2},
                    "params_per_segment": 2,
                    "cancel_check_every": 1000,
                }
            }
        },
        "cost": "L2",
        "constraints": {"min_segment_len": 2},
        "stopping": {"KnownK": 2},
        "seed": None,
    }

    pipelined = cpd.detect_offline(x, pipeline=pipeline)
    explicit = cpd.detect_offline(
        x,
        detector="pelt",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
    )

    assert pipelined.breakpoints == explicit.breakpoints


def test_detect_offline_pipeline_repro_mode_matches_explicit_path() -> None:
    x = _three_regime_signal()
    pipeline = {
        "detector": {"kind": "pelt", "params_per_segment": 2},
        "cost": "l2",
        "constraints": {"min_segment_len": 2},
        "stopping": {"n_bkps": 2},
    }

    pipelined = cpd.detect_offline(x, pipeline=pipeline, repro_mode="fast")
    explicit = cpd.detect_offline(
        x,
        detector="pelt",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
        repro_mode="fast",
    )

    assert pipelined.breakpoints == explicit.breakpoints
    assert not any(
        "currently uses balanced reproducibility mode" in note
        for note in pipelined.diagnostics.notes
    )


def test_detect_offline_accepts_extended_constraints_surface() -> None:
    x = _three_regime_signal()
    result = cpd.detect_offline(
        x,
        detector="pelt",
        cost="l2",
        constraints={
            "min_segment_len": 2,
            "jump": 1,
            "max_change_points": 4,
            "max_depth": None,
            "candidate_splits": None,
            "time_budget_ms": None,
            "max_cost_evals": None,
            "memory_budget_bytes": 20_000_000,
            "max_cache_bytes": 10_000_000,
            "cache_policy": {"kind": "budgeted", "max_bytes": 10_000_000},
            "degradation_plan": [
                {"kind": "increase_jump", "factor": 2, "max_jump": 8},
                {"kind": "disable_uncertainty_bands"},
                {"kind": "switch_cache_policy", "cache_policy": {"kind": "full"}},
            ],
            "allow_algorithm_fallback": False,
        },
        stopping={"n_bkps": 2},
        repro_mode="balanced",
    )
    assert result.breakpoints == [40, 80, 120]


def test_detect_offline_rejects_invalid_parameters() -> None:
    x = _three_regime_signal()

    with pytest.raises(ValueError, match="unsupported detector"):
        cpd.detect_offline(x, detector="nope")

    with pytest.raises(ValueError, match="exactly one"):
        cpd.detect_offline(x, stopping={"n_bkps": 2, "pen": 1.0})

    with pytest.raises(ValueError, match="unsupported constraints key"):
        cpd.detect_offline(x, constraints={"not_a_real_key": 1}, stopping={"n_bkps": 2})

    with pytest.raises(ValueError, match="either pipeline"):
        cpd.detect_offline(
            x,
            pipeline={"detector": "pelt", "stopping": {"n_bkps": 2}},
            detector="binseg",
        )

    with pytest.raises(ValueError, match="requires cost='l2'"):
        cpd.detect_offline(x, detector="fpop", cost="normal", stopping={"n_bkps": 2})


def test_detect_offline_rejects_preprocess_without_feature() -> None:
    x = _three_regime_signal()
    with pytest.raises(ValueError, match="preprocess feature"):
        cpd.detect_offline(x, stopping={"n_bkps": 2}, preprocess={"winsorize": {}})


def test_detect_offline_error_paths_are_clear() -> None:
    with pytest.raises((ValueError, RuntimeError)):
        cpd.detect_offline(np.array([], dtype=np.float64), stopping={"n_bkps": 1})


def test_binseg_releases_gil_during_predict() -> None:
    n = 100_000
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 3.0

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
            cpd.Binseg(model="l2", min_segment_len=50, jump=50, max_change_points=8)
            .fit(values)
            .predict(pen=1.0)
        )
    finally:
        state["in_predict"] = False
        state["running"] = False
        thread.join(timeout=2.0)

    assert state["during_predict_ticks"] > 0


def test_detect_offline_releases_gil_during_compute() -> None:
    n = 100_000
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 3.0

    state = {"running": True, "in_call": False, "ticks": 0}

    def worker() -> None:
        while state["running"]:
            if state["in_call"]:
                state["ticks"] += 1
            time.sleep(0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.01)

    try:
        state["in_call"] = True
        _ = cpd.detect_offline(
            values,
            detector="pelt",
            cost="l2",
            constraints={"min_segment_len": 50, "jump": 50, "max_change_points": 8},
            stopping={"pen": 1.0},
        )
    finally:
        state["in_call"] = False
        state["running"] = False
        thread.join(timeout=2.0)

    assert state["ticks"] > 0

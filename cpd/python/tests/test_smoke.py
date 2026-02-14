from importlib import resources

import numpy as np

import cpd


def test_smoke_detector_fit_predict_roundtrip() -> None:
    detector = cpd.SmokeDetector().fit([0.0, 1.0, 2.0])
    assert detector.predict() == [3]


def test_smoke_detect_one_shot() -> None:
    assert cpd.smoke_detect([0.0, 1.0]) == [2]


def test_mvp_a_high_level_and_low_level_smoke() -> None:
    values = np.concatenate(
        [
            np.zeros(50, dtype=np.float64),
            np.full(50, 5.0, dtype=np.float64),
        ]
    )

    pelt = cpd.Pelt(model="l2").fit(values).predict(n_bkps=1)
    binseg = cpd.Binseg(model="l2").fit(values).predict(n_bkps=1)
    low = cpd.detect_offline(
        values,
        detector="pelt",
        cost="l2",
        stopping={"n_bkps": 1},
    )

    assert pelt.breakpoints == [50, 100]
    assert binseg.breakpoints == [50, 100]
    assert low.breakpoints == [50, 100]
    assert low.change_points == [50]
    assert low.diagnostics.algorithm == "pelt"
    assert low.diagnostics.cost_model == "l2_mean"
    assert low.diagnostics.blas_backend is None


def test_typed_marker_present() -> None:
    marker = resources.files("cpd").joinpath("py.typed")
    assert marker.is_file()

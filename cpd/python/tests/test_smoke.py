from importlib import resources

import cpd


def test_smoke_detector_fit_predict_roundtrip() -> None:
    detector = cpd.SmokeDetector().fit([0.0, 1.0, 2.0])
    assert detector.predict() == [3]


def test_smoke_detect_one_shot() -> None:
    assert cpd.smoke_detect([0.0, 1.0]) == [2]


def test_typed_marker_present() -> None:
    marker = resources.files("cpd").joinpath("py.typed")
    assert marker.is_file()

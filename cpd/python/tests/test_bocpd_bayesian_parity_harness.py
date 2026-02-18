import json
from pathlib import Path
import sys

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import bocpd_parity_harness as bh


MANIFEST_PATH = TESTS_DIR / "fixtures" / "parity" / "bocpd_manifest.v1.json"


def _base_case(case_id: str) -> dict:
    return {
        "id": case_id,
        "category": "step_single",
        "expected_behavior": "x",
        "signal_spec": {
            "kind": "piecewise_constant",
            "seed": 1,
            "segments": [{"length": 10, "mean": 0.0}, {"length": 10, "mean": 1.0}],
            "noise": {"distribution": "normal", "std": 0.1},
        },
        "hazard_mean_run_length": 200.0,
        "target_change_points": [10],
        "tolerance": {"index": 2},
        "profile_tags": ["smoke"],
    }


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest = {
        "manifest_version": 1,
        "cases": [_base_case("dup"), _base_case("dup")],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(bh.ManifestValidationError, match="duplicate case id"):
        bh.load_manifest(path)


def test_manifest_rejects_invalid_profile_tag(tmp_path: Path) -> None:
    bad = _base_case("bad_profile")
    bad["profile_tags"] = ["smoke", "nightly_only"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": 1, "cases": [bad]}), encoding="utf-8")

    with pytest.raises(bh.ManifestValidationError, match="unsupported profile tags"):
        bh.load_manifest(path)


def test_manifest_rejects_invalid_target_change_points(tmp_path: Path) -> None:
    bad = _base_case("bad_cp")
    bad["target_change_points"] = [12, 10]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": 1, "cases": [bad]}), encoding="utf-8")

    with pytest.raises(
        bh.ManifestValidationError, match="target_change_points must be strictly increasing"
    ):
        bh.load_manifest(path)


def test_signal_generation_is_deterministic() -> None:
    cases = bh.load_manifest(MANIFEST_PATH)
    case = next(c for c in cases if c.id == "step_single_01")

    first = bh.generate_signal(case)
    second = bh.generate_signal(case)
    np.testing.assert_allclose(first, second)


def test_peak_extraction_enforces_min_separation() -> None:
    p_change = np.asarray([0.0, 0.1, 0.2, 0.95, 0.1, 0.90, 0.2, 0.88, 0.1], dtype=np.float64)
    peaks = bh.extract_top_k_change_points(
        p_change, k=2, warmup_skip=0, min_separation=3
    )
    assert peaks == (3, 7)


def test_tolerance_and_jaccard_behavior() -> None:
    tolerant, matches, jaccard = bh.compare_change_points((20, 50), (18, 48), tolerance=2)
    assert tolerant
    assert matches == 2
    assert jaccard == pytest.approx(1.0)

    tolerant2, matches2, jaccard2 = bh.compare_change_points((20, 50), (70,), tolerance=2)
    assert not tolerant2
    assert matches2 == 0
    assert jaccard2 == pytest.approx(0.0)


def test_reference_adapter_normalizes_run_length_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeOnlineChangepointDetection:
        @staticmethod
        def online_changepoint_detection(
            values: np.ndarray, hazard, observation
        ) -> tuple[np.ndarray, np.ndarray]:
            _ = hazard(np.arange(4))
            _ = observation
            n = values.shape[0]
            run_length = np.zeros((n + 1, n + 1), dtype=np.float64)
            run_length[0, 1:] = np.linspace(0.1, 0.9, n, dtype=np.float64)
            return run_length, np.zeros(n, dtype=np.float64)

    class _FakeHazardFunctions:
        @staticmethod
        def constant_hazard(mean_run_length: float, run_length: np.ndarray) -> np.ndarray:
            values = np.asarray(run_length, dtype=np.float64)
            return np.full(values.shape, 1.0 / mean_run_length, dtype=np.float64)

    class _FakeStudentT:
        def __init__(self, alpha: float, beta: float, kappa: float, mu: float) -> None:
            self.params = (alpha, beta, kappa, mu)

    class _FakeOnlineLikelihoods:
        StudentT = _FakeStudentT

    def _fake_import(name: str):
        if name == "bayesian_changepoint_detection.online_changepoint_detection":
            return _FakeOnlineChangepointDetection
        if name == "bayesian_changepoint_detection.hazard_functions":
            return _FakeHazardFunctions
        if name == "bayesian_changepoint_detection.online_likelihoods":
            return _FakeOnlineLikelihoods
        raise AssertionError(f"unexpected module import {name}")

    monkeypatch.setattr(bh.importlib, "import_module", _fake_import)

    values = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    p_change = bh.run_reference_case(values=values, hazard_mean_run_length=200.0)
    assert p_change.shape == (values.shape[0],)
    assert np.all(np.isfinite(p_change))
    assert p_change[0] == pytest.approx(0.1)
    assert p_change[-1] == pytest.approx(0.9)

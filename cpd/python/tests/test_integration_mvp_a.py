import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

import numpy as np
import pytest

import cpd

CPD_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
MIGRATION_RESULT_FIXTURE_DIR = CPD_ROOT / "tests" / "fixtures" / "migrations" / "result"


def _three_regime_signal() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(40, dtype=np.float64),
            np.full(40, 8.0, dtype=np.float64),
            np.full(40, -4.0, dtype=np.float64),
        ]
    )


def _load_example_module(script_name: str) -> ModuleType:
    script_path = EXAMPLES_DIR / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"cpd_example_{script_name}", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fixture_payload(fixture_name: str) -> str:
    return (MIGRATION_RESULT_FIXTURE_DIR / fixture_name).read_text(encoding="utf-8")


class _FakeAxis:
    def __init__(self) -> None:
        self.figure = None

    def plot(self, _y, **_kwargs) -> None:
        return None

    def axvline(self, _x, **_kwargs) -> None:
        return None

    def set_title(self, _title) -> None:
        return None

    def set_ylabel(self, _label) -> None:
        return None

    def set_xlabel(self, _label) -> None:
        return None

    def legend(self, *_args, **_kwargs) -> None:
        return None


class _FakeFigure:
    def __init__(self) -> None:
        self.axes = []

    def tight_layout(self) -> None:
        return None

    def savefig(self, out_path, **_kwargs) -> None:
        Path(out_path).write_bytes(b"fake-png")


def _install_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    pyplot = ModuleType("matplotlib.pyplot")

    def _subplots(rows: int, cols: int = 1, **_kwargs):
        fig = _FakeFigure()
        axes_grid = np.empty((rows, cols), dtype=object)
        for row in range(rows):
            for col in range(cols):
                axis = _FakeAxis()
                axis.figure = fig
                axes_grid[row, col] = axis
                fig.axes.append(axis)
        return fig, axes_grid

    pyplot.subplots = _subplots  # type: ignore[attr-defined]

    matplotlib = ModuleType("matplotlib")
    matplotlib.use = lambda _backend: None  # type: ignore[attr-defined]
    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)


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


def test_detector_outputs_roundtrip_through_result_json_api() -> None:
    x = _three_regime_signal()
    outputs = [
        cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2),
        cpd.Binseg(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2),
        cpd.Fpop(min_segment_len=2).fit(x).predict(n_bkps=2),
        cpd.detect_offline(
            x,
            detector="pelt",
            cost="l2",
            constraints={"min_segment_len": 2},
            stopping={"n_bkps": 2},
            repro_mode="balanced",
        ),
    ]

    for result in outputs:
        payload = result.to_json()
        decoded = json.loads(payload)
        restored = cpd.OfflineChangePointResult.from_json(payload)

        assert restored.breakpoints == result.breakpoints
        assert restored.change_points == result.change_points
        assert restored.diagnostics.algorithm == decoded["diagnostics"]["algorithm"]
        assert decoded["diagnostics"]["schema_version"] >= 1


@pytest.mark.parametrize(
    ("fixture_name", "expected_schema_version"),
    [
        ("offline_result.v1.json", 1),
        ("offline_result.v2.additive.json", 2),
    ],
)
def test_result_json_backcompat_fixtures_parse_in_python_api(
    fixture_name: str, expected_schema_version: int
) -> None:
    payload = _load_fixture_payload(fixture_name)
    fixture = json.loads(payload)
    parsed = cpd.OfflineChangePointResult.from_json(payload)

    assert parsed.breakpoints == fixture["breakpoints"]
    assert parsed.change_points == fixture["change_points"]
    assert parsed.diagnostics.schema_version == expected_schema_version


def test_additive_fixture_unknown_fields_roundtrip_through_python_api() -> None:
    parsed = cpd.OfflineChangePointResult.from_json(
        _load_fixture_payload("offline_result.v2.additive.json")
    )
    roundtrip = json.loads(parsed.to_json())

    assert roundtrip["future_result_flag"] == "additive-v2"
    assert roundtrip["diagnostics"]["future_diagnostics_flag"]["source"] == "v2"


def test_examples_synthetic_and_csv_run_integration_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    synthetic_signal = _load_example_module("synthetic_signal")
    csv_detect = _load_example_module("csv_detect")

    assert synthetic_signal.main() == 0

    csv_path = tmp_path / "signal.csv"
    np.savetxt(csv_path, _three_regime_signal(), delimiter=",")
    monkeypatch.setattr(
        sys,
        "argv",
        ["csv_detect.py", "--csv", str(csv_path), "--column", "0", "--n-bkps", "2"],
    )
    assert csv_detect.main() == 0


def test_plot_example_runs_with_fake_matplotlib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_matplotlib(monkeypatch)
    plot_breakpoints = _load_example_module("plot_breakpoints")

    out_path = tmp_path / "plot.png"
    monkeypatch.setattr(sys, "argv", ["plot_breakpoints.py", "--out", str(out_path)])
    assert plot_breakpoints.main() == 0
    assert out_path.is_file()


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


def test_detect_offline_rejects_pipeline_fpop_with_non_l2_cost() -> None:
    x = _three_regime_signal()
    pipeline = {
        "detector": {"kind": "fpop"},
        "cost": "normal",
        "stopping": {"n_bkps": 2},
    }
    with pytest.raises(
        ValueError,
        match=r"(pipeline\.detector='fpop' requires pipeline\.cost='l2'|detector=fpop requires cost=l2)",
    ):
        cpd.detect_offline(x, pipeline=pipeline)


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

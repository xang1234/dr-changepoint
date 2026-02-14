# cpd-rs Python Bindings (MVP-A)

`cpd-rs` exposes fast offline change-point detection from Rust into Python.

For citation and provenance policy, see [`../CITATION.cff`](../CITATION.cff)
and [`../docs/clean_room_policy.md`](../docs/clean_room_policy.md).

## Install

For local development from this repository:

```bash
cd cpd/python
python -m pip install --upgrade pip maturin numpy
maturin develop --release --manifest-path ../crates/cpd-python/Cargo.toml
```

Apple Silicon contributors should run the architecture checks and sanity path in
[`../docs/python_apple_silicon_toolchain.md`](../docs/python_apple_silicon_toolchain.md)
before debugging `pyo3`/linker errors.

## API Map

- `cpd.Pelt`: high-level PELT detector.
- `cpd.Binseg`: high-level Binary Segmentation detector.
- `cpd.detect_offline`: low-level API for explicit detector/cost/constraints/stopping/preprocess selection.
- `cpd.OfflineChangePointResult`: typed result object with breakpoints and diagnostics.

## Quickstart

See [`QUICKSTART.md`](./QUICKSTART.md) for a full walkthrough.

## Reproducibility Modes

`detect_offline(..., repro_mode=...)` supports `strict`, `balanced` (default),
and `fast`.
For deterministic contracts, cross-platform expectations, and tolerance gates,
see [`../docs/reproducibility_modes.md`](../docs/reproducibility_modes.md).

Minimal example:

```python
import numpy as np
import cpd

x = np.concatenate([
    np.zeros(40, dtype=np.float64),
    np.full(40, 8.0, dtype=np.float64),
    np.full(40, -4.0, dtype=np.float64),
])

pelt = cpd.Pelt(model="l2").fit(x).predict(n_bkps=2)
binseg = cpd.Binseg(model="l2").fit(x).predict(n_bkps=2)
low = cpd.detect_offline(
    x,
    detector="pelt",
    cost="l2",
    constraints={"min_segment_len": 2},
    stopping={"n_bkps": 2},
    preprocess={
        "detrend": {"method": "polynomial", "degree": 2},
        "deseasonalize": {"method": "stl_like", "period": 4},
        "winsorize": {"lower_quantile": 0.05, "upper_quantile": 0.95},
        "robust_scale": {"mad_epsilon": 1e-9, "normal_consistency": 1.4826},
    },  # optional; requires preprocess feature
)
```

## Preprocess Config Contract

`detect_offline(..., preprocess=...)` validates keys and method payloads.
Unknown preprocess stage keys fail with `ValueError`.

Canonical shape:

```python
preprocess = {
    "detrend": {"method": "linear"},  # or {"method": "polynomial", "degree": 2}
    "deseasonalize": {"method": "differencing", "period": 2},  # or method="stl_like" (period >= 2)
    "winsorize": {"lower_quantile": 0.05, "upper_quantile": 0.95},  # optional fields
    "robust_scale": {"mad_epsilon": 1e-9, "normal_consistency": 1.4826},  # optional fields
}
```

Validation details:

- `detrend.method`: `"linear"` or `"polynomial"` (`degree` required for polynomial).
- `deseasonalize.method`: `"differencing"` (`period >= 1`) or `"stl_like"` (`period >= 2`).
- `winsorize`: defaults to `lower_quantile=0.01`, `upper_quantile=0.99` when omitted.
- `robust_scale`: defaults to `mad_epsilon=1e-9`, `normal_consistency=1.4826` when omitted.

## Example Scripts

- `examples/synthetic_signal.py`: synthetic step-function detection with all MVP-A APIs.
- `examples/csv_detect.py`: detect breakpoints from a CSV column.
- `examples/plot_breakpoints.py`: render detected breakpoints over a synthetic signal.

Run from repo root:

```bash
cpd/python/.venv/bin/python cpd/python/examples/synthetic_signal.py
cpd/python/.venv/bin/python cpd/python/examples/csv_detect.py --csv /path/to/data.csv --column 0
cpd/python/.venv/bin/python cpd/python/examples/plot_breakpoints.py --out /tmp/cpd_breakpoints.png
```

## Ruptures Parity Suite

To run the differential parity suite locally (after installing `ruptures` in the active
environment):

```bash
cd cpd/python
CPD_PARITY_PROFILE=smoke pytest -q tests/test_ruptures_parity.py
CPD_PARITY_PROFILE=full CPD_PARITY_REPORT_OUT=/tmp/cpd-parity-report.json pytest -q tests/test_ruptures_parity.py
```

See [`../docs/parity_ruptures.md`](../docs/parity_ruptures.md) for corpus structure,
tolerance rules, and CI thresholds.

## Wheel CI Policy

Cross-platform wheel hardening is enforced by
[`../../.github/workflows/wheel-build.yml`](../../.github/workflows/wheel-build.yml)
and [`../../.github/workflows/wheel-smoke.yml`](../../.github/workflows/wheel-smoke.yml).

- Build backend: `cibuildwheel`
- Platforms:
  - Linux manylinux x86_64
  - macOS universal2 (validated on `macos-13` and `macos-14`)
  - Windows amd64 (`windows-2022`)
- Python matrix:
  - Full (`main`/nightly/tag): `3.9`, `3.10`, `3.11`, `3.12`, `3.13`
  - Tiered (`pull_request`): representative subset with at least one `3.13` row
- NumPy matrix:
  - `1.26.*` and `2.*`
  - `3.13 + numpy 1.26.*` is excluded
- Python `3.13` rows are marked `experimental` and soft-gated (`continue-on-error`)

Default wheels are BLAS-free by policy:

- Native dependency reports are gated by
  [`../../.github/scripts/wheel_dependency_gate.py`](../../.github/scripts/wheel_dependency_gate.py)
  using `auditwheel` (Linux), `delocate` (macOS), and `delvewheel` (Windows).
- Runtime smoke asserts `low.diagnostics.blas_backend is None` for default wheel installs.

## Troubleshooting

1. `TypeError: expected float32 or float64`
Cause: integer/object arrays are passed into `.fit(...)` or `detect_offline(...)`.
Fix: cast first, e.g. `x = np.asarray(x, dtype=np.float64)`.

2. Input contains NaN/missing values and detection fails
Cause: MVP-A Python APIs reject missing values under `MissingPolicy::Error`.
Fix: impute/drop NaNs before calling detectors.

3. `RuntimeError: fit(...) must be called before predict(...)`
Cause: `.predict(...)` called on an unfitted high-level detector.
Fix: always call `.fit(x)` first.

4. Extension import fails after Rust/Python upgrade
Cause: wheel/extension built against a different interpreter environment.
Fix: rebuild via `maturin develop --release` in the active environment.

5. Apple Silicon linker mismatch (`arm64` vs `x86_64`)
Cause: host shell/interpreter/libpython architectures do not match.
Fix: follow
[`../docs/python_apple_silicon_toolchain.md`](../docs/python_apple_silicon_toolchain.md)
to verify architecture and run the CI-aligned local sanity flow.

## API Reference Outline

- `Pelt(model="l2"|"normal", min_segment_len, jump, max_change_points)`
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `Binseg(model="l2"|"normal", min_segment_len, jump, max_change_points, max_depth)`
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `detect_offline(x, detector, cost, constraints, stopping, preprocess, repro_mode, return_diagnostics)`
- `OfflineChangePointResult`
  - fields: `breakpoints`, `change_points`, `scores`, `segments`, `diagnostics`
  - helper: `to_json()` (`from_json(...)` is planned and not yet implemented)

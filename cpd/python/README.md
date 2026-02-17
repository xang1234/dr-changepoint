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
- `cpd.Fpop`: high-level FPOP detector (L2 cost only).
- `cpd.detect_offline`: low-level API for explicit detector/cost/constraints/stopping/preprocess selection.
- `cpd.OfflineChangePointResult`: typed result object with breakpoints and diagnostics.

## Streaming `update()` vs `update_many()` Policy

`update_many()` now uses a size-aware GIL strategy in Rust bindings:

- Workloads with `< 16` scalar work items (`n * d`) keep the GIL (lower overhead for tiny micro-batches).
- Workloads with `>= 16` scalar work items (`n * d`) release the GIL (`py.allow_threads`) for throughput and thread fairness.

To reproduce the benchmark snapshot used for this policy:

```bash
cd cpd/python
python -m pip install --upgrade pytest
pytest -q tests/test_streaming_perf_contract.py
```

Optional controls:

- `CPD_PY_STREAMING_PERF_ENFORCE=1`: enable stricter ratio gates.
- `CPD_PY_STREAMING_PERF_REPORT_OUT=/tmp/cpd-python-streaming-perf.json`: write JSON metrics.

The perf contract uses median latency with outlier-triggered retry rounds to reduce scheduler-noise flakiness.

Reference run (local dev machine, `tests/test_streaming_perf_contract.py`, median ms):

| Batch size | `update()` median ms | `update_many()` median ms | `update_many()` speedup vs `update()` |
| --- | ---: | ---: | ---: |
| 1 | 0.0035 | 0.0097 | 0.36x |
| 8 | 0.0177 | 0.0194 | 0.91x |
| 16 | 0.0356 | 0.0310 | 1.15x |
| 64 | 0.1308 | 0.0891 | 1.47x |
| 4096 | 7.8216 | 4.4616 | 1.75x |

## Masking Risk Guidance

If BinSeg diagnostics indicate masking risk (for example warnings that closely
spaced weaker changes may be hidden), prefer Wild Binary Segmentation (WBS) in
Rust/offline flows (`cpd-offline::Wbs`) for stronger recovery.

Python high-level APIs expose `cpd.Pelt`, `cpd.Binseg`, and `cpd.Fpop`.
WBS is not yet exposed as a Python high-level detector.

## Quickstart

See [`QUICKSTART.md`](./QUICKSTART.md) for a full walkthrough.

## Reproducibility Modes

`detect_offline(..., repro_mode=...)` supports `strict`, `balanced` (default),
and `fast`.
For deterministic contracts, cross-platform expectations, and tolerance gates,
see [`../docs/reproducibility_modes.md`](../docs/reproducibility_modes.md).

## Result JSON Contract

`OfflineChangePointResult.to_json()` / `OfflineChangePointResult.from_json(...)`
follow the versioned contract in
[`../docs/result_json_contract.md`](../docs/result_json_contract.md), with the
canonical schema marker at `diagnostics.schema_version`.

In `0.x`, schema compatibility follows the bounded version window documented in
[`../VERSIONING.md`](../VERSIONING.md): readers accept only supported
schema-marker versions (currently `1..=2` for offline result fixtures).

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
fpop = cpd.Fpop(min_segment_len=2).fit(x).predict(n_bkps=2)
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

payload = pelt.to_json()
restored = cpd.OfflineChangePointResult.from_json(payload)
assert restored.breakpoints == pelt.breakpoints

try:
    fig = restored.plot(x, title="Detected breakpoints")
except ImportError:
    # Plotting remains optional.
    # Install with: python -m pip install matplotlib
    fig = None
```

## Stopping and Penalty Guide

Ruptures-compatible naming is supported in Python:

- `n_bkps`: exact number of change points (`Stopping::KnownK`)
- `pen`: manual penalty scalar (`Stopping::Penalized(Penalty::Manual(...))`)
- `min_segment_len`: minimum segment size (`Constraints.min_segment_len`)

When to use each stopping style:

- `n_bkps` (`KnownK`): use when you know the expected number of changes and need an exact count.
- `pen="bic"`: good default when you want automatic model-selection behavior that scales with sample size.
- `pen="aic"`: less conservative than BIC; can recover weaker changes but may over-segment noisy data.
- `pen=<float>`: use when you need tight operational control over sensitivity (lower finds more changes, higher finds fewer).
- `stopping={"PenaltyPath": [...]}` (pipeline serde form): request multiple penalties in one PELT sweep and inspect diagnostics notes for each path entry.

BIC/AIC complexity terms are model-aware by default:

- `l2` uses `params_per_segment=2` (mean + residual variance proxy)
- `normal` uses `params_per_segment=3` (mean + variance + residual term)
- `normal_full_cov` uses model-aware effective complexity for BIC/AIC: `1 + d + d(d+1)/2` (mean vector + full covariance + residual term)

Advanced users can still override `params_per_segment` in low-level pipeline detector config.

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

- `Pelt(model="l2"|"normal"|"normal_full_cov", min_segment_len, jump, max_change_points)`
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `Binseg(model="l2"|"normal"|"normal_full_cov", min_segment_len, jump, max_change_points, max_depth)`
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `Fpop(min_segment_len, jump, max_change_points)` (`l2` only)
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `detect_offline(x, pipeline=None, detector, cost, constraints, stopping, preprocess, repro_mode, return_diagnostics)`
  - `detector` accepts `pelt`, `binseg`, or `fpop` (`fpop` requires `cost="l2"`).
  - `cost` accepts `l1_median`, `l2`, `normal`, `normal_full_cov`, and (pipeline-only) `nig`.
  - `pipeline` accepts both simplified Python dicts (for example `{"detector": {"kind": "pelt"}}`) and Rust `PipelineSpec` serde shape (for example `{"detector": {"Offline": {"Pelt": {...}}}, ...}`).
- `OfflineChangePointResult`
  - fields: `breakpoints`, `change_points`, `scores`, `segments`, `diagnostics`
  - helpers: `to_json()`, `from_json(payload)`, `plot(values=None, *, ax=None, title=...)`

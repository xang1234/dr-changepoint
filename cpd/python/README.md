# cpd-rs Python Bindings (MVP-A)

`cpd-rs` exposes fast offline change-point detection from Rust into Python.

## Install

For local development from this repository:

```bash
cd cpd/python
python -m pip install --upgrade pip maturin numpy
maturin develop --release --manifest-path ../crates/cpd-python/Cargo.toml
```

## API Map

- `cpd.Pelt`: high-level PELT detector.
- `cpd.Binseg`: high-level Binary Segmentation detector.
- `cpd.detect_offline`: low-level API for explicit detector/cost/constraints/stopping selection.
- `cpd.OfflineChangePointResult`: typed result object with breakpoints and diagnostics.

## Quickstart

See [`QUICKSTART.md`](./QUICKSTART.md) for a full walkthrough.

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
)
```

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

## API Reference Outline

- `Pelt(model="l2"|"normal", min_segment_len, jump, max_change_points)`
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `Binseg(model="l2"|"normal", min_segment_len, jump, max_change_points, max_depth)`
  - `.fit(x)` -> detector
  - `.predict(pen=..., n_bkps=...)` -> `OfflineChangePointResult`
- `detect_offline(x, detector, cost, constraints, stopping, repro_mode, return_diagnostics)`
- `OfflineChangePointResult`
  - fields: `breakpoints`, `change_points`, `scores`, `segments`, `diagnostics`
  - helper: `to_json()` (`from_json(...)` is planned and not yet implemented)

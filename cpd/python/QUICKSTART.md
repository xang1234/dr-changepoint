# Quickstart (MVP-A)

## 1. Build and import the extension

```bash
cd cpd/python
python -m pip install --upgrade pip maturin numpy
maturin develop --release --manifest-path ../crates/cpd-python/Cargo.toml
python -c "import cpd; print(cpd.__version__)"
```

If you are on Apple Silicon, run the architecture verification and troubleshooting
steps in
[`../docs/python_apple_silicon_toolchain.md`](../docs/python_apple_silicon_toolchain.md)
before or after this build step when diagnosing linker/import failures.

## 2. Detect change points with high-level APIs

```python
import numpy as np
import cpd

x = np.concatenate([
    np.zeros(50, dtype=np.float64),
    np.full(50, 5.0, dtype=np.float64),
    np.full(50, -2.0, dtype=np.float64),
])

pelt_result = cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
binseg_result = cpd.Binseg(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)

print("PELT breakpoints:", pelt_result.breakpoints)
print("BinSeg breakpoints:", binseg_result.breakpoints)
```

Expected breakpoints:

```text
[50, 100, 150]
```

## 3. Use low-level `detect_offline(...)`

```python
low_level = cpd.detect_offline(
    x,
    detector="pelt",
    cost="l2",
    constraints={
        "min_segment_len": 2,
        "jump": 1,
        "max_change_points": 4,
    },
    stopping={"n_bkps": 2},
    preprocess={
        "detrend": {"method": "linear"},
        "deseasonalize": {"method": "differencing", "period": 2},
        "winsorize": {},  # defaults to lower_quantile=0.01, upper_quantile=0.99
        "robust_scale": {},  # defaults to mad_epsilon=1e-9, normal_consistency=1.4826
    },  # optional; requires preprocess feature
    repro_mode="balanced",
    return_diagnostics=True,
)

print("Low-level breakpoints:", low_level.breakpoints)
print("Algorithm:", low_level.diagnostics.algorithm)
print("Cost model:", low_level.diagnostics.cost_model)
```

`preprocess` is strictly validated: unsupported keys or invalid method/parameter
combinations raise `ValueError`.

## 4. Run provided examples

```bash
cpd/python/.venv/bin/python cpd/python/examples/synthetic_signal.py
cpd/python/.venv/bin/python cpd/python/examples/csv_detect.py --csv /path/to/data.csv --column 0
cpd/python/.venv/bin/python cpd/python/examples/plot_breakpoints.py --out /tmp/cpd_breakpoints.png
```

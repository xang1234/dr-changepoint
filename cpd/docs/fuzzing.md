# Fuzzing Guide

This repository uses `cargo-fuzz` to harden panic/crash behavior for core offline, online, and Python interop surfaces.

## Fuzz Package

Fuzz targets live under `cpd/fuzz`.

Current targets:

- `timeseries_view_no_panic`
- `cost_segment_cost_no_panic`
- `offline_detectors_no_panic`
- `online_bocpd_update_no_panic`
- `numpy_interop_no_panic`

`offline_detectors_no_panic` currently exercises offline detector paths for:

- `Pelt`
- `BinSeg`
- `Wbs`

## Local Setup

```bash
cargo install --locked cargo-fuzz
python -m pip install --upgrade pip numpy
```

## Running Targets Locally

Run from `cpd/fuzz`:

```bash
cargo fuzz run timeseries_view_no_panic -- -max_total_time=60
cargo fuzz run cost_segment_cost_no_panic -- -max_total_time=60
cargo fuzz run offline_detectors_no_panic -- -max_total_time=60
cargo fuzz run online_bocpd_update_no_panic -- -max_total_time=60
cargo fuzz run numpy_interop_no_panic -- -max_total_time=60
```

For the NumPy interop target, set `PYO3_PYTHON` when needed:

```bash
PYO3_PYTHON="$(python -c 'import sys; print(sys.executable)')" \
  cargo fuzz run numpy_interop_no_panic -- -max_total_time=60
```

## Corpus and Artifacts

`cargo-fuzz` writes runtime state to:

- `cpd/fuzz/corpus/<target>`
- `cpd/fuzz/artifacts/<target>`

These paths are ignored in git.

## Crash Reproduction Workflow

If fuzzing reports a crash, reproduce first:

```bash
cargo fuzz run <target> cpd/fuzz/artifacts/<target>/<crash-file>
```

Then:

1. Minimize the reproducer input when possible.
2. Add a regression test in the affected crate.
3. Fix the bug before merging.
4. Re-run the target to confirm no crash.

## CI Policy

Nightly CI runs all five active targets for 900 seconds each (4500 seconds total).

If a nightly fuzz job fails:

1. Download uploaded fuzz artifacts.
2. Reproduce locally.
3. Land the fix and regression test in the same PR when feasible.

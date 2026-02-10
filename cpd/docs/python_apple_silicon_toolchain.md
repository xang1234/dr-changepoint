# Apple Silicon Python Toolchain Guide

This guide documents the supported local setup for `cpd-python` contributors on
Apple Silicon and how to debug `pyo3`/`libpython` architecture mismatch errors.

## Supported Setup Policy

- On Apple Silicon hosts, use an `arm64` shell and an `arm64` Python
  interpreter/libpython pair for local `cpd-python` development.
- Do not mix `arm64` host/toolchain with `x86_64` Python binaries unless you are
  intentionally running the entire workflow under Rosetta.
- Keep Rust, Python, and `PYO3_PYTHON` pointed at the same interpreter.

## Verify Architecture Alignment

Run these checks before building `cpd-python`:

```bash
# Host and current shell architecture (expected on Apple Silicon: arm64)
uname -m
arch

# Python interpreter path and architecture
PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
echo "${PYTHON_BIN}"
file "${PYTHON_BIN}"
python -c "import platform, sys; print('platform.machine=', platform.machine()); print('sys.executable=', sys.executable)"

# Linked libpython path and architecture
LIBPYTHON="$(python -c 'import pathlib, sysconfig; libdir=sysconfig.get_config_var(\"LIBDIR\"); lib=sysconfig.get_config_var(\"LDLIBRARY\"); print(pathlib.Path(libdir, lib) if libdir and lib else \"\")')"
echo "${LIBPYTHON}"
if [ -n "${LIBPYTHON}" ]; then
  file "${LIBPYTHON}"
fi

# Optional: inspect linked dynamic libraries from the interpreter
otool -L "${PYTHON_BIN}" | rg -i libpython || true
```

If host/shell is `arm64` but interpreter or `libpython` shows `x86_64`, fix the
environment before continuing.

## Common Mismatch Symptoms

- `ld: warning: ignoring file ... libpython... built for macOS-x86_64`
- `ld: symbol(s) not found for architecture arm64`
- `pyo3-build-config` selecting an interpreter architecture that does not match
  the active target

## Remediation

1. Ensure you are in a native Apple Silicon shell (`arch` must print `arm64`).
2. Select an `arm64` Python interpreter (for Homebrew installs this is usually
   under `/opt/homebrew/bin/python3`, not `/usr/local/bin/python3`).
3. Recreate your virtual environment with that interpreter.
4. Set `PYO3_PYTHON` explicitly when running Rust tests for `cpd-python`.
5. Rebuild extension artifacts in the corrected environment.

Example reset:

```bash
cd cpd/python
rm -rf .venv
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin numpy pytest
```

## Local Sanity Path (Matches CI Expectations)

This mirrors the `python-mvp-a-api` job in
`/Users/admin/Documents/Work/claude-doctor-changepoint/.github/workflows/pr-checks.yml`.

From repository root:

```bash
cd cpd/python
python -m pip install --upgrade pip maturin numpy pytest
maturin build --release \
  --manifest-path ../crates/cpd-python/Cargo.toml \
  --features extension-module \
  --interpreter python
python -m pip install --force-reinstall ../target/wheels/cpd_rs-*.whl
pytest -q \
  tests/test_pelt.py \
  tests/test_integration_mvp_a.py \
  tests/test_smoke.py

cd ../
PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
PYO3_PYTHON="${PYTHON_BIN}" cargo test -p cpd-python --lib
PYO3_PYTHON="${PYTHON_BIN}" cargo test -p cpd-python --features preprocess --lib
PYO3_PYTHON="${PYTHON_BIN}" cargo test -p cpd-python --features serde --lib result_objects::tests::
```

This sequence verifies:

- wheel build/install succeeds against the active interpreter
- Python API smoke/integration tests pass
- Rust `cpd-python` lib tests resolve `pyo3` against the same interpreter

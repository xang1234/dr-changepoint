# Releasing `changepoint-doctor` to PyPI

This project publishes Python artifacts with GitHub Actions via
`/.github/workflows/release.yml`.

## Target package/version

- Distribution name: `changepoint-doctor`
- Release version: `0.0.2`
- Import package: `cpd`

## Local artifact sanity check

From repository root:

```bash
cd cpd/python
python -m pip install --upgrade pip maturin
maturin build --release --interpreter python
maturin sdist --out dist
ls -1 ../target/wheels/changepoint_doctor-0.0.2-*.whl
ls -1 dist/changepoint_doctor-0.0.2.tar.gz
```

## CI/CD release flow

1. Push a git tag for the release:
   - `git tag v0.0.2`
   - `git push origin v0.0.2`
2. The `release` workflow builds:
   - Linux wheels
   - macOS wheels
   - Windows wheels
   - sdist
3. The workflow signs and verifies the release bundle.
4. The `publish-pypi` job uploads artifacts from `release-artifacts/dist` to PyPI.

## PyPI environment requirement

Configure the `pypi` GitHub environment with Trusted Publishing for this
repository before first publish.

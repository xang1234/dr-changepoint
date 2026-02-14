# Release Security Verification

This project publishes security metadata for each tagged release:

- Artifact checksums: `SHA256SUMS.txt`
- Sigstore keyless signatures and certificates: `*.sig`, `*.pem`
- CycloneDX SBOMs:
  - `sbom-rust-workspace.cdx.json`
  - `sbom-crates.cdx.json`
  - `sbom-wheels.cdx.json`
- GitHub build provenance attestations
- Pre-publish crate artifact alignment checks for crates.io publication

Release assets are attached to the corresponding GitHub release tag (`v*`).

## Wheel hardening controls

Release wheels are built with `cibuildwheel` and aligned with non-release wheel CI:

- Linux manylinux x86_64
- macOS universal2
- Windows amd64
- Python `3.9` through `3.13`

Dependency/linkage gates run per wheel artifact:

- Linux: `auditwheel show`
- macOS: `delocate-listdeps`
- Windows: `delvewheel show`

All reports are checked by `.github/scripts/wheel_dependency_gate.py`, which blocks
unexpected BLAS/LAPACK-style dependencies (for example OpenBLAS/MKL) in default
wheels. Linux gates additionally require a `manylinux` tag to be reported by
`auditwheel`.

## 1. Verify checksums

Download release assets and run:

```bash
sha256sum -c SHA256SUMS.txt
```

Every line must report `OK`.

## 2. Verify Sigstore keyless signatures

Install [cosign](https://github.com/sigstore/cosign), then verify each signed file:

```bash
REPO="OWNER/REPO"
for artifact in dist/* crates/*.crate sbom/*.json checksums/SHA256SUMS.txt; do
  base="$(basename "$artifact")"
  cosign verify-blob \
    --signature "signatures/${base}.sig" \
    --certificate "signatures/${base}.pem" \
    --certificate-identity-regexp "^https://github.com/${REPO}/.github/workflows/release.yml@refs/(heads|tags)/.*$" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
    "$artifact"
done
```

Verification succeeds only if signature, certificate identity, and OIDC issuer all match.

## 3. Inspect SBOMs

SBOMs are CycloneDX JSON files. You can inspect them directly with `jq` or import into an SBOM tool.

```bash
jq '.metadata.component.name, .components | length' sbom/sbom-crates.cdx.json
```

## 4. Verify build provenance attestation

GitHub release builds publish SLSA provenance subjects for:

- `dist/*`
- `crates/*.crate`
- `sbom/*.json`
- `checksums/SHA256SUMS.txt`

Use GitHub's attestation UI for the workflow run, or the GitHub CLI:

```bash
gh attestation verify <artifact-path-or-uri> --repo OWNER/REPO
```

Refer to the workflow definition in `.github/workflows/release.yml` for exact subject paths.

## 5. crates.io publication controls

The release workflow publishes only the core Rust crates (`cpd-core`, `cpd-costs`, `cpd-preprocess`, `cpd-offline`, `cpd-online`, `cpd-doctor`) and enforces SHA256 alignment against signed release `.crate` assets before upload.

For full publish policy, gating, auth fallback, and rerun guidance, see `cpd/docs/crates_publish.md`.

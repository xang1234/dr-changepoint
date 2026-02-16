# Versioning, Deprecation, and Schema Compatibility

This document is the source of truth for compatibility policy in `cpd-rs`.

## Scope

This policy covers:

- Rust public APIs in workspace crates.
- Python public APIs in `cpd`.
- Serialized JSON artifacts (result/config/checkpoint payloads).

## SemVer Policy

- During `0.x`, iteration may be faster, but avoid gratuitous breaking changes.
- At `1.0.0`, freeze the public Python API and Rust core trait contracts.
- Any planned break must include a migration path and release-note callout.

## Deprecation Policy

Deprecation uses a two-step policy:

1. Deprecate in minor release `N`.
2. Preserve behavior through `N`.
3. Remove in major release `N+1`.

Every deprecation must include:

- A release-note entry.
- A migration note that shows the replacement path.

## JSON Schema Versioning Policy

### Canonical marker in `0.x`

For offline result payloads in `0.x`, the canonical schema marker is:

- `diagnostics.schema_version`

Current persisted offline result contract is schema version `1`.

### Top-level envelope

A top-level envelope marker for result payloads is planned as a future migration
for `1.x` contracts. Until then, readers and writers should rely on
`diagnostics.schema_version`.

### Config payload contract in `0.x` (`pipeline_spec.v0`)

`pipeline_spec.v0` now includes an explicit optional
`payload.preprocess` object. Canonical preprocess stage keys are:

- `detrend`
- `deseasonalize`
- `winsorize`
- `robust_scale`

Method variants:

- `detrend`: `{"method":"linear"}` or
  `{"method":"polynomial","degree":<int>=1}`
- `deseasonalize`: `{"method":"differencing","period":<int>=1}` or
  `{"method":"stl_like","period":<int>=2}`

Parameter defaults and validation semantics:

- `winsorize` defaults to `lower_quantile=0.01`, `upper_quantile=0.99`,
  with required invariant `0.0 <= lower < upper <= 1.0`.
- `robust_scale` defaults to `mad_epsilon=1e-9`,
  `normal_consistency=1.4826`; both must be finite and `> 0`.
- Unknown keys inside `payload.preprocess` and inside each preprocess stage
  are rejected by runtime parsers.

### Unknown fields

- Readers should ignore unknown fields by default.
- Writers should preserve unknown fields where roundtrip APIs exist.

## N-1 Read Compatibility Policy

- Readers must accept the current schema version `N` and previous schema version
  `N-1`.
- This issue establishes the policy and CI fixtures; runtime migration shims for
  future schema bumps are tracked as follow-up implementation work.

## Checkpoint Compatibility Policy

Required envelope fields:

- `detector_id`
- `state_schema_version`
- `engine_fingerprint`
- `created_at_ns`
- `payload_codec`
- `payload_crc32`
- `payload`

Compatibility rules:

- Runtime writers emit v1 canonical envelopes (`state_schema_version=1`).
- Runtime readers accept v1 and legacy v0 (`schema_version=0`) envelopes
  under the N-1 read compatibility policy.
- Checksum mismatch/corruption must fail fast with `CpdError::InvalidInput`.
- Unsupported checkpoint schema versions must fail explicitly with actionable
  errors and migration guidance.

## Migration Process

When bumping any schema version, create a migration record from:

- `cpd/docs/templates/schema_migration.md`

Include this record in PR/release notes and add migration fixtures in CI.

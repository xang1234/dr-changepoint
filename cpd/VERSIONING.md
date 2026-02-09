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

### Unknown fields

- Readers should ignore unknown fields by default.
- Writers should preserve unknown fields where roundtrip APIs exist.

## N-1 Read Compatibility Policy

- Readers must accept the current schema version `N` and previous schema version
  `N-1`.
- This issue establishes the policy and CI fixtures; runtime migration shims for
  future schema bumps are tracked as follow-up implementation work.

## Checkpoint Compatibility Policy (Provisional)

Online checkpoint payloads are currently policy-level while online checkpoint
serialization is under development.

Required envelope fields:

- `schema_version`
- `detector_id`
- `engine_version`
- `created_at_ns`
- `payload_codec`
- `payload_crc32`
- `payload`

Compatibility rules:

- Apply the same N-1 read policy used for config/result payloads.
- Checksum mismatch/corruption must fail fast with `CpdError`.
- Unsupported checkpoint schema versions must fail explicitly with actionable
  errors.

## Migration Process

When bumping any schema version, create a migration record from:

- `cpd/docs/templates/schema_migration.md`

Include this record in PR/release notes and add migration fixtures in CI.

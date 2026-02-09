# Schema Migration Record Template

Use this template whenever a schema version is bumped.

## Summary

- Artifact: `<result|config|checkpoint>`
- From version: `<N-1>`
- To version: `<N>`
- Effective release: `<x.y.z>`
- Owner: `<team/person>`

## Breaking/Non-Breaking Assessment

- Classification: `<non-breaking|breaking>`
- Reason:
  - `<explain why this is or is not a breaking schema change>`

## Field-Level Changes

1. Added:
   - `<field path>`
2. Changed:
   - `<field path + old/new semantics>`
3. Removed/Deprecated:
   - `<field path + replacement>`

## Reader Compatibility

- N reader accepts:
  - `<N payload>`
  - `<N-1 payload>`
- Behavior on unsupported version:
  - `<error type + message contract>`
- Behavior on malformed/corrupt payload:
  - `<error type + message contract>`

## Migration Strategy

- Transform steps:
  1. `<step 1>`
  2. `<step 2>`
- Lossy fields (if any):
  - `<field + impact>`

## Fixture and CI Updates

- Added fixtures:
  - `<path/to/new_fixture>`
- Updated fixtures:
  - `<path/to/updated_fixture>`
- CI coverage:
  - `<tests and gates updated>`

## Release Notes Entry

`<short release-notes text for users>`

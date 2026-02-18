# `cpd-rs` BOCPD vs `bayesian_changepoint_detection` Parity Contract

This document defines the BOCPD online parity suite against
`hildensia/bayesian_changepoint_detection`.

## Scope

- Detector under test: `cpd.Bocpd`
- Model: `gaussian_nig`
- Hazard: constant, configured via `hazard=1/mean_run_length`
- Maximum state bound: `max_run_length=2000`
- Reference implementation:
  `https://github.com/hildensia/bayesian_changepoint_detection`
- Reference pin:
  `f3f8f03af0de7f4f98bd54c7ca0b5f6d0b0f6f8c`

No runtime API behavior is changed by this suite; this is a differential testing
contract only.

## Corpus Layout

Manifest path:
`/Users/admin/Documents/Work/claude-doctor-changepoint/cpd/python/tests/fixtures/parity/bocpd_manifest.v1.json`

Total cases: 12 (`full`), with 4 tagged for `smoke`.

Category minimums:

- `step_single`: 4
- `step_double`: 4
- `autocorrelated_single`: 4

## Comparison Logic

For each case:

1. Generate deterministic univariate signal from manifest `signal_spec`.
2. Run `cpd.Bocpd` and collect per-step `p_change`.
3. Run reference `online_changepoint_detection` using:
   - `constant_hazard`
   - `StudentT(alpha=1.0, beta=1.0, kappa=1.0, mu=0.0)`
4. Normalize reference run-length output to a 1D `p_change` vector of length `n`.
5. Extract top-`k` change-point peaks (`k = len(target_change_points)`) with:
   - local maxima candidates,
   - warmup skip = 5,
   - min separation = 8.
6. Compare cpd vs reference change-point sets under tolerance `±tolerance.index`.

## Thresholds

Suite-level gates:

- `tolerant_rate >= 0.85`
- `mean_jaccard >= 0.80`
- `median_primary_peak_delta <= 10`

Where:

- `tolerant_rate`: share of cases with one-to-one tolerance matches.
- `mean_jaccard`: tolerance-adjusted Jaccard similarity mean.
- `median_primary_peak_delta`: median absolute index delta between top-ranked
  cpd and reference peaks.

## Profiles and Env Vars

- `CPD_BOCPD_PARITY_PROFILE=smoke|full`
- `CPD_BOCPD_PARITY_REPORT_OUT=/path/to/report.json` (optional)

## CI Policy

- PR checks: not wired (no new PR gate).
- Nightly: job `python-bocpd-bayesian-parity-full` runs full profile and uploads
  `nightly-bocpd-bayesian-parity-report`.

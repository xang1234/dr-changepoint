from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

_ALLOWED_PROFILES = {"smoke", "full"}
_WARMUP_SKIP = 5
_MIN_SEPARATION = 8


@dataclass(frozen=True)
class BocpdParityCase:
    id: str
    category: str
    expected_behavior: str
    signal_spec: dict[str, Any]
    hazard_mean_run_length: float
    target_change_points: tuple[int, ...]
    tolerance_index: int
    profile_tags: tuple[str, ...]


@dataclass(frozen=True)
class BocpdParityResult:
    case_id: str
    category: str
    tolerance_index: int
    target_change_points: tuple[int, ...]
    cpd_change_points: tuple[int, ...]
    reference_change_points: tuple[int, ...]
    cpd_primary_peak: int | None
    reference_primary_peak: int | None
    primary_peak_delta: int | None
    tolerant_match: bool
    tolerant_matches: int
    tolerant_jaccard: float


class ManifestValidationError(ValueError):
    pass


def load_manifest(path: Path) -> list[BocpdParityCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestValidationError("manifest root must be an object")
    if payload.get("manifest_version") != 1:
        raise ManifestValidationError(
            f"manifest_version must be 1, got {payload.get('manifest_version')!r}"
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestValidationError("manifest.cases must be a non-empty list")

    out: list[BocpdParityCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ManifestValidationError(f"case at index {index} must be an object")
        case = _parse_case(raw_case, index=index)
        if case.id in seen_ids:
            raise ManifestValidationError(f"duplicate case id {case.id!r}")
        seen_ids.add(case.id)
        out.append(case)
    return out


def _parse_case(raw_case: dict[str, Any], index: int) -> BocpdParityCase:
    required = {
        "id",
        "category",
        "expected_behavior",
        "signal_spec",
        "hazard_mean_run_length",
        "target_change_points",
        "tolerance",
        "profile_tags",
    }
    missing = sorted(required - set(raw_case))
    if missing:
        raise ManifestValidationError(
            f"case index {index} missing required fields: {', '.join(missing)}"
        )

    case_id = raw_case["id"]
    if not isinstance(case_id, str) or not case_id:
        raise ManifestValidationError(f"case index {index} has invalid id {case_id!r}")

    category = raw_case["category"]
    if not isinstance(category, str) or not category:
        raise ManifestValidationError(f"case {case_id!r} category must be a non-empty string")

    expected_behavior = raw_case["expected_behavior"]
    if not isinstance(expected_behavior, str) or not expected_behavior.strip():
        raise ManifestValidationError(f"case {case_id!r} must define expected_behavior")

    signal_spec = raw_case["signal_spec"]
    if not isinstance(signal_spec, dict):
        raise ManifestValidationError(f"case {case_id!r} signal_spec must be an object")
    kind = signal_spec.get("kind")
    if kind not in {"piecewise_constant", "ar1_piecewise_mean"}:
        raise ManifestValidationError(
            f"case {case_id!r} has unsupported signal_spec.kind {kind!r}"
        )

    mean_run_length = raw_case["hazard_mean_run_length"]
    if not isinstance(mean_run_length, (int, float)) or not math.isfinite(mean_run_length):
        raise ManifestValidationError(
            f"case {case_id!r} hazard_mean_run_length must be a finite number"
        )
    if float(mean_run_length) <= 1.0:
        raise ManifestValidationError(
            f"case {case_id!r} hazard_mean_run_length must be > 1.0"
        )

    target_change_points = raw_case["target_change_points"]
    if not isinstance(target_change_points, list) or not target_change_points:
        raise ManifestValidationError(
            f"case {case_id!r} target_change_points must be a non-empty list"
        )
    try:
        cps = tuple(int(point) for point in target_change_points)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"case {case_id!r} target_change_points must contain integers"
        ) from exc
    if any(point < 0 for point in cps):
        raise ManifestValidationError(
            f"case {case_id!r} target_change_points must be >= 0"
        )
    if tuple(sorted(set(cps))) != cps:
        raise ManifestValidationError(
            f"case {case_id!r} target_change_points must be strictly increasing and unique"
        )

    tolerance = raw_case["tolerance"]
    if not isinstance(tolerance, dict):
        raise ManifestValidationError(f"case {case_id!r} tolerance must be an object")
    tolerance_index = tolerance.get("index")
    if not isinstance(tolerance_index, int) or tolerance_index < 0:
        raise ManifestValidationError(
            f"case {case_id!r} tolerance.index must be an integer >= 0"
        )

    profile_tags = raw_case["profile_tags"]
    if not isinstance(profile_tags, list) or not profile_tags:
        raise ManifestValidationError(f"case {case_id!r} profile_tags must be non-empty")
    profiles = tuple(str(tag) for tag in profile_tags)
    unknown = sorted(set(profiles) - _ALLOWED_PROFILES)
    if unknown:
        raise ManifestValidationError(
            f"case {case_id!r} has unsupported profile tags: {', '.join(unknown)}"
        )

    return BocpdParityCase(
        id=case_id,
        category=category,
        expected_behavior=expected_behavior,
        signal_spec=signal_spec,
        hazard_mean_run_length=float(mean_run_length),
        target_change_points=cps,
        tolerance_index=tolerance_index,
        profile_tags=profiles,
    )


def category_counts(cases: list[BocpdParityCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return counts


def select_cases(cases: list[BocpdParityCase], profile: str) -> list[BocpdParityCase]:
    if profile not in _ALLOWED_PROFILES:
        raise ValueError(f"unsupported parity profile {profile!r}")
    return [case for case in cases if profile in case.profile_tags]


def generate_signal(case: BocpdParityCase) -> np.ndarray:
    spec = case.signal_spec
    kind = spec["kind"]
    if kind == "piecewise_constant":
        values = _piecewise_constant_signal(spec)
    elif kind == "ar1_piecewise_mean":
        values = _ar1_piecewise_signal(spec)
    else:
        raise ValueError(f"unsupported signal_spec.kind {kind!r} for case {case.id}")

    if not np.isfinite(values).all():
        raise ValueError(f"case {case.id} generated non-finite values")
    return values.astype(np.float64, copy=False)


def _piecewise_constant_signal(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("piecewise_constant requires non-empty segments")

    values = np.concatenate(
        [
            np.full(int(segment["length"]), float(segment["mean"]), dtype=np.float64)
            for segment in segments
        ]
    )

    noise = spec.get("noise", {"distribution": "normal", "std": 0.0})
    distribution = str(noise.get("distribution", "normal")).lower()
    std = float(noise.get("std", 0.0))
    if distribution != "normal":
        raise ValueError(f"unsupported noise distribution {distribution!r}")
    return values + rng.normal(loc=0.0, scale=std, size=values.shape[0])


def _ar1_piecewise_signal(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    phi = float(spec.get("phi", 0.4))
    sigma = float(spec.get("sigma", 1.0))
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("ar1_piecewise_mean requires non-empty segments")

    means = np.concatenate(
        [
            np.full(int(segment["length"]), float(segment["mean"]), dtype=np.float64)
            for segment in segments
        ]
    )
    out = np.empty_like(means)
    out[0] = means[0] + rng.normal(scale=sigma)
    prev_mean = means[0]
    for idx in range(1, means.shape[0]):
        mean = means[idx]
        innovation = rng.normal(scale=sigma)
        out[idx] = mean + phi * (out[idx - 1] - prev_mean) + innovation
        prev_mean = mean
    return out


def run_cpd_case(values: np.ndarray, hazard_mean_run_length: float) -> np.ndarray:
    cpd = importlib.import_module("cpd")
    detector = cpd.Bocpd(
        model="gaussian_nig",
        hazard=1.0 / float(hazard_mean_run_length),
        max_run_length=2_000,
    )
    steps = detector.update_many(values)
    p_change = np.asarray([float(step.p_change) for step in steps], dtype=np.float64)
    if p_change.shape[0] != values.shape[0]:
        raise ValueError(
            f"cpd p_change length mismatch: got {p_change.shape[0]}, expected {values.shape[0]}"
        )
    return p_change


def run_reference_case(values: np.ndarray, hazard_mean_run_length: float) -> np.ndarray:
    oncd = importlib.import_module(
        "bayesian_changepoint_detection.online_changepoint_detection"
    )
    hazard_functions = importlib.import_module(
        "bayesian_changepoint_detection.hazard_functions"
    )
    online_likelihoods = importlib.import_module(
        "bayesian_changepoint_detection.online_likelihoods"
    )

    hazard = partial(hazard_functions.constant_hazard, float(hazard_mean_run_length))
    observation = online_likelihoods.StudentT(alpha=1.0, beta=1.0, kappa=1.0, mu=0.0)
    raw = oncd.online_changepoint_detection(np.asarray(values, dtype=np.float64), hazard, observation)
    return _normalize_reference_p_change(raw, n_samples=values.shape[0])


def _normalize_reference_p_change(raw: Any, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    matrix_like: np.ndarray | None = None
    if isinstance(raw, np.ndarray):
        matrix_like = np.asarray(raw, dtype=np.float64)
    elif isinstance(raw, (list, tuple)):
        for candidate in raw:
            if isinstance(candidate, np.ndarray):
                matrix_like = np.asarray(candidate, dtype=np.float64)
                break
            if isinstance(candidate, (list, tuple)):
                try:
                    candidate_arr = np.asarray(candidate, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if candidate_arr.ndim >= 1:
                    matrix_like = candidate_arr
                    break

    if matrix_like is None:
        raise ValueError("unable to locate reference BOCPD run-length matrix in output")

    if matrix_like.ndim == 1:
        return _fit_vector_length(matrix_like, n_samples=n_samples)
    if matrix_like.ndim != 2:
        raise ValueError(f"unsupported reference output ndim={matrix_like.ndim}")

    row0 = np.asarray(matrix_like[0, :], dtype=np.float64)
    col0 = np.asarray(matrix_like[:, 0], dtype=np.float64)
    row_fit = _fit_vector_length(row0, n_samples=n_samples)
    col_fit = _fit_vector_length(col0, n_samples=n_samples)

    row_dynamic = float(np.max(row_fit) - np.min(row_fit))
    col_dynamic = float(np.max(col_fit) - np.min(col_fit))
    selected = row_fit if row_dynamic >= col_dynamic else col_fit
    return np.clip(selected, 0.0, 1.0)


def _fit_vector_length(vector: np.ndarray, n_samples: int) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    length = vec.shape[0]
    if length == n_samples:
        return vec
    if length == n_samples + 1:
        return vec[1:]
    if length > n_samples:
        return vec[:n_samples]
    if length == 0:
        raise ValueError("reference vector is empty")

    out = np.empty(n_samples, dtype=np.float64)
    out[:length] = vec
    out[length:] = vec[-1]
    return out


def extract_top_k_change_points(
    p_change: np.ndarray,
    k: int,
    warmup_skip: int = _WARMUP_SKIP,
    min_separation: int = _MIN_SEPARATION,
) -> tuple[int, ...]:
    if k <= 0:
        return ()
    values = np.asarray(p_change, dtype=np.float64).reshape(-1)
    n = values.shape[0]
    start = max(0, min(n, int(warmup_skip)))

    candidates: list[int] = []
    for idx in range(start, n):
        left = values[idx - 1] if idx > 0 else -np.inf
        right = values[idx + 1] if idx + 1 < n else -np.inf
        if values[idx] >= left and values[idx] >= right:
            candidates.append(idx)

    if not candidates:
        candidates = list(range(start, n))

    ranked = sorted(candidates, key=lambda idx: (-values[idx], idx))
    chosen: list[int] = []
    for idx in ranked:
        if all(abs(idx - selected) >= min_separation for selected in chosen):
            chosen.append(idx)
        if len(chosen) == k:
            break

    if len(chosen) < k:
        for idx in ranked:
            if idx in chosen:
                continue
            chosen.append(idx)
            if len(chosen) == k:
                break

    return tuple(chosen[:k])


def compare_change_points(
    left: tuple[int, ...] | list[int],
    right: tuple[int, ...] | list[int],
    tolerance: int,
) -> tuple[bool, int, float]:
    left_sorted = tuple(sorted(int(x) for x in left))
    right_sorted = tuple(sorted(int(x) for x in right))
    i = 0
    j = 0
    matches = 0
    while i < len(left_sorted) and j < len(right_sorted):
        delta = left_sorted[i] - right_sorted[j]
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < -tolerance:
            i += 1
        else:
            j += 1

    union = len(left_sorted) + len(right_sorted) - matches
    jaccard = 1.0 if union == 0 else matches / union
    tolerant_match = matches == len(left_sorted) == len(right_sorted)
    return tolerant_match, matches, jaccard


def run_parity_suite(cases: list[BocpdParityCase]) -> list[BocpdParityResult]:
    results: list[BocpdParityResult] = []
    for case in cases:
        values = generate_signal(case)
        cpd_p_change = run_cpd_case(values, hazard_mean_run_length=case.hazard_mean_run_length)
        reference_p_change = run_reference_case(
            values, hazard_mean_run_length=case.hazard_mean_run_length
        )
        target_k = len(case.target_change_points)

        cpd_ranked = extract_top_k_change_points(cpd_p_change, k=target_k)
        reference_ranked = extract_top_k_change_points(reference_p_change, k=target_k)
        cpd_sorted = tuple(sorted(cpd_ranked))
        reference_sorted = tuple(sorted(reference_ranked))
        tolerant_match, tolerant_matches, tolerant_jaccard = compare_change_points(
            cpd_sorted,
            reference_sorted,
            tolerance=case.tolerance_index,
        )

        cpd_primary = cpd_ranked[0] if cpd_ranked else None
        reference_primary = reference_ranked[0] if reference_ranked else None
        primary_delta = (
            abs(cpd_primary - reference_primary)
            if cpd_primary is not None and reference_primary is not None
            else None
        )

        results.append(
            BocpdParityResult(
                case_id=case.id,
                category=case.category,
                tolerance_index=case.tolerance_index,
                target_change_points=case.target_change_points,
                cpd_change_points=cpd_sorted,
                reference_change_points=reference_sorted,
                cpd_primary_peak=cpd_primary,
                reference_primary_peak=reference_primary,
                primary_peak_delta=primary_delta,
                tolerant_match=tolerant_match,
                tolerant_matches=tolerant_matches,
                tolerant_jaccard=tolerant_jaccard,
            )
        )

    return results


def summarize_results(results: list[BocpdParityResult]) -> dict[str, float]:
    if not results:
        raise ValueError("cannot summarize empty results")
    total = float(len(results))
    tolerant_pass = float(sum(1 for result in results if result.tolerant_match))
    mean_jaccard = float(sum(result.tolerant_jaccard for result in results) / total)
    deltas = [result.primary_peak_delta for result in results if result.primary_peak_delta is not None]
    median_primary_peak_delta = float(median(deltas)) if deltas else float("nan")
    return {
        "total": total,
        "tolerant_pass": tolerant_pass,
        "tolerant_rate": tolerant_pass / total,
        "mean_jaccard": mean_jaccard,
        "median_primary_peak_delta": median_primary_peak_delta,
    }


def results_to_jsonable(results: list[BocpdParityResult]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": result.case_id,
            "category": result.category,
            "tolerance_index": result.tolerance_index,
            "target_change_points": list(result.target_change_points),
            "cpd_change_points": list(result.cpd_change_points),
            "reference_change_points": list(result.reference_change_points),
            "cpd_primary_peak": result.cpd_primary_peak,
            "reference_primary_peak": result.reference_primary_peak,
            "primary_peak_delta": result.primary_peak_delta,
            "tolerant_match": result.tolerant_match,
            "tolerant_matches": result.tolerant_matches,
            "tolerant_jaccard": result.tolerant_jaccard,
        }
        for result in results
    ]

#!/usr/bin/env python3
"""Gate schema and fixture compatibility contracts for CPD-kvd.7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
CRC32_RE = re.compile(r"^[0-9a-f]{8}$")

RESULT_REQUIRED_ROOT = {"breakpoints", "change_points", "diagnostics"}
RESULT_DIAGNOSTICS_REQUIRED = {
    "n",
    "d",
    "schema_version",
    "algorithm",
    "cost_model",
    "repro_mode",
}
SEGMENT_REQUIRED = {"start", "end", "count", "missing_count"}
CONFIG_REQUIRED = {"schema_version", "kind", "payload"}
CHECKPOINT_REQUIRED = {
    "schema_version",
    "detector_id",
    "engine_version",
    "created_at_ns",
    "payload_codec",
    "payload_crc32",
    "payload",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
CPD_ROOT = REPO_ROOT / "cpd"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _as_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"missing file: {path}")
    except OSError as exc:  # pragma: no cover - defensive CI guard
        raise ValueError(f"failed to read {path}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc


def _require_keys(payload: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_result_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == SCHEMA_DRAFT_2020_12,
        "result schema must declare JSON Schema Draft 2020-12",
    )
    _require(schema.get("type") == "object", "result schema root type must be object")
    _require(
        schema.get("additionalProperties") is True,
        "result schema must allow additionalProperties=true",
    )

    required = set(_as_list(schema.get("required"), "result schema.required"))
    missing_required = RESULT_REQUIRED_ROOT - required
    _require(
        not missing_required,
        f"result schema.required missing: {', '.join(sorted(missing_required))}",
    )

    properties = _as_dict(schema.get("properties"), "result schema.properties")
    for field in RESULT_REQUIRED_ROOT.union({"scores", "segments"}):
        _require(field in properties, f"result schema.properties missing '{field}'")

    defs = _as_dict(schema.get("$defs"), "result schema.$defs")
    diagnostics = _as_dict(defs.get("diagnostics"), "result schema.$defs.diagnostics")
    diag_required = set(_as_list(diagnostics.get("required"), "result diagnostics.required"))
    missing_diag_required = RESULT_DIAGNOSTICS_REQUIRED - diag_required
    _require(
        not missing_diag_required,
        "result diagnostics.required missing: "
        + ", ".join(sorted(missing_diag_required)),
    )

    diag_properties = _as_dict(
        diagnostics.get("properties"), "result diagnostics.properties"
    )
    schema_version = _as_dict(
        diag_properties.get("schema_version"), "result diagnostics.schema_version"
    )
    _require(
        schema_version.get("type") == "integer",
        "result diagnostics.schema_version type must be integer",
    )


def validate_config_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == SCHEMA_DRAFT_2020_12,
        "config schema must declare JSON Schema Draft 2020-12",
    )
    _require(schema.get("type") == "object", "config schema root type must be object")
    _require(
        schema.get("additionalProperties") is True,
        "config schema must allow additionalProperties=true",
    )

    required = set(_as_list(schema.get("required"), "config schema.required"))
    _require_keys({key: True for key in required}, CONFIG_REQUIRED, "config schema.required")

    properties = _as_dict(schema.get("properties"), "config schema.properties")
    schema_version = _as_dict(properties.get("schema_version"), "config schema.schema_version")
    kind = _as_dict(properties.get("kind"), "config schema.kind")
    _require(
        schema_version.get("const") == 0,
        "config schema.schema_version const must be 0",
    )
    _require(
        kind.get("const") == "pipeline_spec",
        "config schema.kind const must be 'pipeline_spec'",
    )


def validate_checkpoint_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == SCHEMA_DRAFT_2020_12,
        "checkpoint schema must declare JSON Schema Draft 2020-12",
    )
    _require(
        schema.get("type") == "object", "checkpoint schema root type must be object"
    )
    _require(
        schema.get("additionalProperties") is True,
        "checkpoint schema must allow additionalProperties=true",
    )

    required = set(_as_list(schema.get("required"), "checkpoint schema.required"))
    missing_required = CHECKPOINT_REQUIRED - required
    _require(
        not missing_required,
        "checkpoint schema.required missing: " + ", ".join(sorted(missing_required)),
    )

    properties = _as_dict(schema.get("properties"), "checkpoint schema.properties")
    schema_version = _as_dict(
        properties.get("schema_version"), "checkpoint schema.schema_version"
    )
    payload_crc32 = _as_dict(
        properties.get("payload_crc32"), "checkpoint schema.payload_crc32"
    )
    _require(
        schema_version.get("const") == 0,
        "checkpoint schema.schema_version const must be 0",
    )
    _require(
        payload_crc32.get("pattern") == CRC32_RE.pattern,
        "checkpoint schema.payload_crc32 pattern must be ^[0-9a-f]{8}$",
    )


def validate_result_fixture(payload: dict[str, Any]) -> None:
    _require_keys(payload, RESULT_REQUIRED_ROOT, "result fixture")

    breakpoints = _as_list(payload.get("breakpoints"), "result fixture.breakpoints")
    change_points = _as_list(payload.get("change_points"), "result fixture.change_points")
    diagnostics = _as_dict(payload.get("diagnostics"), "result fixture.diagnostics")
    _require_keys(diagnostics, RESULT_DIAGNOSTICS_REQUIRED, "result fixture.diagnostics")

    n = diagnostics.get("n")
    d = diagnostics.get("d")
    schema_version = diagnostics.get("schema_version")
    _require(isinstance(n, int) and n >= 0, "result fixture diagnostics.n must be int >= 0")
    _require(isinstance(d, int) and d >= 0, "result fixture diagnostics.d must be int >= 0")
    _require(
        isinstance(schema_version, int) and schema_version == 1,
        "result fixture diagnostics.schema_version must be exactly 1",
    )

    _require(
        isinstance(diagnostics.get("algorithm"), str)
        and bool(diagnostics.get("algorithm")),
        "result fixture diagnostics.algorithm must be a non-empty string",
    )
    _require(
        isinstance(diagnostics.get("cost_model"), str)
        and bool(diagnostics.get("cost_model")),
        "result fixture diagnostics.cost_model must be a non-empty string",
    )
    _require(
        isinstance(diagnostics.get("repro_mode"), str)
        and bool(diagnostics.get("repro_mode")),
        "result fixture diagnostics.repro_mode must be a non-empty string",
    )

    _require(
        all(isinstance(bp, int) and bp >= 0 for bp in breakpoints),
        "result fixture.breakpoints must contain int >= 0",
    )
    _require(
        all(
            breakpoints[idx] > breakpoints[idx - 1]
            for idx in range(1, len(breakpoints))
        ),
        "result fixture.breakpoints must be strictly increasing",
    )
    if n == 0:
        _require(not breakpoints, "result fixture.breakpoints must be empty when n=0")
    else:
        _require(breakpoints, "result fixture.breakpoints must be non-empty when n>0")
        _require(
            breakpoints[-1] == n,
            "result fixture.breakpoints must include n as final element",
        )
        _require(
            all(bp > 0 for bp in breakpoints),
            "result fixture.breakpoints must be > 0 when n>0",
        )

    _require(
        all(isinstance(cp, int) and cp >= 0 for cp in change_points),
        "result fixture.change_points must contain int >= 0",
    )
    expected_change_points = [bp for bp in breakpoints if bp < n]
    _require(
        change_points == expected_change_points,
        "result fixture.change_points must equal breakpoints excluding n",
    )

    scores = payload.get("scores")
    if scores is not None:
        scores_array = _as_list(scores, "result fixture.scores")
        _require(
            all(_is_number(score) for score in scores_array),
            "result fixture.scores must contain numbers",
        )
        _require(
            len(scores_array) == len(change_points),
            "result fixture.scores length must equal change_points length",
        )

    segments = payload.get("segments")
    if segments is None:
        return

    segments_array = _as_list(segments, "result fixture.segments")
    _require(
        len(segments_array) == len(breakpoints),
        "result fixture.segments length must equal breakpoints length",
    )

    expected_start = 0
    for idx, segment_value in enumerate(segments_array):
        segment = _as_dict(segment_value, f"result fixture.segments[{idx}]")
        _require_keys(segment, SEGMENT_REQUIRED, f"result fixture.segments[{idx}]")
        start = segment.get("start")
        end = segment.get("end")
        count = segment.get("count")
        missing_count = segment.get("missing_count")

        _require(
            isinstance(start, int) and isinstance(end, int) and start >= 0 and end >= start,
            f"result fixture.segments[{idx}] start/end must satisfy 0 <= start <= end",
        )
        _require(
            isinstance(count, int) and count == (end - start),
            f"result fixture.segments[{idx}].count must equal end-start",
        )
        _require(
            isinstance(missing_count, int) and 0 <= missing_count <= count,
            f"result fixture.segments[{idx}].missing_count must be within [0, count]",
        )

        expected_end = breakpoints[idx]
        _require(
            start == expected_start and end == expected_end,
            f"result fixture.segments[{idx}] boundaries must match breakpoints",
        )
        expected_start = expected_end

        for field in ("mean", "variance"):
            value = segment.get(field)
            if value is None:
                continue
            series = _as_list(value, f"result fixture.segments[{idx}].{field}")
            _require(
                all(_is_number(item) for item in series),
                f"result fixture.segments[{idx}].{field} must contain numbers",
            )
            if d > 0:
                _require(
                    len(series) == d,
                    f"result fixture.segments[{idx}].{field} length must equal diagnostics.d",
                )


def validate_config_fixture(payload: dict[str, Any]) -> None:
    _require_keys(payload, CONFIG_REQUIRED, "config fixture")
    _require(
        payload.get("schema_version") == 0, "config fixture schema_version must be 0"
    )
    _require(
        payload.get("kind") == "pipeline_spec",
        "config fixture kind must be 'pipeline_spec'",
    )
    _as_dict(payload.get("payload"), "config fixture.payload")


def validate_checkpoint_fixture(payload: dict[str, Any]) -> None:
    _require_keys(payload, CHECKPOINT_REQUIRED, "checkpoint fixture")
    _require(
        payload.get("schema_version") == 0,
        "checkpoint fixture schema_version must be 0",
    )
    _require(
        isinstance(payload.get("detector_id"), str) and bool(payload.get("detector_id")),
        "checkpoint fixture detector_id must be a non-empty string",
    )
    _require(
        isinstance(payload.get("engine_version"), str)
        and bool(payload.get("engine_version")),
        "checkpoint fixture engine_version must be a non-empty string",
    )
    created_at_ns = payload.get("created_at_ns")
    _require(
        isinstance(created_at_ns, int) and created_at_ns >= 0,
        "checkpoint fixture created_at_ns must be int >= 0",
    )
    _require(
        isinstance(payload.get("payload_codec"), str) and bool(payload.get("payload_codec")),
        "checkpoint fixture payload_codec must be a non-empty string",
    )
    payload_crc32 = payload.get("payload_crc32")
    _require(
        isinstance(payload_crc32, str) and bool(CRC32_RE.fullmatch(payload_crc32)),
        "checkpoint fixture payload_crc32 must match ^[0-9a-f]{8}$",
    )
    _as_dict(payload.get("payload"), "checkpoint fixture.payload")


def _validate_required_coverage(
    fixture: dict[str, Any], schema: dict[str, Any], context: str
) -> None:
    required = _as_list(schema.get("required"), f"{context} schema.required")
    for key in required:
        _require(key in fixture, f"{context} fixture missing required schema field '{key}'")


def validate_repo(repo_root: Path) -> list[str]:
    cpd_root = repo_root / "cpd"
    result_schema_path = (
        cpd_root / "schemas" / "result" / "offline_change_point_result.v1.schema.json"
    )
    config_schema_path = cpd_root / "schemas" / "config" / "pipeline_spec.v0.schema.json"
    checkpoint_schema_path = (
        cpd_root
        / "schemas"
        / "checkpoint"
        / "online_detector_checkpoint.v0.schema.json"
    )
    result_fixture_path = (
        cpd_root / "crates" / "cpd-python" / "tests" / "fixtures" / "offline_result_v1.json"
    )
    config_fixture_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "schemas"
        / "config"
        / "pipeline_spec.v0.stub.json"
    )
    checkpoint_fixture_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "schemas"
        / "checkpoint"
        / "online_detector_checkpoint.v0.stub.json"
    )

    errors: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    json_objects = {
        "result_schema": result_schema_path,
        "config_schema": config_schema_path,
        "checkpoint_schema": checkpoint_schema_path,
        "result_fixture": result_fixture_path,
        "config_fixture": config_fixture_path,
        "checkpoint_fixture": checkpoint_fixture_path,
    }

    for label, path in json_objects.items():
        try:
            loaded[label] = _as_dict(_read_json(path), str(path))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        return errors

    try:
        validate_result_schema(loaded["result_schema"])
    except ValueError as exc:
        errors.append(f"{result_schema_path}: {exc}")

    try:
        validate_config_schema(loaded["config_schema"])
    except ValueError as exc:
        errors.append(f"{config_schema_path}: {exc}")

    try:
        validate_checkpoint_schema(loaded["checkpoint_schema"])
    except ValueError as exc:
        errors.append(f"{checkpoint_schema_path}: {exc}")

    try:
        validate_result_fixture(loaded["result_fixture"])
    except ValueError as exc:
        errors.append(f"{result_fixture_path}: {exc}")

    try:
        validate_config_fixture(loaded["config_fixture"])
    except ValueError as exc:
        errors.append(f"{config_fixture_path}: {exc}")

    try:
        validate_checkpoint_fixture(loaded["checkpoint_fixture"])
    except ValueError as exc:
        errors.append(f"{checkpoint_fixture_path}: {exc}")

    if errors:
        return errors

    try:
        _validate_required_coverage(
            loaded["result_fixture"], loaded["result_schema"], "result"
        )
        result_defs = _as_dict(loaded["result_schema"].get("$defs"), "result schema.$defs")
        result_diag_schema = _as_dict(
            result_defs.get("diagnostics"), "result schema.$defs.diagnostics"
        )
        result_diag = _as_dict(
            loaded["result_fixture"].get("diagnostics"), "result fixture.diagnostics"
        )
        _validate_required_coverage(result_diag, result_diag_schema, "result diagnostics")
        _validate_required_coverage(
            loaded["config_fixture"], loaded["config_schema"], "config"
        )
        _validate_required_coverage(
            loaded["checkpoint_fixture"], loaded["checkpoint_schema"], "checkpoint"
        )
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate schema + fixture compatibility contracts."
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root containing cpd/ (default: auto-detected).",
    )
    args = parser.parse_args(argv)

    errors = validate_repo(Path(args.repo_root))
    if errors:
        print("BLOCK: schema/fixture contract checks failed")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("PASS: schema/fixture contracts validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

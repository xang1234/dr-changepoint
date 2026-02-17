import json
from pathlib import Path


CPD_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_PATH = (
    CPD_ROOT / "schemas" / "result" / "offline_change_point_result.v1.schema.json"
)
RESULT_V1_FIXTURE_PATH = (
    CPD_ROOT / "tests" / "fixtures" / "migrations" / "result" / "offline_result.v1.json"
)
RESULT_V2_FIXTURE_PATH = (
    CPD_ROOT
    / "tests"
    / "fixtures"
    / "migrations"
    / "result"
    / "offline_result.v2.additive.json"
)
PYTHON_V1_FIXTURE_PATH = (
    CPD_ROOT
    / "crates"
    / "cpd-python"
    / "tests"
    / "fixtures"
    / "offline_result_v1.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_result_schema_declares_required_contract_fields() -> None:
    schema = _load_json(RESULT_SCHEMA_PATH)

    assert schema["required"] == ["breakpoints", "change_points", "diagnostics"]
    diagnostics = schema["$defs"]["diagnostics"]
    assert diagnostics["required"] == [
        "n",
        "d",
        "schema_version",
        "algorithm",
        "cost_model",
        "repro_mode",
    ]


def test_migration_fixtures_cover_current_and_additive_compatibility() -> None:
    v1 = _load_json(RESULT_V1_FIXTURE_PATH)
    v2 = _load_json(RESULT_V2_FIXTURE_PATH)

    assert v1["diagnostics"]["schema_version"] == 1
    assert v2["diagnostics"]["schema_version"] == 2

    for payload in (v1, v2):
        assert "breakpoints" in payload
        assert "change_points" in payload
        assert "diagnostics" in payload
        assert "algorithm" in payload["diagnostics"]
        assert "cost_model" in payload["diagnostics"]
        assert "repro_mode" in payload["diagnostics"]


def test_python_result_fixture_matches_canonical_v1_fixture() -> None:
    python_v1 = _load_json(PYTHON_V1_FIXTURE_PATH)
    migration_v1 = _load_json(RESULT_V1_FIXTURE_PATH)

    assert python_v1 == migration_v1


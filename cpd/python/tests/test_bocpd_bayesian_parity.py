import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import bocpd_parity_harness as bh

if importlib.util.find_spec("bayesian_changepoint_detection") is None:
    pytest.skip(
        "bayesian_changepoint_detection is required for BOCPD parity tests",
        allow_module_level=True,
    )
if importlib.util.find_spec("cpd") is None:
    pytest.skip("cpd is required for BOCPD parity tests", allow_module_level=True)

import cpd as _cpd_import_check  # noqa: F401


MANIFEST_PATH = TESTS_DIR / "fixtures" / "parity" / "bocpd_manifest.v1.json"


def _required_category_minimums() -> dict[str, int]:
    return {
        "step_single": 4,
        "step_double": 4,
        "autocorrelated_single": 4,
    }


def test_bocpd_bayesian_parity_contract() -> None:
    cases = bh.load_manifest(MANIFEST_PATH)
    assert len(cases) >= 12

    counts = bh.category_counts(cases)
    for category, minimum in _required_category_minimums().items():
        assert counts.get(category, 0) >= minimum, f"insufficient cases for {category}"

    profile = os.environ.get("CPD_BOCPD_PARITY_PROFILE", "smoke").strip().lower()
    selected = bh.select_cases(cases, profile=profile)
    assert selected, f"no parity cases selected for profile={profile!r}"

    results = bh.run_parity_suite(selected)
    assert results, "BOCPD parity suite returned no results"

    summary = bh.summarize_results(results)
    assert summary["tolerant_rate"] >= 0.85
    assert summary["mean_jaccard"] >= 0.80
    assert not math.isnan(summary["median_primary_peak_delta"])
    assert summary["median_primary_peak_delta"] <= 10.0

    report_out = os.environ.get("CPD_BOCPD_PARITY_REPORT_OUT")
    if report_out:
        payload = {
            "profile": profile,
            "manifest_path": str(MANIFEST_PATH),
            "summary": summary,
            "results": bh.results_to_jsonable(results),
        }
        out_path = Path(report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

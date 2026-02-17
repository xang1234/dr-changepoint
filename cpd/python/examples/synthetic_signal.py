#!/usr/bin/env python3
"""Run MVP-A APIs on a synthetic piecewise-constant signal."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import cpd  # noqa: E402


def build_signal() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(40, dtype=np.float64),
            np.full(40, 8.0, dtype=np.float64),
            np.full(40, -4.0, dtype=np.float64),
        ]
    )


def main() -> int:
    values = build_signal()

    pelt = cpd.Pelt(model="l2", min_segment_len=2).fit(values).predict(n_bkps=2)
    binseg = cpd.Binseg(model="l2", min_segment_len=2).fit(values).predict(n_bkps=2)
    fpop = cpd.Fpop(min_segment_len=2).fit(values).predict(n_bkps=2)
    low = cpd.detect_offline(
        values,
        detector="pelt",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
        repro_mode="balanced",
    )

    print("pelt breakpoints:", pelt.breakpoints)
    print("binseg breakpoints:", binseg.breakpoints)
    print("fpop breakpoints:", fpop.breakpoints)
    print("detect_offline breakpoints:", low.breakpoints)
    print("detect_offline diagnostics.algorithm:", low.diagnostics.algorithm)
    print("detect_offline diagnostics.cost_model:", low.diagnostics.cost_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

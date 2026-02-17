#!/usr/bin/env python3
"""Plot detected breakpoints for a synthetic signal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import cpd  # noqa: E402

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for this script. Install with: python -m pip install matplotlib"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="/tmp/cpd_breakpoints.png",
        help="Output image path (default: /tmp/cpd_breakpoints.png)",
    )
    return parser.parse_args()


def build_signal() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(60, dtype=np.float64),
            np.full(60, 6.0, dtype=np.float64),
            np.full(60, -3.0, dtype=np.float64),
        ]
    )


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    values = build_signal()
    result = cpd.Pelt(model="l2", min_segment_len=2).fit(values).predict(n_bkps=2)
    matplotlib.use("Agg")
    fig = result.plot(values, title="MVP-A offline change-point detection")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

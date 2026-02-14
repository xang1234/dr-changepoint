#!/usr/bin/env python3
"""Gate wheel dependency reports for unexpected BLAS/LAPACK-style linkages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DENY_SUBSTRINGS = (
    "openblas",
    "blas",
    "lapack",
    "mkl",
    "atlas",
    "cblas",
)

LIB_NAME_PATTERN = re.compile(
    r"(?i)(?:^|[\\/\s:])([A-Za-z0-9_.+-]+\.(?:so(?:\.[0-9]+)*|dylib|dll))(?=$|[\s,:)])"
)


def parse_libraries(report_text: str) -> set[str]:
    """Extract shared library basenames from a wheel dependency report."""
    found: set[str] = set()
    for line in report_text.splitlines():
        for match in LIB_NAME_PATTERN.findall(line):
            found.add(Path(match).name.lower())
    return found


def contains_manylinux_tag(report_text: str) -> bool:
    return "manylinux" in report_text.lower()


def find_blocked_libraries(libraries: set[str]) -> list[str]:
    blocked = [
        lib
        for lib in sorted(libraries)
        if any(token in lib for token in DENY_SUBSTRINGS)
    ]
    return blocked


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if wheel dependency reports include blocked native libraries."
    )
    parser.add_argument(
        "--tool",
        choices=("auditwheel", "delocate", "delvewheel"),
        required=True,
        help="Dependency inspection tool that produced the report.",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to report text file, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--wheel",
        required=False,
        default="(unknown wheel)",
        help="Wheel path for logging.",
    )
    return parser.parse_args(argv)


def _load_report(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        report_text = _load_report(args.report)
    except OSError as exc:
        print(f"BLOCK: failed to load report for {args.wheel}: {exc}", file=sys.stderr)
        return 1

    libraries = parse_libraries(report_text)
    blocked = find_blocked_libraries(libraries)
    if blocked:
        print(
            "BLOCK: "
            f"{args.wheel} via {args.tool} references blocked libraries: {', '.join(blocked)}"
        )
        return 1

    if args.tool == "auditwheel" and not contains_manylinux_tag(report_text):
        print(f"BLOCK: {args.wheel} auditwheel output did not report a manylinux tag")
        return 1

    print(
        "PASS: "
        f"{args.wheel} via {args.tool} has no blocked BLAS/LAPACK dependencies "
        f"(detected libraries: {len(libraries)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

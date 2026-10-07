#!/usr/bin/env python
"""Enforce the coverage floors the documentation claims are enforced.

``coverage.py`` supports a single global ``fail_under``. This project also documents
per-module floors — 100% on the leakage-critical and metric-defining packages — and for a
whole milestone those floors were documentation only: ``make test`` exited 0 at 82% against
a stated 90%, with ``splits/plan.py`` at 86% against a stated 100%.

Floors live in ``[tool.ocr_risk.coverage]`` in ``pyproject.toml``, so the number in the
gate and the number in the docs are the same number. Run after ``pytest --cov``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _floors() -> tuple[float, dict[str, float]]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    section = config["tool"]["ocr_risk"]["coverage"]
    return float(section["overall"]), {k: float(v) for k, v in section["paths"].items()}


def _coverage_json() -> dict[str, object]:
    """The .coverage database the last ``pytest --cov`` run wrote, as JSON."""
    report = subprocess.run(
        # --fail-under=0 because THIS script is the gate; `coverage json` inherits
        # fail_under from pyproject and exits 2 below it, which would turn a reportable
        # shortfall into a crash with no message about which floor was missed.
        [sys.executable, "-m", "coverage", "json", "-o", "-", "--quiet", "--fail-under=0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(report.stdout))


def main() -> int:
    overall_floor, path_floors = _floors()
    payload = _coverage_json()
    files: dict[str, dict[str, dict[str, float]]] = payload["files"]  # type: ignore[assignment]
    measured = {
        path.replace("\\", "/"): float(entry["summary"]["percent_covered"])
        for path, entry in files.items()
    }
    if not measured:
        print("no coverage data; run `pytest --cov` first", file=sys.stderr)
        return 2

    totals: dict[str, float] = payload["totals"]  # type: ignore[assignment]
    total_statements = int(totals["num_statements"])
    covered_statements = int(totals["covered_lines"])

    failures: list[str] = []
    overall = float(totals["percent_covered"])
    if overall + 1e-9 < overall_floor:
        failures.append(
            f"overall {overall:.1f}% < {overall_floor:.1f}% "
            f"({covered_statements}/{total_statements} statements)"
        )

    for prefix, floor in sorted(path_floors.items()):
        # A floor may name a file or a package; a package floor applies to the package as
        # a whole, so one thoroughly tested module cannot carry an untested sibling.
        matched = {p: c for p, c in measured.items() if p == prefix or p.startswith(f"{prefix}/")}
        if not matched:
            failures.append(f"{prefix}: floor declared but no such file is measured")
            continue
        for path, percent in sorted(matched.items()):
            if percent + 1e-9 < floor:
                failures.append(f"{path} {percent:.1f}% < {floor:.1f}%")

    for line in failures:
        print(f"COVERAGE FLOOR: {line}", file=sys.stderr)
    if failures:
        print(
            f"\n{len(failures)} coverage floor(s) not met. Raise coverage with tests that "
            "would fail if the behaviour were wrong; do not lower the floor.",
            file=sys.stderr,
        )
        return 1

    print(f"coverage floors met: overall {overall:.1f}% >= {overall_floor:.1f}%")
    for prefix, floor in sorted(path_floors.items()):
        print(f"  {prefix:<40} >= {floor:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

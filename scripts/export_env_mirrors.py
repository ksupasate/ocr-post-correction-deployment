#!/usr/bin/env python3
"""Regenerate environment.yml and requirements.txt from the authoritative uv.lock.

uv.lock is the source of truth (ADR-001). These mirrors exist so a collaborator on a
conda or pip workflow has a supported entry point, and they carry a header saying they
are generated so nobody edits them by hand and expects it to stick.
"""

from __future__ import annotations

import subprocess
import sys

from ocr_risk.io.paths import project_root

HEADER = (
    "# GENERATED FILE - do not edit by hand.\n"
    "# uv.lock is authoritative (see docs/adr/0001-uv-primary.md).\n"
    "# Regenerate with: make env-mirrors\n"
)


def main() -> int:
    root = project_root()

    result = subprocess.run(
        ["uv", "export", "--no-hashes", "--no-dev", "--format", "requirements-txt"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"uv export failed: {result.stderr}", file=sys.stderr)
        return 1

    requirements = root / "requirements.txt"
    requirements.write_text(HEADER + result.stdout, encoding="utf-8")
    print(f"wrote {requirements.relative_to(root)}")

    pip_lines = [
        f"      - {line.strip()}"
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]
    environment = root / "environment.yml"
    environment.write_text(
        HEADER
        + "name: ocr-risk\n"
        + "channels:\n  - conda-forge\n"
        + "dependencies:\n  - python=3.11\n  - pip\n  - pip:\n"
        + "\n".join(pip_lines)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {environment.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

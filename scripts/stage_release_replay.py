#!/usr/bin/env python3
"""Stage code plus the identifier/numeric OR3 bundle in a NEW replay directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from release_artifact_checks import sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Choose a new output directory"
    code = Path(__file__).resolve().parents[1]
    shutil.copytree(
        code,
        args.output,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".coverage",
            ".hypothesis",
            "htmlcov",
            "*.egg-info",
        ),
    )
    manifest = json.loads((args.resource / "MANIFEST.json").read_text())
    for record in manifest["files"]:
        original = record.get("original_path")
        if not original:
            continue
        relative = Path(original)
        assert not relative.is_absolute() and ".." not in relative.parts, original
        source = args.resource / record["relative_path"]
        assert sha(source) == record["sha256"], record["relative_path"]
        target = args.output / relative
        assert not target.exists(), f"Conflicting staged file: {original}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print("Staged verified frozen files; no experiments executed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate that every path the runtime manifest names actually exists.

The manifest is the repository's claim about what its canonical entrypoints, configs, and
manifests are. A claim nobody checks drifts, and a reproducibility package that lists a
script it no longer contains is worse than one that lists nothing.
"""

from __future__ import annotations

import json
import sys

from ocr_risk.io.paths import project_root

SECTIONS = (
    "canonical_entrypoints",
    "canonical_scripts",
    "canonical_configs",
    "canonical_manifests",
    "canonical_docs",
)


def main() -> int:
    root = project_root()
    manifest_path = root / "runtime_manifest.json"
    if not manifest_path.is_file():
        print(f"missing {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    checked = 0

    for section in SECTIONS:
        for entry in manifest.get(section, []):
            checked += 1
            if not (root / entry).exists():
                missing.append(f"  {section}: {entry}")

    for entry in missing:
        print(f"MISSING {entry}", file=sys.stderr)

    if missing:
        print(
            f"\nruntime manifest is stale: {len(missing)} of {checked} paths missing",
            file=sys.stderr,
        )
        return 1

    print(f"runtime manifest OK: {checked} paths verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

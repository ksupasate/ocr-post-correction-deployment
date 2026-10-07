#!/usr/bin/env python3
"""Replay the existing Markdown table projection; do not evaluate new outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_artifact_checks import sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "results/paper3_vf2/run1/tables/scientific_results.json"
    freeze = json.loads((root / "results/paper3_vf2/run1/manifests/result_freeze.json").read_text())
    assert sha(source) == freeze["files_sha256"][str(source.relative_to(root))]
    # Module import defines the original projection function; no stage entrypoint is called.
    import paper3_vf2_reporting as reporting

    rendered = reporting.scientific_tables(source.parent.parent, json.loads(source.read_text()))
    original = root / "docs/paper3/journal_track_2027/validity_recovery/P3_VF2_RESULT_TABLES.md"
    assert rendered == original.read_text(), "Frozen Markdown table mismatch"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print("PASS: full-pool Markdown tables byte-identical; no new performance computed")


if __name__ == "__main__":
    main()

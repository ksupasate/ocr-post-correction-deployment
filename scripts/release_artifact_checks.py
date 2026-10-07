#!/usr/bin/env python3
"""Verify release hashes and registered values; never fit or evaluate a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numerical(resource: Path) -> None:
    registry = json.loads((resource / "numerical_registry/P3_VF2_RESULT_REGISTRY.json").read_text())
    entries = {row["key"]: row for row in registry["entries"]}
    assert len(entries) == len(registry["entries"]), "Duplicate registry keys"
    with (resource / "NUMERICAL_AUDIT.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        value = entries[row["source_key"]]["value"]
        if row["rounding"] == "classification":
            displayed = str(value)
        else:
            scale, decimals = row["rounding"].split(":")
            displayed = f"{value * float(scale):.{int(decimals)}f}"
        assert displayed == row["expected_value"], (row["source_key"], displayed)
        assert str(value) == row["registry_value"], row["source_key"]
        assert row["status"] == "PASS", row["source_key"]
    print(f"PASS: {len(rows)} numerical and classification checks")


def hashes(manifest: Path) -> None:
    data = json.loads(manifest.read_text())
    for row in data["files"]:
        path = manifest.parent / row["relative_path"]
        assert path.is_file(), f"Missing file: {row['relative_path']}"
        assert path.stat().st_size == row["byte_size"], row["relative_path"]
        assert sha(path) == row["sha256"], row["relative_path"]
    print(f"PASS: {len(data['files'])} manifest files")


def checksums(path: Path) -> None:
    root = path.parent.parent
    count = 0
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(root / name) == digest, name
        count += 1
    print(f"PASS: {count} SHA-256 checksums")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resource", type=Path)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--checksums", type=Path)
    args = parser.parse_args()
    if args.resource:
        numerical(args.resource)
    elif args.manifest:
        hashes(args.manifest)
    else:
        checksums(args.checksums)


if __name__ == "__main__":
    main()

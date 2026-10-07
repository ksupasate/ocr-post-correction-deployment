#!/usr/bin/env python3
"""Download and extract the FUNSD release.

FUNSD is distributed for research use and is **not redistributed** by this repository.
This script fetches it from the official host and verifies the archive against the digest
recorded in ``manifests/datasets/funsd.json``.

    uv run python scripts/download_funsd.py
    uv run python scripts/download_funsd.py --verify-checksums

Resumable and idempotent: re-running on a complete corpus performs no network IO.
"""

from __future__ import annotations

import argparse
import json
import sys

from ocr_risk.datasets.download import ChecksumMismatchError, DownloadSpec, download, extract_zip
from ocr_risk.datasets.funsd import FunsdDataset
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import project_root, raw_source_dir

MANIFEST = project_root() / "manifests" / "datasets" / "funsd.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="verify an existing download without fetching anything",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"missing dataset manifest: {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    root = raw_source_dir("funsd")
    archive = root / manifest["archive"]["filename"]
    expected = manifest["archive"].get("sha256")

    if args.verify_checksums:
        if not archive.is_file():
            print(f"nothing to verify: {archive} is not present", file=sys.stderr)
            return 1
        observed = file_sha256(archive)
        if expected and observed != expected:
            print(f"MISMATCH\n  expected {expected}\n  actual   {observed}", file=sys.stderr)
            return 1
        print(f"verified {archive.name}: {observed}")
        if not expected:
            print("  (no digest was recorded in the manifest; add the value above)")
        return 0

    print(f"FUNSD -> {root}")
    print(f"  licence: {manifest['license']['id']} ({manifest['license']['redistribution']})")
    print(f"  {manifest['license']['notes']}")

    try:
        download(
            DownloadSpec(
                url=manifest["archive"]["url"],
                destination=archive,
                sha256=expected,
                description="FUNSD release archive",
            ),
            force=args.force,
            progress=lambda message: print(f"  {message}"),
        )
    except ChecksumMismatchError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"\ndownload failed: {error}", file=sys.stderr)
        print(f"  fetch manually from {manifest['archive']['url']} and place it at {archive}")
        return 1

    extract_zip(archive, root, force=args.force)
    report = FunsdDataset().preflight()
    print(f"  extracted; preflight: available={report.available} documents={report.n_documents}")
    for problem in report.problems:
        print(f"    {problem}")
    return 0 if report.available else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download OCR-D / SBB historical-print ground truth.

Two things must be settled before this corpus enters a benchmark, and the script reports
both rather than assuming them:

1. **Licensing.** The OCR-D ``gt_structure_text`` repository is published under
   CC-BY-SA-4.0; the value recorded in ``manifests/datasets/ocrd_sbb.json`` was read from
   the GitHub repository metadata, with the date. Individual volumes originate from
   several holding libraries, so a publication should still confirm per volume.
2. **Recognition language.** These are German prints, largely Fraktur. Recognizing them
   with an English model produces plausible-looking garbage, so the preflight reports
   which Tesseract language packs are installed and what is missing.

    uv run python scripts/download_ocrd_sbb.py --volumes 12
    uv run python scripts/download_ocrd_sbb.py --verify-checksums

Resumable and idempotent: a volume already present and matching its recorded digest is
neither re-downloaded nor re-extracted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from ocr_risk.datasets.download import ChecksumMismatchError, DownloadSpec, download, extract_zip
from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import project_root, raw_source_dir

MANIFEST = project_root() / "manifests" / "datasets" / "ocrd_sbb.json"


def _tesseract_languages() -> list[str]:
    if shutil.which("tesseract") is None:
        return []
    result = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True, check=False
    )
    return [line.strip() for line in result.stdout.splitlines()[1:] if line.strip()]


def _report_languages(required: list[str]) -> bool:
    installed = _tesseract_languages()
    if not installed:
        print("  tesseract binary not found; skipping the language check")
        return True
    missing = [lang for lang in required if lang not in installed]
    if missing:
        print(f"\n  MISSING tesseract language data: {missing}")
        print("  These are German prints, largely Fraktur. Recognizing them with an")
        print("  English model produces plausible-looking garbage that is worse than a")
        print("  failure, because it looks like a result.")
        print("  Install with: brew install tesseract-lang")
        return False
    print(f"  tesseract language data present: {required}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument(
        "--volumes", type=int, default=0, help="fetch only the first N recorded volumes"
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="verify existing downloads without fetching anything",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"missing dataset manifest: {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    root = raw_source_dir("ocrd_sbb")
    volumes = manifest["volumes"]
    if args.volumes:
        volumes = volumes[: args.volumes]

    print(f"OCR-D / SBB -> {root}")
    print(f"  licence: {manifest['license']['id']} ({manifest['license']['redistribution']})")
    print(f"  release: {manifest['source']['release_tag']}")
    print(f"  volumes: {len(volumes)}")

    if args.verify_checksums:
        failures = 0
        for volume in volumes:
            archive = root / "_archive" / volume["filename"]
            if not archive.is_file():
                print(f"  MISSING {volume['filename']}", file=sys.stderr)
                failures += 1
                continue
            observed = file_sha256(archive)
            if observed != volume["sha256"]:
                print(f"  MISMATCH {volume['filename']}: {observed}", file=sys.stderr)
                failures += 1
        print(f"  verified {len(volumes) - failures}/{len(volumes)}")
        return 1 if failures else 0

    for volume in volumes:
        archive = root / "_archive" / volume["filename"]
        try:
            download(
                DownloadSpec(
                    url=volume["url"],
                    destination=archive,
                    sha256=volume["sha256"],
                    description=volume["volume_id"],
                ),
                force=args.force,
                progress=lambda message: print(f"  {message}"),
            )
        except ChecksumMismatchError as error:
            print(f"\n{error}", file=sys.stderr)
            return 1
        except Exception as error:
            print(f"  FAILED {volume['volume_id']}: {error}", file=sys.stderr)
            continue
        extract_zip(archive, root / volume["volume_id"], force=args.force)

    report = OcrdSbbDataset().preflight()
    print(f"\n  preflight: available={report.available} documents={report.n_documents}")
    for problem in report.problems:
        print(f"    {problem}")

    languages_ok = _report_languages(manifest["recognition"]["required_tesseract_languages"])
    return 0 if report.available and languages_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

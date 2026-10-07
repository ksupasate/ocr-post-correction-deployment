#!/usr/bin/env python3
"""Download and materialize the CORD v2 public release.

CORD is CC BY 4.0, so redistribution would be permitted — it is still fetched by script
rather than vendored, to keep the provenance chain to the upstream release explicit.

Upstream ships parquet shards per split with the page image inlined as a bytes column.
The adapter reads a per-file layout, so this script materializes it:

    data/raw/source/cord/<split>/image/<document>.png
    data/raw/source/cord/<split>/json/<document>.json

    uv run python scripts/download_cord.py                       # every split
    uv run python scripts/download_cord.py --split train         # one split
    uv run python scripts/download_cord.py --verify-checksums    # no network IO

Every archive's sha256 is pinned in manifests/datasets/cord.json before this script is
run; for the train/validation shards the digest is the upstream LFS oid, so the manifest
never depends on a local-only checksum. A fetched file that disagrees with its pinned
digest is a hard error, never a warning.

Resumable and idempotent: re-running on a complete, verified corpus performs no network
IO and rewrites nothing. Materialization is keyed on a digest-stamped marker covering
every shard of the split, so a partial extraction is redone rather than mistaken for a
complete one, and an upstream shard change forces a rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from ocr_risk.datasets.cord import CordDataset, shard_document_plan
from ocr_risk.datasets.download import ChecksumMismatchError, DownloadSpec, download
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import project_root, raw_source_dir
from ocr_risk.io.raw_store import RawStore

MANIFEST = project_root() / "manifests" / "datasets" / "cord.json"
ACQUISITION_RECORD = (
    project_root()
    / "results"
    / "generated"
    / "sgv1"
    / "corpus_qualification"
    / "cord_acquisition_record.json"
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_acquisition_record(record: dict) -> None:
    ACQUISITION_RECORD.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    temporary = ACQUISITION_RECORD.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(ACQUISITION_RECORD)


def _load_acquisition_record(manifest: dict) -> dict:
    if ACQUISITION_RECORD.is_file():
        return json.loads(ACQUISITION_RECORD.read_text(encoding="utf-8"))
    return {
        "schema_version": "sgv1-cord-acquisition-v1",
        "dataset": manifest["source"]["repo_id"],
        "dataset_release": manifest["source"]["dataset_release"],
        "revision": manifest["source"]["revision"],
        "canonical_source": manifest["source"]["verification"]["repository_api"],
        "license_source": manifest["license"]["source_url"],
        "license_id": manifest["license"]["id"],
        "raw_data_policy": manifest["license"]["raw_image_policy"],
        "archives": {},
        "splits": {},
        "runs": [],
    }


def _validate_archive_manifest(archives: list[dict]) -> None:
    """Fail before network I/O if an archive pin is incomplete or inconsistent."""
    seen: set[tuple[str, str]] = set()
    for entry in archives:
        key = (entry["split"], entry["shard"])
        if key in seen:
            raise ValueError(f"duplicate CORD archive entry: {key}")
        seen.add(key)
        if entry.get("checksum_kind") != "upstream_verified_checksum":
            raise ValueError(f"{entry['filename']} lacks an upstream_verified_checksum declaration")
        digest = entry.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{entry['filename']} has no valid pinned sha256")
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0:
            raise ValueError(f"{entry['filename']} has no valid pinned size")


def _materialize_split(root: Path, split: str, archives: list[dict]) -> tuple[int, str]:
    """Expand one split with write-once checks; return document count and marker hash."""
    parquets = [root / "_archive" / a["filename"] for a in archives]
    marker_digest = hashlib.sha256(
        "|".join(f"{a['filename']}:{a['sha256']}" for a in archives).encode("utf-8")
    ).hexdigest()
    images_dir, json_dir = root / split / "image", root / split / "json"
    marker = root / split / ".materialized.sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == marker_digest:
        return len(sorted(json_dir.glob("*.json"))), marker_digest

    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    store = RawStore()

    row_counts = [pq.read_metadata(p).num_rows for p in parquets]
    offsets = shard_document_plan(row_counts)
    written = 0
    for parquet, offset in zip(parquets, offsets):
        table = pq.read_table(parquet)
        for index in range(table.num_rows):
            # Named by the global split number alone: the adapter prefixes
            # "cord-<split>-"; the cumulative offset supplies the numeric suffix.
            document = f"{offset + index:04d}"
            payload = json.loads(table["ground_truth"][index].as_py())
            image_bytes = table["image"][index].as_py()["bytes"]

            # Decoded and re-encoded as PNG so page geometry is read from the same
            # pixels the engines will see, rather than trusting a declared size.
            with Image.open(io.BytesIO(image_bytes)) as image:
                encoded = io.BytesIO()
                image.save(encoded, format="PNG")
            store.write_bytes(images_dir / f"{document}.png", encoded.getvalue())
            store.write_bytes(
                json_dir / f"{document}.json",
                (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
            written += 1

    # The caller writes the marker only after census/preflight succeeds. A crash or
    # failed census therefore leaves a resumable write-once extraction, not a false
    # completion stamp.
    return written, marker_digest


def _split_census(root: Path, split: str) -> dict[str, int]:
    image_files = sorted((root / split / "image").glob("*.png"))
    annotation_files = sorted((root / split / "json").glob("*.json"))
    extracted_size = sum(p.stat().st_size for p in (*image_files, *annotation_files))
    return {
        "images": len(image_files),
        "annotations": len(annotation_files),
        "files": len(image_files) + len(annotation_files),
        "extracted_size_bytes": extracted_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and re-materialize")
    parser.add_argument("--split", help="process only this split (test|train|validation)")
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
    archives: list[dict] = manifest["archives"]
    root = raw_source_dir("cord")

    if args.split:
        archives = [a for a in archives if a["split"] == args.split]
        if not archives:
            known = sorted({a["split"] for a in manifest["archives"]})
            print(f"unknown split {args.split!r}; known: {known}", file=sys.stderr)
            return 2

    try:
        _validate_archive_manifest(archives)
    except (KeyError, ValueError) as error:
        print(f"invalid CORD archive manifest: {error}", file=sys.stderr)
        return 2

    if args.verify_checksums:
        failures = 0
        for entry in archives:
            archive = root / "_archive" / entry["filename"]
            if not archive.is_file():
                print(f"MISSING {archive}", file=sys.stderr)
                failures += 1
                continue
            observed = file_sha256(archive)
            if observed != entry["sha256"]:
                print(f"MISMATCH {entry['filename']}", file=sys.stderr)
                print(f"  pinned  {entry['sha256']}", file=sys.stderr)
                print(f"  actual  {observed}", file=sys.stderr)
                failures += 1
            else:
                print(f"verified {entry['filename']}: {observed}")
        return 1 if failures else 0

    if args.force and any((root / "_archive" / a["filename"]).is_file() for a in archives):
        print(
            "--force cannot replace an existing raw archive; raw evidence is write-once",
            file=sys.stderr,
        )
        return 2

    print(f"CORD -> {root}")
    print(f"  licence: {manifest['license']['id']} ({manifest['license']['redistribution']})")
    print(f"  revision: {manifest['source']['revision']}")

    acquisition = _load_acquisition_record(manifest)
    run = {
        "started_utc": _utc_now(),
        "completed_utc": None,
        "pid": os.getpid(),
        "git_head": _git_head(),
        "command": [sys.executable, *sys.argv],
        "requested_split": args.split or "all",
        "script_sha256": file_sha256(Path(__file__)),
        "status": "running",
        "partial_files_observed": [],
    }
    acquisition["runs"].append(run)
    _write_acquisition_record(acquisition)

    for entry in archives:
        archive = root / "_archive" / entry["filename"]
        partial = archive.with_suffix(archive.suffix + ".part")
        archive_existed = archive.is_file()
        partial_size = partial.stat().st_size if partial.is_file() else 0
        if partial.is_file():
            run["partial_files_observed"].append(
                {
                    "path": partial.relative_to(project_root()).as_posix(),
                    "size_bytes": partial_size,
                    "resume_policy": (
                        "HTTP Range resume when non-zero; a zero-byte partial is opened "
                        "fresh and never treated as verified evidence"
                    ),
                }
            )
            _write_acquisition_record(acquisition)
        try:
            download(
                DownloadSpec(
                    url=entry["url"],
                    destination=archive,
                    sha256=entry["sha256"],
                    description=f"CORD v2 {entry['split']} split ({entry['shard']})",
                ),
                force=args.force,
                progress=lambda message: print(f"  {message}"),
            )
        except ChecksumMismatchError as error:
            run["status"] = "failed_checksum"
            run["completed_utc"] = _utc_now()
            run["error"] = str(error)
            _write_acquisition_record(acquisition)
            print(f"\n{error}", file=sys.stderr)
            return 1
        except Exception as error:
            run["status"] = "failed_download"
            run["completed_utc"] = _utc_now()
            run["error"] = repr(error)
            _write_acquisition_record(acquisition)
            print(f"\ndownload failed: {error}", file=sys.stderr)
            print(f"  fetch manually from {entry['url']} and place it at {archive}")
            return 1

        local_sha = file_sha256(archive)
        actual_size = archive.stat().st_size
        if actual_size != entry["size_bytes"]:
            run["status"] = "failed_size"
            run["completed_utc"] = _utc_now()
            run["error"] = f"{entry['filename']} size {actual_size} != pinned {entry['size_bytes']}"
            _write_acquisition_record(acquisition)
            print(run["error"], file=sys.stderr)
            return 1
        previous_archive_record = acquisition["archives"].get(entry["filename"], {})
        first_retrieval_utc = previous_archive_record.get("retrieval_utc")
        if first_retrieval_utc is None:
            first_retrieval_utc = entry.get("retrieved_utc") if archive_existed else _utc_now()
        acquisition["archives"][entry["filename"]] = {
            "canonical_url": entry["url"],
            "split": entry["split"],
            "shard": entry["shard"],
            "checksum_kind": entry["checksum_kind"],
            "upstream_verified_checksum": entry["sha256"],
            "locally_observed_checksum": local_sha,
            "checksum_match": local_sha == entry["sha256"],
            "expected_size_bytes": entry["size_bytes"],
            "actual_size_bytes": actual_size,
            "archive_preexisted_this_run": archive_existed,
            "partial_size_before_run": partial_size,
            "retrieval_utc": first_retrieval_utc,
            "last_verified_utc": _utc_now(),
        }
        _write_acquisition_record(acquisition)

    failures = 0
    for split in dict.fromkeys(a["split"] for a in archives):
        split_archives = [a for a in archives if a["split"] == split]
        if args.force:
            (root / split / ".materialized.sha256").unlink(missing_ok=True)
        written, marker_digest = _materialize_split(root, split, split_archives)
        print(f"  materialized {written} document(s) into {root / split}")

        report = CordDataset(split=split).preflight()
        print(f"  preflight: available={report.available} documents={report.n_documents}")
        for problem in report.problems:
            print(f"    {problem}")

        expected_n = manifest.get("expected", {}).get(f"{split}_documents")
        census = _split_census(root, split)
        complete = (
            expected_n is not None
            and report.n_documents == expected_n
            and census["images"] == expected_n
            and census["annotations"] == expected_n
            and report.available
        )
        acquisition["splits"][split] = {
            "requested_by_run_started_utc": run["started_utc"],
            "expected_documents": expected_n,
            "actual_documents": report.n_documents,
            **census,
            "materialization_marker_sha256": marker_digest,
            "materializer": "scripts/download_cord.py:_materialize_split",
            "extraction_command": run["command"],
            "script_sha256": run["script_sha256"],
            "complete": complete,
        }
        _write_acquisition_record(acquisition)
        if expected_n is not None and report.n_documents != expected_n:
            print(
                f"  WARNING: manifest expects {expected_n} documents, found {report.n_documents}",
                file=sys.stderr,
            )
            failures += 1
        elif census["images"] != expected_n or census["annotations"] != expected_n:
            print(
                f"  WARNING: split file census is images={census['images']} "
                f"annotations={census['annotations']}, expected {expected_n} each",
                file=sys.stderr,
            )
            failures += 1
        elif not report.available:
            failures += 1
        else:
            marker = root / split / ".materialized.sha256"
            marker.write_text(marker_digest + "\n", encoding="utf-8")

    run["status"] = "complete" if not failures else "failed_preflight"
    run["completed_utc"] = _utc_now()
    _write_acquisition_record(acquisition)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

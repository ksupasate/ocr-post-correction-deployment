"""Resumable, checksum-verified dataset downloads.

Three rules, each with a reason:

- **Verify before use.** A missing or mismatched checksum is a hard error, never a
  warning. A corrupted corpus produces plausible-looking numbers.
- **Resume, do not restart.** Partial downloads continue from a ``.part`` file; a
  half-finished multi-hundred-megabyte fetch should not cost the whole thing.
- **Idempotent.** Re-running on a complete, verified corpus performs no network IO, so
  ``make`` targets and CI can call it freely.

Nothing here is invoked automatically. Downloads are explicit because they are slow,
licensed, and sometimes large.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ocr_risk.io.hashing import file_sha256

__all__ = ["ChecksumMismatchError", "DownloadSpec", "download", "extract_zip"]

_CHUNK = 1 << 20
_USER_AGENT = "ocr-post-correction-deployment/0.1 (research; +https://github.com/ksupasate/ocr-post-correction-deployment)"


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file does not match its recorded digest."""


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """One file to fetch, and how to know it arrived intact."""

    url: str
    destination: Path
    sha256: str | None = None
    """``None`` means the digest is not yet recorded. The first successful download
    prints the observed value so it can be added to the manifest — the file is still
    usable, but the run is marked unverified rather than silently trusted."""
    description: str = ""


def _verified(path: Path, expected: str | None) -> bool:
    if not path.is_file():
        return False
    if expected is None:
        return True
    return file_sha256(path) == expected


def download(
    spec: DownloadSpec,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fetch one file, resuming a partial download and verifying the result."""
    say = progress or (lambda _: None)
    destination = spec.destination

    if not force and _verified(destination, spec.sha256):
        say(f"already present and verified: {destination.name}")
        return destination
    if destination.is_file() and spec.sha256 and not _verified(destination, spec.sha256):
        msg = (
            f"{destination} exists but does not match its recorded sha256.\n"
            f"  expected {spec.sha256}\n  actual   {file_sha256(destination)}\n"
            "Refusing to use it: a corrupted corpus produces plausible-looking numbers. "
            "Delete the file to re-download."
        )
        raise ChecksumMismatchError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.is_file() else 0

    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
        say(f"resuming {destination.name} from {existing / 1e6:.1f} MB")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            # A server that ignores Range restarts the file, so the partial must be
            # discarded rather than appended to.
            append = existing > 0 and response.status == 206
            mode = "ab" if append else "wb"
            if not append and existing:
                say("server ignored the range request; restarting the download")
            with partial.open(mode) as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
    except urllib.error.HTTPError as error:
        if error.code == 416 and existing:  # already complete
            pass
        else:
            raise

    if spec.sha256:
        observed = file_sha256(partial)
        if observed != spec.sha256:
            partial.unlink(missing_ok=True)
            msg = (
                f"checksum mismatch for {spec.url}\n"
                f"  expected {spec.sha256}\n  actual   {observed}\n"
                "The partial file has been removed."
            )
            raise ChecksumMismatchError(msg)
    else:
        say(
            f"UNVERIFIED: no sha256 recorded for {destination.name}. Observed "
            f"{file_sha256(partial)} — add it to the dataset manifest."
        )

    partial.replace(destination)
    say(f"downloaded {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def extract_zip(archive: Path, target: Path, *, force: bool = False) -> Path:
    """Extract an archive into ``target``, skipping the work if already done.

    "Already done" is decided by a marker file stamped with the archive's digest, not by
    the target directory existing. The download creates that directory, so an
    existence check silently skips extraction — and leaves a corpus that looks downloaded
    and is empty.
    """
    marker = target / ".extracted.sha256"
    digest = file_sha256(archive)

    if not force and marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return target

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            # Refuse path traversal: an archive entry must not escape the target.
            resolved = (target / member).resolve()
            if not str(resolved).startswith(str(target.resolve())):
                msg = f"archive member escapes the target directory: {member}"
                raise ValueError(msg)
        handle.extractall(target)

    marker.write_text(digest + "\n", encoding="utf-8")
    return target

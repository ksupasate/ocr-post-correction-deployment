"""Write-once storage for raw source files and verbatim OCR engine responses.

``data/raw`` is the evidence layer. Everything downstream is a re-derivable opinion about
it, so the one property that must hold is that it was never edited after the fact. The
store enforces this by refusing any write that would change existing bytes — an
overwrite attempt raises rather than succeeding quietly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ocr_risk.io.hashing import canonical_json, faithful_json, file_sha256, sha256_of_bytes
from ocr_risk.io.paths import raw_ocr_dir, raw_source_dir
from ocr_risk.schemas.spans import RawEngineResponse

__all__ = ["ImmutableWriteError", "RawStore"]


class ImmutableWriteError(RuntimeError):
    """Raised when a write would modify bytes already present in the raw layer."""


class RawStore:
    """Content-checked, write-once access to ``data/raw``."""

    def write_bytes(self, path: Path, data: bytes) -> str:
        """Write ``data`` to ``path`` unless different bytes are already there.

        Re-writing byte-identical content is a no-op and succeeds: pipelines re-run, and
        an idempotent re-download should not be an error. Writing *different* content to
        an existing path always raises.
        """
        digest = sha256_of_bytes(data)
        if path.exists():
            existing = file_sha256(path)
            if existing == digest:
                return digest
            msg = (
                f"refusing to overwrite immutable raw file {path}\n"
                f"  existing sha256: {existing}\n"
                f"  incoming sha256: {digest}\n"
                "Raw evidence is write-once. Write a new fingerprinted path instead."
            )
            raise ImmutableWriteError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        # Atomic rename: a crash mid-write must not leave a truncated file that later
        # passes an existence check and silently becomes "the evidence".
        tmp.replace(path)
        return digest

    def write_json(self, path: Path, payload: Any) -> str:
        return self.write_bytes(path, canonical_json(payload).encode("utf-8"))

    def write_engine_response(self, response: RawEngineResponse) -> Path:
        """Persist one engine response under its fingerprinted, immutable location."""
        directory = raw_ocr_dir(
            response.dataset_id, response.engine_id, response.engine_fingerprint
        )
        path = directory / f"{response.document_id}.json"
        # faithful_json, not canonical_json: the raw layer stores evidence, and
        # canonical encoding turns every measurement into a string.
        self.write_bytes(path, faithful_json(response).encode("utf-8"))
        return path

    def read_engine_response(self, path: Path) -> RawEngineResponse:
        return RawEngineResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def engine_response_path(
        self, dataset_id: str, engine_id: str, fingerprint: str, doc: str
    ) -> Path:
        return raw_ocr_dir(dataset_id, engine_id, fingerprint) / f"{doc}.json"

    def has_engine_response(
        self, dataset_id: str, engine_id: str, fingerprint: str, doc: str
    ) -> bool:
        return self.engine_response_path(dataset_id, engine_id, fingerprint, doc).exists()

    def list_engine_responses(
        self, dataset_id: str, engine_id: str, fingerprint: str
    ) -> list[Path]:
        directory = raw_ocr_dir(dataset_id, engine_id, fingerprint)
        return sorted(directory.glob("*.json")) if directory.exists() else []

    def fingerprint_record_path(self, dataset_id: str, engine_id: str) -> Path:
        """Sidecar naming the fingerprint a completed recognition pass used.

        Recorded so a later stage can find the raw responses **without constructing the
        engine**. That is not an optimization: PaddlePaddle and PyTorch cannot be
        imported into one process on this platform (paddle's allocator leaves torch
        unable to initialize a tensor), so a run that touched all four backends in
        sequence would crash. With the fingerprint on disk, recognition happens one
        engine per process and everything downstream reads bytes.
        """
        return raw_ocr_dir(dataset_id, engine_id, "_meta").parent / "fingerprint.json"

    def record_fingerprint(self, dataset_id: str, engine_id: str, payload: Any) -> Path:
        path = self.fingerprint_record_path(dataset_id, engine_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately not write_bytes: this is a pointer into the immutable store, not
        # evidence, and it is rewritten whenever a recognition pass completes.
        path.write_text(canonical_json(payload), encoding="utf-8")
        return path

    def read_fingerprint(self, dataset_id: str, engine_id: str) -> dict[str, Any] | None:
        path = self.fingerprint_record_path(dataset_id, engine_id)
        if not path.is_file():
            return None
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return payload

    def source_path(self, dataset_id: str, relative: str) -> Path:
        return raw_source_dir(dataset_id) / relative

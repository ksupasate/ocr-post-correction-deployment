"""Tesseract adapter.

Uses Tesseract's TSV output, which is the only mode that reports per-word bounding boxes
*and* a per-word confidence together. The alternatives lose one or the other, and this
project needs both on every span.

Native confidence is Tesseract's own 0-100 word confidence, stored verbatim. Tesseract
emits ``-1`` for entries it did not score; that is recorded as ``None`` rather than
clamped to 0, because "unscored" and "scored zero" are different observations.

Requires the ``tesseract`` extra plus the binary::

    uv sync --extra tesseract
    brew install tesseract           # macOS
    brew install tesseract-lang      # non-English traineddata (deu, frk, ...)
"""

from __future__ import annotations

import csv
import io
import platform
import shutil
import socket
import subprocess
import time
from typing import Any

from ocr_risk.engines.base import (
    Availability,
    OCREngineAdapter,
    PageInput,
    ParsedSpan,
    make_fingerprint,
)
from ocr_risk.engines.registry import register_engine
from ocr_risk.io.hashing import sha256_of_bytes
from ocr_risk.schemas.base import BBox, ConfidenceScale
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = ["TESSERACT_SCALE", "TesseractEngine"]

ADAPTER_NAME = "tesseract"
ADAPTER_VERSION = "1"
PAYLOAD_FORMAT = "tesseract_tsv_v5"
_TIMEOUT_SECONDS = 180

TESSERACT_SCALE = ConfidenceScale(
    name="tesseract_word_conf_0_100",
    minimum=0.0,
    maximum=100.0,
    higher_is_better=True,
)


@register_engine(ADAPTER_NAME)
class TesseractEngine(OCREngineAdapter):
    """Real Tesseract via the CLI, producing TSV."""

    confidence_scale = TESSERACT_SCALE

    def __init__(
        self,
        engine_id: str = "tesseract",
        lang: str = "eng",
        psm: int = 3,
        oem: int = 3,
        extra_config: tuple[str, ...] = (),
        binary: str = "tesseract",
    ) -> None:
        self.engine_id = engine_id
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.extra_config = tuple(extra_config)
        self.binary = binary

    # --- contract ----------------------------------------------------------------
    def availability(self) -> Availability:
        path = shutil.which(self.binary)
        if path is None:
            return Availability(
                engine_id=self.engine_id,
                installed=False,
                detail=f"{self.binary!r} not found on PATH",
                remediation="brew install tesseract  (and: uv sync --extra tesseract)",
            )
        version = self._version()
        langs = self._languages()
        if self.lang not in langs:
            return Availability(
                engine_id=self.engine_id,
                installed=False,
                version=version,
                detail=f"language {self.lang!r} not installed; available: {', '.join(langs)}",
                remediation=(
                    "brew install tesseract-lang, or place <lang>.traineddata in TESSDATA_PREFIX"
                ),
            )
        return Availability(engine_id=self.engine_id, installed=True, version=version)

    def _version(self) -> str:
        try:
            out = subprocess.run(
                [self.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        first = out.stdout.splitlines()[0] if out.stdout else ""
        return first.replace("tesseract ", "").strip() or "unknown"

    def _languages(self) -> tuple[str, ...]:
        try:
            out = subprocess.run(
                [self.binary, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
        return tuple(line.strip() for line in out.stdout.splitlines()[1:] if line.strip())

    def fingerprint(self) -> EngineFingerprint:
        return make_fingerprint(
            engine_id=self.engine_id,
            engine_version=self._version(),
            model_ids=(f"{self.lang}.traineddata",),
            config={
                "psm": self.psm,
                "oem": self.oem,
                "lang": self.lang,
                "extra_config": list(self.extra_config),
            },
        )

    def recognize(self, page: PageInput) -> RawEngineResponse:
        self.availability().require()
        started = time.monotonic()
        command = [
            self.binary,
            str(page.image_path),
            "stdout",
            "-l",
            self.lang,
            "--psm",
            str(self.psm),
            "--oem",
            str(self.oem),
            *self.extra_config,
            "tsv",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, check=False
        )
        if result.returncode != 0:
            msg = f"tesseract failed on {page.document_id}: {result.stderr.strip()}"
            raise RuntimeError(msg)

        # The TSV text is stored verbatim. Parsing happens in parse(), so a parser fix can
        # be re-applied later without re-running the engine.
        payload: dict[str, Any] = {
            "tsv": result.stdout,
            "command": command,
            "stderr": result.stderr,
            "image_sha256": page.image_sha256,
        }
        return RawEngineResponse(
            document_id=page.document_id,
            dataset_id=page.dataset_id,
            engine_id=self.engine_id,
            engine_fingerprint=self.fingerprint().fingerprint,
            payload_format=PAYLOAD_FORMAT,
            payload=payload,
            payload_sha256=sha256_of_bytes(result.stdout.encode("utf-8")),
            adapter_version=ADAPTER_VERSION,
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_seconds=max(0.0, time.monotonic() - started),
            host=socket.gethostname(),
            platform=platform.platform(),
        )

    def parse(self, raw: RawEngineResponse) -> list[ParsedSpan]:
        """Parse stored TSV. Pure function of the payload; needs no Tesseract installed."""
        if raw.payload_format != PAYLOAD_FORMAT:
            msg = f"{self.engine_id}: cannot parse payload format {raw.payload_format!r}"
            raise ValueError(msg)

        reader = csv.DictReader(
            io.StringIO(raw.payload["tsv"]), delimiter="\t", quoting=csv.QUOTE_NONE
        )
        spans: list[ParsedSpan] = []
        for index, row in enumerate(reader):
            # level 5 is the word level; the coarser levels repeat the same pixels as
            # block/paragraph/line aggregates and would double-count every span.
            if row.get("level") != "5":
                continue
            text = row.get("text") or ""
            if not text.strip():
                continue
            left, top = float(row["left"]), float(row["top"])
            width, height = float(row["width"]), float(row["height"])
            conf = float(row["conf"])
            spans.append(
                ParsedSpan(
                    text=text,
                    raw_index=index,
                    bbox=BBox(x0=left, y0=top, x1=left + width, y1=top + height),
                    line_id=f"b{row['block_num']}:p{row['par_num']}:l{row['line_num']}",
                    block_id=f"b{row['block_num']}",
                    # -1 means Tesseract did not score this word. Recording None keeps
                    # "unscored" distinguishable from "scored zero".
                    native_conf_recognition=None if conf < 0 else conf,
                    reading_order_hint=len(spans),
                )
            )
        return spans

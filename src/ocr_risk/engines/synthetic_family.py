"""Simulated OCR engines with deliberately different error profiles.

These exist for one reason the smoke test genuinely needs: **cross-engine shift must be
present, not merely labelled.** Four adapters that corrupted text identically would let a
leave-one-engine-out experiment pass while testing nothing, because there would be no
distribution to shift.

Each profile therefore differs along the axes that drive the research question:

======================  ==========================================================
axis                    why it matters
======================  ==========================================================
confusion family        which characters get confused decides which *candidates*
                        are plausible, and therefore what a text-only verifier can
                        infer without looking at pixels
split / merge rate      segmentation disagreement is what makes alignment hard and
                        makes spans non-comparable across engines
deletion / insertion    omissions and hallucinations are the alignment cases that
                        cannot be represented as substitutions
confidence shape        bias and informativeness set how well confidence predicts
                        correctness, so calibration fitted on one engine genuinely
                        mis-transfers to another (this is what H1 is about)
======================  ==========================================================

SYNTHETIC. Nothing measured with these engines is a research result.
"""

from __future__ import annotations

import json
import platform
import socket
import time
from dataclasses import dataclass
from typing import Any

from numpy.random import Generator, default_rng

from ocr_risk.engines.base import (
    Availability,
    OCREngineAdapter,
    PageInput,
    ParsedSpan,
    make_fingerprint,
)
from ocr_risk.engines.registry import register_engine
from ocr_risk.io.hashing import hash_str, sha256_of_bytes
from ocr_risk.io.paths import raw_source_dir
from ocr_risk.schemas.base import BBox, ConfidenceScale
from ocr_risk.schemas.spans import EngineFingerprint, RawEngineResponse

__all__ = ["CONFUSION_PROFILES", "SyntheticEngine"]

ADAPTER_NAME = "synthetic"
ADAPTER_VERSION = "1"
PAYLOAD_FORMAT = "synthetic_words_v1"

SCALES: dict[str, ConfidenceScale] = {
    "synthetic_0_1": ConfidenceScale(
        name="synthetic_0_1", minimum=0.0, maximum=1.0, higher_is_better=True
    ),
    "synthetic_0_100": ConfidenceScale(
        name="synthetic_0_100", minimum=0.0, maximum=100.0, higher_is_better=True
    ),
}

# Directed character substitutions per profile. Multi-character keys model the classic
# scanner failures ("rn" read as "m") that single-character noise cannot produce.
CONFUSION_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "visual": {
        "rn": ("m",),
        "m": ("rn",),
        "l": ("1", "I"),
        "I": ("l", "1"),
        "1": ("l",),
        "O": ("0",),
        "0": ("O",),
        "S": ("5",),
        "5": ("S",),
        "cl": ("d",),
        "d": ("cl",),
        "B": ("8",),
        "e": ("c",),
        "c": ("e",),
        "a": ("o",),
        "n": ("h",),
    },
    "segmentation": {
        "l": ("1",),
        "i": ("l",),
        "o": ("0",),
        "u": ("v",),
        "t": ("f",),
    },
    "numeric": {
        # Digit confusions plus decimal-point damage: a misplaced or dropped separator is
        # the error that turns 0.015 into 0.15, which is the harmful case this project
        # cares most about detecting.
        "0": ("8", "6"),
        "1": ("7", "4"),
        "3": ("8",),
        "5": ("6", "8"),
        "6": ("5",),
        "8": ("3", "0"),
        "9": ("4",),
        ".": ("", ","),
        ",": (".",),
        "g": ("q", "9"),
        "m": ("n",),
        "u": ("v", "µ"),
    },
    "mixed": {
        "rn": ("m",),
        "m": ("rn", "n"),
        "l": ("1",),
        "O": ("0",),
        "0": ("O",),
        "5": ("S",),
        "8": ("3",),
        ".": (",",),
        "e": ("c",),
    },
}

_HALLUCINATIONS = ("the", "of", ".", "-", "l", "|", "'", "1")


@dataclass(slots=True)
class SyntheticParams:
    """Error-profile parameters for one simulated engine."""

    seed: int = 1000
    char_error_rate: float = 0.03
    confusion_profile: str = "mixed"
    split_rate: float = 0.01
    merge_rate: float = 0.01
    deletion_rate: float = 0.004
    insertion_rate: float = 0.003
    confidence_scale: str = "synthetic_0_1"
    confidence_bias: float = 0.0
    confidence_noise: float = 0.1
    confidence_informativeness: float = 0.7
    """How strongly confidence tracks correctness. 1.0 is an oracle, 0.0 is pure noise.
    Varying this across engines is what makes calibration transfer a real problem."""

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}


def _corrupt_text(text: str, rng: Generator, table: dict[str, tuple[str, ...]], rate: float) -> str:
    """Apply character-level confusions, preferring longer (multi-char) matches."""
    if rate <= 0.0 or not text:
        return text
    out: list[str] = []
    i = 0
    keys_by_length = sorted(table, key=len, reverse=True)
    while i < len(text):
        applied = False
        if rng.random() < rate:
            for key in keys_by_length:
                if text.startswith(key, i):
                    choices = table[key]
                    replacement = choices[int(rng.integers(0, len(choices)))] if choices else ""
                    out.append(replacement)
                    i += len(key)
                    applied = True
                    break
        if not applied:
            out.append(text[i])
            i += 1
    return "".join(out)


def _split_bbox(box: BBox, fraction: float) -> tuple[BBox, BBox]:
    cut = box.x0 + box.width * fraction
    return (
        BBox(x0=box.x0, y0=box.y0, x1=cut, y1=box.y1),
        BBox(x0=cut, y0=box.y0, x1=box.x1, y1=box.y1),
    )


def _union_bbox(a: BBox, b: BBox) -> BBox:
    return BBox(x0=min(a.x0, b.x0), y0=min(a.y0, b.y0), x1=max(a.x1, b.x1), y1=max(a.y1, b.y1))


@register_engine(ADAPTER_NAME)
class SyntheticEngine(OCREngineAdapter):
    """A deterministic simulated recognizer.

    Reads the ground truth of a synthetic page and produces a corrupted reading of it. It
    is a *simulation of an engine*, not a recognizer: it never looks at pixels.
    """

    def __init__(self, engine_id: str, **params: Any) -> None:
        self.engine_id = engine_id
        known = set(SyntheticParams.__slots__)
        unknown = set(params) - known
        if unknown:
            msg = f"unknown synthetic engine parameters: {sorted(unknown)}"
            raise TypeError(msg)
        self.params = SyntheticParams(**params)
        if self.params.confusion_profile not in CONFUSION_PROFILES:
            msg = (
                f"unknown confusion profile {self.params.confusion_profile!r}; "
                f"known: {sorted(CONFUSION_PROFILES)}"
            )
            raise ValueError(msg)
        self.confidence_scale = SCALES[self.params.confidence_scale]

    # --- contract ----------------------------------------------------------------
    def availability(self) -> Availability:
        return Availability(
            engine_id=self.engine_id,
            installed=True,
            version=ADAPTER_VERSION,
            detail="simulated engine; always available",
        )

    def fingerprint(self) -> EngineFingerprint:
        return make_fingerprint(
            engine_id=self.engine_id,
            engine_version=ADAPTER_VERSION,
            model_ids=(f"profile:{self.params.confusion_profile}",),
            config=self.params.as_dict(),
        )

    def recognize(self, page: PageInput) -> RawEngineResponse:
        """Simulate recognition of one page from its stored ground truth."""
        started = time.monotonic()
        gt_path = raw_source_dir(page.dataset_id) / f"{page.document_id}.gt.json"
        if not gt_path.is_file():
            msg = (
                f"synthetic engine needs ground truth at {gt_path}; "
                "it simulates a reading of known text rather than recognizing pixels"
            )
            raise FileNotFoundError(msg)
        truth = json.loads(gt_path.read_text(encoding="utf-8"))
        words = self._simulate(truth["tokens"], page)

        payload: dict[str, Any] = {
            "engine_id": self.engine_id,
            "document_id": page.document_id,
            "image_sha256": page.image_sha256,
            "words": words,
            "profile": self.params.confusion_profile,
            "confidence_scale": self.params.confidence_scale,
        }
        return RawEngineResponse(
            document_id=page.document_id,
            dataset_id=page.dataset_id,
            engine_id=self.engine_id,
            engine_fingerprint=self.fingerprint().fingerprint,
            payload_format=PAYLOAD_FORMAT,
            payload=payload,
            payload_sha256=sha256_of_bytes(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ),
            adapter_version=ADAPTER_VERSION,
            # Recognition is simulated, so wall-clock facts are recorded for provenance
            # symmetry with real engines but carry no measurement meaning.
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_seconds=max(0.0, time.monotonic() - started),
            host=socket.gethostname(),
            platform=platform.platform(),
        )

    def parse(self, raw: RawEngineResponse) -> list[ParsedSpan]:
        if raw.payload_format != PAYLOAD_FORMAT:
            msg = f"{self.engine_id}: cannot parse payload format {raw.payload_format!r}"
            raise ValueError(msg)
        spans: list[ParsedSpan] = []
        for index, word in enumerate(raw.payload["words"]):
            x0, y0, x1, y1 = word["bbox"]
            spans.append(
                ParsedSpan(
                    text=word["text"],
                    raw_index=index,
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    line_id=word.get("line_id"),
                    native_conf_recognition=word.get("conf"),
                    native_conf_detection=word.get("det_conf"),
                    reading_order_hint=index,
                )
            )
        return spans

    # --- simulation ---------------------------------------------------------------
    def _confidence(self, correct: bool, rng: Generator) -> float:
        """Confidence whose *relationship to correctness* is an engine property.

        A well-calibrated engine and an overconfident one differ here, not in accuracy,
        which is precisely the shift a cross-engine calibration study must contend with.
        """
        p = self.params
        signal = p.confidence_informativeness * (1.0 if correct else 0.0) + (
            1.0 - p.confidence_informativeness
        ) * float(rng.random())
        value = signal + p.confidence_bias + float(rng.normal(0.0, p.confidence_noise))
        unit = min(max(value, 0.0), 1.0)
        scale = self.confidence_scale
        return round(scale.minimum + unit * (scale.maximum - scale.minimum), 6)

    def _simulate(self, tokens: list[dict[str, Any]], page: PageInput) -> list[dict[str, Any]]:
        p = self.params
        # Seeded by (engine, document) so every engine reads every page reproducibly, and
        # so adding a document never perturbs the others.
        # Stable digest, NOT builtin hash(): str hashing is randomized by PYTHONHASHSEED,
        # which would make the corpus differ between processes.
        doc_seed = int(hash_str(page.document_id)[:8], 16)
        rng = default_rng([p.seed, doc_seed])
        table = CONFUSION_PROFILES[p.confusion_profile]

        words: list[dict[str, Any]] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            text = str(token["text"])
            box = BBox(
                x0=token["bbox"][0], y0=token["bbox"][1], x1=token["bbox"][2], y1=token["bbox"][3]
            )
            line_id = f"line:{token['line']:03d}"

            if rng.random() < p.deletion_rate:
                index += 1
                continue  # OCR omission: the token simply does not appear.

            # Merge with the following token on the same line (drops the space).
            if (
                rng.random() < p.merge_rate
                and index + 1 < len(tokens)
                and tokens[index + 1]["line"] == token["line"]
            ):
                nxt = tokens[index + 1]
                nxt_box = BBox(
                    x0=nxt["bbox"][0], y0=nxt["bbox"][1], x1=nxt["bbox"][2], y1=nxt["bbox"][3]
                )
                merged = text + str(nxt["text"])
                emitted = _corrupt_text(merged, rng, table, p.char_error_rate)
                words.append(
                    {
                        "text": emitted,
                        "bbox": _bbox_list(_union_bbox(box, nxt_box)),
                        "line_id": line_id,
                        "conf": self._confidence(correct=False, rng=rng),
                        "det_conf": round(float(rng.uniform(0.5, 1.0)), 6),
                    }
                )
                index += 2
                continue

            emitted = _corrupt_text(text, rng, table, p.char_error_rate)

            # Split into two spans at an interior character boundary.
            if rng.random() < p.split_rate and len(emitted) >= 4:
                cut = int(rng.integers(1, len(emitted)))
                left_box, right_box = _split_bbox(box, cut / len(emitted))
                for piece, piece_box in ((emitted[:cut], left_box), (emitted[cut:], right_box)):
                    words.append(
                        {
                            "text": piece,
                            "bbox": _bbox_list(piece_box),
                            "line_id": line_id,
                            "conf": self._confidence(correct=False, rng=rng),
                            "det_conf": round(float(rng.uniform(0.5, 1.0)), 6),
                        }
                    )
                index += 1
                continue

            words.append(
                {
                    "text": emitted,
                    "bbox": _bbox_list(box),
                    "line_id": line_id,
                    "conf": self._confidence(correct=emitted == text, rng=rng),
                    "det_conf": round(float(rng.uniform(0.5, 1.0)), 6),
                }
            )

            # Hallucinate an extra span beside the current one.
            if rng.random() < p.insertion_rate:
                ghost = _HALLUCINATIONS[int(rng.integers(0, len(_HALLUCINATIONS)))]
                words.append(
                    {
                        "text": ghost,
                        "bbox": _bbox_list(
                            BBox(x0=box.x1 + 1.0, y0=box.y0, x1=box.x1 + 9.0, y1=box.y1)
                        ),
                        "line_id": line_id,
                        "conf": self._confidence(correct=False, rng=rng),
                        "det_conf": round(float(rng.uniform(0.3, 0.8)), 6),
                    }
                )
            index += 1
        return words


def _bbox_list(box: BBox) -> list[float]:
    return [round(box.x0, 3), round(box.y0, 3), round(box.x1, 3), round(box.y1, 3)]

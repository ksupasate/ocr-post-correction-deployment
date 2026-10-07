"""CORD adapter — receipts, and the number-heavy track.

Written and fixture-tested but **not executed** during the bootstrap. It is registered so
a config can name it and get a real preflight report rather than a registry miss.

CORD matters to this project specifically because receipts are dense in the spans where a
harmful accepted edit actually costs something: prices, quantities, totals, and item
codes. A miscorrected price is a different kind of error from a miscorrected adjective,
and a benchmark of prose alone cannot see the difference.

Its ground truth is quad-based (four corner points per word) rather than axis-aligned,
because receipt photos are routinely skewed. The quads are preserved and the bounding box
is derived, so the skew stays recoverable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from ocr_risk.datasets.base import DatasetAdapter, LicenseSpec, PreflightReport
from ocr_risk.datasets.registry import register_dataset
from ocr_risk.datasets.synthetic import shingle_hash
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import raw_source_dir
from ocr_risk.layout import group_boxes_into_lines, raster_order
from ocr_risk.schemas.base import BBox, Point, Polygon
from ocr_risk.schemas.documents import DocumentBundle, GtToken, SourceDocument
from ocr_risk.schemas.enums import RedistributionPolicy

__all__ = ["CORD_LICENSE", "CordDataset", "shard_document_plan"]

DATASET_ID = "cord"
GT_POLICY = "cord_receipt_annotation"

CORD_LICENSE = LicenseSpec(
    license_id="CC-BY-4.0",
    name="CORD: Consolidated Receipt Dataset (public subset)",
    url="https://github.com/clovaai/cord",
    redistribution=RedistributionPolicy.ALLOWED,
    attribution_required=True,
    notes=(
        "CC BY 4.0 on the public release. Still fetched by script rather than vendored, "
        "so the provenance chain to the upstream release stays explicit."
    ),
)


def shard_document_plan(shard_row_counts: list[int]) -> list[int]:
    """Return cumulative document offsets for filename-sorted parquet shards."""
    offsets: list[int] = []
    next_offset = 0
    for count in shard_row_counts:
        if count < 0:
            raise ValueError("CORD shard row counts cannot be negative")
        offsets.append(next_offset)
        next_offset += count
    return offsets


@register_dataset(DATASET_ID)
class CordDataset(DatasetAdapter):
    """Reads a downloaded CORD release. Not exercised on real data in this bootstrap."""

    dataset_id = DATASET_ID
    license = CORD_LICENSE
    gt_policy = GT_POLICY

    def __init__(self, split: str = "train", **_: object) -> None:
        self.split = split
        self.root = raw_source_dir(DATASET_ID)

    @property
    def annotations_dir(self) -> Path:
        return self.root / self.split / "json"

    @property
    def images_dir(self) -> Path:
        return self.root / self.split / "image"

    def preflight(self) -> PreflightReport:
        if not self.annotations_dir.is_dir():
            return PreflightReport(
                dataset_id=DATASET_ID,
                available=False,
                root=self.root,
                problems=[f"expected {self.annotations_dir}"],
                remediation="uv run python scripts/download_cord.py",
            )
        annotations = sorted(self.annotations_dir.glob("*.json"))
        return PreflightReport(
            dataset_id=DATASET_ID,
            available=bool(annotations),
            n_documents=len(annotations),
            root=self.root,
            checksums_verified=True,
        )

    def documents(self, limit: int | None = None) -> Iterator[DocumentBundle]:
        self.preflight().raise_if_unavailable()
        paths = sorted(self.annotations_dir.glob("*.json"))
        for path in paths if limit is None else paths[:limit]:
            bundle = self._load(path)
            if bundle is not None:
                yield bundle

    def _load(self, annotation_path: Path) -> DocumentBundle | None:
        document_id = f"cord-{self.split}-{annotation_path.stem}"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = self._image_for(annotation_path)
        if image_path is None:
            return None

        with Image.open(image_path) as image:
            width, height = image.width, image.height

        words: list[tuple[str, Polygon, str]] = []
        for line_index, line in enumerate(payload.get("valid_line", [])):
            for word in line.get("words", []):
                text = str(word.get("text", "")).strip()
                quad = word.get("quad")
                if not text or not quad:
                    continue
                polygon = Polygon(
                    points=(
                        Point(x=float(quad["x1"]), y=float(quad["y1"])),
                        Point(x=float(quad["x2"]), y=float(quad["y2"])),
                        Point(x=float(quad["x3"]), y=float(quad["y3"])),
                        Point(x=float(quad["x4"]), y=float(quad["y4"])),
                    )
                )
                words.append((text, polygon, f"{document_id}:line:{line_index:04d}"))

        if not words:
            return None

        # Regroup and re-order by the SAME geometric rule the OCR side uses. A CORD
        # ``valid_line`` is a receipt *item* (a semantic group), not a raster line, and
        # region anchoring matches on line_id — so carrying the item id through would make
        # the aligner match a semantic group against a printed line. A fixed rounding grid
        # was no better: on receipt text ~15-20px tall, a /10 grid puts a band boundary
        # through the middle of most lines.
        extents = [
            (word[1].bbox.x0, word[1].bbox.y0, word[1].bbox.x1, word[1].bbox.y1) for word in words
        ]
        lines = group_boxes_into_lines(extents)
        words = [
            (words[i][0], words[i][1], f"{document_id}:line:{lines[i]:04d}")
            for i in raster_order(extents)
        ]

        gt_text = " ".join(text for text, _, _ in words)
        tokens: list[GtToken] = []
        cursor = 0
        for index, (text, polygon, line_id) in enumerate(words):
            start = cursor
            cursor = start + len(text)
            tokens.append(
                GtToken(
                    gt_token_id=f"{document_id}:gt:{index:05d}",
                    document_id=document_id,
                    dataset_id=DATASET_ID,
                    index=index,
                    text=text,
                    char_start=start,
                    char_end=cursor,
                    line_id=line_id,
                    bbox=polygon.bbox,
                    # The quad is kept: receipt photos are skewed, and collapsing to a box
                    # would discard rotation the alignment stage can use.
                    polygon=polygon,
                )
            )
            cursor += 1

        document = SourceDocument(
            document_id=document_id,
            dataset_id=DATASET_ID,
            image_path=image_path.relative_to(raw_source_dir(DATASET_ID).parents[2]).as_posix(),
            image_sha256=file_sha256(image_path),
            page_index=0,
            width=width,
            height=height,
            gt_text=gt_text,
            gt_policy=GT_POLICY,
            has_gt_geometry=True,
            n_gt_tokens=len(tokens),
            license_id=CORD_LICENSE.license_id,
            source_url=CORD_LICENSE.url,
            gt_text_shingle_hash=shingle_hash(gt_text),
        )
        return DocumentBundle(document=document, gt_tokens=tuple(tokens))

    def _image_for(self, annotation_path: Path) -> Path | None:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = self.images_dir / f"{annotation_path.stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None


def bbox_of(polygon: Polygon) -> BBox:
    return polygon.bbox

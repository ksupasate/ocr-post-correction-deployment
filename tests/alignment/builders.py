"""Builders for alignment inputs from compact fixture declarations.

A plain module rather than a conftest: these are factories the battery calls with
different arguments per case, not per-test fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence

from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.documents import GtToken, SourceDocument
from ocr_risk.schemas.spans import CanonicalSpan

CHAR_WIDTH = 9.0
LINE_HEIGHT = 20.0
PAGE_WIDTH = 900
PAGE_HEIGHT = 300


def make_document(document_id: str = "doc-fixture", *, gt_text: str = "") -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        dataset_id="fixture",
        image_path=f"raw/source/fixture/{document_id}.png",
        image_sha256="0" * 64,
        page_index=0,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        gt_text=gt_text,
        gt_policy="fixture_exact",
        has_gt_geometry=True,
        n_gt_tokens=0,
        license_id="synthetic-generated",
    )


def _laid_out(texts: Sequence[str], *, line: int = 0) -> list[BBox]:
    """Left-to-right layout with proportional widths, so geometry is plausible."""
    boxes: list[BBox] = []
    x = 40.0
    y = 40.0 + line * LINE_HEIGHT
    for text in texts:
        width = max(len(text), 1) * CHAR_WIDTH
        boxes.append(BBox(x0=x, y0=y, x1=x + width, y1=y + LINE_HEIGHT - 4))
        x += width + CHAR_WIDTH
    return boxes


def make_gt_tokens(
    texts: Sequence[str], document_id: str = "doc-fixture", *, with_geometry: bool = True
) -> tuple[list[GtToken], str]:
    """Build ground-truth tokens with exact character offsets into the joined text."""
    boxes = _laid_out(texts)
    tokens: list[GtToken] = []
    cursor = 0
    for index, text in enumerate(texts):
        start = cursor
        cursor = start + len(text)
        tokens.append(
            GtToken(
                gt_token_id=f"{document_id}:gt:{index:04d}",
                document_id=document_id,
                dataset_id="fixture",
                index=index,
                text=text,
                char_start=start,
                char_end=cursor,
                line_id=f"{document_id}:gtline:000",
                bbox=boxes[index] if with_geometry else None,
            )
        )
        cursor += 1  # the joining space
    return tokens, " ".join(texts)


def make_spans(
    texts: Sequence[str],
    document_id: str = "doc-fixture",
    engine_id: str = "fixture_engine",
    *,
    with_geometry: bool = True,
) -> list[CanonicalSpan]:
    boxes = _laid_out(texts)
    spans: list[CanonicalSpan] = []
    cursor = 0
    for index, text in enumerate(texts):
        start = cursor
        cursor = start + len(text)
        spans.append(
            CanonicalSpan(
                span_id=f"{document_id}:{engine_id}:{index:05d}",
                document_id=document_id,
                dataset_id="fixture",
                engine_id=engine_id,
                engine_fingerprint="f" * 64,
                text=text,
                reading_order=index,
                line_id=f"{document_id}:ocrline:000",
                bbox=boxes[index] if with_geometry else None,
                native_conf_recognition=90.0,
                conf_scale="tesseract_word_conf_0_100",
                char_start=start,
                char_end=cursor,
                raw_ref="raw/ocr/fixture.json",
                raw_index=index,
            )
        )
        cursor += 1
    return spans

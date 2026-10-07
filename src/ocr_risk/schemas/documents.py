"""Source documents and ground-truth tokens.

A ``SourceDocument`` is the unit the benchmark holds constant across OCR engines: the same
page, with one shared ground truth, is read by every engine. That matched-source property
is what keeps engine shift from being confounded with document shift.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from ocr_risk.schemas.base import BBox, Polygon, RecordModel

__all__ = ["DocumentBundle", "GtToken", "SourceDocument"]


class GtToken(RecordModel):
    """One ground-truth token with its character offsets into the linearized GT text.

    Geometry is optional: plain-text ground truth (common in historical corpora) has none,
    and the alignment stage degrades to a text-only path rather than inventing boxes.
    """

    gt_token_id: str
    document_id: str
    dataset_id: str
    index: int = Field(ge=0)
    """Position in ground-truth reading order."""
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    """Exclusive end offset into ``SourceDocument.gt_text``."""
    line_id: str | None = None
    bbox: BBox | None = None
    polygon: Polygon | None = None


class SourceDocument(RecordModel):
    """A single page of source imagery plus its ground-truth transcription.

    ``image_path`` is relative to the data root and is never committed; ``image_sha256``
    is the stable identity used for split assignment, duplicate detection, and crop
    recipes, so a result stays traceable even when the image itself cannot be shared.
    """

    document_id: str
    dataset_id: str
    image_path: str
    image_sha256: str
    page_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    gt_text: str
    gt_policy: str
    """Identifier of the transcription convention actually used by this corpus, e.g.
    ``funsd_word_annotation`` or ``ocrd_diplomatic``. Corpora with different policies must
    not be silently merged into one benchmark track."""
    has_gt_geometry: bool
    n_gt_tokens: int = Field(ge=0)
    license_id: str
    work_id: str | None = None
    """The larger work a page belongs to -- a book, a volume, a multi-page form.

    Pages of one work share a typeface, a scan session, a binding and often a running
    head, so they must be assigned to a split role **as a unit**. The image-hash and
    shingle duplicate checks cannot see this: two pages of one book are genuinely
    different pages with different content, so nothing marks them as related, and a
    verifier fitted on pages 3-7 of a volume and evaluated on page 8 has seen that exact
    printing. ``None`` means the page stands alone.
    """
    source_url: str | None = None
    gt_text_shingle_hash: str | None = None
    """Hash of the GT text's shingle set, used to force near-duplicate pages into the same
    split bucket (leakage vector L8)."""


@dataclass(frozen=True, slots=True)
class DocumentBundle:
    """In-memory pairing of a document with its ground-truth tokens.

    Deliberately *not* a stored record: the two are persisted as separate normalized
    tables (``documents`` and ``gt_tokens``). This container exists only so a dataset
    adapter can yield both together.
    """

    document: SourceDocument
    gt_tokens: tuple[GtToken, ...]

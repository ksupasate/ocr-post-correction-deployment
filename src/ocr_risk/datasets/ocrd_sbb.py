"""OCR-D / SBB adapter — historical print, from PAGE-XML.

Written and fixture-tested but **not executed** during the bootstrap.

This is the hard track, and the one that most tests the architecture:

- **Line-level ground truth**, not word-level, so the aligner's segmentation handling is
  genuinely exercised rather than assumed.
- **Fraktur**, which needs ``deu`` or ``frk`` traineddata. A stock macOS Tesseract ships
  only ``eng``/``osd``/``snum``, and the adapter's preflight says so rather than letting
  someone recognize German with an English model and wonder at the CER.
- **Per-volume licensing**, commonly CC-BY or CC-BY-SA but not uniform, which is why the
  registry records this corpus as ``unclear`` and the audit treats it as prohibited until
  determined.
- **A diplomatic transcription policy**, which is why it must not be pooled with FUNSD or
  CORD without saying so: the same page transcribed under two policies is two different
  ground truths.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

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

__all__ = ["OCRD_SBB_LICENSE", "OcrdSbbDataset", "parse_page_xml"]

DATASET_ID = "ocrd_sbb"
GT_POLICY = "ocrd_diplomatic_line"

OCRD_SBB_LICENSE = LicenseSpec(
    license_id="CC-BY-SA-4.0",
    name="OCR-D ground truth, historical German prints (gt_structure_text)",
    url="https://github.com/OCR-D/gt_structure_text",
    redistribution=RedistributionPolicy.ALLOWED,
    attribution_required=True,
    notes=(
        "CC-BY-SA-4.0, read from the GitHub repository metadata on 2026-08-18 "
        "(license.spdx_id). This replaces an earlier 'unclear-per-volume' placeholder: "
        "'terms unknown' and 'terms forbid it' are different statements, and this one was "
        "resolvable by reading. The corpus aggregates works from several holding "
        "libraries, so a publication should still confirm terms per volume; nothing is "
        "redistributed from this repository in any case."
    ),
)

_PAGE_NAMESPACE = re.compile(r"\{(?P<uri>.+?)\}")


def _points(value: str) -> tuple[Point, ...]:
    """Parse a PAGE-XML ``points`` attribute: ``"x1,y1 x2,y2 ..."``."""
    parsed: list[Point] = []
    for pair in value.split():
        if "," not in pair:
            continue
        x, _, y = pair.partition(",")
        parsed.append(Point(x=float(x), y=float(y)))
    return tuple(parsed)


def parse_page_xml(path: Path) -> list[tuple[str, Polygon | None, str]]:
    """Extract ``(text, polygon, line_id)`` per transcription unit from a PAGE-XML file.

    Two things here were wrong in a way that produced a plausible-looking corpus.

    **The text was read with ``.//pc:TextEquiv``.** ``.//`` matches any descendant, and a
    PAGE-XML ``TextLine`` contains ``Word`` children that each carry their own
    ``TextEquiv`` -- so the search returned the first *word* of the line, not the line.
    Every line was silently truncated to its first word: a page whose ground truth is
    1138 characters was recorded as 132, and the engines' correct reading of the rest
    scored as hallucination. Measured CER against that ground truth was 5.9, i.e. the
    "errors" outnumbered the reference sixfold. The line's own text is its **direct**
    ``TextEquiv`` child.

    **Word-level units are used where the annotation provides them.** OCR-D describes
    itself as a diplomatic *line* transcription, and where only lines are annotated that
    is what is returned. But these volumes do annotate words, with their own coordinates,
    so word units are read rather than invented -- which keeps the historical track at
    the same granularity as the other two corpora and as the normalized OCR spans. A
    granularity mismatch between corpora would confound the cross-corpus stratum exactly
    as a mismatch between engines would confound the transfer comparison.
    """
    tree = ElementTree.parse(path)
    root = tree.getroot()
    match = _PAGE_NAMESPACE.match(root.tag)
    if match is None:
        # Without the namespace the TextEquiv lookup below is skipped entirely and every
        # line comes back empty, so the page silently contributes nothing. An
        # unrecognized schema is a fact about the input, not an empty page.
        msg = (
            f"{path} has no XML namespace on its root element <{root.tag}>; this does not "
            "look like PAGE-XML, and parsing it would silently yield an empty page"
        )
        raise ValueError(msg)
    namespace = {"pc": match.group("uri")}

    def _text_of(element: ElementTree.Element) -> str:
        """This element's OWN transcription, never a descendant's."""
        equiv = element.find("pc:TextEquiv/pc:Unicode", namespace)
        return (equiv.text or "").strip() if equiv is not None else ""

    def _polygon_of(element: ElementTree.Element) -> Polygon | None:
        coords = element.find("pc:Coords", namespace)
        if coords is None or not coords.get("points"):
            return None
        points = _points(coords.get("points", ""))
        return Polygon(points=points) if len(points) >= 3 else None

    units: list[tuple[str, Polygon | None, str]] = []
    for index, line in enumerate(root.iter(f"{{{namespace['pc']}}}TextLine")):
        line_id = line.get("id") or f"line-{index:04d}"
        # findall, not iter: direct children only. A nested descendant search duplicated
        # text where a Word appeared under more than one ancestor.
        words = line.findall("pc:Word", namespace)
        word_units = [(_text_of(word), _polygon_of(word), line_id) for word in words]
        word_units = [unit for unit in word_units if unit[0]]
        if word_units:
            units.extend(word_units)
            continue
        text = _text_of(line)
        if text:
            units.append((text, _polygon_of(line), line_id))
    return units


@register_dataset(DATASET_ID)
class OcrdSbbDataset(DatasetAdapter):
    """Reads OCR-D PAGE-XML ground truth. Not exercised on real data in this bootstrap."""

    dataset_id = DATASET_ID
    license = OCRD_SBB_LICENSE
    gt_policy = GT_POLICY

    def __init__(self, volume: str = "", **_: object) -> None:
        self.volume = volume
        self.root = raw_source_dir(DATASET_ID) / volume if volume else raw_source_dir(DATASET_ID)

    # Both prerequisites are reported together, always. Someone missing the corpus is
    # usually also missing the language data, and finding that out only after downloading
    # several gigabytes is a poor experience.
    REMEDIATION = (
        "uv run python scripts/download_ocrd_sbb.py --volume <id>; "
        "German Fraktur also needs deu/frk traineddata: brew install tesseract-lang"
    )

    # OCR-D BagIt volumes ship the SAME PAGE-XML under up to three fileGrp directories.
    # They are byte-identical: on one real volume the three copies hash alike. Globbing
    # *.xml therefore finds 3N + 1 files for an N-page volume (the +1 being mets.xml),
    # and since document_id is built from the file stem, all three copies collide on one
    # id -- which downstream concatenates a page's ground truth with two copies of
    # itself. Preference order is most-granular-first; not every volume ships all three.
    _GT_FILE_GROUPS = ("OCR-D-GT-SEG-LINE", "OCR-D-GT-SEG-BLOCK", "OCR-D-GT-SEG-PAGE")
    _IMAGE_GROUP = "OCR-D-IMG"

    def _page_files(self) -> list[Path]:
        """PAGE-XML files from exactly one fileGrp **per volume**, most granular available.

        Per volume, not per corpus: volumes differ in which fileGrps they ship. Choosing
        one group for the whole corpus silently dropped every volume that lacks it -- 14
        of 102 pages on the pilot selection.
        """
        by_volume: dict[Path, list[Path]] = {}
        for group in self._GT_FILE_GROUPS:
            for path in sorted(self.root.rglob(f"{group}/*.xml")):
                if path.name.lower() == "mets.xml":
                    continue
                # The volume root is the grandparent of the fileGrp directory:
                #   <volume>/data/<GROUP>/<page>.xml
                volume = path.parent.parent
                if volume in by_volume and by_volume[volume][0].parent.name != group:
                    continue  # a more granular group already claimed this volume
                by_volume.setdefault(volume, []).append(path)
        return sorted(page for pages in by_volume.values() for page in pages)

    def preflight(self) -> PreflightReport:
        if not self.root.is_dir():
            return PreflightReport(
                dataset_id=DATASET_ID,
                available=False,
                root=self.root,
                problems=[f"expected a volume directory at {self.root}"],
                remediation=self.REMEDIATION,
            )
        pages = self._page_files()
        problems: list[str] = []
        if self.license.redistribution is RedistributionPolicy.UNCLEAR:
            problems.append("licensing is per-volume and not yet determined; treated as prohibited")
        if not pages:
            problems.append(
                f"no PAGE-XML under any of {list(self._GT_FILE_GROUPS)}; is this an "
                "extracted OCR-D BagIt volume?"
            )
        # Count pages that will actually LOAD, not files on disk. Counting files reported
        # a healthy corpus while documents() yielded nothing, because _load returns None
        # when the image cannot be found -- a silent empty corpus.
        loadable = [path for path in pages if self._image_for(path) is not None]
        if pages and not loadable:
            problems.append(
                f"found {len(pages)} PAGE-XML file(s) but no matching image under "
                f"{self._IMAGE_GROUP}/; the corpus would load as empty"
            )
        elif len(loadable) < len(pages):
            problems.append(f"{len(pages) - len(loadable)} page(s) have no matching image")
        return PreflightReport(
            dataset_id=DATASET_ID,
            available=bool(loadable),
            n_documents=len(loadable),
            root=self.root,
            problems=problems,
            remediation=self.REMEDIATION,
        )

    def documents(self, limit: int | None = None) -> Iterator[DocumentBundle]:
        self.preflight().raise_if_unavailable()
        paths = self._page_files()
        for path in paths if limit is None else paths[:limit]:
            bundle = self._load(path)
            if bundle is not None:
                yield bundle

    def _load(self, page_path: Path) -> DocumentBundle | None:
        image_path = self._image_for(page_path)
        if image_path is None:
            return None
        lines = parse_page_xml(page_path)
        if not lines:
            return None

        with Image.open(image_path) as image:
            width, height = image.width, image.height

        # The work identifier comes from the path, not from the constructor: a corpus
        # loaded from the root would otherwise label every page "root" and lose the volume
        # entirely. Pages of one book share a typeface, a scan session and a binding, so
        # the work must be recoverable for same-work grouping in the document partition.
        work_id = self._work_of(page_path)
        document_id = f"ocrd-{work_id}-{page_path.stem}"

        # Re-order lines into raster reading order from their own geometry, the same
        # derivation the OCR side uses. PAGE-XML document order follows the annotation's
        # region structure, which is not the order a reader -- or an engine -- traverses
        # the page, and using it would make the ground-truth stream disagree with every
        # engine for reasons of annotation convention rather than recognition.
        boxes: list[tuple[float, float, float, float] | None] = [
            None
            if polygon is None
            else (polygon.bbox.x0, polygon.bbox.y0, polygon.bbox.x1, polygon.bbox.y1)
            for _, polygon, _ in lines
        ]
        if any(box is not None for box in boxes):
            # Re-derive line_id from geometry with the SHARED grouper, as FUNSD and CORD
            # already do. Keeping the PAGE-XML TextLine id made OCR-D the one corpus where
            # the two sides of an alignment were grouped by different rules -- region
            # anchoring matches on line_id, so it compared geometric OCR line-groups
            # against annotation TextLine groups. On 27 of 102 pages those disagree on more
            # than 10% of adjacent token pairs (multi-column and marginalia chain
            # together), and the disagreement correlates with the out-of-region rate at
            # rho = -0.46 to -0.61 per engine, unequally across engines.
            grouped = group_boxes_into_lines(boxes)
            order = raster_order(boxes)
            lines = [
                (lines[i][0], lines[i][1], f"{document_id}:line:{grouped[i]:04d}") for i in order
            ]

        # Words are separated by a space and lines by a newline. Joining every unit with
        # a newline -- correct when the units were lines -- would make a word boundary
        # indistinguishable from a line break in the character stream the aligner walks.
        separators = [
            "\n" if index and lines[index - 1][2] != line_id else " "
            for index, (_, _, line_id) in enumerate(lines)
        ]
        gt_text = "".join(
            (separator if index else "") + text
            for index, ((text, _, _), separator) in enumerate(zip(lines, separators, strict=True))
        )

        tokens: list[GtToken] = []
        cursor = 0
        for index, (text, polygon, line_id) in enumerate(lines):
            if index:
                cursor += 1  # the separator written before this unit
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
                    bbox=polygon.bbox if polygon else None,
                    polygon=polygon,
                )
            )

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
            has_gt_geometry=any(t.bbox is not None for t in tokens),
            n_gt_tokens=len(tokens),
            license_id=OCRD_SBB_LICENSE.license_id,
            work_id=work_id,
            source_url=OCRD_SBB_LICENSE.url,
            gt_text_shingle_hash=shingle_hash(gt_text),
        )
        return DocumentBundle(document=document, gt_tokens=tuple(tokens))

    def _work_of(self, page_path: Path) -> str:
        """The volume a page belongs to, from its position in the BagIt layout."""
        if self.volume:
            return self.volume
        # <root>/<volume>/data/<GROUP>/<page>.xml
        volume_dir = page_path.parent.parent.parent
        root = raw_source_dir(DATASET_ID)
        return volume_dir.name if volume_dir != root else "root"

    def _image_for(self, page_path: Path) -> Path | None:
        """Find the page image for a PAGE-XML file.

        A real OCR-D BagIt volume keeps images in a sibling ``OCR-D-IMG/`` directory, as
        ``.tif`` in some volumes and ``.jpg`` in others -- not beside the XML, and not in
        a directory called ``images``. Looking only in the wrong two places returned None
        for every page, and since ``_load`` treats that as "skip", the adapter yielded an
        empty corpus while preflight reported it available.

        ``Path.stem`` rather than ``with_suffix``: a stem like ``PPN123_0001.nrm`` would
        have ``with_suffix`` replace ``.nrm``, so the two lookups disagreed on the name.
        """
        stem = page_path.stem
        directories = [
            page_path.parent.parent / self._IMAGE_GROUP,
            page_path.parent,
            page_path.parent.parent / "images",
        ]
        for directory in directories:
            for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                candidate = directory / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
        return None


def bbox_of(polygon: Polygon) -> BBox:
    return polygon.bbox

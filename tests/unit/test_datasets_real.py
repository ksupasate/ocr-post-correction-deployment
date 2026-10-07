"""Real-corpus adapters, tested from fixtures so CI needs no download.

The FUNSD adapter is also exercised against the actual corpus when it happens to be
present locally, marked so CI skips it.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from ocr_risk.datasets import ChecksumMismatchError, DownloadSpec, download
from ocr_risk.datasets.download import extract_zip
from ocr_risk.datasets.funsd import FunsdDataset
from ocr_risk.io.hashing import file_sha256, sha256_of_bytes
from ocr_risk.schemas.enums import RedistributionPolicy


def _write_funsd_fixture(root: Path, split: str = "training_data") -> None:
    """A miniature FUNSD release, including the quirks the adapter must handle."""
    annotations = root / "dataset" / split / "annotations"
    images = root / "dataset" / split / "images"
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    Image.new("L", (600, 400), color=255).save(images / "form1.png")
    (annotations / "form1.json").write_text(
        json.dumps(
            {
                "form": [
                    # Entity order deliberately disagrees with raster order: the adapter
                    # must re-sort, or GT would disagree with every engine's reading.
                    {
                        "id": 0,
                        "words": [
                            {"text": "SIGNATURE", "box": [40, 300, 160, 320]},
                            {"text": "Smith", "box": [180, 300, 260, 320]},
                        ],
                    },
                    {
                        "id": 1,
                        "words": [
                            {"text": "Dose:", "box": [40, 60, 100, 80]},
                            {"text": "0.015", "box": [110, 60, 170, 80]},
                            # An inverted box, which FUNSD does contain.
                            {"text": "mg", "box": [220, 80, 180, 60]},
                        ],
                    },
                    {"id": 2, "words": [{"text": "   ", "box": [0, 0, 10, 10]}]},
                ]
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def funsd_root(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("funsd")
    _write_funsd_fixture(root)
    return root


# --- FUNSD adapter --------------------------------------------------------------------
def test_funsd_licence_forbids_redistribution() -> None:
    """The corpus is research-use only; the audit relies on this being recorded."""
    assert FunsdDataset().license.redistribution is RedistributionPolicy.PROHIBITED
    assert FunsdDataset().license.attribution_required


def test_funsd_preflight_reports_remediation_when_absent(isolated_env: Path) -> None:
    report = FunsdDataset().preflight()
    assert not report.available
    assert "download_funsd" in report.remediation


def test_funsd_loads_a_document(funsd_root: Path) -> None:
    bundles = list(FunsdDataset().documents())
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.document.has_gt_geometry
    assert bundle.document.gt_policy == "funsd_word_annotation"
    assert bundle.document.n_gt_tokens == len(bundle.gt_tokens)


def test_funsd_reorders_ground_truth_into_raster_order(funsd_root: Path) -> None:
    """Entity order is a structural annotation, not a transcription order. Using it would
    make GT disagree with every engine's reading purely by convention."""
    bundle = next(iter(FunsdDataset().documents()))
    texts = [t.text for t in bundle.gt_tokens]
    # The Dose: line is at y=60 and the signature at y=300, so it must come first even
    # though it is the second entity in the file.
    assert texts.index("Dose:") < texts.index("SIGNATURE")


def test_funsd_gt_offsets_index_the_gt_text(funsd_root: Path) -> None:
    bundle = next(iter(FunsdDataset().documents()))
    text = bundle.document.gt_text
    for token in bundle.gt_tokens:
        assert text[token.char_start : token.char_end] == token.text


def test_funsd_normalizes_inverted_boxes(funsd_root: Path) -> None:
    """FUNSD contains boxes stored with corners reversed; the schema would reject them."""
    bundle = next(iter(FunsdDataset().documents()))
    for token in bundle.gt_tokens:
        assert token.bbox is not None
        assert token.bbox.x1 >= token.bbox.x0
        assert token.bbox.y1 >= token.bbox.y0


def test_funsd_drops_whitespace_only_words(funsd_root: Path) -> None:
    bundle = next(iter(FunsdDataset().documents()))
    assert all(t.text.strip() for t in bundle.gt_tokens)


def test_funsd_keeps_entity_structure_as_line_ids(funsd_root: Path) -> None:
    """Re-ordering must not discard the form structure."""
    bundle = next(iter(FunsdDataset().documents()))
    assert len({t.line_id for t in bundle.gt_tokens}) >= 2


def test_funsd_limit_is_respected(funsd_root: Path) -> None:
    assert list(FunsdDataset().documents(limit=0)) == []


# --- downloads -----------------------------------------------------------------------------
def test_download_verifies_the_checksum(tmp_path: Path) -> None:
    payload = b"corpus bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    destination = tmp_path / "out" / "source.bin"

    spec = DownloadSpec(
        url=source.as_uri(), destination=destination, sha256=sha256_of_bytes(payload)
    )
    assert download(spec) == destination
    assert file_sha256(destination) == spec.sha256


def test_download_rejects_a_corrupted_file(tmp_path: Path) -> None:
    """A corrupted corpus produces plausible-looking numbers, so this must be an error."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"actual bytes")
    destination = tmp_path / "out.bin"
    spec = DownloadSpec(url=source.as_uri(), destination=destination, sha256="0" * 64)
    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        download(spec)
    assert not destination.exists()


def test_download_refuses_an_existing_mismatched_file(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"stale")
    source = tmp_path / "source.bin"
    source.write_bytes(b"fresh")
    spec = DownloadSpec(
        url=source.as_uri(), destination=destination, sha256=sha256_of_bytes(b"fresh")
    )
    with pytest.raises(ChecksumMismatchError, match="does not match its recorded sha256"):
        download(spec)


def test_download_is_idempotent(tmp_path: Path) -> None:
    payload = b"stable"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    destination = tmp_path / "out.bin"
    spec = DownloadSpec(
        url=source.as_uri(), destination=destination, sha256=sha256_of_bytes(payload)
    )
    download(spec)
    first = destination.stat().st_mtime_ns
    messages: list[str] = []
    download(spec, progress=messages.append)
    assert destination.stat().st_mtime_ns == first
    assert any("already present" in m for m in messages)


def test_extraction_is_keyed_on_the_archive_digest(tmp_path: Path) -> None:
    """Not on the target directory existing: the download creates that directory, so an
    existence check silently skips extraction and leaves an empty corpus."""
    archive = tmp_path / "a.zip"
    target = tmp_path / "target"
    target.mkdir()  # as the download would have done
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/file.txt", "content")

    extract_zip(archive, target)
    assert (target / "dataset" / "file.txt").is_file()
    assert (target / ".extracted.sha256").is_file()


def test_extraction_skips_when_already_done(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    target = tmp_path / "target"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("f.txt", "v1")
    extract_zip(archive, target)
    (target / "f.txt").write_text("edited")
    extract_zip(archive, target)
    assert (target / "f.txt").read_text() == "edited"


def test_extraction_refuses_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "gotcha")
    with pytest.raises(ValueError, match="escapes the target directory"):
        extract_zip(archive, tmp_path / "target")


# --- the real corpus, when it is present locally --------------------------------------------
def _real_funsd_present() -> bool:
    from ocr_risk.io.paths import raw_source_dir

    return (raw_source_dir("funsd") / "dataset" / "training_data" / "annotations").is_dir()


@pytest.mark.requires_network
@pytest.mark.skipif(not _real_funsd_present(), reason="FUNSD is not downloaded")
def test_real_funsd_loads_and_is_well_formed() -> None:
    bundles = list(FunsdDataset().documents(limit=10))
    assert len(bundles) == 10
    for bundle in bundles:
        assert bundle.gt_tokens
        text = bundle.document.gt_text
        for token in bundle.gt_tokens:
            assert text[token.char_start : token.char_end] == token.text
            assert token.bbox is not None
            assert 0 <= token.bbox.x0 <= bundle.document.width


# --- CORD and OCR-D: written and fixture-tested, not executed on real data -----------------
def test_cord_parses_quad_geometry(isolated_env: Path) -> None:
    """Receipt photos are skewed, so the quad is preserved and the box derived from it."""
    from ocr_risk.datasets.cord import CordDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("cord")
    (root / "train" / "json").mkdir(parents=True)
    (root / "train" / "image").mkdir(parents=True)
    Image.new("RGB", (300, 500), color="white").save(root / "train" / "image" / "r1.png")
    (root / "train" / "json" / "r1.json").write_text(
        json.dumps(
            {
                "valid_line": [
                    {
                        "words": [
                            {
                                "text": "TOTAL",
                                "quad": {
                                    "x1": 10,
                                    "y1": 100,
                                    "x2": 70,
                                    "y2": 104,
                                    "x3": 70,
                                    "y3": 124,
                                    "x4": 10,
                                    "y4": 120,
                                },
                            },
                            {
                                "text": "12.50",
                                "quad": {
                                    "x1": 200,
                                    "y1": 100,
                                    "x2": 260,
                                    "y2": 104,
                                    "x3": 260,
                                    "y3": 124,
                                    "x4": 200,
                                    "y4": 120,
                                },
                            },
                        ]
                    }
                ]
            }
        )
    )

    bundle = next(iter(CordDataset().documents()))
    assert [t.text for t in bundle.gt_tokens] == ["TOTAL", "12.50"]
    assert all(t.polygon is not None for t in bundle.gt_tokens)
    # The box is derived from the quad, so skew stays recoverable.
    assert bundle.gt_tokens[0].bbox == bundle.gt_tokens[0].polygon.bbox  # type: ignore[union-attr]


def test_cord_licence_permits_redistribution() -> None:
    from ocr_risk.datasets.cord import CordDataset

    assert CordDataset().license.may_redistribute
    assert CordDataset().license.attribution_required


def test_ocrd_parses_line_level_page_xml(isolated_env: Path, tmp_path: Path) -> None:
    """Line-level, because that is what OCR-D ground truth actually provides. Splitting
    into pseudo-words would invent a segmentation the annotation does not assert."""
    from ocr_risk.datasets.ocrd_sbb import parse_page_xml

    page = tmp_path / "p1.xml"
    page.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageWidth="800" imageHeight="1200">
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords points="40,60 400,60 400,90 40,90"/>
        <TextEquiv><Unicode>Der Patient erhielt eine Dosis</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l2">
        <Coords points="40,100 380,100 380,130 40,130"/>
        <TextEquiv><Unicode>von 0,015 mg täglich.</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l3">
        <Coords points="40,140 380,140 380,170 40,170"/>
        <TextEquiv><Unicode></Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>""",
        encoding="utf-8",
    )
    lines = parse_page_xml(page)
    assert len(lines) == 2, "an empty TextEquiv must not become a ground-truth line"
    assert lines[0][0] == "Der Patient erhielt eine Dosis"
    assert lines[0][1] is not None
    assert lines[0][2] == "l1"


def test_ocrd_licence_was_determined_and_matches_the_registry() -> None:
    """`unclear` was a placeholder for "not yet looked up", and it was looked up.

    The terms were read from the upstream repository metadata (CC-BY-SA-4.0) and recorded
    with the date. What must not drift is the adapter and the registry disagreeing: the
    licence audit reads the registry, so a permissive adapter beside a restrictive
    registry entry would let a corpus through on the wrong terms.
    """
    import yaml

    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import project_root

    spec = OcrdSbbDataset().license
    assert spec.redistribution is RedistributionPolicy.ALLOWED
    assert spec.attribution_required, "CC-BY-SA requires attribution"

    registry = yaml.safe_load(
        (project_root() / "manifests" / "licenses" / "registry.yaml").read_text("utf-8")
    )
    entry = registry["datasets"]["ocrd_sbb"]
    assert entry["license_id"] == spec.license_id
    assert entry["redistribution"] == spec.redistribution.value


def test_an_undetermined_licence_is_treated_as_prohibited() -> None:
    """The `unclear` path must stay unsafe for the next corpus that needs it.

    "We could not determine the terms" and "the terms forbid it" are different
    statements, and the audit must treat the unresolved one as unsafe rather than
    optimistic.
    """
    from ocr_risk.datasets.base import LicenseSpec

    spec = LicenseSpec(
        license_id="unknown",
        name="hypothetical corpus",
        url="",
        redistribution=RedistributionPolicy.UNCLEAR,
        attribution_required=True,
    )
    assert not spec.may_redistribute


def test_ocrd_preflight_warns_about_fraktur_traineddata(isolated_env: Path) -> None:
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset

    report = OcrdSbbDataset().preflight()
    assert "deu/frk" in report.remediation


# --- OCR-D BagIt layout: the file-selection logic that already lost pages twice ----------


def _page_xml(text_by_line: dict[str, str]) -> str:
    lines = "\n".join(
        f"""      <TextLine id="{line_id}">
        <Coords points="40,{60 + 40 * i} 400,{60 + 40 * i} 400,{90 + 40 * i} 40,{90 + 40 * i}"/>
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>
      </TextLine>"""
        for i, (line_id, text) in enumerate(text_by_line.items())
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageWidth="800" imageHeight="1200">
    <TextRegion id="r1">
{lines}
    </TextRegion>
  </Page>
</PcGts>"""


def _ocrd_volume(
    root: Path,
    volume: str,
    *,
    groups: tuple[str, ...],
    pages: tuple[str, ...] = ("PPN_0001",),
    image_suffix: str | None = ".tif",
) -> None:
    """Write one BagIt-shaped volume: ``<root>/<volume>/data/<GROUP>/<page>.xml``."""
    from PIL import Image

    for group in groups:
        directory = root / volume / "data" / group
        directory.mkdir(parents=True, exist_ok=True)
        for page in pages:
            (directory / f"{page}.xml").write_text(
                _page_xml({"l1": f"{group} erste Zeile", "l2": "zweite Zeile"}), encoding="utf-8"
            )
    if image_suffix is not None:
        images = root / volume / "data" / "OCR-D-IMG"
        images.mkdir(parents=True, exist_ok=True)
        for page in pages:
            Image.new("RGB", (800, 1200), "white").save(images / f"{page}{image_suffix}")


def test_ocrd_reads_the_most_granular_file_group_each_volume_actually_ships(
    isolated_env: Path,
) -> None:
    """Per volume, not per corpus. Choosing one group corpus-wide dropped 14 of 102 pages.

    Volume A ships SEG-LINE and SEG-PAGE; volume B ships only SEG-PAGE. A corpus-wide
    choice of SEG-LINE loses B entirely, and a corpus-wide SEG-PAGE reads A at the coarser
    granularity while a finer annotation sits unused next to it.
    """
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("ocrd_sbb")
    _ocrd_volume(root, "vol_a", groups=("OCR-D-GT-SEG-LINE", "OCR-D-GT-SEG-PAGE"))
    _ocrd_volume(root, "vol_b", groups=("OCR-D-GT-SEG-PAGE",))

    bundles = {b.document.work_id: b for b in OcrdSbbDataset().documents()}
    assert set(bundles) == {"vol_a", "vol_b"}, "a volume without the finest group was dropped"
    assert "OCR-D-GT-SEG-LINE" in bundles["vol_a"].document.gt_text
    assert "OCR-D-GT-SEG-PAGE" in bundles["vol_b"].document.gt_text


def test_ocrd_never_reads_one_page_from_three_copies_of_itself(isolated_env: Path) -> None:
    """The three GT fileGrps are byte-identical copies, so globbing *.xml triples a page."""
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("ocrd_sbb")
    _ocrd_volume(
        root,
        "vol_a",
        groups=("OCR-D-GT-SEG-LINE", "OCR-D-GT-SEG-BLOCK", "OCR-D-GT-SEG-PAGE"),
        pages=("PPN_0001", "PPN_0002"),
    )
    documents = [b.document for b in OcrdSbbDataset().documents()]
    assert len(documents) == 2
    assert len({d.document_id for d in documents}) == 2
    for document in documents:
        assert document.gt_text.count("erste Zeile") == 1


def test_ocrd_preflight_refuses_a_corpus_whose_pages_have_no_images(isolated_env: Path) -> None:
    """Counting XML files reported a healthy corpus while ``documents()`` yielded nothing."""
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("ocrd_sbb")
    _ocrd_volume(root, "vol_a", groups=("OCR-D-GT-SEG-LINE",), image_suffix=None)
    report = OcrdSbbDataset().preflight()
    assert not report.available
    assert report.n_documents == 0
    assert any("would load as empty" in problem for problem in report.problems)


def test_ocrd_preflight_counts_only_the_pages_that_will_load(isolated_env: Path) -> None:
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("ocrd_sbb")
    _ocrd_volume(root, "vol_a", groups=("OCR-D-GT-SEG-LINE",), pages=("PPN_0001", "PPN_0002"))
    (root / "vol_a" / "data" / "OCR-D-IMG" / "PPN_0002.tif").unlink()
    report = OcrdSbbDataset().preflight()
    assert report.available
    assert report.n_documents == 1
    assert any("no matching image" in problem for problem in report.problems)


@pytest.mark.parametrize("suffix", [".tif", ".jpg", ".png"])
def test_ocrd_finds_the_page_image_whatever_the_volume_encoded_it_as(
    isolated_env: Path, suffix: str
) -> None:
    """Volumes differ: some ship TIFF, some JPEG. Looking for one of them found none."""
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("ocrd_sbb")
    _ocrd_volume(root, "vol_a", groups=("OCR-D-GT-SEG-LINE",), image_suffix=suffix)
    assert OcrdSbbDataset().preflight().n_documents == 1


def test_ocrd_ground_truth_separates_lines_by_newline_and_carries_geometry(
    isolated_env: Path,
) -> None:
    """A word boundary and a line break must not be the same character in the GT stream."""
    from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
    from ocr_risk.io.paths import raw_source_dir

    root = raw_source_dir("ocrd_sbb")
    _ocrd_volume(root, "vol_a", groups=("OCR-D-GT-SEG-LINE",))
    bundle = next(iter(OcrdSbbDataset().documents()))
    assert "\n" in bundle.document.gt_text
    assert bundle.document.has_gt_geometry
    assert bundle.document.work_id == "vol_a"
    # Every token's char span must index the text it came from, or alignment is nonsense.
    for token in bundle.gt_tokens:
        assert bundle.document.gt_text[token.char_start : token.char_end] == token.text


def test_ocrd_rejects_a_file_that_is_not_page_xml_rather_than_reading_it_as_empty(
    isolated_env: Path, tmp_path: Path
) -> None:
    from ocr_risk.datasets.ocrd_sbb import parse_page_xml

    page = tmp_path / "not_page.xml"
    page.write_text("<root><TextLine/></root>", encoding="utf-8")
    with pytest.raises(ValueError, match="no XML namespace"):
        parse_page_xml(page)

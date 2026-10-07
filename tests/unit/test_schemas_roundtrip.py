"""Schema parity and lossless round-trip.

Arrow schemas are hand-written so the physical artifact layout is reviewable in one place.
That only stays safe if drift is impossible, which is what these tests enforce.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from ocr_risk.schemas import (
    TABLES,
    AlignmentRecord,
    AlignmentRelation,
    AlignmentStatus,
    BBox,
    Candidate,
    CandidateLabel,
    CanonicalSpan,
    ConfidenceFeatures,
    CorrectionSite,
    Decision,
    DecisionAction,
    EvidenceBundle,
    EvidenceField,
    GeometryFeatures,
    GtToken,
    OutcomeIfAccepted,
    OutcomeIfRejected,
    Point,
    Polygon,
    Prediction,
    SiteKind,
    SourceDocument,
    TableSpec,
    records_to_table,
    table_to_records,
)
from ocr_risk.schemas.arrow import allows_none

BOX = BBox(x0=10.0, y0=20.0, x1=110.0, y1=48.0)
POLY = Polygon.from_bbox(BOX)


def _example(spec: TableSpec) -> object:
    """One fully populated record per table, with every nullable field set.

    Nullables are populated here and blanked in the null-round-trip test, so both paths
    are exercised rather than only the convenient one.
    """
    builders = {
        "documents": lambda: SourceDocument(
            document_id="doc-0001",
            dataset_id="synthetic",
            image_path="raw/source/synthetic/doc-0001.png",
            image_sha256="a" * 64,
            page_index=0,
            width=1240,
            height=1754,
            gt_text="Total 0.015 mg",
            gt_policy="synthetic_exact",
            has_gt_geometry=True,
            n_gt_tokens=3,
            license_id="synthetic-generated",
            source_url="https://example.invalid/doc-0001",
            gt_text_shingle_hash="b" * 64,
        ),
        "gt_tokens": lambda: GtToken(
            gt_token_id="doc-0001:gt:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            index=2,
            text="mg",
            char_start=12,
            char_end=14,
            line_id="line-0",
            bbox=BOX,
            polygon=POLY,
        ),
        "spans": lambda: CanonicalSpan(
            span_id="doc-0001:tesseract:00002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            engine_fingerprint="c" * 64,
            text="rng",
            reading_order=2,
            line_id="line-0",
            block_id="block-0",
            bbox=BOX,
            polygon=POLY,
            native_conf_recognition=87.0,
            native_conf_detection=0.93,
            conf_scale="tesseract_word_conf_0_100",
            char_start=12,
            char_end=15,
            raw_ref="raw/ocr/synthetic/tesseract/abcd1234/doc-0001.json",
            raw_index=2,
        ),
        "alignments": lambda: AlignmentRecord(
            alignment_id="doc-0001:tesseract:al:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            ocr_span_ids=("doc-0001:tesseract:00002",),
            gt_token_ids=("doc-0001:gt:0002",),
            relation=AlignmentRelation.ONE_TO_ONE,
            status=AlignmentStatus.RESOLVED,
            ocr_text="rng",
            gt_text="mg",
            align_confidence=0.72,
            char_agreement=0.34,
            geom_score=0.91,
            uniqueness_margin=0.55,
            edit_distance=2,
            region_id="region-0",
            bbox=BOX,
            reason_code=None,
            diagnostics={"geometry": "present", "band_width": "12"},
        ),
        "sites": lambda: CorrectionSite(
            site_id="doc-0001:tesseract:site:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            alignment_ids=("doc-0001:tesseract:al:0002",),
            ocr_span_ids=("doc-0001:tesseract:00002",),
            gt_token_ids=("doc-0001:gt:0002",),
            ocr_text="rng",
            gt_text="mg",
            d_before=2,
            site_kind=SiteKind.SUBSTITUTION,
            evaluable=True,
            char_start=12,
            char_end=15,
            reading_order_start=2,
            bbox=BOX,
            min_align_confidence=0.72,
        ),
        "candidates": lambda: Candidate(
            candidate_id="doc-0001:tesseract:site:0002:lexical:0",
            site_id="doc-0001:tesseract:site:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            candidate_text="mg",
            generator_id="lexical",
            generator_version="1",
            generator_rank=0,
            generator_score=-1.25,
            is_synthetic_hard_negative=False,
            hard_negative_family=None,
            metadata={"source": "lexicon"},
        ),
        "labels": lambda: CandidateLabel(
            candidate_id="doc-0001:tesseract:site:0002:lexical:0",
            site_id="doc-0001:tesseract:site:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            d_before=2,
            d_after=0,
            delta=2,
            outcome_if_accepted=OutcomeIfAccepted.TRUE_CORRECTION,
            outcome_if_rejected=OutcomeIfRejected.MISSED_ERROR,
            gt_text="mg",
        ),
        "evidence": lambda: EvidenceBundle(
            candidate_id="doc-0001:tesseract:site:0002:lexical:0",
            site_id="doc-0001:tesseract:site:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            original_ocr="rng",
            candidate_text="mg",
            context_before="Total 0.015 ",
            context_after="",
            crop_recipe_sha256="d" * 64,
            conf_features=ConfidenceFeatures(
                native_min=87.0,
                native_mean=87.0,
                conf_normalized=0.87,
                conf_rank_in_line=0.5,
                conf_zscore_within_engine=-1.2,
                n_spans=1,
                has_native_confidence=True,
            ),
            geom_features=GeometryFeatures(
                width=100.0,
                height=28.0,
                aspect_ratio=100.0 / 28.0,
                relative_x=0.008,
                relative_y=0.011,
                relative_area=0.0013,
                n_spans=1,
                has_geometry=True,
            ),
            available_fields=(EvidenceField.ORIGINAL_OCR, EvidenceField.IMAGE_CROP),
            masked_fields=(EvidenceField.SPATIAL,),
        ),
        "predictions": lambda: Prediction(
            candidate_id="doc-0001:tesseract:site:0002:lexical:0",
            site_id="doc-0001:tesseract:site:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            verifier_id="feature_lr",
            evidence_config="v6_full",
            fold_id="loeo:tesseract",
            raw_score=1.83,
            calibrated_score=0.86,
            calibrator_id="isotonic:abc123",
        ),
        "decisions": lambda: Decision(
            site_id="doc-0001:tesseract:site:0002",
            document_id="doc-0001",
            dataset_id="synthetic",
            engine_id="tesseract",
            method_id="feature_lr|v6_full|isotonic",
            fold_id="loeo:tesseract",
            epsilon=0.01,
            tau=0.82,
            action=DecisionAction.CORRECT,
            accepted_candidate_id="doc-0001:tesseract:site:0002:lexical:0",
            accepted_score=0.86,
            n_candidates_considered=4,
        ),
    }
    return builders[spec.name]()


ALL_SPECS = pytest.mark.parametrize("spec", TABLES.values(), ids=list(TABLES))


@ALL_SPECS
def test_field_names_and_order_match(spec: TableSpec) -> None:
    """The arrow schema must mirror the model exactly, including declaration order."""
    assert list(spec.schema.names) == list(spec.model.model_fields)


@ALL_SPECS
def test_nullability_matches_optionality(spec: TableSpec) -> None:
    """A column is nullable iff the model field admits ``None``.

    Catches the drift that matters most: a field made optional in the model while storage
    still rejects null, or storage silently accepting null the model would refuse.
    """
    for name, info in spec.model.model_fields.items():
        arrow_field = spec.schema.field(spec.schema.get_field_index(name))
        assert arrow_field.nullable == allows_none(info.annotation), (
            f"{spec.name}.{name}: arrow nullable={arrow_field.nullable} "
            f"but model allows_none={allows_none(info.annotation)}"
        )


@ALL_SPECS
def test_primary_key_columns_exist(spec: TableSpec) -> None:
    for column in (*spec.primary_key, *spec.partition_by):
        assert column in spec.schema.names, f"{spec.name}: missing key column {column!r}"


@ALL_SPECS
def test_roundtrip_through_parquet(spec: TableSpec) -> None:
    """model -> arrow -> parquet bytes -> arrow -> model must be lossless."""
    original = _example(spec)
    table = records_to_table(spec, [original])

    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    restored_table = pq.read_table(buffer)

    restored = table_to_records(spec, restored_table)
    assert len(restored) == 1
    assert restored[0] == original


@ALL_SPECS
def test_roundtrip_with_nulls(spec: TableSpec) -> None:
    """Every nullable field set to ``None`` must survive the round trip as ``None``."""
    original = _example(spec)
    blanked = {
        name: None for name, info in spec.model.model_fields.items() if allows_none(info.annotation)
    }
    if not blanked:
        pytest.skip(f"{spec.name} has no nullable fields")
    record = original.model_copy(update=blanked)  # type: ignore[attr-defined]

    restored = table_to_records(spec, records_to_table(spec, [record]))
    assert restored[0] == record


@ALL_SPECS
def test_empty_batch_keeps_declared_types(spec: TableSpec) -> None:
    """An empty stage output must still produce a correctly typed, readable table."""
    table = records_to_table(spec, [])
    assert table.num_rows == 0
    assert table.schema == spec.schema
    assert table_to_records(spec, table) == []


@ALL_SPECS
def test_read_rejects_foreign_schema(spec: TableSpec) -> None:
    """Reading a table whose schema drifted must fail loudly, not coerce."""
    foreign = pa.table({"unexpected": pa.array([1, 2, 3])})
    with pytest.raises(ValueError, match="schema mismatch"):
        table_to_records(spec, foreign)


def test_records_are_immutable() -> None:
    """Artifacts are immutable; the in-memory records that build them are too."""
    doc = _example(TABLES["documents"])
    with pytest.raises(ValidationError):
        doc.document_id = "mutated"  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    """A typo'd column must fail at validation rather than be silently dropped."""
    with pytest.raises(ValidationError):
        Point.model_validate({"x": 1.0, "y": 2.0, "z": 3.0})


def test_bbox_rejects_inverted_corners() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=10.0, y0=0.0, x1=5.0, y1=10.0)


def test_bbox_allows_degenerate_box() -> None:
    """Thin glyphs legitimately produce zero-area boxes; refusing them would falsify
    the raw evidence."""
    box = BBox(x0=5.0, y0=5.0, x1=5.0, y1=12.0)
    assert box.width == 0.0
    assert box.area == 0.0


def test_polygon_bbox_is_the_vertex_hull() -> None:
    poly = Polygon(points=(Point(x=3.0, y=9.0), Point(x=11.0, y=4.0), Point(x=7.0, y=15.0)))
    assert poly.bbox == BBox(x0=3.0, y0=4.0, x1=11.0, y1=15.0)


def test_whitespace_in_text_is_preserved() -> None:
    """OCR whitespace is a measurement; stripping it would silently alter the data."""
    span = _example(TABLES["spans"]).model_copy(update={"text": "  rng "})  # type: ignore[attr-defined]
    restored = table_to_records(TABLES["spans"], records_to_table(TABLES["spans"], [span]))
    assert restored[0].text == "  rng "


def test_an_artifact_written_before_a_defaulted_column_existed_still_reads(
    tmp_path: Path,
) -> None:
    """Artifact longevity is an invariant, so one new optional field may not orphan a table.

    The H1 pilot's tables are immutable by policy: regenerating them is not available, and
    reading them years later is a requirement. A schema addition with a default is
    backfilled on read; anything else — a dropped column, a changed type — still raises.
    """
    from ocr_risk.io.parquet import read_records
    from ocr_risk.schemas.arrow import get_table

    spec = get_table("candidates")
    full = pa.table(
        {
            "candidate_id": ["c1"],
            "site_id": ["s1"],
            "document_id": ["d1"],
            "dataset_id": ["funsd"],
            "engine_id": ["tesseract"],
            "candidate_text": ["address"],
            "generator_id": ["lexical"],
            "generator_version": ["1"],
            "generator_rank": pa.array([0], type=pa.int32()),
            "generator_score": pa.array([None], type=pa.float64()),
            "is_synthetic_hard_negative": [False],
            "hard_negative_family": pa.array([None], type=pa.string()),
            "metadata": pa.array([[]], type=spec.schema.field("metadata").type),
        },
        schema=spec.schema,
    )
    dropped = full.drop_columns(["generator_score"])
    path = tmp_path / "legacy.parquet"
    pq.write_table(dropped, path)
    records = read_records(path, spec)
    assert len(records) == 1
    assert records[0].generator_score is None

    # A column the model cannot default is still a hard error.
    unfillable = full.drop_columns(["candidate_id"])
    broken = tmp_path / "broken.parquet"
    pq.write_table(unfillable, broken)
    with pytest.raises(ValueError, match="does not match the current declaration"):
        read_records(broken, spec)

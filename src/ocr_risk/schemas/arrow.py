"""Storage schemas: the on-disk form of every canonical table.

Arrow schemas are written out explicitly rather than derived from the pydantic models, so
that the physical layout of a research artifact is something a reviewer can read in one
place. The cost is drift, which ``tests/unit/test_schemas_roundtrip.py`` removes by
asserting field-for-field parity (names, order, and nullability) against the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

import pyarrow as pa

from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.base import RecordModel
from ocr_risk.schemas.candidates import Candidate, CandidateLabel
from ocr_risk.schemas.decisions import Decision
from ocr_risk.schemas.documents import GtToken, SourceDocument
from ocr_risk.schemas.evidence import EvidenceBundle
from ocr_risk.schemas.predictions import Prediction
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = [
    "BBOX",
    "POINT",
    "POLYGON",
    "TABLES",
    "TableSpec",
    "allows_none",
    "get_table",
    "records_to_table",
    "table_to_records",
]

M = TypeVar("M", bound=RecordModel)

# --- shared physical types ------------------------------------------------------------
POINT = pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
BBOX = pa.struct(
    [
        pa.field("x0", pa.float64()),
        pa.field("y0", pa.float64()),
        pa.field("x1", pa.float64()),
        pa.field("y1", pa.float64()),
    ]
)
POLYGON = pa.struct([pa.field("points", pa.list_(POINT))])
STR_MAP = pa.map_(pa.string(), pa.string())
STR_LIST = pa.list_(pa.string())

CONF_FEATURES = pa.struct(
    [
        pa.field("native_min", pa.float64(), nullable=True),
        pa.field("native_mean", pa.float64(), nullable=True),
        pa.field("conf_normalized", pa.float64(), nullable=True),
        pa.field("conf_rank_in_line", pa.float64(), nullable=True),
        pa.field("conf_zscore_within_engine", pa.float64(), nullable=True),
        pa.field("n_spans", pa.int32()),
        pa.field("has_native_confidence", pa.bool_()),
    ]
)
GEOM_FEATURES = pa.struct(
    [
        pa.field("width", pa.float64(), nullable=True),
        pa.field("height", pa.float64(), nullable=True),
        pa.field("aspect_ratio", pa.float64(), nullable=True),
        pa.field("relative_x", pa.float64(), nullable=True),
        pa.field("relative_y", pa.float64(), nullable=True),
        pa.field("relative_area", pa.float64(), nullable=True),
        pa.field("n_spans", pa.int32()),
        pa.field("has_geometry", pa.bool_()),
    ]
)


def _f(name: str, type_: pa.DataType, *, null: bool = False) -> pa.Field:
    return pa.field(name, type_, nullable=null)


DOCUMENTS_SCHEMA = pa.schema(
    [
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("image_path", pa.string()),
        _f("image_sha256", pa.string()),
        _f("page_index", pa.int32()),
        _f("width", pa.int32()),
        _f("height", pa.int32()),
        _f("gt_text", pa.string()),
        _f("gt_policy", pa.string()),
        _f("has_gt_geometry", pa.bool_()),
        _f("n_gt_tokens", pa.int32()),
        _f("license_id", pa.string()),
        _f("work_id", pa.string(), null=True),
        _f("source_url", pa.string(), null=True),
        _f("gt_text_shingle_hash", pa.string(), null=True),
    ]
)

GT_TOKENS_SCHEMA = pa.schema(
    [
        _f("gt_token_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("index", pa.int32()),
        _f("text", pa.string()),
        _f("char_start", pa.int32()),
        _f("char_end", pa.int32()),
        _f("line_id", pa.string(), null=True),
        _f("bbox", BBOX, null=True),
        _f("polygon", POLYGON, null=True),
    ]
)

SPANS_SCHEMA = pa.schema(
    [
        _f("span_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("engine_fingerprint", pa.string()),
        _f("text", pa.string()),
        _f("reading_order", pa.int32()),
        _f("line_id", pa.string(), null=True),
        _f("block_id", pa.string(), null=True),
        _f("bbox", BBOX, null=True),
        _f("polygon", POLYGON, null=True),
        # float64, not float32: native confidences are copied verbatim and must not be
        # perturbed by a storage-precision decision.
        _f("native_conf_recognition", pa.float64(), null=True),
        _f("native_conf_detection", pa.float64(), null=True),
        _f("conf_scale", pa.string(), null=True),
        _f("char_start", pa.int32()),
        _f("char_end", pa.int32()),
        _f("raw_ref", pa.string()),
        _f("raw_index", pa.int32()),
        _f("granularity_split", pa.bool_()),
    ]
)

ALIGNMENTS_SCHEMA = pa.schema(
    [
        _f("alignment_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("ocr_span_ids", STR_LIST),
        _f("gt_token_ids", STR_LIST),
        _f("relation", pa.string()),
        _f("status", pa.string()),
        _f("ocr_text", pa.string()),
        _f("gt_text", pa.string()),
        _f("align_confidence", pa.float64()),
        _f("char_agreement", pa.float64()),
        _f("geom_score", pa.float64(), null=True),
        _f("uniqueness_margin", pa.float64()),
        _f("edit_distance", pa.int32()),
        _f("region_id", pa.string(), null=True),
        _f("bbox", BBOX, null=True),
        _f("reason_code", pa.string(), null=True),
        _f("diagnostics", STR_MAP),
    ]
)

SITES_SCHEMA = pa.schema(
    [
        _f("site_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("alignment_ids", STR_LIST),
        _f("ocr_span_ids", STR_LIST),
        _f("gt_token_ids", STR_LIST),
        _f("ocr_text", pa.string()),
        _f("gt_text", pa.string()),
        _f("d_before", pa.int32()),
        _f("site_kind", pa.string()),
        _f("evaluable", pa.bool_()),
        _f("char_start", pa.int32()),
        _f("char_end", pa.int32()),
        _f("reading_order_start", pa.int32()),
        _f("bbox", BBOX, null=True),
        _f("min_align_confidence", pa.float64()),
    ]
)

CANDIDATES_SCHEMA = pa.schema(
    [
        _f("candidate_id", pa.string()),
        _f("site_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("candidate_text", pa.string()),
        _f("generator_id", pa.string()),
        _f("generator_version", pa.string()),
        _f("generator_rank", pa.int32()),
        _f("generator_score", pa.float64(), null=True),
        _f("is_synthetic_hard_negative", pa.bool_()),
        _f("hard_negative_family", pa.string(), null=True),
        _f("metadata", STR_MAP),
    ]
)

LABELS_SCHEMA = pa.schema(
    [
        _f("candidate_id", pa.string()),
        _f("site_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("d_before", pa.int32()),
        _f("d_after", pa.int32()),
        _f("delta", pa.int32()),
        _f("outcome_if_accepted", pa.string()),
        _f("outcome_if_rejected", pa.string()),
        _f("gt_text", pa.string()),
    ]
)

EVIDENCE_SCHEMA = pa.schema(
    [
        _f("candidate_id", pa.string()),
        _f("site_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("original_ocr", pa.string()),
        _f("candidate_text", pa.string()),
        _f("context_before", pa.string()),
        _f("context_after", pa.string()),
        _f("crop_recipe_sha256", pa.string(), null=True),
        _f("conf_features", CONF_FEATURES),
        _f("geom_features", GEOM_FEATURES),
        _f("available_fields", STR_LIST),
        _f("masked_fields", STR_LIST),
    ]
)

PREDICTIONS_SCHEMA = pa.schema(
    [
        _f("candidate_id", pa.string()),
        _f("site_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("verifier_id", pa.string()),
        _f("evidence_config", pa.string()),
        _f("fold_id", pa.string()),
        _f("raw_score", pa.float64()),
        _f("calibrated_score", pa.float64(), null=True),
        _f("calibrator_id", pa.string(), null=True),
    ]
)

DECISIONS_SCHEMA = pa.schema(
    [
        _f("site_id", pa.string()),
        _f("document_id", pa.string()),
        _f("dataset_id", pa.string()),
        _f("engine_id", pa.string()),
        _f("method_id", pa.string()),
        _f("fold_id", pa.string()),
        _f("epsilon", pa.float64()),
        _f("tau", pa.float64()),
        _f("action", pa.string()),
        _f("accepted_candidate_id", pa.string(), null=True),
        _f("accepted_score", pa.float64(), null=True),
        _f("n_candidates_considered", pa.int32()),
    ]
)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Binding between a validation model and its physical storage form."""

    name: str
    model: type[RecordModel]
    schema: pa.Schema
    version: int
    primary_key: tuple[str, ...]
    partition_by: tuple[str, ...] = ()


TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in (
        TableSpec(
            "documents", SourceDocument, DOCUMENTS_SCHEMA, 1, ("document_id",), ("dataset_id",)
        ),
        TableSpec("gt_tokens", GtToken, GT_TOKENS_SCHEMA, 1, ("gt_token_id",), ("dataset_id",)),
        TableSpec(
            "spans", CanonicalSpan, SPANS_SCHEMA, 1, ("span_id",), ("dataset_id", "engine_id")
        ),
        TableSpec(
            "alignments",
            AlignmentRecord,
            ALIGNMENTS_SCHEMA,
            1,
            ("alignment_id",),
            ("dataset_id", "engine_id"),
        ),
        TableSpec(
            "sites", CorrectionSite, SITES_SCHEMA, 1, ("site_id",), ("dataset_id", "engine_id")
        ),
        TableSpec(
            "candidates",
            Candidate,
            CANDIDATES_SCHEMA,
            1,
            ("candidate_id",),
            ("dataset_id", "engine_id"),
        ),
        TableSpec(
            "labels",
            CandidateLabel,
            LABELS_SCHEMA,
            1,
            ("candidate_id",),
            ("dataset_id", "engine_id"),
        ),
        TableSpec(
            "evidence",
            EvidenceBundle,
            EVIDENCE_SCHEMA,
            1,
            ("candidate_id",),
            ("dataset_id", "engine_id"),
        ),
        TableSpec(
            "predictions",
            Prediction,
            PREDICTIONS_SCHEMA,
            1,
            ("candidate_id", "verifier_id", "evidence_config", "fold_id"),
            ("dataset_id", "engine_id"),
        ),
        TableSpec(
            "decisions",
            Decision,
            DECISIONS_SCHEMA,
            1,
            ("site_id", "method_id", "fold_id", "epsilon"),
            ("dataset_id", "engine_id"),
        ),
    )
}


def get_table(name: str) -> TableSpec:
    """Look up a table spec, failing loudly on an unknown name."""
    try:
        return TABLES[name]
    except KeyError:
        known = ", ".join(sorted(TABLES))
        msg = f"unknown table {name!r}; known tables: {known}"
        raise KeyError(msg) from None


def allows_none(annotation: Any) -> bool:
    """Whether a resolved pydantic annotation admits ``None``."""
    if get_origin(annotation) in (Union, UnionType):
        return any(arg is type(None) for arg in get_args(annotation))
    return annotation is type(None)


def _to_storage(value: Any) -> Any:
    """Normalize a ``model_dump`` value into something pyarrow accepts."""
    if isinstance(value, tuple):
        return [_to_storage(v) for v in value]
    if isinstance(value, list):
        return [_to_storage(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_storage(v) for k, v in value.items()}
    return value


def records_to_table(spec: TableSpec, records: list[Any]) -> pa.Table:
    """Materialize validated records into an Arrow table with the declared schema.

    Passing the schema explicitly (rather than letting Arrow infer it) means an empty
    batch still produces correctly typed columns, and a type mismatch fails here instead
    of surfacing as a silent cast in a later join.
    """
    rows = [
        {k: _to_storage(v) for k, v in rec.model_dump(mode="python").items()} for rec in records
    ]
    return pa.Table.from_pylist(rows, schema=spec.schema)


def _from_storage(value: Any) -> Any:
    """Invert :func:`_to_storage`; Arrow returns map columns as key/value tuple lists."""
    if isinstance(value, list):
        if value and all(isinstance(v, tuple) and len(v) == 2 for v in value):
            return {k: _from_storage(v) for k, v in value}
        return [_from_storage(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_storage(v) for k, v in value.items()}
    return value


def table_to_records(spec: TableSpec, table: pa.Table) -> list[Any]:
    """Validate an Arrow table back into records, re-running every model constraint."""
    if table.schema != spec.schema:
        msg = (
            f"table {spec.name!r} schema mismatch on read.\n"
            f"expected: {spec.schema}\nactual:   {table.schema}"
        )
        raise ValueError(msg)
    out: list[Any] = []
    for row in table.to_pylist():
        payload = {k: _from_storage(v) for k, v in row.items()}
        # Map columns arrive as [] when empty; the models want {}.
        for key, field_info in spec.model.model_fields.items():
            if (
                field_info.annotation is not None
                and payload.get(key) == []
                and _is_mapping(spec, key)
            ):
                payload[key] = {}
        out.append(spec.model.model_validate(payload))
    return out


def _is_mapping(spec: TableSpec, field_name: str) -> bool:
    idx = spec.schema.get_field_index(field_name)
    return idx >= 0 and bool(pa.types.is_map(spec.schema.field(idx).type))

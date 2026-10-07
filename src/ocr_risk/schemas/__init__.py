"""Canonical data model: pydantic validation models paired with pyarrow storage schemas.

Layer 0. No logic beyond validation, no imports from sibling packages.

``SCHEMA_VERSION`` is bumped whenever any stored field changes name, type, or nullability,
or whenever an enum member is added or removed. It is written into every run record, so an
artifact produced under an older model stays identifiable rather than silently
misinterpreted.
"""

from __future__ import annotations

from ocr_risk.schemas.alignment import AlignmentRecord, RegionAnchor
from ocr_risk.schemas.arrow import TABLES, TableSpec, get_table, records_to_table, table_to_records
from ocr_risk.schemas.base import (
    BBox,
    ConfidenceScale,
    Point,
    Polygon,
    Probability,
    RecordModel,
    UnitInterval,
)
from ocr_risk.schemas.candidates import Candidate, CandidateLabel
from ocr_risk.schemas.decisions import Decision
from ocr_risk.schemas.documents import DocumentBundle, GtToken, SourceDocument
from ocr_risk.schemas.enums import (
    AlignmentRelation,
    AlignmentStatus,
    CoverageUnit,
    DecisionAction,
    EvidenceField,
    HarmPolicy,
    OutcomeIfAccepted,
    OutcomeIfRejected,
    RedistributionPolicy,
    SiteKind,
    SplitRole,
    StageName,
    harmful_outcomes,
)
from ocr_risk.schemas.evidence import (
    ConfidenceFeatures,
    CropRecipe,
    EvidenceBundle,
    GeometryFeatures,
    ObservationView,
)
from ocr_risk.schemas.predictions import Prediction
from ocr_risk.schemas.run_record import (
    ArtifactRef,
    EnvironmentState,
    GitState,
    RunRecord,
    SplitDescriptor,
)
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan, EngineFingerprint, RawEngineResponse

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "TABLES",
    "AlignmentRecord",
    "AlignmentRelation",
    "AlignmentStatus",
    "ArtifactRef",
    "BBox",
    "Candidate",
    "CandidateLabel",
    "CanonicalSpan",
    "ConfidenceFeatures",
    "ConfidenceScale",
    "CorrectionSite",
    "CoverageUnit",
    "CropRecipe",
    "Decision",
    "DecisionAction",
    "DocumentBundle",
    "EngineFingerprint",
    "EnvironmentState",
    "EvidenceBundle",
    "EvidenceField",
    "GeometryFeatures",
    "GitState",
    "GtToken",
    "HarmPolicy",
    "ObservationView",
    "OutcomeIfAccepted",
    "OutcomeIfRejected",
    "Point",
    "Polygon",
    "Prediction",
    "Probability",
    "RawEngineResponse",
    "RecordModel",
    "RedistributionPolicy",
    "RegionAnchor",
    "RunRecord",
    "SiteKind",
    "SourceDocument",
    "SplitDescriptor",
    "SplitRole",
    "StageName",
    "TableSpec",
    "UnitInterval",
    "get_table",
    "harmful_outcomes",
    "records_to_table",
    "table_to_records",
]

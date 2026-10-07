"""Ground truth must be structurally unreachable from the evidence layer.

Leakage vector L9. A verifier that saw the answer would score perfectly and mean nothing,
and the failure would be invisible in every downstream number — the metrics would simply
look excellent.

Three independent checks, because each catches a different way it could creep back:
reflection over the types, signature inspection of the builder, and a source scan of the
whole package.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from ocr_risk.evidence import EvidenceBuilder
from ocr_risk.schemas.evidence import (
    ConfidenceFeatures,
    CropRecipe,
    EvidenceBundle,
    GeometryFeatures,
    ObservationView,
    field_names_leaking_ground_truth,
)

EVIDENCE_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "ocr_risk" / "evidence"
EVIDENCE_TYPES = [ObservationView, EvidenceBundle, ConfidenceFeatures, GeometryFeatures, CropRecipe]


@pytest.mark.parametrize("model", EVIDENCE_TYPES, ids=lambda m: m.__name__)
def test_no_evidence_type_declares_a_ground_truth_field(model: type) -> None:
    """Reflection over the declared fields: the answer cannot be a column."""
    offenders = field_names_leaking_ground_truth(model.model_fields)  # type: ignore[attr-defined]
    assert not offenders, (
        f"{model.__name__} declares ground-truth-shaped field(s) {offenders}. "
        "The verifier must see only what a deployed system would see."
    )


def test_observation_view_carries_no_label_columns() -> None:
    """The builder's input type is the narrow gate; spell out what it must not have."""
    forbidden = {"gt_text", "d_before", "d_after", "delta", "outcome_if_accepted", "harmful"}
    assert not (set(ObservationView.model_fields) & forbidden)


def test_builder_accepts_only_the_ground_truth_free_view() -> None:
    """Signature inspection: widening ``build`` to take a site or a label record would be
    a refactor that quietly reintroduces the leak."""
    signature = inspect.signature(EvidenceBuilder.build)
    view_param = signature.parameters["view"]
    assert view_param.annotation in (ObservationView, "ObservationView")

    banned = {"CorrectionSite", "CandidateLabel", "gt_text", "label"}
    for name, parameter in signature.parameters.items():
        annotation = str(parameter.annotation)
        assert not (banned & {annotation, name}), (
            f"EvidenceBuilder.build parameter {name!r}: {annotation} can carry ground truth"
        )


def test_evidence_package_never_imports_label_types() -> None:
    """A source scan over the whole package, in case a helper reaches around the types."""
    banned_imports = {"CandidateLabel", "CorrectionSite"}
    violations: list[str] = []
    for path in sorted(EVIDENCE_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in banned_imports:
                        violations.append(f"  {path.name} imports {alias.name}")
    assert not violations, "evidence/ imports label-bearing types:\n" + "\n".join(violations)


def test_evidence_source_mentions_no_gt_attribute_access() -> None:
    """Catches ``something.gt_text`` reached through an untyped object."""
    violations: list[str] = []
    for path in sorted(EVIDENCE_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "gt_text",
                "gt_token_ids",
                "d_before",
                "d_after",
                "outcome_if_accepted",
            }:
                violations.append(f"  {path.name}:{node.lineno} reads .{node.attr}")
    assert not violations, "evidence/ reads ground-truth attributes:\n" + "\n".join(violations)

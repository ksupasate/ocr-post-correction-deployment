"""Fairness and crop-integrity guards for the SGV1 development pilot.

The pilot's whole claim is that V2 differs from V1 in the image channel and in nothing
else, and that the pixels it reads are the pixels the candidate's own OCR geometry names.
Both are checkable, so both are checked here rather than asserted in a docstring.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from ocr_risk.evidence.fields import get_evidence_config
from ocr_risk.schemas.enums import EvidenceField
from ocr_risk.schemas.evidence import ObservationView

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sgv1_verifier_pilot.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sgv1_verifier_pilot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_row(**overrides: object) -> object:
    base = {
        "candidate_id": "c-0",
        "site_id": "s-0",
        "generator_rank": 1,
        "generator_score": 0.75,
        "suspicion_score": 0.4,
        "char_start": 10,
        "char_end": 15,
        "generator_source": "g3_edit_aware",
        "operation": "substitution",
        "anchor_kind": "token",
        "site_type": "substitution",
    }
    base.update(overrides)
    return next(pd.DataFrame([base]).itertuples())


# --- the primary contrast differs in the image channel and nothing else ---------------


def test_sgv1_ladder_nests_and_isolates_the_image_channel() -> None:
    module = _script()
    module.validate_ladder()
    v0 = get_evidence_config("sgv1_v0").fields
    v1 = get_evidence_config("sgv1_v1").fields
    v2 = get_evidence_config("sgv1_v2").fields
    assert v0 < v1 < v2
    assert v2 - v1 == {EvidenceField.IMAGE_CROP}


def test_provenance_block_is_given_to_v1_and_every_v2_arm_but_not_v0() -> None:
    """A V1 denied the provenance channel would be a baseline chosen to flatter V2."""
    module = _script()
    assert {"V1", "V2_correct", "V2_shuffled", "V2_masked"} == module.PROVENANCE_ARMS
    assert "V0" not in module.PROVENANCE_ARMS
    assert set(module.ARMS) == {"V0", *module.PROVENANCE_ARMS}


def test_every_v2_arm_shares_v1s_evidence_configuration() -> None:
    module = _script()
    v2_arms = {a: c for a, (c, _) in module.ARMS.items() if a.startswith("V2")}
    assert set(v2_arms.values()) == {"sgv1_v2"}
    assert module.ARMS["V1"][0] == "sgv1_v1"
    assert {mode for _, mode in module.ARMS.values()} == {"none", "correct", "shuffled", "masked"}


def test_provenance_features_are_all_frozen_candidate_columns() -> None:
    """Red team: nothing in the provenance block may come from a label."""
    module = _script()
    source = inspect.getsource(module.provenance_block)
    for forbidden in ("outcome", "is_harmful", "d_before", "d_after", "region_gt", "gt_"):
        assert forbidden not in source
    block = module.provenance_block(_candidate_row(), site_candidate_count=3)
    assert len(block.names) == len(block.values)
    assert block.names == module.PROVENANCE_NAMES
    assert np.isfinite(block.values).all()


def test_provenance_block_marks_a_missing_generator_score_rather_than_imputing_zero() -> None:
    module = _script()
    present = module.provenance_block(_candidate_row(generator_score=0.9), 2)
    missing = module.provenance_block(_candidate_row(generator_score=float("nan")), 2)
    index = module.PROVENANCE_NAMES.index("prov_generator_score_missing")
    assert present.values[index] == 0.0
    assert missing.values[index] == 1.0
    assert np.isfinite(missing.values).all()


def test_sgv1_verifier_appends_provenance_only_when_it_is_supplied() -> None:
    module = _script()
    plain = module.SGV1Verifier(verifier_id="v", evidence_config="sgv1_v1")
    assert plain._provenance == {}
    with_provenance = module.SGV1Verifier(
        verifier_id="v", evidence_config="sgv1_v1", provenance={"c-0": np.zeros(3)}
    )
    assert with_provenance._provenance


# --- crop integrity -------------------------------------------------------------------


def test_observation_view_has_no_ground_truth_field() -> None:
    """The evidence path is typed so a label cannot reach a feature."""
    for forbidden in ("gt_text", "region_gt", "outcome", "is_harmful", "d_before", "d_after"):
        assert forbidden not in ObservationView.model_fields


def test_page_image_carries_no_annotation() -> None:
    """Red team: the only thing the crop path learns about a document is its pixels."""
    module = _script()
    fields = set(module.PageImage.__dataclass_fields__)
    assert fields == {"document_id", "path", "image_sha256", "width", "height"}
    source = inspect.getsource(module._page_images)
    assert "gt_tokens" not in source
    assert "gt_text" not in source


def test_crop_geometry_comes_from_ocr_spans_not_ground_truth() -> None:
    module = _script()
    source = inspect.getsource(module.run_evidence)
    assert "_union_bbox(anchor_spans)" in source
    assert "gt_tokens" not in source
    lineage_source = inspect.getsource(module.run_evidence)
    assert '"geometry_source": "ocr_span_bbox_union"' in lineage_source


def test_union_bbox_covers_every_anchor_span() -> None:
    module = _script()
    from ocr_risk.schemas.base import BBox

    class _Span:
        def __init__(self, bbox: BBox | None) -> None:
            self.bbox = bbox

    boxes = [
        _Span(BBox(x0=10.0, y0=20.0, x1=30.0, y1=40.0)),
        _Span(BBox(x0=5.0, y0=25.0, x1=25.0, y1=50.0)),
    ]
    union = module._union_bbox(boxes)
    assert (union.x0, union.y0, union.x1, union.y1) == (5.0, 20.0, 30.0, 50.0)
    assert module._union_bbox([_Span(None)]) is None


def test_shuffled_donor_is_always_a_different_document_and_deterministic() -> None:
    """The control only controls if the donor crop cannot be the real one."""
    module = _script()
    documents = [f"cord-train-{i:04d}" for i in range(25)]
    for index in range(200):
        candidate_id = f"candidate-{index}"
        home = documents[index % len(documents)]
        donor = module._shuffled_donor(candidate_id, home, documents)
        assert donor != home
        assert donor in documents
        assert donor == module._shuffled_donor(candidate_id, home, documents)


def test_masked_arm_receives_no_crops_at_all() -> None:
    module = _script()
    lineage = pd.DataFrame(
        [
            {
                "candidate_id": "c-0",
                "document_id": "doc-a",
                "crop_recipe_sha256": "a" * 64,
                "shuffled_crop_recipe_sha256": "b" * 64,
            }
        ]
    )
    assert module._crop_map("masked", lineage) == {}
    assert module._crop_map("none", lineage) == {}
    correct = module._crop_map("correct", lineage)
    shuffled = module._crop_map("shuffled", lineage)
    assert correct["c-0"] != shuffled["c-0"]
    assert correct["c-0"].name.startswith("a" * 8)
    assert shuffled["c-0"].name.startswith("b" * 8)


# --- endpoints ------------------------------------------------------------------------


def test_pair_ranking_is_hand_computed_with_ties_as_half() -> None:
    module = _script()
    scores = {"p1": 0.9, "m1": 0.1, "p2": 0.4, "m2": 0.4, "p3": 0.2, "m3": 0.8}
    pairs = pd.DataFrame(
        [
            {"plus_candidate_id": "p1", "minus_candidate_id": "m1"},
            {"plus_candidate_id": "p2", "minus_candidate_id": "m2"},
            {"plus_candidate_id": "p3", "minus_candidate_id": "m3"},
        ]
    )
    # one win, one tie (0.5), one loss -> 1.5 / 3
    assert module._pair_ranking(scores, pairs) == pytest.approx(0.5)
    assert np.isnan(module._pair_ranking(scores, pairs.iloc[:0]))


def test_c3_is_inconclusive_rather_than_passing_when_an_artifact_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent input must never read as a pass."""
    module = _script()
    for name in ("EVIDENCE_RECORD", "PILOT_RECORD", "SCORES_TABLE", "CROP_LINEAGE"):
        monkeypatch.setattr(module, name, tmp_path / name.lower())
    monkeypatch.setattr(module, "C3_CERTIFICATE", tmp_path / "c3.json")
    monkeypatch.setattr(module, "_git_head", lambda: "0" * 40)
    assert module.run_c3() == 0
    import json

    certificate = json.loads((tmp_path / "c3.json").read_text())
    assert certificate["verdict"] == "C3_INCONCLUSIVE"


def test_pilot_declares_the_document_held_out_axis_not_an_engine_one() -> None:
    """Every engine is in every role here, and the record has to say so."""
    module = _script()
    source = inspect.getsource(module.run_pilot)
    assert '"held_out_engine": None' in source
    assert "cross-engine shift" in source

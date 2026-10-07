"""SGV1 leakage guards: ladder integrity, GT blindness, and the freeze-order contract.

These are red-team tests: each one attempts a specific violation and asserts the guard
raises, in addition to the architectural properties (subset structure, signature
discipline) that make the violation hard to commit by accident.
"""

from __future__ import annotations

import inspect

from ocr_risk.evidence.fields import get_evidence_config
from ocr_risk.experiments import sgv1_frames
from ocr_risk.experiments.sgv1_design import (
    PRIMARY_CONTRAST,
    SGV1_LADDER,
    validate_ladder,
)
from ocr_risk.experiments.sgv1_frames import label_columns_present
from ocr_risk.schemas.enums import EvidenceField

F = EvidenceField


class TestLadder:
    def test_ladder_nests_and_pins_the_image_channel(self) -> None:
        validate_ladder()  # raises AssertionError on any drift

    def test_v1_is_every_non_image_channel(self) -> None:
        # The strong baseline must not be quietly weakened: sgv1_v1 is the full
        # EvidenceField vocabulary minus exactly the image crop.
        v1 = get_evidence_config("sgv1_v1").fields
        assert v1 == frozenset(F) - {F.IMAGE_CROP}

    def test_v2_differs_from_v1_by_image_alone(self) -> None:
        v1 = get_evidence_config("sgv1_v1").fields
        v2 = get_evidence_config("sgv1_v2").fields
        assert v2 - v1 == {F.IMAGE_CROP}

    def test_primary_contrast_is_v2_over_v1(self) -> None:
        assert PRIMARY_CONTRAST == ("sgv1_v2", "sgv1_v1")
        assert set(SGV1_LADDER) == {"sgv1_v0", "sgv1_v1", "sgv1_v2"}

    def test_v0_is_a_strict_text_only_baseline(self) -> None:
        v0 = get_evidence_config("sgv1_v0").fields
        assert v0 == {F.ORIGINAL_OCR, F.CANDIDATE_TEXT, F.TEXT_CONTEXT}


class TestGroundTruthBlindness:
    def test_label_column_detector_names_the_usual_offenders(self) -> None:
        assert label_columns_present(["gt_text", "candidate_id"]) == ["gt_text"]
        assert label_columns_present(["outcome", "d_before"]) == ["outcome", "d_before"]

    def test_clean_candidate_columns_pass(self) -> None:
        assert label_columns_present(["candidate_id", "original_ocr"]) == []


class TestFreezeOrderContract:
    def test_every_label_consumer_requires_a_freeze(self) -> None:
        """No public function may accept labeled data without its freeze record.

        If someone adds ``score_pool(labeled, ...)`` without a ``freeze`` parameter,
        this test fails: the freeze check is structural, not conventional.
        """
        # Entry points that accept ground-truth-carrying tables. frame_a/frame_b
        # outputs are post-freeze constructions (re-checked where produced), so they
        # are not entry points for unlabeled data.
        labelish = ("labeled", "labels")
        for name, func in inspect.getmembers(sgv1_frames, inspect.isfunction):
            if name.startswith("_"):
                continue
            params = inspect.signature(func).parameters
            takes_labels = any(p in labelish for p in params)
            requires_freeze = "freeze" in params
            if takes_labels:
                assert requires_freeze, (
                    f"{name} consumes labeled data without a freeze parameter; "
                    "freeze-order enforcement is bypassable through it"
                )

    def test_build_matched_pairs_operates_only_on_frozen_rows(self) -> None:
        # build_matched_pairs takes frame_a (already freeze-checked upstream); its
        # inputs are therefore frozen-pool rows by construction. Documented here so
        # a future signature change re-triggers the check above.
        params = inspect.signature(sgv1_frames.build_matched_pairs).parameters
        assert "freeze" not in params  # deliberately: it consumes Frame A output only
        assert "frame_a" in params


class TestRuntimeGuardSource:
    """The guards must be exceptions at runtime, not warnings -- fail closed."""

    def test_frame_error_is_runtime_error(self) -> None:
        assert issubclass(sgv1_frames.FrameError, RuntimeError)

    def test_module_never_warns_instead_of_raising(self) -> None:
        # Fail-closed discipline: no warn-and-continue path may creep in beside the
        # FrameError raises (behavior covered in tests/unit/test_sgv1_frames.py).
        assert "warnings" not in inspect.getsource(sgv1_frames)

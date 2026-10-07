"""Verifiers V0-V6 and their featurizers.

The load-bearing checks are the ablation ones: masking must actually change what a
verifier can see, or the whole V3-vs-V6 comparison measures nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr_risk.evidence import EvidenceMask
from ocr_risk.schemas.enums import EvidenceField
from ocr_risk.schemas.evidence import ConfidenceFeatures, EvidenceBundle, GeometryFeatures
from ocr_risk.verify import (
    NotFittedVerifierError,
    VerificationInput,
    available_verifiers,
    blocks_for,
    build_verifier,
)
from ocr_risk.verify.featurizers import confidence_block, geometry_block, image_block, text_block


def _bundle(
    original: str = "rng",
    candidate: str = "mg",
    *,
    conf: float | None = 0.8,
    geometry: bool = True,
    crop: str | None = "c" * 64,
) -> EvidenceBundle:
    return EvidenceBundle(
        candidate_id=f"c:{original}:{candidate}",
        site_id="s",
        document_id="d",
        dataset_id="ds",
        engine_id="e",
        original_ocr=original,
        candidate_text=candidate,
        context_before="Dose: ",
        context_after=" total",
        crop_recipe_sha256=crop,
        conf_features=ConfidenceFeatures(
            native_min=conf,
            conf_normalized=conf,
            conf_zscore_within_engine=0.1 if conf is not None else None,
            n_spans=1,
            has_native_confidence=conf is not None,
        ),
        geom_features=GeometryFeatures(
            width=0.05,
            height=0.02,
            aspect_ratio=2.5,
            relative_x=0.1,
            relative_y=0.2,
            relative_area=0.001,
            n_spans=1,
            has_geometry=geometry,
        ),
        available_fields=tuple(EvidenceField),
    )


def _inputs(n: int = 60, crop: Path | None = None) -> tuple[list[VerificationInput], list[bool]]:
    """A separable toy problem: harmful edits change a digit, safe ones do not."""
    items: list[VerificationInput] = []
    harmful: list[bool] = []
    for index in range(n):
        is_harmful = index % 2 == 0
        bundle = _bundle(
            original="0.015" if is_harmful else "rng",
            candidate="0.15" if is_harmful else "mg",
            conf=0.3 if is_harmful else 0.95,
        ).model_copy(update={"candidate_id": f"c{index}"})
        items.append(VerificationInput(bundle=bundle, crop=crop))
        harmful.append(is_harmful)
    return items, harmful


# --- registry -----------------------------------------------------------------------------
def test_verifiers_are_registered() -> None:
    assert set(available_verifiers()) == {"accept_all", "feature_logistic", "torch_multimodal"}


def test_unknown_verifier_lists_known_ones() -> None:
    with pytest.raises(KeyError, match="registered:"):
        build_verifier("nope", verifier_id="x", evidence_config="v6")


# --- V0 -----------------------------------------------------------------------------------
def test_accept_all_scores_everything_maximally() -> None:
    """Not a straw man: it is the operating point every curve is read against."""
    verifier = build_verifier("accept_all", verifier_id="v0", evidence_config="v0")
    items, harmful = _inputs(10)
    verifier.fit(items, harmful)
    batch = verifier.score(items)
    assert np.all(batch.scores == 1.0)
    assert batch.candidate_ids == tuple(i.candidate_id for i in items)


# --- feature verifier ------------------------------------------------------------------------
def test_verifier_must_be_fitted_before_scoring() -> None:
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    with pytest.raises(NotFittedVerifierError, match="before fit"):
        verifier.score(_inputs(4)[0])


def test_verifier_learns_a_separable_signal() -> None:
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    items, harmful = _inputs(80)
    verifier.fit(items, harmful)
    scores = verifier.score(items).scores

    harmful_scores = scores[np.asarray(harmful)]
    safe_scores = scores[~np.asarray(harmful)]
    # Higher score means safer to accept.
    assert safe_scores.mean() > harmful_scores.mean()


def test_scores_stay_aligned_to_candidate_ids() -> None:
    """A score that drifted from its row would be undetectable downstream."""
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    items, harmful = _inputs(40)
    verifier.fit(items, harmful)
    batch = verifier.score(items)
    assert batch.candidate_ids == tuple(i.candidate_id for i in items)
    assert batch.scores.size == len(items)


def test_single_class_training_falls_back_to_a_constant() -> None:
    """One class only means there is nothing to learn; a fitted model would encode an
    accident of the split."""
    items, _ = _inputs(20)
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    verifier.fit(items, [True] * len(items))
    scores = verifier.score(items).scores
    assert len(set(scores.tolist())) == 1


def test_empty_training_set_is_survivable() -> None:
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    verifier.fit([], [])
    assert verifier.score(_inputs(3)[0]).scores.size == 3


def test_scoring_an_empty_batch_returns_nothing() -> None:
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    items, harmful = _inputs(20)
    verifier.fit(items, harmful)
    assert verifier.score([]).scores.size == 0


def test_mismatched_label_count_is_rejected() -> None:
    verifier = build_verifier("feature_logistic", verifier_id="v3", evidence_config="v3")
    items, _ = _inputs(5)
    with pytest.raises(ValueError, match="inputs but"):
        verifier.fit(items, [True, False])


def test_gradient_boosting_model_also_fits() -> None:
    verifier = build_verifier(
        "feature_logistic", verifier_id="v6", evidence_config="v6", model="gradient_boosting"
    )
    items, harmful = _inputs(80)
    verifier.fit(items, harmful)
    assert verifier.score(items).scores.size == 80


# --- the ablation actually ablates ---------------------------------------------------------------
def test_masking_changes_the_feature_vector() -> None:
    """If V3 and V6 saw the same features, the central comparison would measure nothing."""
    bundle = _bundle()
    crop = None
    v3 = blocks_for(
        EvidenceMask.from_key("v3").apply(bundle), EvidenceMask.from_key("v3").spec.fields, crop
    )
    v6 = blocks_for(
        EvidenceMask.from_key("v6").apply(bundle), EvidenceMask.from_key("v6").spec.fields, crop
    )
    v3_names = [n for block in v3 for n in block.names]
    v6_names = [n for block in v6 for n in block.names]
    assert set(v3_names) < set(v6_names)
    assert any(n.startswith("img_") for n in v6_names)
    assert not any(n.startswith("img_") for n in v3_names)


def test_v1_sees_only_confidence_features() -> None:
    mask = EvidenceMask.from_key("v1")
    names = [n for b in blocks_for(mask.apply(_bundle()), mask.spec.fields, None) for n in b.names]
    assert names
    assert all(n.startswith("conf_") for n in names)


def test_v4_sees_only_image_features() -> None:
    mask = EvidenceMask.from_key("v4")
    names = [n for b in blocks_for(mask.apply(_bundle()), mask.spec.fields, None) for n in b.names]
    assert names
    assert all(n.startswith("img_") for n in names)


def test_v0_sees_no_features_at_all() -> None:
    mask = EvidenceMask.from_key("v0")
    assert blocks_for(mask.apply(_bundle()), mask.spec.fields, None) == []


def test_v0_verifier_still_scores_with_no_features() -> None:
    """The degenerate configuration must not crash the pipeline."""
    verifier = build_verifier("feature_logistic", verifier_id="v0f", evidence_config="v0")
    mask = EvidenceMask.from_key("v0")
    items = [VerificationInput(bundle=mask.apply(i.bundle)) for i in _inputs(20)[0]]
    verifier.fit(items, _inputs(20)[1])
    assert verifier.score(items).scores.size == 20


# --- feature blocks ----------------------------------------------------------------------------------
def test_text_block_flags_a_digit_change() -> None:
    """The signature of the magnitude errors this project cares most about."""
    block = text_block(_bundle("0.015", "0.15"))
    features = dict(zip(block.names, block.values.tolist(), strict=True))
    assert features["text_digit_changed"] == 1.0
    assert features["text_edit_distance"] == 1.0


def test_text_block_on_a_case_only_change() -> None:
    block = text_block(_bundle("Smith", "smith"))
    features = dict(zip(block.names, block.values.tolist(), strict=True))
    assert features["text_case_only_change"] == 1.0
    assert features["text_digit_changed"] == 0.0


def test_missing_confidence_is_flagged_not_imputed() -> None:
    """Imputing 0.0 would let the model read 'no confidence reported' as 'confidence
    zero', which are different statements about the world."""
    block = confidence_block(_bundle(conf=None))
    features = dict(zip(block.names, block.values.tolist(), strict=True))
    assert features["conf_missing"] == 1.0


def test_present_confidence_is_not_flagged_missing() -> None:
    block = confidence_block(_bundle(conf=0.8))
    features = dict(zip(block.names, block.values.tolist(), strict=True))
    assert features["conf_missing"] == 0.0
    assert features["conf_normalized"] == pytest.approx(0.8)


def test_missing_geometry_is_flagged() -> None:
    block = geometry_block(_bundle(geometry=False))
    assert dict(zip(block.names, block.values.tolist(), strict=True))["geom_missing"] == 1.0


def test_image_block_without_a_crop_is_flagged_missing() -> None:
    block = image_block(None)
    assert dict(zip(block.names, block.values.tolist(), strict=True))["img_missing"] == 1.0


def test_image_block_distinguishes_ink_density(tmp_path: Path) -> None:
    """The image channel must carry real signal, or V6-over-V3 could never be observed
    even in principle."""
    sparse = tmp_path / "sparse.png"
    dense = tmp_path / "dense.png"
    Image.new("L", (60, 20), color=255).save(sparse)
    Image.new("L", (60, 20), color=40).save(dense)

    sparse_features = dict(
        zip(image_block(sparse).names, image_block(sparse).values.tolist(), strict=True)
    )
    dense_features = dict(
        zip(image_block(dense).names, image_block(dense).values.tolist(), strict=True)
    )
    assert sparse_features["img_mean"] > dense_features["img_mean"]
    assert sparse_features["img_missing"] == 0.0


def test_image_features_are_finite_on_a_one_pixel_crop(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.png"
    Image.new("L", (1, 1), color=128).save(tiny)
    values = image_block(tiny).values
    assert np.all(np.isfinite(values))


# --- torch slot -------------------------------------------------------------------------------------------
def test_torch_verifier_refuses_rather_than_returning_constants() -> None:
    """A stub scoring 0.5 would produce a complete-looking risk-coverage curve for a model
    that does not exist."""
    verifier = build_verifier("torch_multimodal", verifier_id="t", evidence_config="v6")
    with pytest.raises(NotImplementedError, match="not implemented"):
        verifier.fit([], [])
    with pytest.raises(NotImplementedError, match="not implemented"):
        verifier.score([])


# --- contracts that would fail silently if they stopped holding ----------------------


def test_scored_batch_refuses_scores_that_do_not_line_up_with_ids() -> None:
    """A misaligned score array would attach every score to the wrong candidate.

    Nothing downstream could detect that: the ids are well-formed, the scores are
    well-formed, and every metric would compute cleanly on the permuted pairing. The
    constructor is the only place it can be caught.
    """
    from ocr_risk.verify.base import ScoredBatch

    with pytest.raises(ValueError, match="3 ids but 2 scores"):
        ScoredBatch(candidate_ids=("a", "b", "c"), scores=np.array([0.1, 0.2]))

    batch = ScoredBatch(candidate_ids=("a", "b"), scores=np.array([0.1, 0.2]))
    assert batch.candidate_ids == ("a", "b")


def test_registering_two_verifiers_under_one_key_is_refused() -> None:
    """Silently replacing a registered verifier would swap the model behind a config key."""
    from ocr_risk.verify.registry import _REGISTRY, register_verifier

    key = "test_only_duplicate_probe"
    try:

        @register_verifier(key)
        class _First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_verifier(key)
            class _Second:
                pass

    finally:
        # The registry is process-global; leaving the probe behind would make
        # `available_verifiers()` differ depending on whether this test ran.
        _REGISTRY.pop(key, None)


def test_text_block_is_empty_rather_than_zero_width_when_there_is_no_text() -> None:
    """An empty block contributes no columns; a zero-valued block would contribute width.

    The difference matters because feature-matrix width is what makes V1 and V2 comparable:
    a block that silently returned zeros for absent text would change the column count for
    some rows and not others.
    """
    from ocr_risk.verify.featurizers import FeatureBlock

    empty = FeatureBlock.empty()
    assert empty.names == ()
    assert empty.values.size == 0

    block = text_block(_bundle(original="", candidate=""))
    assert block.names == ()
    assert block.values.size == 0

    populated = text_block(_bundle(original="tota1", candidate="total"))
    assert len(populated.names) == len(populated.values) > 0


def test_feature_names_are_exposed_and_match_the_matrix_width() -> None:
    """The recorded feature_names are what a run record claims the model saw."""
    verifier = build_verifier("feature_logistic", verifier_id="v", evidence_config="v3")
    assert verifier.feature_names == ()
    inputs, harmful = _inputs()
    verifier.fit(inputs, harmful)
    assert len(verifier.feature_names) > 0
    assert len(verifier.feature_names) == verifier._matrix(inputs).shape[1]

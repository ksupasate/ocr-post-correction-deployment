"""Candidate generation and evidence construction."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ocr_risk.candidates import (
    HARD_NEGATIVE_FAMILIES,
    GenerationContext,
    available_generators,
    build_generator,
)
from ocr_risk.candidates.stubs import LlmGenerator
from ocr_risk.evidence import (
    EVIDENCE_CONFIGS,
    ConfidenceFeaturizer,
    CropPolicy,
    EvidenceBuilder,
    EvidenceMask,
    MaskedFieldAccessError,
    NotFittedError,
    build_recipe,
    context_window,
    crop_path,
    geometry_features,
    get_evidence_config,
    materialize,
)
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.enums import EvidenceField, SiteKind
from ocr_risk.schemas.evidence import ObservationView
from ocr_risk.schemas.sites import CorrectionSite


def _context(text: str) -> GenerationContext:
    return GenerationContext(
        site_id="s",
        document_id="d",
        dataset_id="ds",
        engine_id="e",
        original_ocr=text,
        context_before="Dose: ",
        context_after=" total",
        native_confidences=(0.8,),
        conf_scale="synthetic_0_1",
    )


# --- generator registry -------------------------------------------------------------------
def test_all_generators_are_registered() -> None:
    assert set(available_generators()) == {
        "byt5",
        "composite",
        "edit_aware",
        "error_gated",
        "hard_negative",
        "identity",
        "lexical",
        "llm",
        "multimodal_rewriter",
        "structural",
        "structural_v2",
    }


def test_unknown_generator_lists_known_ones() -> None:
    with pytest.raises(KeyError, match="registered:"):
        build_generator("nope", generator_id="x")


def test_generation_context_from_site_drops_ground_truth() -> None:
    """The projection is the gate: a generator never receives the site itself."""
    site = CorrectionSite(
        site_id="s",
        document_id="d",
        dataset_id="ds",
        engine_id="e",
        alignment_ids=("a",),
        ocr_span_ids=("sp",),
        gt_token_ids=("t",),
        ocr_text="rng",
        gt_text="mg",
        d_before=2,
        site_kind=SiteKind.SUBSTITUTION,
        evaluable=True,
        char_start=0,
        char_end=3,
        reading_order_start=0,
        min_align_confidence=0.9,
    )
    context = GenerationContext.from_site(site, "before ", " after")
    assert context.original_ocr == "rng"
    assert "mg" not in str(context), "ground truth reached the generator context"
    assert not hasattr(context, "gt_text")


# --- identity ----------------------------------------------------------------------------
def test_identity_proposes_the_original() -> None:
    proposals = build_generator("identity", generator_id="id").propose(_context("mg"), 3)
    assert [p.text for p in proposals] == ["mg"]


# --- lexical -----------------------------------------------------------------------------
def test_lexical_needs_fitting_first() -> None:
    """An unfitted lexicon proposes nothing rather than inventing a vocabulary."""
    generator = build_generator("lexical", generator_id="lex")
    assert generator.propose(_context("patlent"), 3) == []


def test_lexical_proposes_near_vocabulary_forms() -> None:
    generator = build_generator("lexical", generator_id="lex")
    generator.fit(["the patient received a dose"] * 3)
    proposals = generator.propose(_context("patlent"), 3)
    assert "patient" in [p.text for p in proposals]


def test_lexical_never_proposes_the_original() -> None:
    generator = build_generator("lexical", generator_id="lex")
    generator.fit(["patient patient patient"])
    assert all(p.text != "patient" for p in generator.propose(_context("patient"), 3))


def test_lexical_respects_the_edit_distance_budget() -> None:
    generator = build_generator("lexical", generator_id="lex", max_edit_distance=1)
    generator.fit(["patient patient"])
    assert generator.propose(_context("xxxxxxx"), 3) == []


def test_lexical_drops_rare_forms_from_the_lexicon() -> None:
    """A lexicon containing every OCR error it ever saw would happily 'correct' a word to
    another engine's mistake."""
    generator = build_generator("lexical", generator_id="lex", min_lexicon_frequency=3)
    generator.fit(["common common common rare"])
    assert generator.lexicon_size == 1


def test_lexical_fit_is_scoped_to_the_given_corpus() -> None:
    """The generator reads only what it is handed; it cannot reach for more (vector L5)."""
    generator = build_generator("lexical", generator_id="lex")
    generator.fit(["alpha alpha beta beta"])
    assert generator.lexicon_size == 2
    generator.fit(["gamma gamma"])
    assert generator.lexicon_size == 1


# --- hard negatives -------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "family", "expected"),
    [
        ("0.015", "numeric_magnitude", "0.15"),
        ("0.015 mg", "unit_substitution", "0.015 ng"),
        ("Ti-6Al-4V", "homoglyph", "Ti-6A1-4V"),
        ("Smith", "plausible_lexical", "Smyth"),
        ("Ti-6Al-4V", "truncation", "Ti-4V"),
    ],
)
def test_hard_negative_families_produce_their_signature_case(
    text: str, family: str, expected: str
) -> None:
    """These are the discriminations the project exists to test: each is linguistically
    well-formed and not supported by the pixels."""
    generator = build_generator("hard_negative", generator_id="hn", families=(family,))
    assert expected in [p.text for p in generator.propose(_context(text), 6)]


def test_hard_negatives_are_tagged() -> None:
    generator = build_generator("hard_negative", generator_id="hn")
    for proposal in generator.propose(_context("0.015 mg"), 6):
        assert proposal.is_synthetic_hard_negative
        assert proposal.hard_negative_family in HARD_NEGATIVE_FAMILIES


def test_hard_negative_never_proposes_the_original() -> None:
    generator = build_generator("hard_negative", generator_id="hn")
    assert all(p.text != "0.015 mg" for p in generator.propose(_context("0.015 mg"), 8))


def test_hard_negative_proposals_are_unique() -> None:
    generator = build_generator("hard_negative", generator_id="hn")
    texts = [p.text for p in generator.propose(_context("0.015 mg"), 8)]
    assert len(texts) == len(set(texts))


def test_unknown_hard_negative_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown hard-negative families"):
        build_generator("hard_negative", generator_id="hn", families=("made_up",))


def test_unit_substitution_requires_a_whole_token() -> None:
    """Substituting inside a word would produce nonsense rather than a plausible edit."""
    generator = build_generator("hard_negative", generator_id="hn", families=("unit_substitution",))
    assert generator.propose(_context("amgle"), 4) == []


def test_hard_negatives_are_deterministic() -> None:
    generator = build_generator("hard_negative", generator_id="hn")
    first = [p.text for p in generator.propose(_context("0.015 mg Smith"), 8)]
    second = [p.text for p in generator.propose(_context("0.015 mg Smith"), 8)]
    assert first == second


# --- stubs ----------------------------------------------------------------------------------
def test_unimplemented_generator_raises_rather_than_returning_nothing() -> None:
    """Silently returning no candidates would be indistinguishable from a model that
    found none — an absent component must not look like a null result."""
    generator = build_generator("multimodal_rewriter", generator_id="multimodal_rewriter")
    assert not generator.available()
    with pytest.raises(NotImplementedError, match="not implemented"):
        generator.propose(_context("x"), 1)


def test_the_byte_model_reports_why_it_is_unavailable_rather_than_only_that_it_is() -> None:
    """``byt5`` is real now, against a pinned public checkpoint. When it cannot run, the
    reason has to be actionable — "unavailable" alone leaves a reader unable to tell a
    missing extra from a missing download from a broken revision."""
    generator = build_generator("byt5", generator_id="byt5")
    if generator.available():  # pragma: no cover - only when the model is cached locally
        pytest.skip("the checkpoint is present, so there is no failure message to check")
    assert generator.load_error
    assert "byt5" in generator.spec.identity


def test_llm_generator_reports_unavailable_without_credentials() -> None:
    assert not LlmGenerator().available()


# --- evidence configurations --------------------------------------------------------------------
def test_the_full_ablation_ladder_exists() -> None:
    # The CGV3-era ablation ladder plus the SGV1 verifier rungs; an accidental key
    # (typo, half-renamed rung) shows up here rather than at run time.
    assert set(EVIDENCE_CONFIGS) == {
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "sgv1_v0",
        "sgv1_v1",
        "sgv1_v2",
    }


def test_v0_sees_nothing_and_v6_sees_everything() -> None:
    assert get_evidence_config("v0").fields == frozenset()
    assert get_evidence_config("v6").fields == frozenset(EvidenceField)


def test_v3_is_text_plus_confidence_without_pixels() -> None:
    """The left-hand side of the central comparison."""
    spec = get_evidence_config("v3")
    assert not spec.uses_image
    assert spec.uses_text
    assert EvidenceField.OCR_CONFIDENCE in spec.fields


def test_v4_isolates_pixels() -> None:
    spec = get_evidence_config("v4")
    assert spec.uses_image
    assert not spec.uses_text


def test_unknown_evidence_config_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown evidence configuration"):
        get_evidence_config("v9")


# --- masking -------------------------------------------------------------------------------------
def _bundle(builder: EvidenceBuilder) -> object:
    view = ObservationView(
        site_id="s",
        candidate_id="c",
        document_id="d",
        dataset_id="ds",
        engine_id="e",
        original_ocr="rng",
        candidate_text="mg",
        ocr_span_ids=("sp",),
        image_sha256="a" * 64,
        image_width=900,
        image_height=560,
        bbox=BBox(x0=10.0, y0=10.0, x1=60.0, y1=30.0),
    )
    return builder.build(
        view,
        ocr_stream="Dose: rng total",
        char_start=6,
        char_end=9,
        native_confidences=[0.8],
        conf_scale="synthetic_0_1",
    )


@pytest.fixture
def builder() -> EvidenceBuilder:
    featurizer = ConfidenceFeaturizer()
    featurizer.fit([("e", 0.5, "synthetic_0_1"), ("e", 0.9, "synthetic_0_1")])
    return EvidenceBuilder(
        crop_policy=CropPolicy(), context_chars=8, confidence_featurizer=featurizer
    )


def test_v6_mask_keeps_everything(builder: EvidenceBuilder) -> None:
    masked = EvidenceMask.from_key("v6").apply(_bundle(builder))  # type: ignore[arg-type]
    assert masked.original_ocr == "rng"
    assert masked.candidate_text == "mg"
    assert masked.crop_recipe_sha256 is not None
    assert masked.conf_features.has_native_confidence
    assert masked.geom_features.has_geometry
    assert masked.masked_fields == ()


def test_v3_mask_removes_the_crop(builder: EvidenceBuilder) -> None:
    """The structural half of the ablation: V3 cannot read pixels even if a later
    featurizer change would happily use them."""
    masked = EvidenceMask.from_key("v3").apply(_bundle(builder))  # type: ignore[arg-type]
    assert masked.crop_recipe_sha256 is None
    assert EvidenceField.IMAGE_CROP in masked.masked_fields
    assert masked.original_ocr == "rng"
    assert masked.conf_features.has_native_confidence


def test_v2_mask_removes_confidence(builder: EvidenceBuilder) -> None:
    masked = EvidenceMask.from_key("v2").apply(_bundle(builder))  # type: ignore[arg-type]
    assert not masked.conf_features.has_native_confidence
    assert masked.conf_features.conf_normalized is None
    assert EvidenceField.OCR_CONFIDENCE in masked.masked_fields


def test_v4_mask_removes_all_text(builder: EvidenceBuilder) -> None:
    masked = EvidenceMask.from_key("v4").apply(_bundle(builder))  # type: ignore[arg-type]
    assert masked.original_ocr == ""
    assert masked.candidate_text == ""
    assert masked.context_before == ""
    assert masked.crop_recipe_sha256 is not None


def test_v0_mask_removes_everything(builder: EvidenceBuilder) -> None:
    masked = EvidenceMask.from_key("v0").apply(_bundle(builder))  # type: ignore[arg-type]
    assert masked.original_ocr == ""
    assert masked.crop_recipe_sha256 is None
    assert not masked.conf_features.has_native_confidence
    assert not masked.geom_features.has_geometry
    assert set(masked.masked_fields) == set(EvidenceField)


def test_masking_narrows_available_fields(builder: EvidenceBuilder) -> None:
    """A reader must not conclude a channel is present merely because the engine
    originally provided it."""
    masked = EvidenceMask.from_key("v1").apply(_bundle(builder))  # type: ignore[arg-type]
    assert set(masked.available_fields) <= {EvidenceField.OCR_CONFIDENCE}


def test_require_raises_on_a_masked_channel() -> None:
    mask = EvidenceMask.from_key("v3")
    mask.require(EvidenceField.OCR_CONFIDENCE)  # allowed
    with pytest.raises(MaskedFieldAccessError, match="excludes"):
        mask.require(EvidenceField.IMAGE_CROP)


# --- context windows -------------------------------------------------------------------------------
def test_context_comes_from_the_ocr_stream() -> None:
    before, after = context_window("Dose: 0.015 mg total", 6, 11, width=6)
    assert before == "Dose: "
    assert after == " mg to"


def test_context_clamps_at_the_stream_edges() -> None:
    before, after = context_window("abc", 0, 3, width=10)
    assert before == ""
    assert after == ""


def test_zero_width_context_is_empty() -> None:
    assert context_window("abcdef", 2, 4, width=0) == ("", "")


# --- confidence featurizer -----------------------------------------------------------------------------
def test_featurizer_must_be_fitted_first() -> None:
    """Estimating statistics at transform time would pool the split being evaluated into
    its own features (leakage vector L3)."""
    with pytest.raises(NotFittedError, match="before fit"):
        ConfidenceFeaturizer().transform([0.5], "synthetic_0_1", "e")


def test_featurizer_standardizes_within_each_engine() -> None:
    """Pooling a 0-100 engine with a 0-1 engine into one z-score would compare
    incomparable quantities."""
    featurizer = ConfidenceFeaturizer()
    featurizer.fit(
        [
            ("a", 0.1, "synthetic_0_1"),
            ("a", 0.9, "synthetic_0_1"),
            ("b", 10.0, "synthetic_0_100"),
            ("b", 90.0, "synthetic_0_100"),
        ]
    )
    assert set(featurizer.engines) == {"a", "b"}
    low = featurizer.transform([0.1], "synthetic_0_1", "a")
    high = featurizer.transform([0.9], "synthetic_0_1", "a")
    assert low.conf_zscore_within_engine is not None
    assert high.conf_zscore_within_engine is not None
    assert low.conf_zscore_within_engine < 0 < high.conf_zscore_within_engine


def test_featurizer_preserves_the_native_value() -> None:
    featurizer = ConfidenceFeaturizer()
    featurizer.fit([("e", 50.0, "tesseract_word_conf_0_100")])
    features = featurizer.transform([87.0], "tesseract_word_conf_0_100", "e")
    assert features.native_min == 87.0
    assert features.conf_normalized == pytest.approx(0.87)


def test_missing_confidence_is_reported_as_absent_not_zero() -> None:
    featurizer = ConfidenceFeaturizer()
    featurizer.fit([("e", 0.5, "synthetic_0_1")])
    features = featurizer.transform([None, None], None, "e")
    assert not features.has_native_confidence
    assert features.conf_normalized is None
    assert features.n_spans == 2


def test_site_confidence_summary_is_the_weakest_span() -> None:
    """The weakest span drives the risk of a multi-span edit."""
    featurizer = ConfidenceFeaturizer()
    featurizer.fit([("e", 0.5, "synthetic_0_1")])
    features = featurizer.transform([0.9, 0.2, 0.7], "synthetic_0_1", "e")
    assert features.conf_normalized == pytest.approx(0.2)


# --- geometry features ----------------------------------------------------------------------------------
def test_geometry_features_are_scale_free() -> None:
    """The same document at two resolutions must give the same features, or a verifier
    would learn its training corpus's dpi."""
    small = geometry_features(BBox(x0=10, y0=10, x1=60, y1=30), 100, 100)
    large = geometry_features(BBox(x0=20, y0=20, x1=120, y1=60), 200, 200)
    assert small.width == pytest.approx(large.width)
    assert small.relative_x == pytest.approx(large.relative_x)
    assert small.aspect_ratio == pytest.approx(large.aspect_ratio)


def test_geometry_absent_when_there_is_no_box() -> None:
    features = geometry_features(None, 100, 100)
    assert not features.has_geometry
    assert features.width is None


def test_degenerate_box_reports_no_aspect_ratio() -> None:
    features = geometry_features(BBox(x0=5, y0=5, x1=15, y1=5), 100, 100)
    assert features.has_geometry
    assert features.aspect_ratio is None


# --- crops -----------------------------------------------------------------------------------------------
def test_recipe_hash_is_deterministic() -> None:
    box = BBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    a = build_recipe("sha", box, CropPolicy())
    b = build_recipe("sha", box, CropPolicy())
    assert a.recipe_sha256 == b.recipe_sha256


def test_recipe_hash_changes_with_policy() -> None:
    """Two crops taken under different padding are different evidence."""
    box = BBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    a = build_recipe("sha", box, CropPolicy(padding_ratio=0.1))
    b = build_recipe("sha", box, CropPolicy(padding_ratio=0.5))
    assert a.recipe_sha256 != b.recipe_sha256


def test_recipe_hash_changes_with_the_source_image() -> None:
    box = BBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    assert (
        build_recipe("image_a", box, CropPolicy()).recipe_sha256
        != build_recipe("image_b", box, CropPolicy()).recipe_sha256
    )


def test_materialize_reproduces_identical_pixels(isolated_env: Path, tmp_path: Path) -> None:
    """The whole point of storing recipes: the same recipe reconstructs the same bytes."""
    source = tmp_path / "page.png"
    Image.new("L", (200, 100), color=255).save(source)
    recipe = build_recipe("sha", BBox(x0=20, y0=20, x1=80, y1=60), CropPolicy())

    first = materialize(recipe, source)
    first_bytes = first.read_bytes()
    first.unlink()
    second = materialize(recipe, source)
    assert second.read_bytes() == first_bytes


def test_materialize_uses_the_cache(isolated_env: Path, tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("L", (200, 100), color=255).save(source)
    recipe = build_recipe("sha", BBox(x0=20, y0=20, x1=80, y1=60), CropPolicy())
    path = materialize(recipe, source)
    assert path == crop_path(recipe)
    assert path.is_file()


def test_crop_is_resized_to_the_target_height(isolated_env: Path, tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("L", (400, 200), color=255).save(source)
    recipe = build_recipe(
        "sha", BBox(x0=0, y0=0, x1=100, y1=20), CropPolicy(target_height=48, padding_ratio=0.0)
    )
    with Image.open(materialize(recipe, source)) as crop:
        assert crop.height == 48


def test_crop_near_the_margin_is_clamped(isolated_env: Path, tmp_path: Path) -> None:
    """A box at the page edge must not shift inward, which would change what the verifier
    sees depending on where the span happens to sit."""
    source = tmp_path / "page.png"
    Image.new("L", (100, 50), color=255).save(source)
    recipe = build_recipe(
        "sha", BBox(x0=0, y0=0, x1=20, y1=10), CropPolicy(padding_ratio=1.0, target_height=None)
    )
    with Image.open(materialize(recipe, source)) as crop:
        assert crop.width <= 100
        assert crop.height <= 50


def test_degenerate_box_still_produces_a_crop(isolated_env: Path, tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("L", (100, 50), color=255).save(source)
    recipe = build_recipe(
        "sha", BBox(x0=10, y0=10, x1=10, y1=20), CropPolicy(padding_ratio=0.0, padding_min_px=0)
    )
    with Image.open(materialize(recipe, source)) as crop:
        assert crop.width >= 1
        assert crop.height >= 1

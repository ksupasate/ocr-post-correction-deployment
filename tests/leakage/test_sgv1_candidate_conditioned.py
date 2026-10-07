"""Guards for the SGV1 candidate-conditioned representation.

Phase 2 ended at a structural ceiling: every representation it tested was site-conditioned,
so within a matched pair the image features were byte-identical and cancelled exactly. This
stage claims to break that. The claim is one number -- how many of the 794 pairs now differ
-- and it is checkable, so it is checked here rather than asserted in a docstring.

The second load-bearing claim is the R0 control. R0 is candidate-BLIND by construction; if
it showed any within-pair variation it would be seeing the candidate, and the comparison
against it would be meaningless.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sgv1_candidate_conditioned.py"
OUT = REPO / "results/generated/sgv1/candidate_conditioned"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sgv1_candidate_conditioned", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module() -> ModuleType:
    return _script()


def _artifact(name: str) -> dict:
    path = OUT / name
    if not path.is_file():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text())


# --- the acceptance test for the whole phase ------------------------------------------


def test_candidate_conditioning_breaks_the_phase_2_ceiling() -> None:
    """Phase 2: 0 of 794 pairs varied. Anything at or near zero here is a failed phase."""
    audit = _artifact("within_pair_variation.json")
    assert audit["pairs"] == 794
    assert audit["phase_2_baseline_pairs_with_visual_variation"] == 0
    assert audit["pairs_with_any_variation"] == audit["pairs"]
    # The visual family is the one Phase 2 proved could not vary. It has to now.
    assert audit["pairs_with_visual_variation"] > 0.9 * audit["pairs"]


def test_r0_is_candidate_blind_which_is_what_makes_it_a_control() -> None:
    """R0 must show EXACTLY zero within-pair variation, or it is not candidate-blind."""
    audit = _artifact("within_pair_variation.json")
    assert audit["pairs_with_r0_variation"] == 0


def test_r0_block_never_reads_the_candidate(module: ModuleType) -> None:
    """Red team at the source, not just the artifact."""
    source = inspect.getsource(module.r0_block)
    assert "candidate_text" not in source
    assert "row.candidate" not in source
    # It may read the OCR observation and its geometry, and nothing about the proposal.
    assert "original_ocr" in source


# --- fitted resources are TRAIN-only and ground-truth-free ----------------------------


def test_every_fitted_resource_is_train_only(module: ModuleType) -> None:
    for function in (module.fit_resources, module.fit_glyph_prototypes):
        source = inspect.getsource(function)
        assert "train_documents" in source
        assert "CALIBRATION" not in source
        assert "DEVELOPMENT" not in source


def test_glyph_prototypes_are_labelled_by_ocr_not_ground_truth(module: ModuleType) -> None:
    """The supervision is the engine's own reading, which a deployed system has."""
    source = inspect.getsource(module.fit_glyph_prototypes)
    assert "original_ocr" in source
    for forbidden in ("region_gt", "gt_text", "outcome", "is_harmful"):
        assert forbidden not in source


def test_no_feature_family_reads_a_label(module: ModuleType) -> None:
    for function in (
        module.r0_block,
        module.edit_block,
        module.plausibility_block,
        module.visual_block,
        module.context_block,
    ):
        source = inspect.getsource(function)
        for forbidden in (
            "is_harmful",
            "beneficial",
            "outcome",
            "d_before",
            "d_after",
            "region_gt",
        ):
            assert forbidden not in source, f"{function.__name__} mentions {forbidden}"


def test_feature_record_declares_ground_truth_was_not_used() -> None:
    record = _artifact("feature_record.json")
    assert record["ground_truth_used_for_features"] is False
    # Row selection IS a ground-truth fact and the record must say so rather than imply
    # the stage never touched a label at all.
    assert record["ground_truth_used_for_row_selection"] is True


# --- the visual family must actually depend on the candidate --------------------------


def test_visual_block_varies_with_the_candidate_not_only_its_length(module: ModuleType) -> None:
    """The bug this phase had to design around.

    Cell statistics keyed on ``len(Y)`` are identical for same-length candidates, which is
    exactly the O/0, I/1, S/5 substitution case. The glyph-prototype term is what makes
    same-length candidates separable, so a synthetic crop plus two same-length candidates
    must produce different vectors.
    """
    rng = np.random.default_rng(0)
    ink = np.zeros((16, 64), dtype=np.float64)
    for start in range(2, 60, 8):
        ink[4:12, start : start + 4] = rng.uniform(0.5, 1.0, size=(8, 4))
    resources = module.FittedResources(
        lm={},
        lm_backoff=-5.0,
        lexicon=frozenset(),
        glyphs={"0": rng.normal(size=32), "8": rng.normal(size=32)},
        width_per_char=8.0,
        documents=1,
    )
    _, a = module.visual_block(ink, "1208", "1200", resources)
    _, b = module.visual_block(ink, "1208", "1288", resources)
    assert len(a) == len(b) == len(module.VISUAL_NAMES)
    assert a != b, "same-length candidates produced identical visual features"


def test_visual_block_reports_missing_rather_than_faking_zeros(module: ModuleType) -> None:
    names, values = module.visual_block(
        None,
        "abc",
        "abd",
        module.FittedResources(
            lm={}, lm_backoff=-5.0, lexicon=frozenset(), glyphs={}, width_per_char=1.0, documents=0
        ),
    )
    assert names[-1] == "vis_missing"
    assert values[-1] == 1.0


# --- the three actions are a policy, not a fabricated label ---------------------------


def test_abstain_is_a_threshold_band_not_a_trained_class(module: ModuleType) -> None:
    """There is no abstain label in this corpus; training one would fabricate a target."""
    record = _artifact("fit_record.json")
    assert record["actions"] == ["CORRECT", "PRESERVE", "ABSTAIN"]
    assert "not a trained class" in record["action_note"]
    source = inspect.getsource(module._select_thresholds)
    assert "CALIBRATION" in inspect.getsource(module.run_fit) or "cal_" in source


def test_thresholds_are_chosen_on_calibration_only() -> None:
    record = _artifact("fit_record.json")
    for arm in record["arms"].values():
        assert arm["threshold_scope"] == "CALIBRATION rows only"
        assert arm["tau_lo"] <= arm["tau_hi"]


def test_every_arm_shares_the_same_rows_and_roles() -> None:
    record = _artifact("fit_record.json")
    arms = record["arms"].values()
    for key in ("fit_rows", "calibration_rows", "evaluation_rows"):
        assert len({a[key] for a in arms}) == 1, f"{key} differs across arms"


def test_the_fit_asserts_role_disjointness(module: ModuleType) -> None:
    source = inspect.getsource(module.run_fit)
    assert "a document appears in both the fit and evaluation role" in source
    assert "a document appears in both the calibration and evaluation role" in source


# --- reported dimensions must match the fitted matrix ---------------------------------


def test_recorded_feature_dimension_matches_the_families_it_claims() -> None:
    """A record that misstates its own width is a provenance defect even if the model is
    right. Phase 2's hardcoded embedding names made exactly that mistake here."""
    record = _artifact("fit_record.json")
    v1 = record["arms"]["V1"]["feature_dimension"]
    for name, arm in record["arms"].items():
        if arm["kind"] != "v1_plus":
            continue
        assert arm["feature_dimension"] == v1 + arm["extra_feature_columns"], name


def test_ablations_are_strictly_smaller_than_the_full_representation() -> None:
    record = _artifact("fit_record.json")
    full = record["arms"]["R1"]["feature_dimension"]
    for name, arm in record["arms"].items():
        if name.startswith("R1_no_"):
            assert arm["feature_dimension"] < full, name


# --- discrimination uses the raw score, not the calibrated probability ----------------


def test_discrimination_uses_the_raw_score(module: ModuleType) -> None:
    """Isotonic calibration is a step function: it collapses ~1,400 distinct scores to ~47
    and the ties depress AUC. Using the raw score is what makes V1 reproduce Phase 2."""
    source = inspect.getsource(module._discrimination)
    assert 'f"score_{arm}"' in source
    assert 'f"psafe_{arm}"' not in source


def test_v1_reproduces_the_phase_2_core_auc() -> None:
    """The comparability check. If V1 has drifted, no contrast here means anything."""
    phase2 = REPO / "results/generated/sgv1/representation/representation_metrics.json"
    if not phase2.is_file():
        pytest.skip("phase-2 metrics absent")
    expected = json.loads(phase2.read_text())["by_arm"]["V1"]["core_roc_auc"]
    baseline = _artifact("baseline_results.json")
    assert baseline["core"]["V1"]["roc_auc"] == pytest.approx(expected, abs=1e-4)


# --- honest reporting ------------------------------------------------------------------


def test_population_sensitivity_is_recorded_because_the_result_reverses() -> None:
    """The headline gain is on a stratified frame and inverts on the natural pool. A record
    that reported only the favourable population would be misleading."""
    robustness = _artifact("robustness_results.json")
    sensitivity = robustness["population_sensitivity"]
    assert {"full_development_pool", "frame_a_core_primary"} <= set(sensitivity)
    assert sensitivity["frame_a_core_primary"]["R1_minus_V1"] > 0
    assert sensitivity["full_development_pool"]["R1_minus_V1"] < 0
    assert "reverses" in robustness["population_sensitivity_note"]


def test_harm_question_is_answered_at_matched_coverage() -> None:
    """The arms operate at different coverage, so the raw joint rate is not a comparison."""
    decision = _artifact("research_decision.json")
    q3 = decision["q3_reduces_harmful_corrections"]
    assert "coverage_matched" in q3
    assert "at_each_arms_own_operating_point" in q3
    assert q3["coverage_matched"]["aurc_R1"] < q3["coverage_matched"]["aurc_V1"]


def test_decision_issues_no_hypothesis_verdict() -> None:
    decision = _artifact("research_decision.json")
    assert decision["hypothesis_verdict"] is None
    assert decision["c2_status"] == "DEFERRED"
    assert decision["q5_publishable_contribution"]["answer"].startswith("NOT YET")


def test_no_confirmatory_document_reaches_any_artifact() -> None:
    lock = json.loads((REPO / "manifests/sgv1/confirmatory_reserve_lock.json").read_text())
    reserved = {
        str(d) for d in (lock.get("document_ids") or lock.get("confirmatory_document_ids") or [])
    }
    if not reserved:
        pytest.skip("reserve lock does not enumerate document ids")
    path = OUT / "decision_scores.parquet"
    if not path.is_file():
        pytest.skip("scores not produced yet")
    documents = set(pd.read_parquet(path, columns=["document_id"])["document_id"].astype(str))
    assert not reserved & documents


def test_every_record_declares_no_confirmatory_access() -> None:
    for name in (
        "feature_record.json",
        "fit_record.json",
        "baseline_results.json",
        "candidate_results.json",
        "ablation_results.json",
        "calibration_results.json",
        "category_results.json",
        "robustness_results.json",
        "research_decision.json",
        "within_pair_variation.json",
        "figure_manifest.json",
    ):
        path = OUT / name
        if not path.is_file():
            continue
        assert json.loads(path.read_text())["confirmatory_accessed"] is False, name


def test_error_categories_come_from_the_edit_not_the_label(module: ModuleType) -> None:
    source = inspect.getsource(module._error_category)
    for forbidden in ("outcome", "is_harmful", "beneficial", "d_before"):
        assert forbidden not in source

    class _Row:
        original_ocr, candidate_text, anchor_kind, operation = (
            "12O8",
            "1208",
            "token",
            "substitution",
        )

    assert module._error_category(_Row()) == "character_confusion"

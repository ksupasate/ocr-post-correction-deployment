"""Guards for the SGV1 target-alignment stage.

This stage exists because the Phase-3 headline was scored on a target the models were not
trained for. The tests that matter here are the ones that stop that recurring: the three
targets must be distinguishable, the accept-score convention must follow the target, and
the primary endpoint must be the objective rather than whichever metric flatters an arm.
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
SCRIPT = REPO / "scripts/sgv1_target_alignment.py"
OUT = REPO / "results/generated/sgv1/target_alignment"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sgv1_target_alignment", SCRIPT)
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


# --- the three targets are genuinely different -----------------------------------------


def test_outcome_class_separates_neutral_from_safe(module: ModuleType) -> None:
    """The whole defect: a binary harm target cannot express 'changed nothing'."""
    frame = pd.DataFrame(
        {
            "is_harmful": [True, False, False],
            "beneficial": [False, False, True],
        }
    )
    assert module._outcome_class(frame).tolist() == [0, 1, 2]


def test_the_three_targets_are_not_the_same_vector(module: ModuleType) -> None:
    frame = pd.DataFrame({"is_harmful": [True, False, False], "beneficial": [False, False, True]})
    harm = module._target_vector(frame, "harm")
    benefit = module._target_vector(frame, "benefit")
    three = module._target_vector(frame, "three")
    assert harm.tolist() != benefit.tolist()
    assert set(np.unique(three).tolist()) == {0, 1, 2}


def test_accept_score_follows_the_target_convention(module: ModuleType) -> None:
    """Higher must always mean 'more worth accepting', whatever the label set."""

    class _Model:
        classes_ = np.array([0, 1])

        @staticmethod
        def predict_proba(matrix: np.ndarray) -> np.ndarray:
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    harm = module._accept_score(_Model(), np.zeros((2, 1)), "harm")
    benefit = module._accept_score(_Model(), np.zeros((2, 1)), "benefit")
    # For the harm target, class 0 is "not harmful" -> row 0 is the safer one.
    assert harm[0] > harm[1]
    # For the benefit target, class 1 is "beneficial" -> row 1 is the better one.
    assert benefit[1] > benefit[0]


# --- the primary endpoint is the objective, not the flattering metric -------------------


def test_primary_endpoint_is_repair_recall_under_a_harm_bound() -> None:
    results = _artifact("alignment_results.json")
    assert results["primary_endpoint"] == "repair_recall_at_bounded_harm"
    assert "WORSE at the job" in results["primary_endpoint_note"]


def test_repair_recall_is_hand_computable(module: ModuleType) -> None:
    """Six rows, ranked by score: check the bound is applied where it actually binds."""
    frame = pd.DataFrame(
        {
            "score_x": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "is_harmful": [False, False, True, False, True, True],
            "beneficial": [True, True, False, True, False, False],
        }
    )
    out = module._repair_recall_at_bounded_harm(frame, "x")
    assert out["total_beneficial"] == 3
    # Top 2 are both beneficial, harm rate 0/2 = 0. Top 3 -> 1/3 = 0.333 > 0.20.
    assert out["epsilon_20"]["n_accepted"] == 2
    assert out["epsilon_20"]["repair_recall"] == pytest.approx(2 / 3)
    # Top 4 -> 1/4 = 0.25, still above 0.20; the endpoint must not accept it.
    assert out["epsilon_20"]["realized_harm_rate"] <= 0.20


def test_the_recommended_arm_is_chosen_by_the_objective_not_by_auc() -> None:
    """The two disagree here, and the decision must say which one governs."""
    decision = _artifact("research_decision.json")
    q = decision["q_should_the_target_be_changed"]
    assert q["answer"].startswith("NO")
    assert q["best_by_repair_recall_at_epsilon_5"] != decision["best_arm_by_auc_on_beneficial"]
    assert "NOT the recommended arm" in decision["best_arm_note"]


# --- the reversal test ------------------------------------------------------------------


def test_the_reversal_exists_only_under_the_mismatch() -> None:
    results = _artifact("alignment_results.json")
    reversal = results["reversal_test"]
    assert reversal["harm"]["reversal_present"] is True
    assert reversal["benefit"]["reversal_present"] is False
    assert reversal["three"]["reversal_present"] is False


def test_h2_is_recorded_as_untestable_rather_than_failed() -> None:
    decision = _artifact("research_decision.json")
    q = decision["q_is_adaptive_routing_warranted"]
    assert q["answer"] == "no"
    assert "zero positives" in q["why"]
    assert "NOT TESTABLE" in decision["hypothesis_note"]


def test_the_degeneracy_proof_is_recorded_with_its_evidence() -> None:
    path = (
        REPO
        / "results/generated/sgv1/candidate_conditioned/incidents/sgv1_r17_target_mismatch.json"
    )
    if not path.is_file():
        pytest.skip("defect record absent")
    record = json.loads(path.read_text())
    proof = record["h2_degeneracy_proof"]
    assert (
        proof["definition_a_site_has_any_beneficial_candidate"][
            "beneficial_rows_outside_a_supported_site"
        ]
        == 0
    )
    assert (
        proof["definition_b_d_before_greater_than_zero"]["beneficial_rows_with_d_before_zero"] == 0
    )
    assert record["severity"] == "HIGH"


def test_the_phase_3_core_result_is_recorded_as_unaffected() -> None:
    """Checked rather than assumed: Frame A core has no neutral rows, so the two targets
    coincide there and the +0.0876 carries no mismatch."""
    scores = REPO / "results/generated/sgv1/candidate_conditioned/decision_scores.parquet"
    frame_a = REPO / "results/generated/sgv1/dev_frames/frame_a.parquet"
    if not scores.is_file() or not frame_a.is_file():
        pytest.skip("phase-3 artifacts absent")
    s = pd.read_parquet(scores)
    ids = set(pd.read_parquet(frame_a)["candidate_id"].astype(str))
    core = s[s["candidate_id"].astype(str).isin(ids) & ~s["region_is_whitespace_only"]]
    neutral = (~core["is_harmful"]) & (~core["beneficial"])
    assert int(neutral.sum()) == 0
    assert bool((core["beneficial"] == ~core["is_harmful"]).all())


# --- role and reserve discipline --------------------------------------------------------


def test_the_fit_asserts_role_disjointness(module: ModuleType) -> None:
    source = inspect.getsource(module.run_fit)
    assert "a document appears in both the fit and evaluation role" in source
    assert "a document appears in both the calibration and evaluation role" in source


def test_every_arm_shares_the_same_rows() -> None:
    record = _artifact("fit_record.json")
    for key in ("fit_rows", "calibration_rows", "evaluation_rows"):
        assert len({a[key] for a in record["arms"].values()}) == 1, key


def test_all_arms_are_calibrated_against_the_same_event() -> None:
    """Scores must be comparable across targets even though the labels are not."""
    record = _artifact("fit_record.json")
    assert {a["calibrated_against"] for a in record["arms"].values()} == {"beneficial"}


def test_no_confirmatory_document_reaches_any_artifact() -> None:
    lock = json.loads((REPO / "manifests/sgv1/confirmatory_reserve_lock.json").read_text())
    reserved = {
        str(d) for d in (lock.get("document_ids") or lock.get("confirmatory_document_ids") or [])
    }
    if not reserved:
        pytest.skip("reserve lock does not enumerate document ids")
    path = OUT / "alignment_scores.parquet"
    if not path.is_file():
        pytest.skip("scores not produced yet")
    documents = set(pd.read_parquet(path, columns=["document_id"])["document_id"].astype(str))
    assert not reserved & documents


def test_every_record_declares_no_confirmatory_access() -> None:
    for name in (
        "fit_record.json",
        "alignment_results.json",
        "decision_results.json",
        "research_decision.json",
        "figure_manifest.json",
    ):
        path = OUT / name
        if not path.is_file():
            continue
        assert json.loads(path.read_text())["confirmatory_accessed"] is False, name


def test_decision_issues_no_hypothesis_verdict() -> None:
    decision = _artifact("research_decision.json")
    assert decision["hypothesis_verdict"] is None
    assert decision["c2_status"] == "DEFERRED"

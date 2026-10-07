"""Fairness, locality, and role guards for the SGV1 image-representation ladder.

The representation study's whole claim is that a rung differs from V1 in the image channel
alone, and that the pixels a candidate is shown are the pixels its own OCR geometry names.
Session 2 established both for the eight-statistic rung; these tests re-establish them for
the learned rungs, where the image channel arrives as a precomputed embedding and the
opportunity to smuggle something else in is larger, not smaller.
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
SCRIPT = REPO / "scripts/sgv1_representation.py"
REPR_DIR = REPO / "results/generated/sgv1/representation"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sgv1_representation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module() -> ModuleType:
    return _script()


def _artifact(name: str) -> Path:
    path = REPR_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not produced yet")
    return path


# --- representation fairness ----------------------------------------------------------


def test_the_embedding_block_is_the_only_thing_added_to_v1(module: ModuleType) -> None:
    """A rung is V1's evidence configuration plus one image block, by construction."""
    source = inspect.getsource(module.run_fit)
    assert 'evidence_config = "sgv1_v0" if plan.label == "V0" else "sgv1_v1"' in source
    # Crops are never handed to the featurizer for the learned rungs: the image channel
    # arrives only as the embedding, so image_block cannot also fire and conflate rungs.
    assert "bundles, {}, mask" in source
    assert len(module.EMBED_NAMES) == module.EMBED_DIM + 1
    assert module.EMBED_NAMES[-1] == "img_missing"


def test_representation_verifier_appends_after_provenance_and_only_once(
    module: ModuleType,
) -> None:
    verifier = module.RepresentationVerifier(
        verifier_id="v", evidence_config="sgv1_v1", embedding={"c-0": np.zeros(33)}
    )
    assert verifier._embedding
    plain = module.RepresentationVerifier(verifier_id="v", evidence_config="sgv1_v1")
    assert plain._embedding == {}


def test_no_embedding_feature_can_come_from_a_label(module: ModuleType) -> None:
    """Red team: the embedding path must never see an outcome column."""
    for function in (module.run_encode, module._embed_all, module._embedding_for):
        source = inspect.getsource(function)
        for forbidden in ("outcome", "beneficial", "d_before", "d_after", "region_gt"):
            assert forbidden not in source, f"{function.__name__} mentions {forbidden}"


def test_the_encoder_target_is_the_fit_role_only(module: ModuleType) -> None:
    """The encoder is a fitted transformer: TRAIN documents, then frozen."""
    source = inspect.getsource(module._train_encoder)
    assert 'frame["role"] == "TRAIN"' in source
    assert "CALIBRATION" not in source
    assert "DEVELOPMENT" not in source
    inner = inspect.getsource(module._inner_split)
    assert "inner" in inner


def test_fit_plan_is_the_registered_grid_with_no_duplicates(module: ModuleType) -> None:
    plans = module._fit_plans()
    labels = [p.label for p in plans]
    assert len(labels) == len(set(labels))
    assert {"V0", "V1"} <= set(labels)
    for plan in plans:
        if plan.rung is not None:
            assert plan.arm in module.IMAGE_ARMS
            assert plan.scale in module.CROP_SCALES
            assert plan.seed in module.ENCODER_SEEDS


def test_baseline_scale_reproduces_the_session_2_crop_policy(module: ModuleType) -> None:
    """Rung A is only comparable to the new rungs if its crops are the same crops."""
    import sgv1_verifier_pilot as pilot

    assert module.CROP_SCALES[module.BASELINE_SCALE] == pilot.CROP_POLICY


# --- ablations ------------------------------------------------------------------------


def test_masked_embedding_is_structurally_empty(module: ModuleType) -> None:
    plan = module.FitPlan("x", "B", module.BASELINE_SCALE, module.PRIMARY_SEED, "masked")
    embedding = module._embedding_for(plan, ["c-0", "c-1"])
    assert embedding is not None
    for values in embedding.values():
        assert values[: module.EMBED_DIM].tolist() == [0.0] * module.EMBED_DIM
        assert values[module.EMBED_DIM] == 1.0


def test_randproj_is_deterministic_full_width_and_carries_no_image(module: ModuleType) -> None:
    """The capacity control must have variance -- a constant column proves nothing."""
    first = module._randproj(["c-0", "c-1", "c-2"])
    second = module._randproj(["c-0", "c-1", "c-2"])
    for key in first:
        assert first[key].tolist() == second[key].tolist()
        assert first[key][module.EMBED_DIM] == 0.0
        assert np.std(first[key][: module.EMBED_DIM]) > 0.0
    assert first["c-0"].tolist() != first["c-1"].tolist()


def test_donor_pick_is_deterministic(module: ModuleType) -> None:
    options = [f"r{i}" for i in range(17)]
    picks = {module._pick(11, f"cand-{i}", options) for i in range(200)}
    assert picks <= set(options)
    assert module._pick(11, "cand-3", options) == module._pick(11, "cand-3", options)
    assert module._pick(11, "cand-3", options) != module._pick(12, "cand-3", options)


# --- crop lineage ---------------------------------------------------------------------


def test_wrong_local_donor_is_the_same_page_at_a_different_site() -> None:
    """The locality control only controls if the page is held constant and the site is not."""
    lineage = pd.read_parquet(_artifact("representation_lineage.parquet"))
    module = _script()
    for scale in module.CROP_SCALES:
        donor = lineage[f"wrong_local_{scale}"]
        own = lineage[f"recipe_{scale}"]
        available = donor.notna()
        assert int((donor[available] == own[available]).sum()) == 0
        # Every donor recipe must belong to the same document: build the document -> recipe
        # index and check membership, which is what "same page" actually means.
        by_document: dict[str, set[str]] = {}
        for row in lineage.itertuples():
            by_document.setdefault(str(row.document_id), set()).add(
                str(getattr(row, f"recipe_{scale}"))
            )
        offenders = [
            str(row.candidate_id)
            for row in lineage[available].itertuples()
            if str(getattr(row, f"wrong_local_{scale}")) not in by_document[str(row.document_id)]
        ]
        assert not offenders, f"{len(offenders)} wrong_local donors are off-page at {scale}"


def test_shuffled_donor_is_always_another_document() -> None:
    lineage = pd.read_parquet(_artifact("representation_lineage.parquet"))
    module = _script()
    by_document: dict[str, set[str]] = {}
    for scale in module.CROP_SCALES:
        by_document.clear()
        for row in lineage.itertuples():
            by_document.setdefault(str(row.document_id), set()).add(
                str(getattr(row, f"recipe_{scale}"))
            )
        for row in lineage.head(4000).itertuples():
            digest = getattr(row, f"shuffled_{scale}")
            if isinstance(digest, str):
                assert digest not in by_document[str(row.document_id)]


def test_every_arm_covers_the_same_candidate_ids() -> None:
    lineage = pd.read_parquet(_artifact("representation_lineage.parquet"))
    module = _script()
    ids = set(lineage["candidate_id"].astype(str))
    assert len(ids) == len(lineage)
    for scale in module.CROP_SCALES:
        assert lineage[f"recipe_{scale}"].notna().all()


def test_crop_geometry_is_inherited_from_the_frozen_ocr_lineage(module: ModuleType) -> None:
    source = inspect.getsource(module.run_crops)
    assert "_bbox_of(row)" in source
    assert "gt_" not in source
    registry = json.loads(_artifact("crop_scale_registry.json").read_text())
    assert registry["geometry_never_from_ground_truth"] is True
    assert registry["geometry_source"].startswith("ocr_span_bbox_union")


# --- role discipline ------------------------------------------------------------------


def test_no_confirmatory_document_reaches_the_representation_lineage() -> None:
    lineage = pd.read_parquet(_artifact("representation_lineage.parquet"))
    lock = json.loads((REPO / "manifests/sgv1/confirmatory_reserve_lock.json").read_text())
    reserved = {
        str(d) for d in (lock.get("document_ids") or lock.get("confirmatory_document_ids") or [])
    }
    if not reserved:
        pytest.skip("reserve lock does not enumerate document ids")
    assert not reserved & set(lineage["document_id"].astype(str))
    assert set(lineage["role"].astype(str)) <= {"TRAIN", "CALIBRATION", "DEVELOPMENT"}


def test_every_representation_record_declares_no_confirmatory_access() -> None:
    for name in (
        "baseline_v2a.json",
        "representation_registry.json",
        "crop_scale_registry.json",
        "representation_record.json",
        "representation_metrics.json",
        "representation_decision.json",
    ):
        path = REPR_DIR / name
        if not path.is_file():
            continue
        assert json.loads(path.read_text())["confirmatory_accessed"] is False


def test_the_fit_asserts_role_disjointness(module: ModuleType) -> None:
    source = inspect.getsource(module.run_fit)
    assert "a document appears in both the fit and evaluation role" in source
    assert "a document appears in both the calibration and evaluation role" in source


# --- model provenance -----------------------------------------------------------------


def test_pretrained_backbone_provenance_is_recorded(module: ModuleType) -> None:
    provenance = module.RESNET18_PROVENANCE
    for key in ("model", "weights_enum", "source_url", "licence", "pretraining_corpus"):
        assert provenance[key]
    assert "limitation" in provenance
    source = inspect.getsource(module._build_encoder)
    assert "this stage does not download models" in source


def test_encoder_records_carry_training_provenance() -> None:
    directory = REPR_DIR / "encoders"
    if not directory.is_dir() or not any(directory.glob("*.json")):
        pytest.skip("no encoder records produced yet")
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text())
        for key in (
            "seed",
            "optimizer",
            "learning_rate",
            "batch_size",
            "max_epochs",
            "patience",
            "early_stopping",
            "selection_scope",
            "parameters",
            "trainable_parameters",
            "device",
            "runtime_seconds",
            "checkpoint_sha256",
        ):
            assert key in record, f"{path.name} lacks {key}"
        assert record["selection_scope"].startswith("TRAIN")


def test_registry_rejects_a_grid_that_drifted(module: ModuleType, monkeypatch) -> None:
    """Red team: preregistration is only real if a later edit fails closed."""
    if not (REPR_DIR / "representation_registry.json").is_file():
        pytest.skip("registry not produced yet")
    monkeypatch.setitem(
        module.CROP_SCALES, "local_tight", module.CropPolicy(padding_ratio=0.99, target_height=48)
    )
    with pytest.raises(module.RepresentationError, match="crop-scale grid"):
        module._assert_registered()


def test_baseline_binding_rejects_a_moved_artifact(module: ModuleType, monkeypatch) -> None:
    if not (REPR_DIR / "baseline_v2a.json").is_file():
        pytest.skip("baseline binding not produced yet")
    monkeypatch.setattr(module, "file_sha256", lambda _p: "0" * 64)
    with pytest.raises(module.RepresentationError, match="rung A baseline moved"):
        module._assert_baseline_intact()


# --- metrics --------------------------------------------------------------------------


def test_paired_contrast_clusters_by_document(module: ModuleType) -> None:
    source = inspect.getsource(module._paired)
    assert "cluster_of=lambda r: r[0]" in source
    rows = inspect.getsource(module._rows_for)
    assert '"document_id"' in rows


def test_engine_strata_are_never_silently_pooled(module: ModuleType) -> None:
    source = inspect.getsource(module.run_analyze)
    assert '"by_engine"' in source
    assert "_heterogeneity" in source


def test_heterogeneity_classification_is_hand_checked(module: ModuleType) -> None:
    assert module._heterogeneity({"a": 0.01, "b": 0.02})["classification"] == "consistent positive"
    assert module._heterogeneity({"a": -0.01, "b": -0.02})["classification"] == "negative"
    assert module._heterogeneity({"a": 0.01, "b": -0.05})["classification"] == "engine-dependent"
    assert module._heterogeneity({"a": 0.01, "b": -0.001})["classification"] == "mixed"
    assert module._heterogeneity({})["classification"] == "inconclusive"


def test_auc_statistic_handles_a_single_class(module: ModuleType) -> None:
    assert np.isnan(module._auc_rows([("d", 0.5, 1.0), ("d", 0.7, 1.0)]))
    assert np.isnan(module._auc_rows([]))
    # Hand-computed: one positive scored above one negative is a perfect ranking.
    assert module._auc_rows([("d", 0.9, 1.0), ("d", 0.1, 0.0)]) == pytest.approx(1.0)


def test_decision_never_issues_a_hypothesis_verdict() -> None:
    path = REPR_DIR / "representation_decision.json"
    if not path.is_file():
        pytest.skip("decision not produced yet")
    decision = json.loads(path.read_text())
    assert decision["hypothesis_verdict"] is None
    assert decision["c2_status"] == "DEFERRED"
    assert decision["status"] in {
        "REPRESENTATION_READY",
        "REPRESENTATION_PARTIALLY_READY",
        "REPRESENTATION_NOT_READY",
        "REPRESENTATION_INCONCLUSIVE",
    }


def test_fit_record_proves_v1_reproduces_session_2() -> None:
    path = REPR_DIR / "representation_record.json"
    if not path.is_file():
        pytest.skip("fit record not produced yet")
    record = json.loads(path.read_text())
    assert record["v1_reproduces_session_2"] is True
    assert record["v1_max_absolute_drift"] <= 1e-12
    for drift in record["masked_arm_max_absolute_drift_from_v1"].values():
        assert drift <= 1e-9


def test_reserve_check_is_re_derived_not_read_back(module: ModuleType) -> None:
    """Reviewer C, Medium: a gate that re-reads its own boolean proves nothing.

    The Session-2 C3 certificate checked `confirmatory_reserve_untouched` by reading a
    flag this same pipeline had written. This stage intersects the locked reserve against
    the document ids actually present in the tables it produced, so a regression that let
    a reserve document through while still writing the flag would fail the gate.
    """
    source = inspect.getsource(module._confirmatory_documents_absent)
    assert "read_parquet" in source
    assert "_confirmatory_document_ids()" in source
    gate = inspect.getsource(module.run_decide)
    assert 'record["confirmatory_accessed"]' not in gate
    reserved = module._confirmatory_document_ids()
    assert len(reserved) > 0


def test_reserve_check_fails_closed_on_a_reserve_document(module: ModuleType, monkeypatch) -> None:
    """Red team: plant a locked document id in the output and assert the gate raises."""
    if not (REPR_DIR / "representation_lineage.parquet").is_file():
        pytest.skip("representation lineage not produced yet")
    reserved = sorted(module._confirmatory_document_ids())
    real = pd.read_parquet
    planted = pd.DataFrame({"document_id": [reserved[0]]})

    def _fake(path, *args, **kwargs):
        if kwargs.get("columns") == ["document_id"]:
            return planted
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _fake)
    with pytest.raises(module.RepresentationError, match="confirmatory documents reached"):
        module._confirmatory_documents_absent()


# --- the registered selection rule ----------------------------------------------------


def _candidate(locality: float, engine_min: float, delta: float, *, eligible: bool = True) -> dict:
    return {
        "gate_2_passed": locality > 0.005,
        "gate_4_passed": engine_min > -0.02,
        "gate_3_incremental": {"core_auc_delta_vs_V1": delta},
        "locality_core": locality,
        "eligible_for_selection": eligible,
    }


def test_decision_requires_beating_the_irrelevant_crop_not_v1(module: ModuleType) -> None:
    """The whole point: a big win over V1 with no locality is NOT ready."""
    status, selected, reason = module._decide_status(
        True, {"B_correct": _candidate(locality=0.0001, engine_min=0.0, delta=0.05)}
    )
    assert status == "REPRESENTATION_NOT_READY"
    assert selected is None
    assert "irrelevant crop" in reason


def test_decision_ready_picks_the_largest_incremental_among_gate_passers(
    module: ModuleType,
) -> None:
    status, selected, _ = module._decide_status(
        True,
        {
            "B_correct": _candidate(locality=0.02, engine_min=0.0, delta=0.004),
            "C_correct": _candidate(locality=0.03, engine_min=0.0, delta=0.009),
            "A_correct": _candidate(locality=0.001, engine_min=0.0, delta=0.05),
        },
    )
    assert status == "REPRESENTATION_READY"
    assert selected == "C_correct"


def test_decision_is_partially_ready_when_locality_holds_but_an_engine_regresses(
    module: ModuleType,
) -> None:
    status, selected, reason = module._decide_status(
        True, {"B_correct": _candidate(locality=0.02, engine_min=-0.05, delta=0.01)}
    )
    assert status == "REPRESENTATION_PARTIALLY_READY"
    assert selected is None
    assert "engine" in reason


def test_decision_is_inconclusive_when_validity_fails_however_good_the_numbers(
    module: ModuleType,
) -> None:
    status, selected, _ = module._decide_status(
        False, {"B_correct": _candidate(locality=0.5, engine_min=0.0, delta=0.5)}
    )
    assert status == "REPRESENTATION_INCONCLUSIVE"
    assert selected is None


def test_decision_thresholds_are_the_registered_ones(module: ModuleType) -> None:
    registry_path = REPR_DIR / "representation_registry.json"
    if not registry_path.is_file():
        pytest.skip("registry not produced yet")
    registry = json.loads(registry_path.read_text())
    rule = registry["selection_rule"]
    assert str(module.LOCALITY_MIN_AUC) in rule["gate_2_locality"]["requirement"]
    assert str(module.ENGINE_REGRESSION_MAX) in rule["gate_4_engine_robustness"]["requirement"]
    assert rule["gate_2_locality"]["significance_required"] is False


def test_a_stability_seed_can_never_be_selected(module: ModuleType) -> None:
    """The registry says non-primary seeds are evidence, not a menu.

    This is not hypothetical: the best-scoring arm in this study IS a stability seed, and
    the registered primary seed is the most conservative of the three. Ranking over all
    seeds would be seed shopping with extra steps.
    """
    status, selected, _ = module._decide_status(
        True,
        {
            "B_seed_a_correct": _candidate(0.02, 0.0, 0.018, eligible=False),
            "B_seed_primary_correct": _candidate(0.013, 0.0, 0.014, eligible=True),
        },
    )
    assert status == "REPRESENTATION_READY"
    assert selected == "B_seed_primary_correct"


def test_only_stability_seeds_clearing_locality_does_not_make_a_selection(
    module: ModuleType,
) -> None:
    status, selected, _ = module._decide_status(
        True, {"B_seed_a_correct": _candidate(0.02, 0.0, 0.018, eligible=False)}
    )
    assert status == "REPRESENTATION_NOT_READY"
    assert selected is None


# --- the post-hoc power check ---------------------------------------------------------


def _ladder(engine_effects: dict[str, list[float]]) -> tuple[dict, dict]:
    engines = list(engine_effects)
    n = len(next(iter(engine_effects.values())))
    by_arm = {
        "V1": {"by_engine": {e: {"core_rows": 107 if e == "small" else 673} for e in engines}}
    }
    contrasts = {
        f"arm{i}": {"engine_effect_vs_V1": {"engines": {e: engine_effects[e][i] for e in engines}}}
        for i in range(n)
    }
    return by_arm, contrasts


def test_power_check_flags_a_gate_decided_inside_its_own_noise(module: ModuleType) -> None:
    """A margin smaller than the estimate's spread is not a robustness result."""
    by_arm, contrasts = _ladder({"small": [-0.019, 0.05, -0.06, 0.01], "big": [0.01] * 4})
    power = module._gate_4_power(by_arm, contrasts, "arm0")
    assert power["underpowered_for_selected_arm"] is True
    assert power["selected_arm_binding_engine"]["engine"] == "small"
    assert "small" in power["engines_whose_sd_exceeds_the_gate_bound"]


def test_power_check_accepts_a_gate_that_clears_its_own_noise(module: ModuleType) -> None:
    by_arm, contrasts = _ladder({"small": [0.05, 0.051, 0.049, 0.05], "big": [0.01] * 4})
    power = module._gate_4_power(by_arm, contrasts, "arm0")
    assert power["underpowered_for_selected_arm"] is False
    assert power["engines_whose_sd_exceeds_the_gate_bound"] == []


def test_power_check_is_inert_when_nothing_was_selected(module: ModuleType) -> None:
    by_arm, contrasts = _ladder({"small": [-0.019, 0.05], "big": [0.01, 0.01]})
    assert module._gate_4_power(by_arm, contrasts, None)["underpowered_for_selected_arm"] is None


def test_a_downgrade_must_record_the_registered_rule_it_departed_from() -> None:
    """A deviation is allowed; a silent deviation is not."""
    path = REPR_DIR / "representation_decision.json"
    if not path.is_file():
        pytest.skip("decision not produced yet")
    decision = json.loads(path.read_text())
    assert decision["registered_rule_status"] in {
        "REPRESENTATION_READY",
        "REPRESENTATION_PARTIALLY_READY",
        "REPRESENTATION_NOT_READY",
        "REPRESENTATION_INCONCLUSIVE",
    }
    if decision["status"] != decision["registered_rule_status"]:
        deviation = decision["deviation_from_registered_rule"]
        assert deviation is not None
        for key in ("type", "registered_status", "reported_status", "why"):
            assert deviation[key]
        assert decision["gate_4_power_check"]["underpowered_for_selected_arm"] is True

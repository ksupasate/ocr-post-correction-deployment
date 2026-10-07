"""Scientific-integrity checks; mutation fixtures are synthetic, not research results."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vf2", ROOT / "scripts/paper3_vf2_clean_execution.py")
assert SPEC and SPEC.loader
vf2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vf2)


@pytest.fixture(scope="module")
def registry():
    return json.loads(
        (
            ROOT / "docs/paper3/journal_track_2027/validity_recovery/P3_VF2_EXECUTION_REGISTRY.json"
        ).read_text()
    )


def test_real_clean_registry_and_minimum_scope(registry):
    gate = vf2.validate(registry)
    assert gate["models_requested"] == 492
    assert sum(c["arm"] == "b2_rk2_target_only" for c in registry["cells"]) == 240
    assert sum(c["arm"] == "M1_TARGET_LABEL_PERMUTATION" for c in registry["cells"]) == 10


@pytest.mark.parametrize("role", ["source_fit", "target_adapt", "target_calibrate", "resource_fit"])
def test_other_page_from_test_volume_is_rejected(registry, role):
    # SYNTHETIC mutation of membership only.
    r = copy.deepcopy(registry)
    c = r["cells"][0]
    p = c[role + "_pages"][0] if c[role + "_pages"] else "synthetic-other-page"
    if not c[role + "_pages"]:
        c[role + "_pages"].append(p)
    r["page_to_group"][p] = r["page_to_group"][c["test_pages"][0]]
    with pytest.raises(AssertionError):
        vf2.validate(r)


def test_calibration_group_cannot_fit_its_predictor(registry):
    r = copy.deepcopy(registry)
    c = next(c for c in r["cells"] if c["budget"] == "pool")
    r["page_to_group"][c["target_adapt_pages"][0]] = r["page_to_group"][
        c["target_calibrate_pages"][0]
    ]
    with pytest.raises(AssertionError):
        vf2.validate(r)


def test_no_outcome_columns_enter_feature_projection():
    forbidden = {
        "grade",
        "ground_truth",
        "exact",
        "is_harmful",
        "outcome",
        "beneficial",
        "d_before",
        "d_after",
    }
    assert forbidden.isdisjoint(vf2.BLIND_CANDIDATE_COLUMNS)
    assert forbidden.isdisjoint(vf2.rk3.OBSERVATION_COLUMNS)


def test_output_writer_refuses_silent_replacement(tmp_path):
    p = tmp_path / "record.json"
    vf2.write(p, {"synthetic": True})
    with pytest.raises(FileExistsError):
        vf2.write(p, {"synthetic": True, "changed": True})


def test_frozen_learner_and_control_seed_match_native():
    assert vf2.rk3.ROUNDS == vf2.rk1.LAMBDAMART_ROUNDS
    assert vf2.rk3.MIN_LEAF == vf2.rk1.LAMBDAMART_MIN_LEAF
    assert vf2.pf1.rk4.PERMUTATION_SEED == 20260931


def test_resources_and_feature_schema_after_rebuild():
    root = ROOT / "results/paper3_vf2/run1"
    if not (root / "manifests/feature_manifest.json").exists():
        pytest.skip("pre-resource/feature stage")
    manifest = vf2.read(root / "manifests/feature_manifest.json")
    assert manifest["GT_feature_input_columns"] == []
    assert len(vf2.rk3.columns_for(vf2.rk3.R5)) == 146
    assert all(v["unexpected_differences"] == 0 for v in manifest["verification"])
    assert manifest["feature_sha256"] == vf2.sha(root / "features/clean_feature_matrix.parquet")

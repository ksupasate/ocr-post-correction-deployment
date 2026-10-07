"""Pre-GT guards for the SGV1 development site and candidate freeze."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from ocr_risk.experiments.sgv1_frames import FrameError, freeze_candidates
from ocr_risk.experiments.sgv1_reserve import CONFIRMATORY, ReserveAccessError

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sgv1_development_candidates.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sgv1_development_candidates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "site_id": "site-1",
                "document_id": "cord-train-0000",
                "engine_id": "tesseract",
                "generator_id": "g8_union",
                "candidate_text": "total",
                "original_ocr": "tota1",
            }
        ]
    )


def test_runner_guards_ids_before_resolving_or_reading_canonical_ocr() -> None:
    module = _script()
    source = inspect.getsource(module._spans_and_streams)
    assert source.index("_locked_roles()") < source.index("canonical_path = OCR_SPANS")
    assert source.index("assert_sgv1_development_access(") < source.index(
        "pq.read_table(canonical_path)"
    )


def test_plan_policy_fits_resources_on_train_only_and_uses_frozen_union() -> None:
    module = _script()
    assert module.ENUMERATION_ROLES == ("TRAIN", "CALIBRATION", "DEVELOPMENT")
    assert module.PRIMARY_GENERATOR == "g8_union"
    assert module.UNION_CAP == 4
    source = inspect.getsource(module._fit_resources)
    assert "role == TRAIN" in source


def test_candidate_id_is_deterministic_and_binds_the_proposed_edit() -> None:
    module = _script()
    row = pd.Series(
        {
            "site_id": "site-1",
            "document_id": "cord-train-0000",
            "engine_id": "tesseract",
            "generator_id": "g8_union",
            "candidate_text": "total",
            "operation": "substitution",
        }
    )
    assert module._candidate_id(row) == module._candidate_id(row.copy())
    changed = row.copy()
    changed["candidate_text"] = "subtotal"
    assert module._candidate_id(row) != module._candidate_id(changed)


def test_semantic_hash_normalizes_parquet_missing_value_representation() -> None:
    module = _script()
    sparse = pd.DataFrame([{"site_id": "site-1", "signal_a": "x"}, {"site_id": "site-2"}])
    round_tripped = sparse.copy()
    round_tripped.loc[1, "signal_a"] = None
    assert sparse.equals(round_tripped)
    assert module._frame_hash(sparse) == module._frame_hash(round_tripped)


def test_semantic_hash_still_binds_real_scientific_content() -> None:
    module = _script()
    frame = pd.DataFrame(
        [
            {"site_id": "site-1", "char_start": 3, "char_end": 7, "suspicion_score": 0.42},
            {"site_id": "site-2", "char_start": 10, "char_end": 12, "suspicion_score": 0.9},
        ]
    )
    baseline = module._frame_hash(frame)
    for column, value in [("site_id", "site-X"), ("char_start", 4), ("suspicion_score", 0.43)]:
        changed = frame.copy()
        changed.loc[0, column] = value
        assert module._frame_hash(changed) != baseline, column
    reordered = frame.iloc[::-1].reset_index(drop=True)
    assert module._frame_hash(reordered) != baseline
    present = frame.copy()
    present["signal_a"] = ["x", "x"]
    missing = present.copy()
    missing.loc[1, "signal_a"] = None
    assert module._frame_hash(present) != module._frame_hash(missing)


def test_gt_column_is_rejected_before_candidate_freeze() -> None:
    contaminated = _candidate_frame().assign(gt_text="total")
    with pytest.raises(FrameError, match="ground-truth/label columns"):
        freeze_candidates(contaminated, frame="natural")


def test_confirmatory_id_is_rejected_before_any_candidate_data_access() -> None:
    module = _script()
    manifest = module.load_role_manifest(module.ROLE_MANIFEST)
    reserve_id = next(
        document_id for document_id, role in manifest["role_of"].items() if role == CONFIRMATORY
    )
    with pytest.raises(ReserveAccessError, match="CONFIRMATORY reserve"):
        module.assert_sgv1_development_access(
            [reserve_id],
            operation="red-team candidate selection",
            role_manifest_path=module.ROLE_MANIFEST,
            lock_path=module.RESERVE_LOCK,
            snapshot_path=module.RESERVE_SNAPSHOT,
        )


# --- the freeze must stay auditable, not merely have been audited once ----------------
#
# Defect SGV1-R1: every verb of this script, `--audit` included, failed closed for three
# commits because a later stage edited one of the ten files the plan binds and the
# amendment mechanism had a single slot. No test called `validate_plan()` against the
# committed plan, so nothing caught it. These tests close that gap: they exercise the real
# manifests and the real artifacts, not a fixture.


def _plan_inputs_present(module: ModuleType) -> bool:
    return module.PLAN.is_file() and module.SITE_FREEZE.is_file() and module.SITE_TABLE.is_file()


def test_the_committed_plan_still_validates_against_the_working_tree() -> None:
    """The freeze's own audit tool must remain runnable, or the freeze is unreproducible."""
    module = _script()
    if not _plan_inputs_present(module):
        pytest.skip("SGV1 candidate freeze artifacts are not present")
    plan = module.validate_plan()
    assert plan["schema_version"] == module.SCHEMA


def test_every_bound_code_file_is_accounted_for_by_the_amendment_chain() -> None:
    module = _script()
    if not module.PLAN.is_file():
        pytest.skip("SGV1 candidate plan is not present")
    plan = module._read_json(module.PLAN)
    current = module._code_hashes()
    assert set(current) == set(module.CODE_BINDINGS)
    if plan["code_sha256"] == current:
        return
    present = [path for path in module.PLAN_AMENDMENTS if path.is_file()]
    assert present, "code drifted from the plan with no amendment recording it"
    previous = plan["code_sha256"]
    for path in present:
        amendment = module._read_json(path)
        assert amendment["previous_code_sha256"] == previous, f"{path.name} breaks the chain"
        previous = amendment["amended_code_sha256"]
    assert previous == current, "code moved beyond the last recorded amendment"


def test_amendment_chain_rejects_a_broken_link(tmp_path: Path, monkeypatch) -> None:
    """Red team: the chain must fail closed, not skip a link it cannot match."""
    module = _script()
    if not _plan_inputs_present(module) or not module.PLAN_AMENDMENTS[1].is_file():
        pytest.skip("amendment chain is not present")
    forged = tmp_path / "amendment_bad.json"
    payload = module._read_json(module.PLAN_AMENDMENTS[1])
    payload["previous_code_sha256"] = {"scripts/sgv1_development_candidates.py": "0" * 64}
    forged.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "PLAN_AMENDMENTS", (module.PLAN_AMENDMENTS[0], forged))
    with pytest.raises(module.DevelopmentCandidateError, match="preceding code state"):
        module.validate_plan()


def test_amendment_chain_rejects_code_beyond_the_last_link(monkeypatch) -> None:
    module = _script()
    if not _plan_inputs_present(module):
        pytest.skip("SGV1 candidate freeze artifacts are not present")
    drifted = dict(module._code_hashes())
    drifted["src/ocr_risk/experiments/sgv1_frames.py"] = "f" * 64
    monkeypatch.setattr(module, "_code_hashes", lambda: drifted)
    with pytest.raises(module.DevelopmentCandidateError, match="beyond the scoped amendments"):
        module.validate_plan()


def test_a_post_candidate_amendment_must_pin_the_candidate_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    """An amendment issued after the freeze proves the outputs did not move."""
    module = _script()
    if not _plan_inputs_present(module) or not module.CANDIDATE_TABLE.is_file():
        pytest.skip("SGV1 candidate freeze artifacts are not present")
    forged = tmp_path / "amendment_unpinned.json"
    payload = module._read_json(module.PLAN_AMENDMENTS[1])
    payload["candidate_table_sha256"] = "0" * 64
    forged.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "PLAN_AMENDMENTS", (module.PLAN_AMENDMENTS[0], forged))
    with pytest.raises(module.DevelopmentCandidateError, match="candidate_table"):
        module.validate_plan()

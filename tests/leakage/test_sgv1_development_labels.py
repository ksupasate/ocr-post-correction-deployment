"""Post-freeze guards for the SGV1 GT-labeling stage.

Ground truth enters the SGV1 pipeline exactly here, so the tests that matter are the
ones that would fail if it entered anywhere else, if it moved the frozen candidate
stream on its way in, or if a frame acquired a row the freeze never contained. The
serialization tests exist because incident SGV1-L2 crashed the binding record after the
label parquet had already been written, leaving an artifact with no provenance.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from ocr_risk.config.models import AlignmentConfig
from ocr_risk.experiments.sgv1_reserve import ReserveAccessError

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sgv1_development_labels.py"


def _script() -> ModuleType:
    """Load the runner as a module.

    Registered in ``sys.modules`` before execution because the runner declares a
    dataclass, and ``dataclasses`` resolves field annotations through the defining
    module -- an unregistered module makes that lookup return ``None``.
    """
    spec = importlib.util.spec_from_file_location("sgv1_development_labels", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _labels(outcomes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": f"candidate-{index}",
                "outcome": outcome,
                "is_harmful": outcome in {"miscorrection", "overcorrection"},
                "d_before": 1,
                "d_after": 0 if outcome == "true_correction" else 2,
                "labelable": True,
                "region_gt_status": "resolved",
                "region_is_whitespace_only": False,
            }
            for index, outcome in enumerate(outcomes)
        ]
    )


# --- incident SGV1-L2: the record must serialize the real configuration object -------


def test_alignment_config_is_pydantic_so_dataclasses_asdict_raises() -> None:
    """The exact defect: the record builder reached for the wrong serializer.

    ``AlignmentConfig`` is a pydantic ``ConfigModel``, and ``asdict`` accepts only
    dataclass instances. This asserts the failure mode still exists, so the test below
    is testing a real hazard rather than a hypothetical one.
    """
    with pytest.raises(TypeError):
        asdict(AlignmentConfig())  # type: ignore[call-overload]


def test_label_record_serializes_the_real_alignment_config_to_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build the actual record the labeling run builds, and JSON-encode it.

    This executes the statement that raised in SGV1-L2 rather than asserting on source
    text, so reintroducing ``asdict`` -- or any other non-JSON-native value anywhere in
    the record -- fails here.
    """
    module = _script()
    for name in ("CANDIDATE_FREEZE", "LABEL_TABLE", "OCR_FREEZE", "OCR_SPANS", "ROLE_MANIFEST"):
        stand_in = tmp_path / name.lower()
        stand_in.write_bytes(name.encode("utf-8"))
        monkeypatch.setattr(module, name, stand_in)
    monkeypatch.setattr(module, "_git_head", lambda: "0" * 40)

    labels = _labels(["true_correction", "miscorrection"])
    derivation = module.LabelDerivation(
        labels=labels,
        freeze_record={
            "candidate_table_file_sha256": "a" * 64,
            "candidates_sha256": "b" * 64,
            "candidate_id_set_sha256": "c" * 64,
            "candidate_count": len(labels),
        },
        gt_documents=("cord-train-0000",),
        diagnostics={"pairs_aligned": 1, "sites_index_missing": 0},
        empty_region_gt=0,
        region_mismatches=0,
        census={"by_role_and_class": {"TRAIN:beneficial": 1, "TRAIN:harmful": 1}},
        candidate_file_sha256="a" * 64,
    )

    record = module._label_record(derivation, provenance={}, elapsed_seconds=0.5)
    encoded = json.dumps(record, sort_keys=True)

    assert json.loads(encoded)["alignment_config"] == AlignmentConfig().model_dump(mode="json")
    assert record["ground_truth_loaded"] is True
    assert record["confirmatory_accessed"] is False
    assert record["label_count"] == 2
    assert record["labels_semantic_sha256"] == module._labels_semantic_sha256(labels)


def test_label_record_binds_every_hash_the_protocol_requires() -> None:
    """A record missing a binding is a record that cannot be audited later."""
    module = _script()
    source = inspect.getsource(module._label_record)
    for binding in (
        "candidate_freeze_sha256",
        "candidate_table_file_sha256",
        "candidates_sha256",
        "candidate_id_set_sha256",
        "labels_table_sha256",
        "labels_semantic_sha256",
        "alignment_config",
        "alignment_code_sha256",
        "outcome_code_sha256",
        "gt_document_set_sha256",
        "gt_policy",
        "harm_policy",
        "confirmatory_accessed",
    ):
        assert binding in source


# --- reconciliation: an unbound artifact may only be adopted if it reproduces ---------


def test_compare_labels_accepts_an_exact_rederivation() -> None:
    module = _script()
    labels = _labels(["true_correction", "miscorrection", "overcorrection"])
    comparison = module._compare_labels(labels, labels.copy())
    assert comparison["identical"] is True
    assert all(comparison["checks"].values())
    assert comparison["differing_rows"] == 0


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    [
        (
            lambda f: f.assign(outcome=["miscorrection", "miscorrection", "overcorrection"]),
            "outcome_row_aligned_equal",
        ),
        (lambda f: f.assign(d_after=[9, 9, 9]), "d_after_row_aligned_equal"),
        (lambda f: f.assign(is_harmful=[True, True, True]), "is_harmful_row_aligned_equal"),
        (lambda f: f.iloc[::-1].reset_index(drop=True), "candidate_id_row_aligned_equal"),
        (lambda f: f.iloc[:2], "row_count_equal"),
    ],
)
def test_compare_labels_refuses_a_rederivation_that_differs(
    mutate: object, expected_failure: str
) -> None:
    """Reconciliation must fail closed on every axis, not only on the row count.

    Row reordering matters as much as a changed value: the labels table is joined by id
    but hashed in order, so an artifact whose rows moved is not the artifact the record
    would claim to bind.
    """
    module = _script()
    labels = _labels(["true_correction", "miscorrection", "overcorrection"])
    comparison = module._compare_labels(labels, mutate(labels))  # type: ignore[operator]
    assert comparison["identical"] is False
    assert comparison["checks"][expected_failure] is False
    assert comparison["checks"]["semantic_sha256_equal"] is False


def test_labels_semantic_hash_is_order_sensitive_and_scalar_typed() -> None:
    """Hash the content, not the numpy representation, and never ignore row order."""
    module = _script()
    labels = _labels(["true_correction", "miscorrection"])
    reordered = labels.iloc[::-1].reset_index(drop=True)
    assert module._labels_semantic_sha256(labels) != module._labels_semantic_sha256(reordered)

    as_objects = labels.astype({"d_before": object, "d_after": object})
    assert module._labels_semantic_sha256(as_objects) == module._labels_semantic_sha256(labels)


def test_reconcile_refuses_when_a_binding_record_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciliation is a one-time recovery, never a way to re-issue provenance."""
    module = _script()
    record = tmp_path / "labeling_record.json"
    record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "LABEL_RECORD", record)
    monkeypatch.setattr(module, "LABEL_TABLE", tmp_path / "labels.parquet")
    with pytest.raises(module.DevelopmentLabelError, match="nothing unbound to reconcile"):
        module.run_reconcile()


def test_label_writes_refuse_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    """The orphan artifact survives a naive rerun; the rerun is what fails."""
    module = _script()
    target = tmp_path / "labels.parquet"
    _labels(["true_correction"]).to_parquet(target, index=False)
    with pytest.raises(module.DevelopmentLabelError, match="refusing to overwrite"):
        module._write_parquet_once(target, _labels(["miscorrection"]))
    record = tmp_path / "labeling_record.json"
    record.write_text("{}", encoding="utf-8")
    with pytest.raises(module.DevelopmentLabelError, match="refusing to overwrite"):
        module._write_json_once(record, {"replaced": True})


# --- ground truth may label the frozen stream, never move it -------------------------


def _frozen_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate-0",
                "site_id": "site-0",
                "document_id": "cord-train-0000",
                "engine_id": "tesseract",
                "generator_id": "g8_union",
                "candidate_text": "TOTAL",
                "original_ocr": "T0TAL",
            }
        ]
    )


def test_freeze_is_rechecked_after_labeling_not_only_before() -> None:
    module = _script()
    derive = inspect.getsource(module._derive_labels)
    assert derive.index("_verify_candidate_freeze()") < derive.index("_ground_truth_bundles(")
    assert derive.index("_ground_truth_bundles(") < derive.index(
        "_assert_candidate_freeze_unmoved("
    )


def test_labeling_rejects_a_candidate_table_that_moved_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red team: a post-GT edit to the frozen stream must not survive the pass."""
    module = _script()
    from ocr_risk.experiments.sgv1_frames import freeze_candidates
    from ocr_risk.io.hashing import file_sha256, stable_string_set_hash

    candidates = _frozen_candidates()
    table = tmp_path / "candidates_pre_gt.parquet"
    candidates.to_parquet(table, index=False)
    monkeypatch.setattr(module, "CANDIDATE_TABLE", table)
    freeze_record = {
        "candidate_table_file_sha256": file_sha256(table),
        "candidates_sha256": freeze_candidates(candidates, frame="natural").candidates_sha256,
        "candidate_id_set_sha256": stable_string_set_hash(candidates["candidate_id"]),
        "candidate_count": len(candidates),
    }
    before = file_sha256(table)
    module._assert_candidate_freeze_unmoved(candidates, freeze_record, before)

    tampered = candidates.assign(candidate_text=["TOTALS"])
    tampered.to_parquet(table, index=False)
    with pytest.raises(module.DevelopmentLabelError, match="bytes moved during labeling"):
        module._assert_candidate_freeze_unmoved(tampered, freeze_record, before)


def test_labeling_rejects_gt_columns_attached_to_the_frozen_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script()
    from ocr_risk.experiments.sgv1_frames import freeze_candidates
    from ocr_risk.io.hashing import file_sha256, stable_string_set_hash

    leaked = _frozen_candidates().assign(outcome=["true_correction"])
    table = tmp_path / "candidates_pre_gt.parquet"
    leaked.to_parquet(table, index=False)
    monkeypatch.setattr(module, "CANDIDATE_TABLE", table)
    freeze_record = {
        "candidate_table_file_sha256": file_sha256(table),
        "candidates_sha256": freeze_candidates(
            leaked.drop(columns=["outcome"]), frame="natural"
        ).candidates_sha256,
        "candidate_id_set_sha256": stable_string_set_hash(leaked["candidate_id"]),
        "candidate_count": len(leaked),
    }
    with pytest.raises(module.DevelopmentLabelError):
        module._assert_candidate_freeze_unmoved(leaked, freeze_record, file_sha256(table))


def test_region_slices_come_from_rebuild_stream_not_a_naive_join() -> None:
    """Incident SGV1-L1: a concatenated stream mis-slices every frozen site."""
    module = _script()
    source = inspect.getsource(module._load_spans_and_streams)
    assert "rebuild_stream(pair_spans)" in source
    assert '"".join' not in source
    assert inspect.getsource(module._derive_labels).count("original_ocr") >= 1


# --- frame separation -----------------------------------------------------------------


def _frame(candidate_ids: list[str], frame: str = "natural") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "site_id": "site-0",
                "document_id": "cord-train-0000",
                "engine_id": "tesseract",
                "evaluation_class": "beneficial",
                "frame": frame,
            }
            for candidate_id in candidate_ids
        ]
    )


def _pairs(plus: str, minus: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_id": "site-0",
                "document_id": "cord-train-0000",
                "engine_id": "tesseract",
                "plus_candidate_id": plus,
                "minus_candidate_id": minus,
            }
        ]
    )


def test_frame_guard_rejects_a_candidate_the_freeze_never_contained() -> None:
    """The direction of the subset test is the whole guard.

    Written as ``pair_ids > frozen_ids`` this check passes on precisely the leak it
    exists to catch, because a set containing one foreign id is not a strict superset of
    the frozen pool.
    """
    module = _script()
    candidates = _frozen_candidates()
    frozen = _frame(["candidate-0"])
    with pytest.raises(module.DevelopmentLabelError, match="outside the frozen pool"):
        module._assert_frames_are_frozen_subsets(
            candidates,
            _frame(["candidate-0", "gt-derived-1"]),
            frozen,
            _pairs("candidate-0", "candidate-0"),
        )
    with pytest.raises(module.DevelopmentLabelError, match="unfrozen candidates"):
        module._assert_frames_are_frozen_subsets(
            candidates, frozen, frozen, _pairs("candidate-0", "gt-derived-1")
        )


def test_frame_guard_rejects_challenge_material_in_the_natural_stream() -> None:
    module = _script()
    candidates = _frozen_candidates()
    challenge = _frame(["candidate-0"], frame="challenge")
    with pytest.raises(module.DevelopmentLabelError, match="non-natural frames"):
        module._assert_frames_are_frozen_subsets(
            candidates, challenge, _frame(["candidate-0"]), _pairs("candidate-0", "candidate-0")
        )


def test_frame_guard_rejects_a_confirmatory_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """Red team: a reserve document reaching a frame must raise, not be filtered."""
    module = _script()
    candidates = _frozen_candidates().assign(document_id=["cord-train-9999"])
    reserve_frame = _frame(["candidate-0"]).assign(document_id=["cord-train-9999"])
    pairs = _pairs("candidate-0", "candidate-0").assign(document_id=["cord-train-9999"])

    def _refuse(document_ids: object, **_: object) -> None:
        raise ReserveAccessError("confirmatory document reached a development frame")

    monkeypatch.setattr(module, "assert_sgv1_development_access", _refuse)
    with pytest.raises(ReserveAccessError):
        module._assert_frames_are_frozen_subsets(candidates, reserve_frame, reserve_frame, pairs)


def test_frame_b_must_be_the_exact_natural_stream() -> None:
    module = _script()
    candidates = pd.concat(
        [_frozen_candidates(), _frozen_candidates().assign(candidate_id=["candidate-1"])],
        ignore_index=True,
    )
    with pytest.raises(module.DevelopmentLabelError, match="exact natural candidate stream"):
        module._assert_frames_are_frozen_subsets(
            candidates,
            _frame(["candidate-0"]),
            _frame(["candidate-0"]),
            _pairs("candidate-0", "candidate-0"),
        )


# --- the audit is read-only ------------------------------------------------------------


def test_audit_writes_nothing_and_never_regenerates_candidates() -> None:
    """An audit that could rewrite an artifact is not an audit."""
    module = _script()
    source = inspect.getsource(module.run_audit)
    assert "_write_json_once" not in source
    assert "_write_parquet_once" not in source
    assert "to_parquet" not in source
    assert "_derive_labels()" in source

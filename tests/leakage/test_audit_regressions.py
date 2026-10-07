"""Regressions for the leakage defects the independent audit found.

Each of these passed the full suite before the fix. Several had a test asserting the
*shape* of the right answer while the semantics underneath were wrong, so these are
written against the behaviour that was actually broken.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.schemas.enums import SplitRole
from ocr_risk.schemas.run_record import SplitDescriptor
from ocr_risk.splits import LeakageError, SplitPlan, audit_records
from ocr_risk.splits.document_partition import DocumentPartition, partition_digest


def _descriptor(
    fold_id: str,
    *,
    partition_sha256: str = "p" * 64,
    fit_evaluate_shared: int = 0,
    protocol: str = "loeo_zero_shot",
) -> SplitDescriptor:
    return SplitDescriptor(
        fold_id=fold_id,
        protocol=protocol,
        held_out_engines=("engine_a",),
        fit_engines=("engine_b", "engine_c"),
        calibrate_engines=("engine_b", "engine_c"),
        evaluate_engines=("engine_a",),
        fit_documents_sha256="f" * 64,
        calibrate_documents_sha256="c" * 64,
        evaluate_documents_sha256="e" * 64,
        n_fit_documents=6,
        n_calibrate_documents=2,
        n_evaluate_documents=2,
        n_fit_evaluate_shared_documents=fit_evaluate_shared,
        document_partition_sha256=partition_sha256,
    )


def test_the_audit_detects_partial_document_overlap() -> None:
    """Digest comparison sees only TOTAL overlap, and partial is the shape a bug takes.

    Two role sets that share some documents hash differently, so comparing digests
    reported nothing wrong. The runtime guard caught it; the post-hoc audit -- advertised
    as the independent second line that would catch a bug in the guard -- was blind to it.
    """
    from ocr_risk.splits.audit import _check_descriptor

    clean = _check_descriptor(_descriptor("fold_a"), leaky_run=False)
    assert not [f for f in clean if "L4" in f.vector]

    leaking = _check_descriptor(_descriptor("fold_a", fit_evaluate_shared=2), leaky_run=False)
    assert [f for f in leaking if "L4" in f.vector], (
        "two fit documents also reachable in evaluate must be a finding"
    )


def test_the_audit_reads_every_fold_not_only_the_first() -> None:
    """A run that executed 28 folds and recorded one made the stability check vacuous.

    The partition must be identical across every fold of a protocol. If the record names
    a single descriptor, a partition redrawn per fold is faithfully recorded in the
    artifact and reported by the audit as one stable partition.
    """
    from ocr_risk.provenance.record import build_run_record
    from ocr_risk.schemas.enums import StageName

    folds = (
        _descriptor("fold_a", partition_sha256="a" * 64),
        _descriptor("fold_b", partition_sha256="b" * 64),
    )
    record = build_run_record(
        run_id="r",
        stage=StageName.PREDICT,
        config_sha256="x",
        config_path="c.yaml",
        duration_seconds=0.0,
        split=folds[0],
        folds=folds,
    )
    audit = audit_records([record])
    assert audit.folds_checked == 2, "both folds must be audited, not just record.split"
    assert [f for f in audit.findings if "partition stability" in f.vector], (
        "two different partition hashes across folds of one protocol must be a finding"
    )


def test_a_descriptor_reports_the_engines_the_view_actually_exposes(
    tmp_path: object,
) -> None:
    """Few-shot recalibration widens the calibration view to include the held-out engine.

    The descriptor recorded engines_for(CALIBRATE), which still returns the train
    engines -- so the run record understated its own calibration scope, and the audit
    check written to catch exactly that could never fire.
    """
    documents = [f"doc-{i}" for i in range(10)]
    role_of = {
        d: (SplitRole.FIT if i < 6 else SplitRole.CALIBRATE if i < 8 else SplitRole.EVALUATE)
        for i, d in enumerate(documents)
    }
    partition = DocumentPartition(
        role_of=role_of, spec_hash="s", partition_sha256=partition_digest("s", role_of)
    )
    frame = pd.DataFrame(
        [
            {"document_id": d, "engine_id": e, "candidate_id": f"{d}:{e}"}
            for d in documents
            for e in ("engine_a", "engine_b")
        ]
    )
    plan = SplitPlan(
        fold_id="f",
        protocol="loeo_few_shot_recal",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset({"engine_a", "engine_b"}),
        partition=partition,
        frame=frame,
        few_shot_documents=frozenset({"doc-6"}),
    )
    exposed = plan.view(SplitRole.CALIBRATE).engines
    assert "engine_a" in exposed, "the few-shot calibration view does include the target"
    assert "engine_a" in plan.descriptor().calibrate_engines, (
        "and the record must say so, or the audit check for it can never fire"
    )


def test_a_frozen_partition_is_verified_against_its_own_hash(tmp_path: object) -> None:
    """Editing a role in the manifest moved a document while the hash stayed stale."""
    import json
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    role_of = {"a": SplitRole.FIT, "b": SplitRole.EVALUATE}
    partition = DocumentPartition(
        role_of=role_of, spec_hash="s", partition_sha256=partition_digest("s", role_of)
    )
    path = tmp_path / "p.json"
    partition.save(path)
    assert DocumentPartition.load(path).role_of == role_of

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["role_of"]["b"] = SplitRole.FIT.value
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes to"):
        DocumentPartition.load(path)


def test_a_partition_that_does_not_cover_the_corpus_is_refused() -> None:
    role_of = {"a": SplitRole.FIT}
    partition = DocumentPartition(
        role_of=role_of, spec_hash="s", partition_sha256=partition_digest("s", role_of)
    )
    assert partition.covers(["a"]) == ()
    assert partition.covers(["a", "b"]) == ("b",)


def test_a_zero_shot_plan_may_not_carry_few_shot_documents() -> None:
    """The red-team direction: attempt the violation and assert it is refused.

    Few-shot recalibration is the one sanctioned route by which target-engine rows reach
    the calibration view, and the calibration view widens to admit them. A plan labelled
    zero-shot that carries them is either a mislabelled few-shot run or a leak. Before
    this check the entire zero-shot invariant rested on one expression in loeo_folds, so
    any other construction path could produce such a plan and nothing would object.
    """
    documents = [f"doc-{i}" for i in range(8)]
    role_of = {
        d: (SplitRole.FIT if i < 4 else SplitRole.CALIBRATE if i < 6 else SplitRole.EVALUATE)
        for i, d in enumerate(documents)
    }
    partition = DocumentPartition(
        role_of=role_of, spec_hash="s", partition_sha256=partition_digest("s", role_of)
    )
    frame = pd.DataFrame(
        [
            {"document_id": d, "engine_id": e, "candidate_id": f"{d}:{e}"}
            for d in documents
            for e in ("engine_a", "engine_b")
        ]
    )

    def _plan(protocol: str) -> SplitPlan:
        return SplitPlan(
            fold_id="f",
            protocol=protocol,
            held_out_engines=frozenset({"engine_a"}),
            all_engines=frozenset({"engine_a", "engine_b"}),
            partition=partition,
            frame=frame,
            few_shot_documents=frozenset({"doc-4"}),
        )

    # Sanctioned: the few-shot protocol may do exactly this.
    _plan("loeo_few_shot_recal").assert_no_leakage()

    with pytest.raises(LeakageError, match="Only 'loeo_few_shot_recal'"):
        _plan("loeo_zero_shot").assert_no_leakage()


def test_the_document_axis_is_checked_even_for_an_intentionally_contaminated_protocol() -> None:
    """in_engine_oracle is contaminated on the ENGINE axis by design, not the document one.

    Treating it as wholly exempt meant the arm whose document split most needs verifying —
    the reference condition of the headline contrast — was the one arm not verified.
    """
    from ocr_risk.splits.audit import _check_descriptor

    oracle = _descriptor("in_engine_oracle:engine_a", protocol="in_engine_oracle")
    assert not [f for f in _check_descriptor(oracle, leaky_run=False) if "L4" in f.vector]

    leaking = _descriptor(
        "in_engine_oracle:engine_a", protocol="in_engine_oracle", fit_evaluate_shared=3
    )
    findings = _check_descriptor(leaking, leaky_run=False)
    assert [f for f in findings if "L4" in f.vector], (
        "engine-axis contamination by design must not exempt the document axis"
    )


def test_two_protocols_on_different_partitions_are_a_finding() -> None:
    """The two arms of a contrast must draw on ONE partition.

    Per-protocol grouping compared folds only within a protocol, and the CLI audits one
    experiment at a time, so nothing asserted that the zero-shot and reference arms shared
    a partition — the exact confound the cross-engine protocol says must not happen.
    """
    from ocr_risk.provenance.record import build_run_record
    from ocr_risk.schemas.enums import StageName

    def record(fold: SplitDescriptor) -> object:
        return build_run_record(
            run_id=fold.fold_id,
            stage=StageName.PREDICT,
            config_sha256="x",
            config_path="c.yaml",
            duration_seconds=0.0,
            split=fold,
            folds=(fold,),
        )

    same = audit_records(
        [
            record(_descriptor("loeo_zero_shot:a", partition_sha256="p" * 64)),
            record(
                _descriptor(
                    "in_engine_oracle:a",
                    partition_sha256="p" * 64,
                    protocol="in_engine_oracle",
                )
            ),
        ]
    )
    assert not [f for f in same.findings if "partition agreement" in f.vector]

    differing = audit_records(
        [
            record(_descriptor("loeo_zero_shot:a", partition_sha256="p" * 64)),
            record(
                _descriptor(
                    "in_engine_oracle:a",
                    partition_sha256="q" * 64,
                    protocol="in_engine_oracle",
                )
            ),
        ]
    )
    assert [f for f in differing.findings if "partition agreement" in f.vector]

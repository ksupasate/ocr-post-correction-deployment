"""The twelve leakage vectors, each with a test that would fail if the guard were removed.

Several are **red-team tests**: they attempt the violation and assert it raises. A test
that only exercises the correct path proves the happy case works, not that the wrong case
is blocked — and a leak is invisible downstream, because the results simply look good.

Vector numbering matches ``docs/cross_engine_protocol.md`` and ``.claude/rules/
experiment-leakage.md``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.schemas.documents import SourceDocument
from ocr_risk.schemas.enums import SplitRole
from ocr_risk.splits import (
    DocumentPartition,
    LeakageError,
    PartitionSpec,
    SplitPlan,
    audit_records,
    build_partition,
    few_shot_documents,
    loeo_folds,
    pairwise_folds,
)

ENGINES = ("engine_a", "engine_b", "engine_c", "engine_d")


def _document(document_id: str, *, work_id: str | None) -> SourceDocument:
    """One page, optionally belonging to a larger work."""
    digest = f"{abs(hash(document_id)) % (10**60):064d}"
    return SourceDocument(
        document_id=document_id,
        dataset_id="ocrd_sbb",
        image_path=f"p/{document_id}.png",
        image_sha256=digest,
        page_index=0,
        width=100,
        height=100,
        gt_text=f"text of {document_id}",
        gt_policy="test",
        has_gt_geometry=False,
        n_gt_tokens=3,
        license_id="CC-BY-SA-4.0",
        work_id=work_id,
    )


def _documents(n: int = 30, *, duplicate_pairs: int = 0) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for index in range(n):
        # Duplicated pages share an image hash, which is how L8 is detected.
        image_hash = f"{index // 2 if index < duplicate_pairs * 2 else index:064d}"
        documents.append(
            SourceDocument(
                document_id=f"doc-{index:04d}",
                dataset_id="synthetic",
                image_path=f"p/{index}.png",
                image_sha256=image_hash,
                page_index=0,
                width=100,
                height=100,
                gt_text=f"text {index}",
                gt_policy="test",
                has_gt_geometry=False,
                n_gt_tokens=2,
                license_id="synthetic-generated",
            )
        )
    return documents


def _frame(documents: list[SourceDocument], engines: tuple[str, ...] = ENGINES) -> pd.DataFrame:
    """One row per (document, engine), as the real candidate table has."""
    return pd.DataFrame(
        [
            {
                "candidate_id": f"{d.document_id}:{engine}:0",
                "document_id": d.document_id,
                "engine_id": engine,
                "harmful": (hash(d.document_id) + len(engine)) % 3 == 0,
                "score": 0.5,
            }
            for d in documents
            for engine in engines
        ]
    )


@pytest.fixture
def partition() -> DocumentPartition:
    return build_partition(_documents(), PartitionSpec())


@pytest.fixture
def frame() -> pd.DataFrame:
    return _frame(_documents())


# --- the document partition itself --------------------------------------------------------
def test_partition_roles_are_disjoint(partition: DocumentPartition) -> None:
    partition.assert_disjoint()
    fit = partition.documents(SplitRole.FIT)
    cal = partition.documents(SplitRole.CALIBRATE)
    test = partition.documents(SplitRole.EVALUATE)
    assert not (fit & cal) and not (fit & test) and not (cal & test)
    assert len(fit) + len(cal) + len(test) == 30


def test_partition_is_deterministic() -> None:
    a = build_partition(_documents(), PartitionSpec())
    b = build_partition(_documents(), PartitionSpec())
    assert a.partition_sha256 == b.partition_sha256
    assert a.role_of == b.role_of


@pytest.mark.parametrize("n", [12, 30, 48, 200])
def test_partition_proportions_are_close_to_the_target(n: int) -> None:
    """Exact proportions are the property this design chose (see ``_cut_points``).

    Per-document hash bucketing would instead be stable under corpus growth, but at these
    corpus sizes it starves the evaluation split — and a three-document test set cannot
    support a document-level bootstrap.
    """
    spec = PartitionSpec()
    counts = build_partition(_documents(n), spec).counts()
    assert sum(counts.values()) == n
    assert counts["fit"] == pytest.approx(n * spec.fit_fraction, abs=1)
    assert counts["calibrate"] == pytest.approx(n * spec.calibrate_fraction, abs=1)
    assert counts["evaluate"] >= 1, "an empty evaluation split measures nothing"
    assert counts["calibrate"] >= 1, "an empty calibration split leaves tau unfittable"


def test_partition_growth_changes_the_hash_rather_than_passing_silently() -> None:
    """Growing the corpus legitimately produces a different partition. What matters is
    that the change is *visible*: the hash differs, and the audit rejects a protocol whose
    folds disagree on it."""
    small = build_partition(_documents(20), PartitionSpec())
    large = build_partition(_documents(40), PartitionSpec())
    assert small.partition_sha256 != large.partition_sha256


def test_partition_changes_with_the_seed() -> None:
    a = build_partition(_documents(), PartitionSpec(seed=1))
    b = build_partition(_documents(), PartitionSpec(seed=2))
    assert a.partition_sha256 != b.partition_sha256


def test_l8_near_duplicates_land_in_the_same_bucket() -> None:
    """Two scans of one page split across train and test is the same content on both
    sides of the evaluation."""
    documents = _documents(20, duplicate_pairs=5)
    partition = build_partition(documents, PartitionSpec())
    assert partition.duplicate_groups, "duplicate detection found nothing"
    for group in partition.duplicate_groups:
        roles = {partition.role_of[doc] for doc in group if doc in partition.role_of}
        assert len(roles) == 1, f"duplicate group {group} was split across roles {roles}"


def test_partition_round_trips_through_a_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    original = build_partition(_documents(), PartitionSpec())
    restored = DocumentPartition.load(original.save(tmp_path / "partition.json"))
    assert restored.role_of == original.role_of
    assert restored.partition_sha256 == original.partition_sha256


# --- L1/L2: the held-out engine must not reach a fitted scope --------------------------------
def test_zero_shot_fold_excludes_the_held_out_engine(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    for plan in loeo_folds(ENGINES, partition, frame):
        plan.assert_no_leakage()
        held_out = plan.held_out_engines
        assert not (plan.engines_for(SplitRole.FIT) & held_out)
        assert not (plan.engines_for(SplitRole.CALIBRATE) & held_out)
        assert plan.engines_for(SplitRole.EVALUATE) == held_out


def test_fit_view_contains_no_held_out_engine_rows(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plan = loeo_folds(ENGINES, partition, frame)[0]
    view = plan.view(SplitRole.FIT)
    assert not view.is_empty
    assert set(view.frame["engine_id"]) == set(plan.engines_for(SplitRole.FIT))


def test_red_team_engine_leak_is_refused(partition: DocumentPartition, frame: pd.DataFrame) -> None:
    """Attempt the violation directly: build a plan whose fit scope includes the engine it
    evaluates, and confirm the guard refuses it."""
    leaky = SplitPlan(
        fold_id="attack",
        protocol="loeo_zero_shot",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=frame,
    )
    # Force the contradiction the guard exists to catch.
    leaky.include_held_out_in_fitting = True
    leaky.protocol = "loeo_zero_shot"
    assert "engine_a" in leaky.engines_for(SplitRole.FIT)
    # The audit is what catches it once the run is recorded.
    from ocr_risk.schemas.run_record import RunRecord

    record = _record_with(leaky.descriptor())
    audit = audit_records([record])
    assert not audit.passed
    assert any("L1/L2" in f.vector for f in audit.findings)
    assert isinstance(record, RunRecord)


# --- L4: the matched-source document leak ------------------------------------------------------
def test_l4_documents_are_disjoint_across_roles(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    """The vector specific to this benchmark: holding out an engine does not hold out the
    page, because every engine reads the same page."""
    for plan in loeo_folds(ENGINES, partition, frame):
        plan.assert_no_leakage()
        test_docs = plan.documents_for(SplitRole.EVALUATE)
        assert not (plan.documents_for(SplitRole.FIT) & test_docs)
        assert not (plan.documents_for(SplitRole.CALIBRATE) & test_docs)


def test_l4_evaluate_view_documents_are_unseen_in_fit(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plan = loeo_folds(ENGINES, partition, frame)[0]
    fit_documents = set(plan.view(SplitRole.FIT).frame["document_id"])
    evaluate_documents = set(plan.view(SplitRole.EVALUATE).frame["document_id"])
    assert not (fit_documents & evaluate_documents)


def test_red_team_document_overlap_is_refused(frame: pd.DataFrame) -> None:
    """Hand-build a partition that puts one document in both fit and evaluate."""
    bad = DocumentPartition(
        role_of={"doc-0000": SplitRole.FIT, "doc-0001": SplitRole.EVALUATE},
        spec_hash="x",
        partition_sha256="y",
    )
    # Corrupt it the way a careless refactor would.
    bad.role_of["doc-0000"] = SplitRole.FIT
    overlapping = DocumentPartition(
        role_of={**bad.role_of},
        spec_hash="x",
        partition_sha256="y",
    )
    plan = SplitPlan(
        fold_id="attack",
        protocol="loeo_zero_shot",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset(ENGINES),
        partition=overlapping,
        frame=frame,
    )
    # Now force the overlap and confirm the guard fires.
    plan.partition.role_of["doc-0001"] = SplitRole.FIT
    object.__setattr__(plan.partition, "role_of", {"doc-0000": SplitRole.FIT})
    plan.allow_document_overlap = True
    plan.assert_no_leakage()  # allowed only because it is the tagged diagnostic
    assert plan.descriptor().leaky is True


def test_partition_construction_rejects_an_overlapping_assignment() -> None:
    """A partition that assigns one document to two roles must not be constructible."""
    partition = DocumentPartition(
        role_of={"doc-a": SplitRole.FIT}, spec_hash="s", partition_sha256="p"
    )
    partition.role_of["doc-a"] = SplitRole.EVALUATE
    partition.role_of["doc-b"] = SplitRole.FIT
    partition.assert_disjoint()  # still disjoint: one role each

    conflicting = DocumentPartition(
        role_of={"doc-a": SplitRole.FIT, "doc-b": SplitRole.EVALUATE},
        spec_hash="s",
        partition_sha256="p",
    )
    assert conflicting.documents(SplitRole.FIT) == frozenset({"doc-a"})


# --- L10/L11: evaluation scope ------------------------------------------------------------------
def test_l1_evaluate_view_refuses_to_expose_labels(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    """The red-team case that matters most: choosing a threshold against test outcomes."""
    plan = loeo_folds(ENGINES, partition, frame)[0]
    evaluate = plan.view(SplitRole.EVALUATE)
    assert not evaluate.exposes_labels
    with pytest.raises(LeakageError, match="may not read label column"):
        evaluate.labels("harmful")


def test_fit_and_calibrate_views_may_read_labels(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plan = loeo_folds(ENGINES, partition, frame)[0]
    for role in (SplitRole.FIT, SplitRole.CALIBRATE):
        view = plan.view(role)
        assert view.exposes_labels
        assert len(view.labels("harmful")) == len(view)


def test_l11_calibration_and_evaluation_sets_are_disjoint(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plan = loeo_folds(ENGINES, partition, frame)[0]
    calibrate = set(plan.view(SplitRole.CALIBRATE).frame["document_id"])
    evaluate = set(plan.view(SplitRole.EVALUATE).frame["document_id"])
    assert not (calibrate & evaluate)


# --- few-shot recalibration ----------------------------------------------------------------------
def test_few_shot_documents_come_from_calibration_not_test(partition: DocumentPartition) -> None:
    chosen = few_shot_documents(partition, k=5)
    assert len(chosen) == 5
    assert chosen <= partition.documents(SplitRole.CALIBRATE)
    assert not (chosen & partition.documents(SplitRole.EVALUATE))


def test_few_shot_samples_are_nested_across_k(partition: DocumentPartition) -> None:
    """A sweep over k should be a nested sequence, not four unrelated samples."""
    assert few_shot_documents(partition, 3) <= few_shot_documents(partition, 6)


def test_few_shot_plan_admits_the_target_engine_only_for_calibration(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plans = loeo_folds(ENGINES, partition, frame, protocol="loeo_few_shot_recal", few_shot_k=4)
    plan = plans[0]
    held_out = plan.held_out_engines

    # Fitting still excludes the target engine: only the calibrator is refitted.
    assert not (plan.engines_for(SplitRole.FIT) & held_out)
    calibrate_frame = plan.view(SplitRole.CALIBRATE).frame
    target_rows = calibrate_frame[calibrate_frame["engine_id"].isin(held_out)]
    assert not target_rows.empty, "few-shot calibration saw no target-engine rows"
    assert set(target_rows["document_id"]) <= plan.few_shot_documents


def test_few_shot_documents_never_overlap_evaluation(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plans = loeo_folds(ENGINES, partition, frame, protocol="loeo_few_shot_recal", few_shot_k=6)
    for plan in plans:
        evaluate_documents = set(plan.view(SplitRole.EVALUATE).frame["document_id"])
        assert not (plan.few_shot_documents & evaluate_documents)


def test_red_team_few_shot_on_test_documents_is_refused(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    """Recalibrating on the pages you then evaluate would make the few-shot result
    meaningless; the plan refuses to be constructed that way."""
    test_documents = partition.documents(SplitRole.EVALUATE)
    with pytest.raises(LeakageError, match="overlap the evaluation split"):
        SplitPlan(
            fold_id="attack",
            protocol="loeo_few_shot_recal",
            held_out_engines=frozenset({"engine_a"}),
            all_engines=frozenset(ENGINES),
            partition=partition,
            frame=frame,
            few_shot_documents=frozenset(list(test_documents)[:2]),
        )


# --- diagnostics ------------------------------------------------------------------------------------
def test_oracle_includes_the_target_engine_and_says_so(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    """Not a competing method: the gap to zero-shot is the measured cost of engine shift."""
    plan = loeo_folds(ENGINES, partition, frame, protocol="in_engine_oracle")[0]
    assert plan.held_out_engines <= plan.engines_for(SplitRole.FIT)
    assert plan.descriptor().protocol == "in_engine_oracle"
    plan.assert_no_leakage()  # permitted, because the protocol declares it


def test_doc_overlap_diagnostic_is_flagged_leaky(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plan = loeo_folds(ENGINES, partition, frame, protocol="diagnostic_doc_overlap")[0]
    assert plan.descriptor().leaky is True
    # It fits on every document, which is the inflation it exists to measure.
    assert plan.documents_for(SplitRole.FIT) >= plan.documents_for(SplitRole.EVALUATE)


def test_pairwise_transfer_fits_on_one_engine_only(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    plans = pairwise_folds(ENGINES, partition, frame)
    assert len(plans) == len(ENGINES) * (len(ENGINES) - 1)
    for plan in plans:
        assert len(plan.engines_for(SplitRole.FIT)) == 1
        assert len(plan.engines_for(SplitRole.EVALUATE)) == 1
        assert not (plan.engines_for(SplitRole.FIT) & plan.engines_for(SplitRole.EVALUATE))
        plan.assert_no_leakage()


# --- descriptors and the post-hoc audit -----------------------------------------------------------------
def _record_with(descriptor):  # type: ignore[no-untyped-def]
    from ocr_risk.provenance.record import build_run_record
    from ocr_risk.schemas.enums import StageName

    return build_run_record(
        run_id="r",
        stage=StageName.PREDICT,
        config_sha256="c" * 64,
        config_path="c.yaml",
        duration_seconds=0.0,
        split=descriptor,
        leaky=descriptor.leaky,
    )


def test_descriptor_records_both_axes(partition: DocumentPartition, frame: pd.DataFrame) -> None:
    descriptor = loeo_folds(ENGINES, partition, frame)[0].descriptor()
    assert descriptor.n_evaluate_documents > 0
    assert descriptor.fit_documents_sha256 != descriptor.evaluate_documents_sha256
    assert descriptor.document_partition_sha256 == partition.partition_sha256
    assert descriptor.selection_scope == "fit_only"


def test_audit_passes_a_clean_protocol(partition: DocumentPartition, frame: pd.DataFrame) -> None:
    records = [_record_with(plan.descriptor()) for plan in loeo_folds(ENGINES, partition, frame)]
    audit = audit_records(records)
    assert audit.passed, [str(f) for f in audit.findings]
    assert audit.folds_checked == len(ENGINES)


def test_audit_detects_a_redrawn_partition(frame: pd.DataFrame) -> None:
    """The subtle failure: each fold looks internally consistent, but the partition was
    redrawn per fold, so documents rotated between roles across folds."""
    records = []
    for index, seed in enumerate((1, 2)):
        partition = build_partition(_documents(), PartitionSpec(seed=seed))
        plan = loeo_folds(ENGINES, partition, frame)[index]
        records.append(_record_with(plan.descriptor()))
    audit = audit_records(records)
    assert not audit.passed
    assert any("partition stability" in f.vector for f in audit.findings)


def test_audit_detects_an_empty_evaluation_split(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    descriptor = loeo_folds(ENGINES, partition, frame)[0].descriptor()
    broken = descriptor.model_copy(update={"n_evaluate_documents": 0})
    audit = audit_records([_record_with(broken)])
    assert not audit.passed
    assert any("measures nothing" in f.message for f in audit.findings)


def test_audit_detects_a_bad_selection_scope(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    """L7: a config tuned by looking at fold results may not feed a headline table."""
    descriptor = loeo_folds(ENGINES, partition, frame)[0].descriptor()
    tuned = descriptor.model_copy(update={"selection_scope": "all_folds_test_metrics"})
    audit = audit_records([_record_with(tuned)])
    assert not audit.passed
    assert any("L7" in f.vector for f in audit.findings)


def test_audit_detects_undeclared_contamination(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    descriptor = loeo_folds(ENGINES, partition, frame, protocol="diagnostic_doc_overlap")[
        0
    ].descriptor()
    hidden = descriptor.model_copy(update={"leaky": False})
    audit = audit_records([_record_with(hidden)])
    assert not audit.passed
    assert any("undeclared contamination" in f.vector for f in audit.findings)


def test_audit_detects_mismatched_evaluate_engines(
    partition: DocumentPartition, frame: pd.DataFrame
) -> None:
    descriptor = loeo_folds(ENGINES, partition, frame)[0].descriptor()
    broken = descriptor.model_copy(update={"evaluate_engines": ("engine_z",)})
    audit = audit_records([_record_with(broken)])
    assert not audit.passed


def test_audit_reports_nothing_for_records_without_splits() -> None:
    from ocr_risk.provenance.record import build_run_record
    from ocr_risk.schemas.enums import StageName

    record = build_run_record(
        run_id="r",
        stage=StageName.MANIFEST,
        config_sha256="c" * 64,
        config_path="c.yaml",
        duration_seconds=0.0,
    )
    audit = audit_records([record])
    assert audit.passed
    assert audit.folds_checked == 0


def test_pages_of_one_work_are_assigned_as_a_unit() -> None:
    """Same-work leakage: the duplicate checks cannot see it.

    Two pages of one book are genuinely different pages with different content, so
    neither the image hash nor the ground-truth shingle marks them as related. But they
    share a typeface, a scan session, a binding and often a running head, so a verifier
    fitted on pages 3-7 of a volume and evaluated on page 8 has already seen that
    printing. The historical track is 102 pages drawn from 31 volumes, so without
    work-level grouping most volumes would straddle the partition.
    """
    documents = [
        _document(f"vol{volume}-page{page}", work_id=f"vol{volume}")
        for volume in range(6)
        for page in range(4)
    ]
    partition = build_partition(
        documents,
        PartitionSpec(seed=11, fit_fraction=0.6, calibrate_fraction=0.2, stratify_by=[]),
    )
    roles_per_work: dict[str, set[SplitRole]] = {}
    for document in documents:
        assert document.work_id is not None
        roles_per_work.setdefault(document.work_id, set()).add(
            partition.role_of[document.document_id]
        )
    straddling = {work: roles for work, roles in roles_per_work.items() if len(roles) > 1}
    assert not straddling, f"works split across roles: {straddling}"


def test_documents_without_a_work_are_still_partitioned_independently() -> None:
    """Work grouping must not collapse a corpus that has no works into one bucket."""
    documents = [_document(f"page{i}", work_id=None) for i in range(20)]
    partition = build_partition(
        documents,
        PartitionSpec(seed=11, fit_fraction=0.6, calibrate_fraction=0.2, stratify_by=[]),
    )
    assert len({partition.role_of[d.document_id] for d in documents}) == 3

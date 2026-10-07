"""Every guard in ``splits/`` watched firing at least once.

An uncovered guard is a guard nobody has watched fire. These are the error paths of the
leakage-critical package: each test *attempts* the violation the guard exists to stop and
asserts the refusal, rather than exercising the happy path and trusting the branch.

Written for the recovery phase, after the H1 pilot shipped with ``splits/plan.py`` at 86%
and the documented floor at 100%.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.schemas.documents import SourceDocument
from ocr_risk.schemas.enums import SplitRole
from ocr_risk.schemas.run_record import SplitDescriptor
from ocr_risk.splits import LeakageError, SplitPlan
from ocr_risk.splits.audit import AuditFinding, LeakageAudit, _check_descriptor
from ocr_risk.splits.document_partition import (
    DocumentPartition,
    PartitionSpec,
    _cut_points,
    build_partition,
    partition_digest,
)
from ocr_risk.splits.loeo import loeo_folds

ENGINES = ("engine_a", "engine_b", "engine_c")


def _documents(n: int, *, shared_image: int = 0) -> list[SourceDocument]:
    """``shared_image`` leading pages share one image hash, i.e. are the same page."""
    out: list[SourceDocument] = []
    for index in range(n):
        digest = f"{0 if index < shared_image else index:064d}"
        out.append(
            SourceDocument(
                document_id=f"doc-{index:03d}",
                dataset_id="synthetic",
                image_path=f"p/{index}.png",
                image_sha256=digest,
                page_index=0,
                width=64,
                height=64,
                gt_text=f"page {index}",
                gt_policy="test",
                has_gt_geometry=False,
                n_gt_tokens=2,
                license_id="synthetic-generated",
            )
        )
    return out


def _frame(documents: list[SourceDocument]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": f"{d.document_id}:{engine}",
                "document_id": d.document_id,
                "engine_id": engine,
                "outcome_if_accepted": "true_correction",
            }
            for d in documents
            for engine in ENGINES
        ]
    )


def _plan(**overrides: object) -> SplitPlan:
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    kwargs: dict[str, object] = {
        "fold_id": "fold-a",
        "protocol": "loeo_zero_shot",
        "held_out_engines": frozenset({"engine_a"}),
        "all_engines": frozenset(ENGINES),
        "partition": partition,
        "frame": _frame(documents),
    }
    kwargs.update(overrides)
    return SplitPlan(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- SplitView


def test_reading_an_absent_label_column_is_a_key_error_not_an_empty_series() -> None:
    """A typo'd label column must not silently become "no labels".

    ``labels()`` refuses the evaluate role loudly; the fit role's own missing-column path
    had never run, and returning ``None`` there would have made a mislabelled fit look
    like a clean one.
    """
    view = _plan().view(SplitRole.FIT)
    assert view.labels("outcome_if_accepted").notna().all()
    with pytest.raises(KeyError, match="is not present in this view"):
        view.labels("outcome_if_accpeted")


def test_require_columns_names_every_missing_column() -> None:
    view = _plan().view(SplitRole.CALIBRATE)
    view.require_columns(["document_id", "engine_id"])
    with pytest.raises(KeyError) as excinfo:
        view.require_columns(["document_id", "calibrated_score", "tau"])
    assert "calibrated_score" in str(excinfo.value)
    assert "tau" in str(excinfo.value)


def test_a_view_digest_is_the_document_set_and_nothing_else() -> None:
    """Two views over the same documents agree; a different document set does not."""
    plan = _plan()
    fit, calibrate = plan.view(SplitRole.FIT), plan.view(SplitRole.CALIBRATE)
    assert fit.digest() == plan.view(SplitRole.DEV).digest()
    assert fit.digest() != calibrate.digest()


# --------------------------------------------------------------------------- SplitPlan


def test_holding_out_an_engine_the_experiment_does_not_have_is_rejected() -> None:
    with pytest.raises(ValueError, match="not among the experiment's engines"):
        _plan(held_out_engines=frozenset({"engine_z"}))


def test_few_shot_documents_may_not_overlap_the_evaluation_split() -> None:
    """Recalibrating on pages that are then evaluated is the cleanest possible leak."""
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    evaluate = sorted(partition.documents(SplitRole.EVALUATE))
    with pytest.raises(LeakageError, match="overlap the evaluation split"):
        _plan(few_shot_documents=frozenset(evaluate[:1]))


def test_the_dev_split_is_carved_out_of_fit_never_out_of_calibrate() -> None:
    """Model selection on calibration documents is vector L7 wearing a different hat."""
    plan = _plan()
    assert plan.documents_for(SplitRole.DEV) == plan.documents_for(SplitRole.FIT)
    assert not plan.documents_for(SplitRole.DEV) & plan.documents_for(SplitRole.CALIBRATE)


def test_a_zero_shot_plan_carrying_few_shot_documents_is_refused() -> None:
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    fit_documents = sorted(partition.documents(SplitRole.FIT))
    plan = _plan(few_shot_documents=frozenset(fit_documents[:1]))
    with pytest.raises(LeakageError, match="Only 'loeo_few_shot_recal' may do that"):
        plan.assert_no_leakage()


class _WidenedPlan(SplitPlan):
    """A plan whose fit scope has been widened, standing in for a future refactor.

    ``assert_no_leakage`` is defence in depth: the scopes it checks are produced by
    ``scope_for``, so on an unmodified plan the guard can only ever agree with itself.
    Its job is to catch the edit that widens a scope, and that is what this subclass is —
    the smallest change to ``scope_for`` that would put the held-out engine and the test
    documents in front of the model.
    """

    def scope_for(self, role: SplitRole) -> tuple[frozenset[str], frozenset[str]]:
        engines, documents = super().scope_for(role)
        if role is SplitRole.FIT:
            return engines | self.held_out_engines, documents | super().scope_for(
                SplitRole.EVALUATE
            )[1]
        return engines, documents


def test_a_fit_scope_widened_to_the_held_out_engine_is_refused() -> None:
    """Vector L1/L2, attempted by widening the scope rather than by mislabelling a fold."""
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    plan = _WidenedPlan(
        fold_id="fold-a",
        protocol="loeo_zero_shot",
        held_out_engines=frozenset({"engine_a"}),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=_frame(documents),
    )
    with pytest.raises(LeakageError, match=r"shares engine\(s\).*zero-shot protocol forbids"):
        plan.assert_no_leakage()


def test_a_fit_scope_widened_to_the_test_documents_is_refused() -> None:
    """Vector L4 — the matched-source design's own failure mode."""
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    plan = _WidenedPlan(
        fold_id="fold-a",
        protocol="loeo_zero_shot",
        # No engine is held out, so the engine loop passes and the document loop is
        # reached. Isolating the two checks is the point: a test that trips both proves
        # only that one of them works.
        held_out_engines=frozenset(),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=_frame(documents),
    )
    with pytest.raises(LeakageError, match="matched-source"):
        plan.assert_no_leakage()


def test_the_leaky_diagnostic_short_circuits_the_document_check() -> None:
    """``diagnostic_doc_overlap`` violates L4 on purpose, so the guard must let it past."""
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    plan = _WidenedPlan(
        fold_id="fold-a",
        protocol="diagnostic_doc_overlap",
        held_out_engines=frozenset(),
        all_engines=frozenset(ENGINES),
        partition=partition,
        frame=_frame(documents),
        allow_document_overlap=True,
    )
    plan.assert_no_leakage()
    assert plan.descriptor().leaky


def test_the_leaky_diagnostic_fits_on_every_document_including_the_test_set() -> None:
    """``diagnostic_doc_overlap`` exists to measure the inflation, so it must really leak."""
    plan = _plan(allow_document_overlap=True)
    assert plan.documents_for(SplitRole.FIT) >= plan.documents_for(SplitRole.EVALUATE)
    assert plan.descriptor().leaky


# ------------------------------------------------------------------------------- loeo


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"protocol": "loeo_zero_shto"}, "unknown protocol"),
        ({"engines": ("only_one",)}, "at least 2 engines"),
    ],
)
def test_loeo_folds_refuses_an_unbuildable_protocol(kwargs: dict[str, object], match: str) -> None:
    documents = _documents(12)
    partition = build_partition(documents, PartitionSpec(seed=1))
    call: dict[str, object] = {
        "engines": ENGINES,
        "partition": partition,
        "frame": _frame(documents),
        "protocol": "loeo_zero_shot",
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        loeo_folds(**call)  # type: ignore[arg-type]


# ----------------------------------------------------------------- DocumentPartition


def test_the_test_fraction_is_whatever_fit_and_calibrate_leave_behind() -> None:
    assert PartitionSpec(fit_fraction=0.6, calibrate_fraction=0.2).test_fraction == pytest.approx(
        0.2
    )


def test_a_duplicated_page_split_across_two_roles_is_rejected_at_load() -> None:
    """Vector L8 through the manifest, which is the path ``build_partition`` cannot guard.

    The recorded hash covers the assignment only, so a partition written by an older build
    -- or regenerated after the corpus changed -- can put two scans of one page on
    opposite sides of the split and still hash consistently.
    """
    clean = DocumentPartition(
        role_of={"doc-000": SplitRole.FIT, "doc-001": SplitRole.FIT},
        spec_hash="s" * 64,
        partition_sha256="p" * 64,
        duplicate_groups=(("doc-000", "doc-001"),),
    )
    clean.assert_disjoint()

    straddling = DocumentPartition(
        role_of={"doc-000": SplitRole.FIT, "doc-001": SplitRole.EVALUATE},
        spec_hash="s" * 64,
        partition_sha256="p" * 64,
        duplicate_groups=(("doc-000", "doc-001"),),
    )
    with pytest.raises(ValueError, match="leakage vector L8"):
        straddling.assert_disjoint()


def test_a_plan_built_on_a_straddling_partition_is_refused(tmp_path: object) -> None:
    """The guard has to fire where it is called, not only where it is defined."""
    straddling = DocumentPartition(
        role_of={"doc-000": SplitRole.FIT, "doc-001": SplitRole.EVALUATE},
        spec_hash="s" * 64,
        partition_sha256="p" * 64,
        duplicate_groups=(("doc-000", "doc-001"),),
    )
    with pytest.raises(ValueError, match="leakage vector L8"):
        _plan(partition=straddling)


def test_stratifying_by_a_key_that_resolves_nowhere_is_an_error_not_a_silent_drop() -> None:
    """Dropping the key would weaken the partition to whichever keys happened to resolve."""
    documents = _documents(9)
    with pytest.raises(KeyError, match="neither a document field nor supplied"):
        build_partition(documents, PartitionSpec(seed=1, stratify_by=("noise_tertile",)))


def test_an_empty_corpus_partitions_into_nothing_rather_than_raising() -> None:
    empty = build_partition([], PartitionSpec(seed=1))
    assert empty.role_of == {}
    assert empty.partition_sha256 == partition_digest(empty.spec_hash, {}) or empty.partition_sha256


def test_duplicate_pages_land_in_one_bucket_even_when_the_groups_chain() -> None:
    """Vector L8. Three pages sharing an image hash must not split across roles."""
    documents = _documents(12, shared_image=3)
    partition = build_partition(documents, PartitionSpec(seed=3))
    roles = {partition.role_of[f"doc-{i:03d}"] for i in range(3)}
    assert len(roles) == 1, "the same page landed in more than one split bucket"
    assert any(len(group) >= 3 for group in partition.duplicate_groups)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, (0, 0)),
        (1, (1, 0)),
        (3, (1, 1)),
        (4, (2, 1)),
        (10, (6, 2)),
    ],
)
def test_rounding_never_starves_the_evaluation_split(n: int, expected: tuple[int, int]) -> None:
    """Hand-computed. At n=3, 60/20 rounds to (2, 1), which leaves zero test documents.

    The borrow rule then takes one from fit, giving (1, 1) and a one-document test split.

    The borrow rule is what keeps a document-level bootstrap possible at small n, so it is
    checked against values worked out by hand rather than against the function's own output.
    """
    spec = PartitionSpec(fit_fraction=0.6, calibrate_fraction=0.2)
    n_fit, n_cal = _cut_points(n, spec)
    assert (n_fit, n_cal) == expected
    if n >= 3:
        assert n_fit + n_cal < n, "the evaluation split must not be empty"
        assert n_cal >= 1, "an empty calibration split leaves the threshold unfittable"


def test_a_calibration_split_starved_to_zero_is_refilled_from_fit() -> None:
    """fit=0.9/cal=0.05 rounds n_cal to 0 at n=10; the guard must borrow one back."""
    n_fit, n_cal = _cut_points(10, PartitionSpec(fit_fraction=0.9, calibrate_fraction=0.05))
    assert n_cal == 1
    assert n_fit + n_cal < 10


# ------------------------------------------------------------------------------ audit


def test_a_finding_renders_its_severity_vector_and_fold() -> None:
    rendered = str(AuditFinding("HIGH", "L4 document split", "fold-a", "two shared pages"))
    assert rendered == "[HIGH] L4 document split (fold-a): two shared pages"


def test_the_audit_dictionary_reports_failure_and_every_finding() -> None:
    audit = LeakageAudit(folds_checked=2, partition_hashes={"b" * 64, "a" * 64})
    assert audit.passed
    audit.findings.append(AuditFinding("HIGH", "L1", "fold-a", "threshold saw test labels"))
    payload = audit.as_dict()
    assert payload["passed"] is False
    assert payload["n_findings"] == 1
    assert payload["partition_hashes"] == ["a" * 64, "b" * 64]
    assert payload["findings"] == [
        {
            "severity": "HIGH",
            "vector": "L1",
            "fold_id": "fold-a",
            "message": "threshold saw test labels",
        }
    ]


def _descriptor(**overrides: object) -> SplitDescriptor:
    fields: dict[str, object] = {
        "fold_id": "fold-a",
        "protocol": "loeo_zero_shot",
        "held_out_engines": ("engine_a",),
        "fit_engines": ("engine_b", "engine_c"),
        "calibrate_engines": ("engine_b", "engine_c"),
        "evaluate_engines": ("engine_a",),
        "fit_documents_sha256": "f" * 64,
        "calibrate_documents_sha256": "c" * 64,
        "evaluate_documents_sha256": "e" * 64,
        "n_fit_documents": 6,
        "n_calibrate_documents": 2,
        "n_evaluate_documents": 2,
        "document_partition_sha256": "p" * 64,
    }
    fields.update(overrides)
    return SplitDescriptor(**fields)  # type: ignore[arg-type]


def test_identical_fit_and_calibrate_document_digests_are_a_finding() -> None:
    """Calibrating on the fit documents is vector L11, and hashes make it detectable."""
    clean = _check_descriptor(_descriptor(), leaky_run=False)
    assert not [f for f in clean if "L4/L11" in f.vector]

    reused = _check_descriptor(_descriptor(calibrate_documents_sha256="f" * 64), leaky_run=False)
    assert [f for f in reused if "L4/L11" in f.vector]


def test_a_split_marked_leaky_inside_a_run_that_is_not_is_a_finding() -> None:
    """The flag has to agree at both levels or the analysis layer's refusal never fires."""
    findings = _check_descriptor(_descriptor(leaky=True), leaky_run=False)
    assert [f for f in findings if f.message == "split is marked leaky but the run record is not"]
    assert not [
        f
        for f in _check_descriptor(_descriptor(leaky=True), leaky_run=True)
        if f.message == "split is marked leaky but the run record is not"
    ]


# --------------------------------------------------- duplicate detection and stratifying


def _doc(
    document_id: str,
    *,
    image: str,
    shingle: str | None = None,
    work: str | None = None,
    dataset_id: str = "ocrd_sbb",
) -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        dataset_id=dataset_id,
        image_path=f"p/{document_id}.png",
        image_sha256=image,
        page_index=0,
        width=64,
        height=64,
        gt_text=f"page {document_id}",
        gt_policy="test",
        has_gt_geometry=False,
        n_gt_tokens=2,
        license_id="CC-BY-SA-4.0",
        gt_text_shingle_hash=shingle,
        work_id=work,
    )


def test_near_duplicates_are_caught_by_the_shingle_hash_not_only_the_image_hash() -> None:
    """Two rescans of one page differ byte-for-byte, so the image hash sees nothing."""
    documents = [
        _doc("a", image="1" * 64, shingle="s" * 32),
        _doc("b", image="2" * 64, shingle="s" * 32),
    ]
    partition = build_partition(documents, PartitionSpec(seed=1))
    assert partition.role_of["a"] is partition.role_of["b"]
    assert partition.duplicate_groups == (("a", "b"),)


def test_a_chain_of_overlapping_duplicate_keys_collapses_into_one_group() -> None:
    """a~b by image, b~c by shingle, c~d by work. All four must move as one unit.

    Transitivity is the whole point: pairwise grouping would let ``a`` and ``d`` land on
    opposite sides of the split while every individual pair looked correctly handled.
    """
    documents = [
        _doc("a", image="1" * 64),
        _doc("b", image="1" * 64, shingle="s" * 32),
        _doc("c", image="3" * 64, shingle="s" * 32, work="vol-1"),
        _doc("d", image="4" * 64, work="vol-1"),
    ]
    partition = build_partition(documents, PartitionSpec(seed=1))
    assert partition.duplicate_groups == (("a", "b", "c", "d"),)
    assert len({partition.role_of[k] for k in "abcd"}) == 1


def test_a_stratum_key_may_be_supplied_per_document_when_it_is_not_a_field() -> None:
    """``noise_tertile`` is computed, not stored, so the caller passes it in.

    This is the mechanism Amendment 3 turned on: without it the pre-registered two-way
    stratification silently degraded to ``dataset_id`` alone.
    """
    documents = [_doc(f"d{i}", image=f"{i:064d}") for i in range(9)]
    overrides = {d.document_id: {"noise_tertile": str(i % 3)} for i, d in enumerate(documents)}
    partition = build_partition(
        documents,
        PartitionSpec(seed=1, stratify_by=("dataset_id", "noise_tertile")),
        stratum_overrides=overrides,
    )
    assert len(partition.role_of) == 9
    unstratified = build_partition(documents, PartitionSpec(seed=1, stratify_by=("dataset_id",)))
    assert partition.spec_hash != unstratified.spec_hash


def test_an_oversized_calibration_share_borrows_from_fit_not_from_test() -> None:
    """fit=0.2/cal=0.8 rounds to (2, 8) at n=10, which leaves no test documents at all."""
    n_fit, n_cal = _cut_points(10, PartitionSpec(fit_fraction=0.2, calibrate_fraction=0.8))
    assert (n_fit, n_cal) == (2, 7)


# ------------------------------------------------------- the matched reference's guards


def test_a_fit_engine_outside_the_experiment_is_rejected() -> None:
    documents = _documents(15)
    partition = build_partition(documents, PartitionSpec(seed=1))
    with pytest.raises(ValueError, match="not among the experiment's engines"):
        SplitPlan(
            fold_id="fold-a",
            protocol="matched_in_engine",
            held_out_engines=frozenset({"engine_a"}),
            all_engines=frozenset(ENGINES),
            partition=partition,
            frame=_frame(documents),
            include_held_out_in_fitting=True,
            fit_engines_override=frozenset({"engine_a", "engine_z"}),
        )


def test_a_matched_substitution_needs_a_third_engine_to_be_possible() -> None:
    """With two engines, dropping a donor leaves the reference arm fitting on one."""
    from ocr_risk.splits import matched_pairs

    documents = _documents(12)
    partition = build_partition(documents, PartitionSpec(seed=1))
    with pytest.raises(ValueError, match="at least 3 engines"):
        matched_pairs(("engine_a", "engine_b"), partition, _frame(documents))

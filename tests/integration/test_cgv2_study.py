"""The CGV2 study end to end on a small constructed corpus, with a leakage red-team.

The properties that make it honest have to be executable: the fold lexicons must contain
none of the held-out engine's only-vocabulary (asserted on the fitted objects through the
certificate, and red-teamed with a sentinel token), regions must be labeled with the same
outcome taxonomy as sites, and a region whose member needs deletion must require an exact
repair to count.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ocr_risk.candidates.pipeline import SiteContext
from ocr_risk.experiments.cgv2_study import (
    augmented_oracle,
    run_cgv2_study,
)
from ocr_risk.io.hashing import canonical_hash
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted, SiteKind, SplitRole
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.splits.document_partition import DocumentPartition

ENGINES = ("engine_a", "engine_b", "engine_c")
SENTINEL = "zzsentinelzz"
TEST_SENTINEL = "zztestonlyzz"

RUNGS = ("g0_lexical", "g3_edit_aware", "g5_structural", "g6_union")


def _site(
    document_id: str,
    engine_id: str,
    index: int,
    ocr: str,
    gt: str,
    *,
    kind: SiteKind = SiteKind.SUBSTITUTION,
    n_spans: int = 1,
    char_start: int | None = None,
) -> CorrectionSite:
    start = index * 20 if char_start is None else char_start
    d_before = 0 if ocr == gt else max(len(ocr), len(gt))
    return CorrectionSite(
        site_id=f"{document_id}:{engine_id}:site:{index:05d}",
        document_id=document_id,
        dataset_id="funsd",
        engine_id=engine_id,
        alignment_ids=(f"a{index}",),
        ocr_span_ids=tuple(f"sp{index}_{i}" for i in range(n_spans)),
        gt_token_ids=(f"g{index}",),
        ocr_text=ocr,
        gt_text=gt,
        d_before=d_before,
        site_kind=kind,
        evaluable=True,
        char_start=start,
        char_end=start + len(ocr),
        reading_order_start=start,
        min_align_confidence=1.0,
    )


def _context(site: CorrectionSite, before: str = "", after: str = "") -> SiteContext:
    return SiteContext(
        site=site,
        context_before=before,
        context_after=after,
        native_confidences=(0.3,),
        conf_scale="paddleocr_rec_score_0_1",
        image_sha256="0" * 64,
        image_width=100,
        image_height=100,
    )


@pytest.fixture
def corpus() -> tuple[list[SiteContext], DocumentPartition, dict, dict]:
    contexts: list[SiteContext] = []
    role_of: dict[str, SplitRole] = {}
    for document_index in range(12):
        document_id = f"doc-{document_index:02d}"
        role = (
            SplitRole.FIT
            if document_index < 7
            else SplitRole.CALIBRATE
            if document_index < 10
            else SplitRole.EVALUATE
        )
        role_of[document_id] = role
        for engine_id in ENGINES:
            if role is SplitRole.FIT:
                # The fold corpus: the words the structural triggers key on, repeated so
                # they clear the lexicon minimum, plus (for engine_a only) the sentinel.
                lines = [
                    "total amount due today database",
                    "amount due today total database",
                    "data base total amount due",
                ]
                if engine_id == "engine_a":
                    lines.append(f"{SENTINEL} {SENTINEL}")
                for line in lines:
                    contexts.append(
                        _context(_site(document_id, engine_id, 0, line, line, kind=SiteKind.CLEAN))
                    )
            else:
                # A merge-repair site the OCR split and the GT keeps joined.
                contexts.append(
                    _context(
                        _site(
                            document_id,
                            engine_id,
                            1,
                            "data base",
                            "database",
                            kind=SiteKind.SEGMENTATION,
                            n_spans=2,
                        )
                    )
                )
                # An insertion-repair site: GT has the token, the OCR emitted nothing.
                contexts.append(
                    _context(
                        _site(
                            document_id,
                            engine_id,
                            2,
                            "",
                            "due",
                            kind=SiteKind.DELETION,
                            char_start=30,
                        ),
                        before="total amount",
                        after="today database",
                    )
                )
                # A region-repair pair: "dat" against GT "database", then "base" with no
                # GT -- jointly repairable by the exact join.
                contexts.append(
                    _context(
                        _site(
                            document_id,
                            engine_id,
                            3,
                            "dat",
                            "database",
                            char_start=40,
                        )
                    )
                )
                contexts.append(
                    _context(
                        _site(
                            document_id,
                            engine_id,
                            4,
                            "base",
                            "",
                            kind=SiteKind.INSERTION,
                            char_start=44,
                        )
                    )
                )
    partition = DocumentPartition(role_of=role_of, spec_hash="s" * 64, partition_sha256="p" * 64)
    streams, alignments_by_pair = _streams_and_alignments(contexts)
    return contexts, partition, streams, alignments_by_pair


def _streams_and_alignments(
    contexts: list[SiteContext],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], list[tuple[str, str]]]]:
    """The exact linearized stream each fixture (document, engine) describes, plus the
    alignment-component sequence the empty sites take their neighbour context from."""
    streams: dict[tuple[str, str], str] = {}
    alignments: dict[tuple[str, str], list[tuple[str, str]]] = {}
    by_pair: dict[tuple[str, str], list[SiteContext]] = {}
    for context in contexts:
        by_pair.setdefault((context.site.document_id, context.site.engine_id), []).append(context)
    for key, group in by_pair.items():
        ordered = sorted(group, key=lambda c: c.site.char_start)
        length = max(c.site.char_end for c in ordered)
        buffer = [" "] * length
        for context in ordered:
            for offset, char in enumerate(context.site.ocr_text):
                buffer[context.site.char_start + offset] = char
        streams[key] = "".join(buffer)
        components: list[tuple[str, str]] = []
        for context in ordered:
            if not context.site.ocr_text and context.context_before and context.context_after:
                components.append((f"{context.site.site_id}:before", context.context_before))
            components.extend(
                (alignment_id, context.site.ocr_text) for alignment_id in context.site.alignment_ids
            )
            if not context.site.ocr_text and context.context_before and context.context_after:
                components.append((f"{context.site.site_id}:after", context.context_after))
        alignments[key] = components
    return streams, alignments


def test_certificates_assert_no_engine_only_vocabulary_in_any_fold(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """Red-team: engine_a's sentinel is forbidden vocabulary in engine_a's fold.

    The certificate must (a) name the sentinel as engine-only vocabulary, and (b) assert
    that every rung actually fitted for that fold excludes it. The same must hold for the
    other engines' folds against their own engine-only tokens, if any.
    """
    contexts, partition, streams, alignments = corpus
    run = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    cert_a = run.certificates["engine_a"]
    assert cert_a["pass"] is True
    assert cert_a["fold_id"] == "loeo_zero_shot:held_out=engine_a"
    assert cert_a["held_out_engine"] == "engine_a"
    assert cert_a["training_engines"] == ["engine_b", "engine_c"]
    assert cert_a["partition_sha256"] == partition.partition_sha256
    assert cert_a["source_checks"] == {
        "target_engine_records_selected": 0,
        "non_fit_document_records_selected": 0,
        "calibrate_document_records_selected": 0,
        "evaluate_document_records_selected": 0,
        "scored_document_records_selected": 0,
    }
    assert cert_a["sources_clean"] is True

    fit_documents = partition.documents(SplitRole.FIT)
    all_counts: Counter[str] = Counter()
    fold_counts: Counter[str] = Counter()
    for context in contexts:
        if context.site.document_id not in fit_documents:
            continue
        all_counts.update(context.site.ocr_text.split())
        if context.site.engine_id != "engine_a":
            fold_counts.update(context.site.ocr_text.split())
    forbidden = {
        token for token, count in all_counts.items() if count >= 2 and fold_counts[token] == 0
    }
    assert SENTINEL in forbidden
    assert cert_a["engine_only_vocabulary"] == len(forbidden)
    assert cert_a["engine_only_vocabulary_sha256"] == canonical_hash(sorted(forbidden))
    rungs = cert_a["rungs"]
    assert isinstance(rungs, dict)
    for entry in rungs.values():
        assert entry.get("clean", False), entry
    for engine_id in ENGINES:
        for entry in run.certificates[engine_id]["rungs"].values():  # type: ignore[index]
            assert entry.get("clean", False)


def test_sentinel_cannot_reach_the_fold_lexicon_that_scores_its_engine(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """Removing a target-engine-only source must not change any target proposals."""
    contexts, partition, streams, alignments = corpus
    with_sentinel = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    without_contexts = [
        context
        for context in contexts
        if not (
            context.site.engine_id == "engine_a"
            and context.site.document_id in partition.documents(SplitRole.FIT)
            and SENTINEL in context.site.ocr_text
        )
    ]
    without_streams, without_alignments = _streams_and_alignments(without_contexts)
    without_sentinel = run_cgv2_study(
        without_contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        rungs=RUNGS,
        streams=without_streams,
        alignments_by_pair=without_alignments,
    )

    def _site_rows(run: object) -> list[dict[str, object]]:
        proposals = run.site_run.proposals  # type: ignore[attr-defined]
        return [proposal.as_dict() for proposal in proposals if proposal.engine_id == "engine_a"]

    def _region_rows(run: object) -> list[dict[str, object]]:
        regions = run.region_records  # type: ignore[attr-defined]
        return [record.as_dict() for record in regions if record.engine_id == "engine_a"]

    assert _site_rows(with_sentinel) == _site_rows(without_sentinel)
    assert _region_rows(with_sentinel) == _region_rows(without_sentinel)
    assert SENTINEL not in with_sentinel.fold_tokens["engine_a"]


def test_test_only_token_cannot_change_existing_target_engine_candidates(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """Red-team the document axis with a token contributed only by held-out test data."""
    contexts, partition, streams, alignments = corpus
    baseline = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    injected_site = _site(
        "doc-10",
        "engine_a",
        9,
        TEST_SENTINEL,
        TEST_SENTINEL,
        kind=SiteKind.CLEAN,
        char_start=100,
    )
    injected_contexts = [*contexts, _context(injected_site)]
    injected_streams, injected_alignments = _streams_and_alignments(injected_contexts)
    injected = run_cgv2_study(
        injected_contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
        rungs=RUNGS,
        streams=injected_streams,
        alignments_by_pair=injected_alignments,
    )

    original_site_ids = {context.site.site_id for context in contexts}
    before = [
        proposal.as_dict()
        for proposal in baseline.site_run.proposals
        if proposal.engine_id == "engine_a"
    ]
    after = [
        proposal.as_dict()
        for proposal in injected.site_run.proposals
        if proposal.engine_id == "engine_a" and proposal.site_id in original_site_ids
    ]
    assert before == after
    assert TEST_SENTINEL not in injected.fold_tokens["engine_a"]
    certificate = injected.certificates["engine_a"]
    assert certificate["source_checks"]["evaluate_document_records_selected"] == 0  # type: ignore[index]
    assert certificate["pass"] is True


def test_structural_rung_reaches_the_shapes_no_other_rung_reaches(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition, streams, alignments = corpus
    run = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    g5 = [p for p in run.site_run.proposals if p.generator_id == "g5_structural"]
    shapes = {p.generator_meta.get("edit_shape") for p in g5}
    assert "merge" in shapes  # the multi-span "data base" site
    assert "insertion" in shapes  # the empty-OCR site between attested neighbours

    insertions = [p for p in g5 if p.generator_meta.get("edit_shape") == "insertion"]
    assert {p.candidate_text for p in insertions} == {"due"}
    assert all(p.outcome is OutcomeIfAccepted.TRUE_CORRECTION for p in insertions)

    # g3 cannot propose at an empty site: zero proposals there, by construction.
    g3_at_empty = [
        p
        for p in run.site_run.proposals
        if p.generator_id == "g3_edit_aware" and p.original_ocr == ""
    ]
    assert g3_at_empty == []


def test_regions_are_labeled_with_the_site_taxonomy(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition, streams, alignments = corpus
    run = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    assert run.region_census["eligible"] > 0
    joins = [
        r
        for r in run.region_records
        if r.left_kind == "substitution" and r.right_kind == "insertion"
    ]
    assert joins, "the dat/base pair must produce a region candidate"
    repair = [r for r in joins if r.candidate_text == "database" and r.region_gt == "database"]
    assert repair
    assert all(r.outcome is OutcomeIfAccepted.TRUE_CORRECTION for r in repair)
    assert all(r.d_after == 0 for r in repair)


def test_augmented_oracle_credits_region_members(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition, streams, alignments = corpus
    run = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    result = augmented_oracle(
        run.census,
        run.site_run.proposals,
        run.region_records,
        "g6_union",
        HarmPolicy.STRICT_WORSENING,
    )
    for engine_id in ENGINES:
        row = result[engine_id]
        assert row["n_evaluable_sites"] > 0
        assert 0 <= row["oracle_accepted_edits"] <= row["n_evaluable_sites"]
        assert 0.0 <= row["oracle_safe_coverage"] <= 1.0


def test_region_census_counts_every_bucket(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition, streams, alignments = corpus
    run = run_cgv2_study(
        contexts,
        partition,
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
        rungs=RUNGS,
        streams=streams,
        alignments_by_pair=alignments,
    )
    assert sum(run.region_census.values()) > 0
    assert run.role == "evaluate"

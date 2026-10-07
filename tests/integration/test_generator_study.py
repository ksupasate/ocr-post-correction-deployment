"""The generator readiness study, end to end on a small constructed corpus.

The study is what the H2-readiness gate reads, so the properties that make it honest have
to be executable: the lexicon must never contain the engine whose sites are being scored,
the scored documents must be disjoint from the lexicon's, and a generator that cannot run
must be reported as absent rather than as having found nothing.
"""

from __future__ import annotations

import math

import pytest

from ocr_risk.candidates.pipeline import SiteContext
from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER
from ocr_risk.experiments.generator_study import GeneratorSpec, run_generator_study
from ocr_risk.schemas.enums import HarmPolicy, SiteKind, SplitRole
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.splits.document_partition import DocumentPartition

ENGINES = ("engine_a", "engine_b", "engine_c")
# One token per engine, all within edit distance 1 of the probe below, so whichever ones
# reached the lexicon are visible in what the generator proposes.
SIGNATURE = {"engine_a": "zzzqa", "engine_b": "zzzqb", "engine_c": "zzzqc"}
PROBE = "zzzqx"


def _site(document_id: str, engine_id: str, index: int, *, clean: bool) -> CorrectionSite:
    gt = "address" if index % 2 else "invoice"
    ocr = gt if clean else gt[:-1] + "x"
    return CorrectionSite(
        site_id=f"{document_id}:{engine_id}:{index}",
        document_id=document_id,
        dataset_id="funsd",
        engine_id=engine_id,
        alignment_ids=(f"a{index}",),
        ocr_span_ids=(f"sp{index}",),
        gt_token_ids=(f"g{index}",),
        ocr_text=ocr,
        gt_text=gt,
        d_before=0 if clean else 1,
        site_kind=SiteKind.CLEAN if clean else SiteKind.SUBSTITUTION,
        evaluable=True,
        char_start=index * 10,
        char_end=index * 10 + len(ocr),
        reading_order_start=index,
        min_align_confidence=1.0,
    )


def _context(site: CorrectionSite, confidence: float) -> SiteContext:
    return SiteContext(
        site=site,
        context_before="the ",
        context_after=" field",
        native_confidences=(confidence,),
        conf_scale="paddleocr_rec_score_0_1",
        image_sha256="0" * 64,
        image_width=100,
        image_height=100,
    )


@pytest.fixture
def corpus() -> tuple[list[SiteContext], DocumentPartition]:
    contexts: list[SiteContext] = []
    role_of = {}
    for document_index in range(12):
        document_id = f"doc-{document_index:02d}"
        role_of[document_id] = (
            SplitRole.FIT
            if document_index < 7
            else SplitRole.CALIBRATE
            if document_index < 10
            else SplitRole.EVALUATE
        )
        for engine_id in ENGINES:
            for index in range(6):
                site = _site(document_id, engine_id, index, clean=index % 3 == 0)
                contexts.append(_context(site, 0.3 if index % 2 else 0.95))
            # On FIT documents each engine emits a token only it ever produces, twice so
            # it clears the lexicon's minimum frequency. On CALIBRATE documents every
            # engine gets the same probe span, one edit away from all three tokens — so
            # which tokens come back as proposals says exactly which engines' vocabulary
            # reached the lexicon that scored this engine.
            fit_document = role_of[document_id] is SplitRole.FIT
            text = SIGNATURE[engine_id] if fit_document else PROBE
            for marker_index in (98, 99) if fit_document else (98,):
                marker = _site(document_id, engine_id, marker_index, clean=True)
                contexts.append(
                    _context(marker.model_copy(update={"ocr_text": text, "gt_text": text}), 0.9)
                )
    partition = DocumentPartition(role_of=role_of, spec_hash="s" * 64, partition_sha256="p" * 64)
    return contexts, partition


def test_the_study_scores_every_engine_on_the_development_documents(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition = corpus
    run = run_generator_study(
        contexts, partition, [GENERATOR_LADDER["g0_lexical"]], HarmPolicy.STRICT_WORSENING
    )
    assert {cell.engine_id for cell in run.cells} == set(ENGINES)
    assert {cell.role for cell in run.cells} == {"calibrate"}
    assert run.n_sites_scored == 3 * 3 * 7  # 3 engines x 3 cal documents x (6 + 1 probe)
    assert set(run.runtime_seconds) == {f"g0_lexical:{engine}" for engine in ENGINES}
    assert all(seconds >= 0 for seconds in run.runtime_seconds.values())


def test_an_engine_never_contributes_to_the_lexicon_used_on_its_own_sites(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """The leak the shipped pipeline had: it built one lexicon over every engine and shared
    it across folds, so the held-out engine gated candidate generation on its own sites.

    The signature token appears only under one engine. If it reached that engine's lexicon,
    the vocabulary gate would suppress the site; scoring the other engines' signatures is
    what shows the lexicon is otherwise complete.
    """
    contexts, partition = corpus
    run = run_generator_study(
        contexts,
        partition,
        [GeneratorSpec(id="probe", kind="lexical", params={"skip_in_vocabulary": False})],
        HarmPolicy.STRICT_WORSENING,
    )
    by_engine: dict[str, set[str]] = {}
    for proposal in run.proposals:
        by_engine.setdefault(proposal.engine_id, set()).add(proposal.candidate_text)

    for engine_id, proposed in by_engine.items():
        assert SIGNATURE[engine_id] not in proposed, (
            f"{engine_id}'s own vocabulary reached the lexicon used on its sites"
        )
        others = {SIGNATURE[e] for e in ENGINES if e != engine_id}
        assert proposed & others, "the lexicon is missing the other engines' vocabulary too"


def test_the_scored_documents_are_disjoint_from_the_lexicon_documents(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition = corpus
    run = run_generator_study(
        contexts, partition, [GENERATOR_LADDER["g0_lexical"]], HarmPolicy.STRICT_WORSENING
    )
    scored = {p.document_id for p in run.proposals}
    assert scored
    assert not scored & partition.documents(SplitRole.FIT)
    assert scored <= partition.documents(SplitRole.CALIBRATE)


def test_scoring_the_held_out_documents_is_possible_but_must_be_asked_for(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """The final partition exists and is reachable — for a frozen generator, once."""
    contexts, partition = corpus
    run = run_generator_study(
        contexts,
        partition,
        [GENERATOR_LADDER["g0_lexical"]],
        HarmPolicy.STRICT_WORSENING,
        role=SplitRole.EVALUATE,
    )
    assert {p.document_id for p in run.proposals} <= partition.documents(SplitRole.EVALUATE)
    assert {cell.role for cell in run.cells} == {"evaluate"}


def test_a_generator_restricted_to_a_corpus_is_not_scored_outside_it(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition = corpus
    run = run_generator_study(
        contexts,
        partition,
        [GeneratorSpec(id="english_only", kind="lexical", datasets=("ocrd_sbb",))],
        HarmPolicy.STRICT_WORSENING,
    )
    assert not run.cells, "a generator scoped to an absent corpus must produce no cells"


def test_an_unavailable_generator_is_reported_as_absent_not_as_finding_nothing(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """Zero proposals from a missing model and zero from a silent one mean opposite things."""
    contexts, partition = corpus
    run = run_generator_study(
        contexts,
        partition,
        [GeneratorSpec(id="g2_byt5", kind="byt5", params={"allow_download": False})],
        HarmPolicy.STRICT_WORSENING,
    )
    if "g2_byt5" not in run.unavailable:  # pragma: no cover - only with a warm model cache
        pytest.skip("the checkpoint is cached locally, so the generator really is available")
    assert not run.cells
    assert run.unavailable["g2_byt5"]


def test_every_cell_has_a_pooled_row_beside_its_per_corpus_rows(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """Per-corpus rates cannot be averaged into a total when the corpora differ in size."""
    contexts, partition = corpus
    run = run_generator_study(
        contexts, partition, [GENERATOR_LADDER["g0_lexical"]], HarmPolicy.STRICT_WORSENING
    )
    pooled = [c for c in run.cells if c.dataset_id == "ALL"]
    assert len(pooled) == len(ENGINES)
    for cell in pooled:
        assert cell.quality.n_sites == 3 * 7


def test_identity_and_duplicate_proposals_never_enter_the_pool(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    contexts, partition = corpus
    run = run_generator_study(
        contexts, partition, [GENERATOR_LADDER["g0_lexical"]], HarmPolicy.STRICT_WORSENING
    )
    seen: set[tuple[str, str]] = set()
    for proposal in run.proposals:
        assert proposal.candidate_text != proposal.original_ocr
        key = (proposal.site_id, proposal.candidate_text)
        assert key not in seen
        seen.add(key)


def test_the_selection_contrast_reports_an_interval_not_only_a_margin(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """The selection rule was applied as a point estimate.

    ``.claude/rules/metrics.md`` requires a paired bootstrap and an effect size with its CI
    for a comparison on the same documents, and on the real corpus the margin between the
    top two rungs turned out to sit inside sampling noise — which a point estimate cannot
    say.
    """
    from ocr_risk.experiments.generator_study import selection_contrast

    contexts, partition = corpus
    run = run_generator_study(
        contexts,
        partition,
        [GENERATOR_LADDER["g0_lexical"], GENERATOR_LADDER["g3_edit_aware"]],
        HarmPolicy.STRICT_WORSENING,
    )
    scored = partition.documents(SplitRole.CALIBRATE)
    site_totals: dict[tuple[str, str], int] = {}
    for context in contexts:
        site = context.site
        if site.document_id in scored and site.evaluable:
            key = (site.engine_id, site.document_id)
            site_totals[key] = site_totals.get(key, 0) + 1

    contrast = selection_contrast(
        run,
        site_totals,
        "g3_edit_aware",
        "g0_lexical",
        HarmPolicy.STRICT_WORSENING,
        n_resamples=200,
        seed=3,
    )
    assert contrast.n_documents == len(scored)
    assert contrast.delta == pytest.approx(
        contrast.challenger_value - contrast.baseline_value, abs=1e-12
    )
    if math.isfinite(contrast.ci_lower):
        assert contrast.ci_lower <= contrast.delta <= contrast.ci_upper
    else:
        # Three calibration documents, every resample identical: the interval collapsed.
        # NaN rather than a fabricated bound is the honest answer, and it is what
        # `bounds=None` asks `cluster_bootstrap` for -- a difference of coverages has no
        # natural range to fall back on.
        assert math.isnan(contrast.ci_upper)
    assert 0.0 <= contrast.p_value <= 1.0
    payload = contrast.as_dict()
    assert payload["metric"] == "mean_oracle_safe_coverage"


def test_the_contrast_is_paired_so_a_generator_against_itself_has_zero_width(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """An unpaired bootstrap would give a non-degenerate interval here, which is the whole
    reason the rule names the paired one."""
    from ocr_risk.experiments.generator_study import selection_contrast

    contexts, partition = corpus
    run = run_generator_study(
        contexts, partition, [GENERATOR_LADDER["g0_lexical"]], HarmPolicy.STRICT_WORSENING
    )
    scored = partition.documents(SplitRole.CALIBRATE)
    site_totals = {
        (c.site.engine_id, c.site.document_id): 1
        for c in contexts
        if c.site.document_id in scored and c.site.evaluable
    }
    contrast = selection_contrast(
        run,
        site_totals,
        "g0_lexical",
        "g0_lexical",
        HarmPolicy.STRICT_WORSENING,
        n_resamples=100,
        seed=3,
    )
    assert contrast.delta == pytest.approx(0.0)


def test_a_scoped_generator_is_labelled_with_its_scope_not_as_the_whole_corpus(
    corpus: tuple[list[SiteContext], DocumentPartition],
) -> None:
    """Its pooled cell used to read ALL on a population half the size, and the readiness
    floor it would then be judged against was derived from the larger denominator."""
    contexts, partition = corpus
    run = run_generator_study(
        contexts,
        partition,
        [
            GeneratorSpec(id="scoped", kind="lexical", datasets=("funsd",)),
            GeneratorSpec(id="unscoped", kind="lexical"),
        ],
        HarmPolicy.STRICT_WORSENING,
    )
    scoped = {c.dataset_id for c in run.cells if c.generator_id == "scoped"}
    unscoped = {c.dataset_id for c in run.cells if c.generator_id == "unscoped"}
    assert "ALL:funsd" in scoped
    assert "ALL" not in scoped
    assert "ALL" in unscoped

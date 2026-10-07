"""The generators added for the recovery phase, and the failure taxonomy over their output.

The pilot's generator was measured only through a verifier, so "73%-88% harmful" was the
whole diagnosis. These cover the parts that produce a better diagnosis: the detector gate,
the edit shapes a substitution-only corrector cannot express, and the classifier that says
*why* a proposal was harmful.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.analysis.candidate_failures import (
    FAILURE_CLASSES,
    classify_failure,
    failure_table,
    stratified_failure_table,
)
from ocr_risk.candidates.base import GenerationContext
from ocr_risk.candidates.detector import error_signals
from ocr_risk.candidates.registry import build_generator

CORPUS = [
    "the invoice total amount",
    "the invoice total amount",
    "please provide the address",
    "please provide the address",
    "New York office",
    "New York office",
]


def _context(
    text: str,
    *,
    confidence: float | None = 0.95,
    before: str = "the ",
    after: str = " field",
) -> GenerationContext:
    return GenerationContext(
        site_id="s1",
        document_id="d1",
        dataset_id="funsd",
        engine_id="tesseract",
        original_ocr=text,
        context_before=before,
        context_after=after,
        native_confidences=() if confidence is None else (confidence,),
        conf_scale="paddleocr_rec_score_0_1" if confidence is not None else None,
    )


# ------------------------------------------------------------------------------ detector


def test_the_detector_reads_only_signals_a_deployed_system_would_have() -> None:
    signals = error_signals(_context("l0022", confidence=0.4), frozenset({"invoice"}))
    assert signals.mixed_alnum
    assert not signals.in_vocabulary
    assert signals.normalized_confidence == pytest.approx(0.4)
    assert signals.length == 5


def test_a_missing_confidence_is_not_read_as_confidence_in_correctness() -> None:
    """An engine that reports nothing must not become ungated by default."""
    generator = build_generator("error_gated", generator_id="g")
    generator.fit(CORPUS)
    silent = generator.suspicion(error_signals(_context("wxyz", confidence=None)))
    confident = generator.suspicion(error_signals(_context("wxyz", confidence=0.99)))
    assert silent > confident


def test_the_gate_suppresses_a_high_confidence_known_word_and_admits_a_doubtful_one() -> None:
    """The pilot's dominant failure: 75% of high-confidence proposals were on correct spans."""
    generator = build_generator("error_gated", generator_id="g")
    generator.fit(CORPUS)
    assert generator.propose(_context("address", confidence=0.99), 4) == []
    assert generator.propose(_context("addres", confidence=0.30), 4)


def test_the_gate_records_why_it_let_a_site_through() -> None:
    generator = build_generator("error_gated", generator_id="g")
    generator.fit(CORPUS)
    proposals = generator.propose(_context("invoce", confidence=0.20), 4)
    assert proposals
    assert "gate_suspicion" in proposals[0].metadata


def test_the_gate_does_not_double_apply_the_inner_vocabulary_filter() -> None:
    """Otherwise the ablation is "gate" versus "gate twice", not "gate" versus "no gate"."""
    generator = build_generator("error_gated", generator_id="g")
    assert generator._inner.skip_in_vocabulary is False


# ---------------------------------------------------------------------------- edit shapes


def test_the_edit_aware_generator_can_propose_a_deletion() -> None:
    """22.8% of the pilot's pool sat on sites whose only correct edit is a deletion, which
    a substitution-only corrector cannot express at all."""
    generator = build_generator("edit_aware", generator_id="g")
    generator.fit(CORPUS)
    proposals = generator.propose(_context("xqz", confidence=0.10), 4)
    assert "" in [p.text for p in proposals]
    shapes = {p.metadata.get("edit_shape") for p in proposals}
    assert "deletion" in shapes


def test_deletion_is_withheld_from_confident_and_from_known_spans() -> None:
    """Deleting real text is the most damaging edit available and has no partial credit."""
    generator = build_generator("edit_aware", generator_id="g")
    generator.fit(CORPUS)
    assert "" not in [p.text for p in generator.propose(_context("xqz", confidence=0.95), 4)]
    assert "" not in [p.text for p in generator.propose(_context("address", confidence=0.10), 4)]


def test_a_merged_token_is_split_only_where_both_halves_are_known_words() -> None:
    generator = build_generator("edit_aware", generator_id="g")
    generator.fit(CORPUS)
    proposals = generator.propose(_context("theaddress", confidence=0.5), 4)
    assert "the address" in [p.text for p in proposals]
    assert not [
        p
        for p in generator.propose(_context("qqqqwwww", confidence=0.5), 4)
        if p.metadata.get("edit_shape") == "split"
    ]


def test_a_span_that_is_too_long_to_be_a_hallucination_is_not_deleted() -> None:
    generator = build_generator("edit_aware", generator_id="g")
    generator.fit(CORPUS)
    long_span = "qqqqwwwweeeerrrrtttt"
    assert "" not in [p.text for p in generator.propose(_context(long_span, confidence=0.05), 4)]


# ------------------------------------------------------------------------- byt5 projection


@pytest.mark.parametrize(
    ("source", "rewritten", "start", "end", "expected"),
    [
        # Hand-worked. The span is bracketed in the comment.
        ("the T0TAL is", "the TOTAL is", 4, 9, "TOTAL"),  # [T0TAL]
        ("a piovided b", "a provided b", 2, 10, "provided"),  # [piovided], one deletion
        ("x addres y", "x address y", 2, 8, "address"),  # [addres ], insertion inside
        ("the cat sat", "the cat sat", 4, 7, "cat"),  # unchanged
        # An edit before the span shifts its boundaries but is not part of the proposal.
        ("teh cat sat", "the cat sat", 4, 7, "cat"),
        # An insertion at the right edge completes the token when it has no whitespace,
        # and starts a new one when it does.
        ("x cat y", "x cat dog y", 2, 5, "cat"),
        # The mirror at the LEFT edge, which used to fall through to "before the span"
        # and shift both boundaries -- so a prefix repair came back unchanged and was
        # reported as the model declining to propose.
        ("ddress line", "address line", 0, 6, "address"),
        ("the ddress", "the address", 4, 10, "address"),
        ("x cat y", "x big cat y", 2, 5, "cat"),
        # An edit straddling the span boundary is not attributable to this site:
        # "cd " (3..6) is replaced wholesale by "X", so part of the change is outside.
        ("ab cd ef", "ab Xef", 3, 5, None),
    ],
)
def test_a_window_rewrite_is_projected_back_onto_the_span_it_belongs_to(
    source: str, rewritten: str, start: int, end: int, expected: str | None
) -> None:
    """Returning the whole window would let a candidate be "correct" by rewriting text the
    site does not own, and the label describes the site."""
    from ocr_risk.candidates.byt5 import project_span

    assert project_span(source, rewritten, start, end) == expected


def test_projecting_out_of_an_empty_source_is_refused() -> None:
    from ocr_risk.candidates.byt5 import project_span

    assert project_span("", "anything", 0, 0) is None


def test_the_byte_model_declares_the_languages_it_may_be_scored_on() -> None:
    """It is fine-tuned on English; scoring it on Fraktur measures a language mismatch."""
    from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER

    generator = build_generator("byt5", generator_id="byt5")
    assert generator.applies_to("eng")
    assert not generator.applies_to("deu")
    assert GENERATOR_LADDER["g2_byt5"].datasets == ("funsd",)
    assert GENERATOR_LADDER["g2_byt5"].applies_to("funsd")
    assert not GENERATOR_LADDER["g2_byt5"].applies_to("ocrd_sbb")
    assert GENERATOR_LADDER["g0_lexical"].applies_to("ocrd_sbb"), "no scope means every corpus"


# --------------------------------------------------------------------- failure taxonomy


@pytest.mark.parametrize(
    ("original", "candidate", "gt", "kind", "delta", "expected"),
    [
        ("addres", "address", "address", "substitution", 1, "beneficial"),
        ("Gary", "Gard", "Gary", "clean", -4, "source_already_correct"),
        ("cies.", "files.", "", "insertion", -1, "site_needs_deletion"),
        ("", "text", "text", "deletion", 0, "site_needs_insertion"),
        ("10022", "10017", "10021", "substitution", -2, "numeric_corruption"),
        ("mg", "ng", "mL", "substitution", 0, "unit_corruption"),
        ("Smith", "Smyth", "Smithe", "substitution", -1, "capitalized_token_rewrite"),
        # ALL-CAPS is pervasive on forms and receipts and is a character corruption,
        # not a proper-noun rewrite.
        ("TKE", "DUE", "THE", "substitution", -2, "worsened_existing_error"),
        ("Reps.", "Reps", "Reps:", "substitution", 0, "punctuation_only"),
        ("NewYork", "New York", "Newark", "segmentation", -2, "token_merge_or_split"),
        ("piovided", "divided", "provided", "substitution", -2, "worsened_existing_error"),
        ("fummo", "fimo", "summo", "substitution", 0, "no_improvement"),
    ],
)
def test_every_failure_class_is_reachable_and_lands_where_it_should(
    original: str, candidate: str, gt: str, kind: str, delta: int, expected: str
) -> None:
    assert classify_failure(original, candidate, gt, kind, delta) == expected


def test_structural_classes_take_precedence_over_content_ones() -> None:
    """A numeric rewrite of an already-correct span is *the span was already correct*.

    Precedence matters because the two call for opposite responses: a content failure says
    build a better corrector, a structural one says this pool cannot be verified into
    usefulness however good the verifier is.
    """
    assert classify_failure("1969", "1968", "1969", "clean", -4) == "source_already_correct"
    assert classify_failure("1966", "1996", "", "insertion", 0) == "site_needs_deletion"
    # Structural precedence over the beneficial check: at an empty-ground-truth site edit
    # distance makes any SHORTER wrong string an improvement, and only the empty candidate
    # actually performs the deletion the site needs.
    assert classify_failure("243,000", "23,000", "", "insertion", 1) == "site_needs_deletion"
    assert classify_failure("243,000", "", "", "insertion", 7) == "beneficial"


def _proposals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": "g0",
                "original_ocr": "Gary",
                "candidate_text": "Gard",
                "gt_text": "Gary",
                "site_kind": "clean",
                "delta": -4,
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "normalized_confidence": 0.95,
            },
            {
                "generator_id": "g0",
                "original_ocr": "addres",
                "candidate_text": "address",
                "gt_text": "address",
                "site_kind": "substitution",
                "delta": 1,
                "dataset_id": "funsd",
                "engine_id": "tesseract",
                "normalized_confidence": 0.20,
            },
            {
                "generator_id": "g1",
                "original_ocr": "cies.",
                "candidate_text": "files.",
                "gt_text": "",
                "site_kind": "insertion",
                "delta": -1,
                "dataset_id": "ocrd_sbb",
                "engine_id": "doctr",
                "normalized_confidence": None,
            },
        ]
    )


def test_the_failure_table_shares_sum_to_one_within_each_generator() -> None:
    table = failure_table(_proposals())
    assert set(table["failure_class"]) <= set(FAILURE_CLASSES)
    for _, group in table.groupby("generator_id"):
        assert group["share"].sum() == pytest.approx(1.0)


def test_a_missing_confidence_gets_its_own_band_rather_than_joining_low() -> None:
    """An engine that reports nothing is a different situation from one reporting doubt."""
    table = stratified_failure_table(_proposals())
    bands = set(table[table["stratum"] == "confidence_band"]["level"])
    assert bands == {"high", "low", "not_reported"}


def test_the_stratified_table_covers_every_requested_stratum() -> None:
    table = stratified_failure_table(_proposals(), strata=("dataset_id", "engine_id"))
    assert set(table["stratum"]) == {"dataset_id", "engine_id", "confidence_band"}
    for _, group in table.groupby(["generator_id", "stratum", "level"]):
        assert group["share"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------- the composite rung


def test_the_union_proposes_what_either_member_proposes() -> None:
    """It is the configuration ``candidates/pipeline.py`` builds from a multi-generator
    config, and the rungs were being scored individually instead."""
    from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER

    union = build_generator("composite", generator_id="u", **GENERATOR_LADDER["g4_union"].params)
    union.fit(CORPUS)
    lexical = build_generator("lexical", generator_id="l")
    lexical.fit(CORPUS)
    edit_aware = build_generator("edit_aware", generator_id="e")
    edit_aware.fit(CORPUS)

    context = _context("addres", confidence=0.20)
    members = {p.text for p in lexical.propose(context, 4)} | {
        p.text for p in edit_aware.propose(context, 4)
    }
    assert members, "the fixture must make at least one member propose"
    assert {p.text for p in union.propose(context, 8)} == members


def test_the_union_de_duplicates_the_same_edit_from_two_members() -> None:
    """Both members wrap the same corrector, so most of their output coincides. Counting
    an edit twice would inflate the pool it is a share of."""
    from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER

    union = build_generator("composite", generator_id="u", **GENERATOR_LADDER["g4_union"].params)
    union.fit(CORPUS)
    texts = [p.text for p in union.propose(_context("invoce", confidence=0.20), 8)]
    assert len(texts) == len(set(texts))


def test_a_composite_with_no_members_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        build_generator("composite", generator_id="u", members=[])


def test_the_union_reports_a_lexicon_size_rather_than_a_corpus_count() -> None:
    """Three of four rungs used to record `len(set(corpus))`, so a reader would conclude
    they had four times the lexicon exposure of the one rung that reported honestly."""
    from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER

    union = build_generator("composite", generator_id="u", **GENERATOR_LADDER["g4_union"].params)
    union.fit(CORPUS)
    lexical = build_generator("lexical", generator_id="l")
    lexical.fit(CORPUS)
    assert union.lexicon_size == lexical.lexicon_size
    # The distinguishing property: it is the fitted lexicon, not a count of corpus rows.
    # On this fixture those happen to differ by a factor of three in the other direction,
    # so the check is against the real object rather than against a magnitude.
    assert union.lexicon_size != len(set(CORPUS))


def test_the_gate_and_the_corrector_agree_on_what_a_known_word_is() -> None:
    """They did not. The detector split the raw corpus — every hapax, every two-character
    fragment — while the corrector kept tokens of length >= 3 seen at least twice, so the
    gate treated another engine's one-off misreading as a known word."""
    gated = build_generator("error_gated", generator_id="g")
    gated.fit([*CORPUS, "zqx one-off token"])
    inner = gated._inner
    assert gated.vocabulary_size == inner.lexicon_size
    assert "zqx" not in gated._vocabulary
    assert "invoice" in gated._vocabulary

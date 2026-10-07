"""Hand-computed trigger tests for the structural generator (CGV2 protocol §5.1)."""

from __future__ import annotations

import pytest

from ocr_risk.candidates.base import GenerationContext
from ocr_risk.candidates.structural import StructuralGenerator


class _Ctx:
    """A stand-in with exactly GenerationContext's generator-visible fields."""

    def __init__(
        self,
        ocr: str,
        before: str = "",
        after: str = "",
        n_spans: int = 1,
    ) -> None:
        self.original_ocr = ocr
        self.context_before = before
        self.context_after = after
        self.n_spans = n_spans


def _fitted(corpus: list[str]) -> StructuralGenerator:
    generator = StructuralGenerator()
    generator.fit(corpus)
    return generator


@pytest.fixture
def generator() -> StructuralGenerator:
    # "database" appears joined once and split once; both halves clear the lexicon on
    # their own. The amount/due/today bigrams repeat so insertion is attested both ways.
    return _fitted(
        [
            "total amount due today",
            "amount due today database",
            "amount due today again",
            "data base total",
            "database total",
        ]
    )


def test_insertion_proposes_the_attested_continuation(generator: StructuralGenerator) -> None:
    context = _Ctx("", before="amount", after="today")
    proposals = generator.propose(context, 8)
    assert [p.text for p in proposals] == ["due"]
    meta = proposals[0].metadata
    assert meta["edit_shape"] == "insertion"
    # (amount, due) appears 3x, (due, today) 3x -- the winner's own evidence, not the
    # last loop iteration's.
    assert meta["bigram_forward"] == "3" and meta["bigram_backward"] == "3"


def test_insertion_is_one_sided_when_the_backward_bigram_is_unattested(
    generator: StructuralGenerator,
) -> None:
    """Amendment A2: a missing form value follows its label, so forward attestation alone
    is evidence. The backward count rides along in the metadata as zero."""
    context = _Ctx("", before="amount", after="never")
    proposals = generator.propose(context, 8)
    assert [p.text for p in proposals] == ["due"]
    assert proposals[0].metadata["bigram_backward"] == "0"
    assert proposals[0].metadata["bigram_forward"] == "3"


def test_insertion_vocabulary_includes_short_tokens() -> None:
    """The empty-OCR stratum's ground truth is 3 characters at the median -- '-', '1',
    'of' -- so the insertion vocabulary must not apply the lexicon's min length."""
    generator = _fitted(["name - value", "name - total", "name - of"])
    context = _Ctx("", before="name", after="value")
    proposals = generator.propose(context, 8)
    assert [p.text for p in proposals] == ["-"]


def test_insertion_refuses_without_both_neighbours(generator: StructuralGenerator) -> None:
    assert generator.propose(_Ctx("", before="amount"), 8) == []
    assert generator.propose(_Ctx("", after="today"), 8) == []


def test_insertion_refuses_at_a_non_empty_site(generator: StructuralGenerator) -> None:
    assert generator.propose(_Ctx("something", before="amount", after="today"), 8) == []


def test_merge_proposes_the_known_join_at_multi_span_sites(
    generator: StructuralGenerator,
) -> None:
    proposals = generator.propose(_Ctx("data base", n_spans=2), 8)
    assert [(p.text, p.metadata["edit_shape"]) for p in proposals] == [("database", "merge")]


def test_merge_refuses_at_single_span_sites(generator: StructuralGenerator) -> None:
    assert generator.propose(_Ctx("data base", n_spans=1), 8) == []


def test_merge_refuses_when_the_join_is_not_a_word(generator: StructuralGenerator) -> None:
    assert generator.propose(_Ctx("total today", n_spans=2), 8) == []


def test_region_join_fires_on_the_known_word(generator: StructuralGenerator) -> None:
    proposals = generator.propose_region("data", "base", None)
    assert [(p.text, p.metadata["edit_shape"]) for p in proposals] == [("database", "region_join")]


def test_region_substitution_fires_within_edit_distance(generator: StructuralGenerator) -> None:
    # The fold lexicon is unigram, so the trigger matches the separator-free join
    # against single words: a noisy split ("dat base") near a known join ("database").
    proposals = generator.propose_region("dat", "base", None)
    assert proposals, "database is distance 1 from the separator-free join datbase"
    assert proposals[0].text == "database"
    assert proposals[0].metadata["edit_shape"] == "region_substitution"


def test_region_refuses_an_empty_member(generator: StructuralGenerator) -> None:
    assert generator.propose_region("", "base", None) == []
    assert generator.propose_region("data", "  ", None) == []


def test_region_respects_the_limit(generator: StructuralGenerator) -> None:
    assert len(generator.propose_region("dat", "base", 1)) == 1


def test_proposals_are_deterministic(generator: StructuralGenerator) -> None:
    first = generator.propose(_Ctx("", before="amount", after="today"), 8)
    second = generator.propose(_Ctx("", before="amount", after="today"), 8)
    assert first == second


def test_smaller_budget_is_a_prefix_of_the_larger(generator: StructuralGenerator) -> None:
    """The K-grid truncates by rank, so propose(ctx, k) must be a prefix of propose(ctx, 8)."""
    context = _Ctx("data base", n_spans=2)
    full = generator.propose(context, 8)
    for k in (1, 2, 4):
        assert generator.propose(context, k) == full[:k]


def test_generation_context_projection_has_no_ground_truth() -> None:
    """The real projection type carries no GT field the generator could read."""
    assert not any(
        "gt" in field
        for field in GenerationContext.__dataclass_fields__  # type: ignore[attr-defined]
    )


def test_fitting_is_confined_to_the_given_corpus() -> None:
    generator = _fitted(["database total", "database total"])
    other = _fitted(["something else entirely", "something else entirely"])
    assert "database" in generator._vocabulary
    assert "database" not in other._vocabulary

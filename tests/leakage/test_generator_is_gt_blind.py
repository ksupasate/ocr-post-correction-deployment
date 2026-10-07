"""Ground truth must not reach a candidate generator, by construction and by attempt.

The generator decides *which* edits exist. A generator that could see the answer would
manufacture a candidate pool no deployed system could produce, and every downstream number
— repair opportunity, oracle coverage, the whole H2 premise — would be measured on a pool
that cannot exist. Unlike a verifier leak, this one is invisible in the results: the pool
simply looks unusually good.

Every generator on the ladder is checked, not only the one the pilot used.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Any

import pytest

from ocr_risk.candidates.base import CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import build_generator
from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER
from ocr_risk.schemas.enums import SiteKind
from ocr_risk.schemas.sites import CorrectionSite

FORBIDDEN = frozenset(
    {"gt_text", "gt_token_ids", "d_before", "d_after", "delta", "outcome_if_accepted", "label"}
)
CORPUS = [
    "the quick brown fox",
    "the quick brown fox",
    "invoice total amount",
    "invoice total amount",
    "please provide the address",
    "please provide the address",
]


def _site() -> CorrectionSite:
    return CorrectionSite(
        site_id="s1",
        document_id="d1",
        dataset_id="funsd",
        engine_id="tesseract",
        alignment_ids=("a1",),
        ocr_span_ids=("sp1",),
        gt_token_ids=("g1",),
        ocr_text="addres",
        gt_text="address",
        d_before=1,
        site_kind=SiteKind.SUBSTITUTION,
        evaluable=True,
        char_start=0,
        char_end=6,
        reading_order_start=0,
        min_align_confidence=1.0,
    )


def test_the_generation_context_has_no_field_that_could_carry_the_answer() -> None:
    names = {f.name for f in fields(GenerationContext)}
    assert not names & FORBIDDEN, f"GenerationContext exposes {sorted(names & FORBIDDEN)}"


def test_projecting_a_site_drops_its_ground_truth() -> None:
    """``CorrectionSite`` carries ``gt_text``; the projection is what makes it unreachable."""
    site = _site()
    assert site.gt_text == "address"
    context = GenerationContext.from_site(site, "the ", " field")
    for name in FORBIDDEN:
        assert not hasattr(context, name)


@pytest.mark.parametrize("study_id", sorted(GENERATOR_LADDER))
def test_no_generator_on_the_ladder_accepts_a_site_in_place_of_a_context(
    study_id: str,
) -> None:
    """Red team: hand each generator the GT-bearing record and require a refusal.

    ``GenerationContext`` and ``CorrectionSite`` have overlapping field names, so a
    generator that reached for ``context.gt_text`` would work when handed a site and fail
    only in production. Passing the site must raise, not quietly succeed.
    """
    spec = GENERATOR_LADDER[study_id]
    generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
    if not generator.available():
        pytest.skip(f"{study_id} is not available in this environment")
    generator.fit(CORPUS)
    with pytest.raises((AttributeError, TypeError)):
        generator.propose(_site(), spec.max_candidates)  # type: ignore[arg-type]


@pytest.mark.parametrize("study_id", sorted(GENERATOR_LADDER))
def test_a_generator_proposal_does_not_depend_on_ground_truth(study_id: str) -> None:
    """The same context must produce the same proposals whatever the truth happens to be.

    Nothing in the type system stops a generator from consulting a global, so this checks
    the property rather than the plumbing: two runs whose only difference is the invisible
    ground truth must agree exactly.
    """
    spec = GENERATOR_LADDER[study_id]
    generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
    if not generator.available():
        pytest.skip(f"{study_id} is not available in this environment")
    generator.fit(CORPUS)

    first = GenerationContext.from_site(_site(), "the ", " field")
    second = GenerationContext.from_site(
        _site().model_copy(update={"gt_text": "COMPLETELY DIFFERENT"}), "the ", " field"
    )

    def texts(proposals: list[CandidateProposal]) -> list[str]:
        return [p.text for p in proposals]

    assert texts(generator.propose(first, spec.max_candidates)) == texts(
        generator.propose(second, spec.max_candidates)
    )


@pytest.mark.parametrize("study_id", sorted(GENERATOR_LADDER))
def test_no_generator_signature_mentions_ground_truth(study_id: str) -> None:
    """A parameter named for the answer is the cheapest possible way to smuggle it in."""
    spec = GENERATOR_LADDER[study_id]
    generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
    for method_name in ("propose", "propose_batch", "fit"):
        method: Any = getattr(generator, method_name, None)
        if method is None:
            continue
        parameters = set(inspect.signature(method).parameters)
        assert not parameters & FORBIDDEN, f"{study_id}.{method_name} takes {parameters}"


def test_the_study_runner_only_ever_hands_a_generator_a_projection() -> None:
    """The one call site that could pass the wrong object, checked in the source.

    A type annotation is not a guarantee at runtime, and this loop is the only place in
    the repository where sites and generators meet.
    """
    from ocr_risk.experiments import generator_study

    source = inspect.getsource(generator_study.run_generator_study)
    assert "c.generation_context()" in source, "the runner must project before proposing"
    before_generation, _, _ = source.partition("proposals = _propose_all(")
    assert ".gt_text" not in before_generation, (
        "ground truth is read before the generator runs, not only when labelling after it"
    )

"""Candidate-generator quality, measured independently of any verifier.

The H1 pilot entangled three questions and could not separate them: whether the generator
proposes anything worth accepting (**Q1**), whether a verifier can tell the good proposals
from the bad (**Q2**), and whether its scores convert into decisions meeting a harm target
(**Q3**). Coverage came out identically zero, and with one number for all three there was
no way to attribute that. Everything in this module answers Q1 only, and none of it
involves a model.

Two of these are **ground-truth oracle** quantities: they ask what a perfect verifier
could do with a pool, which is an upper bound no deployable system reaches. They are
analysis-only, they carry ``oracle`` in the name, and the report that prints them is
required to label them. Ground truth never reaches a generator.

Denominators, stated once because they are what makes the rates comparable across
generators:

- ``BCR``/``HCR``/``no-change rate`` are over **non-identity candidates**, so a generator
  that emits the identity option at every site is not rewarded for it.
- ``ERO`` is over **OCR error sites**, so a generator that proposes nowhere scores 0
  rather than being undefined.
- ``CSPR`` is over **already-correct OCR sites**, which is the population where
  overcorrection is the only thing that can happen.

One further rule, on what counts as a repair. At a site whose ground truth is empty — the
OCR invented text and the correct edit is a deletion — edit distance makes *any shorter
wrong string* an improvement. Under the plain outcome taxonomy those score
``PARTIAL_IMPROVEMENT`` and enter every oracle ceiling. They are arithmetically real and
operationally empty: a verifier cannot be asked to prefer a shorter hallucination. The
three oracle-facing metrics (``ERO``, ``oracle_safe_coverage``, ``oracle_repair_recall``)
therefore require an *exact* repair at such sites. Counting them inflated the pilot
generator's oracle safe coverage from 0.028 to 0.043 — a 55% overstatement of the ceiling
the H2 gate is read against. ``BCR``/``HCR`` keep the taxonomy's own definition, so the
outcome shares still sum as the taxonomy says they do.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ocr_risk.edits.outcome import is_beneficial, is_harmful
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted

__all__ = [
    "GeneratorQuality",
    "SiteProposals",
    "beneficial_candidate_rate",
    "clean_span_proposal_rate",
    "error_repair_opportunity",
    "generator_quality",
    "harmful_candidate_rate",
    "oracle_repair_recall",
    "oracle_safe_coverage",
]


@dataclass(frozen=True, slots=True)
class SiteProposals:
    """One correction site and every non-identity candidate proposed for it.

    ``outcomes`` and ``deltas`` are parallel. A site with an empty ``outcomes`` received no
    proposal, which is a real and common state — it is the difference between "the
    generator declined" and "the generator was wrong", and collapsing the two is how a
    quiet generator gets mistaken for an accurate one.
    """

    site_id: str
    d_before: int
    outcomes: tuple[OutcomeIfAccepted, ...] = ()
    deltas: tuple[int, ...] = ()
    needs_deletion: bool = False
    """The ground truth here is empty: the OCR invented text and the only correct edit is
    to remove it.

    It matters to the oracle metrics because edit distance makes *any shorter wrong string*
    an improvement at such a site — ``243,000`` proposed as ``23,000`` against an empty
    reference scores ``PARTIAL_IMPROVEMENT``. That is arithmetically correct and
    operationally meaningless: no verifier can be asked to prefer a shorter hallucination,
    and counting it inflated the pilot generator's oracle safe coverage substantially."""

    @property
    def is_clean(self) -> bool:
        """The OCR was already correct here, so any accepted edit is overcorrection."""
        return self.d_before == 0

    @property
    def n_proposals(self) -> int:
        return len(self.outcomes)

    def repairable(self, policy: HarmPolicy) -> bool:
        """Whether a perfect verifier could take this site to a genuinely better state."""
        return any(self._counts_as_repair(o, policy) for o in self.outcomes)

    def _counts_as_repair(self, outcome: OutcomeIfAccepted, policy: HarmPolicy) -> bool:
        if not is_beneficial(outcome, policy):
            return False
        # At a deletion-shaped site only the exact edit -- the empty string -- is a repair.
        return not self.needs_deletion or outcome is OutcomeIfAccepted.TRUE_CORRECTION


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def beneficial_candidate_rate(outcomes: Iterable[OutcomeIfAccepted], policy: HarmPolicy) -> float:
    """Beneficial share of non-identity candidates. NaN when nothing was proposed."""
    total = 0
    good = 0
    for outcome in outcomes:
        if outcome is OutcomeIfAccepted.IDENTITY:
            continue
        total += 1
        good += is_beneficial(outcome, policy)
    return _rate(good, total)


def harmful_candidate_rate(outcomes: Iterable[OutcomeIfAccepted], policy: HarmPolicy) -> float:
    """Harmful share of non-identity candidates. NaN when nothing was proposed."""
    total = 0
    bad = 0
    for outcome in outcomes:
        if outcome is OutcomeIfAccepted.IDENTITY:
            continue
        total += 1
        bad += is_harmful(outcome, policy)
    return _rate(bad, total)


def error_repair_opportunity(sites: Sequence[SiteProposals], policy: HarmPolicy) -> float:
    """Share of OCR error sites where at least one beneficial candidate exists.

    The ceiling on repair for this pool: no verifier, however good, can fix an error the
    generator never proposed a fix for. When this is low the bottleneck is Q1, and a better
    verifier cannot move it.
    """
    errors = [s for s in sites if not s.is_clean]
    return _rate(sum(site.repairable(policy) for site in errors), len(errors))


def clean_span_proposal_rate(sites: Sequence[SiteProposals]) -> float:
    """Share of already-correct OCR sites that received a non-identity proposal.

    Every one of these is an opportunity to overcorrect that the generator created. It is
    the quantity a detector-then-corrector design exists to reduce, and it is measurable
    without any verifier.
    """
    clean = [s for s in sites if s.is_clean]
    return _rate(sum(s.n_proposals > 0 for s in clean), len(clean))


def oracle_safe_coverage(sites: Sequence[SiteProposals], policy: HarmPolicy) -> float:
    """**GROUND-TRUTH ORACLE — NOT DEPLOYABLE.** Sites a perfect verifier could correct.

    Denominator is every site considered, clean ones included, matching the deployed
    coverage denominator. A perfect verifier accepts a beneficial candidate wherever one
    exists and preserves everywhere else, so this is the largest coverage any selective
    method could reach on this pool **at zero risk**. If it is near zero, a risk-coverage
    curve cannot exist and there is nothing for a verifier to be good at.
    """
    if not sites:
        return float("nan")
    return _rate(sum(site.repairable(policy) for site in sites), len(sites))


def oracle_repair_recall(sites: Sequence[SiteProposals], policy: HarmPolicy) -> float:
    """**GROUND-TRUTH ORACLE — NOT DEPLOYABLE.** Share of error characters repairable.

    Weighted by ``d_before`` rather than by site, because a site with one wrong character
    and a site with twenty are not the same repair. Uses the best available candidate per
    site, so it is the character-level ceiling matching :func:`oracle_safe_coverage`.
    """
    total = sum(s.d_before for s in sites)
    if not total:
        return float("nan")
    repaired = 0
    for site in sites:
        gains = [
            delta
            for outcome, delta in zip(site.outcomes, site.deltas, strict=True)
            if site._counts_as_repair(outcome, policy) and delta > 0
        ]
        repaired += max(gains, default=0)
    return _rate(repaired, total)


@dataclass(frozen=True, slots=True)
class GeneratorQuality:
    """Every Q1 endpoint for one (generator, engine, dataset) cell."""

    n_sites: int
    n_clean_sites: int
    n_error_sites: int
    n_nonidentity_candidates: int
    beneficial_rate: float
    harmful_rate: float
    no_change_rate: float
    error_repair_opportunity: float
    clean_span_proposal_rate: float
    oracle_safe_coverage: float
    oracle_repair_recall: float
    exact_correction_rate: float
    partial_improvement_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_sites": self.n_sites,
            "n_clean_sites": self.n_clean_sites,
            "n_error_sites": self.n_error_sites,
            "n_nonidentity_candidates": self.n_nonidentity_candidates,
            "beneficial_rate": self.beneficial_rate,
            "harmful_rate": self.harmful_rate,
            "no_change_rate": self.no_change_rate,
            "error_repair_opportunity": self.error_repair_opportunity,
            "clean_span_proposal_rate": self.clean_span_proposal_rate,
            "oracle_safe_coverage": self.oracle_safe_coverage,
            "oracle_repair_recall": self.oracle_repair_recall,
            "exact_correction_rate": self.exact_correction_rate,
            "partial_improvement_rate": self.partial_improvement_rate,
        }


def generator_quality(sites: Sequence[SiteProposals], policy: HarmPolicy) -> GeneratorQuality:
    """Every Q1 endpoint at once, from one pass over the pool."""
    outcomes = [o for site in sites for o in site.outcomes if o is not OutcomeIfAccepted.IDENTITY]
    n_candidates = len(outcomes)
    # "No change" is LATERAL_CHANGE: the text moved but the distance to ground truth did
    # not. Under strict_worsening it is neither beneficial nor harmful, which makes it
    # invisible in the two headline rates -- and it was 24.5% of the pilot's pool.
    no_change = sum(o is OutcomeIfAccepted.LATERAL_CHANGE for o in outcomes)
    exact = sum(o is OutcomeIfAccepted.TRUE_CORRECTION for o in outcomes)
    partial = sum(o is OutcomeIfAccepted.PARTIAL_IMPROVEMENT for o in outcomes)
    return GeneratorQuality(
        n_sites=len(sites),
        n_clean_sites=sum(s.is_clean for s in sites),
        n_error_sites=sum(not s.is_clean for s in sites),
        n_nonidentity_candidates=n_candidates,
        beneficial_rate=beneficial_candidate_rate(outcomes, policy),
        harmful_rate=harmful_candidate_rate(outcomes, policy),
        no_change_rate=_rate(no_change, n_candidates),
        error_repair_opportunity=error_repair_opportunity(sites, policy),
        clean_span_proposal_rate=clean_span_proposal_rate(sites),
        oracle_safe_coverage=oracle_safe_coverage(sites, policy),
        oracle_repair_recall=oracle_repair_recall(sites, policy),
        exact_correction_rate=_rate(exact, n_candidates),
        partial_improvement_rate=_rate(partial, n_candidates),
    )

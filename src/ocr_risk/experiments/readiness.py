"""Three separate gates: the generator, H2 readiness, and RH1.

They are kept apart because the H1 pilot merged them and could not attribute its own
result. One gate reporting "coverage was zero" is compatible with a generator that
proposes nothing useful, a verifier that cannot rank, and a controller that cannot certify
a target — three different problems with three different next steps.

Each gate returns one verdict from a closed set. None of them may be softened because the
answer is disappointing: a NOT READY that stops a milestone is the gate working.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "GENERATOR_INCONCLUSIVE",
    "GENERATOR_NOT_READY",
    "GENERATOR_PARTIALLY_READY",
    "GENERATOR_READY",
    "H2_INCONCLUSIVE",
    "H2_NOT_READY",
    "H2_READY",
    "RH1_INCONCLUSIVE",
    "RH1_NOT_SUPPORTED",
    "RH1_PARTIALLY_SUPPORTED",
    "RH1_SUPPORTED",
    "Criterion",
    "EngineCell",
    "GateOutcome",
    "evaluate_generator_gate",
    "evaluate_h2_readiness",
    "evaluate_rh1_gate",
    "generator_criteria",
    "h2_readiness_criteria",
]

GENERATOR_READY = "GENERATOR READY"
GENERATOR_PARTIALLY_READY = "GENERATOR PARTIALLY READY"
GENERATOR_NOT_READY = "GENERATOR NOT READY"
GENERATOR_INCONCLUSIVE = "GENERATOR INCONCLUSIVE"

H2_READY = "H2 READY"
H2_NOT_READY = "H2 NOT READY"
H2_INCONCLUSIVE = "H2 INCONCLUSIVE"

RH1_SUPPORTED = "DISCRIMINATION DEGRADATION SUPPORTED"
RH1_PARTIALLY_SUPPORTED = "PARTIALLY SUPPORTED"
RH1_NOT_SUPPORTED = "NOT SUPPORTED"
RH1_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class Criterion:
    """One pre-registered condition, its threshold, and what was measured."""

    key: str
    question: str
    threshold: str
    observed: str
    met: bool
    measurable: bool = True
    """False when the evidence needed to judge it was not produced. Distinguished from
    ``met=False`` throughout: "we looked and it failed" and "we could not look" lead to
    different next steps, and the pilot's gate collapsed them."""

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "question": self.question,
            "threshold": self.threshold,
            "observed": self.observed,
            "met": self.met,
            "measurable": self.measurable,
        }


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """One gate's verdict, with every criterion it rests on."""

    gate: str
    verdict: str
    criteria: tuple[Criterion, ...]
    notes: tuple[str, ...] = ()
    context: dict[str, object] = field(default_factory=dict)

    @property
    def n_met(self) -> int:
        return sum(c.met for c in self.criteria)

    @property
    def n_measurable(self) -> int:
        return sum(c.measurable for c in self.criteria)

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "verdict": self.verdict,
            "n_criteria": len(self.criteria),
            "n_measurable": self.n_measurable,
            "n_met": self.n_met,
            "criteria": [c.as_dict() for c in self.criteria],
            "notes": list(self.notes),
            "context": self.context,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def _verdict(
    criteria: Sequence[Criterion], ready: str, partial: str, not_ready: str, inconclusive: str
) -> str:
    """The shared decision rule, so three gates cannot drift into three rules.

    Unmeasurable criteria dominate: a gate that cannot see one of its conditions is
    INCONCLUSIVE regardless of how the others came out, because the missing one could have
    changed the answer. Only once everything is visible does the count decide.
    """
    if any(not c.measurable for c in criteria):
        return inconclusive
    met = sum(c.met for c in criteria)
    if met == len(criteria):
        return ready
    return not_ready if met == 0 else partial


def evaluate_generator_gate(
    criteria: Sequence[Criterion], notes: Sequence[str] = ()
) -> GateOutcome:
    """Is any generator on the ladder good enough to build a study on?"""
    return GateOutcome(
        gate="generator",
        verdict=_verdict(
            criteria,
            GENERATOR_READY,
            GENERATOR_PARTIALLY_READY,
            GENERATOR_NOT_READY,
            GENERATOR_INCONCLUSIVE,
        ),
        criteria=tuple(criteria),
        notes=tuple(notes),
    )


def evaluate_h2_readiness(
    criteria: Sequence[Criterion],
    notes: Sequence[str] = (),
    context: dict[str, object] | None = None,
) -> GateOutcome:
    """Can the incremental value of source-image verification be measured at all here?

    There is no PARTIALLY READY. H2 is a single experiment that either can or cannot be
    run: a pool that satisfies three of five conditions does not support three-fifths of
    an experiment, and a partial verdict is the kind of wording that lets a milestone
    proceed on evidence that does not support it.
    """
    if any(not c.measurable for c in criteria):
        verdict = H2_INCONCLUSIVE
    else:
        verdict = H2_READY if all(c.met for c in criteria) else H2_NOT_READY
    return GateOutcome(
        gate="h2_readiness",
        verdict=verdict,
        criteria=tuple(criteria),
        notes=tuple(notes),
        context=dict(context or {}),
    )


def evaluate_rh1_gate(
    n_engines_degraded: int,
    n_engines_tested: int,
    primary_metric: str,
    supporting: Sequence[Criterion] = (),
    notes: Sequence[str] = (),
    context: dict[str, object] | None = None,
    harm_policy_disagreement: bool = False,
    n_engines_unmeasurable: int = 0,
    sensitivity_evidence_missing: bool = False,
) -> GateOutcome:
    """Does unseen-engine transfer reduce edit discrimination, on matched arms?

    Read per engine and never averaged: four OCR engines are four fixed environments, not
    four draws from a population of engines. The thresholds are majority and any, declared
    before the run, because "how many engines" is the only summary a fixed-factor design
    supports.

    Three rules sit above the engine counts, in precedence order:

    1. **Evidence not produced makes the gate INCONCLUSIVE** (protocol section 5). An
       engine whose pooled interval is missing or degenerate is "could not look", not
       "looked and found nothing" -- the distinction this project's own pilot got wrong.
       The same applies to the pre-registered sensitivity tables: a missing or unpooled
       sensitivity table is not a measured zero.
    2. **The pre-committed harm-policy rule** (protocol section 4): when the sensitivity
       policies disagree with the primary about how many engines degraded, the verdict is
       PARTIALLY SUPPORTED *whatever the engine counts say* -- including from SUPPORTED.
       The criterion keeps the true counts; only the verdict is overridden. Downgrading
       by re-deriving the verdict from adjusted counts was the first implementation and
       is a no-op from SUPPORTED, where it matters most.
    """
    if n_engines_tested == 0 or n_engines_unmeasurable > 0 or sensitivity_evidence_missing:
        verdict = RH1_INCONCLUSIVE
    else:
        if n_engines_degraded == n_engines_tested:
            verdict = RH1_SUPPORTED
        elif n_engines_degraded == 0:
            verdict = RH1_NOT_SUPPORTED
        else:
            verdict = RH1_PARTIALLY_SUPPORTED
        if harm_policy_disagreement and verdict in {RH1_SUPPORTED, RH1_NOT_SUPPORTED}:
            verdict = RH1_PARTIALLY_SUPPORTED

    observed = f"{n_engines_degraded} of {n_engines_tested} engines"
    if n_engines_unmeasurable:
        observed += f" ({n_engines_unmeasurable} unmeasurable)"
    criteria = (
        Criterion(
            key="rh1_primary",
            question=(f"Does {primary_metric} degrade under transfer, per engine, after Holm?"),
            threshold="all engines = SUPPORTED; some = PARTIALLY; none = NOT SUPPORTED; "
            "harm-policy disagreement = PARTIALLY; evidence missing = INCONCLUSIVE",
            observed=observed,
            met=n_engines_degraded > 0,
            measurable=n_engines_tested > 0 and n_engines_unmeasurable == 0,
        ),
        *supporting,
    )
    return GateOutcome(
        gate="rh1",
        verdict=verdict,
        criteria=criteria,
        notes=tuple(notes),
        context=dict(context or {}),
    )


# --------------------------------------------------------------------- the frozen criteria

# Every number here is the one in docs/h1_recovery/h2_readiness_protocol.md, frozen before
# the held-out partition was scored. They are derived from the sample-size table in that
# document's section 2, not from the development results.
MIN_ORACLE_ACCEPTED = 245
"""Accepted edits needed to certify risk <= 0.05 with harm at epsilon/2 (Bentkus, delta=0.1)."""
MIN_ORACLE_COVERAGE = 0.05
"""The same requirement as a rate, at the 4101-4890 sites per engine this corpus holds."""
MIN_ENGINES = 3
"""Of four. Results are reported per engine and never averaged."""
MIN_NATURAL_CANDIDATES = 1000
MIN_TEST_DOCUMENTS = 20
"""The document-level bootstrap's resampling unit; below ~20 clusters a percentile
interval is not trustworthy."""
MIN_CLEAN_SITE_SHARE = 0.30
MIN_CLEAN_PROPOSALS = 100
"""Overcorrection is observable only where the OCR was right and something was proposed."""


@dataclass(frozen=True, slots=True)
class EngineCell:
    """One engine's measured quantities, typed so the criteria are not stringly-indexed."""

    engine_id: str
    oracle_accepted: int
    oracle_safe_coverage: float
    n_nonidentity_candidates: int
    n_sites: int
    n_clean_sites: int
    clean_span_proposal_rate: float

    @property
    def clean_share(self) -> float:
        return self.n_clean_sites / self.n_sites if self.n_sites else 0.0

    @property
    def clean_proposals(self) -> float:
        return self.clean_span_proposal_rate * self.n_clean_sites


def h2_readiness_criteria(
    *,
    per_engine: Sequence[EngineCell],
    n_test_documents: int,
    all_natural: bool | None,
    all_pairs_matched: bool | None,
    alignment_sensitivity_reported: bool,
) -> list[Criterion]:
    """Assemble the frozen criteria from one generator's measured cells.

    ``all_pairs_matched`` is ``None`` when no match certificate was produced, which makes
    criterion F unmeasurable rather than failed. ``all_natural`` is ``None`` when the
    scored pool's manifest cannot be read, which makes criterion E unmeasurable rather
    than passed on faith.
    """
    engines = list(per_engine)
    n_a = sum(e.oracle_accepted >= MIN_ORACLE_ACCEPTED for e in engines)
    n_b = sum(e.oracle_safe_coverage >= MIN_ORACLE_COVERAGE for e in engines)
    n_candidates = min((e.n_nonidentity_candidates for e in engines), default=0)
    clean_shares = [e.clean_share for e in engines]
    clean_proposals = [e.clean_proposals for e in engines]

    return [
        Criterion(
            key="A_repair_opportunity",
            question="Does the pool admit enough beneficial edits to certify any target?",
            threshold=(
                f">= {MIN_ORACLE_ACCEPTED} oracle-accepted edits on >= {MIN_ENGINES}/4 engines"
            ),
            observed=(
                f"{n_a}/{len(engines)} engines; per engine "
                + ", ".join(f"{e.engine_id}={e.oracle_accepted}" for e in engines)
            ),
            met=n_a >= MIN_ENGINES,
        ),
        Criterion(
            key="B_oracle_safe_coverage",
            question="Would a perfect verifier trace a risk-coverage curve worth reading?",
            threshold=(
                f"oracle safe coverage >= {MIN_ORACLE_COVERAGE} on >= {MIN_ENGINES}/4 engines"
            ),
            observed=(
                f"{n_b}/{len(engines)} engines; per engine "
                + ", ".join(f"{e.engine_id}={e.oracle_safe_coverage:.4f}" for e in engines)
            ),
            met=n_b >= MIN_ENGINES,
        ),
        Criterion(
            key="C_sample_size",
            question="Are there enough candidates and enough document clusters?",
            threshold=(
                f">= {MIN_NATURAL_CANDIDATES} natural candidates per engine and "
                f">= {MIN_TEST_DOCUMENTS} test documents"
            ),
            observed=f"min {n_candidates} candidates, {n_test_documents} documents",
            met=n_candidates >= MIN_NATURAL_CANDIDATES and n_test_documents >= MIN_TEST_DOCUMENTS,
        ),
        Criterion(
            key="D_overcorrection_observable",
            question="Can overcorrection still be measured on this pool?",
            threshold=(
                f">= {MIN_CLEAN_SITE_SHARE:.0%} of sites clean and >= {MIN_CLEAN_PROPOSALS} "
                "proposals on clean sites per engine"
            ),
            observed=(
                f"clean share {min(clean_shares, default=0.0):.2f}-"
                f"{max(clean_shares, default=0.0):.2f}, clean proposals "
                f"{min(clean_proposals, default=0.0):.0f}-{max(clean_proposals, default=0.0):.0f}"
            ),
            met=(
                bool(clean_shares)
                and min(clean_shares) >= MIN_CLEAN_SITE_SHARE
                and min(clean_proposals) >= MIN_CLEAN_PROPOSALS
            ),
        ),
        Criterion(
            key="E_natural_pool",
            question="Are the headline candidates naturally generated, not adversarial?",
            threshold="100% pool == natural",
            observed=(
                "scored pool not verifiable"
                if all_natural is None
                else "natural only"
                if all_natural
                else "challenge candidates present"
            ),
            met=bool(all_natural),
            measurable=all_natural is not None,
        ),
        Criterion(
            key="F_no_unresolved_confound",
            question="Are the arms matched, and is the alignment confound reported?",
            threshold="every match certificate matched, alignment sensitivity reported",
            observed=(
                "no match certificate produced"
                if all_pairs_matched is None
                else f"matched={all_pairs_matched}, "
                f"alignment_sensitivity={alignment_sensitivity_reported}"
            ),
            met=bool(all_pairs_matched) and alignment_sensitivity_reported,
            measurable=all_pairs_matched is not None,
        ),
    ]


# The generator gate asks a narrower question than H2 readiness: did the ladder move the
# pool at all, relative to the generator the H1 pilot actually ran? Its criteria are
# RELATIVE and were written after the measurements — stated plainly here, because unlike
# the H2 criteria they were not frozen in advance. The H2 verdict does not depend on them.
GENERATOR_CSPR_IMPROVEMENT = 2.0
"""How much a rung must cut the clean-span proposal rate to count as having helped."""


def generator_criteria(
    *,
    baseline: Sequence[EngineCell],
    best: Sequence[EngineCell],
    baseline_id: str,
    best_id: str,
    baseline_clean_span_rate: float,
    best_clean_span_rate: float,
    baseline_repair_opportunity: float,
    best_repair_opportunity: float,
    meets_h2_repair: bool,
    meets_h2_coverage: bool,
) -> list[Criterion]:
    """Did the ladder produce a pool better than the one the pilot ran on?"""
    by_engine = {cell.engine_id: cell for cell in baseline}
    improved = [
        cell.engine_id
        for cell in best
        if cell.engine_id in by_engine
        and cell.oracle_safe_coverage > by_engine[cell.engine_id].oracle_safe_coverage
    ]
    ratio = (
        baseline_clean_span_rate / best_clean_span_rate
        if best_clean_span_rate > 0
        else float("inf")
    )
    return [
        Criterion(
            key="G1_more_repair_reachable",
            question=f"Does {best_id} raise oracle safe coverage over {baseline_id}?",
            threshold=f"improved on > {len(list(best)) // 2} of {len(list(best))} engines",
            observed=f"{len(improved)}/{len(list(best))} engines improved: {sorted(improved)}",
            met=len(improved) > len(list(best)) // 2,
        ),
        Criterion(
            key="G2_less_overcorrection_offered",
            question="Does it cut clean-span proposals without giving up repair reach?",
            threshold=(
                f"clean-span proposal rate cut >= {GENERATOR_CSPR_IMPROVEMENT}x with repair "
                "opportunity not reduced"
            ),
            observed=(
                f"clean-span {baseline_clean_span_rate:.3f} -> {best_clean_span_rate:.3f} "
                f"({ratio:.1f}x), repair opportunity {baseline_repair_opportunity:.3f} -> "
                f"{best_repair_opportunity:.3f}"
            ),
            met=ratio >= GENERATOR_CSPR_IMPROVEMENT
            and best_repair_opportunity >= baseline_repair_opportunity,
        ),
        Criterion(
            key="G3_clears_the_h2_floor",
            question="Is the resulting pool one the H2 study could actually be run on?",
            threshold="meets the frozen H2 criteria A and B",
            observed=f"A={meets_h2_repair}, B={meets_h2_coverage}",
            met=meets_h2_repair and meets_h2_coverage,
        ),
    ]

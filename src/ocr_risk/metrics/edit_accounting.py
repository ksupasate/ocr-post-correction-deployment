"""Edit accounting: what a set of accept/preserve decisions actually did.

This is the layer where the project's framing lives. CER and WER say how good the final
text is; edit accounting says *how much genuine repair was safely automated*, which is the
question the risk-coverage comparison is built on.

Every rate here has an explicitly named denominator, because the interesting ones differ:

- **edit precision** — beneficial accepted / accepted. What fraction of what we did helped.
- **correction recall** — beneficial accepted / sites that needed repair. How much of the
  available repair we captured.
- **overcorrection rate** — overcorrections / clean sites. Conditioned on sites that were
  already right, so it is not diluted by how noisy the corpus is.
- **accepted-edit risk** — harmful accepted / accepted. The quantity being bounded.
- **joint harm rate** — harmful accepted / sites considered. A stable companion, because
  the selective denominator vanishes as the threshold rises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ocr_risk.edits.outcome import is_beneficial, is_harmful
from ocr_risk.schemas.enums import (
    CoverageUnit,
    HarmPolicy,
    OutcomeIfAccepted,
    OutcomeIfRejected,
)

__all__ = ["EditAccounting", "EditDecision", "account"]


@dataclass(frozen=True, slots=True)
class EditDecision:
    """One site's decision and its counterfactual outcomes.

    Carries both branches so the accounting can attribute a rejected site to preservation
    or to a missed error without re-deriving anything from text.
    """

    site_id: str
    document_id: str
    accepted: bool
    d_before: int
    outcome_if_accepted: OutcomeIfAccepted | None
    """``None`` when no candidate was available at this site."""
    outcome_if_rejected: OutcomeIfRejected
    delta: int = 0
    """``d_before - d_after`` for the candidate under consideration."""
    n_candidates: int = 0
    n_beneficial_available: int = 0
    """Beneficial candidates the generator offered at this site, accepted or not.

    The denominator of :attr:`EditAccounting.beneficial_edit_coverage`. Without it the only
    answerable question is "how much of the *corpus* was repaired", which conflates a
    verifier that rejected good candidates with a generator that never produced any."""


@dataclass(slots=True)
class EditAccounting:
    """Counts and rates for one set of decisions under one harm policy."""

    harm_policy: HarmPolicy
    coverage_unit: CoverageUnit

    n_sites: int = 0
    n_sites_with_error: int = 0
    n_clean_sites: int = 0
    n_candidates: int = 0
    n_beneficial_available: int = 0

    n_accepted: int = 0
    n_true_corrections: int = 0
    n_partial_improvements: int = 0
    n_lateral_changes: int = 0
    n_miscorrections: int = 0
    n_overcorrections: int = 0
    n_beneficial: int = 0
    n_harmful: int = 0

    n_preservations: int = 0
    n_missed_errors: int = 0

    characters_repaired: int = 0
    """Sum of ``delta`` over accepted edits: net characters moved toward ground truth.
    Negative contributions from harmful edits are included, which is the point."""
    characters_broken: int = 0
    total_error_characters: int = 0

    by_outcome: dict[str, int] = field(default_factory=dict)

    # --- rates -----------------------------------------------------------------------
    @property
    def coverage(self) -> float:
        """Accepted decisions over decisions considered.

        Site-level by default: sites are the decision points, whereas candidates-per-site
        is an artifact of the generator and would make generators incomparable.
        """
        denominator = self.n_sites if self.coverage_unit is CoverageUnit.SITE else self.n_candidates
        return self.n_accepted / denominator if denominator else 0.0

    @property
    def accepted_edit_risk(self) -> float:
        """Harmful accepted edits over accepted edits. The quantity being bounded."""
        return self.n_harmful / self.n_accepted if self.n_accepted else 0.0

    @property
    def joint_harm_rate(self) -> float:
        """Harmful accepted edits over all sites considered.

        Reported alongside the selective risk because that denominator vanishes as the
        threshold rises, making the ratio unstable exactly where a method looks safest.
        """
        return self.n_harmful / self.n_sites if self.n_sites else 0.0

    @property
    def edit_precision(self) -> float:
        return self.n_beneficial / self.n_accepted if self.n_accepted else 0.0

    @property
    def correction_recall(self) -> float:
        """Beneficial accepted edits over sites that actually needed repair."""
        return self.n_beneficial / self.n_sites_with_error if self.n_sites_with_error else 0.0

    @property
    def repair_coverage(self) -> float:
        """**The headline coverage measure**: how much genuine repair was safely automated.

        Three coverages are reported and they answer different questions. This one is the
        headline because it is the one the research question is about; the other two are
        diagnostics that say *where* it went wrong.

        =============================  ==========================================
        measure                        denominator
        =============================  ==========================================
        :attr:`coverage`               every site a decision was taken at
        :attr:`repair_coverage`        sites that actually needed repair
        :attr:`beneficial_edit_cov...` beneficial candidates the generator offered
        =============================  ==========================================
        """
        return self.correction_recall

    @property
    def beneficial_edit_coverage(self) -> float:
        """Beneficial accepted edits over beneficial candidates available. Diagnostic.

        Separates the two ways repair coverage can be low. If this is high while
        :attr:`repair_coverage` is low, the verifier is accepting what it was offered and
        the generator did not offer much — a Q1 failure. If this is low, the verifier is
        rejecting good candidates — a Q2 failure. The pilot could not tell those apart.
        """
        return (
            self.n_beneficial / self.n_beneficial_available if self.n_beneficial_available else 0.0
        )

    @property
    def preservation_rate(self) -> float:
        """Clean sites correctly left alone, over clean sites."""
        return self.n_preservations / self.n_clean_sites if self.n_clean_sites else 0.0

    @property
    def overcorrection_rate(self) -> float:
        """Clean sites broken, over clean sites.

        Conditioning on clean sites keeps this from being diluted by corpus noise: a
        method is not safer merely because it was run on messier text.
        """
        return self.n_overcorrections / self.n_clean_sites if self.n_clean_sites else 0.0

    @property
    def missed_error_rate(self) -> float:
        return self.n_missed_errors / self.n_sites_with_error if self.n_sites_with_error else 0.0

    @property
    def net_repair_gain(self) -> float:
        """Net characters moved toward ground truth, as a share of the errors present.

        Can be negative: a method that breaks more than it fixes should report a negative
        gain rather than a small positive one.
        """
        return (
            self.characters_repaired / self.total_error_characters
            if self.total_error_characters
            else 0.0
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "harm_policy": self.harm_policy.value,
            "coverage_unit": self.coverage_unit.value,
            "n_sites": self.n_sites,
            "n_sites_with_error": self.n_sites_with_error,
            "n_clean_sites": self.n_clean_sites,
            "n_candidates": self.n_candidates,
            "n_accepted": self.n_accepted,
            "n_beneficial": self.n_beneficial,
            "n_harmful": self.n_harmful,
            "n_true_corrections": self.n_true_corrections,
            "n_partial_improvements": self.n_partial_improvements,
            "n_lateral_changes": self.n_lateral_changes,
            "n_miscorrections": self.n_miscorrections,
            "n_overcorrections": self.n_overcorrections,
            "n_preservations": self.n_preservations,
            "n_missed_errors": self.n_missed_errors,
            "coverage": self.coverage,
            "accepted_edit_risk": self.accepted_edit_risk,
            "joint_harm_rate": self.joint_harm_rate,
            "edit_precision": self.edit_precision,
            "correction_recall": self.correction_recall,
            "repair_coverage": self.repair_coverage,
            "beneficial_edit_coverage": self.beneficial_edit_coverage,
            "n_beneficial_available": self.n_beneficial_available,
            "preservation_rate": self.preservation_rate,
            "overcorrection_rate": self.overcorrection_rate,
            "missed_error_rate": self.missed_error_rate,
            "net_repair_gain": self.net_repair_gain,
            "characters_repaired": self.characters_repaired,
            "characters_broken": self.characters_broken,
        }


_COUNTER_FIELD = {
    OutcomeIfAccepted.TRUE_CORRECTION: "n_true_corrections",
    OutcomeIfAccepted.PARTIAL_IMPROVEMENT: "n_partial_improvements",
    OutcomeIfAccepted.LATERAL_CHANGE: "n_lateral_changes",
    OutcomeIfAccepted.MISCORRECTION: "n_miscorrections",
    OutcomeIfAccepted.OVERCORRECTION: "n_overcorrections",
}


def account(
    decisions: Sequence[EditDecision],
    harm_policy: HarmPolicy,
    coverage_unit: CoverageUnit = CoverageUnit.SITE,
) -> EditAccounting:
    """Tally a set of decisions under one harm policy."""
    result = EditAccounting(harm_policy=harm_policy, coverage_unit=coverage_unit)

    for decision in decisions:
        result.n_sites += 1
        result.n_candidates += decision.n_candidates
        result.n_beneficial_available += decision.n_beneficial_available
        result.total_error_characters += decision.d_before
        if decision.d_before > 0:
            result.n_sites_with_error += 1
        else:
            result.n_clean_sites += 1

        if not decision.accepted:
            if decision.outcome_if_rejected is OutcomeIfRejected.PRESERVATION:
                result.n_preservations += 1
            else:
                result.n_missed_errors += 1
            continue

        outcome = decision.outcome_if_accepted
        if outcome is None:
            msg = f"site {decision.site_id} is marked accepted but carries no outcome"
            raise ValueError(msg)

        result.n_accepted += 1
        result.by_outcome[outcome.value] = result.by_outcome.get(outcome.value, 0) + 1
        counter = _COUNTER_FIELD.get(outcome)
        if counter:
            setattr(result, counter, getattr(result, counter) + 1)

        if is_harmful(outcome, harm_policy):
            result.n_harmful += 1
        if is_beneficial(outcome, harm_policy):
            result.n_beneficial += 1

        result.characters_repaired += decision.delta
        if decision.delta < 0:
            result.characters_broken += -decision.delta

    return result

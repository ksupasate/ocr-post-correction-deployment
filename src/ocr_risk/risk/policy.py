"""Turning calibrated scores into per-site decisions.

At most **one** candidate is accepted per site. Sites are the decision points; accepting
two edits at one site would either conflict textually or double-count a single repair in
both coverage and harm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["SITE_POLICIES", "SiteDecision", "decide_sites"]

SITE_POLICIES = ("argmax_above_tau", "top1_above_tau", "none")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One candidate with its calibrated score, as the policy sees it."""

    candidate_id: str
    site_id: str
    document_id: str
    score: float
    generator_rank: int = 0


@dataclass(frozen=True, slots=True)
class SiteDecision:
    """What was decided at one site."""

    site_id: str
    document_id: str
    accepted_candidate_id: str | None
    accepted_score: float | None
    n_candidates_considered: int

    @property
    def accepted(self) -> bool:
        return self.accepted_candidate_id is not None


def decide_sites(
    candidates: Sequence[ScoredCandidate], tau: float, policy: str = "argmax_above_tau"
) -> list[SiteDecision]:
    """Apply the acceptance policy at every site.

    ``argmax_above_tau`` (default) takes the best-scoring candidate if it clears the
    threshold. ``top1_above_tau`` considers only the generator's own first choice, which
    is the stricter reading of "does the pipeline's preferred edit pass?" and is available
    as a sensitivity check. ``none`` preserves everything — the no-correction baseline.
    """
    if policy not in SITE_POLICIES:
        msg = f"unknown site policy {policy!r}; known: {', '.join(SITE_POLICIES)}"
        raise ValueError(msg)

    by_site: dict[str, list[ScoredCandidate]] = {}
    for candidate in candidates:
        by_site.setdefault(candidate.site_id, []).append(candidate)

    decisions: list[SiteDecision] = []
    for site_id in sorted(by_site):
        group = by_site[site_id]
        document_id = group[0].document_id

        if policy == "none":
            chosen = None
        elif policy == "top1_above_tau":
            # Ties on rank are broken by score, then by id, so the choice is deterministic.
            first = min(group, key=lambda c: (c.generator_rank, -c.score, c.candidate_id))
            chosen = first if first.score >= tau else None
        else:
            best = max(group, key=lambda c: (c.score, -c.generator_rank, c.candidate_id))
            chosen = best if best.score >= tau else None

        decisions.append(
            SiteDecision(
                site_id=site_id,
                document_id=document_id,
                accepted_candidate_id=chosen.candidate_id if chosen else None,
                accepted_score=chosen.score if chosen else None,
                n_candidates_considered=len(group),
            )
        )
    return decisions

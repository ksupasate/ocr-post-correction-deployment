"""Does the alignment layer's per-engine ambiguity distort what the benchmark measures?

EasyOCR resolves far fewer components than the other three — 9 612 ambiguous against
3 334 for PaddleOCR on the pilot corpus — and an ambiguous component yields no evaluable
site. Clean sites are where overcorrection is observable and error sites are where repair
is, so an engine that resolves less is measured on a *different* population, not merely a
smaller one. Every cross-engine comparison then carries that difference inside it.

The control is a **common-support** subset: the ground-truth tokens that *every* engine
resolved. The inclusion rule is engine-independent by construction (it is the same set of
positions for all four) and ground-truth-safe (it reads alignment status, never
correctness). Restricting to it answers "do the conclusions hold where all four engines
are measured on the same text?"

Nothing here removes ambiguous EasyOCR examples from the headline. Both populations are
reported side by side, and where they disagree the disagreement is the finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ocr_risk.schemas.enums import AlignmentStatus, HarmPolicy, OutcomeIfAccepted, harmful_outcomes

__all__ = ["SupportComparison", "common_support_tokens", "support_sensitivity_table"]


@dataclass(frozen=True, slots=True)
class SupportComparison:
    """One engine, measured on the full population and on common support."""

    engine_id: str
    population: str
    n_sites: int
    n_clean_sites: int
    clean_share: float
    n_candidates: int
    harmful_rate: float
    beneficial_rate: float
    mean_d_before: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "engine_id": self.engine_id,
            "population": self.population,
            "n_sites": self.n_sites,
            "n_clean_sites": self.n_clean_sites,
            "clean_share": self.clean_share,
            "n_candidates": self.n_candidates,
            "harmful_rate": self.harmful_rate,
            "beneficial_rate": self.beneficial_rate,
            "mean_d_before": self.mean_d_before,
        }


def common_support_tokens(alignments: pd.DataFrame) -> set[tuple[str, str]]:
    """``(document_id, gt_token_id)`` pairs every engine resolved.

    Resolved, not merely present: an ambiguous component is carried in the artifact but
    contributes to no metric, so a token only one engine resolved is a position where the
    others are silent rather than wrong.
    """
    engines = set(alignments["engine_id"].unique())
    if not engines:
        return set()
    resolved = alignments[alignments["status"] == AlignmentStatus.RESOLVED.value]
    seen: dict[tuple[str, str], set[str]] = {}
    for document_id, engine, tokens in zip(
        resolved["document_id"], resolved["engine_id"], resolved["gt_token_ids"], strict=True
    ):
        for token in tokens if tokens is not None else ():
            seen.setdefault((str(document_id), str(token)), set()).add(str(engine))
    return {key for key, found in seen.items() if found == engines}


def support_sensitivity_table(
    alignments: pd.DataFrame,
    sites: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    policy: HarmPolicy = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """Per engine, the same quantities on the full population and on common support."""
    support = common_support_tokens(alignments)
    harmful_values = {o.value for o in harmful_outcomes(policy)}
    beneficial_values = {
        OutcomeIfAccepted.TRUE_CORRECTION.value,
        OutcomeIfAccepted.PARTIAL_IMPROVEMENT.value,
    }

    evaluable = sites[sites["evaluable"]].copy()

    def _covered(document_id: object, tokens: Sequence[str] | None) -> bool:
        if tokens is None or len(tokens) == 0:
            return False
        return all((str(document_id), str(token)) in support for token in tokens)

    evaluable["in_support"] = [
        _covered(document_id, tokens)
        for document_id, tokens in zip(
            evaluable["document_id"], evaluable["gt_token_ids"], strict=True
        )
    ]

    pool = candidates.merge(
        labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    )
    pool = pool[~pool["is_synthetic_hard_negative"]]

    rows: list[dict[str, float | int | str]] = []
    for engine_id, engine_sites in evaluable.groupby("engine_id", sort=True):
        for population, subset in (
            ("full", engine_sites),
            ("common_support", engine_sites[engine_sites["in_support"]]),
        ):
            site_ids = set(subset["site_id"])
            engine_pool = pool[pool["site_id"].isin(site_ids)]
            outcomes = engine_pool["outcome_if_accepted"]
            rows.append(
                SupportComparison(
                    engine_id=str(engine_id),
                    population=population,
                    n_sites=len(subset),
                    n_clean_sites=int((subset["d_before"] == 0).sum()),
                    clean_share=float((subset["d_before"] == 0).mean()) if len(subset) else 0.0,
                    n_candidates=len(engine_pool),
                    harmful_rate=(
                        float(outcomes.isin(harmful_values).mean()) if len(engine_pool) else 0.0
                    ),
                    beneficial_rate=(
                        float(outcomes.isin(beneficial_values).mean()) if len(engine_pool) else 0.0
                    ),
                    mean_d_before=float(subset["d_before"].mean()) if len(subset) else 0.0,
                ).as_dict()
            )
    return pd.DataFrame(rows)

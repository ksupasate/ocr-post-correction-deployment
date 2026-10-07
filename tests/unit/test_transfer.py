"""The H1 transfer contrast.

H1 is a paired comparison between two protocols, and the ways it can be got wrong are
specific: comparing unmatched rows, using a metric confounded by base rate, or reducing
four fixed engines to a mean. Each has a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ocr_risk.analysis.tables import AnalysisInput
from ocr_risk.analysis.transfer import h1_transfer_table, held_out_engine_of
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted


def _arm(
    *, engines: tuple[str, ...], scores: dict[str, float], n_per_engine: int = 40
) -> AnalysisInput:
    """One protocol's predictions over a fixed candidate set.

    ``scores`` gives each engine's central predicted probability. Real per-document
    variation is added around it, deliberately: a *constant* score makes the paired
    difference algebraically independent of which documents are drawn, so the bootstrap
    has zero sampling variability and the guard in paired_cluster_bootstrap correctly
    refuses to call it significant. A fixture without variation would test nothing.
    """
    rng = np.random.default_rng(17)
    predictions, labels, candidates = [], [], []
    for engine in engines:
        for index in range(n_per_engine):
            candidate_id = f"{engine}:cand:{index:03d}"
            # Alternating rather than random, so every engine AND every document has a
            # base harm rate of exactly 0.5. A random rate makes the calibration error
            # differ by engine for reasons the test is not about.
            harmful = index % 2 == 0
            score = float(np.clip(scores[engine] + rng.normal(0.0, 0.03), 0.01, 0.99))
            predictions.append(
                {
                    "candidate_id": candidate_id,
                    "document_id": f"{engine}:doc:{index // 4:02d}",
                    "fold_id": f"loeo_zero_shot:{engine}",
                    "verifier_id": "v3_text_conf",
                    "raw_score": score,
                    "calibrated_score": score,
                }
            )
            labels.append(
                {
                    "candidate_id": candidate_id,
                    "outcome_if_accepted": (
                        OutcomeIfAccepted.MISCORRECTION.value
                        if harmful
                        else OutcomeIfAccepted.TRUE_CORRECTION.value
                    ),
                }
            )
            candidates.append({"candidate_id": candidate_id, "is_synthetic_hard_negative": False})
    return AnalysisInput(
        predictions=pd.DataFrame(predictions),
        decisions=pd.DataFrame(),
        labels=pd.DataFrame(labels),
        candidates=pd.DataFrame(candidates),
    )


def test_held_out_engine_is_read_off_the_fold_id() -> None:
    assert held_out_engine_of("loeo_zero_shot:tesseract") == "tesseract"
    assert held_out_engine_of("in_engine_oracle:doctr") == "doctr"


def test_a_degraded_arm_is_reported_per_engine_with_a_paired_interval() -> None:
    """The zero-shot arm is badly calibrated on every engine; the oracle is well calibrated.

    The base harm rate is ~0.5 on every engine, so a raw-Brier contrast and a calibration
    contrast agree here -- which is what makes this a check of the machinery rather than
    of the metric choice.
    """
    engines = ("engine_a", "engine_b", "engine_c", "engine_d")
    zero_shot = _arm(engines=engines, scores=dict.fromkeys(engines, 0.95))
    oracle = _arm(engines=engines, scores=dict.fromkeys(engines, 0.5))

    table = h1_transfer_table(zero_shot, oracle, HarmPolicy.STRICT_WORSENING, n_bootstrap=2000)
    primary = table[table["metric"] == "calibration_error"]

    assert set(primary["held_out_engine"]) == set(engines), "one row per target engine"
    assert (primary["delta"] > 0).all(), "zero-shot minus oracle must be positive when worse"
    assert primary["degraded_after_holm"].all()
    assert "n_documents" in primary and (primary["n_documents"] > 1).all()


def test_an_undegraded_arm_is_not_reported_as_degraded() -> None:
    """Identical arms must produce no finding. The gate reads this column directly."""
    engines = ("engine_a", "engine_b", "engine_c", "engine_d")
    scores = dict.fromkeys(engines, 0.5)
    table = h1_transfer_table(
        _arm(engines=engines, scores=scores),
        _arm(engines=engines, scores=scores),
        HarmPolicy.STRICT_WORSENING,
        n_bootstrap=200,
    )
    primary = table[table["metric"] == "calibration_error"]
    assert not primary["degraded_after_holm"].any()
    assert np.allclose(primary["delta"].to_numpy(), 0.0)


def test_only_matched_candidates_are_compared() -> None:
    """A paired contrast on different rows is not paired.

    The oracle arm here carries twice as many candidates per engine. The comparison must
    use the intersection, not each arm's own population -- otherwise a difference in what
    was evaluated is reported as a difference in what was learned.
    """
    engines = ("engine_a", "engine_b")
    zero_shot = _arm(engines=engines, scores=dict.fromkeys(engines, 0.9), n_per_engine=20)
    oracle = _arm(engines=engines, scores=dict.fromkeys(engines, 0.5), n_per_engine=40)
    table = h1_transfer_table(zero_shot, oracle, HarmPolicy.STRICT_WORSENING, n_bootstrap=100)
    assert (table["n_candidates"] == 20).all(), "the intersection is the smaller arm"


def test_no_statistic_is_computed_across_engines() -> None:
    """Four engines are four fixed environments, not four draws from a population.

    The table must expose per-engine rows and nothing that aggregates them; a mean or a
    p-value over the four would be population inference the design cannot support.
    """
    engines = ("engine_a", "engine_b", "engine_c", "engine_d")
    table = h1_transfer_table(
        _arm(engines=engines, scores=dict.fromkeys(engines, 0.9)),
        _arm(engines=engines, scores=dict.fromkeys(engines, 0.5)),
        HarmPolicy.STRICT_WORSENING,
        n_bootstrap=100,
    )
    assert "held_out_engine" in table.columns
    assert len(table[table["metric"] == "brier"]) == len(engines), "per engine, not pooled"
    forbidden = {"mean_delta", "pooled_delta", "across_engine_p_value", "engine_mean"}
    assert not forbidden & set(table.columns)


def test_the_holm_family_is_the_four_engines_not_engines_times_verifiers() -> None:
    """The family is declared in the pre-registration and must not silently widen.

    Folding all seven ablation arms into one family applies a correction for comparisons
    H1 is not making -- V3 against V6 is H2's question -- and is strictly more
    conservative than what was promised. With four engines Holm needs p <= alpha/4 for the
    smallest; with 28 it needs alpha/28, which no attainable bootstrap p would clear at a
    sane resample count. The distinction decides whether H1 can return SUPPORTED at all.
    """
    engines = ("engine_a", "engine_b", "engine_c", "engine_d")
    table = h1_transfer_table(
        _arm(engines=engines, scores=dict.fromkeys(engines, 0.95)),
        _arm(engines=engines, scores=dict.fromkeys(engines, 0.5)),
        HarmPolicy.STRICT_WORSENING,
        n_bootstrap=2000,
    )
    family = table[
        (table["metric"] == "calibration_error") & (table["verifier_id"] == "v3_text_conf")
    ]
    assert len(family) == len(engines)

    # Holm's largest adjustment factor is the family size. With four engines all at the
    # same p, the smallest adjusted value is 4p; a family of 28 would give 28p.
    raw = family["p_value"].min()
    adjusted = family["holm_adjusted_p"].min()
    assert adjusted <= len(engines) * raw + 1e-12, (
        f"adjusted {adjusted} exceeds a four-test correction of raw {raw}"
    )


def test_a_constant_score_arm_is_not_reported_as_degraded() -> None:
    """The defect this guards against was live in the shipped H1 table.

    A constant-score arm — the accept-everything reference emits one — makes the paired
    difference identical on every document draw. The bootstrap p then falls to its floor
    2/(B+1) on ZERO sampling variability, and three rows were flagged degraded with an
    interval 1e-16 wide. Maximal significance from no information.
    """
    engines = ("engine_a", "engine_b")
    zero_shot = _arm_constant(engines=engines, score=0.9)
    oracle = _arm_constant(engines=engines, score=0.5)
    table = h1_transfer_table(zero_shot, oracle, HarmPolicy.STRICT_WORSENING, n_bootstrap=300)
    assert not table["degraded_after_holm"].any(), (
        "a difference that cannot vary under resampling is not evidence"
    )
    assert table["delta_ci_lower"].isna().all(), "a degenerate draw yields no interval"


def _arm_constant(*, engines: tuple[str, ...], score: float) -> AnalysisInput:
    """An arm whose verifier emits one constant score, like v0_accept_all."""
    predictions, labels, candidates = [], [], []
    for engine in engines:
        for index in range(40):
            candidate_id = f"{engine}:cand:{index:03d}"
            predictions.append(
                {
                    "candidate_id": candidate_id,
                    "document_id": f"{engine}:doc:{index // 4:02d}",
                    "fold_id": f"loeo_zero_shot:{engine}",
                    "verifier_id": "v0_accept_all",
                    "raw_score": score,
                    "calibrated_score": score,
                }
            )
            labels.append(
                {
                    "candidate_id": candidate_id,
                    "outcome_if_accepted": (
                        OutcomeIfAccepted.MISCORRECTION.value
                        if index % 2 == 0
                        else OutcomeIfAccepted.TRUE_CORRECTION.value
                    ),
                }
            )
            candidates.append({"candidate_id": candidate_id, "is_synthetic_hard_negative": False})
    return AnalysisInput(
        predictions=pd.DataFrame(predictions),
        decisions=pd.DataFrame(),
        labels=pd.DataFrame(labels),
        candidates=pd.DataFrame(candidates),
    )

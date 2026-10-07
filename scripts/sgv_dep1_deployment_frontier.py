#!/usr/bin/env python3
"""SGV-DEP1: deployment frontier analysis for safe OCR correction automation.

SGV-RK4 left the cutoff as the deployment bottleneck: the adapted ranker orders a new generator's
candidates well from five labelled pages on, and no cutoff that accounts for page-clustered harm
accepts anything within the pages this universe holds. This stage asks a narrower, practical
question on frozen scores alone: under which review policy can any part of the work be taken off
a human's desk safely, and what does each labelled page buy?

**1. No model is fitted.** Every score is SGV-RK4's (a new generator, by labelled page budget) or
SGV-RK3's and SGV-DS1's (known generators). The stage chooses cutoffs and routes decisions.

**2. Four policies.** Full automation with a plug-in cutoff (P0); full automation with the
page-corrected Clopper-Pearson cutoff (P1); a three-way abstention policy (P2) that accepts above
the conservative cutoff, rejects below a mirrored conservative cutoff and sends the rest to a human;
and a cost-aware three-way policy (P3) that chooses both cutoffs by plug-in estimates to maximize
the share of decisions no human sees, subject to harm <= 0.1.

**3. A rejected site can lose a repair but cannot cause harm.** An automatic reject keeps the OCR
text. The reject cutoff is therefore held to a mirrored constraint: a rejected decision may carry an
exact repair no more often than an accepted one may carry harm.

    --reconstruct   section 0: DS1, TH1, TH2, RK3 and RK4 re-read from their own artifacts
    --freeze        upstream hashes and the frozen scores this stage consumes
    --preregister   policies, constraints, practical and deployable definitions, criteria
    --cells         one decision per site for every frozen cell, label-free
    --policies      every policy's bands on every cell, read on the sealed test pages
    --curves        label-budget curves over draws
    --frontier      deployable and practical frontiers and their page budgets
    --tradeoffs     ANALYSIS ONLY harm-coverage, reject and human-review tradeoff curves
    --controls      random-score control through every policy
    --stats         exact sign tests over paired draws, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / DEPLOYMENT METHODOLOGY ONLY. No candidate is generated, no OCR output is regenerated,
no model is fitted, nothing is certified and nothing is production-ready. The human reviewer is an
idealization that accepts exact repairs and rejects everything else; no human was measured. The
confirmatory reserve stays LOCKED and `ready_for_external_confirmation` is false by construction.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_rk4_generator_adaptive_ranking as rk4
from ocr_risk.io.hashing import canonical_hash, file_sha256

rk3 = rk4.rk3
th1 = rk4.th1
th2 = rk4.th2
rk2 = rk4.rk2
rk1 = rk4.rk1
ds1 = rk4.ds1
rl1 = rk4.rl1
s15 = rk4.s15

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/generated/sgv_dep1"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"

DECISIONS = OUT / "decisions.parquet"
CELL_REGISTRY = OUT / "cell_registry.json"
POLICY_OUTCOMES = OUT / "policy_outcomes.parquet"
POLICY_REGISTRY = OUT / "policy_registry.json"
CURVES = OUT / "label_budget_curves.json"
FRONTIER = OUT / "deployment_frontier.json"
TRADEOFFS = OUT / "tradeoff_curves.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_dep1/deployment_frontier.md"

SCHEMA_VERSION = 1
STAGE = "sgv_dep1_deployment_frontier"
HYPOTHESIS = "SGV-DEP1-D1"
STAGE_KIND = "DEVELOPMENT / DEPLOYMENT METHODOLOGY"

PhaseError = rk4.PhaseError
_relative = rk4._relative
_git = rk4._git
_write_json_once = rk4._write_json_once
_write_parquet_once = rk4._write_parquet_once
cc_read_json = rk4.cc_read_json
_require = rk4._require
_forbid = rk4._forbid
_share = rk4._share

BOOTSTRAP_RESAMPLES = rk4.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rk4.BOOTSTRAP_SEED
CONTROL_SEED = 20261001

# ------------------------------------------------------------------ the frozen design

DIRECTIONS = rk4.DIRECTIONS
DIRECTION_SHORT = rk4.DIRECTION_SHORT
BUDGETS = rk4.BUDGETS
POOL = rk4.POOL
DECISION_BUDGET = rk4.DECISION_BUDGET
DRAWS = rk4.DRAWS
NOT_MEASURABLE = rk4.BRIEF_BUDGETS_NOT_MEASURABLE

EPSILON = rk4.PRIMARY_EPSILON
ETA = (
    EPSILON  # a rejected decision may carry an exact repair no more often than an accepted one harm
)
DELTA = rk4.DELTA
VIOLATION_CEILING = rk4.VIOLATION_CEILING
WORKING_FLOOR = rk4.WORKING_FLOOR
REVIEW_REDUCTION_FLOOR = 0.25
REVIEW_REDUCTION_SENSITIVITY = (0.10, 0.25, 0.50)
HARM_TOLERANCE = rk4.HARM_TOLERANCE

SCENARIO_NEW = "new_generator"
SCENARIO_KNOWN = "known_generators"
NEW_ARM = rk4.M1
ZERO_ARM = rk4.A0
KNOWN_MODELS = (rk3.M_RK3, rk3.M_DS1)
KNOWN_LABEL = {rk3.M_RK3: "RK3 risk-aware LambdaRank", rk3.M_DS1: "DS1 reliability score"}

R_PLUG_IN = "plug_in"
R_PAGE_BOUND = "page_bound"
R_ALL_REST = "reject_all_rest"

P0 = "p0_full_automation"
P1 = "p1_conservative_threshold"
P2 = "p2_abstention"
P3 = "p3_cost_aware"
POLICIES = (P0, P1, P2, P3)
FULL_AUTOMATION = (P0, P1)
THREE_WAY = (P2, P3)
POLICY_RULES: dict[str, tuple[str, str]] = {
    P0: (R_PLUG_IN, R_ALL_REST),
    P1: (R_PAGE_BOUND, R_ALL_REST),
    P2: (R_PAGE_BOUND, R_PAGE_BOUND),
    P3: (R_PLUG_IN, R_PLUG_IN),
}
POLICY_LABEL = {
    P0: (
        "full automation: one cutoff by RK1's plug-in rule; above it the correction is applied, "
        "below it the OCR text is kept; no human"
    ),
    P1: (
        "conservative full automation: one cutoff by the page-corrected Clopper-Pearson bound at "
        "delta 0.1; no human"
    ),
    P2: (
        "abstention: accept above the page-corrected bound cutoff, reject below a mirrored "
        "page-corrected bound cutoff on the exact-repair rate, and send the rest to a human"
    ),
    P3: (
        "cost-aware: accept above the plug-in cutoff and reject below the mirrored plug-in cutoff, "
        "which maximizes the share of decisions no human sees subject to harm <= 0.1 and the "
        "mirrored reject constraint, both estimated on the calibration pages"
    ),
}
POLICY_SHORT = {
    P0: "P0 full automation",
    P1: "P1 conservative",
    P2: "P2 abstention",
    P3: "P3 cost-aware",
}
PRIMARY_FULL = P1
PRIMARY_THREE_WAY = P2

OUTCOME_TAXONOMY = {
    "A": "conservative full automation is deployable within the measured page budgets",
    "B": (
        "full automation is not, but conservative abstention with human review is practical "
        "within the measured page budgets in both directions"
    ),
    "C": "a conservative policy is deployable or practical in one direction only",
    "D": "no conservative policy is deployable or practical within the measured page budgets",
}
NEXT_STAGE = {
    "A": "NEXT: CONFIRM CONSERVATIVE FULL AUTOMATION ON NEW PAGES",
    "B": "NEXT: CONFIRM HUMAN-IN-THE-LOOP TRIAGE ON NEW PAGES BEFORE ANY AUTOMATION CLAIM",
    "C": "NEXT: DIRECTION-SPECIFIC TRIAGE FOR THE GENERATOR THAT REACHED A SAFE POLICY",
    "D": "NEXT: EXPAND THE LABELLED PAGE POOL BEFORE ANY DEPLOYMENT POLICY",
}

PRIMARY_FAMILY = (
    "T1_any_violation_p2_vs_p3_at_pool",
    "T2_review_reduction_p3_vs_p2_at_pool",
    "T3_review_reduction_p2_pool_vs_10_pages",
    "T4_total_recall_p3_vs_p0_at_pool",
)
FAMILY_SPEC: dict[str, dict[str, str]] = {
    PRIMARY_FAMILY[0]: {
        "left": P2,
        "right": P3,
        "metric": "any_violation",
        "left_budget": POOL,
        "right_budget": POOL,
    },
    PRIMARY_FAMILY[1]: {
        "left": P3,
        "right": P2,
        "metric": "review_reduction",
        "left_budget": POOL,
        "right_budget": POOL,
    },
    PRIMARY_FAMILY[2]: {
        "left": P2,
        "right": P2,
        "metric": "review_reduction",
        "left_budget": POOL,
        "right_budget": DECISION_BUDGET,
    },
    PRIMARY_FAMILY[3]: {
        "left": P3,
        "right": P0,
        "metric": "total_recall",
        "left_budget": POOL,
        "right_budget": POOL,
    },
}

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_rk4": {
        "outcome": "B",
        "recommended_next_stage": "NEXT: EXPAND THE LABELLED PAGE POOL FOR THE ADAPTED RANKER",
        "ready_for_external_confirmation": False,
        "deployment_claim_permitted": False,
    },
    "sgv_rk3": {"outcome": "B", "deployment_claim_permitted": False},
    "sgv_th2": {"outcome": "D"},
    "sgv_th1": {"outcome": "D"},
    "sgv_ds1": {"outcome": "B"},
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_dep1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "production_ready": False,
        "certified": False,
        "confirmatory_reserve_consumed": False,
    }


def deployable(accept_violation_share: float | None, working_share: float | None) -> bool:
    """Full automation: TH1's definition on the accepts."""
    return rk4.deployable(accept_violation_share, working_share)


def practical(
    accept_violation_share: float | None,
    reject_violation_share: float | None,
    review_reduction: float | None,
    floor: float = REVIEW_REDUCTION_FLOOR,
) -> bool:
    """Three-way: both constraints held in all but a tenth of draws, and review reduced enough."""
    return bool(
        accept_violation_share is not None
        and reject_violation_share is not None
        and review_reduction is not None
        and accept_violation_share <= VIOLATION_CEILING
        and reject_violation_share <= VIOLATION_CEILING
        and review_reduction >= floor
    )


def assign_outcome(full: dict[str, str | None], three_way: dict[str, str | None]) -> str:
    """A: P1 deployable in both directions; B: P2 practical in both; C: either in one; D: none."""
    if all(full[d] is not None for d in DIRECTIONS):
        return "A"
    if all(three_way[d] is not None for d in DIRECTIONS):
        return "B"
    if any(full[d] is not None or three_way[d] is not None for d in DIRECTIONS):
        return "C"
    return "D"


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    files = list(rk4._upstream_files())
    files.append(REPO / "scripts/sgv_rk4_generator_adaptive_ranking.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(rk4.OUT.glob(pattern)))
    files.append(rk4.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    _check = rk3.xr1._check
    checks = list(rk4.upstream_checks())
    decisions = {
        "sgv_rk4": cc_read_json(rk4.DECISION),
        "sgv_rk3": cc_read_json(rk3.DECISION),
        "sgv_th2": cc_read_json(th2.DECISION),
        "sgv_th1": cc_read_json(th1.DECISION),
        "sgv_ds1": cc_read_json(ds1.DECISION),
    }
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    checks.append(
        _check("sgv_rk4.determinism", cc_read_json(rk4.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_rk4.report_traceability",
            cc_read_json(rk4.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
            0,
        )
    )
    for stage, decision in decisions.items():
        checks.append(
            _check(
                f"{stage}.confirmatory_reserve_consumed",
                decision.get("confirmatory_reserve_consumed"),
                False,
            )
        )
    return checks


def run_reconstruct() -> int:
    started = time.monotonic()
    _forbid(RESEARCH_FREEZE)
    checks = upstream_checks()
    failed = [c for c in checks if not c["agrees"]]
    if failed:
        raise PhaseError(
            f"{len(failed)} upstream checks failed: {[c['check'] for c in failed][:5]}"
        )
    print(f"reconstruct: {len(checks)} upstream checks pass ({time.monotonic() - started:.0f}s)")
    return 0


def _labels_exist() -> bool:
    return any(p.exists() for p in (POLICY_OUTCOMES, DECISION))


def run_freeze() -> int:
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("a DEP1 endpoint already exists; the freeze must precede every one")
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {_relative(p): file_sha256(p) for p in _upstream_files()}
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "upstream_sha256": hashes,
            "upstream_file_count": len(hashes),
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short"),
                "diff_stat": _git("diff", "--stat"),
            },
            "fits_a_model": False,
            "regenerates_a_candidate": False,
            "regenerates_ocr": False,
            "changes_an_upstream_outcome": False,
            "uses_ground_truth": False,
        },
    )
    rk4_decision = cc_read_json(rk4.DECISION)
    rk4_curves = cc_read_json(rk4.CURVES)["policies"]
    key = rl1.epsilon_key(EPSILON)
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "rk4_cell_scores": _relative(rk4.CELL_SCORES),
            "rk4_primary_arm": rk4_decision["primary_arm"],
            "rk4_outcome": rk4_decision["outcome"],
            "rk4_ranking_ready": rk4_decision["ranking_ready_primary"],
            "rk4_n_star_page_rule": rk4_decision["n_star_primary"],
            "rk4_naive_rule_at_pool": {
                d: {
                    k: rk4_curves[d][NEW_ARM][rk4.R_NAIVE][key][POOL][k]
                    for k in ("violation_share", "working_share", "median_recall")
                }
                for d in DIRECTIONS
            },
            "rk3_ranking_scores": _relative(rk3.RANKING_SCORES),
            "rk3_in_distribution_recall": cc_read_json(rk3.DEPLOYMENT)["by_protocol"][rk3.P_IN][
                rk3.M_RK3
            ][rk3.V_FULL][key]["repair_recall"],
            "ds1_recall_same_rule": cc_read_json(rk3.DEPLOYMENT)["by_protocol"][rk3.P_IN][
                rk3.M_DS1
            ][rk3.V_FULL][key]["repair_recall"],
            "th1_deployable_definition": {
                "violation_ceiling": VIOLATION_CEILING,
                "working_floor": WORKING_FLOOR,
                "delta": DELTA,
            },
            "th2_outcome": cc_read_json(th2.DECISION)["outcome"],
            "uses_ground_truth": False,
        },
    )
    print(f"freeze: {len(hashes)} upstream files hashed ({time.monotonic() - started:.0f}s)")
    return 0


def run_preregister() -> int:
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("a DEP1 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_questions": {
                "RQ1": "how does labelled page budget affect safe automation coverage?",
                "RQ2": (
                    "can abstention and human-review policies enable practical deployment before "
                    "full automation?"
                ),
                "RQ3": (
                    "what is the tradeoff between automation rate, harm rate and annotation cost?"
                ),
            },
            "primary_hypothesis": (
                "before any page budget in this universe supports safe full automation, a "
                "three-way policy that rejects the clearly unsafe decisions and reviews the "
                "uncertain ones removes a useful share of the human workload without automated "
                "harm above 0.1 and without discarding exact repairs above the mirrored rate"
            ),
            "fits_a_model": False,
            "scenarios": {
                SCENARIO_NEW: (
                    f"SGV-RK4's primary arm ({NEW_ARM}) on the held-out generator, by labelled "
                    "page budget and draw, with RK4's held-back calibration pages choosing every "
                    "cutoff and RK4's sealed test pages evaluating; budget 0 is RK3's zero-shot "
                    "ranker with the source calibration block"
                ),
                SCENARIO_KNOWN: (
                    "SGV-RK3's primary ranker and SGV-DS1's reliability score on the generators "
                    "they were fitted on: DS1's threshold block chooses cutoffs, its test block "
                    "evaluates. One realization, reported as the reference"
                ),
            },
            "budgets": {
                "measured": list(BUDGETS),
                "not_measurable": list(NOT_MEASURABLE),
                "draws": DRAWS,
                "source": "SGV-RK4's purchases, unchanged",
            },
            "decisions": (
                "one per site: the candidate with the highest score among the site's scored "
                "candidates, the candidate id breaking ties, as every SGV stage since DS1"
            ),
            "policies": POLICY_LABEL,
            "policy_rules": {p: {"accept": a, "reject": r} for p, (a, r) in POLICY_RULES.items()},
            "constraints": {
                "harm": f"selective harm among automatic accepts at most {EPSILON}",
                "reject": (
                    f"share of automatic rejects that carry an exact repair at most {ETA}: a "
                    "rejected decision may carry an exact repair no more often than an accepted "
                    "one may carry harm"
                ),
                "mirrored_cutoff": (
                    "the reject cutoff is the accept rule applied to negated scores with exact "
                    "repairs in place of harm; for the page-corrected bound, the design effect is "
                    "that of the exact-repair rate across calibration pages"
                ),
                "precedence": (
                    "a decision at or above the accept cutoff is accepted, never rejected"
                ),
                "delta": DELTA,
            },
            "human_reviewer": (
                "an idealization: a reviewed decision is corrected when it is an exact repair and "
                "left as OCR otherwise. No human was measured; recall through review is an upper "
                "bound on what a reviewer could recover"
            ),
            "metrics": {
                "acceptance_coverage": "share of decision sites corrected automatically",
                "automation_rate": "share of decision sites decided without a human",
                "review_rate": "share of decision sites sent to a human",
                "review_reduction": "one minus the review rate: the work taken off a reviewer",
                "harm": "selective harm among automatic accepts",
                "automated_recall": (
                    "error sites repaired by automatic accepts, over every test error site"
                ),
                "review_recall": "error sites whose exact repair sits in the review band",
                "lost_recall": "error sites whose exact repair is rejected automatically",
                "total_recall": "automated plus review recall, under the idealized reviewer",
                "label_cost": "labelled target pages",
            },
            "definitions": {
                "accept_violation": "a draw accepts something and its test harm exceeds 0.1",
                "reject_violation": (
                    "a three-way draw rejects something and the test share of rejects carrying an "
                    "exact repair exceeds 0.1"
                ),
                "works": "a draw accepts something, holds 0.1 and repairs an error site",
                "deployable": (
                    f"full automation: accept-violation share at most {VIOLATION_CEILING} and "
                    f"working share at least {WORKING_FLOOR} over the draws, TH1's definition"
                ),
                "practical": (
                    f"three-way: accept- and reject-violation shares each at most "
                    f"{VIOLATION_CEILING}, and median review reduction at least "
                    f"{REVIEW_REDUCTION_FLOOR}"
                ),
                "review_reduction_floor": (
                    "a quarter of the decision sites taken off the reviewer. It is a convention, "
                    "not a derived value, so the practical frontier is also reported at "
                    f"{list(REVIEW_REDUCTION_SENSITIVITY)}"
                ),
                "n_star": (
                    "the smallest measured budget from which the flag holds at every larger "
                    "measured budget"
                ),
            },
            "criteria": {
                "F1": (
                    f"{PRIMARY_FULL} is deployable within the measured budgets in both directions"
                ),
                "F2": (
                    f"{PRIMARY_THREE_WAY} is practical within the measured budgets in both "
                    "directions"
                ),
                "deployment_claim": (
                    "a claim is permitted only for a conservative policy whose criterion is met, "
                    "and only for the mode it tested: full automation for F1, triage with human "
                    "review for F2"
                ),
            },
            "outcome_rule": OUTCOME_TAXONOMY,
            "next_stage_rule": NEXT_STAGE,
            "statistical_family": {name: FAMILY_SPEC[name] for name in PRIMARY_FAMILY},
            "statistical_plan": {
                "unit": "draw, paired, both directions pooled",
                "test": "exact two-sided sign test; tied draws carry no information",
                "effect": "median paired difference, or the difference in violation shares",
                "interval": "draw resamples stratified by direction",
                "resamples": BOOTSTRAP_RESAMPLES,
                "multiplicity": "Holm within the frozen family of four",
                "scope": "conditional on the sealed test pages",
            },
            "disclosures": {
                "known_from_rk4": (
                    "P0's and P1's accept sides are RK4's plug-in and page-bound rules on RK4's "
                    "own "
                    "cells, so their accept outcomes were published by RK4 before this record: the "
                    "plug-in rule met TH1's definition only at the pool, and the page bound "
                    "accepted nothing for the primary arm. F1's answer is therefore known in "
                    "advance; the three-way policies and the reject constraint are new"
                ),
            },
            "controls": {
                "random": "seeded random scores through every policy at every adapted budget",
                "random_rule": (
                    "if random scores are practical at the primary review-reduction floor at a "
                    "budget and direction, that budget's practical flag for the same policy is not "
                    "counted: a reject band that works with random scores measures a low base rate "
                    "of exact repairs, not the ranker"
                ),
            },
            "non_goals": [
                "no model is fitted and no score is changed",
                "no candidate is generated and no OCR output is regenerated",
                "no page budget beyond the pool is projected",
                "no human reviewer is measured or simulated beyond the stated idealization",
                "the confirmatory reserve is not touched",
                "no certification, production claim or external confirmation",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ decisions

KNOWN_UNIT = "known"
KNOWN_BUDGET = "fit_block"
SET_CALIBRATION = "calibration"
SET_TEST = "test"


def _top_per_site(frame: pd.DataFrame) -> pd.DataFrame:
    return ds1.top_per_site(frame, "rank_score", False).reset_index(drop=True)


def decision_rows(population: pd.DataFrame) -> pd.DataFrame:
    """One decision per site for every frozen cell. Reads scores and site keys, never a label."""
    sites = population.set_index("candidate_id")[["site_key", "document_id"]]
    frames: list[pd.DataFrame] = []
    scores = pd.read_parquet(rk4.CELL_SCORES)
    new = scores[
        ((scores["arm"] == NEW_ARM) | (scores["arm"] == ZERO_ARM))
        & scores["evaluation_set"].isin([SET_CALIBRATION, SET_TEST])
    ]
    for (direction, arm, budget, draw, kind), group in new.groupby(
        ["direction", "arm", "budget", "draw", "evaluation_set"], sort=True
    ):
        joined = group[["candidate_id", "score"]].join(sites, on="candidate_id", how="inner")
        joined = joined.assign(rank_score=joined["score"], safety=joined["score"])
        top = _top_per_site(joined)
        frames.append(
            pd.DataFrame(
                {
                    "scenario": SCENARIO_NEW,
                    "unit": str(direction),
                    "model": str(arm),
                    "budget": str(budget),
                    "draw": int(draw),
                    "evaluation_set": str(kind),
                    "candidate_id": top["candidate_id"].astype(str).to_numpy(),
                    "site_key": top["site_key"].astype(str).to_numpy(),
                    "document_id": top["document_id"].astype(str).to_numpy(),
                    "rank_score": top["rank_score"].to_numpy(np.float64),
                    "safety": top["safety"].to_numpy(np.float64),
                }
            )
        )
    known = pd.read_parquet(rk3.RANKING_SCORES)
    known = known[
        (known["protocol"] == rk3.P_IN)
        & known["model"].isin(KNOWN_MODELS)
        & (known["variant"] == rk3.V_FULL)
    ]
    for (model, kind), group in known.groupby(["model", "evaluation_set"], sort=True):
        joined = group[["candidate_id", "rank_score", "safety"]].join(
            sites, on="candidate_id", how="inner"
        )
        top = _top_per_site(joined)
        frames.append(
            pd.DataFrame(
                {
                    "scenario": SCENARIO_KNOWN,
                    "unit": KNOWN_UNIT,
                    "model": str(model),
                    "budget": KNOWN_BUDGET,
                    "draw": 0,
                    "evaluation_set": SET_CALIBRATION if kind == "threshold" else SET_TEST,
                    "candidate_id": top["candidate_id"].astype(str).to_numpy(),
                    "site_key": top["site_key"].astype(str).to_numpy(),
                    "document_id": top["document_id"].astype(str).to_numpy(),
                    "rank_score": top["rank_score"].to_numpy(np.float64),
                    "safety": top["safety"].to_numpy(np.float64),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(
        ["scenario", "unit", "model", "budget", "draw", "evaluation_set", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def run_cells() -> int:
    """Every frozen cell's decisions, label-free, written once."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (DECISIONS, CELL_REGISTRY):
        _forbid(path)
    table = decision_rows(rk4.load_population())
    _write_parquet_once(DECISIONS, table)
    counts = (
        table.groupby(["scenario", "unit", "model", "budget", "evaluation_set"])["draw"]
        .agg(["nunique", "size"])
        .reset_index()
    )
    _write_json_once(
        CELL_REGISTRY,
        {
            **_envelope("cell_registry"),
            "cells": [
                {
                    "scenario": r.scenario,
                    "unit": r.unit,
                    "model": r.model,
                    "budget": r.budget,
                    "evaluation_set": r.evaluation_set,
                    "draws": int(r.nunique),
                    "decisions": int(r.size),
                }
                for r in counts.itertuples(index=False)
            ],
            "decision_rows": len(table),
            "reads_labels": False,
            "rule": cc_read_json(DESIGN_RECORD)["decisions"],
        },
    )
    print(f"cells: {len(table)} decisions ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the policies


def accept_cutoff(rule: str, calibration: pd.DataFrame, epsilon: float) -> float | None:
    """The loosest cutoff a rule allows on calibration decisions; None when it allows none."""
    if calibration.empty:
        return None
    if rule == R_PLUG_IN:
        return rk1.choose_threshold(calibration, epsilon)
    if rule == R_PAGE_BOUND:
        deff = th1.page_design_effect(calibration)
        if deff is None:
            return None
        cutoff, _bound = th1.bound_threshold(calibration, epsilon, DELTA, deff)
        return cutoff
    raise PhaseError(f"unknown accept rule {rule}")


def mirrored(calibration: pd.DataFrame) -> pd.DataFrame:
    """Negated scores with exact repairs in place of harm: the reject side as an accept problem."""
    return calibration.assign(
        safety=-calibration["safety"].astype(np.float64),
        is_harmful=calibration["exact"].astype(bool),
    )


def reject_cutoff(rule: str, calibration: pd.DataFrame, eta: float) -> float | None:
    """The highest score at or below which decisions may be rejected; None when none may be."""
    cutoff = accept_cutoff(rule, mirrored(calibration), eta)
    return None if cutoff is None else -cutoff


def bands(
    policy: str, calibration: pd.DataFrame, test: pd.DataFrame, usable: bool = True
) -> dict[str, Any]:
    """A policy's accept, review and reject masks on the test decisions, and its two cutoffs."""
    accept_rule, reject_rule = POLICY_RULES[policy]
    safety = test["safety"].to_numpy(np.float64)
    high = accept_cutoff(accept_rule, calibration, EPSILON) if usable else None
    accept = safety >= high if high is not None else np.zeros(safety.size, dtype=bool)
    low: float | None = None
    if reject_rule == R_ALL_REST:
        reject = ~accept
    else:
        low = reject_cutoff(reject_rule, calibration, ETA) if usable else None
        reject = (
            (~accept) & (safety <= low) if low is not None else np.zeros(safety.size, dtype=bool)
        )
    return {
        "accept": accept,
        "reject": reject,
        "review": ~accept & ~reject,
        "accept_cutoff": high,
        "reject_cutoff": low,
    }


def evaluate_policy(
    policy: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    keys: set[str],
    mapping: dict[str, set[str]],
    errors: pd.DataFrame,
    usable: bool = True,
) -> dict[str, Any]:
    """One policy on one cell: cutoffs from calibration decisions, outcomes on test decisions."""
    b = bands(policy, calibration, test, usable)
    accepted, rejected, reviewed = test[b["accept"]], test[b["reject"]], test[b["review"]]
    n = len(test)

    def repaired(part: pd.DataFrame) -> set[str]:
        return ds1._repaired_flags(test, part, errors, mapping)

    auto, review, lost = repaired(accepted), repaired(reviewed), repaired(rejected)
    harm = _share(int(accepted["is_harmful"].sum()), len(accepted))
    reject_exact = _share(int(rejected["exact"].sum()), len(rejected))
    accept_violation = bool(
        len(accepted) > 0 and harm is not None and harm > EPSILON + HARM_TOLERANCE
    )
    reject_violation = bool(
        policy in THREE_WAY
        and len(rejected) > 0
        and reject_exact is not None
        and reject_exact > ETA + HARM_TOLERANCE
    )
    denominator = max(len(keys), 1)
    return {
        "policy": policy,
        "decisions": n,
        "accepted": len(accepted),
        "reviewed": len(reviewed),
        "rejected": len(rejected),
        "accept_cutoff": b["accept_cutoff"],
        "reject_cutoff": b["reject_cutoff"],
        "acceptance_coverage": _share(len(accepted), n),
        "automation_rate": _share(len(accepted) + len(rejected), n),
        "review_rate": _share(len(reviewed), n),
        "review_reduction": None if n == 0 else 1.0 - len(reviewed) / n,
        "harm": harm,
        "joint_harm": _share(int(accepted["is_harmful"].sum()), n),
        "reject_exact_rate": reject_exact,
        "automated_recall": len(auto) / denominator,
        "review_recall": len(review) / denominator,
        "lost_recall": len(lost) / denominator,
        "total_recall": len(auto | review) / denominator,
        "repaired_automatically": len(auto),
        "accept_violation": accept_violation,
        "reject_violation": reject_violation,
        "any_violation": bool(accept_violation or reject_violation),
        "works": bool(len(accepted) > 0 and not accept_violation and len(auto) >= 1),
        "safe_coverage": 0.0 if accept_violation else (_share(len(accepted), n) or 0.0),
    }


@dataclass(slots=True)
class Evaluation:
    """Labelled decisions per cell, and the recall context, loaded once."""

    cells: dict[tuple[str, str, str, str, int], dict[str, pd.DataFrame]]
    keys: set[str]
    mapping: dict[str, set[str]]
    errors: pd.DataFrame


def load_evaluation() -> Evaluation:
    population = rk4.load_population()
    labels = population.set_index("candidate_id")[["is_harmful", "exact", "beneficial"]]
    cells: dict[tuple[str, str, str, str, int], dict[str, pd.DataFrame]] = {}
    for key, group in pd.read_parquet(DECISIONS).groupby(
        ["scenario", "unit", "model", "budget", "draw", "evaluation_set"], sort=True
    ):
        scenario, unit, model, budget, draw, kind = key
        joined = group.join(labels, on="candidate_id", how="inner")
        if len(joined) != len(group):
            raise PhaseError(f"{key}: a decision has no label row")
        cells.setdefault((str(scenario), str(unit), str(model), str(budget), int(draw)), {})[
            str(kind)
        ] = joined.reset_index(drop=True)
    errors = ds1.load_error_population()
    test_errors = errors[errors["block"] == "test"].reset_index(drop=True)
    return Evaluation(
        cells=cells,
        keys=set(test_errors["error_key"].astype(str)),
        mapping=ds1.site_to_errors(errors),
        errors=test_errors,
    )


def run_policies() -> int:
    """Every policy on every cell."""
    started = time.monotonic()
    _require(CELL_REGISTRY, "cells")
    for path in (POLICY_OUTCOMES, POLICY_REGISTRY):
        _forbid(path)
    evaluation = load_evaluation()
    rows: list[dict[str, Any]] = []
    for key, cell in sorted(evaluation.cells.items()):
        scenario, unit, model, budget, draw = key
        test = cell[SET_TEST]
        usable = not rk2._constant(test["safety"].to_numpy(np.float64))
        for policy in POLICIES:
            rows.append(
                {
                    "scenario": scenario,
                    "unit": unit,
                    "model": model,
                    "budget": budget,
                    "draw": draw,
                    **evaluate_policy(
                        policy,
                        cell.get(SET_CALIBRATION, test.head(0)),
                        test,
                        evaluation.keys,
                        evaluation.mapping,
                        evaluation.errors,
                        usable,
                    ),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["scenario", "unit", "model", "budget", "draw", "policy"], kind="stable"
    )
    _write_parquet_once(POLICY_OUTCOMES, table.reset_index(drop=True))
    _write_json_once(
        POLICY_REGISTRY,
        {
            **_analysis_envelope("policy_registry"),
            "policies": POLICY_LABEL,
            "rules": {p: list(r) for p, r in POLICY_RULES.items()},
            "cells": len(evaluation.cells),
            "rows": len(table),
            "epsilon": EPSILON,
            "eta": ETA,
            "delta": DELTA,
            "recall_denominator": len(evaluation.keys),
            "cutoffs_chosen_without_test_labels": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"policies: {len(table)} outcomes on {len(evaluation.cells)} cells "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ curves and frontier


def _median(values: Sequence[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.median(finite)) if finite else None


def cell_rows(table: pd.DataFrame, unit: str, budget: str, policy: str) -> pd.DataFrame:
    """A new-generator budget's rows for one policy; budget 0 is the zero-shot ranker."""
    model = ZERO_ARM if budget == "0" else NEW_ARM
    return table[
        (table["scenario"] == SCENARIO_NEW)
        & (table["unit"] == unit)
        & (table["model"] == model)
        & (table["budget"] == budget)
        & (table["policy"] == policy)
    ]


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    accepting = rows[rows["accepted"] > 0]
    rejecting = rows[rows["rejected"] > 0]

    def med(column: str, frame: pd.DataFrame = rows) -> float | None:
        return _median(frame[column].tolist())

    return {
        "draws": len(rows),
        "accept_violation_share": float(rows["accept_violation"].mean()) if len(rows) else None,
        "reject_violation_share": float(rows["reject_violation"].mean()) if len(rows) else None,
        "any_violation_share": float(rows["any_violation"].mean()) if len(rows) else None,
        "working_share": float(rows["works"].mean()) if len(rows) else None,
        "median_acceptance_coverage": med("acceptance_coverage"),
        "median_automation_rate": med("automation_rate"),
        "median_review_rate": med("review_rate"),
        "median_review_reduction": med("review_reduction"),
        "median_safe_coverage": med("safe_coverage"),
        "median_automated_recall": med("automated_recall"),
        "median_review_recall": med("review_recall"),
        "median_lost_recall": med("lost_recall"),
        "median_total_recall": med("total_recall"),
        "median_harm": med("harm", accepting),
        "median_reject_exact_rate": med("reject_exact_rate", rejecting),
        "accepting_draws": len(accepting),
        "rejecting_draws": len(rejecting),
    }


def flags(summary: dict[str, Any], policy: str) -> dict[str, bool]:
    if policy in FULL_AUTOMATION:
        return {
            "deployable": deployable(summary["accept_violation_share"], summary["working_share"])
        }
    return {
        f"practical_at_{floor}": practical(
            summary["accept_violation_share"],
            summary["reject_violation_share"],
            summary["median_review_reduction"],
            floor,
        )
        for floor in REVIEW_REDUCTION_SENSITIVITY
    }


def run_curves() -> int:
    """Label-budget curves for the new generator, and the known-generator reference."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(CURVES)
    table = pd.read_parquet(POLICY_OUTCOMES)
    new: dict[str, Any] = {}
    for unit in DIRECTIONS:
        for policy in POLICIES:
            for budget in BUDGETS:
                summary = summarize(cell_rows(table, unit, budget, policy))
                new.setdefault(unit, {}).setdefault(policy, {})[budget] = summary | flags(
                    summary, policy
                )
    known: dict[str, Any] = {}
    for model in KNOWN_MODELS:
        for policy in POLICIES:
            rows = table[
                (table["scenario"] == SCENARIO_KNOWN)
                & (table["model"] == model)
                & (table["policy"] == policy)
            ]
            row = rows.iloc[0].to_dict()
            known.setdefault(model, {})[policy] = {
                k: (
                    None
                    if isinstance(v, float) and np.isnan(v)
                    else (v.item() if hasattr(v, "item") else v)
                )
                for k, v in row.items()
                if k not in ("scenario", "unit", "model", "budget", "draw", "policy")
            }
    _write_json_once(
        CURVES,
        {
            **_analysis_envelope("label_budget_curves"),
            "new_generator": new,
            "known_generators": known,
            "known_models": KNOWN_LABEL,
            "budgets": list(BUDGETS),
            "pool_pages": cc_read_json(rk4.CURVES)["pool_pages"],
            "zero_budget_note": (
                "budget 0 is RK3's zero-shot ranker with the source calibration block"
            ),
        },
    )
    p2 = {d: new[d][P2][POOL]["median_review_reduction"] for d in DIRECTIONS}
    print(
        f"curves: P2 median review reduction at the pool {p2} ({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_frontier() -> int:
    """N* per policy: deployable for full automation, practical for the three-way policies."""
    started = time.monotonic()
    _require(CURVES, "curves")
    _forbid(FRONTIER)
    curves = cc_read_json(CURVES)["new_generator"]
    chance = set(cc_read_json(CONTROL_RESULTS)["random_practical_cells"])
    n_star: dict[str, Any] = {}
    unmasked: dict[str, Any] = {}
    for unit in DIRECTIONS:
        for policy in POLICIES:
            names = (
                ["deployable"]
                if policy in FULL_AUTOMATION
                else [f"practical_at_{f}" for f in REVIEW_REDUCTION_SENSITIVITY]
            )
            for name in names:
                raw = {b: curves[unit][policy][b][name] for b in BUDGETS}
                masked = {
                    b: flag and f"{unit}|{policy}|{b}" not in chance for b, flag in raw.items()
                }
                n_star.setdefault(unit, {}).setdefault(policy, {})[name] = rk4.minimal_budget(
                    masked
                )
                unmasked.setdefault(unit, {}).setdefault(policy, {})[name] = rk4.minimal_budget(raw)
    primary = {
        "full_automation": {d: n_star[d][PRIMARY_FULL]["deployable"] for d in DIRECTIONS},
        "three_way": {
            d: n_star[d][PRIMARY_THREE_WAY][f"practical_at_{REVIEW_REDUCTION_FLOOR}"]
            for d in DIRECTIONS
        },
    }
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("deployment_frontier"),
            "n_star": n_star,
            "n_star_before_the_random_control": unmasked,
            "random_practical_cells": sorted(chance),
            "primary": primary,
            "definitions": cc_read_json(DESIGN_RECORD)["definitions"],
            "measured_budgets": list(BUDGETS),
            "not_measurable": list(NOT_MEASURABLE),
        },
    )
    print(f"frontier: primary {primary} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ tradeoff curves

TRADEOFF_BUDGETS = ("0", "10", POOL)
REJECT_GRID = rk3.HARM_GRID


def reject_sweep(test: pd.DataFrame, keys: set[str], mapping: dict[str, set[str]]) -> pd.DataFrame:
    """ANALYSIS ONLY: every reject cutoff from the bottom: share rejected, exact rate, loss."""
    ordered = test.sort_values(["safety", "candidate_id"], ascending=[True, True], kind="stable")
    safety = ordered["safety"].to_numpy(np.float64)
    exact = ordered["exact"].to_numpy(bool)
    lost: set[str] = set()
    counts = np.zeros(len(ordered), dtype=np.int64)
    for i, (site, is_exact) in enumerate(zip(ordered["site_key"].astype(str), exact, strict=True)):
        if is_exact:
            lost |= mapping.get(site, set()) & keys
        counts[i] = len(lost)
    rejected = np.arange(1, len(ordered) + 1)
    ends = np.flatnonzero(np.append(safety[1:] != safety[:-1], True))
    return pd.DataFrame(
        {
            "reject_share": rejected[ends] / max(len(ordered), 1),
            "exact_rate": np.cumsum(exact)[ends] / rejected[ends],
            "lost_recall": counts[ends] / max(len(keys), 1),
        }
    )


def reject_frontier(points: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for level in REJECT_GRID:
        within = points[points["exact_rate"] <= level + 1e-12]
        out.append(
            {
                "exact_rate": level,
                "reject_share": float(within["reject_share"].max()) if len(within) else 0.0,
            }
        )
    return out


def run_tradeoffs() -> int:
    """ANALYSIS ONLY: harm-coverage and reject frontiers read off the test labels, over draws."""
    started = time.monotonic()
    _require(FRONTIER, "frontier")
    _forbid(TRADEOFFS)
    evaluation = load_evaluation()
    out: dict[str, Any] = {}
    targets = [
        (SCENARIO_NEW, d, ZERO_ARM if b == "0" else NEW_ARM, b)
        for d in DIRECTIONS
        for b in TRADEOFF_BUDGETS
    ] + [(SCENARIO_KNOWN, KNOWN_UNIT, m, KNOWN_BUDGET) for m in KNOWN_MODELS]
    for scenario, unit, model, budget in targets:
        draws = range(1) if (scenario == SCENARIO_KNOWN or budget == "0") else range(DRAWS)
        coverage_curves, harm_frontiers, reject_frontiers = [], [], []
        for draw in draws:
            test = evaluation.cells[(scenario, unit, model, budget, draw)][SET_TEST]
            summary = rk3.curve_summary(rk3.sweep(test, evaluation.keys, evaluation.mapping))
            coverage_curves.append(summary["coverage_curve"])
            harm_frontiers.append(summary["frontier"])
            reject_frontiers.append(
                reject_frontier(reject_sweep(test, evaluation.keys, evaluation.mapping))
            )
        key = f"{unit}|{budget}" if scenario == SCENARIO_NEW else f"{KNOWN_UNIT}|{model}"
        out[key] = {
            "scenario": scenario,
            "model": model,
            "draws": len(coverage_curves),
            "harm_vs_coverage": [
                {
                    "coverage_grid": row["grid"],
                    "median_selective_harm": _median(
                        [c[i]["selective_harm"] for c in coverage_curves]
                    ),
                    "median_recall": _median([c[i]["recall"] for c in coverage_curves]),
                }
                for i, row in enumerate(coverage_curves[0])
            ],
            "automation_frontier": [
                {
                    "harm": row["harm"],
                    "median_acceptance_coverage": _median(
                        [f[i]["coverage"] for f in harm_frontiers]
                    ),
                    "median_recall": _median([f[i]["recall"] for f in harm_frontiers]),
                }
                for i, row in enumerate(harm_frontiers[0])
            ],
            "reject_frontier": [
                {
                    "exact_rate": row["exact_rate"],
                    "median_reject_share": _median(
                        [f[i]["reject_share"] for f in reject_frontiers]
                    ),
                }
                for i, row in enumerate(reject_frontiers[0])
            ],
        }
    _write_json_once(
        TRADEOFFS,
        {
            **_analysis_envelope("tradeoff_curves"),
            "by_cell": out,
            "analysis_only": (
                "every curve sweeps the test decisions over every distinct cutoff and reads the "
                "test "
                "labels; none chooses a cutoff and no criterion reads one"
            ),
            "coverage_grid": list(rk3.COVERAGE_GRID),
            "harm_grid": list(rk3.HARM_GRID),
        },
    )
    print(f"tradeoffs: {len(out)} cells ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ controls


def run_controls() -> int:
    """Seeded random scores through every policy at every adapted budget."""
    started = time.monotonic()
    _require(POLICY_REGISTRY, "policies")
    _forbid(CONTROL_RESULTS)
    evaluation = load_evaluation()
    rows = []
    for unit in DIRECTIONS:
        for budget in rk4.ADAPTED_BUDGETS:
            for draw in range(DRAWS):
                cell = evaluation.cells[(SCENARIO_NEW, unit, NEW_ARM, budget, draw)]
                generator = np.random.default_rng(
                    int(
                        canonical_hash({"seed": CONTROL_SEED, "cell": [unit, budget, draw]})[:8], 16
                    )
                )
                test = cell[SET_TEST].assign(safety=generator.random(len(cell[SET_TEST])))
                calibration = cell[SET_CALIBRATION].assign(
                    safety=generator.random(len(cell[SET_CALIBRATION]))
                )
                for policy in POLICIES:
                    rows.append(
                        {"unit": unit, "budget": budget, "draw": draw}
                        | evaluate_policy(
                            policy,
                            calibration,
                            test,
                            evaluation.keys,
                            evaluation.mapping,
                            evaluation.errors,
                        )
                    )
    table = pd.DataFrame(rows)
    summary = {
        unit: {
            policy: {
                budget: summarize(
                    table[
                        (table["unit"] == unit)
                        & (table["policy"] == policy)
                        & (table["budget"] == budget)
                    ]
                )
                for budget in rk4.ADAPTED_BUDGETS
            }
            for policy in POLICIES
        }
        for unit in DIRECTIONS
    }
    practical_anywhere = [
        f"{u}|{p}|{b}"
        for u in DIRECTIONS
        for p in THREE_WAY
        for b in rk4.ADAPTED_BUDGETS
        if practical(
            summary[u][p][b]["accept_violation_share"],
            summary[u][p][b]["reject_violation_share"],
            summary[u][p][b]["median_review_reduction"],
        )
    ]
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random": summary,
            "random_practical_cells": practical_anywhere,
            "random_any_violation_share_overall": float(table["any_violation"].mean()),
            "seed": CONTROL_SEED,
        },
    )
    print(
        f"controls: random scores practical in {len(practical_anywhere)} cells "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ statistics


def _value(
    table: pd.DataFrame, unit: str, policy: str, budget: str, draw: int, metric: str
) -> float:
    rows = cell_rows(table, unit, budget, policy)
    rows = rows[rows["draw"] == draw]
    if rows.empty or pd.isna(rows.iloc[0][metric]):
        return float("nan")
    return float(rows.iloc[0][metric])


def draw_comparison(name: str, spec: dict[str, str], table: pd.DataFrame) -> dict[str, Any]:
    """Paired over draws of both directions: sign test and a direction-stratified bootstrap."""
    a, b, direction = [], [], []
    for unit in DIRECTIONS:
        for draw in range(DRAWS):
            left = _value(table, unit, spec["left"], spec["left_budget"], draw, spec["metric"])
            right = _value(table, unit, spec["right"], spec["right_budget"], draw, spec["metric"])
            if np.isfinite(left) and np.isfinite(right):
                a.append(left)
                b.append(right)
                direction.append(unit)
    left_values, right_values, strata_of = np.asarray(a), np.asarray(b), np.asarray(direction)
    shares = spec["metric"] == "any_violation"

    def effect(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(x.mean() - y.mean()) if shares else float(np.median(x - y))

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(strata_of == d) for d in sorted(set(strata_of.tolist()))]
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for resample in range(BOOTSTRAP_RESAMPLES):
        index = (
            np.concatenate([generator.choice(s, s.size, replace=True) for s in strata])
            if strata
            else np.empty(0, dtype=np.int64)
        )
        draws[resample] = effect(left_values[index], right_values[index])
    p, positive, negative = th1.sign_test(left_values - right_values)
    finite = draws[np.isfinite(draws)]
    return {
        "comparison": name,
        **spec,
        "units": int(left_values.size),
        "effect": effect(left_values, right_values) if left_values.size else None,
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else None,
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else None,
        "p_value": p,
        "draws_left_higher": positive,
        "draws_right_higher": negative,
        "ties": int(left_values.size - positive - negative),
    }


def run_stats() -> int:
    started = time.monotonic()
    _require(TRADEOFFS, "tradeoffs")
    _forbid(STATISTICAL_TESTS)
    table = pd.read_parquet(POLICY_OUTCOMES)
    family = {name: draw_comparison(name, FAMILY_SPEC[name], table) for name in PRIMARY_FAMILY}
    adjusted = s15.holm(family)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "family": adjusted,
            "family_size": len(adjusted),
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "plan": cc_read_json(DESIGN_RECORD)["statistical_plan"],
        },
    )
    print(
        f"stats: {sum(1 for v in adjusted.values() if v['survives_holm'])}/{len(adjusted)} survive "
        f"Holm ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ falsification

LABEL_FIELDS = rk3.LABEL_FIELDS
FIT_CALLS = frozenset(
    {"fit_boosted", "fit_lambdamart", "fit_logistic", "fit_arm", "fit_model", "LogisticRegression"}
)


def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def _functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _mentions(function_name: str) -> set[str]:
    functions = _functions()
    seen: set[str] = set()
    names: set[str] = set()
    stack = [function_name]
    while stack:
        current = stack.pop()
        if current in seen or current not in functions:
            continue
        seen.add(current)
        for node in ast.walk(functions[current]):
            if isinstance(node, ast.Name):
                names.add(node.id)
                stack.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
    return names


def _called() -> set[str]:
    return {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
    }


def _cutoff_arguments() -> list[str]:
    """What `bands` hands the two cutoff functions as the decisions they are chosen on."""
    found: list[str] = []
    for node in ast.walk(_functions()["bands"]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("accept_cutoff", "reject_cutoff")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        ):
            found.append(node.args[1].id)
    return found


def toy_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A hand-built calibration and test set whose three-way bands can be read by eye."""
    calibration = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(10)],
            "document_id": ["p"] * 5 + ["q"] * 5,
            "safety": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "is_harmful": [False, False, False, False, True, True, True, True, True, True],
            "exact": [True, True, False, True, False, False, False, False, False, False],
        }
    )
    test = pd.DataFrame(
        {
            "candidate_id": [f"t{i}" for i in range(6)],
            "document_id": ["r"] * 6,
            "site_key": [f"s{i}" for i in range(6)],
            "safety": [9.5, 7.5, 6.5, 5.5, 1.5, 0.5],
            "is_harmful": [False, False, True, True, True, True],
            "exact": [True, False, False, False, False, False],
        }
    )
    return calibration, test


def run_negative() -> int:
    """Every claim the stage makes, with a test that would fail if the claim were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    tests: list[dict[str, Any]] = []

    def record(name: str, claim: str, passed: bool, evidence: Any) -> None:
        tests.append({"test": name, "claim": claim, "passed": bool(passed), "evidence": evidence})

    table = pd.read_parquet(POLICY_OUTCOMES)
    decisions = pd.read_parquet(DECISIONS)
    evaluation = load_evaluation()
    rk4_policies = pd.read_parquet(rk4.POLICY_OUTCOMES)

    flipped_same = True
    for unit in DIRECTIONS:
        cell = evaluation.cells[(SCENARIO_NEW, unit, NEW_ARM, POOL, 0)]
        test = cell[SET_TEST]
        inverted = test.assign(is_harmful=~test["is_harmful"], exact=~test["exact"])
        for policy in POLICIES:
            honest = bands(policy, cell[SET_CALIBRATION], test)
            flipped = bands(policy, cell[SET_CALIBRATION], inverted)
            flipped_same &= (honest["accept_cutoff"], honest["reject_cutoff"]) == (
                flipped["accept_cutoff"],
                flipped["reject_cutoff"],
            )
    record(
        "f01_cutoffs_read_calibration_decisions_only",
        "both cutoffs are chosen on calibration decisions, and inverting every test label leaves "
        "them unchanged",
        _cutoff_arguments() == ["calibration", "calibration"] and flipped_same,
        {"cutoff_arguments": _cutoff_arguments()},
    )

    def against_rk4(policy: str, rule: str) -> dict[str, Any]:
        ours = table[(table["scenario"] == SCENARIO_NEW) & (table["policy"] == policy)]
        theirs = rk4_policies[
            (rk4_policies["rule"] == rule)
            & (rk4_policies["epsilon"] == EPSILON)
            & rk4_policies["arm"].isin([NEW_ARM, ZERO_ARM])
        ]
        merged = ours.merge(
            theirs,
            left_on=["unit", "model", "budget", "draw"],
            right_on=["direction", "arm", "budget", "draw"],
            suffixes=("", "_rk4"),
        )
        same = (
            (merged["accepted"] == merged["accepted_rk4"])
            & (merged["accept_violation"] == merged["violation"])
            & (merged["works"] == merged["working"])
            & np.isclose(merged["automated_recall"], merged["repair_recall"], atol=0.0)
        )
        return {
            "cells": len(merged),
            "expected": len(ours),
            "identical": bool(same.all() and len(merged) == len(ours)),
        }

    p0 = against_rk4(P0, rk4.R_NAIVE)
    p1 = against_rk4(P1, rk4.R_PAGE)
    record(
        "f02_p0_accept_side_is_rk4s_plug_in_rule",
        "on every new-generator cell, P0's accepts, violations, working draws and automated recall "
        "equal RK4's plug-in rule outcomes",
        p0["identical"],
        p0,
    )
    record(
        "f03_p1_accept_side_is_rk4s_page_bound",
        "on every new-generator cell, P1's accept side equals RK4's page-bound outcomes",
        p1["identical"],
        p1,
    )
    rk3_deployment = cc_read_json(rk3.DEPLOYMENT)["by_protocol"][rk3.P_IN]
    key = rl1.epsilon_key(EPSILON)
    known_ok = True
    known_evidence = {}
    for model in KNOWN_MODELS:
        row = table[
            (table["scenario"] == SCENARIO_KNOWN)
            & (table["model"] == model)
            & (table["policy"] == P0)
        ].iloc[0]
        published = rk3_deployment[model][rk3.V_FULL][key]
        known_ok &= int(row["accepted"]) == published["accepted"] and bool(
            np.isclose(row["automated_recall"], published["repair_recall"], atol=0.0)
        )
        known_evidence[model] = {
            "accepted": int(row["accepted"]),
            "published": published["accepted"],
        }
    record(
        "f04_known_generator_p0_is_rk3s_deployment",
        "on the known generators, P0 accepts exactly what RK3's deployment phase accepted, for RK3 "
        "and for DS1's score, with the same recall",
        known_ok,
        known_evidence,
    )
    partition = (
        table["accepted"] + table["reviewed"] + table["rejected"] == table["decisions"]
    ).all()
    no_review = (table[table["policy"].isin(FULL_AUTOMATION)]["reviewed"] == 0).all()
    record(
        "f05_bands_partition_the_decisions",
        "accepted, reviewed and rejected decisions add up to every decision, and full automation "
        "sends nothing to a human",
        bool(partition and no_review),
        {"rows": len(table)},
    )
    calibration, test = toy_cells()
    toy = bands(P3, calibration, test)
    record(
        "f06_three_way_bands_by_hand",
        "on a hand-built set the plug-in accept cutoff is 7.0, the mirrored reject cutoff is 6.0, "
        "and the test decisions fall into accept, review and reject as read by eye",
        toy["accept_cutoff"] == 7.0
        and toy["reject_cutoff"] == 6.0
        and toy["accept"].tolist() == [True, True, False, False, False, False]
        and toy["reject"].tolist() == [False, False, False, True, True, True]
        and toy["review"].tolist() == [False, False, True, False, False, False],
        {"accept_cutoff": toy["accept_cutoff"], "reject_cutoff": toy["reject_cutoff"]},
    )
    full = table[table["policy"].isin(FULL_AUTOMATION)]
    record(
        "f07_without_review_total_recall_is_automated_recall",
        "a policy that reviews nothing recovers exactly what it accepts",
        bool(np.isclose(full["total_recall"], full["automated_recall"]).all()),
        {},
    )
    per_site = decisions.groupby(
        ["scenario", "unit", "model", "budget", "draw", "evaluation_set", "site_key"]
    ).size()
    record(
        "f08_one_label_free_decision_per_site",
        "every cell holds one decision per site, and the decision builder mentions no label field",
        bool((per_site == 1).all()) and not (_mentions("decision_rows") & LABEL_FIELDS),
        {"label_fields": sorted(_mentions("decision_rows") & LABEL_FIELDS)},
    )
    record(
        "f09_no_model_is_fitted",
        "the stage calls no fitting function",
        not (_called() & FIT_CALLS),
        {"fit_calls": sorted(_called() & FIT_CALLS)},
    )
    freeze = cc_read_json(RESEARCH_FREEZE)["upstream_sha256"]
    record(
        "f10_frozen_scores_are_unchanged",
        "RK4's cell scores and RK3's ranking scores are the files hashed at the freeze",
        file_sha256(rk4.CELL_SCORES) == freeze[_relative(rk4.CELL_SCORES)]
        and file_sha256(rk3.RANKING_SCORES) == freeze[_relative(rk3.RANKING_SCORES)],
        {},
    )
    crossing = 0
    for key_, cell in evaluation.cells.items():
        if key_[0] != SCENARIO_NEW or SET_CALIBRATION not in cell:
            continue
        crossing += len(
            set(cell[SET_CALIBRATION]["document_id"]) & set(cell[SET_TEST]["document_id"])
        )
    record(
        "f11_calibration_and_test_never_share_a_page",
        "in every new-generator cell the calibration decisions and the test decisions come from "
        "different pages",
        crossing == 0,
        {"shared_pages": crossing},
    )
    controls = cc_read_json(CONTROL_RESULTS)
    frontier = cc_read_json(FRONTIER)
    curves = cc_read_json(CURVES)["new_generator"]
    rederived = {
        unit: rk4.minimal_budget(
            {
                b: curves[unit][PRIMARY_THREE_WAY][b][f"practical_at_{REVIEW_REDUCTION_FLOOR}"]
                and f"{unit}|{PRIMARY_THREE_WAY}|{b}" not in controls["random_practical_cells"]
                for b in BUDGETS
            }
        )
        for unit in DIRECTIONS
    }
    record(
        "f12_the_frontier_re_derives_and_honours_the_random_control",
        "the primary three-way N* follows from the stored flags, with every budget at which random "
        "scores are also practical excluded",
        rederived == frontier["primary"]["three_way"],
        {"rederived": rederived},
    )
    decision_inputs = _mentions("criteria_from_artifacts") | _mentions("run_decide")
    record(
        "f13_no_analysis_only_curve_reaches_a_decision",
        "no criterion or outcome reads a tradeoff curve or an oracle cut",
        not [n for n in decision_inputs if "oracle" in n.lower() or n == "TRADEOFFS"],
        {},
    )
    record(
        "f14_no_certification_and_no_reserve",
        "no certification or reserve machinery is called",
        not [n for n in _called() if "certif" in n.lower() or "reserve" in n.lower()],
        {},
    )
    record(
        "f15_constraints_are_the_frozen_ones",
        "the reject tolerance equals the harm target, and the review-reduction floor and its "
        "sensitivity values are the pre-registered ones",
        ETA == EPSILON == 0.1
        and REVIEW_REDUCTION_FLOOR == 0.25
        and cc_read_json(DESIGN_RECORD)["definitions"]["practical"].endswith(
            str(REVIEW_REDUCTION_FLOOR)
        ),
        {"eta": ETA, "floor": REVIEW_REDUCTION_FLOOR},
    )
    record(
        "f16_budgets_are_rk4s_and_nothing_beyond_the_pool",
        "every evaluated budget is one of RK4's measured budgets",
        set(table[table["scenario"] == SCENARIO_NEW]["budget"]) <= set(BUDGETS),
        {"not_measurable": list(NOT_MEASURABLE)},
    )
    scored = set(pd.read_parquet(rk4.CELL_SCORES)["candidate_id"]) | set(
        pd.read_parquet(rk3.RANKING_SCORES)["candidate_id"]
    )
    record(
        "f17_every_decision_is_a_frozen_scored_candidate",
        "every decision is a candidate RK3 or RK4 already scored",
        set(decisions["candidate_id"]) <= scored,
        {"decisions": len(decisions)},
    )
    one_page = pd.DataFrame(
        {
            "document_id": ["p"] * 30,
            "safety": np.linspace(1.0, 0.0, 30),
            "is_harmful": [False] * 30,
            "exact": [False] * 30,
        }
    )
    record(
        "f18_page_bound_needs_two_pages_on_both_sides",
        "with one calibration page neither page-bound cutoff exists",
        accept_cutoff(R_PAGE_BOUND, one_page, EPSILON) is None
        and reject_cutoff(R_PAGE_BOUND, one_page, ETA) is None,
        {},
    )
    reject_rows = table[table["policy"].isin(FULL_AUTOMATION)]
    record(
        "f19_full_automation_never_claims_the_reject_constraint",
        "only the three-way policies are held to the reject constraint",
        not bool(reject_rows["reject_violation"].any()),
        {},
    )
    record(
        "f20_the_decision_reads_the_masked_frontier",
        "the outcome is assigned from the frontier that excludes random-practical budgets",
        "FRONTIER" in _mentions("criteria_from_artifacts"),
        {},
    )
    passed = sum(t["passed"] for t in tests)
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "all_passed": passed == len(tests),
        },
    )
    if passed != len(tests):
        raise PhaseError(f"falsification failed: {[t['test'] for t in tests if not t['passed']]}")
    print(f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    frontier = cc_read_json(FRONTIER)["primary"]
    full, three_way = frontier["full_automation"], frontier["three_way"]
    criteria = {
        "F1": all(full[d] is not None for d in DIRECTIONS),
        "F2": all(three_way[d] is not None for d in DIRECTIONS),
    }
    outcome = assign_outcome(full, three_way)
    scope = (
        "full automation"
        if criteria["F1"]
        else ("triage with human review" if criteria["F2"] else None)
    )
    stats = cc_read_json(STATISTICAL_TESTS)["family"]
    return {
        "criteria": criteria,
        "full_automation_n_star": full,
        "three_way_n_star": three_way,
        "outcome": outcome,
        "claim_scope": scope,
        "supported": {name: bool(row["survives_holm"]) for name, row in stats.items()},
    }


def outcome_label(state: dict[str, Any]) -> str:
    def shown(value: str | None) -> str:
        return (
            "not reached" if value is None else ("the pool" if value == POOL else f"{value} pages")
        )

    parts = [
        f"{DIRECTION_SHORT[d]}: {PRIMARY_FULL} {shown(state['full_automation_n_star'][d])}, "
        f"{PRIMARY_THREE_WAY} {shown(state['three_way_n_star'][d])}"
        for d in DIRECTIONS
    ]
    return f"{state['outcome']}: {OUTCOME_TAXONOMY[state['outcome']]} -- " + "; ".join(parts)


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    falsification = cc_read_json(FALSIFICATION)
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "outcome": state["outcome"],
            "outcome_label": outcome_label(state),
            "criteria": state["criteria"],
            "full_automation_n_star": state["full_automation_n_star"],
            "three_way_n_star": state["three_way_n_star"],
            "deployment_claim_permitted": bool(state["claim_scope"] is not None),
            "deployment_claim_scope": state["claim_scope"],
            "statistically_supported": state["supported"],
            "statistical_family_surviving": cc_read_json(STATISTICAL_TESTS)["surviving"],
            "primary_policies": {"full_automation": PRIMARY_FULL, "three_way": PRIMARY_THREE_WAY},
            "epsilon": EPSILON,
            "eta": ETA,
            "review_reduction_floor": REVIEW_REDUCTION_FLOOR,
            "human_reviewer": "idealized, not measured",
            "falsification_tests_passed": falsification["passed"],
            "falsification_tests_total": falsification["total"],
            "recommended_next_stage": NEXT_STAGE[state["outcome"]],
            "issued_head": _git("rev-parse", "HEAD"),
            "ready_for_external_confirmation": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {outcome_label(state)}")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = rk3.FIGURE_NOTE
SURFACE = rk3.SURFACE
INK = rk3.INK
INK_SECONDARY = rk3.INK_SECONDARY
GRID = rk3.GRID
POLICY_COLOURS = {P0: "#8a8a8a", P1: "#2a78d6", P2: "#eb6834", P3: "#1baf7a"}
BUDGET_STYLES = {"0": ":", "10": "--", POOL: "-"}
FIGURES = (
    "fig1_label_budget_curves.png",
    "fig2_harm_vs_coverage.png",
    "fig3_automation_frontier.png",
    "fig4_human_review_tradeoff.png",
)


def run_figures() -> int:
    """Four figures, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(DECISION, "decide")
    for path in (FIGURE_DIR, FIGURE_MANIFEST):
        _forbid(path)
    FIGURE_DIR.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "svg.hashsalt": "dep1",
        }
    )
    manifest: dict[str, Any] = {}

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(
            FIGURE_DIR / name,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Software": None},
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {s.name: rk3.xr1._signature(s) for s in sources},
        }

    curves = cc_read_json(CURVES)
    pools = curves["pool_pages"]
    new = curves["new_generator"]

    def pages(unit: str) -> list[int]:
        return [rk4.budget_pages(b, pools[unit]) for b in BUDGETS]

    def series(unit: str, policy: str, field: str) -> list[float]:
        return [
            np.nan if new[unit][policy][b][field] is None else new[unit][policy][b][field]
            for b in BUDGETS
        ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex="row")
    for row, unit in enumerate(DIRECTIONS):
        x = pages(unit)
        for policy in POLICIES:
            axes[row, 0].plot(
                x,
                series(unit, policy, "median_acceptance_coverage"),
                marker="o",
                ms=3,
                color=POLICY_COLOURS[policy],
                label=POLICY_SHORT[policy],
            )
            axes[row, 2].plot(
                x,
                series(unit, policy, "any_violation_share"),
                marker="o",
                ms=3,
                color=POLICY_COLOURS[policy],
            )
        for policy in THREE_WAY:
            axes[row, 1].plot(
                x,
                series(unit, policy, "median_review_reduction"),
                marker="o",
                ms=3,
                color=POLICY_COLOURS[policy],
            )
        axes[row, 1].axhline(REVIEW_REDUCTION_FLOOR, color=INK_SECONDARY, lw=0.8, ls="--")
        axes[row, 2].axhline(VIOLATION_CEILING, color=INK_SECONDARY, lw=0.8, ls="--")
        axes[row, 0].set_ylabel(f"{DIRECTION_SHORT[unit]}")
        for col in range(3):
            axes[row, col].set_ylim(-0.05, 1.05)
    axes[0, 0].set_title("(a) median share corrected automatically", loc="left", fontsize=7.5)
    axes[0, 1].set_title("(b) median review reduction", loc="left", fontsize=7.5)
    axes[0, 2].set_title("(c) share of draws violating a constraint", loc="left", fontsize=7.5)
    for col in range(3):
        axes[1, col].set_xlabel("labelled target pages")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    save(
        fig,
        FIGURES[0],
        (CURVES,),
        "Label-budget curves for the new generator: automatic corrections, review taken off the "
        "reviewer, and violations of the harm or reject constraint, by policy",
    )

    tradeoffs = cc_read_json(TRADEOFFS)["by_cell"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, unit in zip(axes, DIRECTIONS, strict=True):
        for budget, style in BUDGET_STYLES.items():
            rows = tradeoffs[f"{unit}|{budget}"]["harm_vs_coverage"]
            ax.plot(
                [r["coverage_grid"] for r in rows],
                [
                    np.nan if r["median_selective_harm"] is None else r["median_selective_harm"]
                    for r in rows
                ],
                ls=style,
                color=POLICY_COLOURS[P2],
                label=f"{budget if budget != POOL else 'pool'} pages",
            )
        rows = tradeoffs[f"{KNOWN_UNIT}|{rk3.M_RK3}"]["harm_vs_coverage"]
        ax.plot(
            [r["coverage_grid"] for r in rows],
            [
                np.nan if r["median_selective_harm"] is None else r["median_selective_harm"]
                for r in rows
            ],
            color=INK_SECONDARY,
            lw=1.0,
            label="known generators, RK3",
        )
        ax.axhline(EPSILON, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xlabel("share of test decision sites corrected")
        ax.set_title(DIRECTION_SHORT[unit], loc="left", fontsize=8)
    axes[0].set_ylabel("selective harm, median over draws")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(
        fig,
        FIGURES[1],
        (TRADEOFFS,),
        "Harm against coverage, ANALYSIS ONLY: every cutoff swept on the test labels",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex="col")
    for row, unit in enumerate(DIRECTIONS):
        for budget, style in BUDGET_STYLES.items():
            cell = tradeoffs[f"{unit}|{budget}"]
            axes[row, 0].plot(
                [r["harm"] for r in cell["automation_frontier"]],
                [
                    np.nan
                    if r["median_acceptance_coverage"] is None
                    else r["median_acceptance_coverage"]
                    for r in cell["automation_frontier"]
                ],
                ls=style,
                color=POLICY_COLOURS[P3],
                label=f"{budget if budget != POOL else 'pool'} pages",
            )
            axes[row, 1].plot(
                [r["exact_rate"] for r in cell["reject_frontier"]],
                [
                    np.nan if r["median_reject_share"] is None else r["median_reject_share"]
                    for r in cell["reject_frontier"]
                ],
                ls=style,
                color=POLICY_COLOURS[P3],
            )
        axes[row, 0].axvline(EPSILON, color=INK_SECONDARY, lw=0.8, ls=":")
        axes[row, 1].axvline(ETA, color=INK_SECONDARY, lw=0.8, ls=":")
        axes[row, 0].set_ylabel(f"{DIRECTION_SHORT[unit]}\nmedian share of sites")
    axes[0, 0].set_title(
        "(a) most sites correctable at each realized harm", loc="left", fontsize=7.5
    )
    axes[0, 1].set_title(
        "(b) most sites rejectable at each exact-repair rate", loc="left", fontsize=7.5
    )
    axes[1, 0].set_xlabel("realized harm among accepts")
    axes[1, 1].set_xlabel("exact-repair rate among rejects")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(
        fig,
        FIGURES[2],
        (TRADEOFFS,),
        "Automation frontiers, ANALYSIS ONLY: the most any accept cutoff or reject cutoff could "
        "reach on the test labels",
    )

    parts = (
        ("median_automated_recall", "corrected automatically", "#1baf7a"),
        ("median_review_recall", "reachable by a reviewer", "#2a78d6"),
        ("median_lost_recall", "lost to automatic rejection", "#eb6834"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    for row, unit in enumerate(DIRECTIONS):
        labels = [str(p) for p in pages(unit)]
        position = np.arange(len(BUDGETS))
        for col, policy in enumerate(THREE_WAY):
            ax = axes[row, col]
            bottom = np.zeros(len(BUDGETS))
            for field, name, colour in parts:
                values = np.nan_to_num(np.asarray(series(unit, policy, field), dtype=np.float64))
                ax.bar(position, values, bottom=bottom, color=colour, width=0.6, label=name)
                bottom += values
            twin = ax.twinx()
            twin.plot(
                position,
                series(unit, policy, "median_review_reduction"),
                color=INK,
                marker="o",
                ms=3,
                lw=1.0,
                label="review reduction",
            )
            twin.set_ylim(-0.05, 1.05)
            twin.grid(False)
            if col == 1:
                twin.set_ylabel("median review reduction")
            else:
                twin.set_yticklabels([])
            ax.set_xticks(position, labels)
            ax.set_title(
                f"{POLICY_SHORT[policy]}, {DIRECTION_SHORT[unit]}", loc="left", fontsize=7.5
            )
            if row == 1:
                ax.set_xlabel("labelled target pages")
        axes[row, 0].set_ylabel("median share of error sites")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    save(
        fig,
        FIGURES[3],
        (CURVES,),
        "Human-review tradeoff for the three-way policies: bars split the error sites into those "
        "corrected automatically, those a reviewer could still repair, and those lost to automatic "
        "rejection (medians over draws); the line is the work taken off the reviewer. The reviewer "
        "is idealized",
    )
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "figures": manifest,
            "note": FIGURE_NOTE,
            "synthetic": False,
        },
    )
    print(f"figures: {len(manifest)} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism and record

DERIVED_OUTPUTS = (
    "DECISIONS",
    "CELL_REGISTRY",
    "POLICY_OUTCOMES",
    "POLICY_REGISTRY",
    "CONTROL_RESULTS",
    "CURVES",
    "FRONTIER",
    "TRADEOFFS",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("cells", run_cells),
        ("policies", run_policies),
        ("controls", run_controls),
        ("curves", run_curves),
        ("frontier", run_frontier),
        ("tradeoffs", run_tradeoffs),
        ("stats", run_stats),
        ("negative", run_negative),
        ("decide", run_decide),
        ("figures", run_figures),
    )


def _current_signatures() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in DERIVED_OUTPUTS:
        path = globals()[name]
        if isinstance(path, Path) and path.is_file():
            out[path.name] = rk3.xr1._signature(path)
        elif isinstance(path, Path) and path.is_dir():
            for figure in sorted(path.glob("*.png")):
                out[f"figures/{figure.name}"] = file_sha256(figure)
    return out


def _sandboxed(directory: Path) -> dict[str, str]:
    import shutil

    saved = {name: globals()[name] for name in DERIVED_OUTPUTS}
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    try:
        for name in DERIVED_OUTPUTS:
            globals()[name] = directory / saved[name].name
        for _name, run in _derived_phases():
            run()
        return _current_signatures()
    finally:
        for name, path in saved.items():
            globals()[name] = path


def run_determinism() -> int:
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(n for n in published if any(r.get(n) != published[n] for r in runs))
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "derived_artifacts_compared": len(published),
            "derived_regenerations": len(runs),
            "differing_artifacts": differing,
            "all_derived_artifacts_identical": not differing,
            "all_runs_identical": not differing,
            "seeds": {"bootstrap": BOOTSTRAP_SEED, "control": CONTROL_SEED},
            "outcome": cc_read_json(DECISION)["outcome"],
            "uses_ground_truth": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(published)} artifacts x {len(runs)} regenerations, "
        f"identical {not differing} ({time.monotonic() - started:.0f}s)"
    )
    return 0


PRODUCED = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    DESIGN_RECORD,
    *(globals()[name] for name in DERIVED_OUTPUTS if name != "FIGURE_DIR"),
    DETERMINISM,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (FROZEN_CONFIGURATION, RESEARCH_FREEZE),
    "2. Research Questions and Pre-Registration": (DESIGN_RECORD,),
    "3. Frozen Inputs and Decisions": (CELL_REGISTRY, FROZEN_CONFIGURATION, POLICY_REGISTRY),
    "4. Four Policies": (DESIGN_RECORD, POLICY_REGISTRY),
    "5. Consistency with RK3 and RK4": (FALSIFICATION, CURVES),
    "6. Page Budget and Safe Automation Coverage": (CURVES, FRONTIER),
    "7. Abstention and Human Review": (CURVES, FRONTIER, CONTROL_RESULTS),
    "8. The Deployment Frontier": (FRONTIER, CURVES, CONTROL_RESULTS),
    "9. Automation, Harm and Annotation Cost": (TRADEOFFS, CURVES),
    "10. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION),
    "11. Statistical Tests": (STATISTICAL_TESTS,),
    "12. Limitations": (DESIGN_RECORD, CURVES, FRONTIER),
    "13. Research Decision": (DECISION, FRONTIER, STATISTICAL_TESTS, FALSIFICATION),
    "14. Next Research Direction": (DECISION, CURVES, TRADEOFFS),
}


def run_record() -> int:
    started = time.monotonic()
    for path in (PROVENANCE, TRACEABILITY):
        _forbid(path)
    missing = [p for p in PRODUCED if not p.exists() and p not in (PROVENANCE, TRACEABILITY)]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    if not cc_read_json(DETERMINISM)["all_runs_identical"]:
        raise PhaseError("determinism did not pass; the record will not be issued")
    freeze = cc_read_json(RESEARCH_FREEZE)
    moved = sorted(
        p for p, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / p) != sha
    )
    if moved:
        raise PhaseError(f"{len(moved)} upstream files moved since the freeze: {moved[:4]}")
    global DECISION
    saved = DECISION
    rederived: list[bool] = []
    for run in range(2):
        directory = CACHE / "record" / f"run_{run}"
        directory.mkdir(parents=True, exist_ok=True)
        sandbox = directory / saved.name
        sandbox.unlink(missing_ok=True)
        try:
            DECISION = sandbox
            run_decide()
        finally:
            DECISION = saved
        again = cc_read_json(sandbox)
        published = cc_read_json(saved)
        rederived.append(
            all(
                again[k] == published[k]
                for k in published
                if k not in ("issued_utc", "runtime_seconds")
            )
        )
    if not all(rederived):
        raise PhaseError("the decision did not re-derive identically")
    audit = rk3.gen1.audit_report(REPORT, REPORT_SECTIONS)
    if audit["untraceable_numeric_claims"]:
        raise PhaseError(
            f"{audit['untraceable_numeric_claims']} untraceable claims, "
            f"e.g. {audit['untraceable'][:3]}"
        )
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "artifacts": {_relative(p): file_sha256(p) for p in PRODUCED if p.is_file()},
            "artifact_count": sum(1 for p in PRODUCED if p.is_file()),
            "figures": {
                f"figures/{p.name}": file_sha256(p) for p in sorted(FIGURE_DIR.glob("*.png"))
            },
            "upstream_files_rehashed": len(freeze["upstream_sha256"]),
            "upstream_files_moved": moved,
            "decision_rederived_identically": rederived,
            "fits_a_model": False,
            "regenerates_a_candidate": False,
            "regenerates_ocr": False,
            "changes_an_upstream_outcome": False,
            "issued_head": _git("rev-parse", "HEAD"),
            "report": {"path": _relative(REPORT), "sha256": file_sha256(REPORT)},
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        TRACEABILITY,
        {
            **_envelope("traceability"),
            "report": _relative(REPORT),
            "sections": {n: [_relative(p) for p in ps] for n, ps in REPORT_SECTIONS.items()},
            "audit": audit,
            "untraceable": audit["untraceable"],
            "rule": (
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says"
            ),
        },
    )
    print(
        f"record: {sum(1 for p in PRODUCED if p.is_file())} artifacts, report claims "
        f"{audit['numeric_claims']}, untraceable {audit['untraceable_numeric_claims']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    phases: dict[str, Callable[[], int]] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "preregister": run_preregister,
        "cells": run_cells,
        "policies": run_policies,
        "controls": run_controls,
        "curves": run_curves,
        "frontier": run_frontier,
        "tradeoffs": run_tradeoffs,
        "stats": run_stats,
        "negative": run_negative,
        "decide": run_decide,
        "figures": run_figures,
        "determinism": run_determinism,
        "record": run_record,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    for name in phases:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(argv)
    chosen = [name for name in phases if getattr(args, name)]
    if not chosen:
        parser.error("choose at least one phase")
    for name in chosen:
        began = time.monotonic()
        code = phases[name]()
        if code != 0:
            return code
        print(f"  [{name} {time.monotonic() - began:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

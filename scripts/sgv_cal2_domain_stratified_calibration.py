#!/usr/bin/env python3
"""SGV-CAL2: domain-stratified, page-efficient calibration for safe OCR correction.

SGV-PF1 closed with outcome C: with about 135 labelled pages the adapted ranker is strong (harm
AUROC about 0.92 on fresh test pages), yet neither pre-registered conservative policy became both
safe and useful, and the plug-in cutoff overshot in current -> Qwen, where historical pages ran
at about 0.15 accepted harm against about 0.096 on forms. This stage tests one deployment
hypothesis that observation exposed:

    safe automation may fail because calibration evidence from heterogeneous document regimes is
    pooled into one decision boundary, even after ranking quality has become strong.

**1. Nothing upstream moves.** OCR, candidates, labels, features, the RK4 M1 ranker and its PF1
scores, the PF1 page registry, purchases and test labels are read, never rebuilt. CAL2 changes
only the step from a frozen score to a decision boundary.

**2. The comparison is small and fixed in advance.** C0 is PF1's pooled calibration and must
reproduce PF1 exactly before anything else runs. C1 picks every cutoff separately per corpus. C2
is C1 fed by PF1's frozen environment-stratified page purchase. C3 is a hierarchical estimator
that shrinks each corpus towards the pooled rate with a prior worth a fixed number of pages. The
design record, criteria, outcome rubric and stopping rule are written before any comparison.

**3. The evidence is an exploratory replay.** The corpus-stratification hypothesis was motivated by
PF1's breakdown of these same fresh test pages, so they cannot confirm it. Every artifact says so,
and nothing here is external confirmation, certification or a production claim.

    --reconstruct   PF1's state re-read from its artifacts, and every input hashed
    --design        DESIGN.json: methods, priors, criteria, family, rubric, stopping rule
    --manifest      the population manifest and every page-budget draw, label-free
    --reproduce     C0 recomputed from PF1's frozen scores; must equal PF1 exactly
    --calibrate     C0, C1, C3 (and C3's prior sensitivity) on every cell and both purchases
    --controls      random scores, calibration-label permutation, PF1's random control
    --summaries     per-budget summaries, flags and the masked frontier
    --mechanism     M1 risk curves, M2 threshold disagreement, M3 evidence, M4 gap
    --stats         the pre-registered family with Holm, the domain-difference family
    --negative      the falsification suite
    --decide        FINAL_DECISION.json from the frozen rubric
    --figures       every figure and its data from persisted artifacts
    --determinism   the derived phases regenerated twice in subprocess sandboxes
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / CALIBRATION METHODOLOGY. Nothing is certified or production-ready, the confirmatory
reserve stays LOCKED, no OCR, candidate generation or language-model call runs, and the stage
cannot execute an external confirmation.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_pf1_page_frontier_scaling as pf1
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.risk.bounds import clopper_pearson_upper
from ocr_risk.risk.cluster_bounds import design_effect

dep1 = pf1.dep1
rk4 = pf1.rk4
rk3 = pf1.rk3
rk2 = pf1.rk2
rk1 = pf1.rk1
th1 = pf1.th1
th2 = pf1.th2
ds1 = pf1.ds1
s14 = pf1.s14
s15 = pf1.s15
gen1 = pf1.gen1
xr1 = pf1.xr1

REPO = pf1.REPO
DEFAULT_OUT = REPO / "results/generated/sgv_cal2_domain_stratified_calibration"
# The determinism phase re-runs the derived phases in subprocesses whose OUT points at a sandbox;
# nothing ever rebinds a path inside a live module.
OUT = Path(os.environ.get("SGV_CAL2_OUT", str(DEFAULT_OUT))).resolve()
SANDBOXED = DEFAULT_OUT.resolve() != OUT
CACHE = OUT / "cache"

UPSTREAM_STATE = OUT / "upstream_state.json"
DESIGN = OUT / "DESIGN.json"
POPULATION_MANIFEST = OUT / "population_manifest.csv"
PAGE_BUDGET_DRAWS = OUT / "page_budget_draws.csv"
DRAW_REGISTRY = OUT / "draw_registry.json"
C0_REPRODUCTION = OUT / "c0_reproduction.json"
POLICY_ROWS = OUT / "policy_rows.parquet"
CLUSTER_COUNTS = OUT / "cluster_counts.parquet"
THRESHOLD_RECORDS = OUT / "threshold_records.csv"
POOLED_RESULTS = OUT / "pooled_results.csv"
CORPUS_STRATIFIED_RESULTS = OUT / "corpus_stratified_results.csv"
HIERARCHICAL_RESULTS = OUT / "hierarchical_results.csv"
STRATIFIED_PURCHASE_RESULTS = OUT / "stratified_purchase_results.csv"
CALIBRATION_REGISTRY = OUT / "calibration_registry.json"
CONTROL_RESULTS = OUT / "control_results.json"
POLICY_SUMMARY = OUT / "policy_summary.csv"
FRONTIER = OUT / "frontier.json"
DOMAIN_SUMMARY = OUT / "domain_summary.csv"
RISK_BINS = OUT / "risk_bins.parquet"
RISK_CURVES = OUT / "risk_curves.csv"
THRESHOLD_DISAGREEMENT = OUT / "threshold_disagreement.csv"
EFFECTIVE_EVIDENCE = OUT / "effective_evidence.csv"
CALIBRATION_GAP = OUT / "calibration_gap.csv"
MECHANISM = OUT / "mechanism.json"
STATISTICAL_TESTS = OUT / "statistical_tests.csv"
STATISTICAL_SUMMARY = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_results.json"
FINAL_DECISION = OUT / "FINAL_DECISION.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"

REPORT = REPO / "docs/sgv_cal2/domain_stratified_calibration.md"

SCHEMA_VERSION = 1
STAGE = "sgv_cal2_domain_stratified_calibration"
HYPOTHESIS = "SGV-CAL2-D1"
STAGE_KIND = "DEVELOPMENT / CALIBRATION METHODOLOGY (EXPLORATORY REPLAY)"
EVIDENCE_STATUS = "exploratory_development_replay"
EVIDENCE_NOTE = (
    "The corpus-stratification hypothesis was motivated by SGV-PF1's per-corpus breakdown of the "
    "same 66 fresh test pages this stage evaluates on. Those pages are therefore an exploratory "
    "development replay for CAL2, not independent or confirmatory evidence, and no result here is "
    "external confirmation or a production claim."
)

PhaseError = pf1.PhaseError
cc_read_json = pf1.cc_read_json
_share = rk4._share

# ------------------------------------------------------------------ the frozen design

DIRECTIONS = pf1.DIRECTIONS
D_G2Q = rk4.D_G2Q
D_Q2G = rk4.D_Q2G
DIRECTION_SHORT = pf1.DIRECTION_SHORT
A0 = pf1.A0
M1 = pf1.M1
POOL = pf1.POOL
BUDGETS = pf1.BUDGETS
ADAPTED_BUDGETS = pf1.ADAPTED_BUDGETS
STRATIFIED_BUDGETS = pf1.STRATIFIED_BUDGETS
DRAWS = pf1.DRAWS
CALIBRATION_EVERY = pf1.CALIBRATION_EVERY
SCHEME_RANDOM = pf1.SCHEME_RANDOM
SCHEME_STRATIFIED = pf1.SCHEME_STRATIFIED
SCHEMES = pf1.SCHEMES
SET_TEST = pf1.SET_TEST
SET_CALIBRATION = pf1.SET_CALIBRATION

P0 = pf1.P0
P1 = pf1.P1
P2 = pf1.P2
P3 = pf1.P3
POLICIES = pf1.POLICIES
FULL_AUTOMATION = pf1.FULL_AUTOMATION
THREE_WAY = pf1.THREE_WAY
CONSERVATIVE = pf1.CONSERVATIVE
PRIMARY_FLAG = pf1.PRIMARY_FLAG
POLICY_RULES = dep1.POLICY_RULES
R_PLUG_IN = dep1.R_PLUG_IN
R_PAGE_BOUND = dep1.R_PAGE_BOUND
R_ALL_REST = dep1.R_ALL_REST
EPSILON = pf1.EPSILON
ETA = pf1.ETA
DELTA = pf1.DELTA
HARM_TOLERANCE = th1.HARM_TOLERANCE
USEFUL_COVERAGE_FLOOR = pf1.USEFUL_COVERAGE_FLOOR
LOST_REPAIR_CEILING = pf1.LOST_REPAIR_CEILING
REVIEW_REDUCTION_FLOOR = pf1.REVIEW_REDUCTION_FLOOR
VIOLATION_CEILING = pf1.VIOLATION_CEILING
WORKING_FLOOR = pf1.WORKING_FLOOR

DOMAINS = ("funsd", "ocrd_sbb")
DOMAIN_LABEL = {
    "funsd": "FUNSD modern forms (modern_forms)",
    "ocrd_sbb": "OCR-D-SBB historical print (historical_print)",
}
DOMAIN_SHORT = {"funsd": "FUNSD", "ocrd_sbb": "SBB"}
POOLED = "pooled"
OVERALL = "overall"

C0 = "c0_pooled"
C1 = "c1_corpus_stratified"
C3 = "c3_hierarchical"
C3_LOW = "c3_hierarchical_prior_2"
C3_HIGH = "c3_hierarchical_prior_10"
METHODS = (C0, C1, C3, C3_LOW, C3_HIGH)
PRIMARY_METHODS = (C0, C1, C3)
DOMAIN_AWARE = (C1, C3)
C3_SENSITIVITY = (C3_LOW, C3_HIGH)
PRIOR_PAGES = {C3: 5.0, C3_LOW: 2.0, C3_HIGH: 10.0}
C2 = "c2_balanced_purchase_corpus_stratified"
METHOD_LABEL = {
    C0: "C0 pooled: PF1's cutoffs, one boundary over every calibration decision",
    C1: "C1 corpus-stratified: each rule applied to one corpus's calibration pages alone",
    C3: (
        "C3 hierarchical: each corpus's counts at a cutoff plus a prior worth 5 average pooled "
        "calibration pages at the pooled rate"
    ),
    C3_LOW: "C3 sensitivity: the same estimator with a prior worth 2 pages (analysis only)",
    C3_HIGH: "C3 sensitivity: the same estimator with a prior worth 10 pages (analysis only)",
    C2: "C2: C1's rule on PF1's frozen environment-stratified page purchase",
}
METHOD_SHORT = {C0: "C0 pooled", C1: "C1 corpus", C3: "C3 hierarchical", C2: "C2 balanced + C1"}

CONTROL_SEED = pf1.CONTROL_SEED
LABEL_PERMUTATION_SEED = 20261004
LABEL_PERMUTATION_DRAWS = 5
DOMAIN_PERMUTATION_SEED = 20261005
DOMAIN_PERMUTATIONS = 200
SYNTHETIC_SEED = 20261006
SYNTHETIC_REPLICATIONS = 200
SYNTHETIC_BOOTSTRAP = 200
DIAGNOSTIC_REPLICATIONS = 1000
DIAGNOSTIC_STREAM = 7
BOOTSTRAP_RESAMPLES = pf1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = pf1.BOOTSTRAP_SEED
RISK_BIN_COUNT = 10
HOLM_ALPHA = 0.05
FAMILY_BUDGET = POOL
SENSITIVITY_BUDGET = "100"

SYNTHETIC = {
    "calibration_pages": {"a": 33, "b": 12},
    "test_pages": {"a": 52, "b": 14},
    "decisions_per_page_poisson_mean": 45,
    "intercept": 2.0,
    "score_slope": 6.0,
    "page_effect_sd": 0.5,
    "exact_share_of_harmless": 0.5,
    "shift_identical": 0.0,
    "shift_shifted": 1.0,
    "scores": (
        "uniform on [0, 1]; harm probability logistic(intercept - slope*score + page + shift)"
    ),
}

PRIMARY_FAMILY = (
    "S1_p0_violation_c1_vs_c0_pool_g2q",
    "S1_p0_violation_c1_vs_c0_pool_q2g",
    "R1_p1_safe_recall_c1_vs_c0_pool_g2q",
    "R1_p1_safe_recall_c1_vs_c0_pool_q2g",
    "R2_p1_safe_recall_c3_vs_c0_pool_g2q",
    "R2_p1_safe_recall_c3_vs_c0_pool_q2g",
)
METRIC_VIOLATION = "accept_violation"
METRIC_SAFE_RECALL = "safe_automated_recall"
FAMILY_SPEC: dict[str, dict[str, Any]] = {
    name: {
        "direction": D_G2Q if name.endswith("g2q") else D_Q2G,
        "metric": METRIC_VIOLATION if name.startswith("S1") else METRIC_SAFE_RECALL,
        "policy": P0 if name.startswith("S1") else P1,
        "aware": C3 if name.startswith("R2") else C1,
        "pooled": C0,
        "scheme": SCHEME_RANDOM,
        "budget": FAMILY_BUDGET,
    }
    for name in PRIMARY_FAMILY
}
DOMAIN_FAMILY = ("Q1_score_adjusted_domain_harm_g2q", "Q1_score_adjusted_domain_harm_q2g")
SECONDARY: dict[str, dict[str, Any]] = {
    **{
        f"X1_p1_safe_recall_c2_vs_c1_random_{b}_{DIRECTION_SHORT[d]}": {
            "direction": d,
            "metric": METRIC_SAFE_RECALL,
            "policy": P1,
            "left": (C1, SCHEME_STRATIFIED, b),
            "right": (C1, SCHEME_RANDOM, b),
        }
        for d in DIRECTIONS
        for b in ("50", "100")
    },
    **{
        f"X2_p0_violation_c2_vs_c1_random_{b}_{DIRECTION_SHORT[d]}": {
            "direction": d,
            "metric": METRIC_VIOLATION,
            "policy": P0,
            "left": (C1, SCHEME_STRATIFIED, b),
            "right": (C1, SCHEME_RANDOM, b),
        }
        for d in DIRECTIONS
        for b in ("50", "100")
    },
    **{
        f"X3_{metric_short}_c1_vs_c0_{SENSITIVITY_BUDGET}_{DIRECTION_SHORT[d]}": {
            "direction": d,
            "metric": metric,
            "policy": policy,
            "left": (C1, SCHEME_RANDOM, SENSITIVITY_BUDGET),
            "right": (C0, SCHEME_RANDOM, SENSITIVITY_BUDGET),
        }
        for d in DIRECTIONS
        for metric_short, metric, policy in (
            ("p0_violation", METRIC_VIOLATION, P0),
            ("p1_safe_recall", METRIC_SAFE_RECALL, P1),
        )
    },
}

QUESTIONS = {
    "Q1": "Do FUNSD and SBB have measurably different score-to-risk relationships?",
    "Q2": "Does corpus-specific calibration outperform pooled calibration?",
    "Q3": "Does balanced page acquisition improve calibration efficiency?",
    "Q4": "Does hierarchical partial pooling improve the safety-utility frontier?",
    "Q5": "Does any conservative policy become both safe and useful?",
    "Q6": (
        "Is the remaining deployment gap caused primarily by ranking, calibration evidence, "
        "domain heterogeneity, or some combination?"
    ),
    "Q7": "What is the defensible Paper-3 claim?",
    "Q8": "Is a single untouched external confirmation now scientifically justified?",
}
OUTCOME_TAXONOMY = {
    "A": (
        "domain-aware calibration produces a safe and useful conservative boundary in both "
        "generator directions and improves over pooled calibration under the frozen criteria"
    ),
    "B": (
        "domain-aware calibration produces a safe and useful conservative boundary where pooled "
        "calibration does not in at least one direction, without robust bidirectional deployment"
    ),
    "C": (
        "domain-aware calibration improves violation frequency or safe recall under the frozen "
        "tests, but no domain-aware conservative operating point is safe and useful"
    ),
    "D": "domain-aware calibration provides no measurable improvement over the pooled baseline",
}
INTERPRETATION = {
    "A": "Heterogeneous calibration evidence was a major, actionable deployment bottleneck.",
    "B": "Domain-aware calibration is useful but deployment remains conditional.",
    "C": "Heterogeneity matters, but available independent evidence remains insufficient.",
    "D": "Simple domain stratification does not close the deployment gap.",
}
NEXT_STAGE = {
    "confirm": (
        "STOP METHOD DEVELOPMENT: PREPARE THE PAPER-3 MANUSCRIPT AND DECIDE SEPARATELY WHETHER THE "
        "FROZEN METHOD DESERVES ONE UNTOUCHED CONFIRMATION STUDY"
    ),
    "write": (
        "STOP METHOD DEVELOPMENT: WRITE PAPER 3 AROUND THE MEASURED DEPLOYMENT FRONTIER AND ITS "
        "LIMITATIONS; NO FURTHER CALIBRATION SEARCH ON THE SAME DATA"
    ),
}
STOPPING_RULE = {
    "position": "SGV-CAL2 is the final development-stage calibration experiment for Paper 3.",
    "if_A_or_strong_B": (
        "Stop method development. Do not invent CAL3. Prepare the Paper-3 manuscript and decide "
        "separately whether the frozen method deserves one untouched confirmation study."
    ),
    "if_B_C_or_D": (
        "Stop method development. Do not search for additional calibration algorithms on the "
        "same evaluated data. Write Paper 3 around the measured deployment frontier and its "
        "limitations."
    ),
    "written_before_outcomes": True,
}


# ------------------------------------------------------------------ envelopes and write-once io


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"{STAGE}-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "stage_kind": STAGE_KIND,
        "hypothesis_id": HYPOTHESIS,
        "evidence_status": EVIDENCE_STATUS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    return {
        **_envelope(artifact),
        "uses_ground_truth": True,
        "production_ready": False,
        "certified": False,
        "confirmatory_reserve_consumed": False,
        "external_confirmation_executed": False,
        "evidence_note": EVIDENCE_NOTE,
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _plain(value: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN to None, containers recursively."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if number != number else number
    return value


def _require(path: Path, phase: str) -> None:
    if not path.exists():
        raise PhaseError(f"run --{phase} first ({_relative(path)} is missing)")


def _forbid(path: Path) -> None:
    if path.exists():
        raise PhaseError(f"{_relative(path)} already exists; delete it deliberately to rewrite")


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    _forbid(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    _forbid(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    _forbid(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def read_csv(path: Path) -> pd.DataFrame:
    """A CSV artifact back exactly: pandas' default float parser does not round-trip every
    float, so a cutoff read back could differ in its last bit from the one written."""
    return pd.read_csv(path, dtype={"budget": str}, float_precision="round_trip")


def _design_sha() -> str:
    return file_sha256(DESIGN)


def _median(values: Any) -> float | None:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(np.float64)
    return float(np.median(array)) if array.size else None


def _mean(values: Any) -> float | None:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(np.float64)
    return float(array.mean()) if array.size else None


def budget_pages(label: str, direction: str, pool_pages: dict[str, int]) -> int:
    """The labelled pages a budget buys: PF1's rule, with the pool read from PF1's registry."""
    if label == "0":
        return 0
    return pf1.budget_pages(label, int(pool_pages[direction]))


def pf1_pool_pages() -> dict[str, int]:
    return {d: int(v) for d, v in cc_read_json(pf1.SPLIT_REGISTRY)["pool_pages"].items()}


# ------------------------------------------------------------------ cutoff rules, vectorized


def cp_upper(harmful: Any, total: Any, delta: float) -> np.ndarray:
    """`risk.bounds.clopper_pearson_upper` over arrays, with its conventions at the edges."""
    from scipy.stats import beta

    k = np.asarray(harmful, dtype=np.float64)
    n = np.asarray(total, dtype=np.float64)
    out = np.ones(np.broadcast(k, n).shape, dtype=np.float64)
    k, n = np.broadcast_arrays(k, n)
    ok = (n > 0) & (k < n)
    if ok.any():
        out[ok] = beta.ppf(1.0 - delta, k[ok] + 1.0, n[ok] - k[ok])
    return out


@dataclass(frozen=True, slots=True)
class Prefix:
    """Every distinct score in descending order, with the counts at or above it."""

    cutoffs: np.ndarray
    harmful: np.ndarray
    total: np.ndarray
    member_harmful: np.ndarray
    member_total: np.ndarray


def prefix_counts(safety: np.ndarray, harmful: np.ndarray, member: np.ndarray) -> Prefix:
    order = np.argsort(-safety, kind="mergesort")
    ordered = safety[order]
    last = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    hit = harmful[order]
    inside = member[order]
    return Prefix(
        cutoffs=ordered[last],
        harmful=np.cumsum(hit)[last],
        total=last + 1,
        member_harmful=np.cumsum(hit & inside)[last],
        member_total=np.cumsum(inside)[last],
    )


def loosest(cutoffs: np.ndarray, qualifies: np.ndarray) -> float | None:
    """RK1's and TH1's convention: the lowest qualifying cutoff, contiguity not required."""
    index = np.flatnonzero(qualifies)
    return float(cutoffs[index[-1]]) if index.size else None


def page_effect(harmful: np.ndarray, pages: np.ndarray) -> float | None:
    """`th1.page_design_effect` on arrays: pages in sorted order, floored at one."""
    if harmful.size == 0:
        return None
    _unique, inverse = np.unique(pages, return_inverse=True)
    if _unique.size < 2:
        return None
    value = design_effect(
        np.bincount(inverse, weights=harmful.astype(np.float64)),
        np.bincount(inverse).astype(np.float64),
    )
    if not np.isfinite(value):
        return None
    return float(max(1.0, value))


def plug_in_cutoff(safety: np.ndarray, harmful: np.ndarray, epsilon: float) -> float | None:
    """RK1's plug-in rule on arrays."""
    if safety.size == 0:
        return None
    p = prefix_counts(safety, harmful, np.ones(safety.size, dtype=bool))
    return loosest(p.cutoffs, p.harmful / p.total <= epsilon + HARM_TOLERANCE)


def bound_cutoff(
    safety: np.ndarray, harmful: np.ndarray, pages: np.ndarray, epsilon: float
) -> float | None:
    """TH1's page-corrected Clopper-Pearson rule on arrays."""
    deff = page_effect(harmful, pages)
    if deff is None:
        return None
    p = prefix_counts(safety, harmful, np.ones(safety.size, dtype=bool))
    upper = cp_upper(p.harmful / deff, p.total / deff, DELTA)
    return loosest(p.cutoffs, upper <= epsilon + HARM_TOLERANCE)


def hierarchical_cutoff(
    rule: str, calibration: pd.DataFrame, domain: str, epsilon: float, prior_pages: float
) -> tuple[float | None, dict[str, Any]]:
    """C3: a corpus's counts at each pooled cutoff plus a prior worth `prior_pages` pages.

    At a cutoff t the corpus has h harmful among n accepted calibration decisions and the pool has
    H among N over P calibration pages. The prior is `prior_pages` average pooled pages at t:
    H * prior_pages / P harmful among N * prior_pages / P. With no own evidence the estimate is
    the pooled rate; with many own pages it is the corpus's. The plug-in rule compares the
    posterior mean with the target; the page-bound rule applies Clopper-Pearson to the augmented
    counts deflated by the pooled page design effect. Borrowed pseudo-counts come from another
    corpus, so the bound is an approximation, never a finite-sample certificate.
    """
    detail: dict[str, Any] = {"design_effect": None, "pooled_calibration_pages": 0}
    if calibration.empty:
        return None, detail
    safety = calibration["safety"].to_numpy(np.float64)
    harmful = calibration["is_harmful"].to_numpy(bool)
    member = calibration["corpus"].astype(str).to_numpy() == domain
    pages = int(calibration["document_id"].nunique())
    detail["pooled_calibration_pages"] = pages
    p = prefix_counts(safety, harmful, member)
    # Scaled by P so every term is an integer while the prior is a whole number of pages: with
    # no own evidence the ratio is then the pooled rate to the last bit, not a rounding of it.
    k = p.member_harmful * float(pages) + prior_pages * p.harmful
    n = p.member_total * float(pages) + prior_pages * p.total
    if rule == R_PLUG_IN:
        safe_n = np.where(n > 0, n, 1.0)
        return loosest(p.cutoffs, (n > 0) & (k / safe_n <= epsilon + HARM_TOLERANCE)), detail
    if rule == R_PAGE_BOUND:
        deff = th1.page_design_effect(calibration)
        detail["design_effect"] = deff
        if deff is None:
            return None, detail
        upper = cp_upper(k / pages / deff, n / pages / deff, DELTA)
        return loosest(p.cutoffs, upper <= epsilon + HARM_TOLERANCE), detail
    raise PhaseError(f"unknown rule {rule}")


# ------------------------------------------------------------------ the three calibrations


@dataclass(slots=True)
class Bands:
    """A policy's accept and reject masks on test decisions, and every cutoff it chose."""

    accept: np.ndarray
    reject: np.ndarray
    records: list[dict[str, Any]]
    accept_cutoff: float | None = None
    reject_cutoff: float | None = None

    @property
    def review(self) -> np.ndarray:
        return ~self.accept & ~self.reject


def _evidence(calibration: pd.DataFrame) -> dict[str, Any]:
    clusters = calibration["cluster"] if "cluster" in calibration else calibration["document_id"]
    return {
        "calibration_pages": int(calibration["document_id"].nunique()),
        "calibration_groups": int(clusters.nunique()),
        "calibration_decisions": len(calibration),
        "calibration_harmful": int(calibration["is_harmful"].sum()),
        "calibration_exact": int(calibration["exact"].sum()),
    }


def _check_domains(frame: pd.DataFrame) -> None:
    unknown = set(frame["corpus"].astype(str)) - set(DOMAINS)
    if unknown:
        raise PhaseError(f"a decision carries an unknown corpus: {sorted(unknown)}")


def _effects(policy: str, calibration: pd.DataFrame, usable: bool) -> tuple[Any, Any]:
    accept_rule, reject_rule = POLICY_RULES[policy]
    if not usable or calibration.empty:
        return None, None
    accept = th1.page_design_effect(calibration) if accept_rule == R_PAGE_BOUND else None
    reject = (
        th1.page_design_effect(dep1.mirrored(calibration)) if reject_rule == R_PAGE_BOUND else None
    )
    return accept, reject


def pooled_bands(policy: str, calibration: pd.DataFrame, test: pd.DataFrame, usable: bool) -> Bands:
    """C0: DEP1's bands exactly as PF1 applied them."""
    b = dep1.bands(policy, calibration, test, usable)
    accept_deff, reject_deff = _effects(policy, calibration, usable)
    record = {
        "domain": POOLED,
        "accept_cutoff": b["accept_cutoff"],
        "reject_cutoff": b["reject_cutoff"],
        "accept_design_effect": accept_deff,
        "reject_design_effect": reject_deff,
        **_evidence(calibration),
    }
    return Bands(
        np.asarray(b["accept"], dtype=bool),
        np.asarray(b["reject"], dtype=bool),
        [record],
        b["accept_cutoff"],
        b["reject_cutoff"],
    )


def corpus_bands(policy: str, calibration: pd.DataFrame, test: pd.DataFrame, usable: bool) -> Bands:
    """C1: DEP1's bands per corpus, each from that corpus's calibration pages alone."""
    _check_domains(test)
    _check_domains(calibration)
    accept = np.zeros(len(test), dtype=bool)
    reject = np.zeros(len(test), dtype=bool)
    corpus = test["corpus"].astype(str).to_numpy()
    calibration_corpus = calibration["corpus"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        mask = corpus == domain
        own = calibration[calibration_corpus == domain]
        b = dep1.bands(policy, own, test[mask], usable)
        accept[mask] = b["accept"]
        reject[mask] = b["reject"]
        accept_deff, reject_deff = _effects(policy, own, usable)
        records.append(
            {
                "domain": domain,
                "accept_cutoff": b["accept_cutoff"],
                "reject_cutoff": b["reject_cutoff"],
                "accept_design_effect": accept_deff,
                "reject_design_effect": reject_deff,
                **_evidence(own),
            }
        )
    return Bands(accept, reject, records)


def hierarchical_bands(
    policy: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    usable: bool,
    prior_pages: float,
) -> Bands:
    """C3: DEP1's band structure with every cutoff from `hierarchical_cutoff`."""
    _check_domains(test)
    _check_domains(calibration)
    accept_rule, reject_rule = POLICY_RULES[policy]
    accept = np.zeros(len(test), dtype=bool)
    reject = np.zeros(len(test), dtype=bool)
    corpus = test["corpus"].astype(str).to_numpy()
    safety = test["safety"].to_numpy(np.float64)
    calibration_corpus = calibration["corpus"].astype(str).to_numpy()
    mirror = dep1.mirrored(calibration) if reject_rule != R_ALL_REST else None
    records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        mask = corpus == domain
        high, accept_detail = (
            hierarchical_cutoff(accept_rule, calibration, domain, EPSILON, prior_pages)
            if usable
            else (None, {"design_effect": None})
        )
        own_accept = safety[mask] >= high if high is not None else np.zeros(mask.sum(), bool)
        low: float | None = None
        reject_detail: dict[str, Any] = {"design_effect": None}
        if reject_rule == R_ALL_REST:
            own_reject = ~own_accept
        else:
            assert mirror is not None
            mirrored_cut, reject_detail = (
                hierarchical_cutoff(reject_rule, mirror, domain, ETA, prior_pages)
                if usable
                else (None, {"design_effect": None})
            )
            low = None if mirrored_cut is None else -mirrored_cut
            own_reject = (
                (~own_accept) & (safety[mask] <= low)
                if low is not None
                else np.zeros(mask.sum(), dtype=bool)
            )
        accept[mask] = own_accept
        reject[mask] = own_reject
        records.append(
            {
                "domain": domain,
                "accept_cutoff": high,
                "reject_cutoff": low,
                "accept_design_effect": accept_detail.get("design_effect"),
                "reject_design_effect": reject_detail.get("design_effect"),
                "prior_pages": prior_pages,
                **_evidence(calibration[calibration_corpus == domain]),
            }
        )
    return Bands(accept, reject, records)


def method_bands(
    method: str, policy: str, calibration: pd.DataFrame, test: pd.DataFrame, usable: bool
) -> Bands:
    if method == C0:
        return pooled_bands(policy, calibration, test, usable)
    if method == C1:
        return corpus_bands(policy, calibration, test, usable)
    if method in PRIOR_PAGES:
        return hierarchical_bands(policy, calibration, test, usable, PRIOR_PAGES[method])
    raise PhaseError(f"unknown method {method}")


# ------------------------------------------------------------------ PF1's frozen cells

CellKey = tuple[str, str, str, str, int]
ROW_KEYS = ("direction", "arm", "scheme", "budget", "draw")


def cal2_cells() -> list[CellKey]:
    """PF1's A0 and M1 cells in PF1's execution order; B2 is not the ranker under study."""
    return [key for key in pf1.cells() if key[1] in (A0, M1)]


def cell_for(direction: str, scheme: str, budget: str, draw: int) -> CellKey:
    """Budget zero is PF1's A0 cell for every scheme."""
    return pf1.cell_key(direction, M1, scheme, budget, draw)


@dataclass(slots=True)
class TestView:
    """A cell's test decisions with every array an outcome needs, computed once."""

    frame: pd.DataFrame
    harmful: np.ndarray
    exact: np.ndarray
    corpus: np.ndarray
    cluster: np.ndarray
    repairs: list[frozenset[str]]


@dataclass(slots=True)
class Cell:
    key: CellKey
    calibration: pd.DataFrame
    test: TestView
    usable: bool


@dataclass(slots=True)
class Context:
    """PF1's test recall context, with each error site's corpus and cluster."""

    keys: frozenset[str]
    mapping: dict[str, set[str]]
    errors: pd.DataFrame
    clusters: tuple[str, ...]
    cluster_domain: tuple[str, ...]
    error_cluster: dict[str, int]
    error_domain: dict[str, str]
    errors_by_domain: dict[str, int]
    errors_by_cluster: np.ndarray


_GROUPS: dict[tuple[str, str], str] = {}


def group_of(corpus: str, page: str) -> str:
    """SGV14's partition unit: a FUNSD form is its own group, an SBB page its volume."""
    key = (corpus, page)
    if key not in _GROUPS:
        _GROUPS[key] = s14.group_of(corpus, page)
    return _GROUPS[key]


def with_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        cluster=[
            group_of(c, d)
            for c, d in zip(
                frame["corpus"].astype(str), frame["document_id"].astype(str), strict=True
            )
        ]
    )


def test_clusters() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every PF1 test page's group, sorted, and the corpus of each."""
    pool = pf1.load_pool()
    test = pool[pool["split"] == pf1.SPLIT_TEST]
    domain_of: dict[str, str] = {}
    for corpus, page in zip(
        test["corpus"].astype(str), test["document_id"].astype(str), strict=True
    ):
        domain_of[group_of(corpus, page)] = corpus
    clusters = tuple(sorted(domain_of))
    return clusters, tuple(domain_of[c] for c in clusters)


def build_context(context: tuple[set[str], dict[str, set[str]], pd.DataFrame]) -> Context:
    keys, mapping, errors = context
    clusters, cluster_domain = test_clusters()
    index = {c: i for i, c in enumerate(clusters)}
    error_cluster: dict[str, int] = {}
    error_domain: dict[str, str] = {}
    for key, corpus, page in zip(
        errors["error_key"].astype(str),
        errors["corpus"].astype(str),
        errors["document_id"].astype(str),
        strict=True,
    ):
        error_cluster[key] = index[group_of(corpus, page)]
        error_domain[key] = corpus
    if set(error_cluster) != set(keys):
        raise PhaseError("the test error keys and the recall denominator disagree")
    by_cluster = np.bincount(list(error_cluster.values()), minlength=len(clusters)).astype(float)
    return Context(
        keys=frozenset(keys),
        mapping=mapping,
        errors=errors,
        clusters=clusters,
        cluster_domain=cluster_domain,
        error_cluster=error_cluster,
        error_domain=error_domain,
        errors_by_domain={d: sum(1 for v in error_domain.values() if v == d) for d in DOMAINS},
        errors_by_cluster=by_cluster,
    )


def test_view(frame: pd.DataFrame, ctx: Context) -> TestView:
    frame = with_clusters(frame).reset_index(drop=True)
    index = {c: i for i, c in enumerate(ctx.clusters)}
    exact = frame["exact"].to_numpy(bool)
    empty: frozenset[str] = frozenset()
    repairs = [
        frozenset(ctx.mapping.get(site, set()) & ctx.keys) if hit else empty
        for site, hit in zip(frame["site_key"].astype(str), exact, strict=True)
    ]
    return TestView(
        frame=frame,
        harmful=frame["is_harmful"].to_numpy(bool),
        exact=exact,
        corpus=frame["corpus"].astype(str).to_numpy(),
        cluster=np.asarray([index[c] for c in frame["cluster"]], dtype=np.int64),
        repairs=repairs,
    )


def load_cells(keys: Sequence[CellKey] | None = None) -> tuple[dict[CellKey, Cell], Context]:
    """PF1's labelled score frames for the CAL2 cells, as PF1's own loader builds them."""
    evaluation = pf1.load_evaluation()
    ctx = build_context(evaluation.context[SET_TEST])
    out: dict[CellKey, Cell] = {}
    for key in keys if keys is not None else cal2_cells():
        cell = evaluation.cells[key]
        test = rk4.decisions(cell[SET_TEST])
        calibration = rk4.decisions(cell.get(SET_CALIBRATION, cell[SET_TEST].head(0)))
        out[key] = Cell(
            key=key,
            calibration=with_clusters(calibration) if len(calibration) else calibration,
            test=test_view(test, ctx),
            usable=bool(evaluation.registry[key]["trainable"]),
        )
    return out, ctx


# ------------------------------------------------------------------ one policy's outcome


def _union(view: TestView, mask: np.ndarray) -> set[str]:
    out: set[str] = set()
    for row in np.flatnonzero(mask & view.exact):
        out |= view.repairs[row]
    return out


def evaluate(
    policy: str, view: TestView, bands: Bands, ctx: Context
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """DEP1's `evaluate_policy` and PF1's extras on precomputed bands, plus per-corpus and
    per-cluster counts. C0 through this function reproduces PF1's rows exactly (--reproduce)."""
    accept, reject, review = bands.accept, bands.reject, bands.review
    n = len(view.frame)
    auto, reviewed_repairs, rejected_repairs = (
        _union(view, accept),
        _union(view, review),
        _union(view, reject),
    )
    accepted = int(accept.sum())
    rejected = int(reject.sum())
    reviewed = int(review.sum())
    harmful_accepted = int(view.harmful[accept].sum())
    harm = _share(harmful_accepted, accepted)
    reject_exact = _share(int(view.exact[reject].sum()), rejected)
    accept_violation = bool(accepted > 0 and harm is not None and harm > EPSILON + HARM_TOLERANCE)
    reject_violation = bool(
        policy in THREE_WAY
        and rejected > 0
        and reject_exact is not None
        and reject_exact > ETA + HARM_TOLERANCE
    )
    denominator = max(len(ctx.keys), 1)
    row: dict[str, Any] = {
        "policy": policy,
        "decisions": n,
        "accepted": accepted,
        "reviewed": reviewed,
        "rejected": rejected,
        "accept_cutoff": bands.accept_cutoff,
        "reject_cutoff": bands.reject_cutoff,
        "acceptance_coverage": _share(accepted, n),
        "automation_rate": _share(accepted + rejected, n),
        "review_rate": _share(reviewed, n),
        "review_reduction": None if n == 0 else 1.0 - reviewed / n,
        "harm": harm,
        "joint_harm": _share(harmful_accepted, n),
        "reject_exact_rate": reject_exact,
        "automated_recall": len(auto) / denominator,
        "review_recall": len(reviewed_repairs) / denominator,
        "lost_recall": len(rejected_repairs) / denominator,
        "total_recall": len(auto | reviewed_repairs) / denominator,
        "repaired_automatically": len(auto),
        "accept_violation": accept_violation,
        "reject_violation": reject_violation,
        "any_violation": bool(accept_violation or reject_violation),
        "works": bool(accepted > 0 and not accept_violation and len(auto) >= 1),
        "safe_coverage": 0.0 if accept_violation else (_share(accepted, n) or 0.0),
    }
    lost = rejected_repairs - auto - reviewed_repairs
    reachable = auto | reviewed_repairs | lost
    row["lost_repair_share"] = len(lost) / len(reachable) if reachable else None
    accepted_frame = view.frame[accept]
    row.update(th2.page_harm(accepted_frame, EPSILON))
    row.update(th1.environment_harm(accepted_frame, EPSILON))
    row["safe_automated_recall"] = 0.0 if accept_violation else row["automated_recall"]
    worst: list[float] = []
    violating = False
    for domain in DOMAINS:
        mask = view.corpus == domain
        own = accept & mask
        own_accepted = int(own.sum())
        own_harm = _share(int(view.harmful[own].sum()), own_accepted)
        own_repairs = sum(1 for key in auto if ctx.error_domain[key] == domain)
        own_recall = own_repairs / max(ctx.errors_by_domain[domain], 1)
        own_violation = bool(
            own_accepted > 0 and own_harm is not None and own_harm > EPSILON + HARM_TOLERANCE
        )
        violating |= own_violation
        if own_harm is not None:
            worst.append(own_harm)
        row.update(
            {
                f"decisions_{domain}": int(mask.sum()),
                f"accepted_{domain}": own_accepted,
                f"harmful_accepted_{domain}": int(view.harmful[own].sum()),
                f"harm_{domain}": own_harm,
                f"coverage_{domain}": _share(own_accepted, int(mask.sum())),
                f"safe_coverage_{domain}": (
                    0.0 if own_violation else (_share(own_accepted, int(mask.sum())) or 0.0)
                ),
                f"automated_repairs_{domain}": own_repairs,
                f"automated_recall_{domain}": own_recall,
                f"safe_automated_recall_{domain}": 0.0 if own_violation else own_recall,
                f"violation_{domain}": own_violation,
            }
        )
    row["worst_domain_harm"] = max(worst) if worst else None
    row["domain_violation"] = violating
    size = len(ctx.clusters)
    clusters = {
        "decisions": np.bincount(view.cluster, minlength=size),
        "accepted": np.bincount(view.cluster[accept], minlength=size),
        "harmful_accepted": np.bincount(view.cluster[accept & view.harmful], minlength=size),
        "auto_repaired": np.bincount(
            np.asarray([ctx.error_cluster[k] for k in auto], dtype=np.int64), minlength=size
        ),
    }
    return row, clusters


def coverage_of(safety: np.ndarray, cutoff: float | None, above: bool = True) -> float | None:
    """The share of a cell's pooled calibration decisions a cutoff admits: label-free."""
    if cutoff is None or safety.size == 0:
        return None
    return float((safety >= cutoff).mean() if above else (safety <= cutoff).mean())


# ------------------------------------------------------------------ inherited definitions


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    """PF1's summary over draws, plus the per-corpus and safe-recall quantities."""
    summary = pf1.summarize(rows)
    accepting = rows[rows["accepted"] > 0]
    summary.update(
        {
            "median_safe_automated_recall": _median(rows["safe_automated_recall"]),
            "mean_safe_automated_recall": _mean(rows["safe_automated_recall"]),
            "domain_violation_share": (
                float(rows["domain_violation"].mean()) if len(rows) else None
            ),
            "median_worst_domain_harm": _median(accepting["worst_domain_harm"]),
            "worst_draw_domain_harm": (
                float(accepting["worst_domain_harm"].dropna().max())
                if accepting["worst_domain_harm"].notna().any()
                else None
            ),
            "median_worst_environment_harm": _median(accepting["worst_environment_harm"]),
        }
    )
    for domain in DOMAINS:
        own = rows[rows[f"accepted_{domain}"] > 0]
        summary.update(
            {
                f"median_harm_{domain}": _median(own[f"harm_{domain}"]),
                f"median_coverage_{domain}": _median(rows[f"coverage_{domain}"]),
                f"median_safe_coverage_{domain}": _median(rows[f"safe_coverage_{domain}"]),
                f"median_automated_recall_{domain}": _median(rows[f"automated_recall_{domain}"]),
                f"median_safe_automated_recall_{domain}": _median(
                    rows[f"safe_automated_recall_{domain}"]
                ),
                f"violation_share_{domain}": (
                    float(rows[f"violation_{domain}"].mean()) if len(rows) else None
                ),
                f"accepting_draws_{domain}": len(own),
            }
        )
    return summary


def flags(summary: dict[str, Any], policy: str) -> dict[str, bool]:
    """PF1's flags, unchanged: useful for full automation, practical for three-way triage."""
    return pf1.flags(summary, policy)


def minimal_budget(flags_by_budget: dict[str, bool]) -> str | None:
    return pf1.minimal_budget(flags_by_budget)


def earliest(budgets: Sequence[str | None]) -> str | None:
    reached = [b for b in budgets if b is not None]
    return min(reached, key=BUDGETS.index) if reached else None


def assign_outcome(gain: dict[str, bool], favourable: dict[str, bool]) -> tuple[str, bool]:
    """The frozen rubric. `gain[d]`: a domain-aware conservative policy is safe and useful at a
    smaller measured budget than any pooled one (or where no pooled one is). `favourable[d]`: a
    family test in direction d survives Holm in favour of the domain-aware method.

    A: a gain in both directions and a favourable test in both. B: a gain somewhere otherwise;
    strong B when a gained direction also has a favourable test. C: no gain, but a favourable
    test somewhere. D: neither.
    """
    gained = [d for d in DIRECTIONS if gain[d]]
    if len(gained) == len(DIRECTIONS) and all(favourable[d] for d in DIRECTIONS):
        return "A", False
    if gained:
        return "B", any(favourable[d] for d in gained)
    if any(favourable[d] for d in DIRECTIONS):
        return "C", False
    return "D", False


def confirmation_recommended(outcome: str, strong_b: bool) -> bool:
    """Only the rubric recommends a later confirmation, and only for A or strong B."""
    return bool(outcome == "A" or (outcome == "B" and strong_b))


# ------------------------------------------------------------------ page-clustered resampling


def bootstrap_weights(strata: Sequence[str], resamples: int, seed: int) -> np.ndarray:
    """Cluster multiplicities for `resamples` draws, resampling clusters within each stratum."""
    labels = np.asarray(strata)
    size = labels.size
    weights = np.zeros((resamples, size), dtype=np.float64)
    generator = np.random.default_rng(seed)
    for stratum in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == stratum)
        picks = members[generator.integers(0, members.size, size=(resamples, members.size))]
        rows = np.repeat(np.arange(resamples), members.size)
        np.add.at(weights, (rows, picks.ravel()), 1.0)
    return weights


def weighted_harm(weights: np.ndarray, accepted: np.ndarray, harmful: np.ndarray) -> np.ndarray:
    """Accepted harm per resample and draw; NaN where nothing is accepted. Shapes (B,G)x(D,G)."""
    acc = weights @ accepted.T
    bad = weights @ harmful.T
    return np.divide(bad, acc, out=np.full(acc.shape, np.nan), where=acc > 0)


def weighted_metric(
    metric: str, weights: np.ndarray, counts: dict[str, np.ndarray], errors: np.ndarray
) -> np.ndarray:
    """A family metric per resample and draw from per-cluster counts."""
    harm = weighted_harm(weights, counts["accepted"], counts["harmful_accepted"])
    violation = np.nan_to_num(harm, nan=0.0) > EPSILON + HARM_TOLERANCE
    if metric == METRIC_VIOLATION:
        return violation.astype(np.float64)
    if metric == METRIC_SAFE_RECALL:
        recall = (weights @ counts["auto_repaired"].T) / (weights @ errors)[:, None]
        return np.where(violation, 0.0, recall)
    raise PhaseError(f"unknown metric {metric}")


def improvement(metric: str, aware: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    """Positive favours the domain-aware method: fewer violations, more safe recall."""
    return pooled - aware if metric == METRIC_VIOLATION else aware - pooled


def bootstrap_p(resampled: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value: twice the smaller tail mass at zero."""
    finite = resampled[np.isfinite(resampled)]
    if finite.size == 0:
        return 1.0
    below = (int((finite <= 0).sum()) + 1) / (finite.size + 1)
    above = (int((finite >= 0).sum()) + 1) / (finite.size + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def score_adjusted_difference(
    weights: np.ndarray, harmful: np.ndarray, decisions: np.ndarray, sbb: np.ndarray
) -> np.ndarray:
    """Q1's statistic: SBB harm minus FUNSD harm within each score decile, weighted by the
    decile's decisions, over deciles both corpora populate.

    `weights` is (R, G) cluster multiplicities, `harmful` and `decisions` (G, bins) cluster
    counts, `sbb` the (G,) mask of SBB clusters. Returns (R,), NaN where no decile has both.
    """
    h_s = weights[:, sbb] @ harmful[sbb]
    n_s = weights[:, sbb] @ decisions[sbb]
    h_f = weights[:, ~sbb] @ harmful[~sbb]
    n_f = weights[:, ~sbb] @ decisions[~sbb]
    both = (n_s > 0) & (n_f > 0)
    rate_s = np.divide(h_s, n_s, out=np.zeros_like(h_s), where=n_s > 0)
    rate_f = np.divide(h_f, n_f, out=np.zeros_like(h_f), where=n_f > 0)
    weight = np.where(both, n_s + n_f, 0.0)
    total = weight.sum(axis=1)
    return np.divide(
        (weight * (rate_s - rate_f)).sum(axis=1),
        total,
        out=np.full(total.shape, np.nan),
        where=total > 0,
    )


# ------------------------------------------------------------------ section 0: PF1 re-read

PF1_ARTIFACTS = (
    pf1.RESEARCH_FREEZE,
    pf1.DESIGN_RECORD,
    pf1.PAGE_POOL,
    pf1.PAGE_REGISTRY,
    pf1.POPULATION,
    pf1.FEATURE_MATRIX,
    pf1.ERROR_SITES,
    pf1.PROPOSAL_ERROR_LINKS,
    pf1.SPLIT_REGISTRY,
    pf1.PURCHASE_REGISTRY,
    pf1.CELL_SCORES,
    pf1.ADAPTATION_REGISTRY,
    pf1.POLICY_OUTCOMES,
    pf1.RANKING_OUTCOMES,
    pf1.POLICY_REGISTRY,
    pf1.CONTROL_RESULTS,
    pf1.CURVES,
    pf1.FRONTIER,
    pf1.DIVERSITY,
    pf1.BOTTLENECK,
    pf1.STATISTICAL_TESTS,
    pf1.FALSIFICATION,
    pf1.DECISION,
    pf1.DETERMINISM,
    pf1.PROVENANCE,
    pf1.TRACEABILITY,
)
UPSTREAM_CODE = tuple(
    REPO / p
    for p in (
        "scripts/sgv_pf1_page_frontier_scaling.py",
        "scripts/sgv_dep1_deployment_frontier.py",
        "scripts/sgv_rk4_generator_adaptive_ranking.py",
        "scripts/sgv_rk3_risk_aware_ranking.py",
        "scripts/sgv_th1_threshold_calibration.py",
        "scripts/sgv_th2_page_budget_frontier.py",
        "scripts/sgv_ds1_deployment_synthesis.py",
        "src/ocr_risk/risk/bounds.py",
        "src/ocr_risk/risk/cluster_bounds.py",
        "manifests/sgv1/confirmatory_reserve_lock.json",
    )
)
PF1_EXPECTED = {
    "outcome": "C",
    "recommended_next_stage": "NEXT: PAGE-EFFICIENT CALIBRATION EVIDENCE BEFORE ANY NEW RANKER",
    "ready_for_external_confirmation": False,
    "deployment_claim_permitted": False,
}


def upstream_hashes() -> dict[str, str]:
    freeze = cc_read_json(pf1.RESEARCH_FREEZE)["upstream_sha256"]
    paths = set(freeze) | {_relative(p) for p in (*PF1_ARTIFACTS, *UPSTREAM_CODE)}
    return {p: file_sha256(REPO / p) for p in sorted(paths)}


def run_reconstruct() -> int:
    """PF1's state from its artifacts: every value CAL2 inherits is read, checked and hashed."""
    started = time.monotonic()
    _forbid(UPSTREAM_STATE)
    decision = cc_read_json(pf1.DECISION)
    frontier = cc_read_json(pf1.FRONTIER)["primary"]
    splits = cc_read_json(pf1.SPLIT_REGISTRY)
    freeze = cc_read_json(pf1.RESEARCH_FREEZE)
    lock = cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")
    diversity = cc_read_json(pf1.DIVERSITY)["cross_document_robustness_at_the_pool"]
    curves = cc_read_json(pf1.CURVES)["curves"][SET_TEST]
    population = pd.read_parquet(pf1.POPULATION, columns=["corpus", "domain"])
    checks: list[dict[str, Any]] = []

    def check(name: str, expected: Any, observed: Any) -> None:
        checks.append(
            {
                "check": name,
                "expected": _plain(expected),
                "observed": _plain(observed),
                "passed": _plain(expected) == _plain(observed),
            }
        )

    for key, value in PF1_EXPECTED.items():
        check(f"pf1_{key}", value, decision[key])
    for direction in DIRECTIONS:
        check(f"pf1_frontier_{DIRECTION_SHORT[direction]}", None, frontier[direction]["frontier"])
    check("pf1_budget_grid", list(BUDGETS), splits["budgets"])
    check("pf1_draws", DRAWS, len({r["draw"] for r in cc_read_json(pf1.PURCHASE_REGISTRY)["rows"]}))
    check("pf1_determinism", True, cc_read_json(pf1.DETERMINISM)["all_runs_identical"])
    check("pf1_untraceable_claims", 0, cc_read_json(pf1.TRACEABILITY)["untraceable"].__len__())
    check("pf1_falsification", True, cc_read_json(pf1.FALSIFICATION)["passed"] == 23)
    check("corpora", sorted(DOMAINS), sorted(population["corpus"].astype(str).unique()))
    check("reserve_lock", "LOCKED", lock["status"])
    moved = sorted(
        p for p, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / p) != sha
    )
    check("pf1_upstream_unmoved", [], moved)
    failed = [c["check"] for c in checks if not c["passed"]]
    motivating = {
        direction: {
            policy: {
                corpus: diversity[direction][policy]["groups"][f"corpus={corpus}"]
                for corpus in DOMAINS
            }
            for policy in pf1.BREAKDOWN_POLICIES
        }
        for direction in DIRECTIONS
    }
    pool_curves = {
        direction: {
            "median_harm_auroc": curves[direction][M1][SCHEME_RANDOM][POOL]["ranking"][
                "median_harm_auroc"
            ],
            "median_oracle_coverage": curves[direction][M1][SCHEME_RANDOM][POOL]["ranking"][
                "median_oracle_coverage"
            ],
            "p0_median_harm": curves[direction][M1][SCHEME_RANDOM][POOL]["policies"][P0][
                "median_harm"
            ],
            "p1_accepting_draws": curves[direction][M1][SCHEME_RANDOM][POOL]["policies"][P1][
                "accepting_draws"
            ],
        }
        for direction in DIRECTIONS
    }
    _write_json_once(
        UPSTREAM_STATE,
        {
            **_envelope("upstream_state"),
            "pf1": {
                "outcome": decision["outcome"],
                "recommended_next_stage": decision["recommended_next_stage"],
                "frontier": {d: frontier[d]["frontier"] for d in DIRECTIONS},
                "pool_pages": splits["pool_pages"],
                "budgets": splits["budgets"],
                "pages_generated": decision["pages_generated"],
                "pool_ranking_and_policies": pool_curves,
                "motivating_corpus_breakdown_at_the_pool": motivating,
            },
            "checks": checks,
            "failed_checks": failed,
            "upstream_sha256": upstream_hashes(),
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if failed:
        raise PhaseError(f"PF1's state is not the frozen state: {failed}")
    print(f"reconstruct: {len(checks)} checks pass ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the design, before outcomes

RESULT_ARTIFACTS = (C0_REPRODUCTION, POLICY_ROWS, THRESHOLD_RECORDS, CONTROL_RESULTS)


def dependence_structure() -> dict[str, Any]:
    """Label-free: pages and groups a direction fits or calibrates on that share a group with a
    PF1 test page. PF1's folds keep its own volumes whole; U5 pages in RK4's blocks predate it."""
    study = pf1.load_study()
    population = study.population
    documents = population["document_id"].astype(str).to_numpy()
    corpora = population["corpus"].astype(str).to_numpy()
    out: dict[str, Any] = {}
    for direction, b in study.blocks.items():
        fitting_rows = np.concatenate([b.source_fit, b.source_calibration, b.pool])
        fitting = {(corpora[i], documents[i]) for i in fitting_rows}
        tested = {(corpora[i], documents[i]) for i in b.test}
        fit_groups = {group_of(c, d) for c, d in fitting}
        shared = sorted({group_of(c, d) for c, d in tested} & fit_groups)
        out[direction] = {
            "test_pages_also_fitting": len({d for _c, d in tested} & {d for _c, d in fitting}),
            "groups_shared_with_test": shared,
            "test_pages_in_shared_groups": sorted(d for c, d in tested if group_of(c, d) in shared),
            "fitting_pages_in_shared_groups": sorted(
                d for c, d in fitting if group_of(c, d) in shared
            ),
        }
    return out


def design_payload() -> dict[str, Any]:
    """Every choice the analysis makes, fixed before any CAL2 comparison exists."""
    pools = pf1_pool_pages()
    return {
        "question": (
            "Does calibration that respects the pre-existing document regimes (FUNSD forms, SBB "
            "historical print) give a safer and more useful operating boundary than one pooled "
            "boundary, on the same frozen ranker, draws and test pages?"
        ),
        "questions": QUESTIONS,
        "evidence_status": EVIDENCE_STATUS,
        "evidence_note": EVIDENCE_NOTE,
        "frozen_components": [
            "OCR outputs and page images (SGV14's frozen canonical OCR; not re-run)",
            "proposer outputs and correction candidates (PF1; not regenerated)",
            "candidate labels and the R5 feature matrix (PF1 and RK3)",
            "RK3's model specification and RK4's adaptation design (M1, A0 at budget 0)",
            "PF1's page registry, folds, purchases, cell scores and test labels",
            "DEP1's four policies and PF1's lost-repair ceiling",
        ],
        "not_run": ["OCR", "candidate generation", "ranker refits", "language-model calls"],
        "population": {
            "domains": DOMAIN_LABEL,
            "domain_rule": (
                "a decision's domain is its page's corpus, observable at inference and fixed "
                "before any outcome; it is never redefined after results"
            ),
            "strata_use_outcomes": False,
            "evidence_unit": "the partition group: a FUNSD form, or an SBB volume",
        },
        "budgets": {
            "grid": list(BUDGETS),
            "source": "PF1's split_registry.json budgets and pool_pages",
            "pool_pages": pools,
            "draws": DRAWS,
            "calibration_every": CALIBRATION_EVERY,
            "semantics": (
                "PF1's: a budget buys a prefix of the draw's seeded page order; every third "
                "purchased page is held back for calibration; budget 0 is A0 with RK4's source "
                "calibration block; the pool buys every pool page in the draw's order"
            ),
            "stratified_grid": list(STRATIFIED_BUDGETS),
            "not_measured": {
                "c2_at_5_pages_and_the_pool": (
                    "PF1's frozen environment-stratified purchase exists at 10, 25, 50 and 100 "
                    "pages only; no new purchase or refit is made"
                ),
            },
            "extrapolation": "no extrapolated budget may satisfy a criterion",
        },
        "ranker": {
            "arm": M1,
            "zero_budget_arm": A0,
            "scores": "PF1's cell_scores.parquet, read only; the ranker is never refitted",
            "ranking_quality": "identical across methods by construction: the scores are shared",
        },
        "methods": {
            C0: {
                "label": METHOD_LABEL[C0],
                "rule": "dep1.bands on every calibration decision of the cell",
                "gate": "must reproduce PF1's policy_outcomes rows exactly before --calibrate",
            },
            C1: {
                "label": METHOD_LABEL[C1],
                "rule": (
                    "dep1.bands per corpus: each corpus's test decisions get the cutoffs its own "
                    "calibration decisions give; a corpus with none gets no accept cutoff"
                ),
                "same_target_every_domain": True,
            },
            C2: {
                "label": METHOD_LABEL[C2],
                "purchase": pf1.SCHEME_LABEL[SCHEME_STRATIFIED],
                "why_it_balances_corpora": (
                    "each corpus holds five engine environments, so a round robin over the ten "
                    "environments buys the two corpora in equal numbers until one corpus's "
                    "environments run out; the allocation reads page and environment ids only"
                ),
                "frozen_before": "PF1's design record, before any PF1 label",
                "budgets": list(STRATIFIED_BUDGETS),
            },
            C3: {
                "label": METHOD_LABEL[C3],
                "estimator": inspect.getdoc(hierarchical_cutoff),
                "prior_pages": PRIOR_PAGES[C3],
                "sensitivity_prior_pages": [PRIOR_PAGES[m] for m in C3_SENSITIVITY],
                "sensitivity_is_analysis_only": True,
                "lattice": "every distinct pooled calibration score",
                "design_effect": "the pooled calibration pages' page design effect (TH1's)",
                "prior_mean": "the pooled calibration harm rate at the same cutoff",
                "prior_rationale": (
                    "fixed before any CAL2 comparison: five average pages, so a corpus with a "
                    "handful of own calibration pages leans on the pooled rate and one with "
                    "dozens relies on its own; not tuned on any outcome"
                ),
                "guarantee": "approximation; not a finite-sample certificate",
            },
        },
        "policies": dep1.POLICY_LABEL,
        "rules": {p: list(r) for p, r in POLICY_RULES.items()},
        "conservative_policies": list(CONSERVATIVE),
        "targets": {
            "epsilon": EPSILON,
            "eta": ETA,
            "delta": DELTA,
            "harm_tolerance": HARM_TOLERANCE,
            "same_for_every_domain": True,
        },
        "definitions": {
            "inherited_from": "PF1's design record, unchanged",
            "pf1_definitions": cc_read_json(pf1.DESIGN_RECORD)["definitions"],
            "violation_ceiling": VIOLATION_CEILING,
            "working_floor": WORKING_FLOOR,
            "useful_coverage_floor": USEFUL_COVERAGE_FLOOR,
            "review_reduction_floor": REVIEW_REDUCTION_FLOOR,
            "lost_repair_ceiling": LOST_REPAIR_CEILING,
            "primary_flags": PRIMARY_FLAG,
            "safe_automated_recall": (
                "a draw's exact-repair recall of accepted decisions when its accepted harm is "
                "within the target, zero when it violates"
            ),
            "domain_violation": "any corpus whose accepted test harm exceeds the target",
            "frontier": (
                "PF1's N*: the smallest measured budget from which a flag holds at every larger "
                "measured budget, masked by the same method's random-score control"
            ),
            "gain": (
                "a domain-aware conservative policy (C1, C2 or C3 under P1 or P2) reaches the "
                "frontier at a smaller measured budget than any pooled conservative policy, or "
                "reaches it where no pooled one does"
            ),
        },
        "controls": {
            "random_scores": {
                "seed": CONTROL_SEED,
                "rule": "PF1's: uniform safety on test and calibration decisions, every method",
            },
            "calibration_label_permutation": {
                "seed": LABEL_PERMUTATION_SEED,
                "draws": LABEL_PERMUTATION_DRAWS,
                "budget": POOL,
                "rule": "calibration labels permuted within each corpus; test labels untouched",
            },
            "pf1_refit_permutation": "PF1's control_results.json, read, not recomputed",
        },
        "mechanism": {
            "M1_risk_curves": (
                "harm by score decile per corpus at the pool, deciles of each draw's pooled "
                "decisions, on calibration pages (primary) and test pages (exploratory replay); "
                "page-clustered bootstrap bands"
            ),
            "M2_threshold_disagreement": {
                "statistic": (
                    "mean over draws of the pooled-calibration coverage at SBB's C1 cutoff minus "
                    "that at FUNSD's, at the pool"
                ),
                "null": "corpus labels permuted over calibration groups, group counts kept",
                "permutations": DOMAIN_PERMUTATIONS,
                "seed": DOMAIN_PERMUTATION_SEED,
                "rules": [R_PLUG_IN, R_PAGE_BOUND],
            },
            "M3_effective_evidence": "pages, groups, decisions, design effect, n / deff per corpus",
            "M4_gap": (
                "test-oracle frontier per corpus (analysis only) against every deployable "
                "method's safe coverage and safe recall"
            ),
            "risk_bins": RISK_BIN_COUNT,
            "Q6_rule": (
                "analysis only, at the pool, on P1's safe exact-repair recall: the evidence "
                "component is the pooled test-oracle recall minus C0's; the heterogeneity "
                "component is the per-corpus test-oracle recall minus the pooled test-oracle "
                "recall; the gap is 'ranking' when the per-corpus oracle covers under PF1's "
                "oracle floor of 0.05, 'combination' when the smaller positive component is at "
                "least half the larger, and otherwise the larger component"
            ),
        },
        "statistical_plan": {
            "family": FAMILY_SPEC,
            "family_budget": FAMILY_BUDGET,
            "effect": (
                "mean over paired draws of the improvement: pooled minus domain-aware violation, "
                "domain-aware minus pooled safe recall"
            ),
            "p_value": (
                "the larger of the exact two-sided sign test over the paired draws and the "
                "two-sided percentile bootstrap over test groups resampled within corpus, so a "
                "test must hold on the draw axis and on the page axis"
            ),
            "holm_alpha": HOLM_ALPHA,
            "survives_in_favour": "Holm-adjusted p below alpha and a positive effect",
            "domain_family": {
                "tests": list(DOMAIN_FAMILY),
                "statistic": inspect.getdoc(score_adjusted_difference),
                "evidence": "pool calibration pages (primary); test pages reported, not tested",
                "measurably_different": "survives Holm over the two directions",
            },
            "secondary": SECONDARY,
            "secondary_note": "reported with unadjusted p-values; none enters a criterion",
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": "the partition group, resampled within corpus",
            },
            "draws_note": (
                "the 20 draws overlap in their pages and are conditional on the available page "
                "population; they are not 20 independent experiments"
            ),
        },
        "criteria": {
            "C1_safety_improvement": (
                "an S1 test survives Holm in favour of C1 at the pool (fewer P0 violations), "
                "with ranking quality identical by construction"
            ),
            "C2_safe_and_useful": (
                "a domain-aware conservative policy's primary flag holds at a measured budget: "
                "P1 useful at coverage 0.05 (violation share <= 0.1, works in at least half the "
                "draws) or P2 practical (both constraints, review reduction 0.25, lost-repair "
                "share <= 0.1), masked by the random control"
            ),
            "C3_recall_improvement": (
                "an R1 or R2 test survives Holm in favour of the domain-aware method at the pool"
            ),
            "C4_breadth": "every criterion is recorded per direction; none is aggregated",
            "C5_page_efficiency": (
                "C2 reaches a conservative frontier at a smaller measured budget than C1 with "
                "random purchase; otherwise no page-efficiency gain is reported"
            ),
            "no_oracle_in_criteria": True,
        },
        "outcome_rubric": {
            "taxonomy": OUTCOME_TAXONOMY,
            "interpretation": INTERPRETATION,
            "rule": inspect.getdoc(assign_outcome),
            "resolution": (
                "the brief's B and C overlap on calibration quality; the rubric separates them "
                "by whether a domain-aware conservative operating point is safe and useful"
            ),
        },
        "stopping_rule": STOPPING_RULE,
        "confirmation": {
            "ready_for_external_confirmation": (
                "true only if the rubric returns A or strong B; a recommendation, never executed"
            ),
            "confirmatory_reserve": "LOCKED and unread",
            "production_ready": False,
            "certified": False,
        },
        "dependence": {
            "structure": dependence_structure(),
            "sensitivity": (
                "the family's point estimates are recomputed without the test groups that share "
                "a volume with a fitting page; analysis only"
            ),
        },
        "difficulty_stratified_analysis": (
            "not implemented: optional and secondary in the brief; no difficulty strata defined"
        ),
        "synthetic_controls": {
            **SYNTHETIC,
            "seed": SYNTHETIC_SEED,
            "replications": SYNTHETIC_REPLICATIONS,
            "bootstrap": SYNTHETIC_BOOTSTRAP,
            "identical_pass": (
                "C1's mean safe recall exceeds C0's by at most 0.01 under both rules, the Q1 "
                "interval excludes zero in at most 0.10 of replications, and SBB's plug-in "
                "cutoff is the stricter in between 0.3 and 0.7 of them"
            ),
            "shifted_pass": (
                "C0's plug-in SBB violation share is at least 0.8, C1's is at least 0.3 lower, "
                "SBB's plug-in cutoff is the stricter in at least 0.9 of replications, and the "
                "Q1 interval lies above zero in at least 0.8"
            ),
            "synthetic": True,
        },
        "leakage_rules": [
            "cutoffs read calibration decisions only; test labels never enter a boundary",
            "a budget's calibration pages are a prefix-defined subset of the draw's purchase",
            "no test page fits or calibrates in any cell",
            "oracle cutoffs are analysis only and enter no criterion",
            "PF1's test pages are an exploratory replay, not confirmation",
        ],
    }


def run_design() -> int:
    started = time.monotonic()
    _require(UPSTREAM_STATE, "reconstruct")
    _forbid(DESIGN)
    existing = [_relative(p) for p in RESULT_ARTIFACTS if p.exists()]
    if existing:
        raise PhaseError(f"the design must precede every comparison; found {existing}")
    _write_json_once(
        DESIGN,
        {
            **_envelope("design"),
            **design_payload(),
            "upstream_state_sha256": file_sha256(UPSTREAM_STATE),
            "written_before_any_cal2_comparison": True,
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"design: frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ manifest and draws

ROLES = ("source_fit", "source_calibration", "pool", "test")


def run_manifest() -> int:
    """The population manifest and every draw's purchase, before any calibration."""
    started = time.monotonic()
    _require(DESIGN, "design")
    for path in (POPULATION_MANIFEST, PAGE_BUDGET_DRAWS, DRAW_REGISTRY):
        _forbid(path)
    study = pf1.load_study()
    population = study.population
    pf1_pages = set(pf1.load_pool()["document_id"].astype(str))
    frozen = {
        (r["direction"], r["scheme"], r["budget"], r["draw"]): r
        for r in cc_read_json(pf1.PURCHASE_REGISTRY)["rows"]
    }
    dependence = cc_read_json(DESIGN)["dependence"]["structure"]
    manifest: list[pd.DataFrame] = []
    draws: list[dict[str, Any]] = []
    digests_equal = True
    realization: dict[str, Any] = {}
    for direction, b in study.blocks.items():
        shared = set(dependence[direction]["groups_shared_with_test"])
        for role in ROLES:
            rows = population.iloc[getattr(b, role)]
            grouped = (
                rows.groupby(["document_id", "environment", "corpus", "domain"], sort=True)
                .agg(
                    candidates=("candidate_id", "size"),
                    decision_sites=("site_group", "nunique"),
                    exact_repair_candidates_analysis_only=("exact", "sum"),
                )
                .reset_index()
            )
            grouped["document_group"] = [
                group_of(c, d)
                for c, d in zip(grouped["corpus"], grouped["document_id"], strict=True)
            ]
            grouped = grouped.assign(
                direction=direction,
                target_generator=b.target,
                candidate_generator=b.source if role.startswith("source") else b.target,
                split_role=role,
                page_id=grouped["document_id"],
                ocr_engine=grouped["environment"].astype(str).str.split("/").str[-1],
                pf1_page=grouped["document_id"].isin(pf1_pages),
                group_shared_with_test=grouped["document_group"].isin(shared),
            )
            manifest.append(grouped)
        pages = b.pages
        corpus_of = population.groupby("document_id")["corpus"].first().astype(str).to_dict()
        for scheme in SCHEMES:
            for label in pf1.scheme_budgets(scheme)[1:]:
                bought_sets: list[frozenset[str]] = []
                composition: list[dict[str, int]] = []
                for draw in range(DRAWS):
                    fit, calibration = pf1.purchase(
                        pages, draw, label, scheme, study.environment_of
                    )
                    record = frozen[(direction, scheme, label, draw)]
                    digests_equal &= bool(
                        pf1._digest(population, rk4.rows_on(population, b.pool, fit))
                        == record["fit_digest"]
                        and pf1._digest(population, rk4.rows_on(population, b.pool, calibration))
                        == record["calibration_digest"]
                    )
                    order = pf1.page_order(pages, draw, scheme, study.environment_of)
                    position = {p: i for i, p in enumerate(order)}
                    for role, bought in (("fit", fit), ("calibration", calibration)):
                        for page in bought:
                            draws.append(
                                {
                                    "direction": direction,
                                    "scheme": scheme,
                                    "budget": label,
                                    "budget_pages": len(fit) + len(calibration),
                                    "draw": draw,
                                    "page_id": page,
                                    "purchase_position": position[page],
                                    "role": role,
                                    "corpus": corpus_of[page],
                                    "document_group": group_of(corpus_of[page], page),
                                    "environment": study.environment_of[page],
                                }
                            )
                    bought_sets.append(frozenset(fit + calibration))
                    composition.append(
                        {
                            f"{role}_{domain}": sum(1 for p in bought if corpus_of[p] == domain)
                            for role, bought in (("fit", fit), ("calibration", calibration))
                            for domain in DOMAINS
                        }
                    )
                pairs = [
                    len(a & c) / len(a | c)
                    for i, a in enumerate(bought_sets)
                    for c in bought_sets[i + 1 :]
                ]
                frame = pd.DataFrame(composition)
                realization.setdefault(direction, {}).setdefault(scheme, {})[label] = {
                    "pages_bought": len(bought_sets[0]),
                    "distinct_page_sets": len(set(bought_sets)),
                    "mean_pairwise_jaccard": float(np.mean(pairs)),
                    "independent_across_draws": len(set(bought_sets)) == DRAWS
                    and float(np.mean(pairs)) < 1.0,
                    "median_composition": {c: float(frame[c].median()) for c in frame.columns},
                }
        for role, rows in (
            ("source_fit", b.source_fit),
            ("source_calibration", b.source_calibration),
        ):
            for page in sorted(set(population.iloc[rows]["document_id"].astype(str))):
                draws.append(
                    {
                        "direction": direction,
                        "scheme": SCHEME_RANDOM,
                        "budget": "0",
                        "budget_pages": 0,
                        "draw": 0,
                        "page_id": page,
                        "purchase_position": -1,
                        "role": role,
                        "corpus": corpus_of[page],
                        "document_group": group_of(corpus_of[page], page),
                        "environment": study.environment_of.get(page, ""),
                    }
                )
    if not digests_equal:
        raise PhaseError("a purchase differs from PF1's frozen purchase registry")
    table = pd.concat(manifest, ignore_index=True)
    columns = [
        "direction",
        "target_generator",
        "candidate_generator",
        "split_role",
        "corpus",
        "domain",
        "page_id",
        "document_id",
        "document_group",
        "environment",
        "ocr_engine",
        "pf1_page",
        "group_shared_with_test",
        "candidates",
        "decision_sites",
        "exact_repair_candidates_analysis_only",
    ]
    table = table[columns].sort_values(
        ["direction", "split_role", "corpus", "page_id", "environment"], kind="stable"
    )
    draw_table = pd.DataFrame(draws).sort_values(
        ["direction", "scheme", "budget", "draw", "purchase_position", "page_id"], kind="stable"
    )
    _write_csv_once(POPULATION_MANIFEST, table.reset_index(drop=True))
    _write_csv_once(PAGE_BUDGET_DRAWS, draw_table.reset_index(drop=True))
    test = table[table["split_role"] == "test"]
    _write_json_once(
        DRAW_REGISTRY,
        {
            **_envelope("draw_registry"),
            "manifest_rows": len(table),
            "draw_rows": len(draw_table),
            "pages_by_direction_and_role": {
                d: {
                    r: {
                        c: int(
                            table[
                                (table["direction"] == d)
                                & (table["split_role"] == r)
                                & (table["corpus"] == c)
                            ]["page_id"].nunique()
                        )
                        for c in DOMAINS
                    }
                    for r in ROLES
                }
                for d in DIRECTIONS
            },
            "groups_by_direction_and_role": {
                d: {
                    r: {
                        c: int(
                            table[
                                (table["direction"] == d)
                                & (table["split_role"] == r)
                                & (table["corpus"] == c)
                            ]["document_group"].nunique()
                        )
                        for c in DOMAINS
                    }
                    for r in ROLES
                }
                for d in DIRECTIONS
            },
            "test_pages": int(test["page_id"].nunique()),
            "purchases_match_pf1_registry": digests_equal,
            "budget_realization": realization,
            "pool_budget_note": (
                "at the pool every draw buys the same pages; only the order, and so which third "
                "is held back for calibration, differs"
            ),
            "exact_repair_counts": "analysis only; never used to form strata or allocate pages",
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"manifest: {len(table)} rows, {len(draw_table)} purchased-page rows, purchases match "
        f"PF1 {digests_equal} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ C0 reproduces PF1

KEY_COLUMNS = (*ROW_KEYS, "policy")


def c0_rows(cells: dict[CellKey, Cell], ctx: Context) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, cell in cells.items():
        for policy in POLICIES:
            bands = pooled_bands(policy, cell.calibration, cell.test.frame, cell.usable)
            row, _clusters = evaluate(policy, cell.test, bands, ctx)
            rows.append(dict(zip(ROW_KEYS, key, strict=True)) | row)
    return pd.DataFrame(rows)


def _same_values(left: pd.Series, right: pd.Series) -> np.ndarray:
    a = left.to_numpy(object)
    b = right.to_numpy(object)
    out = np.zeros(a.size, dtype=bool)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        x_missing = x is None or (isinstance(x, float) and x != x)
        y_missing = y is None or (isinstance(y, float) and y != y)
        out[i] = (x_missing and y_missing) or (not x_missing and not y_missing and x == y)
    return out


def compare_to_pf1(ours: pd.DataFrame) -> dict[str, Any]:
    theirs = pd.read_parquet(pf1.POLICY_OUTCOMES)
    theirs = theirs[theirs["arm"].isin([A0, M1]) & (theirs["evaluation_set"] == SET_TEST)]
    columns = [c for c in theirs.columns if c not in (*KEY_COLUMNS, "evaluation_set")]
    joined = theirs.merge(ours, on=list(KEY_COLUMNS), how="outer", suffixes=("_pf1", "_cal2"))
    mismatched: dict[str, int] = {}
    for column in columns:
        same = _same_values(joined[f"{column}_pf1"], joined[f"{column}_cal2"])
        if not same.all():
            mismatched[column] = int((~same).sum())
    return {
        "pf1_rows": len(theirs),
        "cal2_rows": len(ours),
        "joined_rows": len(joined),
        "columns_compared": columns,
        "mismatched_columns": mismatched,
        "equal": bool(len(theirs) == len(ours) == len(joined) and not mismatched),
    }


def compare_breakdown(ours: pd.DataFrame) -> dict[str, Any]:
    """PF1's per-corpus breakdown at the pool, from C0's per-corpus columns."""
    published = cc_read_json(pf1.DIVERSITY)["cross_document_robustness_at_the_pool"]
    mismatches: list[str] = []
    for direction in DIRECTIONS:
        for policy in pf1.BREAKDOWN_POLICIES:
            rows = ours[
                (ours["direction"] == direction)
                & (ours["arm"] == M1)
                & (ours["scheme"] == SCHEME_RANDOM)
                & (ours["budget"] == POOL)
                & (ours["policy"] == policy)
            ]
            for domain in DOMAINS:
                entry = published[direction][policy]["groups"][f"corpus={domain}"]
                accepting = rows[rows[f"accepted_{domain}"] > 0]
                mine = {
                    "median_coverage": _median(rows[f"coverage_{domain}"]),
                    "median_harm": _median(accepting[f"harm_{domain}"]),
                    "accepting_draws": len(accepting),
                }
                for name, value in mine.items():
                    if entry[name] != value:
                        mismatches.append(f"{direction}|{policy}|{domain}|{name}")
    return {"mismatches": mismatches, "equal": not mismatches}


def run_reproduce() -> int:
    """C0 from PF1's frozen scores must equal PF1's policy outcomes before CAL2 proceeds."""
    started = time.monotonic()
    _require(DRAW_REGISTRY, "manifest")
    _forbid(C0_REPRODUCTION)
    cells, ctx = load_cells()
    ours = c0_rows(cells, ctx)
    comparison = compare_to_pf1(ours)
    breakdown = compare_breakdown(ours)
    _write_json_once(
        C0_REPRODUCTION,
        {
            **_analysis_envelope("c0_reproduction"),
            **comparison,
            "pf1_corpus_breakdown": breakdown,
            "reproduces_pf1_exactly": bool(comparison["equal"] and breakdown["equal"]),
            "cells": len(cells),
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if not (comparison["equal"] and breakdown["equal"]):
        raise PhaseError(f"C0 does not reproduce PF1: {comparison['mismatched_columns']}")
    print(
        f"reproduce: C0 equals PF1 on {comparison['pf1_rows']} rows x "
        f"{len(comparison['columns_compared'])} columns ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ calibration


COUNT_FIELDS = ("decisions", "accepted", "harmful_accepted", "auto_repaired")


@dataclass(slots=True)
class Run:
    """Every outcome row, cutoff record and cluster count a set of cells produced."""

    rows: list[dict[str, Any]]
    records: list[dict[str, Any]]
    keys: list[tuple[Any, ...]]
    counts: dict[str, list[np.ndarray]]

    @classmethod
    def empty(cls) -> Run:
        return cls([], [], [], {f: [] for f in COUNT_FIELDS})

    def cluster_frame(self, ctx: Context) -> pd.DataFrame:
        size = len(ctx.clusters)
        repeat = np.repeat(np.arange(len(self.keys)), size)
        frame = pd.DataFrame(
            [self.keys[i] for i in range(len(self.keys))],
            columns=[*ROW_KEYS, "method", "policy"],
        ).iloc[repeat]
        frame = frame.reset_index(drop=True).assign(
            cluster=np.tile(np.asarray(ctx.clusters, dtype=object), len(self.keys)),
            cluster_domain=np.tile(np.asarray(ctx.cluster_domain, dtype=object), len(self.keys)),
        )
        for name in COUNT_FIELDS:
            frame[name] = np.concatenate(self.counts[name]).astype(np.int64)
        return frame


def run_one(
    cell: Cell,
    calibration: pd.DataFrame,
    view: TestView,
    ctx: Context,
    methods: Sequence[str],
    run: Run,
    usable: bool | None = None,
) -> None:
    usable = cell.usable if usable is None else usable
    base = dict(zip(ROW_KEYS, cell.key, strict=True))
    pooled = calibration["safety"].to_numpy(np.float64) if len(calibration) else np.empty(0)
    for method in methods:
        for policy in POLICIES:
            bands = method_bands(method, policy, calibration, view.frame, usable)
            row, counts = evaluate(policy, view, bands, ctx)
            head = base | {"method": method, "prior_pages": PRIOR_PAGES.get(method)}
            run.rows.append(head | row)
            accept_rule, reject_rule = POLICY_RULES[policy]
            for record in bands.records:
                run.records.append(
                    head
                    | {
                        "policy": policy,
                        "accept_rule": accept_rule,
                        "reject_rule": reject_rule,
                        "epsilon": EPSILON,
                        "eta": ETA if reject_rule != R_ALL_REST else None,
                        "delta": DELTA,
                        "usable": usable,
                        **record,
                        "accept_cutoff_coverage": coverage_of(pooled, record["accept_cutoff"]),
                        "reject_cutoff_coverage": coverage_of(
                            pooled, record["reject_cutoff"], above=False
                        ),
                    }
                )
            run.keys.append((*cell.key, method, policy))
            for name in COUNT_FIELDS:
                run.counts[name].append(counts[name])


RECORD_COLUMNS = (
    *ROW_KEYS,
    "method",
    "prior_pages",
    "policy",
    "domain",
    "accept_rule",
    "reject_rule",
    "epsilon",
    "eta",
    "delta",
    "usable",
    "accept_cutoff",
    "reject_cutoff",
    "accept_cutoff_coverage",
    "reject_cutoff_coverage",
    "accept_design_effect",
    "reject_design_effect",
    "calibration_pages",
    "calibration_groups",
    "calibration_decisions",
    "calibration_harmful",
    "calibration_exact",
)
ORDER = [*ROW_KEYS, "method", "policy"]


def _sorted(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    return frame.sort_values([*ORDER, *extra], kind="stable").reset_index(drop=True)


def run_calibrate() -> int:
    """C0, C1 and C3 (with its prior sensitivity) on every CAL2 cell and both purchases."""
    started = time.monotonic()
    _require(C0_REPRODUCTION, "reproduce")
    if not cc_read_json(C0_REPRODUCTION)["reproduces_pf1_exactly"]:
        raise PhaseError("C0 did not reproduce PF1; CAL2 does not proceed")
    outputs = (
        POLICY_ROWS,
        CLUSTER_COUNTS,
        THRESHOLD_RECORDS,
        POOLED_RESULTS,
        CORPUS_STRATIFIED_RESULTS,
        HIERARCHICAL_RESULTS,
        STRATIFIED_PURCHASE_RESULTS,
        CALIBRATION_REGISTRY,
    )
    for path in outputs:
        _forbid(path)
    cells, ctx = load_cells()
    run = Run.empty()
    for index, cell in enumerate(cells.values()):
        run_one(cell, cell.calibration, cell.test, ctx, METHODS, run)
        if index % 50 == 0:
            print(f"  {index}/{len(cells)} cells ({time.monotonic() - started:.0f}s)", flush=True)
    rows = _sorted(pd.DataFrame(run.rows))
    pooled = rows[rows["method"] == C0].drop(columns=["method", "prior_pages"])
    again = compare_to_pf1(pooled)
    if not again["equal"]:
        raise PhaseError(f"C0 drifted from PF1 inside --calibrate: {again['mismatched_columns']}")
    records = _sorted(pd.DataFrame(run.records)[list(RECORD_COLUMNS)], ["domain"])
    clusters = _sorted(run.cluster_frame(ctx), ["cluster"])
    _write_parquet_once(POLICY_ROWS, rows)
    _write_parquet_once(CLUSTER_COUNTS, clusters)
    _write_csv_once(THRESHOLD_RECORDS, records)
    random = rows["scheme"] == SCHEME_RANDOM
    _write_csv_once(POOLED_RESULTS, rows[random & (rows["method"] == C0)].reset_index(drop=True))
    _write_csv_once(
        CORPUS_STRATIFIED_RESULTS, rows[random & (rows["method"] == C1)].reset_index(drop=True)
    )
    _write_csv_once(
        HIERARCHICAL_RESULTS,
        rows[random & rows["method"].isin([C3, *C3_SENSITIVITY])].reset_index(drop=True),
    )
    _write_csv_once(
        STRATIFIED_PURCHASE_RESULTS,
        rows[rows["scheme"] == SCHEME_STRATIFIED].reset_index(drop=True),
    )
    _write_json_once(
        CALIBRATION_REGISTRY,
        {
            **_analysis_envelope("calibration_registry"),
            "cells": len(cells),
            "methods": METHOD_LABEL,
            "prior_pages": PRIOR_PAGES,
            "policy_rows": len(rows),
            "threshold_records": len(records),
            "cluster_rows": len(clusters),
            "test_clusters": {d: sum(1 for c in ctx.cluster_domain if c == d) for d in DOMAINS},
            "recall_denominator": {"all": len(ctx.keys), **ctx.errors_by_domain},
            "c0_equals_pf1_inside_calibrate": again["equal"],
            "cutoffs_chosen_without_test_labels": True,
            "epsilon_every_record": sorted(set(records["epsilon"].tolist())),
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"calibrate: {len(rows)} rows, {len(records)} cutoffs over {len(cells)} cells "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def load_rows() -> pd.DataFrame:
    return pd.read_parquet(POLICY_ROWS)


def select(
    rows: pd.DataFrame, direction: str, method: str, scheme: str, budget: str, policy: str
) -> pd.DataFrame:
    """One budget's draws; budget 0 is PF1's A0 cell for every scheme."""
    d, arm, sc, b, _draw = cell_for(direction, scheme, budget, 0)
    block = rows[
        (rows["direction"] == d)
        & (rows["arm"] == arm)
        & (rows["scheme"] == sc)
        & (rows["budget"] == b)
        & (rows["method"] == method)
        & (rows["policy"] == policy)
    ]
    return block.sort_values("draw", kind="stable")


def scheme_budgets(scheme: str) -> tuple[str, ...]:
    return pf1.scheme_budgets(scheme)


# ------------------------------------------------------------------ controls


def random_run(cells: dict[CellKey, Cell], ctx: Context) -> Run:
    """PF1's random-score control, through every primary method, on every M1 cell."""
    run = Run.empty()
    for key, cell in cells.items():
        direction, arm, scheme, label, draw = key
        if arm != M1:
            continue
        generator = (
            pf1._random_generator(direction, label, draw)
            if scheme == SCHEME_RANDOM
            else pf1._random_generator(direction, scheme, label, draw)
        )
        test = cell.test.frame.assign(safety=generator.random(len(cell.test.frame)))
        calibration = cell.calibration.assign(safety=generator.random(len(cell.calibration)))
        view = TestView(
            test,
            cell.test.harmful,
            cell.test.exact,
            cell.test.corpus,
            cell.test.cluster,
            cell.test.repairs,
        )
        run_one(cell, calibration, view, ctx, PRIMARY_METHODS, run, usable=True)
    return run


def permuted_calibration(calibration: pd.DataFrame, direction: str, draw: int) -> pd.DataFrame:
    """Calibration labels permuted within each corpus, jointly, seeded; scores untouched."""
    generator = np.random.default_rng([LABEL_PERMUTATION_SEED, DIRECTIONS.index(direction), draw])
    out = calibration.copy()
    corpus = out["corpus"].astype(str).to_numpy()
    harmful = out["is_harmful"].to_numpy(bool).copy()
    exact = out["exact"].to_numpy(bool).copy()
    for domain in DOMAINS:
        members = np.flatnonzero(corpus == domain)
        order = generator.permutation(members.size)
        harmful[members] = harmful[members][order]
        exact[members] = exact[members][order]
    return out.assign(is_harmful=harmful, exact=exact)


def summary_block(rows: pd.DataFrame, policy: str) -> dict[str, Any]:
    summary = summarize(rows)
    return summary | flags(summary, policy)


def run_controls() -> int:
    started = time.monotonic()
    _require(CALIBRATION_REGISTRY, "calibrate")
    _forbid(CONTROL_RESULTS)
    cells, ctx = load_cells()
    random = pd.DataFrame(random_run(cells, ctx).rows)
    random_summary: dict[str, Any] = {}
    passing: list[str] = []
    for direction in DIRECTIONS:
        for method in PRIMARY_METHODS:
            for scheme in SCHEMES:
                for label in scheme_budgets(scheme)[1:]:
                    for policy in POLICIES:
                        block = random[
                            (random["direction"] == direction)
                            & (random["method"] == method)
                            & (random["scheme"] == scheme)
                            & (random["budget"] == label)
                            & (random["policy"] == policy)
                        ]
                        entry = summary_block(block, policy)
                        random_summary.setdefault(direction, {}).setdefault(method, {}).setdefault(
                            scheme, {}
                        ).setdefault(policy, {})[label] = entry
                        if entry[PRIMARY_FLAG[policy]]:
                            passing.append(f"{direction}|{method}|{scheme}|{policy}|{label}")
    published = cc_read_json(pf1.CONTROL_RESULTS)
    reproduced = all(
        _plain(pf1.summarize(block) | pf1.flags(pf1.summarize(block), policy))
        == published["random"][direction][policy][label]
        for direction in DIRECTIONS
        for policy in POLICIES
        for label in ADAPTED_BUDGETS
        for block in [
            random[
                (random["direction"] == direction)
                & (random["method"] == C0)
                & (random["scheme"] == SCHEME_RANDOM)
                & (random["budget"] == label)
                & (random["policy"] == policy)
            ]
        ]
    )
    permutation: dict[str, Any] = {}
    for direction in DIRECTIONS:
        run = Run.empty()
        auroc: list[dict[str, Any]] = []
        for draw in range(LABEL_PERMUTATION_DRAWS):
            cell = cells[(direction, M1, SCHEME_RANDOM, POOL, draw)]
            shuffled = permuted_calibration(cell.calibration, direction, draw)
            run_one(cell, shuffled, cell.test, ctx, PRIMARY_METHODS, run)
            auroc.append(
                {
                    "draw": draw,
                    "real_calibration_harm_auroc": rk4.harm_auroc_of(cell.calibration),
                    "permuted_calibration_harm_auroc": rk4.harm_auroc_of(shuffled),
                }
            )
        frame = pd.DataFrame(run.rows)
        entry: dict[str, Any] = {"auroc": auroc, "methods": {}}
        for method in PRIMARY_METHODS:
            for policy in POLICIES:
                block = frame[(frame["method"] == method) & (frame["policy"] == policy)]
                entry["methods"].setdefault(method, {})[policy] = summary_block(block, policy)
        values = pd.DataFrame(auroc)
        entry["median_real_calibration_harm_auroc"] = float(
            values["real_calibration_harm_auroc"].median()
        )
        entry["median_permuted_calibration_harm_auroc"] = float(
            values["permuted_calibration_harm_auroc"].median()
        )
        entry["conservative_flags_passing"] = sorted(
            f"{method}|{policy}"
            for method in PRIMARY_METHODS
            for policy in CONSERVATIVE
            if entry["methods"][method][policy][PRIMARY_FLAG[policy]]
        )
        permutation[direction] = entry
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "random": random_summary,
            "random_passing_cells": sorted(passing),
            "random_conservative_passing_cells": sorted(
                c for c in passing if c.split("|")[3] in CONSERVATIVE
            ),
            "c0_random_reproduces_pf1_random_control": reproduced,
            "calibration_label_permutation": permutation,
            "pf1_refit_permutation": {
                d: {k: v for k, v in published["permutation"][d].items() if k.startswith("median_")}
                for d in DIRECTIONS
            },
            "seeds": {
                "random": CONTROL_SEED,
                "calibration_label_permutation": LABEL_PERMUTATION_SEED,
            },
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"controls: random passes {len(passing)} cells, C0 random reproduces PF1 {reproduced} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ summaries and the frontier


def cluster_block(
    counts: pd.DataFrame,
    ctx_clusters: Sequence[str],
    direction: str,
    method: str,
    scheme: str,
    budget: str,
    policy: str,
) -> dict[str, np.ndarray]:
    """(draws, clusters) matrices of one budget's cluster counts, draws in order."""
    d, arm, sc, b, _draw = cell_for(direction, scheme, budget, 0)
    block = counts[
        (counts["direction"] == d)
        & (counts["arm"] == arm)
        & (counts["scheme"] == sc)
        & (counts["budget"] == b)
        & (counts["method"] == method)
        & (counts["policy"] == policy)
    ]
    out: dict[str, np.ndarray] = {}
    for name in COUNT_FIELDS:
        wide = block.pivot(index="draw", columns="cluster", values=name)
        out[name] = wide.reindex(columns=list(ctx_clusters)).to_numpy(np.float64)
    return out


@dataclass(slots=True)
class Resampling:
    clusters: tuple[str, ...]
    domains: tuple[str, ...]
    errors: np.ndarray
    weights: np.ndarray


def resampling() -> Resampling:
    """The test groups, their error counts, and one set of weights shared by every comparison."""
    clusters, domains = test_clusters()
    counts = pd.read_parquet(CLUSTER_COUNTS, columns=["cluster"]).drop_duplicates()
    if sorted(counts["cluster"]) != list(clusters):
        raise PhaseError("the cluster counts do not cover exactly PF1's test groups")
    errors = pf1.error_population()
    errors = errors[errors["block"] == "test"]
    index = {c: i for i, c in enumerate(clusters)}
    per = np.zeros(len(clusters))
    for corpus, page in zip(
        errors["corpus"].astype(str), errors["document_id"].astype(str), strict=True
    ):
        per[index[group_of(corpus, page)]] += 1.0
    return Resampling(
        clusters, domains, per, bootstrap_weights(domains, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    )


def summary_intervals(block: dict[str, np.ndarray], sampling: Resampling) -> dict[str, Any]:
    """Page-clustered 95% intervals for the draw-median accepted harm and safe recall."""
    if block["accepted"].size == 0:
        return {}
    harm = weighted_harm(sampling.weights, block["accepted"], block["harmful_accepted"])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_harm = np.nanmedian(harm, axis=1)
    recall = np.median(
        weighted_metric(METRIC_SAFE_RECALL, sampling.weights, block, sampling.errors), axis=1
    )
    out: dict[str, Any] = {}
    finite = median_harm[np.isfinite(median_harm)]
    out["median_harm_ci_low"] = float(np.percentile(finite, 2.5)) if finite.size else None
    out["median_harm_ci_high"] = float(np.percentile(finite, 97.5)) if finite.size else None
    out["median_safe_automated_recall_ci_low"] = float(np.percentile(recall, 2.5))
    out["median_safe_automated_recall_ci_high"] = float(np.percentile(recall, 97.5))
    return out


def run_summaries() -> int:
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    for path in (POLICY_SUMMARY, FRONTIER):
        _forbid(path)
    rows = load_rows()
    counts = pd.read_parquet(CLUSTER_COUNTS)
    sampling = resampling()
    pools = pf1_pool_pages()
    chance = set(cc_read_json(CONTROL_RESULTS)["random_passing_cells"])
    summaries: list[dict[str, Any]] = []
    flags_of: dict[tuple[str, str, str, str, str], dict[str, bool]] = {}
    for direction in DIRECTIONS:
        for method in METHODS:
            for scheme in SCHEMES:
                for budget in scheme_budgets(scheme):
                    for policy in POLICIES:
                        block = select(rows, direction, method, scheme, budget, policy)
                        entry = summary_block(block, policy)
                        entry.update(
                            summary_intervals(
                                cluster_block(
                                    counts,
                                    sampling.clusters,
                                    direction,
                                    method,
                                    scheme,
                                    budget,
                                    policy,
                                ),
                                sampling,
                            )
                        )
                        flags_of[(direction, method, scheme, budget, policy)] = {
                            k: v for k, v in entry.items() if isinstance(v, bool)
                        }
                        summaries.append(
                            {
                                "direction": direction,
                                "method": method,
                                "scheme": scheme,
                                "budget": budget,
                                "budget_pages": budget_pages(budget, direction, pools),
                                "policy": policy,
                                "random_control_passes": (
                                    f"{direction}|{method}|{scheme}|{policy}|{budget}" in chance
                                ),
                                **entry,
                            }
                        )
    table = pd.DataFrame(summaries)
    _write_csv_once(POLICY_SUMMARY, table)

    def n_star(direction: str, method: str, scheme: str, policy: str, flag: str) -> tuple[Any, Any]:
        raw = {
            b: bool(flags_of[(direction, method, scheme, b, policy)][flag])
            for b in scheme_budgets(scheme)
        }
        masked = {
            b: v and f"{direction}|{method}|{scheme}|{policy}|{b}" not in chance
            for b, v in raw.items()
        }
        return minimal_budget(masked), minimal_budget(raw)

    masked_star: dict[str, Any] = {}
    raw_star: dict[str, Any] = {}
    for direction in DIRECTIONS:
        for method in METHODS:
            for scheme in SCHEMES:
                for policy in POLICIES:
                    for flag in flags_of[(direction, method, scheme, "0", policy)]:
                        masked, raw = n_star(direction, method, scheme, policy, flag)
                        masked_star.setdefault(direction, {}).setdefault(method, {}).setdefault(
                            scheme, {}
                        ).setdefault(policy, {})[flag] = masked
                        raw_star.setdefault(direction, {}).setdefault(method, {}).setdefault(
                            scheme, {}
                        ).setdefault(policy, {})[flag] = raw
    conservative: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    candidates = [
        (C1, SCHEME_RANDOM),
        (C1, SCHEME_STRATIFIED),
        (C3, SCHEME_RANDOM),
    ]
    for direction in DIRECTIONS:
        pooled = {
            policy: masked_star[direction][C0][SCHEME_RANDOM][policy][PRIMARY_FLAG[policy]]
            for policy in CONSERVATIVE
        }
        aware = {
            f"{method}|{scheme}|{policy}": masked_star[direction][method][scheme][policy][
                PRIMARY_FLAG[policy]
            ]
            for method, scheme in candidates
            for policy in CONSERVATIVE
        }
        pooled_frontier = earliest(list(pooled.values()))
        aware_frontier = earliest(list(aware.values()))
        gain = bool(
            aware_frontier is not None
            and (
                pooled_frontier is None
                or BUDGETS.index(aware_frontier) < BUDGETS.index(pooled_frontier)
            )
        )
        conservative[direction] = {
            "pooled": pooled,
            "pooled_frontier": pooled_frontier,
            "domain_aware": aware,
            "domain_aware_frontier": aware_frontier,
            "reached_by": sorted(k for k, v in aware.items() if v == aware_frontier and v),
            "gain": gain,
            "frontier_pages": {
                "pooled": None
                if pooled_frontier is None
                else budget_pages(pooled_frontier, direction, pools),
                "domain_aware": None
                if aware_frontier is None
                else budget_pages(aware_frontier, direction, pools),
            },
        }
        c2 = earliest(
            [
                masked_star[direction][C1][SCHEME_STRATIFIED][p][PRIMARY_FLAG[p]]
                for p in CONSERVATIVE
            ]
        )
        c1 = earliest(
            [masked_star[direction][C1][SCHEME_RANDOM][p][PRIMARY_FLAG[p]] for p in CONSERVATIVE]
        )
        efficiency[direction] = {
            "c2_frontier": c2,
            "c1_random_frontier": c1,
            "gain": bool(c2 is not None and (c1 is None or BUDGETS.index(c2) < BUDGETS.index(c1))),
        }
    published = cc_read_json(pf1.FRONTIER)["n_star"]
    reproduced = all(
        masked_star[d][C0][SCHEME_RANDOM][p][f] == published[d][M1][p][f]
        for d in DIRECTIONS
        for p in POLICIES
        for f in published[d][M1][p]
    )
    _write_json_once(
        FRONTIER,
        {
            **_analysis_envelope("frontier"),
            "n_star": masked_star,
            "n_star_before_the_random_control": raw_star,
            "conservative": conservative,
            "page_efficiency": efficiency,
            "c0_frontier_reproduces_pf1": reproduced,
            "sensitivity_methods_masked_by": "no random control of their own; unmasked values",
            "pool_pages": pools,
            "measured_budgets": list(BUDGETS),
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"summaries: {len(table)} rows; gain "
        f"{ {DIRECTION_SHORT[d]: conservative[d]['gain'] for d in DIRECTIONS} }, C0 frontier "
        f"reproduces PF1 {reproduced} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ mechanism


def risk_bin_frame(cells: dict[CellKey, Cell]) -> pd.DataFrame:
    """At the pool, each draw's decisions in score deciles of that draw's pooled decisions."""
    frames: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        for draw in range(DRAWS):
            cell = cells[(direction, M1, SCHEME_RANDOM, POOL, draw)]
            for kind, frame in ((SET_CALIBRATION, cell.calibration), (SET_TEST, cell.test.frame)):
                ordered = frame.sort_values(
                    ["safety", "candidate_id"], ascending=[False, True], kind="mergesort"
                ).reset_index(drop=True)
                decile = (np.arange(len(ordered)) * RISK_BIN_COUNT) // max(len(ordered), 1)
                grouped = (
                    ordered.assign(bin=decile)
                    .groupby(["cluster", "corpus", "bin"], sort=True)
                    .agg(decisions=("is_harmful", "size"), harmful=("is_harmful", "sum"))
                    .reset_index()
                )
                frames.append(grouped.assign(direction=direction, evaluation_set=kind, draw=draw))
    table = pd.concat(frames, ignore_index=True)
    table["harmful"] = table["harmful"].astype(np.int64)
    return table[
        ["direction", "evaluation_set", "draw", "cluster", "corpus", "bin", "decisions", "harmful"]
    ].sort_values(["direction", "evaluation_set", "draw", "cluster", "bin"], kind="stable")


def bin_matrices(
    bins: pd.DataFrame, direction: str, kind: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Clusters, their SBB mask, and (draws, clusters, bins) harmful and decision counts."""
    block = bins[(bins["direction"] == direction) & (bins["evaluation_set"] == kind)]
    domain_of = block.groupby("cluster")["corpus"].first().astype(str).to_dict()
    clusters = sorted(domain_of)
    index = {c: i for i, c in enumerate(clusters)}
    harmful = np.zeros((DRAWS, len(clusters), RISK_BIN_COUNT))
    decisions = np.zeros((DRAWS, len(clusters), RISK_BIN_COUNT))
    rows = block["cluster"].map(index).to_numpy()
    harmful[block["draw"].to_numpy(), rows, block["bin"].to_numpy()] = block["harmful"].to_numpy()
    decisions[block["draw"].to_numpy(), rows, block["bin"].to_numpy()] = block[
        "decisions"
    ].to_numpy()
    sbb = np.asarray([domain_of[c] == "ocrd_sbb" for c in clusters])
    return clusters, sbb, harmful, decisions


def risk_curve_rows(bins: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for kind in (SET_CALIBRATION, SET_TEST):
            _clusters, sbb, harmful, decisions = bin_matrices(bins, direction, kind)
            domains = ["ocrd_sbb" if s else "funsd" for s in sbb]
            weights = bootstrap_weights(domains, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
            h = harmful.sum(axis=0)
            n = decisions.sum(axis=0)
            for domain in DOMAINS:
                own = sbb if domain == "ocrd_sbb" else ~sbb
                for curve, hh, nn in (
                    ("binned", h, n),
                    ("selective", np.cumsum(h, axis=1), np.cumsum(n, axis=1)),
                ):
                    point_h = hh[own].sum(axis=0)
                    point_n = nn[own].sum(axis=0)
                    boot_h = weights[:, own] @ hh[own]
                    boot_n = weights[:, own] @ nn[own]
                    rate = np.divide(
                        boot_h, boot_n, out=np.full(boot_h.shape, np.nan), where=boot_n > 0
                    )
                    for b in range(RISK_BIN_COUNT):
                        finite = rate[:, b][np.isfinite(rate[:, b])]
                        out.append(
                            {
                                "direction": direction,
                                "evaluation_set": kind,
                                "domain": domain,
                                "curve": curve,
                                "score_decile": b + 1,
                                "pooled_coverage_at_decile_end": (b + 1) / RISK_BIN_COUNT,
                                "decisions_per_draw": float(point_n[b] / DRAWS),
                                "harm": float(point_h[b] / point_n[b]) if point_n[b] else None,
                                "ci_low": float(np.percentile(finite, 2.5))
                                if finite.size
                                else None,
                                "ci_high": (
                                    float(np.percentile(finite, 97.5)) if finite.size else None
                                ),
                                "groups": int(own.sum()),
                                "draws": DRAWS,
                                "exploratory_replay": kind == SET_TEST,
                            }
                        )
    return out


def domain_cutoffs(
    safety: np.ndarray, harmful: np.ndarray, pages: np.ndarray, sbb: np.ndarray
) -> dict[str, tuple[float | None, float | None]]:
    """C1's accept cutoffs per corpus under both rules, from arrays."""
    out: dict[str, tuple[float | None, float | None]] = {}
    for rule in (R_PLUG_IN, R_PAGE_BOUND):
        pair: list[float | None] = []
        for mask in (~sbb, sbb):
            if rule == R_PLUG_IN:
                pair.append(plug_in_cutoff(safety[mask], harmful[mask], EPSILON))
            else:
                pair.append(bound_cutoff(safety[mask], harmful[mask], pages[mask], EPSILON))
        out[rule] = (pair[0], pair[1])
    return out


def disagreement(
    safety: np.ndarray, cuts: tuple[float | None, float | None]
) -> tuple[float | None, float | None, float | None]:
    funsd = coverage_of(safety, cuts[0])
    sbb = coverage_of(safety, cuts[1])
    delta = None if funsd is None or sbb is None else sbb - funsd
    return funsd, sbb, delta


def threshold_disagreement(
    cells: dict[CellKey, Cell],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """M2: how far C1's corpus cutoffs sit apart, against corpus labels permuted over groups."""
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for d_index, direction in enumerate(DIRECTIONS):
        observed: dict[str, list[float]] = {R_PLUG_IN: [], R_PAGE_BOUND: []}
        null = {rule: np.full((DOMAIN_PERMUTATIONS, DRAWS), np.nan) for rule in observed}
        for draw in range(DRAWS):
            calibration = cells[(direction, M1, SCHEME_RANDOM, POOL, draw)].calibration
            safety = calibration["safety"].to_numpy(np.float64)
            harmful = calibration["is_harmful"].to_numpy(bool)
            pages = calibration["document_id"].astype(str).to_numpy()
            groups = calibration["cluster"].astype(str).to_numpy()
            sbb = calibration["corpus"].astype(str).to_numpy() == "ocrd_sbb"
            cuts = domain_cutoffs(safety, harmful, pages, sbb)
            for rule, pair in cuts.items():
                q_f, q_s, delta = disagreement(safety, pair)
                rows.append(
                    {
                        "direction": direction,
                        "draw": draw,
                        "rule": rule,
                        "cutoff_funsd": pair[0],
                        "cutoff_ocrd_sbb": pair[1],
                        "coverage_at_funsd_cutoff": q_f,
                        "coverage_at_sbb_cutoff": q_s,
                        "delta_coverage": delta,
                        "sbb_stricter": None if delta is None else bool(delta < 0),
                    }
                )
                if delta is not None:
                    observed[rule].append(delta)
            unique = sorted(set(groups.tolist()))
            sbb_groups = {g for g, s in zip(groups, sbb, strict=True) if s}
            for r in range(DOMAIN_PERMUTATIONS):
                generator = np.random.default_rng([DOMAIN_PERMUTATION_SEED, d_index, draw, r])
                chosen = set(
                    np.asarray(unique, dtype=object)[generator.permutation(len(unique))][
                        : len(sbb_groups)
                    ].tolist()
                )
                mask = np.asarray([g in chosen for g in groups])
                for rule, pair in domain_cutoffs(safety, harmful, pages, mask).items():
                    delta = disagreement(safety, pair)[2]
                    if delta is not None:
                        null[rule][r, draw] = delta
        summary[direction] = {}
        for rule, values in observed.items():
            statistic = float(np.mean(values)) if values else None
            with np.errstate(all="ignore"):
                counts = np.isfinite(null[rule]).sum(axis=1)
                sums = np.nansum(null[rule], axis=1)
            permuted = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)
            finite = permuted[np.isfinite(permuted)]
            p = (
                (int((np.abs(finite) >= abs(statistic)).sum()) + 1) / (finite.size + 1)
                if statistic is not None and finite.size
                else None
            )
            summary[direction][rule] = {
                "mean_delta_coverage": statistic,
                "draws_with_both_cutoffs": len(values),
                "draws_sbb_stricter": int(sum(1 for v in values if v < 0)),
                "null_mean_abs": float(np.mean(np.abs(finite))) if finite.size else None,
                "null_p97_5_abs": float(np.percentile(np.abs(finite), 97.5))
                if finite.size
                else None,
                "permutation_p_value": p,
                "permutations": DOMAIN_PERMUTATIONS,
            }
    return rows, summary


def oracle_outcome(view: TestView, mask: np.ndarray, ctx: Context) -> dict[str, Any]:
    """ANALYSIS ONLY: what a set of accepted test decisions covers and repairs, per corpus."""
    auto = _union(view, mask)
    out: dict[str, Any] = {
        "coverage": float(mask.mean()) if mask.size else 0.0,
        "automated_recall": len(auto) / max(len(ctx.keys), 1),
        "harm": _share(int(view.harmful[mask].sum()), int(mask.sum())),
    }
    for domain in DOMAINS:
        own = view.corpus == domain
        out[f"coverage_{domain}"] = _share(int((mask & own).sum()), int(own.sum()))
        out[f"automated_recall_{domain}"] = sum(
            1 for k in auto if ctx.error_domain[k] == domain
        ) / max(ctx.errors_by_domain[domain], 1)
        out[f"harm_{domain}"] = _share(int(view.harmful[mask & own].sum()), int((mask & own).sum()))
    return out


def oracle_rows(cells: dict[CellKey, Cell], ctx: Context) -> pd.DataFrame:
    """Per draw: the pooled test-oracle cutoff and the per-corpus test-oracle cutoffs."""
    rows: list[dict[str, Any]] = []
    for key, cell in cells.items():
        direction, _arm, scheme, budget, draw = key
        if scheme != SCHEME_RANDOM:
            continue
        test = cell.test.frame
        safety = test["safety"].to_numpy(np.float64)
        pooled_cut = rk1.choose_threshold(test, EPSILON) if cell.usable else None
        pooled = safety >= pooled_cut if pooled_cut is not None else np.zeros(len(test), bool)
        stratified = np.zeros(len(test), dtype=bool)
        for domain in DOMAINS:
            own = cell.test.corpus == domain
            cut = rk1.choose_threshold(test[own], EPSILON) if cell.usable and own.any() else None
            if cut is not None:
                stratified[own] = safety[own] >= cut
        for name, mask in (("oracle_pooled", pooled), ("oracle_stratified", stratified)):
            rows.append(
                {
                    "direction": direction,
                    "budget": budget,
                    "draw": draw,
                    "frontier": name,
                    **oracle_outcome(cell.test, mask, ctx),
                }
            )
    return pd.DataFrame(rows)


GAP_FRONTIERS = tuple(f"{m}|{p}" for p in (P0, P1) for m in PRIMARY_METHODS)


def gap_rows(
    oracles: pd.DataFrame, summary: pd.DataFrame, pools: dict[str, int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for budget in BUDGETS:
            base = {
                "direction": direction,
                "budget": budget,
                "budget_pages": budget_pages(budget, direction, pools),
            }
            values: dict[tuple[str, str], tuple[float | None, float | None]] = {}
            for name in ("oracle_stratified", "oracle_pooled"):
                block = oracles[
                    (oracles["direction"] == direction)
                    & (oracles["budget"] == budget)
                    & (oracles["frontier"] == name)
                ]
                for domain in (*DOMAINS, OVERALL):
                    suffix = "" if domain == OVERALL else f"_{domain}"
                    values[(name, domain)] = (
                        _median(block[f"coverage{suffix}"]),
                        _median(block[f"automated_recall{suffix}"]),
                    )
            for frontier in GAP_FRONTIERS:
                method, policy = frontier.split("|")
                entry = summary[
                    (summary["direction"] == direction)
                    & (summary["method"] == method)
                    & (summary["scheme"] == SCHEME_RANDOM)
                    & (summary["budget"] == budget)
                    & (summary["policy"] == policy)
                ].iloc[0]
                for domain in (*DOMAINS, OVERALL):
                    suffix = "" if domain == OVERALL else f"_{domain}"
                    values[(frontier, domain)] = (
                        _plain(entry[f"median_safe_coverage{suffix}"]),
                        _plain(entry[f"median_safe_automated_recall{suffix}"]),
                    )
            for (frontier, domain), (coverage, recall) in values.items():
                ceiling = values[("oracle_stratified", domain)][1]
                out.append(
                    base
                    | {
                        "domain": domain,
                        "frontier": frontier,
                        "analysis_only": frontier.startswith("oracle"),
                        "median_coverage": coverage,
                        "median_safe_automated_recall": recall,
                        "gap_to_stratified_oracle_recall": (
                            None if ceiling is None or recall is None else ceiling - recall
                        ),
                    }
                )
    return out


def evidence_rows(
    records: pd.DataFrame, rows: pd.DataFrame, draws: pd.DataFrame, pools: dict[str, int]
) -> list[dict[str, Any]]:
    """M3: per budget and corpus, the independent evidence each boundary rested on."""
    out: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for scheme in SCHEMES:
            for budget in scheme_budgets(scheme):
                d, arm, sc, b, _ = cell_for(direction, scheme, budget, 0)
                for method, domains in ((C0, (POOLED,)), (C1, DOMAINS)):
                    own = records[
                        (records["direction"] == d)
                        & (records["arm"] == arm)
                        & (records["scheme"] == sc)
                        & (records["budget"] == b)
                        & (records["method"] == method)
                        & (records["policy"] == P1)
                    ]
                    for domain in domains:
                        block = own[own["domain"] == domain]
                        deff = pd.to_numeric(block["accept_design_effect"], errors="coerce")
                        effective = block["calibration_decisions"] / deff
                        fit = draws[
                            (draws["direction"] == direction)
                            & (draws["scheme"] == scheme)
                            & (draws["budget"] == budget)
                            & (draws["role"] == "fit")
                        ]
                        if domain != POOLED:
                            fit = fit[fit["corpus"] == domain]
                        fit_pages = (
                            fit.groupby("draw")["page_id"]
                            .nunique()
                            .reindex(range(DRAWS if budget != "0" else 1), fill_value=0)
                        )
                        entry = {
                            "direction": direction,
                            "scheme": scheme,
                            "budget": budget,
                            "budget_pages": budget_pages(budget, direction, pools),
                            "method": method,
                            "domain": domain,
                            "median_fit_pages": float(fit_pages.median()) if budget != "0" else 0.0,
                            "median_calibration_pages": _median(block["calibration_pages"]),
                            "median_calibration_groups": _median(block["calibration_groups"]),
                            "median_calibration_decisions": _median(block["calibration_decisions"]),
                            "median_calibration_harmful": _median(block["calibration_harmful"]),
                            "median_design_effect": _median(deff),
                            "median_effective_decisions": _median(effective),
                            "draws_with_a_design_effect": int(deff.notna().sum()),
                            "draws": len(block),
                        }
                        for policy in (P0, P1):
                            result = select(rows, direction, method, scheme, budget, policy)
                            if domain == POOLED:
                                accepted = result["accepted"]
                                harmful = sum(result[f"harmful_accepted_{d}"] for d in DOMAINS)
                            else:
                                accepted = result[f"accepted_{domain}"]
                                harmful = result[f"harmful_accepted_{domain}"]
                            entry[f"{policy}_median_accepted"] = _median(accepted)
                            entry[f"{policy}_median_harmful_accepted"] = _median(harmful)
                        out.append(entry)
    return out


def dependence_sensitivity(counts: pd.DataFrame, sampling: Resampling) -> dict[str, Any]:
    """The family's effects without the test groups that share a volume with a fitting page."""
    structure = cc_read_json(DESIGN)["dependence"]["structure"]
    out: dict[str, Any] = {}
    for name, spec in FAMILY_SPEC.items():
        shared = set(structure[spec["direction"]]["groups_shared_with_test"])
        keep = np.asarray([c not in shared for c in sampling.clusters])
        ones = keep.astype(np.float64)[None, :]
        values = {}
        for side in ("aware", "pooled"):
            block = cluster_block(
                counts,
                sampling.clusters,
                spec["direction"],
                spec[side],
                spec["scheme"],
                spec["budget"],
                spec["policy"],
            )
            values[side] = weighted_metric(spec["metric"], ones, block, sampling.errors)[0]
        change = improvement(spec["metric"], values["aware"], values["pooled"])
        out[name] = {
            "groups_dropped": sorted(shared),
            "effect_without_them": float(change.mean()),
        }
    return out


def run_mechanism() -> int:
    started = time.monotonic()
    _require(FRONTIER, "summaries")
    outputs = (
        RISK_BINS,
        RISK_CURVES,
        THRESHOLD_DISAGREEMENT,
        EFFECTIVE_EVIDENCE,
        CALIBRATION_GAP,
        DOMAIN_SUMMARY,
        MECHANISM,
    )
    for path in outputs:
        _forbid(path)
    cells, ctx = load_cells()
    pools = pf1_pool_pages()
    bins = risk_bin_frame(cells)
    _write_parquet_once(RISK_BINS, bins.reset_index(drop=True))
    _write_csv_once(RISK_CURVES, pd.DataFrame(risk_curve_rows(bins)))
    disagreement_rows, disagreement_summary = threshold_disagreement(cells)
    _write_csv_once(THRESHOLD_DISAGREEMENT, pd.DataFrame(disagreement_rows))
    rows = load_rows()
    records = read_csv(THRESHOLD_RECORDS)
    draws = read_csv(PAGE_BUDGET_DRAWS)
    _write_csv_once(EFFECTIVE_EVIDENCE, pd.DataFrame(evidence_rows(records, rows, draws, pools)))
    summary = read_csv(POLICY_SUMMARY)
    oracles = oracle_rows(cells, ctx)
    _write_csv_once(CALIBRATION_GAP, pd.DataFrame(gap_rows(oracles, summary, pools)))
    manifest = read_csv(POPULATION_MANIFEST)
    domain_rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for domain in DOMAINS:
            own = manifest[(manifest["direction"] == direction) & (manifest["corpus"] == domain)]
            test_bins = (
                bins[
                    (bins["direction"] == direction)
                    & (bins["evaluation_set"] == SET_TEST)
                    & (bins["corpus"] == domain)
                ]
                .groupby("draw")[["decisions", "harmful"]]
                .sum()
            )
            entry: dict[str, Any] = {
                "direction": direction,
                "domain": domain,
                "label": DOMAIN_LABEL[domain],
                "test_error_sites": ctx.errors_by_domain[domain],
                "median_test_decisions_at_pool": float(test_bins["decisions"].median()),
                "median_test_harmful_share_at_pool_analysis_only": float(
                    (test_bins["harmful"] / test_bins["decisions"]).median()
                ),
            }
            for role in ROLES:
                block = own[own["split_role"] == role]
                entry[f"{role}_pages"] = int(block["page_id"].nunique())
                entry[f"{role}_groups"] = int(block["document_group"].nunique())
            for method in PRIMARY_METHODS:
                for policy in (P0, P1):
                    result = select(rows, direction, method, SCHEME_RANDOM, POOL, policy)
                    accepting = result[result[f"accepted_{domain}"] > 0]
                    entry[f"{method}_{policy}_median_harm_at_pool"] = _median(
                        accepting[f"harm_{domain}"]
                    )
                    entry[f"{method}_{policy}_median_coverage_at_pool"] = _median(
                        result[f"coverage_{domain}"]
                    )
            block = oracles[
                (oracles["direction"] == direction)
                & (oracles["budget"] == POOL)
                & (oracles["frontier"] == "oracle_stratified")
            ]
            entry["oracle_stratified_median_coverage_at_pool_analysis_only"] = _median(
                block[f"coverage_{domain}"]
            )
            domain_rows.append(entry)
    _write_csv_once(DOMAIN_SUMMARY, pd.DataFrame(domain_rows))
    counts = pd.read_parquet(CLUSTER_COUNTS)
    sampling = resampling()
    _write_json_once(
        MECHANISM,
        {
            **_analysis_envelope("mechanism"),
            "m2_threshold_disagreement": disagreement_summary,
            "dependence_sensitivity": dependence_sensitivity(counts, sampling),
            "dependence_structure": cc_read_json(DESIGN)["dependence"]["structure"],
            "oracle_is_analysis_only": True,
            "test_pages_are_exploratory_replay": True,
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "mechanism: M2 plug-in mean delta "
        f"{ {d: disagreement_summary[d][R_PLUG_IN]['mean_delta_coverage'] for d in DIRECTIONS} } "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ statistics


def paired_test(
    name: str,
    direction: str,
    metric: str,
    policy: str,
    left: tuple[str, str, str],
    right: tuple[str, str, str],
    rows: pd.DataFrame,
    counts: pd.DataFrame,
    sampling: Resampling,
) -> dict[str, Any]:
    """Left against right on the same draw indices: sign test over draws, and the percentile
    bootstrap over test groups resampled within corpus. Positive effects favour the left side."""
    frames = [select(rows, direction, m, s, b, policy) for m, s, b in (left, right)]
    if list(frames[0]["draw"]) != list(frames[1]["draw"]):
        raise PhaseError(f"{name}: the two sides are not on the same draws")
    observed = [f[metric].astype(np.float64).to_numpy() for f in frames]
    change = improvement(metric, observed[0], observed[1])
    blocks = [
        cluster_block(counts, sampling.clusters, direction, m, s, b, policy)
        for m, s, b in (left, right)
    ]
    ones = np.ones((1, len(sampling.clusters)))
    for block, values in zip(blocks, observed, strict=True):
        again = weighted_metric(metric, ones, block, sampling.errors)[0]
        if not np.array_equal(again, values):
            raise PhaseError(f"{name}: the cluster counts do not rebuild the observed metric")
    resampled = improvement(
        metric,
        weighted_metric(metric, sampling.weights, blocks[0], sampling.errors),
        weighted_metric(metric, sampling.weights, blocks[1], sampling.errors),
    ).mean(axis=1)
    p_sign, better, worse = th1.sign_test(change)
    p_boot = bootstrap_p(resampled)
    return {
        "test": name,
        "direction": direction,
        "metric": metric,
        "policy": policy,
        "left": "|".join(left),
        "right": "|".join(right),
        "draws": int(change.size),
        "effect": float(change.mean()),
        "ci_low": float(np.percentile(resampled, 2.5)),
        "ci_high": float(np.percentile(resampled, 97.5)),
        "left_mean": float(observed[0].mean()),
        "right_mean": float(observed[1].mean()),
        "p_sign": p_sign,
        "p_bootstrap": p_boot,
        "p_value": max(p_sign, p_boot),
        "draws_left_better": better,
        "draws_right_better": worse,
        "ties": int(change.size - better - worse),
    }


def domain_difference(bins: pd.DataFrame, direction: str, kind: str) -> dict[str, Any]:
    """Q1's statistic per draw, its mean, the sign test and the group bootstrap."""
    clusters, sbb, harmful, decisions = bin_matrices(bins, direction, kind)
    domains = ["ocrd_sbb" if s else "funsd" for s in sbb]
    weights = bootstrap_weights(domains, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    ones = np.ones((1, len(clusters)))
    per_draw = np.asarray(
        [score_adjusted_difference(ones, harmful[d], decisions[d], sbb)[0] for d in range(DRAWS)]
    )
    resampled = np.stack(
        [score_adjusted_difference(weights, harmful[d], decisions[d], sbb) for d in range(DRAWS)],
        axis=1,
    )
    with np.errstate(all="ignore"):
        counts = np.isfinite(resampled).sum(axis=1)
        means = np.divide(
            np.nansum(resampled, axis=1),
            counts,
            out=np.full(counts.shape, np.nan),
            where=counts > 0,
        )
    finite = per_draw[np.isfinite(per_draw)]
    p_sign, positive, negative = th1.sign_test(finite)
    p_boot = bootstrap_p(means)
    boot = means[np.isfinite(means)]
    return {
        "direction": direction,
        "evaluation_set": kind,
        "groups": len(clusters),
        "sbb_groups": int(sbb.sum()),
        "draws": int(finite.size),
        "effect": float(finite.mean()) if finite.size else None,
        "per_draw": [_plain(v) for v in per_draw],
        "ci_low": float(np.percentile(boot, 2.5)) if boot.size else None,
        "ci_high": float(np.percentile(boot, 97.5)) if boot.size else None,
        "p_sign": p_sign,
        "p_bootstrap": p_boot,
        "p_value": max(p_sign, p_boot),
        "draws_sbb_riskier": positive,
        "draws_funsd_riskier": negative,
    }


def run_stats() -> int:
    started = time.monotonic()
    _require(MECHANISM, "mechanism")
    for path in (STATISTICAL_TESTS, STATISTICAL_SUMMARY):
        _forbid(path)
    rows = load_rows()
    counts = pd.read_parquet(CLUSTER_COUNTS)
    sampling = resampling()
    family: dict[str, dict[str, Any]] = {}
    for name, spec in FAMILY_SPEC.items():
        family[name] = paired_test(
            name,
            spec["direction"],
            spec["metric"],
            spec["policy"],
            (spec["aware"], spec["scheme"], spec["budget"]),
            (spec["pooled"], spec["scheme"], spec["budget"]),
            rows,
            counts,
            sampling,
        )
    adjusted = s14.holm(family)
    for entry in adjusted.values():
        entry["favourable"] = bool(entry["survives_holm"] and entry["effect"] > 0)
    bins = pd.read_parquet(RISK_BINS)
    q1 = {
        name: domain_difference(bins, direction, SET_CALIBRATION)
        for name, direction in zip(DOMAIN_FAMILY, DIRECTIONS, strict=True)
    }
    q1_adjusted = s14.holm(q1)
    q1_replay = {
        f"Q1_test_replay_{DIRECTION_SHORT[d]}": domain_difference(bins, d, SET_TEST)
        for d in DIRECTIONS
    }
    secondary = {
        name: paired_test(
            name,
            spec["direction"],
            spec["metric"],
            spec["policy"],
            spec["left"],
            spec["right"],
            rows,
            counts,
            sampling,
        )
        for name, spec in SECONDARY.items()
    }
    table: list[dict[str, Any]] = []
    for family_name, entries in (
        ("primary", adjusted),
        ("q1_domain_difference", q1_adjusted),
        ("q1_test_replay_not_tested", q1_replay),
        ("secondary_unadjusted", secondary),
    ):
        for name, entry in entries.items():
            flat = {k: v for k, v in entry.items() if k != "per_draw"}
            table.append({"test": name, "family": family_name, **flat})
    frame = pd.DataFrame(table)
    ordered = [
        "test",
        "family",
        "direction",
        "evaluation_set",
        "metric",
        "policy",
        "left",
        "right",
        "draws",
        "effect",
        "ci_low",
        "ci_high",
        "left_mean",
        "right_mean",
        "p_sign",
        "p_bootstrap",
        "p_value",
        "holm_adjusted_p",
        "survives_holm",
        "favourable",
        "draws_left_better",
        "draws_right_better",
        "ties",
        "groups",
        "sbb_groups",
        "draws_sbb_riskier",
        "draws_funsd_riskier",
    ]
    frame = frame.reindex(columns=ordered)
    _write_csv_once(STATISTICAL_TESTS, frame)
    favourable = {
        d: any(v["favourable"] for v in adjusted.values() if v["direction"] == d)
        for d in DIRECTIONS
    }
    _write_json_once(
        STATISTICAL_SUMMARY,
        {
            **_analysis_envelope("statistical_tests"),
            "family": adjusted,
            "family_size": len(adjusted),
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "favourable": sorted(k for k, v in adjusted.items() if v["favourable"]),
            "favourable_by_direction": favourable,
            "q1_domain_difference": q1_adjusted,
            "q1_measurably_different": {
                q1_adjusted[name]["direction"]: bool(q1_adjusted[name]["survives_holm"])
                for name in DOMAIN_FAMILY
            },
            "q1_test_replay": q1_replay,
            "secondary": secondary,
            "plan": cc_read_json(DESIGN)["statistical_plan"],
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"stats: favourable {sorted(k for k, v in adjusted.items() if v['favourable'])}; Q1 "
        f"{ {q1_adjusted[n]['direction']: q1_adjusted[n]['survives_holm'] for n in DOMAIN_FAMILY} }"
        " "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ synthetic controls


def synthetic_frame(
    generator: np.random.Generator, pages: dict[str, int], shift: float, prefix: str
) -> pd.DataFrame:
    """SYNTHETIC decisions: uniform scores, logistic harm with a page effect, SBB shifted."""
    frames: list[pd.DataFrame] = []
    for key, corpus in (("a", "funsd"), ("b", "ocrd_sbb")):
        for page in range(pages[key]):
            size = int(generator.poisson(SYNTHETIC["decisions_per_page_poisson_mean"]))
            score = generator.random(size)
            effect = generator.normal(0.0, SYNTHETIC["page_effect_sd"])
            logit = (
                SYNTHETIC["intercept"]
                - SYNTHETIC["score_slope"] * score
                + effect
                + (shift if corpus == "ocrd_sbb" else 0.0)
            )
            harmful = generator.random(size) < 1.0 / (1.0 + np.exp(-logit))
            exact = ~harmful & (generator.random(size) < SYNTHETIC["exact_share_of_harmless"])
            document = f"{prefix}-{corpus}-{page:03d}"
            ids = [f"{document}|{i:03d}" for i in range(size)]
            frames.append(
                pd.DataFrame(
                    {
                        "candidate_id": ids,
                        "site_key": ids,
                        "safety": score,
                        "is_harmful": harmful,
                        "exact": exact,
                        "corpus": corpus,
                        "document_id": document,
                        "cluster": document,
                        "synthetic": True,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def synthetic_deciles(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ordered = frame.sort_values(
        ["safety", "candidate_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    decile = (np.arange(len(ordered)) * RISK_BIN_COUNT) // len(ordered)
    clusters = sorted(ordered["cluster"].unique())
    index = {c: i for i, c in enumerate(clusters)}
    rows = ordered["cluster"].map(index).to_numpy()
    harmful = np.zeros((len(clusters), RISK_BIN_COUNT))
    decisions = np.zeros((len(clusters), RISK_BIN_COUNT))
    np.add.at(harmful, (rows, decile), ordered["is_harmful"].to_numpy(np.float64))
    np.add.at(decisions, (rows, decile), 1.0)
    domain_of = ordered.groupby("cluster")["corpus"].first().to_dict()
    domains = [str(domain_of[c]) for c in clusters]
    sbb = np.asarray([d == "ocrd_sbb" for d in domains])
    return harmful, decisions, sbb, domains


def synthetic_condition(shift: float, condition: int) -> dict[str, Any]:
    """One synthetic condition over every replication: C0 against C1 under P0 and P1."""
    outcomes: dict[str, list[dict[str, Any]]] = {f"{m}|{p}": [] for m in (C0, C1) for p in (P0, P1)}
    stricter: list[bool] = []
    excludes: list[bool] = []
    above: list[bool] = []
    for replication in range(SYNTHETIC_REPLICATIONS):
        generator = np.random.default_rng([SYNTHETIC_SEED, condition, replication])
        calibration = synthetic_frame(generator, SYNTHETIC["calibration_pages"], shift, "cal")
        test = synthetic_frame(generator, SYNTHETIC["test_pages"], shift, "test")
        harmful = test["is_harmful"].to_numpy(bool)
        exact = test["exact"].to_numpy(bool)
        sbb = test["corpus"].to_numpy() == "ocrd_sbb"
        for method in (C0, C1):
            for policy in (P0, P1):
                bands = method_bands(method, policy, calibration, test, True)
                accepted = bands.accept
                harm = harmful[accepted].mean() if accepted.any() else np.nan
                violation = bool(accepted.any() and harm > EPSILON + HARM_TOLERANCE)
                own = accepted & sbb
                sbb_violation = bool(own.any() and harmful[own].mean() > EPSILON + HARM_TOLERANCE)
                recall = float((accepted & exact).sum() / max(int(exact.sum()), 1))
                outcomes[f"{method}|{policy}"].append(
                    {
                        "violation": violation,
                        "sbb_violation": sbb_violation,
                        "safe_recall": 0.0 if violation else recall,
                    }
                )
                if method == C1 and policy == P0:
                    cuts = {r["domain"]: r["accept_cutoff"] for r in bands.records}
                    if cuts["funsd"] is not None and cuts["ocrd_sbb"] is not None:
                        stricter.append(bool(cuts["ocrd_sbb"] > cuts["funsd"]))
        h, n, mask, domains = synthetic_deciles(calibration)
        weights = bootstrap_weights(
            domains, SYNTHETIC_BOOTSTRAP, int(SYNTHETIC_SEED + 1000 * condition + replication)
        )
        resampled = score_adjusted_difference(weights, h, n, mask)
        finite = resampled[np.isfinite(resampled)]
        low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
        excludes.append(bool(low > 0 or high < 0))
        above.append(bool(low > 0))
    summary: dict[str, Any] = {"shift": shift, "replications": SYNTHETIC_REPLICATIONS}
    for policy in (P0, P1):
        paired = np.asarray(
            [
                a["safe_recall"] - b["safe_recall"]
                for a, b in zip(outcomes[f"{C1}|{policy}"], outcomes[f"{C0}|{policy}"], strict=True)
            ]
        )
        summary[f"c1_minus_c0_safe_recall_{policy}"] = {
            "mean": float(paired.mean()),
            "monte_carlo_se": float(paired.std(ddof=1) / np.sqrt(paired.size)),
        }
    for key, values in outcomes.items():
        frame = pd.DataFrame(values)
        summary[key] = {
            "violation_share": float(frame["violation"].mean()),
            "sbb_violation_share": float(frame["sbb_violation"].mean()),
            "mean_safe_recall": float(frame["safe_recall"].mean()),
        }
    summary["plug_in_sbb_cutoff_stricter_share"] = float(np.mean(stricter)) if stricter else None
    summary["q1_interval_excludes_zero_share"] = float(np.mean(excludes))
    summary["q1_interval_above_zero_share"] = float(np.mean(above))
    summary["synthetic"] = True
    summary["label"] = "SYNTHETIC -- NOT A RESEARCH RESULT"
    return summary


def synthetic_difference_diagnostic(replications: int, stream: int) -> dict[str, Any]:
    """POST-HOC DIAGNOSTIC, SYNTHETIC: C1 minus C0 safe recall on identical domains over a fresh
    seed stream. Added after f09 failed on the frozen stream; it never changes f09's verdict."""
    paired: dict[str, list[float]] = {P0: [], P1: []}
    for replication in range(replications):
        generator = np.random.default_rng([SYNTHETIC_SEED, stream, replication])
        calibration = synthetic_frame(generator, SYNTHETIC["calibration_pages"], 0.0, "cal")
        test = synthetic_frame(generator, SYNTHETIC["test_pages"], 0.0, "test")
        harmful = test["is_harmful"].to_numpy(bool)
        exact = test["exact"].to_numpy(bool)
        for policy, values in paired.items():
            recall: list[float] = []
            for method in (C1, C0):
                accepted = method_bands(method, policy, calibration, test, True).accept
                violation = bool(
                    accepted.any() and harmful[accepted].mean() > EPSILON + HARM_TOLERANCE
                )
                share = float((accepted & exact).sum() / max(int(exact.sum()), 1))
                recall.append(0.0 if violation else share)
            values.append(recall[0] - recall[1])
    return {
        "post_hoc_diagnostic": True,
        "synthetic": True,
        "label": "SYNTHETIC -- NOT A RESEARCH RESULT",
        "replications": replications,
        "seed_stream": [SYNTHETIC_SEED, stream],
        **{
            f"c1_minus_c0_safe_recall_{policy}": {
                "mean": float(np.mean(values)),
                "monte_carlo_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                "replications_c1_higher": int(sum(v > 0 for v in values)),
                "replications_c0_higher": int(sum(v < 0 for v in values)),
            }
            for policy, values in paired.items()
        },
        "verdict_unchanged": "f09 keeps its pre-registered verdict",
    }


def synthetic_verdicts(identical: dict[str, Any], shifted: dict[str, Any]) -> dict[str, bool]:
    """The pre-registered pass rules for both synthetic conditions."""
    no_benefit = all(
        identical[f"{C1}|{p}"]["mean_safe_recall"]
        <= identical[f"{C0}|{p}"]["mean_safe_recall"] + 0.01
        for p in (P0, P1)
    )
    share = identical["plug_in_sbb_cutoff_stricter_share"]
    identical_pass = bool(
        no_benefit
        and identical["q1_interval_excludes_zero_share"] <= 0.10
        and share is not None
        and 0.3 <= share <= 0.7
    )
    c0 = shifted[f"{C0}|{P0}"]["sbb_violation_share"]
    c1 = shifted[f"{C1}|{P0}"]["sbb_violation_share"]
    shifted_share = shifted["plug_in_sbb_cutoff_stricter_share"]
    shifted_pass = bool(
        c0 >= 0.8
        and c1 <= c0 - 0.3
        and shifted_share is not None
        and shifted_share >= 0.9
        and shifted["q1_interval_above_zero_share"] >= 0.8
    )
    return {"identical": identical_pass, "shifted": shifted_pass}


# ------------------------------------------------------------------ falsification


def _names_in(function: Callable[..., Any]) -> set[str]:
    tree = ast.parse(inspect.getsource(function).lstrip())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _bands_equal(a: Bands, b: Bands) -> bool:
    return bool(
        np.array_equal(a.accept, b.accept)
        and np.array_equal(a.reject, b.reject)
        and _plain(a.records) == _plain(b.records)
    )


def _relabelled(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return frame.assign(corpus=frame["document_id"].astype(str).map(mapping))


def run_negative() -> int:
    """The falsification suite: each test fails if the claim it guards is false."""
    started = time.monotonic()
    _require(STATISTICAL_SUMMARY, "stats")
    _forbid(FALSIFICATION)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any, synthetic: bool = False) -> None:
        tests.append(
            {"test": name, "passed": bool(passed), "detail": _plain(detail), "synthetic": synthetic}
        )
        print(f"  {name}: {'pass' if passed else 'FAIL'}", flush=True)

    design = cc_read_json(DESIGN)
    reproduction = cc_read_json(C0_REPRODUCTION)
    calibration_registry = cc_read_json(CALIBRATION_REGISTRY)
    controls = cc_read_json(CONTROL_RESULTS)
    frontier = cc_read_json(FRONTIER)
    record(
        "f01_c0_reproduces_pf1",
        reproduction["reproduces_pf1_exactly"]
        and calibration_registry["c0_equals_pf1_inside_calibrate"]
        and controls["c0_random_reproduces_pf1_random_control"]
        and frontier["c0_frontier_reproduces_pf1"],
        {
            "rows": reproduction["pf1_rows"],
            "columns": len(reproduction["columns_compared"]),
            "random_control": controls["c0_random_reproduces_pf1_random_control"],
            "frontier": frontier["c0_frontier_reproduces_pf1"],
        },
    )
    probe_keys = [
        key for key in cal2_cells() if key[4] == 0 or (key[3] == POOL and key[2] == SCHEME_RANDOM)
    ]
    cells, ctx = load_cells(probe_keys)
    probe = cells[(D_G2Q, M1, SCHEME_RANDOM, POOL, 0)]
    pages = sorted(
        set(probe.calibration["document_id"].astype(str)) | set(probe.test.frame["document_id"])
    )
    generator = np.random.default_rng(CONTROL_SEED)
    shuffled = dict(zip(pages, generator.choice(np.asarray(DOMAINS), size=len(pages)), strict=True))
    swapped = {
        p: ("ocrd_sbb" if c == "funsd" else "funsd")
        for p, c in zip(
            probe.test.frame["document_id"].astype(str),
            probe.test.frame["corpus"].astype(str),
            strict=True,
        )
    } | {
        p: ("ocrd_sbb" if c == "funsd" else "funsd")
        for p, c in zip(
            probe.calibration["document_id"].astype(str),
            probe.calibration["corpus"].astype(str),
            strict=True,
        )
    }
    pooled_same = True
    aware_changed: set[str] = set()
    names_invariant = True
    for policy in POLICIES:
        for method in METHODS:
            real = method_bands(method, policy, probe.calibration, probe.test.frame, True)
            moved = method_bands(
                method,
                policy,
                _relabelled(probe.calibration, shuffled),
                _relabelled(probe.test.frame, shuffled),
                True,
            )
            renamed = method_bands(
                method,
                policy,
                _relabelled(probe.calibration, swapped),
                _relabelled(probe.test.frame, swapped),
                True,
            )
            if method == C0:
                pooled_same &= _bands_equal(real, moved) and _bands_equal(real, renamed)
            else:
                if not (
                    np.array_equal(real.accept, moved.accept)
                    and np.array_equal(real.reject, moved.reject)
                ):
                    aware_changed.add(method)
                names_invariant &= bool(
                    np.array_equal(real.accept, renamed.accept)
                    and np.array_equal(real.reject, renamed.reject)
                )
    record(
        "f02_corpus_labels_change_only_the_stratified_path",
        pooled_same and {C1, C3} <= aware_changed and names_invariant,
        {
            "pooled_unchanged": pooled_same,
            "stratified_methods_changed": sorted(aware_changed),
            "renaming_the_corpora_changes_nothing": names_invariant,
        },
    )
    blind_same = True
    probed = 0
    for key, cell in cells.items():
        if key[4] != 0:
            continue
        stripped = cell.test.frame.drop(
            columns=[
                c for c in ("is_harmful", "exact", "beneficial", "grade") if c in cell.test.frame
            ]
        )
        for method in METHODS:
            for policy in POLICIES:
                real = method_bands(method, policy, cell.calibration, cell.test.frame, cell.usable)
                blind = method_bands(method, policy, cell.calibration, stripped, cell.usable)
                blind_same &= _bands_equal(real, blind)
                probed += 1
    record(
        "f03_test_labels_never_reach_a_boundary",
        blind_same,
        {"band_computations_without_test_labels": probed},
    )
    draws = read_csv(PAGE_BUDGET_DRAWS)
    nested = True
    calibration_inside = True
    for (direction, scheme, draw), block in draws[draws["budget"] != "0"].groupby(
        ["direction", "scheme", "draw"]
    ):
        previous_bought: set[str] = set()
        previous_calibration: set[str] = set()
        for budget in scheme_budgets(str(scheme))[1:]:
            own = block[block["budget"] == budget]
            bought = set(own["page_id"])
            calibration_pages = set(own[own["role"] == "calibration"]["page_id"])
            nested &= previous_bought <= bought and previous_calibration <= calibration_pages
            previous_bought, previous_calibration = bought, calibration_pages
            key = (str(direction), M1, str(scheme), budget, int(draw))
            if key in cells:
                used = set(cells[key].calibration["document_id"].astype(str))
                calibration_inside &= used <= calibration_pages
    record(
        "f04_future_pages_never_enter_smaller_budgets",
        nested and calibration_inside,
        {
            "nested": nested,
            "calibration_decisions_on_purchased_calibration_pages": calibration_inside,
        },
    )
    purchased = draws[draws["budget"] != "0"]
    every_third = bool(
        (
            (purchased["role"] == "calibration")
            == (purchased["purchase_position"] % CALIBRATION_EVERY == CALIBRATION_EVERY - 1)
        ).all()
    )
    record(
        "f05_budget_nesting_matches_the_frozen_rule",
        every_third and cc_read_json(DRAW_REGISTRY)["purchases_match_pf1_registry"],
        {"every_third_position_calibrates": every_third},
    )
    structure = design["dependence"]["structure"]
    page_leaks = 0
    for key, cell in cells.items():
        tested = set(cell.test.frame["document_id"].astype(str))
        page_leaks += len(tested & set(cell.calibration["document_id"].astype(str)))
        bought = draws[
            (draws["direction"] == key[0])
            & (draws["scheme"] == key[2])
            & (draws["budget"] == key[3])
            & (draws["draw"] == key[4])
        ]
        page_leaks += len(tested & set(bought["page_id"]))
    groups_as_documented = dependence_structure() == structure
    record(
        "f06_no_test_page_fits_or_calibrates",
        page_leaks == 0
        and groups_as_documented
        and all(structure[d]["test_pages_also_fitting"] == 0 for d in DIRECTIONS)
        and "dependence_sensitivity" in cc_read_json(MECHANISM),
        {
            "page_overlaps": page_leaks,
            "shared_groups": {d: structure[d]["groups_shared_with_test"] for d in DIRECTIONS},
            "groups_as_documented": groups_as_documented,
        },
    )
    record(
        "f07_random_scores_are_never_conservatively_deployable",
        not controls["random_conservative_passing_cells"],
        {
            "conservative_passing": controls["random_conservative_passing_cells"],
            "all_passing": controls["random_passing_cells"],
        },
    )
    permutation = controls["calibration_label_permutation"]
    record(
        "f08_calibration_label_permutation_destroys_the_relationship",
        all(
            0.4 <= permutation[d]["median_permuted_calibration_harm_auroc"] <= 0.6
            and permutation[d]["median_real_calibration_harm_auroc"] >= 0.7
            and not permutation[d]["conservative_flags_passing"]
            for d in DIRECTIONS
        ),
        {
            d: {
                "permuted_auroc": permutation[d]["median_permuted_calibration_harm_auroc"],
                "real_auroc": permutation[d]["median_real_calibration_harm_auroc"],
                "conservative_passing": permutation[d]["conservative_flags_passing"],
            }
            for d in DIRECTIONS
        },
    )
    identical = synthetic_condition(SYNTHETIC["shift_identical"], 0)
    shifted = synthetic_condition(SYNTHETIC["shift_shifted"], 1)
    verdicts = synthetic_verdicts(identical, shifted)
    diagnostic = (
        synthetic_difference_diagnostic(DIAGNOSTIC_REPLICATIONS, DIAGNOSTIC_STREAM)
        if not verdicts["identical"]
        else None
    )
    record(
        "f09_identical_synthetic_domains_show_no_benefit",
        verdicts["identical"],
        {**identical, "post_hoc_diagnostic": diagnostic},
        synthetic=True,
    )
    record(
        "f10_a_shifted_synthetic_domain_is_detected", verdicts["shifted"], shifted, synthetic=True
    )
    records = read_csv(THRESHOLD_RECORDS)
    same_target = bool(
        (records["epsilon"] == EPSILON).all()
        and (records["eta"].dropna() == ETA).all()
        and (records["delta"] == DELTA).all()
    )
    recomputed = 0
    agree = True
    for key, cell in cells.items():
        if key[3] != POOL or key[2] != SCHEME_RANDOM:
            continue
        for domain in DOMAINS:
            own = cell.calibration[cell.calibration["corpus"] == domain]
            for policy in (P0, P1):
                rule = POLICY_RULES[policy][0]
                expected = dep1.accept_cutoff(rule, own, EPSILON)
                stored = records[
                    (records["direction"] == key[0])
                    & (records["scheme"] == key[2])
                    & (records["budget"] == key[3])
                    & (records["draw"] == key[4])
                    & (records["method"] == C1)
                    & (records["policy"] == policy)
                    & (records["domain"] == domain)
                ]["accept_cutoff"]
                value = stored.iloc[0]
                agree &= (expected is None and pd.isna(value)) or (
                    expected is not None and float(value) == expected
                )
                recomputed += 1
    record(
        "f11_every_domain_uses_the_same_target",
        same_target and agree,
        {"records": len(records), "recomputed_with_epsilon": recomputed, "agree": agree},
    )
    decision_names = (
        _names_in(criteria_from_artifacts) | _names_in(assign_outcome) | _names_in(run_decide)
    )
    oracle_free = not any("oracle" in n.lower() for n in decision_names) and not any(
        "oracle" in c for c in load_rows().columns
    )
    record(
        "f12_no_oracle_quantity_enters_a_criterion",
        oracle_free and "CALIBRATION_GAP" not in decision_names,
        {"decision_names_checked": len(decision_names)},
    )
    _calibration_toy, test_toy, context_toy = pf1.toy_policy_frames()
    everything = Bands(np.zeros(len(test_toy), dtype=bool), np.ones(len(test_toy), dtype=bool), [])
    toy_ctx = Context(
        keys=frozenset(context_toy[0]),
        mapping=context_toy[1],
        errors=context_toy[2].assign(corpus="funsd", document_id="p0"),
        clusters=("p0",),
        cluster_domain=("funsd",),
        error_cluster=dict.fromkeys(context_toy[0], 0),
        error_domain=dict.fromkeys(context_toy[0], "funsd"),
        errors_by_domain={"funsd": len(context_toy[0]), "ocrd_sbb": 0},
        errors_by_cluster=np.asarray([float(len(context_toy[0]))]),
    )
    toy_view = test_view(
        test_toy.assign(corpus="funsd", environment="toy", document_id="p0"), toy_ctx
    )
    toy_row, _ = evaluate(P2, toy_view, everything, toy_ctx)
    summary = read_csv(POLICY_SUMMARY)
    practical_rows = summary[
        summary["policy"].isin(list(THREE_WAY)) & summary[f"practical_at_{REVIEW_REDUCTION_FLOOR}"]
    ]
    record(
        "f13_triage_cannot_pass_by_discarding_repairs",
        toy_row["lost_repair_share"] == 1.0
        and not pf1.practical(0.0, 0.0, 1.0, toy_row["lost_repair_share"])
        and dep1.practical(0.0, 0.0, 1.0)
        and bool((practical_rows["median_lost_repair_share"] <= LOST_REPAIR_CEILING).all()),
        {
            "toy_lost_repair_share": toy_row["lost_repair_share"],
            "practical_rows": len(practical_rows),
        },
    )
    rows = load_rows()
    record(
        "f14_priors_were_frozen_before_results",
        design["issued_utc"] <= reproduction["issued_utc"]
        and design["issued_utc"] <= calibration_registry["issued_utc"]
        and design["methods"][C3]["prior_pages"] == PRIOR_PAGES[C3]
        and set(rows[rows["method"] == C3]["prior_pages"]) == {PRIOR_PAGES[C3]}
        and calibration_registry["design_sha256"] == _design_sha(),
        {
            "design_issued": design["issued_utc"],
            "first_result_issued": reproduction["issued_utc"],
            "prior_pages": PRIOR_PAGES[C3],
        },
    )
    again = Run.empty()
    for key, cell in cells.items():
        if key[4] == 0:
            run_one(cell, cell.calibration, cell.test, ctx, METHODS, again)
    fresh = _sorted(pd.DataFrame(again.rows))
    stored = rows.merge(fresh[ORDER], on=ORDER, how="inner")
    stored = _sorted(stored)
    identical_rows = len(fresh) == len(stored) and all(
        _same_values(fresh[c], stored[c]).all() for c in fresh.columns
    )
    record(
        "f15_regeneration_reproduces_the_stored_rows",
        identical_rows,
        {"rows_regenerated": len(fresh)},
    )
    lone = probe.calibration[probe.calibration["corpus"] == "funsd"]
    c3_cut, _detail = hierarchical_cutoff(R_PLUG_IN, lone, "ocrd_sbb", EPSILON, PRIOR_PAGES[C3])
    empty_domains = records[(records["method"] == C3) & (records["calibration_decisions"] == 0)]
    pooled_records = records[records["method"] == C0][[*ORDER, "accept_cutoff"]]
    matched = empty_domains[empty_domains["accept_rule"] == R_PLUG_IN].merge(
        pooled_records, on=ORDER, suffixes=("", "_pooled")
    )
    record(
        "f16_c3_without_own_evidence_is_the_pooled_plug_in",
        c3_cut == dep1.accept_cutoff(R_PLUG_IN, lone, EPSILON)
        and bool(_same_values(matched["accept_cutoff"], matched["accept_cutoff_pooled"]).all()),
        {"constructed": c3_cut, "real_records_without_own_evidence": len(matched)},
    )
    one = dict.fromkeys(pages, "funsd")
    single = all(
        np.array_equal(
            method_bands(
                C1, p, _relabelled(probe.calibration, one), _relabelled(probe.test.frame, one), True
            ).accept,
            method_bands(C0, p, probe.calibration, probe.test.frame, True).accept,
        )
        and np.array_equal(
            method_bands(
                C1, p, _relabelled(probe.calibration, one), _relabelled(probe.test.frame, one), True
            ).reject,
            method_bands(C0, p, probe.calibration, probe.test.frame, True).reject,
        )
        for p in POLICIES
    )
    record("f17_stratifying_one_corpus_is_pooling", single, {"policies": len(POLICIES)})
    disagreement_rows = read_csv(THRESHOLD_DISAGREEMENT)
    vector_agree = True
    for row in disagreement_rows.itertuples(index=False):
        policy = P0 if row.rule == R_PLUG_IN else P1
        for domain, value in (("funsd", row.cutoff_funsd), ("ocrd_sbb", row.cutoff_ocrd_sbb)):
            stored = records[
                (records["direction"] == row.direction)
                & (records["scheme"] == SCHEME_RANDOM)
                & (records["budget"] == POOL)
                & (records["draw"] == row.draw)
                & (records["method"] == C1)
                & (records["policy"] == policy)
                & (records["domain"] == domain)
            ]["accept_cutoff"].iloc[0]
            vector_agree &= bool(
                (pd.isna(stored) and pd.isna(value)) or float(stored) == float(value)
            )
    grid = [(k, n) for n in (1.0, 2.0, 7.5, 40.0, 311.25) for k in (0.0, 0.5, 1.0, 3.0, n)]
    scalar = np.asarray([clopper_pearson_upper(k, n, DELTA) for k, n in grid])  # type: ignore[arg-type]
    vector = cp_upper([k for k, _ in grid], [n for _, n in grid], DELTA)
    record(
        "f18_vectorized_rules_match_the_upstream_rules",
        vector_agree and bool(np.array_equal(scalar, vector)),
        {"cutoffs_compared": 2 * len(disagreement_rows), "bound_grid": len(grid)},
    )
    splits = cc_read_json(pf1.SPLIT_REGISTRY)
    record(
        "f19_the_budget_grid_is_pf1s",
        design["budgets"]["grid"] == splits["budgets"] == list(BUDGETS)
        and design["budgets"]["pool_pages"] == splits["pool_pages"]
        and design["budgets"]["draws"] == DRAWS,
        {"grid": splits["budgets"], "pool_pages": splits["pool_pages"]},
    )
    manifest = read_csv(POPULATION_MANIFEST)
    registry = cc_read_json(pf1.PAGE_REGISTRY)
    pool_table = pf1.load_pool()
    test_counts = {
        c: int(
            manifest[(manifest["split_role"] == "test") & (manifest["corpus"] == c)][
                "page_id"
            ].nunique()
        )
        for c in DOMAINS
    }
    expected_counts = {
        c: int(((pool_table["split"] == pf1.SPLIT_TEST) & (pool_table["corpus"] == c)).sum())
        for c in DOMAINS
    }
    pf1_pool = int(
        manifest[(manifest["split_role"] == "pool") & manifest["pf1_page"]]["page_id"].nunique()
    )
    record(
        "f20_the_manifest_matches_pf1s_registry",
        test_counts == expected_counts
        and sum(test_counts.values()) == registry["by_split"][pf1.SPLIT_TEST]
        and pf1_pool == registry["by_split"][pf1.SPLIT_POOL],
        {"test_pages": test_counts, "pf1_pool_pages": pf1_pool},
    )
    role = cc_read_json(REPO / "manifests/sgv1/role_manifest.json")
    lock = cc_read_json(REPO / "manifests/sgv1/confirmatory_reserve_lock.json")
    every_page = set(manifest["page_id"].astype(str))
    record(
        "f21_no_confirmatory_page_is_read",
        not every_page & set(role["role_of"])
        and lock["status"] == "LOCKED"
        and set(manifest["corpus"]) == set(DOMAINS),
        {"cord_pages": len(every_page & set(role["role_of"])), "lock": lock["status"]},
    )
    scores = pd.read_parquet(pf1.CELL_SCORES)
    difference = 0.0
    compared = 0
    for key, cell in cells.items():
        for kind, frame in ((SET_TEST, cell.test.frame), (SET_CALIBRATION, cell.calibration)):
            if frame.empty:
                continue
            published = scores[
                (scores["direction"] == key[0])
                & (scores["arm"] == key[1])
                & (scores["scheme"] == key[2])
                & (scores["budget"] == key[3])
                & (scores["draw"] == key[4])
                & (scores["evaluation_set"] == kind)
            ].set_index("candidate_id")["score"]
            used = frame.set_index("candidate_id")["safety"]
            joined = used.to_frame().join(published, how="left")
            if joined["score"].isna().any():
                difference = float("inf")
            else:
                difference = max(
                    difference, float((joined["safety"] - joined["score"]).abs().max())
                )
            compared += len(joined)
    record(
        "f22_the_ranker_is_pf1s_and_shared_by_every_method",
        difference == 0.0,
        {"decisions_compared": compared, "max_absolute_score_difference": difference},
    )
    order_names = _names_in(pf1.page_order) | _names_in(pf1.purchase)
    label_free = not order_names & {"is_harmful", "exact", "beneficial", "grade", "harm"}
    domain_is_corpus = bool(
        (
            probe.test.frame["corpus"]
            == probe.test.frame["document_id"].map(
                manifest.drop_duplicates("page_id").set_index("page_id")["corpus"]
            )
        ).all()
    )
    record(
        "f23_strata_and_purchases_read_no_outcome",
        label_free and domain_is_corpus,
        {"purchase_reads_labels": not label_free, "domain_is_the_page_corpus": domain_is_corpus},
    )
    passed = sum(t["passed"] for t in tests)
    _write_json_once(
        FALSIFICATION,
        {
            **_analysis_envelope("falsification_results"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "all_passed": passed == len(tests),
            "synthetic_tests": [t["test"] for t in tests if t["synthetic"]],
            "synthetic_label": "SYNTHETIC -- NOT A RESEARCH RESULT (f09 and f10 only)",
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the decision

CLAIMS = {
    "A": (
        "On development pages, domain-aware calibration gave a safe and useful conservative "
        "boundary in both generator directions where pooled calibration did not; the frozen "
        "method now needs one untouched confirmation before any deployment statement."
    ),
    "B": (
        "Domain-aware calibration reduces the ranking-to-deployment gap, but safe automation "
        "remains conditional on sufficient independent page-level evidence."
    ),
    "C": (
        "Heterogeneity matters, but available independent evidence remains insufficient: "
        "domain-aware calibration improves the frozen tests without producing a useful "
        "conservative operating point."
    ),
    "D": "The deployment gap cannot be explained by simple corpus pooling alone.",
}


def criteria_from_artifacts() -> dict[str, Any]:
    """The pre-registered criteria, read from the frontier, the family and the suite."""
    frontier = cc_read_json(FRONTIER)
    stats = cc_read_json(STATISTICAL_SUMMARY)
    falsification = cc_read_json(FALSIFICATION)
    family = stats["family"]
    gain = {d: bool(frontier["conservative"][d]["gain"]) for d in DIRECTIONS}
    favourable = {d: bool(stats["favourable_by_direction"][d]) for d in DIRECTIONS}
    outcome, strong_b = assign_outcome(gain, favourable)

    def passing(prefixes: tuple[str, ...], direction: str) -> bool:
        return any(
            family[n]["favourable"]
            for n in PRIMARY_FAMILY
            if n.startswith(prefixes) and FAMILY_SPEC[n]["direction"] == direction
        )

    return {
        "outcome": outcome,
        "strong_b": strong_b,
        "gain": gain,
        "favourable": favourable,
        "C1_safety_improvement": {d: passing(("S1",), d) for d in DIRECTIONS},
        "C2_safe_and_useful": {
            d: {
                "domain_aware_frontier": frontier["conservative"][d]["domain_aware_frontier"],
                "reached_by": frontier["conservative"][d]["reached_by"],
                "pooled_frontier": frontier["conservative"][d]["pooled_frontier"],
            }
            for d in DIRECTIONS
        },
        "C3_recall_improvement": {d: passing(("R1", "R2"), d) for d in DIRECTIONS},
        "C4_breadth": {
            "directions_with_gain": [d for d in DIRECTIONS if gain[d]],
            "directions_with_a_favourable_test": [d for d in DIRECTIONS if favourable[d]],
            "bidirectional": all(gain.values()) and all(favourable.values()),
        },
        "C5_page_efficiency": {d: bool(frontier["page_efficiency"][d]["gain"]) for d in DIRECTIONS},
        "stats_surviving": stats["surviving"],
        "stats_favourable": stats["favourable"],
        "falsification_passed": falsification["passed"],
        "falsification_total": falsification["total"],
    }


def q6_decomposition() -> dict[str, Any]:
    """ANALYSIS ONLY (Q6): at the pool, P1's safe recall against the test-oracle frontiers."""
    gap = read_csv(CALIBRATION_GAP)
    out: dict[str, Any] = {}
    for direction in DIRECTIONS:
        block = gap[
            (gap["direction"] == direction) & (gap["budget"] == POOL) & (gap["domain"] == OVERALL)
        ].set_index("frontier")

        def recall(name: str, frame: pd.DataFrame = block) -> float:
            return float(frame.loc[name, "median_safe_automated_recall"])

        stratified = recall("oracle_stratified")
        pooled = recall("oracle_pooled")
        deployed = recall(f"{C0}|{P1}")
        aware = max(recall(f"{C1}|{P1}"), recall(f"{C3}|{P1}"))
        coverage = float(block.loc["oracle_stratified", "median_coverage"])
        evidence = pooled - deployed
        heterogeneity = stratified - pooled
        if coverage < pf1.ORACLE_COVERAGE_FLOOR:
            label = "ranking"
        else:
            larger, smaller = max(evidence, heterogeneity), min(evidence, heterogeneity)
            if smaller > 0 and smaller >= 0.5 * larger:
                label = "combination"
            else:
                label = (
                    "calibration evidence" if evidence >= heterogeneity else "domain heterogeneity"
                )
        out[direction] = {
            "stratified_oracle_recall": stratified,
            "stratified_oracle_coverage": coverage,
            "pooled_oracle_recall": pooled,
            "c0_p1_safe_recall": deployed,
            "best_domain_aware_p1_safe_recall": aware,
            "evidence_component": evidence,
            "heterogeneity_component": heterogeneity,
            "recovered_by_domain_aware": aware - deployed,
            "primary": label,
        }
    return out


def decision_payload() -> dict[str, Any]:
    criteria = criteria_from_artifacts()
    outcome = criteria["outcome"]
    recommended = confirmation_recommended(outcome, criteria["strong_b"])
    all_pass = criteria["falsification_passed"] == criteria["falsification_total"]
    stats = cc_read_json(STATISTICAL_SUMMARY)
    frontier = cc_read_json(FRONTIER)
    mechanism = cc_read_json(MECHANISM)
    return {
        "outcome": outcome,
        "outcome_label": OUTCOME_TAXONOMY[outcome],
        "interpretation": INTERPRETATION[outcome],
        "strong_b": criteria["strong_b"],
        "criteria": criteria,
        "answers": {
            "Q1": {
                "question": QUESTIONS["Q1"],
                "measurably_different": stats["q1_measurably_different"],
                "calibration_pages": {
                    v["direction"]: {
                        k: v[k] for k in ("effect", "ci_low", "ci_high", "holm_adjusted_p")
                    }
                    for v in stats["q1_domain_difference"].values()
                },
                "test_replay": {
                    v["direction"]: {k: v[k] for k in ("effect", "ci_low", "ci_high")}
                    for v in stats["q1_test_replay"].values()
                },
                "threshold_disagreement": mechanism["m2_threshold_disagreement"],
            },
            "Q2": {
                "question": QUESTIONS["Q2"],
                "safety": criteria["C1_safety_improvement"],
                "recall": {
                    d: any(
                        stats["family"][n]["favourable"]
                        for n in PRIMARY_FAMILY
                        if n.startswith("R1") and FAMILY_SPEC[n]["direction"] == d
                    )
                    for d in DIRECTIONS
                },
            },
            "Q3": {"question": QUESTIONS["Q3"], "page_efficiency": frontier["page_efficiency"]},
            "Q4": {
                "question": QUESTIONS["Q4"],
                "recall": {
                    d: any(
                        stats["family"][n]["favourable"]
                        for n in PRIMARY_FAMILY
                        if n.startswith("R2") and FAMILY_SPEC[n]["direction"] == d
                    )
                    for d in DIRECTIONS
                },
                "frontier": {
                    d: {
                        m: frontier["n_star"][d][m][SCHEME_RANDOM][P1][PRIMARY_FLAG[P1]]
                        for m in (C3, *C3_SENSITIVITY)
                    }
                    for d in DIRECTIONS
                },
            },
            "Q5": {"question": QUESTIONS["Q5"], "conservative": criteria["C2_safe_and_useful"]},
            "Q6": {"question": QUESTIONS["Q6"], "decomposition": q6_decomposition()},
            "Q7": {"question": QUESTIONS["Q7"], "claim": CLAIMS[outcome]},
            "Q8": {
                "question": QUESTIONS["Q8"],
                "recommended": recommended,
                "reason": (
                    "the rubric returned A or strong B"
                    if recommended
                    else "the rubric returned neither A nor strong B"
                ),
            },
        },
        "claim": CLAIMS[outcome],
        "stopping_rule": STOPPING_RULE,
        "method_development_stopped": True,
        "recommended_next_stage": NEXT_STAGE["confirm" if recommended else "write"],
        "external_confirmation_recommended": recommended,
        "ready_for_external_confirmation": bool(recommended and all_pass),
        "external_confirmation_executed": False,
        "deployment_claim_permitted": False,
        "deployment_claim_reason": (
            "the evidence is an exploratory replay on PF1's test pages, which motivated the "
            "hypothesis; no deployment claim follows from it"
        ),
        "production_ready": False,
        "certified": False,
        "confirmatory_reserve_consumed": False,
        "falsification_all_passed": all_pass,
    }


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(FINAL_DECISION)
    payload = decision_payload()
    _write_json_once(
        FINAL_DECISION,
        {
            **_analysis_envelope("final_decision"),
            **payload,
            "design_sha256": _design_sha(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"decide: outcome {payload['outcome']} (strong B {payload['strong_b']}), next "
        f"{payload['recommended_next_stage']} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = (
    rk1.FIGURE_NOTE + "; exploratory replay on SGV-PF1's development pages, SGV-CAL2; not synthetic"
)
SURFACE = rk1.SURFACE
INK = rk1.INK
INK_SECONDARY = rk1.INK_SECONDARY
GRID = rk1.GRID
METHOD_COLOURS = {C0: "#2a78d6", C1: "#eb6834", C3: "#7d52c7"}
DOMAIN_COLOURS = {"funsd": "#2a78d6", "ocrd_sbb": "#eb6834"}
ORACLE_COLOUR = INK_SECONDARY
FIGURES = (
    "fig1_domain_risk_curves.png",
    "fig2_budget_frontier.png",
    "fig3_gap_decomposition.png",
    "fig4_annotation_allocation.png",
)
FIGURE_CAPTIONS = {
    FIGURES[0]: (
        "Realized harm by score decile of each draw's pooled decisions, FUNSD against SBB, at "
        "the full pool; top row calibration pages, bottom row the test pages (exploratory "
        "replay); bands are 95% page-clustered bootstrap intervals."
    ),
    FIGURES[1]: (
        "Median safe exact-repair recall at harm 0.1 by labelled pages for pooled (C0), "
        "corpus-stratified (C1) and hierarchical (C3) calibration; the dashed line is the "
        "test-label oracle, an analysis upper bound that is not deployable."
    ),
    FIGURES[2]: (
        "At the full pool: the per-corpus test-label oracle (analysis only) against pooled and "
        "domain-aware calibrated frontiers, safe exact-repair recall by corpus."
    ),
    FIGURES[3]: (
        "Corpus-stratified calibration with random against environment-stratified (balanced) "
        "page purchase: median safe exact-repair recall by labelled pages."
    ),
}


def run_figures() -> int:
    """Four figures and the data behind each, every value read from a persisted artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    _require(FINAL_DECISION, "decide")
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
            "svg.hashsalt": "cal2",
        }
    )
    manifest: dict[str, Any] = {}
    pools = pf1_pool_pages()

    def save(fig: Any, name: str, data: pd.DataFrame, sources: Sequence[Path]) -> None:
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
        data_name = name.replace(".png", "_data.csv")
        _write_csv_once(FIGURE_DIR / data_name, data.reset_index(drop=True))
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "data": f"{FIGURE_DIR.name}/{data_name}",
            "caption": FIGURE_CAPTIONS[name],
            "synthetic": False,
            "analysis_only": True,
            "exploratory_replay": True,
            "sources": {s.name: file_sha256(s) for s in sources},
        }

    curves = read_csv(RISK_CURVES)
    binned = curves[curves["curve"] == "binned"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharey=True)
    for row, kind in enumerate((SET_CALIBRATION, SET_TEST)):
        for col, direction in enumerate(DIRECTIONS):
            axis = axes[row][col]
            for domain in DOMAINS:
                block = binned[
                    (binned["direction"] == direction)
                    & (binned["evaluation_set"] == kind)
                    & (binned["domain"] == domain)
                ].sort_values("score_decile")
                x = block["score_decile"].to_numpy()
                axis.fill_between(
                    x,
                    block["ci_low"].astype(float),
                    block["ci_high"].astype(float),
                    color=DOMAIN_COLOURS[domain],
                    alpha=0.15,
                    linewidth=0,
                )
                axis.plot(
                    x,
                    block["harm"].astype(float),
                    color=DOMAIN_COLOURS[domain],
                    marker="o" if domain == "funsd" else "s",
                    markersize=3,
                    linewidth=1.5,
                    label=DOMAIN_LABEL[domain],
                )
            axis.axhline(EPSILON, color=INK_SECONDARY, linestyle=":", linewidth=1)
            set_name = "calibration pages" if kind == SET_CALIBRATION else "test pages (replay)"
            axis.set_title(f"{direction.replace('_', ' ')}: {set_name}", fontsize=8, color=INK)
            axis.set_xticks(range(1, RISK_BIN_COUNT + 1))
            if row == 1:
                axis.set_xlabel("score decile of the draw's pooled decisions (1 = highest)")
            if col == 0:
                axis.set_ylabel("realized harm among decisions")
    axes[0][0].legend(frameon=False, fontsize=7)
    save(fig, FIGURES[0], binned, [RISK_CURVES])

    summary = read_csv(POLICY_SUMMARY)
    gap = read_csv(CALIBRATION_GAP)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharey="row")
    frames: list[pd.DataFrame] = []
    for row, policy in enumerate((P1, P0)):
        for col, direction in enumerate(DIRECTIONS):
            axis = axes[row][col]
            positions = list(range(len(BUDGETS)))
            labels = [str(budget_pages(b, direction, pools)) for b in BUDGETS]
            oracle = (
                gap[
                    (gap["direction"] == direction)
                    & (gap["domain"] == OVERALL)
                    & (gap["frontier"] == "oracle_stratified")
                ]
                .set_index("budget")
                .reindex(list(BUDGETS))
            )
            axis.plot(
                positions,
                oracle["median_safe_automated_recall"].astype(float),
                color=ORACLE_COLOUR,
                linestyle="--",
                linewidth=1,
                label="test-label oracle per corpus (analysis upper bound, not deployable)",
            )
            for method in PRIMARY_METHODS:
                block = (
                    summary[
                        (summary["direction"] == direction)
                        & (summary["method"] == method)
                        & (summary["scheme"] == SCHEME_RANDOM)
                        & (summary["policy"] == policy)
                    ]
                    .set_index("budget")
                    .reindex(list(BUDGETS))
                )
                axis.plot(
                    positions,
                    block["median_safe_automated_recall"].astype(float),
                    color=METHOD_COLOURS[method],
                    marker={C0: "o", C1: "s", C3: "^"}[method],
                    markersize=3.5,
                    linewidth=1.5,
                    label=METHOD_SHORT[method],
                )
                frames.append(
                    block.reset_index()[
                        [
                            "direction",
                            "method",
                            "policy",
                            "budget",
                            "budget_pages",
                            "median_safe_automated_recall",
                        ]
                    ]
                )
            frames.append(
                oracle.reset_index()[
                    [
                        "direction",
                        "frontier",
                        "budget",
                        "budget_pages",
                        "median_safe_automated_recall",
                    ]
                ]
                .rename(columns={"frontier": "method"})
                .assign(policy="analysis_only")
            )
            axis.set_xticks(positions, labels)
            axis.set_title(
                f"{direction.replace('_', ' ')}: {dep1.POLICY_SHORT[policy]}", fontsize=8, color=INK
            )
            if row == 1:
                axis.set_xlabel("labelled pages")
            if col == 0:
                axis.set_ylabel("median safe exact-repair recall")
    axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, FIGURES[1], pd.concat(frames, ignore_index=True), [POLICY_SUMMARY, CALIBRATION_GAP])

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharey=True)
    pool_gap = gap[gap["budget"] == POOL]
    frontiers = ("oracle_stratified", *(f"{m}|{{policy}}" for m in PRIMARY_METHODS))
    width = 0.2
    for row, policy in enumerate((P1, P0)):
        for col, direction in enumerate(DIRECTIONS):
            axis = axes[row][col]
            block = pool_gap[pool_gap["direction"] == direction]
            for offset, template in enumerate(frontiers):
                name = template.format(policy=policy)
                values = [
                    float(
                        block[(block["frontier"] == name) & (block["domain"] == domain)][
                            "median_safe_automated_recall"
                        ].iloc[0]
                    )
                    for domain in (*DOMAINS, OVERALL)
                ]
                is_oracle = name.startswith("oracle")
                method = None if is_oracle else name.split("|")[0]
                axis.bar(
                    np.arange(3) + (offset - 1.5) * width,
                    values,
                    width=width * 0.92,
                    color="none" if is_oracle else METHOD_COLOURS[str(method)],
                    edgecolor=ORACLE_COLOUR if is_oracle else SURFACE,
                    hatch="///" if is_oracle else None,
                    linewidth=0.8,
                    label="test-label oracle (analysis only)"
                    if is_oracle
                    else METHOD_SHORT[str(method)],
                )
            axis.set_xticks(range(3), ["FUNSD", "SBB", "overall"])
            axis.set_title(
                f"{direction.replace('_', ' ')}: {dep1.POLICY_SHORT[policy]} at the full pool",
                fontsize=8,
                color=INK,
            )
            if col == 0:
                axis.set_ylabel("median safe exact-repair recall")
    axes[0][0].legend(frameon=False, fontsize=6.5)
    save(fig, FIGURES[2], pool_gap, [CALIBRATION_GAP])

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharey="row")
    frames = []
    for row, policy in enumerate((P1, P0)):
        for col, direction in enumerate(DIRECTIONS):
            axis = axes[row][col]
            for scheme, marker, style in (
                (SCHEME_RANDOM, "o", "-"),
                (SCHEME_STRATIFIED, "s", "--"),
            ):
                block = (
                    summary[
                        (summary["direction"] == direction)
                        & (summary["method"] == C1)
                        & (summary["scheme"] == scheme)
                        & (summary["policy"] == policy)
                        & summary["budget"].isin(list(STRATIFIED_BUDGETS))
                    ]
                    .set_index("budget")
                    .reindex(list(STRATIFIED_BUDGETS))
                )
                axis.plot(
                    block["budget_pages"].astype(float),
                    block["median_safe_automated_recall"].astype(float),
                    color=METHOD_COLOURS[C1],
                    linestyle=style,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.5,
                    label=f"C1, {scheme} purchase",
                )
                frames.append(
                    block.reset_index()[
                        [
                            "direction",
                            "method",
                            "scheme",
                            "policy",
                            "budget",
                            "budget_pages",
                            "median_safe_automated_recall",
                        ]
                    ]
                )
            axis.set_title(
                f"{direction.replace('_', ' ')}: {dep1.POLICY_SHORT[policy]}", fontsize=8, color=INK
            )
            if row == 1:
                axis.set_xlabel("labelled pages")
            if col == 0:
                axis.set_ylabel("median safe exact-repair recall")
    axes[0][0].legend(frameon=False, fontsize=7)
    save(fig, FIGURES[3], pd.concat(frames, ignore_index=True), [POLICY_SUMMARY])

    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_analysis_envelope("figure_manifest"),
            "figures": manifest,
            "note": FIGURE_NOTE,
            "palette": {"methods": METHOD_COLOURS, "domains": DOMAIN_COLOURS},
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(manifest)} ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism

DERIVED_PHASES = (
    "manifest",
    "reproduce",
    "calibrate",
    "controls",
    "summaries",
    "mechanism",
    "stats",
    "negative",
    "decide",
    "figures",
)
DETERMINISM_INPUTS = (UPSTREAM_STATE, DESIGN)


def derived_outputs(root: Path) -> list[Path]:
    """Every artifact the derived phases write under `root`, the figures included."""
    names = [
        POPULATION_MANIFEST,
        PAGE_BUDGET_DRAWS,
        DRAW_REGISTRY,
        C0_REPRODUCTION,
        POLICY_ROWS,
        CLUSTER_COUNTS,
        THRESHOLD_RECORDS,
        POOLED_RESULTS,
        CORPUS_STRATIFIED_RESULTS,
        HIERARCHICAL_RESULTS,
        STRATIFIED_PURCHASE_RESULTS,
        CALIBRATION_REGISTRY,
        CONTROL_RESULTS,
        POLICY_SUMMARY,
        FRONTIER,
        DOMAIN_SUMMARY,
        RISK_BINS,
        RISK_CURVES,
        THRESHOLD_DISAGREEMENT,
        EFFECTIVE_EVIDENCE,
        CALIBRATION_GAP,
        MECHANISM,
        STATISTICAL_TESTS,
        STATISTICAL_SUMMARY,
        FALSIFICATION,
        FINAL_DECISION,
        FIGURE_MANIFEST,
    ]
    out = [root / p.relative_to(OUT) for p in names]
    figures = root / FIGURE_DIR.relative_to(OUT)
    if figures.is_dir():
        out.extend(sorted(figures.iterdir()))
    return out


def signature(path: Path) -> str:
    """Byte identity for CSV and PNG; JSON without clock fields; parquet as ordered records."""
    if path.suffix == ".json":
        return str(canonical_hash(gen1._stable_view(cc_read_json(path))))
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        return str(canonical_hash(frame.to_json(orient="records", default_handler=str)))
    return file_sha256(path)


def signatures(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): signature(p) for p in derived_outputs(root) if p.is_file()
    }


def run_determinism() -> int:
    """The derived phases regenerated twice from frozen inputs, each in its own subprocess."""
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    if SANDBOXED:
        raise PhaseError("determinism runs from the published directory only")
    published = signatures(OUT)
    sandboxes = [CACHE / "determinism" / f"run_{i}" for i in range(2)]
    processes = []
    for sandbox in sandboxes:
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True)
        for source in DETERMINISM_INPUTS:
            shutil.copy2(source, sandbox / source.name)
        log = (sandbox / "run.log").open("w")
        processes.append(
            (
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--derived"],
                    cwd=REPO,
                    env={**os.environ, "SGV_CAL2_OUT": str(sandbox)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
                log,
            )
        )
    codes = []
    for process, log in processes:
        codes.append(process.wait())
        log.close()
    if any(codes):
        raise PhaseError(f"a regeneration failed: exit codes {codes}")
    runs = [signatures(s) for s in sandboxes]
    differing = sorted(n for n in published if any(r.get(n) != published[n] for r in runs))
    missing = sorted(n for n in published if any(n not in r for r in runs))
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "artifacts_compared": len(published),
            "regenerations": len(runs),
            "differing_artifacts": differing,
            "missing_in_a_regeneration": missing,
            "all_identical": not differing and not missing,
            "comparison": (
                "CSV and PNG byte for byte; JSON after removing clock fields (issued_utc, "
                "runtime_seconds) and UTC stamps; parquet as ordered records"
            ),
            "isolated_nondeterministic_metadata": ["issued_utc", "runtime_seconds"],
            "inputs_copied": [p.name for p in DETERMINISM_INPUTS],
            "seeds": {
                "page": pf1.PAGE_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "control": CONTROL_SEED,
                "calibration_label_permutation": LABEL_PERMUTATION_SEED,
                "domain_permutation": DOMAIN_PERMUTATION_SEED,
                "synthetic": SYNTHETIC_SEED,
            },
            "outcome": cc_read_json(FINAL_DECISION)["outcome"],
            "uses_ground_truth": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(published)} artifacts x {len(runs)} regenerations, identical "
        f"{not differing and not missing} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ record and traceability

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Status and Scope": (UPSTREAM_STATE, DESIGN, FINAL_DECISION),
    "2. What PF1 Left Open": (UPSTREAM_STATE, C0_REPRODUCTION),
    "3. Pre-Registered Design": (DESIGN,),
    "4. Population and Evidence Units": (DRAW_REGISTRY, POPULATION_MANIFEST, DESIGN, MECHANISM),
    "5. C0 Reproduces PF1": (C0_REPRODUCTION, CONTROL_RESULTS, FRONTIER),
    "6. Q1: Do FUNSD and SBB Differ in Score-to-Risk?": (
        STATISTICAL_SUMMARY,
        RISK_CURVES,
        MECHANISM,
        THRESHOLD_DISAGREEMENT,
    ),
    "7. Q2: Corpus-Stratified against Pooled Calibration": (
        STATISTICAL_SUMMARY,
        POLICY_SUMMARY,
        DOMAIN_SUMMARY,
    ),
    "8. Q3: Balanced Page Acquisition": (
        FRONTIER,
        POLICY_SUMMARY,
        STATISTICAL_SUMMARY,
        DRAW_REGISTRY,
    ),
    "9. Q4: Hierarchical Partial Pooling": (STATISTICAL_SUMMARY, POLICY_SUMMARY, FRONTIER, DESIGN),
    "10. Q5: Does Any Conservative Policy Become Safe and Useful?": (
        FRONTIER,
        POLICY_SUMMARY,
        DESIGN,
    ),
    "11. Q6: Where the Remaining Gap Lies": (
        FINAL_DECISION,
        CALIBRATION_GAP,
        EFFECTIVE_EVIDENCE,
        POLICY_SUMMARY,
    ),
    "12. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION, DESIGN),
    "13. Statistical Tests": (STATISTICAL_SUMMARY, STATISTICAL_TESTS, MECHANISM),
    "14. Limitations": (DESIGN, DRAW_REGISTRY, MECHANISM, DETERMINISM),
    "15. Q7: The Defensible Paper-3 Claim": (FINAL_DECISION,),
    "16. Q8: Is External Confirmation Justified?": (FINAL_DECISION,),
    "17. Research Decision and Stopping Rule": (
        FINAL_DECISION,
        FALSIFICATION,
        DETERMINISM,
        FRONTIER,
    ),
}


def _number_pool(paths: Sequence[Path]) -> set[float]:
    """gen1's pool over JSON, extended to every numeric cell of a CSV artifact."""
    import csv

    sink: set[float] = set()
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            gen1._walk_numbers(json.loads(path.read_text(encoding="utf-8")), sink)
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    for cell in row:
                        try:
                            value = float(cell)
                        except ValueError:
                            continue
                        if value == value and abs(value) != float("inf"):
                            sink.add(value)
    derived = {v * 100.0 for v in sink} | {v / 100.0 for v in sink} | {-v for v in sink}
    return sink | derived | {abs(v) for v in sink}


def audit_report(report: Path, sections: dict[str, Sequence[Path]]) -> dict[str, Any]:
    """gen1's audit: every number in a section traced to that section's artifacts."""
    counted = {"decimal": 0, "grouped": 0, "percent": 0, "integer": 0}
    untraceable: list[str] = []
    for name, body in gen1._report_sections(report.read_text(encoding="utf-8")):
        paths = sections.get(name)
        if paths is None:
            untraceable.append(f"{name}: section not indexed")
            continue
        pool = _number_pool(paths)
        text = gen1._strip_structure(body)
        for kind, pattern in (
            ("decimal", gen1._DECIMAL),
            ("grouped", gen1._GROUPED),
            ("percent", gen1._PERCENT),
            ("integer", gen1._INTEGER),
        ):
            for match in pattern.findall(text):
                literal = match.replace(",", "")
                value = float(literal)
                digits = len(literal.split(".")[1]) if "." in literal else 0
                tolerance = max(0.5 * 10.0 ** (-digits) if digits else 1e-9, 1e-12)
                counted[kind] += 1
                if not any(abs(value - candidate) <= tolerance for candidate in pool):
                    untraceable.append(f"{name}: {kind} {match}")
    return {
        "numeric_claims": sum(counted.values()),
        "by_class": counted,
        "untraceable_numeric_claims": len(untraceable),
        "untraceable": untraceable,
    }


def produced() -> list[Path]:
    return [UPSTREAM_STATE, DESIGN, *derived_outputs(OUT), DETERMINISM]


def run_record() -> int:
    started = time.monotonic()
    for path in (PROVENANCE, TRACEABILITY):
        _forbid(path)
    missing = [_relative(p) for p in produced() if not p.exists()]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {missing[:5]}")
    if not cc_read_json(DETERMINISM)["all_identical"]:
        raise PhaseError("determinism did not pass; the record will not be issued")
    state = cc_read_json(UPSTREAM_STATE)
    moved = sorted(p for p, sha in state["upstream_sha256"].items() if file_sha256(REPO / p) != sha)
    if moved:
        raise PhaseError(f"{len(moved)} upstream files moved since --reconstruct: {moved[:4]}")
    published = cc_read_json(FINAL_DECISION)
    again = decision_payload()
    rederived = all(_plain(again[k]) == published[k] for k in again)
    if not rederived:
        raise PhaseError("the decision did not re-derive identically")
    audit = audit_report(REPORT, REPORT_SECTIONS)
    if audit["untraceable_numeric_claims"]:
        raise PhaseError(
            f"{audit['untraceable_numeric_claims']} untraceable claims, "
            f"e.g. {audit['untraceable'][:3]}"
        )
    artifacts = {_relative(p): file_sha256(p) for p in produced() if p.is_file()}
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "upstream_files_rehashed": len(state["upstream_sha256"]),
            "upstream_files_moved": moved,
            "decision_rederived_identically": rederived,
            "regenerates_ocr": False,
            "generates_new_candidates": False,
            "refits_the_ranker": False,
            "calls_a_language_model": False,
            "changes_an_upstream_artifact": False,
            "issued_head": pf1._git("rev-parse", "HEAD"),
            "script": {
                "path": _relative(Path(__file__)),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
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
                "every number in the report is read from one of the artifacts its section names "
                "(JSON values or CSV cells), and from the field that means what the sentence says"
            ),
        },
    )
    print(
        f"record: {len(artifacts)} artifacts, report claims {audit['numeric_claims']}, "
        f"untraceable {audit['untraceable_numeric_claims']}"
    )
    return 0


# ------------------------------------------------------------------ entry point

PHASES: dict[str, Callable[[], int]] = {
    "reconstruct": run_reconstruct,
    "design": run_design,
    "manifest": run_manifest,
    "reproduce": run_reproduce,
    "calibrate": run_calibrate,
    "controls": run_controls,
    "summaries": run_summaries,
    "mechanism": run_mechanism,
    "stats": run_stats,
    "negative": run_negative,
    "decide": run_decide,
    "figures": run_figures,
    "determinism": run_determinism,
    "record": run_record,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--derived", action="store_true", help="every derived phase, in order")
    args = parser.parse_args(argv)
    chosen = list(DERIVED_PHASES) if args.derived else [n for n in PHASES if getattr(args, n)]
    if not chosen:
        parser.print_help()
        return 2
    if SANDBOXED and any(n not in DERIVED_PHASES for n in chosen):
        raise PhaseError("a sandboxed run may execute the derived phases only")
    for name in chosen:
        PHASES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

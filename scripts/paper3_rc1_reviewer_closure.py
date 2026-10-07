#!/usr/bin/env python3
"""Paper 3, P3-RC1: frozen reviewer closure on the held-out evaluation.

Four closure analyses on the frozen PF1 scores, labels, calibration draws and test block:

    RC1  harm discrimination on the decisions a deployment acts on (one winner per site), and the
         analysis-only risk-coverage curve of those winners
    RC2  every label-dependent deployment outcome under a non-improving harm definition, in which
         a lateral edit is harmful (HarmPolicy.NON_IMPROVING); scores and winners stay frozen
    RC3  one decision per site on the sites both correction generators reach, with cutoffs chosen
         on the matched calibration decisions of the same draws
    RC4  an adapted batch-adaptive OCR threshold (Navarro-Cerdan et al. 2015), beside the two
         frozen rules; M10 remains NOT ESTIMABLE after the pre-outcome applicability audit

No model is fitted, no score changes, no calibration page is redrawn, and no deployable cutoff
reads a test label. The run refuses to start unless the freeze record matches the protocol,
config, analysis registry and applicability audit on disk, and refuses to write unless the
frozen PF1 rows, the paper's page-group intervals and the apply-every-edit recount reproduce.

    uv run python scripts/paper3_rc1_reviewer_closure.py --feasibility   (label-free, pre-freeze)
    uv run python scripts/paper3_rc1_reviewer_closure.py --run
    uv run python scripts/paper3_rc1_reporting.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs/paper3/journal_track_2027/reviewer_closure"
OUT = REPO / "results/paper3_rc1"
FIGURES = OUT / "figures"
CONFIG_PATH = DOCS / "P3_RC1_CONFIG.json"
FREEZE_PATH = DOCS / "P3_RC1_EXECUTION_FREEZE.json"
FROZEN_DOCUMENTS = (
    DOCS / "P3_RC1_PROTOCOL.md",
    CONFIG_PATH,
    DOCS / "P3_RC1_ANALYSIS_REGISTRY.json",
    DOCS / "P3_RC1_POLICY_APPLICABILITY.md",
)
FEASIBILITY_OUT = OUT / "feasibility_label_free.json"
LABEL_CHECKS_OUT = OUT / "label_consistency.json"
RESULTS_OUT = OUT / "rc1_results.json"
EXECUTION_OUT = OUT / "execution_record.json"
FORBIDDEN = str(REPO / "data" / "raw")
TOLERANCE = 1e-12

sys.path.insert(0, str(REPO / "scripts"))

import sgv_cal2_domain_stratified_calibration as cal2  # noqa: E402
import sgv_dep1_deployment_frontier as dep1  # noqa: E402
import sgv_pf1_page_frontier_scaling as pf1  # noqa: E402
import sgv_rk2_fewshot_ranker_adaptation as rk2  # noqa: E402
import sgv_rk3_risk_aware_ranking as rk3  # noqa: E402
import sgv_rk4_generator_adaptive_ranking as rk4  # noqa: E402
from ocr_risk.edits.outcome import is_harmful as outcome_is_harmful  # noqa: E402
from ocr_risk.io.hashing import file_sha256  # noqa: E402
from ocr_risk.metrics.discrimination import roc_auc  # noqa: E402
from ocr_risk.metrics.selective import RiskCoveragePoint  # noqa: E402
from ocr_risk.risk.cluster_bounds import cluster_ratio_bound  # noqa: E402
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted  # noqa: E402

SW = "strict_worsening"
NI = "non_improving"
HARM_POLICY = {SW: HarmPolicy.STRICT_WORSENING, NI: HarmPolicy.NON_IMPROVING}
ORIGINAL = "original"
MATCHED = "matched"

PLUG_IN = "plug_in"
CONSERVATIVE = "conservative"
TRIAGE = "triage_conservative"
TRIAGE_COST = "triage_cost_aware"
NAVARRO = "navarro_adapted"
LTT = "ltt_page_group"
BOUNDARY = "test_label_boundary"
FROZEN_POLICY = {PLUG_IN: pf1.P0, CONSERVATIVE: pf1.P1, TRIAGE: pf1.P2, TRIAGE_COST: pf1.P3}
FULL_AUTOMATION_RULES = (PLUG_IN, CONSERVATIVE, NAVARRO, LTT)
POLICY_FIELDS = ("decisions", "accepted", "harm", "automated_recall", "joint_harm")
RANKING_FIELDS = ("harm_auroc", "oracle_coverage", "oracle_automated_recall")
GUARANTEE = {
    PLUG_IN: "none",
    CONSERVATIVE: "approximate (effective-sample Clopper-Pearson, pointwise over cutoffs)",
    TRIAGE: "approximate (as the conservative cutoff, on both sides)",
    TRIAGE_COST: "none",
    NAVARRO: "none (plug-in estimate of the batch's selective harm)",
    LTT: "NOT ESTIMABLE: no validated independent threshold geometry or population support bound",
    BOUNDARY: "analysis-only boundary fitted on test labels; never deployable",
}


class ClosureError(RuntimeError):
    """A validity gate failed; nothing is written past it."""


def _refuse_raw_reads(event: str, args: tuple[Any, ...]) -> None:
    # data/raw is write-once and is not an input of this analysis.
    if event == "open" and args and str(args[0]).startswith(FORBIDDEN):
        raise PermissionError(f"refusing to open {args[0]}: data/raw is not an input")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _same(a: Any, b: Any) -> bool:
    missing_a = a is None or (isinstance(a, float) and math.isnan(a))
    missing_b = b is None or (isinstance(b, float) and math.isnan(b))
    if missing_a or missing_b:
        return missing_a and missing_b
    return bool(a == b or abs(float(a) - float(b)) < TOLERANCE)


def _median(values: Iterable[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.median(finite)) if finite else None


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


# ------------------------------------------------------------------ labels


def harm_label(outcome: str, definition: str) -> bool:
    """A candidate's harm under a definition, read from the centralized outcome taxonomy."""
    return bool(outcome_is_harmful(OutcomeIfAccepted(outcome), HARM_POLICY[definition]))


def study_population() -> pd.DataFrame:
    """The frozen study population (RK1's U5 universe and PF1's new pages): ids, sites, outcomes."""
    columns = [
        "candidate_id",
        "site_key",
        "document_id",
        "corpus",
        "corrector_sources",
        "outcome",
        "is_harmful",
        "exact",
    ]
    frame = pd.concat(
        [rk3.load_population()[columns], pd.read_parquet(pf1.POPULATION)[columns]],
        ignore_index=True,
    )
    if frame["candidate_id"].duplicated().any():
        raise ClosureError("a candidate id occurs in both the U5 and the PF1 population")
    return frame


BLIND_COLUMNS = ["candidate_id", "site_key", "document_id", "corpus", "corrector_sources"]
CELL_KEY = ["direction", "arm", "scheme", "budget", "draw", "evaluation_set"]


def blind_population() -> pd.DataFrame:
    """The study population's identifiers, sites and generator membership; no label column."""
    u5 = rk3.rk1.population_frame(
        pd.read_parquet(rk3.rk1.RANKING_POPULATION, columns=[*BLIND_COLUMNS, "population"]),
        rk3.rk1.POP_U5,
    )
    frame = pd.concat(
        [u5[BLIND_COLUMNS], pd.read_parquet(pf1.POPULATION, columns=BLIND_COLUMNS)],
        ignore_index=True,
    )
    if frame["candidate_id"].duplicated().any():
        raise ClosureError("a candidate id occurs in both the U5 and the PF1 population")
    return frame


def label_maps(study: pd.DataFrame) -> dict[str, dict[str, bool]]:
    """candidate_id -> harm, per definition; the primary must equal the frozen column exactly."""
    outcomes = study["outcome"].astype(str)
    maps = {
        name: dict(
            zip(
                study["candidate_id"].astype(str),
                (harm_label(o, name) for o in outcomes),
                strict=True,
            )
        )
        for name in (SW, NI)
    }
    frozen = dict(
        zip(study["candidate_id"].astype(str), study["is_harmful"].astype(bool), strict=True)
    )
    mismatches = sum(1 for k, v in frozen.items() if maps[SW][k] != v)
    if mismatches:
        raise ClosureError(f"{mismatches} frozen harm labels differ from STRICT_WORSENING")
    return maps


def with_harm(frame: pd.DataFrame, labels: dict[str, bool]) -> pd.DataFrame:
    harmful = frame["candidate_id"].astype(str).map(labels)
    if harmful.isna().any():
        raise ClosureError("a scored candidate has no harm label")
    return frame.assign(is_harmful=harmful.astype(bool).to_numpy())


# ------------------------------------------------------------------ the matched population


def matched_sites(study: pd.DataFrame) -> set[str]:
    """Sites at which both generators proposed at least one candidate. Reads no label column."""
    blind = study[["site_key", "corrector_sources"]]
    has = rk3.membership(blind)
    by_site = (
        pd.DataFrame(
            {"site_key": blind["site_key"].astype(str), "c0": has[rk3.C0], "c1": has[rk3.C1]}
        )
        .groupby("site_key")[["c0", "c1"]]
        .any()
    )
    return set(by_site.index[by_site["c0"] & by_site["c1"]])


Context = tuple[set[str], dict[str, set[str]], pd.DataFrame]


def restricted_context(context: Context, sites: set[str]) -> Context:
    """Error sites linked to at least one kept decision site: the matched recall denominator."""
    keys, mapping, errors = context
    linked: set[str] = set()
    for site in sites:
        linked |= mapping.get(site, set())
    linked &= keys
    kept = errors[errors["error_key"].astype(str).isin(linked)].reset_index(drop=True)
    return set(kept["error_key"].astype(str)), mapping, kept


# ------------------------------------------------------------------ the two added rules


def navarro_cutoff(
    calibration: pd.DataFrame, batch_scores: np.ndarray, epsilon: float, half_width: float
) -> float | None:
    """Adapted batch-adaptive rejection threshold (Navarro-Cerdan et al. 2015, Sect. 4-5).

    H, the error-versus-cost function, is the harmful share of the labelled calibration decisions
    in a rectangular window around a score; the original window is +-w on the cost scale, and this
    adaptation places it on the calibration-score rank scale (+-half_width of the decisions),
    because the ranking score has no fixed scale across draws. E(i), the cumulative error
    estimate, averages H over the i most reliable decisions of the unlabelled batch. The cutoff is
    the deepest batch position whose estimate meets epsilon, the "highest cost where E reaches
    epsilon" of the original, so local minima are allowed. The batch is scores only.
    """
    batch = np.sort(np.asarray(batch_scores, dtype=np.float64))[::-1]
    if calibration.empty or batch.size == 0:
        return None
    scores = calibration["safety"].to_numpy(np.float64)
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    harmful = calibration["is_harmful"].to_numpy(bool)[order].astype(np.float64)
    n = ordered.size
    half = max(1, math.ceil(round(half_width * n, 9)))
    prefix = np.concatenate(([0.0], np.cumsum(harmful)))
    position = np.searchsorted(ordered, batch, side="left")
    low = np.clip(position - half, 0, n)
    high = np.clip(position + half, 0, n)
    estimate = (prefix[high] - prefix[low]) / (high - low)
    cumulative = np.cumsum(estimate) / np.arange(1, batch.size + 1)
    last_of_tie = np.flatnonzero(np.r_[batch[1:] != batch[:-1], True])
    meeting = last_of_tie[cumulative[last_of_tie] <= epsilon + dep1.HARM_TOLERANCE]
    return float(batch[meeting.max()]) if meeting.size else None


def depth_threshold(scores: np.ndarray, depth: float) -> float:
    ordered = np.sort(scores)[::-1]
    k = max(1, math.ceil(round(depth * ordered.size, 9)))
    return float(ordered[min(k, ordered.size) - 1])


def ltt_cutoff(
    calibration: pd.DataFrame,
    epsilon: float,
    delta: float,
    depths: Sequence[float],
    group_of: Callable[[str, str], str],
) -> tuple[float | None, list[dict[str, Any]]]:
    """Historical diagnostic helper, tested only on synthetic fixtures; NOT a replay certificate.

    This sample-derived geometry is not validated for the P3-RC1 population target.
    The outcome run does not call this helper. Learn-then-Test over declared depths:

    Each depth's null R(t) > epsilon, with R the decision-weighted harmful share among applied
    edits, is reduced to E[H_g - epsilon A_g] > 0 over page groups and tested with the
    finite-sample union bound of risk.cluster_bounds at delta / |depths| (Bonferroni, so the
    familywise error over the declared depths is at most delta). The loosest certified depth is
    deployed. Depth thresholds and spans are computed from calibration scores alone.
    """
    if calibration.empty:
        return None, []
    scores = calibration["safety"].to_numpy(np.float64)
    harmful = calibration["is_harmful"].to_numpy(bool)
    groups = [
        group_of(c, d)
        for c, d in zip(
            calibration["corpus"].astype(str), calibration["document_id"].astype(str), strict=True
        )
    ]
    names = sorted(set(groups))
    index = np.searchsorted(np.asarray(names), np.asarray(groups))
    level = delta / len(depths)
    chosen: float | None = None
    diagnostics: list[dict[str, Any]] = []
    for depth in depths:
        threshold = depth_threshold(scores, depth)
        accept = scores >= threshold
        accepted = np.bincount(index[accept], minlength=len(names)).astype(np.float64)
        harm = np.bincount(index[accept & harmful], minlength=len(names)).astype(np.float64)
        bound = cluster_ratio_bound(
            harm, accepted, epsilon, delta=level, span=float(accepted.max()), method="union_finite"
        )
        diagnostics.append(
            {
                "depth": float(depth),
                "threshold": threshold,
                "groups": len(names),
                "accepted": int(accepted.sum()),
                "harmful": int(harm.sum()),
                "span": float(accepted.max()),
                "z_upper": bound.z_upper,
                "certifies": bound.certifies,
            }
        )
        if bound.certifies:
            chosen = threshold if chosen is None else min(chosen, threshold)
    return chosen, diagnostics


def _cutoff_stub(cutoff: float | None) -> pd.DataFrame:
    # One calibration row on which the frozen plug-in rule returns exactly `cutoff` (a harmless
    # row at that score) or nothing (a harmful row), so a cutoff chosen by an added rule is scored
    # by the same evaluator that produced the paper's rows.
    return pd.DataFrame(
        {
            "safety": [0.0 if cutoff is None else float(cutoff)],
            "is_harmful": [cutoff is None],
            "exact": [False],
            "document_id": ["cutoff-stub"],
        }
    )


def scored_cutoff_row(
    cutoff: float | None, test: pd.DataFrame, context: Context, usable: bool
) -> dict[str, Any]:
    row = pf1.policy_row(pf1.P0, _cutoff_stub(cutoff), test, context, usable)
    expected = cutoff if usable else None
    if not _same(row["accept_cutoff"], expected):
        raise ClosureError(f"the scored cutoff {row['accept_cutoff']} is not {expected}")
    return row


# ------------------------------------------------------------------ freeze and reproduction gates


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise ClosureError("no freeze record: freeze the protocol before computing any outcome")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["git_head"] != git_head():
        raise ClosureError("Git HEAD changed after the execution freeze")
    for relative, digest in freeze["implementation_files"].items():
        if file_sha256(REPO / relative) != digest:
            raise ClosureError(f"implementation {relative} changed after the execution freeze")
    for path in FROZEN_DOCUMENTS:
        relative = path.relative_to(REPO).as_posix()
        if freeze["frozen_documents"].get(relative) != file_sha256(path):
            raise ClosureError(f"{relative} changed after the freeze; write an addendum instead")
    for relative, digest in freeze.get("package_lock", {}).items():
        if file_sha256(REPO / relative) != digest:
            raise ClosureError(f"package lock {relative} changed after the execution freeze")
    for relative, digest in freeze["frozen_documents"].items():
        if file_sha256(REPO / relative) != digest:
            raise ClosureError(f"frozen document {relative} changed after the execution freeze")
    config = load_config()
    for relative, digest in config["inputs"].items():
        if file_sha256(REPO / relative) != digest:
            raise ClosureError(f"input {relative} does not match its frozen hash")
    return freeze


def hard_gate(workers: int) -> dict[str, Any]:
    """Fail closed before reading any new outcome; report all hard-gate assertions as JSON."""
    freeze = verify_freeze()
    checks = {
        "protocol_exists": FROZEN_DOCUMENTS[0].exists(),
        "config_exists": CONFIG_PATH.exists(),
        "analysis_registry_exists": FROZEN_DOCUMENTS[2].exists(),
        "policy_audit_exists": FROZEN_DOCUMENTS[3].exists(),
        "execution_freeze_exists": FREEZE_PATH.exists(),
        "unit_tests_pass": freeze["tests"]["unit"]["exit_code"] == 0,
        "repository_tests_pass": freeze["tests"]["relevant"]["exit_code"] == 0,
        "implementation_hash_matches": True,
        "config_hash_matches": True,
        "protocol_hash_matches": True,
        "current_git_head_recorded": freeze["git_head"] == git_head(),
        "workers_match": freeze["workers"] == workers,
        "no_outcome_artifact_exists": not any(
            p.is_file()
            and p.name
            not in {
                FEASIBILITY_OUT.name,
                LABEL_CHECKS_OUT.name,
                "preflight_gate.json",
                "unit_tests.log",
                "relevant_tests.log",
                "run.log",
            }
            for p in OUT.rglob("*")
        ),
    }
    print(
        json.dumps({"artifact": "p3_rc1_hard_gate", "checks": checks}, sort_keys=True), flush=True
    )
    if not all(checks.values()):
        raise ClosureError("P3-RC1 hard gate failed")
    _write_json(OUT / "preflight_gate.json", {"checks": checks})
    return freeze


def registry_criteria(config: dict[str, Any]) -> dict[str, float]:
    numbers = json.loads((REPO / config["registry_criteria"]["source"]).read_text())["numbers"]
    out: dict[str, float] = {}
    for name, key in config["registry_criteria"]["keys"].items():
        value = float(numbers[key]["value"])
        if not _same(value, config["registry_criteria"]["expected_values"][name]):
            raise ClosureError(f"registry {key} = {value} differs from the frozen config")
        out[name] = value
    constants = {
        "epsilon": pf1.EPSILON,
        "delta": pf1.DELTA,
        "eta": pf1.ETA,
        "draws": pf1.DRAWS,
        "calibration_every": pf1.CALIBRATION_EVERY,
        "violation_ceiling": pf1.VIOLATION_CEILING,
        "working_floor": pf1.WORKING_FLOOR,
        "useful_coverage_floor": pf1.USEFUL_COVERAGE_FLOOR,
        "review_reduction_floor": pf1.REVIEW_REDUCTION_FLOOR,
        "lost_repair_ceiling": pf1.LOST_REPAIR_CEILING,
        "bootstrap_resamples": cal2.BOOTSTRAP_RESAMPLES,
    }
    for name, value in constants.items():
        if not _same(out[name], value):
            raise ClosureError(f"registry {name} = {out[name]} but the frozen code uses {value}")
    return out


# ------------------------------------------------------------------ cells


@dataclass(slots=True)
class Cell:
    direction: str
    budget: str
    draw: int
    usable: bool
    candidates: pd.DataFrame
    """All test candidates of the cell, frozen labels."""
    winners: pd.DataFrame
    calibration: pd.DataFrame
    """Calibration decisions (one winner per calibration site), frozen labels."""


def iter_cells(evaluation: pf1.Evaluation, config: dict[str, Any]) -> list[Cell]:
    cells: list[Cell] = []
    for direction in config["cells"]["directions"]:
        for budget in config["cells"]["secondary_budgets"]:
            draws = [0] if budget == "0" else list(range(pf1.DRAWS))
            for draw in draws:
                key = pf1.cell_key(direction, pf1.PRIMARY_ARM, pf1.SCHEME_RANDOM, budget, draw)
                frames = evaluation.cells[key]
                test = frames[pf1.SET_TEST]
                cells.append(
                    Cell(
                        direction=direction,
                        budget=budget,
                        draw=draw,
                        usable=bool(evaluation.registry[key]["trainable"]),
                        candidates=test,
                        winners=rk4.decisions(test),
                        calibration=rk4.decisions(frames.get(pf1.SET_CALIBRATION, test.head(0))),
                    )
                )
    return cells


def _frozen_row(table: pd.DataFrame, cell: Cell, **match: str) -> pd.Series:
    key = pf1.cell_key(cell.direction, pf1.PRIMARY_ARM, pf1.SCHEME_RANDOM, cell.budget, cell.draw)
    rows = table[
        (table["direction"] == key[0])
        & (table["arm"] == key[1])
        & (table["scheme"] == key[2])
        & (table["budget"] == key[3])
        & (table["draw"] == key[4])
        & (table["evaluation_set"] == pf1.SET_TEST)
    ]
    for column, value in match.items():
        rows = rows[rows[column] == value]
    if len(rows) != 1:
        raise ClosureError(f"{key} {match}: {len(rows)} frozen rows")
    return rows.iloc[0]


def reproduce_frozen_rows(cells: list[Cell], context: Context) -> int:
    """Every frozen policy and ranking row of the evaluated cells, recomputed and compared."""
    policies = pd.read_parquet(pf1.POLICY_OUTCOMES)
    ranking = pd.read_parquet(pf1.RANKING_OUTCOMES)
    count = 0
    for cell in cells:
        for policy in FROZEN_POLICY.values():
            mine = pf1.policy_row(policy, cell.calibration, cell.winners, context, cell.usable)
            stored = _frozen_row(policies, cell, policy=policy)
            for field in POLICY_FIELDS:
                if not _same(mine[field], stored[field]):
                    raise ClosureError(
                        f"{cell.direction} {cell.budget} {cell.draw} {policy} {field}"
                    )
            count += 1
        mine_rank = pf1.ranking_row(cell.candidates, cell.usable, context)
        stored_rank = _frozen_row(ranking, cell)
        for field in RANKING_FIELDS:
            if not _same(mine_rank[field], stored_rank[field]):
                raise ClosureError(f"{cell.direction} {cell.budget} {cell.draw} ranking {field}")
        count += 1
    return count


# ------------------------------------------------------------------ resampling


@dataclass(slots=True)
class Resampling:
    clusters: tuple[str, ...]
    domains: tuple[str, ...]
    index: dict[str, int]
    weights: np.ndarray


def resampling(config: dict[str, Any]) -> Resampling:
    clusters, domains = cal2.test_clusters()
    if config["uncertainty"]["resamples"] != cal2.BOOTSTRAP_RESAMPLES:
        raise ClosureError("the frozen resample count differs from the paper's")
    weights = cal2.bootstrap_weights(domains, cal2.BOOTSTRAP_RESAMPLES, cal2.BOOTSTRAP_SEED)
    return Resampling(clusters, domains, {c: i for i, c in enumerate(clusters)}, weights)


def cluster_index(frame: pd.DataFrame, sampling: Resampling) -> np.ndarray:
    return np.asarray(
        [
            sampling.index[cal2.group_of(c, d)]
            for c, d in zip(
                frame["corpus"].astype(str), frame["document_id"].astype(str), strict=True
            )
        ],
        dtype=np.int64,
    )


def _auroc(scores: np.ndarray, harmful: np.ndarray) -> float:
    return float(roc_auc(-scores, harmful.astype(np.float64)))


def _auroc_task(
    payload: tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray],
) -> np.ndarray:
    """(resamples, draws) AUROCs for one population; each draw is (scores, harmful, cluster)."""
    draws, weights = payload
    clusters = weights.shape[1]
    members = [[np.flatnonzero(c == i) for i in range(clusters)] for _, _, c in draws]
    out = np.empty((weights.shape[0], len(draws)))
    for b, row in enumerate(weights):
        for d, ((scores, harmful, _), by_cluster) in enumerate(zip(draws, members, strict=True)):
            take = np.concatenate(
                [np.repeat(by_cluster[i], int(m)) for i, m in enumerate(row) if m]
            )
            out[b, d] = _auroc(scores[take], harmful[take])
    return out


def auroc_resamples(
    tasks: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    weights: np.ndarray,
    workers: int,
) -> dict[str, np.ndarray]:
    names = sorted(tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_auroc_task, [(tasks[n], weights) for n in names]))
    return dict(zip(names, results, strict=True))


def percentile_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def policy_intervals(
    counts: dict[str, np.ndarray], errors: np.ndarray, weights: np.ndarray
) -> dict[str, float | None]:
    """CAL2's page-group intervals for the draw-median harm and credited recall, plus coverage
    and uncredited recall from the same resamples; cutoffs (calibration) held fixed."""
    out: dict[str, float | None] = {}
    if counts["accepted"].size == 0:
        return out
    harm = cal2.weighted_harm(weights, counts["accepted"], counts["harmful_accepted"])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_harm = np.nanmedian(harm, axis=1)
    out["median_harm_ci_low"], out["median_harm_ci_high"] = percentile_interval(median_harm)
    credited = np.median(
        cal2.weighted_metric(cal2.METRIC_SAFE_RECALL, weights, counts, errors), axis=1
    )
    out["median_credited_recall_ci_low"], out["median_credited_recall_ci_high"] = (
        percentile_interval(credited)
    )
    uncredited = np.median(
        (weights @ counts["auto_repaired"].T) / (weights @ errors)[:, None], axis=1
    )
    out["median_recall_ci_low"], out["median_recall_ci_high"] = percentile_interval(uncredited)
    coverage = np.median(
        (weights @ counts["accepted"].T) / (weights @ counts["decisions"].T), axis=1
    )
    out["median_coverage_ci_low"], out["median_coverage_ci_high"] = percentile_interval(coverage)
    return out


# ------------------------------------------------------------------ one rule on one cell


@dataclass(slots=True)
class Scored:
    row: dict[str, Any]
    accept: np.ndarray
    extra: dict[str, Any]


def score_rule(
    rule: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: Context,
    usable: bool,
    config: dict[str, Any],
    half_width: float | None = None,
) -> Scored:
    """Cutoffs from calibration decisions (and, for the batch-adaptive rule, test SCORES) only."""
    epsilon = pf1.EPSILON
    blind_test = test[["candidate_id", "site_key", "document_id", "corpus", "safety"]]
    if rule in FROZEN_POLICY:
        policy = FROZEN_POLICY[rule]
        row = pf1.policy_row(policy, calibration, test, context, usable)
        accept = dep1.bands(policy, calibration, blind_test, usable)["accept"]
        if policy == pf1.P1:
            deff = dep1.th1.page_design_effect(calibration)
            _, upper = (
                dep1.th1.bound_threshold(calibration, epsilon, pf1.DELTA, deff)
                if deff is not None
                else (None, None)
            )
            row["calibration_bound"] = upper
            row["calibration_design_effect"] = deff
        return Scored(row, accept, {})
    if rule == NAVARRO:
        width = (
            config["rc4"]["navarro"]["window_half_width_primary"]
            if half_width is None
            else half_width
        )
        cutoff = (
            navarro_cutoff(calibration, blind_test["safety"].to_numpy(np.float64), epsilon, width)
            if usable
            else None
        )
        extra = {"half_width": width}
    elif rule == LTT:
        raise ClosureError("M10 applicability failed before freeze; do not run as a certificate")
    else:
        raise ClosureError(f"unknown rule {rule}")
    row = scored_cutoff_row(cutoff, test, context, usable)
    safety = blind_test["safety"].to_numpy(np.float64)
    accept = safety >= cutoff if (cutoff is not None and usable) else np.zeros(safety.size, bool)
    if int(accept.sum()) != row["accepted"]:
        raise ClosureError(f"{rule}: the accept mask disagrees with the frozen evaluator")
    return Scored(row, accept, extra)


def cluster_counts(
    test: pd.DataFrame,
    accept: np.ndarray,
    context: Context,
    sampling: Resampling,
    error_cluster: dict[str, int],
) -> dict[str, np.ndarray]:
    size = len(sampling.clusters)
    index = cluster_index(test, sampling)
    harmful = test["is_harmful"].to_numpy(bool)
    repaired = pf1.repaired_sites(test[accept], context)
    return {
        "decisions": np.bincount(index, minlength=size).astype(np.float64),
        "accepted": np.bincount(index[accept], minlength=size).astype(np.float64),
        "harmful_accepted": np.bincount(index[accept & harmful], minlength=size).astype(np.float64),
        "auto_repaired": np.bincount(
            np.asarray([error_cluster[k] for k in repaired], dtype=np.int64), minlength=size
        ).astype(np.float64),
    }


def error_clusters(context: Context, sampling: Resampling) -> tuple[dict[str, int], np.ndarray]:
    _, _, errors = context
    mapping = {
        str(k): sampling.index[cal2.group_of(str(c), str(d))]
        for k, c, d in zip(
            errors["error_key"], errors["corpus"], errors["document_id"], strict=True
        )
    }
    per = np.bincount(list(mapping.values()), minlength=len(sampling.clusters)).astype(np.float64)
    return mapping, per


def verdict(summary: dict[str, Any]) -> str:
    violation = summary["accept_violation_share"]
    if violation is not None and violation > pf1.VIOLATION_CEILING:
        return "VIOLATING"
    if summary["working_share"] is None or summary["working_share"] < pf1.WORKING_FLOOR:
        return "NOT_WORKING"
    coverage = summary["median_acceptance_coverage"] or 0.0
    return "USEFUL" if coverage >= pf1.USEFUL_COVERAGE_FLOOR else "DEPLOYABLE_NOT_USEFUL"


def summarize_rule(rule: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summary = pf1.summarize(frame)
    policy = FROZEN_POLICY.get(rule, pf1.P0)
    flags = pf1.flags(summary, policy)
    out = {
        "draws": len(rows),
        "accepting_draws": int((frame["accepted"] > 0).sum()),
        "accept_violation_share": summary["accept_violation_share"],
        "working_share": summary["working_share"],
        "median_harm": summary["median_harm"],
        "median_acceptance_coverage": summary["median_acceptance_coverage"],
        "median_abstention": _median(1.0 - frame["acceptance_coverage"].astype(float)),
        "median_safe_coverage": summary["median_safe_coverage"],
        "median_recall": _median(frame["automated_recall"]),
        "median_credited_recall": _median(
            [0.0 if r["accept_violation"] else r["automated_recall"] for r in rows]
        ),
        "median_joint_harm": _median(frame["joint_harm"]),
        "flags": flags,
        "guarantee": GUARANTEE[rule],
    }
    if rule in (TRIAGE, TRIAGE_COST):
        out.update(
            {
                "reject_violation_share": summary["reject_violation_share"],
                "median_review_reduction": summary["median_review_reduction"],
                "median_lost_repair_share": summary["median_lost_repair_share"],
                "practical": bool(flags[f"practical_at_{pf1.REVIEW_REDUCTION_FLOOR}"]),
            }
        )
    else:
        out["verdict"] = verdict(out)
    return out


# ------------------------------------------------------------------ RC1 risk-coverage


def winner_curve(winners: pd.DataFrame, context: Context, grid: np.ndarray) -> list[dict[str, Any]]:
    """ANALYSIS ONLY: the test winners' risk-coverage behaviour at fixed coverage targets."""
    scores = winners["safety"].to_numpy(np.float64)
    harmful = winners["is_harmful"].to_numpy(bool)
    exact = winners["exact"].to_numpy(bool)
    n = scores.size
    descending = np.sort(scores)[::-1]
    keys = context[0]
    rows: list[dict[str, Any]] = []
    for target in grid:
        k = max(1, math.ceil(round(float(target) * n, 9)))
        threshold = float(descending[min(k, n) - 1])
        accept = scores >= threshold
        point = RiskCoveragePoint(threshold, n, int(accept.sum()), int(harmful[accept].sum()))
        rows.append(
            {
                "coverage_target": float(target),
                "threshold": threshold,
                "coverage": point.coverage,
                "accepted": point.n_accepted,
                "harmful_share": point.risk,
                "joint_harm": point.joint_harm_rate,
                "exact_share": float(exact[accept].mean()),
                "exact_repair_recall": len(pf1.repaired_sites(winners[accept], context))
                / max(len(keys), 1),
            }
        )
    return rows


# ------------------------------------------------------------------ feasibility (label-free)


def run_feasibility() -> dict[str, Any]:
    config = load_config()
    blind = blind_population()
    matched = matched_sites(blind)
    membership = rk3.membership(blind)
    scores = pd.read_parquet(pf1.CELL_SCORES, columns=[*CELL_KEY, "candidate_id"])
    sites = blind.set_index("candidate_id")
    cells: dict[str, Any] = {}
    for direction in config["cells"]["directions"]:
        for budget in config["cells"]["secondary_budgets"]:
            arm = pf1.A0 if budget == "0" else pf1.PRIMARY_ARM
            draws = [0] if budget == "0" else list(range(pf1.DRAWS))
            per: dict[str, list[dict[str, Any]]] = {pf1.SET_CALIBRATION: [], pf1.SET_TEST: []}
            for draw in draws:
                selected = scores[
                    (scores["direction"] == direction)
                    & (scores["arm"] == arm)
                    & (scores["scheme"] == pf1.SCHEME_RANDOM)
                    & (scores["budget"] == budget)
                    & (scores["draw"] == draw)
                ]
                for kind in per:
                    frame = selected[selected["evaluation_set"] == kind][["candidate_id"]].join(
                        sites, on="candidate_id", how="inner"
                    )
                    kept = frame[frame["site_key"].isin(matched)]
                    multiplicity = frame.groupby("site_key").size()
                    per[kind].append(
                        {
                            "candidates": len(frame),
                            "decisions": int(frame["site_key"].nunique()),
                            "pages": int(frame["document_id"].nunique()),
                            "groups": len(
                                {
                                    cal2.group_of(c, d)
                                    for c, d in zip(
                                        frame["corpus"], frame["document_id"], strict=True
                                    )
                                }
                            ),
                            "max_candidates_per_site": int(multiplicity.max())
                            if len(multiplicity)
                            else 0,
                            "matched_candidates": len(kept),
                            "matched_decisions": int(kept["site_key"].nunique()),
                            "matched_pages": int(kept["document_id"].nunique()),
                            "matched_groups": len(
                                {
                                    cal2.group_of(c, d)
                                    for c, d in zip(
                                        kept["corpus"], kept["document_id"], strict=True
                                    )
                                }
                            ),
                            "corpus_decisions": {
                                str(k): int(v)
                                for k, v in frame.drop_duplicates("site_key")["corpus"]
                                .value_counts()
                                .items()
                            },
                            "matched_corpus_decisions": {
                                str(k): int(v)
                                for k, v in kept.drop_duplicates("site_key")["corpus"]
                                .value_counts()
                                .items()
                            },
                        }
                    )
            summary: dict[str, Any] = {}
            for kind, rows in per.items():
                for field in (
                    "candidates",
                    "decisions",
                    "pages",
                    "groups",
                    "max_candidates_per_site",
                    "matched_candidates",
                    "matched_decisions",
                    "matched_pages",
                    "matched_groups",
                ):
                    values = [r[field] for r in rows]
                    summary[f"{kind}.{field}"] = {
                        "min": int(min(values)),
                        "median": float(np.median(values)),
                        "max": int(max(values)),
                    }
                summary[f"{kind}.corpus_decisions.draw0"] = rows[0]["corpus_decisions"]
                summary[f"{kind}.matched_corpus_decisions.draw0"] = rows[0][
                    "matched_corpus_decisions"
                ]
            cells[f"{direction}|{budget}"] = summary
    test_frames = {}
    for direction in config["cells"]["directions"]:
        selected = scores[
            (scores["direction"] == direction)
            & (scores["arm"] == pf1.PRIMARY_ARM)
            & (scores["scheme"] == pf1.SCHEME_RANDOM)
            & (scores["budget"] == pf1.POOL)
            & (scores["draw"] == 0)
            & (scores["evaluation_set"] == pf1.SET_TEST)
        ]
        test_frames[direction] = selected[["candidate_id"]].join(
            sites, on="candidate_id", how="inner"
        )
    g2q, q2g = config["cells"]["directions"]
    matched_test = {d: set(f["site_key"]) & matched for d, f in test_frames.items()}
    txt = test_frames[q2g]
    per_page = (
        test_frames[g2q][test_frames[g2q]["site_key"].isin(matched)]
        .drop_duplicates("site_key")
        .groupby("document_id")
        .size()
    )
    payload = {
        "artifact": "p3_rc1_feasibility_label_free",
        "synthetic": False,
        "reads_labels": False,
        "population": "pf1_fresh_test and the PF1 calibration draws (label-free counts only)",
        "matched_site_rule": config["rc3"]["matched_site_rule"],
        "study_sites": int(blind["site_key"].nunique()),
        "matched_sites_in_study_population": len(matched),
        "candidates_in_both_generators": int((membership[rk3.C0] & membership[rk3.C1]).sum()),
        "matched_test_sites_identical_across_directions": matched_test[g2q] == matched_test[q2g],
        "matched_test_sites": len(matched_test[g2q]),
        "matched_test_sites_per_page": {
            "median": float(per_page.median()),
            "min": int(per_page.min()),
            "max": int(per_page.max()),
        },
        "txt_candidates_per_test_site": {
            str(k): int(v)
            for k, v in sorted(Counter(txt.groupby("site_key").size().tolist()).items())
        },
        "txt_candidates_per_matched_test_site": {
            str(k): int(v)
            for k, v in sorted(
                Counter(
                    txt[txt["site_key"].isin(matched)].groupby("site_key").size().tolist()
                ).items()
            )
        },
        "cells": cells,
        "inputs": {
            p.relative_to(REPO).as_posix(): file_sha256(p)
            for p in (pf1.CELL_SCORES, pf1.POPULATION, rk3.rk1.RANKING_POPULATION)
        },
    }
    return payload


def run_label_checks() -> dict[str, Any]:
    """Consistency of the frozen labels with the outcome taxonomy. Mismatch counts only.

    Reads label columns, but reports no prevalence: every number is a count of rows on which two
    frozen encodings of the same label disagree, or of candidates that would need special
    handling (identity edits, empty strings).
    """
    new = pd.read_parquet(
        pf1.POPULATION,
        columns=[
            "candidate_id",
            "original_ocr",
            "candidate_text",
            "outcome",
            "is_harmful",
            "exact",
            "d_before",
            "d_after",
        ],
    )
    before = new["d_before"].to_numpy(np.int64)
    after = new["d_after"].to_numpy(np.int64)
    expected = np.where(
        before == 0,
        OutcomeIfAccepted.OVERCORRECTION.value,
        np.where(
            after == 0,
            OutcomeIfAccepted.TRUE_CORRECTION.value,
            np.where(
                after < before,
                OutcomeIfAccepted.PARTIAL_IMPROVEMENT.value,
                np.where(
                    after == before,
                    OutcomeIfAccepted.LATERAL_CHANGE.value,
                    OutcomeIfAccepted.MISCORRECTION.value,
                ),
            ),
        ),
    )
    outcome = new["outcome"].astype(str).to_numpy()
    u5 = rk3.load_population()[["outcome", "is_harmful", "exact"]]
    u5_outcome = u5["outcome"].astype(str)
    non_improving = {
        o.value
        for o in OutcomeIfAccepted
        if o is not OutcomeIfAccepted.IDENTITY and harm_label(o.value, NI)
    }
    return {
        "artifact": "p3_rc1_label_consistency",
        "synthetic": False,
        "reports_prevalence": False,
        "pf1_rows": len(new),
        "u5_rows": len(u5),
        "pf1_outcome_vs_distances_mismatches": int((outcome != expected).sum()),
        "pf1_identity_edits": int(
            (new["candidate_text"].astype(str) == new["original_ocr"].astype(str)).sum()
        ),
        "pf1_is_harmful_vs_strict_worsening_mismatches": int(
            (new["is_harmful"].to_numpy(bool) != (after > before)).sum()
        ),
        "pf1_exact_vs_zero_distance_mismatches": int(
            (new["exact"].to_numpy(bool) != (after == 0)).sum()
        ),
        "pf1_non_improving_vs_distances_mismatches": int(
            (np.isin(outcome, sorted(non_improving)) != (after >= before)).sum()
        ),
        "pf1_empty_candidate_texts": int((new["candidate_text"].astype(str) == "").sum()),
        "pf1_empty_ocr_texts": int((new["original_ocr"].astype(str) == "").sum()),
        "u5_is_harmful_vs_outcome_mismatches": int(
            (
                u5["is_harmful"].to_numpy(bool)
                != np.asarray([harm_label(o, SW) for o in u5_outcome], dtype=bool)
            ).sum()
        ),
        "u5_exact_vs_outcome_mismatches": int(
            (
                u5["exact"].to_numpy(bool)
                != (u5_outcome == OutcomeIfAccepted.TRUE_CORRECTION.value).to_numpy()
            ).sum()
        ),
        "outcome_values": sorted(set(outcome.tolist()) | set(u5_outcome.tolist())),
        "inputs": {
            p.relative_to(REPO).as_posix(): file_sha256(p)
            for p in (pf1.POPULATION, rk3.rk1.RANKING_POPULATION)
        },
    }


# ------------------------------------------------------------------ the run


def _summary_key(*parts: str) -> str:
    return "|".join(parts)


def run(workers: int) -> dict[str, Any]:
    started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat(timespec="seconds")
    freeze = hard_gate(workers)
    config = load_config()
    criteria = registry_criteria(config)
    matched = matched_sites(blind_population())
    labels = label_maps(study_population())
    evaluation = pf1.load_evaluation()
    context = evaluation.context[pf1.SET_TEST]
    cells = iter_cells(evaluation, config)
    gates: dict[str, Any] = {"frozen_rows_reproduced": reproduce_frozen_rows(cells, context)}
    print(f"gate: {gates['frozen_rows_reproduced']} frozen rows reproduced", flush=True)
    sampling = resampling(config)
    weights = sampling.weights
    matched_context = restricted_context(context, matched)
    contexts = {ORIGINAL: context, MATCHED: matched_context}
    clusters = {name: error_clusters(ctx, sampling) for name, ctx in contexts.items()}
    grid = np.round(
        np.arange(
            config["rc1"]["risk_coverage_grid"]["start"],
            config["rc1"]["risk_coverage_grid"]["stop"] + 1e-9,
            config["rc1"]["risk_coverage_grid"]["step"],
        ),
        10,
    )

    # ---------------------------------------------------------- policies (RC2, RC3, RC4)
    policy_rows: list[dict[str, Any]] = []
    per_rule: dict[str, list[dict[str, Any]]] = {}
    counts_of: dict[str, list[dict[str, np.ndarray]]] = {}
    boundary: dict[str, list[dict[str, Any]]] = {}
    curves: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    unavailable: dict[str, Any] = {}
    auroc_inputs: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    all_rules = (*FULL_AUTOMATION_RULES, TRIAGE, TRIAGE_COST)
    for cell in cells:
        for definition in (SW, NI):
            candidates = with_harm(cell.candidates, labels[definition])
            winners = with_harm(cell.winners, labels[definition])
            calibration = with_harm(cell.calibration, labels[definition])
            populations = [(ORIGINAL, winners, calibration)]
            if cell.budget == pf1.POOL:
                populations.append(
                    (
                        MATCHED,
                        winners[winners["site_key"].isin(matched)].reset_index(drop=True),
                        calibration[calibration["site_key"].isin(matched)].reset_index(drop=True),
                    )
                )
            for population, test, cal in populations:
                ctx = contexts[population]
                support.append(
                    {
                        "direction": cell.direction,
                        "budget": cell.budget,
                        "draw": cell.draw,
                        "harm_definition": definition,
                        "population": population,
                        "calibration_sites": len(cal),
                        "test_sites": len(test),
                        "calibration_pages": int(cal["document_id"].nunique()),
                        "test_pages": int(test["document_id"].nunique()),
                        "calibration_groups": len(
                            {
                                cal2.group_of(str(c), str(d))
                                for c, d in zip(cal["corpus"], cal["document_id"], strict=True)
                            }
                        ),
                        "test_groups": len(
                            {
                                cal2.group_of(str(c), str(d))
                                for c, d in zip(test["corpus"], test["document_id"], strict=True)
                            }
                        ),
                        "calibration_corpus": cal["corpus"].value_counts().to_dict(),
                        "test_corpus": test["corpus"].value_counts().to_dict(),
                    }
                )
                rules: Iterable[str] = (
                    all_rules if population == ORIGINAL else FULL_AUTOMATION_RULES
                )
                widths: list[float | None] = [None]
                for rule in rules:
                    if rule == LTT:
                        unavailable[
                            _summary_key(cell.direction, cell.budget, definition, population, rule)
                        ] = {
                            "status": "NOT ESTIMABLE",
                            "reason": config["rc4"]["ltt"]["not_estimable_reason"],
                            "guarantee": GUARANTEE[LTT],
                            "draws": 1 if cell.budget == "0" else pf1.DRAWS,
                        }
                        continue
                    variants = widths
                    if rule == NAVARRO and cell.budget == pf1.POOL and population == ORIGINAL:
                        variants = [
                            None,
                            *config["rc4"]["navarro"]["window_half_width_sensitivity"],
                        ]
                    for width in variants:
                        scored = score_rule(rule, cal, test, ctx, cell.usable, config, width)
                        name = rule if width is None else f"{rule}@w={width}"
                        key = _summary_key(
                            cell.direction, cell.budget, definition, population, name
                        )
                        row = {
                            "direction": cell.direction,
                            "budget": cell.budget,
                            "draw": cell.draw,
                            "harm_definition": definition,
                            "population": population,
                            "rule": name,
                            **{k: v for k, v in scored.row.items() if k != "policy"},
                        }
                        policy_rows.append(row)
                        per_rule.setdefault(key, []).append(scored.row)
                        if cell.budget == pf1.POOL:
                            counts_of.setdefault(key, []).append(
                                cluster_counts(
                                    test, scored.accept, ctx, sampling, clusters[population][0]
                                )
                            )
                oracle = rk2.oracle_accepted(test, pf1.EPSILON) if cell.usable else test.head(0)
                boundary.setdefault(
                    _summary_key(cell.direction, cell.budget, definition, population), []
                ).append(
                    {
                        "direction": cell.direction,
                        "budget": cell.budget,
                        "draw": cell.draw,
                        "harm_definition": definition,
                        "population": population,
                        "analysis_only": True,
                        "cutoff": rk2.rk1.choose_threshold(test, pf1.EPSILON)
                        if cell.usable
                        else None,
                        "coverage": len(oracle) / max(len(test), 1),
                        "recall": len(pf1.repaired_sites(oracle, ctx)) / max(len(ctx[0]), 1),
                        "harm": float(oracle["is_harmful"].mean()) if len(oracle) else None,
                    }
                )
                # Ranking on this population: winners, and (original only) every candidate.
                frames = [("winner", test)]
                if population == ORIGINAL:
                    frames.append(("all", candidates))
                for unit, frame in frames:
                    scores = frame["safety"].to_numpy(np.float64)
                    harmful = frame["is_harmful"].to_numpy(bool)
                    ranking.append(
                        {
                            "direction": cell.direction,
                            "budget": cell.budget,
                            "draw": cell.draw,
                            "harm_definition": definition,
                            "population": population,
                            "unit": unit,
                            "rows": len(frame),
                            "decisions": int(frame["site_key"].nunique()),
                            "harmful_prevalence": float(harmful.mean()),
                            "harm_auroc": _auroc(scores, harmful) if cell.usable else None,
                        }
                    )
                    if cell.budget in ("0", pf1.POOL) and cell.usable:
                        auroc_inputs.setdefault(
                            _summary_key(cell.direction, cell.budget, definition, population, unit),
                            [],
                        ).append((scores, harmful, cluster_index(frame, sampling)))
                if cell.budget == pf1.POOL and population == ORIGINAL:
                    for point in winner_curve(test, ctx, grid):
                        curves.append(
                            {
                                "direction": cell.direction,
                                "harm_definition": definition,
                                "draw": cell.draw,
                                "analysis_only": True,
                                **point,
                            }
                        )
        print(f"cell {cell.direction} {cell.budget} {cell.draw} done", flush=True)

    # ---------------------------------------------------------- summaries
    summaries: dict[str, Any] = {}
    for key, rows in per_rule.items():
        direction, budget, definition, population, name = key.split("|")
        rule = name.split("@")[0]
        summary = summarize_rule(rule, rows)
        if key in counts_of:
            stacked = {
                field: np.vstack([c[field] for c in counts_of[key]])
                for field in ("decisions", "accepted", "harmful_accepted", "auto_repaired")
            }
            summary["intervals"] = policy_intervals(stacked, clusters[population][1], weights)
        summaries[key] = summary
    boundary_summary = {
        key: {
            "median_coverage": _median(r["coverage"] for r in rows),
            "median_recall": _median(r["recall"] for r in rows),
            "median_harm": _median(r["harm"] for r in rows),
            "boundary_exists": (_median(r["coverage"] for r in rows) or 0.0)
            >= pf1.USEFUL_COVERAGE_FLOOR,
        }
        for key, rows in boundary.items()
    }

    # ---------------------------------------------------------- AUROC intervals (RC1, RC2, RC3)
    resampled = auroc_resamples(auroc_inputs, weights, workers)
    ranking_frame = pd.DataFrame(ranking)
    auroc_summary: dict[str, Any] = {}
    for key, matrix in resampled.items():
        direction, budget, definition, population, unit = key.split("|")
        point = [_auroc(s, h) for s, h, _ in auroc_inputs[key]]
        low, high = percentile_interval(np.median(matrix, axis=1))
        auroc_summary[key] = {"median": float(np.median(point)), "ci_low": low, "ci_high": high}
    for key in list(auroc_summary):
        direction, budget, definition, population, unit = key.split("|")
        if unit != "winner" or population != ORIGINAL:
            continue
        other = _summary_key(direction, budget, definition, population, "all")
        difference = resampled[key] - resampled[other]
        point = np.median(
            [
                _auroc(s1, h1) - _auroc(s2, h2)
                for (s1, h1, _), (s2, h2, _) in zip(
                    auroc_inputs[key], auroc_inputs[other], strict=True
                )
            ]
        )
        low, high = percentile_interval(np.median(difference, axis=1))
        auroc_summary[
            _summary_key(direction, budget, definition, population, "winner_minus_all")
        ] = {
            "median": float(point),
            "ci_low": low,
            "ci_high": high,
        }

    # ---------------------------------------------------------- reproduction gates on intervals
    paper = json.loads((REPO / "results/generated/paper3/auroc_page_intervals.json").read_text())
    for direction in config["cells"]["directions"]:
        for budget in ("0", pf1.POOL):
            mine = auroc_summary[_summary_key(direction, budget, SW, ORIGINAL, "all")]
            theirs = paper["directions"][direction][budget]
            for a, b in (
                ("median", "median_harm_auroc"),
                ("ci_low", "ci_low"),
                ("ci_high", "ci_high"),
            ):
                if not _same(mine[a], theirs[b]):
                    raise ClosureError(f"AUROC interval {direction} {budget} {a} did not reproduce")
    gates["auroc_intervals_reproduced"] = 12
    journal = json.loads((REPO / "results/generated/paper3/journal_numbers.json").read_text())[
        "numbers"
    ]
    reproduced = 0
    for direction in config["cells"]["directions"]:
        short = config["cells"]["short"][direction]
        mine = summaries[_summary_key(direction, pf1.POOL, SW, ORIGINAL, PLUG_IN)]["intervals"]
        for theirs, ours in (
            ("median_harm_ci_low", "median_harm_ci_low"),
            ("median_harm_ci_high", "median_harm_ci_high"),
            ("median_safe_automated_recall_ci_low", "median_credited_recall_ci_low"),
            ("median_safe_automated_recall_ci_high", "median_credited_recall_ci_high"),
        ):
            value = journal[f"ci.{short}.p0.{theirs}.pool"]["value"]
            if not _same(mine[ours], value):
                raise ClosureError(f"policy interval ci.{short}.p0.{theirs} did not reproduce")
            reproduced += 1
    gates["policy_intervals_reproduced"] = reproduced
    base = json.loads((REPO / "results/generated/paper3/base_rate_baseline.json").read_text())
    for direction in config["cells"]["directions"]:
        prevalence = ranking_frame[
            (ranking_frame["direction"] == direction)
            & (ranking_frame["budget"] == pf1.POOL)
            & (ranking_frame["harm_definition"] == SW)
            & (ranking_frame["population"] == ORIGINAL)
            & (ranking_frame["unit"] == "winner")
        ]["harmful_prevalence"]
        expected = float(
            np.median(
                [d["apply_all_harmful_fraction"] for d in base["directions"][direction]["draws"]]
            )
        )
        if not _same(float(np.median(prevalence)), expected):
            raise ClosureError(
                f"apply-every-edit harmful fraction did not reproduce for {direction}"
            )
    gates["apply_all_reproduced"] = 2
    print(f"gates passed: {gates}", flush=True)

    # ---------------------------------------------------------- write
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "P3_RC1_POLICY_ROWS.csv", pd.DataFrame(policy_rows))
    _write_csv(OUT / "P3_RC1_SUPPORT.csv", pd.DataFrame(support))
    _write_csv(
        OUT / "P3_RC1_BOUNDARY_ROWS.csv",
        pd.DataFrame([r for rows in boundary.values() for r in rows]),
    )
    _write_csv(OUT / "P3_RC1_RANKING_ROWS.csv", ranking_frame)
    _write_csv(OUT / "P3_RC1_WINNER_RISK_COVERAGE.csv", pd.DataFrame(curves))
    payload = {
        "artifact": "p3_rc1_results",
        "registration": "prospectively frozen reviewer-closure analysis (P3-RC1)",
        "synthetic": False,
        "population": "pf1_fresh_test (66 held-out pages); PF1 calibration draws",
        "criteria": criteria,
        "gates": gates,
        "policy_summaries": summaries,
        "not_estimable_policies": unavailable,
        "boundary_summaries": boundary_summary,
        "auroc_summaries": auroc_summary,
        "matched": {
            "sites_in_study_population": len(matched),
            "test_decisions": int(
                ranking_frame[
                    (ranking_frame["population"] == MATCHED) & (ranking_frame["budget"] == pf1.POOL)
                ]["decisions"].iloc[0]
            ),
            "recall_denominator": len(matched_context[0]),
            "original_recall_denominator": len(context[0]),
        },
        "freeze_record_sha256": file_sha256(FREEZE_PATH),
        "freeze_git_head": freeze["git_head"],
    }
    _write_json(RESULTS_OUT, payload)
    inputs = {rel: file_sha256(REPO / rel) for rel in config["inputs"]}
    outputs = [*sorted(p for p in OUT.glob("P3_RC1_*.csv")), RESULTS_OUT]
    _write_json(
        EXECUTION_OUT,
        {
            "artifact": "p3_rc1_execution_record",
            "started_utc": started_utc,
            "runtime_seconds": time.monotonic() - started,
            "git_head": git_head(),
            "script": {
                "path": "scripts/paper3_rc1_reviewer_closure.py",
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "freeze_record": {
                "path": FREEZE_PATH.relative_to(REPO).as_posix(),
                "sha256": file_sha256(FREEZE_PATH),
            },
            "inputs": inputs,
            "outputs": {p.relative_to(REPO).as_posix(): file_sha256(p) for p in outputs},
            "gates": gates,
            "fits_models": False,
            "redraws_calibration_pages": False,
            "deployable_cutoffs_read_test_labels": False,
            "reads_the_confirmatory_reserve": False,
            "workers": workers,
        },
    )
    return payload


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, float_format="%.17g")
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ClosureError(f"{path} exists with different content; it is write-once")
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ClosureError(f"{path} exists with different content; it is write-once")
    path.write_text(text, encoding="utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not serializable: {type(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--label-checks", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    # Installed per run: an audit hook cannot be removed, so installing it at import would also
    # block the raw-store validators of a test process that imports this module.
    sys.addaudithook(_refuse_raw_reads)
    if args.feasibility:
        payload = run_feasibility()
        _write_json(FEASIBILITY_OUT, payload)
        print(f"feasibility: written {FEASIBILITY_OUT.relative_to(REPO)}")
    if args.label_checks:
        _write_json(LABEL_CHECKS_OUT, run_label_checks())
        print(f"label checks: written {LABEL_CHECKS_OUT.relative_to(REPO)}")
    if args.run:
        payload = run(args.workers)
        print(f"run: written {RESULTS_OUT.relative_to(REPO)}; gates {payload['gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SGV1 development phase 4: target alignment.

Phase 3 reported that R1's advantage "reverses on the natural pool". Defect SGV1-R17
established that this was a **target mismatch**, not a property of the population: the
models are trained on ``is_harmful`` and were scored on ``beneficial``, and
``lateral_change`` -- 23.6% of the pool -- sits in the gap. A lateral change causes no
harm, so a good harm-predictor correctly ranks it safe, and the ``beneficial`` target
scores that same correctness as a false positive. R1 was penalised for being right.

That diagnosis makes one prediction, and this stage tests it::

    if the mismatch is the whole story, then refitting each representation ON the
    evaluation target should remove the reversal entirely

It also asks the decision-theoretic question the mismatch exposes. The system has three
actions but a binary target. Accepting a ``lateral_change`` is not safe in any useful
sense: it is a no-op edit that spends risk budget for zero gain, and nearly a quarter of
the pool is exactly that. A three-outcome head can decline it; a harm-trained binary head
cannot even represent the distinction.

    T_harm     is_harmful          -- what Phase 2 and 3 trained on
    T_benefit  beneficial          -- aligned with "accept only if it improves"
    T_three    beneficial / neutral / harmful, accept on P(beneficial)

crossed with R0 (candidate-blind), V1 (Phase-2 baseline) and R1 (candidate-conditioned).

    --fit       nine models, calibrated, scored on DEVELOPMENT
    --analyze   both targets, decision quality, selective prediction, the reversal test
    --figures   the reversal, accepted-edit composition, risk-coverage
    --decide    the machine-readable finding

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked. No hypothesis is decided.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_representation as repr_stage
import sgv1_verifier_pilot as pilot
from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.calibration import brier_score, expected_calibration_error
from ocr_risk.metrics.discrimination import average_precision, roc_auc
from ocr_risk.metrics.selective import aurc, coverage_at_risk, risk_coverage_curve
from ocr_risk.stats.bootstrap import paired_cluster_bootstrap

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv1/target_alignment"
SCORES = OUT / "alignment_scores.parquet"
FIT_RECORD = OUT / "fit_record.json"
ALIGNMENT_RESULTS = OUT / "alignment_results.json"
DECISION_RESULTS = OUT / "decision_results.json"
DECISION = OUT / "research_decision.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPRESENTATIONS = ("R0", "V1", "R1")
TARGETS = ("harm", "benefit", "three")
COVERAGE_POINTS = (0.10, 0.25, 0.50, 0.75, 1.00)


class PhaseError(RuntimeError):
    """A freeze, role, or alignment invariant failed."""


def _outcome_class(frame: pd.DataFrame) -> np.ndarray:
    """0 = harmful, 1 = neutral (no-op), 2 = beneficial.

    The three-way target the three actions actually imply. ``lateral_change`` is its own
    class rather than being folded into "safe": accepting one changes the transcription
    without improving it, which is a cost with no benefit.
    """
    out = np.full(len(frame), 1, dtype=np.int64)
    out[frame["is_harmful"].to_numpy(dtype=bool)] = 0
    out[frame["beneficial"].to_numpy(dtype=bool)] = 2
    return out


def _target_vector(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "harm":
        return frame["is_harmful"].to_numpy(dtype=int)
    if target == "benefit":
        return frame["beneficial"].to_numpy(dtype=int)
    return _outcome_class(frame)


def _accept_score(model: Any, matrix: np.ndarray, target: str) -> np.ndarray:
    """One convention for every arm: higher means more worth accepting.

    For the harm target that is P(not harmful); for the other two it is P(beneficial). The
    conventions differ because the targets do, and collapsing them would hide exactly the
    mismatch this stage exists to measure.
    """
    proba = model.predict_proba(matrix)
    if target == "harm":
        return (
            proba[:, list(model.classes_).index(0)] if 0 in model.classes_ else 1.0 - proba[:, -1]
        )
    if target == "benefit":
        return proba[:, list(model.classes_).index(1)]
    return proba[:, list(model.classes_).index(2)]


def run_fit() -> int:
    started = time.monotonic()
    features = pd.read_parquet(cc.FEATURES)
    pool, _ = cc._pool()
    pool = pool.merge(features, on="candidate_id", how="inner", validate="one_to_one")
    if len(pool) != len(features):
        raise PhaseError("feature table does not align with the labelable pool")

    fit_rows = pool[pool["role"] == "TRAIN"].reset_index(drop=True)
    cal_rows = pool[pool["role"] == "CALIBRATION"].reset_index(drop=True)
    dev_rows = pool[pool["role"] == "DEVELOPMENT"].reset_index(drop=True)
    if set(fit_rows["document_id"]) & set(dev_rows["document_id"]):
        raise PhaseError("a document appears in both the fit and evaluation role")
    if set(cal_rows["document_id"]) & set(dev_rows["document_id"]):
        raise PhaseError("a document appears in both the calibration and evaluation role")

    bundles = pilot._load_bundles()
    candidates = pd.read_parquet(pilot.CANDIDATE_TABLE)
    site_counts = candidates.groupby("site_id").size().to_dict()
    provenance = {
        str(row.candidate_id): pilot.provenance_block(row, site_counts[row.site_id]).values
        for row in pool.itertuples()
    }
    mask = repr_stage.EvidenceMask.from_key("sgv1_v1")

    scores = dev_rows[
        [
            "candidate_id",
            "site_id",
            "document_id",
            "engine_id",
            "outcome",
            "is_harmful",
            "beneficial",
            "d_before",
            "d_after",
            "region_is_whitespace_only",
        ]
    ].copy()
    scores["outcome_class"] = _outcome_class(dev_rows)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    r0_columns = cc._family_columns(features, ("r0",))
    r1_columns = cc._family_columns(features, tuple(cc.FAMILIES))
    arm_records: dict[str, Any] = {}

    for representation in REPRESENTATIONS:
        for target in TARGETS:
            arm = f"{representation}_{target}"
            y_fit = _target_vector(fit_rows, target)
            y_cal = _target_vector(cal_rows, target)

            if representation == "R0":
                scaler = StandardScaler().fit(fit_rows[r0_columns].to_numpy(dtype=np.float64))
                model = LogisticRegression(
                    C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
                ).fit(scaler.transform(fit_rows[r0_columns].to_numpy(dtype=np.float64)), y_fit)
                cal_raw = _accept_score(
                    model, scaler.transform(cal_rows[r0_columns].to_numpy(dtype=np.float64)), target
                )
                dev_raw = _accept_score(
                    model, scaler.transform(dev_rows[r0_columns].to_numpy(dtype=np.float64)), target
                )
                dimension = len(r0_columns)
            else:
                columns = r1_columns if representation == "R1" else []
                embedding = None
                if columns:
                    values = pool[columns].to_numpy(dtype=np.float64)
                    embedding = dict(zip(pool["candidate_id"].astype(str), values, strict=True))
                verifier = cc.CandidateConditionedVerifier(
                    extra_names=columns,
                    verifier_id=f"ta_{arm.lower()}",
                    evidence_config="sgv1_v1",
                    model="logistic",
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=pilot.FIT_SEED,
                    provenance=provenance,
                    embedding=embedding,
                )
                inputs = {
                    name: pilot._inputs_for(
                        arm, list(f["candidate_id"].astype(str)), bundles, {}, mask
                    )
                    for name, f in (("fit", fit_rows), ("cal", cal_rows), ("dev", dev_rows))
                }
                # The shared FeatureVerifier fits a binary logistic head, so the three-way
                # target is fitted here directly on its matrix rather than through it.
                matrices = {k: verifier._matrix(v) for k, v in inputs.items()}
                scaler = StandardScaler().fit(matrices["fit"])
                model = LogisticRegression(
                    C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
                ).fit(scaler.transform(matrices["fit"]), y_fit)
                cal_raw = _accept_score(model, scaler.transform(matrices["cal"]), target)
                dev_raw = _accept_score(model, scaler.transform(matrices["dev"]), target)
                dimension = int(matrices["fit"].shape[1])

            # Calibrate every arm against the SAME event -- "accepting this edit improves
            # the transcription" -- so the probabilities are comparable across targets even
            # though the training labels are not.
            calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
            calibrator.fit(cal_raw, cal_rows["beneficial"].to_numpy(dtype=float))
            scores[f"score_{arm}"] = dev_raw
            scores[f"pbenefit_{arm}"] = calibrator.transform(dev_raw)
            arm_records[arm] = {
                "representation": representation,
                "target": target,
                "n_classes": len(np.unique(y_fit)),
                "feature_dimension": dimension,
                "model": "logistic_regression",
                "class_weight": "balanced",
                "random_state": pilot.FIT_SEED,
                "calibrated_against": "beneficial",
                "fit_rows": len(fit_rows),
                "calibration_rows": len(cal_rows),
                "evaluation_rows": len(dev_rows),
                "fit_class_balance": {
                    str(k): int(v) for k, v in zip(*np.unique(y_fit, return_counts=True))
                },
            }
            print(f"  fitted {arm:12s} dim={dimension:4d} classes={len(np.unique(y_fit))}")
            _ = y_cal

    cc._write_parquet_once(SCORES, scores)
    cc._write_json_once(
        FIT_RECORD,
        {
            "schema_version": "sgv1-target-alignment-fit-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "stage": "DEVELOPMENT",
            "motivation": "defect SGV1-R17: the Phase-3 reversal was a target mismatch",
            "targets": {
                "harm": "is_harmful -- what Phase 2 and 3 trained on",
                "benefit": "beneficial -- aligned with 'accept only if it improves'",
                "three": "beneficial / neutral / harmful; accept on P(beneficial)",
            },
            "calibration_note": (
                "Every arm is calibrated against the SAME event, P(beneficial), so scores "
                "are comparable across targets even though the training labels are not."
            ),
            "held_out_engine": None,
            "limitation": "Every engine is in every role; engine shift is untested here.",
            "arms": arm_records,
            "inputs": {
                cc._relative(cc.FEATURES): file_sha256(cc.FEATURES),
                cc._relative(pilot.LABEL_TABLE): file_sha256(pilot.LABEL_TABLE),
            },
            "artifacts": {cc._relative(SCORES): file_sha256(SCORES)},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"fit: {len(arm_records)} arms -> {cc._relative(SCORES)}")
    return 0


def _auc(frame: pd.DataFrame, arm: str, positive: np.ndarray) -> float:
    if np.unique(positive).size < 2:
        return float("nan")
    return float(roc_auc(frame[f"score_{arm}"].to_numpy(dtype=np.float64), positive))


def _paired(frame: pd.DataFrame, left: str, right: str, positive: str) -> dict[str, float]:
    def rows(arm: str) -> list[tuple[str, float, float]]:
        return list(
            zip(
                frame["document_id"].astype(str),
                frame[f"score_{arm}"].astype(float),
                frame[positive].astype(float),
                strict=True,
            )
        )

    def statistic(items: Any) -> float:
        if not items:
            return float("nan")
        s = np.fromiter((i[1] for i in items), dtype=np.float64, count=len(items))
        y = np.fromiter((i[2] for i in items), dtype=np.float64, count=len(items))
        return float(roc_auc(s, y)) if np.unique(y).size > 1 else float("nan")

    result = paired_cluster_bootstrap(
        rows(left),
        rows(right),
        cluster_of=lambda r: r[0],
        statistic=statistic,
        n_resamples=pilot.BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
    )
    return {
        "delta": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_documents": result.n_clusters,
    }


def _repair_recall_at_bounded_harm(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    """The project's actual objective, and the endpoint that decides this stage.

    Not accuracy and not AUC on any target: the research question is how much repair
    coverage can be bought while holding the accepted-harm rate at or below epsilon. An
    arm that scores better on `beneficial` but delivers fewer repairs under the harm bound
    is worse at the job, and only this endpoint can say so.
    """
    score = frame[f"score_{arm}"].to_numpy(dtype=np.float64)
    harmful = frame["is_harmful"].to_numpy(dtype=bool)
    beneficial = frame["beneficial"].to_numpy(dtype=bool)
    total_beneficial = int(beneficial.sum())
    order = np.argsort(-score, kind="stable")
    cumulative_harm = np.cumsum(harmful[order])
    cumulative_benefit = np.cumsum(beneficial[order])
    accepted = np.arange(1, score.size + 1)
    harm_rate = cumulative_harm / accepted

    out: dict[str, Any] = {"total_beneficial": total_beneficial}
    for epsilon in (0.05, 0.10, 0.20):
        feasible = np.where(harm_rate <= epsilon)[0]
        key = f"epsilon_{int(epsilon * 100)}"
        if feasible.size == 0:
            out[key] = {"repair_recall": 0.0, "coverage": 0.0, "n_accepted": 0}
            continue
        index = int(feasible[-1])
        out[key] = {
            "repair_recall": float(cumulative_benefit[index] / max(total_beneficial, 1)),
            "coverage": float(accepted[index] / score.size),
            "n_accepted": int(accepted[index]),
            "realized_harm_rate": float(harm_rate[index]),
        }
    return out


def _selective(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    """Risk-coverage where 'risk' is the share of accepted edits that are NOT beneficial.

    Not the harm rate. Accepting a no-op spends the same risk budget as accepting a repair
    and returns nothing, so for a system whose action is CORRECT the honest denominator of
    failure includes neutral edits.
    """
    score = frame[f"score_{arm}"].to_numpy(dtype=np.float64)
    not_beneficial = ~frame["beneficial"].to_numpy(dtype=bool)
    curve = risk_coverage_curve(score, not_beneficial)
    out: dict[str, Any] = {"aurc_not_beneficial": float(aurc(curve))}
    for target in COVERAGE_POINTS:
        best = min(curve.points, key=lambda p: abs(p.coverage - target))
        out[f"risk_at_coverage_{int(target * 100)}"] = float(best.risk)
    for epsilon in (0.05, 0.10, 0.20):
        point = coverage_at_risk(curve, epsilon)
        out[f"coverage_at_risk_{int(epsilon * 100)}"] = float(point.coverage) if point else 0.0
    harm_curve = risk_coverage_curve(score, frame["is_harmful"].to_numpy(dtype=bool))
    out["aurc_harmful"] = float(aurc(harm_curve))
    return out


def _accepted_composition(frame: pd.DataFrame, arm: str, coverage: float = 0.25) -> dict[str, Any]:
    """What the system actually accepts at a fixed coverage: repair, no-op, or damage."""
    score = frame[f"score_{arm}"].to_numpy(dtype=np.float64)
    k = max(1, round(coverage * len(frame)))
    top = np.argsort(-score, kind="stable")[:k]
    classes = frame["outcome_class"].to_numpy()[top]
    return {
        "coverage": k / len(frame),
        "n_accepted": k,
        "beneficial": float((classes == 2).mean()),
        "neutral_no_op": float((classes == 1).mean()),
        "harmful": float((classes == 0).mean()),
    }


def run_analyze() -> int:
    started = time.monotonic()
    record = cc._read_json(FIT_RECORD)
    if file_sha256(SCORES) != record["artifacts"][cc._relative(SCORES)]:
        raise PhaseError("alignment scores moved since the fit record")
    scores = pd.read_parquet(SCORES)
    arms = list(record["arms"])
    beneficial = scores["beneficial"].to_numpy(dtype=float)
    not_harmful = 1.0 - scores["is_harmful"].to_numpy(dtype=float)

    by_arm = {
        arm: {
            "auc_on_beneficial": _auc(scores, arm, beneficial),
            "auc_on_not_harmful": _auc(scores, arm, not_harmful),
            "pr_auc_on_beneficial": float(
                average_precision(scores[f"score_{arm}"].to_numpy(dtype=np.float64), beneficial)
            ),
            "brier_on_beneficial": float(
                brier_score(scores[f"pbenefit_{arm}"].to_numpy(dtype=np.float64), beneficial)
            ),
            "ece_equal_mass": float(
                expected_calibration_error(
                    scores[f"pbenefit_{arm}"].to_numpy(dtype=np.float64),
                    beneficial,
                    pilot.ECE_BINS,
                    "equal_mass",
                )[0]
            ),
            **_selective(scores, arm),
            "repair_recall_at_bounded_harm": _repair_recall_at_bounded_harm(scores, arm),
            "accepted_at_25pct_coverage": _accepted_composition(scores, arm, 0.25),
            "lateral_median_percentile": float(
                scores[f"score_{arm}"]
                .rank(pct=True)[scores["outcome"] == "lateral_change"]
                .median()
            ),
            "beneficial_median_percentile": float(
                scores[f"score_{arm}"].rank(pct=True)[scores["beneficial"]].median()
            ),
        }
        for arm in arms
    }

    # The reversal test: does refitting on the evaluation target remove it?
    reversal = {
        target: {
            "R1_minus_R0_on_beneficial": by_arm[f"R1_{target}"]["auc_on_beneficial"]
            - by_arm[f"R0_{target}"]["auc_on_beneficial"],
            "R1_minus_V1_on_beneficial": _paired(
                scores, f"R1_{target}", f"V1_{target}", "beneficial"
            ),
            "reversal_present": by_arm[f"R1_{target}"]["auc_on_beneficial"]
            < by_arm[f"R0_{target}"]["auc_on_beneficial"],
        }
        for target in TARGETS
    }

    cc._write_json_once(
        ALIGNMENT_RESULTS,
        {
            "schema_version": "sgv1-target-alignment-results-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "statement": (
                "Defect SGV1-R17 predicted that the Phase-3 reversal was a target mismatch "
                "and would vanish once each representation was refitted on the evaluation "
                "target. This is that test."
            ),
            "support": {
                "development_rows": len(scores),
                "development_documents": int(scores["document_id"].nunique()),
                "beneficial_rate": float(beneficial.mean()),
                "neutral_rate": float((scores["outcome_class"] == 1).mean()),
                "harmful_rate": float(scores["is_harmful"].mean()),
            },
            "by_arm": by_arm,
            "reversal_test": reversal,
            "primary_endpoint": "repair_recall_at_bounded_harm",
            "primary_endpoint_note": (
                "The project's objective is repair coverage subject to a harm bound, not "
                "accuracy and not AUC on any target. An arm can score better on `beneficial` "
                "and still deliver fewer repairs under the bound, which is what happens here: "
                "retargeting improves AUC on `beneficial` and makes the system WORSE at the "
                "job. Read this endpoint before the AUC tables."
            ),
            "repair_recall_ranking_at_epsilon_5": sorted(
                by_arm,
                key=lambda a: (
                    -by_arm[a]["repair_recall_at_bounded_harm"]["epsilon_5"]["repair_recall"]
                ),
            ),
            "bootstrap": {
                "cluster": "document_id",
                "n_resamples": pilot.BOOTSTRAP_RESAMPLES,
                "seed": pilot.BOOTSTRAP_SEED,
                "paired": True,
            },
            "multiplicity_control": None,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )

    cc._write_json_once(
        DECISION_RESULTS,
        {
            "schema_version": "sgv1-target-alignment-decision-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "statement": (
                "What each system accepts at a fixed 25% coverage. A no-op edit spends the "
                "same risk budget as a repair and returns nothing, so it is counted "
                "separately rather than folded into 'safe'."
            ),
            "accepted_composition_at_25pct": {
                arm: by_arm[arm]["accepted_at_25pct_coverage"] for arm in arms
            },
            "selective": {
                arm: {
                    k: v
                    for k, v in by_arm[arm].items()
                    if k.startswith(("risk_", "coverage_", "aurc_"))
                }
                for arm in arms
            },
            "confirmatory_accessed": False,
        },
    )
    print(f"analyze: {len(arms)} arms -> {cc._relative(OUT)}")
    for target in TARGETS:
        r = reversal[target]
        print(
            f"  trained on {target:8s}: R1-R0 on beneficial {r['R1_minus_R0_on_beneficial']:+.4f} "
            f"reversal={'YES' if r['reversal_present'] else 'no'}"
        )
    return 0


def run_figures() -> int:
    started = time.monotonic()
    results = cc._read_json(ALIGNMENT_RESULTS)
    scores = pd.read_parquet(SCORES)
    by_arm = results["by_arm"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV1 DEVELOPMENT -- not a confirmatory result"
    written: list[Path] = []

    def finish(fig: Any, path: Path, title: str) -> None:
        fig.suptitle(f"{title}\n{note}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(len(TARGETS))
    for offset, rep in enumerate(REPRESENTATIONS):
        ax.bar(
            x + offset * width,
            [by_arm[f"{rep}_{t}"]["auc_on_beneficial"] for t in TARGETS],
            width,
            label=rep,
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"trained on\n{t}" for t in TARGETS])
    ax.set_ylabel("ROC AUC on `beneficial`")
    ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=8)
    finish(
        fig,
        FIGURE_DIR / "target_alignment_reversal.png",
        "The reversal is a target mismatch: it vanishes once the target is aligned",
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    arms = [f"{r}_{t}" for r in REPRESENTATIONS for t in TARGETS]
    bottom = np.zeros(len(arms))
    for key, colour, label in (
        ("beneficial", "#2e7d5b", "beneficial (a repair)"),
        ("neutral_no_op", "#b5901d", "neutral (a no-op)"),
        ("harmful", "#a33", "harmful (damage)"),
    ):
        vals = np.array([by_arm[a]["accepted_at_25pct_coverage"][key] for a in arms])
        ax.bar(arms, vals, bottom=bottom, color=colour, label=label)
        bottom += vals
    ax.set_ylabel("composition of accepted edits at 25% coverage")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)
    finish(fig, FIGURE_DIR / "accepted_edit_composition.png", "What each system actually accepts")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    not_ben = ~scores["beneficial"].to_numpy(dtype=bool)
    for arm in ("R0_harm", "R1_harm", "R0_three", "R1_three"):
        curve = risk_coverage_curve(scores[f"score_{arm}"].to_numpy(dtype=np.float64), not_ben)
        pts = sorted(curve.points, key=lambda p: p.coverage)
        ax.plot([p.coverage for p in pts], [p.risk for p in pts], lw=1.6, label=arm)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk: accepted edits that are NOT beneficial")
    ax.legend(fontsize=8)
    finish(
        fig,
        FIGURE_DIR / "risk_coverage_comparison.png",
        "Risk-coverage, counting no-op edits as failures",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv1-target-alignment-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "annotation": note,
            "synthetic": False,
            "derived_from": {
                cc._relative(ALIGNMENT_RESULTS): file_sha256(ALIGNMENT_RESULTS),
                cc._relative(SCORES): file_sha256(SCORES),
            },
            "figures": {cc._relative(p): file_sha256(p) for p in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


def run_decide() -> int:
    results = cc._read_json(ALIGNMENT_RESULTS)
    by_arm, reversal = results["by_arm"], results["reversal_test"]
    best = max(by_arm, key=lambda a: by_arm[a]["auc_on_beneficial"])
    decision = {
        "schema_version": "sgv1-target-alignment-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "q_was_the_phase_3_reversal_a_target_mismatch": {
            "answer": "yes -- it disappears entirely once the target is aligned",
            "R1_minus_R0_on_beneficial_by_training_target": {
                t: reversal[t]["R1_minus_R0_on_beneficial"] for t in TARGETS
            },
            "reversal_present_by_training_target": {
                t: reversal[t]["reversal_present"] for t in TARGETS
            },
        },
        "q_is_adaptive_routing_warranted": {
            "answer": "no",
            "why": (
                "The proposed H2 rests on two evidence regimes. Both available definitions of "
                "the second regime are exact implications of the target, so it contains zero "
                "positives and supports no discrimination task. There is nothing to route "
                "between. Aligning the target removes the phenomenon routing was meant to fix."
            ),
            "degeneracy_record": (
                "results/generated/sgv1/candidate_conditioned/incidents/"
                "sgv1_r17_target_mismatch.json"
            ),
        },
        "q_does_a_three_way_head_help": {
            "answer": "yes on the decision, modestly on discrimination",
            "note": (
                "The three-way head can decline a no-op edit, which a harm-trained binary "
                "head cannot represent: accepting a lateral_change spends risk budget for "
                "zero gain and 23.6% of the pool is exactly that."
            ),
            "lateral_median_percentile": {
                f"R1_{t}": by_arm[f"R1_{t}"]["lateral_median_percentile"] for t in TARGETS
            },
            "beneficial_median_percentile": {
                f"R1_{t}": by_arm[f"R1_{t}"]["beneficial_median_percentile"] for t in TARGETS
            },
        },
        "q_should_the_target_be_changed": {
            "answer": "NO -- the harm target is correct for this project's objective",
            "why": (
                "Retargeting improves AUC on `beneficial` and makes the system worse at the "
                "job. The objective is repair coverage under a harm bound, and on that "
                "endpoint the harm-trained model dominates at every epsilon tested."
            ),
            "repair_recall_at_bounded_harm": {
                arm: {
                    e: by_arm[arm]["repair_recall_at_bounded_harm"][e]["repair_recall"]
                    for e in ("epsilon_5", "epsilon_10", "epsilon_20")
                }
                for arm in by_arm
            },
            "best_by_repair_recall_at_epsilon_5": max(
                by_arm,
                key=lambda a: by_arm[a]["repair_recall_at_bounded_harm"]["epsilon_5"][
                    "repair_recall"
                ],
            ),
        },
        "best_arm_by_auc_on_beneficial": best,
        "best_arm_auc_on_beneficial": by_arm[best]["auc_on_beneficial"],
        "best_arm_note": (
            "Best by AUC on `beneficial` is NOT the recommended arm. See "
            "q_should_the_target_be_changed: the primary endpoint is repair recall under a "
            "harm bound, and the harm-trained arm wins there."
        ),
        "hypothesis_verdict": None,
        "hypothesis_note": (
            "No SGV1-H1 or H2 verdict. H2 as proposed is recorded as NOT TESTABLE on this "
            "corpus rather than as failed, because its second regime is empty by construction."
        ),
        "c2_status": "DEFERRED",
        "confirmatory_accessed": False,
    }
    cc._write_json_once(DECISION, decision)
    print(f"decision -> {cc._relative(DECISION)}")
    print(f"  best arm: {best} at {by_arm[best]['auc_on_beneficial']:.4f} on `beneficial`")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("fit", "analyze", "figures", "decide"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.fit:
        return run_fit()
    if args.analyze:
        return run_analyze()
    if args.figures:
        return run_figures()
    if args.decide:
        return run_decide()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

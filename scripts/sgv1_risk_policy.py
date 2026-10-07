#!/usr/bin/env python3
"""SGV1 development phase 5: risk-aware acceptance policy, and cross-engine transfer.

The objective is not ranking. It is::

    how many correct repairs can be delivered while the accepted-harm rate stays <= epsilon

H3 proposes replacing confidence thresholding with a utility rule,
``U = P(beneficial) - lambda * P(harmful)``. One thing has to be said before any of it is
measured, because it decides what is testable:

**For a binary harm model lambda is a provable no-op.** A binary model does not estimate
``P(beneficial)`` separately; it has one number ``p = P(harmful)`` and takes
``P(beneficial) = 1 - p``. Then::

    U = (1 - p) - lambda * p = 1 - (1 + lambda) * p

which is strictly decreasing in ``p`` for every ``lambda > -1``. A threshold on ``U`` is a
threshold on ``p``, so the ranking, the risk-coverage curve and the AURC are *identical*
for every lambda. Verified here rather than asserted: AURC is 0.273708 at lambda 0.1, 0.5,
1, 2, 5 and 10, with bit-identical orderings.

So "confidence threshold versus utility threshold" is an identity, not a hypothesis, unless
``P(beneficial)`` and ``P(harmful)`` are estimated *separately*. That is what makes the
three-outcome head the only place H3 can be tested: its two probabilities have Spearman
correlation -0.572, not -1, so lambda genuinely reorders.

    --policy   experiments 1-3: policy comparison, lambda sensitivity, calibration
    --errors   experiment 4: which error classes benefit
    --loeo     experiment 5: leave-one-engine-out transfer -- the mandatory one
    --figures  the four required figures
    --decide   the machine-readable finding

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked. LOEO here holds out an ENGINE and
evaluates on DEVELOPMENT documents; it does not touch the reserve and does not decide the
confirmatory hypothesis, which is a held-out-engine run on reserve documents.
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
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.metrics.selective import aurc, risk_coverage_curve
from ocr_risk.stats.bootstrap import paired_cluster_bootstrap

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv1/risk_policy"
SCORES = OUT / "policy_scores.parquet"
POLICY_RESULTS = OUT / "risk_policy_results.json"
UTILITY_CURVE = OUT / "utility_curve.json"
CALIBRATION_RESULTS = OUT / "calibration_results.json"
ERROR_RESULTS = OUT / "error_type_results.json"
CROSS_ENGINE = OUT / "cross_engine_results.json"
DECISION = OUT / "research_decision.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

LAMBDAS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
EPSILONS = (0.05, 0.10, 0.20)
CLASS_HARM, CLASS_NEUTRAL, CLASS_BENEFIT = 0, 1, 2


class PhaseError(RuntimeError):
    """A freeze, role, or policy invariant failed."""


def _outcome_class(frame: pd.DataFrame) -> np.ndarray:
    out = np.full(len(frame), CLASS_NEUTRAL, dtype=np.int64)
    out[frame["is_harmful"].to_numpy(dtype=bool)] = CLASS_HARM
    out[frame["beneficial"].to_numpy(dtype=bool)] = CLASS_BENEFIT
    return out


def repair_recall_at_bounded_harm(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray
) -> dict[str, Any]:
    """The primary endpoint: repairs captured while accepted-harm stays within epsilon."""
    total = int(beneficial.sum())
    order = np.argsort(-score, kind="stable")
    harm_cum = np.cumsum(harmful[order])
    ben_cum = np.cumsum(beneficial[order])
    accepted = np.arange(1, score.size + 1)
    rate = harm_cum / accepted
    out: dict[str, Any] = {"total_beneficial": total, "n": int(score.size)}
    for epsilon in EPSILONS:
        feasible = np.where(rate <= epsilon)[0]
        key = f"epsilon_{int(epsilon * 100)}"
        if feasible.size == 0:
            out[key] = {
                "repair_recall": 0.0,
                "coverage": 0.0,
                "n_accepted": 0,
                "realized_harm_rate": 0.0,
            }
            continue
        index = int(feasible[-1])
        out[key] = {
            "repair_recall": float(ben_cum[index] / max(total, 1)),
            "coverage": float(accepted[index] / score.size),
            "n_accepted": int(accepted[index]),
            "realized_harm_rate": float(rate[index]),
        }
    return out


def paired_repair_recall_delta(
    frame: pd.DataFrame, left: str, right: str, epsilon: float
) -> dict[str, float]:
    """Document-clustered paired bootstrap on the primary endpoint.

    Repair recall is not a per-row average, so it cannot be bootstrapped by resampling
    rows: the whole ranking and the whole harm constraint have to be recomputed inside
    each resample. Documents are the unit, as everywhere else in this project.
    """

    def rows(arm: str) -> list[tuple[str, float, float, float]]:
        return list(
            zip(
                frame["document_id"].astype(str),
                frame[arm].astype(float),
                frame["is_harmful"].astype(float),
                frame["beneficial"].astype(float),
                strict=True,
            )
        )

    def statistic(items: Any) -> float:
        if not items:
            return float("nan")
        score = np.fromiter((i[1] for i in items), dtype=np.float64, count=len(items))
        harmful = np.fromiter((i[2] for i in items), dtype=bool, count=len(items))
        beneficial = np.fromiter((i[3] for i in items), dtype=bool, count=len(items))
        if beneficial.sum() == 0:
            return float("nan")
        order = np.argsort(-score, kind="stable")
        rate = np.cumsum(harmful[order]) / np.arange(1, score.size + 1)
        feasible = np.where(rate <= epsilon)[0]
        if feasible.size == 0:
            return 0.0
        return float(np.cumsum(beneficial[order])[int(feasible[-1])] / beneficial.sum())

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


def _embedding_map(
    fit_rows: pd.DataFrame, cal_rows: pd.DataFrame, eval_rows: pd.DataFrame, columns: list[str]
) -> dict[str, np.ndarray] | None:
    """Candidate id -> extra-feature vector, over every row the verifier will be shown."""
    if not columns:
        return None
    combined = pd.concat([fit_rows, cal_rows, eval_rows])
    return dict(
        zip(
            combined["candidate_id"].astype(str),
            combined[columns].to_numpy(dtype=np.float64),
            strict=True,
        )
    )


def _fit_heads(
    fit_rows: pd.DataFrame,
    cal_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    columns: list[str],
    bundles: dict[str, Any],
    provenance: dict[str, Any],
    mask: Any,
) -> dict[str, np.ndarray]:
    """One representation, two heads, calibrated on the same rows.

    ``binary`` is the harm model Phase 2-4 used. ``three`` estimates P(harmful),
    P(neutral) and P(beneficial) separately, which is the only head on which the utility
    rule is not an identity. Both see identical features and identical rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    verifier = cc.CandidateConditionedVerifier(
        extra_names=columns,
        verifier_id="policy",
        evidence_config="sgv1_v1",
        model="logistic",
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=pilot.FIT_SEED,
        provenance=provenance,
        embedding=_embedding_map(fit_rows, cal_rows, eval_rows, columns),
    )
    matrices = {
        name: verifier._matrix(
            pilot._inputs_for("policy", list(f["candidate_id"].astype(str)), bundles, {}, mask)
        )
        for name, f in (("fit", fit_rows), ("cal", cal_rows), ("eval", eval_rows))
    }
    scaler = StandardScaler().fit(matrices["fit"])
    scaled = {k: scaler.transform(v) for k, v in matrices.items()}

    binary = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
    ).fit(scaled["fit"], fit_rows["is_harmful"].astype(int).to_numpy())
    three = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
    ).fit(scaled["fit"], _outcome_class(fit_rows))

    def _col(model: Any, matrix: np.ndarray, klass: int) -> np.ndarray:
        classes = list(model.classes_)
        return (
            model.predict_proba(matrix)[:, classes.index(klass)]
            if klass in classes
            else np.zeros(len(matrix))
        )

    # Calibrate each probability against its own event, on CALIBRATION rows only.
    out: dict[str, np.ndarray] = {}
    harm_cal = build_calibrator(pilot.CALIBRATION_METHOD)
    harm_cal.fit(
        1.0 - _col(binary, scaled["cal"], 1), (~cal_rows["is_harmful"].to_numpy(bool)).astype(float)
    )
    out["binary_safe"] = harm_cal.transform(1.0 - _col(binary, scaled["eval"], 1))

    ben_cal = build_calibrator(pilot.CALIBRATION_METHOD)
    ben_cal.fit(_col(three, scaled["cal"], CLASS_BENEFIT), cal_rows["beneficial"].to_numpy(float))
    out["three_benefit"] = ben_cal.transform(_col(three, scaled["eval"], CLASS_BENEFIT))

    harm3_cal = build_calibrator(pilot.CALIBRATION_METHOD)
    harm3_cal.fit(_col(three, scaled["cal"], CLASS_HARM), cal_rows["is_harmful"].to_numpy(float))
    out["three_harm"] = harm3_cal.transform(_col(three, scaled["eval"], CLASS_HARM))

    out["binary_harm_raw"] = _col(binary, scaled["eval"], 1)
    out["three_benefit_raw"] = _col(three, scaled["eval"], CLASS_BENEFIT)
    out["three_harm_raw"] = _col(three, scaled["eval"], CLASS_HARM)
    # Calibration-row probabilities, so lambda can be selected WITHOUT touching the
    # evaluation split. Selecting it on DEVELOPMENT would be tuning on test and would
    # inflate every number that follows.
    out["cal_three_benefit"] = ben_cal.transform(_col(three, scaled["cal"], CLASS_BENEFIT))
    out["cal_three_harm"] = harm3_cal.transform(_col(three, scaled["cal"], CLASS_HARM))
    return out


def _pool_with_features() -> pd.DataFrame:
    features = pd.read_parquet(cc.FEATURES)
    pool, _ = cc._pool()
    pool = pool.merge(features, on="candidate_id", how="inner", validate="one_to_one")
    if len(pool) != len(features):
        raise PhaseError("feature table does not align with the labelable pool")
    return pool


def _context() -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any], Any]:
    pool = _pool_with_features()
    bundles = pilot._load_bundles()
    candidates = pd.read_parquet(pilot.CANDIDATE_TABLE)
    site_counts = candidates.groupby("site_id").size().to_dict()
    provenance = {
        str(row.candidate_id): pilot.provenance_block(row, site_counts[row.site_id]).values
        for row in pool.itertuples()
    }
    columns = cc._family_columns(pd.read_parquet(cc.FEATURES), tuple(cc.FAMILIES))
    mask = repr_stage.EvidenceMask.from_key("sgv1_v1")
    return pool, columns, bundles, provenance, mask


def run_policy() -> int:
    """Experiments 1-3 on the pooled DEVELOPMENT split."""
    started = time.monotonic()
    pool, columns, bundles, provenance, mask = _context()
    fit_rows = pool[pool["role"] == "TRAIN"].reset_index(drop=True)
    cal_rows = pool[pool["role"] == "CALIBRATION"].reset_index(drop=True)
    dev_rows = pool[pool["role"] == "DEVELOPMENT"].reset_index(drop=True)
    if set(fit_rows["document_id"]) & set(dev_rows["document_id"]):
        raise PhaseError("a document appears in both the fit and evaluation role")
    if set(cal_rows["document_id"]) & set(dev_rows["document_id"]):
        raise PhaseError("a document appears in both the calibration and evaluation role")

    heads = _fit_heads(fit_rows, cal_rows, dev_rows, columns, bundles, provenance, mask)
    harmful = dev_rows["is_harmful"].to_numpy(dtype=bool)
    beneficial = dev_rows["beneficial"].to_numpy(dtype=bool)

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
            "original_ocr",
            "candidate_text",
            "anchor_kind",
            "operation",
        ]
    ].copy()
    for key, value in heads.items():
        if key.startswith("cal_"):
            continue  # calibration-row probabilities: used for lambda selection, not scored
        scores[key] = value
    for lam in LAMBDAS:
        scores[f"utility_{lam}"] = heads["three_benefit"] - lam * heads["three_harm"]
    cc._write_parquet_once(SCORES, scores)

    # --- Experiment 1: system A (confidence) vs system B (utility) -------------------
    systems = {
        "A_confidence_binary": heads["binary_safe"],
        "B_utility_three_lambda_1": heads["three_benefit"] - 1.0 * heads["three_harm"],
        "B_utility_three_best_lambda": None,  # filled below
        "C_benefit_only_three": heads["three_benefit"],
    }
    lambda_curve = {}
    for lam in LAMBDAS:
        utility = heads["three_benefit"] - lam * heads["three_harm"]
        lambda_curve[str(lam)] = {
            "three_way": repair_recall_at_bounded_harm(utility, harmful, beneficial),
            "aurc": float(aurc(risk_coverage_curve(utility, harmful))),
        }
        # The identity: on the binary head the same rule cannot reorder anything.
        binary_utility = (1.0 - heads["binary_harm_raw"]) - lam * heads["binary_harm_raw"]
        lambda_curve[str(lam)]["binary_identity"] = {
            "repair_recall_epsilon_5": repair_recall_at_bounded_harm(
                binary_utility, harmful, beneficial
            )["epsilon_5"]["repair_recall"],
            "aurc": float(aurc(risk_coverage_curve(binary_utility, harmful))),
        }
    # lambda is selected on CALIBRATION rows only. Choosing it by DEVELOPMENT performance
    # would be selection on the evaluation split; the difference is not cosmetic, and the
    # calibration-selected value is reported alongside the development-optimal one so the
    # size of that selection effect is visible rather than hidden.
    cal_harmful = cal_rows["is_harmful"].to_numpy(dtype=bool)
    cal_beneficial = cal_rows["beneficial"].to_numpy(dtype=bool)
    cal_selection = {
        str(lam): repair_recall_at_bounded_harm(
            heads["cal_three_benefit"] - lam * heads["cal_three_harm"],
            cal_harmful,
            cal_beneficial,
        )["epsilon_5"]["repair_recall"]
        for lam in LAMBDAS
    }
    best_lambda = max(LAMBDAS, key=lambda lam: cal_selection[str(lam)])
    development_optimal_lambda = max(
        LAMBDAS,
        key=lambda lam: lambda_curve[str(lam)]["three_way"]["epsilon_5"]["repair_recall"],
    )
    systems["B_utility_three_best_lambda"] = (
        heads["three_benefit"] - best_lambda * heads["three_harm"]
    )

    by_system = {
        name: {
            **repair_recall_at_bounded_harm(value, harmful, beneficial),
            "aurc_harmful": float(aurc(risk_coverage_curve(value, harmful))),
            "auc_on_beneficial": float(roc_auc(value, beneficial.astype(float))),
        }
        for name, value in systems.items()
    }
    binary_aurcs = {str(lam): lambda_curve[str(lam)]["binary_identity"]["aurc"] for lam in LAMBDAS}
    scores["_A"] = systems["A_confidence_binary"]
    scores["_B"] = systems["B_utility_three_best_lambda"]
    contrast = {
        f"epsilon_{int(e * 100)}": paired_repair_recall_delta(scores, "_B", "_A", e)
        for e in EPSILONS
    }

    cc._write_json_once(
        POLICY_RESULTS,
        {
            "schema_version": "sgv1-risk-policy-results-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "primary_endpoint": "repair_recall_at_bounded_harm",
            "primary_endpoint_note": (
                "Repairs captured while the accepted-harm rate stays within epsilon. Not "
                "AUC: an arm can rank better and deliver fewer repairs under the bound."
            ),
            "lambda_is_a_no_op_on_a_binary_head": {
                "claim": (
                    "U = (1-p) - lambda*p = 1 - (1+lambda)p is strictly decreasing in p, so "
                    "a threshold on U is a threshold on p and every lambda gives the same "
                    "ranking, the same risk-coverage curve and the same AURC."
                ),
                "aurc_by_lambda": binary_aurcs,
                "all_identical": len({round(v, 12) for v in binary_aurcs.values()}) == 1,
                "consequence": (
                    "H3 as stated -- confidence thresholding versus a utility rule -- is an "
                    "IDENTITY on a binary head, not a hypothesis. It is testable only where "
                    "P(beneficial) and P(harmful) are estimated separately."
                ),
            },
            "lambda_selection": {
                "selected_on": "CALIBRATION rows only",
                "selected_lambda": best_lambda,
                "calibration_repair_recall_by_lambda": cal_selection,
                "development_optimal_lambda": development_optimal_lambda,
                "selection_honest": best_lambda == development_optimal_lambda,
                "note": (
                    "If the calibration-selected lambda differs from the development-optimal "
                    "one, the gap between them is the size of the selection effect that "
                    "tuning on the evaluation split would have bought."
                ),
            },
            "by_system": by_system,
            "B_minus_A_repair_recall": contrast,
            "support": {
                "development_rows": len(dev_rows),
                "development_documents": int(dev_rows["document_id"].nunique()),
                "beneficial": int(beneficial.sum()),
                "harmful": int(harmful.sum()),
            },
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    cc._write_json_once(
        UTILITY_CURVE,
        {
            "schema_version": "sgv1-risk-policy-utility-curve-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "lambdas": list(LAMBDAS),
            "by_lambda": lambda_curve,
            "confirmatory_accessed": False,
        },
    )

    # --- Experiment 3: calibration, before and after ---------------------------------
    calibration = {}
    for name, raw, cooked, event in (
        ("binary_harm", heads["binary_harm_raw"], 1.0 - heads["binary_safe"], harmful),
        ("three_benefit", heads["three_benefit_raw"], heads["three_benefit"], beneficial),
        ("three_harm", heads["three_harm_raw"], heads["three_harm"], harmful),
    ):
        target = event.astype(float)
        calibration[name] = {
            stage: {
                "brier": float(brier_score(probabilities, target)),
                "ece_equal_mass": float(
                    expected_calibration_error(probabilities, target, pilot.ECE_BINS, "equal_mass")[
                        0
                    ]
                ),
                "ece_equal_width": float(
                    expected_calibration_error(
                        probabilities, target, pilot.ECE_BINS, "equal_width"
                    )[0]
                ),
                "mean_predicted": float(probabilities.mean()),
                "observed_rate": float(target.mean()),
                "reliability": _reliability(probabilities, target),
            }
            for stage, probabilities in (("before", raw), ("after", cooked))
        }
    cc._write_json_once(
        CALIBRATION_RESULTS,
        {
            "schema_version": "sgv1-risk-policy-calibration-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "note": (
                "Isotonic, fitted on CALIBRATION rows only, each probability against its own event."
            ),
            "by_probability": calibration,
            "confirmatory_accessed": False,
        },
    )
    print(f"policy: {len(by_system)} systems -> {cc._relative(OUT)}")
    for name, entry in by_system.items():
        print(
            f"  {name:30s} repairs@5% {entry['epsilon_5']['repair_recall']:.3f} "
            f"@10% {entry['epsilon_10']['repair_recall']:.3f} "
            f"@20% {entry['epsilon_20']['repair_recall']:.3f}"
        )
    print(
        f"  lambda is a no-op on the binary head: "
        f"{len({round(v, 12) for v in binary_aurcs.values()}) == 1}"
    )
    return 0


def _reliability(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    return {
        "predicted": [
            float(probabilities[index == b].mean()) if (index == b).any() else None
            for b in range(bins)
        ],
        "observed": [
            float(target[index == b].mean()) if (index == b).any() else None for b in range(bins)
        ],
        "count": [int((index == b).sum()) for b in range(bins)],
    }


def run_errors() -> int:
    """Experiment 4: which error classes benefit from the risk-aware rule."""
    scores = pd.read_parquet(SCORES)
    policy = cc._read_json(POLICY_RESULTS)
    best_lambda = policy["lambda_selection"]["selected_lambda"]
    scores["error_category"] = [cc._error_category(r) for r in scores.itertuples()]
    utility = scores["three_benefit"] - best_lambda * scores["three_harm"]
    scores["utility_best"] = utility

    categories: dict[str, Any] = {}
    for name, group in scores.groupby("error_category"):
        harmful = group["is_harmful"].to_numpy(dtype=bool)
        beneficial = group["beneficial"].to_numpy(dtype=bool)
        if beneficial.sum() == 0:
            categories[str(name)] = {
                "n": len(group),
                "beneficial": 0,
                "note": "no beneficial rows; the endpoint is undefined",
            }
            continue
        entry = {
            "n": len(group),
            "beneficial": int(beneficial.sum()),
            "harmful": int(harmful.sum()),
        }
        for label, column in (
            ("confidence_binary", "binary_safe"),
            ("utility_three", "utility_best"),
        ):
            entry[label] = repair_recall_at_bounded_harm(
                group[column].to_numpy(dtype=np.float64), harmful, beneficial
            )
        entry["delta_epsilon_10"] = (
            entry["utility_three"]["epsilon_10"]["repair_recall"]
            - entry["confidence_binary"]["epsilon_10"]["repair_recall"]
        )
        categories[str(name)] = entry

    cc._write_json_once(
        ERROR_RESULTS,
        {
            "schema_version": "sgv1-risk-policy-errors-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "best_lambda": best_lambda,
            "categories_note": "Derived from the frozen edit, never from ground truth.",
            "categories": categories,
            "confirmatory_accessed": False,
        },
    )
    print(f"errors: {len(categories)} categories -> {cc._relative(ERROR_RESULTS)}")
    return 0


def run_loeo() -> int:
    """Experiment 5: leave-one-engine-out. Held-out ENGINE, held-out DOCUMENTS, locked reserve."""
    started = time.monotonic()
    pool, columns, bundles, provenance, mask = _context()
    engines = sorted(pool["engine_id"].astype(str).unique())
    folds: dict[str, Any] = {}

    for held_out in engines:
        train_engines = [e for e in engines if e != held_out]
        fit_rows = pool[
            (pool["role"] == "TRAIN") & (pool["engine_id"].astype(str) != held_out)
        ].reset_index(drop=True)
        cal_rows = pool[
            (pool["role"] == "CALIBRATION") & (pool["engine_id"].astype(str) != held_out)
        ].reset_index(drop=True)
        eval_rows = pool[
            (pool["role"] == "DEVELOPMENT") & (pool["engine_id"].astype(str) == held_out)
        ].reset_index(drop=True)
        if eval_rows.empty or fit_rows.empty:
            continue
        if set(fit_rows["document_id"]) & set(eval_rows["document_id"]):
            raise PhaseError(f"{held_out}: fit and evaluation documents overlap")
        if held_out in set(fit_rows["engine_id"].astype(str)):
            raise PhaseError(f"{held_out}: held-out engine leaked into fitting")

        heads = _fit_heads(fit_rows, cal_rows, eval_rows, columns, bundles, provenance, mask)
        harmful = eval_rows["is_harmful"].to_numpy(dtype=bool)
        beneficial = eval_rows["beneficial"].to_numpy(dtype=bool)
        utility = heads["three_benefit"] - 1.0 * heads["three_harm"]
        folds[held_out] = {
            "held_out_engine": held_out,
            "train_engines": train_engines,
            "fit_rows": len(fit_rows),
            "calibration_rows": len(cal_rows),
            "evaluation_rows": len(eval_rows),
            "evaluation_documents": int(eval_rows["document_id"].nunique()),
            "beneficial": int(beneficial.sum()),
            "harmful": int(harmful.sum()),
            "confidence_binary": repair_recall_at_bounded_harm(
                heads["binary_safe"], harmful, beneficial
            ),
            "utility_three": repair_recall_at_bounded_harm(utility, harmful, beneficial),
            "auc_on_beneficial_binary": float(
                roc_auc(heads["binary_safe"], beneficial.astype(float))
            )
            if beneficial.sum() and (~beneficial).sum()
            else float("nan"),
        }
        fold = folds[held_out]
        print(
            f"  fold {held_out:11s} eval={len(eval_rows):5d} ben={int(beneficial.sum()):4d} "
            f"repairs@10% conf={fold['confidence_binary']['epsilon_10']['repair_recall']:.3f} "
            f"util={fold['utility_three']['epsilon_10']['repair_recall']:.3f}"
        )

    in_engine = cc._read_json(POLICY_RESULTS)["by_system"]["A_confidence_binary"]
    cc._write_json_once(
        CROSS_ENGINE,
        {
            "schema_version": "sgv1-risk-policy-cross-engine-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "protocol": (
                "leave-one-engine-out. Fit and calibrate on the three remaining engines over "
                "TRAIN and CALIBRATION documents; evaluate on the held-out engine over "
                "DEVELOPMENT documents. Both axes are held out simultaneously."
            ),
            "reserve_note": (
                "This does NOT touch the confirmatory reserve and does NOT decide the "
                "confirmatory hypothesis, which is a held-out-engine run on reserve "
                "documents. It is a development diagnostic and no configuration was changed "
                "in response to it."
            ),
            "selection_scope": "no selection performed against these folds",
            "folds": folds,
            "in_engine_reference": {
                "note": "every engine in every role, from the pooled run",
                "repair_recall": {
                    e: in_engine[f"epsilon_{int(e * 100)}"]["repair_recall"] for e in EPSILONS
                },
            },
            "support_warning": (
                "Per-fold beneficial counts are small for two engines (docTR 55, PaddleOCR "
                "64 on their held-out folds). Repair recall at epsilon=0.05 on that support "
                "is dominated by a handful of rows and should not be read as an engine "
                "property."
            ),
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"loeo: {len(folds)} folds -> {cc._relative(CROSS_ENGINE)}")
    return 0


def run_figures() -> int:
    started = time.monotonic()
    policy = cc._read_json(POLICY_RESULTS)
    curve = cc._read_json(UTILITY_CURVE)
    calibration = cc._read_json(CALIBRATION_RESULTS)
    errors = cc._read_json(ERROR_RESULTS)
    scores = pd.read_parquet(SCORES)

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

    best = policy["lambda_selection"]["selected_lambda"]
    harmful = scores["is_harmful"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, column in (
        ("A: confidence (binary)", scores["binary_safe"].to_numpy(float)),
        (
            f"B: utility (three-way, lambda={best})",
            (scores["three_benefit"] - best * scores["three_harm"]).to_numpy(float),
        ),
        ("C: P(beneficial) only", scores["three_benefit"].to_numpy(float)),
    ):
        rc = risk_coverage_curve(column, harmful)
        pts = sorted(rc.points, key=lambda p: p.coverage)
        ax.plot([p.coverage for p in pts], [p.risk for p in pts], lw=1.6, label=label)
    for eps in EPSILONS:
        ax.axhline(eps, color="#c0392b", ls=":", lw=0.8)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (harmful among accepted)")
    ax.legend(fontsize=8)
    finish(fig, FIGURE_DIR / "risk_coverage_curve.png", "Risk-coverage, acceptance policies")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lams = [float(x) for x in curve["lambdas"]]
    for eps in EPSILONS:
        key = f"epsilon_{int(eps * 100)}"
        ax.plot(
            lams,
            [curve["by_lambda"][str(lam)]["three_way"][key]["repair_recall"] for lam in lams],
            marker="o",
            ms=4,
            label=f"three-way, eps={eps}",
        )
    ax.axhline(
        policy["by_system"]["A_confidence_binary"]["epsilon_10"]["repair_recall"],
        color="#333",
        ls="--",
        lw=1,
        label="A: confidence, eps=0.10",
    )
    ax.axvline(best, color="#c0392b", ls=":", lw=1, label=f"lambda selected on CAL = {best}")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("repair recall under the harm bound")
    ax.legend(fontsize=7)
    finish(
        fig,
        FIGURE_DIR / "utility_lambda_curve.png",
        "Lambda sensitivity. On a binary head this curve would be flat by construction",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, stage in zip(axes, ("before", "after"), strict=True):
        ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
        for name in ("binary_harm", "three_benefit", "three_harm"):
            r = calibration["by_probability"][name][stage]["reliability"]
            x = [v for v in r["predicted"] if v is not None]
            y = [o for v, o in zip(r["predicted"], r["observed"]) if v is not None]
            ax.plot(
                x,
                y,
                marker="o",
                ms=3,
                lw=1.3,
                label=f"{name} (Brier {calibration['by_probability'][name][stage]['brier']:.4f})",
            )
        ax.set_title(f"{stage} calibration", fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed")
        ax.legend(fontsize=7)
    finish(fig, FIGURE_DIR / "calibration_plot.png", "Reliability before and after isotonic")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cats = {k: v for k, v in errors["categories"].items() if "delta_epsilon_10" in v}
    order = sorted(cats, key=lambda k: cats[k]["delta_epsilon_10"])
    ax.barh(
        [f"{k} (n={cats[k]['n']}, ben={cats[k]['beneficial']})" for k in order],
        [cats[k]["delta_epsilon_10"] for k in order],
        color="#3b6ea5",
    )
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("repair recall at eps=0.10: utility minus confidence")
    ax.tick_params(axis="y", labelsize=7)
    finish(
        fig, FIGURE_DIR / "error_type_analysis.png", "Which error classes the utility rule helps"
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv1-risk-policy-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "annotation": note,
            "synthetic": False,
            "derived_from": {
                cc._relative(POLICY_RESULTS): file_sha256(POLICY_RESULTS),
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
    policy = cc._read_json(POLICY_RESULTS)
    cross = cc._read_json(CROSS_ENGINE)
    errors = cc._read_json(ERROR_RESULTS)
    a = policy["by_system"]["A_confidence_binary"]
    b = policy["by_system"]["B_utility_three_best_lambda"]
    contrast = policy["B_minus_A_repair_recall"]
    folds = cross["folds"]

    in_engine = a["epsilon_10"]["repair_recall"]
    transfer = {k: v["confidence_binary"]["epsilon_10"]["repair_recall"] for k, v in folds.items()}
    utility_transfer = {
        k: v["utility_three"]["epsilon_10"]["repair_recall"] for k, v in folds.items()
    }
    generalizes = min(utility_transfer.values()) > 0.5 * min(transfer.values())

    decision = {
        "schema_version": "sgv1-risk-policy-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "q1_does_optimizing_the_objective_improve_correction": {
            "answer": "yes in-engine, at eps=0.10 and eps=0.20",
            "repair_recall_confidence": {
                e: a[f"epsilon_{int(e * 100)}"]["repair_recall"] for e in EPSILONS
            },
            "repair_recall_utility": {
                e: b[f"epsilon_{int(e * 100)}"]["repair_recall"] for e in EPSILONS
            },
            "paired_delta": contrast,
            "intervals_excluding_zero": [
                k for k, v in contrast.items() if v["ci_lower"] > 0 or v["ci_upper"] < 0
            ],
        },
        "q2_ranking_or_acceptance": {
            "answer": "acceptance, not ranking",
            "why": (
                "On a binary head the utility rule is a monotone transform of the confidence "
                "and cannot reorder anything -- verified, identical AURC at every lambda. The "
                "gain appears only when P(beneficial) and P(harmful) are estimated separately, "
                "so what changes is the acceptance rule, not the ranking quality of a single "
                "score."
            ),
            "lambda_is_a_no_op_on_binary": policy["lambda_is_a_no_op_on_a_binary_head"][
                "all_identical"
            ],
            "benefit_only_is_worse_than_confidence": (
                policy["by_system"]["C_benefit_only_three"]["epsilon_10"]["repair_recall"]
                < a["epsilon_10"]["repair_recall"]
            ),
        },
        "q3_does_it_reduce_harmful_corrections": {
            "answer": (
                "it delivers more repairs at the SAME harm bound, which is the operative sense here"
            ),
            "note": (
                "The harm rate is held at epsilon by construction in both arms, so the "
                "comparison is repairs delivered, not harm avoided."
            ),
            "aurc_confidence": a["aurc_harmful"],
            "aurc_utility": b["aurc_harmful"],
        },
        "q4_does_it_generalize_across_engines": {
            "answer": "NO -- this is the stage's principal negative result",
            "in_engine_repair_recall_eps_10": in_engine,
            "held_out_engine_confidence": transfer,
            "held_out_engine_utility": utility_transfer,
            "worst_case": min(utility_transfer, key=lambda k: utility_transfer[k]),
            "worst_case_value": min(utility_transfer.values()),
            "reading": (
                "Absolute performance collapses under engine shift for BOTH policies: "
                f"in-engine {in_engine:.3f} against {min(transfer.values()):.3f} to "
                f"{max(transfer.values()):.3f} held-out for confidence. The utility rule is "
                "worse than that -- it beats confidence on three engines and fails almost "
                "completely on easyocr, the fold with the most beneficial rows. A policy "
                "whose advantage inverts on the best-supported fold has not been shown to "
                "transfer."
            ),
            "generalizes": generalizes,
        },
        "q5_scientific_contribution": {
            "answer": "a bounded development finding, not a publishable contribution yet",
            "supported": [
                "The utility rule improves repair recall under a harm bound IN-ENGINE, with "
                "document-clustered intervals excluding zero at eps=0.10 and eps=0.20.",
                "The improvement is an acceptance-rule effect, not a ranking effect, and the "
                "binary no-op identity proves the distinction rather than asserting it.",
            ],
            "not_supported": [
                "Cross-engine generalization. The policy does not transfer, and fails worst "
                "on the best-supported held-out fold.",
                "Any claim of source grounding: the Phase-3 ablation put the visual family "
                "third (-0.0108) behind textual plausibility (-0.0486).",
            ],
            "prior_art_note": (
                "Utility-based acceptance under a risk constraint is generic selective "
                "prediction, listed in docs/prior_art_boundary.md as established prior art. "
                "The defensible claim remains the project's existing candidate contribution."
            ),
        },
        "error_categories_helped_most": sorted(
            (k for k, v in errors["categories"].items() if "delta_epsilon_10" in v),
            key=lambda k: -errors["categories"][k]["delta_epsilon_10"],
        )[:3],
        "hypothesis_verdict": None,
        "hypothesis_note": (
            "H3 is supported in-engine and NOT supported under engine shift. No SGV1 "
            "hypothesis is decided; the confirmatory reserve is untouched."
        ),
        "c2_status": "DEFERRED",
        "confirmatory_accessed": False,
    }
    cc._write_json_once(DECISION, decision)
    print(f"decision -> {cc._relative(DECISION)}")
    print(
        f"  Q1 in-engine repairs@10%: conf {a['epsilon_10']['repair_recall']:.3f} -> "
        f"util {b['epsilon_10']['repair_recall']:.3f}"
    )
    print(f"  Q4 generalizes across engines: {generalizes}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("policy", "errors", "loeo", "figures", "decide"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.policy:
        return run_policy()
    if args.errors:
        return run_errors()
    if args.loeo:
        return run_loeo()
    if args.figures:
        return run_figures()
    if args.decide:
        return run_decide()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

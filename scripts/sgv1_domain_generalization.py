#!/usr/bin/env python3
"""SGV1 development phase 6: does harm-risk estimation transfer across OCR engines?

Phase 5 ended on a negative: the utility acceptance rule improves repair coverage under a
harm bound in-engine, and does not survive leave-one-engine-out. This stage asks what
broke, and whether anything short of retraining fixes it.

**The hypothesis is not called H4 here.** `docs/sgv1/protocol.md` already binds SGV1-H4 to
"does the V2-V1 image increment persist under held-out-engine evaluation", and that
protocol says prior IDs are never reused. The brief for this stage uses the same letter for
a different claim, so it is recorded as **SGV1-DG1** and the collision is written down
rather than resolved by overwriting a frozen ID.

    SGV1-DG1: a calibrated harm model built on engine-invariant features maintains
    bounded-risk repair coverage on an OCR engine it was never fitted on.

Three things have to be said before any number is read, because each one decides what the
numbers can mean.

**1. Phase 5's headline comparison was confounded.** It put a POOLED in-engine repair
recall (0.701, every engine mixed into one evaluation population) beside PER-ENGINE
held-out folds (0.101-0.328). Those denominators are not the same population: beneficial
prevalence runs 2.3% on docTR against 31.3% on Tesseract, and harm prevalence runs 38.8%
to 77.4%, so most of that drop is arithmetic, not transfer. The control Phase 5 never ran
is an in-engine model evaluated on the *identical* held-out-engine rows. It is run first
here (`--loeo`), and the transfer gap survives it -- larger, on the engine Phase 5 read as
its best case.

**2. No monotone recalibration can move the achievable frontier.** Repair recall at a harm
bound is computed from a ranking; Platt, isotonic and temperature scaling are monotone
maps, so they leave that ranking, and therefore the whole risk-coverage curve, unchanged
except where isotonic collapses distinct scores into ties. Recalibration can only move the
*operating point*. So "can recalibration recover performance?" splits into two questions
with different answers, and this stage reports them separately: it can restore the risk
guarantee, and it cannot restore capability.

**3. A distribution-free risk bound assumes exchangeability, which engine shift breaks by
construction.** `risk/controller.py` says so in its own module docstring. The certified
threshold is therefore not expected to hold on an unseen engine; the measurement is how far
it misses, and against an in-engine control that isolates the shift from the ordinary
finite-sample looseness of the threshold rule itself.

    --shift      experiment 1: where the distributions move, and whether the conditional moves too
    --loeo       experiments 0 and 4: the missing control, then domain-generalization arms
    --transfer   experiments 2 and 5: calibration transfer, certified thresholds, label budget
    --ablation   experiment 3: which feature families carry transfer
    --figures    the three required figures
    --decide     the machine-readable finding

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; neither axis is relaxed anywhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_risk_policy as policy
import sgv1_verifier_pilot as pilot
from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    murphy_decomposition,
)
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.metrics.selective import aurc, risk_coverage_curve
from ocr_risk.risk.controller import CONTROLLERS, select_threshold
from ocr_risk.stats.bootstrap import cluster_bootstrap_indices

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv1/domain_generalization"
DESIGN_MATRIX = OUT / "design_matrix.parquet"
SHIFT_RESULTS = OUT / "engine_shift_analysis.json"
LOEO_RESULTS = OUT / "leave_one_engine_out.json"
TRANSFER_RESULTS = OUT / "calibration_transfer.json"
ABLATION_RESULTS = OUT / "feature_ablation.json"
DECISION = OUT / "research_decision.json"
FIT_RECORD = OUT / "fit_record.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"
INCIDENT_DIR = OUT / "incidents"

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = 0.10
DELTA = 0.10
ECE_BINS = 10
BINNINGS = ("equal_width", "equal_mass")
DIVERGENCE_BINS = 20
DIVERGENCE_FLOOR = 1e-6
BOOTSTRAP_RESAMPLES = 2000
NULL_SPLITS = 20
BUDGETS = (2, 5, 10, 20, 40, 80)
BUDGET_REPEATS = 20
CALIBRATION_METHODS = ("identity", "platt", "isotonic", "temperature")

# The families the base representation is assembled from, keyed by the prefix the feature
# names actually carry. `conf_` and `geom_` are the two blocks that are properties of the
# engine's own output rather than of the proposed edit: the recognizer's confidence on its
# native scale, and the detector's box. Everything else is computed from text, the crop, or
# the frozen candidate provenance.
FAMILIES: dict[str, str] = {
    "text": "text_",
    "conf": "conf_",
    "geom": "geom_",
    "prov": "prov_",
    "edit": "edit_",
    "plaus": "plaus_",
    "vis": "vis_",
    "ctx": "ctx_",
}
ENGINE_OWNED = ("conf", "geom")


class PhaseError(RuntimeError):
    """A freeze, role, split, or selection invariant failed."""


# ------------------------------------------------------------------ the design matrix


@dataclass(slots=True)
class Design:
    """One frozen feature matrix plus the row metadata every stage splits on.

    Built once and cached. Phase 5 rebuilt it inside every fold, which is 4x the work and,
    worse, makes it impossible to assert that two arms saw byte-identical inputs.
    """

    matrix: np.ndarray
    names: tuple[str, ...]
    meta: pd.DataFrame

    def rows(
        self, *, role: str | None = None, engine: str | None = None, exclude: str | None = None
    ) -> np.ndarray:
        mask = np.ones(len(self.meta), dtype=bool)
        if role is not None:
            mask &= self.meta["role"].to_numpy(str) == role
        if engine is not None:
            mask &= self.meta["engine_id"].to_numpy(str) == engine
        if exclude is not None:
            mask &= self.meta["engine_id"].to_numpy(str) != exclude
        return np.flatnonzero(mask)

    @property
    def engines(self) -> list[str]:
        return sorted(self.meta["engine_id"].astype(str).unique())

    @property
    def harmful(self) -> np.ndarray:
        return self.meta["is_harmful"].to_numpy(dtype=bool)

    @property
    def beneficial(self) -> np.ndarray:
        return self.meta["beneficial"].to_numpy(dtype=bool)

    @property
    def documents(self) -> np.ndarray:
        return self.meta["document_id"].to_numpy(str)

    def columns(self, families: tuple[str, ...]) -> list[int]:
        prefixes = tuple(FAMILIES[f] for f in families)
        return [i for i, n in enumerate(self.names) if n.startswith(prefixes)]

    def without(self, families: tuple[str, ...]) -> list[int]:
        prefixes = tuple(FAMILIES[f] for f in families)
        return [i for i, n in enumerate(self.names) if not n.startswith(prefixes)]


META_COLUMNS = [
    "candidate_id",
    "site_id",
    "document_id",
    "engine_id",
    "role",
    "outcome",
    "is_harmful",
    "beneficial",
]


def build_design() -> Design:
    """The Phase-3 candidate-conditioned representation, materialized once.

    Identical construction to Phase 5 -- same evidence mask, same provenance block, same
    extra-feature columns -- so this stage's baseline is Phase 5's model and not a
    reimplementation that could differ for uninteresting reasons.
    """
    pool, columns, bundles, provenance, mask = policy._context()
    verifier = cc.CandidateConditionedVerifier(
        extra_names=columns,
        verifier_id="domain_generalization",
        evidence_config="sgv1_v1",
        model="logistic",
        provenance=provenance,
        embedding=policy._embedding_map(pool, pool.head(0), pool.head(0), columns),
    )
    ids = list(pool["candidate_id"].astype(str))
    matrix = verifier._matrix(pilot._inputs_for("domain_generalization", ids, bundles, {}, mask))
    names = tuple(verifier.feature_names)
    if matrix.shape[1] != len(names):
        raise PhaseError(f"matrix width {matrix.shape[1]} disagrees with {len(names)} names")
    meta = pool[META_COLUMNS].reset_index(drop=True)
    if set(meta["role"]) - {"TRAIN", "CALIBRATION", "DEVELOPMENT"}:
        raise PhaseError("a non-development role reached the design matrix")
    return Design(matrix=matrix, names=names, meta=meta)


def load_design() -> Design:
    if DESIGN_MATRIX.is_file():
        frame = pd.read_parquet(DESIGN_MATRIX)
        names = tuple(c for c in frame.columns if c not in META_COLUMNS)
        return Design(
            matrix=frame[list(names)].to_numpy(dtype=np.float64),
            names=names,
            meta=frame[META_COLUMNS].reset_index(drop=True),
        )
    design = build_design()
    frame = pd.concat(
        [design.meta, pd.DataFrame(design.matrix, columns=list(design.names))], axis=1
    )
    cc._write_parquet_once(DESIGN_MATRIX, frame)
    return design


# ------------------------------------------------------------------ divergences

# PSI and symmetrized KL are the SAME quantity on the same bins:
#   PSI = sum (p - q) log(p/q) = KL(p||q) + KL(q||p).
# They are reported both ways because the brief asks for both, and the identity is asserted
# in the artifact so no reader treats them as two independent pieces of evidence. Wasserstein
# is genuinely independent: it measures how far the mass moved on the value axis, which a
# binned divergence cannot see.


def _bin_edges(reference: np.ndarray, n_bins: int = DIVERGENCE_BINS) -> np.ndarray:
    """Bin edges from the reference population: quantiles, except where they collapse.

    Quantiles rather than equal width, because most of these features are heavily skewed and
    an equal-width grid spends its resolution where there is no data. But quantiles fail
    silently on a low-cardinality feature: for a binary indicator that is 0 on 85% of rows,
    every quantile from 0 to 0.85 is 0.0, the deduplicated edge array is [0, 1], and after
    the infinite tails there is exactly ONE bin -- so the divergence is 0 no matter how far
    apart the two rates are. 38 of the 96 features here are in that regime.

    So a feature with few distinct values is binned on those values instead, using midpoints,
    which gives a binary indicator two real bins and makes a rate difference measurable.
    """
    unique = np.unique(reference)
    if unique.size < 2:
        # Genuinely constant. One bin is correct here: there is nothing that can move.
        return np.array([-np.inf, np.inf])
    if unique.size <= n_bins:
        interior = (unique[:-1] + unique[1:]) / 2.0
        return np.concatenate([[-np.inf], interior, [np.inf]])
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        interior = (unique[:-1] + unique[1:]) / 2.0
        step = max(1, interior.size // n_bins)
        return np.concatenate([[-np.inf], interior[::step], [np.inf]])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(counts.size, 1.0 / counts.size)
    proportions = counts / total
    # Smoothing floor: an empty bin makes log(p/q) infinite, and an infinite divergence
    # reported as a number is worse than a slightly biased finite one. The floor is
    # recorded in the artifact so the bias is visible.
    proportions = np.maximum(proportions, DIVERGENCE_FLOOR)
    return proportions / proportions.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def population_stability_index(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((p - q) * np.log(p / q)))


def divergences(left: np.ndarray, right: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    from scipy.stats import wasserstein_distance

    p, q = _histogram(left, edges), _histogram(right, edges)
    scale = float(np.std(np.concatenate([left, right])))
    return {
        "kl_left_right": kl_divergence(p, q),
        "kl_right_left": kl_divergence(q, p),
        "psi": population_stability_index(p, q),
        # Standardized by the pooled spread, so the number is comparable across features
        # that live on different units. An unstandardized Wasserstein would simply rank
        # features by their variance.
        "wasserstein_standardized": float(wasserstein_distance(left, right) / scale)
        if scale > 1e-12
        else 0.0,
    }


# ------------------------------------------------------------------ endpoints

# The primary endpoint has exactly one implementation in this repository and it is Phase
# 5's. Importing it rather than restating it is what keeps two functions called "repair
# recall" from ever reporting two different numbers; `tests/leakage` asserts the identity.
achievable_repair_recall = policy.repair_recall_at_bounded_harm


def frontier(score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray) -> dict[str, Any]:
    """What the ranking could deliver if the operating point were chosen with hindsight.

    The threshold here is read off the evaluation labels, so this is an achievable frontier
    and never a deployment number. It is the right instrument for asking whether the
    *ranking* transferred, and the wrong one for asking whether the risk bound held -- that
    is `deployed_point`, below, and the two are never compared without their harm rates.
    """
    result = achievable_repair_recall(score, harmful, beneficial)
    curve = risk_coverage_curve(score, harmful)
    result["aurc"] = float(aurc(curve))
    result["auc_safe"] = (
        float(roc_auc(score, (~harmful).astype(float)))
        if harmful.any() and (~harmful).any()
        else float("nan")
    )
    result["auc_beneficial"] = (
        float(roc_auc(score, beneficial.astype(float)))
        if beneficial.any() and (~beneficial).any()
        else float("nan")
    )
    return result


def deployed_point(
    tau: float, score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, epsilon: float
) -> dict[str, Any]:
    """What a threshold frozen before evaluation actually delivers, and what it costs."""
    from ocr_risk.metrics.precision import attainable_epsilon

    accepted = score >= tau
    n = int(accepted.sum())
    if n == 0:
        return {
            "n_accepted": 0,
            "coverage": 0.0,
            "repair_recall": 0.0,
            "realized_harm_rate": 0.0,
            "realized_harm_ucb": 1.0,
            "bound_violated": False,
            "harm_rate_excess": -epsilon,
        }
    realized = float(harmful[accepted].mean())
    return {
        "n_accepted": n,
        "coverage": float(n / score.size),
        "repair_recall": float(beneficial[accepted].sum() / max(beneficial.sum(), 1)),
        "realized_harm_rate": realized,
        "realized_harm_ucb": float(attainable_epsilon(n, DELTA, int(harmful[accepted].sum()))),
        "bound_violated": bool(realized > epsilon),
        "harm_rate_excess": float(realized - epsilon),
    }


def _repair_recall_statistic(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, epsilon: float
) -> Any:
    """Closure over row indices for the cluster bootstrap.

    Repair recall is not a row mean: the ranking and the harm constraint have to be rebuilt
    inside every resample, which is why the whole vector is captured rather than a per-row
    contribution.
    """

    def statistic(index: np.ndarray) -> float:
        if index.size == 0 or beneficial[index].sum() == 0:
            return float("nan")
        order = np.argsort(-score[index], kind="stable")
        rows = index[order]
        rate = np.cumsum(harmful[rows]) / np.arange(1, rows.size + 1)
        feasible = np.flatnonzero(rate <= epsilon)
        if feasible.size == 0:
            return 0.0
        return float(np.cumsum(beneficial[rows])[feasible[-1]] / beneficial[index].sum())

    return statistic


def clustered_interval(
    score: np.ndarray,
    harmful: np.ndarray,
    beneficial: np.ndarray,
    documents: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    result = cluster_bootstrap_indices(
        list(documents),
        _repair_recall_statistic(score, harmful, beneficial, epsilon),
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_documents": result.n_clusters,
    }


# ------------------------------------------------------------------ models and arms


@dataclass(slots=True)
class HarmModel:
    """A fitted scaler and harm classifier, plus the columns it was fitted on."""

    scaler: Any
    model: Any
    columns: tuple[int, ...]

    def safety(self, design: Design, index: np.ndarray) -> np.ndarray:
        """P(not harmful). Higher is safer, matching the risk controller's accept rule."""
        block = design.matrix[np.ix_(index, list(self.columns))]
        scaled = self.scaler.transform(block)
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.ones(len(index))
        return 1.0 - self.model.predict_proba(scaled)[:, classes.index(1)]


def fit_harm_model(
    design: Design,
    fit_index: np.ndarray,
    columns: list[int],
    *,
    sample_weight: np.ndarray | None = None,
    fit_transform: Any = None,
) -> HarmModel:
    """Fit on ``fit_index``; ``fit_transform`` moves ONLY the fit rows.

    The asymmetry is the point of an alignment arm: CORAL maps the source distribution onto
    the target's, so applying it again at scoring time would move the target away from
    itself. Scoring is therefore always in raw target coordinates.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    block = design.matrix[np.ix_(fit_index, columns)]
    if fit_transform is not None:
        block = fit_transform(block)
    scaler = StandardScaler().fit(block)
    model = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
    ).fit(
        scaler.transform(block), design.harmful[fit_index].astype(int), sample_weight=sample_weight
    )
    return HarmModel(scaler=scaler, model=model, columns=tuple(columns))


def engine_balanced_weights(design: Design, fit_index: np.ndarray) -> np.ndarray:
    """One unit of influence per engine, not per row.

    Without this the pooled fit is 40% EasyOCR and 20% PaddleOCR simply because those
    engines proposed different numbers of candidates, so "the pooled model" is really a
    model of whichever engine was most verbose.
    """
    engines = design.meta["engine_id"].to_numpy(str)[fit_index]
    counts = pd.Series(engines).value_counts()
    return np.array([len(fit_index) / (len(counts) * counts[e]) for e in engines])


def coral_transform(source: np.ndarray, target: np.ndarray) -> Any:
    """Second-order alignment: recolor the source covariance to the target's.

    Uses only *unlabeled* target features, so it is admissible for an engine whose labels do
    not exist -- which is the deployment situation this stage is about. It moves the fit
    rows, not the target rows, so the target is never transformed by a statistic estimated
    from itself.

    **Where the target statistics come from is a split decision, not a detail.** Estimating
    them on the evaluation rows is transductive: it computes a covariance over the frame
    being scored, which `.claude/rules/experiment-leakage.md` forbids for fitted
    transformers. The deployable source is the held-out engine's TRAIN-role documents --
    other pages that engine has read, no labels touched, document partition intact. Both are
    run and reported separately, because the gap between them is the size of the transductive
    advantage and hiding it inside one arm would overstate what an operator can have.
    """
    mean_s, mean_t = source.mean(0), target.mean(0)
    ridge = np.eye(source.shape[1])
    cov_s = np.cov(source - mean_s, rowvar=False) + ridge
    cov_t = np.cov(target - mean_t, rowvar=False) + ridge

    def power(matrix: np.ndarray, exponent: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        return vectors @ np.diag(np.clip(values, 1e-8, None) ** exponent) @ vectors.T

    whiten = power(cov_s, -0.5) @ power(cov_t, 0.5)

    def apply(block: np.ndarray) -> np.ndarray:
        # Only the fit population is moved; target rows are already in target coordinates.
        # Distinguishing them by shape would be fragile, so the map is applied to the fit
        # block at fit time and is the identity afterwards.
        return (block - mean_s) @ whiten + mean_t

    return apply


def nested_invariant_families(design: Design, held_out: str) -> tuple[str, ...]:
    """Choose a feature-family subset WITHOUT looking at the held-out engine.

    A subset picked by held-out performance is not a domain-generalization method, it is
    selection on the test set wearing one. The choice is made by an inner leave-one-engine-out
    over the three fit engines only; the held-out engine contributes nothing to it, which
    `tests/leakage` asserts by re-deriving the selection from the recorded inner scores.
    """
    train_engines = [e for e in design.engines if e != held_out]
    candidates: list[tuple[str, ...]] = [
        tuple(FAMILIES),
        tuple(f for f in FAMILIES if f not in ENGINE_OWNED),
        tuple(f for f in FAMILIES if f != "conf"),
        ("vis", "edit", "plaus"),
        ("vis", "edit", "plaus", "ctx", "text"),
    ]
    scores: dict[tuple[str, ...], float] = {}
    for families in candidates:
        columns = design.columns(families)
        inner: list[float] = []
        for inner_held in train_engines:
            fit_index = np.flatnonzero(
                (design.meta["role"].to_numpy(str) == "TRAIN")
                & np.isin(
                    design.meta["engine_id"].to_numpy(str),
                    [e for e in train_engines if e != inner_held],
                )
            )
            eval_index = design.rows(role="CALIBRATION", engine=inner_held)
            if fit_index.size == 0 or eval_index.size == 0:
                continue
            model = fit_harm_model(design, fit_index, columns)
            score = model.safety(design, eval_index)
            inner.append(
                achievable_repair_recall(
                    score, design.harmful[eval_index], design.beneficial[eval_index]
                )[f"epsilon_{int(PRIMARY_EPSILON * 100)}"]["repair_recall"]
            )
        scores[families] = float(np.mean(inner)) if inner else float("nan")
    best = max(scores, key=lambda k: scores[k])
    return best


def loeo_arms(design: Design, held_out: str) -> tuple[dict[str, HarmModel], tuple[str, ...]]:
    """Every arm evaluated on one held-out engine, and the two controls that bound it."""
    train_role = design.meta["role"].to_numpy(str) == "TRAIN"
    engine_of = design.meta["engine_id"].to_numpy(str)
    fit_loeo = np.flatnonzero(train_role & (engine_of != held_out))
    fit_in = np.flatnonzero(train_role & (engine_of == held_out))
    fit_all = np.flatnonzero(train_role)
    eval_index = design.rows(role="DEVELOPMENT", engine=held_out)
    every = design.columns(tuple(FAMILIES))

    arms: dict[str, HarmModel] = {
        "in_engine_ceiling": fit_harm_model(design, fit_in, every),
        "pooled_engine_seen": fit_harm_model(design, fit_all, every),
        "loeo_pooled": fit_harm_model(design, fit_loeo, every),
        "loeo_engine_balanced": fit_harm_model(
            design, fit_loeo, every, sample_weight=engine_balanced_weights(design, fit_loeo)
        ),
        # Inductive: target statistics from OTHER pages the held-out engine read, never the
        # pages being scored. This is the arm an operator could actually deploy.
        "loeo_coral": fit_harm_model(
            design,
            fit_loeo,
            every,
            fit_transform=coral_transform(
                design.matrix[np.ix_(fit_loeo, every)],
                design.matrix[np.ix_(design.rows(role="TRAIN", engine=held_out), every)],
            ),
        ),
        # Transductive: target statistics from the evaluation rows themselves. Reported as a
        # diagnostic upper reference on what second-order alignment could buy, never as a
        # deployable arm.
        "loeo_coral_transductive": fit_harm_model(
            design,
            fit_loeo,
            every,
            fit_transform=coral_transform(
                design.matrix[np.ix_(fit_loeo, every)], design.matrix[np.ix_(eval_index, every)]
            ),
        ),
    }
    selected = nested_invariant_families(design, held_out)
    arms["loeo_invariant_subset"] = fit_harm_model(design, fit_loeo, design.columns(selected))
    return arms, selected


# ------------------------------------------------------------------ experiment 1: shift


def _split_half_documents(documents: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(documents.tolist())))
    order = np.random.default_rng(seed).permutation(unique.size)
    left = set(unique[order[: unique.size // 2]].tolist())
    mask = np.isin(documents, list(left))
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def run_shift() -> int:
    """Experiment 1. Where do the distributions move, and does the conditional move too?

    A divergence on its own says nothing: the same statistic computed between two disjoint
    halves of ONE engine is not zero, and without that floor a PSI of 0.3 is uninterpretable.
    Every measure here is reported against its own within-engine null.
    """
    started = time.monotonic()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    design = load_design()
    engines = design.engines
    train = design.rows(role="TRAIN")
    engine_of = design.meta["engine_id"].to_numpy(str)
    documents = design.documents

    per_feature: dict[str, Any] = {}
    for column, name in enumerate(design.names):
        reference = design.matrix[train, column]
        edges = _bin_edges(reference)
        cell: dict[str, Any] = {"engines": {}, "within_engine_null": {}}
        for engine in engines:
            here = np.flatnonzero(engine_of == engine)
            rest = np.flatnonzero(engine_of != engine)
            here = np.intersect1d(here, train, assume_unique=False)
            rest = np.intersect1d(rest, train, assume_unique=False)
            cell["engines"][engine] = divergences(
                design.matrix[here, column], design.matrix[rest, column], edges
            )
        nulls: list[float] = []
        wasserstein_nulls: list[float] = []
        for engine in engines:
            here = np.intersect1d(np.flatnonzero(engine_of == engine), train)
            for seed in range(NULL_SPLITS // len(engines)):
                left, right = _split_half_documents(documents[here], pilot.BOOTSTRAP_SEED + seed)
                measured = divergences(
                    design.matrix[here[left], column], design.matrix[here[right], column], edges
                )
                nulls.append(measured["psi"])
                wasserstein_nulls.append(measured["wasserstein_standardized"])
        cell["within_engine_null"] = {
            "psi_mean": float(np.mean(nulls)),
            "psi_p95": float(np.quantile(nulls, 0.95)),
            "wasserstein_mean": float(np.mean(wasserstein_nulls)),
            "wasserstein_p95": float(np.quantile(wasserstein_nulls, 0.95)),
            "n_splits": len(nulls),
        }
        worst = max(cell["engines"].values(), key=lambda v: v["psi"])
        cell["max_psi"] = worst["psi"]
        # A ratio against a zero floor is not a large number, it is an undefined one. A
        # constant feature has nothing to divide and says so.
        floor = cell["within_engine_null"]["psi_p95"]
        cell["psi_over_null_p95"] = (
            worst["psi"] / floor
            if floor > 0
            else (float("nan") if worst["psi"] == 0.0 else float("inf"))
        )
        cell["is_constant"] = bool(np.unique(design.matrix[train, column]).size < 2)
        cell["family"] = next(
            (f for f, prefix in FAMILIES.items() if name.startswith(prefix)), "other"
        )
        per_feature[name] = cell

    # --- how separable are the engines at all: a domain classifier ----------------------
    scaler = StandardScaler().fit(design.matrix[train])
    scaled = scaler.transform(design.matrix)
    evaluation = design.rows(role="DEVELOPMENT")
    domain_classifier = {}
    for engine in engines:
        model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
        ).fit(scaled[train], (engine_of[train] == engine).astype(int))
        domain_classifier[engine] = float(
            roc_auc(
                model.predict_proba(scaled[evaluation])[:, 1],
                (engine_of[evaluation] == engine).astype(float),
            )
        )

    # --- does the CONDITIONAL move: coefficient geometry against a document null ---------
    def coefficients(index: np.ndarray) -> np.ndarray:
        model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
        ).fit(scaled[index], design.harmful[index].astype(int))
        return np.asarray(model.coef_).ravel()

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    per_engine_coefficients = {
        engine: coefficients(np.intersect1d(np.flatnonzero(engine_of == engine), train))
        for engine in engines
    }
    between = {
        f"{a}|{b}": cosine(per_engine_coefficients[a], per_engine_coefficients[b])
        for i, a in enumerate(engines)
        for b in engines[i + 1 :]
    }
    within: dict[str, dict[str, float]] = {}
    for engine in engines:
        here = np.intersect1d(np.flatnonzero(engine_of == engine), train)
        values = []
        for seed in range(5):
            left, right = _split_half_documents(documents[here], pilot.BOOTSTRAP_SEED + seed)
            values.append(cosine(coefficients(here[left]), coefficients(here[right])))
        within[engine] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "n_splits": len(values),
        }

    # --- the sharpest single instrument: sign of the univariate association -------------
    associations: dict[str, dict[str, float]] = {}
    for column, name in enumerate(design.names):
        row = {}
        for engine in engines:
            here = np.intersect1d(np.flatnonzero(engine_of == engine), train)
            values = scaled[here, column]
            if np.std(values) < 1e-12:
                row[engine] = 0.0
                continue
            row[engine] = float(np.corrcoef(values, design.harmful[here].astype(float))[0, 1])
        associations[name] = row
    flipped = {
        name: row
        for name, row in associations.items()
        if max(row.values()) > 0.05 and min(row.values()) < -0.05
    }

    # --- the sharpest statement about C: does the confidence channel invert? -------------
    confidence_columns = design.columns(("conf",))
    confidence_only = {}
    for engine in engines:
        fit_index = np.flatnonzero(
            (engine_of != engine) & (design.meta["role"].to_numpy(str) == "TRAIN")
        )
        eval_index = design.rows(role="DEVELOPMENT", engine=engine)
        model = fit_harm_model(design, fit_index, confidence_columns)
        confidence_only[engine] = float(
            roc_auc(
                model.safety(design, eval_index),
                (~design.harmful[eval_index]).astype(float),
            )
        )

    # How many features the naive quantile binning would have silently reported as unmoved.
    collapsed = 0
    for column in range(design.matrix.shape[1]):
        reference = design.matrix[train, column]
        naive = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, DIVERGENCE_BINS + 1)))
        if naive.size < 3 and np.unique(reference).size >= 2:
            collapsed += 1

    ranked = sorted(per_feature, key=lambda n: -per_feature[n]["max_psi"])
    by_family: dict[str, float] = {}
    for family in FAMILIES:
        members = [n for n in design.names if per_feature[n]["family"] == family]
        by_family[family] = float(np.mean([per_feature[n]["max_psi"] for n in members]))

    cc._write_json_once(
        SHIFT_RESULTS,
        {
            "schema_version": "sgv1-domain-generalization-shift-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV1-DG1",
            "measured_on": "TRAIN documents only, all four engines",
            "binning": {
                "n_bins": DIVERGENCE_BINS,
                "scheme": "quantile edges from the pooled TRAIN distribution of each feature",
                "smoothing_floor": DIVERGENCE_FLOOR,
            },
            "psi_kl_identity": {
                "claim": ("PSI = KL(p||q) + KL(q||p) on the same bins; one measure, not two"),
                "max_abs_residual": float(
                    max(
                        abs(v["psi"] - v["kl_left_right"] - v["kl_right_left"])
                        for cell in per_feature.values()
                        for v in cell["engines"].values()
                    )
                ),
            },
            "per_feature": per_feature,
            "features_ranked_by_max_psi": ranked[:25],
            "mean_max_psi_by_family": by_family,
            "domain_classifier_auc": domain_classifier,
            "domain_classifier_note": (
                "One engine against the rest, fitted on TRAIN and scored on DEVELOPMENT. "
                "This is a magnitude for covariate shift only: it says the engines are "
                "distinguishable from the features, not that the harm relationship differs."
            ),
            "conditional_shift": {
                "question": "does P(harm | features) itself differ by engine, or only P(features)?",
                "between_engine_coefficient_cosine": between,
                "between_engine_mean": float(np.mean(list(between.values()))),
                "within_engine_split_half_null": within,
                "every_within_engine_mean_exceeds_every_between_engine_pair": bool(
                    min(v["mean"] for v in within.values()) > max(between.values())
                ),
                "every_within_engine_minimum_exceeds_every_between_engine_pair": bool(
                    min(v["min"] for v in within.values()) > max(between.values())
                ),
                "reading": (
                    "The null resamples DOCUMENTS inside one engine, so it carries the same "
                    "sampling noise as the between-engine comparison and differs only in "
                    "whether the engine changed. The separation holds on means -- every "
                    "within-engine mean is above every between-engine pair -- and not on the "
                    "worst single split, where one engine's noisiest half-split dips below "
                    "the closest engine pair. The weaker statement is the one that is true."
                ),
            },
            "confidence_only_transfer": {
                "question": (
                    "Is the confidence channel merely rescaled between engines, or does its "
                    "relationship to harm reverse?"
                ),
                "protocol": (
                    "Fit a harm model on the confidence family alone over the three fit "
                    "engines' TRAIN rows; score the held-out engine's DEVELOPMENT rows. "
                    "Reported as AUC for ranking NOT-harmful, so 0.5 is chance and below 0.5 "
                    "is an inverted relationship."
                ),
                "auc_safe_by_held_out_engine": confidence_only,
                "n_confidence_features": len(confidence_columns),
                "inverted_on": [e for e, v in confidence_only.items() if v < 0.5],
            },
            "quantile_binning_collapse": {
                "n_features_a_naive_quantile_grid_would_report_as_unmoved": collapsed,
                "why": (
                    "On a low-cardinality feature the quantile edges deduplicate to a single "
                    "interior point, leaving one bin, and one bin makes every divergence zero "
                    "no matter how far the rates differ. Those features are binned on their "
                    "own values instead."
                ),
            },
            "univariate_sign_flips": flipped,
            "n_features_with_sign_flip": len(flipped),
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        f"shift: {len(per_feature)} features, {len(flipped)} sign flips"
        f" -> {cc._relative(SHIFT_RESULTS)}"
    )
    return 0


# ------------------------------------------------------------------ experiments 0 and 4


def run_loeo() -> int:
    """The control Phase 5 never ran, then the domain-generalization arms.

    Every arm in a fold is scored on the SAME evaluation rows -- the held-out engine's
    DEVELOPMENT documents -- so nothing in this table can be explained by a change of
    population. That is the whole reason the in-engine ceiling is here: it is the only
    number that separates "the transfer failed" from "this engine is harder".
    """
    started = time.monotonic()
    design = load_design()
    folds: dict[str, Any] = {}
    for held_out in design.engines:
        eval_index = design.rows(role="DEVELOPMENT", engine=held_out)
        fit_loeo = np.flatnonzero(
            (design.meta["role"].to_numpy(str) == "TRAIN")
            & (design.meta["engine_id"].to_numpy(str) != held_out)
        )
        if set(design.documents[fit_loeo]) & set(design.documents[eval_index]):
            raise PhaseError(f"{held_out}: fit and evaluation documents overlap")
        if held_out in set(design.meta["engine_id"].to_numpy(str)[fit_loeo]):
            raise PhaseError(f"{held_out}: the held-out engine reached the fit rows")

        harmful = design.harmful[eval_index]
        beneficial = design.beneficial[eval_index]
        documents = design.documents[eval_index]
        arms, selected = loeo_arms(design, held_out)
        scores = {name: model.safety(design, eval_index) for name, model in arms.items()}

        results: dict[str, Any] = {}
        for name, score in scores.items():
            results[name] = frontier(score, harmful, beneficial)
            results[name]["interval_primary_epsilon"] = clustered_interval(
                score, harmful, beneficial, documents, PRIMARY_EPSILON
            )
        baseline = results["loeo_pooled"][f"epsilon_{int(PRIMARY_EPSILON * 100)}"]["repair_recall"]
        ceiling = results["in_engine_ceiling"][f"epsilon_{int(PRIMARY_EPSILON * 100)}"][
            "repair_recall"
        ]
        folds[held_out] = {
            "held_out_engine": held_out,
            "train_engines": [e for e in design.engines if e != held_out],
            "evaluation_rows": int(eval_index.size),
            "evaluation_documents": len(set(documents.tolist())),
            "beneficial": int(beneficial.sum()),
            "harmful": int(harmful.sum()),
            "beneficial_prevalence": float(beneficial.mean()),
            "harm_prevalence": float(harmful.mean()),
            "arms": results,
            "selected_invariant_families": list(selected),
            "transfer_gap_primary_epsilon": float(ceiling - baseline),
            "closed_fraction": {
                name: float(
                    (
                        results[name][f"epsilon_{int(PRIMARY_EPSILON * 100)}"]["repair_recall"]
                        - baseline
                    )
                    / (ceiling - baseline)
                )
                if ceiling > baseline
                else float("nan")
                for name in results
            },
        }
        print(
            f"  fold {held_out:11s} n={eval_index.size:5d} ben={int(beneficial.sum()):4d} "
            f"loeo={baseline:.3f} in-engine={ceiling:.3f} gap={ceiling - baseline:+.3f}"
        )

    # --- the confound, stated in numbers -------------------------------------------------
    pooled_eval = design.rows(role="DEVELOPMENT")
    pooled_model = fit_harm_model(
        design, design.rows(role="TRAIN"), design.columns(tuple(FAMILIES))
    )
    pooled_raw = pooled_model.safety(design, pooled_eval)
    pooled_frontier = frontier(
        pooled_raw, design.harmful[pooled_eval], design.beneficial[pooled_eval]
    )
    # Provenance check, not a result: Phase 5's pooled arm was the isotonic-calibrated score.
    # Reproducing its three published numbers exactly is what licenses every comparison in
    # this stage to be read as a comparison against Phase 5 rather than against a rebuild
    # that happens to resemble it.
    pooled_cal = design.rows(role="CALIBRATION")
    calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    calibrator.fit(
        pooled_model.safety(design, pooled_cal), (~design.harmful[pooled_cal]).astype(float)
    )
    pooled_calibrated = frontier(
        calibrator.transform(pooled_raw),
        design.harmful[pooled_eval],
        design.beneficial[pooled_eval],
    )
    predecessor = {
        "epsilon_5": 0.5581151832460733,
        "epsilon_10": 0.7010471204188482,
        "epsilon_20": 0.8612565445026178,
    }

    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    cc._write_json_once(
        LOEO_RESULTS,
        {
            "schema_version": "sgv1-domain-generalization-loeo-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV1-DG1",
            "protocol": (
                "Leave one engine out AND leave documents out, simultaneously and always. "
                "Fit on TRAIN documents of the remaining engines; evaluate on DEVELOPMENT "
                "documents of the held-out engine. Every arm in a fold sees identical "
                "evaluation rows."
            ),
            "endpoint": (
                "Achievable repair recall at a harm bound: the operating point is read off "
                "the evaluation labels, so this measures whether the RANKING transferred. "
                "Whether the risk bound holds is a different question and lives in "
                "calibration_transfer.json."
            ),
            "folds": folds,
            "fold_means": {
                name: float(np.mean([folds[e]["arms"][name][key]["repair_recall"] for e in folds]))
                for name in folds[design.engines[0]]["arms"]
            },
            "confound_in_the_predecessor_comparison": {
                "what_phase_5_compared": (
                    "a POOLED in-engine repair recall against PER-ENGINE held-out folds, "
                    "which are different evaluation populations"
                ),
                "pooled_development_repair_recall": {
                    f"epsilon_{int(e * 100)}": pooled_frontier[f"epsilon_{int(e * 100)}"][
                        "repair_recall"
                    ]
                    for e in EPSILONS
                },
                "per_engine_prevalence": {
                    engine: {
                        "beneficial_prevalence": folds[engine]["beneficial_prevalence"],
                        "harm_prevalence": folds[engine]["harm_prevalence"],
                    }
                    for engine in folds
                },
                "pooled_development_repair_recall_isotonic_calibrated": {
                    name: pooled_calibrated[name]["repair_recall"] for name in predecessor
                },
                "reproduces_predecessor_exactly": {
                    name: bool(abs(pooled_calibrated[name]["repair_recall"] - value) < 1e-12)
                    for name, value in predecessor.items()
                },
                "predecessor_values": predecessor,
                "raw_versus_calibrated_note": (
                    "The raw and isotonic-calibrated pooled numbers differ in the fourth "
                    "decimal because isotonic collapses 11,998 distinct scores to 96, and "
                    "ties move the frontier. That is the same tie effect measured directly "
                    "in calibration_transfer.json, not a different model."
                ),
                "gap_decomposition": {
                    "definition": (
                        "Phase 5's quoted drop is pooled_reference -> loeo_pooled(e). It "
                        "splits exactly into a COMPOSITION step (pooled_reference -> "
                        "in_engine_ceiling(e): same model class, different evaluation "
                        "population) and a TRANSFER step (in_engine_ceiling(e) -> "
                        "loeo_pooled(e): same population, engine now unseen)."
                    ),
                    "pooled_reference": pooled_frontier[key]["repair_recall"],
                    "by_engine": {
                        engine: {
                            "composition": float(
                                folds[engine]["arms"]["in_engine_ceiling"][key]["repair_recall"]
                                - pooled_frontier[key]["repair_recall"]
                            ),
                            "transfer": float(
                                folds[engine]["arms"]["loeo_pooled"][key]["repair_recall"]
                                - folds[engine]["arms"]["in_engine_ceiling"][key]["repair_recall"]
                            ),
                            "total": float(
                                folds[engine]["arms"]["loeo_pooled"][key]["repair_recall"]
                                - pooled_frontier[key]["repair_recall"]
                            ),
                        }
                        for engine in folds
                    },
                    "engines_where_composition_works_against_the_drop": [
                        engine
                        for engine in folds
                        if folds[engine]["arms"]["in_engine_ceiling"][key]["repair_recall"]
                        > pooled_frontier[key]["repair_recall"]
                    ],
                    "reading": (
                        "Composition is not a uniform inflation of the drop. On two engines "
                        "the single-engine population is HARDER than the pool and composition "
                        "adds to the drop; on the other two it is EASIER and composition works "
                        "against it, so the confounded comparison understated the transfer "
                        "loss there. Transfer is negative on all four."
                    ),
                },
                "correct_control": (
                    "in_engine_ceiling, fitted on the held-out engine, scored on identical rows"
                ),
                "gap_survives_the_control": bool(
                    all(f["transfer_gap_primary_epsilon"] > 0 for f in folds.values())
                ),
                "note": (
                    "The direction of Phase 5's conclusion survives; the magnitude it quoted "
                    "was a mixture of transfer loss and population composition. On Tesseract "
                    "the properly controlled gap is larger than the confounded one."
                ),
            },
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"loeo: {len(folds)} folds -> {cc._relative(LOEO_RESULTS)}")
    return 0


# ------------------------------------------------------------------ experiments 2 and 5


def _calibration_quality(probabilities: np.ndarray, safe: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "brier": float(brier_score(probabilities, safe)),
        "distinct_values": int(np.unique(probabilities).size),
    }
    for binning in BINNINGS:
        ece, mce = expected_calibration_error(probabilities, safe, ECE_BINS, binning)
        out[f"ece_{binning}"] = float(ece)
        out[f"mce_{binning}"] = float(mce)
    calibration, refinement, uncertainty = murphy_decomposition(
        probabilities, safe, ECE_BINS, "equal_mass"
    )
    out["murphy"] = {
        "calibration": float(calibration),
        "refinement": float(refinement),
        "uncertainty": float(uncertainty),
    }
    return out


DESCRIPTOR_FEATURES = ("conf_normalized", "text_len_original", "geom_width", "geom_height")


def engine_descriptor_block(design: Design) -> np.ndarray:
    """A low-dimensional, LABEL-FREE summary of how each engine behaves, broadcast per row.

    Engine identity cannot condition a model on an engine it has never seen -- the one-hot
    for a new engine is a column of zeros in training and a column of ones at evaluation,
    which is not a conditioning variable but an extrapolation. Descriptors are the version
    of "engine-aware" that is defined out of sample: mean and spread of the engine's own
    unlabeled output, so a new engine has them on day one, before a single label exists.

    Estimated on that engine's TRAIN rows only, which keeps the statistic off the
    evaluation documents even though it needs no labels.
    """
    columns = [design.names.index(n) for n in DESCRIPTOR_FEATURES]
    engine_of = design.meta["engine_id"].to_numpy(str)
    train = design.meta["role"].to_numpy(str) == "TRAIN"
    block = np.zeros((len(design.meta), 2 * len(columns)), dtype=np.float64)
    for engine in design.engines:
        reference = design.matrix[np.ix_(np.flatnonzero(train & (engine_of == engine)), columns)]
        block[engine_of == engine] = np.concatenate([reference.mean(0), reference.std(0)])
    return block


def run_transfer() -> int:
    """Experiments 2 and 5: what recalibration can and cannot repair."""
    started = time.monotonic()
    design = load_design()
    every = design.columns(tuple(FAMILIES))
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in design.engines:
        fit_index = np.flatnonzero(
            (design.meta["role"].to_numpy(str) == "TRAIN")
            & (design.meta["engine_id"].to_numpy(str) != held_out)
        )
        source_cal = np.flatnonzero(
            (design.meta["role"].to_numpy(str) == "CALIBRATION")
            & (design.meta["engine_id"].to_numpy(str) != held_out)
        )
        target_cal = design.rows(role="CALIBRATION", engine=held_out)
        eval_index = design.rows(role="DEVELOPMENT", engine=held_out)
        for name, block in (("source_cal", source_cal), ("target_cal", target_cal)):
            if set(design.documents[block]) & set(design.documents[eval_index]):
                raise PhaseError(f"{held_out}: {name} shares documents with evaluation")

        model = fit_harm_model(design, fit_index, every)
        raw = {
            "eval": model.safety(design, eval_index),
            "source": model.safety(design, source_cal),
            "target": model.safety(design, target_cal),
        }
        harmful = design.harmful[eval_index]
        beneficial = design.beneficial[eval_index]
        safe_eval = (~harmful).astype(float)
        raw_frontier = frontier(raw["eval"], harmful, beneficial)

        methods: dict[str, Any] = {
            "uncalibrated": {
                **_calibration_quality(raw["eval"], safe_eval),
                "achievable_repair_recall": raw_frontier[key]["repair_recall"],
                "auc_safe": raw_frontier["auc_safe"],
            }
        }
        deployed: dict[str, Any] = {}
        for method in CALIBRATION_METHODS:
            for source_name, cal_index, cal_raw in (
                ("source_engines", source_cal, raw["source"]),
                ("target_engine", target_cal, raw["target"]),
            ):
                calibrator = build_calibrator(method)
                calibrator.fit(cal_raw, (~design.harmful[cal_index]).astype(float))
                probabilities = calibrator.transform(raw["eval"])
                on_calibration = calibrator.transform(cal_raw)
                measured = frontier(probabilities, harmful, beneficial)
                label = f"{method}__{source_name}"
                methods[label] = {
                    **_calibration_quality(probabilities, safe_eval),
                    "achievable_repair_recall": measured[key]["repair_recall"],
                    "auc_safe": measured["auc_safe"],
                    "frontier_moved_by_calibration": float(
                        measured[key]["repair_recall"] - raw_frontier[key]["repair_recall"]
                    ),
                }
                for controller in CONTROLLERS:
                    decision = select_threshold(
                        on_calibration,
                        design.harmful[cal_index],
                        PRIMARY_EPSILON,
                        delta=DELTA,
                        controller=controller,
                    )
                    deployed[f"{label}__{controller}"] = {
                        "threshold": decision.as_dict(),
                        "on_held_out_engine": deployed_point(
                            decision.tau, probabilities, harmful, beneficial, PRIMARY_EPSILON
                        ),
                    }

        # --- the control that separates engine shift from an ordinary loose threshold ----
        in_engine_model = fit_harm_model(design, design.rows(role="TRAIN", engine=held_out), every)
        in_engine_cal = in_engine_model.safety(design, target_cal)
        in_engine_eval = in_engine_model.safety(design, eval_index)
        in_engine_deployed = {}
        for controller in CONTROLLERS:
            calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
            calibrator.fit(in_engine_cal, (~design.harmful[target_cal]).astype(float))
            decision = select_threshold(
                calibrator.transform(in_engine_cal),
                design.harmful[target_cal],
                PRIMARY_EPSILON,
                delta=DELTA,
                controller=controller,
            )
            in_engine_deployed[controller] = {
                "threshold": decision.as_dict(),
                "on_held_out_engine": deployed_point(
                    decision.tau,
                    calibrator.transform(in_engine_eval),
                    harmful,
                    beneficial,
                    PRIMARY_EPSILON,
                ),
            }

        folds[held_out] = {
            "held_out_engine": held_out,
            "evaluation_rows": int(eval_index.size),
            "source_calibration_rows": int(source_cal.size),
            "target_calibration_rows": int(target_cal.size),
            "target_calibration_documents": len(set(design.documents[target_cal].tolist())),
            "calibration_quality": methods,
            "deployed_operating_points": deployed,
            "in_engine_threshold_control": in_engine_deployed,
            "label_budget": _label_budget_curve(design, model, held_out, target_cal, eval_index),
            "engine_aware": _engine_aware(design, held_out, every),
        }
        print(
            f"  fold {held_out:11s} calibrated {len(methods)} arms,"
            f" {len(deployed)} operating points"
        )

    cc._write_json_once(
        TRANSFER_RESULTS,
        {
            "schema_version": "sgv1-domain-generalization-transfer-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV1-DG1",
            "monotone_invariance": {
                "claim": (
                    "Platt, temperature and isotonic scaling are monotone non-decreasing, so "
                    "they cannot reorder scores and cannot move the achievable risk-coverage "
                    "frontier. Isotonic is the one exception and only through ties: it maps "
                    "distinct scores onto a step function, and rows that were ordered become "
                    "equal."
                ),
                "max_abs_frontier_move_by_method": {
                    method: float(
                        max(
                            abs(
                                folds[engine]["calibration_quality"][arm][
                                    "frontier_moved_by_calibration"
                                ]
                            )
                            for engine in folds
                            for arm in folds[engine]["calibration_quality"]
                            if arm.startswith(f"{method}__")
                        )
                    )
                    for method in CALIBRATION_METHODS
                },
                "consequence": (
                    "'Can recalibration recover performance?' has two answers. It can restore "
                    "the operating point and therefore the risk guarantee. It cannot restore "
                    "capability, because capability is the ranking and recalibration does not "
                    "touch it."
                ),
            },
            "exchangeability_note": (
                "A distribution-free risk bound certifies P(risk <= epsilon) >= 1 - delta on "
                "EXCHANGEABLE calibration and test data. A held-out engine is not exchangeable "
                "with the fit engines by construction, so the certificate does not apply and is "
                "reported here as a measurement of how far it misses, never as a guarantee."
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"transfer: {len(folds)} folds -> {cc._relative(TRANSFER_RESULTS)}")
    return 0


def _label_budget_curve(
    design: Design,
    model: HarmModel,
    held_out: str,
    target_cal: np.ndarray,
    eval_index: np.ndarray,
) -> dict[str, Any]:
    """How many labelled documents from the new engine buy back the risk guarantee?

    This is the deployable form of "engine-adaptive calibration": the harm model is frozen
    -- the new engine never enters fitting -- and only the calibrator and the threshold are
    re-estimated, on documents disjoint from evaluation. It answers the brief's question
    about avoiding a retrain with a price rather than a yes or no.
    """
    harmful = design.harmful[eval_index]
    beneficial = design.beneficial[eval_index]
    raw_eval = model.safety(design, eval_index)
    documents = np.array(sorted(set(design.documents[target_cal].tolist())))
    rng = np.random.default_rng(pilot.FIT_SEED)
    curve: dict[str, Any] = {}

    for budget in (*BUDGETS, len(documents)):
        if budget > len(documents):
            continue
        repeats = 1 if budget == len(documents) else BUDGET_REPEATS
        draws: list[dict[str, Any]] = []
        for _ in range(repeats):
            chosen = set(rng.choice(documents, size=budget, replace=False).tolist())
            rows = target_cal[np.isin(design.documents[target_cal], list(chosen))]
            if design.harmful[rows].sum() == 0 or (~design.harmful[rows]).sum() == 0:
                continue
            calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
            calibrator.fit(model.safety(design, rows), (~design.harmful[rows]).astype(float))
            decision = select_threshold(
                calibrator.transform(model.safety(design, rows)),
                design.harmful[rows],
                PRIMARY_EPSILON,
                delta=DELTA,
                controller="ltt_bentkus",
            )
            draws.append(
                deployed_point(
                    decision.tau,
                    calibrator.transform(raw_eval),
                    harmful,
                    beneficial,
                    PRIMARY_EPSILON,
                )
            )
        if not draws:
            continue
        curve[str(budget)] = _budget_cell(budget, draws)
    # Budget 0 is the whole point of the comparison: no target labels at all.
    calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    source_cal = np.flatnonzero(
        (design.meta["role"].to_numpy(str) == "CALIBRATION")
        & (design.meta["engine_id"].to_numpy(str) != held_out)
    )
    calibrator.fit(model.safety(design, source_cal), (~design.harmful[source_cal]).astype(float))
    decision = select_threshold(
        calibrator.transform(model.safety(design, source_cal)),
        design.harmful[source_cal],
        PRIMARY_EPSILON,
        delta=DELTA,
        controller="ltt_bentkus",
    )
    zero = deployed_point(
        decision.tau, calibrator.transform(raw_eval), harmful, beneficial, PRIMARY_EPSILON
    )
    curve["0"] = _budget_cell(0, [zero])
    return curve


def _budget_cell(budget: int, draws: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one budget, keeping abstention separate from safety.

    A draw where the controller accepted nothing has a realized harm rate of zero, and
    averaging it in with draws that accepted edits would report a *safe* system where there
    is in fact *no* system. Abstention is counted, and the harm mean is taken over the draws
    that actually accepted something.
    """
    accepting = [d for d in draws if d["n_accepted"] > 0]
    realized = np.array([d["realized_harm_rate"] for d in accepting])
    return {
        "n_documents": budget,
        "n_draws": len(draws),
        "n_draws_abstained": len(draws) - len(accepting),
        "abstention_rate": float((len(draws) - len(accepting)) / len(draws)),
        "realized_harm_mean_when_accepting": float(realized.mean())
        if realized.size
        else float("nan"),
        "realized_harm_p90_when_accepting": float(np.quantile(realized, 0.9))
        if realized.size
        else float("nan"),
        "violation_rate": float(np.mean([d["bound_violated"] for d in draws])),
        "repair_recall_mean": float(np.mean([d["repair_recall"] for d in draws])),
        "coverage_mean": float(np.mean([d["coverage"] for d in draws])),
    }


def _engine_aware(design: Design, held_out: str, columns: list[int]) -> dict[str, Any]:
    """Experiment 5: is engine identity a useful conditioning variable?

    Split in two, because the literal question has two regimes with different answers:

    * **seen engine** -- add a one-hot and evaluate on an engine that was in the fit set.
      Well posed, and answered on the pooled DEVELOPMENT split.
    * **unseen engine** -- the one-hot is a constant zero in training and a constant one at
      evaluation. There is no fitted coefficient for it, so this is not a conditioning
      variable but an extrapolation, and it is reported as UNDEFINED rather than run and
      quoted. The defined substitute is a label-free engine descriptor.
    """
    role = design.meta["role"].to_numpy(str)
    engine_of = design.meta["engine_id"].to_numpy(str)
    engines = design.engines
    fit_loeo = np.flatnonzero((role == "TRAIN") & (engine_of != held_out))
    eval_index = design.rows(role="DEVELOPMENT", engine=held_out)
    harmful, beneficial = design.harmful[eval_index], design.beneficial[eval_index]
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"

    def one_hot(index: np.ndarray) -> np.ndarray:
        return np.stack([(engine_of[index] == e).astype(float) for e in engines], axis=1)

    agnostic = fit_harm_model(design, fit_loeo, columns)
    agnostic_score = agnostic.safety(design, eval_index)

    # Descriptor arm: the model sees WHAT the engine is like, never WHICH engine it is.
    descriptors = engine_descriptor_block(design)
    augmented = np.hstack([design.matrix[:, columns], descriptors])
    descriptor_design = Design(
        matrix=augmented,
        names=(
            *[design.names[c] for c in columns],
            *[f"engine_descriptor_{i}" for i in range(descriptors.shape[1])],
        ),
        meta=design.meta,
    )
    descriptor_model = fit_harm_model(descriptor_design, fit_loeo, list(range(augmented.shape[1])))
    descriptor_score = descriptor_model.safety(descriptor_design, eval_index)

    # Seen-engine regime, on the pooled split where the one-hot is defined for every row.
    pooled_fit = np.flatnonzero(role == "TRAIN")
    pooled_eval = design.rows(role="DEVELOPMENT")
    with_id = Design(
        matrix=np.hstack([design.matrix[:, columns], one_hot(np.arange(len(design.meta)))]),
        names=(*[design.names[c] for c in columns], *[f"engine_is_{e}" for e in engines]),
        meta=design.meta,
    )
    seen_agnostic = fit_harm_model(design, pooled_fit, columns)
    seen_aware = fit_harm_model(with_id, pooled_fit, list(range(with_id.matrix.shape[1])))

    return {
        "A_engine_agnostic": frontier(agnostic_score, harmful, beneficial)[key]["repair_recall"],
        "B_engine_identity_unseen": {
            "status": "UNDEFINED",
            "why": (
                "The held-out engine's indicator is identically zero on every fit row, so no "
                "coefficient is estimated for it, and identically one on every evaluation row. "
                "Running it would produce a number that measures the intercept shift, not "
                "conditioning."
            ),
        },
        "B_prime_engine_descriptor_unseen": frontier(descriptor_score, harmful, beneficial)[key][
            "repair_recall"
        ],
        "descriptor_note": (
            "Descriptors are constant within an engine, so on a single held-out engine they "
            "are a constant vector: they can shift the intercept but cannot reorder rows. The "
            "arm is reported to show that, not because it was expected to help."
        ),
        "seen_engine_regime": {
            "engine_agnostic": frontier(
                seen_agnostic.safety(design, pooled_eval),
                design.harmful[pooled_eval],
                design.beneficial[pooled_eval],
            )[key]["repair_recall"],
            "engine_identity_conditioned": frontier(
                seen_aware.safety(with_id, pooled_eval),
                design.harmful[pooled_eval],
                design.beneficial[pooled_eval],
            )[key]["repair_recall"],
            "population": "pooled DEVELOPMENT, all four engines, every indicator defined",
        },
    }


# ------------------------------------------------------------------ experiment 3


ABLATION_ARMS: dict[str, tuple[str, ...]] = {
    "M1_all_features": tuple(FAMILIES),
    "M2_minus_ocr_confidence": tuple(f for f in FAMILIES if f != "conf"),
    "M3_minus_engine_owned": tuple(f for f in FAMILIES if f not in ENGINE_OWNED),
    "M4_visual_and_edit_only": ("vis", "edit"),
    "M5_language_only": ("plaus",),
}


def run_ablation() -> int:
    """Experiment 3: which families carry transfer, with intervals, under LOEO.

    Leave-one-family-out is reported alongside the brief's five models, because a subset's
    score confounds "this family helps" with "the other families were removed". Only the
    difference between a model and the same model minus one family attributes anything.
    """
    started = time.monotonic()
    design = load_design()
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    arms = dict(ABLATION_ARMS)
    for family in FAMILIES:
        arms[f"LOFO_minus_{family}"] = tuple(f for f in FAMILIES if f != family)

    folds: dict[str, Any] = {}
    for held_out in design.engines:
        fit_index = np.flatnonzero(
            (design.meta["role"].to_numpy(str) == "TRAIN")
            & (design.meta["engine_id"].to_numpy(str) != held_out)
        )
        eval_index = design.rows(role="DEVELOPMENT", engine=held_out)
        harmful, beneficial = design.harmful[eval_index], design.beneficial[eval_index]
        documents = design.documents[eval_index]
        cell: dict[str, Any] = {}
        for name, families in arms.items():
            model = fit_harm_model(design, fit_index, design.columns(families))
            score = model.safety(design, eval_index)
            measured = frontier(score, harmful, beneficial)
            cell[name] = {
                "families": list(families),
                "n_features": len(design.columns(families)),
                "repair_recall": {
                    f"epsilon_{int(e * 100)}": measured[f"epsilon_{int(e * 100)}"]["repair_recall"]
                    for e in EPSILONS
                },
                "auc_safe": measured["auc_safe"],
                "aurc": measured["aurc"],
                "interval_primary_epsilon": clustered_interval(
                    score, harmful, beneficial, documents, PRIMARY_EPSILON
                ),
            }
        folds[held_out] = cell
        best = max(cell, key=lambda n: cell[n]["repair_recall"][key])
        print(f"  fold {held_out:11s} best={best} ({cell[best]['repair_recall'][key]:.3f})")

    fold_means = {
        name: float(np.mean([folds[e][name]["repair_recall"][key] for e in folds])) for name in arms
    }
    sign_agreement = {}
    for name in arms:
        if name == "M1_all_features":
            continue
        deltas = [
            folds[e][name]["repair_recall"][key] - folds[e]["M1_all_features"]["repair_recall"][key]
            for e in folds
        ]
        sign_agreement[name] = {
            "per_engine_delta": dict(zip(folds, deltas, strict=True)),
            "engines_improved": int(sum(d > 0 for d in deltas)),
            "unanimous": bool(all(d > 0 for d in deltas) or all(d < 0 for d in deltas)),
        }

    cc._write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv1-domain-generalization-ablation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV1-DG1",
            "arms": {name: list(families) for name, families in arms.items()},
            "engine_owned_families": list(ENGINE_OWNED),
            "engine_owned_note": (
                "conf_ is the recognizer's own confidence on its own native scale and geom_ "
                "is the detector's own box. Both are properties of the engine, not of the "
                "proposed edit, which is what 'engine-specific' means operationally here."
            ),
            "folds": folds,
            "fold_means": fold_means,
            "sign_agreement_against_all_features": sign_agreement,
            "inference_note": (
                "Four engines is not a sample to do inference over. The intervals are "
                "document-clustered WITHIN a fold and are the only intervals reported; the "
                "fold mean is a summary, and sign agreement across engines is the honest "
                "substitute for a cross-engine p-value."
            ),
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"ablation: {len(arms)} arms x {len(folds)} folds -> {cc._relative(ABLATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ figures


def run_figures() -> int:
    started = time.monotonic()
    shift = cc._read_json(SHIFT_RESULTS)
    loeo = cc._read_json(LOEO_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV1 DEVELOPMENT -- not a confirmatory result"
    written: list[Path] = []
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    engines = sorted(loeo["folds"])

    def finish(fig: Any, path: Path, title: str) -> None:
        fig.suptitle(f"{title}\n{note}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # --- feature_shift.png ---------------------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    ranked = shift["features_ranked_by_max_psi"][:18][::-1]
    values = [shift["per_feature"][n]["max_psi"] for n in ranked]
    nulls = [shift["per_feature"][n]["within_engine_null"]["psi_p95"] for n in ranked]
    colours = {
        "conf": "#a33",
        "geom": "#c87a2b",
        "vis": "#2e7d5b",
        "plaus": "#3a5f9e",
        "text": "#6d6d6d",
        "prov": "#7b52ab",
        "edit": "#2a8f9e",
        "ctx": "#9e9e2a",
    }
    positions = np.arange(len(ranked))
    left.barh(
        positions,
        values,
        color=[colours.get(shift["per_feature"][n]["family"], "#888") for n in ranked],
    )
    left.plot(nulls, positions, "k.", markersize=5, label="within-engine null (p95)")
    left.set_yticks(positions)
    left.set_yticklabels(ranked, fontsize=7)
    left.set_xlabel("max PSI, one engine against the other three")
    left.legend(fontsize=8)
    left.set_title("Which features move between engines", fontsize=10)

    between = shift["conditional_shift"]["between_engine_coefficient_cosine"]
    within = shift["conditional_shift"]["within_engine_split_half_null"]
    right.scatter(
        [0.0] * len(between),
        list(between.values()),
        color="#a33",
        s=45,
        label="between engines",
        zorder=3,
    )
    right.scatter(
        [1.0] * len(within),
        [v["mean"] for v in within.values()],
        color="#2e7d5b",
        s=45,
        label="within one engine\n(disjoint documents)",
        zorder=3,
    )
    for label, value in between.items():
        right.annotate(label, (0.02, value), fontsize=6, va="center")
    for label, value in within.items():
        right.annotate(label, (1.02, value["mean"]), fontsize=6, va="center")
    right.set_xlim(-0.35, 1.7)
    right.set_xticks([0.0, 1.0])
    right.set_xticklabels(["between", "within"])
    right.set_ylabel("cosine similarity of fitted harm-model coefficients")
    right.legend(fontsize=8, loc="lower left")
    right.set_title("The harm relationship itself differs by engine", fontsize=10)
    finish(
        fig,
        FIGURE_DIR / "feature_shift.png",
        "Engine shift is not only where the features sit, but what they mean",
    )

    # --- risk_coverage_transfer.png ------------------------------------------------------
    fig, axes = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 4.6), sharey=True)
    arms = (
        ("in_engine_ceiling", "#2e7d5b", "in-engine ceiling"),
        ("pooled_engine_seen", "#3a5f9e", "pooled (engine seen)"),
        ("loeo_pooled", "#a33", "LOEO pooled"),
        ("loeo_engine_balanced", "#c87a2b", "LOEO engine-balanced"),
        ("loeo_coral", "#7b52ab", "LOEO CORAL (inductive)"),
        ("loeo_coral_transductive", "#c9a7e0", "LOEO CORAL (transductive,\nnot deployable)"),
        ("loeo_invariant_subset", "#6d6d6d", "LOEO invariant subset"),
    )
    width = 0.8 / len(arms)
    for axis, engine in zip(axes, engines, strict=True):
        fold = loeo["folds"][engine]
        for offset, (arm, colour, label) in enumerate(arms):
            cell = fold["arms"][arm]
            interval = cell["interval_primary_epsilon"]
            axis.bar(
                offset * width,
                cell[key]["repair_recall"],
                width,
                color=colour,
                label=label if engine == engines[0] else None,
            )
            axis.errorbar(
                offset * width,
                cell[key]["repair_recall"],
                yerr=[
                    [max(cell[key]["repair_recall"] - interval["ci_lower"], 0.0)],
                    [max(interval["ci_upper"] - cell[key]["repair_recall"], 0.0)],
                ],
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=2,
            )
        axis.set_xticks([])
        axis.set_title(f"held out: {engine}\n{fold['beneficial']} beneficial rows", fontsize=9)
        axis.set_ylim(0, 1.0)
    axes[0].set_ylabel(f"achievable repair recall at harm <= {PRIMARY_EPSILON:.0%}")
    axes[0].legend(fontsize=7, loc="upper left")
    finish(
        fig,
        FIGURE_DIR / "risk_coverage_transfer.png",
        "No domain-generalization arm closes the gap to the in-engine ceiling",
    )

    # --- calibration_comparison.png ------------------------------------------------------
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(15.5, 5.0))
    methods = [
        f"{m}__{origin}"
        for m in CALIBRATION_METHODS
        for origin in ("source_engines", "target_engine")
    ]
    positions = np.arange(len(methods))
    bar_width = 0.8 / len(engines)
    for index, engine in enumerate(engines):
        quality = transfer["folds"][engine]["calibration_quality"]
        left.bar(
            positions + index * bar_width,
            [quality[m]["ece_equal_mass"] for m in methods],
            bar_width,
            label=engine,
        )
    left.set_xticks(positions + 0.4)
    left.set_xticklabels(
        [m.replace("__", "\n").replace("_engine", "").replace("_engines", "") for m in methods],
        fontsize=7,
        rotation=40,
        ha="right",
    )
    left.set_ylabel("ECE (equal-mass, 10 bins) on the held-out engine")
    left.legend(fontsize=8)
    left.set_title("Target labels fix calibration", fontsize=10)

    # Harm and coverage are plotted together on purpose. A harm curve alone would show the
    # certified controller looking safest exactly where it accepted nothing at all, and a
    # figure that reports abstention as safety is the wrong figure.
    for axis, field, label, title in (
        (
            middle,
            "realized_harm_mean_when_accepting",
            "realized harm among accepted edits\n(draws that accepted anything)",
            "...and the bound mostly holds",
        ),
        (
            right,
            "repair_recall_mean",
            "repair recall delivered",
            "...by accepting almost nothing",
        ),
    ):
        for engine in engines:
            budget = transfer["folds"][engine]["label_budget"]
            keys = sorted(budget, key=int)
            xs = [int(k) for k in keys]
            axis.plot(xs, [budget[k][field] for k in keys], marker="o", markersize=4, label=engine)
            abstaining = [
                (int(k), budget[k][field]) for k in keys if budget[k]["abstention_rate"] > 0.0
            ]
            if abstaining:
                axis.scatter(
                    [x for x, _ in abstaining],
                    [y for _, y in abstaining],
                    marker="x",
                    s=55,
                    color="black",
                    zorder=4,
                )
        axis.set_xscale("symlog")
        axis.axvline(1.0, color="#bbbbbb", linewidth=0.8)
        axis.annotate(
            "no target\nlabels", (0.02, 0.02), xycoords="axes fraction", fontsize=6, color="#666"
        )
        axis.set_xlabel("labelled documents from the new engine used to recalibrate")
        axis.set_ylabel(label, fontsize=9)
        axis.set_title(title, fontsize=10)
    middle.axhline(PRIMARY_EPSILON, color="black", linestyle="--", linewidth=1)
    middle.annotate(
        f"nominal bound {PRIMARY_EPSILON:.0%}", (1.5, PRIMARY_EPSILON * 1.06), fontsize=8
    )
    middle.scatter([], [], marker="x", s=55, color="black", label="some draws abstained")
    middle.legend(fontsize=7)
    right.legend(fontsize=8)
    finish(
        fig,
        FIGURE_DIR / "calibration_comparison.png",
        "Calibration transfer: what target labels buy, and what they do not",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv1-domain-generalization-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "synthetic": False,
            "annotation": note,
            "derived_from": {
                cc._relative(path): file_sha256(path)
                for path in (SHIFT_RESULTS, LOEO_RESULTS, TRANSFER_RESULTS)
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ the finding


def run_decide() -> int:
    """The five interpretation questions, answered from the artifacts and nothing else."""
    started = time.monotonic()
    shift = cc._read_json(SHIFT_RESULTS)
    loeo = cc._read_json(LOEO_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    engines = sorted(loeo["folds"])

    gaps = {e: loeo["folds"][e]["transfer_gap_primary_epsilon"] for e in engines}
    ceilings = {
        e: loeo["folds"][e]["arms"]["in_engine_ceiling"][key]["repair_recall"] for e in engines
    }
    baselines = {e: loeo["folds"][e]["arms"]["loeo_pooled"][key]["repair_recall"] for e in engines}

    between = shift["conditional_shift"]["between_engine_coefficient_cosine"]
    within = shift["conditional_shift"]["within_engine_split_half_null"]
    conditional_separates_on_means = max(between.values()) < min(v["mean"] for v in within.values())
    conditional_separates_on_worst_split = max(between.values()) < min(
        v["min"] for v in within.values()
    )

    zero_budget = {
        e: transfer["folds"][e]["label_budget"]["0"]["realized_harm_mean_when_accepting"]
        for e in engines
    }
    full_budget_key = {
        e: max(transfer["folds"][e]["label_budget"], key=lambda k: int(k)) for e in engines
    }
    full_budget = {
        e: transfer["folds"][e]["label_budget"][full_budget_key[e]][
            "realized_harm_mean_when_accepting"
        ]
        for e in engines
    }
    minimum_useful_budget = {
        e: next(
            (
                int(k)
                for k in sorted(transfer["folds"][e]["label_budget"], key=lambda x: int(x))
                if int(k) > 0 and transfer["folds"][e]["label_budget"][k]["abstention_rate"] == 0.0
            ),
            None,
        )
        for e in engines
    }
    in_engine_control = {
        e: transfer["folds"][e]["in_engine_threshold_control"]["ltt_bentkus"]["on_held_out_engine"][
            "realized_harm_rate"
        ]
        for e in engines
    }
    frontier_moves = transfer["monotone_invariance"]["max_abs_frontier_move_by_method"]

    # The descriptor arm lives in the transfer artifact rather than the LOEO one, and it is
    # the arm that came closest. Folding it into the same comparison is the point: an arm
    # that did better must not be easier to overlook than the arms that did not.
    fold_means = dict(loeo["fold_means"])
    descriptor_by_engine = {
        e: transfer["folds"][e]["engine_aware"]["B_prime_engine_descriptor_unseen"] for e in engines
    }
    fold_means["loeo_engine_descriptor"] = float(np.mean(list(descriptor_by_engine.values())))
    # The transductive CORAL arm is excluded from the support test on purpose: it estimates
    # its alignment on the rows it is scored on, so it is an upper reference and not a
    # candidate method. It is reported in fold_means so the exclusion is visible.
    dg_arms = [
        a
        for a in fold_means
        if a.startswith("loeo_") and a not in {"loeo_pooled", "loeo_coral_transductive"}
    ]
    best_dg = max(dg_arms, key=lambda a: fold_means[a])

    def arm_by_engine(arm: str) -> dict[str, float]:
        if arm == "loeo_engine_descriptor":
            return descriptor_by_engine
        return {e: loeo["folds"][e]["arms"][arm][key]["repair_recall"] for e in engines}

    dg_beats_baseline_everywhere = {
        arm: bool(all(arm_by_engine(arm)[e] > baselines[e] for e in engines)) for arm in dg_arms
    }
    dg_per_engine = {arm: arm_by_engine(arm) for arm in dg_arms}

    # A hypothesis is only supported if the arm that is supposed to support it beats the
    # baseline on every engine AND closes a material part of the gap. Neither holds.
    dg1_supported = bool(
        any(dg_beats_baseline_everywhere.values())
        and fold_means[best_dg] >= 0.5 * float(np.mean(list(ceilings.values())))
    )

    payload = {
        "schema_version": "sgv1-domain-generalization-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "c2_status": "DEFERRED",
        "confirmatory_accessed": False,
        "hypothesis_verdict": None,
        "hypothesis_id": "SGV1-DG1",
        "hypothesis_id_note": (
            "The brief names this hypothesis H4. docs/sgv1/protocol.md already binds SGV1-H4 "
            "to the V2-V1 image increment under held-out-engine evaluation and states that "
            "prior IDs are never reused, so this claim is filed as SGV1-DG1. No frozen ID was "
            "overwritten."
        ),
        "sgv1_dg1": {
            "claim": (
                "A calibrated harm model built on engine-invariant features maintains "
                "bounded-risk repair coverage on an unseen OCR engine."
            ),
            "supported": dg1_supported,
            "best_arm": best_dg,
            "best_arm_fold_mean": fold_means[best_dg],
            "baseline_fold_mean": fold_means["loeo_pooled"],
            "in_engine_ceiling_fold_mean": fold_means["in_engine_ceiling"],
            "arms_beating_baseline_on_every_engine": dg_beats_baseline_everywhere,
            "arm_fold_means": fold_means,
            "excluded_from_support_test": {
                "loeo_coral_transductive": (
                    "estimates its alignment on the evaluation rows; an upper reference, not "
                    "a deployable arm"
                ),
                "in_engine_ceiling": "the control being compared against",
                "pooled_engine_seen": "the engine is not held out",
            },
            "arm_by_engine": dg_per_engine,
            "why_not_supported": (
                "Support requires an arm that beats the pooled baseline on EVERY held-out "
                "engine and reaches at least half the in-engine ceiling. The best arm does "
                "neither. A fold mean that improves while one engine degrades is a "
                "heterogeneity result, not a generalization result."
            ),
        },
        "q1_why_does_the_policy_fail_under_engine_shift": {
            "answer": (
                "Because the harm model's ranking does not transfer, not because the "
                "population changed. On identical evaluation rows an in-engine model reaches "
                "repair recalls of "
                + ", ".join(f"{e} {ceilings[e]:.3f}" for e in engines)
                + " where the leave-one-engine-out model reaches "
                + ", ".join(f"{e} {baselines[e]:.3f}" for e in engines)
                + "."
            ),
            "transfer_gap_by_engine": gaps,
            "gap_positive_on_every_engine": bool(all(v > 0 for v in gaps.values())),
            "correction_to_the_predecessor": loeo["confound_in_the_predecessor_comparison"],
        },
        "q2_miscalibration_or_feature_instability": {
            "answer": (
                "Both are present and they damage different things. Miscalibration decides "
                "whether the risk bound holds; feature and conditional instability decide how "
                "many repairs are reachable at that bound. Neither substitutes for the other."
            ),
            "calibration_is_real_and_repairable": {
                "ece_source_calibrated": {
                    e: transfer["folds"][e]["calibration_quality"]["isotonic__source_engines"][
                        "ece_equal_mass"
                    ]
                    for e in engines
                },
                "ece_target_calibrated": {
                    e: transfer["folds"][e]["calibration_quality"]["isotonic__target_engine"][
                        "ece_equal_mass"
                    ]
                    for e in engines
                },
            },
            "but_calibration_cannot_move_the_frontier": {
                "max_abs_frontier_move_by_method": frontier_moves,
                "why": transfer["monotone_invariance"]["claim"],
            },
            "covariate_shift_magnitude": shift["domain_classifier_auc"],
            "conditional_shift_magnitude": {
                "between_engine_coefficient_cosine": between,
                "within_engine_null": within,
                "between_is_below_every_within_engine_mean": conditional_separates_on_means,
                "between_is_below_every_within_engine_worst_split": (
                    conditional_separates_on_worst_split
                ),
                "n_features_whose_harm_association_flips_sign": shift["n_features_with_sign_flip"],
            },
        },
        "q3_can_engine_invariant_features_recover_transfer": {
            "answer": (
                "Not in any form tested here. Engine-balanced sampling, unsupervised CORAL "
                "alignment, and a nested-selected invariant feature subset all leave the fold "
                "mean far below the in-engine ceiling, and none of them improves every engine."
            ),
            "fold_means": fold_means,
            "arms_beating_baseline_on_every_engine": dg_beats_baseline_everywhere,
            "per_engine": dg_per_engine,
            "ablation_fold_means": ablation["fold_means"],
            "ablation_sign_agreement": {
                name: cell["unanimous"]
                for name, cell in ablation["sign_agreement_against_all_features"].items()
            },
        },
        "q4_should_engine_identity_be_modelled_explicitly": {
            "answer": (
                "Not for an unseen engine, where it is undefined rather than unhelpful. What "
                "is defined and does matter is target-engine recalibration, which needs labels "
                "rather than identity."
            ),
            "per_engine": {e: transfer["folds"][e]["engine_aware"] for e in engines},
        },
        "q5_what_is_the_contribution": {
            "answer": "a bounded negative development finding plus one measured deployment cost",
            "supported": [
                (
                    "The cross-engine failure of the harm model survives the control the "
                    "predecessor stage omitted: an in-engine model on identical evaluation "
                    "rows, which is better on every engine."
                ),
                (
                    "The failure decomposes. Recalibration is provably unable to move the "
                    "achievable frontier because monotone maps do not reorder; measured "
                    "frontier movement is bounded by the isotonic tie effect alone."
                ),
                (
                    "Fitted harm-model coefficient geometry differs more between engines than "
                    "between disjoint document halves of one engine, which is conditional "
                    "shift and not sampling noise."
                ),
                (
                    "A distribution-free threshold certified at epsilon on the fit engines "
                    "does not hold on an unseen engine, and the size of that miss is measured "
                    "against an in-engine control of the same threshold rule."
                ),
            ],
            "not_supported": [
                (
                    "SGV1-DG1. No engine-invariant representation tested restores "
                    "bounded-risk coverage."
                ),
                (
                    "Any claim that engine-balanced sampling, CORAL, or feature pruning is a "
                    "remedy. None improves every held-out engine."
                ),
                (
                    "Any deployment claim on an unseen engine. Two of four engines still miss "
                    "the bound at full target-label budget."
                ),
            ],
            "prior_art_note": (
                "Covariate/concept shift diagnosis, importance-style alignment, recalibration "
                "under shift, and selective prediction are all established prior art and are "
                "listed as such in docs/prior_art_boundary.md. Nothing here is claimed as new "
                "method; the contribution is a measurement on this benchmark."
            ),
            "realized_harm_at_nominal_epsilon": {
                "epsilon": PRIMARY_EPSILON,
                "zero_target_labels": zero_budget,
                "full_target_label_budget": full_budget,
                "in_engine_same_threshold_rule": in_engine_control,
                "smallest_budget_with_no_abstaining_draw": minimum_useful_budget,
                "note": (
                    "Harm rates are taken over draws that accepted at least one edit. Below "
                    "the budgets listed the certified controller abstains entirely, which is "
                    "zero harm and zero system, and averaging the two together would report "
                    "safety where there is no deployment."
                ),
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    cc._write_json_once(DECISION, payload)
    print(f"decide: SGV1-DG1 supported={dg1_supported} -> {cc._relative(DECISION)}")
    return 0


def run_design() -> int:
    """Materialize the design matrix and record what produced it."""
    started = time.monotonic()
    design = load_design()
    engine_of = design.meta["engine_id"].to_numpy(str)
    role = design.meta["role"].to_numpy(str)
    cc._write_json_once(
        FIT_RECORD,
        {
            "schema_version": "sgv1-domain-generalization-fit-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV1-DG1",
            "representation": "Phase 3 candidate-conditioned R1, unchanged",
            "evidence_config": "sgv1_v1",
            "n_rows": len(design.meta),
            "n_features": len(design.names),
            "feature_names": list(design.names),
            "families": {
                family: [n for n in design.names if n.startswith(prefix)]
                for family, prefix in FAMILIES.items()
            },
            "rows_by_role_and_engine": {
                f"{r}|{e}": int(np.count_nonzero((role == r) & (engine_of == e)))
                for r in sorted(set(role.tolist()))
                for e in design.engines
            },
            "model": {
                "estimator": "sklearn LogisticRegression",
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": pilot.FIT_SEED,
                "target": "is_harmful",
            },
            "calibration_method_default": pilot.CALIBRATION_METHOD,
            "risk_controllers": list(CONTROLLERS),
            "delta": DELTA,
            "epsilon_grid": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "bootstrap": {
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": pilot.BOOTSTRAP_SEED,
                "unit": "document",
            },
            "ground_truth_used_for_features": False,
            "confirmatory_accessed": False,
            "inputs": {
                cc._relative(path): file_sha256(path)
                for path in (pilot.CANDIDATE_TABLE, pilot.LABEL_TABLE, cc.FEATURES)
            },
            "artifacts": {cc._relative(DESIGN_MATRIX): file_sha256(DESIGN_MATRIX)},
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"design: {design.matrix.shape} -> {cc._relative(DESIGN_MATRIX)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("design", "shift", "loeo", "transfer", "ablation", "figures", "decide"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    stages = (
        ("design", run_design),
        ("shift", run_shift),
        ("loeo", run_loeo),
        ("transfer", run_transfer),
        ("ablation", run_ablation),
        ("figures", run_figures),
        ("decide", run_decide),
    )
    selected = [fn for name, fn in stages if getattr(args, name)]
    if not selected:
        parser.print_help()
        return 2
    for stage in selected:
        code = stage()
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

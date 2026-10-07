#!/usr/bin/env python3
"""SGV6: how much target-engine supervision does a safe correction policy actually need?

SGV5 left one number standing above all the others. On the four leave-one-engine-out folds
the deployable model reached repair recall 0.636 / 0.009 / 0.641 / 0.738 at a 10% harm bound,
and the same model class fitted on the held-out engine's OWN labelled training documents --
the in-engine oracle -- reached 0.782 / 0.761 / 0.938 / 0.986. Mean transfer cost 0.361
against mean residual 0.133. The representation is not the binding constraint and the
ambiguity is not irreducible; what is missing is that the model has never seen a label from
the engine it is being asked to judge.

So this stage stops trying to make zero-shot transfer work and asks the question the oracle
gap poses directly:

    SGV6-T1: a reliability model fitted on several OCR engines can recover a useful part of
    the in-engine oracle's bounded-risk repair coverage on an unseen engine from a small
    number of labelled examples of that engine, and the number of labels required is
    measurable.

**The hypothesis is recorded as SGV6-T1, not H1.** `docs/sgv1/protocol.md` binds SGV1-H1
through SGV1-H4 and says a frozen ID is never reused for a different claim; SGV2, SGV3, SGV4
and SGV5 each restarted their own family for the same reason. The brief's SGV6-H1/H2/H3 are
carried as sub-claims T1a (adaptation beats zero-shot), T1b (the endpoint rises with the
budget) and T1c (the oracle gap closes), all under T1.

Five decisions determine what the numbers below can mean.

**1. The label budget is drawn from a pool that already exists, and the oracle is its
endpoint.** The adaptation pool is the held-out engine's TRAIN documents -- exactly the rows
`sgv5.in_engine_fold` fits the oracle on. A budget of N is a prefix of that pool. So the
curve does not merely approach the oracle asymptotically by analogy: at N equal to the whole
pool it *is* the oracle's fitting set, and every intermediate point is a real subset of it.
The document partition is untouched, so no budget row shares a page with an evaluation row.

**2. The budget pays for everything, including the choice of adapter.** A comparison in which
the model is adapted on N target labels and then the adapter family, its shrinkage and the
deployment threshold are chosen with a further pool of target labels is not a comparison at
budget N. Every choice this stage makes on the target side is made from grouped
cross-validation inside the same N rows, and the grouping is by document.

**3. Every adapter family contains the zero-shot model exactly.** Each family's shrinkage
grid includes an identity element, and the arm selection includes the unadapted model itself.
At N = 0 no fitting happens at all and the score vector is SGV5's, bit for bit; the identity
is asserted against SGV5's frozen table rather than assumed. The consequence is the property
this stage needs to be honest: when the budget carries no usable signal, the procedure returns
SGV5 rather than returning noise, and a measured gain therefore cannot be a selection artifact
of having more knobs.

**4. Class prevalence in the target pool, not the budget size, is what a label costs.** The
adaptation pools are 3.0% beneficial on docTR and 3.3% on PaddleOCR against 25.7% on
Tesseract. A random budget of 100 rows is 3 beneficial examples on one engine and 26 on
another; the same nominal budget is not the same experiment. This is why SGV6-B is not an
optional extra here -- it is the difference between the budget being spendable and not -- and
why the label-efficiency curves are reported per engine and never pooled.

**5. A single draw of 10 rows is not an estimate.** The stochastic acquisition is repeated
over five seeds fixed in advance, the endpoint is computed per seed, and the spread across
seeds is reported next to the document-clustered interval. Where the two disagree about the
size of the uncertainty, the larger one is the honest one.

    --adapt        every model this stage fits: the frozen SGV5 base, the budgets, the
                   adapters, the acquisitions, and the freezing ablation
    --curves       the risk-coverage frontier at every epsilon for every arm and budget
    --efficiency   the adaptation curve, its monotonicity, and the recovery ratio
    --oracle       the oracle gap per engine and the budget that closes it
    --active       SGV6-B: does the acquisition rule change the price of a label?
    --ablation     feature freezing, label budget, engine difficulty
    --figures      the required figures
    --decide       the machine-readable finding
    --record       provenance for every artifact

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; neither axis is relaxed anywhere, and
the target-engine labels this stage spends come from documents no evaluation row belongs to.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_candidate_conditioned as cc
import sgv1_domain_generalization as dg
import sgv1_risk_policy as policy
import sgv1_verifier_pilot as pilot
import sgv2_reliability_layer as rl
import sgv3_self_aware as sa
import sgv5_candidate_reliability as c5
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.risk.controller import CONTROLLERS, select_threshold

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv6_target_engine_adaptation"
SCORES = OUT / "adaptation_scores.parquet"
SELECTION_RECORD = OUT / "selection_record.json"
ADAPTATION_RESULTS = OUT / "adaptation_results.json"
CURVE_RESULTS = OUT / "risk_coverage_curves.json"
EFFICIENCY_RESULTS = OUT / "label_efficiency.json"
ORACLE_GAP = OUT / "oracle_gap.json"
ACTIVE_RESULTS = OUT / "active_learning_results.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance_manifest.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# The endpoint, the restriction that keeps a rejected row rejected inside it, the interval and
# the paired delta have exactly one implementation in this repository. Six stages now import
# them rather than restating them; `tests/leakage` asserts the identity.
achievable_repair_recall = rl.achievable_repair_recall
restricted_frontier = sa._restricted_frontier
restricted_interval = sa._restricted_interval
arm_summary = sa._arm_summary
paired_repair_recall_delta = sa._paired_repair_recall_delta
harm_at_matched_coverage = rl.harm_at_matched_coverage
deployed_point = c5._deployed
build_fold = rl.build_fold
Fold = rl.Fold

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
PRIMARY_KEY = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
MATCHED_COVERAGES = rl.MATCHED_COVERAGES
CROSS_ENGINE = sa.CROSS_ENGINE

# The brief's grid, unchanged. Zero is not a padding entry: it is the arm the whole stage is
# measured against, and it is produced by running no fitting at all rather than by fitting
# something on an empty budget.
BUDGETS = (0, 10, 25, 50, 100, 250, 500, 1000)
NONZERO_BUDGETS = tuple(n for n in BUDGETS if n > 0)
SEED_REPEATS = 5
BUDGET_SEED = 20260903

RANDOM = "random"
UNCERTAINTY = "uncertainty"
DIVERSITY = "diversity"
EXPECTED_HARM_REDUCTION = "expected_harm_reduction"
ACQUISITIONS = (RANDOM, UNCERTAINTY, DIVERSITY, EXPECTED_HARM_REDUCTION)

# The adapter reads the frozen model's two calibrated heads plus a low-dimensional projection
# of the representation the frozen model already sees. Eight components is the whole point of
# the constraint: an adapter with as many parameters as the base model is retraining, and
# `freeze_c_target_only` is in this script precisely to measure what that costs.
ADAPTER_COMPONENTS = 8
INNER_FOLDS = 5
PROB_FLOOR = 1e-6

ZERO_SHOT = "zero_shot"
A1_LAMBDA = "a1_lambda"
A1_PLATT = "a1_platt"
A1_TEMPERATURE = "a1_temperature"
A1_ISOTONIC = "a1_isotonic"
A2_LOGISTIC = "a2_logistic"
A2_RIDGE = "a2_ridge"
A2_BOOSTED = "a2_boosted"
A3_BAYES = "a3_bayes"
ADAPTER_ARMS = (
    ZERO_SHOT,
    A1_LAMBDA,
    A1_PLATT,
    A1_TEMPERATURE,
    A1_ISOTONIC,
    A2_LOGISTIC,
    A2_RIDGE,
    A2_BOOSTED,
    A3_BAYES,
)
SELECTED_ARM = "sgv6"
PERMUTED_ARM = "sgv6_permuted_budget"

FREEZE_ADAPTER = "freeze_a_adapter"
FREEZE_JOINT_NAIVE = "freeze_b_joint_naive"
FREEZE_JOINT_PARITY = "freeze_b_joint_parity"
FREEZE_TARGET_ONLY = "freeze_c_target_only"
FREEZE_ARMS = (FREEZE_ADAPTER, FREEZE_JOINT_NAIVE, FREEZE_JOINT_PARITY, FREEZE_TARGET_ONLY)

# How much of the fitted correction to apply, for the families whose fit does not itself
# depend on a penalty. Zero is the identity and is always in the grid.
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# Gaussian prior precision on the correction coefficients. An infinite precision is the
# identity element and it is EXACT rather than numerically small: the coefficient vector is
# zero, so the arm returns the frozen model's probability to the last bit. That exactness is
# what lets this stage say "a budget carrying no signal returns SGV5" as a fact about the
# numbers rather than as an approximation.
TAU_GRID = (float("inf"), 1e6, 1e4, 1e3, 1e2, 1e1, 1.0, 1e-1)
# The evidence cannot score an infinite precision -- its Occam term is the log determinant of
# a zero covariance -- so the Bayesian family reads the finite part of the same grid.
EVIDENCE_GRID = TAU_GRID[1:]
# The harm-aversion grid every stage since SGV1 has used. `a1_lambda` re-picks from it on the
# target budget, which is the cheapest possible use of a target label: one parameter, no
# features, and the identity element is whatever SGV5 already chose for this fold.
LAMBDA_GRID = policy.LAMBDAS

# The frozen baselines this stage reads rather than refits.
BASELINE_COLUMNS = (
    "arm__no_correction",
    "arm__harm_only",
    "arm__harm_aware",
    "arm__selfaware",
    "arm__env_aware",
    "arm__sgv5",
    "arm__oracle_in_engine",
)
SGV5_ARM = "arm__sgv5"
SGV1_ARM = "arm__harm_aware"
ORACLE_ARM = "arm__oracle_in_engine"


class PhaseError(RuntimeError):
    """A freeze, role, split, budget or selection invariant failed."""


# ------------------------------------------------------------------ the frozen SGV5 base


def _stable_seed(*parts: str) -> int:
    """A seed derived from content, never from `hash()`, which PYTHONHASHSEED can move."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:7], "big")


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1.0 - PROB_FLOOR)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))), dtype=float)


def load_base() -> tuple[dg.Design, np.ndarray, list[int]]:
    """SGV5's fold-invariant inputs: the extended matrix, the edit signatures, the columns.

    Rebuilt rather than read, because block E is fold-dependent and cannot live in a
    corpus-wide table. Everything here is a pure function of artifacts SGV5 already froze, and
    `--adapt` asserts the rebuilt model reproduces SGV5's published score vector exactly.
    """
    design = dg.load_design()
    if not c5.FEATURES.is_file():
        raise PhaseError("SGV5's candidate_features.parquet is missing; run SGV5 --features")
    structural = pd.read_parquet(c5.FEATURES)
    published = list(design.meta["candidate_id"].astype(str))
    if list(structural["candidate_id"].astype(str)) != published:
        raise PhaseError("the structural feature table is not aligned with the design matrix")
    base = c5.base_design(design, structural[list(c5.STRUCTURAL_NAMES)])
    ordered = policy._pool_with_features().set_index("candidate_id").loc[published]
    signatures = np.array(
        [
            c5.edit_signature(o, y, op)
            for o, y, op in zip(
                ordered["original_ocr"].astype(str),
                ordered["candidate_text"].astype(str),
                ordered["operation"].astype(str),
                strict=True,
            )
        ]
    )
    retrieval_columns = [i for i, name in enumerate(base.names) if not name.startswith("retr_")]
    return base, signatures, retrieval_columns


def frozen_selection() -> dict[str, dict[str, Any]]:
    """SGV5's per-fold choice of representation, model class and lambda, read not re-made.

    Re-running SGV5's inner leave-one-engine-out here would cost twenty minutes and could only
    reproduce a choice that is already frozen, hashed and published. Reading it is also the
    stricter option: a selection that drifted between the two stages would be invisible if
    each stage made its own, and is a hard failure when one reads the other's record.
    """
    if not c5.POLICY_SELECTION.is_file():
        raise PhaseError("SGV5's policy_selection.json is missing; run SGV5 --scores")
    record = cc._read_json(c5.POLICY_SELECTION)
    folds = record["folds"]
    return {engine: dict(folds[engine]["selected"]) for engine in sorted(folds)}


def load_frozen_scores() -> pd.DataFrame:
    """SGV5's cross-engine rows, verified against the hash SGV5's own record carries."""
    fit_record = cc._read_json(c5.FIT_RECORD)
    published = dict(fit_record["artifacts"])
    relative = cc._relative(c5.PREDICTIONS)
    if relative not in published:
        raise PhaseError(f"{relative} is not in SGV5's provenance record")
    if file_sha256(c5.PREDICTIONS) != published[relative]:
        raise PhaseError(f"{relative} does not match the hash SGV5 recorded for it")
    frame = pd.read_parquet(c5.PREDICTIONS)
    frame = frame[frame["evaluation_mode"] == CROSS_ENGINE].reset_index(drop=True)
    missing = [name for name in BASELINE_COLUMNS if name not in frame.columns]
    if missing:
        raise PhaseError(f"SGV5's table is missing {missing}")
    return frame


@dataclass(slots=True)
class TargetPool:
    """The held-out engine's own labelled TRAIN rows: the only place a budget may come from.

    This is the same index block `sgv5.in_engine_fold` uses as its fitting set, which is what
    makes the oracle the endpoint of this stage's budget axis rather than a separate quantity.
    Its documents are disjoint from the evaluation documents by the outer fold's construction,
    and `--adapt` re-asserts that rather than inheriting it.
    """

    index: np.ndarray
    documents: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    p_harm: np.ndarray
    p_benefit: np.ndarray
    utility: np.ndarray
    adapter_input: np.ndarray

    @property
    def size(self) -> int:
        return int(self.index.size)


@dataclass(slots=True)
class BaseFold:
    """One fold's frozen SGV5 model plus everything the adapters need to sit on top of it."""

    held_out: str
    fold: Fold
    selected: dict[str, Any]
    lambda_: float
    model: c5.Reliability
    extended: c5.FoldDesign
    design: dg.Design
    projection: Any
    scaler: Any
    pool: TargetPool
    evaluation: dict[str, np.ndarray]
    calibration: dict[str, np.ndarray]
    identity: dict[str, Any]
    diagnostics: dict[str, Any]


def _fill_target_block(
    base: dg.Design,
    extended: c5.FoldDesign,
    retrieval: c5.Retrieval,
    fold: Fold,
    signatures: np.ndarray,
    retrieval_columns: list[int],
) -> c5.FoldDesign:
    """Materialise block E on the target pool without touching any value SGV5 already set.

    SGV5 fills block E on the four blocks its own experiments score and leaves the rest
    unfilled, so the target pool -- which SGV5 never scores -- would trip its `filled` guard
    here. The retrieval index is built from the FIT engines' rows only, so scoring a target row
    against it reads no target label; `leave_one_out` is False because a target row is not in
    the index and cannot be its own neighbour.
    """
    matrix = extended.design.matrix.copy()
    filled = extended.filled.copy()
    index = fold.unlabeled_target
    if index.size:
        block = retrieval.features(base, index, retrieval_columns, signatures, leave_one_out=False)
        matrix[np.ix_(index, range(len(base.names), matrix.shape[1]))] = np.nan_to_num(block)
        filled[index] = True
    return c5.FoldDesign(
        design=dg.Design(matrix=matrix, names=extended.names, meta=extended.design.meta),
        filled=filled,
        names=extended.names,
    )


def _adapter_input(
    model: c5.Reliability,
    design: dg.Design,
    index: np.ndarray,
    projection: Any,
    scaler: Any,
    p_harm: np.ndarray,
    p_benefit: np.ndarray,
) -> np.ndarray:
    """What an adapter is allowed to see: the two frozen heads and a frozen low-rank view.

    The projection and the standardisation are fitted on the FIT engines' training rows and
    frozen before any target row is touched, so the adapter's input space carries no target
    statistic at all -- not a mean, not a variance, not a principal direction. The only target
    information in an adapter is the handful of labels the budget paid for.
    """
    components = projection.transform(model._scaled(design, index))
    raw = np.column_stack([_logit(p_harm), _logit(p_benefit), components])
    return np.asarray(scaler.transform(raw), dtype=float)


def build_base_fold(
    base: dg.Design,
    held_out: str,
    signatures: np.ndarray,
    retrieval_columns: list[int],
    selection: dict[str, Any],
    frozen: pd.DataFrame,
) -> BaseFold:
    """Rebuild SGV5's selected model for one fold and prove it is SGV5's, not a near-copy."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fold = build_fold(base, held_out)
    extended_raw, retrieval = c5.extend_fold(base, fold, signatures, retrieval_columns)
    extended = _fill_target_block(
        base, extended_raw, retrieval, fold, signatures, retrieval_columns
    )
    design = extended.design
    representation, model_kind = selection["representation"], selection["model"]
    if model_kind == "ranker":  # pragma: no cover - no fold selected the ranker
        raise PhaseError(
            "the adapter families are defined over two calibrated probability heads and the "
            "ranker exposes neither; a fold that selected it would need its own construction"
        )
    lambda_ = float(selection["lambda"])
    model = c5.fit_reliability(
        design, fold, extended.columns(representation), model_kind, representation
    )

    fit_scaled = model._scaled(design, fold.fit)
    components = min(ADAPTER_COMPONENTS, fit_scaled.shape[1], fit_scaled.shape[0])
    projection = PCA(n_components=components, random_state=pilot.FIT_SEED).fit(fit_scaled)

    def heads(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = model.probabilities(design, extended.block(index))
        return probabilities["harm"], probabilities["benefit"]

    fit_harm, fit_benefit = heads(fold.fit)
    scaler = StandardScaler().fit(
        np.column_stack([_logit(fit_harm), _logit(fit_benefit), projection.transform(fit_scaled)])
    )

    blocks: dict[str, dict[str, np.ndarray]] = {}
    for name, index in (
        ("evaluation", fold.eval),
        ("calibration", fold.source_cal),
        ("pool", fold.unlabeled_target),
    ):
        harm, benefit = heads(index)
        blocks[name] = {
            "index": index,
            "p_harm": harm,
            "p_benefit": benefit,
            "utility": benefit - lambda_ * harm,
            "adapter_input": _adapter_input(
                model, design, index, projection, scaler, harm, benefit
            ),
            "harmful": design.harmful[index],
            "beneficial": design.beneficial[index],
            "documents": design.documents[index],
        }

    identifiers = design.meta["candidate_id"].astype(str).to_numpy()
    reference = frozen[frozen["held_out_engine"] == held_out].set_index("candidate_id")
    published = reference.loc[identifiers[fold.eval], SGV5_ARM].to_numpy(dtype=float)
    difference = float(np.abs(blocks["evaluation"]["utility"] - published).max())
    if difference != 0.0:
        raise PhaseError(
            f"{held_out}: the rebuilt SGV5 arm differs from the published one by {difference:g}; "
            "every SGV6 number is a delta against that arm and a drifted baseline invalidates "
            "all of them"
        )

    pool_documents = set(blocks["pool"]["documents"].tolist())
    evaluation_documents = set(blocks["evaluation"]["documents"].tolist())
    if pool_documents & evaluation_documents:
        raise PhaseError(f"{held_out}: the adaptation pool shares documents with the evaluation")
    pool = TargetPool(
        index=blocks["pool"]["index"],
        documents=blocks["pool"]["documents"],
        harmful=blocks["pool"]["harmful"],
        beneficial=blocks["pool"]["beneficial"],
        p_harm=blocks["pool"]["p_harm"],
        p_benefit=blocks["pool"]["p_benefit"],
        utility=blocks["pool"]["utility"],
        adapter_input=blocks["pool"]["adapter_input"],
    )
    oracle_fold = c5.in_engine_fold(base, held_out)
    diagnostics = {
        "rows": {
            "fit": int(fold.fit.size),
            "source_calibration": int(fold.source_cal.size),
            "cross_engine_development": int(fold.eval.size),
            "adaptation_pool": pool.size,
        },
        "documents": {
            "adaptation_pool": len(pool_documents),
            "cross_engine_development": len(evaluation_documents),
        },
        "adaptation_pool_prevalence": {
            "beneficial": float(pool.beneficial.mean()),
            "harmful": float(pool.harmful.mean()),
            "note": (
                "the price of a random label. A budget of N rows buys N * this many beneficial "
                "examples in expectation, and the endpoint is a recall over beneficial rows, so "
                "two engines with the same nominal budget are not running the same experiment."
            ),
        },
        "oracle_budget": {
            "fit_rows": int(oracle_fold.fit.size),
            "calibration_rows": int(oracle_fold.source_cal.size),
            "note": (
                "what the in-engine oracle spends. Its fitting set IS this fold's adaptation "
                "pool, so the budget axis ends at the oracle rather than merely approaching it; "
                "the oracle additionally spends the held-out engine's CALIBRATION rows, which no "
                "SGV6 arm may touch, and that difference is recorded rather than absorbed."
            ),
        },
        "adapter_input": {
            "components": int(components),
            "columns": ["logit_p_harm", "logit_p_benefit"]
            + [f"pc_{i + 1}" for i in range(components)],
            "explained_variance_ratio": float(projection.explained_variance_ratio_.sum()),
            "fitted_on": "fit engines, TRAIN documents",
        },
    }
    return BaseFold(
        held_out=held_out,
        fold=fold,
        selected=dict(selection),
        lambda_=lambda_,
        model=model,
        extended=extended,
        design=design,
        projection=projection,
        scaler=scaler,
        pool=pool,
        evaluation=blocks["evaluation"],
        calibration=blocks["calibration"],
        identity={
            "max_abs_difference_from_published_sgv5": difference,
            "rows": int(fold.eval.size),
        },
        diagnostics=diagnostics,
    )


# ------------------------------------------------------------------ SGV6-B: acquisition


def acquisition_order(pool: TargetPool, rule: str, threshold: float, seed: int) -> np.ndarray:
    """Positions in the pool, in the order a human would be asked to label them.

    Every rule returns one complete ordering and every budget is a PREFIX of it, so the
    label-efficiency curve is a pure budget increase and the N = 250 experiment contains the
    N = 100 experiment. That also fixes what this stage can and cannot claim about active
    learning: the ordering is computed once from the frozen zero-shot model and is never
    revised as labels arrive, so these curves are a lower bound on what a sequential
    acquisition loop could deliver, not an estimate of it.

    No rule may read a target label. `random` reads nothing; the other three read only the
    frozen model's own outputs on unlabelled target rows, which is information an operator has
    before annotating anything.
    """
    size = pool.size
    if rule == RANDOM:
        return np.asarray(np.random.default_rng(seed).permutation(size), dtype=int)
    if rule == UNCERTAINTY:
        # Closest to the operating point the frozen policy would actually deploy at. A row far
        # inside either region is one the model is already sure about and a label there buys
        # little; the decision boundary is where a label changes an accept.
        key = np.abs(pool.utility - threshold)
        return np.asarray(np.lexsort((np.arange(size), key)), dtype=int)
    if rule == DIVERSITY:
        return _farthest_first(pool.adapter_input)
    if rule == EXPECTED_HARM_REDUCTION:
        # Two factors, each doing one job. p(1-p) is the variance of the harm indicator under
        # the frozen model -- how much there is to learn about this row -- and the kernel is
        # how much learning it could change, measured in the same units as the decision. It is
        # a proxy for expected value of information and is reported as one; the exact quantity
        # would require integrating over the posterior the acquisition does not yet have.
        spread = float(np.median(np.abs(pool.utility - np.median(pool.utility)))) or 1.0
        learnable = pool.p_harm * (1.0 - pool.p_harm)
        value = learnable * np.exp(-np.abs(pool.utility - threshold) / spread)
        return np.asarray(np.lexsort((np.arange(size), -value)), dtype=int)
    raise PhaseError(f"unknown acquisition rule {rule!r}")


def _farthest_first(points: np.ndarray) -> np.ndarray:
    """Greedy farthest-point traversal, started at the row nearest the pool centroid.

    Deterministic, prefix-nested, and it spends the budget on covering the target engine's own
    representation rather than on whatever the frozen model happens to be unsure about. The
    start is the centroid's nearest row rather than a random row so the ordering does not
    depend on a seed the other rules do not have.
    """
    centre = points.mean(axis=0)
    distance = np.linalg.norm(points - centre, axis=1)
    first = int(np.argmin(distance))
    order = [first]
    nearest = np.linalg.norm(points - points[first], axis=1)
    nearest[first] = -np.inf
    for _ in range(points.shape[0] - 1):
        pick = int(np.argmax(nearest))
        order.append(pick)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[pick], axis=1))
        nearest[pick] = -np.inf
    return np.asarray(order, dtype=int)


# ------------------------------------------------------------------ the adapter families


@dataclass(slots=True)
class Budget:
    """One drawn budget: which rows, what they cost, and how they split for inner validation.

    `folds` is a grouped k-fold over the budget's own DOCUMENTS. Grouping matters more here
    than anywhere else in the project: a budget of 50 rows can easily contain eight candidates
    from one receipt, and an ungrouped split would let a page appear on both sides of the inner
    validation and make every adapter look better than it is at exactly the budgets where the
    stage's claim is most fragile.
    """

    positions: np.ndarray
    documents: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    p_harm: np.ndarray
    p_benefit: np.ndarray
    adapter_input: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    n_documents: int

    @property
    def size(self) -> int:
        return int(self.positions.size)


def draw_budget(pool: TargetPool, order: np.ndarray, size: int) -> Budget:
    positions = order[:size]
    documents = pool.documents[positions]
    groups = sorted(set(documents.tolist()))
    k = min(INNER_FOLDS, len(groups))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if k >= 2:
        assignment = {name: i % k for i, name in enumerate(groups)}
        block = np.array([assignment[name] for name in documents.tolist()])
        for i in range(k):
            held = np.flatnonzero(block == i)
            rest = np.flatnonzero(block != i)
            if held.size and rest.size:
                folds.append((rest, held))
    return Budget(
        positions=positions,
        documents=documents,
        harmful=pool.harmful[positions],
        beneficial=pool.beneficial[positions],
        p_harm=pool.p_harm[positions],
        p_benefit=pool.p_benefit[positions],
        adapter_input=pool.adapter_input[positions],
        folds=folds,
        n_documents=len(groups),
    )


def _design_row(z: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((z.shape[0], 1)), z])


def _penalised_logistic(
    z: np.ndarray, y: np.ndarray, offset: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """MAP coefficients of `sigmoid(offset + w . [1, z])` under a N(0, 1/tau) prior, and the
    Laplace covariance of w.

    Solved with L-BFGS on an objective that is strictly convex for every tau > 0, so the
    result depends on no step size and no stopping heuristic beyond the tolerance. The prior
    is centred on ZERO CORRECTION, not on zero log-odds: at large tau the arm returns the
    frozen model, which is the behaviour a few-shot method has to have when the few shots say
    nothing.
    """
    from scipy.optimize import minimize

    X = _design_row(z)
    if not np.isfinite(tau):
        return np.zeros(X.shape[1]), np.zeros((X.shape[1], X.shape[1]))
    target = np.asarray(y, dtype=float)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + X @ w
        # log(1 + exp(eta)) computed stably; the gradient is the usual (p - y) X.
        loss = float(np.sum(np.logaddexp(0.0, eta) - target * eta) + 0.5 * tau * float(w @ w))
        gradient = X.T @ (_sigmoid(eta) - target) + tau * w
        return loss, np.asarray(gradient, dtype=float)

    start = np.zeros(X.shape[1])
    result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    w = np.asarray(result.x, dtype=float)
    p = _sigmoid(offset + X @ w)
    hessian = X.T @ (X * (p * (1.0 - p))[:, None]) + tau * np.eye(X.shape[1])
    return w, np.linalg.inv(hessian)


def _log_evidence(
    z: np.ndarray, y: np.ndarray, offset: np.ndarray, tau: float, w: np.ndarray, cov: np.ndarray
) -> float:
    """Laplace approximation to log p(y | tau): fit, prior penalty, and Occam volume."""
    X = _design_row(z)
    eta = offset + X @ w
    target = np.asarray(y, dtype=float)
    log_likelihood = float(np.sum(target * eta - np.logaddexp(0.0, eta)))
    dimension = X.shape[1]
    sign, log_det_cov = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    return float(
        log_likelihood
        - 0.5 * tau * float(w @ w)
        + 0.5 * dimension * float(np.log(tau))
        + 0.5 * float(log_det_cov)
    )


class Family:
    """One adapter family: a fit that does not depend on the setting, and a setting grid.

    Splitting the two is what keeps the inner cross-validation affordable and, more
    importantly, what keeps it honest: for the blend families the underlying model is fitted
    once per inner fold and every setting is evaluated against the SAME fitted object, so the
    grid measures the amount of correction and nothing else.
    """

    name: str = ""
    settings: tuple[float, ...] = ()
    identity: float = 0.0
    # How much adaptation a setting represents, for the tie-break that prefers less of it.
    ascending: bool = True
    # A family that re-picks the harm-aversion weight rather than transforming a head. Its
    # setting is a lambda, so `fit_arm` reads the grid and the identity from the fold.
    is_lambda: bool = False

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        raise NotImplementedError

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def amount(self, setting: float) -> float:
        if self.ascending:
            return float(setting)
        # A descending family is indexed by a prior precision, so more precision is LESS
        # adaptation, and an infinite precision is none at all: it must win every tie.
        return -float(np.log10(max(setting, 1e-12)))


class ZeroShot(Family):
    name = ZERO_SHOT
    settings = (0.0,)
    identity = 0.0

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        return None

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


class BlendFamily(Family):
    """A family whose fitted map is blended with the frozen one in log-odds space.

    Blending in log-odds rather than probability is the choice that makes the identity element
    exact for every member: alpha = 0 returns `logit(p)` unchanged, so the arm returns SGV5's
    score to the last bit rather than to within a rounding of it.
    """

    settings = ALPHA_GRID
    identity = 0.0
    ascending = True

    def mapped(self, core: Any, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        if core is None or setting == 0.0:
            return np.asarray(p, dtype=float)
        base = _logit(p)
        return _sigmoid(base + setting * (_logit(self.mapped(core, p, z)) - base))


class PlattFamily(BlendFamily):
    """`sigmoid(a * logit(p) + b)`, the two-parameter recalibration the brief names first."""

    name = A1_PLATT

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        if np.unique(y).size < 2:
            return None
        # A ridge of 1e-6 on (a, b) only keeps a separable budget from running the coefficients
        # to infinity; it is six orders below the smallest penalty the A2 grid ever selects and
        # cannot be doing the family's work for it.
        feature = _logit(p)[:, None]
        w, _ = _penalised_logistic(feature, y, np.zeros(len(p)), 1e-6)
        return w

    def mapped(self, core: Any, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        w = np.asarray(core, dtype=float)
        return _sigmoid(w[0] + w[1] * _logit(p))


class TemperatureFamily(BlendFamily):
    """`sigmoid(logit(p) / T)`: one parameter, so it can move sharpness but not location."""

    name = A1_TEMPERATURE

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        from scipy.optimize import minimize_scalar

        if np.unique(y).size < 2:
            return None
        base, target = _logit(p), np.asarray(y, dtype=float)

        def loss(log_t: float) -> float:
            eta = base / float(np.exp(log_t))
            return float(np.sum(np.logaddexp(0.0, eta) - target * eta))

        result = minimize_scalar(loss, bounds=(-4.0, 4.0), method="bounded")
        return float(np.exp(result.x))

    def mapped(self, core: Any, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(p) / float(core))


class IsotonicFamily(BlendFamily):
    """A free monotone map. The most expressive recalibration and the most label-hungry."""

    name = A1_ISOTONIC

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        from sklearn.isotonic import IsotonicRegression

        if np.unique(y).size < 2:
            return None
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        )

    def mapped(self, core: Any, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(core.predict(np.asarray(p, dtype=float)), dtype=float)


class BoostedFamily(BlendFamily):
    """A gradient-boosted correction over the adapter input, blended with the frozen head."""

    name = A2_BOOSTED

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        if np.unique(y).size < 2 or len(y) < 2 * INNER_FOLDS:
            return None
        model = HistGradientBoostingClassifier(
            max_iter=c5.BOOSTING_ITERATIONS, random_state=pilot.FIT_SEED
        ).fit(z, np.asarray(y, dtype=int))
        return model

    def mapped(self, core: Any, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        classes = list(core.classes_)
        if 1 not in classes:  # pragma: no cover - guarded by the single-class check above
            return np.asarray(p, dtype=float)
        return np.asarray(core.predict_proba(z)[:, classes.index(1)], dtype=float)


class LogisticAdapter(Family):
    """`logit(p) + w . [1, z]`: the frozen score as an offset, a small correction on top.

    The offset parameterisation is the whole design. A logistic regression that took `logit(p)`
    as a FEATURE would have to relearn the coefficient 1 from the budget and would be worse
    than the frozen model at small N; as an offset, the coefficient is fixed at 1 and the
    budget is spent entirely on the correction, whose prior mean is zero.
    """

    name = A2_LOGISTIC
    settings = TAU_GRID
    identity = TAU_GRID[0]
    ascending = False

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        if np.unique(y).size < 2:
            return None
        offset = _logit(p)
        return {tau: _penalised_logistic(z, y, offset, tau)[0] for tau in TAU_GRID}

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        # A zero correction returns `p` itself rather than sigmoid(logit(p)). The round trip
        # through the log-odds is accurate to about 1e-16 and that is not the same thing as
        # exact: the identity element has to be an identity, or "budget zero IS SGV5" becomes
        # a claim about rounding.
        if core is None or not np.any(core[setting]):
            return np.asarray(p, dtype=float)
        return _sigmoid(_logit(p) + _design_row(z) @ core[setting])


class RidgeAdapter(Family):
    """A linear correction to the PROBABILITY, not the log-odds: `p + delta(z)`, clipped.

    Kept as a separate family rather than folded into the logistic one because the two make
    different mistakes at small N. A log-odds correction of fixed size moves a probability near
    zero hardly at all and a probability near a half a great deal; a probability-space
    correction does the opposite. Which of those is the right inductive bias for a harm head
    that lives near zero is an empirical question, so both are on the grid.

    It is also the one family that stays defined when the budget carries a single label class.
    A logistic fit on ten rows with no harmful example diverges and returns the frozen head; a
    least-squares fit on the residual does not, and reads those ten rows as evidence that the
    frozen head over-predicts harm here. That is real information and it is not discarded, but
    it is why this family alone can move a head the others leave alone.
    """

    name = A2_RIDGE
    settings = TAU_GRID
    identity = TAU_GRID[0]
    ascending = False

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        X = _design_row(z)
        residual = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
        gram = X.T @ X
        cross = X.T @ residual
        return {
            tau: (
                np.zeros(X.shape[1])
                if not np.isfinite(tau)
                else np.linalg.solve(gram + tau * np.eye(X.shape[1]), cross)
            )
            for tau in TAU_GRID
        }

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        if core is None or not np.any(core[setting]):
            return np.asarray(p, dtype=float)
        return np.clip(np.asarray(p, dtype=float) + _design_row(z) @ core[setting], 0.0, 1.0)


class BayesAdapter(Family):
    """The logistic adapter with the shrinkage chosen by evidence rather than by validation.

    Two things differ from `a2_logistic`, and they are the two the brief asks about. The prior
    precision is selected by maximising the Laplace approximation to the marginal likelihood,
    which spends no validation rows; and the correction is integrated over its own posterior
    rather than taken at the mode.

    The predictive step is applied to the CORRECTION only:

        p_new = sigmoid( logit(p_base) + (mu . x) / sqrt(1 + pi * s^2 / 8) )

    and not to the whole linear predictor. The usual probit approximation would regress an
    uncertain prediction towards one half, which is the right target when the prior belief is
    "no information" and the wrong one here, where the prior belief is "no correction". Under
    this form an uncertain correction decays towards the frozen model, which is the behaviour
    the family is being tested for. It is an approximation and is labelled as one.
    """

    name = A3_BAYES
    settings = (1.0,)
    identity = 1.0

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        if np.unique(y).size < 2:
            return None
        offset = _logit(p)
        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for tau in EVIDENCE_GRID:
            w, cov = _penalised_logistic(z, y, offset, tau)
            evidence = _log_evidence(z, y, offset, tau, w, cov)
            if best is None or evidence > best[0]:
                best = (evidence, tau, w, cov)
        assert best is not None
        return {"tau": best[1], "w": best[2], "cov": best[3], "log_evidence": best[0]}

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        if core is None or not np.any(core["w"]):
            return np.asarray(p, dtype=float)
        X = _design_row(z)
        mean = X @ core["w"]
        variance = np.einsum("ij,jk,ik->i", X, core["cov"], X)
        shrunk = mean / np.sqrt(1.0 + np.pi * np.maximum(variance, 0.0) / 8.0)
        return _sigmoid(_logit(p) + shrunk)


class LambdaRetune(Family):
    """Re-pick the harm-aversion weight on the target budget. One parameter, no features.

    This is the cheapest thing a target label can buy and it is here to bound everything else.
    SGV5 chooses lambda by an inner leave-one-engine-out over the SOURCE engines, so the
    weight that balances `P(benefit)` against `P(harm)` is fitted to how those engines fail. If
    a handful of target labels moving that one number recovers most of what a ten-parameter
    adapter recovers, then the transfer failure SGV5 diagnosed is a miscalibrated trade-off and
    not a representation problem, and the stage should say so rather than credit the adapter.

    The heads are untouched: `apply` is the identity, and only the weight combining them moves.
    """

    name = A1_LAMBDA
    is_lambda = True
    settings = LAMBDA_GRID

    def core(self, p: np.ndarray, y: np.ndarray, z: np.ndarray) -> Any:
        return None

    def apply(self, core: Any, setting: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


FAMILIES: dict[str, Family] = {
    family.name: family
    for family in (
        ZeroShot(),
        LambdaRetune(),
        PlattFamily(),
        TemperatureFamily(),
        IsotonicFamily(),
        LogisticAdapter(),
        RidgeAdapter(),
        BoostedFamily(),
        BayesAdapter(),
    )
}


# ------------------------------------------------------------------ inner target validation


@dataclass(slots=True)
class Adapted:
    """One adapter family fitted at one budget, with the setting the budget itself chose.

    `lambda_` travels with the arm because one family re-picks it. Every other family inherits
    SGV5's frozen weight unchanged, so a difference between two arms is never partly a
    difference in harm aversion that nobody declared.
    """

    arm: str
    setting: float
    lambda_: float
    harm_core: Any
    benefit_core: Any
    oof_utility: np.ndarray
    inner: dict[str, Any]

    def score(self, p_harm: np.ndarray, p_benefit: np.ndarray, z: np.ndarray) -> np.ndarray:
        family = FAMILIES[self.arm]
        harm = family.apply(self.harm_core, self.setting, p_harm, z)
        benefit = family.apply(self.benefit_core, self.setting, p_benefit, z)
        return np.asarray(benefit - self.lambda_ * harm, dtype=float)


def _inner_score(
    utility: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray
) -> tuple[float, float]:
    """The endpoint on the budget's own out-of-fold predictions, plus a smoother tie-break.

    The endpoint comes first because it is what the stage is optimising and a proxy that
    disagrees with it would silently change the objective. It is also extremely coarse at
    N = 10 -- a budget with one beneficial row admits exactly two values -- so a ranking AUC
    breaks the ties it leaves, and where both are tied the third key prefers less adaptation.
    """
    if beneficial.sum() == 0:
        return 0.0, float("nan")
    endpoint = float(
        achievable_repair_recall(utility, harmful, beneficial)[PRIMARY_KEY]["repair_recall"]
    )
    discrimination = (
        float(roc_auc(utility, beneficial.astype(float)))
        if beneficial.any() and (~beneficial).any()
        else float("nan")
    )
    return endpoint, discrimination


def fit_arm(arm: str, budget: Budget, lambda_: float) -> Adapted:
    """Fit one family on the budget and let the budget's own grouped folds pick the setting.

    Nothing outside the budget is read. The evaluation rows are not touched, the held-out
    engine's calibration rows are not touched, and the source engines contribute only the
    frozen model the family sits on top of.
    """
    family = FAMILIES[arm]
    z = budget.adapter_input
    harm_core = family.core(budget.p_harm, budget.harmful, z)
    benefit_core = family.core(budget.p_benefit, budget.beneficial, z)
    settings = family.settings
    grid: dict[str, dict[str, float]] = {}
    oof: dict[float, np.ndarray] = {}

    def amount(setting: float) -> float:
        if not family.is_lambda:
            return family.amount(setting)
        # Distance from the weight SGV5 already chose for this fold, in the units the grid is
        # spaced on, so the tie-break prefers leaving the frozen trade-off alone.
        return abs(float(np.log10(setting)) - float(np.log10(lambda_)))

    if budget.folds:
        cores = [
            (
                train,
                held,
                family.core(budget.p_harm[train], budget.harmful[train], z[train]),
                family.core(budget.p_benefit[train], budget.beneficial[train], z[train]),
            )
            for train, held in budget.folds
        ]
        for setting in settings:
            utility = np.full(budget.size, np.nan)
            weight = setting if family.is_lambda else lambda_
            for _, held, harm_fold, benefit_fold in cores:
                harm = family.apply(harm_fold, setting, budget.p_harm[held], z[held])
                benefit = family.apply(benefit_fold, setting, budget.p_benefit[held], z[held])
                utility[held] = benefit - weight * harm
            covered = np.isfinite(utility)
            endpoint, discrimination = _inner_score(
                utility[covered], budget.harmful[covered], budget.beneficial[covered]
            )
            grid[f"s_{setting:g}"] = {
                "inner_repair_recall": endpoint,
                "inner_auc_beneficial": discrimination,
                "rows_scored": int(covered.sum()),
            }
            oof[setting] = utility

    def rank(setting: float) -> tuple[float, float, float]:
        cell = grid.get(f"s_{setting:g}")
        if cell is None:
            return (float("-inf"), float("-inf"), 0.0)
        auc = cell["inner_auc_beneficial"]
        return (
            cell["inner_repair_recall"],
            -1.0 if np.isnan(auc) else auc,
            -amount(setting),
        )

    selected = (
        max(settings, key=rank) if grid else (lambda_ if family.is_lambda else family.identity)
    )
    return Adapted(
        arm=arm,
        setting=float(selected),
        lambda_=float(selected) if family.is_lambda else float(lambda_),
        harm_core=harm_core,
        benefit_core=benefit_core,
        oof_utility=oof.get(float(selected), np.full(budget.size, np.nan)),
        inner={
            "selected_setting": float(selected),
            "grid": grid,
            "inner_folds": len(budget.folds),
            "inner_documents": budget.n_documents,
            "harm_head_fittable": harm_core is not None,
            "benefit_head_fittable": benefit_core is not None,
            "inner_validation_available": bool(grid),
            "budget_beneficial": int(budget.beneficial.sum()),
            "budget_harmful": int(budget.harmful.sum()),
        },
    )


def select_arm(fitted: dict[str, Adapted], budget: Budget) -> tuple[str, dict[str, Any]]:
    """Which family the budget itself prefers. Ties resolve towards not adapting at all.

    `zero_shot` is a competitor here, not a fallback bolted on afterwards, and it is first in
    the tie-break order. That is the property that makes a measured SGV6 gain interpretable: a
    budget carrying no signal produces ties, ties resolve to `zero_shot`, and the arm returns
    SGV5's score exactly rather than a perturbation of it.
    """
    table: dict[str, dict[str, float]] = {}
    for arm, adapted in fitted.items():
        covered = np.isfinite(adapted.oof_utility)
        if not covered.any():
            table[arm] = {"inner_repair_recall": 0.0, "inner_auc_beneficial": float("nan")}
            continue
        endpoint, discrimination = _inner_score(
            adapted.oof_utility[covered], budget.harmful[covered], budget.beneficial[covered]
        )
        table[arm] = {
            "inner_repair_recall": endpoint,
            "inner_auc_beneficial": discrimination,
        }

    def rank(arm: str) -> tuple[float, float, int]:
        cell = table[arm]
        auc = cell["inner_auc_beneficial"]
        return (
            cell["inner_repair_recall"],
            -1.0 if np.isnan(auc) else auc,
            -ADAPTER_ARMS.index(arm),
        )

    best = max(table, key=rank)
    return best, {
        "selected_arm": best,
        "per_arm": table,
        "tie_break": (
            "highest inner repair recall on the budget's pooled out-of-fold predictions, then "
            "the ranking AUC, then the earlier arm in a fixed order whose first entry is the "
            "unadapted model. Every key is computed inside the budget; no evaluation row and "
            "no target row outside the budget is read."
        ),
    }


# ------------------------------------------------------------------ the freezing ablation


@dataclass(slots=True)
class Refit:
    """A model re-estimated with target labels rather than corrected by an adapter."""

    name: str
    model: c5.Reliability | None
    note: dict[str, Any]

    def score(self, design: dg.Design, index: np.ndarray, lambda_: float) -> np.ndarray:
        if self.model is None:
            # An unfittable model accepts nothing, and -inf is how this repository spells that.
            # Returning the frozen score instead would credit the ablation with the base
            # model's performance and hide exactly the failure the ablation exists to show.
            return np.full(index.size, -np.inf)
        return self.model.utility(design, index, lambda_)


def _fit_weighted(
    design: dg.Design,
    columns: list[int],
    rows: np.ndarray,
    weight: np.ndarray,
    kind: str,
    representation: str,
    calibration: np.ndarray | None,
    calibration_scores: tuple[np.ndarray, np.ndarray] | None,
) -> c5.Reliability:
    """SGV5's two-head construction with sample weights, and nothing else changed.

    The model class, its hyperparameters and the calibrator are the ones SGV5 selected for this
    fold. Only the fitting rows and their weights differ, which is what makes the freezing
    ablation a statement about where the target labels enter rather than about model choice.
    """
    from sklearn.preprocessing import StandardScaler

    from ocr_risk.calibrate.calibrators import build_calibrator

    block = design.matrix[np.ix_(rows, columns)]
    scaler = StandardScaler().fit(block)
    scaled = np.asarray(scaler.transform(block), dtype=float)
    harmful = design.harmful[rows].astype(int)
    outcome = np.full(rows.size, policy.CLASS_NEUTRAL, dtype=np.int64)
    outcome[design.harmful[rows]] = policy.CLASS_HARM
    outcome[design.beneficial[rows]] = policy.CLASS_BENEFIT

    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        def make() -> Any:
            return LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
            )

    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        balance = "balanced" if kind == c5.BOOSTED_BALANCED else None

        def make() -> Any:
            return HistGradientBoostingClassifier(
                max_iter=c5.BOOSTING_ITERATIONS,
                class_weight=balance,
                random_state=pilot.FIT_SEED,
            )

    fitted = c5.Reliability(
        kind=kind,
        representation=representation,
        columns=tuple(columns),
        scaler=scaler,
        harm=make().fit(scaled, harmful, sample_weight=weight),
        outcome=make().fit(scaled, outcome, sample_weight=weight),
        harm_calibrator=None,
        benefit_calibrator=None,
        weights=None,
    )
    harm_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    benefit_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    if calibration_scores is not None:
        assert calibration is not None
        harm_calibrator.fit(calibration_scores[0], design.harmful[calibration].astype(float))
        benefit_calibrator.fit(calibration_scores[1], design.beneficial[calibration].astype(float))
    else:
        assert calibration is not None
        raw = fitted.raw(design, calibration)
        harm_calibrator.fit(raw["harm_raw"], design.harmful[calibration].astype(float))
        benefit_calibrator.fit(raw["benefit_raw"], design.beneficial[calibration].astype(float))
    fitted.harm_calibrator = harm_calibrator
    fitted.benefit_calibrator = benefit_calibrator
    return fitted


def joint_refit(bf: BaseFold, rows: np.ndarray, parity: bool) -> Refit:
    """Ablation B: re-estimate the whole model on the source rows plus the target budget.

    Two weightings, because the choice is not neutral and tuning it on anything would cost
    target labels the budget has not got. `naive` counts a target row once, which is what
    pooling does by default and which at N = 1000 against 25,000 source rows is a four percent
    perturbation. `parity` gives the budget the weight of one source engine, which is the
    strongest defensible claim that the target should count as much as the engines already
    seen. Both are fixed rules stated in advance, neither is selected.
    """
    columns = bf.extended.columns(bf.selected["representation"])
    combined = np.concatenate([bf.fold.fit, rows])
    weight = np.ones(combined.size)
    per_row = 1.0
    if parity:
        per_row = float(bf.fold.fit.size) / float(max(len(bf.fold.train_engines), 1) * rows.size)
        weight[bf.fold.fit.size :] = per_row
    model = _fit_weighted(
        bf.design,
        columns,
        combined,
        weight,
        bf.selected["model"],
        bf.selected["representation"],
        bf.fold.source_cal,
        None,
    )
    return Refit(
        name=FREEZE_JOINT_PARITY if parity else FREEZE_JOINT_NAIVE,
        model=model,
        note={
            "fit_rows": int(combined.size),
            "target_rows": int(rows.size),
            "target_row_weight": per_row,
            "calibrated_on": "the fit engines' calibration documents, as SGV5 does",
        },
    )


def target_only_refit(bf: BaseFold, rows: np.ndarray, budget: Budget) -> Refit:
    """Ablation C: the same model class fitted on the target budget and nothing else.

    Its calibrators are fitted on the budget's own grouped out-of-fold raw scores, so the
    ablation is not handed an in-sample calibration advantage the other arms do not have. A
    budget whose harm labels are all one value cannot fit the head at all; that is recorded and
    the arm accepts nothing, which is the truthful reading of "this model does not exist yet".
    """
    columns = bf.extended.columns(bf.selected["representation"])
    if np.unique(bf.design.harmful[rows]).size < 2:
        return Refit(
            name=FREEZE_TARGET_ONLY,
            model=None,
            note={"fittable": False, "reason": "the budget's harm labels are a single class"},
        )
    if not budget.folds:
        return Refit(
            name=FREEZE_TARGET_ONLY,
            model=None,
            note={"fittable": False, "reason": "the budget spans too few documents to calibrate"},
        )
    harm_oof = np.full(rows.size, np.nan)
    benefit_oof = np.full(rows.size, np.nan)
    for train, held in budget.folds:
        if np.unique(bf.design.harmful[rows[train]]).size < 2:
            continue
        inner = _fit_weighted(
            bf.design,
            columns,
            rows[train],
            np.ones(train.size),
            bf.selected["model"],
            bf.selected["representation"],
            rows[train],
            None,
        )
        raw = inner.raw(bf.design, rows[held])
        harm_oof[held] = raw["harm_raw"]
        benefit_oof[held] = raw["benefit_raw"]
    covered = np.isfinite(harm_oof) & np.isfinite(benefit_oof)
    if covered.sum() < 2 or np.unique(bf.design.harmful[rows[covered]]).size < 2:
        return Refit(
            name=FREEZE_TARGET_ONLY,
            model=None,
            note={"fittable": False, "reason": "no out-of-fold calibration rows survived"},
        )
    model = _fit_weighted(
        bf.design,
        columns,
        rows,
        np.ones(rows.size),
        bf.selected["model"],
        bf.selected["representation"],
        rows[covered],
        (harm_oof[covered], benefit_oof[covered]),
    )
    return Refit(
        name=FREEZE_TARGET_ONLY,
        model=model,
        note={
            "fittable": True,
            "fit_rows": int(rows.size),
            "calibration_rows": int(covered.sum()),
            "calibrated_on": "the budget's own grouped out-of-fold raw scores",
        },
    )


# ------------------------------------------------------------------ the score table


def cell_name(arm: str, budget: int, acquisition: str, seed: int) -> str:
    return f"sgv6__{arm}__n{budget}__{acquisition}__s{seed}"


def parse_cell(name: str) -> dict[str, Any]:
    arm, budget, acquisition, seed = name[len("sgv6__") :].rsplit("__", 3)
    return {
        "arm": arm,
        "budget": int(budget[1:]),
        "acquisition": acquisition,
        "seed": int(seed[1:]),
    }


def _threshold_block(score: np.ndarray, harmful: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(score)
    if not finite.any() or harmful[finite].size == 0:
        return {controller: {"tau": float("inf"), "feasible": False} for controller in CONTROLLERS}
    return {
        controller: {
            key: value
            for key, value in select_threshold(
                score[finite],
                harmful[finite],
                PRIMARY_EPSILON,
                delta=DELTA,
                controller=controller,
            )
            .as_dict()
            .items()
            if key in ("tau", "feasible", "coverage", "observed_risk", "n_accepted")
        }
        for controller in CONTROLLERS
    }


def _draws() -> list[tuple[str, int]]:
    """Which (acquisition, seed) pairs are drawn. Random is repeated; the rest are not random.

    The three informed rules are deterministic functions of the frozen model, so a seed axis
    would repeat the same draw five times and buy a false impression of stability. Random gets
    five seeds because one draw of ten rows from a pool that is three percent beneficial is not
    an estimate of anything.
    """
    return [(RANDOM, seed) for seed in range(SEED_REPEATS)] + [
        (rule, 0) for rule in ACQUISITIONS if rule != RANDOM
    ]


def run_adapt() -> int:
    """Fit every model this stage uses and write the row-level table the rest of it reads."""
    started = time.monotonic()
    base, signatures, retrieval_columns = load_base()
    selection = frozen_selection()
    frozen = load_frozen_scores()

    frames: list[pd.DataFrame] = []
    record: dict[str, Any] = {}
    for held_out in base.engines:
        bf = build_base_fold(
            base, held_out, signatures, retrieval_columns, selection[held_out], frozen
        )
        evaluation, calibration, pool = bf.evaluation, bf.calibration, bf.pool
        identifiers = bf.design.meta["candidate_id"].astype(str).to_numpy()[bf.fold.eval]

        zero_threshold = _threshold_block(calibration["utility"], calibration["harmful"])
        # Where the frozen policy would actually cut. When no threshold on the source
        # calibration rows can certify the bound the deployed cut does not exist, and the
        # acquisition rules fall back to the pool's own median so that "near the decision" stays
        # defined; which of the two was used is recorded rather than silently substituted.
        feasible = bool(zero_threshold["empirical"]["feasible"])
        certified = float(zero_threshold["empirical"]["tau"])
        operating_point = certified if feasible else float(np.median(pool.utility))

        columns: dict[str, np.ndarray] = {}
        cells: dict[str, Any] = {}
        for acquisition, seed in _draws():
            order = acquisition_order(
                pool, acquisition, operating_point, _stable_seed(held_out, acquisition, str(seed))
            )
            primary_draw = acquisition == RANDOM and seed == 0
            for size in NONZERO_BUDGETS:
                budget = draw_budget(pool, order, size)
                rows = pool.index[budget.positions]
                # Every family is fitted under every acquisition, because the choice of family
                # is part of the method and a rule that changed which families were available
                # would confound the acquisition comparison. Only the columns differ: the
                # informed rules store the selected arm, which is what SGV6-B compares.
                fitted = {arm: fit_arm(arm, budget, bf.lambda_) for arm in ADAPTER_ARMS}
                chosen, note = select_arm(fitted, budget)

                produced: dict[str, np.ndarray] = {}
                if acquisition == RANDOM:
                    for arm, adapted in fitted.items():
                        produced[arm] = adapted.score(
                            evaluation["p_harm"],
                            evaluation["p_benefit"],
                            evaluation["adapter_input"],
                        )
                produced[SELECTED_ARM] = fitted[chosen].score(
                    evaluation["p_harm"],
                    evaluation["p_benefit"],
                    evaluation["adapter_input"],
                )

                if primary_draw:
                    permuted_labels = _permuted_budget(budget, held_out, size)
                    permuted_fitted = {
                        arm: fit_arm(arm, permuted_labels, bf.lambda_) for arm in ADAPTER_ARMS
                    }
                    permuted_choice, _ = select_arm(permuted_fitted, permuted_labels)
                    produced[PERMUTED_ARM] = permuted_fitted[permuted_choice].score(
                        evaluation["p_harm"],
                        evaluation["p_benefit"],
                        evaluation["adapter_input"],
                    )
                    produced[FREEZE_ADAPTER] = produced[SELECTED_ARM]
                    refits = [
                        joint_refit(bf, rows, parity=False),
                        joint_refit(bf, rows, parity=True),
                        target_only_refit(bf, rows, budget),
                    ]
                    for refit in refits:
                        produced[refit.name] = refit.score(bf.design, bf.fold.eval, bf.lambda_)
                    freeze_notes = {refit.name: refit.note for refit in refits}
                else:
                    refits = []
                    freeze_notes = {}
                    permuted_choice = ""

                # Both placements, for every arm that has a threshold at all. The permuted
                # control is a red-team arm and is never deployed, so it is not given one.
                thresholds: dict[str, Any] = {}
                refit_by_name = {refit.name: refit for refit in refits}
                for arm in produced:
                    if arm == PERMUTED_ARM:
                        continue
                    if arm in FAMILIES or arm in (SELECTED_ARM, FREEZE_ADAPTER):
                        adapted = fitted[arm] if arm in FAMILIES else fitted[chosen]
                        source = adapted.score(
                            calibration["p_harm"],
                            calibration["p_benefit"],
                            calibration["adapter_input"],
                        )
                        target_oof = adapted.oof_utility
                    else:
                        refit = refit_by_name[arm]
                        source = refit.score(bf.design, bf.fold.source_cal, bf.lambda_)
                        target_oof = np.full(budget.size, np.nan)
                    covered = np.isfinite(target_oof)
                    thresholds[arm] = {
                        "source_calibrated": _threshold_block(source, calibration["harmful"]),
                        "target_calibrated": (
                            _threshold_block(target_oof[covered], budget.harmful[covered])
                            if covered.any()
                            else {c: {"tau": float("inf"), "feasible": False} for c in CONTROLLERS}
                        ),
                    }

                for arm, score in produced.items():
                    columns[cell_name(arm, size, acquisition, seed)] = score

                cells[cell_name(SELECTED_ARM, size, acquisition, seed)] = {
                    "budget": size,
                    "acquisition": acquisition,
                    "seed": seed,
                    "rows_drawn": int(budget.size),
                    "documents_drawn": budget.n_documents,
                    "beneficial_in_budget": int(budget.beneficial.sum()),
                    "harmful_in_budget": int(budget.harmful.sum()),
                    "arm_selection": note,
                    "selected_arm_inner": fitted[chosen].inner,
                    "settings_per_arm": {
                        arm: adapted.setting for arm, adapted in sorted(fitted.items())
                    },
                    "lambda_per_arm": {
                        arm: adapted.lambda_ for arm, adapted in sorted(fitted.items())
                    },
                    "permuted_control_arm": permuted_choice,
                    "freezing": freeze_notes,
                    "thresholds": thresholds,
                }

        frame = pd.DataFrame(
            {
                "candidate_id": identifiers,
                "document_id": evaluation["documents"],
                "engine_id": bf.design.meta["engine_id"].to_numpy(str)[bf.fold.eval],
                "held_out_engine": held_out,
                "evaluation_mode": CROSS_ENGINE,
                "is_harmful": evaluation["harmful"],
                "beneficial": evaluation["beneficial"],
            }
        )
        reference = frozen[frozen["held_out_engine"] == held_out].set_index("candidate_id")
        # Assembled in one concatenation rather than four hundred insertions: the column order
        # is the same either way, and pandas warns about the fragmentation of the other form.
        frames.append(
            pd.concat(
                [
                    frame,
                    pd.DataFrame(
                        {
                            name: reference.loc[identifiers, name].to_numpy(dtype=float)
                            for name in BASELINE_COLUMNS
                        }
                    ),
                    pd.DataFrame({f"arm__{name}": columns[name] for name in sorted(columns)}),
                ],
                axis=1,
            )
        )

        record[held_out] = {
            "held_out_engine": held_out,
            "train_engines": list(bf.fold.train_engines),
            "sgv5_selection": bf.selected,
            "sgv5_identity_check": bf.identity,
            "zero_shot_thresholds": zero_threshold,
            "acquisition_operating_point": {
                "value": operating_point,
                "source": (
                    "the source-calibrated empirical threshold"
                    if feasible
                    else "the adaptation pool's median utility, because no source threshold "
                    "certified the bound"
                ),
            },
            "diagnostics": bf.diagnostics,
            "cells": cells,
        }
        print(
            f"  fold {held_out:11s} pool={pool.size} rows / "
            f"{bf.diagnostics['documents']['adaptation_pool']} documents  "
            f"beneficial={bf.diagnostics['adaptation_pool_prevalence']['beneficial']:.3f}  "
            f"columns={len(columns)}"
        )

    cc._write_parquet_once(SCORES, pd.concat(frames, ignore_index=True))
    cc._write_json_once(
        SELECTION_RECORD,
        {
            "schema_version": "sgv6-selection-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "selection_scope": (
                "the representation, model class and lambda are SGV5's, read from its frozen "
                "policy_selection.json and not re-made here. Everything SGV6 chooses -- the "
                "adapter family, its shrinkage, and the deployment threshold when it is "
                "target-calibrated -- is chosen by grouped cross-validation INSIDE the drawn "
                "budget, so the label budget pays for the selection as well as the fit. No "
                "evaluation row and no target row outside the budget is read by any choice."
            ),
            "development_only": True,
            "synthetic": False,
            "budget_grid": list(BUDGETS),
            "acquisition_rules": list(ACQUISITIONS),
            "seed_repeats": SEED_REPEATS,
            "engine_identity_used_by_method": False,
            "folds": record,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"adapt: {sum(len(f) for f in frames)} rows -> {cc._relative(SCORES)}")
    return 0


def _permuted_budget(budget: Budget, held_out: str, size: int) -> Budget:
    """The same budget with its labels shuffled: the red-team control for the whole procedure.

    If the adaptation machinery -- the families, the inner selection, the shrinkage grids, the
    threshold placement -- can manufacture a gain from a budget that carries no information,
    it will do so here too. A real effect must collapse to the zero-shot arm.
    """
    rng = np.random.default_rng(_stable_seed(held_out, "permuted", str(size)))
    order = rng.permutation(budget.size)
    return Budget(
        positions=budget.positions,
        documents=budget.documents,
        harmful=budget.harmful[order],
        beneficial=budget.beneficial[order],
        p_harm=budget.p_harm,
        p_benefit=budget.p_benefit,
        adapter_input=budget.adapter_input,
        folds=budget.folds,
        n_documents=budget.n_documents,
    )


# ------------------------------------------------------------------ reading the table


@dataclass(slots=True)
class Slice:
    """One fold's evaluation rows and every score column measured on them."""

    held_out: str
    documents: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    scores: dict[str, np.ndarray]

    def arm(self, name: str) -> np.ndarray:
        if name not in self.scores:
            raise PhaseError(f"{self.held_out}: no column {name!r} in the adaptation table")
        return self.scores[name]

    def cell(self, arm: str, budget: int, acquisition: str = RANDOM, seed: int = 0) -> np.ndarray:
        """The score for one (arm, budget, acquisition, seed). Budget zero IS SGV5's column.

        Routing N = 0 to SGV5's own frozen vector rather than to a stored copy is what makes
        every delta in this stage a delta against the published baseline by construction: there
        is no separate zero-budget arm that could drift away from it.
        """
        if budget == 0:
            return self.arm(SGV5_ARM)
        return self.arm(f"arm__{cell_name(arm, budget, acquisition, seed)}")


def load_scores() -> tuple[dict[str, Slice], dict[str, Any]]:
    if not SCORES.is_file():
        raise PhaseError("run --adapt before any experiment")
    frame = pd.read_parquet(SCORES)
    record = cc._read_json(SELECTION_RECORD)
    slices: dict[str, Slice] = {}
    for held_out, block in frame.groupby("held_out_engine", sort=True):
        block = block.reset_index(drop=True)
        slices[str(held_out)] = Slice(
            held_out=str(held_out),
            documents=block["document_id"].to_numpy(str),
            harmful=block["is_harmful"].to_numpy(dtype=bool),
            beneficial=block["beneficial"].to_numpy(dtype=bool),
            scores={
                name: block[name].to_numpy(dtype=np.float64)
                for name in block.columns
                if name.startswith("arm__")
            },
        )
    return slices, record


def _point(score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray) -> dict[str, Any]:
    """The frontier at every epsilon, with abstention reported as abstention."""
    summary = restricted_frontier(score, harmful, beneficial)
    return {
        "rejected_rows": int(score.size - np.count_nonzero(np.isfinite(score))),
        **{
            f"epsilon_{int(e * 100)}": {
                key: summary[f"epsilon_{int(e * 100)}"][key]
                for key in ("repair_recall", "coverage", "n_accepted", "realized_harm_rate")
            }
            for e in EPSILONS
        },
    }


def _recall(score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray) -> float:
    return float(restricted_frontier(score, harmful, beneficial)[PRIMARY_KEY]["repair_recall"])


def _deployed_pair(piece: Slice, score: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    """What the arm delivers at a threshold nobody chose with hindsight.

    Both placements are reported for every arm. The source-calibrated one is SGV5's deployment
    rule unchanged and costs no target label; the target-calibrated one spends the budget's own
    out-of-fold predictions and is the honest question for a stage about label budgets, because
    an operator with N labels would not leave the threshold on the source engines.
    """
    out: dict[str, Any] = {}
    for placement in ("source_calibrated", "target_calibrated"):
        block = thresholds.get(placement, {})
        out[placement] = {
            controller: {
                "tau": block.get(controller, {}).get("tau", float("inf")),
                "feasible_on_calibration": block.get(controller, {}).get("feasible", False),
                **deployed_point(
                    score,
                    piece.harmful,
                    piece.beneficial,
                    float(block.get(controller, {}).get("tau", float("inf"))),
                ),
            }
            for controller in CONTROLLERS
        }
    return out


# ------------------------------------------------------------------ the risk-coverage frontier


def run_curves() -> int:
    """Every arm at every budget, at each epsilon, with the deployment view beside it."""
    started = time.monotonic()
    slices, record = load_scores()
    folds: dict[str, Any] = {}

    for held_out, piece in slices.items():
        cells = record["folds"][held_out]["cells"]
        baselines = {
            name[len("arm__") :]: _point(piece.arm(name), piece.harmful, piece.beneficial)
            for name in BASELINE_COLUMNS
        }
        arms: dict[str, Any] = {}
        for budget in BUDGETS:
            for arm in (*ADAPTER_ARMS, SELECTED_ARM, PERMUTED_ARM, *FREEZE_ARMS):
                for acquisition, seed in _draws():
                    if budget == 0 and (arm != SELECTED_ARM or acquisition != RANDOM or seed):
                        continue
                    key = cell_name(arm, budget, acquisition, seed)
                    column = f"arm__{key}"
                    if budget and column not in piece.scores:
                        continue
                    score = piece.cell(arm, budget, acquisition, seed)
                    entry = _point(score, piece.harmful, piece.beneficial)
                    holder = cells.get(cell_name(SELECTED_ARM, budget, acquisition, seed), {})
                    thresholds = holder.get("thresholds", {}).get(arm)
                    if thresholds is not None:
                        entry["deployed"] = _deployed_pair(piece, score, thresholds)
                    arms[key] = entry
        folds[held_out] = {
            "held_out_engine": held_out,
            "rows": int(piece.harmful.size),
            "documents": len(set(piece.documents.tolist())),
            "total_beneficial": int(piece.beneficial.sum()),
            "baselines": baselines,
            "budget_cells": arms,
        }
        print(f"  fold {held_out:11s} cells={len(arms)}")

    cc._write_json_once(
        CURVE_RESULTS,
        {
            "schema_version": "sgv6-curves-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "primary_endpoint": "repair_recall_at_bounded_harm, imported from Phase 5",
            "epsilon_grid": list(EPSILONS),
            "endpoint_note": (
                "the frontier reads its cut off the evaluation labels, so it measures whether "
                "the RANKING transferred. `deployed` answers the different question of whether "
                "an operator's threshold held, and no claim in this stage rests on one without "
                "the other."
            ),
            "budget_zero_is": "SGV5's published score vector, read from its frozen table",
            "folds": folds,
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"curves: {len(folds)} folds -> {cc._relative(CURVE_RESULTS)}")
    return 0


# ------------------------------------------------------------------ label efficiency


def _seed_curve(piece: Slice, arm: str, budget: int) -> dict[str, Any]:
    """The endpoint at one budget across every random draw, spread and all.

    A single draw of ten rows from a pool that is three percent beneficial is not an estimate,
    and averaging the five draws away would hide the fact. The mean is reported next to the
    range, and where the range is wider than the document-clustered interval the range is the
    honest uncertainty.
    """
    seeds = [
        seed
        for seed in range(SEED_REPEATS)
        if f"arm__{cell_name(arm, budget, RANDOM, seed)}" in piece.scores
    ]
    values = [
        _recall(piece.cell(arm, budget, RANDOM, seed), piece.harmful, piece.beneficial)
        for seed in seeds
    ]
    return {
        "seeds": seeds,
        "per_seed": values,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _spearman(x: list[float], y: list[float]) -> float:
    if len(set(y)) < 2 or len(set(x)) < 2:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def run_efficiency() -> int:
    """SGV6-T1b: does the endpoint rise with the budget, and by how much per label?"""
    started = time.monotonic()
    slices, _ = load_scores()
    folds: dict[str, Any] = {}

    for held_out, piece in slices.items():
        baseline = piece.arm(SGV5_ARM)
        oracle = piece.arm(ORACLE_ARM)
        zero = _recall(baseline, piece.harmful, piece.beneficial)
        ceiling = _recall(oracle, piece.harmful, piece.beneficial)

        curves: dict[str, Any] = {}
        for arm in (*ADAPTER_ARMS, SELECTED_ARM, PERMUTED_ARM):
            available = [
                budget
                for budget in NONZERO_BUDGETS
                if f"arm__{cell_name(arm, budget, RANDOM, 0)}" in piece.scores
            ]
            if not available:
                continue
            per_budget = {f"n_{budget}": _seed_curve(piece, arm, budget) for budget in available}
            means = [per_budget[f"n_{budget}"]["mean"] for budget in available]
            # One rank correlation per random draw. The arms that exist at only one seed --
            # the permutation control, which is a control and not a curve -- get one.
            drawn = len(per_budget[f"n_{available[0]}"]["per_seed"])
            per_seed_rho = [
                _spearman(
                    list(available),
                    [per_budget[f"n_{budget}"]["per_seed"][position] for budget in available],
                )
                for position in range(drawn)
            ]
            steps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
            curves[arm] = {
                "zero_shot": zero,
                "per_budget": per_budget,
                "rank_correlation_of_mean_curve": _spearman(list(available), means),
                "rank_correlation_per_seed": per_seed_rho,
                "rank_correlation_sign_agrees_across_seeds": bool(
                    all(r > 0 for r in per_seed_rho if not np.isnan(r))
                    and any(not np.isnan(r) for r in per_seed_rho)
                ),
                "non_decreasing_steps": int(sum(1 for s in steps if s >= -1e-12)),
                "steps": len(steps),
                "largest_budget_minus_zero_shot": float(means[-1] - zero),
            }

        deltas = {
            f"n_{budget}": paired_repair_recall_delta(
                piece.cell(SELECTED_ARM, budget, RANDOM, 0),
                baseline,
                piece.harmful,
                piece.beneficial,
                piece.documents,
                PRIMARY_EPSILON,
            )
            for budget in NONZERO_BUDGETS
        }
        largest = NONZERO_BUDGETS[-1]
        per_arm_at_largest = {
            arm: paired_repair_recall_delta(
                piece.cell(arm, largest, RANDOM, 0),
                baseline,
                piece.harmful,
                piece.beneficial,
                piece.documents,
                PRIMARY_EPSILON,
            )
            for arm in (*ADAPTER_ARMS, SELECTED_ARM, PERMUTED_ARM)
            if f"arm__{cell_name(arm, largest, RANDOM, 0)}" in piece.scores
        }
        budget_effect = paired_repair_recall_delta(
            piece.cell(SELECTED_ARM, largest, RANDOM, 0),
            piece.cell(SELECTED_ARM, NONZERO_BUDGETS[0], RANDOM, 0),
            piece.harmful,
            piece.beneficial,
            piece.documents,
            PRIMARY_EPSILON,
        )

        # What the budget could have bought if the right family had been picked. The pick is
        # made on the EVALUATION rows, so it is a ceiling and never a policy; it is here because
        # the difference between it and the deployed selection separates two failures that look
        # identical from outside -- an adapter that cannot help, and an inner validation too
        # small to tell which adapter would.
        selection_ceiling: dict[str, Any] = {}
        costs: list[float] = []
        for budget in NONZERO_BUDGETS:
            per_arm = {
                arm: _recall(piece.cell(arm, budget, RANDOM, 0), piece.harmful, piece.beneficial)
                for arm in ADAPTER_ARMS
                if f"arm__{cell_name(arm, budget, RANDOM, 0)}" in piece.scores
            }
            deployed = _recall(
                piece.cell(SELECTED_ARM, budget, RANDOM, 0), piece.harmful, piece.beneficial
            )
            best_arm = max(per_arm, key=lambda name: per_arm[name])
            costs.append(per_arm[best_arm] - deployed)
            selection_ceiling[f"n_{budget}"] = {
                "deployed_selection": deployed,
                "best_available_arm": best_arm,
                "best_available_repair_recall": per_arm[best_arm],
                "selection_cost": float(per_arm[best_arm] - deployed),
                "per_arm": per_arm,
            }
        selection_ceiling["mean_selection_cost"] = float(np.mean(costs))
        selection_ceiling["note"] = (
            "the best-available arm is chosen with hindsight on the held-out engine's own "
            "evaluation labels. It bounds what the adapter families contain and is never a "
            "deployable arm."
        )

        headroom = ceiling - zero
        recovery = {
            f"n_{budget}": {
                "repair_recall": curves[SELECTED_ARM]["per_budget"][f"n_{budget}"]["mean"],
                "recovery_ratio": (
                    float(
                        (curves[SELECTED_ARM]["per_budget"][f"n_{budget}"]["mean"] - zero)
                        / headroom
                    )
                    if headroom > 0
                    else float("nan")
                ),
            }
            for budget in NONZERO_BUDGETS
        }
        required = {}
        for level in (0.5, 0.75, 0.9):
            reached = [
                budget
                for budget in NONZERO_BUDGETS
                if recovery[f"n_{budget}"]["recovery_ratio"] >= level
            ]
            required[f"recover_{int(level * 100)}_percent"] = int(reached[0]) if reached else None

        folds[held_out] = {
            "held_out_engine": held_out,
            "zero_shot_repair_recall": zero,
            "in_engine_oracle_repair_recall": ceiling,
            "headroom": float(headroom),
            "curves": curves,
            "arm_selection_ceiling": selection_ceiling,
            "paired_delta_against_sgv5": deltas,
            "paired_delta_per_arm_at_largest_budget": per_arm_at_largest,
            "paired_delta_largest_minus_smallest_budget": budget_effect,
            "recovery": recovery,
            "labels_required": required,
        }
        print(
            f"  fold {held_out:11s} zero={zero:.3f} oracle={ceiling:.3f} "
            f"n1000={recovery['n_1000']['repair_recall']:.3f} "
            f"recovery={recovery['n_1000']['recovery_ratio']:.3f}"
        )

    cc._write_json_once(
        EFFICIENCY_RESULTS,
        {
            "schema_version": "sgv6-efficiency-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1b",
            "primary_epsilon": PRIMARY_EPSILON,
            "acquisition": RANDOM,
            "recovery_ratio": "(SGV6(N) - SGV5) / (in-engine oracle - SGV5), per engine",
            "recovery_ratio_note": (
                "the denominator is one engine's own headroom, so a ratio is comparable across "
                "budgets within an engine and NOT across engines: a fold whose headroom is "
                "0.03 and a fold whose headroom is 0.75 do not mean the same thing by 50%."
            ),
            "monotonicity_note": (
                "the rank correlation is taken over the eight-point budget grid, which is a "
                "descriptive statistic over a grid and not a resampled one. The claim that a "
                "budget helps carries a document-clustered paired interval instead, under "
                "`paired_delta_largest_minus_smallest_budget`, and the sign agreement across "
                "the five random draws is reported next to it."
            ),
            "folds": folds,
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"efficiency: {len(folds)} folds -> {cc._relative(EFFICIENCY_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the oracle gap


def run_oracle() -> int:
    """SGV6-T1c: how much of the in-engine oracle's advantage a budget actually buys."""
    started = time.monotonic()
    slices, record = load_scores()
    folds: dict[str, Any] = {}

    for held_out, piece in slices.items():
        baseline = piece.arm(SGV5_ARM)
        oracle = piece.arm(ORACLE_ARM)
        diagnostics = record["folds"][held_out]["diagnostics"]
        largest = NONZERO_BUDGETS[-1]
        best = piece.cell(SELECTED_ARM, largest, RANDOM, 0)
        folds[held_out] = {
            "held_out_engine": held_out,
            "oracle_budget": diagnostics["oracle_budget"],
            "sgv6_budget": {
                "labels": largest,
                "fraction_of_the_oracle_fitting_set": float(
                    largest / diagnostics["rows"]["adaptation_pool"]
                ),
                "note": (
                    "the two are drawn from the SAME pool, so this is a subset fraction and not "
                    "an analogy. The oracle additionally spends the held-out engine's "
                    "calibration documents, which no SGV6 arm is allowed to see."
                ),
            },
            "per_epsilon": {
                f"epsilon_{int(e * 100)}": {
                    "sgv5": restricted_frontier(baseline, piece.harmful, piece.beneficial)[
                        f"epsilon_{int(e * 100)}"
                    ]["repair_recall"],
                    "sgv6": restricted_frontier(best, piece.harmful, piece.beneficial)[
                        f"epsilon_{int(e * 100)}"
                    ]["repair_recall"],
                    "in_engine_oracle": restricted_frontier(
                        oracle, piece.harmful, piece.beneficial
                    )[f"epsilon_{int(e * 100)}"]["repair_recall"],
                }
                for e in EPSILONS
            },
            "remaining_gap_at_primary_epsilon": paired_repair_recall_delta(
                oracle,
                best,
                piece.harmful,
                piece.beneficial,
                piece.documents,
                PRIMARY_EPSILON,
            ),
            "gap_closed_by_the_budget": paired_repair_recall_delta(
                best, baseline, piece.harmful, piece.beneficial, piece.documents, PRIMARY_EPSILON
            ),
            "adaptation_pool_prevalence": diagnostics["adaptation_pool_prevalence"],
        }
        print(
            f"  fold {held_out:11s} remaining gap="
            f"{folds[held_out]['remaining_gap_at_primary_epsilon']['delta_repair_recall']:+.3f}"
        )

    cc._write_json_once(
        ORACLE_GAP,
        {
            "schema_version": "sgv6-oracle-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1c",
            "oracle_is": (
                "SGV5's in-engine oracle, refitted on the held-out engine's own TRAIN documents "
                "with the document partition intact. It is an upper bound and never a deployable "
                "arm: it needs labels for the engine being judged, which is the thing a new "
                "engine does not have."
            ),
            "folds": folds,
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"oracle: {len(folds)} folds -> {cc._relative(ORACLE_GAP)}")
    return 0


# ------------------------------------------------------------------ SGV6-B: acquisition


def run_active() -> int:
    """Does choosing WHICH rows to label change the price of a label?"""
    started = time.monotonic()
    slices, record = load_scores()
    folds: dict[str, Any] = {}

    for held_out, piece in slices.items():
        cells = record["folds"][held_out]["cells"]
        rules: dict[str, Any] = {}
        for rule in ACQUISITIONS:
            per_budget: dict[str, Any] = {}
            for budget in NONZERO_BUDGETS:
                key = cell_name(SELECTED_ARM, budget, rule, 0)
                if f"arm__{key}" not in piece.scores:
                    continue
                score = piece.cell(SELECTED_ARM, budget, rule, 0)
                holder = cells[key]
                entry: dict[str, Any] = {
                    "repair_recall": _recall(score, piece.harmful, piece.beneficial),
                    "beneficial_in_budget": holder["beneficial_in_budget"],
                    "harmful_in_budget": holder["harmful_in_budget"],
                    "documents_drawn": holder["documents_drawn"],
                    "selected_arm": holder["arm_selection"]["selected_arm"],
                }
                if rule != RANDOM:
                    reference = _seed_curve(piece, SELECTED_ARM, budget)
                    entry["random_reference"] = reference
                    entry["exceeds_every_random_draw"] = bool(
                        entry["repair_recall"] > reference["max"]
                    )
                    entry["paired_delta_against_random_seed_0"] = paired_repair_recall_delta(
                        score,
                        piece.cell(SELECTED_ARM, budget, RANDOM, 0),
                        piece.harmful,
                        piece.beneficial,
                        piece.documents,
                        PRIMARY_EPSILON,
                    )
                per_budget[f"n_{budget}"] = entry
            rules[rule] = per_budget
        folds[held_out] = {
            "held_out_engine": held_out,
            "adaptation_pool_beneficial_rate": record["folds"][held_out]["diagnostics"][
                "adaptation_pool_prevalence"
            ]["beneficial"],
            "rules": rules,
        }
        summary = " ".join(
            f"{rule[:4]}={rules[rule]['n_100']['beneficial_in_budget']:3d}"
            for rule in ACQUISITIONS
            if "n_100" in rules[rule]
        )
        print(f"  fold {held_out:11s} beneficial rows bought at N=100: {summary}")

    cc._write_json_once(
        ACTIVE_RESULTS,
        {
            "schema_version": "sgv6-active-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "rules": {
                RANDOM: "a seeded uniform permutation of the pool; five draws",
                UNCERTAINTY: (
                    "smallest distance from the frozen utility to the source-calibrated "
                    "operating point"
                ),
                DIVERSITY: (
                    "greedy farthest-point traversal in the frozen adapter input space, started "
                    "at the row nearest the pool centroid"
                ),
                EXPECTED_HARM_REDUCTION: (
                    "p(1-p) on the frozen harm head times an exponential kernel in the distance "
                    "to the operating point. A proxy for expected value of information, reported "
                    "as a proxy: the exact quantity needs a posterior the rule does not have "
                    "before any label is bought."
                ),
            },
            "batch_mode_note": (
                "every ordering is computed once from the frozen zero-shot model and never "
                "revised as labels arrive, so a budget is a prefix of one ordering. These curves "
                "are therefore a lower bound on what a sequential acquisition loop could deliver "
                "and are not an estimate of one."
            ),
            "no_rule_reads_a_target_label": True,
            "folds": folds,
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"active: {len(folds)} folds -> {cc._relative(ACTIVE_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the ablations


def run_ablation() -> int:
    """Feature freezing, the label-budget grid, the permutation control, engine difficulty."""
    started = time.monotonic()
    slices, record = load_scores()
    folds: dict[str, Any] = {}

    for held_out, piece in slices.items():
        cells = record["folds"][held_out]["cells"]
        baseline = piece.arm(SGV5_ARM)
        oracle = piece.arm(ORACLE_ARM)
        zero = _recall(baseline, piece.harmful, piece.beneficial)

        freezing: dict[str, Any] = {}
        for budget in NONZERO_BUDGETS:
            reference = piece.cell(FREEZE_ADAPTER, budget, RANDOM, 0)
            entry: dict[str, Any] = {
                FREEZE_ADAPTER: {
                    "repair_recall": _recall(reference, piece.harmful, piece.beneficial)
                }
            }
            for arm in FREEZE_ARMS[1:]:
                score = piece.cell(arm, budget, RANDOM, 0)
                entry[arm] = {
                    "repair_recall": _recall(score, piece.harmful, piece.beneficial),
                    "note": cells[cell_name(SELECTED_ARM, budget, RANDOM, 0)]["freezing"].get(arm),
                    "paired_delta_against_adapter_only": paired_repair_recall_delta(
                        score,
                        reference,
                        piece.harmful,
                        piece.beneficial,
                        piece.documents,
                        PRIMARY_EPSILON,
                    ),
                }
            freezing[f"n_{budget}"] = entry

        permutation = {
            f"n_{budget}": {
                "repair_recall": _recall(
                    piece.cell(PERMUTED_ARM, budget, RANDOM, 0), piece.harmful, piece.beneficial
                ),
                "selected_arm": cells[cell_name(SELECTED_ARM, budget, RANDOM, 0)][
                    "permuted_control_arm"
                ],
                "paired_delta_against_sgv5": paired_repair_recall_delta(
                    piece.cell(PERMUTED_ARM, budget, RANDOM, 0),
                    baseline,
                    piece.harmful,
                    piece.beneficial,
                    piece.documents,
                    PRIMARY_EPSILON,
                ),
            }
            for budget in NONZERO_BUDGETS
        }

        largest = NONZERO_BUDGETS[-1]
        folds[held_out] = {
            "held_out_engine": held_out,
            "feature_freezing": freezing,
            "permuted_budget_control": permutation,
            "label_budget": {
                f"n_{budget}": _seed_curve(piece, SELECTED_ARM, budget)
                for budget in NONZERO_BUDGETS
            },
            "engine_difficulty": {
                "zero_shot_repair_recall": zero,
                "in_engine_oracle_repair_recall": _recall(oracle, piece.harmful, piece.beneficial),
                "transfer_gap": float(_recall(oracle, piece.harmful, piece.beneficial) - zero),
                "adaptation_pool_beneficial_rate": record["folds"][held_out]["diagnostics"][
                    "adaptation_pool_prevalence"
                ]["beneficial"],
                "beneficial_rows_bought_at_the_largest_budget": cells[
                    cell_name(SELECTED_ARM, largest, RANDOM, 0)
                ]["beneficial_in_budget"],
            },
            "matched_coverage_harm_reduction_at_the_largest_budget": rl._matched_coverage_contrast(
                baseline,
                piece.cell(SELECTED_ARM, largest, RANDOM, 0),
                piece.harmful,
                piece.beneficial,
                piece.documents,
            ),
        }
        print(
            f"  fold {held_out:11s} permuted at N={largest}: "
            f"{permutation[f'n_{largest}']['repair_recall']:.3f} against SGV5's {zero:.3f}"
        )

    cc._write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv6-ablation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "feature_freezing": {
                FREEZE_ADAPTER: "the SGV6 arm: the frozen model plus an adapter over it",
                FREEZE_JOINT_NAIVE: (
                    "the whole model re-estimated on the source rows plus the budget, each "
                    "target row counted once"
                ),
                FREEZE_JOINT_PARITY: (
                    "the same, with the budget weighted to carry one source engine's influence"
                ),
                FREEZE_TARGET_ONLY: "the same model class fitted on the budget and nothing else",
            },
            "permutation_control": (
                "the identical procedure -- families, inner selection, shrinkage grids, "
                "threshold placement -- run on the same budget rows with their labels shuffled. "
                "A gain that survives here was manufactured by the machinery and not by the "
                "labels."
            ),
            "folds": folds,
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"ablation: {len(folds)} folds -> {cc._relative(ABLATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ SGV6-T1a: the headline


def run_results() -> int:
    """SGV6-T1a: does adaptation beat the zero-shot arm and everything that came before it?"""
    started = time.monotonic()
    slices, record = load_scores()
    folds: dict[str, Any] = {}
    largest = NONZERO_BUDGETS[-1]

    for held_out, piece in slices.items():
        cells = record["folds"][held_out]["cells"]
        best = piece.cell(SELECTED_ARM, largest, RANDOM, 0)
        comparisons = {
            name[len("arm__") :]: paired_repair_recall_delta(
                best,
                piece.arm(name),
                piece.harmful,
                piece.beneficial,
                piece.documents,
                PRIMARY_EPSILON,
            )
            for name in BASELINE_COLUMNS
            if name != ORACLE_ARM
        }
        per_budget = {
            f"n_{budget}": {
                "sgv6": _point(
                    piece.cell(SELECTED_ARM, budget, RANDOM, 0), piece.harmful, piece.beneficial
                ),
                "selected_arm": cells[cell_name(SELECTED_ARM, budget, RANDOM, 0)]["arm_selection"][
                    "selected_arm"
                ],
                "selected_setting": cells[cell_name(SELECTED_ARM, budget, RANDOM, 0)][
                    "selected_arm_inner"
                ]["selected_setting"],
                "deployed": _deployed_pair(
                    piece,
                    piece.cell(SELECTED_ARM, budget, RANDOM, 0),
                    cells[cell_name(SELECTED_ARM, budget, RANDOM, 0)]["thresholds"][SELECTED_ARM],
                ),
            }
            for budget in NONZERO_BUDGETS
        }
        zero_thresholds = {
            "source_calibrated": record["folds"][held_out]["zero_shot_thresholds"],
            "target_calibrated": {c: {"tau": float("inf"), "feasible": False} for c in CONTROLLERS},
        }
        folds[held_out] = {
            "held_out_engine": held_out,
            "budget_zero": {
                "sgv5": _point(piece.arm(SGV5_ARM), piece.harmful, piece.beneficial),
                "deployed": _deployed_pair(piece, piece.arm(SGV5_ARM), zero_thresholds),
            },
            "per_budget": per_budget,
            "paired_delta_at_the_largest_budget": comparisons,
            "seed_spread_at_the_largest_budget": _seed_curve(piece, SELECTED_ARM, largest),
        }
        favourable = sum(1 for cell in comparisons.values() if cell["favours_arm"])
        print(
            f"  fold {held_out:11s} N={largest}: "
            f"{per_budget[f'n_{largest}']['sgv6'][PRIMARY_KEY]['repair_recall']:.3f} "
            f"({favourable}/{len(comparisons)} baselines beaten with an interval)"
        )

    cc._write_json_once(
        ADAPTATION_RESULTS,
        {
            "schema_version": "sgv6-adaptation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1a",
            "primary_endpoint": "repair_recall_at_bounded_harm at epsilon = 0.10",
            "acquisition": RANDOM,
            "seed": 0,
            "baselines": {
                "no_correction": "accept nothing",
                "harm_only": "rank by predicted safety alone",
                "harm_aware": "SGV1's candidate-level utility, the strongest prior arm",
                "selfaware": "SGV3's uncertainty-aware arm",
                "env_aware": "SGV4's environment-applicability arm",
                "sgv5": "the zero-shot arm this stage adapts, and this stage's budget-zero point",
            },
            "folds": folds,
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"results: {len(folds)} folds -> {cc._relative(ADAPTATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ figures


def run_figures() -> int:
    started = time.monotonic()
    results = cc._read_json(ADAPTATION_RESULTS)
    efficiency = cc._read_json(EFFICIENCY_RESULTS)
    oracle = cc._read_json(ORACLE_GAP)
    active = cc._read_json(ACTIVE_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV6 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(results["folds"])
    written: list[Path] = []
    budgets = list(NONZERO_BUDGETS)

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    # --- adaptation_curve.png --------------------------------------------------------------
    # The figure the brief asks for: the endpoint against the number of target labels, with
    # SGV5 and the in-engine oracle as the two horizontal lines the curve lives between.
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4), sharey=True)
    families = {
        A1_LAMBDA: ("#c87a2b", "--"),
        A1_PLATT: ("#7a7a7a", ":"),
        A2_LOGISTIC: ("#3a5f9e", "-"),
        A3_BAYES: ("#2e7d5b", "--"),
        SELECTED_ARM: ("#a33", "-"),
    }
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        block = efficiency["folds"][engine]
        panel.axhline(block["zero_shot_repair_recall"], color="#444", linestyle="-", linewidth=1.2)
        panel.axhline(
            block["in_engine_oracle_repair_recall"], color="#a33", linestyle=":", linewidth=1.2
        )
        for arm, (colour, style) in families.items():
            if arm not in block["curves"]:
                continue
            curve = block["curves"][arm]["per_budget"]
            panel.plot(
                budgets,
                [curve[f"n_{n}"]["mean"] for n in budgets],
                style,
                color=colour,
                linewidth=1.8 if arm == SELECTED_ARM else 1.2,
                marker="o" if arm == SELECTED_ARM else None,
                markersize=3,
                label=arm,
            )
        selected = block["curves"][SELECTED_ARM]["per_budget"]
        panel.fill_between(
            budgets,
            [selected[f"n_{n}"]["min"] for n in budgets],
            [selected[f"n_{n}"]["max"] for n in budgets],
            color="#a33",
            alpha=0.15,
        )
        panel.set_xscale("log")
        panel.set_xticks(budgets)
        panel.set_xticklabels([str(n) for n in budgets], fontsize=7)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("target-engine labels")
        panel.set_ylim(0.0, 1.02)
    np.atleast_1d(panels)[0].set_ylabel("repair recall at a 10% harm bound")
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "adaptation_curve.png",
        "Repair recall against the target-engine label budget. The solid grey line is SGV5 "
        "(budget zero) and the dotted red line is the in-engine oracle; the band is the range "
        "over five random draws",
    )

    # --- oracle_recovery.png ---------------------------------------------------------------
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for engine in engines:
        recovery = efficiency["folds"][engine]["recovery"]
        left.plot(
            budgets,
            [recovery[f"n_{n}"]["recovery_ratio"] for n in budgets],
            "-o",
            markersize=3,
            linewidth=1.4,
            label=engine,
        )
    left.axhline(1.0, color="#a33", linestyle=":", linewidth=1)
    left.set_xscale("log")
    left.set_xticks(budgets)
    left.set_xticklabels([str(n) for n in budgets], fontsize=7)
    left.set_xlabel("target-engine labels")
    left.set_ylabel("(SGV6 - SGV5) / (oracle - SGV5)")
    left.set_title("oracle recovery ratio, per engine", fontsize=9)
    left.legend(fontsize=7, loc="lower right")

    positions = np.arange(len(engines))
    zero = np.array([efficiency["folds"][e]["zero_shot_repair_recall"] for e in engines])
    reached = np.array(
        [
            efficiency["folds"][e]["curves"][SELECTED_ARM]["per_budget"][
                f"n_{NONZERO_BUDGETS[-1]}"
            ]["mean"]
            for e in engines
        ]
    )
    ceiling = np.array([efficiency["folds"][e]["in_engine_oracle_repair_recall"] for e in engines])
    right.bar(positions, zero, 0.55, color="#444", label="SGV5 (budget zero)")
    right.bar(
        positions,
        np.maximum(reached - zero, 0.0),
        0.55,
        bottom=zero,
        color="#a33",
        label=f"bought by {NONZERO_BUDGETS[-1]} labels",
    )
    right.bar(
        positions,
        np.maximum(ceiling - np.maximum(reached, zero), 0.0),
        0.55,
        bottom=np.maximum(reached, zero),
        color="#ddd",
        label="still held by the in-engine oracle",
    )
    right.set_xticks(positions)
    right.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    right.set_ylim(0.0, 1.0)
    right.set_ylabel("repair recall at a 10% harm bound")
    right.set_title("where the oracle gap goes", fontsize=9)
    right.legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "oracle_recovery.png",
        "How much of each engine's own oracle headroom a label budget recovers",
    )

    # --- acquisition_comparison.png --------------------------------------------------------
    # Two panels because they answer different questions: what a rule BUYS (beneficial rows,
    # which is the currency the endpoint is denominated in) and what it DELIVERS.
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    width = 0.8 / len(ACQUISITIONS)
    positions = np.arange(len(engines))
    for offset, rule in enumerate(ACQUISITIONS):
        left.bar(
            positions + offset * width,
            [active["folds"][e]["rules"][rule]["n_100"]["beneficial_in_budget"] for e in engines],
            width,
            label=rule,
        )
        right.bar(
            positions + offset * width,
            [active["folds"][e]["rules"][rule]["n_100"]["repair_recall"] for e in engines],
            width,
            label=rule,
        )
    for panel, label, title in (
        (left, "beneficial rows in a 100-label budget", "what the rule buys"),
        (right, "repair recall at a 10% harm bound", "what the rule delivers"),
    ):
        panel.set_xticks(positions + 0.4 - width / 2)
        panel.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
        panel.set_ylabel(label)
        panel.set_title(title, fontsize=9)
        panel.legend(fontsize=7)
    finish(
        figure,
        FIGURE_DIR / "acquisition_comparison.png",
        "SGV6-B: which rows to label, at a fixed budget of 100",
    )

    # --- feature_freezing.png --------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4), sharey=True)
    styles = {
        FREEZE_ADAPTER: ("#a33", "-"),
        FREEZE_JOINT_NAIVE: ("#3a5f9e", "--"),
        FREEZE_JOINT_PARITY: ("#2e7d5b", "--"),
        FREEZE_TARGET_ONLY: ("#c87a2b", ":"),
    }
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        block = ablation["folds"][engine]["feature_freezing"]
        for arm, (colour, style) in styles.items():
            panel.plot(
                budgets,
                [block[f"n_{n}"][arm]["repair_recall"] for n in budgets],
                style,
                color=colour,
                linewidth=1.5,
                marker="o",
                markersize=3,
                label=arm,
            )
        panel.axhline(
            efficiency["folds"][engine]["zero_shot_repair_recall"],
            color="#444",
            linewidth=1.0,
        )
        panel.set_xscale("log")
        panel.set_xticks(budgets)
        panel.set_xticklabels([str(n) for n in budgets], fontsize=7)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("target-engine labels")
        panel.set_ylim(0.0, 1.02)
    np.atleast_1d(panels)[0].set_ylabel("repair recall at a 10% harm bound")
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "feature_freezing.png",
        "Where the target labels enter: an adapter over a frozen model, a joint refit, or a "
        "model fitted on the budget alone. The grey line is SGV5",
    )

    # --- risk_coverage_curve.png -----------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4), sharey=True)
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        block = oracle["folds"][engine]["per_epsilon"]
        grid = [float(e) for e in EPSILONS]
        for arm, colour, style in (
            ("sgv5", "#444", "-"),
            ("sgv6", "#a33", "-"),
            ("in_engine_oracle", "#a33", ":"),
        ):
            panel.plot(
                grid,
                [block[f"epsilon_{int(e * 100)}"][arm] for e in grid],
                style,
                color=colour,
                linewidth=1.6,
                marker="o",
                markersize=3,
                label=arm,
            )
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("harm bound (epsilon)")
        panel.set_ylim(0.0, 1.02)
    np.atleast_1d(panels)[0].set_ylabel("repair recall at the bound")
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "risk_coverage_curve.png",
        f"Repair recall against the harm bound, at a budget of {NONZERO_BUDGETS[-1]} labels",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv6-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (
                    ADAPTATION_RESULTS,
                    EFFICIENCY_RESULTS,
                    ORACLE_GAP,
                    ACTIVE_RESULTS,
                    ABLATION_RESULTS,
                )
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ the decision


def _favourable(cell: dict[str, Any]) -> bool:
    return bool(cell.get("favours_arm")) and not cell.get("degenerate_interval", False)


def _adverse(cell: dict[str, Any]) -> bool:
    return bool(cell["ci_upper"] < 0.0) and not cell.get("degenerate_interval", False)


def run_decide() -> int:
    """The four criteria, evaluated exactly as they were written before the run."""
    started = time.monotonic()
    results = cc._read_json(ADAPTATION_RESULTS)
    efficiency = cc._read_json(EFFICIENCY_RESULTS)
    oracle = cc._read_json(ORACLE_GAP)
    ablation = cc._read_json(ABLATION_RESULTS)
    active = cc._read_json(ACTIVE_RESULTS)
    record = cc._read_json(SELECTION_RECORD)
    engines = sorted(results["folds"])
    largest = NONZERO_BUDGETS[-1]

    improvement = {
        engine: efficiency["folds"][engine]["paired_delta_against_sgv5"][f"n_{largest}"]
        for engine in engines
    }
    criterion_1 = {
        "rule": (
            "the paired repair-recall delta against SGV5 at the largest budget is positive on "
            "every fold"
        ),
        "per_engine": {engine: improvement[engine]["delta_repair_recall"] for engine in engines},
        "holds": all(improvement[engine]["delta_repair_recall"] > 0.0 for engine in engines),
    }
    criterion_2 = {
        "rule": (
            "that delta's document-clustered 95% interval excludes zero on every fold, and no "
            "fold's interval lies wholly below zero"
        ),
        "per_engine": {
            engine: {
                "delta": improvement[engine]["delta_repair_recall"],
                "ci_lower": improvement[engine]["ci_lower"],
                "ci_upper": improvement[engine]["ci_upper"],
                "favourable": _favourable(improvement[engine]),
                "adverse": _adverse(improvement[engine]),
            }
            for engine in engines
        },
        "holds": all(_favourable(improvement[engine]) for engine in engines),
        "no_fold_is_significantly_worse": not any(_adverse(improvement[e]) for e in engines),
    }
    monotone = {engine: efficiency["folds"][engine]["curves"][SELECTED_ARM] for engine in engines}
    criterion_3 = {
        "rule": (
            "the rank correlation between the budget and the endpoint is positive on the "
            "seed-mean curve of every fold, and its sign agrees across all five random draws"
        ),
        "per_engine": {
            engine: {
                "rank_correlation_of_mean_curve": monotone[engine][
                    "rank_correlation_of_mean_curve"
                ],
                "rank_correlation_per_seed": monotone[engine]["rank_correlation_per_seed"],
                "sign_agrees_across_seeds": monotone[engine][
                    "rank_correlation_sign_agrees_across_seeds"
                ],
                "non_decreasing_steps": monotone[engine]["non_decreasing_steps"],
                "steps": monotone[engine]["steps"],
            }
            for engine in engines
        },
        "holds": all(
            monotone[engine]["rank_correlation_of_mean_curve"] > 0.0
            and monotone[engine]["rank_correlation_sign_agrees_across_seeds"]
            for engine in engines
        ),
    }
    permuted = {
        engine: {
            f"n_{budget}": ablation["folds"][engine]["permuted_budget_control"][f"n_{budget}"][
                "repair_recall"
            ]
            for budget in NONZERO_BUDGETS
        }
        for engine in engines
    }
    criterion_4 = {
        "rule": (
            "the rebuilt SGV5 baseline is bit-identical to the published one on every fold, no "
            "budget document appears in an evaluation block, and the label-permuted budget "
            "never exceeds SGV5 at any budget on any fold"
        ),
        "sgv5_identity_max_difference": {
            engine: record["folds"][engine]["sgv5_identity_check"][
                "max_abs_difference_from_published_sgv5"
            ]
            for engine in engines
        },
        "permuted_budget_never_exceeds_sgv5": {
            engine: bool(
                max(permuted[engine].values())
                <= efficiency["folds"][engine]["zero_shot_repair_recall"] + 1e-12
            )
            for engine in engines
        },
        "permuted_budget_repair_recall": permuted,
        # What the control reached at its own best budget, beside what the real arm reached at
        # the same budget and what SGV5 reached. The rule above is a strict inequality and is
        # left exactly as written; this is the measurement a reader needs to see what a failure
        # of it actually was, and it is added rather than substituted.
        "permuted_high_water_mark": {
            engine: {
                "budget": int(max(permuted[engine], key=lambda k: permuted[engine][k])[2:]),
                "permuted_repair_recall": max(permuted[engine].values()),
                "sgv6_repair_recall_at_the_same_budget": efficiency["folds"][engine]["curves"][
                    SELECTED_ARM
                ]["per_budget"][max(permuted[engine], key=lambda k: permuted[engine][k])][
                    "per_seed"
                ][0],
                "sgv5_repair_recall": efficiency["folds"][engine]["zero_shot_repair_recall"],
            }
            for engine in engines
        },
        "holds": all(
            record["folds"][engine]["sgv5_identity_check"]["max_abs_difference_from_published_sgv5"]
            == 0.0
            and max(permuted[engine].values())
            <= efficiency["folds"][engine]["zero_shot_repair_recall"] + 1e-12
            for engine in engines
        ),
    }

    criteria = {
        "1_improves_over_sgv5": criterion_1,
        "2_survives_document_clustered_intervals": criterion_2,
        "3_monotone_in_the_label_budget": criterion_3,
        "4_no_leakage_detected": criterion_4,
    }
    supported = all(block["holds"] for block in criteria.values())

    ceiling = {
        engine: efficiency["folds"][engine].get("arm_selection_ceiling", {}) for engine in engines
    }
    lambda_only = {
        engine: {
            f"n_{budget}": efficiency["folds"][engine]["curves"][A1_LAMBDA]["per_budget"][
                f"n_{budget}"
            ]["mean"]
            for budget in NONZERO_BUDGETS
        }
        for engine in engines
    }
    bottleneck = {
        "arm_selection": {
            "question": (
                "is the limit what the adapters can do, or what the budget's own validation can "
                "tell about them?"
            ),
            "selected_minus_best_available_per_engine": {
                engine: ceiling[engine].get("mean_selection_cost") for engine in engines
            },
            "note": (
                "the best-available arm is picked on the EVALUATION rows and is a ceiling, never "
                "a deployable policy. It is here because the difference between it and the "
                "deployed selection is the part of the failure that more labels for FITTING "
                "cannot repair."
            ),
        },
        "prevalence": {
            engine: {
                "adaptation_pool_beneficial_rate": record["folds"][engine]["diagnostics"][
                    "adaptation_pool_prevalence"
                ]["beneficial"],
                "beneficial_rows_in_a_1000_label_budget": ablation["folds"][engine][
                    "engine_difficulty"
                ]["beneficial_rows_bought_at_the_largest_budget"],
                "transfer_gap": ablation["folds"][engine]["engine_difficulty"]["transfer_gap"],
            }
            for engine in engines
        },
        "one_parameter_share": {
            "question": (
                "how much of the gain is a single retuned harm-aversion weight rather than a "
                "learned correction?"
            ),
            "lambda_only_repair_recall": lambda_only,
            "sgv6_repair_recall": {
                engine: {
                    f"n_{budget}": efficiency["folds"][engine]["curves"][SELECTED_ARM][
                        "per_budget"
                    ][f"n_{budget}"]["mean"]
                    for budget in NONZERO_BUDGETS
                }
                for engine in engines
            },
        },
        "acquisition": {
            engine: {
                rule: active["folds"][engine]["rules"][rule]["n_100"]["beneficial_in_budget"]
                for rule in ACQUISITIONS
            }
            for engine in engines
        },
        "remaining_oracle_gap": {
            engine: oracle["folds"][engine]["remaining_gap_at_primary_epsilon"][
                "delta_repair_recall"
            ]
            for engine in engines
        },
    }

    cc._write_json_once(
        DECISION,
        {
            "schema_version": "sgv6-decision-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "hypothesis": (
                "a reliability model fitted on several OCR engines can recover a useful part of "
                "the in-engine oracle's bounded-risk repair coverage on an unseen engine from a "
                "small number of labelled examples of that engine, and the number of labels "
                "required is measurable"
            ),
            "verdict": "SUPPORTED" if supported else "NOT SUPPORTED",
            "criteria": criteria,
            "bottleneck": bottleneck,
            "claims_not_made": [
                "solves OCR reliability",
                "domain generalization",
                "universal OCR correction",
            ],
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {'SUPPORTED' if supported else 'NOT SUPPORTED'} -> {cc._relative(DECISION)}")
    for name, block in criteria.items():
        print(f"    {name:45s} {'HOLDS' if block['holds'] else 'FAILS'}")
    return 0


# ------------------------------------------------------------------ provenance


def run_record() -> int:
    started = time.monotonic()
    produced = [
        path
        for path in (
            SCORES,
            SELECTION_RECORD,
            ADAPTATION_RESULTS,
            CURVE_RESULTS,
            EFFICIENCY_RESULTS,
            ORACLE_GAP,
            ACTIVE_RESULTS,
            ABLATION_RESULTS,
            DECISION,
            FIGURE_MANIFEST,
        )
        if path.is_file()
    ]
    figures = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.is_dir() else []
    inputs = [
        dg.DESIGN_MATRIX,
        c5.FEATURES,
        c5.PREDICTIONS,
        c5.POLICY_SELECTION,
        c5.FIT_RECORD,
    ]
    cc._write_json_once(
        PROVENANCE,
        {
            "schema_version": "sgv6-provenance-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV6-T1",
            "stage": "SGV6 -- few-shot target-engine reliability adaptation",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "inputs": {cc._relative(path): file_sha256(path) for path in inputs if path.is_file()},
            "artifacts": {cc._relative(path): file_sha256(path) for path in (*produced, *figures)},
            "budget": {
                "grid": list(BUDGETS),
                "drawn_from": (
                    "the held-out engine's own TRAIN documents -- the same rows SGV5's in-engine "
                    "oracle is fitted on, so the budget axis ends at the oracle"
                ),
                "seed": BUDGET_SEED,
                "seed_repeats": SEED_REPEATS,
                "acquisition_rules": list(ACQUISITIONS),
            },
            "adapters": {
                ZERO_SHOT: "SGV5's frozen arm; no fitting happens",
                A1_LAMBDA: "re-pick the harm-aversion weight on the budget; one parameter",
                A1_PLATT: "sigmoid(a * logit(p) + b) on each frozen head",
                A1_TEMPERATURE: "sigmoid(logit(p) / T) on each frozen head",
                A1_ISOTONIC: "a free monotone map on each frozen head",
                A2_LOGISTIC: (
                    "logit(p) as a fixed offset plus a ridge-penalised linear correction over "
                    "the frozen adapter input"
                ),
                A2_RIDGE: "a ridge-penalised linear correction to the probability itself",
                A2_BOOSTED: (
                    "a gradient-boosted correction over the frozen adapter input, blended with "
                    "the frozen head"
                ),
                A3_BAYES: (
                    "the logistic adapter with the prior precision chosen by Laplace evidence "
                    "and the correction integrated over its own posterior"
                ),
            },
            "frozen_from_sgv5": {
                "representation": "Phase 3's matrix plus SGV5's blocks D and E, rebuilt exactly",
                "model_class_and_lambda": "read from SGV5's policy_selection.json, not re-made",
                "identity_check": "the rebuilt arm must equal SGV5's published vector bit for bit",
            },
            "invariants": {
                "held_out_engine_and_held_out_document": True,
                "target_labels_only_from_train_documents_of_the_held_out_engine": True,
                "every_target_side_choice_made_inside_the_budget": True,
                "oracle_used_only_as_an_upper_bound": True,
                "engine_identity_used_as_a_feature": False,
                "confirmatory_accessed": False,
            },
            "commands": [
                "uv run python scripts/sgv6_target_engine_adaptation.py --adapt",
                "uv run python scripts/sgv6_target_engine_adaptation.py --results",
                "uv run python scripts/sgv6_target_engine_adaptation.py --curves",
                "uv run python scripts/sgv6_target_engine_adaptation.py --efficiency",
                "uv run python scripts/sgv6_target_engine_adaptation.py --oracle",
                "uv run python scripts/sgv6_target_engine_adaptation.py --active",
                "uv run python scripts/sgv6_target_engine_adaptation.py --ablation",
                "uv run python scripts/sgv6_target_engine_adaptation.py --figures",
                "uv run python scripts/sgv6_target_engine_adaptation.py --decide",
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"record: {len(produced) + len(figures)} artifacts -> {cc._relative(PROVENANCE)}")
    return 0


def main() -> int:
    stages = {
        "--adapt": run_adapt,
        "--results": run_results,
        "--curves": run_curves,
        "--efficiency": run_efficiency,
        "--oracle": run_oracle,
        "--active": run_active,
        "--ablation": run_ablation,
        "--figures": run_figures,
        "--decide": run_decide,
        "--record": run_record,
    }
    requested = [flag for flag in sys.argv[1:] if flag in stages]
    unknown = [flag for flag in sys.argv[1:] if flag not in stages]
    if unknown or not requested:
        print(f"usage: {Path(__file__).name} [{' | '.join(stages)}]")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    for flag in stages:
        if flag in requested:
            code = stages[flag]()
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

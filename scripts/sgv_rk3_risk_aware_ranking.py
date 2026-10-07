#!/usr/bin/env python3
"""SGV-RK3: risk-aware candidate ranking for reliable OCR correction under generator shift.

SGV-RK1 showed that a document-level LambdaMART ranker selects exact repairs better than SGV-DS1's
pointwise reliability score on the generators it was trained on, and transfers worse to a
generator it never saw. SGV-RK2, TH1 and TH2 then showed that the decision boundary, not only the
ranking, keeps a new generator from being deployed, and that in one direction the top of the list
is not pure enough for any boundary. This stage asks whether the ranking itself can be made
better, and safer, from two levers the earlier stages held fixed:

**1. The representation.** RL1's R3 describes the edited fragment. RK3 adds label-free evidence
about the edited WORD, the edit's fit to its surrounding text, glyph confusability, the page, and
whether a second, distinct generator proposed the same edit (R5 = R3 + five new families).

**2. The objective.** A risk-aware LambdaRank gain puts a harmful candidate below a neutral one by
exactly what the primary harm target implies: at harm 0.1, one harmful accept costs what nine
neutral ones earn. A pairwise loss with the same utility and a pointwise fit to the same utility
are fitted beside it, with the same trees, so the objective is the only thing that changes.

**3. Two transfer protocols.** The primary one holds out the generator AND the documents: the
source generator's labelled fit pages train, the unseen generator's test pages evaluate, and a
site with both generators stays in, so agreement is measurable. The secondary one is SGV-RK2's
sealed blocks exactly, so the zero-shot ranker can be placed on RK2's label-budget curve.

    --reconstruct   section 0: DS1, RK1 and RK2 re-read from their own artifacts
    --freeze        upstream hashes and the frozen inputs this stage consumes
    --preregister   representation, utility, models, protocols, criteria, outcome rule
    --features      the five new label-free families, R3 copied and verified against RK1
    --splits        in-distribution blocks, both transfer protocols, the within-generator cells
    --reproduce     DS1's score, RK1's ranker and RK2's arms, reproduced exactly before any fit
    --rank          every pre-registered fit, scored strictly out of sample
    --metrics       Top-1, Recall@K, NDCG@K and AUROC per fit
    --deployment    the DS1 policy per fit, and the analysis-only ranking curves and frontier
    --transfer      both transfer protocols, the generator transfer matrix, RK2's label curve
    --ablate        feature-family ablations and the generator-identification classifier
    --controls      grade permutation and random ranking
    --stats         document-clustered paired bootstrap, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / RANKING ONLY. No candidate is generated, no OCR output is regenerated, no generator
identity enters a model, nothing is certified and nothing is production-ready. The confirmatory
reserve stays LOCKED and `ready_for_external_confirmation` is false by construction.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_rk2_fewshot_ranker_adaptation as rk2
from ocr_risk.io.hashing import canonical_hash, file_sha256
from ocr_risk.metrics.text import levenshtein
from ocr_risk.stats.bootstrap import _bootstrap_p_value

rk1 = rk2.rk1
ds1 = rk2.ds1
rl2 = rk2.rl2
rl1 = rk2.rl1
hy1 = rk2.hy1
gen1 = rk2.gen1
xr1 = rk2.xr1
xc1 = rk2.xc1
s15 = rk2.s15
cc = rl1.cc

REPO = rk1.REPO
OUT = REPO / "results/generated/sgv_rk3"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"

FEATURE_MATRIX = OUT / "feature_matrix.parquet"
FEATURE_REGISTRY = OUT / "feature_registry.json"
SPLIT_REGISTRY = OUT / "split_registry.json"
REPRODUCTION = OUT / "reproduction.json"

RANKING_SCORES = OUT / "ranking_scores.parquet"
MODEL_REGISTRY = OUT / "model_registry.json"
TRAINING_REGISTRY = OUT / "training_registry.json"

RANKING_METRICS = OUT / "ranking_metrics.json"
DEPLOYMENT = OUT / "deployment_comparison.json"
CURVES = OUT / "ranking_curves.json"
TRANSFER = OUT / "generator_transfer.json"
ABLATION = OUT / "feature_ablation.json"
LEAKAGE = OUT / "generator_leakage.json"
CONTROL_RESULTS = OUT / "control_results.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_rk3/risk_aware_ranking.md"

SCHEMA_VERSION = 1
STAGE = "sgv_rk3_risk_aware_ranking"
HYPOTHESIS = "SGV-RK3-D1"
STAGE_KIND = "DEVELOPMENT / RANKING"

PhaseError = rl1.PhaseError
_relative = rl1._relative
_git = rl1._git
_write_json_once = rl1._write_json_once
_write_parquet_once = rl1._write_parquet_once
cc_read_json = rl1.cc_read_json
_ratio = rl1._ratio
_require = rl1._require
_forbid = rl1._forbid
auroc = rl1.auroc
_interval = rk1._interval
_share = rk1._share

BOOTSTRAP_RESAMPLES = rk1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rk1.BOOTSTRAP_SEED
FIT_SEED = rk1.FIT_SEED
PERMUTATION_SEED = 20260930
PERMUTATION_DRAWS = 5

# ------------------------------------------------------------------ the frozen design

C0 = rk1.C0
C1 = rk1.C1
GENERATORS = (C0, C1)
GENERATOR_SHORT = {C0: "current", C1: "Qwen"}
D_G2Q = rk2.D_G2Q
D_Q2G = rk2.D_Q2G
DIRECTIONS = (D_G2Q, D_Q2G)
SOURCE_OF = rk2.SOURCE_OF
TARGET_OF = rk2.TARGET_OF
DIRECTION_SHORT = rk2.DIRECTION_SHORT

EPSILONS = rk1.EPSILONS
PRIMARY_EPSILON = rk1.PRIMARY_EPSILON
HARM_TOLERANCE = ds1.HARM_TOLERANCE

REL_EXACT = rk1.REL_EXACT
REL_PARTIAL = rk1.REL_PARTIAL
REL_NEUTRAL = rk1.REL_NEUTRAL
REL_HARMFUL = rk1.REL_HARMFUL

# RK1's gain for every non-harmful grade. The risk-aware utility keeps it and sets the harmful
# gain so that, at the primary target, an accept set whose harm is exactly epsilon and whose other
# accepts are neutral earns nothing: (1 - epsilon) / epsilon neutral gains per harmful accept.
STANDARD_GAIN = {REL_EXACT: 7.0, REL_PARTIAL: 3.0, REL_NEUTRAL: 1.0, REL_HARMFUL: 0.0}
HARM_PENALTY_MULTIPLE = round((1.0 - PRIMARY_EPSILON) / PRIMARY_EPSILON, 12)
RISK_UTILITY = {
    REL_EXACT: STANDARD_GAIN[REL_EXACT],
    REL_PARTIAL: STANDARD_GAIN[REL_PARTIAL],
    REL_NEUTRAL: STANDARD_GAIN[REL_NEUTRAL],
    REL_HARMFUL: -HARM_PENALTY_MULTIPLE * STANDARD_GAIN[REL_NEUTRAL],
}

# Boosting settings are RK1's, fixed before any fit and shared by every tree objective.
ROUNDS = rk1.LAMBDAMART_ROUNDS
LEARNING_RATE = rk1.LAMBDAMART_LEARNING_RATE
MAX_DEPTH = rk1.LAMBDAMART_MAX_DEPTH
MIN_LEAF = rk1.LAMBDAMART_MIN_LEAF
SIGMA = rk1.LAMBDAMART_SIGMA

R3 = "r3"
R5 = "r5"
REPRESENTATIONS = (R3, R5)

OBJ_RANDOM = "random"
OBJ_LOGISTIC = "pointwise_logistic"
OBJ_POINTWISE = "pointwise_boosted"
OBJ_PAIRWISE = "pairwise_boosted"
OBJ_LAMBDA = "lambdarank_standard"
OBJ_RISK = "lambdarank_risk"
OBJECTIVE_LABEL = {
    OBJ_RANDOM: "seeded random scores",
    OBJ_LOGISTIC: (
        "SGV-DS1's pointwise reliability model: logistic P(beneficial) chooses, 1 - P(harmful) "
        "gates"
    ),
    OBJ_POINTWISE: "boosted regression trees fitted pointwise to the risk-aware utility",
    OBJ_PAIRWISE: (
        "boosted trees fitted to a pairwise logistic loss over document pairs, each pair weighted "
        "by its risk-aware utility difference, with no position discount"
    ),
    OBJ_LAMBDA: "LambdaMART with RK1's gain 2^grade - 1, RK1's gradient unchanged",
    OBJ_RISK: (
        "risk-aware LambdaMART: the LambdaRank gradient with the risk-aware utility and a "
        "query normalization by the utility range"
    ),
}

M_RANDOM = "b0_random"
M_DS1 = "b1_ds1_pointwise_r3"
M_RK1 = "b2_rk1_lambdamart_r3"
M_DS1_R5 = "p1_ds1_pointwise_r5"
M_GBP_R3 = "p2_boosted_pointwise_r3"
M_GBP_R5 = "p2_boosted_pointwise_r5"
M_PAIR_R3 = "q_pairwise_r3"
M_PAIR_R5 = "q_pairwise_r5"
M_LR_R5 = "l_lambdarank_r5"
M_RK3_R3 = "r_risk_lambdarank_r3"
M_RK3 = "r_risk_lambdarank_r5"
PRIMARY_MODEL = M_RK3
MODEL_SPEC: dict[str, tuple[str, str]] = {
    M_RANDOM: (OBJ_RANDOM, R3),
    M_DS1: (OBJ_LOGISTIC, R3),
    M_RK1: (OBJ_LAMBDA, R3),
    M_DS1_R5: (OBJ_LOGISTIC, R5),
    M_GBP_R3: (OBJ_POINTWISE, R3),
    M_GBP_R5: (OBJ_POINTWISE, R5),
    M_PAIR_R3: (OBJ_PAIRWISE, R3),
    M_PAIR_R5: (OBJ_PAIRWISE, R5),
    M_LR_R5: (OBJ_LAMBDA, R5),
    M_RK3_R3: (OBJ_RISK, R3),
    M_RK3: (OBJ_RISK, R5),
}
MODELS = tuple(MODEL_SPEC)
MODEL_SHORT = {
    M_RANDOM: "random",
    M_DS1: "DS1",
    M_RK1: "RK1",
    M_DS1_R5: "DS1 form, R5",
    M_GBP_R3: "pointwise, R3",
    M_GBP_R5: "pointwise, R5",
    M_PAIR_R3: "pairwise, R3",
    M_PAIR_R5: "pairwise, R5",
    M_LR_R5: "LambdaRank, R5",
    M_RK3_R3: "risk LambdaRank, R3",
    M_RK3: "RK3",
}

# Protocols. The in-distribution protocol is DS1's three blocks. The primary transfer protocol
# holds out the generator and the documents; the secondary one is RK2's sealed blocks exactly.
P_IN = "in_distribution"
K_IN = "in_distribution"
K_TDOC = "generator_and_document_held_out"
K_WITHIN = "within_generator"
K_STRICT = "rk2_strict_blocks"
TDOC = {d: f"tdoc_{d}" for d in DIRECTIONS}
STRICT = {d: f"strict_{d}" for d in DIRECTIONS}
WITHIN = {g: f"within_{GENERATOR_SHORT[g].lower()}" for g in GENERATORS}

V_FULL = "full"

TRANSFER_MODELS = (M_DS1, M_RK1, M_DS1_R5, M_PAIR_R5, M_LR_R5, M_RK3_R3, M_RK3)
MATRIX_MODELS = (M_DS1, M_RK1, M_RK3)
STRICT_MODELS = (M_DS1, M_RK1, M_RK3)
CURVE_MODELS = (M_DS1, M_RK1, M_DS1_R5, M_GBP_R5, M_PAIR_R5, M_LR_R5, M_RK3_R3, M_RK3)

# The five new families, and the brief's five groups, each mapped onto the frozen R3 families
# as well. Visual evidence is an R3 family the brief does not name; it is ablated on its own.
FAM_EDIT = "rx_edit_"
FAM_UNC = "rx_unc_"
FAM_LANG = "rx_lang_"
FAM_AGREE = "rx_agree_"
FAM_PAGE = "rx_page_"
NEW_FAMILIES = (FAM_EDIT, FAM_UNC, FAM_LANG, FAM_AGREE, FAM_PAGE)
BRIEF_GROUPS: dict[str, dict[str, Any]] = {
    "edit": {"new": FAM_EDIT, "frozen": (rl1.FAM_EDIT,)},
    "ocr_uncertainty": {"new": FAM_UNC, "frozen": (rl1.FAM_CONF,)},
    "generator_independent": {
        "new": FAM_LANG,
        "frozen": (rl1.FAM_TEXT, rl1.FAM_PLAUS, rl1.FAM_CTX),
    },
    "agreement": {"new": FAM_AGREE, "frozen": ()},
    "page_context": {"new": FAM_PAGE, "frozen": (rl1.FAM_GEOM,)},
}
VISUAL_FROZEN = (rl1.FAM_VIS,)

# Ablations of the primary model. `minus_new_*` drops one new family; `minus_group_*` drops a
# whole brief group, frozen and new; the agreement group is new only, so it has one entry.
ABLATIONS: dict[str, tuple[str, ...]] = {
    **{f"minus_new_{name}": (spec["new"],) for name, spec in BRIEF_GROUPS.items()},
    **{
        f"minus_group_{name}": (spec["new"], *spec["frozen"])
        for name, spec in BRIEF_GROUPS.items()
        if spec["frozen"]
    },
    "minus_visual": VISUAL_FROZEN,
}
TRANSFER_ABLATIONS = tuple(f"minus_new_{name}" for name in BRIEF_GROUPS)

RANK_CUTOFFS = rk1.RANK_CUTOFFS
COVERAGE_GRID = tuple(round(0.02 * i, 2) for i in range(1, 51))
HARM_GRID = tuple(round(0.01 * i, 2) for i in range(0, 31))

# Success criteria, frozen before any endpoint. The floor is RK1's criterion-B floor.
TRANSFER_HARM_FLOOR = rk1.TRANSFER_HARM_FLOOR
PERMUTATION_BAND = rk2.NEAR_RANDOM_BAND
SAFETY_BOUND_LEVEL = 0.95

OUTCOME_TAXONOMY = {
    "A": (
        "the richer representation and the risk-aware objective both improve safe selection, "
        "and the ranker transfers to an unseen generator"
    ),
    "B": "selection improves on known generators, but the ranker does not transfer",
    "C": "the ranker transfers, but at most one lever improves selection on known generators",
    "D": "neither the representation, the objective nor transfer improves on the baselines",
    "E": "a lever improves a criterion, but the risk-aware ranker's own cutoff does not hold",
}
NEXT_STAGE = {
    "A": "NEXT: CONFIRM THE RISK-AWARE RANKER ON NEW PAGES BEFORE ANY DEPLOYMENT CLAIM",
    "B": "NEXT: TARGET-GENERATOR ADAPTATION OF THE RISK-AWARE RANKER WITH PAGE-LEVEL CUTOFFS",
    "C": "NEXT: RECOVER IN-DISTRIBUTION SELECTION WITHOUT LOSING TRANSFER",
    "D": "NEXT: NEW VERIFICATION EVIDENCE, NOT ANOTHER RANKING OBJECTIVE OR FEATURE SET",
    "E": "NEXT: A CUTOFF THAT HOLDS FOR THE RISK-AWARE RANKER BEFORE ANY SELECTION CLAIM",
}

PRIMARY_FAMILY = (
    "P1_recall_r5_vs_r3_risk_objective",
    "P2_recall_rk3_vs_pointwise_r5",
    "P3_recall_rk3_vs_rk1",
    "P4_top1_rk3_vs_rk1",
    "P5_harm_auroc_rk3_vs_ds1",
    "P6_transfer_harm_auroc_rk3_vs_rk1",
)

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_ds1": {"outcome": "B", "production_ready": False},
    "sgv_rk1": {
        "outcome": "B",
        "recommended_next_stage": "NEXT: GENERATOR ADAPTATION OF THE RANKER UNDER A LABEL BUDGET",
        "ready_for_external_confirmation": False,
    },
    "sgv_rk2": {
        "outcome": "D",
        "recommended_next_stage": "NEXT: WITHIN-BUDGET THRESHOLD CALIBRATION FOR A NEW GENERATOR",
        "ready_for_external_confirmation": False,
    },
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_rk3-{artifact}-v{SCHEMA_VERSION}",
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


def _num(value: Any) -> float | None:
    return rk2._num(value)


def assign_outcome(criteria: dict[str, bool]) -> str:
    """Exactly one outcome. No gain is D; any gain with an unsafe primary cutoff is E.

    With the primary cutoff holding: transfer plus both in-distribution levers is A, transfer
    with at most one lever is C, and one or both levers without transfer is B.
    """
    gains = criteria["C1"] or criteria["C2"]
    if not (gains or criteria["C3"]):
        return "D"
    if not criteria["S1"]:
        return "E"
    if criteria["C3"]:
        return "A" if (criteria["C1"] and criteria["C2"]) else "C"
    return "B"


def utility_of(grades: np.ndarray, table: dict[int, float]) -> np.ndarray:
    """A grade vector mapped through a frozen utility table."""
    return np.asarray([table[int(g)] for g in grades.tolist()], dtype=np.float64)


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    files = list(rk2._upstream_files())
    files.append(REPO / "scripts/sgv_rk2_fewshot_ranker_adaptation.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(rk2.OUT.glob(pattern)))
    files.append(rk2.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(rk2.upstream_checks())
    decisions = {
        "sgv_ds1": cc_read_json(ds1.DECISION),
        "sgv_rk1": cc_read_json(rk1.DECISION),
        "sgv_rk2": cc_read_json(rk2.DECISION),
    }
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    for stage, module in (("sgv_rk1", rk1), ("sgv_rk2", rk2)):
        checks.append(
            _check(
                f"{stage}.determinism",
                cc_read_json(module.DETERMINISM)["all_runs_identical"],
                True,
            )
        )
        checks.append(
            _check(
                f"{stage}.report_traceability",
                cc_read_json(module.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
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
    return any(p.exists() for p in (RANKING_SCORES, RANKING_METRICS, DECISION))


def run_freeze() -> int:
    started = time.monotonic()
    for path in (RESEARCH_FREEZE, FROZEN_CONFIGURATION):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("an RK3 endpoint already exists; the freeze must precede every one")
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
            "regenerates_a_candidate": False,
            "regenerates_ocr": False,
            "changes_an_upstream_outcome": False,
            "ranking_libraries": cc_read_json(rk1.RESEARCH_FREEZE)["ranking_libraries"],
            "uses_ground_truth": False,
        },
    )
    rk1_decision = cc_read_json(rk1.DECISION)
    rk2_decision = cc_read_json(rk2.DECISION)
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "r3_columns": len(rl1.columns_for(rl1.PRIMARY_REPRESENTATION)),
            "rk1_feature_matrix": _relative(rk1.FEATURE_MATRIX),
            "rk1_ranking_population": _relative(rk1.RANKING_POPULATION),
            "rl1_fitted_resources": _relative(rl1.FITTED_RESOURCES),
            "document_folds": _relative(rl1.GROUP_REGISTRY),
            "ds1_blocks": {
                "fit": list(rk1.FIT_FOLDS),
                "threshold": list(rk1.THRESHOLD_FOLDS),
                "test": list(rk1.TEST_FOLDS),
            },
            "rk1_outcome": rk1_decision["outcome"],
            "rk1_transfer_mean_harm_auroc": cc_read_json(rk1.TRANSFER)["primary_ranker"][
                "mean_logo_harm_auroc"
            ],
            "rk1_deployment_recall": cc_read_json(rk1.DEPLOYMENT)["primary_ranker_recall"],
            "ds1_score_recall_same_rule": cc_read_json(rk1.DEPLOYMENT)[
                "ds1_score_recall_same_rule"
            ],
            "rk2_outcome": rk2_decision["outcome"],
            "rk2_blocks": _relative(rk2.SPLIT_REGISTRY),
            "lambdamart": cc_read_json(rk1.DESIGN_RECORD)["lambdamart"],
            "uses_ground_truth": False,
        },
    )
    print(f"freeze: {len(hashes)} upstream files hashed ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the pre-registration


def run_preregister() -> int:
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("an RK3 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_questions": {
                "RQ1": "can enhanced candidate representations improve exact correction ranking?",
                "RQ2": (
                    "can pairwise or listwise ranking outperform the existing pointwise "
                    "reliability model?"
                ),
                "RQ3": "can the ranking model transfer across correction generators?",
            },
            "primary_hypothesis": (
                "label-free evidence about the edited word, its context, glyph confusability, "
                "the page and cross-generator agreement, ranked with a utility that prices harm "
                "at the primary target, repairs more error sites at harm <= 0.1 than RK1 and than "
                "the pointwise reliability model, and transfers better to an unseen generator"
            ),
            "representations": {
                R3: "SGV-RL1's R3, 93 columns, exactly as SGV-RK1 and SGV-RK2 used it",
                R5: (
                    "R3 plus five new label-free families: rx_edit_ (edit shape and glyph "
                    "confusability), rx_unc_ (page-relative confidence and lexical ambiguity), "
                    "rx_lang_ (the edited word, its fit to context and to the page), rx_agree_ "
                    "(cross-generator agreement at the site) and rx_page_ (page difficulty, "
                    "engine and document statistics)"
                ),
                "new_feature_sources": (
                    "the frozen OCR stream and spans, the frozen HY1 candidate text and spans, "
                    "RL1's ADAPTATION-fitted resources (language model, lexicon, glyph "
                    "prototypes, confidence quantiles) and ADAPTATION page streams. No outcome, "
                    "label, distance to ground truth or ground-truth string is read"
                ),
                "agreement": (
                    "computed over the two distinct generators only. The text-only Qwen "
                    "condition shares the image corrector's checkpoint, so agreement with it "
                    "would be self-agreement; it is not part of this stage"
                ),
                "engine": (
                    "the page's OCR engine family enters as a page-context feature. It would be "
                    "undefined under a held-out engine, which no protocol here uses"
                ),
            },
            "brief_groups": {
                name: {"new": spec["new"], "frozen": list(spec["frozen"])}
                for name, spec in BRIEF_GROUPS.items()
            },
            "utility": {
                "standard_gain": {str(k): v for k, v in STANDARD_GAIN.items()},
                "risk_utility": {str(k): v for k, v in RISK_UTILITY.items()},
                "harm_penalty_multiple": HARM_PENALTY_MULTIPLE,
                "derivation": (
                    "the non-harmful gains are RK1's 2^grade - 1. The harmful gain is minus "
                    "(1 - epsilon) / epsilon neutral gains at the primary target epsilon = 0.1, "
                    "so an accept set whose harm rate is exactly epsilon and whose other accepts "
                    "are neutral has zero total utility. It is derived from the target, not tuned"
                ),
                "grades": "RK1's: exact 3, partial improvement 2, neutral 1, harmful 0",
            },
            "objectives": OBJECTIVE_LABEL,
            "normalization": {
                OBJ_LAMBDA: "RK1's: each query's deltas divided by its ideal DCG",
                OBJ_RISK: (
                    "each query's deltas divided by its utility range, ideal DCG minus worst "
                    "DCG, which stays positive when negative gains make the ideal DCG negative"
                ),
                OBJ_PAIRWISE: "each query's pair weights divided by their sum",
                OBJ_POINTWISE: "none: squared loss on the utility over every fit row",
            },
            "boosting": {
                "rounds": ROUNDS,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "min_samples_leaf": MIN_LEAF,
                "sigma": SIGMA,
                "leaf_values": "Newton step: summed gradients over summed curvature",
                "queries": "documents, as RK1's primary ranker",
                "tuned": False,
                "source": "SGV-RK1's LambdaMART settings, shared by every tree objective",
            },
            "models": {
                model: {"objective": spec[0], "representation": spec[1]}
                for model, spec in MODEL_SPEC.items()
            },
            "primary_model": PRIMARY_MODEL,
            "model_roles": {
                M_DS1: "SGV-DS1's reliability score, reproduced exactly",
                M_RK1: "SGV-RK1's primary ranker, reproduced exactly",
                M_RK3: "the proposed risk-aware LambdaRank on R5",
                M_RK3_R3: "RQ1: the same objective on R3, isolating the representation",
                M_DS1_R5: "RQ2: the pointwise reliability model on R5, isolating the objective",
                M_GBP_R5: "RQ2: a pointwise fit with RK3's trees and utility",
                M_PAIR_R5: "RQ2: the pairwise loss with RK3's trees and utility",
                M_LR_R5: "the standard LambdaRank gain on R5, isolating the risk-aware utility",
            },
            "protocols": {
                K_IN: (
                    "SGV-DS1's blocks unchanged: RL1 folds 0 and 1 fit, fold 2 chooses cutoffs, "
                    "folds 3 and 4 are the test block"
                ),
                K_TDOC: (
                    "PRIMARY transfer protocol. Training rows are the source generator's fit-"
                    "block rows that the held-out generator did not also propose; cutoffs come "
                    "from the source generator's threshold-block rows by the same rule; the "
                    "evaluated rows are every test-block row the held-out generator proposed. "
                    "Documents never cross, so no site does. The held-out generator's unlabelled "
                    "outputs are visible to the agreement features as context on every page; its "
                    "labels and rows are never fitted on"
                ),
                K_WITHIN: (
                    "each generator's own fit rows train and its own test rows evaluate; the "
                    "diagonal of the transfer matrix"
                ),
                K_STRICT: (
                    "SECONDARY. SGV-RK2's sealed blocks exactly: RL1's strict site rule, source "
                    "fit on folds 0-1, source calibration on fold 2, target test on folds 3-4. "
                    "There a target site never carries a source candidate, so agreement features "
                    "are constant; the protocol places RK3's zero-shot ranker on RK2's label curve"
                ),
            },
            "deployment_policy": (
                "SGV-DS1's policy with each fit's score: one decision per site, the top candidate "
                "among the evaluated rows, accepted when its safety score clears the loosest "
                "cutoff meeting the harm target on the threshold rows. Cutoffs are chosen on "
                "threshold rows alone and applied unchanged"
            ),
            "recall_denominator": {
                K_IN: "every test-block evaluable error site, as SGV-DS1 and SGV-RK1",
                K_TDOC: "every test-block evaluable error site, the same for every cell",
                K_WITHIN: "every test-block evaluable error site",
                K_STRICT: "SGV-RK2's target-reachable error sites, as RK2",
            },
            "metrics": {
                "primary": [
                    "exact-repair recall at harm <= 0.1",
                    "Top-1 on choice sites",
                    "harm AUROC",
                    "mean harm AUROC over the two held-out-generator directions",
                ],
                "definitions": cc_read_json(rk1.DESIGN_RECORD)["metrics"],
                "curves": (
                    "ANALYSIS ONLY: the test decisions swept over every distinct cutoff. The "
                    "coverage curve reads recall and harm at a fixed share accepted; the "
                    "harm-recall frontier reads the highest recall any cutoff reaches at each "
                    "realized harm. Neither chooses a cutoff"
                ),
            },
            "ablations": {name: list(drop) for name, drop in ABLATIONS.items()},
            "transfer_ablations": list(TRANSFER_ABLATIONS),
            "criteria": {
                "C1": (
                    f"RQ1: at harm <= {PRIMARY_EPSILON} on the test block, {M_RK3} repairs "
                    f"more error sites than {M_RK3_R3}, both holding the target"
                ),
                "C2": (
                    f"RQ2: at harm <= {PRIMARY_EPSILON}, {M_RK3} repairs more error sites than "
                    f"{M_DS1_R5}, both holding the target"
                ),
                "C3": (
                    f"RQ3: {M_RK3}'s mean harm AUROC over the two generator-and-document held-"
                    f"out directions exceeds {TRANSFER_HARM_FLOOR}"
                ),
                "S1": f"{M_RK3}'s in-distribution operating point holds harm <= {PRIMARY_EPSILON}",
                "S2": (
                    f"the one-sided {SAFETY_BOUND_LEVEL} document-clustered bootstrap bound on "
                    f"that operating point's test harm is at most {PRIMARY_EPSILON}"
                ),
                "S3": (
                    "in both held-out-generator directions, the source-chosen cutoff holds the "
                    "target on the unseen generator and repairs at least one error site"
                ),
                "deployment_claim": (
                    "no deployment statement is made unless S1, S2 and S3 all hold, and even "
                    "then it is development evidence, never a production claim"
                ),
                "reading": (
                    "criteria are thresholds applied as written. Whether a met criterion is also "
                    "statistically supported is recorded beside it and named in the outcome label"
                ),
            },
            "outcome_rule": OUTCOME_TAXONOMY,
            "next_stage_rule": NEXT_STAGE,
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": "document, resampled within environment, one draw shared by paired arms",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "statistic": "pooled over the evaluated units, as the criteria name them",
                "cutoffs": "chosen once on the threshold rows and held fixed in every resample",
                "p_value": "two-sided bootstrap p with the +1 correction",
                "multiplicity": "Holm within the frozen family of six",
            },
            "controls": {
                "permutation": (
                    f"{PERMUTATION_DRAWS} refits of {M_RK3} with grades shuffled inside each "
                    f"training document; mean test harm AUROC must fall in {list(PERMUTATION_BAND)}"
                ),
                "random": f"{M_RANDOM}, seeded, scored on every protocol's test rows",
            },
            "disclosures": {
                "label_peek": (
                    "during exploration, before this record, two label summaries were printed "
                    "over the whole U5 universe: the exact, beneficial and harmful rates of the "
                    "12 candidates two generators proposed identically (0.75, 0.83 and 0.08 "
                    "against 0.09, 0.24 and 0.56 elsewhere), and the same rates by generator. "
                    "The agreement family is the brief's own and was not chosen from them, but "
                    "they were seen"
                ),
                "label_blind_probes": (
                    "site counts by generator mix (316 of 2,835 sites carry both generators), "
                    "the label-free similarity of the generators' candidates at those sites, "
                    "HY1's proposal strata by generator and environment load times"
                ),
            },
            "non_goals": [
                "no candidate is generated and no OCR output is regenerated",
                "no generator identity enters any model",
                "no large model is introduced",
                "no hyperparameter is searched",
                "the confirmatory reserve is not touched",
                "no certification, production claim or external confirmation",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the new features

WORD_CONTEXT_CHARS = 8
LOW_CONFIDENCE_POINT = 100  # index of the 10th percentile in RL1's 1001-point quantile grid
ENGINE_FAMILIES = ("doctr", "easyocr", "paddleocr", "tesseract")
HISTORICAL_CORPUS = "ocrd_sbb"

EDIT_NAMES: tuple[tuple[str, str], ...] = (
    ("rx_edit_levenshtein", "character edit distance from the OCR text to the candidate text"),
    ("rx_edit_normalized_cost", "that distance over the longer of the two texts"),
    (
        "rx_edit_word_normalized_cost",
        "edit distance between the edited word before and after the edit, over the longer word",
    ),
    ("rx_edit_sub_share", "substitutions among the edit operations"),
    ("rx_edit_ins_share", "insertions among the edit operations"),
    ("rx_edit_del_share", "deletions among the edit operations"),
    (
        "rx_edit_glyph_similarity_mean",
        "mean ADAPTATION glyph-prototype similarity of the substituted character pairs",
    ),
    ("rx_edit_glyph_similarity_min", "lowest glyph-prototype similarity of a substituted pair"),
    (
        "rx_edit_glyph_pairs_available",
        "share of substituted pairs whose two characters both have a glyph prototype",
    ),
    (
        "rx_edit_char_logfreq_gain",
        "mean ADAPTATION log-frequency of the characters introduced minus of those removed",
    ),
    (
        "rx_edit_unseen_char_introduced",
        "1 when the edit introduces a character the engine never output on ADAPTATION pages",
    ),
)
UNC_NAMES: tuple[tuple[str, str], ...] = (
    (
        "rx_unc_anchor_page_percentile",
        "the anchor's mean confidence as a mid-rank percentile among its own page's spans",
    ),
    (
        "rx_unc_low_confidence_anchor",
        "1 when the anchor's mean confidence is below the ADAPTATION 10th percentile",
    ),
    (
        "rx_unc_lexicon_neighbours_original",
        "log of one plus the ADAPTATION lexicon words one edit away from the OCR word",
    ),
    ("rx_unc_lexicon_neighbours_candidate", "the same for the candidate word"),
    (
        "rx_unc_candidate_is_neighbour",
        "1 when the candidate word is a lexicon word one edit away from the OCR word",
    ),
    (
        "rx_unc_unique_neighbour_repair",
        "1 when the OCR word has exactly one lexicon neighbour and the candidate word is it",
    ),
)
LANG_NAMES: tuple[tuple[str, str], ...] = (
    ("rx_lang_word_len_original", "characters in the OCR word the edit falls in"),
    ("rx_lang_word_len_candidate", "characters in that word after the edit"),
    ("rx_lang_word_lexicon_original", "share of the OCR word's tokens in the ADAPTATION lexicon"),
    ("rx_lang_word_lexicon_candidate", "the same share after the edit"),
    ("rx_lang_word_lexicon_gain", "lexicon share gained by the edit"),
    (
        "rx_lang_word_page_support_original",
        "occurrences of the OCR word's tokens elsewhere on the page",
    ),
    (
        "rx_lang_word_page_support_candidate",
        "occurrences of the edited word's tokens elsewhere on the page",
    ),
    ("rx_lang_word_page_support_gain", "page support gained by the edit"),
    (
        "rx_lang_context_lm_gain",
        "ADAPTATION language-model log-probability of the candidate inside its surrounding "
        "text minus that of the OCR text",
    ),
    ("rx_lang_context_lm_gain_per_char", "that gain over the longer of the two edited texts"),
    (
        "rx_lang_page_trigram_support_gain",
        "share of the edited word's character trigrams seen elsewhere on the page, minus the "
        "OCR word's",
    ),
    (
        "rx_lang_page_unseen_chars_gain",
        "characters of the OCR word seen nowhere else on the page, minus the edited word's",
    ),
    (
        "rx_lang_case_consistency_gain",
        "how much closer the edited word's upper-case share is to its line's than the OCR word's",
    ),
    (
        "rx_lang_mixed_class_gain",
        "1 when the edit removes a letter-digit mixture from the word, -1 when it creates one",
    ),
)
AGREE_NAMES: tuple[tuple[str, str], ...] = (
    ("rx_agree_exact", "1 when two distinct generators proposed this exact candidate"),
    ("rx_agree_source_diversity", "distinct generators with any candidate at this site"),
    (
        "rx_agree_other_generator_present",
        "1 when a generator that did not propose this candidate proposed something at the site",
    ),
    (
        "rx_agree_best_similarity",
        "highest normalized similarity of this candidate's text over the site's union window to "
        "another generator's, 1 for an exact agreement, 0 with no other generator",
    ),
    (
        "rx_agree_same_span",
        "1 when another generator replaces exactly the same character span, or agrees exactly",
    ),
    (
        "rx_agree_same_result",
        "1 when another generator's candidate yields the same text over the site's union window",
    ),
)
PAGE_NAMES: tuple[tuple[str, str], ...] = (
    ("rx_page_log_tokens", "log of one plus the tokens in the page stream"),
    ("rx_page_log_chars", "log of one plus the characters in the page stream"),
    ("rx_page_digit_share", "digits among the page's non-space characters"),
    ("rx_page_upper_share", "upper-case letters among the page's letters"),
    ("rx_page_nonascii_share", "non-ASCII characters among the page's non-space characters"),
    ("rx_page_mean_token_len", "mean token length on the page"),
    (
        "rx_page_low_confidence_share",
        "page spans whose confidence is below the ADAPTATION 10th percentile",
    ),
    ("rx_page_confidence_missing_share", "page spans reporting no confidence"),
    ("rx_page_oov_share", "page tokens absent from the ADAPTATION lexicon"),
    (
        "rx_page_lm_per_char",
        "the page stream's length-normalized ADAPTATION language-model log-probability",
    ),
    ("rx_page_site_density", "candidate-bearing proposal sites per hundred page tokens"),
    *(
        (f"rx_page_engine_{family}", f"1 when the page was read by the {family} engine family")
        for family in ENGINE_FAMILIES
    ),
    ("rx_page_domain_historical", "1 for the historical newspaper corpus"),
)
NEW_NAMES: tuple[tuple[str, str], ...] = (
    *EDIT_NAMES,
    *UNC_NAMES,
    *LANG_NAMES,
    *AGREE_NAMES,
    *PAGE_NAMES,
)
NEW_COLUMNS: tuple[str, ...] = tuple(name for name, _definition in NEW_NAMES)
OBSERVATION_COLUMNS = (
    "candidate_id",
    "environment",
    "document_id",
    "site_group",
    "site_id",
    "corrector_sources",
)
LABEL_FIELDS = frozenset(
    {
        "outcome",
        "is_harmful",
        "beneficial",
        "exact",
        "grade",
        "d_before",
        "d_after",
        "error_kind",
    }
)


def r3_columns() -> list[str]:
    return rl1.columns_for(rl1.PRIMARY_REPRESENTATION)


def columns_for(representation: str, drop: Sequence[str] = ()) -> list[str]:
    """One representation's columns in frozen order, less any dropped family prefixes."""
    base = [*r3_columns(), *(NEW_COLUMNS if representation == R5 else ())]
    return [c for c in base if not c.startswith(tuple(drop))] if drop else base


def family_of(column: str) -> str:
    for prefix in NEW_FAMILIES:
        if column.startswith(prefix):
            return prefix
    return rl1.family_of(column)


def observation_view(population: pd.DataFrame) -> pd.DataFrame:
    """The label-free projection the featurizer is allowed to read."""
    return population[list(OBSERVATION_COLUMNS)].drop_duplicates("candidate_id").copy()


def generators_of(sources: str) -> frozenset[str]:
    return frozenset(s for s in str(sources).split("|") if s in GENERATORS)


def word_window(stream: str, start: int, end: int) -> tuple[int, int]:
    """The edited span widened to the whitespace on either side of it."""
    left = start
    while left > 0 and not stream[left - 1].isspace():
        left -= 1
    right = end
    while right < len(stream) and not stream[right].isspace():
        right += 1
    return left, right


def _similarity(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    return 1.0 - levenshtein(left, right) / longest if longest else 1.0


def _upper_share(text: str) -> float | None:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else None


def _mixed(text: str) -> bool:
    return any(c.isalpha() for c in text) and any(c.isdigit() for c in text)


def _trigrams(text: str) -> list[str]:
    return [t[i : i + 3] for t in cc._tokens(text) for i in range(max(len(t) - 2, 0))]


class Lexicon:
    """One environment's ADAPTATION lexicon with a single-deletion index for distance-1 lookups."""

    __slots__ = ("_cache", "_index", "words")

    def __init__(self, words: frozenset[str]) -> None:
        self.words = words
        self._index: dict[str, set[str]] = {}
        for word in sorted(words):
            for variant in {word, *(word[:i] + word[i + 1 :] for i in range(len(word)))}:
                self._index.setdefault(variant, set()).add(word)
        self._cache: dict[str, tuple[str, ...]] = {}

    def neighbours(self, query: str) -> tuple[str, ...]:
        """Lexicon words exactly one edit away, sorted. Candidates share a deletion variant."""
        if query not in self._cache:
            keys = {query, *(query[:i] + query[i + 1 :] for i in range(len(query)))}
            found: set[str] = set()
            for key in keys:
                found |= self._index.get(key, set())
            self._cache[query] = tuple(sorted(w for w in found if levenshtein(query, w) == 1))
        return self._cache[query]


@dataclass(slots=True)
class EnvironmentEvidence:
    """What one environment contributes to every new feature, built once from label-free data."""

    lexicon: Lexicon
    fitted: Any
    prototypes: dict[str, np.ndarray]
    char_counts: Counter[str]
    char_total: int
    low_confidence: float | None
    engine: str
    historical: bool


def environment_evidence(environment: Any, record: dict[str, Any]) -> EnvironmentEvidence:
    """RL1's ADAPTATION resources for one environment, plus its ADAPTATION character counts."""
    engine = environment.spec["base_engine"]
    counts: Counter[str] = Counter()
    for document in record["documents"]:
        page = environment.pages.get((document, engine))
        if page is not None:
            counts.update(c for c in page.stream if not c.isspace())
    prototypes = {}
    for character, grid in record["glyph_prototypes"].items():
        vector = np.asarray(grid, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            prototypes[str(character)] = vector / norm
    quantiles = record["conf_quantiles"]
    return EnvironmentEvidence(
        lexicon=Lexicon(frozenset(str(t) for t in record["lexicon"])),
        fitted=rl1._resources_of(record),
        prototypes=prototypes,
        char_counts=counts,
        char_total=int(sum(counts.values())),
        low_confidence=float(quantiles[LOW_CONFIDENCE_POINT]) if quantiles else None,
        engine=str(engine),
        historical=bool(environment.spec["corpus"] == HISTORICAL_CORPUS),
    )


def _logfreq(evidence: EnvironmentEvidence, character: str) -> float:
    vocabulary = len(evidence.char_counts) + 1
    return math.log(
        (evidence.char_counts.get(character, 0) + 1) / (evidence.char_total + vocabulary)
    )


def edit_features(
    original: str, candidate: str, o_word: str, y_word: str, evidence: EnvironmentEvidence
) -> list[float]:
    """Edit shape, glyph confusability and character rarity. Nothing here reads a label."""
    from rapidfuzz.distance import Levenshtein as Alignment

    operations = Alignment.editops(original, candidate)
    counts = Counter(op[0] for op in operations)
    total = max(len(operations), 1)
    pairs: list[tuple[str, str]] = []
    introduced: list[str] = []
    removed: list[str] = []
    for tag, source, target in operations:
        if tag == "replace":
            pairs.append((original[source], candidate[target]))
            introduced.append(candidate[target])
            removed.append(original[source])
        elif tag == "insert":
            introduced.append(candidate[target])
        elif tag == "delete":
            removed.append(original[source])
    similarities = [
        float(evidence.prototypes[a] @ evidence.prototypes[b])
        for a, b in pairs
        if a in evidence.prototypes and b in evidence.prototypes
    ]
    distance = levenshtein(original, candidate)
    introduced_freq = [_logfreq(evidence, c) for c in introduced if not c.isspace()]
    removed_freq = [_logfreq(evidence, c) for c in removed if not c.isspace()]
    return [
        float(distance),
        distance / max(len(original), len(candidate), 1),
        levenshtein(o_word, y_word) / max(len(o_word), len(y_word), 1),
        counts.get("replace", 0) / total,
        counts.get("insert", 0) / total,
        counts.get("delete", 0) / total,
        float(np.mean(similarities)) if similarities else 0.0,
        float(min(similarities)) if similarities else 0.0,
        len(similarities) / len(pairs) if pairs else 0.0,
        (float(np.mean(introduced_freq)) if introduced_freq else 0.0)
        - (float(np.mean(removed_freq)) if removed_freq else 0.0),
        1.0
        if any(evidence.char_counts.get(c, 0) == 0 for c in introduced if not c.isspace())
        else 0.0,
    ]


def uncertainty_features(
    anchor_confidence: float | None,
    page_confidences: Sequence[float],
    o_word: str,
    y_word: str,
    evidence: EnvironmentEvidence,
) -> list[float]:
    """The anchor's confidence relative to its own page, and how crowded the lexicon is here."""
    if anchor_confidence is None or not page_confidences:
        percentile = 0.5
    else:
        values = np.asarray(page_confidences, dtype=np.float64)
        below = float((values < anchor_confidence).sum())
        equal = float((values == anchor_confidence).sum())
        percentile = (below + 0.5 * equal) / values.size
    low = (
        1.0
        if (
            anchor_confidence is not None
            and evidence.low_confidence is not None
            and anchor_confidence < evidence.low_confidence
        )
        else 0.0
    )
    o_key, y_key = o_word.strip(), y_word.strip()
    o_neighbours = evidence.lexicon.neighbours(o_key) if o_key else ()
    y_neighbours = evidence.lexicon.neighbours(y_key) if y_key else ()
    is_neighbour = bool(y_key in evidence.lexicon.words and y_key in o_neighbours)
    return [
        percentile,
        low,
        math.log1p(len(o_neighbours)),
        math.log1p(len(y_neighbours)),
        1.0 if is_neighbour else 0.0,
        1.0 if (is_neighbour and len(o_neighbours) == 1) else 0.0,
    ]


def language_features(
    original: str,
    candidate: str,
    o_word: str,
    y_word: str,
    before: str,
    after: str,
    line_context: str,
    page: dict[str, Any],
    evidence: EnvironmentEvidence,
) -> list[float]:
    """The edited word, its fit to the surrounding text and to the rest of the page."""
    lexicon = evidence.lexicon.words
    o_tokens, y_tokens = cc._tokens(o_word), cc._tokens(y_word)

    def lexicon_share(tokens: Sequence[str]) -> float:
        return sum(t in lexicon for t in tokens) / len(tokens) if tokens else 0.0

    own_tokens = Counter(o_tokens)

    def page_support(tokens: Sequence[str]) -> float:
        return float(sum(max(page["tokens"].get(t, 0) - own_tokens.get(t, 0), 0) for t in tokens))

    own_grams = Counter(_trigrams(o_word))

    def trigram_support(word: str) -> float:
        grams = _trigrams(word)
        if not grams:
            return 0.0
        seen = sum(1 for g in grams if page["trigrams"].get(g, 0) - own_grams.get(g, 0) >= 1)
        return seen / len(grams)

    own_chars = Counter(c for c in o_word if not c.isspace())

    def unseen(word: str) -> float:
        characters = {c for c in word if not c.isspace()}
        return float(sum(1 for c in characters if page["chars"].get(c, 0) - own_chars[c] <= 0))

    def context_total(text: str) -> float:
        window = before + text + after
        return evidence.fitted.logprob(window) * max(len(window), 1)

    gain = context_total(candidate) - context_total(original)
    reference = _upper_share(line_context)
    o_upper, y_upper = _upper_share(o_word), _upper_share(y_word)
    consistency = (
        abs(o_upper - reference) - abs(y_upper - reference)
        if (reference is not None and o_upper is not None and y_upper is not None)
        else 0.0
    )
    o_lex, y_lex = lexicon_share(o_tokens), lexicon_share(y_tokens)
    o_support, y_support = page_support(o_tokens), page_support(y_tokens)
    return [
        float(len(o_word)),
        float(len(y_word)),
        o_lex,
        y_lex,
        y_lex - o_lex,
        o_support,
        y_support,
        y_support - o_support,
        gain,
        gain / max(len(original), len(candidate), 1),
        trigram_support(y_word) - trigram_support(o_word),
        unseen(o_word) - unseen(y_word),
        consistency,
        float(_mixed(o_word)) - float(_mixed(y_word)),
    ]


def agreement_features(site_rows: pd.DataFrame) -> dict[str, list[float]]:
    """Cross-generator agreement for every candidate at one site, symmetric in the generators.

    Only the two distinct generators count. A candidate's comparison set is the site's other
    candidates carrying a generator this one lacks, so a generator never agrees with itself.
    """
    generators = [generators_of(s) for s in site_rows["corrector_sources"]]
    present = frozenset().union(*generators)
    results = site_rows["site_result"].astype(str).tolist()
    spans = list(
        zip(site_rows["char_start"].astype(int), site_rows["char_end"].astype(int), strict=True)
    )
    out: dict[str, list[float]] = {}
    for i, candidate in enumerate(site_rows["candidate_id"].astype(str)):
        agreed = len(generators[i]) >= 2
        others = [j for j in range(len(generators)) if not generators[j] <= generators[i]]
        similarity = (
            1.0
            if agreed
            else (max(_similarity(results[i], results[j]) for j in others) if others else 0.0)
        )
        out[candidate] = [
            1.0 if agreed else 0.0,
            float(len(present)),
            1.0 if others else 0.0,
            similarity,
            1.0 if (agreed or any(spans[j] == spans[i] for j in others)) else 0.0,
            1.0 if (agreed or any(results[j] == results[i] for j in others)) else 0.0,
        ]
    return out


def page_summary(
    environment: Any, document: str, evidence: EnvironmentEvidence, sites: int
) -> dict[str, Any]:
    """One page's own statistics. Every one is read from this page and ADAPTATION resources."""
    page = environment.pages[(document, evidence.engine)]
    stream = page.stream
    tokens = cc._tokens(stream)
    characters = [c for c in stream if not c.isspace()]
    letters = [c for c in characters if c.isalpha()]
    confidences: list[float] = []
    missing = 0
    for span_id in page.order:
        span = environment.spans.get(str(span_id))
        if span is None or span.native_conf_recognition is None:
            missing += 1
        else:
            confidences.append(float(span.native_conf_recognition))
    low = evidence.low_confidence
    features = [
        math.log1p(len(tokens)),
        math.log1p(len(stream)),
        _ratio(sum(c.isdigit() for c in characters), len(characters)) if characters else 0.0,
        _ratio(sum(c.isupper() for c in letters), len(letters)) if letters else 0.0,
        _ratio(sum(ord(c) > 127 for c in characters), len(characters)) if characters else 0.0,
        float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        (
            sum(1 for v in confidences if v < low) / len(confidences)
            if (confidences and low is not None)
            else 0.0
        ),
        missing / max(len(page.order), 1),
        (sum(1 for t in tokens if t not in evidence.lexicon.words) / len(tokens))
        if tokens
        else 0.0,
        evidence.fitted.logprob(stream),
        100.0 * sites / max(len(tokens), 1),
        *(1.0 if evidence.engine == family else 0.0 for family in ENGINE_FAMILIES),
        1.0 if evidence.historical else 0.0,
    ]
    return {
        "features": features,
        "tokens": Counter(tokens),
        "trigrams": Counter(_trigrams(stream)),
        "chars": Counter(characters),
        "confidences": confidences,
    }


def _anchor_confidence(environment: Any, anchors: Sequence[str]) -> float | None:
    values = [
        float(environment.spans[str(a)].native_conf_recognition)
        for a in anchors
        if str(a) in environment.spans
        and environment.spans[str(a)].native_conf_recognition is not None
    ]
    return float(np.mean(values)) if values else None


def build_new_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Every new feature of every candidate, from label-free inputs only.

    The loop reads the label-free observation view, the frozen HY1 candidate text and spans, the
    frozen proposal lattice, the frozen OCR of each page and RL1's ADAPTATION-fitted resources.
    It never reads an outcome, a label, a distance or a ground-truth string.
    """
    texts = rl1._hy1_candidates()[
        ["candidate_id", "original_ocr", "candidate_text", "char_start", "char_end"]
    ].drop_duplicates("candidate_id")
    frame = observations.merge(texts, on="candidate_id", how="left", validate="one_to_one")
    if frame["candidate_text"].isna().any():
        raise PhaseError("a candidate has no frozen HY1 text")
    strata = pd.read_parquet(hy1.PROPOSAL_STRATA)
    resources = cc_read_json(rl1.FITTED_RESOURCES)["by_environment"]
    blocks: list[pd.DataFrame] = []
    for spec in rl1.environment_specs():
        name = spec["environment"]
        block = frame[frame["environment"] == name].sort_values("candidate_id", kind="stable")
        if block.empty:
            continue
        began = time.monotonic()
        environment = rl1._environment(spec)
        evidence = environment_evidence(environment, resources[name])
        sites = strata[strata["environment"] == name].set_index("site_id")
        pages = {
            document: page_summary(
                environment,
                document,
                evidence,
                int(block.loc[block["document_id"] == document, "site_group"].nunique()),
            )
            for document in sorted(set(block["document_id"].astype(str)))
        }
        windows: dict[str, tuple[int, int]] = {}
        results: dict[str, str] = {}
        words: dict[str, tuple[str, str, int, int]] = {}
        for row in block.itertuples(index=False):
            stream = environment.pages[(str(row.document_id), evidence.engine)].stream
            start, end = int(row.char_start), int(row.char_end)
            left, right = word_window(stream, start, end)
            words[str(row.candidate_id)] = (
                stream[left:start] + str(row.original_ocr) + stream[end:right],
                stream[left:start] + str(row.candidate_text) + stream[end:right],
                left,
                right,
            )
            low, high = windows.get(str(row.site_group), (left, right))
            windows[str(row.site_group)] = (min(low, left), max(high, right))
        for row in block.itertuples(index=False):
            stream = environment.pages[(str(row.document_id), evidence.engine)].stream
            low, high = windows[str(row.site_group)]
            start, end = int(row.char_start), int(row.char_end)
            results[str(row.candidate_id)] = (
                stream[low:start] + str(row.candidate_text) + stream[end:high]
            )
        block = block.assign(site_result=[results[c] for c in block["candidate_id"].astype(str)])
        agreement: dict[str, list[float]] = {}
        for _site, site_rows in block.groupby("site_group", sort=True):
            agreement.update(agreement_features(site_rows))
        rows: list[list[float]] = []
        for row in block.itertuples(index=False):
            document = str(row.document_id)
            page = environment.pages[(document, evidence.engine)]
            stream = page.stream
            site = sites.loc[str(row.site_id)]
            anchors = gen1.anchors_of(str(site.anchor_ref))
            original, candidate = str(row.original_ocr), str(row.candidate_text)
            start, end = int(row.char_start), int(row.char_end)
            o_word, y_word, left, right = words[str(row.candidate_id)]
            line_start, line_end = (
                page.lines[int(site.line_index)]
                if int(site.line_index) < len(page.lines)
                else (left, right)
            )
            line_context = stream[int(line_start) : left] + stream[right : int(line_end)]
            summary = pages[document]
            values = [
                *edit_features(original, candidate, o_word, y_word, evidence),
                *uncertainty_features(
                    _anchor_confidence(environment, anchors),
                    summary["confidences"],
                    o_word,
                    y_word,
                    evidence,
                ),
                *language_features(
                    original,
                    candidate,
                    o_word,
                    y_word,
                    stream[max(0, start - WORD_CONTEXT_CHARS) : start],
                    stream[end : end + WORD_CONTEXT_CHARS],
                    line_context,
                    summary,
                    evidence,
                ),
                *agreement[str(row.candidate_id)],
                *summary["features"],
            ]
            if len(values) != len(NEW_COLUMNS):
                raise PhaseError(
                    f"{row.candidate_id}: {len(values)} values for {len(NEW_COLUMNS)} features"
                )
            rows.append(values)
        part = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=list(NEW_COLUMNS))
        part.insert(0, "environment", name)
        part.insert(0, "candidate_id", block["candidate_id"].astype(str).to_numpy())
        blocks.append(part)
        print(f"  {name}: {len(part)} candidates ({time.monotonic() - began:.0f}s)", flush=True)
        del environment
    matrix = pd.concat(blocks, ignore_index=True)
    if not np.isfinite(matrix[list(NEW_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise PhaseError("a new feature value is not finite")
    return matrix.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def load_population() -> pd.DataFrame:
    """SGV-RK1's U5 ranking population, in RK1's own row order, with its blocks and grades."""
    return rk1.population_frame(pd.read_parquet(rk1.RANKING_POPULATION), rk1.POP_U5)


def run_features() -> int:
    """The new families for every U5 candidate, and R3 copied from RK1 and verified."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (FEATURE_MATRIX, FEATURE_REGISTRY):
        _forbid(path)
    population = load_population()
    ids = set(population["candidate_id"].astype(str))
    frozen = pd.read_parquet(rk1.FEATURE_MATRIX)
    frozen = frozen[frozen["candidate_id"].isin(ids)]
    published = pd.read_parquet(rl1.FEATURE_MATRIX)
    published = published[published["candidate_id"].isin(ids)]
    columns = r3_columns()
    shared = sorted(set(frozen["candidate_id"]) & set(published["candidate_id"]))
    difference = float(
        np.abs(
            frozen.set_index("candidate_id").loc[shared, columns].to_numpy(np.float64)
            - published.set_index("candidate_id").loc[shared, columns].to_numpy(np.float64)
        ).max()
    )
    if difference != 0.0 or len(shared) != len(ids):
        raise PhaseError(f"R3 does not reproduce RL1 on U5: {difference}, {len(shared)} rows")
    new = build_new_features(observation_view(population))
    matrix = frozen[["candidate_id", "environment", *columns]].merge(
        new.drop(columns=["environment"]), on="candidate_id", how="inner", validate="one_to_one"
    )
    matrix = matrix.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if len(matrix) != len(ids):
        raise PhaseError("the feature matrix does not cover every U5 candidate")
    _write_parquet_once(FEATURE_MATRIX, matrix)
    mixed = population.groupby("site_group")["corrector_sources"].agg(
        lambda s: len(frozenset().union(*(generators_of(x) for x in s)))
    )
    at_mixed = population["site_group"].map(mixed).to_numpy() >= 2
    ordered = matrix.set_index("candidate_id").loc[population["candidate_id"].astype(str)]
    _write_json_once(
        FEATURE_REGISTRY,
        {
            **_envelope("feature_registry"),
            "new_features": [
                {
                    "name": name,
                    "definition": definition,
                    "family": family_of(name),
                    "brief_group": next(
                        g for g, spec in BRIEF_GROUPS.items() if spec["new"] == family_of(name)
                    ),
                    "distinct_values": int(matrix[name].nunique()),
                }
                for name, definition in NEW_NAMES
            ],
            "new_feature_count": len(NEW_COLUMNS),
            "r3_columns": len(columns),
            "r5_columns": len(columns_for(R5)),
            "r3_reproduces_rl1": {"rows": len(shared), "max_absolute_difference": difference},
            "constant_new_features": sorted(n for n in NEW_COLUMNS if matrix[n].nunique() <= 1),
            "agreement_varies_only_at_mixed_sites": {
                "mixed_site_candidates": int(at_mixed.sum()),
                "agreement_nonconstant_outside_mixed_sites": sorted(
                    n
                    for n, _d in AGREE_NAMES
                    if n != "rx_agree_source_diversity" and ordered.loc[~at_mixed, n].nunique() > 1
                ),
            },
            "reads": list(OBSERVATION_COLUMNS),
            "label_fields_read": [],
            "sources": cc_read_json(DESIGN_RECORD)["representations"]["new_feature_sources"],
            "uses_ground_truth": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"features: {len(matrix)} candidates, {len(NEW_COLUMNS)} new columns, R3 reproduces "
        f"RL1 on {len(shared)} rows ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the protocols

ORDER_RK1 = "rk1"
ORDER_RK2 = "rk2"


@dataclass(slots=True)
class Protocol:
    """One protocol's frozen rows, as index arrays into the population it is ordered by."""

    name: str
    kind: str
    ordering: str
    train: np.ndarray
    threshold: np.ndarray
    test: np.ndarray
    sources: tuple[str, ...]
    target: str | None


def strict_population() -> pd.DataFrame:
    """SGV-RK2's population, in RK2's own row order, which its sealed blocks index into."""
    return rk2.load_population()


def frames() -> dict[str, pd.DataFrame]:
    return {ORDER_RK1: load_population(), ORDER_RK2: strict_population()}


def membership(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Which rows each distinct generator proposed; a doubly proposed candidate is in both."""
    sets = [generators_of(s) for s in frame["corrector_sources"].astype(str)]
    return {g: np.asarray([g in s for s in sets], dtype=bool) for g in GENERATORS}


def build_protocols(populations: dict[str, pd.DataFrame]) -> dict[str, Protocol]:
    """Every protocol, from the frozen blocks and generator membership alone. No label is read."""
    population = populations[ORDER_RK1]
    block = population["block"].to_numpy(dtype=str)
    fit, threshold, test = block == "fit", block == "threshold", block == "test"
    has = membership(population)
    out = {
        P_IN: Protocol(
            P_IN,
            K_IN,
            ORDER_RK1,
            np.flatnonzero(fit),
            np.flatnonzero(threshold),
            np.flatnonzero(test),
            GENERATORS,
            None,
        )
    }
    for direction in DIRECTIONS:
        source, target = SOURCE_OF[direction], TARGET_OF[direction]
        source_only = has[source] & ~has[target]
        out[TDOC[direction]] = Protocol(
            TDOC[direction],
            K_TDOC,
            ORDER_RK1,
            np.flatnonzero(fit & source_only),
            np.flatnonzero(threshold & source_only),
            np.flatnonzero(test & has[target]),
            (source,),
            target,
        )
    for generator in GENERATORS:
        out[WITHIN[generator]] = Protocol(
            WITHIN[generator],
            K_WITHIN,
            ORDER_RK1,
            np.flatnonzero(fit & has[generator]),
            np.flatnonzero(threshold & has[generator]),
            np.flatnonzero(test & has[generator]),
            (generator,),
            generator,
        )
    for direction, blocks in rk2.all_blocks(populations[ORDER_RK2]).items():
        out[STRICT[direction]] = Protocol(
            STRICT[direction],
            K_STRICT,
            ORDER_RK2,
            blocks.source_fit,
            blocks.source_calibration,
            blocks.target_test,
            (blocks.source,),
            blocks.target,
        )
    return out


def run_splits() -> int:
    """Every protocol's rows and isolation counts, frozen before any fit."""
    started = time.monotonic()
    _require(FEATURE_REGISTRY, "features")
    _forbid(SPLIT_REGISTRY)
    populations = frames()
    protocols = build_protocols(populations)
    strict_digests = {
        name: row["target_test_digest"]
        for name, row in cc_read_json(rk2.SPLIT_REGISTRY)["directions"].items()
    }
    rows: dict[str, Any] = {}
    for name, protocol in protocols.items():
        frame = populations[protocol.ordering]
        parts = {"train": protocol.train, "threshold": protocol.threshold, "test": protocol.test}
        has = membership(frame)

        def shared(
            column: str, left: np.ndarray, right: np.ndarray, f: pd.DataFrame = frame
        ) -> int:
            return len(
                set(f.iloc[left][column].astype(str)) & set(f.iloc[right][column].astype(str))
            )

        row: dict[str, Any] = {
            "kind": protocol.kind,
            "ordering": protocol.ordering,
            "sources": list(protocol.sources),
            "target": protocol.target,
            **{
                part: {
                    "rows": int(index.size),
                    "documents": int(frame.iloc[index]["document_id"].nunique()),
                    "sites": int(frame.iloc[index]["site_group"].nunique()),
                    "environments": int(frame.iloc[index]["environment"].nunique()),
                    "by_generator": {g: int(has[g][index].sum()) for g in GENERATORS},
                }
                for part, index in parts.items()
            },
            "isolation": {
                "documents_train_vs_test": shared("document_id", protocol.train, protocol.test),
                "documents_train_vs_threshold": shared(
                    "document_id", protocol.train, protocol.threshold
                ),
                "documents_threshold_vs_test": shared(
                    "document_id", protocol.threshold, protocol.test
                ),
                "sites_train_vs_test": shared("site_group", protocol.train, protocol.test),
                "candidates_train_vs_test": shared("candidate_id", protocol.train, protocol.test),
            },
            "test_harmful": int(frame.iloc[protocol.test]["is_harmful"].sum()),
            "test_exact": int(frame.iloc[protocol.test]["exact"].sum()),
        }
        if protocol.target is not None and protocol.kind in (K_TDOC, K_STRICT):
            row["held_out_rows_in_training"] = int(
                has[protocol.target][np.concatenate([protocol.train, protocol.threshold])].sum()
            )
            row["test_rows_not_proposed_by_the_target"] = int(
                (~has[protocol.target][protocol.test]).sum()
            )
        if protocol.kind == K_TDOC:
            source = protocol.sources[0]
            source_sites = set(frame.loc[has[source], "site_group"].astype(str))
            row["test_rows_at_sites_the_source_also_proposed"] = int(
                frame.iloc[protocol.test]["site_group"].astype(str).isin(source_sites).sum()
            )
        if protocol.kind == K_STRICT:
            direction = next(d for d in DIRECTIONS if STRICT[d] == name)
            digest = rk2._digest(frame, protocol.test)
            row["test_digest"] = digest
            row["equals_rk2_sealed_test"] = bool(digest == strict_digests[direction])
        rows[name] = row
    isolated = all(
        value == 0
        for row in rows.values()
        for key, value in row["isolation"].items()
        if key != "documents_train_vs_threshold" or row["kind"] != K_STRICT
    )
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "protocols": rows,
            "every_protocol_isolated": isolated,
            "strict_blocks_equal_rk2": all(
                row["equals_rk2_sealed_test"] for row in rows.values() if row["kind"] == K_STRICT
            ),
            "rules": cc_read_json(DESIGN_RECORD)["protocols"],
            "frozen_before_any_fit": not RANKING_SCORES.exists(),
        },
    )
    if not isolated:
        raise PhaseError("a protocol shares a document, site or candidate across its blocks")
    print(
        "splits: "
        + ", ".join(f"{name} test {row['test']['rows']}" for name, row in rows.items())
        + f" ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the objectives


def _queries(groups: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    """RK1's query rule: one query per group, kept only if it carries two distinct values."""
    keys = pd.Series(groups).astype(str)
    return [
        np.asarray(index, dtype=np.int64)
        for _key, index in sorted(keys.groupby(keys).groups.items())
        if len(set(values[np.asarray(index)].tolist())) >= 2
    ]


def ranking_gradients(
    scores: np.ndarray,
    utility: np.ndarray,
    queries: Sequence[np.ndarray],
    discount: bool,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise logistic gradients weighted by utility differences, one query at a time.

    With a position discount this is the LambdaRank gradient: a pair's weight is what swapping it
    would change in DCG. Given RK1's gain and ideal-DCG normalization it is RK1's gradient term for
    term. The risk-aware utility gives harmful candidates a negative gain, which can make the
    ideal DCG itself negative, so it normalizes by the utility range instead. Without a discount
    every pair is weighted by its utility difference alone, normalized to sum to one per query.
    """
    lambdas = np.zeros(scores.size, dtype=np.float64)
    weights = np.zeros(scores.size, dtype=np.float64)
    for index in queries:
        gain = utility[index]
        size = index.size
        score = scores[index]
        better = gain[:, None] > gain[None, :]
        if discount:
            discount_ideal = 1.0 / np.log2(np.arange(size) + 2.0)
            ideal = float((np.sort(gain)[::-1] * discount_ideal).sum())
            if normalization == "ideal":
                scale = ideal
            elif normalization == "range":
                scale = ideal - float((np.sort(gain) * discount_ideal).sum())
            else:
                raise PhaseError(f"unknown normalization {normalization}")
            if scale <= 0.0:
                continue
            order = np.argsort(-score, kind="mergesort")
            position = np.empty(size, dtype=np.int64)
            position[order] = np.arange(size)
            place = 1.0 / np.log2(position + 2.0)
            delta = (
                np.abs((gain[:, None] - gain[None, :]) * (place[:, None] - place[None, :])) / scale
            )
        else:
            weight = np.abs(gain[:, None] - gain[None, :]) * better
            total = float(weight.sum())
            if total <= 0.0:
                continue
            delta = weight / total
        difference = np.clip(score[:, None] - score[None, :], -50.0, 50.0)
        rho = 1.0 / (1.0 + np.exp(SIGMA * difference))
        pair = SIGMA * rho * delta * better
        curvature = (SIGMA**2) * rho * (1.0 - rho) * delta * better
        lambdas[index] += pair.sum(axis=1) - pair.sum(axis=0)
        weights[index] += curvature.sum(axis=1) + curvature.sum(axis=0)
    return lambdas, weights


def fit_boosted(design: np.ndarray, utility: np.ndarray, groups: np.ndarray, objective: str) -> Any:
    """Boosted regression trees with Newton leaves, under one of three objectives.

    The loop is RK1's LambdaMART loop with the gradient swapped: squared error on the utility for
    the pointwise fit, the undiscounted pairwise gradient, or the risk-aware LambdaRank gradient.
    Trees, depth, leaf size, rounds and learning rate are RK1's for all three.
    """
    from sklearn.tree import DecisionTreeRegressor

    empty = rk1.LambdaMART(trees=[], leaf_values=[], learning_rate=LEARNING_RATE)
    if objective == OBJ_POINTWISE:
        if np.unique(utility).size < 2:
            return empty
        used = np.arange(design.shape[0], dtype=np.int64)

        def gradient(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return utility - scores, np.ones_like(scores)

    elif objective in (OBJ_PAIRWISE, OBJ_RISK):
        queries = _queries(groups, utility)
        if not queries:
            return empty
        used = np.sort(np.concatenate(queries))
        discount = objective == OBJ_RISK
        normalization = "range" if discount else "pairs"

        def gradient(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return ranking_gradients(scores, utility, queries, discount, normalization)

    else:
        raise PhaseError(f"fit_boosted does not fit {objective}")
    scores = np.zeros(design.shape[0], dtype=np.float64)
    trees: list[Any] = []
    leaf_values: list[np.ndarray] = []
    for _round in range(ROUNDS):
        lambdas, curvature = gradient(scores)
        if not np.any(lambdas[used]):
            break
        tree = DecisionTreeRegressor(
            max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=FIT_SEED
        ).fit(design[used], lambdas[used])
        leaves = tree.apply(design[used])
        values = np.zeros(tree.tree_.node_count, dtype=np.float64)
        for leaf in np.unique(leaves):
            member = leaves == leaf
            values[leaf] = float(lambdas[used][member].sum()) / (
                float(curvature[used][member].sum()) + 1e-9
            )
        scores += LEARNING_RATE * values[tree.apply(design)]
        trees.append(tree)
        leaf_values.append(values)
    return rk1.LambdaMART(trees=trees, leaf_values=leaf_values, learning_rate=LEARNING_RATE)


def fit_model(
    model: str, design: np.ndarray, frame: pd.DataFrame, train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Fit one pre-registered model on `train` and return its scorer: (rank score, safety)."""
    objective, _representation = MODEL_SPEC[model]
    if objective == OBJ_LOGISTIC:
        benefit_labels = frame["beneficial"].to_numpy(dtype=bool)[train]
        harm_labels = frame["is_harmful"].to_numpy(dtype=bool)[train]

        def logistic(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            benefit = rl1.fit_and_score(rl1.M0, design[train], benefit_labels, design[rows])
            harm = rl1.fit_and_score(rl1.M0, design[train], harm_labels, design[rows])
            return benefit, 1.0 - harm

        return logistic
    grades = frame["grade"].to_numpy(dtype=np.int64)[train]
    groups = frame["document_id"].astype(str).to_numpy()[train]
    if objective == OBJ_LAMBDA:
        fitted = rk1.fit_lambdamart(design[train], grades, groups)
    else:
        fitted = fit_boosted(design[train], utility_of(grades, RISK_UTILITY), groups, objective)

    def ranked(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = fitted.score(design[rows])
        return value, value

    ranked.trees = len(fitted.trees)  # type: ignore[attr-defined]
    return ranked


def random_scores(protocol: str, rows: int) -> np.ndarray:
    generator = np.random.default_rng(
        int(canonical_hash({"seed": FIT_SEED, "protocol": protocol, "rows": rows})[:8], 16)
    )
    return generator.random(rows)


def score_job(
    model: str,
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    protocol: Protocol,
    drop: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(scored rows, rank score, safety, trees) for one fit: threshold rows first, then test."""
    rows = np.concatenate([protocol.threshold, protocol.test])
    objective, representation = MODEL_SPEC[model]
    if objective == OBJ_RANDOM:
        value = random_scores(protocol.name, int(rows.size))
        return rows, value, value, 0
    design = rl1._matrix_for(frame, matrix, columns_for(representation, drop))
    scorer = fit_model(model, design, frame, protocol.train)
    rank_score, safety = scorer(rows)
    return rows, rank_score, safety, int(getattr(scorer, "trees", 0))


# ------------------------------------------------------------------ reproduction gates

REPRODUCED_BUDGETS = (100, 250, 500)


def _max_difference(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    shared = left.index.intersection(right.index)
    if not len(shared):
        return 0, float("inf")
    return len(shared), float(np.abs(left.loc[shared] - right.loc[shared]).max())


def run_reproduce() -> int:
    """DS1's score, RK1's primary ranker and RK2's arms, reproduced exactly before any RK3 fit."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    _forbid(REPRODUCTION)
    populations = frames()
    matrix = pd.read_parquet(FEATURE_MATRIX)
    protocols = build_protocols(populations)
    frame = populations[ORDER_RK1]
    base = protocols[P_IN]
    out: dict[str, Any] = {}

    rows, rank_score, safety, _trees = score_job(M_DS1, frame, matrix, base)
    ids = frame.iloc[rows]["candidate_id"].astype(str).to_numpy()
    published = pd.read_parquet(ds1.DEPLOYMENT_SCORES).set_index("candidate_id")
    count, rank_gap = _max_difference(pd.Series(rank_score, index=ids), published["score_benefit"])
    _count, safety_gap = _max_difference(pd.Series(safety, index=ids), published["safety"])
    out["ds1_reliability_score"] = {
        "rows": count,
        "rank_score_max_difference": rank_gap,
        "safety_max_difference": safety_gap,
        "identical": bool(rank_gap == 0.0 and safety_gap == 0.0 and count == len(ids)),
    }

    rows, rank_score, _safety, _trees = score_job(M_RK1, frame, matrix, base)
    rk1_scores = pd.read_parquet(rk1.RANKING_SCORES)
    cell = rk1_scores[
        (rk1_scores["split"] == rk1.SPLIT_BLOCKS)
        & (rk1_scores["model"] == rk1.PRIMARY_RANKER)
        & (rk1_scores["variant"] == rk1.V_FULL)
    ].set_index("candidate_id")["rank_score"]
    count, gap = _max_difference(pd.Series(rank_score, index=ids), cell)
    out["rk1_primary_ranker"] = {
        "rows": count,
        "published_rows": len(cell),
        "max_absolute_difference": gap,
        "identical": bool(gap == 0.0 and count == len(ids) == len(cell)),
    }

    strict = populations[ORDER_RK2]
    design = rl1._matrix_for(strict, matrix, r3_columns())
    grades = strict["grade"].to_numpy(dtype=np.int64)
    groups = strict["document_id"].astype(str).to_numpy()
    calibration = rk2.calibration_mask(strict["site_group"].astype(str).tolist())
    strict_ids = strict["candidate_id"].astype(str).to_numpy()
    adapted = pd.read_parquet(rk2.ADAPTED_SCORES)
    cells: dict[str, Any] = {}

    def compare(
        direction: str, arm: str, budget: int, kind: str, index: np.ndarray, model: Any
    ) -> dict[str, Any]:
        stored = adapted[
            (adapted["direction"] == direction)
            & (adapted["arm"] == arm)
            & (adapted["budget"] == budget)
            & (adapted["draw"] == 0)
            & (adapted["evaluation_set"] == kind)
        ].set_index("candidate_id")["score"]
        rebuilt = pd.Series(model.score(design[index]), index=strict_ids[index])
        count, gap = _max_difference(rebuilt, stored)
        return {
            "rows": count,
            "stored_rows": len(stored),
            "max_absolute_difference": gap,
            "identical": bool(gap == 0.0 and count == len(stored) == index.size),
        }

    for direction, blocks in rk2.all_blocks(strict).items():
        source_model = rk2.fit_arm(
            rk2.A0, design, grades, groups, blocks.source_fit, blocks.source_fit[:0], None
        )
        cells[f"{direction}|{rk2.A0}|0|test"] = compare(
            direction, rk2.A0, 0, "test", blocks.target_test, source_model
        )
        cells[f"{direction}|{rk2.A0}|0|calibration"] = compare(
            direction, rk2.A0, 0, "calibration", blocks.source_calibration, source_model
        )
        order = rl2.draw_order(strict, blocks.frozen, 0)
        for budget in REPRODUCED_BUDGETS:
            bought = rl2.purchased(order, budget)
            fit_rows, cal_rows = rk2.split_purchase(bought, calibration)
            model = rk2.fit_arm(
                rk2.A1, design, grades, groups, blocks.source_fit, fit_rows, source_model
            )
            cells[f"{direction}|{rk2.A1}|{budget}|test"] = compare(
                direction, rk2.A1, budget, "test", blocks.target_test, model
            )
            cells[f"{direction}|{rk2.A1}|{budget}|calibration"] = compare(
                direction, rk2.A1, budget, "calibration", cal_rows, model
            )
    out["rk2_arms"] = {
        "cells": cells,
        "draw": 0,
        "budgets": list(REPRODUCED_BUDGETS),
        "identical": all(c["identical"] for c in cells.values()),
    }
    identical = all(block["identical"] for block in out.values())
    _write_json_once(
        REPRODUCTION,
        {
            **_analysis_envelope("reproduction"),
            "by_baseline": out,
            "all_identical": identical,
            "note": (
                "DS1's score and RK1's ranker are refitted on DS1's fit block and compared on "
                "every threshold and test row; RK2's zero-shot arm and its target-only arm on "
                "draw 0 are refitted on RK2's own sealed blocks and purchases"
            ),
        },
    )
    if not identical:
        raise PhaseError(
            f"a baseline did not reproduce: { {k: v['identical'] for k, v in out.items()} }"
        )
    print(
        f"reproduce: DS1, RK1 and {len(cells)} RK2 cells reproduce exactly "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ every fit


def rank_jobs() -> list[dict[str, Any]]:
    """Every pre-registered fit, fixed before any score exists."""
    jobs: list[dict[str, Any]] = []

    def add(protocol: str, model: str, variant: str = V_FULL, drop: tuple[str, ...] = ()) -> None:
        jobs.append({"protocol": protocol, "model": model, "variant": variant, "drop": drop})

    for model in MODELS:
        add(P_IN, model)
    for name, drop in ABLATIONS.items():
        add(P_IN, M_RK3, name, drop)
    for direction in DIRECTIONS:
        for model in (M_RANDOM, *TRANSFER_MODELS):
            add(TDOC[direction], model)
        for name in TRANSFER_ABLATIONS:
            add(TDOC[direction], M_RK3, name, ABLATIONS[name])
    for generator in GENERATORS:
        for model in MATRIX_MODELS:
            add(WITHIN[generator], model)
    for direction in DIRECTIONS:
        for model in (M_RANDOM, *STRICT_MODELS):
            add(STRICT[direction], model)
    return jobs


def run_rank() -> int:
    """Every pre-registered fit, trained on its protocol's train rows and scored on the rest."""
    started = time.monotonic()
    _require(REPRODUCTION, "reproduce")
    for path in (RANKING_SCORES, MODEL_REGISTRY, TRAINING_REGISTRY):
        _forbid(path)
    populations = frames()
    matrix = pd.read_parquet(FEATURE_MATRIX)
    protocols = build_protocols(populations)
    tables: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    for job in rank_jobs():
        began = time.monotonic()
        protocol = protocols[job["protocol"]]
        frame = populations[protocol.ordering]
        rows, rank_score, safety, trees = score_job(
            job["model"], frame, matrix, protocol, job["drop"]
        )
        kinds = np.concatenate(
            [
                np.full(protocol.threshold.size, "threshold", dtype=object),
                np.full(protocol.test.size, "test", dtype=object),
            ]
        )
        tables.append(
            pd.DataFrame(
                {
                    "protocol": protocol.name,
                    "model": job["model"],
                    "variant": job["variant"],
                    "evaluation_set": kinds,
                    "candidate_id": frame.iloc[rows]["candidate_id"].astype(str).to_numpy(),
                    "rank_score": rank_score,
                    "safety": safety,
                }
            )
        )
        columns = (
            []
            if MODEL_SPEC[job["model"]][0] == OBJ_RANDOM
            else columns_for(MODEL_SPEC[job["model"]][1], job["drop"])
        )
        registry.append(
            {
                "protocol": protocol.name,
                "model": job["model"],
                "variant": job["variant"],
                "objective": MODEL_SPEC[job["model"]][0],
                "representation": MODEL_SPEC[job["model"]][1],
                "columns": len(columns),
                "dropped": list(job["drop"]),
                "carries_source_columns": bool(
                    [c for c in columns if c.startswith(rl1.FAM_SOURCE)]
                ),
                "train_rows": int(protocol.train.size),
                "threshold_rows": int(protocol.threshold.size),
                "test_rows": int(protocol.test.size),
                "trees": trees,
                "constant_on_test": bool(np.unique(safety[protocol.threshold.size :]).size <= 1),
            }
        )
        print(
            f"  {protocol.name} {job['model']} {job['variant']}: {time.monotonic() - began:.1f}s",
            flush=True,
        )
    table = pd.concat(tables, ignore_index=True)
    table = table.sort_values(
        ["protocol", "model", "variant", "evaluation_set", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    _write_parquet_once(RANKING_SCORES, table)
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "models": {
                model: {
                    "objective": spec[0],
                    "objective_label": OBJECTIVE_LABEL[spec[0]],
                    "representation": spec[1],
                    "columns": 0 if spec[0] == OBJ_RANDOM else len(columns_for(spec[1])),
                }
                for model, spec in MODEL_SPEC.items()
            },
            "primary_model": PRIMARY_MODEL,
            "utility": cc_read_json(DESIGN_RECORD)["utility"],
            "boosting": cc_read_json(DESIGN_RECORD)["boosting"],
            "established_prior_art": (
                "LambdaMART, LambdaRank, RankNet, pairwise and pointwise learning-to-rank, and "
                "cost-sensitive gains are established methods; this stage applies them and "
                "claims none of them"
            ),
            "uses_ground_truth": True,
        },
    )
    _write_json_once(
        TRAINING_REGISTRY,
        {
            **_analysis_envelope("training_registry"),
            "jobs": registry,
            "job_count": len(registry),
            "scored_rows": len(table),
            "untrainable": [
                f"{r['protocol']}|{r['model']}|{r['variant']}"
                for r in registry
                if r["constant_on_test"] and r["objective"] != OBJ_RANDOM
            ],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"rank: {len(registry)} fits, {len(table)} scored rows ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ shared evaluation helpers

LABEL_COLUMNS = (
    "site_group",
    "site_key",
    "document_id",
    "environment",
    "corrector_source",
    "corrector_sources",
    "is_harmful",
    "beneficial",
    "exact",
    "grade",
)


@dataclass(slots=True)
class Context:
    """Everything an evaluation phase reads, loaded once."""

    populations: dict[str, pd.DataFrame]
    protocols: dict[str, Protocol]
    cells: dict[tuple[str, str, str, str], pd.DataFrame]
    errors: pd.DataFrame
    mapping: dict[str, set[str]]


def load_context() -> Context:
    populations = frames()
    protocols = build_protocols(populations)
    scores = pd.read_parquet(RANKING_SCORES)
    labelled = {
        ordering: frame[["candidate_id", *LABEL_COLUMNS]].set_index("candidate_id")
        for ordering, frame in populations.items()
    }
    cells: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for key, group in scores.groupby(["protocol", "model", "variant", "evaluation_set"], sort=True):
        protocol, model, variant, kind = (str(k) for k in key)
        labels = labelled[protocols[protocol].ordering]
        joined = group[["candidate_id", "rank_score", "safety"]].join(
            labels, on="candidate_id", how="inner"
        )
        if len(joined) != len(group):
            raise PhaseError(f"{key}: a scored candidate has no label row")
        cells[(protocol, model, variant, kind)] = joined.reset_index(drop=True)
    errors = ds1.load_error_population()
    return Context(populations, protocols, cells, errors, ds1.site_to_errors(errors))


def restrict(cell: pd.DataFrame, generator: str | None) -> pd.DataFrame:
    """A cell's rows that one generator proposed, or every row."""
    if generator is None:
        return cell
    keep = [generator in generators_of(s) for s in cell["corrector_sources"].astype(str)]
    return cell[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def decisions(frame: pd.DataFrame) -> pd.DataFrame:
    return rk2.decisions(frame)


def error_keys(context: Context, protocol: Protocol, test: pd.DataFrame) -> set[str]:
    """The recall denominator: every test error site, or RK2's reachable ones on its blocks."""
    if protocol.kind == K_STRICT:
        return rk2.reachable_error_keys(
            set(test["site_key"].astype(str)), context.errors, context.mapping
        )
    return set(context.errors[context.errors["block"] == "test"]["error_key"].astype(str))


def _constant(values: np.ndarray) -> bool:
    return bool(values.size == 0 or np.unique(values).size <= 1)


def policy(
    context: Context,
    protocol: Protocol,
    threshold: pd.DataFrame,
    test: pd.DataFrame,
    epsilon: float,
    keys: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """DS1's policy: a cutoff on threshold decisions, applied unchanged to test decisions."""
    trainable = not _constant(test["safety"].to_numpy(np.float64))
    chosen_on = decisions(threshold)
    tested = decisions(test)
    cutoff, accepted = rk2.gate(chosen_on, tested, epsilon, trainable)
    row = rk2.policy_outcome(tested, accepted, keys, context.mapping, epsilon)
    neutral = ~accepted["exact"] & ~accepted["beneficial"] & ~accepted["is_harmful"]
    row.update(
        {
            "threshold": cutoff,
            "abstains": cutoff is None,
            "accepts_nothing_on_test": bool(len(accepted) == 0),
            "threshold_decisions": len(chosen_on),
            "accepted_exact_share": _share(int(accepted["exact"].sum()), len(accepted)),
            "accepted_neutral_share": _share(int(neutral.sum()), len(accepted)),
            "recall_denominator": len(keys),
        }
    )
    return row, accepted


def cell_pair(
    context: Context, protocol: str, model: str, variant: str = V_FULL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        context.cells[(protocol, model, variant, "threshold")],
        context.cells[(protocol, model, variant, "test")],
    )


def jobs_present(context: Context) -> list[tuple[str, str, str]]:
    return sorted({(p, m, v) for p, m, v, _k in context.cells})


# ------------------------------------------------------------------ ranking metrics


def run_metrics() -> int:
    """Top-1, Recall@K, NDCG@K and AUROC for every fit, on its protocol's test rows."""
    started = time.monotonic()
    _require(RANKING_SCORES, "rank")
    _forbid(RANKING_METRICS)
    context = load_context()
    out: dict[str, Any] = {}
    for protocol, model, variant in jobs_present(context):
        _threshold, test = cell_pair(context, protocol, model, variant)
        out.setdefault(protocol, {}).setdefault(model, {})[variant] = rk1.ranking_metrics(test)
    by_kind = {model: rk1.top1_by_site_kind(cell_pair(context, P_IN, model)[1]) for model in MODELS}
    by_generator = {
        model: {
            GENERATOR_SHORT[g]: {
                key: value
                for key, value in rk1.ranking_metrics(
                    restrict(cell_pair(context, P_IN, model)[1], g)
                ).items()
                if key in ("rows", "choice_sites", "top1", "harm_auroc", "benefit_auroc")
            }
            for g in GENERATORS
        }
        for model in MODELS
    }
    _write_json_once(
        RANKING_METRICS,
        {
            **_analysis_envelope("ranking_metrics"),
            "by_protocol": out,
            "top1_by_choice_site_kind": by_kind,
            "in_distribution_by_generator": by_generator,
            "choice_site_definition": cc_read_json(rk1.DESIGN_RECORD)["metrics"]["choice_site"],
            "undefined_note": (
                "a constant scorer orders candidates by the id tie-break alone, and a set with no "
                "choice site has no Top-1; both are recorded as undefined, never as chance"
            ),
        },
    )
    primary = out[P_IN][M_RK3][V_FULL]
    reference = out[P_IN][M_RK1][V_FULL]
    print(
        f"metrics: in-distribution Top-1 {primary['top1']:.4f} vs RK1 {reference['top1']:.4f}, "
        f"harm AUROC {primary['harm_auroc']:.4f} vs RK1 {reference['harm_auroc']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ deployment and curves


def run_deployment() -> int:
    """DS1's policy for every fit at every harm target, and the analysis-only oracle cut."""
    started = time.monotonic()
    _require(RANKING_METRICS, "metrics")
    _forbid(DEPLOYMENT)
    context = load_context()
    out: dict[str, Any] = {}
    for protocol_name, model, variant in jobs_present(context):
        protocol = context.protocols[protocol_name]
        threshold, test = cell_pair(context, protocol_name, model, variant)
        keys = error_keys(context, protocol, test)
        rows: dict[str, Any] = {}
        for epsilon in EPSILONS:
            rows[rl1.epsilon_key(epsilon)], _accepted = policy(
                context, protocol, threshold, test, epsilon, keys
            )
        tested = decisions(test)
        oracle = rk2.oracle_accepted(tested, PRIMARY_EPSILON)
        rows["oracle_at_primary_target"] = rk2.policy_outcome(
            tested, oracle, keys, context.mapping, PRIMARY_EPSILON
        ) | {"analysis_only": True}
        out.setdefault(protocol_name, {}).setdefault(model, {})[variant] = rows
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    _write_json_once(
        DEPLOYMENT,
        {
            **_analysis_envelope("deployment_comparison"),
            "by_protocol": out,
            "primary_epsilon": PRIMARY_EPSILON,
            "threshold_rule": cc_read_json(rk1.DEPLOYMENT)["threshold_rule"],
            "oracle_note": (
                "the oracle cut is chosen on the test labels themselves; it is analysis only, "
                "never a deployable policy, and no criterion reads it"
            ),
            "rk1_published_recall": cc_read_json(rk1.DEPLOYMENT)["primary_ranker_recall"],
        },
    )
    primary = out[P_IN][M_RK3][V_FULL][key]
    reference = out[P_IN][M_RK1][V_FULL][key]
    print(
        f"deployment: at harm <= {PRIMARY_EPSILON}, RK3 recall {primary['repair_recall']:.4f} "
        f"(harm {primary['selective_harm_rate']}) vs RK1 {reference['repair_recall']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def sweep(tested: pd.DataFrame, keys: set[str], mapping: dict[str, set[str]]) -> pd.DataFrame:
    """Every distinct cutoff over the test decisions: coverage, selective harm and recall."""
    ordered = tested.sort_values(["safety", "candidate_id"], ascending=[False, True], kind="stable")
    safety = ordered["safety"].to_numpy(np.float64)
    harmful = ordered["is_harmful"].to_numpy(bool)
    repaired: set[str] = set()
    counts = np.zeros(len(ordered), dtype=np.int64)
    for i, (site, exact) in enumerate(
        zip(ordered["site_key"].astype(str), ordered["exact"].to_numpy(bool), strict=True)
    ):
        if exact:
            repaired |= mapping.get(site, set()) & keys
        counts[i] = len(repaired)
    accepted = np.arange(1, len(ordered) + 1)
    ends = np.flatnonzero(np.append(safety[1:] != safety[:-1], True))
    return pd.DataFrame(
        {
            "accepted": accepted[ends],
            "coverage": accepted[ends] / max(len(ordered), 1),
            "selective_harm": np.cumsum(harmful)[ends] / accepted[ends],
            "recall": counts[ends] / max(len(keys), 1),
        }
    )


def curve_summary(points: pd.DataFrame) -> dict[str, Any]:
    """The coverage curve on a fixed grid and the harm-recall frontier on a fixed harm grid."""
    coverage = []
    for level in COVERAGE_GRID:
        within = points[points["coverage"] <= level + 1e-12]
        row = within.iloc[-1] if len(within) else None
        coverage.append(
            {
                "grid": level,
                "coverage": float(row["coverage"]) if row is not None else 0.0,
                "recall": float(row["recall"]) if row is not None else 0.0,
                "selective_harm": float(row["selective_harm"]) if row is not None else None,
            }
        )
    frontier = []
    for level in HARM_GRID:
        within = points[points["selective_harm"] <= level + HARM_TOLERANCE + 1e-12]
        frontier.append(
            {
                "harm": level,
                "recall": float(within["recall"].max()) if len(within) else 0.0,
                "coverage": float(
                    within.loc[within["recall"].idxmax(), "coverage"] if len(within) else 0.0
                ),
            }
        )
    return {"coverage_curve": coverage, "frontier": frontier, "cutoffs": len(points)}


def run_curves() -> int:
    """ANALYSIS ONLY: ranking curves and the harm-recall frontier, read off the test labels."""
    started = time.monotonic()
    _require(DEPLOYMENT, "deployment")
    _forbid(CURVES)
    context = load_context()
    deployment = cc_read_json(DEPLOYMENT)["by_protocol"]
    out: dict[str, Any] = {}
    targets = [(P_IN, m) for m in CURVE_MODELS] + [
        (TDOC[d], m) for d in DIRECTIONS for m in MATRIX_MODELS
    ]
    for protocol_name, model in targets:
        protocol = context.protocols[protocol_name]
        _threshold, test = cell_pair(context, protocol_name, model)
        keys = error_keys(context, protocol, test)
        summary = curve_summary(sweep(decisions(test), keys, context.mapping))
        summary["operating_points"] = {
            k: {
                "selective_harm": row["selective_harm_rate"],
                "recall": row["repair_recall"],
                "coverage": row["coverage"],
                "holds": row["holds_its_target"],
            }
            for k, row in deployment[protocol_name][model][V_FULL].items()
            if k != "oracle_at_primary_target"
        }
        out.setdefault(protocol_name, {})[model] = summary
    primary = out[P_IN][M_RK3]["frontier"]
    at_target = next(r for r in primary if abs(r["harm"] - PRIMARY_EPSILON) < 1e-9)
    _write_json_once(
        CURVES,
        {
            **_analysis_envelope("ranking_curves"),
            "by_protocol": out,
            "coverage_grid": list(COVERAGE_GRID),
            "harm_grid": list(HARM_GRID),
            "analysis_only": (
                "every curve sweeps the test decisions over every distinct cutoff and reads the "
                "test labels; none chooses a cutoff and no criterion reads one"
            ),
        },
    )
    print(
        f"curves: {len(targets)} curves; RK3's frontier reaches recall "
        f"{at_target['recall']:.4f} at harm {PRIMARY_EPSILON} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ transfer


def resample_draws(
    frames_: Sequence[pd.DataFrame],
    statistic: Callable[[list[np.ndarray]], float],
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Documents resampled within environment, one draw shared by every frame and arm."""
    generator = np.random.default_rng(seed)
    environments = sorted(set().union(*(set(f["environment"].astype(str)) for f in frames_)))
    plan = []
    for environment in environments:
        documents = sorted(
            set().union(
                *(
                    set(
                        f.loc[f["environment"].astype(str) == environment, "document_id"].astype(
                            str
                        )
                    )
                    for f in frames_
                )
            )
        )
        index = [
            {
                document: np.flatnonzero(
                    (f["environment"].astype(str).to_numpy() == environment)
                    & (f["document_id"].astype(str).to_numpy() == document)
                )
                for document in documents
            }
            for f in frames_
        ]
        picks = generator.integers(0, len(documents), size=(BOOTSTRAP_RESAMPLES, len(documents)))
        plan.append((documents, index, picks))
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for resample in range(BOOTSTRAP_RESAMPLES):
        chosen: list[list[np.ndarray]] = [[] for _ in frames_]
        for documents, index, picks in plan:
            for position in picks[resample]:
                for f in range(len(frames_)):
                    chosen[f].append(index[f][documents[position]])
        rows = [np.concatenate(c) if c else np.empty(0, dtype=np.int64) for c in chosen]
        draws[resample] = statistic(rows)
    return draws


def harm_auroc(
    frame: pd.DataFrame, column: str = "safety", rows: np.ndarray | None = None
) -> float:
    part = frame if rows is None else frame.iloc[rows]
    return auroc(-part[column].to_numpy(np.float64), part["is_harmful"].to_numpy(bool))


def paired_safety(context: Context, protocol: str, models: Sequence[str]) -> pd.DataFrame:
    """One protocol's test rows with every named model's safety score side by side."""
    base: pd.DataFrame | None = None
    for model in models:
        test = cell_pair(context, protocol, model)[1]
        column = test[["candidate_id", "safety"]].rename(columns={"safety": model})
        base = (
            test[["candidate_id", "environment", "document_id", "is_harmful"]].merge(
                column, on="candidate_id", validate="one_to_one"
            )
            if base is None
            else base.merge(column, on="candidate_id", validate="one_to_one")
        )
    if base is None:
        raise PhaseError("no model named")
    return base.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def _matrix_cell(
    context: Context,
    protocol_name: str,
    model: str,
    generator: str | None,
    deployment: dict[str, Any],
) -> dict[str, Any]:
    protocol = context.protocols[protocol_name]
    threshold, test = cell_pair(context, protocol_name, model)
    evaluated = restrict(test, generator)
    metrics = rk1.ranking_metrics(evaluated)
    if generator is None or protocol.kind != K_IN:
        row = deployment[protocol_name][model][V_FULL][rl1.epsilon_key(PRIMARY_EPSILON)]
    else:
        row, _accepted = policy(
            context,
            protocol,
            threshold,
            evaluated,
            PRIMARY_EPSILON,
            error_keys(context, protocol, evaluated),
        )
    return {
        "protocol": protocol_name,
        "rows": metrics["rows"],
        "harm_auroc": metrics["harm_auroc"],
        "benefit_auroc": metrics["benefit_auroc"],
        "top1": metrics.get("top1"),
        "choice_sites": metrics.get("choice_sites"),
        "repair_recall": row["repair_recall"],
        "selective_harm_rate": row["selective_harm_rate"],
        "accepted": row["accepted"],
        "holds_its_target": row["holds_its_target"],
        "working_safe_gate": row["working_safe_gate"],
    }


def rk2_label_equivalent(direction: str, value: float | None) -> dict[str, Any]:
    """Where a zero-shot harm AUROC sits on RK2's target-only label curve (medians over draws)."""
    curve = cc_read_json(rk2.RANKING_CURVES)["by_direction"][direction]
    zero = curve[rk2.A0]["0"]["harm_auroc"]["median"]
    # RK2's own reporting rule: a median resting on fewer than half the draws is not a median.
    medians = {
        int(b): (
            summary["median"]
            if summary["defined"] >= rk2.DRAW_MAJORITY * summary["draws"]
            else None
        )
        for b in rk2.BUDGETS[1:]
        if str(b) in curve[rk2.A1]
        for summary in (curve[rk2.A1][str(b)]["harm_auroc"],)
    }
    reached = [
        b for b, m in sorted(medians.items()) if m is not None and value is not None and m >= value
    ]
    return {
        "rk2_zero_shot_harm_auroc": zero,
        "rk2_target_only_median_harm_auroc": {str(b): m for b, m in medians.items()},
        "value": value,
        "above_rk2_zero_shot": bool(value is not None and value > zero),
        "smallest_rk2_budget_whose_median_reaches_it": reached[0] if reached else None,
        "exceeds_every_rk2_median": bool(value is not None and not reached),
        "median_rule": (
            "RK2's own: a budget whose median rests on fewer than half its draws has no median. "
            "This rule was applied after a first run had read the 25- and 50-label medians, which "
            "rest on one or two of twenty draws"
        ),
    }


def run_transfer() -> int:
    """RQ3: both transfer protocols, the generator transfer matrix and RK2's label curve."""
    started = time.monotonic()
    _require(CURVES, "curves")
    _forbid(TRANSFER)
    context = load_context()
    metrics = cc_read_json(RANKING_METRICS)["by_protocol"]
    deployment = cc_read_json(DEPLOYMENT)["by_protocol"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)

    def row_of(protocol: str, model: str, variant: str = V_FULL) -> dict[str, Any]:
        m = metrics[protocol][model][variant]
        d = deployment[protocol][model][variant]
        return {
            "rows": m["rows"],
            "harm_auroc": m["harm_auroc"],
            "benefit_auroc": m["benefit_auroc"],
            "top1": m["top1"],
            "choice_sites": m["choice_sites"],
            "repair_recall": d[key]["repair_recall"],
            "selective_harm_rate": d[key]["selective_harm_rate"],
            "accepted": d[key]["accepted"],
            "holds_its_target": d[key]["holds_its_target"],
            "working_safe_gate": d[key]["working_safe_gate"],
            "abstains": d[key]["abstains"],
            "oracle_recall": d["oracle_at_primary_target"]["repair_recall"],
        }

    tdoc = {
        direction: {model: row_of(TDOC[direction], model) for model in (M_RANDOM, *TRANSFER_MODELS)}
        for direction in DIRECTIONS
    }
    mean_harm: dict[str, float | None] = {}
    for model in (M_RANDOM, *TRANSFER_MODELS):
        values = [tdoc[d][model]["harm_auroc"] for d in DIRECTIONS]
        mean_harm[model] = None if any(v is None for v in values) else float(np.mean(values))
    folds = [paired_safety(context, TDOC[d], MATRIX_MODELS) for d in DIRECTIONS]
    intervals: dict[str, Any] = {}
    for model in MATRIX_MODELS:
        draws = resample_draws(
            folds,
            lambda rows, m=model: float(
                np.mean([harm_auroc(f, m, r) for f, r in zip(folds, rows, strict=True)])
            ),
        )
        low, high = _interval(draws)
        intervals[model] = {"mean_harm_auroc": mean_harm[model], "ci_low": low, "ci_high": high}
    primary = mean_harm[M_RK3]
    matrix: dict[str, Any] = {}
    for model in MATRIX_MODELS:
        cells = {
            f"train_{GENERATOR_SHORT[C0]}|eval_{GENERATOR_SHORT[C0]}": (WITHIN[C0], None),
            f"train_{GENERATOR_SHORT[C0]}|eval_{GENERATOR_SHORT[C1]}": (TDOC[D_G2Q], None),
            f"train_{GENERATOR_SHORT[C1]}|eval_{GENERATOR_SHORT[C0]}": (TDOC[D_Q2G], None),
            f"train_{GENERATOR_SHORT[C1]}|eval_{GENERATOR_SHORT[C1]}": (WITHIN[C1], None),
            f"train_both|eval_{GENERATOR_SHORT[C0]}": (P_IN, C0),
            f"train_both|eval_{GENERATOR_SHORT[C1]}": (P_IN, C1),
        }
        matrix[model] = {
            name: _matrix_cell(context, protocol, model, generator, deployment)
            for name, (protocol, generator) in cells.items()
        }
    strict = {
        direction: {model: row_of(STRICT[direction], model) for model in (M_RANDOM, *STRICT_MODELS)}
        for direction in DIRECTIONS
    }
    zero_shot = {
        direction: {
            "rk1_here": strict[direction][M_RK1]["harm_auroc"],
            "rk2_published": cc_read_json(rk2.RANKING_CURVES)["by_direction"][direction][rk2.A0][
                "0"
            ]["harm_auroc"]["median"],
        }
        for direction in DIRECTIONS
    }
    for row in zero_shot.values():
        row["identical"] = bool(
            row["rk1_here"] is not None
            and row["rk2_published"] is not None
            and abs(row["rk1_here"] - row["rk2_published"]) < 1e-12
        )
    placement = {
        direction: {
            model: rk2_label_equivalent(direction, strict[direction][model]["harm_auroc"])
            for model in (M_DS1, M_RK3)
        }
        for direction in DIRECTIONS
    }
    _write_json_once(
        TRANSFER,
        {
            **_analysis_envelope("generator_transfer"),
            "primary_protocol": cc_read_json(DESIGN_RECORD)["protocols"][K_TDOC],
            "tdoc": tdoc,
            "tdoc_mean_harm_auroc": mean_harm,
            "tdoc_intervals": intervals,
            "primary_model": {
                "model": M_RK3,
                "mean_harm_auroc": primary,
                "floor": TRANSFER_HARM_FLOOR,
                "meets_the_floor": bool(primary is not None and primary > TRANSFER_HARM_FLOOR),
                "interval_clears_the_floor": bool(
                    intervals[M_RK3]["ci_low"] is not None
                    and intervals[M_RK3]["ci_low"] > TRANSFER_HARM_FLOOR
                ),
            },
            "matrix": matrix,
            "matrix_note": (
                "rows name the generators whose labels trained the ranker, columns the generator "
                "whose test candidates are evaluated. Diagonal cells hold out documents only; "
                "off-diagonal cells hold out the generator and the documents; the both-generators "
                "row is the in-distribution fit restricted to one generator's candidates"
            ),
            "strict": strict,
            "strict_zero_shot_reproduces_rk2": zero_shot,
            "rk2_label_curve_placement": placement,
            "bootstrap": {
                "unit": "document, resampled within environment, one draw for both directions",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
        },
    )
    print(
        f"transfer: RK3 mean held-out-generator harm AUROC {primary:.4f} "
        f"[{intervals[M_RK3]['ci_low']:.4f}, {intervals[M_RK3]['ci_high']:.4f}] vs RK1 "
        f"{mean_harm[M_RK1]:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ ablations and leakage


def run_ablate() -> int:
    """Which families carry RK3's gains, and how much the evidence identifies the generator."""
    started = time.monotonic()
    _require(TRANSFER, "transfer")
    for path in (ABLATION, LEAKAGE):
        _forbid(path)
    metrics = cc_read_json(RANKING_METRICS)["by_protocol"]
    deployment = cc_read_json(DEPLOYMENT)["by_protocol"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)

    def summary(protocol: str, variant: str) -> dict[str, Any]:
        m = metrics[protocol][M_RK3][variant]
        d = deployment[protocol][M_RK3][variant][key]
        return {
            "top1": m["top1"],
            "harm_auroc": m["harm_auroc"],
            "benefit_auroc": m["benefit_auroc"],
            "repair_recall": d["repair_recall"],
            "selective_harm_rate": d["selective_harm_rate"],
            "holds_its_target": d["holds_its_target"],
        }

    full = summary(P_IN, V_FULL)
    in_distribution = {}
    for name in ABLATIONS:
        row = summary(P_IN, name)
        row["delta"] = {
            k: rk1._delta(row[k], full[k])
            for k in ("top1", "harm_auroc", "benefit_auroc", "repair_recall")
        }
        in_distribution[name] = row
    transfer = {}
    full_transfer = {d: metrics[TDOC[d]][M_RK3][V_FULL]["harm_auroc"] for d in DIRECTIONS}
    full_mean = float(np.mean(list(full_transfer.values())))
    for name in TRANSFER_ABLATIONS:
        values = {d: metrics[TDOC[d]][M_RK3][name]["harm_auroc"] for d in DIRECTIONS}
        mean = float(np.mean(list(values.values())))
        transfer[name] = {"by_direction": values, "mean": mean, "delta_mean": mean - full_mean}
    _write_json_once(
        ABLATION,
        {
            **_analysis_envelope("feature_ablation"),
            "full": full,
            "in_distribution": in_distribution,
            "transfer_full": {"by_direction": full_transfer, "mean": full_mean},
            "transfer": transfer,
            "families": {name: list(drop) for name, drop in ABLATIONS.items()},
            "descriptive": (
                "no ablation is in the statistical family; on the test block one choice site "
                "moves Top-1 by more than two points"
            ),
        },
    )
    population = load_population()
    matrix = pd.read_parquet(FEATURE_MATRIX)
    classifiers = {
        R3: rk1.source_classifier(population, matrix, columns_for(R3)),
        R5: rk1.source_classifier(population, matrix, columns_for(R5)),
        **{
            f"r3_plus_{family.rstrip('_')}": rk1.source_classifier(
                population,
                matrix,
                [*r3_columns(), *(c for c in NEW_COLUMNS if c.startswith(family))],
            )
            for family in NEW_FAMILIES
        },
        **{
            f"only_{family.rstrip('_')}": rk1.source_classifier(
                population, matrix, [c for c in NEW_COLUMNS if c.startswith(family)]
            )
            for family in NEW_FAMILIES
        },
    }
    _write_json_once(
        LEAKAGE,
        {
            **_analysis_envelope("generator_leakage"),
            "classifiers": classifiers,
            "rk1_published_r3_source_auroc": cc_read_json(rk1.LEAKAGE)["rl1_r3_source_auroc"],
            "r3_reproduces_rk1": bool(
                abs(
                    classifiers[R3]["source_auroc"]
                    - cc_read_json(rk1.LEAKAGE)["rl1_r3_source_auroc"]
                )
                < 1e-12
            ),
            "reading": (
                "a logistic classifier trained to name the generator, out of fold over RL1's "
                "document folds. No generator column enters any RK3 model; this measures how "
                "much the evidence identifies the generator anyway"
            ),
        },
    )
    print(
        f"ablate: {len(in_distribution)} in-distribution and {len(transfer)} transfer ablations; "
        f"source AUROC R3 {classifiers[R3]['source_auroc']:.4f}, R5 "
        f"{classifiers[R5]['source_auroc']:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ controls


def shuffled_grades(grades: np.ndarray, groups: np.ndarray, draw: int) -> np.ndarray:
    """Grades permuted inside each document query, seeded per draw and document."""
    out = grades.copy()
    for document in sorted(set(groups.tolist())):
        index = np.flatnonzero(groups == document)
        generator = np.random.default_rng(
            int(
                canonical_hash({"seed": PERMUTATION_SEED, "draw": draw, "document": document})[:8],
                16,
            )
        )
        out[index] = grades[index][generator.permutation(index.size)]
    return out


def run_controls() -> int:
    """Grade permutation for the primary model, and the seeded random ranking."""
    started = time.monotonic()
    _require(LEAKAGE, "ablate")
    _forbid(CONTROL_RESULTS)
    population = load_population()
    matrix = pd.read_parquet(FEATURE_MATRIX)
    protocol = build_protocols(frames())[P_IN]
    design = rl1._matrix_for(population, matrix, columns_for(R5))
    grades = population["grade"].to_numpy(dtype=np.int64)
    groups = population["document_id"].astype(str).to_numpy()
    train, test = protocol.train, protocol.test
    harmful = population["is_harmful"].to_numpy(bool)[test]
    beneficial = population["beneficial"].to_numpy(bool)[test]
    draws = []
    for draw in range(PERMUTATION_DRAWS):
        permuted = shuffled_grades(grades[train], groups[train], draw)
        model = fit_boosted(
            design[train], utility_of(permuted, RISK_UTILITY), groups[train], OBJ_RISK
        )
        score = model.score(design[test])
        draws.append(
            {
                "draw": draw,
                "harm_auroc": auroc(-score, harmful),
                "benefit_auroc": auroc(score, beneficial),
                "grades_changed": int((permuted != grades[train]).sum()),
            }
        )
    mean_harm = float(np.mean([d["harm_auroc"] for d in draws]))
    metrics = cc_read_json(RANKING_METRICS)["by_protocol"]
    random = {
        protocol_name: metrics[protocol_name][M_RANDOM][V_FULL]["harm_auroc"]
        for protocol_name in metrics
        if M_RANDOM in metrics[protocol_name]
    }
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "permutation": {
                "draws": draws,
                "mean_harm_auroc": mean_harm,
                "mean_benefit_auroc": float(np.mean([d["benefit_auroc"] for d in draws])),
                "band": list(PERMUTATION_BAND),
                "inside_the_band": bool(PERMUTATION_BAND[0] <= mean_harm <= PERMUTATION_BAND[1]),
                "honest_harm_auroc": metrics[P_IN][M_RK3][V_FULL]["harm_auroc"],
                "rule": cc_read_json(DESIGN_RECORD)["controls"]["permutation"],
            },
            "random_harm_auroc_by_protocol": random,
        },
    )
    print(
        f"controls: permuted-grade harm AUROC {mean_harm:.4f} over {PERMUTATION_DRAWS} draws "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ statistics


def _comparison(
    name: str,
    left: str,
    right: str,
    effect: float | None,
    draws: np.ndarray,
    units: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finite = draws[np.isfinite(draws)]
    low, high = _interval(draws)
    defined = bool(effect is not None and finite.size > 0 and np.isfinite(effect))
    return {
        "comparison": name,
        "left": left,
        "right": right,
        "units": units,
        "resamples": BOOTSTRAP_RESAMPLES,
        "defined": defined,
        "effect": float(effect) if defined else None,
        "ci_low": low if defined else None,
        "ci_high": high if defined else None,
        "p_value": float(_bootstrap_p_value(finite)) if defined else 1.0,
        **(extra or {}),
    }


def _test_errors(context: Context) -> pd.DataFrame:
    errors = context.errors[context.errors["block"] == "test"]
    return errors.sort_values("error_key", kind="stable").reset_index(drop=True)


def repaired_by(
    context: Context, protocol_name: str, model: str
) -> tuple[set[str], dict[str, Any]]:
    """The error sites one fit's primary-target policy repairs, and that policy's outcome."""
    protocol = context.protocols[protocol_name]
    threshold, test = cell_pair(context, protocol_name, model)
    row, accepted = policy(
        context, protocol, threshold, test, PRIMARY_EPSILON, error_keys(context, protocol, test)
    )
    return ds1._repaired_flags(
        decisions(test), accepted, _test_errors(context), context.mapping
    ), row


def recall_comparison(context: Context, name: str, left: str, right: str) -> dict[str, Any]:
    left_set, left_row = repaired_by(context, P_IN, left)
    right_set, right_row = repaired_by(context, P_IN, right)
    frame = ds1._recall_frame(_test_errors(context), left_set, right_set, "error_site")
    a, b = frame["a"].to_numpy(np.float64), frame["b"].to_numpy(np.float64)
    draws = resample_draws([frame], lambda rows: float(a[rows[0]].mean() - b[rows[0]].mean()))
    return _comparison(
        name,
        left,
        right,
        float(a.mean() - b.mean()),
        draws,
        len(frame),
        {
            "left_selective_harm": left_row["selective_harm_rate"],
            "right_selective_harm": right_row["selective_harm_rate"],
            "left_holds": left_row["holds_its_target"],
            "right_holds": right_row["holds_its_target"],
        },
    )


def top1_frame(context: Context, left: str, right: str) -> pd.DataFrame:
    """The in-distribution test choice sites, with whether each model puts an exact repair first."""
    tests = {m: cell_pair(context, P_IN, m)[1] for m in (left, right)}
    rows = []
    for site, group in tests[left].groupby("site_group", sort=True):
        exact = group["exact"].to_numpy(bool)
        if len(group) >= 2 and exact.any() and not exact.all():
            other = tests[right][tests[right]["site_group"] == site]
            rows.append(
                {
                    "environment": str(group["environment"].iloc[0]),
                    "document_id": str(group["document_id"].iloc[0]),
                    "a": float(rk1._ordered(group)["exact"].iloc[0]),
                    "b": float(rk1._ordered(other)["exact"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def safety_bound(context: Context) -> dict[str, Any]:
    """S2: a one-sided document-clustered bound on RK3's in-distribution test harm."""
    _repaired, row = repaired_by(context, P_IN, M_RK3)
    threshold, test = cell_pair(context, P_IN, M_RK3)
    _row, accepted = policy(
        context,
        context.protocols[P_IN],
        threshold,
        test,
        PRIMARY_EPSILON,
        error_keys(context, context.protocols[P_IN], test),
    )
    if accepted.empty:
        return {"accepted": 0, "selective_harm": None, "upper": None, "within_target": False}
    harmful = accepted["is_harmful"].to_numpy(np.float64)
    draws = resample_draws(
        [accepted.reset_index(drop=True)],
        lambda rows: float(harmful[rows[0]].mean()) if rows[0].size else float("nan"),
    )
    finite = draws[np.isfinite(draws)]
    upper = float(np.percentile(finite, 100.0 * SAFETY_BOUND_LEVEL)) if finite.size else None
    return {
        "accepted": len(accepted),
        "selective_harm": row["selective_harm_rate"],
        "upper": upper,
        "level": SAFETY_BOUND_LEVEL,
        "within_target": bool(upper is not None and upper <= PRIMARY_EPSILON + HARM_TOLERANCE),
    }


def run_stats() -> int:
    """The frozen family of six, document-clustered, paired, Holm-adjusted; and the S2 bound."""
    started = time.monotonic()
    _require(CONTROL_RESULTS, "controls")
    _forbid(STATISTICAL_TESTS)
    context = load_context()
    family: dict[str, dict[str, Any]] = {
        PRIMARY_FAMILY[0]: recall_comparison(context, PRIMARY_FAMILY[0], M_RK3, M_RK3_R3),
        PRIMARY_FAMILY[1]: recall_comparison(context, PRIMARY_FAMILY[1], M_RK3, M_DS1_R5),
        PRIMARY_FAMILY[2]: recall_comparison(context, PRIMARY_FAMILY[2], M_RK3, M_RK1),
    }
    choice = top1_frame(context, M_RK3, M_RK1)
    a, b = choice["a"].to_numpy(np.float64), choice["b"].to_numpy(np.float64)
    family[PRIMARY_FAMILY[3]] = _comparison(
        PRIMARY_FAMILY[3],
        M_RK3,
        M_RK1,
        float(a.mean() - b.mean()) if len(choice) else None,
        resample_draws([choice], lambda rows: float(a[rows[0]].mean() - b[rows[0]].mean())),
        len(choice),
    )
    paired = paired_safety(context, P_IN, (M_RK3, M_DS1))
    family[PRIMARY_FAMILY[4]] = _comparison(
        PRIMARY_FAMILY[4],
        M_RK3,
        M_DS1,
        harm_auroc(paired, M_RK3) - harm_auroc(paired, M_DS1),
        resample_draws(
            [paired],
            lambda rows: harm_auroc(paired, M_RK3, rows[0]) - harm_auroc(paired, M_DS1, rows[0]),
        ),
        len(paired),
    )
    folds = [paired_safety(context, TDOC[d], (M_RK3, M_RK1)) for d in DIRECTIONS]

    def transfer_difference(rows: list[np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    harm_auroc(f, M_RK3, r) - harm_auroc(f, M_RK1, r)
                    for f, r in zip(folds, rows, strict=True)
                ]
            )
        )

    family[PRIMARY_FAMILY[5]] = _comparison(
        PRIMARY_FAMILY[5],
        M_RK3,
        M_RK1,
        transfer_difference([np.arange(len(f)) for f in folds]),
        resample_draws(folds, transfer_difference),
        int(sum(len(f) for f in folds)),
    )
    adjusted = s15.holm(family)
    bound = safety_bound(context)
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "family": adjusted,
            "family_size": len(adjusted),
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "safety_bound": bound,
            "plan": cc_read_json(DESIGN_RECORD)["statistical_plan"],
            "recall_note": (
                "each recall comparison records both policies' realized harm and whether each "
                "held the target, because recall alone says nothing about safety"
            ),
        },
    )
    print(
        f"stats: {sum(1 for v in adjusted.values() if v['survives_holm'])}/{len(adjusted)} survive "
        f"Holm; RK3 harm bound {bound['upper']} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ falsification

FEATURIZER_FUNCTIONS = (
    "build_new_features",
    "edit_features",
    "uncertainty_features",
    "language_features",
    "agreement_features",
    "page_summary",
    "environment_evidence",
    "observation_view",
    "word_window",
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
    """Every name, attribute and string a function mentions, following calls in this module."""
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


def _gate_first_arguments() -> list[str]:
    """What `policy` hands to RK2's gate as the decisions a cutoff is chosen on, from the AST."""
    found: list[str] = []
    for node in ast.walk(_functions()["policy"]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "gate"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            found.append(node.args[0].id)
    return found


def hand_built_query_set(rows_per_query: int = 20, queries: int = 12) -> tuple[Any, ...]:
    """A synthetic set whose single informative feature orders the four grades perfectly."""
    generator = np.random.default_rng(FIT_SEED)
    grades = np.tile(np.repeat(np.arange(4), rows_per_query // 4), queries).astype(np.int64)
    noise = generator.normal(0.0, 1.0, size=(grades.size, 3))
    design = np.column_stack([grades + generator.uniform(-0.2, 0.2, grades.size), noise])
    groups = np.repeat([f"q{i:02d}" for i in range(queries)], rows_per_query)
    return design, grades, groups


def _mean_by_grade(scores: np.ndarray, grades: np.ndarray) -> list[float]:
    return [float(scores[grades == g].mean()) for g in range(4)]


def _hand_built_site() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "corrector_sources": [C0, C1, f"{C0}|{C1}", C0],
            "site_result": ["Hause", "House", "Haufe", "Hause"],
            "char_start": [3, 3, 3, 2],
            "char_end": [4, 4, 4, 4],
        }
    )


def run_negative() -> int:
    """Every claim the stage makes, with a test that would fail if the claim were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    tests: list[dict[str, Any]] = []

    def record(name: str, claim: str, passed: bool, evidence: Any) -> None:
        tests.append({"test": name, "claim": claim, "passed": bool(passed), "evidence": evidence})

    registry = cc_read_json(TRAINING_REGISTRY)
    splits = cc_read_json(SPLIT_REGISTRY)
    reproduction = cc_read_json(REPRODUCTION)
    features = cc_read_json(FEATURE_REGISTRY)
    metrics = cc_read_json(RANKING_METRICS)["by_protocol"]
    transfer = cc_read_json(TRANSFER)
    controls = cc_read_json(CONTROL_RESULTS)
    decision_inputs = _mentions("criteria_from_artifacts") | _mentions("run_decide")
    population = load_population()
    matrix = pd.read_parquet(FEATURE_MATRIX)

    carrying = [j for j in registry["jobs"] if j["carries_source_columns"]]
    record(
        "t01_no_generator_identity_in_any_model",
        "no fit carries a source column, and no R5 column names a generator",
        not carrying
        and not [c for c in columns_for(R5) if c.startswith(rl1.FAM_SOURCE) or "corrector" in c],
        {"fits_with_source_columns": len(carrying), "r5_columns": len(columns_for(R5))},
    )
    mentioned = set().union(*(_mentions(f) for f in FEATURIZER_FUNCTIONS))
    record(
        "t02_featurizer_reads_no_label",
        "no featurizer function, or anything it calls here, mentions a label field",
        not (mentioned & LABEL_FIELDS),
        {"label_fields_mentioned": sorted(mentioned & LABEL_FIELDS)},
    )
    view = observation_view(population)
    record(
        "t03_observation_view_is_label_free",
        "the only frame the featurizer receives carries no label column",
        not (set(view.columns) & LABEL_FIELDS) and not (set(OBSERVATION_COLUMNS) & LABEL_FIELDS),
        {"columns": list(view.columns)},
    )
    site = _hand_built_site()
    features_here = agreement_features(site)
    swapped = site.assign(
        corrector_sources=[
            "|".join(sorted({C0: C1, C1: C0}[g] for g in s.split("|")))
            for s in site["corrector_sources"]
        ]
    )
    features_swapped = agreement_features(swapped)
    expected_a = [0.0, 2.0, 1.0, 1.0 - 1.0 / 5.0, 1.0, 0.0]
    record(
        "t04_agreement_is_symmetric_and_hand_computed",
        "swapping the two generators' names changes no agreement feature, and a hand-built site "
        "gives the hand-computed values",
        features_here == features_swapped
        and features_here["a"] == expected_a
        and features_here["c"] == [1.0, 2.0, 0.0, 1.0, 1.0, 1.0]
        and features_here["d"][4] == 0.0,
        {"a": features_here["a"], "expected_a": expected_a, "c": features_here["c"]},
    )
    frozen = pd.read_parquet(rk1.FEATURE_MATRIX).set_index("candidate_id")
    ours = matrix.set_index("candidate_id")
    gap = float(
        np.abs(
            ours[r3_columns()].to_numpy(np.float64)
            - frozen.loc[ours.index, r3_columns()].to_numpy(np.float64)
        ).max()
    )
    record(
        "t05_r3_is_rk1s_r3",
        "every R3 value in RK3's matrix equals RK1's, and RK1's equals RL1's",
        gap == 0.0 and features["r3_reproduces_rl1"]["max_absolute_difference"] == 0.0,
        {"max_absolute_difference": gap},
    )
    for name, key in (
        ("t06_ds1_reproduces", "ds1_reliability_score"),
        ("t07_rk1_reproduces", "rk1_primary_ranker"),
        ("t08_rk2_reproduces", "rk2_arms"),
    ):
        record(
            name,
            f"{key} is rebuilt exactly before any RK3 fit",
            reproduction["by_baseline"][key]["identical"],
            {k: v for k, v in reproduction["by_baseline"][key].items() if k != "cells"},
        )
    scores = pd.read_parquet(RANKING_SCORES)
    in_scores = scores[(scores["protocol"] == P_IN) & (scores["variant"] == V_FULL)]
    ds1_published = pd.read_parquet(ds1.DEPLOYMENT_SCORES).set_index("candidate_id")
    ours_ds1 = in_scores[in_scores["model"] == M_DS1].set_index("candidate_id")
    rk1_published = pd.read_parquet(rk1.RANKING_SCORES)
    rk1_cell = rk1_published[
        (rk1_published["split"] == rk1.SPLIT_BLOCKS)
        & (rk1_published["model"] == rk1.PRIMARY_RANKER)
        & (rk1_published["variant"] == rk1.V_FULL)
    ].set_index("candidate_id")["rank_score"]
    ours_rk1 = in_scores[in_scores["model"] == M_RK1].set_index("candidate_id")["rank_score"]
    ds1_gap = _max_difference(ours_ds1["safety"], ds1_published["safety"])[1]
    rk1_gap = _max_difference(ours_rk1, rk1_cell)[1]
    zero_shot = transfer["strict_zero_shot_reproduces_rk2"]
    record(
        "t09_the_fits_themselves_reproduce_the_baselines",
        "the rank phase's own DS1 and RK1 scores equal the published ones, and RK1 on RK2's "
        "blocks is RK2's zero-shot arm",
        ds1_gap == 0.0 and rk1_gap == 0.0 and all(r["identical"] for r in zero_shot.values()),
        {"ds1_gap": ds1_gap, "rk1_gap": rk1_gap, "zero_shot": zero_shot},
    )
    context = load_context()
    threshold, test = cell_pair(context, P_IN, M_RK3)
    keys = error_keys(context, context.protocols[P_IN], test)
    honest = [
        policy(context, context.protocols[P_IN], threshold, test, e, keys)[0]["threshold"]
        for e in EPSILONS
    ]
    inverted = test.assign(is_harmful=~test["is_harmful"], exact=~test["exact"])
    flipped = [
        policy(context, context.protocols[P_IN], threshold, inverted, e, keys)[0]["threshold"]
        for e in EPSILONS
    ]
    record(
        "t10_cutoffs_read_threshold_rows_only",
        "the gate is handed threshold decisions only, and inverting every test label leaves "
        "every cutoff unchanged",
        _gate_first_arguments() == ["chosen_on"] and honest == flipped,
        {"gate_first_arguments": _gate_first_arguments(), "cutoffs": honest},
    )
    tdoc_rows = [splits["protocols"][TDOC[d]] for d in DIRECTIONS]
    record(
        "t11_protocols_are_isolated",
        "no protocol shares a document, site or candidate across its blocks; no held-out "
        "generator row trains; every held-out test row was proposed by the held-out generator",
        splits["every_protocol_isolated"]
        and all(r["held_out_rows_in_training"] == 0 for r in tdoc_rows)
        and all(r["test_rows_not_proposed_by_the_target"] == 0 for r in tdoc_rows),
        {
            d: {k: splits["protocols"][TDOC[d]][k] for k in ("held_out_rows_in_training",)}
            for d in DIRECTIONS
        },
    )
    record(
        "t12_strict_blocks_are_rk2s",
        "the secondary protocol's test blocks are RK2's sealed blocks, digest for digest",
        splits["strict_blocks_equal_rk2"],
        {d: splits["protocols"][STRICT[d]]["test_digest"][:12] for d in DIRECTIONS},
    )
    protocol = build_protocols(frames())[P_IN]
    grades = population["grade"].to_numpy(dtype=np.int64)[protocol.train]
    groups = population["document_id"].astype(str).to_numpy()[protocol.train]
    queries = _queries(groups, grades)
    probe = np.random.default_rng(FIT_SEED).normal(size=grades.size)
    theirs = rk1._lambda_gradients(probe, grades, queries)
    mine = ranking_gradients(probe, utility_of(grades, STANDARD_GAIN), queries, True, "ideal")
    record(
        "t13_standard_gradient_is_rk1s",
        "with RK1's gain and normalization the shared gradient equals RK1's, element for element",
        bool(np.array_equal(theirs[0], mine[0]) and np.array_equal(theirs[1], mine[1])),
        {"queries": len(queries), "rows": int(grades.size)},
    )
    pair = [np.arange(2)]
    standard = ranking_gradients(np.zeros(2), np.asarray([1.0, 0.0]), pair, True, "ideal")[0]
    risky = ranking_gradients(
        np.zeros(2), np.asarray([RISK_UTILITY[1], RISK_UTILITY[0]]), pair, True, "range"
    )[0]
    swap = 1.0 - 1.0 / math.log2(3.0)
    record(
        "t14_risk_gradient_is_hand_computed",
        "a tied neutral-over-harmful pair gets 0.5 * |delta discount| of push under RK1's gain "
        "and exactly 0.5 under the risk-aware utility, whose range normalization is the pair's "
        "own weight",
        bool(
            abs(standard[0] - 0.5 * swap) < 1e-12
            and abs(standard[1] + 0.5 * swap) < 1e-12
            and abs(risky[0] - 0.5) < 1e-12
            and abs(risky[1] + 0.5) < 1e-12
        ),
        {"standard": standard.tolist(), "risk": risky.tolist(), "expected_standard": 0.5 * swap},
    )
    utility = utility_of(grades, RISK_UTILITY)
    discount = [1.0 / np.log2(np.arange(q.size) + 2.0) for q in queries]
    ideal = [
        float((np.sort(utility[q])[::-1] * d).sum()) for q, d in zip(queries, discount, strict=True)
    ]
    worst = [float((np.sort(utility[q]) * d).sum()) for q, d in zip(queries, discount, strict=True)]
    record(
        "t15_range_normalization_is_positive",
        "every training query's utility range is positive, including queries whose ideal DCG "
        "the negative harm gain drives below zero",
        all(i - w > 0 for i, w in zip(ideal, worst, strict=True)),
        {"queries": len(queries), "negative_ideal_dcg": sum(1 for i in ideal if i <= 0)},
    )
    design, hand_grades, hand_groups = hand_built_query_set()
    orders = {}
    for objective in (OBJ_RISK, OBJ_PAIRWISE, OBJ_POINTWISE):
        fitted = fit_boosted(design, utility_of(hand_grades, RISK_UTILITY), hand_groups, objective)
        orders[objective] = _mean_by_grade(fitted.score(design), hand_grades)
    record(
        "t16_every_objective_orders_a_hand_built_set",
        "on a set where one feature orders the grades, every tree objective scores exact above "
        "partial above neutral above harmful",
        all(all(np.diff(means) > 0) for means in orders.values()),
        orders,
    )
    flat = fit_boosted(design, np.ones(hand_grades.size), hand_groups, OBJ_RISK)
    choice_free = metrics[TDOC[D_G2Q]][M_RK3][V_FULL]
    record(
        "t17_untrainable_and_choiceless_are_undefined",
        "a single-utility training set yields no tree, and Top-1 is undefined where the "
        "evaluated generator offers one candidate per site",
        not flat.trees and choice_free["top1"] is None and choice_free["choice_sites"] == 0,
        {"trees": len(flat.trees), "choice_sites": choice_free["choice_sites"]},
    )
    record(
        "t18_permutation_control_in_band",
        "refitted on grades shuffled inside each document, the primary model ranks harm at chance",
        controls["permutation"]["inside_the_band"],
        {"mean_harm_auroc": controls["permutation"]["mean_harm_auroc"]},
    )
    record(
        "t19_utility_is_derived_from_the_target",
        "the harmful gain is minus (1 - epsilon) / epsilon neutral gains, and the design record "
        "froze that value",
        RISK_UTILITY[REL_HARMFUL]
        == -round((1.0 - PRIMARY_EPSILON) / PRIMARY_EPSILON, 12) * STANDARD_GAIN[REL_NEUTRAL]
        and cc_read_json(DESIGN_RECORD)["utility"]["risk_utility"][str(REL_HARMFUL)]
        == RISK_UTILITY[REL_HARMFUL],
        {"risk_utility": {str(k): v for k, v in RISK_UTILITY.items()}},
    )
    sizes_match = all(
        j["train_rows"] == splits["protocols"][j["protocol"]]["train"]["rows"]
        and j["test_rows"] == splits["protocols"][j["protocol"]]["test"]["rows"]
        for j in registry["jobs"]
    )
    record(
        "t20_every_fit_trains_on_its_protocol_rows",
        "every fit's train and test sizes are its protocol's frozen ones",
        sizes_match,
        {"fits": registry["job_count"]},
    )
    resources = cc_read_json(rl1.FITTED_RESOURCES)["by_environment"]
    used = set(population["document_id"].astype(str))
    overlap = {
        e: sorted(set(r["documents"]) & used)
        for e, r in resources.items()
        if set(r["documents"]) & used
    }
    record(
        "t21_resources_come_from_pages_without_candidates",
        "every fitted resource the new features read comes from ADAPTATION pages that carry no "
        "U5 candidate",
        not overlap,
        {"environments_with_overlap": sorted(overlap)},
    )
    generated = set(rl1._hy1_candidates()["candidate_id"].astype(str))
    record(
        "t22_every_candidate_is_one_hy1_generated",
        "no candidate is generated here",
        set(matrix["candidate_id"].astype(str)) <= generated,
        {"candidates": len(matrix)},
    )
    decision_flags = cc_read_json(DESIGN_RECORD)["ready_for_external_confirmation"] is False
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
    }
    certification = sorted(n for n in called if "certif" in n.lower() or "reserve" in n.lower())
    record(
        "t23_no_certification_and_no_reserve",
        "nothing is certified, no reserve is consumed and no certification machinery is invoked",
        decision_flags and not certification,
        {"certification_names": certification},
    )
    values = matrix[columns_for(R5)].to_numpy(np.float64)
    record(
        "t24_feature_matrix_is_complete_and_finite",
        "every U5 candidate has every R5 value, and every value is finite",
        bool(np.isfinite(values).all() and len(matrix) == len(population)),
        {"rows": len(matrix), "columns": values.shape[1]},
    )
    record(
        "t25_oracle_never_reaches_a_decision",
        "no criterion or outcome reads an oracle cut",
        not [n for n in decision_inputs if "oracle" in n.lower()],
        {"decision_inputs_mentioning_oracle": sorted(n for n in decision_inputs if "oracle" in n)},
    )
    spec = rl1.environment_specs()[0]
    environment = rl1._environment(spec)
    evidence = environment_evidence(environment, resources[spec["environment"]])
    block = population[population["environment"] == spec["environment"]]
    document = sorted(set(block["document_id"].astype(str)))[0]
    alone = page_summary(
        environment,
        document,
        evidence,
        int(block.loc[block["document_id"] == document, "site_group"].nunique()),
    )["features"]
    member = block[block["document_id"] == document]["candidate_id"].astype(str).iloc[0]
    page_columns = [name for name, _d in PAGE_NAMES]
    stored = ours.loc[member, page_columns].to_numpy(np.float64)
    record(
        "t26_page_features_are_read_from_the_page_alone",
        "a page's features recomputed from that page alone equal the matrix values",
        bool(np.array_equal(np.asarray(alone, dtype=np.float64), stored)),
        {"environment": spec["environment"], "document": document},
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
        failed = [t["test"] for t in tests if not t["passed"]]
        raise PhaseError(f"falsification failed: {failed}")
    print(f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    deployment = cc_read_json(DEPLOYMENT)["by_protocol"][P_IN]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    rows = {m: deployment[m][V_FULL][key] for m in (M_RK3, M_RK3_R3, M_DS1_R5, M_RK1, M_DS1)}

    def safely_better(left: str, right: str) -> bool:
        return bool(
            rows[left]["repair_recall"] > rows[right]["repair_recall"]
            and rows[left]["holds_its_target"]
            and rows[right]["holds_its_target"]
        )

    transfer = cc_read_json(TRANSFER)
    stats = cc_read_json(STATISTICAL_TESTS)
    family = stats["family"]
    tdoc = transfer["tdoc"]
    criteria = {
        "C1": safely_better(M_RK3, M_RK3_R3),
        "C2": safely_better(M_RK3, M_DS1_R5),
        "C3": bool(transfer["primary_model"]["meets_the_floor"]),
        "S1": bool(rows[M_RK3]["holds_its_target"]),
        "S2": bool(stats["safety_bound"]["within_target"]),
        "S3": all(bool(tdoc[d][M_RK3]["working_safe_gate"]) for d in DIRECTIONS),
    }
    supported = {
        "C1": bool(family[PRIMARY_FAMILY[0]]["survives_holm"]),
        "C2": bool(family[PRIMARY_FAMILY[1]]["survives_holm"]),
        "C3": bool(transfer["primary_model"]["interval_clears_the_floor"]),
    }
    measured = {
        "recall": {m: rows[m]["repair_recall"] for m in rows},
        "selective_harm": {m: rows[m]["selective_harm_rate"] for m in rows},
        "holds": {m: rows[m]["holds_its_target"] for m in rows},
        "transfer_mean_harm_auroc": transfer["tdoc_mean_harm_auroc"],
        "transfer_interval": transfer["tdoc_intervals"][M_RK3],
        "safety_bound": stats["safety_bound"],
        "transfer_working": {d: tdoc[d][M_RK3]["working_safe_gate"] for d in DIRECTIONS},
    }
    return {"criteria": criteria, "supported": supported, "measured": measured}


def outcome_label(outcome: str, criteria: dict[str, bool], supported: dict[str, bool]) -> str:
    parts = []
    for name in ("C1", "C2", "C3"):
        if criteria[name]:
            parts.append(f"{name} met ({'supported' if supported[name] else 'not supported'})")
        else:
            parts.append(f"{name} not met")
    parts.append("S1 holds" if criteria["S1"] else "S1 fails")
    return f"{outcome}: {OUTCOME_TAXONOMY[outcome]} -- " + "; ".join(parts)


def run_decide() -> int:
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    criteria, supported = state["criteria"], state["supported"]
    outcome = assign_outcome(criteria)
    falsification = cc_read_json(FALSIFICATION)
    stats = cc_read_json(STATISTICAL_TESTS)
    claim = bool(criteria["S1"] and criteria["S2"] and criteria["S3"])
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "outcome": outcome,
            "outcome_label": outcome_label(outcome, criteria, supported),
            "criteria": criteria,
            "statistically_supported": supported,
            "measured": state["measured"],
            "deployment_claim_permitted": claim,
            "deployment_claim_rule": cc_read_json(DESIGN_RECORD)["criteria"]["deployment_claim"],
            "statistical_family_surviving": stats["surviving"],
            "falsification_tests_passed": falsification["passed"],
            "falsification_tests_total": falsification["total"],
            "primary_model": PRIMARY_MODEL,
            "primary_epsilon": PRIMARY_EPSILON,
            "recommended_next_stage": NEXT_STAGE[outcome],
            "issued_head": _git("rev-parse", "HEAD"),
            "ready_for_external_confirmation": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {outcome_label(outcome, criteria, supported)}")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = rk1.FIGURE_NOTE
SURFACE = rk1.SURFACE
INK = rk1.INK
INK_SECONDARY = rk1.INK_SECONDARY
GRID = rk1.GRID
MODEL_COLOURS = {
    M_DS1: "#8a8a8a",
    M_RK1: "#2a78d6",
    M_DS1_R5: "#b58b00",
    M_GBP_R5: "#9c6ade",
    M_PAIR_R5: "#1baf7a",
    M_LR_R5: "#5fa8d3",
    M_RK3_R3: "#f2a07b",
    M_RK3: "#eb6834",
}
FIGURES = (
    "fig1_ranking_curves.png",
    "fig2_harm_recall_frontier.png",
    "fig3_generator_transfer_matrix.png",
    "fig4_feature_ablation.png",
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
            "svg.hashsalt": "rk3",
        }
    )
    manifest: dict[str, Any] = {}

    def save(fig: Any, name: str, sources: Sequence[Path], caption: str) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = FIGURE_DIR / name
        fig.savefig(
            path, dpi=200, bbox_inches="tight", pad_inches=0.08, metadata={"Software": None}
        )
        plt.close(fig)
        manifest[name] = {
            "path": f"{FIGURE_DIR.name}/{name}",
            "caption": caption,
            "sources": {s.name: xr1._signature(s) for s in sources},
        }

    curves = cc_read_json(CURVES)["by_protocol"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    shown = (M_DS1, M_RK1, M_DS1_R5, M_RK3_R3, M_RK3)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for model in shown:
        rows = curves[P_IN][model]["coverage_curve"]
        x = [r["coverage"] for r in rows]
        axes[0].plot(
            x,
            [r["recall"] for r in rows],
            color=MODEL_COLOURS[model],
            lw=1.2,
            label=MODEL_SHORT[model],
        )
        harm = [
            (r["coverage"], r["selective_harm"]) for r in rows if r["selective_harm"] is not None
        ]
        axes[1].plot([h[0] for h in harm], [h[1] for h in harm], color=MODEL_COLOURS[model], lw=1.2)
        point = curves[P_IN][model]["operating_points"][key]
        if point["coverage"] is not None and point["recall"] is not None:
            axes[0].plot(point["coverage"], point["recall"], "o", color=MODEL_COLOURS[model], ms=4)
    axes[1].axhline(PRIMARY_EPSILON, color=INK_SECONDARY, lw=0.8, ls="--")
    axes[0].set_xlabel("share of test decision sites accepted")
    axes[0].set_ylabel("exact-repair recall")
    axes[1].set_xlabel("share of test decision sites accepted")
    axes[1].set_ylabel("selective harm")
    axes[0].set_title("(a) repairs as the cutoff loosens", loc="left", fontsize=8)
    axes[1].set_title("(b) harm as the cutoff loosens", loc="left", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(
        fig,
        FIGURES[0],
        (CURVES, DEPLOYMENT),
        "In-distribution ranking curves on the DS1 test block, every distinct cutoff swept; dots "
        "are each model's operating point at harm <= 0.1, chosen on the threshold block",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for model in (M_DS1, M_RK1, M_RK3):
        rows = curves[P_IN][model]["frontier"]
        axes[0].plot(
            [r["harm"] for r in rows],
            [r["recall"] for r in rows],
            color=MODEL_COLOURS[model],
            lw=1.2,
            label=MODEL_SHORT[model],
        )
        for eps_key, point in curves[P_IN][model]["operating_points"].items():
            if point["selective_harm"] is not None:
                axes[0].plot(
                    point["selective_harm"],
                    point["recall"],
                    "o",
                    color=MODEL_COLOURS[model],
                    ms=3.5,
                    mfc="none" if eps_key != key else None,
                )
        for direction, style in zip(DIRECTIONS, ("-", "--"), strict=True):
            rows = curves[TDOC[direction]][model]["frontier"]
            axes[1].plot(
                [r["harm"] for r in rows],
                [r["recall"] for r in rows],
                color=MODEL_COLOURS[model],
                lw=1.1,
                ls=style,
                label=f"{MODEL_SHORT[model]}, {DIRECTION_SHORT[direction]}",
            )
    for ax in axes:
        ax.axvline(PRIMARY_EPSILON, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xlabel("realized selective harm")
    axes[0].set_ylabel("highest exact-repair recall")
    axes[0].set_title("(a) known generators", loc="left", fontsize=8)
    axes[1].set_title("(b) held-out generator and documents", loc="left", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper left")
    save(
        fig,
        FIGURES[1],
        (CURVES,),
        "Harm-recall frontier, ANALYSIS ONLY: the highest recall any cutoff reaches at each "
        "realized harm on the test labels; open dots are the deployed operating points at the "
        "other targets, the filled dot the primary one",
    )

    matrix = cc_read_json(TRANSFER)["matrix"]
    rows_order = ("current", "Qwen", "both")
    cols_order = ("current", "Qwen")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, model in zip(axes, MATRIX_MODELS, strict=True):
        grid = np.full((3, 2), np.nan)
        for i, r in enumerate(rows_order):
            for j, c in enumerate(cols_order):
                value = matrix[model][f"train_{r}|eval_{c}"]["harm_auroc"]
                grid[i, j] = np.nan if value is None else value
        ax.imshow(grid, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        ax.grid(False)
        for i in range(3):
            for j in range(2):
                text = "undefined" if np.isnan(grid[i, j]) else f"{grid[i, j]:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(2), [f"eval {c}" for c in cols_order])
        ax.set_yticks(range(3), [f"train {r}" for r in rows_order])
        ax.set_title(MODEL_SHORT[model], loc="left", fontsize=8)
    save(
        fig,
        FIGURES[2],
        (TRANSFER,),
        "Generator transfer matrix, test harm AUROC. Off-diagonal cells hold out the generator "
        "and the documents; diagonal cells hold out the documents only",
    )

    from matplotlib.ticker import MaxNLocator

    ablation = cc_read_json(ABLATION)
    names = list(ablation["in_distribution"])
    transfer_names = list(ablation["transfer"])
    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(len(names), len(transfer_names) + 1))
    top = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    bottom = fig.add_subplot(grid[1, :])

    def bars(ax: Any, labels: Sequence[str], values: Sequence[float], title: str) -> None:
        y = np.arange(len(labels))
        ax.barh(y, values, color=[MODEL_COLOURS[M_RK3] if v < 0 else "#1baf7a" for v in values])
        ax.axvline(0.0, color=INK_SECONDARY, lw=0.8)
        ax.set_yticks(y, [n.replace("_", " ") for n in labels])
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(title)

    for ax, metric, title in (
        (top[0], "repair_recall", "(a) change in recall at harm <= 0.1"),
        (top[1], "harm_auroc", "(b) change in harm AUROC"),
    ):
        bars(
            ax,
            names,
            [ablation["in_distribution"][n]["delta"][metric] or 0.0 for n in names],
            title,
        )
    top[1].set_yticklabels([])
    bars(
        bottom,
        transfer_names,
        [ablation["transfer"][n]["delta_mean"] for n in transfer_names],
        "(c) change in mean held-out-generator harm AUROC",
    )
    save(
        fig,
        FIGURES[3],
        (ABLATION,),
        "Feature ablations of RK3, descriptive: each bar is the ablated model minus the full one",
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
    "FEATURE_MATRIX",
    "FEATURE_REGISTRY",
    "SPLIT_REGISTRY",
    "REPRODUCTION",
    "RANKING_SCORES",
    "MODEL_REGISTRY",
    "TRAINING_REGISTRY",
    "RANKING_METRICS",
    "DEPLOYMENT",
    "CURVES",
    "TRANSFER",
    "ABLATION",
    "LEAKAGE",
    "CONTROL_RESULTS",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("features", run_features),
        ("splits", run_splits),
        ("reproduce", run_reproduce),
        ("rank", run_rank),
        ("metrics", run_metrics),
        ("deployment", run_deployment),
        ("curves", run_curves),
        ("transfer", run_transfer),
        ("ablate", run_ablate),
        ("controls", run_controls),
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
            out[path.name] = xr1._signature(path)
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
            "seeds": {
                "fit": FIT_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "permutation": PERMUTATION_SEED,
            },
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
    "3. The Enhanced Representation": (FEATURE_REGISTRY, DESIGN_RECORD, LEAKAGE),
    "4. Risk-Aware LambdaRank": (DESIGN_RECORD, MODEL_REGISTRY),
    "5. Protocols and Reproduction": (SPLIT_REGISTRY, REPRODUCTION, DESIGN_RECORD),
    "6. Ranking Results": (RANKING_METRICS, TRAINING_REGISTRY),
    "7. Safe Selection at Harm 0.1": (DEPLOYMENT, STATISTICAL_TESTS),
    "8. Ranking Curves and the Harm-Recall Frontier": (CURVES, DEPLOYMENT),
    "9. Generator Transfer": (TRANSFER, SPLIT_REGISTRY),
    "10. Feature Ablation": (ABLATION, LEAKAGE),
    "11. Statistical Tests": (STATISTICAL_TESTS,),
    "12. Controls and Falsification": (CONTROL_RESULTS, FALSIFICATION),
    "13. Limitations": (SPLIT_REGISTRY, DESIGN_RECORD, STATISTICAL_TESTS, TRANSFER),
    "14. Research Decision": (DECISION, STATISTICAL_TESTS, TRANSFER, FALSIFICATION),
    "15. Next Research Direction": (DECISION, TRANSFER, CURVES, DEPLOYMENT),
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
    audit = gen1.audit_report(REPORT, REPORT_SECTIONS)
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
        "features": run_features,
        "splits": run_splits,
        "reproduce": run_reproduce,
        "rank": run_rank,
        "metrics": run_metrics,
        "deployment": run_deployment,
        "curves": run_curves,
        "transfer": run_transfer,
        "ablate": run_ablate,
        "controls": run_controls,
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

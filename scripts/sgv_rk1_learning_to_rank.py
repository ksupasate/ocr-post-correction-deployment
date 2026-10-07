#!/usr/bin/env python3
"""SGV-RK1: can a generator-agnostic ranking model select the right OCR correction candidate?

SGV-DS1 closed with outcome B and named ranking quality as the next constraint: the hybrid
candidate universe reaches exact-repair recall 0.1449 under a perfect selector, and the deployed
reliability gate reaches 0.0191 at the project's primary harm target. This stage asks whether a
model trained with a RANKING objective, on the same frozen generator-agnostic evidence, closes that
gap -- and it measures, before training anything, where in the pipeline the gap actually sits.

**1. Nothing upstream moves.** Candidates, labels, the R3 feature schema, the document folds and the
DS1 three-block deployment protocol are read from SGV-HY1, SGV-RL1 and SGV-DS1 and hash-verified.

**2. The objective is the variable.** The representation is held fixed at RL1's R3 so that any
difference between the rankers and DS1's pointwise reliability score is attributable to the
training objective, not to new features. Within-site relative features are a declared secondary
arm, because a candidate's site size identifies its generator in this universe.

**3. Two query granularities.** A site-level query asks "which candidate at this site is right?"
and carries the ranking metrics. A document-level query asks "which of this page's candidates are
safest to accept?" and carries the deployment endpoint. Measured before any ranker was trained,
perfect within-site ranking adds nothing at DS1's operating point while a perfect gate recovers
most of the gap, so the second granularity is where a ranking objective could matter.

**4. Generator identity never reaches the primary model.** It is used to build transfer splits and
to report strata, and a generator-aware ranker is fitted only as a leakage reference.

    --reconstruct   section 0: HY1, RL1, RL2 and DS1 re-read from their own artifacts
    --freeze        upstream hashes and the frozen inputs this stage consumes
    --preregister   queries, relevance grades, models, splits, criteria, outcome rules
    --headroom      where the DS1 gap sits: within-site ranking versus the accept gate
    --population    the ranking population, including the third candidate source
    --features      R3 for the third source, reproduced exactly against RL1 for the rest
    --splits        the fit / threshold / test blocks and the transfer folds
    --rank          every pre-registered ranker, scored strictly out of sample
    --metrics       Top-1, Recall@K, NDCG@K and AUROC per ranker and granularity
    --deployment    the DS1 policy re-run with each ranker's ordering
    --transfer      leave-one-generator-out and leave-one-source-out
    --ablate        feature-family ablations, single-family models, the leakage classifier
    --stats         document-clustered paired bootstrap, Holm within the frozen family
    --negative      the falsification suite
    --decide        the frozen outcome rule
    --figures       every figure from persisted artifacts
    --determinism   every derived phase twice from the frozen upstream inputs
    --record        provenance, upstream re-hash, decision re-derivation, traceability

DEVELOPMENT / RANKING ONLY. No new candidate is generated, no new model family beyond the declared
rankers is searched, nothing is certified and nothing is production-ready. The confirmatory
reserve stays LOCKED and `ready_for_external_confirmation` is false by construction.
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

import sgv_ds1_deployment_synthesis as ds1
from ocr_risk.io.hashing import canonical_hash, file_sha256

rl2 = ds1.rl2
rl1 = ds1.rl1
hy1 = rl1.hy1
lp1 = rl1.lp1
gen1 = rl1.gen1
xr1 = rl1.xr1
xc1 = rl1.xc1
s15 = rl1.s15

REPO = rl1.REPO
OUT = REPO / "results/generated/sgv_rk1_learning_to_rank"
CACHE = OUT / "cache"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"
HEADROOM = OUT / "ranking_headroom.json"

RANKING_POPULATION = OUT / "ranking_population.parquet"
POPULATION_INVENTORY = OUT / "ranking_population_inventory.json"
FEATURE_MATRIX = OUT / "feature_matrix.parquet"
FEATURE_REPRODUCTION = OUT / "feature_reproduction.json"
SPLIT_REGISTRY = OUT / "split_registry.json"

RANKING_SCORES = OUT / "ranking_scores.parquet"
MODEL_REGISTRY = OUT / "model_registry.json"
TRAINING_REGISTRY = OUT / "training_registry.json"

RANKING_METRICS = OUT / "ranking_metrics.json"
DISCRIMINATION = OUT / "discrimination.json"
DEPLOYMENT = OUT / "deployment_comparison.json"
TRANSFER = OUT / "generator_transfer.json"
ABLATION = OUT / "feature_ablation.json"
SINGLE_FAMILY = OUT / "single_family_models.json"
LEAKAGE = OUT / "generator_leakage.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
FALSIFICATION = OUT / "falsification_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_rk1/learning_to_rank.md"

SCHEMA_VERSION = 1
STAGE = "sgv_rk1_learning_to_rank"
HYPOTHESIS = "SGV-RK1-D1"
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
_finite = rl1._finite

BOOTSTRAP_RESAMPLES = rl1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = rl1.BOOTSTRAP_SEED
ALPHA = rl1.ALPHA
FIT_SEED = rl1.FIT_SEED

# ------------------------------------------------------------------ the frozen design

REPRESENTATION = rl1.PRIMARY_REPRESENTATION
C0 = rl1.C0
C1 = rl1.C1
C1_TEXT = hy1.C1_TEXT
SOURCES = (C0, C1, C1_TEXT)
SOURCE_LABEL = {
    C0: "frozen current generator",
    C1: "Qwen3-VL image-conditioned corrector",
    C1_TEXT: "Qwen3-VL text-only condition (same checkpoint as the image corrector)",
}
# The third source shares the image corrector's weights. Holding it out tests an unseen CONDITION
# of a seen model, not an unseen generator, and is reported as exactly that.
DISTINCT_GENERATORS = (C0, C1)

# Blocks are DS1's, unchanged, so the deployment endpoint is directly comparable with DS1's.
FIT_FOLDS = ds1.FIT_FOLDS
THRESHOLD_FOLDS = ds1.THRESHOLD_FOLDS
TEST_FOLDS = ds1.TEST_FOLDS
BLOCKS = ds1.BLOCKS
EPSILONS = ds1.EPSILONS
PRIMARY_EPSILON = ds1.PRIMARY_EPSILON
THRESHOLD_GRID = ds1.THRESHOLD_GRID

# Graded relevance. Exact beats a partial improvement beats a neutral change beats harm.
REL_EXACT = 3
REL_PARTIAL = 2
REL_NEUTRAL = 1
REL_HARMFUL = 0

# Query granularities.
Q_SITE = "site"
Q_DOCUMENT = "document"
GRANULARITIES = (Q_SITE, Q_DOCUMENT)

# Rankers. B0-B2 are references; R* are the proposed generator-agnostic rankers.
B0 = "b0_random"
B1 = "b1_ds1_reliability"
B2 = "b2_generator_aware"
RN = "r_ranknet_linear"
LM_SITE = "r_lambdamart_site"
LM_DOC = "r_lambdamart_document"
LM_REL = "r_lambdamart_document_relative"
MODELS = (B0, B1, B2, RN, LM_SITE, LM_DOC, LM_REL)
PRIMARY_RANKER = LM_DOC
MODEL_LABEL = {
    B0: "random ranking",
    B1: "SGV-DS1 reliability score: logistic P(beneficial), gated by 1 - P(harmful)",
    B2: "generator-aware LambdaMART, R4 with source identity -- leakage reference only",
    RN: "linear RankNet, the repository's SGV5 pairwise ranker",
    LM_SITE: "LambdaMART with site-level queries",
    LM_DOC: "LambdaMART with document-level queries",
    LM_REL: "LambdaMART with document-level queries and within-site relative features",
}
GENERATOR_AGNOSTIC = (B1, RN, LM_SITE, LM_DOC, LM_REL)

# LambdaMART, fixed before any fit. Small trees and a modest learning rate: the scientific variable
# is the objective, and nothing here is tuned.
LAMBDAMART_ROUNDS = 150
LAMBDAMART_LEARNING_RATE = 0.1
LAMBDAMART_MAX_DEPTH = 3
LAMBDAMART_MIN_LEAF = 20
LAMBDAMART_SIGMA = 1.0
# RankNet: the repository's SGV5 configuration, reused unchanged.
RANKNET_PAIRS = 60000

RANK_CUTOFFS = (1, 3, 5)
NDCG_CUTOFFS = (1, 3, 5)

# Success criteria, frozen before any RK1 endpoint. A and C are the prompt's; B's floor is the
# prompt's too, measured on the strict two-generator split so it is comparable with SGV-RL1.
TOP1_RELATIVE_GAIN = 0.10
TRANSFER_HARM_FLOOR = 0.75
FAMILY_SIZE = 4

OUTCOME_TAXONOMY = {
    "A": "generator-agnostic ranking improves selection and deployment and transfers",
    "B": "ranking improves selection or deployment, but not transfer to an unseen generator",
    "C": "a ranking objective does not improve on the DS1 reliability score",
}
NEXT_STAGE = {
    "A": "NEXT: TARGETED EXTERNAL CONFIRMATION OF THE RANKED DEPLOYMENT POLICY",
    "B": "NEXT: GENERATOR ADAPTATION OF THE RANKER UNDER A LABEL BUDGET",
    "C": "NEXT: RICHER CANDIDATE VERIFICATION EVIDENCE, NOT A NEW OBJECTIVE",
}

PRIMARY_FAMILY = (
    "P1_top1_ranker_vs_ds1_choice_sites",
    "P2_deployment_recall_ranker_vs_ds1",
    "P3_deployment_joint_harm_ranker_vs_ds1",
    "P4_transfer_harm_auroc_ranker_vs_ds1",
)

UPSTREAM_EXPECTED: dict[str, dict[str, Any]] = {
    "sgv_ds1": {
        "outcome": "B",
        "recommended_next_stage": "NEXT: RANKING-QUALITY WORK BEFORE ANY AUTOMATION CLAIM",
        "production_ready": False,
        "ready_for_external_confirmation": False,
        "falsification_tests_passed": 22,
        "falsification_tests_total": 22,
    },
    "sgv_rl1": {"outcome": "B"},
    "sgv_rl2": {"outcome": "B"},
}


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_rk1-{artifact}-v{SCHEMA_VERSION}",
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


def assign_outcome(criteria: dict[str, bool]) -> str:
    """Exactly one outcome, in the frozen precedence A, B, C.

    A needs all three: better selection, better deployment and transfer to an unseen generator. B
    is the widened case in which the ranking objective does improve selection or deployment on
    known generators but the transfer floor is not met. C is the case in which the ranking
    objective improves neither, and the report must then say that the objective is not the lever.
    """
    if criteria["A"] and criteria["B"] and criteria["C"]:
        return "A"
    if criteria["A"] or criteria["C"]:
        return "B"
    return "C"


def relevance(frame: pd.DataFrame) -> np.ndarray:
    """The frozen graded relevance: exact 3, partial 2, neutral 1, harmful 0."""
    exact = frame["exact"].to_numpy(dtype=bool)
    beneficial = frame["beneficial"].to_numpy(dtype=bool)
    harmful = frame["is_harmful"].to_numpy(dtype=bool)
    grade = np.full(len(frame), REL_NEUTRAL, dtype=np.int64)
    grade[beneficial & ~exact] = REL_PARTIAL
    grade[exact] = REL_EXACT
    grade[harmful] = REL_HARMFUL
    return grade


# ------------------------------------------------------------------ section 0: frozen state


def _upstream_files() -> list[Path]:
    files = list(ds1._upstream_files())
    files.append(REPO / "scripts/sgv_ds1_deployment_synthesis.py")
    for pattern in ("*.json", "*.parquet"):
        files.extend(sorted(ds1.OUT.glob(pattern)))
    files.append(ds1.REPORT)
    return sorted({p for p in files if p.is_file()})


def upstream_checks() -> list[dict[str, Any]]:
    """Every upstream value this stage rests on, re-read from the stage that produced it."""
    _check = xr1._check
    checks = list(ds1.upstream_checks())
    decisions = {
        "sgv_ds1": cc_read_json(ds1.DECISION),
        "sgv_rl1": cc_read_json(rl1.DECISION),
        "sgv_rl2": cc_read_json(rl2.DECISION),
    }
    for stage, expected in UPSTREAM_EXPECTED.items():
        for key, value in expected.items():
            checks.append(_check(f"{stage}.{key}", decisions[stage].get(key), value))
    ds1_decision = decisions["sgv_ds1"]
    checks.append(
        _check(
            "sgv_ds1.gate_is_far_below_the_oracle",
            bool(
                ds1_decision["operating_point"]["repair_recall"]
                < ds1_decision["pipeline_repair_recall"][ds1.P3]
            ),
            True,
        )
    )
    checks.append(
        _check("sgv_ds1.determinism", cc_read_json(ds1.DETERMINISM)["all_runs_identical"], True)
    )
    checks.append(
        _check(
            "sgv_ds1.report_traceability",
            cc_read_json(ds1.TRACEABILITY)["audit"]["untraceable_numeric_claims"],
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
        raise PhaseError("an RK1 endpoint already exists; the freeze must precede every one")
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
            "changes_the_feature_schema": False,
            "changes_an_upstream_outcome": False,
            "ranking_libraries": {
                "lightgbm": "not installed and not added: the stage's test module imports this "
                "script, so a hard dependency would break CI collection",
                "xgboost": "not installed and not added, for the same reason",
                "implementation": (
                    "LambdaMART is implemented directly on sklearn regression trees fitted to "
                    "LambdaRank gradients; RankNet reuses the repository's SGV5 pairwise ranker"
                ),
            },
            "uses_ground_truth": False,
        },
    )
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "representation": REPRESENTATION,
            "representation_columns": len(rl1.columns_for(REPRESENTATION)),
            "rl1_feature_matrix": _relative(rl1.FEATURE_MATRIX),
            "rl1_fitted_resources": _relative(rl1.FITTED_RESOURCES),
            "document_folds": _relative(rl1.GROUP_REGISTRY),
            "ds1_blocks": {
                "fit": list(FIT_FOLDS),
                "threshold": list(THRESHOLD_FOLDS),
                "test": list(TEST_FOLDS),
            },
            "ds1_operating_point": cc_read_json(ds1.DECISION)["operating_point"],
            "ds1_pipeline_repair_recall": cc_read_json(ds1.DECISION)["pipeline_repair_recall"],
            "hyperparameters": dict(cc_read_json(rl1.HYPERPARAMETER_REGISTRY)[rl1.PRIMARY_MODEL]),
            "sources": {name: SOURCE_LABEL[name] for name in SOURCES},
            "uses_ground_truth": False,
        },
    )
    print(f"freeze: {len(hashes)} upstream files hashed ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the pre-registration

FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "text": (rl1.FAM_TEXT,),
    "ocr": (rl1.FAM_CONF, rl1.FAM_GEOM),
    "edit": (rl1.FAM_EDIT,),
    "language": (rl1.FAM_PLAUS, rl1.FAM_CTX),
    "visual": (rl1.FAM_VIS,),
}
ABLATIONS: dict[str, tuple[str, ...]] = {
    "a1_no_visual": FAMILY_PREFIXES["visual"],
    "a2_no_ocr_confidence": FAMILY_PREFIXES["ocr"],
    "a3_no_language": FAMILY_PREFIXES["language"],
}
# The prompt's feature list names a masked language model, a crop embedding and a vision-language
# similarity score. None exists in the frozen R3 schema, and each would need a new large model; the
# frozen schema's lighter implementation of each category is used and the gap is recorded.
SCHEMA_MAPPING: dict[str, dict[str, Any]] = {
    "ocr": {
        "requested": [
            "OCR confidence",
            "token confidence",
            "character uncertainty",
            "engine confidence",
            "token position",
            "bounding box",
        ],
        "frozen_families": ["conf_", "geom_"],
        "not_available": ["character-level uncertainty: the engines report word-level confidence"],
    },
    "edit": {
        "requested": [
            "edit distance",
            "insertions",
            "deletions",
            "substitutions",
            "segmentation change",
            "affected character ratio",
        ],
        "frozen_families": ["edit_", "txt_"],
        "not_available": [],
    },
    "language": {
        "requested": [
            "masked language model probability",
            "token likelihood",
            "perplexity difference",
            "grammatical consistency",
        ],
        "frozen_families": ["plaus_", "ctx_"],
        "not_available": [
            "masked language model probability: the frozen schema uses a character 3-gram model "
            "fitted on adaptation pages; a masked LM would be a new large model",
            "grammatical consistency: not represented",
        ],
    },
    "visual": {
        "requested": ["crop embedding", "OCR-image consistency", "vision-language similarity"],
        "frozen_families": ["vis_"],
        "not_available": [
            "crop embedding and vision-language similarity: the frozen schema uses ink "
            "segmentation and glyph-prototype matching; an embedding would be a new large model",
        ],
    },
    "context": {
        "requested": ["neighbouring words", "document structure", "line position", "page region"],
        "frozen_families": ["ctx_", "txt_", "geom_"],
        "not_available": [],
    },
}


def columns_without(drop: Sequence[str]) -> list[str]:
    return [c for c in rl1.columns_for(REPRESENTATION) if not c.startswith(tuple(drop))]


def columns_only(keep: Sequence[str]) -> list[str]:
    return [c for c in rl1.columns_for(REPRESENTATION) if c.startswith(tuple(keep))]


def run_preregister() -> int:
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("an RK1 endpoint exists; the design is frozen before every one")
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "research_question": (
                "can a generator-agnostic ranking model reliably select beneficial OCR "
                "corrections from candidates produced by different correction systems?"
            ),
            "primary_hypothesis": (
                "a ranking objective on candidate-level evidence -- OCR uncertainty, edit shape, "
                "language plausibility, visual evidence and context -- selects better candidates "
                "than DS1's pointwise reliability score, and generalizes across generators"
            ),
            "the_variable_is_the_objective": (
                "the representation is RL1's R3 unchanged, so any difference between a ranker and "
                "DS1's reliability score is attributable to the training objective"
            ),
            "relevance_grades": {
                "exact": REL_EXACT,
                "partial_improvement": REL_PARTIAL,
                "neutral": REL_NEUTRAL,
                "harmful": REL_HARMFUL,
            },
            "granularities": {
                Q_SITE: "one query per proposal site: which candidate at this site is right?",
                Q_DOCUMENT: (
                    "one query per document: which of this page's candidates are safest to "
                    "accept? This is the ordering the deployment gate uses"
                ),
            },
            "models": MODEL_LABEL,
            "primary_ranker": PRIMARY_RANKER,
            "generator_agnostic_models": list(GENERATOR_AGNOSTIC),
            "lambdamart": {
                "rounds": LAMBDAMART_ROUNDS,
                "learning_rate": LAMBDAMART_LEARNING_RATE,
                "max_depth": LAMBDAMART_MAX_DEPTH,
                "min_samples_leaf": LAMBDAMART_MIN_LEAF,
                "sigma": LAMBDAMART_SIGMA,
                "gain": "2^grade - 1",
                "leaf_values": "Newton step: summed lambdas over summed second-order weights",
                "tuned": False,
            },
            "ranknet": {"pairs": RANKNET_PAIRS, "source": "SGV5's pairwise ranker, unchanged"},
            "relative_features": (
                "for four frozen features, the candidate's value minus its site mean and its "
                "within-site rank fraction. Site size is never a feature, and the leakage these "
                "features introduce is measured, because in this universe a site's size partly "
                "identifies its generator"
            ),
            "blocks": {
                "fit": list(FIT_FOLDS),
                "threshold": list(THRESHOLD_FOLDS),
                "test": list(TEST_FOLDS),
                "rule": "SGV-DS1's three blocks, unchanged, so the deployment endpoint compares",
            },
            "deployment_policy": (
                "DS1's policy with each model's score in place of DS1's: one decision per site, "
                "the top candidate by the model's score, accepted when the score clears a "
                "threshold chosen on the threshold block to meet the harm target"
            ),
            "transfer": {
                "primary": (
                    "SGV-RL1's strict leave-one-generator-out rule over the two distinct "
                    "generators, so the transfer floor is comparable with RL1's own result"
                ),
                "secondary": (
                    "leave-one-source-out over three candidate sources. The text-only Qwen "
                    "condition shares the image corrector's checkpoint, so holding it out tests "
                    "an unseen condition of a seen model, and it is reported as that"
                ),
            },
            "families": {name: list(prefixes) for name, prefixes in FAMILY_PREFIXES.items()},
            "ablations": {name: list(drop) for name, drop in ABLATIONS.items()},
            "schema_mapping": SCHEMA_MAPPING,
            "metrics": {
                "top1": "share of choice sites whose top-ranked candidate is exact",
                "choice_site": (
                    "a site with at least two candidates, at least one exact, and not all exact; "
                    "the only sites where ranking can change which candidate is chosen"
                ),
                "recall_at_k": (
                    "share of exact-bearing multi-candidate sites with an exact in the top k"
                ),
                "ndcg_at_k": "graded NDCG over sites with at least two distinct grades",
                "auroc": "harm-oriented and benefit-oriented AUROC of the model score",
                "cutoffs": list(RANK_CUTOFFS),
            },
            "criteria": {
                "A": (
                    f"the primary ranker's test Top-1 on choice sites is at least "
                    f"{1 + TOP1_RELATIVE_GAIN} times DS1's"
                ),
                "B": (
                    f"the primary ranker's mean strict leave-one-generator-out harm AUROC "
                    f"exceeds {TRANSFER_HARM_FLOOR}"
                ),
                "C": (
                    f"at harm <= {PRIMARY_EPSILON} the primary ranker's test exact-repair recall "
                    "exceeds DS1's, with both policies holding the target or abstaining"
                ),
                "reading": (
                    "A, B and C are the thresholds the brief set and are applied as written. "
                    "Whether each met criterion is also statistically supported is recorded "
                    "beside it and named in the outcome label, so a noise-level difference cannot "
                    "be read as an improvement"
                ),
            },
            "outcome_rule": OUTCOME_TAXONOMY,
            "statistical_family": list(PRIMARY_FAMILY),
            "statistical_plan": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm within the frozen family",
            },
            "disclosure_a_headroom_probe_was_run": (
                "before this record was written, a probe on SGV-DS1's frozen scores measured "
                "that perfect within-site ranking adds nothing at DS1's operating point while a "
                "perfect accept gate recovers most of the gap, and that DS1's within-site Top-1 on "
                "the test block's choice sites is 0.6522. That measurement motivated the "
                "document-level granularity. The criteria are the brief's own and were not chosen "
                "from the probe, but the probe happened"
            ),
            "non_goals": [
                "no candidate is generated and no OCR output is regenerated",
                "the R3 feature schema is not extended in the primary models",
                "no generator identity enters a primary model",
                "no large model is introduced",
                "no certification, production claim or external confirmation",
            ],
            "ready_for_external_confirmation": False,
            "uses_ground_truth": False,
        },
    )
    print(f"preregister: design frozen ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ where the DS1 gap sits


def run_headroom() -> int:
    """Before any ranker: how much of DS1's gap is within-site ranking and how much is the gate."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    _forbid(HEADROOM)
    scores = pd.read_parquet(ds1.DEPLOYMENT_SCORES)
    errors = ds1.load_error_population()
    mapping = ds1.site_to_errors(errors)
    candidates = pd.read_parquet(ds1.DECISION_POPULATION)
    arm = set(ds1.arm_candidates(candidates, ds1.PIPELINE_ARM[ds1.P2])["candidate_id"])
    thresholds = cc_read_json(ds1.THRESHOLD_REGISTRY)["by_target"]
    out: dict[str, Any] = {}
    for block in ("threshold", "test"):
        context = ds1.block_context(scores, errors, block)
        frame = context["scores"][context["scores"]["candidate_id"].isin(arm)].copy()
        frame["grade"] = relevance(frame)
        by_site = frame.groupby("site_key").agg(
            candidates=("candidate_id", "size"), exact=("exact", "sum")
        )
        choice = set(
            by_site[
                (by_site["candidates"] >= 2)
                & (by_site["exact"] >= 1)
                & (by_site["exact"] < by_site["candidates"])
            ].index
        )
        ds1_top = ds1.top_per_site(frame, "score_benefit", False)
        oracle_top = ds1.top_per_site(frame.assign(_g=frame["grade"]), "_g", False)
        rows = {}
        for key, row in sorted(thresholds.items()):
            tau = row["threshold"]
            gate_ds1 = (
                ds1_top[ds1_top["safety"] >= float(tau)] if tau is not None else ds1_top.head(0)
            )
            gate_oracle_rank = (
                oracle_top[oracle_top["safety"] >= float(tau)]
                if tau is not None
                else oracle_top.head(0)
            )

            def recall(
                decisions: pd.DataFrame,
                accepted: pd.DataFrame,
                keys: set[str] = context["error_keys"],
                documents: int = context["documents"],
            ) -> float:
                return ds1.evaluate_policy(
                    decisions, accepted, keys, mapping, documents
                ).repair_recall

            rows[key] = {
                "ds1_ranking_ds1_gate": recall(ds1_top, gate_ds1),
                "perfect_ranking_ds1_gate": recall(oracle_top, gate_oracle_rank),
                "ds1_ranking_perfect_gate": recall(ds1_top, ds1_top[~ds1_top["is_harmful"]]),
                "perfect_ranking_perfect_gate": recall(
                    oracle_top, oracle_top[~oracle_top["is_harmful"]]
                ),
            }
            total = rows[key]["perfect_ranking_perfect_gate"] - rows[key]["ds1_ranking_ds1_gate"]
            rows[key]["gap"] = total
            rows[key]["share_closed_by_perfect_ranking"] = _ratio(
                rows[key]["perfect_ranking_ds1_gate"] - rows[key]["ds1_ranking_ds1_gate"], total
            )
            rows[key]["share_closed_by_perfect_gate"] = _ratio(
                rows[key]["ds1_ranking_perfect_gate"] - rows[key]["ds1_ranking_ds1_gate"], total
            )
        chosen = ds1_top[ds1_top["site_key"].isin(choice)]
        out[block] = {
            "decision_sites": int(by_site.shape[0]),
            "single_candidate_sites": int((by_site["candidates"] == 1).sum()),
            "multi_candidate_sites": int((by_site["candidates"] >= 2).sum()),
            "choice_sites": len(choice),
            "ds1_top1_on_choice_sites": float(chosen["exact"].mean()) if len(chosen) else None,
            "by_target": rows,
        }
    population = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    universe = population[population["population"] == rl1.PRIMARY_POPULATION]
    size = universe.groupby("site_group").size()
    _write_json_once(
        HEADROOM,
        {
            **_analysis_envelope("ranking_headroom"),
            "by_block": out,
            "universe": {
                "sites": len(size),
                "candidates_per_site": {
                    str(k): int(v) for k, v in size.value_counts().sort_index().items()
                },
                "single_candidate_share": _ratio(int((size == 1).sum()), len(size)),
            },
            "reading": (
                "a site with one candidate offers a ranker nothing to reorder. Perfect within-"
                "site ranking under DS1's own gate is the most any within-site ranker could add; "
                "a perfect gate under DS1's own ranking is the most any accept rule could add"
            ),
            "measured_before_any_ranker": True,
        },
    )
    test = out["test"]["by_target"][rl1.epsilon_key(PRIMARY_EPSILON)]
    print(
        f"headroom: at the primary target, perfect ranking closes "
        f"{test['share_closed_by_perfect_ranking']:.4f} and a perfect gate "
        f"{test['share_closed_by_perfect_gate']:.4f} of the gap ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the ranking population

POP_U5 = "u5_dual_union"
POP_3SRC = "three_source"


def _three_source_rows() -> pd.DataFrame:
    """HY1's dual-union arm plus the text-only Qwen condition, deduplicated by RL1's own rule."""
    candidates = hy1._attach_membership(
        pd.read_parquet(hy1.CANDIDATES), pd.read_parquet(hy1.PROPOSAL_STRATA)
    )
    union = hy1.arm_candidates(candidates, hy1.H5)
    text = hy1.arm_candidates(candidates, hy1.HT)
    text = text[text["corrector_source"].astype(str) == C1_TEXT]
    return rl1.deduplicate(pd.concat([union, text], ignore_index=True))


def _as_population(block: pd.DataFrame, name: str) -> pd.DataFrame:
    """RL1's population columns, built by RL1's own rules, for a block RL1 never built."""
    kinds = rl1._error_kind_of_site()
    specs = {spec["environment"]: spec for spec in rl1.environment_specs()}
    keys = list(
        zip(block["environment"].astype(str), block["lattice_site_id"].astype(str), strict=True)
    )
    out = pd.DataFrame(
        {
            "population": name,
            "candidate_id": block["candidate_id"].astype(str).to_numpy(),
            "environment": block["environment"].astype(str).to_numpy(),
            "corpus": block["corpus"].astype(str).to_numpy(),
            "domain": [rl1._domain(specs[e]["corpus"]) for e in block["environment"].astype(str)],
            "document_id": block["document_id"].astype(str).to_numpy(),
            "site_id": block["lattice_site_id"].astype(str).to_numpy(),
            "proposal_stratum": block["proposal_stratum"].astype(str).to_numpy(),
            "corrector_source": block["corrector_source"].astype(str).to_numpy(),
            "corrector_sources": block["corrector_sources"].astype(str).to_numpy(),
            "multi_source": block["multi_source"].astype(bool).to_numpy(),
            "generator_rank": block["generator_rank"].astype(int).to_numpy(),
            "error_kind": [kinds.get(k, "none") for k in keys],
            "outcome": block["outcome"].astype(str).to_numpy(),
            "is_harmful": block["is_harmful"].astype(bool).to_numpy(),
            "beneficial": block["beneficial"].astype(bool).to_numpy(),
            "exact": block["exact"].astype(bool).to_numpy(),
        }
    )
    out["site_group"] = out["environment"] + "|" + out["site_id"]
    return out


def attach_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    folds = cc_read_json(rl1.GROUP_REGISTRY)["document_folds"]
    mapped = frame["document_id"].astype(str).map(folds)
    if mapped.isna().any():
        raise PhaseError("a candidate's document carries no RL1 fold assignment")
    frame = frame.copy()
    frame["fold"] = mapped.to_numpy(dtype=np.int64)
    frame["block"] = [ds1.block_of(int(f)) for f in frame["fold"]]
    frame["site_key"] = frame["site_group"]
    frame["grade"] = relevance(frame)
    return frame


def run_population() -> int:
    """The primary population is RL1's U5 exactly; the third source joins only for transfer."""
    started = time.monotonic()
    _require(HEADROOM, "headroom")
    for path in (RANKING_POPULATION, POPULATION_INVENTORY):
        _forbid(path)
    frozen = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    u5 = frozen[frozen["population"] == rl1.PRIMARY_POPULATION].reset_index(drop=True)
    u5 = u5.assign(population=POP_U5)
    three = _as_population(_three_source_rows(), POP_3SRC)
    columns = list(three.columns)
    frame = pd.concat([u5[columns], three], ignore_index=True)
    frame = attach_blocks(frame)
    frame = frame.sort_values(["population", "environment", "candidate_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    _write_parquet_once(RANKING_POPULATION, frame)
    inventory: dict[str, Any] = {}
    for name, block in frame.groupby("population", sort=True):
        size = block.groupby("site_group").size()
        inventory[str(name)] = {
            "candidates": len(block),
            "sites": len(size),
            "documents": int(block["document_id"].nunique()),
            "by_source": {
                str(k): int(v)
                for k, v in block["corrector_source"].value_counts().sort_index().items()
            },
            "multi_source_candidates": int(block["multi_source"].sum()),
            "single_candidate_sites": int((size == 1).sum()),
            "multi_candidate_sites": int((size >= 2).sum()),
            "by_grade": {
                str(k): int(v) for k, v in block["grade"].value_counts().sort_index().items()
            },
        }
    _write_json_once(
        POPULATION_INVENTORY,
        {
            **_analysis_envelope("ranking_population_inventory"),
            "populations": inventory,
            "primary": POP_U5,
            "u5_is_rl1s_population_exactly": True,
            "third_source": {
                "source": C1_TEXT,
                "label": SOURCE_LABEL[C1_TEXT],
                "arm": hy1.HT,
                "note": (
                    "the text-only Qwen condition shares the image corrector's checkpoint; it is "
                    "used only in the secondary leave-one-source-out analysis"
                ),
            },
            "never_regenerated_here": True,
        },
    )
    print(
        f"population: {inventory[POP_U5]['candidates']} primary, "
        f"{inventory[POP_3SRC]['candidates']} three-source ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the frozen features

RELATIVE_BASE = ("plaus_lm_delta", "edit_changed_fraction", "vis_glyph_gain", "txt_len_delta")
FAM_RELATIVE = "rel_"


def relative_features(frame: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Within-site relative evidence: value minus site mean, and within-site rank fraction.

    Site size is deliberately NOT one of them. Both still take a fixed value at a single-candidate
    site, and single-candidate sites are disproportionately one generator's in this universe, so
    these features carry some generator information; the leakage classifier measures how much.
    """
    joined = frame[["candidate_id", "site_group"]].merge(
        matrix[["candidate_id", *RELATIVE_BASE]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    out = pd.DataFrame({"candidate_id": joined["candidate_id"].to_numpy()})
    for name in RELATIVE_BASE:
        grouped = joined.groupby("site_group")[name]
        out[f"{FAM_RELATIVE}{name}_minus_site_mean"] = (
            joined[name] - grouped.transform("mean")
        ).to_numpy(dtype=np.float64)
        rank = grouped.rank(method="average", ascending=True)
        size = grouped.transform("size")
        out[f"{FAM_RELATIVE}{name}_site_rank_fraction"] = np.where(
            size > 1, (rank - 1.0) / (size - 1.0).clip(lower=1.0), 0.5
        )
    return out


def _rebuild_features(candidate_ids: set[str]) -> pd.DataFrame:
    """RL1's own featurizer, pointed at a candidate set RL1 never featurized."""
    import shutil

    sandbox = CACHE / "featurize"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    wanted = sandbox / "wanted.parquet"
    pd.DataFrame({"candidate_id": sorted(candidate_ids)}).to_parquet(wanted, index=False)
    saved = rl1.CANDIDATE_POPULATION
    try:
        rl1.CANDIDATE_POPULATION = wanted
        return rl1.build_features()
    finally:
        rl1.CANDIDATE_POPULATION = saved


def run_features() -> int:
    """R3 for every ranking candidate, rebuilt by RL1's featurizer and reproduced against RL1."""
    started = time.monotonic()
    _require(RANKING_POPULATION, "population")
    for path in (FEATURE_MATRIX, FEATURE_REPRODUCTION):
        _forbid(path)
    population = pd.read_parquet(RANKING_POPULATION)
    rebuilt = _rebuild_features(set(population["candidate_id"].astype(str)))
    published = pd.read_parquet(rl1.FEATURE_MATRIX)
    columns = list(rl1.FEATURE_NAMES)
    overlap = sorted(set(published["candidate_id"]) & set(rebuilt["candidate_id"]))
    left = published.set_index("candidate_id").loc[overlap, columns].to_numpy(dtype=np.float64)
    right = rebuilt.set_index("candidate_id").loc[overlap, columns].to_numpy(dtype=np.float64)
    difference = float(np.abs(left - right).max()) if overlap else float("nan")
    if difference != 0.0:
        raise PhaseError(f"RL1's featurizer did not reproduce RL1's matrix: {difference}")
    relative = relative_features(population.drop_duplicates("candidate_id"), rebuilt)
    matrix = rebuilt.merge(relative, on="candidate_id", how="left", validate="one_to_one")
    matrix = matrix.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    _write_parquet_once(FEATURE_MATRIX, matrix)
    _write_json_once(
        FEATURE_REPRODUCTION,
        {
            **_envelope("feature_reproduction"),
            "featurizer": "sgv_rl1_generator_agnostic_reliability.build_features, unchanged",
            "rows_rebuilt": len(rebuilt),
            "rows_already_in_rl1": len(overlap),
            "rows_new_to_rk1": len(rebuilt) - len(overlap),
            "max_absolute_difference_on_rl1_rows": difference,
            "identical_on_rl1_rows": bool(difference == 0.0),
            "relative_features": [c for c in relative.columns if c != "candidate_id"],
            "r3_columns": len(rl1.columns_for(REPRESENTATION)),
            "uses_ground_truth": False,
        },
    )
    print(
        f"features: {len(rebuilt)} rows, {len(overlap)} reproduce RL1 exactly "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ the splits

SPLIT_BLOCKS = "ds1_blocks"


def _sites_carrying(frame: pd.DataFrame, source: str) -> set[str]:
    direct = frame["corrector_source"].astype(str) == source
    listed = frame["corrector_sources"].astype(str).str.contains(source, regex=False)
    return set(frame.loc[direct | listed, "site_group"].astype(str))


@dataclass(slots=True)
class Split:
    """One train/evaluate partition over a population, as index arrays into that population."""

    name: str
    population: str
    train: np.ndarray
    test: np.ndarray
    held_out: str | None


def build_splits(population: pd.DataFrame) -> list[Split]:
    """DS1's blocks, RL1's strict two-generator transfer, and the three-source extension."""
    out: list[Split] = []
    u5 = population[population["population"] == POP_U5].reset_index(drop=True)
    block = u5["block"].to_numpy(str)
    out.append(
        Split(
            SPLIT_BLOCKS,
            POP_U5,
            np.flatnonzero(block == "fit"),
            np.flatnonzero(block != "fit"),
            None,
        )
    )
    for held in DISTINCT_GENERATORS:
        other = [s for s in DISTINCT_GENERATORS if s != held]
        held_sites = _sites_carrying(u5, held)
        other_sites = set().union(*(_sites_carrying(u5, s) for s in other))
        corrector = u5["corrector_source"].astype(str).to_numpy()
        site = u5["site_group"].astype(str).to_numpy()
        out.append(
            Split(
                f"logo_{held}",
                POP_U5,
                np.flatnonzero(np.isin(corrector, other) & ~np.isin(site, list(held_sites))),
                np.flatnonzero((corrector == held) & ~np.isin(site, list(other_sites))),
                held,
            )
        )
    three = population[population["population"] == POP_3SRC].reset_index(drop=True)
    for held in SOURCES:
        other = [s for s in SOURCES if s != held]
        held_sites = _sites_carrying(three, held)
        other_sites = set().union(*(_sites_carrying(three, s) for s in other))
        corrector = three["corrector_source"].astype(str).to_numpy()
        site = three["site_group"].astype(str).to_numpy()
        out.append(
            Split(
                f"loso_{held}",
                POP_3SRC,
                np.flatnonzero(np.isin(corrector, other) & ~np.isin(site, list(held_sites))),
                np.flatnonzero((corrector == held) & ~np.isin(site, list(other_sites))),
                held,
            )
        )
    return out


def population_frame(population: pd.DataFrame, name: str) -> pd.DataFrame:
    return population[population["population"] == name].reset_index(drop=True)


def run_splits() -> int:
    started = time.monotonic()
    _require(FEATURE_MATRIX, "features")
    _forbid(SPLIT_REGISTRY)
    population = pd.read_parquet(RANKING_POPULATION)
    rows = []
    for split in build_splits(population):
        frame = population_frame(population, split.population)
        train = frame.iloc[split.train]
        test = frame.iloc[split.test]

        def shared(column: str, left: pd.DataFrame = train, right: pd.DataFrame = test) -> int:
            return len(set(left[column].astype(str)) & set(right[column].astype(str)))

        rows.append(
            {
                "split": split.name,
                "population": split.population,
                "held_out": split.held_out,
                "train_rows": int(split.train.size),
                "test_rows": int(split.test.size),
                "train_sources": sorted(set(train["corrector_source"].astype(str))),
                "test_sources": sorted(set(test["corrector_source"].astype(str))),
                "shared_sites": shared("site_group"),
                "shared_documents": shared("document_id"),
                "test_harmful": int(test["is_harmful"].sum()),
                "test_beneficial": int(test["beneficial"].sum()),
                "test_exact": int(test["exact"].sum()),
            }
        )
    _write_json_once(
        SPLIT_REGISTRY,
        {
            **_analysis_envelope("split_registry"),
            "splits": rows,
            "no_site_crosses_any_split": all(row["shared_sites"] == 0 for row in rows),
            "documents_crossing_the_ds1_blocks": next(
                row["shared_documents"] for row in rows if row["split"] == SPLIT_BLOCKS
            ),
            "transfer_shares_documents_by_design": (
                "the generator splits follow SGV-RL1's strict rule, which holds out the generator "
                "and the sites but not the documents, so the transfer floor is comparable with "
                "RL1's own result"
            ),
        },
    )
    print(f"splits: {len(rows)} splits ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ the rankers


@dataclass(slots=True)
class LambdaMART:
    """A fitted LambdaMART ensemble: regression trees with Newton leaf values."""

    trees: list[Any]
    leaf_values: list[np.ndarray]
    learning_rate: float

    def score(self, design: np.ndarray) -> np.ndarray:
        out = np.zeros(design.shape[0], dtype=np.float64)
        for tree, values in zip(self.trees, self.leaf_values, strict=True):
            out += self.learning_rate * values[tree.apply(design)]
        return out


def _lambda_gradients(
    scores: np.ndarray, grades: np.ndarray, queries: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """LambdaRank gradients and second-order weights, one query at a time.

    For every pair with a higher-graded i and lower-graded j, the gradient pushes i up and j down
    by sigma * rho * |delta NDCG|, where rho is the probability the current scores order the pair
    wrongly and delta NDCG is what swapping the two would change. Queries whose candidates share
    one grade contribute nothing, which is the point: they offer no ordering to learn.
    """
    lambdas = np.zeros(scores.size, dtype=np.float64)
    weights = np.zeros(scores.size, dtype=np.float64)
    for index in queries:
        grade = grades[index]
        gain = np.power(2.0, grade) - 1.0
        size = index.size
        discount_ideal = 1.0 / np.log2(np.arange(size) + 2.0)
        ideal = float((np.sort(gain)[::-1] * discount_ideal).sum())
        if ideal <= 0.0:
            continue
        score = scores[index]
        order = np.argsort(-score, kind="mergesort")
        position = np.empty(size, dtype=np.int64)
        position[order] = np.arange(size)
        discount = 1.0 / np.log2(position + 2.0)
        better = grade[:, None] > grade[None, :]
        delta = (
            np.abs((gain[:, None] - gain[None, :]) * (discount[:, None] - discount[None, :]))
            / ideal
        )
        difference = np.clip(score[:, None] - score[None, :], -50.0, 50.0)
        rho = 1.0 / (1.0 + np.exp(LAMBDAMART_SIGMA * difference))
        pair = LAMBDAMART_SIGMA * rho * delta * better
        curvature = (LAMBDAMART_SIGMA**2) * rho * (1.0 - rho) * delta * better
        lambdas[index] += pair.sum(axis=1) - pair.sum(axis=0)
        weights[index] += curvature.sum(axis=1) + curvature.sum(axis=0)
    return lambdas, weights


def fit_lambdamart(design: np.ndarray, grades: np.ndarray, groups: np.ndarray) -> LambdaMART:
    """LambdaMART (Burges 2010) on sklearn regression trees, fixed before any fit.

    Only queries with at least two distinct grades carry a gradient, so trees are fitted on the
    rows of those queries alone; every row can still be scored afterwards. Leaf values are the
    Newton step -- summed gradients over summed curvature -- rather than the tree's own mean.
    """
    from sklearn.tree import DecisionTreeRegressor

    keys = pd.Series(groups).astype(str)
    queries = [
        np.asarray(index, dtype=np.int64)
        for _key, index in sorted(keys.groupby(keys).groups.items())
        if len(set(grades[np.asarray(index)].tolist())) >= 2
    ]
    if not queries:
        return LambdaMART(trees=[], leaf_values=[], learning_rate=LAMBDAMART_LEARNING_RATE)
    used = np.sort(np.concatenate(queries))
    scores = np.zeros(design.shape[0], dtype=np.float64)
    trees: list[Any] = []
    leaf_values: list[np.ndarray] = []
    for _round in range(LAMBDAMART_ROUNDS):
        lambdas, weights = _lambda_gradients(scores, grades, queries)
        if not np.any(lambdas[used]):
            break
        tree = DecisionTreeRegressor(
            max_depth=LAMBDAMART_MAX_DEPTH,
            min_samples_leaf=LAMBDAMART_MIN_LEAF,
            random_state=FIT_SEED,
        ).fit(design[used], lambdas[used])
        leaves = tree.apply(design[used])
        values = np.zeros(tree.tree_.node_count, dtype=np.float64)
        for leaf in np.unique(leaves):
            member = leaves == leaf
            values[leaf] = float(lambdas[used][member].sum()) / (
                float(weights[used][member].sum()) + 1e-9
            )
        scores += LAMBDAMART_LEARNING_RATE * values[tree.apply(design)]
        trees.append(tree)
        leaf_values.append(values)
    return LambdaMART(trees=trees, leaf_values=leaf_values, learning_rate=LAMBDAMART_LEARNING_RATE)


def fit_ranknet(design: np.ndarray, frame: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """The repository's SGV5 pairwise ranker: RankNet with a linear scorer, solved not descended.

    Pairs are beneficial-versus-harmful, within a site first because that is the question an
    operator faces, then filled with cross-site pairs to a fixed count.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(design)
    scaled = np.asarray(scaler.transform(design), dtype=np.float64)
    beneficial = np.flatnonzero(frame["beneficial"].to_numpy(dtype=bool))
    harmful = np.flatnonzero(frame["is_harmful"].to_numpy(dtype=bool))
    if beneficial.size == 0 or harmful.size == 0:
        return scaler, np.zeros(design.shape[1])
    sites = frame["site_group"].astype(str).to_numpy()
    by_site: dict[str, tuple[list[int], list[int]]] = {}
    for row in beneficial.tolist():
        by_site.setdefault(sites[row], ([], []))[0].append(row)
    for row in harmful.tolist():
        by_site.setdefault(sites[row], ([], []))[1].append(row)
    within_good: list[int] = []
    within_bad: list[int] = []
    for key in sorted(by_site):
        good, bad = by_site[key]
        for a in good:
            for b in bad:
                within_good.append(a)
                within_bad.append(b)
    rng = np.random.default_rng(FIT_SEED)
    remaining = max(RANKNET_PAIRS - len(within_good), 0)
    positive = np.concatenate(
        [np.asarray(within_good, dtype=np.int64), rng.choice(beneficial, remaining)]
    )
    negative = np.concatenate(
        [np.asarray(within_bad, dtype=np.int64), rng.choice(harmful, remaining)]
    )
    difference = scaled[positive] - scaled[negative]
    model = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(
        np.vstack([difference, -difference]),
        np.concatenate([np.ones(len(difference)), np.zeros(len(difference))]),
    )
    return scaler, np.asarray(model.coef_[0], dtype=np.float64)


def _design(frame: pd.DataFrame, matrix: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return rl1._matrix_for(frame, matrix, list(columns))


def model_columns(model: str, drop: Sequence[str] = (), keep: Sequence[str] = ()) -> list[str]:
    if model == B2:
        return rl1.columns_for(rl1.R4)
    base = columns_only(keep) if keep else columns_without(drop)
    if model == LM_REL:
        return [*base, *relative_columns()]
    return base


def relative_columns() -> list[str]:
    return [
        f"{FAM_RELATIVE}{name}_{suffix}"
        for name in RELATIVE_BASE
        for suffix in ("minus_site_mean", "site_rank_fraction")
    ]


def score_model(
    model: str,
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """One model fitted on `train` and scored on `test`: (rank score, safety score).

    The rank score picks the candidate within a site; the safety score gates it. A ranker trained
    on graded relevance puts harm at the bottom of its scale, so for the rankers both are the same
    learned score. The DS1 reliability score keeps its own two heads, exactly as DS1 used them.
    """
    if model == B0:
        generator = np.random.default_rng(
            int(canonical_hash({"seed": FIT_SEED, "rows": int(test.size)})[:8], 16)
        )
        value = generator.random(test.size)
        return value, value
    design = _design(frame, matrix, columns)
    if model == B1:
        benefit = rl1.fit_and_score(
            rl1.M0, design[train], frame["beneficial"].to_numpy(bool)[train], design[test]
        )
        harm = rl1.fit_and_score(
            rl1.M0, design[train], frame["is_harmful"].to_numpy(bool)[train], design[test]
        )
        return benefit, 1.0 - harm
    if model == RN:
        scaler, weights = fit_ranknet(design[train], frame.iloc[train].reset_index(drop=True))
        value = np.asarray(scaler.transform(design[test]), dtype=np.float64) @ weights
        return value, value
    grades = frame["grade"].to_numpy(dtype=np.int64)
    granularity = "site_group" if model == LM_SITE else "document_id"
    groups = frame[granularity].astype(str).to_numpy()
    fitted = fit_lambdamart(design[train], grades[train], groups[train])
    value = fitted.score(design[test])
    return value, value


V_FULL = "full"
FAMILIES = tuple(FAMILY_PREFIXES)


def rank_jobs() -> list[dict[str, Any]]:
    """Every pre-registered fit, fixed before any score exists."""
    jobs: list[dict[str, Any]] = []
    for model in MODELS:
        jobs.append(
            {"split": SPLIT_BLOCKS, "model": model, "variant": V_FULL, "drop": (), "keep": ()}
        )
    for name, drop in ABLATIONS.items():
        jobs.append(
            {
                "split": SPLIT_BLOCKS,
                "model": PRIMARY_RANKER,
                "variant": name,
                "drop": drop,
                "keep": (),
            }
        )
    for family in FAMILIES:
        jobs.append(
            {
                "split": SPLIT_BLOCKS,
                "model": PRIMARY_RANKER,
                "variant": f"only_{family}",
                "drop": (),
                "keep": FAMILY_PREFIXES[family],
            }
        )
    for held in DISTINCT_GENERATORS:
        for model in (B1, B2, RN, LM_SITE, LM_DOC, LM_REL):
            jobs.append(
                {"split": f"logo_{held}", "model": model, "variant": V_FULL, "drop": (), "keep": ()}
            )
    for held in SOURCES:
        for model in (B1, PRIMARY_RANKER):
            jobs.append(
                {"split": f"loso_{held}", "model": model, "variant": V_FULL, "drop": (), "keep": ()}
            )
    return jobs


def run_rank() -> int:
    """Every pre-registered ranker, fitted on its training rows and scored on its test rows."""
    started = time.monotonic()
    _require(SPLIT_REGISTRY, "splits")
    for path in (RANKING_SCORES, MODEL_REGISTRY, TRAINING_REGISTRY):
        _forbid(path)
    population = pd.read_parquet(RANKING_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    splits = {split.name: split for split in build_splits(population)}
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    for job in rank_jobs():
        began = time.monotonic()
        split = splits[job["split"]]
        frame = population_frame(population, split.population)
        columns = model_columns(job["model"], job["drop"], job["keep"])
        rank_score, safety = score_model(
            job["model"], frame, matrix, split.train, split.test, columns
        )
        frames.append(
            pd.DataFrame(
                {
                    "population": split.population,
                    "split": split.name,
                    "model": job["model"],
                    "variant": job["variant"],
                    "candidate_id": frame.iloc[split.test]["candidate_id"].to_numpy(),
                    "rank_score": rank_score,
                    "safety": safety,
                }
            )
        )
        registry.append(
            {
                "split": split.name,
                "model": job["model"],
                "variant": job["variant"],
                "columns": len(columns),
                "carries_source_columns": bool(
                    [c for c in columns if c.startswith(rl1.FAM_SOURCE)]
                ),
                "train_rows": int(split.train.size),
                "test_rows": int(split.test.size),
                "runtime_seconds": round(time.monotonic() - began, 3),
            }
        )
        print(
            f"  {job['split']} {job['model']} {job['variant']}: {time.monotonic() - began:.1f}s",
            flush=True,
        )
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(
        ["population", "split", "model", "variant", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    _write_parquet_once(RANKING_SCORES, table)
    published = pd.read_parquet(ds1.DEPLOYMENT_SCORES).set_index("candidate_id")
    b1 = table[
        (table["split"] == SPLIT_BLOCKS) & (table["model"] == B1) & (table["variant"] == V_FULL)
    ].set_index("candidate_id")
    shared = b1.index.intersection(published.index[published["block"] != "fit"])
    reproduction = {
        "rows": len(shared),
        "rank_score_max_difference": float(
            np.abs(b1.loc[shared, "rank_score"] - published.loc[shared, "score_benefit"]).max()
        ),
        "safety_max_difference": float(
            np.abs(b1.loc[shared, "safety"] - published.loc[shared, "safety"]).max()
        ),
    }
    reproduction["identical"] = bool(
        reproduction["rank_score_max_difference"] == 0.0
        and reproduction["safety_max_difference"] == 0.0
    )
    if not reproduction["identical"]:
        raise PhaseError(f"the DS1 reliability score did not reproduce: {reproduction}")
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "models": MODEL_LABEL,
            "primary_ranker": PRIMARY_RANKER,
            "generator_agnostic": list(GENERATOR_AGNOSTIC),
            "leakage_reference": B2,
            "lambdamart": cc_read_json(DESIGN_RECORD)["lambdamart"],
            "ranknet": cc_read_json(DESIGN_RECORD)["ranknet"],
            "ds1_reproduction": reproduction,
            "established_prior_art": (
                "LambdaMART, RankNet and pairwise learning-to-rank are established methods; this "
                "stage applies them and does not claim them"
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
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"rank: {len(registry)} jobs, DS1 score reproduced exactly on {reproduction['rows']} rows "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ ranking metrics


def scored_frame(
    scores: pd.DataFrame, population: pd.DataFrame, split: str, model: str, variant: str
) -> pd.DataFrame:
    """One model's scores joined to the labels, restricted to the rows it is evaluated on."""
    cell = scores[
        (scores["split"] == split) & (scores["model"] == model) & (scores["variant"] == variant)
    ]
    if cell.empty:
        return cell
    name = str(cell["population"].iloc[0])
    frame = population_frame(population, name)
    joined = cell.merge(
        frame[
            [
                "candidate_id",
                "site_group",
                "document_id",
                "environment",
                "block",
                "corrector_source",
                "is_harmful",
                "beneficial",
                "exact",
                "grade",
            ]
        ],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if split == SPLIT_BLOCKS:
        joined = joined[joined["block"] == "test"]
    return joined.reset_index(drop=True)


def _ordered(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(["rank_score", "candidate_id"], ascending=[False, True], kind="stable")


def _dcg(grades: np.ndarray, cutoff: int) -> float:
    head = grades[:cutoff]
    return float(((np.power(2.0, head) - 1.0) / np.log2(np.arange(head.size) + 2.0)).sum())


def ranking_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Top-1, Recall@K and NDCG@K over the sites where ranking can matter, plus AUROC."""
    if frame.empty:
        return {"rows": 0}
    top1_hits: list[bool] = []
    recall: dict[int, list[bool]] = {k: [] for k in RANK_CUTOFFS}
    ndcg: dict[int, list[float]] = {k: [] for k in NDCG_CUTOFFS}
    for _site, group in frame.groupby("site_group", sort=True):
        ordered = _ordered(group)
        exact = ordered["exact"].to_numpy(dtype=bool)
        grades = ordered["grade"].to_numpy(dtype=np.int64)
        if len(ordered) >= 2 and exact.any() and not exact.all():
            top1_hits.append(bool(exact[0]))
        if len(ordered) >= 2 and exact.any():
            for k in RANK_CUTOFFS:
                recall[k].append(bool(exact[:k].any()))
        if len(set(grades.tolist())) >= 2:
            ideal = np.sort(grades)[::-1]
            for k in NDCG_CUTOFFS:
                best = _dcg(ideal, k)
                if best > 0:
                    ndcg[k].append(_dcg(grades, k) / best)
    # A constant scorer orders candidates by the id tie-break alone, which is not a ranking, so
    # every metric of it is undefined rather than chance.
    defined = bool(frame["rank_score"].nunique() > 1)

    def share(values: Sequence[Any]) -> float | None:
        return float(np.mean(values)) if (values and defined) else None

    return {
        "rows": len(frame),
        "choice_sites": len(top1_hits),
        "top1": share(top1_hits),
        "recall_at_k": {str(k): share(v) for k, v in recall.items()},
        "recall_sites": len(recall[RANK_CUTOFFS[0]]),
        "ndcg_at_k": {str(k): share(v) for k, v in ndcg.items()},
        "ndcg_sites": len(ndcg[NDCG_CUTOFFS[0]]),
        "harm_auroc": auroc(
            -frame["safety"].to_numpy(np.float64), frame["is_harmful"].to_numpy(bool)
        )
        if defined
        else None,
        "benefit_auroc": auroc(
            frame["rank_score"].to_numpy(np.float64), frame["beneficial"].to_numpy(bool)
        )
        if defined
        else None,
        "constant_score": not defined,
    }


def top1_by_site_kind(frame: pd.DataFrame) -> dict[str, Any]:
    """Top-1 split by whether a choice site's candidates come from one generator or both.

    A document-level ranker could win Top-1 simply by preferring one generator wherever both
    propose. At a single-generator site that preference is useless, so the split shows which kind
    of site any Top-1 difference comes from.
    """
    out: dict[str, list[bool]] = {"mixed_generators": [], "single_generator": []}
    if frame.empty or frame["rank_score"].nunique() <= 1:
        return {kind: {"sites": 0, "top1": None} for kind in out}
    for _site, group in frame.groupby("site_group", sort=True):
        exact = group["exact"].to_numpy(dtype=bool)
        if len(group) >= 2 and exact.any() and not exact.all():
            kind = (
                "mixed_generators"
                if group["corrector_source"].nunique() > 1
                else "single_generator"
            )
            out[kind].append(bool(_ordered(group)["exact"].iloc[0]))
    return {
        kind: {"sites": len(hits), "top1": float(np.mean(hits)) if hits else None}
        for kind, hits in out.items()
    }


def _graded_queries(frame: pd.DataFrame, rows: np.ndarray, key: str) -> int:
    block = frame.iloc[rows]
    return int(sum(1 for _k, group in block.groupby(key) if group["grade"].nunique() >= 2))


def run_metrics() -> int:
    """RQ1: Top-1, Recall@K, NDCG@K and AUROC for every model, on every split's test rows."""
    started = time.monotonic()
    _require(RANKING_SCORES, "rank")
    for path in (RANKING_METRICS, DISCRIMINATION):
        _forbid(path)
    scores = pd.read_parquet(RANKING_SCORES)
    population = pd.read_parquet(RANKING_POPULATION)
    splits = {split.name: split for split in build_splits(population)}
    out: dict[str, Any] = {}
    trainable: dict[str, Any] = {}
    for (split, model, variant), _group in scores.groupby(["split", "model", "variant"], sort=True):
        frame = scored_frame(scores, population, str(split), str(model), str(variant))
        out.setdefault(str(split), {}).setdefault(str(model), {})[str(variant)] = ranking_metrics(
            frame
        )
        if str(model) in (LM_SITE, LM_DOC, LM_REL, B2) and str(variant) == V_FULL:
            spec = splits[str(split)]
            base = population_frame(population, spec.population)
            key = "site_group" if str(model) == LM_SITE else "document_id"
            queries = _graded_queries(base, spec.train, key)
            trainable[f"{split}|{model}"] = {
                "graded_training_queries": queries,
                "trainable": queries > 0,
            }
    _write_json_once(
        RANKING_METRICS,
        {
            **_analysis_envelope("ranking_metrics"),
            "by_split": out,
            "evaluated_rows": "the test block for the DS1 split; every test row elsewhere",
            "choice_site_definition": cc_read_json(DESIGN_RECORD)["metrics"]["choice_site"],
            "trainability": trainable,
            "untrainable_note": (
                "a LambdaMART ranker learns only from queries whose candidates carry at least two "
                "grades. The image corrector contributes one candidate per site, so a site-level "
                "ranker trained on its rows alone has nothing to learn and scores every candidate "
                "the same; its metrics are recorded as undefined rather than as chance"
            ),
            "cutoffs": {"recall": list(RANK_CUTOFFS), "ndcg": list(NDCG_CUTOFFS)},
        },
    )
    blocks = out[SPLIT_BLOCKS]
    by_kind = {
        model: top1_by_site_kind(scored_frame(scores, population, SPLIT_BLOCKS, model, V_FULL))
        for model in MODELS
    }
    _write_json_once(
        DISCRIMINATION,
        {
            **_analysis_envelope("discrimination"),
            "split": SPLIT_BLOCKS,
            "block": "test",
            "by_model": {
                model: {
                    "harm_auroc": blocks[model][V_FULL]["harm_auroc"],
                    "benefit_auroc": blocks[model][V_FULL]["benefit_auroc"],
                }
                for model in MODELS
                if model in blocks
            },
            "orientation": (
                "harm AUROC ranks harmful candidates below safe ones by the safety score; benefit "
                "AUROC ranks beneficial candidates above the rest by the rank score"
            ),
            "top1_by_choice_site_kind": by_kind,
        },
    )
    primary = blocks[PRIMARY_RANKER][V_FULL]
    reference = blocks[B1][V_FULL]
    print(
        f"metrics: test Top-1 {primary['top1']:.4f} vs DS1 {reference['top1']:.4f} on "
        f"{primary['choice_sites']} choice sites ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ RQ4: deployment


def choose_threshold(decisions: pd.DataFrame, epsilon: float) -> float | None:
    """The loosest cutoff on the threshold block whose realized harm meets the target.

    Every distinct safety value is a candidate cutoff, so no model is disadvantaged by a lattice
    tuned to another model's scale. Accept sets are "safety at or above the cutoff". Otherwise
    this is SGV-DS1's rule: the loosest cutoff that accepts something and meets the target.
    """
    safety = decisions["safety"].to_numpy(dtype=np.float64)
    harmful = decisions["is_harmful"].to_numpy(dtype=bool)
    chosen: float | None = None
    for cutoff in np.unique(safety)[::-1]:
        accepted = safety >= cutoff
        if float(harmful[accepted].mean()) <= epsilon + ds1.HARM_TOLERANCE:
            chosen = float(cutoff)
    return chosen


def _share(numerator: int, denominator: int) -> float | None:
    """A proportion that is None, not NaN, when nothing was counted: NaN breaks JSON equality."""
    return float(numerator / denominator) if denominator else None


def decisions_for(
    scores: pd.DataFrame, population: pd.DataFrame, model: str, variant: str
) -> dict[str, pd.DataFrame]:
    """One decision per site on the threshold and test blocks: the model's top candidate."""
    cell = scores[
        (scores["split"] == SPLIT_BLOCKS)
        & (scores["model"] == model)
        & (scores["variant"] == variant)
    ]
    frame = population_frame(population, POP_U5)
    joined = cell.merge(
        frame[
            [
                "candidate_id",
                "site_group",
                "document_id",
                "environment",
                "block",
                "corrector_source",
                "is_harmful",
                "beneficial",
                "exact",
            ]
        ],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"site_group": "site_key"})
    return {
        block: joined[joined["block"] == block]
        .sort_values(["rank_score", "candidate_id"], ascending=[False, True], kind="stable")
        .groupby("site_key", sort=False)
        .head(1)
        .reset_index(drop=True)
        for block in ("threshold", "test")
    }


def accepted_at(
    decisions: dict[str, pd.DataFrame], epsilon: float
) -> tuple[float | None, pd.DataFrame]:
    """The threshold chosen on the threshold block, and what it accepts on the test block."""
    cutoff = choose_threshold(decisions["threshold"], epsilon)
    test = decisions["test"]
    return cutoff, (test[test["safety"] >= cutoff] if cutoff is not None else test.head(0))


def deployment_for(
    scores: pd.DataFrame,
    population: pd.DataFrame,
    model: str,
    variant: str,
    errors: pd.DataFrame,
    mapping: dict[str, set[str]],
) -> dict[str, Any]:
    """DS1's policy with this model's scores: top per site, gated, threshold from its own block."""
    decisions = decisions_for(scores, population, model, variant)
    test = decisions["test"]
    keys = set(errors[errors["block"] == "test"]["error_key"].astype(str))
    documents = int(test["document_id"].nunique())
    out: dict[str, Any] = {}
    for epsilon in EPSILONS:
        cutoff, accepted = accepted_at(decisions, epsilon)
        row = ds1._as_dict(ds1.evaluate_policy(test, accepted, keys, mapping, documents))
        row["threshold"] = cutoff
        row["abstains"] = cutoff is None
        # A cutoff chosen on the threshold block can accept nothing on the test block. An empty
        # accept set incurs no harm, so it holds the bound, and it is flagged so that it is never
        # read as a working gate.
        row["accepts_nothing_on_test"] = bool(len(accepted) == 0)
        row["holds_its_target"] = bool(
            cutoff is None
            or len(accepted) == 0
            or (
                row["selective_harm_rate"] is not None
                and row["selective_harm_rate"] <= epsilon + ds1.HARM_TOLERANCE
            )
        )
        neutral = ~accepted["exact"] & ~accepted["beneficial"] & ~accepted["is_harmful"]
        row["accepted_exact_share"] = _share(int(accepted["exact"].sum()), len(accepted))
        row["accepted_neutral_share"] = _share(int(neutral.sum()), len(accepted))
        partial = accepted["beneficial"] & ~accepted["exact"]
        row["accepted_partial_share"] = _share(int(partial.sum()), len(accepted))
        row["accepted_by_source"] = {
            source: {
                "accepted": int((accepted["corrector_source"] == source).sum()),
                "exact": int(((accepted["corrector_source"] == source) & accepted["exact"]).sum()),
                "harmful": int(
                    ((accepted["corrector_source"] == source) & accepted["is_harmful"]).sum()
                ),
            }
            for source in DISTINCT_GENERATORS
        }
        out[rl1.epsilon_key(epsilon)] = row
    out["open_loop"] = ds1._as_dict(ds1.evaluate_policy(test, test, keys, mapping, documents))
    return out


def run_deployment() -> int:
    """RQ4 and criterion C: does a ranking objective buy more safe automation than DS1's score?"""
    started = time.monotonic()
    _require(RANKING_METRICS, "metrics")
    _forbid(DEPLOYMENT)
    scores = pd.read_parquet(RANKING_SCORES)
    population = pd.read_parquet(RANKING_POPULATION)
    errors = ds1.load_error_population()
    mapping = ds1.site_to_errors(errors)
    results = {
        model: deployment_for(scores, population, model, V_FULL, errors, mapping)
        for model in MODELS
    }
    ablations = {
        variant: deployment_for(scores, population, PRIMARY_RANKER, variant, errors, mapping)
        for variant in (*ABLATIONS, *(f"only_{f}" for f in FAMILIES))
    }
    published = cc_read_json(ds1.DECISION)["operating_point"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    _write_json_once(
        DEPLOYMENT,
        {
            **_analysis_envelope("deployment_comparison"),
            "by_model": results,
            "by_variant_of_the_primary_ranker": ablations,
            "threshold_rule": (
                "every distinct safety value on the threshold block is a candidate cutoff and the "
                "loosest one meeting the harm target is kept; the same rule for every model"
            ),
            "ds1_published_operating_point": published,
            "ds1_published_used_a_coarser_lattice": (
                "SGV-DS1 searched a 0.005 lattice on its [0, 1] safety score. This stage searches "
                "every distinct value so no model is advantaged by a lattice fitted to another's "
                "scale; the DS1 reliability score is re-run under the same rule as B1"
            ),
            "primary_epsilon": PRIMARY_EPSILON,
            "primary_ranker_recall": results[PRIMARY_RANKER][key]["repair_recall"],
            "ds1_score_recall_same_rule": results[B1][key]["repair_recall"],
        },
    )
    print(
        f"deployment: at harm <= {PRIMARY_EPSILON}, ranker recall "
        f"{results[PRIMARY_RANKER][key]['repair_recall']:.4f} vs DS1 score "
        f"{results[B1][key]['repair_recall']:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ RQ2: generator transfer

LOGO_SPLITS = tuple(f"logo_{held}" for held in DISTINCT_GENERATORS)
LOSO_SPLITS = tuple(f"loso_{held}" for held in SOURCES)
LOGO_MODELS = (B1, B2, RN, LM_SITE, LM_DOC, LM_REL)


def _paired_fold_frame(
    scores: pd.DataFrame, population: pd.DataFrame, split: str, models: Sequence[str]
) -> pd.DataFrame:
    """One held-out fold's test rows, with every named model's safety score side by side."""
    base: pd.DataFrame | None = None
    for model in models:
        frame = scored_frame(scores, population, split, model, V_FULL)
        column = frame[["candidate_id", "safety"]].rename(columns={"safety": model})
        if base is None:
            base = frame[
                ["candidate_id", "environment", "document_id", "is_harmful", "beneficial"]
            ].merge(column, on="candidate_id", validate="one_to_one")
        else:
            base = base.merge(column, on="candidate_id", validate="one_to_one")
    if base is None:
        raise PhaseError("no model named for a paired fold frame")
    return base.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def fold_mean_auroc_draws(
    folds: Sequence[pd.DataFrame], columns: Sequence[str]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Mean-over-folds harm AUROC per score column, with document-clustered bootstrap draws.

    The transfer folds share their documents (the generator is held out, not the page), so one
    draw of documents -- with replacement, inside each environment -- serves every fold and every
    column at once. The draw is paired across columns and across folds. A pooled AUROC over the
    two folds would compare scores from two differently-fitted models on one scale, so the
    statistic is the mean of the per-fold AUROCs, which is what criterion B names.
    """

    def harm(frame: pd.DataFrame, column: str, index: np.ndarray | None = None) -> float:
        rows = frame if index is None else frame.iloc[index]
        return auroc(-rows[column].to_numpy(np.float64), rows["is_harmful"].to_numpy(bool))

    effect = {c: float(np.mean([harm(fold, c) for fold in folds])) for c in columns}
    environments = sorted(set().union(*(set(f["environment"].astype(str)) for f in folds)))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    plan = []
    for environment in environments:
        documents = sorted(
            set().union(
                *(
                    set(f.loc[f["environment"] == environment, "document_id"].astype(str))
                    for f in folds
                )
            )
        )
        index = [
            {
                document: np.flatnonzero(
                    (fold["environment"].to_numpy(str) == environment)
                    & (fold["document_id"].to_numpy(str) == document)
                )
                for document in documents
            }
            for fold in folds
        ]
        picks = generator.integers(0, len(documents), size=(BOOTSTRAP_RESAMPLES, len(documents)))
        plan.append((documents, index, picks))
    draws = {c: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for c in columns}
    for resample in range(BOOTSTRAP_RESAMPLES):
        chosen: list[list[np.ndarray]] = [[] for _ in folds]
        for documents, index, picks in plan:
            for position in picks[resample]:
                for f, _fold in enumerate(folds):
                    chosen[f].append(index[f][documents[position]])
        rows = [np.concatenate(c) if c else np.empty(0, dtype=np.int64) for c in chosen]
        for column in columns:
            draws[column][resample] = float(
                np.mean([harm(fold, column, rows[f]) for f, fold in enumerate(folds)])
            )
    return effect, draws


def _interval(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None, None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def sign_flip_diagnostic(population: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, Any]:
    """Which evidence separates exact from non-exact candidates, generator by generator.

    Standardized mean differences (exact minus the rest, in units of the pooled U5 deviation), for
    the frozen generator's candidates at its own choice sites and for the image corrector's
    candidates. A feature whose sign flips points the same way for one generator's good edits as
    for the other's bad ones, which no generator-agnostic scorer can use for both.
    """
    frame = population_frame(population, POP_U5).merge(
        matrix.drop(columns=["environment"]), on="candidate_id", validate="one_to_one"
    )
    columns = columns_without(())
    spread = frame[columns].std().replace(0.0, np.nan)
    c0 = frame[frame["corrector_source"] == C0]
    c1 = frame[frame["corrector_source"] == C1]
    size = c0.groupby("site_group")["exact"].agg(["size", "sum"])
    choice = size[(size["size"] >= 2) & (size["sum"] > 0) & (size["sum"] < size["size"])].index
    at_choice = c0[c0["site_group"].isin(choice)]

    def shift(block: pd.DataFrame) -> pd.Series:
        exact = block["exact"].to_numpy(bool)
        return (block[columns][exact].mean() - block[columns][~exact].mean()) / spread

    left = shift(at_choice)
    right = shift(c1)
    table = pd.DataFrame({"c0_choice_sites": left, "c1_all": right}).dropna()
    table["sign_flip"] = np.sign(table["c0_choice_sites"]) != np.sign(table["c1_all"])
    strongest = table.reindex(table["c0_choice_sites"].abs().sort_values(ascending=False).index)
    top = strongest.head(15)
    return {
        "c0_choice_sites": len(choice),
        "c0_candidates_at_choice_sites": len(at_choice),
        "c1_candidates": len(c1),
        "features_compared": len(table),
        "features_with_a_sign_flip": int(table["sign_flip"].sum()),
        "sign_flips_among_the_15_strongest_c0_features": int(top["sign_flip"].sum()),
        "strongest": [
            {
                "feature": str(name),
                "c0_choice_sites_smd": float(row["c0_choice_sites"]),
                "c1_smd": float(row["c1_all"]),
                "sign_flip": bool(row["sign_flip"]),
            }
            for name, row in top.iterrows()
        ],
        "unit": "standardized mean difference, exact minus non-exact, pooled U5 deviation",
        "is_a_test": False,
    }


def run_transfer() -> int:
    """RQ2 and criterion B: harm discrimination on a generator no training row came from."""
    started = time.monotonic()
    _require(DEPLOYMENT, "deployment")
    _forbid(TRANSFER)
    scores = pd.read_parquet(RANKING_SCORES)
    population = pd.read_parquet(RANKING_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    metrics = cc_read_json(RANKING_METRICS)["by_split"]
    trainability = cc_read_json(RANKING_METRICS)["trainability"]
    by_split = {
        split: {
            model: {
                key: metrics[split][model][V_FULL][key]
                for key in ("rows", "choice_sites", "top1", "harm_auroc", "benefit_auroc")
            }
            | {"trainable": trainability.get(f"{split}|{model}", {"trainable": True})["trainable"]}
            for model in (LOGO_MODELS if split in LOGO_SPLITS else (B1, PRIMARY_RANKER))
        }
        for split in (*LOGO_SPLITS, *LOSO_SPLITS)
    }

    def fold_mean(model: str, key: str) -> float | None:
        values = [by_split[split][model][key] for split in LOGO_SPLITS]
        return None if any(v is None for v in values) else float(np.mean(values))

    logo_mean = {
        model: {
            "harm_auroc": fold_mean(model, "harm_auroc"),
            "benefit_auroc": fold_mean(model, "benefit_auroc"),
        }
        for model in LOGO_MODELS
    }
    folds = [
        _paired_fold_frame(scores, population, split, (B1, PRIMARY_RANKER)) for split in LOGO_SPLITS
    ]
    effect, draws = fold_mean_auroc_draws(folds, (B1, PRIMARY_RANKER))
    low, high = _interval(draws[PRIMARY_RANKER])
    b1_low, b1_high = _interval(draws[B1])
    rl1_logo = cc_read_json(rl1.LEAVE_GENERATOR_OUT)["by_split"]
    rl1_strict = rl1_logo[rl1.SPLIT_B][rl1.M0][REPRESENTATION]["mean_harm_auroc"]
    rl1_document = rl1_logo[rl1.SPLIT_B_DOC][rl1.M0][REPRESENTATION]["mean_harm_auroc"]
    reproduces = bool(abs(effect[B1] - float(rl1_strict)) < 1e-12)
    if not reproduces:
        raise PhaseError(
            f"the DS1 score's transfer AUROC {effect[B1]} does not reproduce RL1's {rl1_strict}"
        )
    primary = effect[PRIMARY_RANKER]
    _write_json_once(
        TRANSFER,
        {
            **_analysis_envelope("generator_transfer"),
            "primary_rule": cc_read_json(DESIGN_RECORD)["transfer"]["primary"],
            "secondary_rule": cc_read_json(DESIGN_RECORD)["transfer"]["secondary"],
            "by_split": by_split,
            "logo_mean": logo_mean,
            "primary_ranker": {
                "model": PRIMARY_RANKER,
                "mean_logo_harm_auroc": primary,
                "ci_low": low,
                "ci_high": high,
                "floor": TRANSFER_HARM_FLOOR,
                "meets_the_floor": bool(primary > TRANSFER_HARM_FLOOR),
                "interval_clears_the_floor": bool(low is not None and low > TRANSFER_HARM_FLOOR),
            },
            "ds1_score": {
                "model": B1,
                "mean_logo_harm_auroc": effect[B1],
                "ci_low": b1_low,
                "ci_high": b1_high,
                "reproduces_rl1_strict_logo": reproduces,
                "rl1_strict_logo_mean_harm_auroc": rl1_strict,
                "rl1_document_held_out_logo_mean_harm_auroc": rl1_document,
            },
            "documents_are_shared_by_design": (
                "the strict split holds out the generator and its sites but not the page. SGV-RL1 "
                "measured the document-held-out variant for its own score and it was no higher, so "
                "the shared pages are not what keeps transfer below the floor"
            ),
            "untrainable": sorted(key for key, row in trainability.items() if not row["trainable"]),
            "sign_flip_diagnostic": sign_flip_diagnostic(population, matrix),
            "loso_caveat": cc_read_json(DESIGN_RECORD)["transfer"]["secondary"],
            "bootstrap": {
                "unit": "document, resampled within environment, one draw shared by both folds",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
        },
    )
    print(
        f"transfer: primary ranker mean LOGO harm AUROC {primary:.4f} [{low:.4f}, {high:.4f}] "
        f"vs DS1 {effect[B1]:.4f} ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ RQ3: feature contribution


def _variant_summary(metrics: dict[str, Any], deployment: dict[str, Any]) -> dict[str, Any]:
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    return {
        "top1": metrics["top1"],
        "ndcg_at_3": metrics["ndcg_at_k"]["3"],
        "harm_auroc": metrics["harm_auroc"],
        "benefit_auroc": metrics["benefit_auroc"],
        "repair_recall_at_primary_target": deployment[key]["repair_recall"],
        "selective_harm_at_primary_target": deployment[key]["selective_harm_rate"],
        "holds_primary_target": deployment[key]["holds_its_target"],
    }


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def source_classifier(
    frame: pd.DataFrame, matrix: pd.DataFrame, columns: Sequence[str]
) -> dict[str, Any]:
    """How well a representation identifies the generator, out of fold over RL1's document folds."""
    design = _design(frame, matrix, columns)
    target = (frame["corrector_source"].astype(str) == C1).to_numpy(dtype=bool)
    fold = frame["fold"].to_numpy(dtype=np.int64)
    predicted = np.full(len(frame), np.nan)
    for held in sorted(set(fold.tolist())):
        train = fold != held
        predicted[~train] = rl1.fit_and_score(rl1.M0, design[train], target[train], design[~train])
    return {
        "columns": len(columns),
        "source_auroc": auroc(predicted, target),
        "source_accuracy": float(((predicted >= 0.5) == target).mean()),
        "majority_class_accuracy": float(max(target.mean(), 1.0 - target.mean())),
    }


def run_ablate() -> int:
    """RQ3: family ablations, single-family rankers, and how much generator the evidence carries."""
    started = time.monotonic()
    _require(TRANSFER, "transfer")
    for path in (ABLATION, SINGLE_FAMILY, LEAKAGE):
        _forbid(path)
    metrics = cc_read_json(RANKING_METRICS)["by_split"][SPLIT_BLOCKS]
    deployment = cc_read_json(DEPLOYMENT)
    full = _variant_summary(metrics[PRIMARY_RANKER][V_FULL], deployment["by_model"][PRIMARY_RANKER])

    def variant(name: str) -> dict[str, Any]:
        row = _variant_summary(
            metrics[PRIMARY_RANKER][name], deployment["by_variant_of_the_primary_ranker"][name]
        )
        return {
            **row,
            "delta_vs_full": {
                key: _delta(row[key], full[key])
                for key in (
                    "top1",
                    "ndcg_at_3",
                    "harm_auroc",
                    "benefit_auroc",
                    "repair_recall_at_primary_target",
                )
            },
        }

    _write_json_once(
        ABLATION,
        {
            **_analysis_envelope("feature_ablation"),
            "model": PRIMARY_RANKER,
            "split": SPLIT_BLOCKS,
            "block": "test",
            "full": full,
            "ablations": {name: variant(name) for name in ABLATIONS},
            "dropped": {name: list(drop) for name, drop in ABLATIONS.items()},
            "not_tested": (
                "no ablation is in the statistical family; the deltas are descriptive, and at 46 "
                "choice sites a Top-1 difference of one site is 0.0217"
            ),
        },
    )
    _write_json_once(
        SINGLE_FAMILY,
        {
            **_analysis_envelope("single_family_models"),
            "model": PRIMARY_RANKER,
            "split": SPLIT_BLOCKS,
            "block": "test",
            "full": full,
            "families": {f"only_{f}": variant(f"only_{f}") for f in FAMILIES},
            "prefixes": {name: list(prefixes) for name, prefixes in FAMILY_PREFIXES.items()},
        },
    )
    population = pd.read_parquet(RANKING_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    frame = population_frame(population, POP_U5)
    base = columns_without(())
    classifiers = {
        "r3": source_classifier(frame, matrix, base),
        "r3_plus_relative": source_classifier(frame, matrix, [*base, *relative_columns()]),
        "relative_only": source_classifier(frame, matrix, relative_columns()),
    }
    rl1_leak = cc_read_json(rl1.IMPLICIT_SOURCE_LEAKAGE)["results"][REPRESENTATION]
    scores = pd.read_parquet(RANKING_SCORES)
    by_model = {}
    for model in MODELS:
        scored = scored_frame(scores, population, SPLIT_BLOCKS, model, V_FULL)
        by_model[model] = auroc(
            scored["rank_score"].to_numpy(np.float64),
            (scored["corrector_source"].astype(str) == C1).to_numpy(bool),
        )
    gap = {
        key: _delta(
            _variant_summary(metrics[B2][V_FULL], deployment["by_model"][B2])[key], full[key]
        )
        for key in ("top1", "harm_auroc", "benefit_auroc", "repair_recall_at_primary_target")
    }
    _write_json_once(
        LEAKAGE,
        {
            **_analysis_envelope("generator_leakage"),
            "target": f"corrector_source == {C1}",
            "population": POP_U5,
            "folds": "RL1's five content-blind document folds",
            "classifiers": classifiers,
            "rl1_r3_source_auroc": rl1_leak["source_auroc"],
            "r3_reproduces_rl1": bool(
                abs(classifiers["r3"]["source_auroc"] - float(rl1_leak["source_auroc"])) < 1e-12
            ),
            "score_as_a_generator_detector": by_model,
            "score_as_a_generator_detector_note": (
                "the AUROC with which each model's own rank score separates the image corrector's "
                "candidates from the frozen generator's on the DS1 test block; 0.5 means the "
                "score carries no generator ordering, and either direction away from it does"
            ),
            "generator_aware_minus_primary": gap,
            "reading": (
                "the evidence identifies the generator almost perfectly whether or not identity is "
                "a column, so 'no generator identity in the model' constrains the columns, not "
                "the information. The generator-aware reference is the upper bound on what "
                "naming the generator adds on known generators"
            ),
        },
    )
    print(
        f"ablate: R3 source AUROC {classifiers['r3']['source_auroc']:.4f}, "
        f"R3+relative {classifiers['r3_plus_relative']['source_auroc']:.4f} "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ statistics


def _top1_frame(scores: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Test choice sites with the primary ranker's hit as a and the DS1 score's hit as b."""
    rows = []
    frames = {
        model: scored_frame(scores, population, SPLIT_BLOCKS, model, V_FULL)
        for model in (PRIMARY_RANKER, B1)
    }
    hits: dict[str, dict[str, bool]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for model, frame in frames.items():
        hits[model] = {}
        for site, group in frame.groupby("site_group", sort=True):
            exact = group["exact"].to_numpy(dtype=bool)
            if len(group) >= 2 and exact.any() and not exact.all():
                hits[model][str(site)] = bool(_ordered(group)["exact"].iloc[0])
                meta[str(site)] = (
                    str(group["environment"].iloc[0]),
                    str(group["document_id"].iloc[0]),
                )
    if set(hits[PRIMARY_RANKER]) != set(hits[B1]):
        raise PhaseError("the two models do not share their choice sites")
    for site in sorted(hits[B1]):
        rows.append(
            {
                "environment": meta[site][0],
                "document_id": meta[site][1],
                "a": float(hits[PRIMARY_RANKER][site]),
                "b": float(hits[B1][site]),
            }
        )
    return pd.DataFrame(rows)


def _auroc_comparison(
    effect: dict[str, float], draws: dict[str, np.ndarray], folds: Sequence[pd.DataFrame]
) -> dict[str, Any]:
    from ocr_risk.stats.bootstrap import _bootstrap_p_value

    difference = draws[PRIMARY_RANKER] - draws[B1]
    finite = difference[np.isfinite(difference)]
    low, high = _interval(difference)
    per_environment: dict[str, float] = {}
    for environment in sorted(set().union(*(set(f["environment"].astype(str)) for f in folds))):
        values = []
        for fold in folds:
            rows = fold[fold["environment"] == environment]
            harmful = rows["is_harmful"].to_numpy(bool)
            values.append(
                auroc(-rows[PRIMARY_RANKER].to_numpy(np.float64), harmful)
                - auroc(-rows[B1].to_numpy(np.float64), harmful)
            )
        finite_values = [v for v in values if np.isfinite(v)]
        if finite_values:
            per_environment[environment] = float(np.mean(finite_values))
    return {
        "comparison": PRIMARY_FAMILY[3],
        "family": "primary",
        "effect": float(effect[PRIMARY_RANKER] - effect[B1]),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(_bootstrap_p_value(finite)),
        "environments_improved": sum(1 for v in per_environment.values() if v > 0),
        "environments_worsened": sum(1 for v in per_environment.values() if v < 0),
        "units": int(sum(len(f) for f in folds)),
        "pooled": False,
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def run_stats() -> int:
    """The frozen family of four, document-clustered and paired, Holm-corrected."""
    started = time.monotonic()
    _require(LEAKAGE, "ablate")
    _forbid(STATISTICAL_TESTS)
    scores = pd.read_parquet(RANKING_SCORES)
    population = pd.read_parquet(RANKING_POPULATION)
    errors = ds1.load_error_population()
    test_errors = errors[errors["block"] == "test"].reset_index(drop=True)
    mapping = ds1.site_to_errors(errors)
    accepted = {}
    thresholds = {}
    decision_sites = {}
    for model in (PRIMARY_RANKER, B1):
        decisions = decisions_for(scores, population, model, V_FULL)
        thresholds[model], accepted[model] = accepted_at(decisions, PRIMARY_EPSILON)
        decision_sites[model] = decisions["test"][["site_key", "environment", "document_id"]]
    if set(decision_sites[B1]["site_key"]) != set(decision_sites[PRIMARY_RANKER]["site_key"]):
        raise PhaseError("the two policies do not decide the same sites")
    sites = decision_sites[B1].sort_values("site_key", kind="stable")
    repaired = {
        model: ds1._repaired_flags(accepted[model], accepted[model], test_errors, mapping)
        for model in accepted
    }
    folds = [
        _paired_fold_frame(scores, population, split, (B1, PRIMARY_RANKER)) for split in LOGO_SPLITS
    ]
    effect, draws = fold_mean_auroc_draws(folds, (B1, PRIMARY_RANKER))
    family = {
        PRIMARY_FAMILY[0]: xr1.comparison(
            _top1_frame(scores, population), PRIMARY_FAMILY[0], "primary"
        ),
        PRIMARY_FAMILY[1]: xr1.comparison(
            ds1._recall_frame(test_errors, repaired[PRIMARY_RANKER], repaired[B1], "error_site"),
            PRIMARY_FAMILY[1],
            "primary",
        ),
        PRIMARY_FAMILY[2]: xr1.comparison(
            ds1._harm_frame(sites, accepted[PRIMARY_RANKER], accepted[B1]),
            PRIMARY_FAMILY[2],
            "primary",
        ),
        PRIMARY_FAMILY[3]: _auroc_comparison(effect, draws, folds),
    }
    adjusted = s15.holm(family)
    frames = {
        PRIMARY_FAMILY[0]: _top1_frame(scores, population),
        PRIMARY_FAMILY[1]: ds1._recall_frame(
            test_errors, repaired[PRIMARY_RANKER], repaired[B1], "error_site"
        ),
        PRIMARY_FAMILY[2]: ds1._harm_frame(sites, accepted[PRIMARY_RANKER], accepted[B1]),
    }
    sensitivity = s15.holm(
        {
            name: xr1.comparison(frame, name, "sensitivity", pooled=True)
            for name, frame in frames.items()
        }
    )
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "bootstrap": {
                "unit": "document, resampled within environment",
                "paired": True,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "alpha": ALPHA,
                "block": "test",
                "epsilon": PRIMARY_EPSILON,
                "thresholds": thresholds,
            },
            "multiplicity": "Holm-Bonferroni inside the frozen primary family",
            "primary_family": adjusted,
            "family_size": len(PRIMARY_FAMILY),
            "declared_before_any_endpoint": True,
            "comparator": B1,
            "model": PRIMARY_RANKER,
            "units": {
                PRIMARY_FAMILY[0]: "test-block choice site; a and b are Top-1 hits",
                PRIMARY_FAMILY[1]: "evaluable OCR error site in the test block",
                PRIMARY_FAMILY[2]: (
                    "decision site in the test block; a and b are 'accepted a harmful edit here'"
                ),
                PRIMARY_FAMILY[3]: (
                    "candidate rows of the two strict leave-one-generator-out folds; the effect is "
                    "the difference of the mean per-fold harm AUROCs"
                ),
            },
            "surviving": sorted(k for k, v in adjusted.items() if v["survives_holm"]),
            "effect_definition": (
                "the repository's default for this bootstrap: the unweighted mean over "
                "environments of each environment's rate difference (a minus b). P4's effect is "
                "the difference of mean per-fold AUROCs"
            ),
            "direction": {
                PRIMARY_FAMILY[0]: "positive favours the ranker",
                PRIMARY_FAMILY[1]: "positive favours the ranker",
                PRIMARY_FAMILY[2]: (
                    "negative favours the ranker; 'environments_improved' counts environments "
                    "where the ranker's joint harm is HIGHER, because the helper counts a - b > 0"
                ),
                PRIMARY_FAMILY[3]: "positive favours the ranker",
            },
            "pooled_sensitivity": sensitivity,
            "pooled_sensitivity_disclosure": (
                "added after the primary family was computed, because criterion A names a pooled "
                "share while the frozen family's statistic averages environments. It is reported "
                "beside the family and never replaces it"
            ),
        },
    )
    print(
        f"stats: {sum(1 for v in adjusted.values() if v['survives_holm'])}/{len(adjusted)} "
        f"survive Holm ({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ falsification

PERMUTATION_DRAWS = 5
PERMUTATION_SEED = 20260921
LABEL_FIELDS = frozenset(
    {"exact", "is_harmful", "beneficial", "outcome", "grade", "d_before", "d_after"}
)


def _module_tree() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def _functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _names_in(function_name: str) -> set[str]:
    """Every name a function mentions, following the calls it makes inside this module."""
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
    return names


def _called_names() -> set[str]:
    """Every name used outside the falsification phase, so its own string table cannot match."""
    names: set[str] = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run_negative":
            continue
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _threshold_block_argument() -> list[str]:
    """Which block's decisions `accepted_at` hands to the threshold chooser, read from the AST."""
    found: list[str] = []
    for node in ast.walk(_functions()["accepted_at"]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "choose_threshold"
        ):
            for argument in node.args:
                if isinstance(argument, ast.Subscript) and isinstance(argument.slice, ast.Constant):
                    found.append(str(argument.slice.value))
    return found


def permutation_control(population: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, Any]:
    """The primary ranker refitted on grades shuffled inside each training query.

    Shuffling inside the query keeps every query's grade mix -- which is all the LambdaMART loss
    sees of a query besides the features -- and destroys only the link between a candidate's
    evidence and its grade. If the ranker's test performance came from anything but that link,
    it would survive the shuffle.
    """
    split = next(s for s in build_splits(population) if s.name == SPLIT_BLOCKS)
    frame = population_frame(population, POP_U5)
    design = _design(frame, matrix, model_columns(PRIMARY_RANKER))
    grades = frame["grade"].to_numpy(dtype=np.int64)
    groups = frame["document_id"].astype(str).to_numpy()
    test = split.test[frame.iloc[split.test]["block"].to_numpy(str) == "test"]
    beneficial = frame["beneficial"].to_numpy(dtype=bool)[test]
    values = []
    for draw in range(PERMUTATION_DRAWS):
        generator = np.random.default_rng(PERMUTATION_SEED + draw)
        shuffled = grades[split.train].copy()
        train_groups = groups[split.train]
        for key in sorted(set(train_groups.tolist())):
            members = np.flatnonzero(train_groups == key)
            shuffled[members] = shuffled[members][generator.permutation(members.size)]
        fitted = fit_lambdamart(design[split.train], shuffled, train_groups)
        values.append(auroc(fitted.score(design[test]), beneficial))
    return {
        "draws": PERMUTATION_DRAWS,
        "seed": PERMUTATION_SEED,
        "benefit_auroc_by_draw": values,
        "mean_benefit_auroc": float(np.mean(values)),
        "near_random_band": [1.0 - rl1.NEAR_RANDOM, rl1.NEAR_RANDOM],
    }


def run_negative() -> int:
    """Twenty-six tests, each of which would fail if the claim it guards were false."""
    started = time.monotonic()
    _require(STATISTICAL_TESTS, "stats")
    _forbid(FALSIFICATION)
    population = pd.read_parquet(RANKING_POPULATION)
    matrix = pd.read_parquet(FEATURE_MATRIX)
    scores = pd.read_parquet(RANKING_SCORES)
    splits = cc_read_json(SPLIT_REGISTRY)
    training = cc_read_json(TRAINING_REGISTRY)["jobs"]
    deployment = cc_read_json(DEPLOYMENT)
    metrics = cc_read_json(RANKING_METRICS)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    agnostic_columns = {
        f"{model}|{variant}": model_columns(model, drop, keep)
        for model in GENERATOR_AGNOSTIC
        for variant, drop, keep in (
            (V_FULL, (), ()),
            *((name, drop, ()) for name, drop in ABLATIONS.items()),
            *((f"only_{f}", (), FAMILY_PREFIXES[f]) for f in FAMILIES),
        )
    }
    carrying = sorted(
        key
        for key, columns in agnostic_columns.items()
        if any(c.startswith(rl1.FAM_SOURCE) for c in columns)
    )
    record("t01_no_generator_identity_column_in_any_agnostic_model", not carrying, carrying)
    with_source = sorted({job["model"] for job in training if job["carries_source_columns"]})
    record(
        "t02_only_the_generator_aware_reference_carries_identity",
        with_source == [B2],
        with_source,
    )
    design_columns = set(matrix.columns) - {"candidate_id", "environment"}
    record(
        "t03_no_label_or_oracle_field_is_a_design_matrix_column",
        not (design_columns & LABEL_FIELDS),
        sorted(design_columns & LABEL_FIELDS),
    )
    u5 = population_frame(population, POP_U5)
    block_of = dict(zip(u5["candidate_id"], u5["block"], strict=True))
    ds1_rows = scores[scores["split"] == SPLIT_BLOCKS]
    scored_fit = int(ds1_rows["candidate_id"].map(block_of).eq("fit").sum())
    record("t04_the_fit_block_is_never_scored", scored_fit == 0, {"fit_rows_scored": scored_fit})
    record(
        "t05_thresholds_are_chosen_on_the_threshold_block_only",
        _threshold_block_argument() == ["threshold"],
        _threshold_block_argument(),
    )
    split_rows = splits["splits"]
    record(
        "t06_no_site_crosses_any_split",
        bool(splits["no_site_crosses_any_split"]),
        {row["split"]: row["shared_sites"] for row in split_rows},
    )
    record(
        "t07_no_document_crosses_the_ds1_blocks",
        int(splits["documents_crossing_the_ds1_blocks"]) == 0,
        splits["documents_crossing_the_ds1_blocks"],
    )
    held_rows = [row for row in split_rows if row["held_out"] is not None]
    record(
        "t08_every_transfer_split_trains_without_its_held_out_source",
        all(
            row["held_out"] not in row["train_sources"] and row["test_sources"] == [row["held_out"]]
            for row in held_rows
        ),
        {row["split"]: [row["train_sources"], row["test_sources"]] for row in held_rows},
    )
    frozen = pd.read_parquet(rl1.CANDIDATE_POPULATION)
    frozen = frozen[frozen["population"] == rl1.PRIMARY_POPULATION]
    record(
        "t09_the_primary_population_is_rl1s_exactly",
        set(u5["candidate_id"]) == set(frozen["candidate_id"]) and len(u5) == len(frozen),
        {"rows": len(u5)},
    )
    reproduction = cc_read_json(FEATURE_REPRODUCTION)
    record(
        "t10_the_features_reproduce_rl1s_matrix_exactly",
        bool(reproduction["identical_on_rl1_rows"]),
        reproduction["max_absolute_difference_on_rl1_rows"],
    )
    record(
        "t11_the_ds1_score_reproduces_exactly",
        bool(cc_read_json(MODEL_REGISTRY)["ds1_reproduction"]["identical"]),
        cc_read_json(MODEL_REGISTRY)["ds1_reproduction"],
    )
    published = cc_read_json(ds1.DECISION)["operating_point"]
    ours = deployment["by_model"][B1][rl1.epsilon_key(PRIMARY_EPSILON)]
    fields = ("repair_recall", "selective_harm_rate", "coverage", "joint_harm_rate")
    record(
        "t12_the_ds1_score_reproduces_ds1s_published_operating_point",
        all(abs(float(ours[f]) - float(published[f])) < 1e-12 for f in fields),
        {f: [ours[f], published[f]] for f in fields},
    )
    transfer = cc_read_json(TRANSFER)
    record(
        "t13_the_ds1_score_reproduces_rl1s_transfer_auroc",
        bool(transfer["ds1_score"]["reproduces_rl1_strict_logo"]),
        transfer["ds1_score"]["mean_logo_harm_auroc"],
    )
    leakage = cc_read_json(LEAKAGE)
    record(
        "t14_the_source_classifier_reproduces_rl1s",
        bool(leakage["r3_reproduces_rl1"]),
        [leakage["classifiers"]["r3"]["source_auroc"], leakage["rl1_r3_source_auroc"]],
    )
    duplicated = {
        model: int(
            decisions_for(scores, population, model, V_FULL)["test"]["site_key"].duplicated().sum()
        )
        for model in MODELS
    }
    record(
        "t15_every_policy_makes_exactly_one_decision_per_site",
        not any(duplicated.values()),
        duplicated,
    )
    regraded = relevance(population)
    record(
        "t16_relevance_grades_follow_the_frozen_rule",
        bool(
            np.array_equal(regraded, population["grade"].to_numpy(dtype=np.int64))
            and not (population["exact"] & population["is_harmful"]).any()
            and set(population.loc[population["exact"], "grade"]) == {REL_EXACT}
            and set(population.loc[population["is_harmful"], "grade"]) == {REL_HARMFUL}
        ),
        {str(k): int(v) for k, v in population["grade"].value_counts().sort_index().items()},
    )
    # A hand-built query set whose first feature orders the grades exactly. A LambdaMART whose
    # gradient sign or pair orientation were wrong would rank it backwards.
    toy_x = np.tile(np.arange(4.0), 30).reshape(-1, 1)
    toy_grade = np.tile(np.array([0, 1, 2, 3], dtype=np.int64), 30)
    toy_group = np.repeat(np.arange(30), 4).astype(str)
    toy = fit_lambdamart(toy_x, toy_grade, toy_group).score(np.arange(4.0).reshape(-1, 1))
    record(
        "t17_lambdamart_learns_a_hand_built_ordering",
        bool(np.all(np.diff(toy) > 0)),
        toy.tolist(),
    )
    # DCG of grades [3, 0, 1] at cutoff 3: (2^3-1)/log2(2) + 0 + (2^1-1)/log2(4) = 7 + 0.5.
    record(
        "t18_dcg_matches_a_hand_computed_value",
        abs(_dcg(np.array([3, 0, 1]), 3) - 7.5) < 1e-12,
        _dcg(np.array([3, 0, 1]), 3),
    )
    hand = pd.DataFrame(
        {
            "site_group": ["s1", "s1", "s2", "s2", "s3"],
            "candidate_id": ["a", "b", "c", "d", "e"],
            "rank_score": [0.9, 0.1, 0.2, 0.8, 0.5],
            "safety": [0.9, 0.1, 0.2, 0.8, 0.5],
            "exact": [True, False, True, False, True],
            "beneficial": [True, False, True, False, True],
            "is_harmful": [False, True, False, True, False],
            "grade": [3, 0, 3, 0, 3],
        }
    )
    computed = ranking_metrics(hand)
    record(
        "t19_top1_and_recall_match_a_hand_computed_frame",
        computed["top1"] == 0.5
        and computed["choice_sites"] == 2
        and computed["recall_at_k"]["3"] == 1.0,
        {"top1": computed["top1"], "choice_sites": computed["choice_sites"]},
    )
    untrainable = metrics["by_split"]["logo_c0_frozen_generator"][LM_SITE][V_FULL]
    record(
        "t20_an_untrainable_ranker_is_reported_undefined_not_chance",
        untrainable["constant_score"]
        and untrainable["top1"] is None
        and untrainable["harm_auroc"] is None,
        {"trainable": metrics["trainability"][f"logo_c0_frozen_generator|{LM_SITE}"]},
    )
    epsilon_leak = (_names_in("score_model") | _names_in("fit_lambdamart")) & {
        "EPSILONS",
        "PRIMARY_EPSILON",
        "epsilon_key",
        "choose_threshold",
    }
    record(
        "t21_no_risk_target_reaches_a_fitted_ranker",
        not epsilon_leak,
        sorted(epsilon_leak),
    )
    forbidden = sorted(
        _called_names()
        & {"select_threshold", "RiskController", "Calibrator", "clopper_pearson", "conformal"}
    )
    record("t22_no_certification_machinery_is_invoked", not forbidden, forbidden)
    generated = set(pd.read_parquet(hy1.CANDIDATES)["candidate_id"].astype(str))
    novel = sorted(set(population["candidate_id"].astype(str)) - generated)
    record(
        "t23_every_candidate_is_one_hy1_already_generated",
        not novel,
        {"candidates": int(population["candidate_id"].nunique()), "not_in_hy1": len(novel)},
    )
    evaluated = scored_frame(scores, population, SPLIT_BLOCKS, PRIMARY_RANKER, V_FULL)
    record(
        "t24_ds1_split_metrics_read_the_test_block_only",
        set(evaluated["block"]) == {"test"},
        sorted(set(evaluated["block"])),
    )
    design_issued = cc_read_json(DESIGN_RECORD)["issued_utc"]
    record(
        "t25_the_design_record_predates_every_endpoint",
        all(
            design_issued <= cc_read_json(path)["issued_utc"]
            for path in (RANKING_METRICS, DEPLOYMENT, TRANSFER, STATISTICAL_TESTS)
        ),
        design_issued,
    )
    control = permutation_control(population, matrix)
    low, high = control["near_random_band"]
    record(
        "t26_a_ranker_fitted_on_shuffled_grades_is_near_random",
        bool(low <= control["mean_benefit_auroc"] <= high),
        control,
    )
    passed = sum(1 for test in tests if test["passed"])
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
    failing = [test["test"] for test in tests if not test["passed"]]
    print(
        f"negative: {passed}/{len(tests)} pass ({time.monotonic() - started:.0f}s)"
        + (f", failing {failing}" if failing else "")
    )
    if failing:
        raise PhaseError(f"{len(failing)} falsification tests failed: {failing}")
    return 0


# ------------------------------------------------------------------ the decision


def criteria_from_artifacts() -> dict[str, Any]:
    """Every success criterion, re-derived from the persisted artifacts and nothing else."""
    metrics = cc_read_json(RANKING_METRICS)["by_split"][SPLIT_BLOCKS]
    deployment = cc_read_json(DEPLOYMENT)["by_model"]
    transfer = cc_read_json(TRANSFER)["primary_ranker"]
    family = cc_read_json(STATISTICAL_TESTS)["primary_family"]
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    top1 = float(metrics[PRIMARY_RANKER][V_FULL]["top1"])
    reference = float(metrics[B1][V_FULL]["top1"])
    ratio = _ratio(top1, reference)
    ranker = deployment[PRIMARY_RANKER][key]
    ds1_point = deployment[B1][key]
    a = bool(ratio >= 1.0 + TOP1_RELATIVE_GAIN)
    b = bool(transfer["meets_the_floor"])
    c = bool(
        ranker["repair_recall"] > ds1_point["repair_recall"]
        and ranker["holds_its_target"]
        and ds1_point["holds_its_target"]
    )
    p1 = family[PRIMARY_FAMILY[0]]
    p2 = family[PRIMARY_FAMILY[1]]
    supported = {
        "A": bool(p1["effect"] > 0 and p1["survives_holm"]),
        "B": bool(transfer["interval_clears_the_floor"]),
        "C": bool(p2["effect"] > 0 and p2["survives_holm"]),
    }
    return {
        "criteria": {"A": a, "B": b, "C": c},
        "supported": supported,
        "top1": {"ranker": top1, "ds1": reference, "ratio": ratio},
        "operating_points": {PRIMARY_RANKER: ranker, B1: ds1_point},
        "transfer": transfer,
        "family": family,
    }


def outcome_label(outcome: str, criteria: dict[str, bool], supported: dict[str, bool]) -> str:
    parts = []
    for name in ("A", "B", "C"):
        if criteria[name]:
            parts.append(
                f"{name} met ({'statistically supported' if supported[name] else 'NOT supported'})"
            )
        else:
            parts.append(f"{name} not met")
    return f"{outcome}: {OUTCOME_TAXONOMY[outcome]} -- " + "; ".join(parts)


def run_decide() -> int:
    """The frozen outcome rule, applied to the persisted artifacts."""
    started = time.monotonic()
    _require(FALSIFICATION, "negative")
    _forbid(DECISION)
    state = criteria_from_artifacts()
    criteria = state["criteria"]
    supported = state["supported"]
    outcome = assign_outcome(criteria)
    negative = cc_read_json(FALSIFICATION)
    headroom = cc_read_json(HEADROOM)["by_block"]["test"]["by_target"][
        rl1.epsilon_key(PRIMARY_EPSILON)
    ]
    deployment = cc_read_json(DEPLOYMENT)["by_model"]
    targets = {
        model: {
            key: {
                "holds": deployment[model][key]["holds_its_target"],
                "abstains": deployment[model][key]["abstains"],
            }
            for key in (rl1.epsilon_key(e) for e in EPSILONS)
        }
        for model in (B1, PRIMARY_RANKER)
    }
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "primary_ranker": PRIMARY_RANKER,
            "comparator": B1,
            "block": "test",
            "primary_risk_target": PRIMARY_EPSILON,
            "top1": state["top1"],
            "operating_points": state["operating_points"],
            "every_target_by_model": targets,
            "every_target_holds_or_abstains": {
                model: all(row["holds"] for row in rows.values()) for model, rows in targets.items()
            },
            "transfer": state["transfer"],
            "headroom_measured_before_any_ranker": {
                "share_closed_by_perfect_ranking": headroom["share_closed_by_perfect_ranking"],
                "share_closed_by_perfect_gate": headroom["share_closed_by_perfect_gate"],
            },
            "criteria_thresholds": {
                "top1_relative_gain": TOP1_RELATIVE_GAIN,
                "transfer_harm_floor": TRANSFER_HARM_FLOOR,
                "harm_tolerance": ds1.HARM_TOLERANCE,
            },
            "criterion_A": criteria["A"],
            "criterion_B": criteria["B"],
            "criterion_C": criteria["C"],
            "statistically_supported": supported,
            "unmet_criteria": sorted(name for name, value in criteria.items() if not value),
            "outcome": outcome,
            "outcome_label": outcome_label(outcome, criteria, supported),
            "statistical_family_surviving": cc_read_json(STATISTICAL_TESTS)["surviving"],
            "production_ready": False,
            "certified": False,
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "falsification_tests_passed": int(negative["passed"]),
            "falsification_tests_total": int(negative["total"]),
            "recommended_next_stage": NEXT_STAGE[outcome],
            "hypothesis_id": HYPOTHESIS,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {outcome_label(outcome, criteria, supported)}")
    return 0


# ------------------------------------------------------------------ figures

FIGURE_NOTE = (
    "ANALYSIS ONLY -- development evidence computed with ground truth; not a production system"
)
SURFACE = rl1.SURFACE
INK = xc1.INK
INK_SECONDARY = xc1.INK_SECONDARY
GRID = xc1.GRID
MODEL_COLOURS = {
    B0: "#b0b0b0",
    B1: "#8a8a8a",
    B2: "#c9a227",
    RN: "#8e6cc0",
    LM_SITE: "#6fa8dc",
    LM_DOC: "#2a78d6",
    LM_REL: "#1baf7a",
}
MODEL_SHORT = {
    B0: "B0 random",
    B1: "B1 DS1 score",
    B2: "B2 generator-aware",
    RN: "RankNet",
    LM_SITE: "LambdaMART site",
    LM_DOC: "LambdaMART document",
    LM_REL: "LambdaMART doc + relative",
}
FIGURES = (
    "ranking_quality.png",
    "risk_coverage.png",
    "generator_transfer.png",
    "feature_contribution.png",
    "accepted_composition.png",
)


def run_figures() -> int:
    """Five figures, minimal academic style, every value read from a persisted artifact."""
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
            "svg.hashsalt": "rk1",
        }
    )
    manifest: dict[str, Any] = {}

    def source_key(path: Path) -> str:
        stage = FIGURE_DIR.parent
        return str(path.relative_to(stage)) if path.is_relative_to(stage) else _relative(path)

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
            "sources": {source_key(s): xr1._signature(s) for s in sources},
        }

    metrics = cc_read_json(RANKING_METRICS)["by_split"][SPLIT_BLOCKS]

    # Figure 1 -- ranking quality on the test block.
    panels = (
        ("top1", "Top-1 on choice sites"),
        ("ndcg3", "NDCG@3"),
        ("benefit_auroc", "benefit AUROC"),
        ("harm_auroc", "harm AUROC"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.9), sharey=True)
    for ax, (field, title) in zip(axes, panels, strict=True):
        values = []
        for model in MODELS:
            row = metrics[model][V_FULL]
            values.append(row["ndcg_at_k"]["3"] if field == "ndcg3" else row[field])
        positions = np.arange(len(MODELS))
        ax.barh(
            positions,
            [np.nan if v is None else v for v in values],
            color=[MODEL_COLOURS[m] for m in MODELS],
            height=0.65,
        )
        ax.set_yticks(positions, [MODEL_SHORT[m] for m in MODELS])
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=8.5)
    axes[0].invert_yaxis()  # the axes share y, so inverting each one would cancel out
    fig.suptitle("Ranking quality, DS1 test block", fontsize=9.5)
    save(fig, FIGURES[0], [RANKING_METRICS], "ranking quality of every model on the test block")

    # Figure 2 -- repair recall against realized harm at every risk target.
    deployment = cc_read_json(DEPLOYMENT)["by_model"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    markers = dict(zip(EPSILONS, ("s", "o", "^"), strict=True))
    for model in (B1, B2, RN, LM_DOC, LM_REL):
        for epsilon in EPSILONS:
            point = deployment[model][rl1.epsilon_key(epsilon)]
            if point["abstains"]:
                continue
            ax.scatter(
                point["selective_harm_rate"],
                point["repair_recall"],
                marker=markers[epsilon],
                s=22,
                color=MODEL_COLOURS[model],
                edgecolors=INK if not point["holds_its_target"] else "none",
                linewidths=0.9,
                label=MODEL_SHORT[model] if epsilon == PRIMARY_EPSILON else None,
            )
    for epsilon in EPSILONS:
        ax.axvline(epsilon, color=GRID, linewidth=0.8, linestyle="--")
        ax.plot(
            [],
            [],
            marker=markers[epsilon],
            linestyle="none",
            color=INK_SECONDARY,
            label=f"target {epsilon}",
        )
    ax.set_xlabel("realized selective harm on the test block")
    ax.set_ylabel("exact-repair recall")
    ax.set_title("Deployment at each risk target (outlined: target missed on test)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.3)
    save(fig, FIGURES[1], [DEPLOYMENT], "repair recall against realized harm per risk target")

    # Figure 3 -- strict leave-one-generator-out harm AUROC.
    transfer = cc_read_json(TRANSFER)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    width = 0.13
    shown = list(LOGO_MODELS)
    for index, model in enumerate(shown):
        values = [transfer["by_split"][split][model]["harm_auroc"] for split in LOGO_SPLITS]
        ax.bar(
            np.arange(len(LOGO_SPLITS)) + (index - len(shown) / 2) * width + width / 2,
            [np.nan if v is None else v for v in values],
            width=width,
            color=MODEL_COLOURS[model],
            label=MODEL_SHORT[model],
        )
    ax.axhline(TRANSFER_HARM_FLOOR, color=INK, linewidth=0.9, linestyle="--")
    ax.axhline(0.5, color=INK_SECONDARY, linewidth=0.6, linestyle=":")
    ax.set_xticks(
        np.arange(len(LOGO_SPLITS)),
        [f"held out: {SOURCE_LABEL[g].split(':')[0]}" for g in DISTINCT_GENERATORS],
    )
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("harm AUROC on the unseen generator")
    ax.set_title("Generator transfer (dashed: criterion B floor)", fontsize=9)
    ax.legend(frameon=False, fontsize=5.8, ncol=2)
    save(fig, FIGURES[2], [TRANSFER], "strict leave-one-generator-out harm AUROC")

    # Figure 4 -- feature contribution: ablations and single families against the full ranker.
    ablation = cc_read_json(ABLATION)
    single = cc_read_json(SINGLE_FAMILY)
    rows = [("full", ablation["full"])]
    rows += [(name, ablation["ablations"][name]) for name in ABLATIONS]
    rows += [(name, single["families"][name]) for name in single["families"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    positions = np.arange(len(rows))
    for ax, field, title in (
        (axes[0], "top1", "Top-1 on choice sites"),
        (axes[1], "repair_recall_at_primary_target", f"recall at harm <= {PRIMARY_EPSILON}"),
    ):
        ax.barh(
            positions,
            [np.nan if r[field] is None else r[field] for _n, r in rows],
            color=[MODEL_COLOURS[LM_DOC] if n == "full" else INK_SECONDARY for n, _r in rows],
            height=0.6,
        )
        ax.set_yticks(positions, [n for n, _r in rows])
        ax.set_title(title, fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle("Feature contribution to the document-level ranker", fontsize=9.5)
    save(fig, FIGURES[3], [ABLATION, SINGLE_FAMILY], "family ablations and single-family rankers")

    # Figure 5 -- what the gate accepts at the primary target.
    key = rl1.epsilon_key(PRIMARY_EPSILON)
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    for index, model in enumerate((B1, LM_DOC)):
        point = deployment[model][key]
        exact = float(point["accepted_exact_share"] or 0.0)
        harm = float(point["selective_harm_rate"] or 0.0)
        neutral = float(point["accepted_neutral_share"] or 0.0)
        partial = float(point["accepted_partial_share"] or 0.0)
        left = 0.0
        for value, colour, label in (
            (exact, "#1baf7a", "exact repair"),
            (partial, "#6fa8dc", "partial improvement"),
            (neutral, "#d0d0d0", "neutral change"),
            (harm, "#eb6834", "harmful"),
        ):
            ax.barh(
                index,
                value,
                left=left,
                color=colour,
                height=0.55,
                label=label if index == 0 else None,
            )
            left += value
        ax.text(
            1.01,
            index,
            f"accepted {point['accepted']}",
            va="center",
            fontsize=6.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks([0, 1], [MODEL_SHORT[B1], MODEL_SHORT[LM_DOC]])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"share of accepted edits at harm <= {PRIMARY_EPSILON}, test block")
    ax.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    save(fig, FIGURES[4], [DEPLOYMENT], "composition of the accepted edits at the primary target")

    missing = [name for name in FIGURES if name not in manifest]
    if missing:
        raise PhaseError(f"{len(missing)} figures were not produced: {missing}")
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_analysis_envelope("figure_manifest"),
            "figures": manifest,
            "count": len(manifest),
            "note": FIGURE_NOTE,
            "synthetic": False,
            "style": "matplotlib only, minimal academic",
            "every_value_read_from_an_artifact": True,
        },
    )
    print(f"figures: {len(manifest)} written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ determinism and record

DERIVED_OUTPUTS = (
    "HEADROOM",
    "RANKING_POPULATION",
    "POPULATION_INVENTORY",
    "FEATURE_MATRIX",
    "FEATURE_REPRODUCTION",
    "SPLIT_REGISTRY",
    "RANKING_SCORES",
    "MODEL_REGISTRY",
    "TRAINING_REGISTRY",
    "RANKING_METRICS",
    "DISCRIMINATION",
    "DEPLOYMENT",
    "TRANSFER",
    "ABLATION",
    "SINGLE_FAMILY",
    "LEAKAGE",
    "STATISTICAL_TESTS",
    "FALSIFICATION",
    "DECISION",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
)


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("headroom", run_headroom),
        ("population", run_population),
        ("features", run_features),
        ("splits", run_splits),
        ("rank", run_rank),
        ("metrics", run_metrics),
        ("deployment", run_deployment),
        ("transfer", run_transfer),
        ("ablate", run_ablate),
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
    """Every derived phase, re-run from the frozen upstream inputs into a directory of its own."""
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
    """Two independent regenerations from the frozen upstream inputs."""
    started = time.monotonic()
    _require(FIGURE_MANIFEST, "figures")
    _forbid(DETERMINISM)
    published = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        name for name in published if any(run.get(name) != published[name] for run in runs)
    )
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
    "1. Motivation": (RESEARCH_FREEZE, FROZEN_CONFIGURATION, HEADROOM),
    "2. Why Ranking Instead of Classification": (DESIGN_RECORD, HEADROOM, POPULATION_INVENTORY),
    "3. Generator-Agnostic Feature Space": (
        DESIGN_RECORD,
        FEATURE_REPRODUCTION,
        POPULATION_INVENTORY,
        LEAKAGE,
    ),
    "4. Learning-to-Rank Methodology": (
        DESIGN_RECORD,
        MODEL_REGISTRY,
        TRAINING_REGISTRY,
        SPLIT_REGISTRY,
        RANKING_METRICS,
    ),
    "5. Ranking Results": (
        RANKING_METRICS,
        DISCRIMINATION,
        STATISTICAL_TESTS,
        DECISION,
    ),
    "6. Cross-Generator Transfer": (TRANSFER, SPLIT_REGISTRY, STATISTICAL_TESTS, RANKING_METRICS),
    "7. Feature Contribution Analysis": (ABLATION, SINGLE_FAMILY, LEAKAGE),
    "8. Deployment Impact": (DEPLOYMENT, STATISTICAL_TESTS, HEADROOM, DECISION),
    "9. Limitations": (
        DESIGN_RECORD,
        SPLIT_REGISTRY,
        RANKING_METRICS,
        DEPLOYMENT,
        TRANSFER,
        STATISTICAL_TESTS,
        FALSIFICATION,
        DECISION,
    ),
    "10. Next Research Direction": (DECISION, TRANSFER, DEPLOYMENT, HEADROOM),
}


def run_record() -> int:
    """Provenance, the upstream re-hash, the decision re-derivation and the traceability index."""
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
        path for path, sha in freeze["upstream_sha256"].items() if file_sha256(REPO / path) != sha
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
                again[key] == published[key]
                for key in published
                if key not in ("issued_utc", "runtime_seconds")
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
            "regenerates_an_ocr_output": False,
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
            "sections": {
                name: [_relative(p) for p in paths] for name, paths in REPORT_SECTIONS.items()
            },
            "audit": audit,
            "untraceable": audit["untraceable"],
            "rule": (
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says"
            ),
        },
    )
    print(
        f"record: {sum(1 for p in PRODUCED if p.is_file())} artifacts, "
        f"report claims {audit['numeric_claims']}, untraceable "
        f"{audit['untraceable_numeric_claims']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    phases: dict[str, Callable[[], int]] = {
        "reconstruct": run_reconstruct,
        "freeze": run_freeze,
        "preregister": run_preregister,
        "headroom": run_headroom,
        "population": run_population,
        "features": run_features,
        "splits": run_splits,
        "rank": run_rank,
        "metrics": run_metrics,
        "deployment": run_deployment,
        "transfer": run_transfer,
        "ablate": run_ablate,
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

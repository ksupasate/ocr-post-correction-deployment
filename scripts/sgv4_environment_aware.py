#!/usr/bin/env python3
"""SGV4 development phase 1: can the system tell whether its risk model applies here?

Three stages have now failed in three different ways. SGV1 phase 6 found that the harm
ranking does not survive an unseen engine and that a certified threshold missed its nominal
bound on one engine of four. SGV2 found that engine shift is easy to detect, that the shift
score does not predict harm within the held-out engine, and that rejecting anomalous rows
costs repairs without buying safety. SGV3 found that per-sample uncertainty -- aleatoric,
epistemic and decision -- carries no signal that survives an engine change, and that split
conformal loses its marginal validity on exactly the engines it was meant to protect.

All three asked a question about a *row*: is this edit risky, is this row anomalous, is this
prediction uncertain. This stage asks a question about a *batch*:

    SGV4-E1: an applicability estimate computed from unlabeled aggregate statistics of a
    batch of OCR output -- never from engine identity -- improves bounded-risk correction on
    an unseen engine relative to the confidence-only, harm-aware, shift-aware and
    uncertainty-aware policies.

**The hypothesis is recorded as SGV4-E1, not H1.** `docs/sgv1/protocol.md` binds SGV1-H1
through SGV1-H4 and says a frozen ID is never reused for a different claim; SGV2 and SGV3
restarted their own families for the same reason. The brief for this stage names its
hypothesis H1, which is already taken.

Six things decide what the numbers below can mean, and all six are settled here rather than
after the measurement.

**1. An environment-constant penalty cannot move a ranking endpoint.** The primary endpoint
is repair recall under a harm bound, which is a property of `argsort(-score)`. If
applicability takes one value across a whole evaluation fold then `U - gamma * (1 - a)` is
`U - constant` and its argsort is the argsort of `U`: the proposed policy would be provably
identical to the harm-aware baseline, whatever gamma is. The method can therefore act
through exactly two channels -- WITHIN-fold variation of applicability across environments,
which reorders; and the deployed threshold, where even a constant shift changes how much is
accepted. Both are measured, and the within-fold dispersion of applicability is reported per
fold as the precondition for the first. A stage that did not check this could report a null
that was arithmetic rather than empirical.

**2. "Environment" has to have more than one instance or Test 1 is not a measurement.** With
one held-out engine per fold there is exactly one environment nobody has seen, and a
correlation over four points is not evidence. An environment here is an (engine, role,
document-shard) cell under a seeded, recorded partition of the document split, which gives
26 shards on TRAIN and
8 each on CALIBRATION and DEVELOPMENT -- 78 fit environments, 24 calibration environments
and 8 unseen environments per fold. Shards partition documents WITHIN a role, so an
environment never straddles the fit/calibrate/evaluate boundary.

**3. Applicability targets a property of the model, not of the sample.** The quantity being
predicted is the risk model's own error on an environment -- its Brier score there, the gap
between the harm it predicts and the harm it gets, whether its source-calibrated threshold
holds -- never the harm label of a row. That is the whole difference from SGV3, and if the
distinction collapses in the implementation the stage measures SGV3 again under a new name.
`tests/leakage` asserts no applicability estimator reads a row label.

**4. Detecting the environment is not the claim.** SGV2 already measured engine separation
at AUC 0.867-0.999, so a high applicability AUROC re-confirms a number in hand and settles
nothing. It is reported as a demoted diagnostic. The load-bearing version of Test 1 is the
correlation between applicability and risk-model reliability computed WITHIN the seen
environments and WITHIN the unseen environments separately: if applicability only tracks
engine identity, both within-group correlations vanish while the pooled one stays large.

**5. Conservatism is not safety, and it is not free.** An arm that abstains everywhere has a
harm rate of zero and repairs nothing. Every cell reports coverage and repair recall beside
the harm rate, and ablation E holds the average penalty fixed while removing its variation,
so "the improvement was just conservatism" is a measurable alternative rather than a caveat.

**6. The environment grouping is induced by engine and shard, and that is a limitation, not
a leak.** No estimator receives an engine label as a feature, at fit time or at inference;
`tests/leakage` permutes the engine column and asserts every applicability score is
unchanged. But which rows form one environment is determined by (engine, shard), so engine
identity is implicit in the *grouping* even though it is absent from the *representation*.
An operator has this too -- a batch is a batch of pages from one pipeline -- so the
assumption is realistic, but it is stated in the limitations rather than buried.

    --environments  stage 1: the environment partition and its descriptors
    --scores        the row-level score table every later stage reads
    --applicability stage 2 and critical test 1: does applicability track reliability?
    --curves        the primary endpoint: risk-coverage, in-domain and cross-engine
    --transfer      critical tests 2 and 3: per-engine transfer and the deployed threshold
    --ablation      the five required ablations
    --figures       the required figures
    --decide        the machine-readable finding
    --record        provenance for every artifact

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; neither axis is relaxed anywhere.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
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
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.calibration import brier_score, expected_calibration_error
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.risk.controller import CONTROLLERS, select_threshold
from ocr_risk.stats.bootstrap import cluster_bootstrap_indices

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv4/environment_aware"
ENVIRONMENT_SPEC = OUT / "environment_embeddings.json"
APPLICABILITY_RESULTS = OUT / "applicability_scores.json"
CURVE_RESULTS = OUT / "risk_coverage_results.json"
TRANSFER_RESULTS = OUT / "transfer_results.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
DECISION = OUT / "research_decision.json"
FIT_RECORD = OUT / "fit_record.json"
SCORES = OUT / "environment_aware_scores.parquet"
SELECTION_RECORD = OUT / "selection_record.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# The primary endpoint has exactly one implementation in this repository and it is Phase 5's.
# Phase 6, SGV2 and SGV3 imported it rather than restating it; so does this stage, and
# `tests/leakage` asserts all five names are the same function object.
achievable_repair_recall = rl.achievable_repair_recall
harm_at_matched_coverage = rl.harm_at_matched_coverage
harm_at_matched_repair_recall = rl.harm_at_matched_repair_recall
build_fold = rl.build_fold
Fold = rl.Fold

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
ECE_BINS = dg.ECE_BINS
BINNINGS = dg.BINNINGS
MATCHED_COVERAGES = rl.MATCHED_COVERAGES

# One environment is about this many documents. Fixed as a DOCUMENT count rather than a
# shard count so that an environment on TRAIN and an environment on DEVELOPMENT carry the
# same sampling noise in their descriptors -- a descriptor estimated from 55 pages and one
# estimated from 17 are not the same measurement, and a density fitted on the first and
# evaluated on the second would be reading that difference as shift.
SHARD_DOCUMENTS = 17
SHARD_SEED = 20260902
# Sensitivity: the whole environment analysis is re-run at these document counts and the
# headline correlations are reported at all three, because a partition granularity chosen
# once is a free parameter until it is shown not to matter.
SHARD_SENSITIVITY = (9, 17, 33)

# Each environment is also split into V disjoint document views. Two views of one
# environment give the only handle on "how much does this descriptor move for reasons that
# are not the environment", which is what the contrastive metric needs and what the
# descriptor stability diagnostic reports.
ENV_VIEWS = 2

LAMBDA_GRID = policy.LAMBDAS
# SYMMETRIC about zero, for the reason SGV2's kappa grid and SGV3's lambda2 grid are: the
# hypothesis says low applicability should be penalised, so a non-negative grid could only
# ever express the direction the hypothesis predicts and a reversal would be unreportable.
GAMMA_GRID = (-1.0, -0.5, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50, 1.00)

KNN_NEIGHBOURS = 5
RETRIEVAL_NEIGHBOURS = 5
PCA_COMPONENTS = 10
CLUSTER_COUNT = 6
CONTRASTIVE_COMPONENTS = 8
COVARIANCE_RIDGE = rl.MAHALANOBIS_RIDGE
SCATTER_RIDGE = 1e-2

# Method A is distance-based, method B is a learned self-supervised embedding, method C is
# retrieval against environments whose reliability has actually been measured. `random` is
# the negative control required by ablation D and is fitted through the identical pipeline,
# so a non-null result from it would indict the pipeline rather than the data.
DISTANCE_METHODS = ("mahalanobis", "knn", "kde")
EMBEDDING_METHODS = ("pca_reconstruction", "cluster", "contrastive")
RETRIEVAL_METHODS = ("retrieval",)
APPLICABILITY_METHODS = DISTANCE_METHODS + EMBEDDING_METHODS + RETRIEVAL_METHODS
CONTROL_METHODS = ("random",)
ORACLE_METHODS = ("oracle_engine", "oracle_reliability")
ALL_METHODS = APPLICABILITY_METHODS + CONTROL_METHODS + ORACLE_METHODS

CROSS_ENGINE = sa.CROSS_ENGINE
IN_DOMAIN = sa.IN_DOMAIN
SOURCE_CALIBRATION = sa.SOURCE_CALIBRATION

BASELINE_ARMS = (
    "no_correction",
    "confidence_only",
    "harm_only",
    "harm_aware",
    "shift_aware",
    "selfaware",
)
PROPOSED_ARM = "env_aware"
# The baselines a positive finding has to beat. `no_correction` is excluded because beating
# "change nothing" is not a claim about reliability, and `confidence_only` is kept because
# the brief names it V1 even though preceding stages measured it at zero repair recall.
SUBSTANTIVE_BASELINES = ("confidence_only", "harm_only", "harm_aware", "shift_aware", "selfaware")
CEILING_ARMS = ("oracle_engine_applicability", "oracle_reliability_applicability")

ABLATION_LABELS = {
    "ablate_environment": "A -- remove the environment representation",
    "ablate_uncertainty_swap": "B -- replace applicability with SGV3's uncertainty composite",
    "oracle_engine_applicability": "C -- engine identity as an upper bound, never deployable",
    "env_random": "D -- random environment embedding, negative control",
    "env_constant": "E -- same average penalty, no variation: conservatism without signal",
}


class PhaseError(RuntimeError):
    """A freeze, role, split, environment or selection invariant failed."""


# ------------------------------------------------------------------ stage 1: environments


@dataclass(frozen=True, slots=True)
class EnvSignalSpec:
    """One coordinate of the environment descriptor, with where it comes from and why.

    The brief requires every feature source to be documented. Holding the documentation in
    the object that computes the number is the only arrangement in which the two cannot
    drift apart: `--environments` writes this table straight into the artifact, so the
    published description of a coordinate is the description the code used.
    """

    name: str
    group: str
    """`process` (how the engine behaves), `structure` (how the page is laid out), or
    `candidate` (how the generator responded)."""
    statistic: str
    source: str
    definition: str
    rationale: str


def _spec(
    name: str, group: str, statistic: str, source: str, definition: str, rationale: str
) -> EnvSignalSpec:
    return EnvSignalSpec(
        name=name,
        group=group,
        statistic=statistic,
        source=source,
        definition=definition,
        rationale=rationale,
    )


# --- group A: how the OCR process behaves -------------------------------------------------
_PROCESS_SIGNALS = (
    _spec(
        "proc_conf_mean",
        "process",
        "masked_mean:conf_normalized:conf_missing",
        "conf_normalized, averaged over rows the engine actually scored",
        "mean recognizer confidence on the engine's own normalized scale, over rows where "
        "conf_missing is 0",
        "The confidence distribution is the most direct summary of how the recognizer is "
        "behaving. Rows without a confidence are excluded rather than filled, because "
        "data-provenance forbids a sentinel that would be indistinguishable from a real "
        "measurement; how often they occur is a separate coordinate.",
    ),
    _spec(
        "proc_conf_sd",
        "process",
        "masked_sd:conf_normalized:conf_missing",
        "conf_normalized, spread over rows the engine actually scored",
        "standard deviation of recognizer confidence over the same rows",
        "An engine that is uniformly unsure and one that is sharply split are different "
        "environments with the same mean.",
    ),
    _spec(
        "proc_conf_p10",
        "process",
        "masked_p10:conf_normalized:conf_missing",
        "conf_normalized, lower tail",
        "10th percentile of recognizer confidence",
        "The lower tail is where correction sites come from, so it moves for reasons the "
        "mean does not.",
    ),
    _spec(
        "proc_conf_missing_share",
        "process",
        "share:conf_missing",
        "conf_missing indicator",
        "fraction of rows for which the engine reported no recognition confidence",
        "Whether an engine reports confidence at all is a property of the engine, and it is "
        "the coordinate most likely to separate environments for a reason that has nothing "
        "to do with how hard the pages are.",
    ),
    _spec(
        "proc_conf_spans_mean",
        "process",
        "mean:conf_n_spans",
        "conf_n_spans",
        "mean number of engine spans the correction site draws its confidence from",
        "A site backed by one span and a site backed by six are different granularities of "
        "the same engine output, which is the tokenisation behaviour the brief asks for.",
    ),
    _spec(
        "proc_token_len_mean",
        "process",
        "mean:text_len_original",
        "text_len_original",
        "mean character length of the OCR token at the correction site",
        "Token length distribution: engines that over-segment produce short tokens and "
        "engines that under-segment produce long ones.",
    ),
    _spec(
        "proc_token_len_sd",
        "process",
        "sd:text_len_original",
        "text_len_original",
        "standard deviation of OCR token length",
        "Separates a uniform tokenisation from a mixed one at the same mean.",
    ),
    _spec(
        "proc_digit_mean",
        "process",
        "mean:text_digits_original",
        "text_digits_original",
        "mean count of digits in the OCR token",
        "The digit/letter mix of what the engine emitted is part of its output signature and "
        "is also the part of the corpus where harm is most expensive.",
    ),
    _spec(
        "proc_punct_mean",
        "process",
        "mean:text_punct_original",
        "text_punct_original",
        "mean count of punctuation characters in the OCR token",
        "Punctuation handling differs sharply between engines and is a whitespace/formatting "
        "behaviour rather than a content property.",
    ),
    _spec(
        "proc_line_tokens_mean",
        "process",
        "mean:ctx_line_tokens",
        "ctx_line_tokens",
        "mean number of tokens on the line the site sits in",
        "Line-level token count is the whitespace behaviour of the engine's segmenter, "
        "measured without needing the pixels.",
    ),
    _spec(
        "proc_whitespace_delta_mean",
        "process",
        "mean:ctx_token_count_delta",
        "ctx_token_count_delta",
        "mean change in token count the proposed edit would cause",
        "Splits and merges are the whitespace errors; how often the generator proposes to "
        "undo one is a direct read on the engine's segmentation behaviour.",
    ),
    _spec(
        "proc_op_substitution_share",
        "process",
        "share:prov_operation_substitution",
        "prov_operation_substitution",
        "fraction of sites whose proposed operation is a substitution",
        "The operation mix is the engine's error-shape signature: substitution-heavy output "
        "is a recognition problem, split/merge-heavy output is a segmentation problem.",
    ),
    _spec(
        "proc_op_insertion_share",
        "process",
        "share:prov_operation_insertion",
        "prov_operation_insertion",
        "fraction of sites whose proposed operation is an insertion",
        "The insertion/deletion/substitution ratio the brief names, split into its parts so "
        "the ratio can be recovered but the components are not forced into one number.",
    ),
    _spec(
        "proc_op_deletion_share",
        "process",
        "share:prov_operation_deletion",
        "prov_operation_deletion",
        "fraction of sites whose proposed operation is a deletion",
        "See proc_op_insertion_share.",
    ),
    _spec(
        "proc_op_merge_share",
        "process",
        "share:prov_operation_merge",
        "prov_operation_merge",
        "fraction of sites whose proposed operation is a merge",
        "Merge and split are the two segmentation operations and separate an over-segmenting "
        "engine from an under-segmenting one.",
    ),
    _spec(
        "proc_op_split_share",
        "process",
        "share:prov_operation_split",
        "prov_operation_split",
        "fraction of sites whose proposed operation is a split",
        "See proc_op_merge_share.",
    ),
    _spec(
        "proc_error_confusion_share",
        "process",
        "class_share:character_confusion",
        "SGV2's error class, imported rather than reimplemented",
        "fraction of sites classified as a same-length confusable-character substitution",
        "This is the confusion-matrix coordinate the brief asks for, at the resolution this "
        "corpus supports: a full character confusion matrix over ~230 rows per environment "
        "would be mostly zeros and would describe the sample, not the engine.",
    ),
    _spec(
        "proc_error_layout_share",
        "process",
        "class_share:layout",
        "SGV2's error class",
        "fraction of sites classified as a layout error (gap insertion or segmentation)",
        "Layout errors are produced by the detector rather than the recognizer, so their "
        "share separates two different failure sources inside one engine.",
    ),
    _spec(
        "proc_error_numeric_share",
        "process",
        "class_share:numeric",
        "SGV2's error class",
        "fraction of sites classified as numeric or price",
        "Numeric fields carry the harm the receipt corpus is about; their share is a "
        "property of what the engine chose to emit as much as of the page.",
    ),
    _spec(
        "proc_error_language_share",
        "process",
        "class_share:language_model",
        "SGV2's error class",
        "fraction of sites classified as a same-length or unclassified lexical error",
        "The residual class. Reported because a descriptor set whose shares do not sum to "
        "one is hiding a category.",
    ),
    _spec(
        "proc_error_formatting_share",
        "process",
        "class_share:formatting_other",
        "SGV2's error class",
        "fraction of sites classified as a formatting error",
        "Formatting behaviour differs between engines independently of accuracy.",
    ),
    _spec(
        "proc_error_datetime_share",
        "process",
        "class_share:datetime",
        "SGV2's error class",
        "fraction of sites whose text matches a date or time pattern",
        "Dates are the second structured field type in this corpus and behave differently "
        "from prices under every engine measured so far.",
    ),
)

# --- group B: how the page is laid out ----------------------------------------------------
_STRUCTURE_SIGNALS = (
    _spec(
        "struct_box_width_mean",
        "structure",
        "mean:geom_width",
        "geom_width",
        "mean detected box width",
        "Bounding-box geometry is the detector's output and is the layout statistic that "
        "needs no pixels to compute.",
    ),
    _spec(
        "struct_box_height_mean",
        "structure",
        "mean:geom_height",
        "geom_height",
        "mean detected box height",
        "See struct_box_width_mean.",
    ),
    _spec(
        "struct_aspect_mean",
        "structure",
        "mean:geom_aspect",
        "geom_aspect",
        "mean box aspect ratio",
        "Aspect ratio separates a line detector from a word detector without knowing which "
        "engine produced either.",
    ),
    _spec(
        "struct_aspect_sd",
        "structure",
        "sd:geom_aspect",
        "geom_aspect",
        "standard deviation of box aspect ratio",
        "A detector with a consistent box shape and one with a variable shape are different "
        "environments at the same mean.",
    ),
    _spec(
        "struct_rel_x_mean",
        "structure",
        "mean:geom_rel_x",
        "geom_rel_x",
        "mean horizontal position of the box on the page, in page units",
        "Where on the page correction sites fall is a layout property of the corpus and the "
        "detector jointly.",
    ),
    _spec(
        "struct_rel_y_mean",
        "structure",
        "mean:geom_rel_y",
        "geom_rel_y",
        "mean vertical position of the box on the page",
        "See struct_rel_x_mean.",
    ),
    _spec(
        "struct_rel_y_sd",
        "structure",
        "sd:geom_rel_y",
        "geom_rel_y",
        "standard deviation of vertical box position",
        "The spatial-consistency coordinate: sites concentrated in one band and sites spread "
        "down the page are different layouts.",
    ),
    _spec(
        "struct_rel_area_mean",
        "structure",
        "mean:geom_rel_area",
        "geom_rel_area",
        "mean fraction of the page covered by the box",
        "Text density, measured through the detector rather than the raster.",
    ),
    _spec(
        "struct_geom_missing_share",
        "structure",
        "share:geom_missing",
        "geom_missing indicator",
        "fraction of rows for which no box was recoverable",
        "An engine that does not localise everything it reads is a different environment "
        "from one that does, and the difference is invisible in the text.",
    ),
    _spec(
        "struct_region_chars_mean",
        "structure",
        "mean:prov_region_chars",
        "prov_region_chars",
        "mean number of characters in the region the site sits in",
        "Region size is the field-structure coordinate: a dense tabular region and a sparse "
        "header region give different numbers.",
    ),
    _spec(
        "struct_page_support_mean",
        "structure",
        "mean:ctx_page_support_original",
        "ctx_page_support_original",
        "mean number of times the OCR token recurs on its own page",
        "Repetition across a page is a structural property of receipts and forms, and it is "
        "the evidence the page-support features were built on.",
    ),
    _spec(
        "struct_sites_per_document",
        "structure",
        "sites_per_document",
        "row count divided by document count in the environment",
        "mean number of correction sites the environment produced per page",
        "How many sites a pipeline raises per page is the single coordinate that most "
        "directly describes 'how much work is there here', and it is not recoverable from "
        "any per-row average.",
    ),
)

# --- group C: how the candidate generator responded ----------------------------------------
_CANDIDATE_SIGNALS = (
    _spec(
        "cand_per_site_mean",
        "candidate",
        "mean:prov_site_candidate_count",
        "prov_site_candidate_count",
        "mean number of candidates generated at the site",
        "How many repairs the generator could think of is the most direct measure of "
        "candidate diversity.",
    ),
    _spec(
        "cand_generation_entropy",
        "candidate",
        "candidate_count_entropy",
        "prov_site_candidate_count, as a distribution over the environment",
        "Shannon entropy (natural log) of the distribution of candidates-per-site",
        "The brief's candidate generation entropy. An environment where every site gets "
        "exactly three candidates and one where the count ranges over 1-8 have the same mean "
        "and different entropy, and the second is the one where the generator is reacting to "
        "the page rather than to its own template.",
    ),
    _spec(
        "cand_generator_rank_mean",
        "candidate",
        "mean:prov_generator_rank",
        "prov_generator_rank",
        "mean rank of the candidate within its site's generated list",
        "Rank distribution says whether the pool is dominated by first choices or reaches "
        "deep into the tail.",
    ),
    _spec(
        "cand_generator_score_mean",
        "candidate",
        "mean:prov_generator_score",
        "prov_generator_score",
        "mean generator score attached to the candidate",
        "The generator's own confidence, which is a different instrument from the engine's.",
    ),
    _spec(
        "cand_generator_score_missing_share",
        "candidate",
        "share:prov_generator_score_missing",
        "prov_generator_score_missing indicator",
        "fraction of candidates carrying no generator score",
        "Which generator produced the candidate determines whether a score exists at all, so "
        "this coordinate is a read on the generator mix.",
    ),
    _spec(
        "cand_suspicion_mean",
        "candidate",
        "mean:prov_suspicion_score",
        "prov_suspicion_score",
        "mean suspicion score assigned to the site by the detector",
        "The site detector's own opinion, aggregated. It is upstream of the risk model, so "
        "it describes the environment the risk model was handed.",
    ),
    _spec(
        "cand_source_g3_share",
        "candidate",
        "share:prov_source_g3_edit_aware",
        "prov_source_g3_edit_aware",
        "fraction of candidates from the edit-aware generator",
        "Generator mix. Two environments with the same candidate counts but different "
        "generator shares are different populations of proposed edits.",
    ),
    _spec(
        "cand_source_g7_share",
        "candidate",
        "share:prov_source_g7_structural_v2",
        "prov_source_g7_structural_v2",
        "fraction of candidates from the structural generator",
        "See cand_source_g3_share.",
    ),
    _spec(
        "cand_edit_distance_mean",
        "candidate",
        "mean:text_edit_distance",
        "text_edit_distance",
        "mean raw character edit distance between the OCR token and the candidate",
        "The edit distance distribution the brief asks for, in raw characters as "
        "`.claude/rules/metrics.md` requires.",
    ),
    _spec(
        "cand_edit_distance_sd",
        "candidate",
        "sd:text_edit_distance",
        "text_edit_distance",
        "standard deviation of raw character edit distance",
        "Distinguishes an environment of uniformly small repairs from one mixing small and "
        "large, which is the shape that matters for harm.",
    ),
    _spec(
        "cand_normalized_distance_mean",
        "candidate",
        "mean:text_normalized_distance",
        "text_normalized_distance",
        "mean length-normalized edit distance",
        "Reported beside the raw distance because a long-token environment and a "
        "large-edit environment are not the same thing.",
    ),
    _spec(
        "cand_changed_fraction_mean",
        "candidate",
        "mean:edit_changed_fraction",
        "edit_changed_fraction",
        "mean fraction of the token the proposed edit would change",
        "How invasive the average proposal is, which is the quantity harm is most sensitive "
        "to in every preceding stage.",
    ),
    _spec(
        "cand_confusable_fraction_mean",
        "candidate",
        "mean:edit_confusable_fraction",
        "edit_confusable_fraction",
        "mean fraction of substitutions that are between confusable characters",
        "Separates repairs the generator proposed because the glyphs look alike from repairs "
        "it proposed for lexical reasons.",
    ),
    _spec(
        "cand_lm_delta_mean",
        "candidate",
        "mean:plaus_lm_delta",
        "plaus_lm_delta",
        "mean language-model log-probability gain of the candidate over the OCR token",
        "The language-model disagreement coordinate: how much the LM prefers the proposal.",
    ),
    _spec(
        "cand_lm_delta_sd",
        "candidate",
        "sd:plaus_lm_delta",
        "plaus_lm_delta",
        "standard deviation of the language-model gain",
        "An environment where the LM is uniformly mildly in favour and one where it is "
        "violently split are different, and the mean cannot tell them apart.",
    ),
    _spec(
        "cand_lexicon_gain_mean",
        "candidate",
        "mean:plaus_lexicon_gain",
        "plaus_lexicon_gain",
        "mean lexicon-membership gain of the candidate over the OCR token",
        "The lexicon is a TRAIN-frozen resource, so its gain distribution says how far the "
        "environment's vocabulary sits from the one the system was built on.",
    ),
)

ENV_SIGNALS: tuple[EnvSignalSpec, ...] = _PROCESS_SIGNALS + _STRUCTURE_SIGNALS + _CANDIDATE_SIGNALS
ENV_SIGNAL_NAMES = tuple(s.name for s in ENV_SIGNALS)
ENV_GROUPS = ("process", "structure", "candidate")


def _seeded_permutation(items: list[str], salt: str) -> list[str]:
    """A deterministic shuffle that depends only on the item names and the salt.

    Not `rng.permutation(len(items))`: that would make the partition depend on the ORDER the
    caller happened to pass, so a future change to how documents are listed would silently
    repartition every environment while the seed stayed the same. Sorting by a keyed digest
    of the name makes the partition a function of the corpus, which is what a recorded,
    reproducible partition has to be.
    """
    keyed = sorted(
        items,
        key=lambda name: hashlib.blake2b(
            f"{salt}|{name}".encode(), digest_size=16, key=str(SHARD_SEED).encode()
        ).digest(),
    )
    return keyed


def shard_partition(design: dg.Design, shard_documents: int = SHARD_DOCUMENTS) -> pd.DataFrame:
    """Assign every document to a shard and a view, within its role.

    Shards partition documents WITHIN a role, never across one, so an environment cannot
    straddle the fit/calibrate/evaluate boundary -- which would put the same page on both
    sides of a density model.
    """
    frame = design.meta[["document_id", "role"]].drop_duplicates().reset_index(drop=True)
    if frame["document_id"].duplicated().any():
        raise PhaseError("a document appears under two roles; the role split is not a partition")
    rows: list[dict[str, Any]] = []
    for role, group in frame.groupby("role", sort=True):
        documents = _seeded_permutation(sorted(group["document_id"].astype(str)), str(role))
        n_shards = max(2, round(len(documents) / shard_documents))
        for position, document in enumerate(documents):
            shard = position * n_shards // len(documents)
            rows.append(
                {
                    "document_id": document,
                    "role": str(role),
                    "shard": int(shard),
                    "view": position % ENV_VIEWS,
                }
            )
    return pd.DataFrame(rows).sort_values("document_id").reset_index(drop=True)


@dataclass(slots=True)
class DescriptorContext:
    """Everything `describe` needs, resolved once so no coordinate re-derives it per call."""

    matrix: np.ndarray
    index_of: dict[str, int]
    error_class: np.ndarray
    documents: np.ndarray

    def column(self, name: str) -> np.ndarray:
        return self.matrix[:, self.index_of[name]]


def build_context(design: dg.Design) -> DescriptorContext:
    return DescriptorContext(
        matrix=design.matrix,
        index_of={name: i for i, name in enumerate(design.names)},
        error_class=rl.error_classes(design)[1],
        documents=design.documents,
    )


def describe(context: DescriptorContext, rows: np.ndarray) -> np.ndarray:
    """The environment descriptor for one set of rows. NaN where a coordinate is undefined.

    Ground-truth-blind by construction: `context` carries the design matrix, the frozen
    error class and the document ids, and nothing else. `tests/leakage` asserts that no name
    bound in this function's globals reaches a label array.
    """
    out = np.full(len(ENV_SIGNALS), np.nan)
    if rows.size == 0:
        return out
    for position, signal in enumerate(ENV_SIGNALS):
        kind, _, argument = signal.statistic.partition(":")
        if kind == "sites_per_document":
            n_documents = len(set(context.documents[rows].tolist()))
            out[position] = float(rows.size) / max(n_documents, 1)
            continue
        if kind == "candidate_count_entropy":
            counts = np.rint(context.column("prov_site_candidate_count")[rows]).astype(int)
            _, frequency = np.unique(counts, return_counts=True)
            share = frequency / frequency.sum()
            out[position] = float(-np.sum(share * np.log(share)))
            continue
        if kind == "class_share":
            out[position] = float(np.mean(context.error_class[rows] == argument))
            continue
        if kind.startswith("masked_"):
            column, mask_column = argument.split(":")
            keep = rows[context.column(mask_column)[rows] <= 0.5]
            if keep.size == 0:
                continue
            values = context.column(column)[keep]
            statistic = kind[len("masked_") :]
        else:
            values = context.column(argument)[rows]
            statistic = kind
        if statistic == "mean" or statistic == "share":
            out[position] = float(values.mean())
        elif statistic == "sd":
            out[position] = float(values.std())
        elif statistic == "p10":
            out[position] = float(np.percentile(values, 10))
        elif statistic == "p90":
            out[position] = float(np.percentile(values, 90))
        else:  # pragma: no cover - the spec table is closed and tested
            raise PhaseError(f"unknown descriptor statistic {signal.statistic!r}")
    return out


@dataclass(slots=True)
class EnvironmentTable:
    """Every environment in the corpus, its descriptor, and the rows it owns.

    One object for all four folds. Building it per fold would make it impossible to assert
    that two folds scored byte-identical descriptors, which is the property that lets the
    cross-fold comparison mean anything.
    """

    ids: tuple[str, ...]
    descriptors: np.ndarray
    view_ids: tuple[str, ...]
    view_of: tuple[str, ...]
    view_descriptors: np.ndarray
    engine: np.ndarray
    role: np.ndarray
    shard: np.ndarray
    n_rows: np.ndarray
    n_documents: np.ndarray
    row_environment: np.ndarray
    """Environment id for every design row, in design order."""
    partition: pd.DataFrame
    shard_documents: int

    @property
    def position(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.ids)}

    def mask(
        self, *, role: str | None = None, engines: tuple[str, ...] | None = None
    ) -> np.ndarray:
        keep = np.ones(len(self.ids), dtype=bool)
        if role is not None:
            keep &= self.role == role
        if engines is not None:
            keep &= np.isin(self.engine, list(engines))
        return keep


def _assemble(
    context: DescriptorContext,
    row_environment: np.ndarray,
    view_index: np.ndarray,
    partition: pd.DataFrame,
    shard_documents: int,
    minimum_rows: int,
) -> EnvironmentTable:
    """Turn a row -> environment assignment into descriptors, once, for any grouping.

    Both the corpus-wide environment table and the per-fold one go through this, so a fold's
    descriptors are computed by the same code as the artifact's and cannot drift from it.
    """
    documents = context.documents
    order = sorted({e for e in row_environment.tolist() if e})
    kept: list[str] = []
    descriptors: list[np.ndarray] = []
    meta: list[tuple[str, str, int, int, int]] = []
    view_ids: list[str] = []
    view_parent: list[str] = []
    view_descriptors: list[np.ndarray] = []
    for identifier in order:
        rows = np.flatnonzero(row_environment == identifier)
        if rows.size < minimum_rows:
            continue
        engine, group, shard = identifier.split("|")[:3]
        kept.append(identifier)
        descriptors.append(describe(context, rows))
        meta.append((engine, group, int(shard), int(rows.size), len(set(documents[rows].tolist()))))
        for view in range(ENV_VIEWS):
            view_rows = rows[view_index[rows] == view]
            if view_rows.size < max(minimum_rows // ENV_VIEWS, 1):
                continue
            view_ids.append(f"{identifier}|v{view}")
            view_parent.append(identifier)
            view_descriptors.append(describe(context, view_rows))

    if not kept:
        raise PhaseError("no environment reached the minimum row count")
    width = len(ENV_SIGNALS)
    return EnvironmentTable(
        ids=tuple(kept),
        descriptors=np.vstack(descriptors),
        view_ids=tuple(view_ids),
        view_of=tuple(view_parent),
        view_descriptors=np.vstack(view_descriptors) if view_ids else np.empty((0, width)),
        engine=np.array([m[0] for m in meta]),
        role=np.array([m[1] for m in meta]),
        shard=np.array([m[2] for m in meta]),
        n_rows=np.array([m[3] for m in meta]),
        n_documents=np.array([m[4] for m in meta]),
        row_environment=row_environment,
        partition=partition,
        shard_documents=shard_documents,
    )


def _shard_and_view(design: dg.Design, partition: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lookup = partition.set_index("document_id")
    shard_of = lookup["shard"].to_dict()
    view_of = lookup["view"].to_dict()
    documents = design.documents
    return (
        np.array([shard_of[d] for d in documents], dtype=int),
        np.array([view_of[d] for d in documents], dtype=int),
    )


def build_environments(
    design: dg.Design,
    context: DescriptorContext,
    shard_documents: int = SHARD_DOCUMENTS,
    *,
    minimum_rows: int = 20,
) -> EnvironmentTable:
    """The corpus-wide environment table: (engine, role, shard), descriptors over all rows."""
    partition = shard_partition(design, shard_documents)
    shard_index, view_index = _shard_and_view(design, partition)
    engines = design.meta["engine_id"].to_numpy(str)
    roles = design.meta["role"].to_numpy(str)
    row_environment = np.array(
        [f"{e}|{r}|{s:02d}" for e, r, s in zip(engines, roles, shard_index, strict=True)]
    )
    return _assemble(context, row_environment, view_index, partition, shard_documents, minimum_rows)


# The four row blocks a fold owns, and the tag each contributes to its environment ids. The
# tag is what keeps an inner fold honest: its reference half and its evaluation half are both
# CALIBRATION rows of the same shard, so without it they would collapse into one environment
# and the percentile reference would be fitted on the pages it is about to score.
FOLD_BLOCKS = (("fit", "fit"), ("source_cal", "cal"), ("seen_eval", "seen"), ("eval", "eval"))


def build_fold_environments(
    design: dg.Design,
    context: DescriptorContext,
    partition: pd.DataFrame,
    fold: Fold,
    *,
    minimum_rows: int = 20,
) -> EnvironmentTable:
    """Environments for one fold, each descriptor computed on that block's rows alone."""
    shard_index, view_index = _shard_and_view(design, partition)
    engines = design.meta["engine_id"].to_numpy(str)
    row_environment = np.full(len(design.meta), "", dtype=object)
    for attribute, tag in FOLD_BLOCKS:
        rows = getattr(fold, attribute)
        row_environment[rows] = [f"{engines[r]}|{tag}|{shard_index[r]:02d}" for r in rows.tolist()]
    return _assemble(
        context,
        np.array(row_environment, dtype=object),
        view_index,
        partition,
        SHARD_DOCUMENTS,
        minimum_rows,
    )


def block_ids(
    env: EnvironmentTable, tag: str, engines: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    keep = env.role == tag
    if engines is not None:
        keep &= np.isin(env.engine, list(engines))
    return tuple(np.array(env.ids)[keep].tolist())


# ------------------------------------------------------------------ stage 2: applicability

# What "reliability" means for the retrieval method and for the reliability oracle. Declared
# once, recorded in the artifact, and deliberately NOT the harm label: it is the gap between
# the harm the risk model predicts on an environment and the harm it actually gets there, so
# a perfect predictor of it is a predictor of the model's own error, not of the outcome.
RETRIEVAL_TARGET = "abs_risk_error"


@dataclass(slots=True)
class Applicability:
    """Applicability for every environment in the table, plus what produced it.

    Precomputed for all environments rather than exposing a `transform`, because two of the
    entries -- the two oracles -- are not functions of the descriptor at all and a common
    interface that pretended otherwise would hide exactly the thing that makes them ceilings.
    """

    method: str
    values: dict[str, float]
    raw: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def of(self, environment_ids: np.ndarray) -> np.ndarray:
        return np.array([self.values.get(str(e), np.nan) for e in environment_ids])


def _standardizer(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill value, centre and scale, all estimated on the FIT environments alone."""
    fill = np.nanmedian(block, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    filled = np.where(np.isfinite(block), block, fill)
    centre = filled.mean(axis=0)
    scale = filled.std(axis=0)
    return fill, centre, np.where(scale < 1e-12, 1.0, scale)


def _prepare(
    block: np.ndarray, fill: np.ndarray, centre: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.where(np.isfinite(block), block, fill) - centre) / scale


def _content_order(block: np.ndarray) -> np.ndarray:
    """Row order determined by the descriptor values, never by the environment's name.

    `_assemble` lists environments in alphabetical id order, so renaming the engines
    reorders the fit block. Means, covariances and sorted distances do not care; a k-means
    fit and a PCA fit do, because their iterations start from the rows they were handed.
    Sorting by content before those two fits makes every applicability score invariant to
    what the engines happen to be called, which is the property `tests/leakage` asserts by
    permuting the names through a bijection.
    """
    return np.lexsort(block.T[::-1])


def _mahalanobis(block: np.ndarray, centre: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    delta = block - centre
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inverse, delta), 0.0))


def _random_descriptors(ids: tuple[str, ...], width: int) -> np.ndarray:
    """Ablation D's descriptors: noise keyed by the environment name.

    Keyed by name rather than drawn in table order so that the control is the same noise
    whichever fold asks for it -- a control that changed between folds would be four
    different controls reported as one.
    """
    out = np.empty((len(ids), width))
    for position, identifier in enumerate(ids):
        digest = hashlib.blake2b(identifier.encode(), digest_size=8, key=b"sgv4-random").digest()
        out[position] = np.random.default_rng(int.from_bytes(digest, "big")).normal(size=width)
    return out


def fit_applicability(
    env: EnvironmentTable,
    method: str,
    *,
    fit_ids: tuple[str, ...],
    calibration_ids: tuple[str, ...],
    fit_engines: tuple[str, ...],
    reliability: dict[str, float],
    oracle_reliability: dict[str, float] | None = None,
) -> Applicability:
    """One applicability estimator, fitted on fit environments and frozen.

    Every estimator sees exactly two things: the descriptors of the fit environments, and --
    for the retrieval method only -- the measured reliability of the CALIBRATION environments
    of the fit engines, which is label information the split plan already permits at that
    role. No estimator sees the held-out engine, and none sees a row label.

    `reliability` and `oracle_reliability` are separate arguments rather than one dictionary
    the caller is trusted to have filtered. The second contains the held-out engine's own
    measured error and only `oracle_reliability` reads it, so the restriction is structural:
    a future edit that let the retrieval method reach held-out information would have to name
    the other argument to do it.
    """
    position = env.position
    descriptors = env.descriptors
    if method == "random":
        descriptors = _random_descriptors(env.ids, descriptors.shape[1])
    fit_rows = np.array([position[i] for i in fit_ids])
    fill, centre, scale = _standardizer(descriptors[fit_rows])
    prepared = _prepare(descriptors, fill, centre, scale)
    reference_block = prepared[fit_rows]
    diagnostics: dict[str, Any] = {"n_fit_environments": int(fit_rows.size)}

    if method in ("mahalanobis", "random"):
        covariance = np.cov(reference_block, rowvar=False) + COVARIANCE_RIDGE * np.eye(
            prepared.shape[1]
        )
        inverse = np.linalg.pinv(covariance)
        raw = _mahalanobis(prepared, reference_block.mean(axis=0), inverse)
    elif method == "knn":
        gaps = np.linalg.norm(prepared[:, None, :] - reference_block[None, :, :], axis=2)
        k = min(KNN_NEIGHBOURS, reference_block.shape[0])
        raw = np.sort(gaps, axis=1)[:, :k].mean(axis=1)
        diagnostics["neighbours"] = k
    elif method == "kde":
        # Scott's rule. Written out rather than taken from scipy so the bandwidth that
        # produced a published number is in this file and cannot move with a dependency.
        n, d = reference_block.shape
        bandwidth = float(n ** (-1.0 / (d + 4)))
        gaps = np.linalg.norm(prepared[:, None, :] - reference_block[None, :, :], axis=2)
        weights = np.exp(-0.5 * (gaps / bandwidth) ** 2).mean(axis=1)
        raw = -np.log(np.maximum(weights, 1e-300))
        diagnostics["bandwidth"] = bandwidth
    elif method == "pca_reconstruction":
        from sklearn.decomposition import PCA

        components = min(PCA_COMPONENTS, reference_block.shape[0] - 1, reference_block.shape[1])
        model = PCA(n_components=components, random_state=pilot.FIT_SEED).fit(
            reference_block[_content_order(reference_block)]
        )
        raw = np.linalg.norm(prepared - model.inverse_transform(model.transform(prepared)), axis=1)
        diagnostics["components"] = int(components)
        diagnostics["explained_variance_ratio"] = float(model.explained_variance_ratio_.sum())
    elif method == "cluster":
        from sklearn.cluster import KMeans

        clusters = min(CLUSTER_COUNT, reference_block.shape[0])
        model = KMeans(n_clusters=clusters, n_init=10, random_state=pilot.FIT_SEED).fit(
            reference_block[_content_order(reference_block)]
        )
        raw = np.linalg.norm(prepared[:, None, :] - model.cluster_centers_[None, :, :], axis=2).min(
            axis=1
        )
        diagnostics["clusters"] = int(clusters)
    elif method == "contrastive":
        raw, contrastive_diagnostics = _contrastive_distance(
            env, prepared, fit_ids, fill, centre, scale
        )
        diagnostics.update(contrastive_diagnostics)
    elif method == "retrieval":
        known = [i for i in calibration_ids if np.isfinite(reliability.get(i, np.nan))]
        if not known:
            raise PhaseError("retrieval has no calibration environment with a measured reliability")
        block = prepared[np.array([position[i] for i in known])]
        errors = np.array([reliability[i] for i in known])
        gaps = np.linalg.norm(prepared[:, None, :] - block[None, :, :], axis=2)
        k = min(RETRIEVAL_NEIGHBOURS, len(known))
        nearest = np.argsort(gaps, axis=1)[:, :k]
        weights = 1.0 / (np.take_along_axis(gaps, nearest, axis=1) + 1e-6)
        raw = (errors[nearest] * weights).sum(axis=1) / weights.sum(axis=1)
        diagnostics["neighbours"] = k
        diagnostics["retrieval_pool"] = len(known)
        diagnostics["target"] = RETRIEVAL_TARGET
    elif method == "oracle_engine":
        raw = np.where(np.isin(env.engine, list(fit_engines)), 0.0, 1.0)
        diagnostics["note"] = "reads engine identity; an upper bound, never a deployable arm"
    elif method == "oracle_reliability":
        if oracle_reliability is None:
            raise PhaseError("the reliability oracle needs the measured reliability it reads")
        raw = np.array([oracle_reliability.get(i, np.nan) for i in env.ids])
        diagnostics["note"] = (
            "reads the environment's own measured reliability, which needs the held-out "
            "engine's labels; the ceiling on what any applicability estimator could deliver"
        )
    else:  # pragma: no cover - the method table is closed and tested
        raise PhaseError(f"unknown applicability method {method!r}")

    # The frozen percentile map. Its reference is the CALIBRATION environments of the fit
    # engines -- fitted-transformer discipline, same as SGV2's shift percentile -- so that a
    # Mahalanobis distance and a retrieved error rate arrive on the same axis and one gamma
    # can mean the same thing for both.
    reference_ids = [i for i in calibration_ids if np.isfinite(raw[position[i]])]
    reference = np.sort(raw[np.array([position[i] for i in reference_ids])])
    percentile = np.searchsorted(reference, raw, side="right") / float(max(reference.size, 1))
    applicability = np.where(np.isfinite(raw), 1.0 - percentile, np.nan)
    diagnostics["percentile_reference_environments"] = len(reference_ids)
    return Applicability(
        method=method,
        values={i: float(applicability[position[i]]) for i in env.ids},
        raw={i: float(raw[position[i]]) for i in env.ids},
        diagnostics=diagnostics,
    )


def _contrastive_distance(
    env: EnvironmentTable,
    prepared: np.ndarray,
    fit_ids: tuple[str, ...],
    fill: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Method B's learned metric: two views of one environment should land near each other.

    The self-supervised signal is the view split. Each fit environment is cut into disjoint
    halves by document, and the objective asks for directions along which two halves of the
    same environment agree and different environments disagree -- a within/between scatter
    ratio, solved as a symmetric generalised eigenproblem rather than by gradient descent so
    the result is exact and cannot depend on an optimiser's seed. No engine label enters:
    the classes are environments, and two shards of the same engine are different classes.
    """
    from scipy.linalg import eigh

    keep = [i for i, parent in enumerate(env.view_of) if parent in set(fit_ids)]
    if not keep:
        raise PhaseError("no fit environment produced a view; the contrastive metric has no signal")
    views = _prepare(env.view_descriptors[keep], fill, centre, scale)
    parents = np.array([env.view_of[i] for i in keep])
    width = views.shape[1]
    within = np.zeros((width, width))
    between = np.zeros((width, width))
    grand = views.mean(axis=0)
    for parent in np.unique(parents):
        block = views[parents == parent]
        gap = block - block.mean(axis=0)
        within += gap.T @ gap
        delta = (block.mean(axis=0) - grand).reshape(-1, 1)
        between += block.shape[0] * (delta @ delta.T)
    regularized = within + SCATTER_RIDGE * float(np.trace(within) / width + 1.0) * np.eye(width)
    values, vectors = eigh(between, regularized)
    components = min(CONTRASTIVE_COMPONENTS, width)
    projection = vectors[:, ::-1][:, :components]
    projected = prepared @ projection
    reference = projected[np.array([env.position[i] for i in fit_ids])]
    covariance = np.cov(reference, rowvar=False) + COVARIANCE_RIDGE * np.eye(components)
    raw = _mahalanobis(projected, reference.mean(axis=0), np.linalg.pinv(covariance))
    return raw, {
        "components": int(components),
        "views_used": len(keep),
        "environments_with_views": int(np.unique(parents).size),
        "top_eigenvalues": [float(v) for v in values[::-1][:components]],
    }


# ------------------------------------------------------------------ reliability of the model


def environment_reliability(
    environment_ids: np.ndarray,
    p_harm: np.ndarray,
    harmful: np.ndarray,
    documents: np.ndarray,
    *,
    score: np.ndarray | None = None,
    tau: float | None = None,
    epsilon: float = PRIMARY_EPSILON,
    minimum_rows: int = 20,
) -> dict[str, dict[str, float]]:
    """How wrong the risk model is on each environment. Never a row label as a target.

    Seven quantities, because they can disagree and a single "reliability" number would let
    whichever one happened to move carry the finding. `abs_risk_error` is the declared
    retrieval and oracle target; the rest are reported beside it. When a threshold is given,
    the accept rule is the risk controller's own -- accept where the score is at or above tau,
    higher being safer -- so the deployment cell here and the deployment cell in `risk/` mean
    the same thing.
    """
    if (tau is None) != (score is None):
        raise PhaseError("a deployed threshold needs the score it thresholds, and vice versa")
    out: dict[str, dict[str, float]] = {}
    for identifier in sorted(set(environment_ids.tolist())):
        rows = np.flatnonzero(environment_ids == identifier)
        if rows.size < minimum_rows:
            continue
        probability = p_harm[rows]
        outcome = harmful[rows].astype(float)
        predicted, realized = float(probability.mean()), float(outcome.mean())
        cell = {
            "n_rows": int(rows.size),
            "n_documents": len(set(documents[rows].tolist())),
            "harm_rate": realized,
            "mean_predicted_harm": predicted,
            "signed_risk_error": predicted - realized,
            "abs_risk_error": abs(predicted - realized),
            "brier": float(brier_score(probability, outcome)),
        }
        for binning in BINNINGS:
            error, maximum = expected_calibration_error(probability, outcome, ECE_BINS, binning)
            cell[f"ece_{binning}"] = float(error)
            cell[f"max_calibration_error_{binning}"] = float(maximum)
        if tau is not None and score is not None:
            # The deployment question: the threshold was chosen on the fit engines'
            # calibration rows to hold harm at epsilon. Does it still hold here?
            accepted = score[rows] >= tau
            cell["coverage_at_source_threshold"] = float(accepted.mean())
            cell["harm_at_source_threshold"] = (
                float(outcome[accepted].mean()) if accepted.any() else float("nan")
            )
            cell["bound_violation"] = (
                float(max(0.0, cell["harm_at_source_threshold"] - epsilon))
                if accepted.any()
                else float("nan")
            )
        out[identifier] = cell
    return out


# ------------------------------------------------------------------ the predecessor's table

SGV3_ARMS = {
    "no_correction": "arm__no_correction",
    "confidence_only": "arm__confidence_only",
    "harm_only": "arm__harm_only",
    "harm_aware": "arm__harm_aware",
    "shift_aware": "arm__shift_aware",
    "selfaware": "arm__selfaware",
}


def load_predecessor() -> pd.DataFrame:
    """SGV3's write-once score table, verified against SGV3's own recorded hash.

    Every baseline in this stage is READ from the predecessor rather than refitted. Two
    reasons, and the second is the important one. Refitting would burn the same compute to
    reproduce numbers that already exist; and a refit that differed by a library version or a
    seed would show up as an SGV4 effect. Reading the artifact makes "the baseline is the
    published baseline" a checkable property instead of an intention, and `tests/leakage`
    checks it by joining on candidate_id.
    """
    if not sa.SCORES.is_file() or not sa.FIT_RECORD.is_file():
        raise PhaseError("SGV3's score table is required; run scripts/sgv3_self_aware.py first")
    record = cc._read_json(sa.FIT_RECORD)
    recorded = record["artifacts"].get(cc._relative(sa.SCORES))
    if recorded is not None and file_sha256(sa.SCORES) != recorded:
        raise PhaseError("SGV3's score table has moved since SGV3 recorded its hash")
    table = pd.read_parquet(
        sa.SCORES,
        columns=[
            "candidate_id",
            "document_id",
            "engine_id",
            "held_out_engine",
            "evaluation_mode",
            "is_harmful",
            "beneficial",
            "p_harm",
            "p_benefit",
            "uncertainty__full",
            *SGV3_ARMS.values(),
        ],
    )
    missing = [name for name in SGV3_ARMS.values() if name not in table.columns]
    if missing:
        raise PhaseError(f"SGV3's table is missing {missing}")
    return table


# ------------------------------------------------------------------ selection

# The pseudo-method the B ablation runs under. It is not an applicability estimator: it puts
# SGV3's per-row uncertainty composite where the applicability penalty goes, so that "swap
# the environment score for the uncertainty score" is measured under SGV4's own selection
# procedure rather than under SGV3's. Comparing against SGV3's published arm confounds the
# penalty with the selection; both comparisons are reported.
UNCERTAINTY_PENALTY = "sgv3_uncertainty"
SELECTABLE = (*APPLICABILITY_METHODS, UNCERTAINTY_PENALTY)


def _resolved(values: np.ndarray) -> np.ndarray:
    """Applicability for rows whose environment was too small to describe.

    Zero, not the fold mean. An environment the estimator could not characterise is one it
    cannot vouch for, and the method's own logic says that is the conservative end of the
    scale. How many rows this touches is recorded per fold, because a policy quietly driven
    by a fallback is not the policy that was described.
    """
    return np.where(np.isfinite(values), values, 0.0)


@dataclass(slots=True)
class Selection:
    """Which applicability estimator, how hard to lean on it, and how it was chosen."""

    method: str
    lambda_: float
    gamma: float
    per_method: dict[str, tuple[float, float]]
    lambda_without_environment: float
    inner: dict[str, Any]


def select_policy(
    design: dg.Design,
    context: DescriptorContext,
    partition: pd.DataFrame,
    outer: Fold,
    columns: list[int],
) -> Selection:
    """Pick (method, lambda, gamma) by an inner leave-one-engine-out over the fit engines.

    gamma = 0 is on the grid on purpose: the procedure is allowed to decide the applicability
    signal is not worth acting on, and a selection that cannot decline is not a selection.
    The outer held-out engine appears in no block of any inner fold, so nothing chosen here
    can have been chosen for performing well on the engine the result is read from.
    """
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    grid: dict[str, list[float]] = {}
    fold_notes: list[dict[str, Any]] = []
    for inner_held in outer.train_engines:
        fold = rl._inner_fold(design, outer, inner_held)
        model = sa.fit_uncertainty(design, fold, columns)
        calibration = model.probabilities(design, fold.source_cal)
        evaluation = model.probabilities(design, fold.eval)
        fold_env = build_fold_environments(design, context, partition, fold)
        reliability = environment_reliability(
            fold_env.row_environment[fold.source_cal],
            calibration["harm"],
            design.harmful[fold.source_cal],
            design.documents[fold.source_cal],
        )
        target = {name: cell[RETRIEVAL_TARGET] for name, cell in reliability.items()}
        eval_environments = fold_env.row_environment[fold.eval]
        penalties: dict[str, np.ndarray] = {}
        for method in APPLICABILITY_METHODS:
            fitted = fit_applicability(
                fold_env,
                method,
                fit_ids=block_ids(fold_env, "fit"),
                calibration_ids=block_ids(fold_env, "cal"),
                fit_engines=fold.train_engines,
                reliability=target,
            )
            penalties[method] = 1.0 - _resolved(fitted.of(eval_environments))
        penalties[UNCERTAINTY_PENALTY] = model.composites(design, fold.eval)[sa.HEADLINE_VARIANT]

        harmful, beneficial = design.harmful[fold.eval], design.beneficial[fold.eval]
        for lambda_ in LAMBDA_GRID:
            utility = evaluation["benefit"] - lambda_ * evaluation["harm"]
            for method, penalty in penalties.items():
                for gamma in GAMMA_GRID:
                    measured = achievable_repair_recall(
                        utility - gamma * penalty, harmful, beneficial
                    )
                    grid.setdefault(f"{method}|l_{lambda_:g}|g_{gamma:g}", []).append(
                        float(measured[key]["repair_recall"])
                    )
        fold_notes.append(
            {
                "inner_held_out": inner_held,
                "inner_train_engines": list(fold.train_engines),
                "rows": {"fit": int(fold.fit.size), "evaluate": int(fold.eval.size)},
                "environments": {tag: len(block_ids(fold_env, tag)) for _, tag in FOLD_BLOCKS},
                "environments_with_measured_reliability": len(target),
            }
        )

    means = {name: float(np.mean(values)) for name, values in grid.items()}

    def parse(name: str) -> tuple[str, float, float]:
        method, lambda_text, gamma_text = name.split("|")
        return method, float(lambda_text[2:]), float(gamma_text[2:])

    def rank(name: str) -> tuple[float, float, float, str]:
        method, lambda_, gamma = parse(name)
        return (means[name], -abs(gamma), -lambda_, method)

    deployable = [n for n in means if parse(n)[0] in APPLICABILITY_METHODS]
    best = max(deployable, key=rank)
    method, lambda_, gamma = parse(best)
    per_method = {}
    for candidate in SELECTABLE:
        names = [n for n in means if parse(n)[0] == candidate]
        chosen = max(names, key=rank)
        per_method[candidate] = (parse(chosen)[1], parse(chosen)[2])
    zero_gamma = [n for n in means if parse(n)[2] == 0.0]
    lambda_without = parse(max(zero_gamma, key=rank))[1]

    return Selection(
        method=method,
        lambda_=lambda_,
        gamma=gamma,
        per_method=per_method,
        lambda_without_environment=lambda_without,
        inner={
            "inner_engines": list(outer.train_engines),
            "inner_evaluation_role": "CALIBRATION",
            "lambda_grid": list(LAMBDA_GRID),
            "gamma_grid": list(GAMMA_GRID),
            "methods": list(SELECTABLE),
            "selected": {"method": method, "lambda": lambda_, "gamma": gamma, "score": means[best]},
            "lambda_without_environment": lambda_without,
            "per_method_selection": {
                name: {"lambda": pair[0], "gamma": pair[1]} for name, pair in per_method.items()
            },
            "mean_repair_recall_at_primary_epsilon": means,
            "per_inner_fold": grid,
            "inner_folds": fold_notes,
            "gamma_at_grid_boundary": bool(gamma in (GAMMA_GRID[0], GAMMA_GRID[-1])),
            "tie_break": (
                "highest inner mean, then smallest |gamma|, then largest lambda, then method "
                "name. Smallest |gamma| first so a tie resolves towards the baseline rather "
                "than towards the hypothesis."
            ),
            "engine_identity_used_by_selection": False,
        },
    )


# ------------------------------------------------------------------ the fold


BLOCK_OF_MODE = {CROSS_ENGINE: "eval", IN_DOMAIN: "seen", SOURCE_CALIBRATION: "cal"}


@dataclass(slots=True)
class FoldModel:
    """Everything one leave-one-engine-out fold fits, so every block goes through one object."""

    held_out: str
    fold: Fold
    environments: EnvironmentTable
    selection: Selection
    applicability: dict[str, Applicability]
    reliability: dict[str, dict[str, dict[str, float]]]
    constant_applicability: float
    diagnostics: dict[str, Any]

    def penalties(self, environment_ids: np.ndarray) -> dict[str, np.ndarray]:
        return {
            method: 1.0 - _resolved(model.of(environment_ids))
            for method, model in self.applicability.items()
        }

    def arms(self, block: pd.DataFrame, environment_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Every acceptance rule this stage compares, on one block of rows.

        The six baselines are columns of SGV3's table, copied rather than recomputed. The
        proposed arm and every ablation are built from the same two calibrated probabilities
        SGV3 used, so the only thing that differs between `harm_aware` and `env_aware` is the
        environment term.
        """
        benefit = block["p_benefit"].to_numpy(dtype=np.float64)
        harm = block["p_harm"].to_numpy(dtype=np.float64)
        uncertainty = block["uncertainty__full"].to_numpy(dtype=np.float64)
        penalty = self.penalties(environment_ids)
        selection = self.selection

        def utility(lambda_: float) -> np.ndarray:
            return benefit - lambda_ * harm

        scores = {
            name: block[column].to_numpy(dtype=np.float64) for name, column in SGV3_ARMS.items()
        }
        for method in APPLICABILITY_METHODS:
            lambda_, gamma = selection.per_method[method]
            scores[f"env_aware__{method}"] = utility(lambda_) - gamma * penalty[method]
        scores[PROPOSED_ARM] = scores[f"env_aware__{selection.method}"]

        # Ablation A: with no environment term the policy IS the harm-aware policy, at the
        # lambda the same inner procedure picks when gamma is pinned to zero. Aliasing rather
        # than recomputing is what lets `tests/leakage` assert the identity bit-for-bit.
        scores["ablate_environment"] = utility(selection.lambda_without_environment)
        lambda_b, gamma_b = selection.per_method[UNCERTAINTY_PENALTY]
        scores["ablate_uncertainty_swap"] = utility(lambda_b) - gamma_b * uncertainty
        scores["env_random"] = utility(selection.lambda_) - selection.gamma * penalty["random"]
        # Ablation E: the same penalty, stripped of its variation. The constant is the
        # row-weighted mean applicability over the fold's CALIBRATION environments, so this
        # arm stays deployable; the realized mean penalty of both arms is recorded beside it
        # so "was the conservatism actually matched" is checkable rather than assumed.
        scores["env_constant"] = utility(selection.lambda_) - selection.gamma * (
            1.0 - self.constant_applicability
        )
        for method, name in (
            ("oracle_engine", "oracle_engine_applicability"),
            ("oracle_reliability", "oracle_reliability_applicability"),
        ):
            scores[name] = utility(selection.lambda_) - selection.gamma * penalty[method]
        return scores

    def oracle_gamma(
        self, block: pd.DataFrame, environment_ids: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """The ceiling: (lambda, gamma) chosen against the held-out engine's own labels.

        Never an arm an operator could run. It exists to separate "the inner selection failed"
        from "the signal is not there", which are different findings and would otherwise be
        reported as the same null -- the distinction SGV3 had to draw for its own lambdas.
        """
        benefit = block["p_benefit"].to_numpy(dtype=np.float64)
        harm = block["p_harm"].to_numpy(dtype=np.float64)
        harmful = block["is_harmful"].to_numpy(dtype=bool)
        beneficial = block["beneficial"].to_numpy(dtype=bool)
        penalty = self.penalties(environment_ids)[self.selection.method]
        key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
        grid: dict[str, float] = {}
        for lambda_ in LAMBDA_GRID:
            utility = benefit - lambda_ * harm
            for gamma in GAMMA_GRID:
                measured = achievable_repair_recall(utility - gamma * penalty, harmful, beneficial)
                grid[f"l_{lambda_:g}|g_{gamma:g}"] = float(measured[key]["repair_recall"])
        best = max(grid, key=lambda name: grid[name])
        lambda_ = float(best.split("|")[0][2:])
        gamma = float(best.split("|")[1][2:])
        return benefit - lambda_ * harm - gamma * penalty, {
            "note": (
                "chosen on the held-out engine's own labels; a ceiling on what perfect "
                "selection of (lambda, gamma) could deliver, never a deployable arm"
            ),
            "selected": best,
            "selected_repair_recall": grid[best],
            "deployable_selection_repair_recall": grid[
                f"l_{self.selection.lambda_:g}|g_{self.selection.gamma:g}"
            ],
            "grid": grid,
        }


def _variance_decomposition(values: np.ndarray, engines: np.ndarray) -> dict[str, float]:
    """How much of the applicability spread is between engines and how much is within one.

    This is the precondition for the ranking channel stated in the module docstring. If
    within-engine variance is ~0 then applicability is constant across a cross-engine
    evaluation block, and the proposed policy is arithmetically identical to the harm-aware
    baseline there whatever gamma is -- a fact about the construction, not about the data,
    and one a reader is entitled to see before the endpoint.
    """
    finite = np.isfinite(values)
    values, engines = values[finite], engines[finite]
    if values.size < 2:
        blank = float("nan")
        return {"total": blank, "between_engine": blank, "within_engine": blank}
    grand = float(values.mean())
    between = 0.0
    within = 0.0
    for engine in np.unique(engines):
        block = values[engines == engine]
        between += block.size * (float(block.mean()) - grand) ** 2
        within += float(((block - block.mean()) ** 2).sum())
    total = between + within
    return {
        "total": float(total / values.size),
        "between_engine": float(between / values.size),
        "within_engine": float(within / values.size),
        "within_engine_share": float(within / total) if total > 0 else float("nan"),
        "sd": float(values.std()),
        "n_environments": int(values.size),
    }


def _blocks(
    table: pd.DataFrame, held_out: str, row_of: dict[str, int]
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """The three evaluation blocks of one fold, each with its design row indices.

    The join is on candidate_id, so a row of SGV3's table and the design row it came from are
    the same edit by identity rather than by position.
    """
    out: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for mode in (CROSS_ENGINE, IN_DOMAIN, SOURCE_CALIBRATION):
        block = table[
            (table["held_out_engine"] == held_out) & (table["evaluation_mode"] == mode)
        ].reset_index(drop=True)
        if block.empty:
            raise PhaseError(f"{held_out}: SGV3's table has no {mode} rows")
        identifiers = block["candidate_id"].astype(str)
        unknown = [i for i in identifiers if i not in row_of]
        if unknown:
            raise PhaseError(f"{held_out}/{mode}: {len(unknown)} rows are not in the design matrix")
        out[mode] = (block, np.array([row_of[i] for i in identifiers]))
    return out


def fit_fold(
    design: dg.Design,
    context: DescriptorContext,
    partition: pd.DataFrame,
    held_out: str,
    table: pd.DataFrame,
    row_of: dict[str, int],
    columns: list[int],
) -> tuple[FoldModel, dict[str, tuple[pd.DataFrame, np.ndarray]]]:
    fold = build_fold(design, held_out)
    fold_env = build_fold_environments(design, context, partition, fold)
    blocks = _blocks(table, held_out, row_of)

    def reliability_of(mode: str, tau: float | None = None, score: np.ndarray | None = None):
        block, rows = blocks[mode]
        return environment_reliability(
            fold_env.row_environment[rows],
            block["p_harm"].to_numpy(dtype=np.float64),
            block["is_harmful"].to_numpy(dtype=bool),
            design.documents[rows],
            score=score,
            tau=tau,
        )

    # The risk model's OWN deployed threshold, chosen on the fit engines' calibration rows to
    # hold harm at epsilon on the calibrated harm probability. This is the quantity Phase 6
    # found missing its nominal bound on one engine of four, and it is what "is my risk model
    # applicable here" is ultimately a question about, so every environment's reliability cell
    # is measured against it.
    calibration_block, calibration_rows = blocks[SOURCE_CALIBRATION]
    calibration_safety = 1.0 - calibration_block["p_harm"].to_numpy(dtype=np.float64)
    controller_thresholds = {
        controller: select_threshold(
            calibration_safety,
            calibration_block["is_harmful"].to_numpy(dtype=bool),
            PRIMARY_EPSILON,
            delta=DELTA,
            controller=controller,
        )
        for controller in CONTROLLERS
    }
    tau = float(controller_thresholds["empirical"].tau)

    reliability = {
        BLOCK_OF_MODE[mode]: reliability_of(
            mode,
            tau=tau,
            score=1.0 - blocks[mode][0]["p_harm"].to_numpy(dtype=np.float64),
        )
        for mode in (SOURCE_CALIBRATION, IN_DOMAIN, CROSS_ENGINE)
    }
    calibration_target = {name: cell[RETRIEVAL_TARGET] for name, cell in reliability["cal"].items()}
    # The oracle's pool is deliberately everything measured, held-out engine included. That
    # is what makes it an oracle; it reaches no other method because no other method is
    # handed this argument.
    oracle_target = {
        name: cell[RETRIEVAL_TARGET]
        for cells in reliability.values()
        for name, cell in cells.items()
    }

    selection = select_policy(design, context, partition, fold, columns)
    applicability = {
        method: fit_applicability(
            fold_env,
            method,
            fit_ids=block_ids(fold_env, "fit"),
            calibration_ids=block_ids(fold_env, "cal"),
            fit_engines=fold.train_engines,
            reliability=calibration_target,
            oracle_reliability=oracle_target if method == "oracle_reliability" else None,
        )
        for method in ALL_METHODS
    }

    calibration_environments = fold_env.row_environment[calibration_rows]
    selected = applicability[selection.method]
    weights = np.array(
        [selected.values.get(str(e), np.nan) for e in calibration_environments], dtype=float
    )
    constant = float(np.nanmean(weights)) if np.isfinite(weights).any() else 0.0

    development = np.array(
        [i for i, tag in zip(fold_env.ids, fold_env.role, strict=True) if tag in ("eval", "seen")]
    )
    development_engines = np.array([i.split("|")[0] for i in development])
    diagnostics = {
        "environments": {tag: len(block_ids(fold_env, tag)) for _, tag in FOLD_BLOCKS},
        "environment_rows": {
            tag: {
                identifier: int(fold_env.n_rows[fold_env.position[identifier]])
                for identifier in block_ids(fold_env, tag)
            }
            for _, tag in FOLD_BLOCKS
        },
        "rows_without_a_described_environment": {
            mode: int(
                np.count_nonzero(
                    ~np.isfinite(applicability[selection.method].of(fold_env.row_environment[rows]))
                )
            )
            for mode, (_, rows) in blocks.items()
        },
        "constant_applicability_from_calibration": constant,
        "applicability_dispersion": {
            method: _variance_decomposition(
                np.array([model.values[i] for i in development]), development_engines
            )
            for method, model in applicability.items()
        },
        "development_applicability": {
            method: {identifier: model.values[identifier] for identifier in development.tolist()}
            for method, model in applicability.items()
        },
        "risk_model_threshold": {
            controller: decision.as_dict() for controller, decision in controller_thresholds.items()
        },
        "estimator_diagnostics": {m: model.diagnostics for m, model in applicability.items()},
    }
    fitted = FoldModel(
        held_out=held_out,
        fold=fold,
        environments=fold_env,
        selection=selection,
        applicability=applicability,
        reliability=reliability,
        constant_applicability=constant,
        diagnostics=diagnostics,
    )
    return fitted, blocks


# ------------------------------------------------------------------ stage 1: the artifact


def _variance_components(views: np.ndarray, parents: np.ndarray) -> dict[str, dict[str, float]]:
    """Three-level decomposition of each descriptor coordinate.

    A coordinate can vary for three reasons and they mean completely different things. If it
    varies BETWEEN ENGINES, it describes the OCR process, which is what the representation is
    for. If it varies BETWEEN ENVIRONMENTS OF ONE ENGINE, it describes the batch, which is
    what makes an applicability score able to reorder anything within a single unseen engine.
    If it varies BETWEEN VIEWS OF ONE ENVIRONMENT -- two disjoint halves of the same pages
    from the same engine -- it is sampling noise. A representation whose variance is almost
    entirely between-engine is a re-encoding of engine identity, which SGV2 already showed is
    easy and does not help.
    """
    engines = np.array([p.split("|")[0] for p in parents])
    grand = views.mean(axis=0)
    width = views.shape[1]
    between_engine = np.zeros(width)
    between_environment = np.zeros(width)
    within_environment = np.zeros(width)
    for engine in np.unique(engines):
        engine_rows = engines == engine
        engine_mean = views[engine_rows].mean(axis=0)
        between_engine += int(engine_rows.sum()) * (engine_mean - grand) ** 2
        for parent in np.unique(parents[engine_rows]):
            block = views[parents == parent]
            between_environment += block.shape[0] * (block.mean(axis=0) - engine_mean) ** 2
            within_environment += ((block - block.mean(axis=0)) ** 2).sum(axis=0)
    total = between_engine + between_environment + within_environment
    out: dict[str, dict[str, float]] = {}
    for position, name in enumerate(ENV_SIGNAL_NAMES):
        denominator = float(total[position])
        if denominator <= 1e-12:
            out[name] = {
                "constant": True,
                "between_engine_share": float("nan"),
                "between_environment_share": float("nan"),
                "within_environment_share": float("nan"),
            }
            continue
        out[name] = {
            "constant": False,
            "between_engine_share": float(between_engine[position] / denominator),
            "between_environment_share": float(between_environment[position] / denominator),
            "within_environment_share": float(within_environment[position] / denominator),
        }
    return out


def _descriptor_diagnostics(env: EnvironmentTable) -> dict[str, Any]:
    """Is the descriptor measuring the environment, or measuring the sample?

    Two views of one environment are disjoint halves of the same pipeline on the same kind of
    page, so a coordinate that separates environments should agree between them. The
    three-level variance decomposition is that agreement per coordinate, and the
    view-retrieval rate is the same question asked of the whole vector at once. Both are
    computed on TRAIN environments so that this diagnostic never reads a page any policy is
    scored on.
    """
    train = np.array([i for i, role in zip(env.ids, env.role, strict=True) if role == "TRAIN"])
    if train.size == 0:
        train = np.array(env.ids)
    fill, centre, scale = _standardizer(env.descriptors[[env.position[i] for i in train]])
    keep = [i for i, parent in enumerate(env.view_of) if parent in set(train.tolist())]
    views = _prepare(env.view_descriptors[keep], fill, centre, scale)
    parents = np.array([env.view_of[i] for i in keep])
    components = _variance_components(views, parents)

    gaps = np.linalg.norm(views[:, None, :] - views[None, :, :], axis=2)
    np.fill_diagonal(gaps, np.inf)
    nearest = np.argmin(gaps, axis=1)
    engines = np.array([p.split("|")[0] for p in parents])
    live = [name for name, cell in components.items() if not cell["constant"]]
    return {
        "views": int(views.shape[0]),
        "environments_with_views": int(np.unique(parents).size),
        "variance_components": components,
        "constant_coordinates": [name for name, cell in components.items() if cell["constant"]],
        "mean_between_engine_share": float(
            np.mean([components[name]["between_engine_share"] for name in live])
        ),
        "mean_between_environment_share": float(
            np.mean([components[name]["between_environment_share"] for name in live])
        ),
        "mean_within_environment_share": float(
            np.mean([components[name]["within_environment_share"] for name in live])
        ),
        "view_retrieval_same_environment": float(np.mean(parents[nearest] == parents)),
        "view_retrieval_same_engine": float(np.mean(engines[nearest] == engines)),
        "chance_same_environment": float(1.0 / np.unique(parents).size),
        "chance_same_engine": float(1.0 / np.unique(engines).size),
        "note": (
            "view_retrieval_same_environment is the fraction of half-environments whose "
            "nearest other half-environment is its own other half. Compared against "
            "view_retrieval_same_engine it says whether the descriptor resolves an "
            "environment or only the engine that produced it, and those two readings support "
            "very different claims."
        ),
    }


def run_environments() -> int:
    """Stage 1: the environment partition, the descriptors, and whether they measure anything."""
    started = time.monotonic()
    design = dg.load_design()
    context = build_context(design)
    env = build_environments(design, context)

    sensitivity: dict[str, Any] = {}
    for shard_documents in SHARD_SENSITIVITY:
        alternative = build_environments(design, context, shard_documents)
        sensitivity[f"shard_documents_{shard_documents}"] = {
            "n_environments": len(alternative.ids),
            "median_rows": float(np.median(alternative.n_rows)),
            "min_rows": int(alternative.n_rows.min()),
            "median_documents": float(np.median(alternative.n_documents)),
            "view_retrieval_same_environment": _descriptor_diagnostics(alternative)[
                "view_retrieval_same_environment"
            ],
        }

    coverage = {
        name: float(np.mean(np.isfinite(env.descriptors[:, position])))
        for position, name in enumerate(ENV_SIGNAL_NAMES)
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cc._write_json_once(
        ENVIRONMENT_SPEC,
        {
            "schema_version": "sgv4-environment-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "synthetic": False,
            "development_only": True,
            "definition": (
                "An environment is one (engine, role, document-shard) cell. Shards partition "
                "documents WITHIN a role under a keyed digest of the document id, so an "
                "environment never straddles the fit / calibrate / evaluate boundary and the "
                "partition is a function of the corpus rather than of iteration order."
            ),
            "engine_identity_used_as_a_feature": False,
            "ground_truth_used": False,
            "partition": {
                "shard_documents": SHARD_DOCUMENTS,
                "seed": SHARD_SEED,
                "views_per_environment": ENV_VIEWS,
                "shards_per_role": {
                    str(role): int(group["shard"].nunique())
                    for role, group in env.partition.groupby("role")
                },
                "documents_per_role": {
                    str(role): len(group) for role, group in env.partition.groupby("role")
                },
            },
            "signals": [
                {
                    "name": s.name,
                    "group": s.group,
                    "statistic": s.statistic,
                    "source": s.source,
                    "definition": s.definition,
                    "rationale": s.rationale,
                    "defined_on_fraction_of_environments": coverage[s.name],
                }
                for s in ENV_SIGNALS
            ],
            "groups": {
                group: [s.name for s in ENV_SIGNALS if s.group == group] for group in ENV_GROUPS
            },
            "environments": {
                identifier: {
                    "engine": env.engine[position],
                    "role": env.role[position],
                    "shard": int(env.shard[position]),
                    "n_rows": int(env.n_rows[position]),
                    "n_documents": int(env.n_documents[position]),
                }
                for position, identifier in enumerate(env.ids)
            },
            "environment_counts": {
                f"{engine}|{role}": int(
                    np.count_nonzero((env.engine == engine) & (env.role == role))
                )
                for engine in design.engines
                for role in ("TRAIN", "CALIBRATION", "DEVELOPMENT")
            },
            "descriptor_diagnostics": _descriptor_diagnostics(env),
            "shard_granularity_sensitivity": sensitivity,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        f"environments: {len(env.ids)} cells, {len(ENV_SIGNALS)} descriptors "
        f"-> {cc._relative(ENVIRONMENT_SPEC)}"
    )
    return 0


# ------------------------------------------------------------------ the score table

ORACLE_GAMMA_ARM = "oracle_gamma"


def run_scores() -> int:
    """Fit every fold once and write the row-level table every later stage reads."""
    started = time.monotonic()
    design = dg.load_design()
    context = build_context(design)
    partition = shard_partition(design)
    table = load_predecessor()
    row_of = {
        identifier: position
        for position, identifier in enumerate(design.meta["candidate_id"].astype(str))
    }
    columns = design.columns(tuple(dg.FAMILIES))
    frames: list[pd.DataFrame] = []
    record: dict[str, Any] = {}

    for held_out in design.engines:
        fitted, blocks = fit_fold(design, context, partition, held_out, table, row_of, columns)
        thresholds: dict[str, Any] = {}
        calibration_block, calibration_rows = blocks[SOURCE_CALIBRATION]
        calibration_environments = fitted.environments.row_environment[calibration_rows]
        calibration_arms = fitted.arms(calibration_block, calibration_environments)
        harmful_calibration = calibration_block["is_harmful"].to_numpy(dtype=bool)
        for name, score in calibration_arms.items():
            finite = np.isfinite(score)
            if not finite.any():
                continue
            thresholds[name] = {
                controller: select_threshold(
                    score[finite],
                    harmful_calibration[finite],
                    PRIMARY_EPSILON,
                    delta=DELTA,
                    controller=controller,
                ).as_dict()
                for controller in CONTROLLERS
            }

        for mode, (block, rows) in blocks.items():
            environments = fitted.environments.row_environment[rows]
            frame = block[
                [
                    "candidate_id",
                    "document_id",
                    "engine_id",
                    "held_out_engine",
                    "evaluation_mode",
                    "is_harmful",
                    "beneficial",
                    "p_harm",
                    "p_benefit",
                    "uncertainty__full",
                ]
            ].copy()
            frame["environment_id"] = environments
            for method, model in fitted.applicability.items():
                frame[f"applicability__{method}"] = model.of(environments)
            for name, score in fitted.arms(block, environments).items():
                frame[f"arm__{name}"] = score
            if mode == CROSS_ENGINE:
                score, oracle = fitted.oracle_gamma(block, environments)
                frame[f"arm__{ORACLE_GAMMA_ARM}"] = score
                fitted.diagnostics["oracle_gamma"] = oracle
            else:
                frame[f"arm__{ORACLE_GAMMA_ARM}"] = np.nan
            frames.append(frame)

        dispersion = fitted.diagnostics["applicability_dispersion"][fitted.selection.method]
        record[held_out] = {
            "held_out_engine": held_out,
            "train_engines": list(fitted.fold.train_engines),
            "rows": {
                "fit": int(fitted.fold.fit.size),
                "source_calibration": int(fitted.fold.source_cal.size),
                "in_domain_development": int(fitted.fold.seen_eval.size),
                "cross_engine_development": int(fitted.fold.eval.size),
            },
            "documents": {
                "fit": len(set(design.documents[fitted.fold.fit].tolist())),
                "source_calibration": len(set(design.documents[fitted.fold.source_cal].tolist())),
                "cross_engine_development": len(set(design.documents[fitted.fold.eval].tolist())),
            },
            "selected": {
                "method": fitted.selection.method,
                "lambda": fitted.selection.lambda_,
                "gamma": fitted.selection.gamma,
                "lambda_without_environment": fitted.selection.lambda_without_environment,
                "per_method": {
                    name: {"lambda": pair[0], "gamma": pair[1]}
                    for name, pair in fitted.selection.per_method.items()
                },
            },
            "inner_selection": fitted.selection.inner,
            "certified_thresholds": thresholds,
            "reliability_by_environment": fitted.reliability,
            "diagnostics": fitted.diagnostics,
        }
        print(
            f"  fold {held_out:11s} method={fitted.selection.method:19s} "
            f"lambda={fitted.selection.lambda_:g} gamma={fitted.selection.gamma:+g} "
            f"within-engine share={dispersion['within_engine_share']:.3f}"
        )

    cc._write_parquet_once(SCORES, pd.concat(frames, ignore_index=True))
    cc._write_json_once(
        SELECTION_RECORD,
        {
            "schema_version": "sgv4-environment-selection-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "selection_scope": (
                "method, lambda and gamma are chosen by an inner leave-one-engine-out over "
                "the FIT engines, evaluated on CALIBRATION documents with the calibration "
                "pages split in half so the frozen percentile reference and the inner "
                "evaluation rows never share a page. The held-out engine contributes nothing "
                "to any choice; tests/leakage re-derives every selection from the recorded "
                "inner scores."
            ),
            "engine_identity_used_by_method": False,
            "baselines_source": (
                "every baseline arm is copied from SGV3's write-once score table, joined on "
                "candidate_id, rather than refitted -- so the comparison is against the "
                "published arm and not against a re-run of it"
            ),
            "folds": record,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"scores: {sum(len(f) for f in frames)} rows -> {cc._relative(SCORES)}")
    return 0


@dataclass(slots=True)
class Slice:
    """One (fold, evaluation mode) block, read back from the write-once score table."""

    held_out: str
    mode: str
    frame: pd.DataFrame
    harmful: np.ndarray
    beneficial: np.ndarray
    documents: np.ndarray
    environments: np.ndarray

    def arm(self, name: str) -> np.ndarray:
        return self.frame[f"arm__{name}"].to_numpy(dtype=np.float64)

    def column(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy(dtype=np.float64)

    @property
    def arm_names(self) -> list[str]:
        return [c[len("arm__") :] for c in self.frame.columns if c.startswith("arm__")]


def load_scores() -> tuple[dict[tuple[str, str], Slice], dict[str, Any]]:
    if not SCORES.is_file() or not SELECTION_RECORD.is_file():
        raise PhaseError("run --scores before any experiment that reads them")
    table = pd.read_parquet(SCORES)
    record = cc._read_json(SELECTION_RECORD)
    slices: dict[tuple[str, str], Slice] = {}
    for (held_out, mode), group in table.groupby(["held_out_engine", "evaluation_mode"], sort=True):
        frame = group.reset_index(drop=True)
        engines = set(frame["engine_id"].astype(str))
        if mode == CROSS_ENGINE and engines != {str(held_out)}:
            raise PhaseError(f"{held_out}: a non-held-out engine is in its cross-engine rows")
        if mode != CROSS_ENGINE and str(held_out) in engines:
            raise PhaseError(f"{held_out}: the held-out engine is in its {mode} rows")
        slices[(str(held_out), str(mode))] = Slice(
            held_out=str(held_out),
            mode=str(mode),
            frame=frame,
            harmful=frame["is_harmful"].to_numpy(dtype=bool),
            beneficial=frame["beneficial"].to_numpy(dtype=bool),
            documents=frame["document_id"].astype(str).to_numpy(),
            environments=frame["environment_id"].astype(str).to_numpy(),
        )
    return slices, record


# ------------------------------------------------------------------ shared endpoint helpers

# Imported, not restated. Phase 5 defines the frontier; SGV3 defines the restriction that
# keeps a rejected row rejected inside it, the interval around it, and the paired delta.
# `tests/leakage` asserts these names are the same function objects in both modules.
restricted_frontier = sa._restricted_frontier
restricted_interval = sa._restricted_interval
arm_summary = sa._arm_summary
selective_metrics = sa._selective_metrics
curve = sa._curve
paired_repair_recall_delta = sa._paired_repair_recall_delta
rate_difference = sa._rate_difference
spearman = rl._spearman


def _spearman_cell(left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Rank correlation with an ENVIRONMENT-clustered interval.

    The resampling unit is the environment, not the document and not the row. Documents are
    nested inside environments here, so resampling environments is the coarser and more
    conservative choice; using documents would treat the ~17 pages of one environment as 17
    independent observations of that environment's reliability, which they are not.
    """
    finite = np.isfinite(left) & np.isfinite(right)
    left, right, labels = left[finite], right[finite], labels[finite]
    if left.size < 4 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return {
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_environments": int(left.size),
            "degenerate_interval": True,
            "interval_available": False,
        }

    def statistic(index: np.ndarray) -> float:
        if np.unique(left[index]).size < 2 or np.unique(right[index]).size < 2:
            return float("nan")
        return float(spearman(left[index], right[index]))

    result = cluster_bootstrap_indices(
        list(labels),
        statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
        bounds=(-1.0, 1.0),
    )
    return {
        "estimate": float(spearman(left, right)),
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "p_value_two_sided": result.p_value_two_sided,
        "degenerate_interval": bool(result.degenerate_interval),
        # An environment-clustered resample of eight environments frequently draws a set on
        # which one of the two ranks is constant, and Spearman is undefined there. The
        # interval then comes back as NaN. That is a property of the design -- one fold has
        # eight unseen environments -- not of the data, so it is flagged rather than
        # silently read as "does not exclude zero", which would let a sample-size limit
        # masquerade as a negative result.
        "interval_available": bool(np.isfinite(result.lower) and np.isfinite(result.upper)),
        "n_environments": int(left.size),
    }


# ------------------------------------------------------------------ stage 2 / critical test 1

RELIABILITY_TARGETS = (
    "abs_risk_error",
    "signed_risk_error",
    "brier",
    "ece_equal_mass",
    "harm_rate",
    "harm_at_source_threshold",
    "bound_violation",
)


def run_applicability() -> int:
    """Critical test 1: does applicability track the risk model's reliability?

    Not "does applicability detect the engine". The pooled correlation cannot separate those
    two, because unseen environments are both less applicable and, on this benchmark, less
    reliable. The correlations computed WITHIN the seen environments and WITHIN the unseen
    ones can: if applicability is only an engine detector, both vanish while the pooled value
    stays large. All three are reported side by side and the decision reads the within-group
    pair.
    """
    started = time.monotonic()
    _, record = load_scores()
    folds: dict[str, Any] = {}
    pooled: dict[str, dict[str, list[tuple[float, float, str]]]] = {
        method: {target: [] for target in RELIABILITY_TARGETS} for method in ALL_METHODS
    }

    for held_out, fold_record in sorted(record["folds"].items()):
        diagnostics = fold_record["diagnostics"]
        reliability = fold_record["reliability_by_environment"]
        measured = {**reliability["eval"], **reliability["seen"]}
        environments = sorted(measured)
        unseen = np.array([e.split("|")[1] == "eval" for e in environments])
        labels = np.array(environments)
        applicability = {
            method: np.array(
                [
                    diagnostics["development_applicability"][method].get(e, np.nan)
                    for e in environments
                ]
            )
            for method in ALL_METHODS
        }

        correlations: dict[str, Any] = {}
        detection: dict[str, Any] = {}
        for method, values in applicability.items():
            finite = np.isfinite(values)
            detection[method] = {
                "auroc_unseen_is_less_applicable": (
                    float(roc_auc(-values[finite], unseen[finite].astype(float)))
                    if finite.any() and unseen[finite].any() and (~unseen[finite]).any()
                    else float("nan")
                ),
                "mean_applicability_seen": float(np.nanmean(values[~unseen])),
                "mean_applicability_unseen": float(np.nanmean(values[unseen])),
                "note": (
                    "demoted on purpose. SGV2 measured engine separation at AUC 0.867-0.999, "
                    "so a high value here re-confirms a number already in hand and is not "
                    "evidence for SGV4-E1."
                ),
            }
            cell: dict[str, Any] = {}
            for target in RELIABILITY_TARGETS:
                outcome = np.array([measured[e].get(target, np.nan) for e in environments])
                cell[target] = {
                    "pooled": _spearman_cell(values, outcome, labels),
                    "within_seen": _spearman_cell(
                        values[~unseen], outcome[~unseen], labels[~unseen]
                    ),
                    "within_unseen": _spearman_cell(
                        values[unseen], outcome[unseen], labels[unseen]
                    ),
                }
                for position in np.flatnonzero(unseen):
                    pooled[method][target].append(
                        (float(values[position]), float(outcome[position]), environments[position])
                    )
            correlations[method] = cell

        folds[held_out] = {
            "held_out_engine": held_out,
            "n_environments": {"seen": int((~unseen).sum()), "unseen": int(unseen.sum())},
            "selected_method": fold_record["selected"]["method"],
            "applicability_dispersion": diagnostics["applicability_dispersion"],
            "estimator_diagnostics": diagnostics["estimator_diagnostics"],
            "risk_model_threshold": diagnostics["risk_model_threshold"],
            "detection_diagnostic": detection,
            "correlation_with_reliability": correlations,
            "environment_cells": {
                environment: {
                    "unseen": bool(unseen[position]),
                    "applicability": {m: float(applicability[m][position]) for m in ALL_METHODS},
                    **{t: measured[environment].get(t, float("nan")) for t in RELIABILITY_TARGETS},
                    "n_rows": measured[environment]["n_rows"],
                    "n_documents": measured[environment]["n_documents"],
                }
                for position, environment in enumerate(environments)
            },
        }

        def within_unseen(method: str, table: dict[str, Any] = correlations) -> float:
            value = table[method]["abs_risk_error"]["within_unseen"]["estimate"]
            return abs(value) if np.isfinite(value) else 0.0

        best = max(ALL_METHODS, key=within_unseen)
        print(
            f"  fold {held_out:11s} strongest within-unseen rho(applicability, |risk error|) "
            f"= {correlations[best]['abs_risk_error']['within_unseen']['estimate']:+.3f} ({best})"
        )

    # The 32 unseen environments across the four folds are genuinely distinct -- each is one
    # engine's development shard, scored by the one fold that held that engine out -- so they
    # pool without repetition. Seen environments do repeat across folds and are NOT pooled.
    pooled_summary: dict[str, Any] = {}
    for method in ALL_METHODS:
        pooled_summary[method] = {}
        for target in RELIABILITY_TARGETS:
            items = pooled[method][target]
            pooled_summary[method][target] = _spearman_cell(
                np.array([i[0] for i in items]),
                np.array([i[1] for i in items]),
                np.array([i[2] for i in items]),
            )

    cc._write_json_once(
        APPLICABILITY_RESULTS,
        {
            "schema_version": "sgv4-applicability-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "synthetic": False,
            "development_only": True,
            "question": (
                "Does the applicability score rank environments by how wrong the risk model "
                "is on them? The target is the model's own error, never a row's harm label."
            ),
            "targets": {
                "abs_risk_error": "|mean predicted harm - realized harm rate| on the environment",
                "signed_risk_error": "mean predicted harm - realized harm rate (direction kept)",
                "brier": "Brier score of the calibrated harm probability on the environment",
                "ece_equal_mass": f"expected calibration error, equal-mass, {ECE_BINS} bins",
                "harm_rate": "realized harmful fraction of all sites in the environment",
                "harm_at_source_threshold": (
                    "realized harm among rows the source-calibrated threshold accepts"
                ),
                "bound_violation": (
                    "how far above epsilon the source-calibrated threshold lands, or zero"
                ),
            },
            "sign_convention": (
                "applicability is high when the environment looks like one the model was "
                "fitted on. SGV4-E1 therefore predicts a NEGATIVE correlation with every "
                "error target: more applicable should mean less wrong."
            ),
            "resampling_unit": "environment",
            "folds": folds,
            "pooled_unseen_environments": pooled_summary,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"applicability: {len(folds)} folds -> {cc._relative(APPLICABILITY_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the primary endpoint

HEADLINE_ARMS = (
    "no_correction",
    "confidence_only",
    "harm_only",
    "harm_aware",
    "shift_aware",
    "selfaware",
    PROPOSED_ARM,
    "ablate_environment",
    "ablate_uncertainty_swap",
    "env_random",
    "env_constant",
    "oracle_engine_applicability",
    "oracle_reliability_applicability",
    ORACLE_GAMMA_ARM,
)


def run_curves() -> int:
    """The risk-coverage curves and the epsilon grid, in-domain and cross-engine."""
    started = time.monotonic()
    slices, _ = load_scores()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}
    for held_out in engines:
        modes: dict[str, Any] = {}
        for mode in (CROSS_ENGINE, IN_DOMAIN):
            block = slices[(held_out, mode)]
            arms: dict[str, Any] = {}
            for name in HEADLINE_ARMS:
                if name not in block.arm_names:
                    continue
                score = block.arm(name)
                if not np.isfinite(score).any():
                    # Only `no_correction` and, in-domain, the cross-engine-only ceilings.
                    # Recorded as abstaining rather than silently dropped.
                    arms[name] = {
                        "frontier": restricted_frontier(score, block.harmful, block.beneficial),
                        "abstains_everywhere": True,
                    }
                    continue
                summary = arm_summary(
                    score,
                    block.harmful,
                    block.beneficial,
                    block.documents,
                    with_interval=(mode == CROSS_ENGINE),
                )
                epsilon_cells = {
                    f"epsilon_{int(e * 100)}": summary.pop(f"epsilon_{int(e * 100)}")
                    for e in EPSILONS
                }
                arms[name] = {
                    **summary,
                    "frontier": {
                        **epsilon_cells,
                        "total_beneficial": summary["total_beneficial"],
                        "n": summary["n"],
                    },
                    "curve": curve(score, block.harmful, block.beneficial, sa.CURVE_POINTS),
                    "abstains_everywhere": False,
                }
            modes[mode] = {
                "rows": len(block.frame),
                "documents": len(set(block.documents.tolist())),
                "environments": len(set(block.environments.tolist())),
                "beneficial": int(block.beneficial.sum()),
                "harmful": int(block.harmful.sum()),
                "arms": arms,
            }
        folds[held_out] = {"held_out_engine": held_out, "modes": modes}
        line = "  ".join(
            f"{name}={folds[held_out]['modes'][CROSS_ENGINE]['arms'][name]['frontier'][key]['repair_recall']:.3f}"
            for name in ("harm_only", "harm_aware", "selfaware", PROPOSED_ARM)
        )
        print(f"  fold {held_out:11s} {line}")

    cc._write_json_once(
        CURVE_RESULTS,
        {
            "schema_version": "sgv4-risk-coverage-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "synthetic": False,
            "development_only": True,
            "primary_endpoint": (
                "repair recall while accepted-harm stays within epsilon, computed on the rows "
                "an arm is willing to accept and rescaled onto all sites, so an arm is never "
                "credited with a repair it rejected"
            ),
            "epsilon_grid": list(EPSILONS),
            "primary_epsilon": PRIMARY_EPSILON,
            "matched_coverages": list(MATCHED_COVERAGES),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"curves: {len(folds)} folds -> {cc._relative(CURVE_RESULTS)}")
    return 0


# ------------------------------------------------------------------ critical tests 2 and 3


def _deployed(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, tau: float
) -> dict[str, float]:
    """What actually happens when the source-calibrated threshold meets the new engine.

    The ranking endpoint asks what the arm COULD deliver if someone chose the cut with
    hindsight. This asks what it DOES deliver at the cut an operator would actually have,
    which is the only number a deployment can be held to and the one Phase 6 found breaking.
    """
    accepted = np.isfinite(score) & (score >= tau)
    total = int(beneficial.sum())
    if not accepted.any():
        return {
            "coverage": 0.0,
            "n_accepted": 0,
            "realized_harm_rate": float("nan"),
            "repair_recall": 0.0,
            "bound_violation": float("nan"),
            "holds_bound": True,
            "accepts_nothing": True,
        }
    realized = float(harmful[accepted].mean())
    return {
        "coverage": float(accepted.mean()),
        "n_accepted": int(accepted.sum()),
        "realized_harm_rate": realized,
        "repair_recall": float(beneficial[accepted].sum() / max(total, 1)),
        "bound_violation": float(max(0.0, realized - PRIMARY_EPSILON)),
        "holds_bound": bool(realized <= PRIMARY_EPSILON),
        "accepts_nothing": False,
    }


def run_transfer() -> int:
    """Critical test 2 (does it improve decisions) and 3 (does it survive an unseen engine)."""
    started = time.monotonic()
    slices, record = load_scores()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        block = slices[(held_out, CROSS_ENGINE)]
        seen = slices[(held_out, IN_DOMAIN)]
        proposed = block.arm(PROPOSED_ARM)

        comparisons: dict[str, Any] = {}
        contrasted = (
            *SUBSTANTIVE_BASELINES,
            "ablate_environment",
            "env_random",
            "env_constant",
        )
        for baseline in contrasted:
            comparisons[baseline] = {
                "paired_repair_recall_delta": paired_repair_recall_delta(
                    proposed,
                    block.arm(baseline),
                    block.harmful,
                    block.beneficial,
                    block.documents,
                    PRIMARY_EPSILON,
                ),
                # SGV2's co-primary, imported with its argument order intact: the
                # BASELINE goes first and a positive harm_reduction means the proposed arm
                # accepted the same number of edits and fewer of them were harmful.
                "matched_coverage_harm_reduction": rl._matched_coverage_contrast(
                    block.arm(baseline),
                    proposed,
                    block.harmful,
                    block.beneficial,
                    block.documents,
                ),
            }

        transfer_gap = {}
        for name in HEADLINE_ARMS:
            if name not in block.arm_names:
                continue
            cross = restricted_frontier(block.arm(name), block.harmful, block.beneficial)
            if np.isfinite(seen.arm(name)).any():
                inside = restricted_frontier(seen.arm(name), seen.harmful, seen.beneficial)
                in_domain = float(inside[key]["repair_recall"])
            else:
                in_domain = float("nan")
            transfer_gap[name] = {
                "in_domain": in_domain,
                "cross_engine": float(cross[key]["repair_recall"]),
                "gap": in_domain - float(cross[key]["repair_recall"]),
            }

        selected_method = record["folds"][held_out]["selected"]["method"]
        applicability_column = f"applicability__{selected_method}"
        thresholds = record["folds"][held_out]["certified_thresholds"]
        deployment: dict[str, Any] = {}
        for name in HEADLINE_ARMS:
            if name not in thresholds:
                continue
            deployment[name] = {
                controller: {
                    "tau": thresholds[name][controller]["tau"],
                    "feasible_on_calibration": thresholds[name][controller]["feasible"],
                    "calibration_coverage": thresholds[name][controller]["coverage"],
                    **_deployed(
                        block.arm(name),
                        block.harmful,
                        block.beneficial,
                        float(thresholds[name][controller]["tau"]),
                    ),
                }
                for controller in CONTROLLERS
            }

        # The volume channel, environment by environment. If the applicability penalty is
        # doing what it claims, the environments it rates least applicable are the ones where
        # the source-calibrated threshold would otherwise have broken its bound.
        per_environment = {}
        for environment in sorted(set(block.environments.tolist())):
            rows = np.flatnonzero(block.environments == environment)
            per_environment[environment] = {
                "n_rows": int(rows.size),
                "applicability": float(block.column(applicability_column)[rows][0]),
                "arms": {
                    name: _deployed(
                        block.arm(name)[rows],
                        block.harmful[rows],
                        block.beneficial[rows],
                        float(thresholds[name]["empirical"]["tau"]),
                    )
                    for name in ("harm_only", "harm_aware", PROPOSED_ARM)
                    if name in thresholds
                },
            }

        # Does the policy become conservative in the RIGHT places? This is the mechanism the
        # brief describes -- low applicability should mean less correction -- and it is a
        # different question from whether the endpoint moved. An arm can tighten by exactly
        # the right total amount and spend all of it on the environments that were already
        # safe, which would look identical in a pooled number and is a failure.
        environments = sorted(per_environment)
        applicable = np.array([per_environment[e]["applicability"] for e in environments])
        violation = np.array(
            [
                per_environment[e]["arms"].get("harm_aware", {}).get("bound_violation", np.nan)
                for e in environments
            ]
        )
        coverage_change = np.array(
            [
                per_environment[e]["arms"].get(PROPOSED_ARM, {}).get("coverage", np.nan)
                - per_environment[e]["arms"].get("harm_aware", {}).get("coverage", np.nan)
                for e in environments
            ]
        )
        targeting = {
            "n_environments": len(environments),
            "applicability_distinct_values": int(np.unique(np.round(applicable, 12)).size),
            "spearman_applicability_vs_baseline_bound_violation": _spearman_cell(
                applicable, violation, np.array(environments)
            ),
            "spearman_applicability_vs_coverage_change": _spearman_cell(
                applicable, coverage_change, np.array(environments)
            ),
            "mean_coverage_change": float(np.nanmean(coverage_change)),
            "environments_where_the_baseline_broke_the_bound": [
                e for e, v in zip(environments, violation, strict=True) if np.isfinite(v) and v > 0
            ],
            "note": (
                "SGV4-E1 predicts the first correlation is NEGATIVE -- less applicable "
                "environments are where the source-calibrated threshold breaks -- and the "
                "second is POSITIVE, because the penalty should remove the most coverage "
                "exactly where applicability is lowest."
            ),
        }

        folds[held_out] = {
            "held_out_engine": held_out,
            "selected": record["folds"][held_out]["selected"],
            "comparisons_against_baselines": comparisons,
            "transfer_gap": transfer_gap,
            "deployed_threshold": deployment,
            "deployed_by_environment": per_environment,
            "conservatism_targeting": targeting,
        }
        delta = comparisons["harm_aware"]["paired_repair_recall_delta"]
        gap = transfer_gap[PROPOSED_ARM]
        print(
            f"  fold {held_out:11s} {PROPOSED_ARM}={gap['cross_engine']:.3f} "
            f"(in-domain {gap['in_domain']:.3f}) vs harm_aware "
            f"delta={delta['delta_repair_recall']:+.3f} "
            f"CI=[{delta['ci_lower']:+.3f},{delta['ci_upper']:+.3f}]"
        )

    cc._write_json_once(
        TRANSFER_RESULTS,
        {
            "schema_version": "sgv4-transfer-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "synthetic": False,
            "development_only": True,
            "test_2": (
                "a success needs higher repair recall AND no worse harm. The paired delta "
                "answers the first at a hindsight-chosen cut; the matched-coverage contrast "
                "answers the second at equal accept volume; the deployed threshold answers "
                "both at the cut an operator would actually have."
            ),
            "test_3": (
                "every number here is measured on the held-out engine only. The in-domain "
                "column of transfer_gap is the same arm on documents held out from engines it "
                "HAS seen, and the distance between the pair is what the engine change costs."
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"transfer: {len(folds)} folds -> {cc._relative(TRANSFER_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the required ablations


def run_ablation() -> int:
    """The five required ablations, plus the estimator family they all sit inside."""
    started = time.monotonic()
    slices, record = load_scores()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        block = slices[(held_out, CROSS_ENGINE)]
        proposed = block.arm(PROPOSED_ARM)
        selected = record["folds"][held_out]["selected"]

        def cell(
            name: str, block: Slice = block, proposed: np.ndarray = proposed
        ) -> dict[str, Any]:
            score = block.arm(name)
            frontier = restricted_frontier(score, block.harmful, block.beneficial)
            out: dict[str, Any] = {
                "repair_recall_at_primary_epsilon": float(frontier[key]["repair_recall"]),
                "coverage_at_primary_epsilon": float(frontier[key]["coverage"]),
                "epsilon_grid": {
                    f"epsilon_{int(e * 100)}": {
                        "repair_recall": float(
                            frontier[f"epsilon_{int(e * 100)}"]["repair_recall"]
                        ),
                        "coverage": float(frontier[f"epsilon_{int(e * 100)}"]["coverage"]),
                    }
                    for e in EPSILONS
                },
                "identical_to_proposed": bool(np.array_equal(score, proposed, equal_nan=True)),
            }
            if not out["identical_to_proposed"]:
                out["paired_delta_proposed_minus_this"] = paired_repair_recall_delta(
                    proposed,
                    score,
                    block.harmful,
                    block.beneficial,
                    block.documents,
                    PRIMARY_EPSILON,
                )
            return out

        ablations = {
            name: {"brief_label": label, **cell(name)}
            for name, label in ABLATION_LABELS.items()
            if name in block.arm_names
        }
        family = {
            method: {
                "selected_lambda": selected["per_method"][method]["lambda"],
                "selected_gamma": selected["per_method"][method]["gamma"],
                **cell(f"env_aware__{method}"),
            }
            for method in APPLICABILITY_METHODS
        }
        mean_penalty = {
            name: float(np.nanmean(1.0 - block.column(f"applicability__{method}")))
            for name, method in (
                ("proposed", selected["method"]),
                ("random", "random"),
                ("oracle_engine", "oracle_engine"),
                ("oracle_reliability", "oracle_reliability"),
            )
        }
        folds[held_out] = {
            "held_out_engine": held_out,
            "selected": selected,
            "proposed": cell(PROPOSED_ARM),
            "ablations": ablations,
            "estimator_family": family,
            "ceilings": {
                name: cell(name)
                for name in (*CEILING_ARMS, ORACLE_GAMMA_ARM)
                if name in block.arm_names
            },
            "mean_penalty_weight": mean_penalty,
            "conservatism_note": (
                "mean_penalty_weight is the average of (1 - applicability) over the held-out "
                "engine's rows. Ablation E holds the penalty constant at the calibration mean "
                "rather than at this one, so the two are reported side by side: if they differ "
                "materially then E did not hold conservatism fixed and its comparison has to "
                "be read with that in mind."
            ),
        }
        line = "  ".join(
            f"{name.replace('ablate_', '')[:12]}={value['repair_recall_at_primary_epsilon']:.3f}"
            for name, value in ablations.items()
        )
        proposed_recall = folds[held_out]["proposed"]["repair_recall_at_primary_epsilon"]
        print(f"  fold {held_out:11s} proposed={proposed_recall:.3f}  {line}")

    cc._write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv4-ablation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "synthetic": False,
            "development_only": True,
            "labels": ABLATION_LABELS,
            "note": (
                "Ablation A is the harm-aware policy at the lambda the same inner procedure "
                "picks with gamma pinned to zero, so it is an alias rather than a refit and "
                "the identity is asserted in tests/leakage. Ablation B is measured twice: "
                "here under SGV4's own selection with SGV3's uncertainty composite in the "
                "penalty slot, and in the transfer artifact against SGV3's published arm."
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"ablation: {len(folds)} folds -> {cc._relative(ABLATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the decision


def _favourable(cell: dict[str, Any]) -> bool:
    """An interval that excludes zero on the favourable side, and was actually resampled.

    A degenerate interval is a grafted one-sided bound produced when every resample returned
    the same number -- which is what happens when two arms are literally the same vector. It
    can never count as evidence in either direction.
    """
    return bool(
        cell.get("ci_lower", float("nan")) > 0.0 and not cell.get("degenerate_interval", False)
    )


def _adverse(cell: dict[str, Any]) -> bool:
    return bool(
        cell.get("ci_upper", float("nan")) < 0.0 and not cell.get("degenerate_interval", False)
    )


def run_decide() -> int:
    """The machine-readable finding, under a rule fixed before the numbers were read."""
    started = time.monotonic()
    applicability = cc._read_json(APPLICABILITY_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)
    curves = cc._read_json(CURVE_RESULTS)
    engines = sorted(transfer["folds"])
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"

    # --- the precondition ------------------------------------------------------------------
    precondition = {}
    for engine in engines:
        method = transfer["folds"][engine]["selected"]["method"]
        dispersion = applicability["folds"][engine]["applicability_dispersion"][method]
        cross = [
            cell["applicability"][method]
            for cell in applicability["folds"][engine]["environment_cells"].values()
            if cell["unseen"]
        ]
        precondition[engine] = {
            "selected_method": method,
            "within_engine_variance_share": dispersion["within_engine_share"],
            "cross_engine_applicability_sd": float(np.std(cross)),
            "cross_engine_applicability_distinct_values": int(np.unique(np.round(cross, 12)).size),
            "ranking_channel_open": bool(np.unique(np.round(cross, 12)).size > 1),
        }

    # --- test 1 ----------------------------------------------------------------------------
    test_1: dict[str, Any] = {"per_fold": {}, "pooled_unseen": {}}
    for engine in engines:
        method = transfer["folds"][engine]["selected"]["method"]
        cell = applicability["folds"][engine]["correlation_with_reliability"][method]
        test_1["per_fold"][engine] = {
            "method": method,
            **{
                target: {
                    "within_unseen": cell[target]["within_unseen"],
                    "within_seen": cell[target]["within_seen"],
                    "pooled": cell[target]["pooled"],
                }
                for target in ("abs_risk_error", "bound_violation", "brier")
            },
            "detection_auroc": applicability["folds"][engine]["detection_diagnostic"][method][
                "auroc_unseen_is_less_applicable"
            ],
        }
    for method in ALL_METHODS:
        test_1["pooled_unseen"][method] = {
            target: applicability["pooled_unseen_environments"][method][target]
            for target in ("abs_risk_error", "bound_violation", "brier")
        }
    # The hypothesis predicts a NEGATIVE correlation, so "supportive" is an interval whose
    # UPPER end is below zero.
    # The rule reads the POOLED unseen correlation, not the per-fold one. Eight unseen
    # environments per fold is too few for an environment-clustered bootstrap: a large
    # fraction of resamples draw a set on which one rank is constant, Spearman is undefined
    # there, and the interval comes back NaN. Treating an uncomputable interval as "fails to
    # exclude zero" would let the sample size decide the finding. The 32 unseen environments
    # across the four folds are distinct and pool without repetition, so the interval exists
    # there; sign agreement across the four folds is required beside it so that a pooled
    # value driven by one engine cannot pass alone.
    per_method: dict[str, Any] = {}
    for method in APPLICABILITY_METHODS:
        estimates = [
            applicability["folds"][engine]["correlation_with_reliability"][method][
                "abs_risk_error"
            ]["within_unseen"]["estimate"]
            for engine in engines
        ]
        pooled_cell = applicability["pooled_unseen_environments"][method]["abs_risk_error"]
        per_method[method] = {
            "pooled_unseen": pooled_cell,
            "per_fold_estimates": dict(zip(engines, estimates, strict=True)),
            "negative_on_every_fold": bool(all(np.isfinite(v) and v < 0.0 for v in estimates)),
            "pooled_excludes_zero_negative": bool(
                np.isfinite(pooled_cell["ci_upper"])
                and pooled_cell["ci_upper"] < 0.0
                and not pooled_cell["degenerate_interval"]
            ),
        }
        per_method[method]["passes"] = bool(
            per_method[method]["negative_on_every_fold"]
            and per_method[method]["pooled_excludes_zero_negative"]
        )
    control = applicability["pooled_unseen_environments"]["random"]["abs_risk_error"]
    test_1["per_method"] = per_method
    test_1["methods_passing"] = [m for m in APPLICABILITY_METHODS if per_method[m]["passes"]]
    test_1["negative_control_pooled"] = control
    test_1["negative_control_is_null"] = bool(
        not (np.isfinite(control["ci_upper"]) and control["ci_upper"] < 0.0)
    )
    test_1["passes"] = bool(test_1["methods_passing"]) and test_1["negative_control_is_null"]
    test_1["rule"] = (
        "at least one deployable estimator must have a pooled within-unseen correlation with "
        "|risk error| whose interval lies entirely below zero, AND a negative point estimate "
        "on all four folds, AND the random-embedding control must not itself pass."
    )

    # --- test 2 ----------------------------------------------------------------------------
    test_2: dict[str, Any] = {"per_baseline": {}}
    for baseline in SUBSTANTIVE_BASELINES:
        per_engine = {}
        for engine in engines:
            comparison = transfer["folds"][engine]["comparisons_against_baselines"][baseline]
            delta = comparison["paired_repair_recall_delta"]
            matched = comparison["matched_coverage_harm_reduction"][
                f"coverage_{int(MATCHED_COVERAGES[1] * 100)}"
            ]
            per_engine[engine] = {
                "delta_repair_recall": delta["delta_repair_recall"],
                "ci_lower": delta["ci_lower"],
                "ci_upper": delta["ci_upper"],
                "degenerate_interval": delta["degenerate_interval"],
                "favours_proposed": _favourable(delta),
                "harm_reduction_at_matched_coverage": matched["harm_reduction"],
                "harm_reduction_ci": [matched["ci_lower"], matched["ci_upper"]],
                "harm_significantly_worse": bool(matched["ci_upper"] < 0.0),
            }
        test_2["per_baseline"][baseline] = {
            "per_engine": per_engine,
            "engines_favouring_proposed": [e for e in engines if per_engine[e]["favours_proposed"]],
            "engines_where_harm_is_worse": [
                e for e in engines if per_engine[e]["harm_significantly_worse"]
            ],
            "unanimous": all(per_engine[e]["favours_proposed"] for e in engines),
        }
    test_2["baselines_beaten_unanimously"] = [
        b for b in SUBSTANTIVE_BASELINES if test_2["per_baseline"][b]["unanimous"]
    ]
    test_2["passes"] = len(test_2["baselines_beaten_unanimously"]) == len(SUBSTANTIVE_BASELINES)

    # --- test 3 ----------------------------------------------------------------------------
    test_3 = {
        "per_fold": {
            engine: {
                "in_domain": transfer["folds"][engine]["transfer_gap"][PROPOSED_ARM]["in_domain"],
                "cross_engine": transfer["folds"][engine]["transfer_gap"][PROPOSED_ARM][
                    "cross_engine"
                ],
                "gap": transfer["folds"][engine]["transfer_gap"][PROPOSED_ARM]["gap"],
                "deployed": transfer["folds"][engine]["deployed_threshold"].get(PROPOSED_ARM, {}),
            }
            for engine in engines
        }
    }
    test_3["engines_holding_the_bound_when_deployed"] = [
        engine
        for engine in engines
        if transfer["folds"][engine]["deployed_threshold"]
        .get(PROPOSED_ARM, {})
        .get("empirical", {})
        .get("holds_bound", False)
    ]
    # Beating a baseline only matters if the arm reaches the unseen engine at all.
    test_3["passes"] = test_2["passes"] and all(
        test_3["per_fold"][engine]["cross_engine"] > 0.0 for engine in engines
    )

    # --- ablations -------------------------------------------------------------------------
    ablation_summary = {
        name: {
            "brief_label": label,
            "per_engine": {
                engine: ablation["folds"][engine]["ablations"][name][
                    "repair_recall_at_primary_epsilon"
                ]
                for engine in engines
                if name in ablation["folds"][engine]["ablations"]
            },
        }
        for name, label in ABLATION_LABELS.items()
    }
    proposed_by_engine = {
        engine: ablation["folds"][engine]["proposed"]["repair_recall_at_primary_epsilon"]
        for engine in engines
    }
    control_beats_proposed = [
        engine
        for engine in engines
        if ablation_summary["env_random"]["per_engine"].get(engine, -1.0)
        >= proposed_by_engine[engine]
    ]

    passes = test_1["passes"] and test_2["passes"] and test_3["passes"]
    verdict = "SUPPORTED" if passes else "NOT SUPPORTED"

    ceilings = {
        engine: {
            name: ablation["folds"][engine]["ceilings"][name]["repair_recall_at_primary_epsilon"]
            for name in ablation["folds"][engine]["ceilings"]
        }
        for engine in engines
    }
    best_baseline = {
        engine: max(
            SUBSTANTIVE_BASELINES,
            key=lambda b, e=engine: curves["folds"][e]["modes"][CROSS_ENGINE]["arms"][b][
                "frontier"
            ][key]["repair_recall"],
        )
        for engine in engines
    }

    cc._write_json_once(
        DECISION,
        {
            "schema_version": "sgv4-decision-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "hypothesis": (
                "An applicability estimate computed from unlabeled aggregate statistics of a "
                "batch of OCR output improves bounded-risk correction on an unseen engine "
                "relative to the confidence-only, harm-aware, shift-aware and "
                "uncertainty-aware policies."
            ),
            "verdict": verdict,
            "synthetic": False,
            "development_only": True,
            "decision_rule": (
                "SUPPORTED requires all three critical tests. Test 1: applicability correlates "
                "with the risk model's own error WITHIN the unseen environments, with an "
                "interval excluding zero on the predicted (negative) side, on every fold -- "
                "the within-group form, because the pooled form cannot be separated from "
                "engine detection. Test 2: the proposed arm beats EVERY substantive baseline "
                "on EVERY held-out engine with a favourable paired interval on the primary "
                "endpoint, and is not significantly more harmful at matched coverage. Test 3: "
                "the arm reaches a non-zero repair recall on every unseen engine. Unanimity "
                "rather than an average, for SGV2's reason: four folds are four experiments "
                "and a mean over them hides the engine where the method broke."
            ),
            "precondition_ranking_channel": precondition,
            "critical_test_1_applicability_tracks_reliability": test_1,
            "critical_test_2_decisions_improve": test_2,
            "critical_test_3_unseen_engine": test_3,
            "ablations": ablation_summary,
            "proposed_repair_recall_at_primary_epsilon": proposed_by_engine,
            "negative_control_matches_or_beats_proposed_on": control_beats_proposed,
            "ceilings": ceilings,
            "strongest_baseline_per_engine": best_baseline,
            "epsilon": PRIMARY_EPSILON,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: {verdict} -> {cc._relative(DECISION)}")
    return 0


# ------------------------------------------------------------------ figures


def run_figures() -> int:
    started = time.monotonic()
    environments = cc._read_json(ENVIRONMENT_SPEC)
    applicability = cc._read_json(APPLICABILITY_RESULTS)
    curves = cc._read_json(CURVE_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV4 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(transfer["folds"])
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    # --- environment_variance.png ----------------------------------------------------------
    # What the descriptor is actually measuring. A bar that is nearly all engine share is a
    # re-encoding of engine identity; only the middle band can reorder anything inside one
    # unseen engine.
    components = environments["descriptor_diagnostics"]["variance_components"]
    live = [n for n in ENV_SIGNAL_NAMES if not components[n]["constant"]]
    order = sorted(live, key=lambda n: -components[n]["between_engine_share"])
    figure, panel = plt.subplots(figsize=(13.5, 6.0))
    positions = np.arange(len(order))
    engine_share = np.array([components[n]["between_engine_share"] for n in order])
    environment_share = np.array([components[n]["between_environment_share"] for n in order])
    within_share = np.array([components[n]["within_environment_share"] for n in order])
    panel.bar(positions, engine_share, color="#3a5f9e", label="between engines")
    panel.bar(
        positions,
        environment_share,
        bottom=engine_share,
        color="#c87a2b",
        label="between environments of one engine",
    )
    panel.bar(
        positions,
        within_share,
        bottom=engine_share + environment_share,
        color="#bbb",
        label="between views of one environment (noise)",
    )
    panel.set_xticks(positions)
    panel.set_xticklabels(order, rotation=90, fontsize=6)
    panel.set_ylabel("share of the coordinate's variance")
    panel.set_ylim(0.0, 1.0)
    panel.legend(fontsize=8, loc="lower left")
    finish(
        figure,
        FIGURE_DIR / "environment_variance.png",
        "What each environment descriptor measures: the engine, the batch, or nothing",
    )

    # --- applicability_vs_reliability.png --------------------------------------------------
    # Drawn for ONE estimator across all four folds rather than for each fold's selected one.
    # The selection optimises the endpoint and picks a different estimator per fold, so a
    # per-fold panel would show four different instruments and could not be read as one
    # result. The fifth panel is the load-bearing form: the 32 unseen environments pooled,
    # which is where the interval in critical test 1 comes from.
    shown = "mahalanobis"
    figure, panels = plt.subplots(1, len(engines) + 1, figsize=(3.5 * (len(engines) + 1), 4.2))
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for panel, engine in zip(np.atleast_1d(panels)[:-1], engines, strict=True):
        cells = applicability["folds"][engine]["environment_cells"]
        for unseen, colour, label in ((False, "#9ab", "seen engines"), (True, "#a33", "unseen")):
            x = [c["applicability"][shown] for c in cells.values() if c["unseen"] == unseen]
            y = [c["abs_risk_error"] for c in cells.values() if c["unseen"] == unseen]
            panel.scatter(x, y, s=26, color=colour, label=label, alpha=0.85)
            if unseen:
                pooled_x.extend(x)
                pooled_y.extend(y)
        rho = applicability["folds"][engine]["correlation_with_reliability"][shown][
            "abs_risk_error"
        ]["within_unseen"]["estimate"]
        panel.set_title(f"held out: {engine}\nwithin-unseen rho = {rho:+.2f}", fontsize=9)
        panel.set_xlabel(f"applicability ({shown})")
        panel.set_xlim(-0.05, 1.05)
    pooled = applicability["pooled_unseen_environments"][shown]["abs_risk_error"]
    last = np.atleast_1d(panels)[-1]
    last.scatter(pooled_x, pooled_y, s=30, color="#a33", alpha=0.85)
    last.set_title(
        f"all {len(pooled_x)} unseen environments\n"
        f"rho = {pooled['estimate']:+.2f} "
        f"[{pooled['ci_lower']:+.2f}, {pooled['ci_upper']:+.2f}]",
        fontsize=9,
    )
    last.set_xlabel(f"applicability ({shown})")
    last.set_xlim(-0.05, 1.05)
    np.atleast_1d(panels)[0].set_ylabel(
        "|predicted harm - realized harm|\non the environment", fontsize=9
    )
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="upper right")
    finish(
        figure,
        FIGURE_DIR / "applicability_vs_reliability.png",
        "Critical test 1: does applicability rank environments by how wrong the risk model is? "
        "(the unseen points are the ones that are not engine detection)",
    )

    # --- coverage_risk_curve.png -----------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4), sharey=True)
    drawn = ("harm_only", "harm_aware", "selfaware", PROPOSED_ARM, "env_random")
    styles = {
        "harm_only": ("#888", "-"),
        "harm_aware": ("#3a5f9e", "-"),
        "selfaware": ("#2e7d5b", "--"),
        PROPOSED_ARM: ("#a33", "-"),
        "env_random": ("#c87a2b", ":"),
    }
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        arms = curves["folds"][engine]["modes"][CROSS_ENGINE]["arms"]
        for name in drawn:
            if name not in arms or "curve" not in arms[name]:
                continue
            colour, style = styles[name]
            panel.plot(
                arms[name]["curve"]["coverage"],
                arms[name]["curve"]["selective_risk"],
                style,
                color=colour,
                linewidth=1.6,
                label=name,
            )
        panel.axhline(PRIMARY_EPSILON, color="#a33", linestyle=":", linewidth=1)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("coverage (fraction of sites accepted)")
        panel.set_xlim(0.0, 1.0)
    np.atleast_1d(panels)[0].set_ylabel("selective risk (harmful accepted / accepted)")
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "coverage_risk_curve.png",
        "Risk against coverage on the held-out engine; the dotted line is the nominal bound",
    )

    # --- transfer_and_deployment.png -------------------------------------------------------
    # Left: what the arm could deliver at a hindsight-chosen cut. Right: what it does deliver
    # at the cut an operator would actually have. The two disagree, and that disagreement is
    # the deployment finding.
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    drawn = ("harm_only", "harm_aware", "selfaware", PROPOSED_ARM)
    width = 0.8 / len(drawn)
    positions = np.arange(len(engines))
    for offset, name in enumerate(drawn):
        left.bar(
            positions + offset * width,
            [transfer["folds"][e]["transfer_gap"][name]["cross_engine"] for e in engines],
            width,
            label=name,
        )
    left.set_xticks(positions + 0.4 - width / 2)
    left.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    left.set_ylabel(f"repair recall at harm <= {PRIMARY_EPSILON:g}")
    left.set_title("hindsight-chosen cut (the primary endpoint)", fontsize=9)
    left.legend(fontsize=7, ncol=2)
    for offset, name in enumerate(drawn):
        cells = [
            transfer["folds"][e]["deployed_threshold"].get(name, {}).get("empirical", {})
            for e in engines
        ]
        values = [c.get("realized_harm_rate", np.nan) for c in cells]
        right.bar(positions + offset * width, values, width, label=name)
        # An arm that accepts nothing has no harm rate and therefore no bar. Left unlabelled
        # that reads as "zero harm", which is the opposite of what it means.
        for index, cell in enumerate(cells):
            if cell.get("accepts_nothing"):
                right.text(
                    positions[index] + offset * width,
                    0.008,
                    "accepts nothing",
                    fontsize=6,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color="#a33",
                )
    right.axhline(PRIMARY_EPSILON, color="#a33", linestyle="--", linewidth=1.3)
    right.text(
        -0.35,
        PRIMARY_EPSILON + 0.012,
        f"nominal bound {PRIMARY_EPSILON:g}",
        fontsize=7,
        color="#a33",
    )
    right.set_xticks(positions + 0.4 - width / 2)
    right.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    right.set_ylabel("realized harm rate among accepted edits")
    right.set_title("source-calibrated threshold (what an operator would run)", fontsize=9)
    finish(
        figure,
        FIGURE_DIR / "transfer_and_deployment.png",
        "The same four arms judged two ways: what they could do, and what they do",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv4-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (
                    ENVIRONMENT_SPEC,
                    APPLICABILITY_RESULTS,
                    CURVE_RESULTS,
                    TRANSFER_RESULTS,
                    SCORES,
                )
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ provenance


def run_record() -> int:
    """What produced every artifact in this stage, with hashes."""
    started = time.monotonic()
    design = dg.load_design()
    produced = [
        path
        for path in (
            SCORES,
            SELECTION_RECORD,
            ENVIRONMENT_SPEC,
            APPLICABILITY_RESULTS,
            CURVE_RESULTS,
            TRANSFER_RESULTS,
            ABLATION_RESULTS,
            DECISION,
            FIGURE_MANIFEST,
        )
        if path.is_file()
    ]
    cc._write_json_once(
        FIT_RECORD,
        {
            "schema_version": "sgv4-environment-fit-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV4-E1",
            "representation": (
                "Phase 3's candidate-conditioned R1 design matrix, reused byte-for-byte from "
                "Phase 6, SGV2 and SGV3 rather than rebuilt, so all five stages score the "
                "same rows. The environment descriptor is an aggregate OF that matrix; no new "
                "row-level feature is introduced by this stage."
            ),
            "n_features": len(design.names),
            "environment_descriptor": {
                "n_coordinates": len(ENV_SIGNALS),
                "coordinates": list(ENV_SIGNAL_NAMES),
                "groups": {
                    group: [s.name for s in ENV_SIGNALS if s.group == group] for group in ENV_GROUPS
                },
                "shard_documents": SHARD_DOCUMENTS,
                "shard_seed": SHARD_SEED,
                "views_per_environment": ENV_VIEWS,
            },
            "applicability_methods": {
                "distance": list(DISTANCE_METHODS),
                "embedding": list(EMBEDDING_METHODS),
                "retrieval": list(RETRIEVAL_METHODS),
                "control": list(CONTROL_METHODS),
                "oracle": list(ORACLE_METHODS),
                "retrieval_target": RETRIEVAL_TARGET,
                "knn_neighbours": KNN_NEIGHBOURS,
                "retrieval_neighbours": RETRIEVAL_NEIGHBOURS,
                "pca_components": PCA_COMPONENTS,
                "clusters": CLUSTER_COUNT,
                "contrastive_components": CONTRASTIVE_COMPONENTS,
                "covariance_ridge": COVARIANCE_RIDGE,
                "scatter_ridge": SCATTER_RIDGE,
            },
            "selection": {
                "lambda_grid": list(LAMBDA_GRID),
                "gamma_grid": list(GAMMA_GRID),
                "scope": "inner leave-one-engine-out over the fit engines, CALIBRATION documents",
                "engine_identity_used_by_method": False,
            },
            "endpoints": {
                "primary": "repair_recall_at_bounded_harm (imported from Phase 5)",
                "co_primary": "harm reduction at matched site coverage (imported from SGV2)",
                "deployment": "realized harm at the source-calibrated threshold",
                "epsilon_grid": list(EPSILONS),
                "primary_epsilon": PRIMARY_EPSILON,
                "delta": DELTA,
            },
            "bootstrap": {
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": pilot.BOOTSTRAP_SEED,
                "unit": "document for row-level endpoints, environment for environment-level ones",
            },
            "ground_truth_used_for_environment_descriptors": False,
            "engine_identity_used_as_a_feature": False,
            "confirmatory_accessed": False,
            "inputs": {
                cc._relative(path): file_sha256(path)
                for path in (
                    pilot.CANDIDATE_TABLE,
                    pilot.LABEL_TABLE,
                    cc.FEATURES,
                    dg.DESIGN_MATRIX,
                    sa.SCORES,
                )
            },
            "artifacts": {cc._relative(path): file_sha256(path) for path in produced},
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"record: {len(produced)} artifacts -> {cc._relative(FIT_RECORD)}")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    stages = (
        ("environments", run_environments),
        ("scores", run_scores),
        ("applicability", run_applicability),
        ("curves", run_curves),
        ("transfer", run_transfer),
        ("ablation", run_ablation),
        ("figures", run_figures),
        ("decide", run_decide),
        ("record", run_record),
    )
    for name, _ in stages:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    selected = [function for name, function in stages if getattr(args, name)]
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

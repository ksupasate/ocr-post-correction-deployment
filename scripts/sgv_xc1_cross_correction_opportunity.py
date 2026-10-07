#!/usr/bin/env python3
"""SGV-XC1: does a stronger correction method raise the candidate opportunity ceiling?

SGV-CG1 measured the ceiling of the frozen correction candidate universe and found it low:
under PERFECT selection the universe repairs 0.0960 of evaluable OCR error sites, the candidate
loss is 0.9040 against 0.0695 for ranking and 0.0141 for deployment, and candidate generation
leads in 10 of 10 environments. Its decision artifact names the next investigation, and this is
it. The question is no longer how to rank or deploy the candidates that exist; it is whether
different correction families produce a materially better set of candidates in the first place.

    SGV-XC1: does replacing the frozen edit-aware generator with stronger correction families
    materially increase the set of OCR errors for which an exact repair is available?

**This stage is diagnostic and it stops at the candidate.** It has no PASS and no FAIL, it does
not rank, it does not deploy, and a higher ceiling here is necessary but not sufficient for a
better system (section 59). Seven decisions fix what the numbers below can mean.

**1. The comparison is opportunity, not accuracy.** Every method is reduced to the project's
atomic edit representation and scored on the same site denominator SGV-CG1 used. Comparing one
method as whole-text CER and another as candidate opportunity would confound generation with
ranking, thresholding and deployment, so nothing here is allowed to be measured end to end.

**2. Discovery and generation are separated everywhere.** A method that repairs well but
localizes badly and a method that localizes well but repairs badly are different findings with
different next stages, and section 44 exists to stop them being averaged into one number. Every
method reports DiscoveryRecall, GenerationSuccess and their product.

**3. Model inference never sees ground truth, and the code is structured so it cannot.** The
`--generate` phase builds its inputs from OCR artifacts alone, writes an immutable raw-output
cache, and never constructs an alignment or an index. Labelling happens afterwards in
`--normalize`, against SGV14's own labelling function. The leakage suite asserts that split
against the source rather than trusting this paragraph.

**4. One deterministic diff serves every generative method.** A corrector that returns a
rewritten line is converted to atomic edits by the repository's own span projection, snapped to
span boundaries so the region a candidate is scored on is a region the alignment can price. Per
method edit-extraction heuristics would make the comparison a comparison of heuristics.

**5. The harmful side is reported beside every ceiling.** A method can raise opportunity by
proposing more useful candidates or merely by proposing more candidates, and section 9's matched
candidate budget is what tells those apart. Nothing here reports an opportunity gain without the
candidate burden that bought it.

**6. Ground truth is allowed in three places and each is labelled.** Outcome labelling, the
oracle localization diagnostic and the perfect candidate oracle. Every artifact built with it
carries `analysis_only = true, uses_ground_truth = true`, and no oracle selects a method,
truncates a candidate set, filters a candidate or admits an environment.

**7. The method family is frozen before any outcome is read.** Section 3's arms, section 26's
prompt, section 27's decoding, section 23's meaningful margin and section 42's criteria are all
written to `design_record.json` and the registries before `--normalize` computes a single label.
An arm that could not be run reproducibly is recorded as unavailable there, not substituted for
another one afterwards.

    --reconstruct   the section-0 freeze: SGV15, SGV15b, SGV-DT1 and SGV-CG1 re-read and hashed
    --freeze        the frozen upstream configuration this stage consumes unchanged
    --preregister   methods, conditions, K, margins, criteria, questions and outcome taxonomy
    --registry      the method, model, prompt and decoding registries, frozen
    --schema        the common atomic edit representation
    --generate      GT-blind inference; the immutable raw-output cache
    --normalize     raw outputs -> atomic edits -> SGV14 labels
    --reproduce     the M0 reproduction gate against SGV-CG1, expected difference 0
    --discovery     site discovery per method
    --generation    conditional generation success per method
    --opportunity   the headline opportunity recall per method and environment
    --matchedk      the matched candidate budget analysis
    --harmful       harmful candidate burden and candidate multiplicity
    --strata        error type, engine and domain stratification
    --oracles       oracle localization, the candidate oracle and its risk-constrained form
    --overlap       pairwise complementarity, unique repairs and the union ceiling
    --nearmiss      secondary candidate distance diagnostics
    --shift         candidate distribution shift, for a later reliability transfer stage
    --cost          runtime and generation cost accounting
    --controls      the seven required controls
    --negative      the twelve falsification tests
    --stats         the pre-registered statistics
    --figures       the sixteen required figures
    --decide        the machine-readable research decision
    --determinism   two independent regenerations from the immutable cache
    --record        provenance and the traceability index

DEVELOPMENT ONLY. `ready_for_external_confirmation` is false by construction (section 58): a new
corrector changes the candidate distribution, so the reliability layer must be revalidated before
any confirmatory stage. The SGV1 CORD confirmatory reserve stays LOCKED and is absent from every
artifact. SGV13, SGV14, SGV15, SGV15b, SGV-DT1 and SGV-CG1 artifacts are read and hash-verified;
none is modified.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_verifier_pilot as pilot
import sgv13_fewshot_prefix_purity_adaptation as s13
import sgv14_confirmatory_validation as s14
import sgv15_target_risk_certification as s15
import sgv15b_document_level_certification as s15b
import sgv_cg1_opportunity_ceiling as cg1
import sgv_dt1_deployment_tradeoff as dt1
from ocr_risk.io.hashing import canonical_hash, file_sha256

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv_xc1_cross_correction_opportunity"
CACHE = OUT / "cache"
RAW_CACHE = OUT / "raw_model_outputs"
# Section 45. Oracle-localization inputs are built from ground truth, so their raw outputs live in
# their own directory and can never be picked up by a phase that reads the GT-blind cache.
RAW_ORACLE_CACHE = OUT / "raw_model_outputs_oracle_localization"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
DESIGN_RECORD = OUT / "design_record.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"

METHOD_REGISTRY = OUT / "method_registry.json"
MODEL_VERSIONS = OUT / "model_versions.json"
PROMPT_REGISTRY = OUT / "prompt_registry.json"
DECODING_REGISTRY = OUT / "decoding_registry.json"

ATOMIC_EDIT_SCHEMA = OUT / "atomic_edit_schema.json"
NORMALIZATION_RESULTS = OUT / "normalization_results.json"
# Controls and the declared out-of-scope diagnostic share one auxiliary table pair. No primary
# phase reads it; `load_arms` reaches it only when a caller names it.
AUXILIARY_CANDIDATES = OUT / "auxiliary_candidates.parquet"
AUXILIARY_LINKS = OUT / "auxiliary_links.parquet"
METHOD_CANDIDATES = OUT / "method_candidates.parquet"
METHOD_LINKS = OUT / "method_links.parquet"
ALIGNMENT_SITES = OUT / "alignment_sites.parquet"
ORACLE_CANDIDATES = OUT / "oracle_localization_candidates.parquet"
ORACLE_LINKS = OUT / "oracle_localization_links.parquet"
ORACLE_NORMALIZATION = OUT / "oracle_localization_normalization.json"
LLM_FORMAT_AUDIT = OUT / "llm_format_audit.json"

CANDIDATE_INVENTORY = OUT / "candidate_inventory_by_method.json"
SITE_INVENTORY = OUT / "site_inventory_by_method.json"
UPSTREAM_REPRODUCTION = OUT / "upstream_reproduction.json"

DISCOVERY_RESULTS = OUT / "discovery_results.json"
GENERATION_RESULTS = OUT / "generation_results.json"
OPPORTUNITY_RESULTS = OUT / "opportunity_results.json"
MATCHED_K_RESULTS = OUT / "matched_k_results.json"

HARMFUL_ANALYSIS = OUT / "harmful_candidate_analysis.json"
CANDIDATE_MULTIPLICITY = OUT / "candidate_multiplicity.json"

ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
ENGINE_ANALYSIS = OUT / "engine_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"

ORACLE_LOCALIZATION = OUT / "oracle_localization.json"
CANDIDATE_ORACLE = OUT / "candidate_oracle.json"
RISK_CANDIDATE_ORACLE = OUT / "risk_candidate_oracle.json"

METHOD_OVERLAP = OUT / "method_overlap.json"
UNIQUE_REPAIRS = OUT / "unique_repairs.json"
UNION_OPPORTUNITY = OUT / "union_opportunity.json"

NEAR_MISS_ANALYSIS = OUT / "near_miss_analysis.json"
CANDIDATE_DISTRIBUTION_SHIFT = OUT / "candidate_distribution_shift.json"
RUNTIME_COST = OUT / "runtime_cost.json"

CONTROL_RESULTS = OUT / "control_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"

DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

REPORT = REPO / "docs/sgv_xc1/cross_correction_opportunity.md"
NORMALIZATION_DOC = REPO / "docs/sgv_xc1/atomic_edit_normalization.md"

SCHEMA_VERSION = 1
STAGE = "sgv_xc1_cross_correction_opportunity"
HYPOTHESIS = "SGV-XC1-D1"

# ------------------------------------------------------------------ imported, never restated

PhaseError = s15.PhaseError
_relative = s15._relative
_git = s15._git
_write_json_once = s15._write_json_once
_write_parquet_once = s15._write_parquet_once
_ignored = s15._ignored
_tracked = s15._tracked
cc_read_json = s15.cc_read_json
_mean = s15._mean
holm = s15.holm

EPSILONS = s15.EPSILONS
PRIMARY_EPSILON = s15.PRIMARY_EPSILON
BOOTSTRAP_RESAMPLES = s13.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = s13.BOOTSTRAP_SEED
ALPHA = 0.05
ENVIRONMENT_COUNT = 10
UNLABELLED_DISTANCE = s14.UNLABELLED_DISTANCE
CONTEXT_CHARS = s14.CONTEXT_CHARS
UNION_CAP = s14.UNION_CAP

# Section 0. The frozen upstream state, re-read from every saved decision artifact.
SGV15_STATE = cg1.SGV15_STATE
SGV15B_STATE = cg1.SGV15B_STATE
SGV_DT1_STATE = cg1.SGV_DT1_STATE
SGV_CG1_STATE = ("CANDIDATE-GENERATION BOTTLENECK -- outcome A", "A", True, False)

# Section 0. SGV-CG1's headline quantities, confirmed from its artifacts rather than the brief.
CG1_OPPORTUNITY = "candidate_opportunity_mean"
CG1_EXPECTED = {
    "alignment_error_sites_evaluation": 32548,
    "discovered_sites_evaluation": 21283,
    "candidate_rows": 73001,
}

UPSTREAM_SCRIPTS = (
    "scripts/sgv13_fewshot_prefix_purity_adaptation.py",
    "scripts/sgv14_confirmatory_validation.py",
    "scripts/sgv15_target_risk_certification.py",
    "scripts/sgv15b_document_level_certification.py",
    "scripts/sgv_dt1_deployment_tradeoff.py",
    "scripts/sgv_cg1_opportunity_ceiling.py",
)

# ------------------------------------------------------------------ the pre-registered registry
#
# Everything from here to the end of this block is section 3, 9, 23, 24, 26, 27, 42 and 43. It is
# written to `design_record.json` and the four registries by `--preregister` and `--registry`,
# both of which refuse to run once `--normalize` has produced a label.

M0 = "m0_frozen_edit_aware"
M1 = "m1_byt5_seq2seq"
M2 = "m2_local_llm"
M3 = "m3_multimodal"
M_UNION = "m_union"

# Section 3. The primary family. `m_union` (section 4/31) is an analysis-only diagnostic and is
# never counted as one of the tested correction families.
PRIMARY_METHODS = (M0, M1, M2)
ALL_METHODS = (M0, M1, M2)

# Section 7 and 46. Three generation conditions. `frozen_localization` is section 7's recommended
# primary: every method is handed the same OCR-only localization the frozen pipeline uses, so
# discovery is identical by construction and only generation can differ. `natural` lets a method
# localize for itself, which is the only condition in which DiscoveryRecall can move. Section 45's
# `oracle_localization` supplies the true error region and withholds the correct text; it is
# analysis-only and may never select a method.
COND_FROZEN = "frozen_localization"
COND_NATURAL = "natural"
COND_ORACLE_LOC = "oracle_localization"
CONDITIONS = (COND_FROZEN, COND_NATURAL, COND_ORACLE_LOC)
PRIMARY_CONDITION = COND_FROZEN

# Section 9 and 40. The matched candidate budget. K=5 is section 9's recommended primary and is
# fixed here, before any outcome is read.
K_GRID = (1, 3, 5, 10)
PRIMARY_K = 5

# Section 23. A scientifically meaningful opportunity gain, declared against SGV-CG1's frozen
# 0.0960 baseline. A method must move the ceiling by at least this much in absolute terms; any
# positive delta is explicitly NOT success.
MEANINGFUL_DELTA = 0.10
# Section 24. The same margin applied within one error type, so "materially improve" is a fixed
# quantity rather than a judgement made after the table is read.
WEAK_TYPE_DELTA = 0.10
# Section 24. The three categories SGV-CG1 found the frozen generator largely misses. The repo's
# frozen taxonomy calls an OCR omission `deletion`; the brief calls it omission. Same category.
WEAK_ERROR_TYPES = ("substitution", "deletion", "segmentation")
WEAK_TYPES_REQUIRED = 2
STRONG_ERROR_TYPE = "insertion"

# Section 42. Breadth, over the ten frozen environments and the two document domains.
BREADTH_MAJORITY = 6
DOMAINS_REQUIRED = 2

# Section 42 criterion 4. The gain must survive an equalized candidate budget: a method whose
# advantage exists only when it is allowed more candidates than the baseline has not raised the
# ceiling, it has bought coverage with burden.
BURDEN_CONDITION_K = PRIMARY_K

# Section 43. Exactly one outcome is assigned.
OUTCOME_TAXONOMY = {
    "A": "strong replacement candidate",
    "B": "complementary correction families",
    "C": "improvement is narrow",
    "D": "no correction family meaningfully raises the ceiling",
}

# Section 41. The six pre-registered questions.
QUESTIONS = {
    "Q1": "Does any correction family materially exceed the frozen candidate opportunity ceiling?",
    "Q2": (
        "Is the improvement caused mainly by better site discovery or better correction "
        "generation after discovery?"
    ),
    "Q3": (
        "Do alternative methods repair substitutions, omissions and segmentation errors that the "
        "current generator largely misses?"
    ),
    "Q4": "Does improved opportunity require an unacceptable explosion in harmful candidates?",
    "Q5": "Are correction methods complementary?",
    "Q6": "Does the best method improve across engines and domains, or only in narrow ones?",
}

# Section 48. The controls.
CONTROLS = (
    "c0_frozen_m0_reproduction",
    "c1_identity_corrector",
    "c2_random_text_perturbation",
    "c3_ground_truth_oracle_corrector",
    "c4_shuffled_method_outputs",
    "c5_candidate_duplication",
    "c6_candidate_truncation",
)
CONTROL_SEED = 20260908

# Section 27. Decoding is frozen here and never searched on an outcome. Beam search with ten
# returned sequences is one pass that serves both the natural condition (the full returned set)
# and every matched-K truncation (a prefix of it), so K is a slice of one frozen decode rather
# than four differently-configured runs.
BEAM_WIDTH = 10
RETURN_SEQUENCES = 10
BYT5_MAX_NEW_TOKENS = 64
BYT5_LINE_MAX_NEW_TOKENS = 320
LLM_MAX_NEW_TOKENS = 96
LLM_LINE_MAX_NEW_TOKENS = 320
GENERATION_BATCH = 32
LINE_BATCH = 8
# Section 40. A whole-line corrector naturally returns one corrected string. Beam-searching ten
# alternative rewritings of a line and calling them ten candidates would fabricate a candidate
# budget the method does not have, so the natural condition decodes greedily and records
# `supports_multi_candidate = false` rather than inventing K values for it.
LINE_BEAMS = 1
LINE_RETURN_SEQUENCES = 1
# Section 27 and 40. The LLM arm decodes greedily and returns one sequence in every condition, so
# it records `supports_multi_candidate = false` rather than being handed fabricated K values. The
# byte model's ten-sequence beam search is not repeated for it: the format audit's measured
# throughput, not any outcome, is the recorded reason. The output cap is a function of the input
# alone -- correcting an N-character fragment needs at most about N tokens plus a margin, and a
# longer answer is a rewrite or an echo -- and requests are sorted by prompt length so that a batch
# pads little. Both orderings are deterministic and both are part of the arm's definition.
LLM_BEAMS = 1
LLM_RETURN_SEQUENCES = 1
LLM_BATCH = 32
LLM_REGION_TOKEN_MARGIN = 8
LLM_LINE_TOKEN_MARGIN = 16

# Section 26. The LLM prompt, frozen before endpoint evaluation. It instructs the model to
# preserve text unless a correction is justified, to answer in a machine-readable form, to avoid
# explanation and to avoid inventing content. It contains no example drawn from any evaluation
# document and no ground truth.
LLM_SYSTEM_PROMPT = (
    "You correct OCR errors. You are given a fragment of text produced by an OCR engine, marked "
    "with [[ and ]]. Follow these rules exactly:\n"
    "1. Preserve the text unchanged unless a correction is clearly justified by an OCR error.\n"
    "2. Answer with the corrected fragment inside [[ and ]], and nothing else.\n"
    "3. Never explain, never add labels, and never repeat the surrounding text.\n"
    "4. Preserve capitalisation, punctuation and spacing where they are correct.\n"
    "5. Never invent words or facts that the fragment and its surroundings do not support.\n"
    "6. If the fragment is only the space between two words, answer with any text missing there, "
    "or with the space unchanged if nothing is missing."
)
LLM_CONTEXT_TEMPLATE = (
    "Text before the fragment: {before}\n"
    "Fragment: [[{region}]]\n"
    "Text after the fragment: {after}\n\n"
    "Corrected fragment:"
)
LLM_LINE_TEMPLATE = "OCR line: [[{region}]]\n\nCorrected line:"
PROMPT_REVISION = 1
# The superseded draft, kept verbatim so the revision is auditable rather than asserted.
PROMPT_DRAFT_0 = {
    "system": (
        "You correct OCR errors. You are given a fragment of text produced by an OCR engine. "
        "Return the corrected fragment and nothing else. Follow these rules exactly:\n1. Preserve "
        "the text unchanged unless a correction is clearly justified by an OCR error.\n2. Return "
        "only the corrected fragment, with no quotes, no labels and no explanation.\n3. Do not "
        "add, remove or reorder content that is not an OCR error.\n4. Preserve capitalisation, "
        "punctuation and layout characters where they are correct.\n5. Never invent words or "
        "facts that the fragment does not support."
    ),
    "region_template": "OCR fragment:\n{region}\n\nCorrected fragment:",
    "context_template": (
        "Surrounding OCR text (context only, do not return "
        "it):\n{before}<<<{region}>>>{after}\n\nOCR fragment to correct:\n{region}\n\nCorrected "
        "fragment:"
    ),
    "line_template": "OCR line:\n{region}\n\nCorrected line:",
}
PROMPT_REVISION_REASON = (
    "the first draft asked for only the corrected fragment but specified no machine-readable "
    "format, which section 26 point 2 requires. A small GT-blind check on requests from one "
    "evaluation environment, run before any label existed, showed the draft returning the "
    "surrounding context instead of the fragment and running to the output cap. It was replaced "
    "once by a delimited format. The replacement was checked on adaptation-split documents only "
    "(llm_format_audit.json) and frozen. No label was read at any point and no second revision "
    "was made."
)

# Section 3 and 28. The model specifications, pinned by revision. `available` is decided by
# `--registry` from the filesystem and the import, and recorded before any label exists.
BYT5_REPO = "yelpfeast/byt5-base-english-ocr-correction"
BYT5_REVISION = "19d5c2fd86b87f0a0febb7d2574878a0d68d5294"
LLM_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# Section 3. The declared language scope of each arm. A checkpoint fine-tuned on English alone is
# not a German corrector, and running it on Fraktur would measure a language mismatch and report
# it as generator quality. Scope is a property of the arm, declared before any result.
LANGUAGE_OF_CORPUS = {"funsd": "eng", "ocrd_sbb": "deu"}
METHOD_LANGUAGES = {
    M0: frozenset({"eng", "deu"}),
    M1: frozenset({"eng"}),
    M2: frozenset({"eng", "deu"}),
}
# Section 3. Outside its declared languages an arm proposes nothing -- `Byt5Generator.applies_to`
# returns nothing there -- so its primary arm records zero opportunity rather than a missing value.
# One environment outside M1's scope is also generated, as a labelled diagnostic of what the
# language mismatch costs. Its outputs go to the auxiliary table and never enter M1's primary arm.
OUT_OF_SCOPE_DIAGNOSTIC: dict[str, tuple[str, ...]] = {M1: ("sbb/doctr",)}
DIAGNOSTIC_SUFFIX = "__out_of_scope_diagnostic"
# Section 28. The LLM's inference dtype. Recorded in the model and decoding registries.
LLM_DTYPE = "bfloat16"

# Section 3 M3. Recorded before endpoint evaluation, with the reason, so that no multimodal model
# can be substituted in after M1 and M2 results are visible.
M3_UNAVAILABLE_REASON = (
    "no reproducible image-aware correction arm is available in this environment: no multimodal "
    "checkpoint is present in the local model cache, the repository ships "
    "`MultimodalRewriterBaseline` as a declared-but-unimplemented stub, and no image-conditioned "
    "correction path exists that could be pinned by revision and regenerated from an immutable "
    "cache. Section 3 requires this to be recorded before endpoint evaluation rather than "
    "substituted for afterwards."
)

# Section 12. Exact repair stays the headline. `partial_improvement` is reported beside it and
# never inside it.
OUTCOME_EXACT = cg1.OUTCOME_EXACT
OUTCOME_PARTIAL = cg1.OUTCOME_PARTIAL
ERROR_OUTCOMES = cg1.ERROR_OUTCOMES

# Section 8. The atomic edit kinds a diff may produce. `gap` is the only kind that can repair an
# OCR omission, which is one of the three categories section 24 cares about, so a diff that
# refused to emit it would make one of the study's primary questions unanswerable.
EDIT_REGION = "region"
EDIT_GAP = "gap"

# Section 48. The controls. C1, C2 and C4 produce candidates, so they pass through the same
# normalization and the same labelling call as the real arms and are written to their own tables;
# no primary analysis can pick one up by accident. Their generation rules and the operational
# form of each expectation are frozen into the method registry before any label exists.
C1_IDENTITY = "c1_identity_corrector"
C2_RANDOM = "c2_random_text_perturbation"
C4_PREFIX = "c4_shuffled_"
C2_PER_SITE = 3
PERTURBATION_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
CONTROL_SPECS = {
    "c0_frozen_m0_reproduction": {
        "rule": "M0 re-run through this stage's own reconstruction, labelling and linking",
        "expectation": "zero differences against SGV-CG1 on every section 5 check",
    },
    "c1_identity_corrector": {
        "rule": "every request, in both GT-blind conditions, answered with its own OCR text",
        "expectation": "no proposal survives normalization, so zero new exact repairs",
    },
    "c2_random_text_perturbation": {
        "rule": (
            f"{C2_PER_SITE} seeded single-character edits per frozen site, drawn from the "
            "region's own characters and [a-z0-9]; reads the OCR region and the seed only"
        ),
        "expectation": (
            "mean opportunity recall below M0's, and harmful candidates outnumbering "
            "beneficial ones"
        ),
    },
    "c3_ground_truth_oracle_corrector": {
        "rule": (
            "analysis only: at each oracle-localized error region, propose the region ground "
            "truth the frozen labeller itself reads"
        ),
        "expectation": (
            "opportunity recall 1.0 on every error site the region representation can express "
            "and label; any shortfall is counted and explained, never forced"
        ),
    },
    "c4_shuffled_method_outputs": {
        "rule": (
            "each arm's frozen-condition outputs reassigned to a seeded permutation of the "
            "environment's frozen requests"
        ),
        "expectation": "mean opportunity recall below the same arm's unshuffled value",
    },
    "c5_candidate_duplication": {
        "rule": "every candidate of an arm duplicated at the analysis layer",
        "expectation": "opportunity recall unchanged while the candidate count doubles",
    },
    "c6_candidate_truncation": {
        "rule": "candidate sets truncated to K in the arm's own frozen order",
        "expectation": "opportunity recall non-increasing as K shrinks",
    },
}


# ------------------------------------------------------------------ small shared helpers


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_xc1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _oracle_envelope(artifact: str, *, oracle_localization: bool = False) -> dict[str, Any]:
    """Section 36. Every artifact built with ground truth declares it in its own header."""
    record = {
        **_envelope(artifact),
        "analysis_only": True,
        "uses_ground_truth": True,
        "may_select_a_method": False,
        "may_truncate_a_candidate_set": False,
        "may_filter_a_candidate": False,
        "may_admit_or_exclude_an_environment": False,
        "may_influence_a_prompt_or_decoding_setting": False,
    }
    if oracle_localization:
        record["oracle_localization"] = True
    return record


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not len(values):
        return {k: float("nan") for k in ("mean", "median", "p90", "p95", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def environment_names() -> list[str]:
    return [spec["environment"] for spec in s14.ENVIRONMENTS]


def _slug(name: str) -> str:
    return name.replace("/", "__")


def language_of(corpus: str) -> str:
    return LANGUAGE_OF_CORPUS[corpus]


def method_applies(method: str, corpus: str) -> bool:
    """Whether an arm's declared language scope covers this corpus. Declared, not discovered.

    A control has no language scope: it is a test of the pipeline, not a corrector.
    """
    languages = METHOD_LANGUAGES.get(method)
    return True if languages is None else language_of(corpus) in languages


def _cg1(artifact: str) -> dict[str, Any]:
    return cc_read_json(cg1.OUT / f"{artifact}.json")


def _require(path: Path, phase: str) -> None:
    if not path.is_file():
        raise PhaseError(f"run --{phase} first ({_relative(path)} is missing)")


def _forbid(path: Path) -> None:
    if path.exists():
        raise PhaseError(f"{_relative(path)} already exists; delete it deliberately")


# ------------------------------------------------------------------ section 0: reconstruct


def run_reconstruct() -> int:
    """Re-read the four frozen decisions this stage is built on, and hash their artifacts.

    Section 0 is explicit that repository evidence is authoritative and that this prompt's
    recollection of it is not. Every verdict, count and headline quantity below is read from a
    saved artifact and compared against the state this stage was designed for; a disagreement
    stops the stage rather than being reconciled in prose.
    """
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    _forbid(RESEARCH_FREEZE)

    states = {
        "sgv15": (
            cg1._decision_state(s15.DECISION, ("verdict", "criteria_met", "ready_for_sgv16")),
            SGV15_STATE,
        ),
        "sgv15b": (
            cg1._decision_state(s15b.DECISION, ("verdict", "criteria_met", "ready_for_sgv16")),
            SGV15B_STATE,
        ),
        "sgv_dt1": (
            cg1._decision_state(
                dt1.DECISION, ("verdict", "criteria_met", "ready_for_external_confirmation")
            ),
            SGV_DT1_STATE,
        ),
    }
    mismatches: list[str] = []
    upstream: dict[str, Any] = {}
    for name, (observed, expected) in states.items():
        if observed != expected:
            mismatches.append(f"{name}: observed {observed!r} != expected {expected!r}")
        upstream[name] = {
            "verdict": observed[0],
            "criteria_met": observed[1],
            "criteria_total": observed[2],
            "outcome": observed[3],
            "onward_gate": observed[4],
        }

    decision = _cg1("research_decision")
    observed_cg1 = (
        str(decision["verdict"]),
        str(decision["outcome"]),
        bool(decision["ready_for_cross_correction_method_study"]),
        bool(decision["ready_for_external_confirmation"]),
    )
    if observed_cg1 != SGV_CG1_STATE:
        mismatches.append(f"sgv_cg1: observed {observed_cg1!r} != expected {SGV_CG1_STATE!r}")
    if not decision["ready_for_cross_correction_method_study"]:
        mismatches.append("sgv_cg1 does not authorize a cross-correction-method study")

    # Section 0's expected headline values, confirmed against SGV-CG1's own artifacts. The
    # opportunity quantities are carried at full precision; the brief's four-decimal forms are
    # what they round to, and the comparison is made at that tolerance rather than on the string.
    inventory = _cg1("environment_inventory")
    counts = {key: int(inventory["totals"][key]) for key in CG1_EXPECTED}
    for key, expected_count in CG1_EXPECTED.items():
        if counts[key] != expected_count:
            mismatches.append(f"sgv_cg1 {key}: observed {counts[key]} != expected {expected_count}")

    headline = {
        "candidate_opportunity_mean": float(decision["candidate_opportunity_mean"]),
        "discovery_recall_mean": float(decision["discovery_recall_mean"]),
        "opportunity_recall_given_discovery_mean": float(
            decision["opportunity_recall_given_discovery_mean"]
        ),
        "candidate_gap": float(decision["candidate_gap"]),
        "ranking_gap": float(decision["ranking_gap"]),
        "deployment_gap": float(decision["deployment_gap"]),
        "environments_candidate_dominant": int(decision["environments_candidate_dominant"]),
    }
    expected_headline = {
        "candidate_opportunity_mean": 0.0960,
        "discovery_recall_mean": 0.3342,
        "opportunity_recall_given_discovery_mean": 0.2270,
        "candidate_gap": 0.9040,
        "ranking_gap": 0.0695,
        "deployment_gap": 0.0141,
    }
    for key, expected_value in expected_headline.items():
        if abs(headline[key] - expected_value) > 5e-5:
            mismatches.append(
                f"sgv_cg1 {key}: observed {headline[key]:.6f} != expected {expected_value}"
            )
    if headline["environments_candidate_dominant"] != ENVIRONMENT_COUNT:
        mismatches.append("sgv_cg1 candidate dominance is not 10 of 10")

    # Section 0 also asks for the frozen error-type pattern: the generator is much stronger on
    # spurious insertion than on the other three categories. Confirmed, not assumed.
    error_types = _cg1("error_type_analysis")
    by_type = {
        str(row["site_kind"]): float(row["opportunity_recall"])
        for row in error_types["per_error_type"]
    }
    if not all(by_type[STRONG_ERROR_TYPE] > by_type[weak] * 10 for weak in WEAK_ERROR_TYPES):
        mismatches.append(f"sgv_cg1 error-type pattern is not as section 0 describes: {by_type}")

    if mismatches:
        raise PhaseError(
            "section 0 reconstruction failed; SGV-XC1 must not begin:\n  " + "\n  ".join(mismatches)
        )

    artifacts = {
        _relative(path): file_sha256(path)
        for path in sorted(cg1.OUT.glob("*.json")) + sorted(cg1.OUT.glob("*.parquet"))
    }
    scripts = {name: file_sha256(REPO / name) for name in UPSTREAM_SCRIPTS}
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "head": _git("rev-parse", "HEAD"),
            "upstream_decisions": upstream,
            "sgv_cg1": {
                "verdict": observed_cg1[0],
                "outcome": observed_cg1[1],
                "outcome_label": str(decision["outcome_label"]),
                "ready_for_cross_correction_method_study": observed_cg1[2],
                "ready_for_external_confirmation": observed_cg1[3],
                "next_investigation": str(decision["next_investigation"]),
                "counts": counts,
                "headline": headline,
                "opportunity_recall_by_error_type": by_type,
                "low_ceiling_threshold": float(decision["low_ceiling_threshold"]),
            },
            "upstream_scripts": scripts,
            "sgv_cg1_artifacts": artifacts,
            "confirmatory_reserve_consumed": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"reconstruct: SGV15 {SGV15_STATE[0]} {SGV15_STATE[1]}/{SGV15_STATE[2]}, "
        f"SGV15b {SGV15B_STATE[1]}/{SGV15B_STATE[2]}, SGV-DT1 outcome {SGV_DT1_STATE[3]}, "
        f"SGV-CG1 outcome {observed_cg1[1]} at opportunity "
        f"{headline['candidate_opportunity_mean']:.4f} -> {len(artifacts)} artifacts hashed"
    )
    return 0


# ------------------------------------------------------------------ section 0: freeze


def run_freeze() -> int:
    """The upstream configuration this stage consumes unchanged.

    SGV-XC1 varies exactly one thing: which correction family proposes the candidate. Everything
    else -- the corpora, the ten environments, the OCR-only discovery enumerator, the alignment,
    the site construction, the labelling function and the harm policy -- is the frozen pipeline,
    recorded here so that a later reader can tell what was held fixed without re-deriving it.
    """
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "reconstruct")
    _forbid(FROZEN_CONFIGURATION)

    from ocr_risk.config.models import AlignmentConfig, SiteConfig
    from ocr_risk.discovery.enumerator import DiscoveryRules

    alignment_config = AlignmentConfig().model_dump(mode="json")
    site_config = SiteConfig().model_dump(mode="json")
    rules = DiscoveryRules()
    discovery_rules = {
        field: getattr(rules, field)
        for field in sorted(getattr(rules, "__dataclass_fields__", {}))
        if isinstance(getattr(rules, field), (int, float, str, bool))
    }

    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "environments": [
                {
                    "environment": spec["environment"],
                    "corpus": spec["corpus"],
                    "base_engine": spec["base_engine"],
                    "engine_id": spec["engine_id"],
                    "language": language_of(spec["corpus"]),
                }
                for spec in s14.ENVIRONMENTS
            ],
            "corpora": sorted({spec["corpus"] for spec in s14.ENVIRONMENTS}),
            "alignment_config": alignment_config,
            "site_config": site_config,
            "discovery_rules": discovery_rules,
            "frozen_generator": {
                "primary_generator": s14.PRIMARY_GENERATOR,
                "union_cap": UNION_CAP,
                "context_chars": CONTEXT_CHARS,
                "components": ["g3_edit_aware", "g7_structural_v2"],
                "note": (
                    "the SGV-CG1 candidate universe. M0 re-runs exactly this and section 5 "
                    "requires it to reproduce SGV-CG1 with zero differences."
                ),
            },
            "labelling": {
                "function": "ocr_risk.experiments.cgv3_study.label_candidate",
                "harm_policy": "STRICT_WORSENING",
                "exact_repair_outcome": OUTCOME_EXACT,
                "unlabelled_distance": UNLABELLED_DISTANCE,
                "note": (
                    "section 4's label definitions are SGV14's and are not redefined here. Every "
                    "method's candidates are labelled by the same function on the same region "
                    "ground truth."
                ),
            },
            "evaluation_split": {
                "role": s14.ROLE_EVALUATION,
                "note": (
                    "the held-out-engine and held-out-document partition is SGV14's global "
                    "partition; SGV-XC1 neither recomputes nor widens it."
                ),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"freeze: {len(s14.ENVIRONMENTS)} environments, generator {s14.PRIMARY_GENERATOR} "
        f"(cap {UNION_CAP}, context {CONTEXT_CHARS}), labelling by SGV14 -> "
        f"{_relative(FROZEN_CONFIGURATION)}"
    )
    return 0


# ------------------------------------------------------------------ section 3, 23, 42: preregister


def run_preregister() -> int:
    """Freeze the design before a single outcome label exists.

    Sections 23, 24, 27 and 42 all say the same thing in different words: the criterion is chosen
    first. This phase refuses to run once `--normalize` has produced labels, so the order is
    enforced by the filesystem rather than promised in prose, and `--record` checks that this
    artifact's timestamp precedes every endpoint artifact's.
    """
    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    _forbid(DESIGN_RECORD)
    if METHOD_CANDIDATES.exists():
        raise PhaseError("labels already exist; the design cannot be registered after the fact")

    baseline = float(_cg1("research_decision")["candidate_opportunity_mean"])
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "head": _git("rev-parse", "HEAD"),
            "stage_kind": "DEVELOPMENT / DIAGNOSTIC",
            "question": (
                "does replacing the frozen edit-aware OCR candidate generator with stronger "
                "correction families materially increase the set of OCR errors for which an "
                "exact repair is available?"
            ),
            "non_goals": [
                "which complete OCR system has the best end-to-end accuracy",
                "whether the reliability layer can rank the new candidates",
                "whether any correction family is deployable",
                "any change to thresholds, certification or human-review policy",
            ],
            "baseline": {
                "method": M0,
                "source": _relative(cg1.OUT / "research_decision.json"),
                "opportunity_recall": baseline,
                "note": (
                    "SGV-CG1's frozen candidate opportunity ceiling, the value every arm is "
                    "measured against."
                ),
            },
            "methods": {
                "primary_family": list(PRIMARY_METHODS),
                "analysis_only": [M_UNION],
                "declared_unavailable": {M3: M3_UNAVAILABLE_REASON},
                "language_scope": {m: sorted(METHOD_LANGUAGES[m]) for m in PRIMARY_METHODS},
            },
            "conditions": {
                "all": list(CONDITIONS),
                "primary": PRIMARY_CONDITION,
                COND_FROZEN: (
                    "section 7's recommended fair condition: every method receives the same "
                    "frozen OCR-only localization the current candidate pipeline uses, so "
                    "discovery is identical by construction and only generation can differ."
                ),
                COND_NATURAL: (
                    "the method localizes for itself. Whole-text correctors rewrite the OCR line "
                    "and the rewrite is diffed into atomic edits. This is the only condition in "
                    "which DiscoveryRecall can move."
                ),
                COND_ORACLE_LOC: (
                    "section 45, analysis only: the true OCR error region is supplied and the "
                    "correct text is withheld. It separates a localization limit from a "
                    "generation limit and may never select a method."
                ),
            },
            "metrics": {
                "discovery_recall": (
                    "OCR error sites touched by the method / evaluable OCR error sites"
                ),
                "generation_success": (
                    "discovered error sites with an exact repair candidate / discovered error sites"
                ),
                "opportunity_recall": "repairable OCR error sites / evaluable OCR error sites",
                "identity": "opportunity_recall == discovery_recall * generation_success",
                "primary_unit": "OCR error site",
                "exact_repair_outcome": OUTCOME_EXACT,
                "note": (
                    "section 12: exact repair stays the headline. Near-miss distance is a "
                    "secondary diagnostic and never enters the primary metric."
                ),
            },
            "candidate_budget": {
                "k_grid": list(K_GRID),
                "primary_k": PRIMARY_K,
                "natural_condition": "each method's frozen decoding configuration, untruncated",
                "note": (
                    "section 9: a generative method can emit arbitrarily many candidates, so the "
                    "headline comparison is repeated at an equalized budget."
                ),
            },
            "thresholds": {
                "meaningful_delta_opportunity": MEANINGFUL_DELTA,
                "weak_error_type_delta": WEAK_TYPE_DELTA,
                "weak_error_types": list(WEAK_ERROR_TYPES),
                "weak_error_types_required": WEAK_TYPES_REQUIRED,
                "breadth_majority": BREADTH_MAJORITY,
                "environment_count": ENVIRONMENT_COUNT,
                "domains_required": DOMAINS_REQUIRED,
                "burden_condition_k": BURDEN_CONDITION_K,
                "alpha": ALPHA,
                "note": (
                    "section 23 is explicit that any positive delta is not success. The margin is "
                    "absolute and is set against a baseline of "
                    f"{baseline:.4f}."
                ),
            },
            "criteria": {
                "c1_opportunity_improvement": (
                    f"at least one arm improves overall opportunity recall by >= "
                    f"{MEANINGFUL_DELTA} in the primary condition"
                ),
                "c2_breadth": (
                    f"that improvement holds on >= {BREADTH_MAJORITY} of {ENVIRONMENT_COUNT} "
                    "environments"
                ),
                "c3_weak_error_type_recovery": (
                    f">= {WEAK_TYPES_REQUIRED} of {list(WEAK_ERROR_TYPES)} improve by >= "
                    f"{WEAK_TYPE_DELTA}"
                ),
                "c4_candidate_burden": (
                    f"the gain survives the equalized budget at K={BURDEN_CONDITION_K}, so it is "
                    "not explained by candidate explosion alone"
                ),
                "c5_domain_robustness": (
                    f"the improvement holds in both of the {DOMAINS_REQUIRED} document domains"
                ),
                "c6_reproducibility": (
                    "the arm's outputs regenerate identically from the immutable raw-output cache"
                ),
            },
            "questions": QUESTIONS,
            "outcome_taxonomy": OUTCOME_TAXONOMY,
            "outcome_rule": {
                "A": "c1 and c2 and c3 and c4 and c5 -- a strong replacement candidate",
                "B": (
                    "not A, and the union of the arms exceeds the best single arm by >= "
                    f"{MEANINGFUL_DELTA} -- the families are complementary"
                ),
                "C": (
                    "not A and not B, and some arm improves by >= "
                    f"{MEANINGFUL_DELTA} in at least one environment, engine, domain or error type"
                ),
                "D": "no arm meaningfully raises the ceiling anywhere",
                "note": (
                    "section 24: an arm that improves only the already-strong spurious-insertion "
                    "category does not satisfy c3 and therefore cannot reach outcome A."
                ),
            },
            "controls": list(CONTROLS),
            "control_seed": CONTROL_SEED,
            "statistics": {
                "inference_unit": "document cluster, within environment",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "multiplicity": "Holm over the pre-declared primary method family",
                "note": (
                    "section 47: candidate rows are not independent inferential units and are "
                    "never resampled as such."
                ),
            },
            "external_confirmation": {
                "ready_for_external_confirmation": False,
                "reason": (
                    "section 58: a new corrector changes the candidate distribution, so the "
                    "reliability layer must be revalidated before any confirmatory stage. This is "
                    "fixed in advance and does not depend on the result."
                ),
                "confirmatory_reserve_consumed": False,
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"preregister: {len(PRIMARY_METHODS)} arms, {len(CONDITIONS)} conditions, K={list(K_GRID)} "
        f"(primary {PRIMARY_K}), meaningful delta {MEANINGFUL_DELTA} against baseline "
        f"{baseline:.4f}, 6 criteria -> {_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------------------------ section 3, 26, 27, 28


LOCAL_MODEL_ROOT = Path(
    os.environ.get("OCR_RISK_MODEL_DIR", str(Path.home() / ".cache/ocr_risk_models"))
)


def _hf_snapshot(repo_id: str, revision: str) -> Path | None:
    """The local directory holding a pinned revision, or None if it is not on this machine.

    Resolved offline in both branches. A model that is not already here is not available to this
    stage, because an arm that silently downloads two gigabytes on first run is not reproducible
    from its own record. Two layouts are accepted: the hub cache, and a plain directory named
    ``<name>@<revision prefix>`` -- the revision is still part of the identity in both, and every
    file is hashed either way, so the layout changes nothing that the record depends on.
    """
    local = LOCAL_MODEL_ROOT / f"{repo_id.split('/')[-1]}@{revision[:12]}"
    if (local / "config.json").is_file():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
    except Exception:
        return None


def _checkpoint_files(snapshot: Path) -> dict[str, str]:
    """Hash every weight and configuration file in the snapshot. Section 28's checkpoint hash."""
    interesting = (".json", ".safetensors", ".bin", ".txt", ".model")
    return {
        path.name: file_sha256(path)
        for path in sorted(snapshot.iterdir())
        if path.is_file() and path.suffix in interesting
    }


def _model_probe(repo_id: str, revision: str, kind: str) -> dict[str, Any]:
    """What is actually on this machine for one arm, decided before any label exists."""
    snapshot = _hf_snapshot(repo_id, revision)
    if snapshot is None:
        return {
            "repo_id": repo_id,
            "revision": revision,
            "kind": kind,
            "available": False,
            "reason": "no local snapshot for this pinned revision",
        }
    config_path = snapshot / "config.json"
    config = cc_read_json(config_path) if config_path.is_file() else {}
    parameters, parameter_source = _parameter_count(snapshot)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "kind": kind,
        "available": True,
        "snapshot": str(snapshot),
        "architecture": (config.get("architectures") or ["unknown"])[0],
        "model_type": str(config.get("model_type", "unknown")),
        "hidden_size": int(config.get("d_model") or config.get("hidden_size") or 0),
        "parameters_millions": parameters,
        "parameter_count_source": parameter_source,
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", True)),
        "torch_dtype": str(config.get("torch_dtype", "")),
        "checkpoint_files": _checkpoint_files(snapshot),
    }


def _parameter_count(snapshot: Path) -> tuple[float | None, str]:
    """Parameter count read from the checkpoint itself, never estimated.

    A safetensors header carries every tensor's shape. A PyTorch state dict is memory-mapped and
    each storage is counted once, so weights the checkpoint ties together are not counted twice.
    A checkpoint with neither yields no count, which is recorded as unavailable rather than zero.
    """
    import json
    import struct

    total = 0
    tensors = sorted(snapshot.glob("*.safetensors"))
    if tensors:
        for path in tensors:
            with path.open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            for name, spec in header.items():
                if name == "__metadata__" or not isinstance(spec, dict):
                    continue
                count = 1
                for dimension in spec.get("shape") or []:
                    count *= int(dimension)
                total += count
        return round(total / 1e6, 3), "safetensors header"
    states = sorted(snapshot.glob("*.bin"))
    if states:
        import torch

        seen: set[tuple[int, int, tuple[int, ...]]] = set()
        for path in states:
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            for tensor in state.values():
                key = (
                    int(tensor.untyped_storage().data_ptr()),
                    int(tensor.storage_offset()),
                    tuple(int(d) for d in tensor.shape),
                )
                if key not in seen:
                    seen.add(key)
                    total += int(tensor.numel())
        return round(total / 1e6, 3), "PyTorch state dict, each storage counted once"
    return None, "no weight file with readable tensor shapes"


def run_registry() -> int:
    """The method, model, prompt and decoding registries, frozen together.

    Section 28 asks for enough to reproduce an arm months later: the checkpoint and its hash, the
    package version, the decoding configuration, the prompt, the hardware and the seed. Section 3
    asks for M3's unavailability to be recorded here, before any endpoint, rather than decided
    after M1 and M2 have been read.
    """
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (METHOD_REGISTRY, MODEL_VERSIONS, PROMPT_REGISTRY, DECODING_REGISTRY):
        _forbid(path)
    if METHOD_CANDIDATES.exists():
        raise PhaseError("labels already exist; registries cannot be frozen after the fact")

    import platform

    byt5 = _model_probe(BYT5_REPO, BYT5_REVISION, "seq2seq_byte_level")
    llm = _model_probe(LLM_REPO, LLM_REVISION, "decoder_only_instruct")
    audit = cc_read_json(LLM_FORMAT_AUDIT) if LLM_FORMAT_AUDIT.is_file() else None
    if llm["available"] and (audit is None or not audit.get("run")):
        raise PhaseError("run --llmaudit first: the LLM prompt is frozen on its audit")
    audit_reference = (
        {"artifact": _relative(LLM_FORMAT_AUDIT), "sha256": file_sha256(LLM_FORMAT_AUDIT)}
        if audit is not None
        else None
    )
    try:
        import torch
        import transformers

        torch_version = str(torch.__version__)
        transformers_version = str(transformers.__version__)
        device = _resolve_device()
    except ImportError as error:
        torch_version = transformers_version = ""
        device = "unavailable"
        for probe in (byt5, llm):
            probe["available"] = False
            probe["reason"] = f"{error}; install with `uv sync --extra torch`"

    methods = {
        M0: {
            "label": "frozen edit-aware OCR candidate generator",
            "family": "rule-based edit-aware + structural union",
            "implementation": f"{s14.PRIMARY_GENERATOR} via ocr_risk.experiments.cgv3_confirmatory",
            "available": True,
            "supports_multi_candidate": True,
            "natural_candidate_cap": UNION_CAP,
            "receives": "OCR text and the frozen OCR-only site anchors",
            "is_baseline": True,
            "conditions": [COND_FROZEN, COND_ORACLE_LOC],
            "note": (
                "the SGV-CG1 universe. It has no whole-text mode, so its natural condition IS the "
                "frozen localization condition; that identity is asserted by the reproduction gate."
            ),
        },
        M1: {
            "label": "byte-level seq2seq corrector",
            "family": "ByT5-style sequence-to-sequence",
            "implementation": (
                "ocr_risk.candidates.byt5.Byt5Generator (checkpoint reused, not refitted)"
            ),
            "available": bool(byt5["available"]),
            "supports_multi_candidate": True,
            "receives": "OCR text only",
            "is_baseline": False,
            "conditions": list(CONDITIONS),
            "language_scope": sorted(METHOD_LANGUAGES[M1]),
            "out_of_scope_diagnostic_environments": list(OUT_OF_SCOPE_DIAGNOSTIC.get(M1, ())),
            "fine_tuned_here": False,
            "note": (
                "a published checkpoint fine-tuned on English wikitext with synthetic OCR noise. "
                "Its declared language scope is English. Outside it the arm proposes nothing, as "
                "Byt5Generator.applies_to does, so its primary arm is zero on the German "
                "environments. One German environment is also generated as a labelled "
                "out-of-scope diagnostic and routed to the auxiliary table."
            ),
        },
        M2: {
            "label": "local text-only instruct LLM corrector",
            "family": "decoder-only generative LLM, text only",
            "implementation": "local pinned checkpoint, greedy/beam decoding, frozen prompt",
            "available": bool(llm["available"]),
            "supports_multi_candidate": True,
            "receives": "OCR text and the frozen local textual context",
            "is_baseline": False,
            "conditions": list(CONDITIONS),
            "language_scope": sorted(METHOD_LANGUAGES[M2]),
            "fine_tuned_here": False,
            "externally_nondeterministic": False,
            "note": (
                "a local checkpoint rather than an external API, so section 27's reproducibility "
                "preference is met exactly: the same weights and the same deterministic decode "
                "regenerate the same raw outputs."
            ),
        },
        M3: {
            "label": "multimodal / VLM corrector",
            "family": "image-aware correction",
            "available": False,
            "reason": M3_UNAVAILABLE_REASON,
            "recorded_before_endpoint_evaluation": True,
            "conditions": [],
        },
        M_UNION: {
            "label": "deduplicated union of all available correction families",
            "family": "analysis-only diagnostic",
            "available": True,
            "analysis_only": True,
            "is_deployable_method": False,
            "note": (
                "section 4 and 31: this measures complementarity and the maximum opportunity "
                "available if the families were combined perfectly. It consumes more generation "
                "resource than any single arm and is never proposed as a method."
            ),
        },
    }
    unavailable = [name for name in PRIMARY_METHODS if not methods[name]["available"]]

    _write_json_once(
        METHOD_REGISTRY,
        {
            **_envelope("method_registry"),
            "methods": methods,
            "primary_family": list(PRIMARY_METHODS),
            "available_primary_family": [m for m in PRIMARY_METHODS if methods[m]["available"]],
            "unavailable_primary_family": unavailable,
            "conditions": list(CONDITIONS),
            "primary_condition": PRIMARY_CONDITION,
            "controls": CONTROL_SPECS,
            "control_seed": CONTROL_SEED,
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        MODEL_VERSIONS,
        {
            **_envelope("model_versions"),
            "models": {M1: byt5, M2: llm},
            "reissue_note": (
                "the four registries were reissued once, before any label existed, to correct the "
                "byte model's parameter count: the first issue recorded 0 because its pinned "
                "checkpoint carries no safetensors header. No method, prompt, decoding setting or "
                "availability changed."
            ),
            "inference_dtype": {M1: "float32", M2: LLM_DTYPE},
            "packages": {"torch": torch_version, "transformers": transformers_version},
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "device": device,
            },
            "external_api_calls": 0,
            "note": (
                "no arm is an external API, so section 28's API identifier, request parameter and "
                "response-hash fields have no subject in this stage. Both arms are local "
                "checkpoints pinned by revision and hashed file by file."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        PROMPT_REGISTRY,
        {
            **_envelope("prompt_registry"),
            "prompts": {
                M0: {"uses_prompt": False, "note": "a rule-based generator takes no prompt."},
                M1: {
                    "uses_prompt": False,
                    "note": (
                        "the checkpoint is fine-tuned for correction and takes the noisy text "
                        "directly, with no instruction prefix. Adding one would change the arm."
                    ),
                },
                M2: {
                    "uses_prompt": True,
                    "revision": PROMPT_REVISION,
                    "system": LLM_SYSTEM_PROMPT,
                    "context_template": LLM_CONTEXT_TEMPLATE,
                    "line_template": LLM_LINE_TEMPLATE,
                    "output_format": "the corrected text inside [[ and ]]",
                    "output_parse": "clean_llm_output, applied during normalization",
                    "contains_evaluation_examples": False,
                    "contains_ground_truth": False,
                    "frozen_before_endpoint_evaluation": True,
                    "revision_history": [
                        {
                            "revision": 0,
                            "status": "superseded before any label existed",
                            "prompt_hash": canonical_hash(PROMPT_DRAFT_0),
                            **PROMPT_DRAFT_0,
                            "reason": PROMPT_REVISION_REASON,
                        }
                    ],
                    "format_audit": audit_reference,
                },
            },
            "prompt_hash": canonical_hash(
                {
                    "system": LLM_SYSTEM_PROMPT,
                    "context": LLM_CONTEXT_TEMPLATE,
                    "line": LLM_LINE_TEMPLATE,
                }
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DECODING_REGISTRY,
        {
            **_envelope("decoding_registry"),
            "shared": {
                "deterministic": True,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": None,
                "seed": CONTROL_SEED,
                "note": (
                    "section 27 prefers reproducibility, so every arm decodes deterministically. "
                    "No decoding setting was searched on an evaluation outcome; these values were "
                    "written before any label existed."
                ),
            },
            "decoding": {
                M0: {
                    "kind": "enumerative",
                    "candidate_cap": UNION_CAP,
                    "note": (
                        "the frozen union generator ranks its own proposals; nothing is sampled."
                    ),
                },
                M1: {
                    "region_conditions": {
                        "num_beams": BEAM_WIDTH,
                        "num_return_sequences": RETURN_SEQUENCES,
                        "max_new_tokens": BYT5_MAX_NEW_TOKENS,
                        "supports_multi_candidate": True,
                    },
                    "natural_line_condition": {
                        "num_beams": LINE_BEAMS,
                        "num_return_sequences": LINE_RETURN_SEQUENCES,
                        "max_new_tokens": BYT5_LINE_MAX_NEW_TOKENS,
                        "supports_multi_candidate": False,
                    },
                    "truncation_max_length": 512,
                    "batch_size": GENERATION_BATCH,
                },
                M2: {
                    "region_conditions": {
                        "num_beams": LLM_BEAMS,
                        "num_return_sequences": LLM_RETURN_SEQUENCES,
                        "max_new_tokens_cap": LLM_MAX_NEW_TOKENS,
                        "max_new_tokens_rule": (
                            f"min(cap, {LLM_REGION_TOKEN_MARGIN} + longest fragment in the "
                            "batch, in characters)"
                        ),
                        "supports_multi_candidate": False,
                    },
                    "natural_line_condition": {
                        "num_beams": LLM_BEAMS,
                        "num_return_sequences": LLM_RETURN_SEQUENCES,
                        "max_new_tokens_cap": LLM_LINE_MAX_NEW_TOKENS,
                        "max_new_tokens_rule": (
                            f"min(cap, {LLM_LINE_TOKEN_MARGIN} + longest line in the batch, in "
                            "characters)"
                        ),
                        "supports_multi_candidate": False,
                    },
                    "batch_size": LLM_BATCH,
                    "ordering": "requests sorted by prompt length, ties by request id",
                    "dtype": LLM_DTYPE,
                    "output_parse": (
                        "clean_llm_output: the text inside the first [[ ]] pair, else the first "
                        "non-empty line; applied in normalization, never in the runner"
                    ),
                    "why_one_greedy_sequence": (
                        "the format audit's measured throughput for greedy single-sequence "
                        "decoding, recorded in the audit artifact. Ten-sequence beam search "
                        "multiplies the decode work, and the arm would not fit this stage's "
                        "compute. Decided before any label existed."
                    ),
                    "reproducibility": (
                        "identical on a repeated run with the same batching, and not invariant "
                        "to batch size in bf16 on this device; the audit records both. The batch "
                        "size and ordering are therefore part of the arm's definition."
                    ),
                    "format_audit": audit_reference,
                },
            },
            "matched_budget": {
                "k_grid": list(K_GRID),
                "primary_k": PRIMARY_K,
                "note": (
                    "one decode returns the full beam set; every K is a prefix of that frozen "
                    "ordering, so matched-K truncation reads no label and re-runs no model."
                ),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"registry: {len([m for m in PRIMARY_METHODS if methods[m]['available']])} of "
        f"{len(PRIMARY_METHODS)} arms available"
        + (f" (unavailable: {unavailable})" if unavailable else "")
        + f"; M3 recorded unavailable; device {device}"
    )
    return 0


def _resolve_device() -> str:
    import torch

    if os.environ.get("OCR_RISK_XC1_DEVICE"):
        return os.environ["OCR_RISK_XC1_DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ section 2, 8: atomic edits


@dataclass(frozen=True, slots=True)
class AtomicEdit:
    """One proposed edit in the project's own representation.

    An edit names a region of the OCR stream by the complete canonical spans it covers, so the
    region a candidate is scored on is a region the alignment can price. `gap` is the separator
    between two adjacent spans, as the repository's GAP anchor defines it, and is the only kind
    that can repair an OCR omission.
    """

    edit_kind: str
    anchor_kind: str
    span_ids: tuple[str, ...]
    char_start: int
    char_end: int
    original: str
    candidate: str
    rank: int

    @property
    def anchor_ref(self) -> str:
        return "\0".join((self.anchor_kind, *self.span_ids))


def _anchor_kind_for(count: int) -> str:
    """The repository's anchor names where they apply, and `token_run` beyond two spans.

    `AnchorKind.TOKEN_PAIR` means exactly two adjacent spans, so a longer run is not labelled as
    one. The labeller and the linker treat every non-gap anchor as the region covering its spans,
    so the name changes nothing either of them computes.
    """
    if count == 1:
        return "token"
    return "token_pair" if count == 2 else "token_run"


def atomic_edits_from_rewrite(
    source: str,
    rewritten: str,
    spans: Sequence[tuple[str, int, int]],
    *,
    rank: int = 0,
) -> tuple[list[AtomicEdit], dict[str, int]]:
    """Convert one rewritten window into atomic edits. Section 8's single shared diff.

    ``spans`` are the canonical spans covering the window, as ``(span_id, start, end)`` in window
    coordinates. The rewrite is aligned to the source once; every non-equal opcode either lands
    inside spans -- which are then snapped out to their full extent, because a candidate scored
    on half a token would be compared against the whole token's ground truth -- or lands in the
    whitespace between two spans, which is a gap insertion.

    Maximal runs of touched spans become one edit each, so two unrelated repairs on the same line
    stay two candidates. Two boundary rules keep every edit set faithful to the rewrite, and
    falsification test f8 checks them by applying the edits back to the source:

    * **A removed separator is a merge.** A run whose rewrite still has text, but no whitespace
      where the source had a separator, has fused with that neighbour, and the neighbour joins
      the run. A run deleted outright merges only when no whitespace survives between its two
      neighbours: a token deleted at a line edge takes its separator with it and is not a merge.
    * **A gap beside a rewritten run is already inside it.** The run's projection spans the
      separator the gap sits in, so emitting the gap as well would insert its text twice.

    Returns the edits and a census of what the conversion did, including the edits it refused to
    emit. A silently dropped edit and a model that proposed nothing are different findings.
    """
    from rapidfuzz.distance import Levenshtein

    census = {
        "opcodes": 0,
        "region_edits": 0,
        "gap_edits": 0,
        "gaps_absorbed_into_a_region": 0,
        "runs_merged_across_a_removed_separator": 0,
        "dropped_unprojectable": 0,
        "dropped_identity": 0,
        "dropped_whitespace_only": 0,
        "dropped_outside_any_span": 0,
    }
    if not spans or rewritten == source:
        return [], census

    ordered = sorted(spans, key=lambda item: (item[1], item[2]))
    starts = [start for _, start, _ in ordered]
    ends = [end for _, _, end in ordered]
    count = len(ordered)
    touched = [False] * count
    gaps: list[tuple[int, str]] = []

    for op in Levenshtein.opcodes(source, rewritten):
        if op.tag == "equal":
            continue
        census["opcodes"] += 1
        s0, s1 = op.src_start, op.src_end
        inserted = rewritten[op.dest_start : op.dest_end]
        if s0 == s1:
            # A pure insertion belongs to a span when it falls strictly inside one, or abuts a span
            # edge without a whitespace boundary -- the rule `project_span` applies, so the two
            # never disagree about who owns an inserted character.
            owner = (
                [i for i in range(count) if starts[i] < s0 < ends[i]]
                or [i for i in range(count) if ends[i] == s0 and inserted[:1].strip()]
                or [i for i in range(count) if starts[i] == s0 and inserted[-1:].strip()]
            )
            if owner:
                touched[owner[0]] = True
            elif not inserted.strip():
                census["dropped_whitespace_only"] += 1
            else:
                gaps.append((s0, inserted.strip()))
            continue
        overlapping = [i for i in range(count) if starts[i] < s1 and ends[i] > s0]
        if overlapping:
            for index in overlapping:
                touched[index] = True
            continue
        # The opcode falls entirely in the whitespace between two spans. Deleting or rewriting that
        # separator is a segmentation repair -- "Sm ith" -> "Smith" -- so both flanks join.
        left = [i for i in range(count) if ends[i] <= s0]
        right = [i for i in range(count) if starts[i] >= s1]
        if left and right and right[0] == left[-1] + 1:
            touched[left[-1]] = touched[right[0]] = True
        else:
            census["dropped_outside_any_span"] += 1

    def runs() -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        position = 0
        while position < count:
            if not touched[position]:
                position += 1
                continue
            last = position
            while last + 1 < count and touched[last + 1]:
                last += 1
            found.append((position, last))
            position = last + 1
        return found

    def padded(first: int, last: int) -> tuple[int, int]:
        # Out to the neighbouring spans: those boundaries lie in whitespace no span owns, so an
        # edit that also consumes a separator does not read as straddling a boundary.
        return (
            ends[first - 1] if first > 0 else 0,
            starts[last + 1] if last + 1 < count else len(source),
        )

    changed = True
    while changed:  # each pass only adds touched spans, so this reaches a fixed point
        changed = False
        for first, last in runs():
            pad_start, pad_end = padded(first, last)
            projected = project_span_shared(source, rewritten, pad_start, pad_end)
            if projected is None:
                continue
            before, after = source[pad_start : starts[first]], source[ends[last] : pad_end]
            left_separated = first > 0 and bool(before) and before.isspace()
            right_separated = last + 1 < count and bool(after) and after.isspace()
            if projected.strip():
                if left_separated and not projected[:1].isspace():
                    touched[first - 1] = changed = True
                    census["runs_merged_across_a_removed_separator"] += 1
                if right_separated and not projected[-1:].isspace():
                    touched[last + 1] = changed = True
                    census["runs_merged_across_a_removed_separator"] += 1
            elif left_separated and right_separated and projected == "":
                touched[first - 1] = touched[last + 1] = changed = True
                census["runs_merged_across_a_removed_separator"] += 2

    edits: list[AtomicEdit] = []
    for first, last in runs():
        pad_start, pad_end = padded(first, last)
        projected = project_span_shared(source, rewritten, pad_start, pad_end)
        if projected is None:
            census["dropped_unprojectable"] += 1
            continue
        proposed = projected.strip()
        original = source[starts[first] : ends[last]]
        if proposed == original:
            census["dropped_identity"] += 1
            continue
        edits.append(
            AtomicEdit(
                edit_kind=EDIT_REGION,
                anchor_kind=_anchor_kind_for(last - first + 1),
                span_ids=tuple(ordered[i][0] for i in range(first, last + 1)),
                char_start=starts[first],
                char_end=ends[last],
                original=original,
                candidate=proposed,
                rank=rank,
            )
        )
        census["region_edits"] += 1

    for offset, inserted in gaps:
        left = [i for i in range(count) if ends[i] <= offset]
        right = [i for i in range(count) if starts[i] >= offset]
        if not left or not right or right[0] != left[-1] + 1:
            # Not on a single span boundary; attributing it to a flank would invent a location.
            census["dropped_outside_any_span"] += 1
            continue
        left_index, right_index = left[-1], right[0]
        if touched[left_index] or touched[right_index]:
            census["gaps_absorbed_into_a_region"] += 1
            continue
        edits.append(
            AtomicEdit(
                edit_kind=EDIT_GAP,
                anchor_kind="gap",
                span_ids=(ordered[left_index][0], ordered[right_index][0]),
                char_start=ends[left_index],
                char_end=starts[right_index],
                # The repository's GAP anchor spans the separator between the two flanks, so that
                # separator is the gap's original text -- the stream slice the labeller checks.
                original=source[ends[left_index] : starts[right_index]],
                candidate=inserted,
                rank=rank,
            )
        )
        census["gap_edits"] += 1
    return edits, census


def project_span_shared(source: str, rewritten: str, start: int, end: int) -> str | None:
    """What the rewrite says in place of ``source[start:end]``.

    Delegates to the repository's own `ocr_risk.candidates.byt5.project_span`, which already has
    the boundary rules and the test suite. Importing it rather than restating it is what keeps
    section 8's "one common deterministic diff" literally true: the ByT5 arm, the LLM arm and the
    controls all project through the same function.
    """
    from ocr_risk.candidates.byt5 import project_span

    return project_span(source, rewritten, start, end)


def run_schema() -> int:
    """Section 2. The common atomic edit representation every arm is reduced to."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    _forbid(ATOMIC_EDIT_SCHEMA)

    fields = {
        "environment": "the (corpus, base engine) pair the edit was proposed in",
        "method": "which correction family proposed it -- section 32's attribution, never dropped",
        "condition": f"one of {list(CONDITIONS)}",
        "document_id": "the page the edit is on",
        "engine_id": "the OCR configuration the edit is against",
        "site_id": "the proposal site, qualified by environment so ids cannot collide",
        "candidate_id": "a content digest over environment, method, condition, site and text",
        "edit_kind": f"{EDIT_REGION} or {EDIT_GAP}",
        "anchor_kind": "token, token_pair or gap -- the repository's own AnchorKind values",
        "span_ids": "the canonical spans the edit covers, in reading order",
        "char_start": "start offset of the region in the linearized OCR stream",
        "char_end": "end offset; for a gap, the right flank's start",
        "original_ocr": (
            "stream[char_start:char_end], the text the edit replaces; for a gap, the separator "
            "between the flanks that the inserted text goes into"
        ),
        "candidate_text": "the proposed replacement",
        "generator_rank": "position in the method's own frozen returned ordering, 0-based",
        "candidate_source": "section 32: the set of methods that proposed this exact edit",
    }
    label_fields = {
        "outcome": "SGV14's OutcomeIfAccepted value",
        "is_harmful": "harm under the frozen STRICT_WORSENING policy",
        "beneficial": "the project's beneficial class (exact repair or partial improvement)",
        "exact": f"outcome == {OUTCOME_EXACT}; the primary success definition",
        "d_before": "levenshtein(region OCR, region GT) in raw characters",
        "d_after": "levenshtein(candidate, region GT) in raw characters",
    }
    _write_json_once(
        ATOMIC_EDIT_SCHEMA,
        {
            **_envelope("atomic_edit_schema"),
            "fields": fields,
            "label_fields": label_fields,
            "reissue_note": (
                "reissued once before any label existed: the first issue described a gap's "
                "region as empty. It is the separator between the two flanks, as the "
                "repository's GAP anchor defines it."
            ),
            "labels_are_computed_by": "ocr_risk.experiments.cgv3_study.label_candidate",
            "shared_diff": {
                "function": "atomic_edits_from_rewrite",
                "projection": "ocr_risk.candidates.byt5.project_span",
                "handles": ["substitution", "insertion", "deletion", "whitespace", "segmentation"],
                "snapping": (
                    "every region edit is snapped out to the complete canonical spans it touches, "
                    "so the region a candidate is scored on is a region the alignment can price."
                ),
                "gap_rule": (
                    "an insertion in the whitespace between two adjacent spans becomes a gap edit "
                    "anchored on both flanks. This is the only edit kind that can repair an OCR "
                    "omission, which is one of the three weak categories section 24 asks about."
                ),
                "note": (
                    "section 8: one diff for every generative method. Per-method edit extraction "
                    "would turn the comparison into a comparison of heuristics."
                ),
            },
            "identity_edits_dropped": True,
            "deduplication": {
                "key": ["environment", "condition", "site_id", "candidate_text"],
                "outcome_labels_may_influence_deduplication": False,
                "attribution_preserved": True,
                "note": (
                    "section 10: two candidates representing the same atomic edit at the same site "
                    "count once, and the surviving row keeps the full set of methods that proposed "
                    "it. A candidate is never removed for being harmful."
                ),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"schema: {len(fields)} edit fields, {len(label_fields)} label fields, one shared diff -> "
        f"{_relative(ATOMIC_EDIT_SCHEMA)}"
    )
    return 0


# ------------------------------------------------------------------ section 36: GT-blind inputs
#
# Everything in this block builds model inputs from OCR artifacts alone. It never imports
# `ocr_risk.align`, never builds an `AlignmentIndex` and never reads a ground-truth token. The
# leakage suite walks this module's call graph and asserts that, so the guarantee is structural
# rather than a promise made in a docstring.

OCR_SPAN_COLUMNS = ("document_id", "engine_id", "span_id", "line_id", "char_start", "char_end")
REQUEST_COLUMNS = (
    "environment",
    "condition",
    "request_id",
    "document_id",
    "engine_id",
    "site_id",
    "anchor_kind",
    "anchor_ref",
    "char_start",
    "char_end",
    "region",
    "context_before",
    "context_after",
    "window_start",
    "window_text",
)


def _ocr_artifacts(
    spec: dict[str, Any], pipeline: Any
) -> tuple[
    dict[tuple[str, str], list[Any]],
    dict[tuple[str, str], str],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
]:
    """Canonical spans, streams, the frozen OCR-only sites and M0's candidate stream.

    This is SGV14's own `_candidate_table` on SGV14's own inputs, with the labelling half not yet
    applied. Calling it rather than re-deriving discovery is what makes the section 5 reproduction
    gate meaningful: M0's candidate ids here are the candidate ids SGV-CG1 published, or the gate
    fails and says so.
    """
    from ocr_risk.canonical import rebuild_stream

    bundles = s14._document_bundles(spec["corpus"])
    roles = s14.partition_of(spec["corpus"], sorted(bundles))
    spans, _record = s14._canonical_spans(spec, bundles)
    streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
    sites, m0_candidates = s14._candidate_table(spec, spans, streams, pipeline, roles)
    return spans, streams, sites, m0_candidates, roles


def _span_table(spans: dict[tuple[str, str], list[Any]]) -> pd.DataFrame:
    rows = [
        {
            "document_id": pair[0],
            "engine_id": pair[1],
            "span_id": str(span.span_id),
            "line_id": str(span.line_id) if span.line_id is not None else "",
            "reading_order": int(span.reading_order),
            "char_start": int(span.char_start),
            "char_end": int(span.char_end),
        }
        for pair, group in spans.items()
        for span in group
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["document_id", "engine_id", "reading_order", "char_start"], kind="stable")
        .reset_index(drop=True)
    )


def _requests_frozen_localization(
    name: str,
    sites: pd.DataFrame,
    streams: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """Section 7's primary condition: one request per frozen discovered site.

    The region and its context window are exactly what the frozen pipeline shows its generators,
    so no arm is handed a wider view of the page than another.
    """
    rows: list[dict[str, Any]] = []
    for site in sites.itertuples(index=False):
        pair = (str(site.document_id), str(site.engine_id))
        stream = streams.get(pair, "")
        start, end = int(site.char_start), int(site.char_end)
        rows.append(
            {
                "environment": name,
                "condition": COND_FROZEN,
                "request_id": f"{name}|{COND_FROZEN}|{site.site_id}",
                "document_id": str(site.document_id),
                "engine_id": str(site.engine_id),
                "site_id": f"{name}|{site.site_id}",
                "anchor_kind": str(site.anchor_kind),
                "anchor_ref": str(site.anchor_ref),
                "char_start": start,
                "char_end": end,
                "region": stream[start:end],
                "context_before": stream[max(0, start - CONTEXT_CHARS) : start],
                "context_after": stream[end : end + CONTEXT_CHARS],
                "window_start": start,
                "window_text": stream[start:end],
            }
        )
    return pd.DataFrame(rows, columns=list(REQUEST_COLUMNS))


def _requests_natural(
    name: str,
    span_table: pd.DataFrame,
    streams: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """The natural condition: one request per OCR line, discovered by the method itself.

    A line is the unit a whole-text corrector actually operates on, and it is defined here from
    the engine's own `line_id` in reading order -- an OCR artifact, not an alignment. The window
    is the stream slice from the first span's start to the last span's end, so every character
    offset the diff produces is already in stream coordinates and needs no remapping.
    """
    rows: list[dict[str, Any]] = []
    for (document_id, engine_id), group in span_table.groupby(
        ["document_id", "engine_id"], sort=True
    ):
        stream = streams.get((str(document_id), str(engine_id)), "")
        if not stream:
            continue
        ordered = group.sort_values(["reading_order", "char_start"], kind="stable")
        # Spans with no line_id are their own line: an engine that does not report line structure
        # must not have every span on the page collapsed into one giant window.
        keys = [
            line_id if line_id else f"__span__{span_id}"
            for line_id, span_id in zip(ordered["line_id"], ordered["span_id"], strict=True)
        ]
        for position, key in enumerate(dict.fromkeys(keys)):
            members = ordered[[k == key for k in keys]]
            start = int(members["char_start"].min())
            end = int(members["char_end"].max())
            if end <= start:
                continue
            rows.append(
                {
                    "environment": name,
                    "condition": COND_NATURAL,
                    "request_id": f"{name}|{COND_NATURAL}|{document_id}|{engine_id}|{position}",
                    "document_id": str(document_id),
                    "engine_id": str(engine_id),
                    "site_id": "",
                    "anchor_kind": "",
                    "anchor_ref": "",
                    "char_start": start,
                    "char_end": end,
                    "region": stream[start:end],
                    "context_before": stream[max(0, start - CONTEXT_CHARS) : start],
                    "context_after": stream[end : end + CONTEXT_CHARS],
                    "window_start": start,
                    "window_text": stream[start:end],
                }
            )
    return pd.DataFrame(rows, columns=list(REQUEST_COLUMNS))


# ------------------------------------------------------------------ section 3, 27: the arms


def _release_device_cache(device: str) -> None:
    """Return the accelerator allocator's cached blocks to the system after a batch.

    PyTorch's MPS allocator keeps freed blocks for reuse. Over thousands of batches of slowly
    growing shapes that cache grew past what this machine could hold and drove it into swap.
    Releasing it changes no computation -- a kernel's result depends on its inputs and shapes, not
    on where its memory lives -- so the frozen batching, and every cached output, is untouched.
    """
    if device == "mps":
        import torch

        torch.mps.empty_cache()


class _Byt5Runner:
    """The byte-level seq2seq arm, loaded once per process from the pinned local snapshot."""

    method = M1

    def __init__(self, snapshot: Path, device: str) -> None:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        model: Any = T5ForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True
        )
        model.eval()
        self.model = model.to(device)
        self.device = device
        torch.set_grad_enabled(False)

    def generate(
        self, prompts: Sequence[str], *, max_new_tokens: int, num_beams: int, num_return: int
    ) -> list[list[str]]:
        import torch

        encoded = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return,
                do_sample=False,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        del encoded, generated
        _release_device_cache(self.device)
        return [
            [text.strip() for text in decoded[i * num_return : (i + 1) * num_return]]
            for i in range(len(prompts))
        ]


class _LlmRunner:
    """The text-only instruct arm. Frozen prompt, deterministic decode, local weights."""

    method = M2

    def __init__(self, snapshot: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        # Decoder-only batching must pad on the left, or the generated continuation starts after
        # a run of pad tokens and the decode is silently wrong for every short row in the batch.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
            str(snapshot), local_files_only=True, dtype=getattr(torch, LLM_DTYPE)
        )
        model.eval()
        self.model = model.to(device)
        self.device = device
        torch.set_grad_enabled(False)

    def _chat(self, user: str) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate(
        self, prompts: Sequence[str], *, max_new_tokens: int, num_beams: int, num_return: int
    ) -> list[list[str]]:
        import torch

        chats = [self._chat(prompt) for prompt in prompts]
        encoded = self.tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        width = int(encoded["input_ids"].shape[1])
        decoded = self.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        del encoded, generated
        _release_device_cache(self.device)
        # The decoded text is cached exactly as produced. Parsing it is normalization and happens in
        # `_neural_edits`, so the raw-output cache holds the model's response rather than a reading
        # of it (section 54), and a determinism check re-runs the parse from that cache.
        return [
            [str(text) for text in decoded[i * num_return : (i + 1) * num_return]]
            for i in range(len(prompts))
        ]


def clean_llm_output(text: str) -> str:
    """Deterministic parse of one generated answer. It reads no label.

    The frozen prompt asks for the corrected text inside `[[` and `]]`. This returns what sits
    inside the first such pair; when the model ignored the delimiters it falls back to the first
    non-empty line without a wrapping quote pair. It applies identically to every row and cannot
    prefer a correct answer to an incorrect one, because it never sees which is which.
    """
    opening = text.find("[[")
    if opening >= 0:
        closing = text.find("]]", opening + 2)
        if closing >= 0:
            return text[opening + 2 : closing].strip()
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    for quote in ('"', "'", "`"):
        if len(first) >= 2 and first.startswith(quote) and first.endswith(quote):
            first = first[1:-1].strip()
    return first.replace("[[", "").replace("]]", "").strip()


def _prompt_for(method: str, condition: str, row: Any) -> str:
    """The exact string an arm receives. No ground truth reaches this function."""
    region = str(row.region)
    if method == M1:
        # The checkpoint was fine-tuned on noisy text directly; an instruction prefix would make
        # it a different arm. It receives the window and nothing else.
        return str(row.window_text) if condition == COND_NATURAL else region
    if condition == COND_NATURAL:
        return LLM_LINE_TEMPLATE.format(region=str(row.window_text))
    # Line separators in the context are flattened to spaces, so the delimited fragment is the only
    # structured thing in the prompt and the answer's first line cannot be a line of context.
    return LLM_CONTEXT_TEMPLATE.format(
        before=str(row.context_before).replace("\n", " "),
        region=region,
        after=str(row.context_after).replace("\n", " "),
    )


RAW_COLUMNS = ("environment", "method", "condition", "request_id", "rank", "output")


def _run_arm(
    runner: Any,
    requests: pd.DataFrame,
    method: str,
    condition: str,
    progress: Callable[[str], None],
    *,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One arm over one condition's requests, batched. Returns raw outputs and cost accounting.

    The byte model runs in request order with fixed caps -- the procedure its cache was produced
    with. The LLM runs over requests sorted by prompt length, so a batch pads little, with an output
    cap set from the longest fragment in the batch. Both orderings are deterministic and both are
    part of the arm's definition: a bf16 decode on this device is reproducible for fixed batching
    and is not invariant to it.
    """
    line = condition == COND_NATURAL
    llm = method == M2
    if llm:
        beams, returns = LLM_BEAMS, LLM_RETURN_SEQUENCES
        cap = LLM_LINE_MAX_NEW_TOKENS if line else LLM_MAX_NEW_TOKENS
        margin = LLM_LINE_TOKEN_MARGIN if line else LLM_REGION_TOKEN_MARGIN
        size = batch_size or LLM_BATCH
    else:
        beams = LINE_BEAMS if line else BEAM_WIDTH
        returns = LINE_RETURN_SEQUENCES if line else RETURN_SEQUENCES
        cap = BYT5_LINE_MAX_NEW_TOKENS if line else BYT5_MAX_NEW_TOKENS
        margin = 0
        size = batch_size or (LINE_BATCH if line else GENERATION_BATCH)

    frame = requests.reset_index(drop=True)
    prompts = [_prompt_for(method, condition, row) for row in frame.itertuples(index=False)]
    widths = frame["window_text" if line else "region"].astype(str).str.len().to_numpy()
    ids = frame["request_id"].astype(str).tolist()
    environments = frame["environment"].astype(str).tolist()
    order = (
        sorted(range(len(frame)), key=lambda i: (len(prompts[i]), ids[i]))
        if llm
        else list(range(len(frame)))
    )
    began = time.monotonic()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(order), size):
        chunk = order[start : start + size]
        max_new = min(cap, margin + int(widths[chunk].max())) if llm else cap
        outputs = runner.generate(
            [prompts[i] for i in chunk], max_new_tokens=max_new, num_beams=beams, num_return=returns
        )
        for i, produced in zip(chunk, outputs, strict=True):
            for rank, text in enumerate(produced):
                rows.append(
                    {
                        "environment": environments[i],
                        "method": method,
                        "condition": condition,
                        "request_id": ids[i],
                        "rank": rank,
                        "output": text,
                    }
                )
        if start and start % (size * 25) == 0:
            done = start + len(chunk)
            rate = done / max(time.monotonic() - began, 1e-9)
            progress(f"    {method}/{condition}: {done}/{len(frame)} ({rate:.1f}/s)")
    elapsed = time.monotonic() - began
    cost = {
        "requests": len(frame),
        "returned_sequences": len(rows),
        "wall_clock_seconds": elapsed,
        "requests_per_second": _ratio(len(frame), elapsed),
        "prompt_characters": sum(len(prompt) for prompt in prompts),
        "num_beams": beams,
        "num_return_sequences": returns,
        "max_new_tokens": cap,
        "max_new_tokens_rule": (
            f"min({cap}, {margin} + longest fragment in the batch, in characters)"
            if llm
            else "fixed"
        ),
        "batch_size": size,
        "ordering": "prompt length, then request id" if llm else "request order",
    }
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS)), cost


def _environment_inputs(
    spec: dict[str, Any], pipeline: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One environment's GT-blind inputs: the span table, both request sets and M0's candidates.

    Pure -- it writes nothing -- so `--prepare` caches its result and `--determinism` rebuilds it
    from the raw OCR and compares the two.
    """
    name = spec["environment"]
    spans, streams, sites, m0_candidates, roles = _ocr_artifacts(spec, pipeline)
    span_table = _span_table(spans)
    evaluation = {d for d, role in roles.items() if role == s14.ROLE_EVALUATION}
    frozen = _requests_frozen_localization(
        name, sites[sites["document_id"].isin(evaluation)], streams
    )
    natural = _requests_natural(
        name, span_table[span_table["document_id"].isin(evaluation)], streams
    )
    return span_table, pd.concat([frozen, natural], ignore_index=True), m0_candidates


def _prepare_environment(spec: dict[str, Any], pipeline: Any) -> pd.DataFrame:
    """Build and cache one environment's GT-blind inputs. No model is loaded here."""
    name = spec["environment"]
    slug = _slug(name)
    request_path = CACHE / f"{slug}.requests.parquet"
    span_path = CACHE / f"{slug}.spans.parquet"
    m0_path = CACHE / f"{slug}.m0_candidates.parquet"
    if request_path.is_file() and span_path.is_file() and m0_path.is_file():
        return pd.read_parquet(request_path)

    began = time.monotonic()
    span_table, requests, m0_candidates = _environment_inputs(spec, pipeline)
    frozen = requests[requests["condition"] == COND_FROZEN]
    natural = requests[requests["condition"] == COND_NATURAL]
    span_table.to_parquet(span_path, index=False)
    requests.to_parquet(request_path, index=False)
    m0_candidates.to_parquet(m0_path, index=False)
    print(
        f"  {name}: {len(frozen)} frozen-site + {len(natural)} line requests, "
        f"{len(m0_candidates)} M0 candidates ({time.monotonic() - began:.0f}s)"
    )
    return requests


def run_prepare() -> int:
    """The GT-blind reconstruction, cached so inference can restart without repeating it.

    Nothing here loads a correction model and nothing here reads a ground-truth token: it is the
    canonical spans, the linearized streams, the frozen OCR-only site enumeration, M0's candidate
    stream and the two request sets built from them.
    """
    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    CACHE.mkdir(parents=True, exist_ok=True)
    pipeline = s14.build_pipeline()
    print(
        f"  frozen pipeline: {len(pipeline.resources.lm)} lm grams, "
        f"{len(pipeline.resources.lexicon)} lexicon tokens, {pipeline.glyph_prototypes} glyphs"
    )
    totals = {COND_FROZEN: 0, COND_NATURAL: 0}
    for spec in s14.ENVIRONMENTS:
        requests = _prepare_environment(spec, pipeline)
        counts = requests["condition"].value_counts().to_dict()
        for condition in totals:
            totals[condition] += int(counts.get(condition, 0))
    print(
        f"prepare: {totals[COND_FROZEN]} frozen-site requests and {totals[COND_NATURAL]} line "
        f"requests over {len(s14.ENVIRONMENTS)} environments in {time.monotonic() - started:.0f}s"
    )
    return 0


def _cost_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.name.replace(".parquet", ".cost.json"))


def _write_cost_sidecar(
    raw_path: Path, name: str, method: str, condition: str, cost: dict[str, Any]
) -> None:
    """Section 29's cost record, written beside the raw file it describes, the moment it exists.

    Written per file rather than once at the end of the run, so that a run killed partway through
    keeps the cost of everything it finished. `time.monotonic` does not advance while the machine
    sleeps, so the wall clock recorded here is compute time and not a suspended session.
    """
    _write_json_once(
        _cost_path(raw_path),
        {
            **_envelope("generation_cost"),
            "environment": name,
            "method": method,
            "condition": condition,
            **cost,
            "raw_output": _relative(raw_path),
            "raw_output_sha256": file_sha256(raw_path),
            "source": "measured by _run_arm at completion",
        },
    )


def run_generate() -> int:
    """Model inference, ground-truth-blind, cached immutably.

    Two conditions run here and both build their inputs from OCR artifacts alone: the frozen
    localization every arm shares, and the natural line condition in which an arm localizes for
    itself. Section 45's oracle localization is NOT generated here -- it needs the true error
    region, so it lives in its own phase and is labelled analysis-only wherever it appears.

    The raw outputs are the immutable record section 54 asks for. Every later phase regenerates
    from this cache rather than from the models, so a determinism check is a check of the
    analysis and not a second sample from a stochastic decoder.
    """
    started = time.monotonic()
    _require(METHOD_REGISTRY, "registry")
    _require(ATOMIC_EDIT_SCHEMA, "schema")
    CACHE.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    registry = cc_read_json(METHOD_REGISTRY)
    arms = [m for m in (M1, M2) if registry["methods"][m]["available"]]
    only = os.environ.get("OCR_RISK_XC1_ONLY", "")
    if only:
        arms = [m for m in arms if m in only.split(",")]
    device = _resolve_device() if arms else "unused"
    runners: dict[str, Any] = {}
    if arms:
        import transformers

        transformers.utils.logging.set_verbosity_error()
        versions = cc_read_json(MODEL_VERSIONS)
        for method in arms:
            snapshot = Path(versions["models"][method]["snapshot"])
            began = time.monotonic()
            runners[method] = (
                _Byt5Runner(snapshot, device) if method == M1 else _LlmRunner(snapshot, device)
            )
            print(f"  loaded {method} on {device} ({time.monotonic() - began:.0f}s)")

    complete = all(
        (CACHE / f"{_slug(spec['environment'])}.{suffix}").is_file()
        for spec in s14.ENVIRONMENTS
        for suffix in ("requests.parquet", "spans.parquet", "m0_candidates.parquet")
    )
    pipeline = None if complete else s14.build_pipeline()
    costs: list[dict[str, Any]] = []
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        slug = _slug(name)
        requests = _prepare_environment(spec, pipeline)

        for method in arms:
            # An arm runs inside its declared language scope, where the implemented generator
            # would; the declared out-of-scope diagnostic environments are the only exception.
            if not method_applies(
                method, spec["corpus"]
            ) and name not in OUT_OF_SCOPE_DIAGNOSTIC.get(method, ()):
                continue
            for condition in (COND_FROZEN, COND_NATURAL):
                raw_path = RAW_CACHE / f"{slug}.{method}.{condition}.parquet"
                if raw_path.is_file():
                    continue
                subset = requests[requests["condition"] == condition]
                if subset.empty:
                    continue
                raw, cost = _run_arm(runners[method], subset, method, condition, print)
                raw.to_parquet(raw_path, index=False)
                _write_cost_sidecar(raw_path, name, method, condition, cost)
                costs.append(
                    {"environment": name, "method": method, "condition": condition, **cost}
                )
                print(
                    f"    {method}/{condition}: {cost['requests']} requests -> "
                    f"{cost['returned_sequences']} sequences in "
                    f"{cost['wall_clock_seconds']:.0f}s ({cost['requests_per_second']:.1f}/s)"
                )

    print(
        f"generate: {len(arms)} neural arms over {len(s14.ENVIRONMENTS)} environments in "
        f"{time.monotonic() - started:.0f}s -> {_relative(RAW_CACHE)}"
    )
    return 0


# ------------------------------------------------------------------ section 8, 10: normalize

METHOD_CANDIDATE_COLUMNS = (
    "environment",
    "corpus",
    "base_engine",
    "method",
    "condition",
    "candidate_id",
    "site_id",
    "document_id",
    "role",
    "edit_kind",
    "anchor_kind",
    "char_start",
    "char_end",
    "original_chars",
    "candidate_chars",
    "original_ocr",
    "candidate_text",
    "generator_rank",
    "candidate_source",
    "outcome",
    "is_harmful",
    "beneficial",
    "exact",
    "d_before",
    "d_after",
)
LABEL_SITE_COLUMNS = (
    "document_id",
    "engine_id",
    "site_id",
    "anchor_kind",
    "anchor_ref",
    "char_start",
    "char_end",
)


def _digest(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}-{canonical_hash(payload)}"


def _neural_edits(
    name: str,
    method: str,
    raw: pd.DataFrame,
    requests: pd.DataFrame,
    spans_by_pair: dict[tuple[str, str], list[tuple[str, int, int]]],
    parse: Callable[[str], str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Turn one arm's raw outputs into sites and candidates in the common representation.

    In the region conditions the arm was handed exactly the region it is scored on, so its output
    is already an atomic candidate. In the natural condition it rewrote a whole line and the
    shared diff converts that rewrite into atomic edits. Both paths end in the same columns, which
    is what makes the arms comparable at all.
    """
    census = {
        "raw_sequences": len(raw),
        "answers_parsed": 0,
        "delimited_answers": 0,
        "declined_unchanged": 0,
        "empty_output": 0,
        "region_candidates": 0,
        "line_rewrites_with_an_edit": 0,
        "opcodes": 0,
        "region_edits": 0,
        "gap_edits": 0,
        "gaps_absorbed_into_a_region": 0,
        "runs_merged_across_a_removed_separator": 0,
        "dropped_unprojectable": 0,
        "dropped_identity": 0,
        "dropped_whitespace_only": 0,
        "dropped_outside_any_span": 0,
        "duplicates_removed": 0,
    }
    by_request = {str(row.request_id): row for row in requests.itertuples(index=False)}
    sites: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []

    for row in raw.itertuples(index=False):
        request = by_request.get(str(row.request_id))
        if request is None:
            continue
        condition = str(row.condition)
        answer = str(row.output)
        if parse is None:
            output = answer
        else:
            # How often the arm honoured the frozen answer format, read with the parser's own test.
            output = parse(answer)
            opening = answer.find("[[")
            census["answers_parsed"] += 1
            census["delimited_answers"] += int(opening >= 0 and answer.find("]]", opening + 2) >= 0)
        rank = int(row.rank)
        pair = (str(request.document_id), str(request.engine_id))

        if condition != COND_NATURAL:
            original = str(request.region)
            # Whitespace-insensitive, so a model answering "no change" to a whitespace-only gap
            # region is read as declining rather than as proposing to delete the separator.
            if output.strip() == original.strip():
                census["declined_unchanged"] += 1
                continue
            if not output and not original:
                census["empty_output"] += 1
                continue
            site_id = str(request.site_id).split("|", 1)[1]
            sites.setdefault(
                site_id,
                {
                    "document_id": pair[0],
                    "engine_id": pair[1],
                    "site_id": site_id,
                    "anchor_kind": str(request.anchor_kind),
                    "anchor_ref": str(request.anchor_ref),
                    "char_start": int(request.char_start),
                    "char_end": int(request.char_end),
                },
            )
            proposals.append(
                {
                    "site_id": site_id,
                    "document_id": pair[0],
                    "engine_id": pair[1],
                    "edit_kind": EDIT_GAP if str(request.anchor_kind) == "gap" else EDIT_REGION,
                    "anchor_kind": str(request.anchor_kind),
                    "char_start": int(request.char_start),
                    "char_end": int(request.char_end),
                    "original_ocr": original,
                    "candidate_text": output,
                    "generator_rank": rank,
                }
            )
            census["region_candidates"] += 1
            continue

        window = str(request.window_text)
        if output.strip() == window.strip():
            census["declined_unchanged"] += 1
            continue
        start = int(request.window_start)
        local = [
            (span_id, span_start - start, span_end - start)
            for span_id, span_start, span_end in spans_by_pair.get(pair, [])
            if span_start >= start and span_end <= int(request.char_end)
        ]
        edits, shard = atomic_edits_from_rewrite(window, output, local, rank=rank)
        # Every counter the diff reports is summed, so a new one cannot be dropped silently.
        for key, value in shard.items():
            census[key] += value
        if edits:
            census["line_rewrites_with_an_edit"] += 1
        for edit in edits:
            anchor_ref = edit.anchor_ref
            site_id = _digest(
                "xc1-site",
                {
                    "schema": "sgv_xc1-natural-site-v1",
                    "environment": name,
                    "document_id": pair[0],
                    "engine_id": pair[1],
                    "anchor_ref": anchor_ref,
                    "char_start": start + edit.char_start,
                    "char_end": start + edit.char_end,
                },
            )
            sites.setdefault(
                site_id,
                {
                    "document_id": pair[0],
                    "engine_id": pair[1],
                    "site_id": site_id,
                    "anchor_kind": edit.anchor_kind,
                    "anchor_ref": anchor_ref,
                    "char_start": start + edit.char_start,
                    "char_end": start + edit.char_end,
                },
            )
            proposals.append(
                {
                    "site_id": site_id,
                    "document_id": pair[0],
                    "engine_id": pair[1],
                    "edit_kind": edit.edit_kind,
                    "anchor_kind": edit.anchor_kind,
                    "char_start": start + edit.char_start,
                    "char_end": start + edit.char_end,
                    "original_ocr": edit.original,
                    "candidate_text": edit.candidate,
                    "generator_rank": rank,
                }
            )

    frame = pd.DataFrame(proposals)
    if frame.empty:
        return (
            pd.DataFrame(columns=list(LABEL_SITE_COLUMNS)),
            pd.DataFrame(
                columns=[*LABEL_SITE_COLUMNS, "original_ocr", "candidate_text", "generator_rank"]
            ),
            census,
        )
    # Section 10. Deduplicate on the atomic edit, keeping the best rank the arm gave it. The key
    # is the site and the proposed text; no label is consulted, so a harmful candidate is never
    # removed and a correct one is never preferred.
    before = len(frame)
    frame = (
        frame.sort_values(["site_id", "candidate_text", "generator_rank"], kind="stable")
        .drop_duplicates(["site_id", "candidate_text"], keep="first")
        .reset_index(drop=True)
    )
    census["duplicates_removed"] = before - len(frame)
    frame["candidate_id"] = [
        _digest(
            "xc1-candidate",
            {
                "schema": "sgv_xc1-candidate-id-v1",
                "environment": name,
                "method": method,
                "site_id": str(row.site_id),
                "candidate_text": str(row.candidate_text),
            },
        )
        for row in frame.itertuples(index=False)
    ]
    site_frame = pd.DataFrame(list(sites.values()), columns=list(LABEL_SITE_COLUMNS))
    return site_frame, frame, census


ALIGNMENT_COLUMNS = (
    "environment",
    "corpus",
    "base_engine",
    "document_id",
    "role",
    "align_site_id",
    "site_kind",
    "evaluable",
    "d_before",
    "char_start",
    "char_end",
    "ocr_chars",
    "gt_chars",
)


def _alignment_population(
    spec: dict[str, Any],
    spans: dict[tuple[str, str], list[Any]],
    bundles: dict[str, Any],
    roles: dict[str, str],
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], Any],
    dict[Any, dict[str, str]],
    dict[Any, dict[str, str]],
    list[tuple[tuple[str, str], Any]],
]:
    """The OCR error site population and the maps that link a proposal to it.

    This is SGV-CG1's section 4 definition unchanged: an evaluable alignment component whose OCR
    text differs from its ground truth, after the adjacent-component merge. It is defined against
    ground truth and not against any generator, which is the only way "what fraction of OCR errors
    is reachable" can be asked of several methods at once.
    """
    from ocr_risk.align import align_document
    from ocr_risk.config.models import AlignmentConfig, SiteConfig
    from ocr_risk.edits.sites import build_sites
    from ocr_risk.experiments.cgv3_study import AlignmentIndex

    name = spec["environment"]
    alignment_config = AlignmentConfig()
    site_config = SiteConfig()
    rows: list[dict[str, Any]] = []
    indexes: dict[tuple[str, str], Any] = {}
    by_span: dict[Any, dict[str, str]] = {}
    by_alignment: dict[Any, dict[str, str]] = {}
    objects: list[tuple[tuple[str, str], Any]] = []
    for pair in sorted(spans):
        document_id, _engine_id = pair
        bundle = bundles[document_id]
        outcome = align_document(bundle.document, spans[pair], bundle.gt_tokens, alignment_config)
        records = [record.model_dump(mode="json") for record in outcome.records]
        indexes.update(
            AlignmentIndex.from_frames(
                pd.DataFrame(records),
                pd.DataFrame([token.model_dump(mode="json") for token in bundle.gt_tokens]),
            )
        )
        span_map: dict[str, str] = {}
        alignment_map: dict[str, str] = {}
        for site in build_sites(outcome.records, spans[pair], site_config):
            objects.append((pair, site))
            align_site_id = f"{name}|{site.site_id}"
            for span_id in site.ocr_span_ids:
                span_map[str(span_id)] = align_site_id
            for alignment_id in site.alignment_ids:
                alignment_map[str(alignment_id)] = align_site_id
            rows.append(
                {
                    "environment": name,
                    "corpus": spec["corpus"],
                    "base_engine": spec["base_engine"],
                    "document_id": document_id,
                    "role": roles.get(document_id, ""),
                    "align_site_id": align_site_id,
                    "site_kind": site.site_kind.value,
                    "evaluable": bool(site.evaluable),
                    "d_before": int(site.d_before),
                    "char_start": int(site.char_start),
                    "char_end": int(site.char_end),
                    "ocr_chars": len(site.ocr_text),
                    "gt_chars": len(site.gt_text),
                }
            )
        by_span[pair] = span_map
        by_alignment[pair] = alignment_map
    return (
        pd.DataFrame(rows, columns=list(ALIGNMENT_COLUMNS)),
        indexes,
        by_span,
        by_alignment,
        objects,
    )


def _m0_sites(m0_candidates: pd.DataFrame) -> pd.DataFrame:
    """M0's site frame, recovered from its own candidate stream.

    `_candidate_table` merges the site columns onto every candidate, so the sites that carry a
    candidate are recoverable exactly. Sites with no candidate cannot change any label and are
    not needed to reproduce one.
    """
    return m0_candidates[list(LABEL_SITE_COLUMNS)].drop_duplicates("site_id").reset_index(drop=True)


def _method_frame(
    spec: dict[str, Any],
    method: str,
    condition: str,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    roles: dict[str, str],
) -> pd.DataFrame:
    """One arm's labelled candidates, in the common output columns."""
    name = spec["environment"]
    merged = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    merged = merged[merged["labelable"]].reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(columns=list(METHOD_CANDIDATE_COLUMNS))
    return pd.DataFrame(
        {
            "environment": name,
            "corpus": spec["corpus"],
            "base_engine": spec["base_engine"],
            "method": method,
            "condition": condition,
            "candidate_id": merged["candidate_id"].astype(str).to_numpy(),
            "site_id": (name + "|" + merged["site_id"].astype(str)).to_numpy(),
            "document_id": merged["document_id"].astype(str).to_numpy(),
            "role": merged["document_id"].map(roles).to_numpy(str),
            "edit_kind": merged["edit_kind"].astype(str).to_numpy(),
            "anchor_kind": merged["anchor_kind"].astype(str).to_numpy(),
            "char_start": merged["char_start"].to_numpy(dtype=np.int64),
            "char_end": merged["char_end"].to_numpy(dtype=np.int64),
            "original_chars": merged["original_ocr"].astype(str).str.len().to_numpy(np.int64),
            "candidate_chars": merged["candidate_text"].astype(str).str.len().to_numpy(np.int64),
            "original_ocr": merged["original_ocr"].astype(str).to_numpy(),
            "candidate_text": merged["candidate_text"].astype(str).to_numpy(),
            "generator_rank": merged["generator_rank"].to_numpy(dtype=np.int64),
            "candidate_source": method,
            "outcome": merged["outcome"].astype(str).to_numpy(),
            "is_harmful": merged["is_harmful"].to_numpy(dtype=bool),
            "beneficial": merged["beneficial"].to_numpy(dtype=bool),
            "exact": (merged["outcome"].astype(str) == OUTCOME_EXACT).to_numpy(dtype=bool),
            "d_before": merged["d_before"].to_numpy(dtype=np.int64),
            "d_after": merged["d_after"].to_numpy(dtype=np.int64),
        },
        columns=list(METHOD_CANDIDATE_COLUMNS),
    )


def _stable_seed(*parts: str) -> int:
    """A seed that depends on its parts and CONTROL_SEED, and on nothing else in the process."""
    return int(canonical_hash({"seed": CONTROL_SEED, "parts": list(parts)})[:16], 16)


def _raw_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def _perturb(region: str, rng: np.random.Generator) -> str:
    """One seeded random character edit, drawn from the region's own characters and [a-z0-9].

    It reads the OCR region and the seed and nothing else, so it cannot prefer a correct edit.
    """
    alphabet = sorted(set(region.replace(" ", "")) | set(PERTURBATION_ALPHABET))
    operation = int(rng.integers(3)) if region else 1
    if operation == 1:
        position = int(rng.integers(len(region) + 1))
        return region[:position] + alphabet[int(rng.integers(len(alphabet)))] + region[position:]
    position = int(rng.integers(len(region)))
    if operation == 0:
        replacement = alphabet[int(rng.integers(len(alphabet)))]
        return region[:position] + replacement + region[position + 1 :]
    return region[:position] + region[position + 1 :]


def _m0_raw(name: str, m0_candidates: pd.DataFrame, frozen_requests: pd.DataFrame) -> pd.DataFrame:
    """M0's evaluation candidates in the raw-output shape, so C4 can shuffle it like any arm."""
    request_id = name + "|" + COND_FROZEN + "|" + m0_candidates["site_id"].astype(str)
    frame = pd.DataFrame(
        {
            "environment": name,
            "method": M0,
            "condition": COND_FROZEN,
            "request_id": request_id.to_numpy(),
            "rank": m0_candidates["generator_rank"].to_numpy(dtype=np.int64),
            "output": m0_candidates["candidate_text"].astype(str).to_numpy(),
        },
        columns=list(RAW_COLUMNS),
    )
    keep = frame["request_id"].isin(set(frozen_requests["request_id"].astype(str)))
    return frame[keep].reset_index(drop=True)


def _shuffle_raw(
    name: str, source: str, raw: pd.DataFrame, request_ids: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    """Section 48 C4: reassign every output to a seeded permutation of the environment's requests.

    Returns the shuffled frame and how many requests the permutation left in place, which is
    reported rather than engineered away.
    """
    ordered = sorted(request_ids)
    rng = np.random.default_rng(_stable_seed(name, "c4", source))
    permuted = [ordered[i] for i in rng.permutation(len(ordered))]
    mapping = dict(zip(ordered, permuted, strict=True))
    fixed = sum(1 for key, value in mapping.items() if key == value)
    shuffled = raw.assign(request_id=raw["request_id"].map(mapping), method=f"{C4_PREFIX}{source}")
    return shuffled.reset_index(drop=True), fixed


def _control_raws(
    name: str,
    frozen_requests: pd.DataFrame,
    natural_requests: pd.DataFrame,
    frozen_raw: dict[str, pd.DataFrame],
) -> list[tuple[str, str, pd.DataFrame, dict[str, Any]]]:
    """The GT-blind controls of section 48, produced in the raw-output shape.

    They pass through the same normalization and the same labelling call as the real arms, which
    is what makes them tests of the pipeline rather than of a separate code path.
    """
    out: list[tuple[str, str, pd.DataFrame, dict[str, Any]]] = []
    for condition, requests, field in (
        (COND_FROZEN, frozen_requests, "region"),
        (COND_NATURAL, natural_requests, "window_text"),
    ):
        out.append(
            (
                C1_IDENTITY,
                condition,
                _raw_frame(
                    [
                        {
                            "environment": name,
                            "method": C1_IDENTITY,
                            "condition": condition,
                            "request_id": str(row.request_id),
                            "rank": 0,
                            "output": str(getattr(row, field)),
                        }
                        for row in requests.itertuples(index=False)
                    ]
                ),
                {},
            )
        )
    perturbed: list[dict[str, Any]] = []
    for row in frozen_requests.itertuples(index=False):
        rng = np.random.default_rng(_stable_seed(name, C2_RANDOM, str(row.request_id)))
        region = str(row.region)
        for rank in range(C2_PER_SITE):
            perturbed.append(
                {
                    "environment": name,
                    "method": C2_RANDOM,
                    "condition": COND_FROZEN,
                    "request_id": str(row.request_id),
                    "rank": rank,
                    "output": _perturb(region, rng),
                }
            )
    out.append((C2_RANDOM, COND_FROZEN, _raw_frame(perturbed), {}))
    ids = frozen_requests["request_id"].astype(str).tolist()
    for source, raw in frozen_raw.items():
        shuffled, fixed = _shuffle_raw(name, source, raw, ids)
        out.append(
            (
                f"{C4_PREFIX}{source}",
                COND_FROZEN,
                shuffled,
                {"shuffled_from": source, "permutation_fixed_points": fixed},
            )
        )
    return out


def run_normalize() -> int:
    """Raw outputs to atomic edits to labels, every arm through SGV14's own labelling function.

    The union of every arm's sites and candidates -- and the controls' -- is labelled in one call
    per environment, so no arm is scored against a different notion of region ground truth than
    another, and M0's labels come out of literally the call that produces M1's and M2's. Site
    linkage is computed once per environment over the union of sites and read per arm: the same
    function on the same rows gives the same answer, and it is paid for once.
    """
    started = time.monotonic()
    _require(METHOD_REGISTRY, "registry")
    for path in (
        METHOD_CANDIDATES,
        METHOD_LINKS,
        AUXILIARY_CANDIDATES,
        AUXILIARY_LINKS,
        ALIGNMENT_SITES,
        NORMALIZATION_RESULTS,
    ):
        _forbid(path)
    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.schemas.enums import AnchorKind

    registry = cc_read_json(METHOD_REGISTRY)
    arms = [m for m in (M1, M2) if registry["methods"][m]["available"]]
    label_columns = [
        "candidate_id",
        "site_id",
        "document_id",
        "engine_id",
        "original_ocr",
        "candidate_text",
        "char_start",
        "char_end",
    ]

    alignments: list[pd.DataFrame] = []
    tables: dict[str, list[pd.DataFrame]] = {
        "candidates": [],
        "links": [],
        "auxiliary_candidates": [],
        "auxiliary_links": [],
    }
    censuses: list[dict[str, Any]] = []
    control_censuses: list[dict[str, Any]] = []
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        slug = _slug(name)
        began = time.monotonic()
        requests = pd.read_parquet(CACHE / f"{slug}.requests.parquet")
        m0_candidates = pd.read_parquet(CACHE / f"{slug}.m0_candidates.parquet")
        span_table = pd.read_parquet(CACHE / f"{slug}.spans.parquet")
        by_condition = {
            condition: requests[requests["condition"] == condition]
            for condition in (COND_FROZEN, COND_NATURAL)
        }

        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        spans, _record = s14._canonical_spans(spec, bundles)
        streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
        alignment, indexes, by_span, by_alignment, _objects = _alignment_population(
            spec, spans, bundles, roles
        )

        spans_by_pair: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
        for row in span_table.itertuples(index=False):
            spans_by_pair.setdefault((str(row.document_id), str(row.engine_id)), []).append(
                (str(row.span_id), int(row.char_start), int(row.char_end))
            )

        site_frames = [_m0_sites(m0_candidates)]
        primary: list[tuple[str, str, pd.DataFrame]] = [
            (
                M0,
                COND_FROZEN,
                m0_candidates.assign(
                    edit_kind=np.where(
                        m0_candidates["anchor_kind"].astype(str) == AnchorKind.GAP.value,
                        EDIT_GAP,
                        EDIT_REGION,
                    )
                ),
            )
        ]
        frozen_raw = {M0: _m0_raw(name, m0_candidates, by_condition[COND_FROZEN])}
        diagnostics: list[tuple[str, str, pd.DataFrame]] = []
        for method in arms:
            in_scope = method_applies(method, spec["corpus"])
            diagnostic = not in_scope and name in OUT_OF_SCOPE_DIAGNOSTIC.get(method, ())
            if not in_scope and not diagnostic:
                censuses.append(
                    {
                        "environment": name,
                        "method": method,
                        "condition": "",
                        "in_declared_language_scope": False,
                        "generated": False,
                        "note": (
                            "outside the declared language scope: the arm proposes nothing here"
                        ),
                    }
                )
                continue
            for condition in (COND_FROZEN, COND_NATURAL):
                raw_path = RAW_CACHE / f"{slug}.{method}.{condition}.parquet"
                if not raw_path.is_file():
                    raise PhaseError(f"missing raw outputs: {_relative(raw_path)}; run --generate")
                raw = pd.read_parquet(raw_path)
                if diagnostic:
                    diagnostics.append((f"{method}{DIAGNOSTIC_SUFFIX}", condition, raw))
                    continue
                if condition == COND_FROZEN:
                    frozen_raw[method] = raw
                site_frame, proposals, census = _neural_edits(
                    name,
                    method,
                    raw,
                    by_condition[condition],
                    spans_by_pair,
                    clean_llm_output if method == M2 else None,
                )
                censuses.append(
                    {
                        "environment": name,
                        "method": method,
                        "condition": condition,
                        "in_declared_language_scope": in_scope,
                        "language": language_of(spec["corpus"]),
                        "declared_scope": sorted(METHOD_LANGUAGES[method]),
                        **census,
                        "proposals": len(proposals),
                    }
                )
                if not proposals.empty:
                    site_frames.append(site_frame)
                    primary.append((method, condition, proposals))

        controls: list[tuple[str, str, pd.DataFrame]] = []
        for control, condition, raw, extra in _control_raws(
            name, by_condition[COND_FROZEN], by_condition[COND_NATURAL], frozen_raw
        ):
            site_frame, proposals, census = _neural_edits(
                name,
                control,
                raw,
                by_condition[condition],
                spans_by_pair,
                clean_llm_output if extra.get("shuffled_from") == M2 else None,
            )
            control_censuses.append(
                {
                    "environment": name,
                    "method": control,
                    "condition": condition,
                    **extra,
                    **census,
                    "proposals": len(proposals),
                }
            )
            if not proposals.empty:
                site_frames.append(site_frame)
                controls.append((control, condition, proposals))
        for label, condition, raw in diagnostics:
            site_frame, proposals, census = _neural_edits(
                name, label, raw, by_condition[condition], spans_by_pair
            )
            control_censuses.append(
                {
                    "environment": name,
                    "method": label,
                    "condition": condition,
                    "out_of_scope_diagnostic": True,
                    **census,
                    "proposals": len(proposals),
                }
            )
            if not proposals.empty:
                site_frames.append(site_frame)
                controls.append((label, condition, proposals))

        all_sites = (
            pd.concat(site_frames, ignore_index=True)
            .drop_duplicates("site_id")
            .reset_index(drop=True)
        )
        label_input = pd.concat(
            [frame[label_columns] for _m, _c, frame in primary + controls], ignore_index=True
        )
        if label_input["candidate_id"].duplicated().any():
            raise PhaseError(f"{name}: candidate ids collide inside the label input")
        labels, _diagnostics = s14._labels(
            spec["corpus"], bundles, spans, streams, all_sites, label_input
        )
        all_links = cg1._site_links(name, all_sites, indexes, by_span, by_alignment, AnchorKind)

        for frames, rows_key, links_key in (
            (primary, "candidates", "links"),
            (controls, "auxiliary_candidates", "auxiliary_links"),
        ):
            for method, condition, frame in frames:
                labelled = _method_frame(spec, method, condition, frame, labels, roles)
                if not labelled.empty:
                    tables[rows_key].append(labelled)
                ids = set((name + "|" + frame["site_id"].astype(str)).tolist())
                links = all_links[all_links["site_id"].isin(ids)]
                if not links.empty:
                    tables[links_key].append(links.assign(method=method, condition=condition))
        alignments.append(alignment)
        print(
            f"  {name}: {len(alignment)} alignment sites, {len(label_input)} labelled proposals "
            f"from {len(primary)} arm-conditions and {len(controls)} controls "
            f"({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    alignment_table = pd.concat(alignments, ignore_index=True)
    candidate_table = pd.concat(tables["candidates"], ignore_index=True)
    link_table = pd.concat(tables["links"], ignore_index=True)
    control_table = pd.concat(tables["auxiliary_candidates"], ignore_index=True)
    control_link_table = pd.concat(tables["auxiliary_links"], ignore_index=True)
    for table, label in ((candidate_table, "candidate"), (control_table, "control candidate")):
        if table["candidate_id"].duplicated().any():
            raise PhaseError(f"{label} ids collide across environments")
    if alignment_table["align_site_id"].duplicated().any():
        raise PhaseError("alignment site ids collide across environments")

    # Section 32. Attribution survives deduplication: an edit two arms both proposed names both of
    # them, and the per-arm rows are kept so a per-arm count stays exact. The key is the atomic
    # edit itself -- region and exact text -- so different texts at one region are never merged.
    key = ["environment", "condition", "document_id", "char_start", "char_end", "candidate_text"]
    pairs = candidate_table[[*key, "method"]].drop_duplicates()
    multi = pairs[pairs.groupby(key, sort=False)["method"].transform("size") > 1]
    shared = (
        multi.sort_values("method", kind="stable")
        .groupby(key, sort=False)["method"]
        .agg("+".join)
        .rename("candidate_source_set")
        .reset_index()
    )
    candidate_table = candidate_table.merge(shared, on=key, how="left", validate="many_to_one")
    candidate_table["candidate_source_set"] = candidate_table["candidate_source_set"].fillna(
        candidate_table["method"]
    )

    _write_parquet_once(ALIGNMENT_SITES, alignment_table)
    _write_parquet_once(METHOD_CANDIDATES, candidate_table)
    _write_parquet_once(METHOD_LINKS, link_table)
    _write_parquet_once(AUXILIARY_CANDIDATES, control_table)
    _write_parquet_once(AUXILIARY_LINKS, control_link_table)

    def by_method_condition(table: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "method": str(method),
                "condition": str(condition),
                "candidates": len(group),
                "candidates_evaluation": int((group["role"] == s14.ROLE_EVALUATION).sum()),
                "sites": int(group["site_id"].nunique()),
                "exact": int(group["exact"].sum()),
            }
            for (method, condition), group in table.groupby(["method", "condition"], sort=True)
        ]

    evaluation = candidate_table["role"] == s14.ROLE_EVALUATION
    _write_json_once(
        NORMALIZATION_RESULTS,
        {
            **_envelope("normalization_results"),
            "shared_diff": (
                "atomic_edits_from_rewrite, projecting through "
                "ocr_risk.candidates.byt5.project_span"
            ),
            "labelling": "one s14._labels call per environment over the union of arms and controls",
            "per_arm": censuses,
            "controls": control_censuses,
            "totals": {
                "alignment_sites": len(alignment_table),
                "candidates": len(candidate_table),
                "candidates_evaluation": int(evaluation.sum()),
                "links": len(link_table),
                "control_candidates": len(control_table),
                "methods": sorted(set(candidate_table["method"])),
                "conditions": sorted(set(candidate_table["condition"])),
            },
            "by_method_condition": by_method_condition(candidate_table),
            "controls_by_method_condition": by_method_condition(control_table),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"normalize: {len(candidate_table)} labelled candidates "
        f"({int(evaluation.sum())} on the evaluation split), {len(control_table)} control "
        f"candidates, {len(alignment_table)} alignment sites, {len(link_table)} links"
    )
    return 0


# ------------------------------------------------------------------ section 6, 11: the arm view

_site_index = cg1._site_index
_any_by_group = cg1._any_by_group
_max_by_group = cg1._max_by_group
_link_structure = cg1._link_structure


@dataclass(frozen=True, slots=True)
class ErrorSites:
    """One environment's OCR error site population -- the denominator every arm shares.

    Defined against ground truth by SGV-CG1's section 4 rule and therefore identical for every
    method. A method-specific denominator would let an arm look better by proposing on fewer
    sites, which is the confusion section 6 exists to prevent.
    """

    environment: str
    corpus: str
    base_engine: str
    ids: np.ndarray
    kind: np.ndarray
    document: np.ndarray
    d_before: np.ndarray
    position: dict[str, int]
    documents: np.ndarray

    @property
    def count(self) -> int:
        return int(self.ids.size)

    @property
    def characters(self) -> int:
        return int(self.d_before.sum())


@dataclass(frozen=True, slots=True)
class Arm:
    """One (method, condition, environment) view, in the units section 11 declares."""

    environment: str
    corpus: str
    base_engine: str
    method: str
    condition: str
    in_language_scope: bool
    errors: ErrorSites

    candidate_id: np.ndarray
    document_id: np.ndarray
    site_of: np.ndarray
    site_ids: np.ndarray
    exact: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    d_before: np.ndarray
    d_after: np.ndarray
    rank: np.ndarray
    anchor_kind: np.ndarray
    edit_kind: np.ndarray
    original_chars: np.ndarray
    candidate_chars: np.ndarray
    site_link_indptr: np.ndarray
    site_link_index: np.ndarray
    covered: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.exact.size)

    @property
    def n_sites(self) -> int:
        return int(self.site_ids.size)

    def linked(self, sites: np.ndarray) -> np.ndarray:
        """The OCR error sites reachable from a boolean mask over this arm's proposal sites."""
        out = np.zeros(self.errors.count, dtype=bool)
        for site in np.flatnonzero(sites):
            start, end = self.site_link_indptr[site], self.site_link_indptr[site + 1]
            out[self.site_link_index[start:end]] = True
        return out

    def budget(self, k: int | None) -> np.ndarray:
        """Section 9's matched budget: the first K proposals per site, by the arm's own order."""
        if k is None:
            return np.ones(self.n_rows, dtype=bool)
        return self.rank < k

    def repaired(self, k: int | None = None) -> np.ndarray:
        """OCR error sites for which this arm offers an exact repair within the budget."""
        keep = self.budget(k)
        return self.linked(_any_by_group(self.exact & keep, self.site_of, self.n_sites))

    def beneficial_sites(self, k: int | None = None) -> np.ndarray:
        keep = self.budget(k)
        return self.linked(_any_by_group(self.beneficial & keep, self.site_of, self.n_sites))

    def discovery_recall(self) -> float:
        return _ratio(int(self.covered.sum()), self.errors.count)

    def opportunity_recall(self, k: int | None = None) -> float:
        return _ratio(int(self.repaired(k).sum()), self.errors.count)

    def generation_success(self, k: int | None = None) -> float:
        return _ratio(int(self.repaired(k).sum()), int(self.covered.sum()))


def _error_sites(alignment: pd.DataFrame, spec: dict[str, Any]) -> ErrorSites:
    name = spec["environment"]
    block = alignment[alignment["environment"] == name]
    evaluable = block[block["evaluable"]]
    errors = evaluable[evaluable["d_before"] > 0].sort_values("align_site_id", kind="stable")
    ids = errors["align_site_id"].astype(str).to_numpy()
    return ErrorSites(
        environment=name,
        corpus=str(spec["corpus"]),
        base_engine=str(spec["base_engine"]),
        ids=ids,
        kind=errors["site_kind"].astype(str).to_numpy(),
        document=errors["document_id"].astype(str).to_numpy(),
        d_before=errors["d_before"].to_numpy(dtype=np.int64),
        position={value: index for index, value in enumerate(ids.tolist())},
        documents=np.asarray(sorted(set(evaluable["document_id"].astype(str))), dtype=str),
    )


def load_arms(
    candidates_path: Path = METHOD_CANDIDATES, links_path: Path = METHOD_LINKS
) -> tuple[dict[str, ErrorSites], dict[tuple[str, str, str], Arm]]:
    """Assemble every arm view from the frozen tables written by `--normalize`.

    The default is the primary family. The controls live in their own tables and are loaded only
    by naming those tables, so a primary analysis cannot pick one up by accident.
    """
    for path in (ALIGNMENT_SITES, candidates_path, links_path):
        if not path.is_file():
            raise PhaseError("run --normalize first")
    alignment = pd.read_parquet(ALIGNMENT_SITES)
    candidates = pd.read_parquet(candidates_path)
    links = pd.read_parquet(links_path)

    alignment = alignment[alignment["role"] == s14.ROLE_EVALUATION]
    candidates = candidates[candidates["role"] == s14.ROLE_EVALUATION]

    errors = {spec["environment"]: _error_sites(alignment, spec) for spec in s14.ENVIRONMENTS}
    arms: dict[tuple[str, str, str], Arm] = {}
    for (name, method, condition), rows in candidates.groupby(
        ["environment", "method", "condition"], sort=True
    ):
        rows = rows.sort_values(["site_id", "generator_rank", "candidate_id"], kind="stable")
        population = errors[str(name)]
        site_ids = rows["site_id"].astype(str).to_numpy()
        names, site_of = _site_index(site_ids)
        block = links[
            (links["environment"] == name)
            & (links["method"] == method)
            & (links["condition"] == condition)
        ]
        indptr, index, covered = _link_structure(block, names, population.position)
        arms[str(name), str(method), str(condition)] = Arm(
            environment=str(name),
            corpus=str(rows["corpus"].iloc[0]),
            base_engine=str(rows["base_engine"].iloc[0]),
            method=str(method),
            condition=str(condition),
            in_language_scope=method_applies(str(method), str(rows["corpus"].iloc[0])),
            errors=population,
            candidate_id=rows["candidate_id"].astype(str).to_numpy(),
            document_id=rows["document_id"].astype(str).to_numpy(),
            site_of=site_of,
            site_ids=names,
            exact=rows["exact"].to_numpy(dtype=bool),
            harmful=rows["is_harmful"].to_numpy(dtype=bool),
            beneficial=rows["beneficial"].to_numpy(dtype=bool),
            d_before=rows["d_before"].to_numpy(dtype=np.int64),
            d_after=rows["d_after"].to_numpy(dtype=np.int64),
            rank=rows["generator_rank"].to_numpy(dtype=np.int64),
            anchor_kind=rows["anchor_kind"].astype(str).to_numpy(),
            edit_kind=rows["edit_kind"].astype(str).to_numpy(),
            original_chars=rows["original_chars"].to_numpy(dtype=np.int64),
            candidate_chars=rows["candidate_chars"].to_numpy(dtype=np.int64),
            site_link_indptr=indptr,
            site_link_index=index,
            covered=covered,
        )
    # An arm that proposed nothing labelable in an environment has opportunity zero there, not an
    # undefined value. Without this fill a per-environment mean would silently average over only
    # the environments an arm happened to reach, and look better for having reached fewer.
    present = {(method, condition) for _name, method, condition in arms}
    for method, condition in sorted(present):
        for name, population in errors.items():
            if (name, method, condition) not in arms:
                arms[name, method, condition] = _empty_arm(population, method, condition)
    return errors, arms


def _empty_arm(population: ErrorSites, method: str, condition: str) -> Arm:
    """An arm with no candidates in this environment: every rate it reports is zero."""
    none_int = np.zeros(0, dtype=np.int64)
    none_bool = np.zeros(0, dtype=bool)
    none_obj = np.zeros(0, dtype=object)
    return Arm(
        environment=population.environment,
        corpus=population.corpus,
        base_engine=population.base_engine,
        method=method,
        condition=condition,
        in_language_scope=method_applies(method, population.corpus),
        errors=population,
        candidate_id=none_obj,
        document_id=none_obj,
        site_of=none_int,
        site_ids=np.zeros(0, dtype=str),
        exact=none_bool,
        harmful=none_bool,
        beneficial=none_bool,
        d_before=none_int,
        d_after=none_int,
        rank=none_int,
        anchor_kind=none_obj,
        edit_kind=none_obj,
        original_chars=none_int,
        candidate_chars=none_int,
        site_link_indptr=np.zeros(1, dtype=np.int64),
        site_link_index=none_int,
        covered=np.zeros(population.count, dtype=bool),
    )


# ------------------------------------------------------------------ section 5: reproduction gate


def _difference(label: str, observed: Any, expected: Any, tolerance: float = 0.0) -> dict[str, Any]:
    if isinstance(observed, float) or isinstance(expected, float):
        agrees = abs(float(observed) - float(expected)) <= tolerance
        delta: float = abs(float(observed) - float(expected))
    else:
        agrees = observed == expected
        delta = 0.0 if agrees else 1.0
    return {
        "check": label,
        "observed": observed,
        "expected": expected,
        "agrees": bool(agrees),
        "absolute_difference": delta,
    }


def run_reproduce() -> int:
    """Section 5. M0 must reproduce SGV-CG1 exactly, or SGV-XC1 does not continue.

    The comparison is against SGV-CG1's own saved tables and artifacts, row by row where the
    tables allow it. Candidate identity is checked through the candidate id, whose digest is
    taken over the candidate text among other fields, so an identical id set is an identical
    text set and no separate text column is needed to prove it.
    """
    started = time.monotonic()
    _forbid(UPSTREAM_REPRODUCTION)
    errors, arms = load_arms()

    mine = pd.read_parquet(METHOD_CANDIDATES)
    theirs = pd.read_parquet(cg1.CANDIDATE_ROWS)
    my_m0 = mine[mine["method"] == M0]
    checks = [
        _difference("m0_candidate_rows", len(my_m0), len(theirs)),
        _difference(
            "m0_candidate_ids",
            len(set(my_m0["candidate_id"]) ^ set(theirs["candidate_id"])),
            0,
        ),
    ]

    joined = my_m0.merge(
        theirs, on="candidate_id", how="inner", suffixes=("_mine", "_theirs"), validate="one_to_one"
    )
    checks.append(_difference("m0_rows_joined", len(joined), len(theirs)))
    for column in ("outcome", "is_harmful", "beneficial", "d_before", "d_after", "char_start"):
        checks.append(
            _difference(
                f"m0_label_{column}",
                int((joined[f"{column}_mine"] != joined[f"{column}_theirs"]).sum()),
                0,
            )
        )

    my_alignment = pd.read_parquet(ALIGNMENT_SITES)
    their_alignment = pd.read_parquet(cg1.ALIGNMENT_SITES)
    checks.append(_difference("alignment_sites", len(my_alignment), len(their_alignment)))
    checks.append(
        _difference(
            "alignment_site_ids",
            len(set(my_alignment["align_site_id"]) ^ set(their_alignment["align_site_id"])),
            0,
        )
    )
    align_join = my_alignment.merge(
        their_alignment, on="align_site_id", suffixes=("_mine", "_theirs"), validate="one_to_one"
    )
    for column in ("site_kind", "evaluable", "d_before"):
        checks.append(
            _difference(
                f"alignment_{column}",
                int((align_join[f"{column}_mine"] != align_join[f"{column}_theirs"]).sum()),
                0,
            )
        )

    # SGV-CG1 linked every discovered site, including sites whose candidates were all unlabelable;
    # this stage links the sites its arms proposed at. They are compared on the sites carrying a
    # labelable candidate in SGV-CG1's own table -- the only sites whose links reach any SGV-CG1
    # metric -- which is the same function on the same rows, so any difference is a real one.
    candidate_sites = set(theirs["site_id"].astype(str))
    link_columns = ["environment", "site_id", "align_site_id", "link_kind"]
    my_links = pd.read_parquet(METHOD_LINKS)
    my_m0_links = my_links[(my_links["method"] == M0) & my_links["site_id"].isin(candidate_sites)]
    their_links = pd.read_parquet(cg1.SITE_LINKS)
    their_links = their_links[their_links["site_id"].isin(candidate_sites)]
    mine_set = set(map(tuple, my_m0_links[link_columns].to_numpy().tolist()))
    theirs_set = set(map(tuple, their_links[link_columns].to_numpy().tolist()))
    checks.append(_difference("m0_site_link_rows", len(mine_set), len(theirs_set)))
    checks.append(_difference("m0_site_links", len(mine_set ^ theirs_set), 0))

    # The headline quantities, recomputed from this stage's own tables and compared with the ones
    # SGV-CG1 published. These are the numbers every SGV-XC1 delta is taken against.
    published = {
        row["environment"]: row for row in _cg1("candidate_opportunity")["per_environment"]
    }
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        arm = arms[name, M0, COND_FROZEN]
        reference = published[name]
        checks.append(
            _difference(
                f"opportunity_recall::{name}",
                arm.opportunity_recall(),
                float(reference["opportunity_recall"]),
                tolerance=1e-12,
            )
        )
        checks.append(
            _difference(
                f"discovery_recall::{name}",
                arm.discovery_recall(),
                float(reference["discovery_recall"]),
                tolerance=1e-12,
            )
        )
        checks.append(
            _difference(
                f"ocr_error_sites::{name}", arm.errors.count, int(reference["ocr_error_sites"])
            )
        )
        checks.append(
            _difference(
                f"candidate_multiplicity::{name}",
                _ratio(arm.n_rows, arm.n_sites),
                float(reference["candidate_multiplicity"]),
                tolerance=1e-12,
            )
        )

    for record in _cg1("error_type_analysis")["per_error_type"]:
        kind = str(record["site_kind"])
        observed = _pooled_error_type_opportunity(arms, errors, M0, COND_FROZEN, kind)
        checks.append(
            _difference(
                f"error_type_opportunity::{kind}",
                observed,
                float(record["opportunity_recall"]),
                tolerance=1e-12,
            )
        )

    failures = [check for check in checks if not check["agrees"]]
    maximum = max((check["absolute_difference"] for check in checks), default=0.0)
    _write_json_once(
        UPSTREAM_REPRODUCTION,
        {
            **_envelope("upstream_reproduction"),
            "gate": (
                "section 5: M0 reproduces SGV-CG1 exactly on candidate identity, candidate "
                "labels, the alignment population, the site links and every headline quantity."
            ),
            "candidate_identity_note": (
                "SGV14's candidate id is a digest over the environment, site, generator, "
                "operation and the candidate text, so an identical id set is an identical "
                "candidate text set."
            ),
            "checks": checks,
            "checks_total": len(checks),
            "checks_agreeing": len(checks) - len(failures),
            "differences": len(failures),
            "maximum_absolute_difference": maximum,
            "failures": failures,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if failures:
        raise PhaseError(
            f"M0 does not reproduce SGV-CG1: {len(failures)} of {len(checks)} checks differ; "
            f"see {_relative(UPSTREAM_REPRODUCTION)}"
        )
    print(
        f"reproduce: {len(checks)} checks, 0 differences, maximum absolute difference "
        f"{maximum:.1f} -- M0 reproduces SGV-CG1"
    )
    return 0


def _pooled_error_type_opportunity(
    arms: dict[tuple[str, str, str], Arm],
    errors: dict[str, ErrorSites],
    method: str,
    condition: str,
    kind: str,
    k: int | None = None,
) -> float:
    """Opportunity recall inside one frozen error-type category, pooled over environments."""
    repaired = 0
    total = 0
    for name in errors:
        arm = arms.get((name, method, condition))
        if arm is None:
            total += int((errors[name].kind == kind).sum())
            continue
        selector = arm.errors.kind == kind
        total += int(selector.sum())
        repaired += int((arm.repaired(k) & selector).sum())
    return _ratio(repaired, total)


# ------------------------------------------------------------------ section 6, 11, 22, 44


def _arm_row(arm: Arm, k: int | None = None) -> dict[str, Any]:
    """Every section-11 quantity for one arm, on the shared OCR error site denominator."""
    keep = arm.budget(k)
    repaired = arm.repaired(k)
    beneficial_sites = arm.beneficial_sites(k)
    covered = int(arm.covered.sum())
    denominator = arm.errors.count
    rows = int(keep.sum())
    sites = int(np.unique(arm.site_of[keep]).size) if rows else 0
    return {
        "environment": arm.environment,
        "corpus": arm.corpus,
        "base_engine": arm.base_engine,
        "method": arm.method,
        "condition": arm.condition,
        "in_declared_language_scope": arm.in_language_scope,
        "k": k,
        "ocr_error_sites": denominator,
        "ocr_error_characters": arm.errors.characters,
        "proposal_sites": sites,
        "candidates": rows,
        "discovered_error_sites": covered,
        "repairable_sites": int(repaired.sum()),
        "beneficial_available_sites": int(beneficial_sites.sum()),
        "discovery_recall": _ratio(covered, denominator),
        "generation_success": _ratio(int(repaired.sum()), covered),
        "opportunity_recall": _ratio(int(repaired.sum()), denominator),
        "opportunity_recall_beneficial": _ratio(int(beneficial_sites.sum()), denominator),
        "unrepairable_fraction": 1.0 - _ratio(int(repaired.sum()), denominator),
        "discovery_loss": 1.0 - _ratio(covered, denominator),
        "generation_loss": _ratio(covered - int(repaired.sum()), denominator),
        "candidates_per_site": _ratio(rows, sites),
        "exact_candidates": int((arm.exact & keep).sum()),
        "beneficial_candidates": int((arm.beneficial & keep).sum()),
        "harmful_candidates": int((arm.harmful & keep).sum()),
        "neutral_candidates": int((~arm.beneficial & ~arm.harmful & keep).sum()),
        "exact_candidates_per_site": _ratio(int((arm.exact & keep).sum()), sites),
        "harmful_candidates_per_site": _ratio(int((arm.harmful & keep).sum()), sites),
        "harmful_to_beneficial_ratio": _ratio(
            int((arm.harmful & keep).sum()), int((arm.beneficial & keep).sum())
        ),
        "opportunity_precision": _ratio(int((arm.beneficial & keep).sum()), rows),
    }


def _rows_for(
    arms: dict[tuple[str, str, str], Arm], condition: str, k: int | None = None
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _arm_row(arm, k)
            for (_name, _method, arm_condition), arm in sorted(arms.items())
            if arm_condition == condition
        ]
    )


def _pooled(frame: pd.DataFrame) -> dict[str, Any]:
    """Pooled over environments: a site-weighted reading, reported beside the per-arm mean."""
    if frame.empty:
        return {}
    return {
        "ocr_error_sites": int(frame["ocr_error_sites"].sum()),
        "discovered_error_sites": int(frame["discovered_error_sites"].sum()),
        "repairable_sites": int(frame["repairable_sites"].sum()),
        "candidates": int(frame["candidates"].sum()),
        "discovery_recall": _ratio(
            int(frame["discovered_error_sites"].sum()), int(frame["ocr_error_sites"].sum())
        ),
        "generation_success": _ratio(
            int(frame["repairable_sites"].sum()), int(frame["discovered_error_sites"].sum())
        ),
        "opportunity_recall": _ratio(
            int(frame["repairable_sites"].sum()), int(frame["ocr_error_sites"].sum())
        ),
        "harmful_candidates": int(frame["harmful_candidates"].sum()),
        "beneficial_candidates": int(frame["beneficial_candidates"].sum()),
    }


def _method_summary(frame: pd.DataFrame, method: str, scope_only: bool = False) -> dict[str, Any]:
    block = frame[frame["method"] == method]
    if scope_only:
        block = block[block["in_declared_language_scope"]]
    if block.empty:
        return {"method": method, "environments": 0}
    return {
        "method": method,
        "environments": len(block),
        "in_declared_language_scope_only": scope_only,
        "mean_discovery_recall": _mean(list(block["discovery_recall"])),
        "mean_generation_success": _mean(list(block["generation_success"])),
        "environments_with_proposals": int((block["proposal_sites"] > 0).sum()),
        "mean_opportunity_recall": _mean(list(block["opportunity_recall"])),
        "mean_opportunity_recall_beneficial": _mean(list(block["opportunity_recall_beneficial"])),
        "mean_candidates_per_site": _mean(list(block["candidates_per_site"])),
        "mean_harmful_candidates_per_site": _mean(list(block["harmful_candidates_per_site"])),
        "mean_opportunity_precision": _mean(list(block["opportunity_precision"])),
        "mean_discovery_loss": _mean(list(block["discovery_loss"])),
        "mean_generation_loss": _mean(list(block["generation_loss"])),
        "pooled": _pooled(block),
    }


def _deltas(frame: pd.DataFrame, method: str, scope_only: bool = False) -> dict[str, Any]:
    """Section 22. The effect, and the decomposition that says why it happened."""
    baseline = frame[frame["method"] == M0].set_index("environment")
    block = frame[frame["method"] == method].set_index("environment")
    if scope_only:
        block = block[block["in_declared_language_scope"]]
    shared = [name for name in block.index if name in baseline.index]
    if not shared:
        return {"method": method, "environments": 0}
    per_environment = [
        {
            "environment": name,
            "in_declared_language_scope": bool(block.loc[name, "in_declared_language_scope"]),
            "opportunity_recall": float(block.loc[name, "opportunity_recall"]),
            "baseline_opportunity_recall": float(baseline.loc[name, "opportunity_recall"]),
            "delta_opportunity_recall": float(
                block.loc[name, "opportunity_recall"] - baseline.loc[name, "opportunity_recall"]
            ),
            "delta_discovery_recall": float(
                block.loc[name, "discovery_recall"] - baseline.loc[name, "discovery_recall"]
            ),
            "delta_generation_success": float(
                block.loc[name, "generation_success"] - baseline.loc[name, "generation_success"]
            ),
            "delta_harmful_candidates_per_site": float(
                block.loc[name, "harmful_candidates_per_site"]
                - baseline.loc[name, "harmful_candidates_per_site"]
            ),
        }
        for name in shared
    ]
    improved = [row for row in per_environment if row["delta_opportunity_recall"] > 0]
    meaningful = [
        row for row in per_environment if row["delta_opportunity_recall"] >= MEANINGFUL_DELTA
    ]
    return {
        "method": method,
        "in_declared_language_scope_only": scope_only,
        "environments": len(shared),
        "per_environment": per_environment,
        "mean_delta_opportunity_recall": _mean(
            [row["delta_opportunity_recall"] for row in per_environment]
        ),
        "mean_delta_discovery_recall": _mean(
            [row["delta_discovery_recall"] for row in per_environment]
        ),
        "mean_delta_generation_success": _mean(
            [row["delta_generation_success"] for row in per_environment]
        ),
        "environments_improved": len(improved),
        "environments_meaningfully_improved": len(meaningful),
        "environments_meaningfully_improved_names": [row["environment"] for row in meaningful],
        "meets_meaningful_delta": bool(
            _mean([row["delta_opportunity_recall"] for row in per_environment]) >= MEANINGFUL_DELTA
        ),
        "meets_breadth": len(meaningful) >= BREADTH_MAJORITY,
    }


def _available_methods(arms: dict[tuple[str, str, str], Arm]) -> list[str]:
    present = {method for _name, method, _condition in arms}
    return [method for method in ALL_METHODS if method in present]


def run_discovery() -> int:
    """Section 6. Where each arm proposes at all, before any question about repair quality."""
    started = time.monotonic()
    _forbid(DISCOVERY_RESULTS)
    _errors, arms = load_arms()
    methods = _available_methods(arms)
    per_condition = {}
    for condition in (COND_FROZEN, COND_NATURAL):
        frame = _rows_for(arms, condition)
        if frame.empty:
            continue
        per_condition[condition] = {
            "per_arm": frame[
                [
                    "environment",
                    "corpus",
                    "base_engine",
                    "method",
                    "in_declared_language_scope",
                    "ocr_error_sites",
                    "proposal_sites",
                    "discovered_error_sites",
                    "discovery_recall",
                    "discovery_loss",
                ]
            ].to_dict("records"),
            "by_method": [_method_summary(frame, method) for method in methods],
            "by_method_in_scope": [
                _method_summary(frame, method, scope_only=True) for method in methods
            ],
        }
    _write_json_once(
        DISCOVERY_RESULTS,
        {
            **_oracle_envelope("discovery_results"),
            "linking_rule": (
                "SGV-CG1's: a proposal site links to every alignment site its anchor spans belong "
                "to, and a gap anchor additionally to the OCR-empty components between its "
                "flanks. Method-agnostic, and the same coupling the labelling stage reads region "
                "ground truth through."
            ),
            "definition": (
                "an OCR error site is discovered by a method when the method proposed at least "
                "one labelable candidate at a site that links to it. In the frozen-localization "
                "condition every arm is offered the same sites, so discovery recall there "
                "measures where an arm chose to propose, not what it was allowed to see."
            ),
            "conditions": per_condition,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    frozen = _rows_for(arms, COND_FROZEN)
    print(
        "discovery: "
        + ", ".join(
            f"{method} {_mean(list(frozen[frozen['method'] == method]['discovery_recall'])):.4f}"
            for method in methods
        )
        + " (frozen localization)"
    )
    return 0


def run_generation() -> int:
    """Section 6. Given that an arm proposed at an error site, how often is the repair exact."""
    started = time.monotonic()
    _forbid(GENERATION_RESULTS)
    _errors, arms = load_arms()
    methods = _available_methods(arms)
    per_condition = {}
    for condition in (COND_FROZEN, COND_NATURAL):
        frame = _rows_for(arms, condition)
        if frame.empty:
            continue
        per_condition[condition] = {
            "per_arm": frame[
                [
                    "environment",
                    "method",
                    "in_declared_language_scope",
                    "discovered_error_sites",
                    "repairable_sites",
                    "generation_success",
                    "generation_loss",
                ]
            ].to_dict("records"),
            "by_method": [_method_summary(frame, method) for method in methods],
        }
    _write_json_once(
        GENERATION_RESULTS,
        {
            **_oracle_envelope("generation_results"),
            "definition": (
                "conditional generation success: discovered error sites carrying an exact repair "
                "candidate, over discovered error sites. Section 44's second term."
            ),
            "decomposition": (
                "1 -> discovery_recall -> opportunity_recall, with discovery_loss = 1 - "
                "discovery_recall and generation_loss = discovery_recall - opportunity_recall. "
                "A high discovery recall with a low generation success is a generation limit; "
                "the reverse is a localization limit."
            ),
            "conditions": per_condition,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    frozen = _rows_for(arms, COND_FROZEN)
    print(
        "generation: "
        + ", ".join(
            f"{method} {_mean(list(frozen[frozen['method'] == method]['generation_success'])):.4f}"
            for method in methods
        )
    )
    return 0


def run_opportunity() -> int:
    """Section 11 and 22. The headline: opportunity recall per arm and its delta against M0."""
    started = time.monotonic()
    _forbid(OPPORTUNITY_RESULTS)
    _errors, arms = load_arms()
    methods = _available_methods(arms)
    baseline = float(_cg1("research_decision")["candidate_opportunity_mean"])

    conditions: dict[str, Any] = {}
    for condition in (COND_FROZEN, COND_NATURAL):
        frame = _rows_for(arms, condition)
        if frame.empty:
            continue
        conditions[condition] = {
            "per_arm": frame.to_dict("records"),
            "by_method": [_method_summary(frame, method) for method in methods],
            "by_method_in_scope": [
                _method_summary(frame, method, scope_only=True) for method in methods
            ],
            "deltas": [_deltas(frame, method) for method in methods if method != M0],
            "deltas_in_scope": [
                _deltas(frame, method, scope_only=True) for method in methods if method != M0
            ],
        }

    primary = conditions[PRIMARY_CONDITION]
    ranking = sorted(
        (
            (summary["method"], float(summary.get("mean_opportunity_recall", float("nan"))))
            for summary in primary["by_method"]
        ),
        key=lambda item: (-item[1], item[0]),
    )
    best_method, best_value = ranking[0]
    _write_json_once(
        OPPORTUNITY_RESULTS,
        {
            **_oracle_envelope("opportunity_results"),
            "primary_condition": PRIMARY_CONDITION,
            "primary_unit": "OCR error site",
            "exact_repair_is_primary": True,
            "baseline": {"method": M0, "sgv_cg1_opportunity_recall": baseline},
            "conditions": conditions,
            "ranking_in_primary_condition": [
                {"method": method, "mean_opportunity_recall": value} for method, value in ranking
            ],
            "best_method": best_method,
            "best_mean_opportunity_recall": best_value,
            "meaningful_delta": MEANINGFUL_DELTA,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "opportunity ("
        + PRIMARY_CONDITION
        + "): "
        + ", ".join(f"{method} {value:.4f}" for method, value in ranking)
    )
    return 0


# ------------------------------------------------------------------ section 9, 40: matched budget


def run_matchedk() -> int:
    """Section 9 and 40. The same comparison at an equalized candidate budget.

    Every K is a prefix of the arm's own frozen ordering, so truncation re-runs no model and
    reads no label. An arm that only ever returns one string is recorded as not supporting a
    multi-candidate budget rather than being given fabricated K values.
    """
    started = time.monotonic()
    _forbid(MATCHED_K_RESULTS)
    _errors, arms = load_arms()
    methods = _available_methods(arms)

    curves: dict[str, Any] = {}
    for condition in (COND_FROZEN, COND_NATURAL):
        present = {method for _n, method, c in arms if c == condition}
        if not present:
            continue
        supports = {
            method: bool(
                max(
                    (
                        int(arm.rank.max()) + 1
                        for (_n, m, c), arm in arms.items()
                        if m == method and c == condition and arm.n_rows
                    ),
                    default=1,
                )
                > 1
            )
            for method in methods
            if method in present
        }
        per_k = []
        for k in K_GRID:
            frame = _rows_for(arms, condition, k)
            for method in methods:
                if method not in present:
                    continue
                summary = _method_summary(frame, method)
                per_k.append(
                    {
                        "k": k,
                        "method": method,
                        "supports_multi_candidate": supports[method],
                        "mean_opportunity_recall": summary["mean_opportunity_recall"],
                        "mean_discovery_recall": summary["mean_discovery_recall"],
                        "mean_generation_success": summary["mean_generation_success"],
                        "mean_candidates_per_site": summary["mean_candidates_per_site"],
                        "mean_harmful_candidates_per_site": summary[
                            "mean_harmful_candidates_per_site"
                        ],
                        "pooled_opportunity_recall": summary["pooled"]["opportunity_recall"],
                    }
                )
        primary_frame = _rows_for(arms, condition, PRIMARY_K)
        curves[condition] = {
            "supports_multi_candidate": supports,
            "per_k": per_k,
            "at_primary_k": {
                "k": PRIMARY_K,
                "by_method": [
                    _method_summary(primary_frame, method)
                    for method in methods
                    if method in present
                ],
                "deltas": [
                    _deltas(primary_frame, method)
                    for method in methods
                    if method != M0 and method in present
                ],
            },
        }

    _write_json_once(
        MATCHED_K_RESULTS,
        {
            **_oracle_envelope("matched_k_results"),
            "k_grid": list(K_GRID),
            "primary_k": PRIMARY_K,
            "truncation_reads_no_label": True,
            "note": (
                "section 9: a generative arm can emit arbitrarily many candidates, so a gain that "
                "exists only at an unequal budget is a burden effect, not a ceiling effect. "
                "Criterion 4 is evaluated at K="
                f"{BURDEN_CONDITION_K}."
            ),
            "conditions": curves,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    primary = curves[PRIMARY_CONDITION]["at_primary_k"]["by_method"]
    print(
        f"matchedk: at K={PRIMARY_K} "
        + ", ".join(f"{s['method']} {s['mean_opportunity_recall']:.4f}" for s in primary)
    )
    return 0


# ------------------------------------------------------------------ section 14, 15: harmful side


def run_harmful() -> int:
    """Sections 14 and 15. The burden that bought whatever opportunity was gained."""
    started = time.monotonic()
    for path in (HARMFUL_ANALYSIS, CANDIDATE_MULTIPLICITY):
        _forbid(path)
    _errors, arms = load_arms()
    methods = _available_methods(arms)

    harmful: dict[str, Any] = {}
    multiplicity: dict[str, Any] = {}
    for condition in (COND_FROZEN, COND_NATURAL):
        frame = _rows_for(arms, condition)
        if frame.empty:
            continue
        harmful[condition] = {
            "per_arm": frame[
                [
                    "environment",
                    "method",
                    "candidates",
                    "exact_candidates",
                    "beneficial_candidates",
                    "harmful_candidates",
                    "neutral_candidates",
                    "harmful_candidates_per_site",
                    "harmful_to_beneficial_ratio",
                    "opportunity_precision",
                    "opportunity_recall",
                ]
            ].to_dict("records"),
            "by_method": [
                {
                    **_method_summary(frame, method),
                    "pooled_harmful_share": _ratio(
                        int(frame[frame["method"] == method]["harmful_candidates"].sum()),
                        int(frame[frame["method"] == method]["candidates"].sum()),
                    ),
                }
                for method in methods
            ],
        }
        per_method = {}
        for method in methods:
            counts: list[float] = []
            for (_n, m, c), arm in sorted(arms.items()):
                if m != method or c != condition or not arm.n_rows:
                    continue
                counts.extend(np.bincount(arm.site_of, minlength=arm.n_sites).tolist())
            if counts:
                per_method[method] = {
                    **_quantiles(counts),
                    "sites": len(counts),
                    "candidates": int(sum(counts)),
                }
        multiplicity[condition] = per_method

    _write_json_once(
        HARMFUL_ANALYSIS,
        {
            **_oracle_envelope("harmful_candidate_analysis"),
            "definition": (
                "harm is the frozen STRICT_WORSENING policy: a candidate is harmful when "
                "accepting it would increase the region's distance to ground truth. Opportunity "
                "precision is beneficial candidates over all valid proposed candidates."
            ),
            "conditions": harmful,
            "note": (
                "section 14: an arm must not win by proposing more. Every opportunity number in "
                "this stage has its harmful burden reported beside it."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        CANDIDATE_MULTIPLICITY,
        {
            **_oracle_envelope("candidate_multiplicity"),
            "unit": "candidates per proposal site",
            "conditions": multiplicity,
            "question": (
                "section 15: does the arm improve opportunity by generating useful candidates or "
                "merely many? The matched-K analysis is what answers it; this is the distribution "
                "that makes the question concrete."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    frozen = harmful.get(PRIMARY_CONDITION, {}).get("by_method", [])
    print(
        "harmful: "
        + ", ".join(
            f"{s['method']} {s['mean_harmful_candidates_per_site']:.2f}/site" for s in frozen
        )
    )
    return 0


# ------------------------------------------------------------------ section 16, 18, 19, 20


def _stratified(
    arms: dict[tuple[str, str, str], Arm],
    errors: dict[str, ErrorSites],
    condition: str,
    method: str,
    selector: Callable[[ErrorSites, str], np.ndarray],
    keys: Sequence[str],
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Opportunity inside a stratum of the error-site population, pooled over environments.

    The stratum is a property of the OCR error site, never of the candidate, so every arm is
    measured against the same partition of the same denominator.
    """
    out: list[dict[str, Any]] = []
    for key in keys:
        total = 0
        covered = 0
        repaired = 0
        characters = 0
        candidates = 0
        harmful = 0
        for name, population in errors.items():
            mask = selector(population, key)
            total += int(mask.sum())
            characters += int(population.d_before[mask].sum())
            arm = arms.get((name, method, condition))
            if arm is None:
                continue
            covered += int((arm.covered & mask).sum())
            repaired += int((arm.repaired(k) & mask).sum())
            candidates += arm.n_rows
            harmful += int(arm.harmful.sum())
        out.append(
            {
                "stratum": key,
                "method": method,
                "ocr_error_sites": total,
                "ocr_error_characters": characters,
                "discovered_error_sites": covered,
                "repairable_sites": repaired,
                "discovery_recall": _ratio(covered, total),
                "generation_success": _ratio(repaired, covered),
                "opportunity_recall": _ratio(repaired, total),
            }
        )
    return out


def _typography(spec: dict[str, Any]) -> str:
    """The engine-side typography configuration, read from the frozen environment spec.

    This describes how the engine was configured, not a claim about the page. Section 20 asks for
    the distinction only where the artifacts support it, and `lang` is what they support.
    """
    language = str(spec.get("params", {}).get("lang", "") or "")
    if language == "frk":
        return "fraktur_configured_engine"
    return "modern_forms" if spec["corpus"] == "funsd" else "historical_print"


def run_strata() -> int:
    """Sections 16, 18, 19 and 20. Error type is a primary analysis and is never aggregated away."""
    started = time.monotonic()
    for path in (ERROR_TYPE_ANALYSIS, ENGINE_ANALYSIS, DOMAIN_ANALYSIS):
        _forbid(path)
    errors, arms = load_arms()
    methods = _available_methods(arms)
    # A stratum is reported only for an arm that exists in that condition. M0 has no whole-text
    # mode, so it has no natural-condition strata -- not strata of zero.
    present = {
        condition: [m for m in methods if any((n, m, condition) in arms for n in errors)]
        for condition in (COND_FROZEN, COND_NATURAL)
    }
    error_types = sorted({kind for population in errors.values() for kind in population.kind})
    engines = sorted({population.base_engine for population in errors.values()})
    domains = sorted({population.corpus for population in errors.values()})
    typography = {spec["environment"]: _typography(spec) for spec in s14.ENVIRONMENTS}

    by_type = {
        condition: {
            method: _stratified(
                arms, errors, condition, method, lambda p, key: p.kind == key, error_types
            )
            for method in present[condition]
        }
        for condition in (COND_FROZEN, COND_NATURAL)
    }
    baseline = {row["stratum"]: row["opportunity_recall"] for row in by_type[PRIMARY_CONDITION][M0]}
    recovery = {}
    for method in methods:
        if method == M0:
            continue
        rows = {
            row["stratum"]: row["opportunity_recall"] for row in by_type[PRIMARY_CONDITION][method]
        }
        improved = [
            kind
            for kind in WEAK_ERROR_TYPES
            if rows.get(kind, 0.0) - baseline.get(kind, 0.0) >= WEAK_TYPE_DELTA
        ]
        recovery[method] = {
            "weak_error_types": list(WEAK_ERROR_TYPES),
            "threshold": WEAK_TYPE_DELTA,
            "delta_by_type": {
                kind: rows.get(kind, 0.0) - baseline.get(kind, 0.0) for kind in error_types
            },
            "weak_types_improved": improved,
            "weak_types_improved_count": len(improved),
            "meets_criterion": len(improved) >= WEAK_TYPES_REQUIRED,
            "improves_only_the_strong_type": bool(
                rows.get(STRONG_ERROR_TYPE, 0.0) - baseline.get(STRONG_ERROR_TYPE, 0.0)
                >= WEAK_TYPE_DELTA
                and not improved
            ),
        }

    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_oracle_envelope("error_type_analysis"),
            "taxonomy_source": (
                "ocr_risk.schemas.enums.SiteKind, assigned by ocr_risk.edits.sites.site_kind_for "
                "from the alignment relation. The frozen names are used unchanged; the brief's "
                "'omission' is this taxonomy's `deletion`."
            ),
            "error_types": error_types,
            "weak_error_types": list(WEAK_ERROR_TYPES),
            "strong_error_type": STRONG_ERROR_TYPE,
            "by_condition": by_type,
            "weak_type_recovery": recovery,
            "matrix_note": (
                "section 38's central table: one row per method, one column per frozen error type."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        ENGINE_ANALYSIS,
        {
            **_oracle_envelope("engine_analysis"),
            "engines": engines,
            "by_condition": {
                condition: {
                    method: _stratified(
                        arms,
                        errors,
                        condition,
                        method,
                        lambda p, key: np.full(p.count, p.base_engine == key, dtype=bool),
                        engines,
                    )
                    for method in present[condition]
                }
                for condition in (COND_FROZEN, COND_NATURAL)
            },
            "question": (
                "section 18: does a correction family help consistently across OCR engines, or "
                "only on one error distribution?"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DOMAIN_ANALYSIS,
        {
            **_oracle_envelope("domain_analysis"),
            "domains": domains,
            "language_of_domain": LANGUAGE_OF_CORPUS,
            "typography_by_environment": typography,
            "by_condition": {
                condition: {
                    method: _stratified(
                        arms,
                        errors,
                        condition,
                        method,
                        lambda p, key: np.full(p.count, p.corpus == key, dtype=bool),
                        domains,
                    )
                    for method in present[condition]
                }
                for condition in (COND_FROZEN, COND_NATURAL)
            },
            "by_typography": {
                condition: {
                    method: _stratified(
                        arms,
                        errors,
                        condition,
                        method,
                        lambda p, key: np.full(
                            p.count, typography[p.environment] == key, dtype=bool
                        ),
                        sorted(set(typography.values())),
                    )
                    for method in present[condition]
                }
                for condition in (COND_FROZEN, COND_NATURAL)
            },
            "note": (
                "section 20 asks for a language and typography split only where the artifacts "
                "support one. `lang` in the frozen environment spec is what they support, so the "
                "split is by engine configuration and is labelled as such rather than inferred "
                "from the page."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"strata: {len(error_types)} error types, {len(engines)} engines, {len(domains)} domains; "
        "weak-type recovery "
        + ", ".join(
            f"{method} {record['weak_types_improved_count']}/{len(WEAK_ERROR_TYPES)}"
            for method, record in recovery.items()
        )
    )
    return 0


# ------------------------------------------------------------------ section 30, 31, 32: overlap

EDIT_KEY = ["environment", "document_id", "char_start", "char_end", "candidate_text"]


def _repaired_by(
    arms: dict[tuple[str, str, str], Arm],
    errors: dict[str, ErrorSites],
    method: str,
    condition: str,
    name: str,
    k: int | None = None,
) -> np.ndarray:
    arm = arms.get((name, method, condition))
    if arm is None:
        return np.zeros(errors[name].count, dtype=bool)
    return arm.repaired(k)


def _union_reading(
    arms: dict[tuple[str, str, str], Arm],
    errors: dict[str, ErrorSites],
    methods: Sequence[str],
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Opportunity if every listed arm's candidates were pooled and selected perfectly.

    An OCR error site is repairable by the union exactly when some arm offers an exact candidate
    linked to it, so the union's opportunity is the OR of the arms' repaired sets on the shared
    denominator. Deduplicating the pooled candidates changes the candidate count and never this.
    """
    per_environment = []
    for name, population in errors.items():
        mask = np.zeros(population.count, dtype=bool)
        for method in methods:
            for condition in conditions:
                mask |= _repaired_by(arms, errors, method, condition, name)
        per_environment.append(
            {
                "environment": name,
                "repairable_sites": int(mask.sum()),
                "ocr_error_sites": population.count,
                "opportunity_recall": _ratio(int(mask.sum()), population.count),
            }
        )
    repaired = sum(row["repairable_sites"] for row in per_environment)
    total = sum(row["ocr_error_sites"] for row in per_environment)
    return {
        "methods": list(methods),
        "conditions": list(conditions),
        "per_environment": per_environment,
        "mean_opportunity_recall": _mean([row["opportunity_recall"] for row in per_environment]),
        "pooled_opportunity_recall": _ratio(repaired, total),
        "repairable_sites": repaired,
        "ocr_error_sites": total,
    }


def _union_burden(
    candidates: pd.DataFrame, methods: Sequence[str], conditions: Sequence[str]
) -> dict[str, Any]:
    """The candidate cost of the union, deduplicated on the atomic edit (section 10)."""
    block = candidates[
        candidates["method"].isin(list(methods)) & candidates["condition"].isin(list(conditions))
    ]
    unique = block.drop_duplicates(EDIT_KEY)
    return {
        "candidates_before_deduplication": len(block),
        "candidates_after_deduplication": len(unique),
        "duplicates_removed": len(block) - len(unique),
        "exact_candidates": int(unique["exact"].sum()),
        "beneficial_candidates": int(unique["beneficial"].sum()),
        "harmful_candidates": int(unique["is_harmful"].sum()),
        "harmful_share": _ratio(int(unique["is_harmful"].sum()), len(unique)),
        "edits_proposed_by_more_than_one_method": int(
            (block.groupby(EDIT_KEY)["method"].nunique() > 1).sum()
        ),
    }


def run_overlap() -> int:
    """Sections 30, 31 and 32. Whether the families are substitutes or complements.

    The union is analysis-only throughout. It spends more generation than any single arm and is
    never a proposed method; what it measures is how much of the ceiling the families leave to
    each other.
    """
    started = time.monotonic()
    for path in (METHOD_OVERLAP, UNIQUE_REPAIRS, UNION_OPPORTUNITY):
        _forbid(path)
    errors, arms = load_arms()
    methods = _available_methods(arms)
    candidates = pd.read_parquet(METHOD_CANDIDATES)
    candidates = candidates[candidates["role"] == s14.ROLE_EVALUATION]
    condition = PRIMARY_CONDITION

    repaired = {
        (method, name): _repaired_by(arms, errors, method, condition, name)
        for method in methods
        for name in errors
    }
    matrix = {
        a: {
            b: int(sum(int((repaired[a, n] & repaired[b, n]).sum()) for n in errors))
            for b in methods
        }
        for a in methods
    }
    pairs = []
    for position, a in enumerate(methods):
        for b in methods[position + 1 :]:
            per_environment = []
            for name, population in errors.items():
                left, right = repaired[a, name], repaired[b, name]
                per_environment.append(
                    {
                        "environment": name,
                        "shared": int((left & right).sum()),
                        "unique_to_a": int((left & ~right).sum()),
                        "unique_to_b": int((right & ~left).sum()),
                        "union": int((left | right).sum()),
                        "ocr_error_sites": population.count,
                        "union_opportunity_recall": _ratio(
                            int((left | right).sum()), population.count
                        ),
                    }
                )
            shared = sum(row["shared"] for row in per_environment)
            union = sum(row["union"] for row in per_environment)
            pairs.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "shared_repairs": shared,
                    "unique_to_a": sum(row["unique_to_a"] for row in per_environment),
                    "unique_to_b": sum(row["unique_to_b"] for row in per_environment),
                    "union_repairs": union,
                    "ocr_error_sites": sum(row["ocr_error_sites"] for row in per_environment),
                    "jaccard": _ratio(shared, union),
                    "mean_union_opportunity_recall": _mean(
                        [row["union_opportunity_recall"] for row in per_environment]
                    ),
                    "per_environment": per_environment,
                }
            )

    unique: dict[str, Any] = {}
    for method in methods:
        others = [other for other in methods if other != method]
        by_kind: dict[str, int] = {}
        only_total = 0
        for name, population in errors.items():
            rest = np.zeros(population.count, dtype=bool)
            for other in others:
                rest |= repaired[other, name]
            only = repaired[method, name] & ~rest
            only_total += int(only.sum())
            for kind in population.kind[only]:
                by_kind[str(kind)] = by_kind.get(str(kind), 0) + 1
        unique[method] = {
            "repairs": int(sum(int(repaired[method, n].sum()) for n in errors)),
            "unique_repairs": only_total,
            "unique_repairs_by_error_type": dict(sorted(by_kind.items())),
        }

    frame = _rows_for(arms, condition)
    singles = {
        method: float(_method_summary(frame, method)["mean_opportunity_recall"])
        for method in methods
    }
    best_single, best_value = max(singles.items(), key=lambda item: (item[1], item[0]))
    union_all = _union_reading(arms, errors, methods, [condition])
    union_new = _union_reading(arms, errors, [m for m in methods if m != M0], [condition])
    union_every_condition = _union_reading(arms, errors, methods, [COND_FROZEN, COND_NATURAL])
    complementarity_gain = float(union_all["mean_opportunity_recall"]) - best_value

    _write_json_once(
        METHOD_OVERLAP,
        {
            **_oracle_envelope("method_overlap"),
            "condition": condition,
            "site_overlap_matrix": matrix,
            "matrix_note": (
                "entry [a][b] is the number of OCR error sites both a and b can repair exactly; "
                "the diagonal is each arm's own repair count."
            ),
            "pairs": pairs,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        UNIQUE_REPAIRS,
        {
            **_oracle_envelope("unique_repairs"),
            "condition": condition,
            "definition": (
                "OCR error sites one arm can repair exactly and no other available arm can."
            ),
            "by_method": unique,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        UNION_OPPORTUNITY,
        {
            **_oracle_envelope("union_opportunity"),
            "is_deployable_method": False,
            "note": (
                "section 31: the maximum opportunity available if the correction families were "
                "combined perfectly. It uses more candidate-generation resource than any single "
                "arm and is not a proposed method."
            ),
            "best_single_method": best_single,
            "best_single_mean_opportunity_recall": best_value,
            "single_method_mean_opportunity_recall": singles,
            "union_all_methods": union_all,
            "union_all_methods_burden": _union_burden(candidates, methods, [condition]),
            "union_new_methods_only": union_new,
            "union_every_gt_blind_condition": union_every_condition,
            "union_every_gt_blind_condition_burden": _union_burden(
                candidates, methods, [COND_FROZEN, COND_NATURAL]
            ),
            "complementarity_gain_over_best_single": complementarity_gain,
            "complementarity_rule": (
                f"outcome B requires the all-method union to exceed the best single arm by at "
                f"least {MEANINGFUL_DELTA} in mean opportunity recall, in the primary condition."
            ),
            "meets_complementarity_rule": complementarity_gain >= MEANINGFUL_DELTA,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"overlap: union {union_all['mean_opportunity_recall']:.4f} vs best single "
        f"{best_single} {best_value:.4f} (gain {complementarity_gain:+.4f}); unique "
        + ", ".join(f"{m} {u['unique_repairs']}" for m, u in unique.items())
    )
    return 0


# ------------------------------------------------------------------ section 13: near miss


def _histogram(values: np.ndarray, top: int = 10) -> dict[str, int]:
    if not values.size:
        return {}
    counts = np.bincount(np.minimum(values.astype(np.int64), top + 1), minlength=top + 2)
    return {**{str(i): int(counts[i]) for i in range(top + 1)}, f"{top + 1}+": int(counts[top + 1])}


def run_nearmiss() -> int:
    """Section 13. How far the non-exact candidates are from ground truth. Secondary only.

    A generative corrector can land one character short of the right answer, which is a
    different failure from proposing something unrelated. Neither is a repair: exact repair stays
    the headline and nothing here enters it. Candidate CER against ground truth is not reported,
    because the frozen label output carries the region distance but not the region ground-truth
    length, and a length that was never saved cannot honestly be reconstructed.
    """
    started = time.monotonic()
    _forbid(NEAR_MISS_ANALYSIS)
    _errors, arms = load_arms()
    methods = _available_methods(arms)
    conditions: dict[str, Any] = {}
    for condition in (COND_FROZEN, COND_NATURAL):
        by_method: dict[str, Any] = {}
        for method in methods:
            selected = [
                arm for (_n, m, c), arm in sorted(arms.items()) if m == method and c == condition
            ]
            if not selected:
                continue
            d_after = np.concatenate([arm.d_after[arm.d_before > 0] for arm in selected])
            d_before = np.concatenate([arm.d_before[arm.d_before > 0] for arm in selected])
            exact = d_after == 0
            near = (d_after == 1) & (d_after < d_before)
            per_environment = []
            for arm in selected:
                near_rows = (arm.d_after == 1) & (arm.d_after < arm.d_before)
                near_sites = arm.linked(_any_by_group(near_rows, arm.site_of, arm.n_sites))
                exact_sites = arm.repaired()
                per_environment.append(
                    {
                        "environment": arm.environment,
                        "exact_opportunity_recall": _ratio(
                            int(exact_sites.sum()), arm.errors.count
                        ),
                        "within_one_opportunity_recall": _ratio(
                            int((exact_sites | near_sites).sum()), arm.errors.count
                        ),
                        "near_miss_only_sites": int((near_sites & ~exact_sites).sum()),
                    }
                )
            size = int(d_after.size)
            by_method[method] = {
                "candidates_at_error_regions": size,
                "exact_share": _ratio(int(exact.sum()), size),
                "near_miss_share": _ratio(int(near.sum()), size),
                "improving_share": _ratio(int((d_after < d_before).sum()), size),
                "lateral_share": _ratio(int((d_after == d_before).sum()), size),
                "worsening_share": _ratio(int((d_after > d_before).sum()), size),
                "d_after_histogram": _histogram(d_after),
                "d_after_non_exact": _quantiles(d_after[~exact].tolist()),
                "residual_error_share": _quantiles((d_after / d_before).tolist()),
                "mean_exact_opportunity_recall": _mean(
                    [row["exact_opportunity_recall"] for row in per_environment]
                ),
                "mean_within_one_opportunity_recall": _mean(
                    [row["within_one_opportunity_recall"] for row in per_environment]
                ),
                "near_miss_only_sites": sum(row["near_miss_only_sites"] for row in per_environment),
                "per_environment": per_environment,
            }
        conditions[condition] = by_method

    _write_json_once(
        NEAR_MISS_ANALYSIS,
        {
            **_oracle_envelope("near_miss_analysis"),
            "secondary_only": True,
            "counts_as_repair": False,
            "definitions": {
                "candidates_at_error_regions": (
                    "labelled candidates whose own region has d_before > 0"
                ),
                "near_miss": (
                    "an improving candidate exactly one character from its region's ground truth: "
                    "d_after == 1 and d_after < d_before"
                ),
                "residual_error_share": (
                    "d_after / d_before -- 0 is an exact repair, below 1 an improvement, 1 a "
                    "lateral change and above 1 a worsening"
                ),
                "within_one_opportunity_recall": (
                    "OCR error sites carrying an exact repair or a near miss. A diagnostic of how "
                    "close a family gets; never a substitute for the primary exact metric."
                ),
            },
            "candidate_cer_reported": False,
            "candidate_cer_reason": (
                "the region ground-truth length is not part of the frozen label output, so a "
                "per-candidate CER would need a quantity this stage did not save."
            ),
            "conditions": conditions,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    frozen = conditions.get(PRIMARY_CONDITION, {})
    print(
        "nearmiss: "
        + ", ".join(
            f"{method} exact {record['exact_share']:.4f} near {record['near_miss_share']:.4f}"
            for method, record in frozen.items()
        )
    )
    return 0


# ------------------------------------------------------------------ section 35: distribution shift


def _edit_type(original: str, candidate: str) -> str:
    """The shape of an edit, named from the edit's side so it cannot be read as a SiteKind.

    The frozen taxonomy names the OCR error (`insertion` is text the engine added). These names
    describe what the proposed edit does to the OCR, which is the opposite direction, so they use
    a separate vocabulary on purpose.
    """
    if not original:
        return "insert_text"
    if not candidate:
        return "delete_text"
    if "".join(original.split()) == "".join(candidate.split()):
        return "whitespace_change"
    if original.casefold() == candidate.casefold():
        return "case_change"
    return "replace_text"


def _shares(values: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in values.value_counts(normalize=True).sort_index().items()}


def run_shift() -> int:
    """Section 35. What each family's candidates look like, stored for a later transfer stage.

    The reliability layer was fitted on M0's candidate distribution. Whether it transfers to a new
    family is a later stage's question, and that stage needs to know how far the distribution
    moved; this is the record of it, in the frozen edit-distance function rather than a new one.
    """
    started = time.monotonic()
    _forbid(CANDIDATE_DISTRIBUTION_SHIFT)
    from ocr_risk.edits.outcome import distance

    candidates = pd.read_parquet(METHOD_CANDIDATES)
    candidates = candidates[candidates["role"] == s14.ROLE_EVALUATION]
    out: dict[str, dict[str, Any]] = {}
    for (method, condition), block in candidates.groupby(["method", "condition"], sort=True):
        originals = block["original_ocr"].astype(str).tolist()
        proposals = block["candidate_text"].astype(str).tolist()
        types = pd.Series(
            [_edit_type(o, c) for o, c in zip(originals, proposals, strict=True)], dtype=str
        )
        edit_distance = [float(distance(o, c)) for o, c in zip(originals, proposals, strict=True)]
        sources = block["candidate_source_set"].astype(str)
        rows = len(block)
        out.setdefault(str(condition), {})[str(method)] = {
            "candidates": rows,
            "edit_type_share": _shares(types),
            "edit_kind_share": _shares(block["edit_kind"].astype(str)),
            "anchor_kind_share": _shares(block["anchor_kind"].astype(str)),
            "original_chars": _quantiles(block["original_chars"].astype(float).tolist()),
            "candidate_chars": _quantiles(block["candidate_chars"].astype(float).tolist()),
            "length_change": _quantiles(
                (block["candidate_chars"] - block["original_chars"]).astype(float).tolist()
            ),
            "edit_distance": _quantiles(edit_distance),
            "candidates_per_site": _quantiles(
                block.groupby("site_id").size().astype(float).tolist()
            ),
            "exact_prevalence": _ratio(int(block["exact"].sum()), rows),
            "beneficial_prevalence": _ratio(int(block["beneficial"].sum()), rows),
            "harmful_prevalence": _ratio(int(block["is_harmful"].sum()), rows),
            "proposed_by_more_than_one_method_share": _ratio(
                int(sources.str.contains("+", regex=False).sum()), rows
            ),
            "candidate_source_set_counts": {
                str(k): int(v) for k, v in sources.value_counts().sort_index().items()
            },
        }
    _write_json_once(
        CANDIDATE_DISTRIBUTION_SHIFT,
        {
            **_oracle_envelope("candidate_distribution_shift"),
            "edit_distance_function": "ocr_risk.edits.outcome.distance",
            "edit_type_vocabulary": {
                "insert_text": "the edit adds text where the OCR had none (a gap edit)",
                "delete_text": "the edit removes the OCR region entirely",
                "whitespace_change": "the edit changes only whitespace -- a segmentation edit",
                "case_change": "the edit changes only letter case",
                "replace_text": "any other change",
            },
            "conditions": out,
            "note": (
                "stored for a future reliability transfer stage (section 57, outcome A). No "
                "reliability model is applied or refitted here."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "shift: "
        + ", ".join(
            f"{m} {record['harmful_prevalence']:.3f} harmful"
            for m, record in out.get(PRIMARY_CONDITION, {}).items()
        )
    )
    return 0


# ------------------------------------------------------------------ section 29: runtime and cost

COST_LOG_LINE = re.compile(
    r"^\s+(?P<environment>\S+)/(?P<condition>frozen_localization|natural|oracle_localization): "
    r"(?P<requests>\d+) req in (?P<seconds>\d+)s \((?P<rate>[\d.]+)/s\)$"
)


def _recovered_costs() -> dict[tuple[str, str, str], dict[str, Any]]:
    """Costs of files whose run was killed before it could persist them, recovered by code.

    The first ByT5 run was interrupted before per-file cost records existed. Its completion lines
    were emitted by `_run_arm` the moment each file finished and are preserved verbatim in the
    stage cache. They are parsed here and never retyped, and `run_cost` accepts one only if its
    request count equals the distinct request count of the raw file it claims to describe.
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for log in sorted(CACHE.glob("generation_log.*.log")):
        method = log.name.split(".")[1]
        digest = file_sha256(log)
        for line in log.read_text().splitlines():
            match = COST_LOG_LINE.match(line)
            if match is None:
                continue
            out[match["environment"], method, match["condition"]] = {
                "requests_in_log": int(match["requests"]),
                "wall_clock_seconds": float(match["seconds"]),
                "wall_clock_resolution_seconds": 1.0,
                "source": f"recovered from {_relative(log)} (sha256 {digest})",
            }
    return out


def run_cost() -> int:
    """Section 29. What each arm's candidates cost to produce. No monetary figure is invented."""
    started = time.monotonic()
    _forbid(RUNTIME_COST)
    versions = cc_read_json(MODEL_VERSIONS)
    candidates = pd.read_parquet(METHOD_CANDIDATES)
    evaluation = candidates[candidates["role"] == s14.ROLE_EVALUATION]
    oracle = pd.read_parquet(ORACLE_CANDIDATES) if ORACLE_CANDIDATES.is_file() else pd.DataFrame()
    recovered = _recovered_costs()

    files: list[dict[str, Any]] = []
    for directory, analysis_only in ((RAW_CACHE, False), (RAW_ORACLE_CACHE, True)):
        for raw_path in sorted(directory.glob("*.parquet")):
            slug, method, condition = raw_path.name[: -len(".parquet")].split(".")
            name = slug.replace("__", "/")
            raw = pd.read_parquet(raw_path, columns=["request_id", "output"])
            requests = int(raw["request_id"].nunique())
            record: dict[str, Any] = {
                "environment": name,
                "method": method,
                "condition": condition,
                "analysis_only": analysis_only,
                "raw_output": _relative(raw_path),
                "requests": requests,
                "returned_sequences": len(raw),
                "output_characters": int(raw["output"].astype(str).str.len().sum()),
            }
            sidecar = _cost_path(raw_path)
            found = recovered.get((name, method, condition))
            if sidecar.is_file():
                measured = cc_read_json(sidecar)
                if requests > int(measured["requests"]):
                    raise PhaseError(f"{_relative(sidecar)} disagrees with its raw file")
                record.update(
                    requests=int(measured["requests"]),
                    requests_with_output=requests,
                )
                record.update(
                    wall_clock_seconds=float(measured["wall_clock_seconds"]),
                    prompt_characters=int(measured["prompt_characters"]),
                    source=str(measured["source"]),
                )
            elif found is not None and found["requests_in_log"] == requests:
                record.update(
                    wall_clock_seconds=found["wall_clock_seconds"],
                    wall_clock_resolution_seconds=found["wall_clock_resolution_seconds"],
                    source=found["source"],
                )
            else:
                record.update(wall_clock_seconds=None, source="not measured")
            files.append(record)

    frame = pd.DataFrame(files)
    summary: list[dict[str, Any]] = []
    if not frame.empty:
        for (method, condition), block in frame.groupby(["method", "condition"], sort=True):
            timed = block[block["wall_clock_seconds"].notna()]
            seconds = float(timed["wall_clock_seconds"].sum())
            source = oracle if condition == COND_ORACLE_LOC else evaluation
            produced = (
                int(((source["method"] == method) & (source["condition"] == condition)).sum())
                if not source.empty
                else 0
            )
            summary.append(
                {
                    "method": str(method),
                    "condition": str(condition),
                    "files": len(block),
                    "files_timed": len(timed),
                    "requests": int(block["requests"].sum()),
                    "returned_sequences": int(block["returned_sequences"].sum()),
                    "normalized_candidates": produced,
                    "wall_clock_seconds": seconds,
                    "wall_clock_hours": seconds / 3600.0,
                    "requests_per_second": _ratio(int(timed["requests"].sum()), seconds),
                    "normalized_candidates_per_second": (
                        _ratio(produced, seconds) if len(timed) == len(block) else float("nan")
                    ),
                }
            )

    _write_json_once(
        RUNTIME_COST,
        {
            **_envelope("runtime_cost"),
            "hardware": versions["hardware"],
            "packages": versions["packages"],
            "files": files,
            "by_method_condition": summary,
            "external_api_calls": 0,
            "monetary_cost": None,
            "monetary_cost_note": (
                "not reported. All inference is local, no charged cost was incurred or recorded, "
                "and section 29 forbids inventing one."
            ),
            "tokens_note": (
                "input and output are recorded in characters, not tokens. ByT5 tokenizes UTF-8 "
                "bytes and the LLM uses a subword vocabulary, so a shared token count would mean "
                "different things for the two arms; characters mean the same thing for both."
            ),
            "accelerator_note": (
                "neural inference ran on the device recorded in `hardware`. Wall clock on that "
                "device is reported as measured and is not converted into GPU-hours of another "
                "accelerator."
            ),
            "clock_note": (
                "wall clock comes from time.monotonic, which on this platform is "
                "mach_absolute_time() and does not advance while the machine sleeps, so a "
                "suspended session is not counted as compute."
            ),
            "throughput_note": (
                "measured on one machine while a model download ran in the background; rates vary "
                "by environment with region length and machine load, and are not a controlled "
                "benchmark."
            ),
            "m0_note": (
                "M0's candidates are produced inside the frozen pipeline's --prepare pass together "
                "with span canonicalization and OCR-only site enumeration. Its generation time is "
                "not isolated there, so it is reported as not measured rather than apportioned."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "cost: "
        + ", ".join(
            f"{row['method']}/{row['condition']} {row['wall_clock_hours']:.2f}h "
            f"({row['files_timed']}/{row['files']} timed)"
            for row in summary
        )
    )
    return 0


# ------------------------------------------------------------------ section 48: controls


def _duplicated(arm: Arm) -> Arm:
    """C5: every candidate twice. Nothing about the candidate set changes except its size."""
    from dataclasses import replace

    def twice(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, values])

    return replace(
        arm,
        candidate_id=twice(arm.candidate_id),
        document_id=twice(arm.document_id),
        site_of=twice(arm.site_of),
        exact=twice(arm.exact),
        harmful=twice(arm.harmful),
        beneficial=twice(arm.beneficial),
        d_before=twice(arm.d_before),
        d_after=twice(arm.d_after),
        rank=twice(arm.rank),
        anchor_kind=twice(arm.anchor_kind),
        edit_kind=twice(arm.edit_kind),
        original_chars=twice(arm.original_chars),
        candidate_chars=twice(arm.candidate_chars),
    )


def _summary_mean(frame: pd.DataFrame, method: str) -> float:
    if frame.empty or not (frame["method"] == method).any():
        return 0.0
    return float(_method_summary(frame, method)["mean_opportunity_recall"])


def run_controls() -> int:
    """Section 48. The seven controls, each against the expectation frozen in the method registry.

    A control that cannot fail proves nothing, so each one is stated as a comparison that the
    pipeline could get wrong: a leak, a mis-link, a label that reads the wrong region or a budget
    that reads a label would each move one of these numbers the wrong way.
    """
    started = time.monotonic()
    _forbid(CONTROL_RESULTS)
    _require(UPSTREAM_REPRODUCTION, "reproduce")
    _require(ORACLE_LOCALIZATION, "oracles")
    _errors, arms = load_arms()
    _control_errors, control_arms = load_arms(AUXILIARY_CANDIDATES, AUXILIARY_LINKS)
    normalization = cc_read_json(NORMALIZATION_RESULTS)
    reproduction = cc_read_json(UPSTREAM_REPRODUCTION)
    oracle = cc_read_json(ORACLE_LOCALIZATION)
    methods = _available_methods(arms)
    frozen = _rows_for(arms, COND_FROZEN)
    controls = _rows_for(control_arms, COND_FROZEN)
    means = {method: _summary_mean(frozen, method) for method in methods}
    results: dict[str, Any] = {}

    results["c0_frozen_m0_reproduction"] = {
        "observed": {
            "checks": reproduction["checks_total"],
            "differences": reproduction["differences"],
            "maximum_absolute_difference": reproduction["maximum_absolute_difference"],
        },
        "discriminates": reproduction["differences"] == 0,
    }

    identity = [row for row in normalization["controls"] if row["method"] == C1_IDENTITY]
    surviving = sum(int(row["proposals"]) for row in identity)
    results["c1_identity_corrector"] = {
        "observed": {
            "requests_answered": sum(int(row["raw_sequences"]) for row in identity),
            "declined_unchanged": sum(int(row["declined_unchanged"]) for row in identity),
            "proposals_surviving_normalization": surviving,
            "new_exact_repairs": 0
            if not surviving
            else int(
                sum(
                    int(arm.repaired().sum())
                    for (_n, m, _c), arm in control_arms.items()
                    if m == C1_IDENTITY
                )
            ),
        },
        "discriminates": surviving == 0,
    }

    random_block = controls[controls["method"] == C2_RANDOM] if not controls.empty else controls
    random_mean = _summary_mean(controls, C2_RANDOM)
    random_harmful = int(random_block["harmful_candidates"].sum()) if len(random_block) else 0
    random_beneficial = int(random_block["beneficial_candidates"].sum()) if len(random_block) else 0
    results["c2_random_text_perturbation"] = {
        "observed": {
            "mean_opportunity_recall": random_mean,
            "baseline_mean_opportunity_recall": means[M0],
            "harmful_candidates": random_harmful,
            "beneficial_candidates": random_beneficial,
            "candidates": int(random_block["candidates"].sum()) if len(random_block) else 0,
        },
        "discriminates": random_mean < means[M0] and random_harmful > random_beneficial,
    }

    ground_truth = oracle["c3_ground_truth_oracle_corrector"]
    results["c3_ground_truth_oracle_corrector"] = {
        "analysis_only": True,
        "observed": ground_truth,
        "discriminates": int(ground_truth["repaired_error_sites"])
        == int(ground_truth["expressible_labelable_error_sites"]),
    }

    shuffled: dict[str, Any] = {}
    for source in methods:
        value = _summary_mean(controls, f"{C4_PREFIX}{source}")
        shuffled[source] = {
            "unshuffled_mean_opportunity_recall": means[source],
            "shuffled_mean_opportunity_recall": value,
            "informative": means[source] > 0.0,
            "collapses": value < means[source],
            "never_rises": value <= means[source],
        }
    informative = [record for record in shuffled.values() if record["informative"]]
    results["c4_shuffled_method_outputs"] = {
        "observed": shuffled,
        "note": (
            "a source arm with no opportunity has nothing to lose, so its shuffle is recorded as "
            "uninformative rather than as a pass."
        ),
        "discriminates": bool(informative)
        and all(record["collapses"] for record in informative)
        and all(record["never_rises"] for record in shuffled.values()),
    }

    duplication = []
    for (name, method, condition), arm in sorted(arms.items()):
        copy = _duplicated(arm)
        duplication.append(
            {
                "environment": name,
                "method": method,
                "condition": condition,
                "opportunity_unchanged": bool(np.array_equal(copy.repaired(), arm.repaired())),
                "candidates_doubled": copy.n_rows == 2 * arm.n_rows,
            }
        )
    results["c5_candidate_duplication"] = {
        "observed": {
            "arms": len(duplication),
            "arms_with_opportunity_unchanged": sum(r["opportunity_unchanged"] for r in duplication),
            "arms_with_candidates_doubled": sum(r["candidates_doubled"] for r in duplication),
        },
        "discriminates": all(
            r["opportunity_unchanged"] and r["candidates_doubled"] for r in duplication
        ),
    }

    truncation = []
    for (name, method, condition), arm in sorted(arms.items()):
        curve = [arm.opportunity_recall(k) for k in sorted(K_GRID)] + [arm.opportunity_recall()]
        finite = [0.0 if np.isnan(value) else value for value in curve]
        truncation.append(
            {
                "environment": name,
                "method": method,
                "condition": condition,
                "curve": dict(
                    zip([str(k) for k in sorted(K_GRID)] + ["natural"], finite, strict=True)
                ),
                "monotone": all(b >= a for a, b in pairwise(finite)),
            }
        )
    results["c6_candidate_truncation"] = {
        "observed": {
            "arms": len(truncation),
            "arms_monotone": sum(row["monotone"] for row in truncation),
            "violations": [row for row in truncation if not row["monotone"]],
        },
        "discriminates": all(row["monotone"] for row in truncation),
    }

    for control, record in results.items():
        record["expectation"] = CONTROL_SPECS[control]["expectation"]
        record["rule"] = CONTROL_SPECS[control]["rule"]
    passed = sum(bool(record["discriminates"]) for record in results.values())
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_oracle_envelope("control_results"),
            "controls": results,
            "discriminating": passed,
            "total": len(CONTROLS),
            "expectations_frozen_in": _relative(METHOD_REGISTRY),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"controls: {passed} of {len(CONTROLS)} discriminate")
    return 0


# ------------------------------------------------------------------ section 47: statistics


def _per_document(population: ErrorSites, mask: np.ndarray) -> np.ndarray:
    """OCR error sites selected by ``mask``, counted per evaluation document."""
    position = {document: index for index, document in enumerate(population.documents.tolist())}
    selected = population.document[mask]
    return np.bincount(
        np.fromiter((position[d] for d in selected), dtype=np.int64, count=selected.size),
        minlength=population.documents.size,
    )


def _interval_bounded(draws: np.ndarray, effect: float) -> dict[str, Any]:
    """A percentile interval with the repository's bootstrap p-value.

    The interval is SGV-CG1's percentile interval. The p-value is not SGV-CG1's: that formula
    doubles the smaller tail mass with no correction, so it can report exactly zero -- a certainty
    a bootstrap of B resamples cannot deliver -- and exceeds one when every draw sits on zero.
    `docs/statistics.md` fixes the method once, in `ocr_risk.stats.bootstrap`: +1 in numerator and
    denominator, clipped at one. That function is used here unchanged.
    """
    from ocr_risk.stats.bootstrap import _bootstrap_p_value

    record: dict[str, Any] = dict(cg1._interval(draws, effect))
    record["p_value"] = _bootstrap_p_value(draws[np.isfinite(draws)])
    record["p_value_method"] = "ocr_risk.stats.bootstrap._bootstrap_p_value"
    return record


def _comparison(
    errors: dict[str, ErrorSites],
    arms: dict[tuple[str, str, str], Arm],
    left: tuple[str, str],
    right: tuple[str, str],
    *,
    selector: Callable[[ErrorSites], np.ndarray] | None = None,
    k: int | None = None,
    pooled: bool = False,
    environments: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A document-clustered paired bootstrap of one opportunity difference.

    Documents are resampled with replacement inside each environment, and the same draw serves
    both arms, so the interval reflects the pairing rather than two independent samples of the
    same pages. The statistic is either the mean over environments of a within-environment site
    rate -- the headline's own estimator -- or a rate pooled over environments, which is what a
    per-error-type comparison needs when some environments hold few sites of a type.
    """
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    rates: list[float] = []
    observed_numerator = 0
    observed_denominator = 0
    for name in errors:
        if environments is not None and name not in environments:
            continue
        population = errors[name]
        mask = np.ones(population.count, dtype=bool) if selector is None else selector(population)
        totals = _per_document(population, mask)
        if not totals.sum():
            continue
        ahead = arms[name, left[0], left[1]].repaired(k) & mask
        behind = arms[name, right[0], right[1]].repaired(k) & mask
        difference = _per_document(population, ahead) - _per_document(population, behind)
        picks = generator.integers(0, totals.size, size=(BOOTSTRAP_RESAMPLES, totals.size))
        numerators.append(difference[picks].sum(axis=1).astype(float))
        denominators.append(totals[picks].sum(axis=1).astype(float))
        rates.append(float(difference.sum() / totals.sum()))
        observed_numerator += int(difference.sum())
        observed_denominator += int(totals.sum())
    if not rates:
        return {"environments_compared": 0}
    numerator = np.vstack(numerators)
    denominator = np.vstack(denominators)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (
            numerator.sum(axis=0) / denominator.sum(axis=0)
            if pooled
            else (numerator / denominator).mean(axis=0)
        )
    effect = observed_numerator / observed_denominator if pooled else float(np.mean(rates))
    return {
        **_interval_bounded(draws, effect),
        "estimator": "pooled site rate" if pooled else "mean over environments of a site rate",
        "environments_compared": len(rates),
        "environments_improved": sum(rate > 0 for rate in rates),
        "environments_worsened": sum(rate < 0 for rate in rates),
    }


def run_stats() -> int:
    """Section 47. The pre-registered comparisons, document-clustered and Holm-corrected.

    The primary family is exactly the new arms against M0 in the primary condition, as the design
    record declares. Every other reading -- the natural condition, the declared language scope,
    the matched budget, the error types -- is its own family, corrected within itself and labelled
    secondary, so no secondary result can borrow the primary family's multiplicity budget.
    """
    started = time.monotonic()
    _forbid(STATISTICAL_TESTS)
    errors, arms = load_arms()
    methods = _available_methods(arms)
    new = [method for method in methods if method != M0]
    baseline = (M0, COND_FROZEN)
    error_types = sorted({kind for population in errors.values() for kind in population.kind})

    tests: list[dict[str, Any]] = []
    for method in new:
        tests.append(
            {
                "comparison": f"{method} - {M0} ({COND_FROZEN})",
                "family": "primary",
                "method": method,
                **_comparison(errors, arms, (method, COND_FROZEN), baseline),
            }
        )
        if any((n, method, COND_NATURAL) in arms for n in errors):
            tests.append(
                {
                    "comparison": f"{method} ({COND_NATURAL}) - {M0} ({COND_FROZEN})",
                    "family": "secondary_natural_condition",
                    "method": method,
                    **_comparison(errors, arms, (method, COND_NATURAL), baseline),
                }
            )
        scope = [n for n, population in errors.items() if method_applies(method, population.corpus)]
        if len(scope) != len(errors):
            tests.append(
                {
                    "comparison": f"{method} - {M0} (declared language scope)",
                    "family": "secondary_language_scope",
                    "method": method,
                    "environments": scope,
                    **_comparison(
                        errors, arms, (method, COND_FROZEN), baseline, environments=scope
                    ),
                }
            )
        tests.append(
            {
                "comparison": f"{method} - {M0} (K={PRIMARY_K})",
                "family": "secondary_matched_budget",
                "method": method,
                **_comparison(errors, arms, (method, COND_FROZEN), baseline, k=PRIMARY_K),
            }
        )
        for kind in error_types:
            tests.append(
                {
                    "comparison": f"{method} - {M0} [{kind}]",
                    "family": "secondary_error_type",
                    "method": method,
                    "error_type": kind,
                    **_comparison(
                        errors,
                        arms,
                        (method, COND_FROZEN),
                        baseline,
                        selector=lambda population, kind=kind: population.kind == kind,
                        pooled=True,
                    ),
                }
            )

    families: dict[str, list[str]] = {}
    for family in sorted({test["family"] for test in tests}):
        members = [test for test in tests if test["family"] == family and "p_value" in test]
        families[family] = [test["comparison"] for test in members]
        if not members:
            continue
        adjusted = holm({test["comparison"]: {"p_value": test["p_value"]} for test in members})
        for test in members:
            test["adjusted_p_value"] = float(adjusted[test["comparison"]]["holm_adjusted_p"])
            test["significant_after_holm"] = bool(adjusted[test["comparison"]]["survives_holm"])

    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_oracle_envelope("statistical_tests"),
            "inference_units": ["environment", "document"],
            "candidate_rows_are_not_inferential_units": True,
            "bootstrap": {
                "kind": "document-clustered, paired, resampled within environment",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile",
                "p_value": "two-sided, from the resample distribution, +1 corrected, at most 1",
            },
            "multiplicity": {
                "method": "Holm",
                "alpha": ALPHA,
                "primary_family": families.get("primary", []),
                "families": families,
                "note": (
                    "the primary family is the new arms against M0 in the primary condition, as "
                    "pre-registered. Each secondary family is corrected within itself."
                ),
            },
            "tests": tests,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    primary = [test for test in tests if test["family"] == "primary" and "p_value" in test]
    print(
        f"stats: {sum(t['significant_after_holm'] for t in primary)} of {len(primary)} primary "
        f"comparisons survive Holm; {len(tests)} tests in {len(families)} families"
    )
    return 0


# -------------------------------------------------------------- section 45, 46: oracle localization

C3_ORACLE = "c3_ground_truth_oracle_corrector"
ORACLE_REQUEST_EXTRA = ("expressible_by_m0", "dataset_id")
ORACLE_TRUTH_COLUMNS = ("request_id", "site_id", "align_site_id", "site_kind", "region_gt")
M0_ANCHOR_KINDS = ("token", "token_pair", "gap")


def _oracle_requests(
    spec: dict[str, Any],
    spans: dict[tuple[str, str], list[Any]],
    streams: dict[tuple[str, str], str],
    roles: dict[str, str],
    indexes: dict[tuple[str, str], Any],
    objects: list[tuple[tuple[str, str], Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Section 45: one request per evaluable OCR error site, located by ground truth.

    The location comes from the alignment and the text never does. A request carries exactly the
    columns a frozen-localization request carries -- the OCR region and its OCR context -- and the
    region ground truth goes to a separate table that only the ground-truth oracle control reads,
    so no model input can reach it through a shared frame.
    """
    name = spec["environment"]
    by_pair = {pair: {span.span_id: span for span in group} for pair, group in spans.items()}
    requests: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    census = {
        "ocr_error_sites": 0,
        "expressible": 0,
        "expressible_by_m0": 0,
        "not_expressible_no_alignment_index": 0,
        "not_expressible_span_missing": 0,
        "not_expressible_gap_without_two_flanks": 0,
    }
    for pair, site in objects:
        if roles.get(pair[0]) != s14.ROLE_EVALUATION or not site.evaluable or site.d_before <= 0:
            continue
        census["ocr_error_sites"] += 1
        index = indexes.get(pair)
        if index is None:
            census["not_expressible_no_alignment_index"] += 1
            continue
        lookup = by_pair.get(pair, {})
        stream = streams.get(pair, "")
        span_ids = [str(span_id) for span_id in site.ocr_span_ids]
        if any(span_id not in lookup for span_id in span_ids):
            census["not_expressible_span_missing"] += 1
            continue
        if span_ids:
            ordered = sorted(
                set(span_ids), key=lambda s: (lookup[s].char_start, lookup[s].char_end, s)
            )
            start = min(lookup[s].char_start for s in ordered)
            end = max(lookup[s].char_end for s in ordered)
            kind = _anchor_kind_for(len(ordered))
            anchors = ordered
            region_gt = str(index.region_truth(ordered)[1])
            dataset_id = str(lookup[ordered[0]].dataset_id)
        else:
            left = index.flanks_for_empty_component(str(site.alignment_ids[0]))[0]
            right = index.flanks_for_empty_component(str(site.alignment_ids[-1]))[1]
            if (
                left is None
                or right is None
                or left not in lookup
                or right not in lookup
                or lookup[right].char_start < lookup[left].char_end
            ):
                census["not_expressible_gap_without_two_flanks"] += 1
                continue
            start, end = lookup[left].char_end, lookup[right].char_start
            kind = "gap"
            anchors = [left, right]
            region_gt = str(index.gap_truth(left, right))
            dataset_id = str(lookup[left].dataset_id)
        census["expressible"] += 1
        census["expressible_by_m0"] += int(kind in M0_ANCHOR_KINDS)
        align_site_id = f"{name}|{site.site_id}"
        site_id = _digest(
            "xc1-oracle-site", {"schema": "sgv_xc1-oracle-site-v1", "align_site_id": align_site_id}
        )
        request_id = f"{name}|{COND_ORACLE_LOC}|{site_id}"
        requests.append(
            {
                "environment": name,
                "condition": COND_ORACLE_LOC,
                "request_id": request_id,
                "document_id": pair[0],
                "engine_id": pair[1],
                "site_id": f"{name}|{site_id}",
                "anchor_kind": kind,
                "anchor_ref": "\0".join((kind, *anchors)),
                "char_start": int(start),
                "char_end": int(end),
                "region": stream[start:end],
                "context_before": stream[max(0, start - CONTEXT_CHARS) : start],
                "context_after": stream[end : end + CONTEXT_CHARS],
                "window_start": int(start),
                "window_text": stream[start:end],
                "expressible_by_m0": kind in M0_ANCHOR_KINDS,
                "dataset_id": dataset_id,
            }
        )
        truth.append(
            {
                "request_id": request_id,
                "site_id": f"{name}|{site_id}",
                "align_site_id": align_site_id,
                "site_kind": site.site_kind.value,
                "region_gt": region_gt,
            }
        )
    return (
        pd.DataFrame(requests, columns=[*REQUEST_COLUMNS, *ORACLE_REQUEST_EXTRA]),
        pd.DataFrame(truth, columns=list(ORACLE_TRUTH_COLUMNS)),
        census,
    )


def _m0_oracle_raw(
    spec: dict[str, Any],
    spans: dict[tuple[str, str], list[Any]],
    streams: dict[tuple[str, str], str],
    requests: pd.DataFrame,
    pipeline: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """M0's frozen ladder, run on the oracle-located regions its anchor vocabulary can express.

    The frozen generators take a `DiscoveredSite`. Three of its fields -- the structural guess, the
    suspicion score and the provenance -- are required by the constructor and read by neither g3
    nor g7, which the leakage suite asserts against their source. They are fixed placeholders, so
    no ground-truth error type can reach the generator through them; only the location does.
    """
    from ocr_risk.experiments.cgv3_confirmatory import generation_pass

    name = spec["environment"]
    expressible = requests[requests["expressible_by_m0"].astype(bool)].reset_index(drop=True)
    began = time.monotonic()
    raw_rows: list[dict[str, Any]] = []
    if not expressible.empty:
        sites = pd.DataFrame(
            {
                "site_id": expressible["site_id"].str.split("|", n=1).str[1].to_numpy(),
                "document_id": expressible["document_id"].to_numpy(),
                "dataset_id": expressible["dataset_id"].to_numpy(),
                "engine_id": expressible["engine_id"].to_numpy(),
                "anchor_kind": expressible["anchor_kind"].to_numpy(),
                "anchor_ref": expressible["anchor_ref"].to_numpy(),
                "char_start": expressible["char_start"].to_numpy(),
                "char_end": expressible["char_end"].to_numpy(),
                "site_type": "oracle_localized",
                "suspicion_score": 0.0,
                "provenance_reason": "low_confidence_token",
            }
        )
        ladder = generation_pass(spans, streams, sites, pipeline.fitted, [spec["base_engine"]])
        if not ladder.empty:
            union = ladder[ladder["generator_id"] == s14.PRIMARY_GENERATOR]
            raw_rows = [
                {
                    "environment": name,
                    "method": M0,
                    "condition": COND_ORACLE_LOC,
                    "request_id": f"{name}|{COND_ORACLE_LOC}|{row.site_id}",
                    "rank": int(row.generator_rank),
                    "output": str(row.candidate_text),
                }
                for row in union.itertuples(index=False)
            ]
    elapsed = time.monotonic() - began
    cost = {
        "requests": len(expressible),
        "returned_sequences": len(raw_rows),
        "wall_clock_seconds": elapsed,
        "requests_per_second": _ratio(len(expressible), elapsed),
        "prompt_characters": int(expressible["region"].astype(str).str.len().sum()),
        "num_beams": None,
        "num_return_sequences": UNION_CAP,
        "max_new_tokens": None,
        "batch_size": None,
    }
    return _raw_frame(raw_rows), cost


def run_oracleprep() -> int:
    """Section 45's inputs, built from ground truth and kept apart from every GT-blind input.

    Requests go to their own cache files, their raw outputs to their own directory, and the region
    ground truth to a third file only the ground-truth control reads. M0's frozen ladder is run
    here because it needs the frozen pipeline rather than a model.
    """
    started = time.monotonic()
    _require(METHOD_REGISTRY, "registry")
    RAW_ORACLE_CACHE.mkdir(parents=True, exist_ok=True)
    pipeline: Any = None
    totals: dict[str, int] = {}
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        slug = _slug(name)
        request_path = CACHE / f"{slug}.oracle_requests.parquet"
        truth_path = CACHE / f"{slug}.oracle_truth.parquet"
        census_path = CACHE / f"{slug}.oracle_census.json"
        m0_path = RAW_ORACLE_CACHE / f"{slug}.{M0}.{COND_ORACLE_LOC}.parquet"
        if not all(path.is_file() for path in (request_path, truth_path, census_path, m0_path)):
            began = time.monotonic()
            bundles = s14._document_bundles(spec["corpus"])
            roles = s14.partition_of(spec["corpus"], sorted(bundles))
            spans, _record = s14._canonical_spans(spec, bundles)
            from ocr_risk.canonical import rebuild_stream

            streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
            _alignment, indexes, _by_span, _by_alignment, objects = _alignment_population(
                spec, spans, bundles, roles
            )
            requests, truth, census = _oracle_requests(
                spec, spans, streams, roles, indexes, objects
            )
            requests.to_parquet(request_path, index=False)
            truth.to_parquet(truth_path, index=False)
            if not census_path.is_file():
                _write_json_once(
                    census_path,
                    {
                        **_oracle_envelope("oracle_census", oracle_localization=True),
                        "environment": name,
                        **census,
                    },
                )
            if not m0_path.is_file():
                if pipeline is None:
                    pipeline = s14.build_pipeline()
                raw, cost = _m0_oracle_raw(spec, spans, streams, requests, pipeline)
                raw.to_parquet(m0_path, index=False)
                _write_cost_sidecar(m0_path, name, M0, COND_ORACLE_LOC, cost)
            print(
                f"  {name}: {census['ocr_error_sites']} error sites, {census['expressible']} "
                f"expressible ({census['expressible_by_m0']} by M0) "
                f"({time.monotonic() - began:.0f}s)",
                flush=True,
            )
        for key, value in cc_read_json(census_path).items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    print(
        f"oracleprep: {totals.get('ocr_error_sites', 0)} OCR error sites, "
        f"{totals.get('expressible', 0)} expressible as a region, "
        f"{totals.get('expressible_by_m0', 0)} in M0's anchor vocabulary "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


def run_oracle_generate() -> int:
    """Section 45 inference: the neural arms on the oracle-located regions, cached apart.

    Each arm runs inside its declared language scope only. There is no out-of-scope diagnostic in
    this condition: the diagnostic exists to price a language mismatch once, not in every view.
    """
    started = time.monotonic()
    _require(METHOD_REGISTRY, "registry")
    RAW_ORACLE_CACHE.mkdir(parents=True, exist_ok=True)
    registry = cc_read_json(METHOD_REGISTRY)
    versions = cc_read_json(MODEL_VERSIONS)
    arms = [m for m in (M1, M2) if registry["methods"][m]["available"]]
    only = os.environ.get("OCR_RISK_XC1_ONLY", "")
    if only:
        arms = [m for m in arms if m in only.split(",")]
    if not arms:
        print("oraclegenerate: no neural arm available")
        return 0
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = _resolve_device()
    for method in arms:
        runner: Any = None
        for spec in s14.ENVIRONMENTS:
            name = spec["environment"]
            slug = _slug(name)
            if not method_applies(method, spec["corpus"]):
                continue
            raw_path = RAW_ORACLE_CACHE / f"{slug}.{method}.{COND_ORACLE_LOC}.parquet"
            if raw_path.is_file():
                continue
            if runner is None:
                snapshot = Path(versions["models"][method]["snapshot"])
                runner = (
                    _Byt5Runner(snapshot, device) if method == M1 else _LlmRunner(snapshot, device)
                )
            requests = pd.read_parquet(CACHE / f"{slug}.oracle_requests.parquet")
            raw, cost = _run_arm(runner, requests, method, COND_ORACLE_LOC, print)
            raw.to_parquet(raw_path, index=False)
            _write_cost_sidecar(raw_path, name, method, COND_ORACLE_LOC, cost)
            print(
                f"    {name}/{COND_ORACLE_LOC}: {cost['requests']} req in "
                f"{cost['wall_clock_seconds']:.0f}s ({cost['requests_per_second']:.1f}/s)",
                flush=True,
            )
        del runner
    print(f"oraclegenerate: done in {time.monotonic() - started:.0f}s")
    return 0


def _perfect_selection(arm: Arm) -> dict[str, int]:
    """The oracle selector: one exact candidate per proposal site that has one, nothing else."""
    chosen = _any_by_group(arm.exact, arm.site_of, arm.n_sites) if arm.n_rows else np.zeros(0, bool)
    return {
        "accepted": int(chosen.sum()),
        "harmful_accepted": 0,
        "exact_and_harmful_candidates": int((arm.exact & arm.harmful).sum()),
    }


def run_oracles() -> int:
    """Section 45's labelling pass: every arm's oracle-localized proposals, labelled and linked.

    Everything here is analysis-only. Oracle localization supplies the true error region and never
    its text. Labelling is the same single `_labels` call per environment the primary arms use, and
    no primary phase reads these tables -- the leakage suite asserts that against the source.
    """
    started = time.monotonic()
    for path in (ORACLE_CANDIDATES, ORACLE_LINKS, ORACLE_NORMALIZATION):
        _forbid(path)
    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.schemas.enums import AnchorKind

    registry = cc_read_json(METHOD_REGISTRY)
    neural = [m for m in (M1, M2) if registry["methods"][m]["available"]]
    label_columns = [
        "candidate_id",
        "site_id",
        "document_id",
        "engine_id",
        "original_ocr",
        "candidate_text",
        "char_start",
        "char_end",
    ]
    candidate_frames: list[pd.DataFrame] = []
    link_frames: list[pd.DataFrame] = []
    shards: list[dict[str, Any]] = []
    expressibility: list[dict[str, Any]] = []
    faithful = {"oracle_sites": 0, "linked_to_own_source_site": 0, "linked_to_more_than_one": 0}
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        slug = _slug(name)
        began = time.monotonic()
        requests = pd.read_parquet(CACHE / f"{slug}.oracle_requests.parquet")
        truth = pd.read_parquet(CACHE / f"{slug}.oracle_truth.parquet")
        census = cc_read_json(CACHE / f"{slug}.oracle_census.json")
        span_table = pd.read_parquet(CACHE / f"{slug}.spans.parquet")
        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        spans, _record = s14._canonical_spans(spec, bundles)
        streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
        _alignment, indexes, by_span, by_alignment, _objects = _alignment_population(
            spec, spans, bundles, roles
        )
        spans_by_pair: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
        for row in span_table.itertuples(index=False):
            spans_by_pair.setdefault((str(row.document_id), str(row.engine_id)), []).append(
                (str(row.span_id), int(row.char_start), int(row.char_end))
            )
        sites = requests[list(LABEL_SITE_COLUMNS)].assign(
            site_id=requests["site_id"].astype(str).str.split("|", n=1).str[1]
        )

        frames: list[tuple[str, pd.DataFrame]] = []
        for method in (M0, *neural):
            raw_path = RAW_ORACLE_CACHE / f"{slug}.{method}.{COND_ORACLE_LOC}.parquet"
            if not raw_path.is_file():
                if method == M0 or method_applies(method, spec["corpus"]):
                    raise PhaseError(f"missing oracle-localization outputs: {_relative(raw_path)}")
                continue
            _site, proposals, shard = _neural_edits(
                name,
                method,
                pd.read_parquet(raw_path),
                requests,
                spans_by_pair,
                clean_llm_output if method == M2 else None,
            )
            shards.append(
                {"environment": name, "method": method, **shard, "proposals": len(proposals)}
            )
            frames.append((method, proposals))
        ground_truth_raw = _raw_frame(
            [
                {
                    "environment": name,
                    "method": C3_ORACLE,
                    "condition": COND_ORACLE_LOC,
                    "request_id": str(row.request_id),
                    "rank": 0,
                    "output": str(row.region_gt),
                }
                for row in truth.itertuples(index=False)
            ]
        )
        _site, c3, shard = _neural_edits(name, C3_ORACLE, ground_truth_raw, requests, spans_by_pair)
        shards.append({"environment": name, "method": C3_ORACLE, **shard, "proposals": len(c3)})
        frames.append((C3_ORACLE, c3))
        frames = [(method, frame) for method, frame in frames if not frame.empty]

        label_input = pd.concat([frame[label_columns] for _m, frame in frames], ignore_index=True)
        labels, _diagnostics = s14._labels(
            spec["corpus"], bundles, spans, streams, sites, label_input
        )
        links = cg1._site_links(name, sites, indexes, by_span, by_alignment, AnchorKind)
        for method, frame in frames:
            labelled = _method_frame(spec, method, COND_ORACLE_LOC, frame, labels, roles)
            if not labelled.empty:
                candidate_frames.append(labelled)
            ids = set((name + "|" + frame["site_id"].astype(str)).tolist())
            chosen = links[links["site_id"].isin(ids)]
            if not chosen.empty:
                link_frames.append(chosen.assign(method=method, condition=COND_ORACLE_LOC))

        pairs = set(
            zip(links["site_id"].astype(str), links["align_site_id"].astype(str), strict=True)
        )
        width = links.groupby("site_id")["align_site_id"].nunique()
        for row in truth.itertuples(index=False):
            faithful["oracle_sites"] += 1
            faithful["linked_to_own_source_site"] += int(
                (str(row.site_id), str(row.align_site_id)) in pairs
            )
            faithful["linked_to_more_than_one"] += int(int(width.get(str(row.site_id), 0)) > 1)
        expressibility.append(
            {
                key: value
                for key, value in census.items()
                if key == "environment" or (isinstance(value, int) and not isinstance(value, bool))
            }
        )
        print(
            f"  {name}: {len(requests)} oracle sites, {len(label_input)} proposals from "
            f"{len(frames)} arms ({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    _write_parquet_once(ORACLE_CANDIDATES, pd.concat(candidate_frames, ignore_index=True))
    _write_parquet_once(ORACLE_LINKS, pd.concat(link_frames, ignore_index=True))
    _write_json_once(
        ORACLE_NORMALIZATION,
        {
            **_oracle_envelope("oracle_localization_normalization", oracle_localization=True),
            "expressibility": expressibility,
            "site_faithfulness": faithful,
            "proposal_census": shards,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"oracles: {sum(int(row['expressible']) for row in expressibility)} expressible oracle "
        f"sites labelled over {len(expressibility)} environments"
    )
    return 0


def run_oracle_analysis() -> int:
    """Sections 21, 45 and 46, read from the oracle tables. Analysis-only, like every oracle here.

    Split from the labelling pass so that section 54 can regenerate the analysis without
    re-aligning every page; the tables it reads are regenerated separately and compared.
    """
    started = time.monotonic()
    for path in (ORACLE_LOCALIZATION, CANDIDATE_ORACLE, RISK_CANDIDATE_ORACLE):
        _forbid(path)
    normalization = cc_read_json(ORACLE_NORMALIZATION)
    expressibility = normalization["expressibility"]
    faithful = normalization["site_faithfulness"]
    shards = normalization["proposal_census"]
    oracle_table = pd.read_parquet(ORACLE_CANDIDATES)
    truth_all = pd.concat(
        [
            pd.read_parquet(CACHE / f"{_slug(spec['environment'])}.oracle_truth.parquet")
            .drop(columns=["region_gt"])
            .assign(environment=spec["environment"])
            for spec in s14.ENVIRONMENTS
        ],
        ignore_index=True,
    )

    errors, oracle_arms = load_arms(ORACLE_CANDIDATES, ORACLE_LINKS)
    _errors, arms = load_arms()
    methods = _available_methods(arms)
    total_errors = sum(population.count for population in errors.values())
    expressible = sum(int(row["expressible"]) for row in expressibility)

    # C3: the ground-truth corrector must repair every error site it can express and label.
    c3_labelled = set(oracle_table.loc[oracle_table["method"] == C3_ORACLE, "site_id"].astype(str))
    reached = truth_all[truth_all["site_id"].isin(c3_labelled)]
    repaired_sources = 0
    for name, population in errors.items():
        repaired_ids = set(
            population.ids[oracle_arms[name, C3_ORACLE, COND_ORACLE_LOC].repaired()].tolist()
        )
        repaired_sources += int(
            reached.loc[reached["environment"] == name, "align_site_id"].isin(repaired_ids).sum()
        )
    identity_regions = sum(int(s["declined_unchanged"]) for s in shards if s["method"] == C3_ORACLE)
    empty_regions = sum(int(s["empty_output"]) for s in shards if s["method"] == C3_ORACLE)
    c3_record = {
        "ocr_error_sites": total_errors,
        "ocr_error_sites_seen_by_prep": sum(int(row["ocr_error_sites"]) for row in expressibility),
        "expressible_error_sites": expressible,
        "identity_regions": identity_regions,
        "empty_regions": empty_regions,
        "expressible_labelable_error_sites": len(reached),
        "repaired_error_sites": repaired_sources,
        "opportunity_recall_on_expressible_labelable": _ratio(repaired_sources, len(reached)),
        "mean_opportunity_recall_all_error_sites": _mean(
            [oracle_arms[name, C3_ORACLE, COND_ORACLE_LOC].opportunity_recall() for name in errors]
        ),
        "shortfall": {
            "not_expressible_as_a_region": total_errors - expressible,
            "region_already_equals_its_ground_truth": identity_regions,
            "empty_region_and_empty_ground_truth": empty_regions,
            "not_labelable": expressible - identity_regions - empty_regions - len(reached),
            "labelled_but_source_site_not_repaired": len(reached) - repaired_sources,
        },
    }

    decomposition: dict[str, Any] = {}
    for method in methods:
        natural_condition = COND_FROZEN if method == M0 else COND_NATURAL
        rows = []
        for name in errors:
            natural = arms[name, method, natural_condition].opportunity_recall()
            frozen = arms[name, method, COND_FROZEN].opportunity_recall()
            located = oracle_arms[name, method, COND_ORACLE_LOC].opportunity_recall()
            rows.append(
                {
                    "environment": name,
                    "natural": natural,
                    "frozen_localization": frozen,
                    "oracle_localization": located,
                    "localization_headroom": located - natural,
                    "generation_headroom": 1.0 - located,
                }
            )
        decomposition[method] = {
            "natural_condition_is": natural_condition,
            "per_environment": rows,
            **{
                f"mean_{key}": _mean([row[key] for row in rows])
                for key in (
                    "natural",
                    "frozen_localization",
                    "oracle_localization",
                    "localization_headroom",
                    "generation_headroom",
                )
            },
            "environments_with_negative_localization_headroom": [
                row["environment"] for row in rows if row["localization_headroom"] < 0
            ],
        }

    located_frame = _rows_for(oracle_arms, COND_ORACLE_LOC)
    per_method = {
        method: {
            **_method_summary(located_frame, method),
            "per_environment": located_frame[located_frame["method"] == method][
                [
                    "environment",
                    "discovered_error_sites",
                    "repairable_sites",
                    "discovery_recall",
                    "generation_success",
                    "opportunity_recall",
                    "candidates_per_site",
                    "harmful_candidates_per_site",
                ]
            ].to_dict("records"),
        }
        for method in methods
    }
    _write_json_once(
        ORACLE_LOCALIZATION,
        {
            **_oracle_envelope("oracle_localization", oracle_localization=True),
            "definition": (
                "section 45: each arm is handed the true OCR error region and its OCR context, "
                "never the correct text. It uses ground-truth localization and no ground-truth "
                "repair content, and it is never compared with a deployable method."
            ),
            "region_representation": {
                "one_span": "token",
                "two_spans": "token_pair",
                "three_or_more_spans": "token_run (outside M0's anchor vocabulary)",
                "no_ocr_span": "gap between the two flanking spans",
            },
            "expressibility": expressibility,
            "site_faithfulness": faithful,
            "per_method": per_method,
            "three_level_decomposition": decomposition,
            "decomposition_note": (
                "section 46: natural opportunity, then the headroom localization would add, then "
                "the headroom that remains even with the true location. A negative localization "
                "headroom is reported as it is and never forced to zero."
            ),
            "c3_ground_truth_oracle_corrector": c3_record,
            "proposal_census": shards,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    candidate_oracle: dict[str, Any] = {}
    for condition, source in (
        (COND_FROZEN, arms),
        (COND_NATURAL, arms),
        (COND_ORACLE_LOC, oracle_arms),
    ):
        frame = _rows_for(source, condition)
        if frame.empty:
            continue
        candidate_oracle[condition] = {
            method: {
                "mean_candidate_oracle_repair_recall": _summary_mean(frame, method),
                "pooled_candidate_oracle_repair_recall": _pooled(
                    frame[frame["method"] == method]
                ).get("opportunity_recall", float("nan")),
            }
            for method in methods
            if (frame["method"] == method).any()
        }
    _write_json_once(
        CANDIDATE_ORACLE,
        {
            **_oracle_envelope("candidate_oracle"),
            "definition": (
                "section 21: if an exact repair exists among an arm's candidates at a site, select "
                "it perfectly. This equals opportunity recall by construction -- perfect selection "
                "repairs exactly the sites a correct candidate exists for -- and is reported under "
                "its own name so the ceiling each arm offers can be read in one place."
            ),
            "conditions": candidate_oracle,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    risk: dict[str, Any] = {}
    conflicts = 0
    for condition, source in (
        (COND_FROZEN, arms),
        (COND_NATURAL, arms),
        (COND_ORACLE_LOC, oracle_arms),
    ):
        for method in methods:
            selected = [
                arm for (_n, m, c), arm in sorted(source.items()) if m == method and c == condition
            ]
            if not selected:
                continue
            tallies = [_perfect_selection(arm) for arm in selected]
            conflicts += sum(t["exact_and_harmful_candidates"] for t in tallies)
            accepted = sum(t["accepted"] for t in tallies)
            value = _mean([arm.opportunity_recall() for arm in selected])
            risk.setdefault(condition, {})[method] = {
                "accepted": accepted,
                "harmful_accepted": 0,
                "automatic_harm": _ratio(0, accepted) if accepted else 0.0,
                "by_epsilon": {
                    str(epsilon): {"feasible": True, "mean_repair_recall": value}
                    for epsilon in EPSILONS
                },
            }
    _write_json_once(
        RISK_CANDIDATE_ORACLE,
        {
            **_oracle_envelope("risk_candidate_oracle"),
            "primary_epsilon": PRIMARY_EPSILON,
            "epsilons": list(EPSILONS),
            "definition": (
                "section 21: the candidate oracle constrained to automatic harm <= epsilon. The "
                "perfect selector accepts only exact candidates, and an exact candidate is never "
                "harmful, so its harm is zero and the constraint never binds: the value equals the "
                "unconstrained candidate oracle at every tolerance."
            ),
            "exact_and_harmful_candidates": conflicts,
            "invariant_across_epsilon": conflicts == 0,
            "conditions": risk,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if conflicts:
        raise PhaseError(
            f"{conflicts} candidates are both exact and harmful; the labels are inconsistent"
        )
    print(
        "oracles: oracle localization "
        + ", ".join(f"{m} {decomposition[m]['mean_oracle_localization']:.4f}" for m in methods)
        + f"; C3 repairs {repaired_sources} of {len(reached)} expressible labelable sites"
    )
    return 0


# ------------------------------------------------------------------ section 26: the prompt audit

AUDIT_ENVIRONMENTS = ("funsd/doctr", "sbb/doctr")
AUDIT_REGIONS = 64
AUDIT_GAPS = 32
AUDIT_LINES = 24


def _adaptation_audit_requests(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Region and line requests from ADAPTATION-split documents only.

    The audit never sees an evaluation document. Regions are M0's own adaptation sites, sampled
    deterministically by site id, with gap anchors drawn separately because a whitespace fragment
    is the case a text-only prompt is most likely to mishandle. Lines are built by the very
    function that builds the natural condition's lines.
    """
    from ocr_risk.canonical import rebuild_stream

    name = spec["environment"]
    slug = _slug(name)
    m0 = pd.read_parquet(CACHE / f"{slug}.m0_candidates.parquet")
    sites = (
        m0[m0["role"] == s14.ROLE_ADAPTATION]
        .drop_duplicates("site_id")
        .sort_values("site_id", kind="stable")
    )
    gap = sites["anchor_kind"].astype(str) == "gap"
    chosen = pd.concat(
        [sites[~gap].head(AUDIT_REGIONS), sites[gap].head(AUDIT_GAPS)], ignore_index=True
    )
    regions = pd.DataFrame(
        {
            "environment": name,
            "condition": COND_FROZEN,
            "request_id": (name + "|audit|" + chosen["site_id"].astype(str)).to_numpy(),
            "document_id": chosen["document_id"].astype(str).to_numpy(),
            "engine_id": chosen["engine_id"].astype(str).to_numpy(),
            "site_id": (name + "|" + chosen["site_id"].astype(str)).to_numpy(),
            "anchor_kind": chosen["anchor_kind"].astype(str).to_numpy(),
            "anchor_ref": chosen["anchor_ref"].astype(str).to_numpy(),
            "char_start": chosen["char_start"].to_numpy(dtype=np.int64),
            "char_end": chosen["char_end"].to_numpy(dtype=np.int64),
            "region": chosen["original_ocr"].astype(str).to_numpy(),
            "context_before": chosen["context_before"].astype(str).to_numpy(),
            "context_after": chosen["context_after"].astype(str).to_numpy(),
            "window_start": chosen["char_start"].to_numpy(dtype=np.int64),
            "window_text": chosen["original_ocr"].astype(str).to_numpy(),
        },
        columns=list(REQUEST_COLUMNS),
    )
    bundles = s14._document_bundles(spec["corpus"])
    roles = s14.partition_of(spec["corpus"], sorted(bundles))
    adaptation = {d for d, role in roles.items() if role == s14.ROLE_ADAPTATION}
    spans, _record = s14._canonical_spans(spec, bundles)
    streams = {
        pair: rebuild_stream(group)
        for pair, group in spans.items()
        if group and pair[0] in adaptation
    }
    span_table = pd.read_parquet(CACHE / f"{slug}.spans.parquet")
    lines = _requests_natural(
        name, span_table[span_table["document_id"].isin(adaptation)], streams
    ).head(AUDIT_LINES)
    return regions, lines


def _format_census(requests: pd.DataFrame, raw: pd.DataFrame, field: str) -> dict[str, Any]:
    """GT-blind format measurements of one set of LLM answers. No label is read."""
    answers = dict(zip(raw["request_id"].astype(str), raw["output"].astype(str), strict=True))
    counts = {"answers": 0, "delimited": 0, "echoes_context": 0, "overlong": 0, "unchanged": 0}
    for row in requests.itertuples(index=False):
        text = answers.get(str(row.request_id), "")
        parsed = clean_llm_output(text)
        source = str(getattr(row, field)).strip()
        before = str(row.context_before).replace("\n", " ").strip()
        after = str(row.context_after).replace("\n", " ").strip()
        counts["answers"] += 1
        counts["delimited"] += int("[[" in text and "]]" in text)
        counts["echoes_context"] += int(
            (len(before) >= 8 and before[-8:] in parsed)
            or (len(after) >= 8 and after[:8] in parsed)
        )
        counts["overlong"] += int(len(parsed) > 3 * len(source) + 12)
        counts["unchanged"] += int(parsed == source)
    total = max(counts["answers"], 1)
    return {
        **counts,
        **{
            f"{key}_share": counts[key] / total
            for key in ("delimited", "echoes_context", "overlong", "unchanged")
        },
    }


def run_llmaudit() -> int:
    """A GT-blind format and throughput audit of the LLM prompt, on adaptation documents only.

    Section 26 requires a machine-readable answer. This checks -- before the registries freeze and
    before any label exists -- whether the frozen prompt elicits one, how fast the arm runs
    under its frozen decoding, and whether a repeat reproduces it. It never touches an evaluation
    document and never reads a label, so it cannot tune the arm toward an outcome.
    """
    started = time.monotonic()
    _forbid(LLM_FORMAT_AUDIT)
    _require(ATOMIC_EDIT_SCHEMA, "schema")
    if METHOD_CANDIDATES.exists():
        raise PhaseError("labels already exist; the prompt audit belongs before them")
    snapshot = _hf_snapshot(LLM_REPO, LLM_REVISION)
    if snapshot is None:
        _write_json_once(
            LLM_FORMAT_AUDIT,
            {**_envelope("llm_format_audit"), "run": False, "reason": "no local snapshot"},
        )
        print("llmaudit: the LLM arm is unavailable; recorded")
        return 0
    import transformers

    transformers.utils.logging.set_verbosity_error()
    runner = _LlmRunner(snapshot, _resolve_device())
    silent: Callable[[str], None] = lambda _message: None  # noqa: E731
    environments: list[dict[str, Any]] = []
    for name in AUDIT_ENVIRONMENTS:
        spec = next(s for s in s14.ENVIRONMENTS if s["environment"] == name)
        regions, lines = _adaptation_audit_requests(spec)
        first, region_cost = _run_arm(runner, regions, M2, COND_FROZEN, silent)
        second, _cost = _run_arm(runner, regions, M2, COND_FROZEN, silent)
        halved, _cost = _run_arm(
            runner, regions, M2, COND_FROZEN, silent, batch_size=LLM_BATCH // 2
        )
        line_raw, line_cost = _run_arm(runner, lines, M2, COND_NATURAL, silent)
        a = dict(zip(first["request_id"], first["output"], strict=True))
        b = dict(zip(second["request_id"], second["output"], strict=True))
        c = dict(zip(halved["request_id"], halved["output"], strict=True))
        record = {
            "environment": name,
            "split": s14.ROLE_ADAPTATION,
            "region_requests": len(regions),
            "gap_requests": int((regions["anchor_kind"] == "gap").sum()),
            "line_requests": len(lines),
            "regions": _format_census(regions, first, "region"),
            "lines": _format_census(lines, line_raw, "window_text"),
            "region_requests_per_second": region_cost["requests_per_second"],
            "line_requests_per_second": line_cost["requests_per_second"],
            "identical_on_repeat": a == b,
            "answers_differing_at_half_batch": sum(a[key] != c.get(key) for key in a),
            "request_id_digest": canonical_hash(
                sorted(regions["request_id"].tolist() + lines["request_id"].tolist())
            ),
        }
        environments.append(record)
        print(
            f"  {name}: regions delimited {record['regions']['delimited_share']:.2f} echo "
            f"{record['regions']['echoes_context_share']:.2f} overlong "
            f"{record['regions']['overlong_share']:.2f} at "
            f"{record['region_requests_per_second']:.1f}/s; lines delimited "
            f"{record['lines']['delimited_share']:.2f} at "
            f"{record['line_requests_per_second']:.1f}/s; "
            f"repeat identical {record['identical_on_repeat']}, "
            f"{record['answers_differing_at_half_batch']} differ at half batch",
            flush=True,
        )
    _write_json_once(
        LLM_FORMAT_AUDIT,
        {
            **_envelope("llm_format_audit"),
            "run": True,
            "ground_truth_read": False,
            "evaluation_documents_used": False,
            "prompt_revision": PROMPT_REVISION,
            "prompt_hash": canonical_hash(
                {
                    "system": LLM_SYSTEM_PROMPT,
                    "context": LLM_CONTEXT_TEMPLATE,
                    "line": LLM_LINE_TEMPLATE,
                }
            ),
            "dtype": LLM_DTYPE,
            "batch_size": LLM_BATCH,
            "definitions": {
                "delimited": "the raw answer contains a [[ ]] pair",
                "echoes_context": (
                    "the parsed answer contains the last 8 characters of the text before the "
                    "fragment or the first 8 of the text after it -- a heuristic for returning "
                    "context instead of the fragment"
                ),
                "overlong": "the parsed answer is longer than 3 x the fragment + 12 characters",
                "unchanged": "the parsed answer equals the fragment",
            },
            "environments": environments,
            "decision_rule": (
                "the prompt is revised at most once and frozen after this audit whatever it "
                "shows; the audit informs the record, not a search"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"llmaudit: {len(environments)} adaptation environments -> {_relative(LLM_FORMAT_AUDIT)}")
    return 0


# ------------------------------------------------------------------ section 49: falsification tests

GROUND_TRUTH_WORDS = ("gt", "truth", "ground", "label", "outcome", "exact", "harm", "benefic")
PRIMARY_PHASE_NAMES = (
    "run_discovery",
    "run_generation",
    "run_opportunity",
    "run_matchedk",
    "run_harmful",
    "run_strata",
    "run_overlap",
    "run_nearmiss",
    "run_shift",
    "run_stats",
)
GT_BLIND_FUNCTION_NAMES = (
    "run_generate",
    "run_prepare",
    "_prepare_environment",
    "_environment_inputs",
    "_ocr_artifacts",
    "_requests_frozen_localization",
    "_requests_natural",
    "_run_arm",
    "_prompt_for",
)
GT_MACHINERY = (
    "align_document",
    "AlignmentIndex",
    "_labels",
    "build_sites",
    "region_truth",
    "gap_truth",
)


_SOURCE_AT_IMPORT = Path(__file__).read_text(encoding="utf-8")
_DEFINITIONS: dict[str, Any] = {}


def _source_tree(function: Callable[..., Any]) -> Any:
    """The syntax tree of a function as this process imported it.

    Read from the source captured at import, not from the file on disk. A structural check run by
    a long-lived process must inspect the code that process is running; a file edited after import
    would otherwise hand it a different function sitting at the same line numbers.
    """
    import ast

    if not _DEFINITIONS:
        for node in ast.parse(_SOURCE_AT_IMPORT).body:
            if isinstance(node, ast.FunctionDef):
                _DEFINITIONS[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef):
                        _DEFINITIONS[f"{node.name}.{member.name}"] = member
    return _DEFINITIONS[function.__qualname__]


def _names_used(function: Callable[..., Any]) -> set[str]:
    """Every bare name and attribute a function's source mentions."""
    import ast

    tree = _source_tree(function)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def _attributes_of(function: Callable[..., Any], variable: str) -> set[str]:
    """The attributes a function reads off one named variable."""
    import ast

    return {
        node.attr
        for node in ast.walk(_source_tree(function))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == variable
    }


def _with_rows(
    arm: Arm, exact: np.ndarray, harmful: np.ndarray, site_of: np.ndarray, rank: np.ndarray
) -> Arm:
    """An arm with extra candidate rows appended; everything else about it unchanged."""
    from dataclasses import replace

    extra = site_of.size
    pad_int = np.zeros(extra, dtype=np.int64)
    pad_obj = np.full(extra, "", dtype=object)
    return replace(
        arm,
        candidate_id=np.concatenate([arm.candidate_id, pad_obj]),
        document_id=np.concatenate([arm.document_id, pad_obj]),
        site_of=np.concatenate([arm.site_of, site_of]),
        exact=np.concatenate([arm.exact, exact]),
        harmful=np.concatenate([arm.harmful, harmful]),
        beneficial=np.concatenate([arm.beneficial, np.zeros(extra, dtype=bool)]),
        d_before=np.concatenate([arm.d_before, pad_int]),
        d_after=np.concatenate([arm.d_after, pad_int]),
        rank=np.concatenate([arm.rank, rank]),
        anchor_kind=np.concatenate([arm.anchor_kind, pad_obj]),
        edit_kind=np.concatenate([arm.edit_kind, pad_obj]),
        original_chars=np.concatenate([arm.original_chars, pad_int]),
        candidate_chars=np.concatenate([arm.candidate_chars, pad_int]),
    )


def _removal_check(arm: Arm) -> dict[str, Any]:
    """Remove the exact candidates at a seeded half of the repairable proposal sites.

    The expected loss is recomputed independently, by inverting the link structure and asking of
    each error site whether every exact-bearing proposal site that reached it was removed. The
    observed loss is the arm's own `repaired`. They must be the same set, not just the same size.
    """
    if not arm.n_rows:
        return {"removed_sites": 0, "expected_lost": 0, "observed_lost": 0, "agrees": True}
    bearing = np.flatnonzero(_any_by_group(arm.exact, arm.site_of, arm.n_sites))
    rng = np.random.default_rng(_stable_seed(arm.environment, arm.method, arm.condition, "removal"))
    removed = (
        set(rng.choice(bearing, size=bearing.size // 2, replace=False).tolist())
        if bearing.size
        else set()
    )
    keep = ~(arm.exact & np.isin(arm.site_of, np.fromiter(removed, dtype=np.int64)))
    after = arm.linked(_any_by_group(arm.exact & keep, arm.site_of, arm.n_sites))
    reach: dict[int, set[int]] = {}
    for site in bearing.tolist():
        start, end = arm.site_link_indptr[site], arm.site_link_indptr[site + 1]
        for error in arm.site_link_index[start:end].tolist():
            reach.setdefault(int(error), set()).add(int(site))
    expected = {error for error, sources in reach.items() if sources <= removed}
    observed = set(np.flatnonzero(arm.repaired() & ~after).tolist())
    return {
        "removed_sites": len(removed),
        "expected_lost": len(expected),
        "observed_lost": len(observed),
        "agrees": expected == observed,
    }


def _apply_edits(source: str, edits: Sequence[AtomicEdit]) -> str:
    """Apply atomic edits to a window, right to left so earlier offsets stay valid."""
    text = source
    for edit in sorted(edits, key=lambda e: (e.char_start, e.char_end), reverse=True):
        replacement = f" {edit.candidate} " if edit.edit_kind == EDIT_GAP else edit.candidate
        text = text[: edit.char_start] + replacement + text[edit.char_end :]
    return text


def _diff_reconstruction() -> dict[str, Any]:
    """Section 49 test 8: the shared diff's edits, applied back, reproduce the rewrite.

    Every natural-condition rewrite in the cache is diffed again and its edits applied to the
    source line. A rewrite whose diff refused an edit -- unprojectable, or outside every span -- is
    untestable by construction and counted separately. The comparison collapses whitespace runs,
    because snapping an edit to its spans leaves the separators around it where they were.
    """
    tested = exact = normalized = untestable = 0
    failures: list[dict[str, str]] = []
    for raw_path in sorted(RAW_CACHE.glob(f"*.{COND_NATURAL}.parquet")):
        slug, method, _condition = raw_path.name[: -len(".parquet")].split(".")
        requests = pd.read_parquet(CACHE / f"{slug}.requests.parquet")
        requests = requests[requests["condition"] == COND_NATURAL].set_index("request_id")
        span_table = pd.read_parquet(CACHE / f"{slug}.spans.parquet")
        spans_by_pair: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
        for row in span_table.itertuples(index=False):
            spans_by_pair.setdefault((str(row.document_id), str(row.engine_id)), []).append(
                (str(row.span_id), int(row.char_start), int(row.char_end))
            )
        parse = clean_llm_output if method == M2 else None
        for row in pd.read_parquet(raw_path).itertuples(index=False):
            request = requests.loc[str(row.request_id)]
            window = str(request["window_text"])
            output = str(row.output) if parse is None else parse(str(row.output))
            if output.strip() == window.strip():
                continue
            start = int(request["window_start"])
            local = [
                (span_id, s - start, e - start)
                for span_id, s, e in spans_by_pair.get(
                    (str(request["document_id"]), str(request["engine_id"])), []
                )
                if s >= start and e <= int(request["char_end"])
            ]
            edits, census = atomic_edits_from_rewrite(window, output, local)
            if census["dropped_unprojectable"] or census["dropped_outside_any_span"]:
                untestable += 1
                continue
            rebuilt = _apply_edits(window, edits)
            tested += 1
            exact += int(rebuilt == output)
            same = " ".join(rebuilt.split()) == " ".join(output.split())
            normalized += int(same)
            if not same and len(failures) < 5:
                failures.append(
                    {
                        "method": method,
                        "source_chars": str(len(window)),
                        "rebuilt_equals_rewrite": "false",
                    }
                )
    return {
        "rewrites_tested": tested,
        "reconstructed_exactly": exact,
        "reconstructed_up_to_whitespace": normalized,
        "untestable_because_the_diff_refused_an_edit": untestable,
        "failures_sample": failures,
        "agrees": tested > 0 and normalized == tested,
    }


def run_negative() -> int:
    """Section 49. Twelve falsification tests, each stated so that it could fail.

    Several are behavioural -- a result recomputed a second, independent way and compared -- and
    several are structural, read from this module's own source, because "the model never saw the
    ground truth" is a claim about code paths and is best checked against the code paths.
    """
    started = time.monotonic()
    _forbid(NEGATIVE_TESTS)
    for path, phase in ((UPSTREAM_REPRODUCTION, "reproduce"), (ORACLE_LOCALIZATION, "oracles")):
        _require(path, phase)
    _errors, arms = load_arms()
    reproduction = cc_read_json(UPSTREAM_REPRODUCTION)
    normalization = cc_read_json(NORMALIZATION_RESULTS)
    oracle = cc_read_json(ORACLE_LOCALIZATION)
    candidates = pd.read_parquet(METHOD_CANDIDATES)
    auxiliary = pd.read_parquet(AUXILIARY_CANDIDATES)
    tests: list[dict[str, Any]] = []

    def record(name: str, claim: str, observed: dict[str, Any], passed: bool) -> None:
        tests.append({"test": name, "claim": claim, "observed": observed, "passed": bool(passed)})

    record(
        "f1_m0_reproduces_sgv_cg1",
        "M0 re-run here reproduces every SGV-CG1 check with zero differences",
        {"checks": reproduction["checks_total"], "differences": reproduction["differences"]},
        reproduction["differences"] == 0,
    )

    identity = [row for row in normalization["controls"] if row["method"] == C1_IDENTITY]
    surviving = sum(int(row["proposals"]) for row in identity)
    record(
        "f2_identity_corrector_creates_no_exact_repair",
        "a corrector that returns its input produces no candidate and so no exact repair",
        {
            "proposals_surviving_normalization": surviving,
            "identity_rows_in_the_auxiliary_table": int((auxiliary["method"] == C1_IDENTITY).sum()),
        },
        surviving == 0 and not (auxiliary["method"] == C1_IDENTITY).any(),
    )

    truth = oracle["c3_ground_truth_oracle_corrector"]
    record(
        "f3_ground_truth_oracle_reaches_one",
        "proposing each region's ground truth repairs every expressible, labelable error site",
        truth,
        int(truth["repaired_error_sites"]) == int(truth["expressible_labelable_error_sites"]),
    )

    duplicated = [
        np.array_equal(_duplicated(arm).repaired(), arm.repaired()) for arm in arms.values()
    ]
    record(
        "f4_duplication_does_not_raise_opportunity",
        "doubling every candidate leaves every arm's repaired set unchanged",
        {"arms": len(duplicated), "unchanged": sum(duplicated)},
        all(duplicated),
    )

    harmful_only = []
    for arm in arms.values():
        if not arm.n_sites:
            harmful_only.append(True)
            continue
        sites = np.arange(arm.n_sites, dtype=np.int64)
        grown = _with_rows(
            arm,
            exact=np.zeros(sites.size, dtype=bool),
            harmful=np.ones(sites.size, dtype=bool),
            site_of=sites,
            rank=np.zeros(sites.size, dtype=np.int64),
        )
        harmful_only.append(bool(np.array_equal(grown.repaired(), arm.repaired())))
    record(
        "f5_harmful_only_additions_do_not_raise_opportunity",
        "adding a harmful, non-exact candidate at every proposal site changes no repaired set",
        {"arms": len(harmful_only), "unchanged": sum(harmful_only)},
        all(harmful_only),
    )

    removal = [_removal_check(arm) for arm in arms.values()]
    record(
        "f6_removing_correct_candidates_reduces_opportunity_exactly",
        "removing exact candidates loses exactly the error sites an inverse-link recomputation "
        "predicts",
        {
            "arms": len(removal),
            "agreeing": sum(r["agrees"] for r in removal),
            "sites_lost": sum(r["observed_lost"] for r in removal),
        },
        all(r["agrees"] for r in removal),
    )

    sources = candidates["candidate_source_set"].astype(str)
    own = [
        m in s.split("+") for m, s in zip(candidates["method"].astype(str), sources, strict=True)
    ]
    key = ["environment", "condition", "document_id", "char_start", "char_end", "candidate_text"]
    proposers = candidates.groupby(key, sort=False)["method"].agg(
        lambda v: "+".join(sorted(set(v)))
    )
    listed = candidates.groupby(key, sort=False)["candidate_source_set"].first()
    record(
        "f7_attribution_survives_deduplication",
        "every row names its own method, and every edit lists exactly the methods that proposed it",
        {
            "rows": len(candidates),
            "rows_naming_their_own_method": int(sum(own)),
            "empty_source_sets": int((sources == "").sum()),
            "edits": len(proposers),
            "edits_listing_exactly_their_proposers": int((proposers == listed).sum()),
        },
        all(own) and not (sources == "").any() and bool((proposers == listed).all()),
    )

    reconstruction = _diff_reconstruction()
    record(
        "f8_whole_text_diff_reconstructs_the_rewrite",
        "applying the shared diff's edits to each source line reproduces the model's rewrite",
        reconstruction,
        bool(reconstruction["agrees"]),
    )

    fields = _attributes_of(_prompt_for, "row")
    leaky_columns = [c for c in REQUEST_COLUMNS if any(word in c for word in GROUND_TRUTH_WORDS)]
    cached = {
        path.name: list(pd.read_parquet(path).columns) == list(REQUEST_COLUMNS)
        for path in sorted(CACHE.glob("*.requests.parquet"))
    }
    oracle_columns = {
        path.name: sorted(
            set(pd.read_parquet(path).columns) - {*REQUEST_COLUMNS, *ORACLE_REQUEST_EXTRA}
        )
        for path in sorted(CACHE.glob("*.oracle_requests.parquet"))
    }
    blind = {
        name: sorted(_names_used(globals()[name]) & set(GT_MACHINERY))
        for name in GT_BLIND_FUNCTION_NAMES
    }
    record(
        "f9_no_method_receives_ground_truth",
        "prompts read only OCR fields, request tables carry no ground-truth column, and the "
        "GT-blind path never touches the alignment or the labeller",
        {
            "prompt_reads": sorted(fields),
            "request_columns_named_like_ground_truth": leaky_columns,
            "request_caches_with_exactly_the_request_columns": sum(cached.values()),
            "request_caches": len(cached),
            "oracle_request_columns_beyond_the_request_columns": oracle_columns,
            "ground_truth_machinery_named_in_the_gt_blind_path": blind,
        },
        fields <= {"region", "window_text", "context_before", "context_after"}
        and not leaky_columns
        and all(cached.values())
        and not any(oracle_columns.values())
        and not any(blind.values()),
    )

    forbidden = {"ORACLE_CANDIDATES", "ORACLE_LINKS", "RAW_ORACLE_CACHE", "ORACLE_LOCALIZATION"}
    reads = {name: sorted(_names_used(globals()[name]) & forbidden) for name in PRIMARY_PHASE_NAMES}
    oracle_rows = int((candidates["condition"] == COND_ORACLE_LOC).sum())
    oracle_raw = [path.name for path in RAW_CACHE.glob(f"*{COND_ORACLE_LOC}*")]
    record(
        "f10_oracle_localization_is_isolated",
        "no primary phase names an oracle artifact, and no oracle row or output sits in a primary "
        "table",
        {
            "primary_phases_naming_an_oracle_artifact": reads,
            "oracle_rows_in_the_primary_table": oracle_rows,
            "oracle_files_in_the_primary_cache": oracle_raw,
        },
        not any(reads.values()) and oracle_rows == 0 and not oracle_raw,
    )

    budget_reads = _attributes_of(Arm.budget, "self")
    shuffled_equal = []
    for arm in arms.values():
        if not arm.n_rows:
            continue
        from dataclasses import replace

        rng = np.random.default_rng(_stable_seed(arm.environment, arm.method, "labels"))
        permuted = replace(
            arm, exact=rng.permutation(arm.exact), harmful=rng.permutation(arm.harmful)
        )
        shuffled_equal.append(
            all(np.array_equal(arm.budget(k), permuted.budget(k)) for k in K_GRID)
        )
    record(
        "f11_matched_k_truncation_reads_no_label",
        "the budget mask reads only the arm's own ranks and is unchanged when labels are permuted",
        {
            "budget_reads": sorted(budget_reads),
            "arms_checked": len(shuffled_equal),
            "unchanged": sum(shuffled_equal),
        },
        budget_reads <= {"rank", "n_rows"} and all(shuffled_equal),
    )

    links_gate = next(c for c in reproduction["checks"] if c["check"] == "m0_site_links")
    linkers = {
        name: "_site_links" in _names_used(globals()[name])
        for name in ("run_normalize", "run_oracles")
    }
    record(
        "f12_linking_uses_the_frozen_alignment_machinery",
        "every link is made by SGV-CG1's _site_links, and M0's links reproduce SGV-CG1's exactly",
        {
            "phases_linking_through_site_links": linkers,
            "m0_link_differences": links_gate["observed"],
        },
        all(linkers.values()) and bool(links_gate["agrees"]),
    )

    passed = sum(test["passed"] for test in tests)
    _write_json_once(
        NEGATIVE_TESTS,
        {
            **_oracle_envelope("negative_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"negative: {passed} of {len(tests)} falsification tests pass")
    return 0


# ----------------------------------------------------------------- section 42, 43, 56: the decision

NEXT_STAGES = {
    "A": (
        "NEXT: RELIABILITY TRANSFER TO NEW CORRECTOR",
        "SGV-XR1 -- can the existing candidate-level reliability framework safely rank the "
        "stronger corrector's candidates? Measured zero-shot before any retraining.",
    ),
    "B": (
        "NEXT: MULTI-CORRECTOR FUSION",
        "multi-corrector candidate fusion with reliability: method identity x candidate "
        "reliability.",
    ),
    "C": (
        "NEXT: CONDITIONAL METHOD ROUTING",
        "conditional correction-method routing, and only after verifying that a route can be "
        "chosen without target outcomes.",
    ),
    "D": (
        "NEXT: SITE-LOCALIZATION / REPRESENTATION INVESTIGATION",
        "site localization, the representation of omissions and segmentation, and whether the "
        "atomic edit formulation itself excludes important repairs. Not a return downstream.",
    ),
}


def _stratum_deltas(analysis: dict[str, Any], method: str) -> dict[str, float]:
    """Per-stratum opportunity of an arm minus M0's, in the primary condition, pooled."""
    block = analysis["by_condition"][PRIMARY_CONDITION]
    mine = {row["stratum"]: float(row["opportunity_recall"]) for row in block.get(method, [])}
    base = {row["stratum"]: float(row["opportunity_recall"]) for row in block[M0]}
    return {stratum: mine.get(stratum, 0.0) - value for stratum, value in base.items()}


def run_decide() -> int:
    """Sections 42, 43 and 56. The research decision, by the rule frozen in the design record.

    Every criterion is read from a saved artifact and every threshold from the design record, so
    the decision is a function of what was measured and of nothing chosen after it. Exactly one
    outcome is assigned. External confirmation stays off whatever the result (section 58).
    """
    started = time.monotonic()
    _forbid(DECISION)
    for path, phase in (
        (OPPORTUNITY_RESULTS, "opportunity"),
        (MATCHED_K_RESULTS, "matchedk"),
        (HARMFUL_ANALYSIS, "harmful"),
        (ERROR_TYPE_ANALYSIS, "strata"),
        (ENGINE_ANALYSIS, "strata"),
        (DOMAIN_ANALYSIS, "strata"),
        (ORACLE_LOCALIZATION, "oracles"),
        (UNION_OPPORTUNITY, "overlap"),
        (UNIQUE_REPAIRS, "overlap"),
        (CONTROL_RESULTS, "controls"),
        (NEGATIVE_TESTS, "negative"),
        (STATISTICAL_TESTS, "stats"),
        (DETERMINISM, "determinism"),
    ):
        _require(path, phase)
    design = cc_read_json(DESIGN_RECORD)
    registry = cc_read_json(METHOD_REGISTRY)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)
    matched = cc_read_json(MATCHED_K_RESULTS)
    error_types = cc_read_json(ERROR_TYPE_ANALYSIS)
    engines = cc_read_json(ENGINE_ANALYSIS)
    domains = cc_read_json(DOMAIN_ANALYSIS)
    oracle = cc_read_json(ORACLE_LOCALIZATION)
    union = cc_read_json(UNION_OPPORTUNITY)
    unique = cc_read_json(UNIQUE_REPAIRS)
    controls = cc_read_json(CONTROL_RESULTS)
    negative = cc_read_json(NEGATIVE_TESTS)
    statistics = cc_read_json(STATISTICAL_TESTS)
    determinism = cc_read_json(DETERMINISM)
    thresholds = design["thresholds"]
    margin = float(thresholds["meaningful_delta_opportunity"])

    primary = opportunity["conditions"][PRIMARY_CONDITION]
    summaries = {s["method"]: s for s in primary["by_method"]}
    deltas = {d["method"]: d for d in primary["deltas"] if d.get("environments")}
    natural = {
        d["method"]: d
        for d in opportunity["conditions"].get(COND_NATURAL, {}).get("deltas", [])
        if d.get("environments")
    }
    at_k = {
        d["method"]: d
        for d in matched["conditions"][PRIMARY_CONDITION]["at_primary_k"]["deltas"]
        if d.get("environments")
    }
    baseline = float(summaries[M0]["mean_opportunity_recall"])

    criteria: dict[str, dict[str, Any]] = {}
    for method, delta in deltas.items():
        domain_delta = _stratum_deltas(domains, method)
        recovery = error_types["weak_type_recovery"].get(method, {})
        regeneration = determinism.get("model_regeneration", {}).get(method, {})
        criteria[method] = {
            "c1_opportunity_improvement": float(delta["mean_delta_opportunity_recall"]) >= margin,
            "c2_breadth": int(delta["environments_meaningfully_improved"])
            >= int(thresholds["breadth_majority"]),
            "c3_weak_error_type_recovery": bool(recovery.get("meets_criterion", False)),
            "c4_candidate_burden": float(
                at_k.get(method, {}).get("mean_delta_opportunity_recall", float("-inf"))
            )
            >= margin,
            "c5_domain_robustness": sum(value >= margin for value in domain_delta.values())
            >= int(thresholds["domains_required"]),
            "c6_reproducibility": bool(determinism.get("all_runs_identical"))
            and bool(regeneration.get("agrees", False)),
            "evidence": {
                "mean_delta_opportunity_recall": float(delta["mean_delta_opportunity_recall"]),
                "environments_meaningfully_improved": int(
                    delta["environments_meaningfully_improved"]
                ),
                "weak_types_improved": list(recovery.get("weak_types_improved", [])),
                "mean_delta_opportunity_recall_at_primary_k": at_k.get(method, {}).get(
                    "mean_delta_opportunity_recall"
                ),
                "domain_deltas": domain_delta,
                "determinism_all_runs_identical": bool(determinism.get("all_runs_identical")),
                "model_regeneration": regeneration,
            },
        }
    outcome_a = [
        m
        for m, c in criteria.items()
        if all(
            c[f"c{i}_{n}"]
            for i, n in (
                (1, "opportunity_improvement"),
                (2, "breadth"),
                (3, "weak_error_type_recovery"),
                (4, "candidate_burden"),
                (5, "domain_robustness"),
            )
        )
    ]
    best = (
        max(outcome_a, key=lambda m: (criteria[m]["evidence"]["mean_delta_opportunity_recall"], m))
        if outcome_a
        else max(deltas, key=lambda m: (float(deltas[m]["mean_delta_opportunity_recall"]), m))
        if deltas
        else None
    )

    narrow: dict[str, list[str]] = {}
    for method, delta in deltas.items():
        hits = [
            f"environment:{row['environment']}"
            for row in delta["per_environment"]
            if row["delta_opportunity_recall"] >= margin
        ]
        hits += [f"engine:{k}" for k, v in _stratum_deltas(engines, method).items() if v >= margin]
        hits += [f"domain:{k}" for k, v in _stratum_deltas(domains, method).items() if v >= margin]
        hits += [
            f"error_type:{k}"
            for k, v in error_types["weak_type_recovery"]
            .get(method, {})
            .get("delta_by_type", {})
            .items()
            if v >= margin
        ]
        narrow[method] = hits

    if outcome_a:
        letter = "A"
    elif bool(union["meets_complementarity_rule"]):
        letter = "B"
    elif any(narrow.values()):
        letter = "C"
    else:
        letter = "D"
    label, next_stage = NEXT_STAGES[letter]

    best_summary = summaries.get(best, {}) if best else {}
    best_delta = deltas.get(best, {}) if best else {}
    met_best = (
        sum(
            bool(criteria[best][key])
            for key in (
                "c1_opportunity_improvement",
                "c2_breadth",
                "c3_weak_error_type_recovery",
                "c4_candidate_burden",
                "c5_domain_robustness",
            )
        )
        if best
        else 0
    )
    primary_tests = [t for t in statistics["tests"] if t["family"] == "primary" and "p_value" in t]
    decomposition = oracle["three_level_decomposition"]
    reason = (
        f"outcome {letter}: the best new arm, {best}, moves mean opportunity recall by "
        f"{float(best_delta.get('mean_delta_opportunity_recall', 0.0)):+.4f} against the frozen "
        f"{baseline:.4f}, where {margin} was declared meaningful; it meets {met_best} of the 5 "
        f"outcome-A criteria, and the all-arm union reaches "
        f"{float(union['union_all_methods']['mean_opportunity_recall']):.4f}."
        if best
        else "outcome D: no new correction arm was available to evaluate."
    )

    _write_json_once(
        DECISION,
        {
            **_envelope("research_decision"),
            "stage": STAGE,
            "stage_kind": "DEVELOPMENT / DIAGNOSTIC",
            "status": "COMPLETE",
            "synthetic": False,
            "issued_head": _git("rev-parse", "HEAD"),
            "methods_evaluated": [
                m for m in PRIMARY_METHODS if registry["methods"][m]["available"]
            ],
            "methods_unavailable": {
                m: registry["methods"][m].get("reason", "")
                for m in (*PRIMARY_METHODS, M3)
                if not registry["methods"][m]["available"]
            },
            "primary_condition": PRIMARY_CONDITION,
            "baseline_opportunity": baseline,
            "baseline_matches_sgv_cg1": abs(
                baseline - float(design["baseline"]["opportunity_recall"])
            )
            < 1e-12,
            "best_method": best,
            "best_method_opportunity": best_summary.get("mean_opportunity_recall"),
            "best_method_delta": best_delta.get("mean_delta_opportunity_recall"),
            "discovery_delta": best_delta.get("mean_delta_discovery_recall"),
            "generation_delta": best_delta.get("mean_delta_generation_success"),
            "natural_condition_deltas": {
                m: {
                    "mean_delta_opportunity_recall": d["mean_delta_opportunity_recall"],
                    "mean_delta_discovery_recall": d["mean_delta_discovery_recall"],
                    "mean_delta_generation_success": d["mean_delta_generation_success"],
                }
                for m, d in natural.items()
            },
            "weak_error_types_improved": criteria.get(best, {})
            .get("evidence", {})
            .get("weak_types_improved", []),
            "candidate_burden_change": (
                float(best_summary.get("mean_candidates_per_site", float("nan")))
                - float(summaries[M0]["mean_candidates_per_site"])
                if best
                else None
            ),
            "harmful_candidate_change": (
                float(best_summary.get("mean_harmful_candidates_per_site", float("nan")))
                - float(summaries[M0]["mean_harmful_candidates_per_site"])
                if best
                else None
            ),
            "environments_improved": best_delta.get("environments_improved"),
            "environments_meaningfully_improved": best_delta.get(
                "environments_meaningfully_improved"
            ),
            "domains_improved": [
                k
                for k, v in criteria.get(best, {})
                .get("evidence", {})
                .get("domain_deltas", {})
                .items()
                if v >= margin
            ],
            "method_complementarity": {
                "union_gain_over_best_single": union["complementarity_gain_over_best_single"],
                "meets_complementarity_rule": union["meets_complementarity_rule"],
                "unique_repairs": {m: r["unique_repairs"] for m, r in unique["by_method"].items()},
            },
            "union_opportunity": union["union_all_methods"]["mean_opportunity_recall"],
            "oracle_localization": {
                m: {
                    "mean_natural": r["mean_natural"],
                    "mean_oracle_localization": r["mean_oracle_localization"],
                    "mean_localization_headroom": r["mean_localization_headroom"],
                    "mean_generation_headroom": r["mean_generation_headroom"],
                }
                for m, r in decomposition.items()
            },
            "criteria": criteria,
            "criteria_operationalized": {
                "c1_opportunity_improvement": (
                    f"mean delta opportunity recall >= {margin}, primary condition"
                ),
                "c2_breadth": (
                    f"environments with a per-environment delta >= {margin} reach "
                    f"{thresholds['breadth_majority']}"
                ),
                "c3_weak_error_type_recovery": (
                    f">= {thresholds['weak_error_types_required']} of "
                    f"{thresholds['weak_error_types']} "
                    f"improve by >= {thresholds['weak_error_type_delta']}, pooled"
                ),
                "c4_candidate_burden": f"mean delta at K={PRIMARY_K} >= {margin}",
                "c5_domain_robustness": (
                    f"pooled delta >= {margin} in each of {thresholds['domains_required']} domains"
                ),
                "c6_reproducibility": (
                    "the derived pipeline regenerates identically from the immutable raw cache and "
                    "the arm's model reproduces its cached outputs on a regenerated sample"
                ),
                "outcome_rule": design["outcome_rule"],
            },
            "narrow_improvements": narrow,
            "outcome": letter,
            "outcome_label": OUTCOME_TAXONOMY[letter],
            "recommended_next_stage": label,
            "next_stage": next_stage,
            "ready_for_reliability_transfer_study": letter == "A",
            "ready_for_external_confirmation": False,
            "external_confirmation_note": design["external_confirmation"]["reason"],
            "confirmatory_reserve_consumed": False,
            "questions": {
                "Q1": {
                    "question": QUESTIONS["Q1"],
                    "methods_meeting_the_meaningful_margin": [
                        m for m, c in criteria.items() if c["c1_opportunity_improvement"]
                    ],
                    "best_delta": best_delta.get("mean_delta_opportunity_recall"),
                },
                "Q2": {
                    "question": QUESTIONS["Q2"],
                    "primary_condition": {
                        m: {
                            "discovery": d["mean_delta_discovery_recall"],
                            "generation": d["mean_delta_generation_success"],
                        }
                        for m, d in deltas.items()
                    },
                    "oracle_localization_headroom": {
                        m: {
                            "localization": r["mean_localization_headroom"],
                            "generation": r["mean_generation_headroom"],
                        }
                        for m, r in decomposition.items()
                    },
                },
                "Q3": {
                    "question": QUESTIONS["Q3"],
                    "weak_types_improved": {
                        m: c["evidence"]["weak_types_improved"] for m, c in criteria.items()
                    },
                },
                "Q4": {
                    "question": QUESTIONS["Q4"],
                    "harmful_candidates_per_site": {
                        m: s["mean_harmful_candidates_per_site"] for m, s in summaries.items()
                    },
                    "survives_matched_budget": {
                        m: c["c4_candidate_burden"] for m, c in criteria.items()
                    },
                },
                "Q5": {
                    "question": QUESTIONS["Q5"],
                    "union_gain_over_best_single": union["complementarity_gain_over_best_single"],
                    "unique_repairs": {
                        m: r["unique_repairs"] for m, r in unique["by_method"].items()
                    },
                },
                "Q6": {
                    "question": QUESTIONS["Q6"],
                    "narrow_improvements": narrow,
                },
            },
            "statistical_support": {
                "primary_family": [t["comparison"] for t in primary_tests],
                "surviving_holm": [
                    t["comparison"] for t in primary_tests if t.get("significant_after_holm")
                ],
            },
            "controls_discriminating": controls["discriminating"],
            "controls_total": controls["total"],
            "falsification_tests_passed": negative["passed"],
            "falsification_tests_total": negative["total"],
            "reason": reason,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {letter} -- {OUTCOME_TAXONOMY[letter]}; {label}")
    return 0


# ------------------------------------------------------------------ section 51: figures

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# One validated categorical slot per correction family, kept in every figure: the reference
# palette's first three slots are the only ones that clear every gate for all pairs at once.
METHOD_COLOR = {M0: "#2a78d6", M1: "#eb6834", M2: "#1baf7a"}
METHOD_LABEL = {M0: "M0 frozen generator", M1: "M1 byte seq2seq", M2: "M2 local LLM"}
# Derived entities are not correction families, so they never borrow a family's hue.
DERIVED_GRAYS = ("#b4b3ad", "#8a8984", "#5f5e5a")
NOT_PROPOSED = "#e6e5e1"
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
FIGURE_NOTE = "analysis-only quantity computed with ground truth; not a deployable method"
REQUIRED_FIGURES = (
    "opportunity_by_method.png",
    "discovery_vs_generation.png",
    "opportunity_by_environment.png",
    "opportunity_by_engine.png",
    "opportunity_by_domain.png",
    "opportunity_by_error_type.png",
    "weak_error_type_recovery.png",
    "candidate_multiplicity_by_method.png",
    "harmful_candidates_by_method.png",
    "opportunity_vs_candidate_burden.png",
    "matched_k_opportunity.png",
    "natural_vs_oracle_localization.png",
    "method_unique_repairs.png",
    "method_overlap_matrix.png",
    "union_opportunity_ceiling.png",
    "near_miss_distance.png",
)


def _tint(color: str, share: float) -> str:
    """The color mixed toward white by ``share``: a lighter step of the same hue, never alpha."""
    channels = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * share):02x}" for c in channels)


def _short(name: str) -> str:
    return (
        name.replace("tesseract", "tess").replace("paddleocr", "paddle").replace("easyocr", "easy")
    )


def run_figures() -> int:
    """Section 51. The sixteen figures. Every plotted number is read from a saved artifact.

    Color follows the entity: each correction family keeps one validated slot in every figure,
    derived entities are neutral grays, and text is ink, never a series color. One axis per panel;
    where two measures share a figure they are separate panels. The aqua slot sits below 3:1
    against the surface, so bars carry direct value labels and the artifacts are the table view.
    """
    started = time.monotonic()
    _forbid(FIGURE_MANIFEST)
    plt = s15b._figure_style()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "grid.color": GRID,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
        }
    )
    design = cc_read_json(DESIGN_RECORD)
    opportunity = cc_read_json(OPPORTUNITY_RESULTS)
    engines = cc_read_json(ENGINE_ANALYSIS)
    domains = cc_read_json(DOMAIN_ANALYSIS)
    error_types = cc_read_json(ERROR_TYPE_ANALYSIS)
    multiplicity = cc_read_json(CANDIDATE_MULTIPLICITY)
    harmful = cc_read_json(HARMFUL_ANALYSIS)
    matched = cc_read_json(MATCHED_K_RESULTS)
    oracle = cc_read_json(ORACLE_LOCALIZATION)
    unique = cc_read_json(UNIQUE_REPAIRS)
    overlap = cc_read_json(METHOD_OVERLAP)
    union = cc_read_json(UNION_OPPORTUNITY)
    near = cc_read_json(NEAR_MISS_ANALYSIS)
    written: list[tuple[Path, tuple[Path, ...]]] = []

    baseline = float(design["baseline"]["opportunity_recall"])
    target = baseline + float(design["thresholds"]["meaningful_delta_opportunity"])
    conditions = [c for c in (COND_FROZEN, COND_NATURAL) if c in opportunity["conditions"]]
    summaries = {
        c: {
            s["method"]: s
            for s in opportunity["conditions"][c]["by_method"]
            if s.get("environments")
        }
        for c in conditions
    }
    methods = [m for m in ALL_METHODS if m in summaries[PRIMARY_CONDITION]]
    new = [m for m in methods if m != M0]
    labels = {**METHOD_LABEL}
    colors = {**METHOD_COLOR}

    def finish(figure: Any, name: str, sources: tuple[Path, ...]) -> None:
        figure.text(
            0.01, 0.003, FIGURE_NOTE, fontsize=5.5, color=INK_SECONDARY, ha="left", va="bottom"
        )
        written.append((s15b._finish(figure, FIGURE_DIR / name), sources))
        plt.close(figure)

    def reference(axis: Any, value: float, label: str) -> None:
        axis.axhline(value, color=INK_SECONDARY, linewidth=0.8, linestyle="--", zorder=1)
        axis.annotate(
            label,
            xy=(1.0, value),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6,
            color=INK_SECONDARY,
        )

    def grouped(
        axis: Any,
        groups: Sequence[str],
        series: dict[str, list[float]],
        *,
        fmt: str = "{:.3f}",
        value_labels: bool = True,
    ) -> None:
        width = 0.8 / max(len(series), 1)
        for position, (key, values) in enumerate(series.items()):
            offset = (position - (len(series) - 1) / 2) * width
            bars = axis.bar(
                np.arange(len(groups)) + offset,
                values,
                width=width,
                color=colors[key],
                edgecolor=SURFACE,
                linewidth=1.0,
                label=labels[key],
                zorder=2,
            )
            if not value_labels:
                continue
            for bar, value in zip(bars, values, strict=True):
                if value is None or not np.isfinite(value):
                    continue
                below = value < 0
                axis.annotate(
                    fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, -2 if below else 2),
                    textcoords="offset points",
                    ha="center",
                    va="top" if below else "bottom",
                    fontsize=5,
                    color=INK,
                )
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels(groups, fontsize=7)
        if len(series) > 1:
            axis.legend(fontsize=6, loc="upper right")

    def stratum(analysis: dict[str, Any], method: str, keys: Sequence[str]) -> list[float]:
        rows = {
            r["stratum"]: r["opportunity_recall"]
            for r in analysis["by_condition"][PRIMARY_CONDITION].get(method, [])
        }
        return [float(rows.get(k, float("nan"))) for k in keys]

    # 1. opportunity by method, per condition
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    grouped(
        axis,
        [
            c.replace("_", " ") + (" (primary)" if c == PRIMARY_CONDITION else "")
            for c in conditions
        ],
        {
            m: [
                float(summaries[c].get(m, {}).get("mean_opportunity_recall", float("nan")))
                for c in conditions
            ]
            for m in methods
        },
    )
    reference(axis, baseline, f"SGV-CG1 baseline {baseline:.4f}")
    reference(axis, target, f"meaningful gain target {target:.4f}")
    axis.set_ylabel("mean opportunity recall")
    axis.set_title("Exact-repair opportunity by correction family")
    axis.text(
        0.01,
        0.98,
        "M0 has no whole-text mode, so it has no natural-condition bar",
        transform=axis.transAxes,
        fontsize=5.5,
        color=INK_SECONDARY,
        va="top",
    )
    finish(figure, "opportunity_by_method.png", (OPPORTUNITY_RESULTS, DESIGN_RECORD))

    # 2. the section 44 decomposition: 1 = opportunity + generation loss + discovery loss
    rows = [(m, c) for c in conditions for m in methods if m in summaries[c]]
    figure, axis = plt.subplots(figsize=(6.8, 0.42 * len(rows) + 1.2))
    for index, (method, condition) in enumerate(rows):
        record = summaries[condition][method]
        parts = [
            float(record["mean_opportunity_recall"]),
            float(record["mean_generation_loss"]),
            float(record["mean_discovery_loss"]),
        ]
        shades = [colors[method], _tint(colors[method], 0.6), NOT_PROPOSED]
        left = 0.0
        for share, shade in zip(parts, shades, strict=True):
            axis.barh(
                index, share, left=left, color=shade, edgecolor=SURFACE, linewidth=1.0, height=0.7
            )
            if share >= 0.06:
                axis.text(
                    left + share / 2,
                    index,
                    f"{share:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=INK,
                )
            left += share
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([f"{labels[m]} / {c.replace('_', ' ')}" for m, c in rows], fontsize=6.5)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("share of OCR error sites")
    axis.set_title("Discovery against generation: where each family loses the error sites")
    axis.legend(
        handles=[
            Patch(color=INK_SECONDARY, label="exact repair available (full hue)"),
            Patch(
                color=_tint(INK_SECONDARY, 0.6), label="proposed at, no exact repair (light step)"
            ),
            Patch(color=NOT_PROPOSED, label="never proposed at"),
        ],
        fontsize=6,
        loc="lower right",
    )
    finish(figure, "discovery_vs_generation.png", (OPPORTUNITY_RESULTS,))

    # 3. by environment
    per_arm = opportunity["conditions"][PRIMARY_CONDITION]["per_arm"]
    names = [spec["environment"] for spec in s14.ENVIRONMENTS]
    lookup = {(r["environment"], r["method"]): float(r["opportunity_recall"]) for r in per_arm}
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    grouped(
        axis,
        [_short(n) for n in names],
        {m: [lookup.get((n, m), float("nan")) for n in names] for m in methods},
        value_labels=False,
    )
    axis.set_xticklabels([_short(n) for n in names], rotation=40, ha="right", fontsize=6)
    axis.set_ylabel("opportunity recall")
    axis.set_title(f"Opportunity by environment ({PRIMARY_CONDITION.replace('_', ' ')})")
    finish(figure, "opportunity_by_environment.png", (OPPORTUNITY_RESULTS,))

    # 4-6. by engine, by domain, by error type
    for analysis, keys_field, name, title, source in (
        (
            engines,
            "engines",
            "opportunity_by_engine.png",
            "Opportunity by OCR engine",
            ENGINE_ANALYSIS,
        ),
        (
            domains,
            "domains",
            "opportunity_by_domain.png",
            "Opportunity by document domain",
            DOMAIN_ANALYSIS,
        ),
        (
            error_types,
            "error_types",
            "opportunity_by_error_type.png",
            "Opportunity by frozen error type",
            ERROR_TYPE_ANALYSIS,
        ),
    ):
        keys = list(analysis[keys_field])
        figure, axis = plt.subplots(figsize=(6.6, 3.6))
        grouped(axis, keys, {m: stratum(analysis, m, keys) for m in methods})
        axis.set_ylabel("opportunity recall (pooled)")
        axis.set_title(f"{title} ({PRIMARY_CONDITION.replace('_', ' ')})")
        finish(figure, name, (source,))

    # 7. weak error-type recovery, as a delta against M0
    recovery = error_types["weak_type_recovery"]
    kinds = [*WEAK_ERROR_TYPES, STRONG_ERROR_TYPE]
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    grouped(
        axis,
        kinds,
        {
            m: [
                float(recovery.get(m, {}).get("delta_by_type", {}).get(k, float("nan")))
                for k in kinds
            ]
            for m in new
        },
        fmt="{:+.3f}",
    )
    axis.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
    reference(axis, WEAK_TYPE_DELTA, f"material improvement {WEAK_TYPE_DELTA}")
    axis.set_ylabel("opportunity recall minus M0's")
    axis.set_title("Recovery of the weak error types (section 24); insertion shown for context")
    finish(figure, "weak_error_type_recovery.png", (ERROR_TYPE_ANALYSIS,))

    # 8. candidate multiplicity: a dot plot, one marker shape per statistic
    rows = [
        (m, c) for c in conditions for m in methods if m in multiplicity["conditions"].get(c, {})
    ]
    figure, axis = plt.subplots(figsize=(6.6, 0.4 * len(rows) + 1.3))
    for index, (method, condition) in enumerate(rows):
        record = multiplicity["conditions"][condition][method]
        for statistic, marker in (("median", "o"), ("mean", "s"), ("p90", "^"), ("p95", "D")):
            axis.plot(
                float(record[statistic]),
                index,
                marker=marker,
                ms=4.5,
                color=colors[method],
                linestyle="none",
                zorder=3,
            )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([f"{labels[m]} / {c.replace('_', ' ')}" for m, c in rows], fontsize=6.5)
    axis.invert_yaxis()
    axis.set_xlabel("candidates per proposal site")
    axis.set_title("Candidate multiplicity by family")
    axis.legend(
        handles=[
            plt.Line2D([], [], marker=m, color=INK_SECONDARY, linestyle="none", ms=4.5, label=s)
            for s, m in (("median", "o"), ("mean", "s"), ("p90", "^"), ("p95", "D"))
        ],
        fontsize=6,
        loc="lower right",
    )
    finish(figure, "candidate_multiplicity_by_method.png", (CANDIDATE_MULTIPLICITY,))

    # 9. the harmful side: two measures, two panels, never two axes
    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    harm = {
        c: {s["method"]: s for s in harmful["conditions"][c]["by_method"] if s.get("environments")}
        for c in conditions
    }
    groups = [c.replace("_", " ") for c in conditions]
    grouped(
        left_axis,
        groups,
        {
            m: [
                float(harm[c].get(m, {}).get("mean_harmful_candidates_per_site", float("nan")))
                for c in conditions
            ]
            for m in methods
        },
        fmt="{:.2f}",
    )
    left_axis.set_ylabel("harmful candidates per proposal site")
    left_axis.set_title("Harmful candidates per site")
    grouped(
        right_axis,
        groups,
        {
            m: [
                float(harm[c].get(m, {}).get("pooled_harmful_share", float("nan")))
                for c in conditions
            ]
            for m in methods
        },
        fmt="{:.2f}",
    )
    right_axis.set_ylabel("harmful share of candidates (pooled)")
    right_axis.set_title("Harmful share of all candidates")
    finish(figure, "harmful_candidates_by_method.png", (HARMFUL_ANALYSIS,))

    # 10. opportunity against candidate burden, along each family's own K
    per_k = matched["conditions"][PRIMARY_CONDITION]["per_k"]
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for method in methods:
        points = sorted(
            {
                (
                    float(r["mean_candidates_per_site"]),
                    float(r["mean_opportunity_recall"]),
                    int(r["k"]),
                )
                for r in per_k
                if r["method"] == method
            },
            key=lambda item: item[2],
        )
        xs = [x for x, _y, _k in points]
        ys = [y for _x, y, _k in points]
        axis.plot(
            xs,
            ys,
            color=colors[method],
            linewidth=1.0,
            marker="o",
            ms=4,
            label=labels[method],
            zorder=3,
        )
        axis.annotate(
            f"K={points[-1][2]}",
            (xs[-1], ys[-1]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=5.5,
            color=INK,
        )
    reference(axis, baseline, f"SGV-CG1 baseline {baseline:.4f}")
    axis.set_xlabel("mean candidates per proposal site")
    axis.set_ylabel("mean opportunity recall")
    axis.set_title("Opportunity against candidate burden (matched budget)")
    axis.legend(fontsize=6, loc="lower right")
    finish(figure, "opportunity_vs_candidate_burden.png", (MATCHED_K_RESULTS,))

    # 11. matched K
    supports = matched["conditions"][PRIMARY_CONDITION]["supports_multi_candidate"]
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    ks = list(K_GRID)
    for method in methods:
        by_k = {
            int(r["k"]): float(r["mean_opportunity_recall"]) for r in per_k if r["method"] == method
        }
        if supports.get(method, True):
            axis.plot(
                range(len(ks)),
                [by_k.get(k, float("nan")) for k in ks],
                color=colors[method],
                linewidth=1.0,
                marker="o",
                ms=4,
                label=labels[method],
                zorder=3,
            )
        else:
            axis.plot(
                [0],
                [by_k.get(ks[0], float("nan"))],
                color=colors[method],
                marker="o",
                ms=5,
                linestyle="none",
                label=f"{labels[method]} (one candidate per site)",
                zorder=3,
            )
    reference(axis, baseline, f"SGV-CG1 baseline {baseline:.4f}")
    reference(axis, target, f"meaningful gain target {target:.4f}")
    axis.set_xticks(range(len(ks)))
    axis.set_xticklabels([str(k) for k in ks])
    axis.set_xlabel("K, candidates kept per site in the family's own order")
    axis.set_ylabel("mean opportunity recall")
    axis.set_title(f"Matched candidate budget (primary K = {PRIMARY_K})")
    axis.legend(fontsize=6, loc="center right")
    finish(figure, "matched_k_opportunity.png", (MATCHED_K_RESULTS, DESIGN_RECORD))

    # 12. natural, frozen and oracle localization
    decomposition = oracle["three_level_decomposition"]
    levels = ("natural", "frozen_localization", "oracle_localization")
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    grouped(
        axis,
        [level.replace("_", " ") for level in levels],
        {
            m: [float(decomposition[m][f"mean_{level}"]) for level in levels]
            for m in methods
            if m in decomposition
        },
    )
    axis.set_ylabel("mean opportunity recall")
    axis.set_title("What the true error location would add (section 46)")
    axis.text(
        0.01,
        0.98,
        "M0's natural condition is its frozen localization",
        transform=axis.transAxes,
        fontsize=5.5,
        color=INK_SECONDARY,
        va="top",
    )
    finish(figure, "natural_vs_oracle_localization.png", (ORACLE_LOCALIZATION,))

    # 13. unique repairs by error type
    kinds_all = list(error_types["error_types"])
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    grouped(
        axis,
        kinds_all,
        {
            m: [
                float(
                    unique["by_method"].get(m, {}).get("unique_repairs_by_error_type", {}).get(k, 0)
                )
                for k in kinds_all
            ]
            for m in methods
        },
        fmt="{:.0f}",
    )
    axis.set_ylabel("OCR error sites only this family can repair")
    axis.set_title(f"Unique repairs by error type ({PRIMARY_CONDITION.replace('_', ' ')})")
    finish(figure, "method_unique_repairs.png", (UNIQUE_REPAIRS, ERROR_TYPE_ANALYSIS))

    # 14. the site overlap matrix: magnitude, so one sequential hue
    matrix = overlap["site_overlap_matrix"]
    order = [m for m in methods if m in matrix]
    shared = np.array([[float(matrix[a][b]) for b in order] for a in order])
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    image = axis.imshow(
        shared, cmap=LinearSegmentedColormap.from_list("xc1_blue", SEQUENTIAL_BLUE), vmin=0
    )
    top = float(shared.max()) if shared.size else 0.0
    for i in range(len(order)):
        for j in range(len(order)):
            axis.text(
                j,
                i,
                f"{int(shared[i, j])}",
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE if shared[i, j] > 0.55 * top else INK,
            )
    axis.set_xticks(range(len(order)))
    axis.set_yticks(range(len(order)))
    axis.set_xticklabels([labels[m] for m in order], rotation=20, ha="right", fontsize=6.5)
    axis.set_yticklabels([labels[m] for m in order], fontsize=6.5)
    axis.grid(False)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label(
        "OCR error sites both can repair", fontsize=6.5
    )
    axis.set_title("Site overlap: shared exact repairs (diagonal: own repairs)")
    finish(figure, "method_overlap_matrix.png", (METHOD_OVERLAP,))

    # 15. the union ceiling
    singles = union["single_method_mean_opportunity_recall"]
    bars = [
        *[(labels[m], float(singles[m]), colors[m]) for m in methods],
        (
            "union, new families",
            float(union["union_new_methods_only"]["mean_opportunity_recall"]),
            DERIVED_GRAYS[0],
        ),
        (
            "union, all families",
            float(union["union_all_methods"]["mean_opportunity_recall"]),
            DERIVED_GRAYS[1],
        ),
        (
            "union, every GT-blind condition",
            float(union["union_every_gt_blind_condition"]["mean_opportunity_recall"]),
            DERIVED_GRAYS[2],
        ),
    ]
    figure, axis = plt.subplots(figsize=(7.0, 3.7))
    for index, (_label, value, shade) in enumerate(bars):
        axis.bar(index, value, color=shade, edgecolor=SURFACE, linewidth=1.0, width=0.7, zorder=2)
        axis.annotate(
            f"{value:.4f}",
            (index, value),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=INK,
        )
    reference(axis, baseline, f"SGV-CG1 baseline {baseline:.4f}")
    reference(axis, target, f"meaningful gain target {target:.4f}")
    axis.set_xticks(range(len(bars)))
    axis.set_xticklabels([label for label, _v, _c in bars], rotation=20, ha="right", fontsize=6.5)
    axis.set_ylabel("mean opportunity recall")
    axis.set_title("The union ceiling: every family's candidates, selected perfectly")
    finish(figure, "union_opportunity_ceiling.png", (UNION_OPPORTUNITY, DESIGN_RECORD))

    # 16. near-miss distance: one panel per family
    histograms = near["conditions"][PRIMARY_CONDITION]
    shown = [m for m in methods if m in histograms]
    figure, axes = plt.subplots(1, len(shown), figsize=(2.9 * len(shown), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, method in zip(axes, shown, strict=True):
        histogram = histograms[method]["d_after_histogram"]
        bins = list(histogram)
        counts = np.array([float(histogram[b]) for b in bins])
        shares = counts / counts.sum() if counts.sum() else counts
        axis.bar(
            range(len(bins)),
            shares,
            color=colors[method],
            edgecolor=SURFACE,
            linewidth=1.0,
            width=0.8,
            zorder=2,
        )
        axis.set_xticks(range(len(bins)))
        axis.set_xticklabels(bins, fontsize=5.5)
        axis.set_title(labels[method], fontsize=7)
        axis.set_xlabel("characters from ground truth (d_after)", fontsize=6)
    axes[0].set_ylabel("share of candidates at error regions")
    figure.suptitle(
        "Near-miss distance: how far non-exact candidates land (secondary only)", fontsize=8
    )
    finish(figure, "near_miss_distance.png", (NEAR_MISS_ANALYSIS,))

    missing = sorted(set(REQUIRED_FIGURES) - {path.name for path, _s in written})
    if missing:
        raise PhaseError(f"required figures not written: {missing}")
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_envelope("figure_manifest"),
            "synthetic": False,
            "scientific_status": "DEVELOPMENT / DIAGNOSTIC",
            "oracle_figures_are_labelled": (
                "every opportunity plotted here is an analysis-only quantity computed with ground "
                "truth, and each figure says so in its footer. None is a deployable method."
            ),
            "palette": {
                "families": METHOD_COLOR,
                "derived_entities": list(DERIVED_GRAYS),
                "sequential": list(SEQUENTIAL_BLUE),
                "rule": (
                    "color follows the entity in every figure; derived entities are neutral; text "
                    "is ink; the aqua slot is relieved by direct labels and the artifact tables"
                ),
            },
            "figures": {
                _relative(path): {
                    "sha256": file_sha256(path),
                    "derived_from": [_relative(source) for source in sources],
                    "source_sha256": {_relative(source): file_sha256(source) for source in sources},
                }
                for path, sources in written
            },
            "figure_count": len(written),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} written -> {_relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ section 50: inventories


def run_inventory() -> int:
    """Section 50's inventories: the environments, and candidates and sites by method."""
    started = time.monotonic()
    for path in (ENVIRONMENT_INVENTORY, CANDIDATE_INVENTORY, SITE_INVENTORY):
        _forbid(path)
    errors, arms = load_arms()
    normalization = cc_read_json(NORMALIZATION_RESULTS)
    candidates = pd.read_parquet(METHOD_CANDIDATES)
    alignment = pd.read_parquet(ALIGNMENT_SITES)

    environments: list[dict[str, Any]] = []
    m0_generated = 0
    for spec in s14.ENVIRONMENTS:
        name = spec["environment"]
        slug = _slug(name)
        population = errors[name]
        counts = pd.read_parquet(CACHE / f"{slug}.requests.parquet", columns=["condition"])[
            "condition"
        ].value_counts()
        oracle_path = CACHE / f"{slug}.oracle_requests.parquet"
        block = alignment[alignment["environment"] == name]
        m0_generated += len(
            pd.read_parquet(CACHE / f"{slug}.m0_candidates.parquet", columns=["candidate_id"])
        )
        environments.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "language": language_of(spec["corpus"]),
                "alignment_sites": len(block),
                "alignment_sites_evaluation": int((block["role"] == s14.ROLE_EVALUATION).sum()),
                "ocr_error_sites": population.count,
                "ocr_error_characters": population.characters,
                "evaluation_documents": int(population.documents.size),
                "frozen_localization_requests": int(counts.get(COND_FROZEN, 0)),
                "natural_line_requests": int(counts.get(COND_NATURAL, 0)),
                "oracle_localization_requests": (
                    len(pd.read_parquet(oracle_path, columns=["request_id"]))
                    if oracle_path.is_file()
                    else None
                ),
                "in_declared_language_scope": {
                    method: method_applies(method, spec["corpus"]) for method in PRIMARY_METHODS
                },
            }
        )
    documents = sorted({d for population in errors.values() for d in population.documents.tolist()})
    totals = {
        "environments": len(environments),
        "alignment_sites": int(sum(row["alignment_sites"] for row in environments)),
        "ocr_error_sites": int(sum(row["ocr_error_sites"] for row in environments)),
        "ocr_error_characters": int(sum(row["ocr_error_characters"] for row in environments)),
        "distinct_evaluation_documents": len(documents),
        "evaluation_document_environment_pairs": int(
            sum(row["evaluation_documents"] for row in environments)
        ),
        "frozen_localization_requests": int(
            sum(row["frozen_localization_requests"] for row in environments)
        ),
        "natural_line_requests": int(sum(row["natural_line_requests"] for row in environments)),
        "oracle_localization_requests": int(
            sum(row["oracle_localization_requests"] or 0 for row in environments)
        ),
        "corpora": sorted({row["corpus"] for row in environments}),
        "engines": sorted({row["base_engine"] for row in environments}),
    }
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            "environments": environments,
            "totals": totals,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    census = pd.DataFrame([row for row in normalization["per_arm"] if row.get("condition")])
    by_arm: list[dict[str, Any]] = []
    for (method, condition), group in candidates.groupby(["method", "condition"], sort=True):
        evaluation = group[group["role"] == s14.ROLE_EVALUATION]
        if method == M0:
            raw = proposals = m0_generated
            declined = duplicates = removed = None
        else:
            shard = census[(census["method"] == method) & (census["condition"] == condition)]
            raw = int(shard["raw_sequences"].sum())
            proposals = int(shard["proposals"].sum())
            declined = int(shard["declined_unchanged"].sum())
            duplicates = int(shard["duplicates_removed"].sum())
            removed = int(
                shard[
                    ["dropped_unprojectable", "dropped_outside_any_span", "dropped_whitespace_only"]
                ]
                .fillna(0)
                .to_numpy()
                .sum()
            )
        by_arm.append(
            {
                "method": str(method),
                "condition": str(condition),
                "raw_generated": raw,
                "declined_unchanged": declined,
                "duplicates_removed": duplicates,
                "removed_by_the_diff": removed,
                "normalized_proposals": proposals,
                "labelled_candidates": len(group),
                "labelled_candidates_evaluation": len(evaluation),
                "unlabelable_removed": proposals - len(group),
                "by_outcome": {
                    k: int(v) for k, v in evaluation["outcome"].value_counts().sort_index().items()
                },
                "by_edit_kind": {
                    k: int(v)
                    for k, v in evaluation["edit_kind"].value_counts().sort_index().items()
                },
                "by_anchor_kind": {
                    k: int(v)
                    for k, v in evaluation["anchor_kind"].value_counts().sort_index().items()
                },
                "exact": int(evaluation["exact"].sum()),
                "beneficial": int(evaluation["beneficial"].sum()),
                "harmful": int(evaluation["is_harmful"].sum()),
            }
        )
    _write_json_once(
        CANDIDATE_INVENTORY,
        {
            **_envelope("candidate_inventory_by_method"),
            "by_method_condition": by_arm,
            "note": (
                "section 10: raw generated, normalized, duplicates removed and invalid removed, "
                "per "
                "arm. M0's raw count is its generated candidate stream on both splits; the neural "
                "arms were generated on the evaluation split only. Labelling removes nothing it "
                "can label -- 'unlabelable' is the frozen RESOLVED gate, never an outcome."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    kinds = sorted({kind for population in errors.values() for kind in population.kind})
    sites: list[dict[str, Any]] = []
    for (name, method, condition), arm in sorted(arms.items()):
        sites.append(
            {
                "environment": name,
                "method": method,
                "condition": condition,
                "proposal_sites": arm.n_sites,
                "proposal_sites_by_anchor_kind": {
                    str(kind): int(np.unique(arm.site_of[arm.anchor_kind == kind]).size)
                    for kind in sorted(set(arm.anchor_kind.tolist()))
                },
                "discovered_error_sites": int(arm.covered.sum()),
                "discovered_error_sites_by_error_type": {
                    kind: int((arm.covered & (arm.errors.kind == kind)).sum()) for kind in kinds
                },
                "repairable_sites": int(arm.repaired().sum()),
            }
        )
    frame = pd.DataFrame(sites)
    _write_json_once(
        SITE_INVENTORY,
        {
            **_oracle_envelope("site_inventory_by_method"),
            "per_arm": sites,
            "by_method_condition": [
                {
                    "method": str(method),
                    "condition": str(condition),
                    "proposal_sites": int(group["proposal_sites"].sum()),
                    "discovered_error_sites": int(group["discovered_error_sites"].sum()),
                    "repairable_sites": int(group["repairable_sites"].sum()),
                }
                for (method, condition), group in frame.groupby(["method", "condition"], sort=True)
            ],
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"inventory: {totals['environments']} environments, {totals['ocr_error_sites']} OCR error "
        f"sites, {len(by_arm)} arm-conditions"
    )
    return 0


# ------------------------------------------------------------------ section 54: determinism

DETERMINISM_RUNS = 3
DETERMINISM_ENVIRONMENTS = ("funsd/doctr", "sbb/doctr")
MODEL_SPOT_ENVIRONMENT = "funsd/doctr"
MODEL_SPOT_BATCHES = 2
# A figure manifest records the file hash of every source artifact, and every artifact carries its
# issue time, so those embedded hashes change on every rerun by construction. Each source's
# timestamp-free content is compared in its own right, so nothing is lost by leaving them out here.
VOLATILE_KEYS = frozenset({"issued_utc", "runtime_seconds", "issued_head", "head", "source_sha256"})
DERIVED_PHASES = (
    "inventory",
    "discovery",
    "generation",
    "opportunity",
    "matchedk",
    "harmful",
    "strata",
    "oracleanalysis",
    "overlap",
    "nearmiss",
    "shift",
    "cost",
    "controls",
    "negative",
    "stats",
    "figures",
)
PHASE_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "inventory": (ENVIRONMENT_INVENTORY, CANDIDATE_INVENTORY, SITE_INVENTORY),
    "discovery": (DISCOVERY_RESULTS,),
    "generation": (GENERATION_RESULTS,),
    "opportunity": (OPPORTUNITY_RESULTS,),
    "matchedk": (MATCHED_K_RESULTS,),
    "harmful": (HARMFUL_ANALYSIS, CANDIDATE_MULTIPLICITY),
    "strata": (ERROR_TYPE_ANALYSIS, ENGINE_ANALYSIS, DOMAIN_ANALYSIS),
    "oracleanalysis": (ORACLE_LOCALIZATION, CANDIDATE_ORACLE, RISK_CANDIDATE_ORACLE),
    "overlap": (METHOD_OVERLAP, UNIQUE_REPAIRS, UNION_OPPORTUNITY),
    "nearmiss": (NEAR_MISS_ANALYSIS,),
    "shift": (CANDIDATE_DISTRIBUTION_SHIFT,),
    "cost": (RUNTIME_COST,),
    "controls": (CONTROL_RESULTS,),
    "negative": (NEGATIVE_TESTS,),
    "stats": (STATISTICAL_TESTS,),
    "figures": (FIGURE_MANIFEST,),
}
TABLE_KEYS = {
    "METHOD_CANDIDATES": ["candidate_id"],
    "AUXILIARY_CANDIDATES": ["candidate_id"],
    "ORACLE_CANDIDATES": ["candidate_id"],
    "METHOD_LINKS": ["environment", "site_id", "align_site_id", "link_kind", "method", "condition"],
    "AUXILIARY_LINKS": [
        "environment",
        "site_id",
        "align_site_id",
        "link_kind",
        "method",
        "condition",
    ],
    "ORACLE_LINKS": ["environment", "site_id", "align_site_id", "link_kind", "method", "condition"],
    "ALIGNMENT_SITES": ["align_site_id"],
}


def _stable_view(value: Any) -> Any:
    """An artifact with its clock and head fields removed -- the part a rerun must reproduce."""
    if isinstance(value, dict):
        return {k: _stable_view(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_stable_view(v) for v in value]
    return value


def _content_hash(path: Path) -> str:
    import hashlib
    import json

    text = json.dumps(_stable_view(cc_read_json(path)), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _derived_signature() -> dict[str, str]:
    signature = {
        _relative(path): _content_hash(path)
        for name in DERIVED_PHASES
        for path in PHASE_ARTIFACTS[name]
    }
    signature.update({_relative(f): file_sha256(f) for f in sorted(FIGURE_DIR.glob("*.png"))})
    return signature


def _frame_equal(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str]) -> bool:
    a = left.sort_values(list(keys), kind="stable").reset_index(drop=True)
    b = right.sort_values(list(keys), kind="stable").reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=False)
    except AssertionError:
        return False
    return True


def _sandboxed(run: Callable[[], int], directory: Path, names: Sequence[str]) -> None:
    """Run a table-writing phase on the determinism environments, into a sandbox directory.

    The module's output paths and SGV14's environment list are swapped for the duration of the call
    and always restored. The phase code itself is unchanged, which is what makes the regenerated
    tables a test of the code that produced the primary ones.
    """
    saved = {name: globals()[name] for name in names}
    environments = s14.ENVIRONMENTS
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for name in names:
            globals()[name] = directory / saved[name].name
        s14.ENVIRONMENTS = tuple(
            spec for spec in environments if spec["environment"] in DETERMINISM_ENVIRONMENTS
        )
        run()
    finally:
        for name, value in saved.items():
            globals()[name] = value
        s14.ENVIRONMENTS = environments


def _table_regeneration() -> dict[str, Any]:
    """Normalization and oracle labels, regenerated from the immutable raw caches and compared."""
    import shutil

    groups = {
        "normalize": (
            run_normalize,
            (
                "METHOD_CANDIDATES",
                "METHOD_LINKS",
                "AUXILIARY_CANDIDATES",
                "AUXILIARY_LINKS",
                "ALIGNMENT_SITES",
                "NORMALIZATION_RESULTS",
            ),
        ),
        "oracles": (run_oracles, ("ORACLE_CANDIDATES", "ORACLE_LINKS", "ORACLE_NORMALIZATION")),
    }
    results: list[dict[str, Any]] = []
    for regeneration in range(2):
        for label, (run, names) in groups.items():
            directory = CACHE / "determinism" / f"{label}_{regeneration}"
            shutil.rmtree(directory, ignore_errors=True)
            _sandboxed(run, directory, names)
            for name in names:
                if name not in TABLE_KEYS:
                    continue
                primary = pd.read_parquet(globals()[name])
                primary = primary[primary["environment"].isin(DETERMINISM_ENVIRONMENTS)]
                regenerated = pd.read_parquet(directory / globals()[name].name)
                results.append(
                    {
                        "regeneration": regeneration,
                        "phase": label,
                        "table": globals()[name].name,
                        "rows": len(regenerated),
                        "identical": _frame_equal(regenerated, primary, TABLE_KEYS[name]),
                    }
                )
            shutil.rmtree(directory, ignore_errors=True)
    return {
        "environments": list(DETERMINISM_ENVIRONMENTS),
        "regenerations": 2,
        "tables": results,
        "identical": all(row["identical"] for row in results),
    }


def _input_rebuild() -> dict[str, Any]:
    """The GT-blind inputs, rebuilt from the raw OCR for the determinism environments."""
    pipeline = s14.build_pipeline()
    rows: list[dict[str, Any]] = []
    for name in DETERMINISM_ENVIRONMENTS:
        spec = next(s for s in s14.ENVIRONMENTS if s["environment"] == name)
        slug = _slug(name)
        span_table, requests, m0_candidates = _environment_inputs(spec, pipeline)
        rows.append(
            {
                "environment": name,
                "spans_identical": _frame_equal(
                    span_table,
                    pd.read_parquet(CACHE / f"{slug}.spans.parquet"),
                    ["document_id", "engine_id", "span_id"],
                ),
                "requests_identical": _frame_equal(
                    requests, pd.read_parquet(CACHE / f"{slug}.requests.parquet"), ["request_id"]
                ),
                "m0_candidates_identical": _frame_equal(
                    m0_candidates,
                    pd.read_parquet(CACHE / f"{slug}.m0_candidates.parquet"),
                    ["candidate_id"],
                ),
                "requests": len(requests),
                "m0_candidates": len(m0_candidates),
            }
        )
    return {
        "rebuilt_from": "the raw OCR store, through the same builder --prepare uses",
        "environments": rows,
        "identical": all(
            row["spans_identical"] and row["requests_identical"] and row["m0_candidates_identical"]
            for row in rows
        ),
    }


def _model_regeneration() -> dict[str, Any]:
    """Each neural arm re-run on its first batches of one environment, in its own frozen batching.

    A bf16 decode is reproducible for fixed batching and not invariant to it, so the check selects
    exactly the requests of the arm's first batches -- request order for the byte model,
    prompt-length order for the LLM -- and compares every returned sequence with the immutable
    cache.
    """
    registry = cc_read_json(METHOD_REGISTRY)
    versions = cc_read_json(MODEL_VERSIONS)
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = _resolve_device()
    slug = _slug(MODEL_SPOT_ENVIRONMENT)
    out: dict[str, Any] = {}
    for method in (M1, M2):
        if not registry["methods"][method]["available"]:
            continue
        snapshot = Path(versions["models"][method]["snapshot"])
        runner: Any = (
            _Byt5Runner(snapshot, device) if method == M1 else _LlmRunner(snapshot, device)
        )
        checks: list[dict[str, Any]] = []
        for condition, directory, stem in (
            (COND_FROZEN, RAW_CACHE, "requests"),
            (COND_NATURAL, RAW_CACHE, "requests"),
            (COND_ORACLE_LOC, RAW_ORACLE_CACHE, "oracle_requests"),
        ):
            raw_path = directory / f"{slug}.{method}.{condition}.parquet"
            if not raw_path.is_file():
                continue
            requests = pd.read_parquet(CACHE / f"{slug}.{stem}.parquet")
            if condition != COND_ORACLE_LOC:
                requests = requests[requests["condition"] == condition]
            requests = requests.reset_index(drop=True)
            if method == M2:
                size = LLM_BATCH
                prompts = [
                    _prompt_for(method, condition, row) for row in requests.itertuples(index=False)
                ]
                ids = requests["request_id"].astype(str).tolist()
                order = sorted(range(len(requests)), key=lambda i: (len(prompts[i]), ids[i]))
            else:
                size = LINE_BATCH if condition == COND_NATURAL else GENERATION_BATCH
                order = list(range(len(requests)))
            subset = requests.iloc[order[: size * MODEL_SPOT_BATCHES]]
            regenerated, _cost = _run_arm(runner, subset, method, condition, lambda _m: None)
            cached = pd.read_parquet(raw_path)
            cached = cached[cached["request_id"].isin(set(subset["request_id"].astype(str)))]
            key = ["request_id", "rank"]
            merged = regenerated[[*key, "output"]].merge(
                cached[[*key, "output"]], on=key, how="outer", suffixes=("_regenerated", "_cached")
            )
            differing = int((merged["output_regenerated"] != merged["output_cached"]).sum())
            checks.append(
                {
                    "environment": MODEL_SPOT_ENVIRONMENT,
                    "condition": condition,
                    "requests": len(subset),
                    "sequences": len(merged),
                    "differing_sequences": differing,
                    "identical": differing == 0,
                }
            )
        out[method] = {
            "checks": checks,
            "agrees": bool(checks) and all(c["identical"] for c in checks),
        }
        del runner
    return out


def run_determinism() -> int:
    """Section 54. Independent regenerations of everything the decision reads, persisted by code.

    Four checks, each a comparison that could fail: the derived analysis regenerated twice more from
    the frozen tables; normalization and oracle labelling regenerated twice from the immutable raw
    caches; the GT-blind inputs rebuilt from the raw OCR; and each neural arm re-run on a sample of
    its own cached requests. Repeated model calls are never presented as analysis nondeterminism --
    the model check is reported on its own line, and the analysis is regenerated from the cache.
    """
    started = time.monotonic()
    _forbid(DETERMINISM)
    lookup = dict(PHASES)
    signatures: list[dict[str, str]] = []
    for run in range(DETERMINISM_RUNS):
        if run:
            for name in DERIVED_PHASES:
                for path in PHASE_ARTIFACTS[name]:
                    path.unlink(missing_ok=True)
            for figure in FIGURE_DIR.glob("*.png"):
                figure.unlink()
            for name in DERIVED_PHASES:
                lookup[name]()
        signatures.append(_derived_signature())
    derived_identical = all(signature == signatures[0] for signature in signatures)
    differing = sorted(
        {
            key
            for signature in signatures[1:]
            for key in signature
            if signature.get(key) != signatures[0].get(key)
        }
    )

    tables = _table_regeneration()
    rebuild = _input_rebuild()
    models = _model_regeneration()
    analysis_identical = derived_identical and tables["identical"] and rebuild["identical"]
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "runs": DETERMINISM_RUNS,
            "derived_regenerations": DETERMINISM_RUNS - 1,
            "derived_phases": list(DERIVED_PHASES),
            "derived_artifacts_compared": len(signatures[0]),
            "derived_signature_hashes": [canonical_hash(signature) for signature in signatures],
            "derived_runs_identical": derived_identical,
            "differing_artifacts": differing,
            "table_regeneration": tables,
            "input_rebuild": rebuild,
            "model_regeneration": models,
            "all_runs_identical": analysis_identical,
            "compared": [
                "raw outputs, by regenerating each neural arm's first batches",
                "normalized edits and candidate inventories, regenerated from the raw caches",
                "site linking, regenerated with the tables",
                "the GT-blind inputs, rebuilt from the raw OCR",
                "opportunity metrics and every other derived analysis, regenerated twice",
                "figures, by content hash",
                "the decision, re-derived twice by --record",
            ],
            "persisted_by": "the regeneration phase itself; this artifact is never hand-written",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if not analysis_identical:
        raise PhaseError(f"determinism failed: {differing or 'a regenerated table differs'}")
    print(
        f"determinism: {DETERMINISM_RUNS - 1} derived regenerations agree over "
        f"{len(signatures[0])} artifacts; tables regenerated identically; inputs rebuilt "
        f"identically; model regeneration "
        + ", ".join(f"{m} {'agrees' if r['agrees'] else 'DIFFERS'}" for m, r in models.items())
    )
    return 0


# ------------------------------------------------------------------ section 53: the record

PRODUCED: tuple[Path, ...] = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    DESIGN_RECORD,
    ATOMIC_EDIT_SCHEMA,
    LLM_FORMAT_AUDIT,
    METHOD_REGISTRY,
    MODEL_VERSIONS,
    PROMPT_REGISTRY,
    DECODING_REGISTRY,
    ALIGNMENT_SITES,
    METHOD_CANDIDATES,
    METHOD_LINKS,
    AUXILIARY_CANDIDATES,
    AUXILIARY_LINKS,
    NORMALIZATION_RESULTS,
    UPSTREAM_REPRODUCTION,
    ENVIRONMENT_INVENTORY,
    CANDIDATE_INVENTORY,
    SITE_INVENTORY,
    DISCOVERY_RESULTS,
    GENERATION_RESULTS,
    OPPORTUNITY_RESULTS,
    MATCHED_K_RESULTS,
    HARMFUL_ANALYSIS,
    CANDIDATE_MULTIPLICITY,
    ERROR_TYPE_ANALYSIS,
    ENGINE_ANALYSIS,
    DOMAIN_ANALYSIS,
    ORACLE_CANDIDATES,
    ORACLE_LINKS,
    ORACLE_NORMALIZATION,
    ORACLE_LOCALIZATION,
    CANDIDATE_ORACLE,
    RISK_CANDIDATE_ORACLE,
    METHOD_OVERLAP,
    UNIQUE_REPAIRS,
    UNION_OPPORTUNITY,
    NEAR_MISS_ANALYSIS,
    CANDIDATE_DISTRIBUTION_SHIFT,
    RUNTIME_COST,
    CONTROL_RESULTS,
    NEGATIVE_TESTS,
    STATISTICAL_TESTS,
    FIGURE_MANIFEST,
    DETERMINISM,
    DECISION,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE,),
    "2. Frozen SGV-CG1 Finding": (RESEARCH_FREEZE, UPSTREAM_REPRODUCTION),
    "3. Research Questions": (DESIGN_RECORD,),
    "4. Non-Goals": (DESIGN_RECORD,),
    "5. Correction Methods": (
        FROZEN_CONFIGURATION,
        METHOD_REGISTRY,
        MODEL_VERSIONS,
        PROMPT_REGISTRY,
        DECODING_REGISTRY,
        LLM_FORMAT_AUDIT,
    ),
    "6. Common Atomic Edit Representation": (ATOMIC_EDIT_SCHEMA, NORMALIZATION_RESULTS),
    "7. Ground-Truth Isolation": (NEGATIVE_TESTS, DESIGN_RECORD),
    "8. Site Discovery": (DISCOVERY_RESULTS, ENVIRONMENT_INVENTORY, SITE_INVENTORY),
    "9. Correction Generation": (GENERATION_RESULTS,),
    "10. Candidate Normalization": (NORMALIZATION_RESULTS, CANDIDATE_INVENTORY),
    "11. Candidate Budget Fairness": (MATCHED_K_RESULTS, DECODING_REGISTRY),
    "12. Overall Opportunity": (OPPORTUNITY_RESULTS,),
    "13. Discovery vs Generation Decomposition": (
        OPPORTUNITY_RESULTS,
        DISCOVERY_RESULTS,
        GENERATION_RESULTS,
    ),
    "14. Error-Type Opportunity": (ERROR_TYPE_ANALYSIS,),
    "15. Engine Analysis": (ENGINE_ANALYSIS,),
    "16. Domain Analysis": (DOMAIN_ANALYSIS,),
    "17. Candidate Multiplicity": (CANDIDATE_MULTIPLICITY,),
    "18. Harmful Candidate Burden": (HARMFUL_ANALYSIS,),
    "19. Matched-K Analysis": (MATCHED_K_RESULTS,),
    "20. Oracle Localization": (ORACLE_LOCALIZATION,),
    "21. Candidate Oracle Ceilings": (CANDIDATE_ORACLE, RISK_CANDIDATE_ORACLE),
    "22. Pairwise Complementarity": (METHOD_OVERLAP, UNIQUE_REPAIRS),
    "23. Union Ceiling": (UNION_OPPORTUNITY,),
    "24. Near-Miss Analysis": (NEAR_MISS_ANALYSIS,),
    "25. Runtime / Generation Cost": (RUNTIME_COST,),
    "26. Controls": (CONTROL_RESULTS,),
    "27. Falsification Tests": (NEGATIVE_TESTS,),
    "28. Statistics": (STATISTICAL_TESTS,),
    "29. Limitations": (
        METHOD_REGISTRY,
        DECODING_REGISTRY,
        LLM_FORMAT_AUDIT,
        NORMALIZATION_RESULTS,
    ),
    "30. Research Decision": (DECISION,),
    "31. Implication for Reliability Transfer": (CANDIDATE_DISTRIBUTION_SHIFT, DECISION),
    "32. Next Stage": (DECISION,),
}


def run_record() -> int:
    """Provenance, the dependency audit, the decision re-derivation and the traceability index."""
    started = time.monotonic()
    for path in (PROVENANCE, TRACEABILITY):
        _forbid(path)
    missing = [p for p in PRODUCED if not p.exists() and p not in (PROVENANCE, TRACEABILITY)]
    if missing:
        raise PhaseError(f"{len(missing)} artifacts missing: {[_relative(p) for p in missing][:5]}")
    determinism = cc_read_json(DETERMINISM)
    if not determinism["all_runs_identical"]:
        raise PhaseError("determinism did not pass; the record will not be issued")

    # Section 54 lists the decision among the things to regenerate. It is a pure function of the
    # artifacts determinism already verified, so it is re-derived twice here and compared.
    rederived: list[bool] = []
    for run in range(2):
        sandbox = CACHE / "record" / f"decision_{run}.json"
        sandbox.parent.mkdir(parents=True, exist_ok=True)
        sandbox.unlink(missing_ok=True)
        saved = globals()["DECISION"]
        try:
            globals()["DECISION"] = sandbox
            run_decide()
        finally:
            globals()["DECISION"] = saved
        rederived.append(
            _stable_view(cc_read_json(sandbox)) == _stable_view(cc_read_json(DECISION))
        )
        sandbox.unlink(missing_ok=True)
    if not all(rederived):
        raise PhaseError("the research decision does not re-derive identically")

    freeze = cc_read_json(RESEARCH_FREEZE)
    cg1_changed = sorted(
        path
        for path, digest in freeze["sgv_cg1_artifacts"].items()
        if file_sha256(REPO / path) != digest
    )
    scripts_changed = sorted(
        name
        for name, digest in freeze["upstream_scripts"].items()
        if file_sha256(REPO / name) != digest
    )
    states_now = {
        "sgv15": cg1._decision_state(s15.DECISION, ("verdict", "criteria_met", "ready_for_sgv16")),
        "sgv15b": cg1._decision_state(
            s15b.DECISION, ("verdict", "criteria_met", "ready_for_sgv16")
        ),
        "sgv_dt1": cg1._decision_state(
            dt1.DECISION, ("verdict", "criteria_met", "ready_for_external_confirmation")
        ),
    }
    decisions_unchanged = all(
        list(state)
        == [
            freeze["upstream_decisions"][name][key]
            for key in ("verdict", "criteria_met", "criteria_total", "outcome", "onward_gate")
        ]
        for name, state in states_now.items()
    )
    existing = [path for path in PRODUCED if path.exists()]
    raw_outputs = sorted(RAW_CACHE.glob("*.parquet")) + sorted(RAW_ORACLE_CACHE.glob("*.parquet"))
    caches = sorted(CACHE.glob("*.parquet"))
    manifest = cc_read_json(FIGURE_MANIFEST)
    decision = cc_read_json(DECISION)
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "issued_head": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
            "artifacts": {_relative(path): file_sha256(path) for path in existing},
            "artifact_count": len(existing),
            "figures": {
                name: record["sha256"] for name, record in sorted(manifest["figures"].items())
            },
            "raw_model_outputs": {_relative(path): file_sha256(path) for path in raw_outputs},
            "raw_model_output_count": len(raw_outputs),
            "input_caches": {_relative(path): file_sha256(path) for path in caches},
            "documents": {
                _relative(REPORT): file_sha256(REPORT) if REPORT.is_file() else None,
                _relative(NORMALIZATION_DOC): (
                    file_sha256(NORMALIZATION_DOC) if NORMALIZATION_DOC.is_file() else None
                ),
            },
            "upstream_inputs": {**freeze["sgv_cg1_artifacts"], **freeze["upstream_scripts"]},
            "upstream_unchanged_since_section_zero": {
                "sgv_cg1_artifacts_changed": cg1_changed,
                "upstream_scripts_changed": scripts_changed,
                "upstream_decisions_unchanged": decisions_unchanged,
            },
            "decision_rederived_identically": rederived,
            "table_rows": {
                _relative(path): len(
                    pd.read_parquet(path, columns=[pd.read_parquet(path).columns[0]])
                )
                for path in existing
                if path.suffix == ".parquet"
            },
            "dependency_audit": {
                "artifacts_tracked_by_git": [_relative(p) for p in existing if _tracked(p)],
                "artifacts_not_git_ignored": [_relative(p) for p in existing if not _ignored(p)],
                "raw_data_written": False,
                "upstream_artifacts_written": bool(cg1_changed or scripts_changed),
                "confirmatory_reserve_consumed": False,
                "external_confirmation_run": False,
                "correction_models_fine_tuned": 0,
                "external_api_calls": 0,
                "ground_truth_used_for": [
                    "outcome labelling",
                    "exact repair determination",
                    "oracle localization inputs (location only, never content)",
                    "oracle and ceiling analysis",
                    "final evaluation",
                ],
                "correction_families_introduced": [
                    m for m in decision["methods_evaluated"] if m != M0
                ],
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        TRACEABILITY,
        {
            **_envelope("traceability"),
            "report": _relative(REPORT),
            "rule": (
                "every number in the report is read from one of the artifacts its section names, "
                "and from the field that means what the sentence says. A number that cannot be "
                "traced does not belong in the report."
            ),
            "audited_classes": [
                "integers",
                "grouped integers",
                "decimals",
                "percentages",
                "effect sizes",
                "confidence intervals",
                "p-values",
            ],
            "sections": {
                name: [_relative(path) for path in paths] for name, paths in REPORT_SECTIONS.items()
            },
        },
    )
    if cg1_changed or scripts_changed or not decisions_unchanged:
        raise PhaseError("an upstream artifact changed since section 0; see provenance.json")
    print(
        f"record: {len(existing)} artifacts, {len(raw_outputs)} raw output files, "
        f"{len(REPORT_SECTIONS)} report sections -> {_relative(PROVENANCE)}"
    )
    return 0


# ------------------------------------------------------------------ the phase table


PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("preregister", run_preregister),
    ("schema", run_schema),
    ("prepare", run_prepare),
    ("llmaudit", run_llmaudit),
    ("registry", run_registry),
    ("generate", run_generate),
    ("oracleprep", run_oracleprep),
    ("oraclegenerate", run_oracle_generate),
    ("normalize", run_normalize),
    ("reproduce", run_reproduce),
    ("inventory", run_inventory),
    ("discovery", run_discovery),
    ("generation", run_generation),
    ("opportunity", run_opportunity),
    ("matchedk", run_matchedk),
    ("harmful", run_harmful),
    ("strata", run_strata),
    ("oracles", run_oracles),
    ("oracleanalysis", run_oracle_analysis),
    ("overlap", run_overlap),
    ("nearmiss", run_nearmiss),
    ("shift", run_shift),
    ("cost", run_cost),
    ("controls", run_controls),
    ("negative", run_negative),
    ("stats", run_stats),
    ("figures", run_figures),
    ("determinism", run_determinism),
    ("decide", run_decide),
    ("record", run_record),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    for name, _ in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--all", action="store_true", help="run every phase in order")
    args = parser.parse_args(list(argv) if argv is not None else None)

    selected = [name for name, _ in PHASES if getattr(args, name)]
    if args.all:
        selected = [name for name, _ in PHASES]
    if not selected:
        parser.print_help()
        return 2
    lookup = dict(PHASES)
    for name in selected:
        began = time.monotonic()
        try:
            status = lookup[name]()
        except PhaseError as error:
            print(f"{name}: FAILED -- {error}", file=sys.stderr)
            return 1
        if status:
            return status
        print(f"  [{name} {time.monotonic() - began:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

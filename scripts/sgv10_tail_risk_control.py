#!/usr/bin/env python3
"""SGV10: is the deployment boundary the remaining bottleneck after SGV1-SGV9?

SGV9 closed by naming what was left. Not representation, not adaptation, not policy selection,
not the deployment decision, and not absolute calibration: what remained was the *variability of
the risk estimate across engines and across blocks of pages*, and the closing sentence said a
stage that attacked it would have to change either what is estimated or what is deployed on,
"not how the probability is transformed."

This stage changes what is deployed on, and it does so with a degree of freedom SGV9 did not
have. Every SGV9 arm was a monotone map of the harm probability, so by construction it could move
the cut and nothing else -- which is why EasyOCR, whose achievable frontier is 0.009 because the
frozen ranking puts harmful candidates at the very top, was declared unreachable there. A
secondary model that reads features and vetoes individual candidates is not a monotone map of the
score. It can reorder. That is the whole mechanism of SGV10 and it is the only reason this stage
can say anything SGV9 could not.

Where the opportunity is, read off SGV9's own published table before this stage was designed
(`results/generated/sgv9/calibrated_scores.parquet`, evaluation blocks, at the deepest prefix
whose realised harm is within 0.10):

    engine      cut rank      realised harm   beneficial in the block   repair recall
    docTR        603/2348           0.0995                55                   0.636
    EasyOCR         5/4146           0.0000               564                   0.009
    PaddleOCR     403/1813           0.0993                64                   0.641
    Tesseract    1974/3917           0.0998              1227                   0.738

On EasyOCR the frozen ranking puts harmful candidates at the very top, so the deepest prefix that
holds the bound is five rows deep and the frontier is 0.009. A veto cannot be a monotone map of
the score -- a monotone map preserves that ranking by construction -- but a secondary model can
reorder, and the counts above say there is a great deal to reorder. Every number in the table is
recomputable from `results/generated/sgv9/calibrated_scores.parquet`, whose hash SGV9's provenance
manifest carries and which this stage records among its own inputs. That is the claim:

    SGV10-T1: a secondary veto model, fitted on the SOURCE engines' boundary rows and applied
    only inside a boundary band of the frozen SGV5 ranking, raises repair recall on an unseen
    OCR engine while the REALISED harm of what it accepts stays within epsilon -- and it does so
    by removing harmful candidates from the accept region rather than by shrinking it.

**The hypothesis is recorded as SGV10-T1, not H1.** `docs/sgv1/protocol.md` binds SGV1-H1 through
SGV1-H4 and forbids reusing a frozen identifier for a different claim; every stage since has
opened its own family. The brief's H1, H2 and H3 are carried as sub-claims T1a (recall improves at
a fixed harm bound), T1b (the improvement comes from boundary harm removal, not from a better
global ranking) and T1c (abstention is selective -- useful corrections are preserved), all under
T1.

Twelve decisions fix what the numbers below can mean. All twelve were made before any SGV10
endpoint was computed; four of them were informed by SGV9's PUBLISHED artifacts, which is stated
here and in `design_record.json` rather than left to be inferred. `design_record.json` carries all
twelve with the prediction each implies.

**1. Nothing upstream moves, and the check is bit-exact against two stages.** SGV5's model, its
features, its candidate pool and its decision score `benefit - lambda * harm` are rebuilt and
compared to SGV5's published score vector; the rebuilt blocks are then compared row by row to
SGV9's published `calibrated_scores.parquet`. A drift in either is a hard failure. SGV9's fold
construction, block construction, stratification, prefix curves, deployed point, harm bound and
resampling engine are IMPORTED from `sgv10`'s predecessor rather than restated, so "the two stages
read the same rows" is checkable rather than aspirational.

**2. The boundary is a band of the DECISION score, not of the harm probability, and both are
reported.** The brief defines the boundary by `|P(harm) - tau| < delta`. SGV9's limitation 6
recorded that the frozen decision score is nearly binary -- lambda = 10 on three folds, the harm
head saturates, and `boundary_analysis.json` records how few distinct utility values carry each
block -- so a band in
harm-probability space and a band in decision space are not the same set. The primary band is a
RANK half-width of the block on which it is fitted, converted to a pair of decision-score bounds
that are carried to the target exactly as a threshold is carried. The brief's literal harm-scale
band is fitted, applied and reported beside it, and negative test 3 sweeps both.

**3. The veto acts only inside the band. That is what makes this a boundary method.** Outside the
band the accept order is SGV5's, unchanged. Inside it, the order is the veto model's. The
composite score is built so that every above-band row outranks every in-band row and every
in-band row outranks every below-band row, and the invariant is asserted on every block of every
fold. Ablation C removes the confinement and lets the veto reorder everything, which is a
different -- and much less falsifiable -- experiment; it is reported as an ablation, never as the
method.

**4. The veto is fitted on the source CALIBRATION rows, not the source fit rows.** The veto reads
the frozen model's own outputs as features. On `fold.fit` those outputs are in-sample for a
boosted model and are far better than they will ever be again, so a veto trained there would learn
that `p_harm` is trustworthy and would be wrong the moment it is deployed. `fold.source_cal` is
where SGV5's isotonic calibrators were fitted and where the frozen model's outputs are honest.
Its model class is chosen by an inner leave-one-source-engine-out replay inside those same rows;
`selection_scope` records it.

**5. Every veto feature is computed within a site or within a page, never across the block.** The
veto sees SGV5's representation, SGV5's two calibrated heads, and aggregates over the other
candidates proposed at the same site and on the same page -- how many there are, where this one
ranks among them, how far it is from the best sibling. No statistic crosses a document boundary,
so a page's features are identical whether it is scored alone or inside a batch of 400, and
`tests/leakage` asserts exactly that by scoring one page both ways. The scaler is fitted on the
source rows and frozen, as `.claude/rules/experiment-leakage.md` requires of any fitted
transformer.

**6. The cut is placed on the unlabelled target pool, never on the evaluation block.** SGV9's
decision 3, unchanged and for the same reason: threshold selection is named in the leakage rules
and there is no exception for reading only the target's scores. The band, the veto ordering and
the cut are all fitted on the held-out engine's TRAIN rows; the resulting pair of band bounds and
the resulting threshold meet the DEVELOPMENT rows once, to be measured.

**7. Repair recall bought by breaking the bound is not credited.** An arm that reorders can pass
the frozen ranking's achievable frontier legitimately, which is the point of the stage -- so
unlike SGV9 this stage cannot use "exceeds the frontier" as a violation flag. It uses the thing
the flag was standing in for: criterion 1 credits a cell only when repair recall improves AND the
REALISED harm rate of the accepted set is within epsilon. Each arm's own achievable frontier is
reported beside it, and so is the frontier of the frozen ranking, so a reader can see which of the
two an arm beat.

**8. CVaR is taken over DOCUMENTS, because a candidate's harm is Bernoulli.** `CVaR_alpha` of a
single Bernoulli outcome is 1 whenever alpha exceeds `1 - p`; the quantity carries no information
at the candidate level. The resampling unit in this project is the document, and the failure the
brief describes -- rare, concentrated harmful corrections -- is a page-level failure. Method D
therefore bounds the mean of the worst `1 - alpha` share of PAGES by their accepted candidates'
harm rate. The CVaR cut searches a rank grid rather than every rank, which is stated as a
limitation and is the only approximation in the cut rules.

**9. Method C bounds; it does not sharpen.** The transfer inflation is SGV9's construction --
each source engine takes a turn as a pseudo-target, the whole pipeline is refitted from the
remaining two, and the inflation is the MAXIMUM of the three errors, which is the only choice
attaining 3/4 distribution-free coverage with three exchangeable calibration points. The inner
folds are SGV9's, reused deliberately rather than re-salted, so the two stages' inflations are
measured on identical splits and are directly comparable. This stage states 3/4 and does not
state `1 - alpha`.

**10. Four references, and three of them are ceilings.** The frozen ranking's achievable frontier
(what SGV9 lived under), each arm's own achievable frontier, a veto fitted on the target engine's
own TRAIN labels (`oracle_pool`), and a veto fitted on the evaluation block itself
(`oracle_eval`). The perfect hindsight accept set -- every beneficial candidate, no harmful one --
is recall 1.000 at harm 0.000 on every fold by definition and is reported once, as arithmetic
rather than as a ceiling anyone could approach.

**11. Criterion 4 is enforced as a floor, not as a sentence.** "Does not win only by rejecting
almost all corrections" is read as: an arm earns credit in a cell only if it accepts at least half
as many candidates as SGV5's published deployment does, and it must beat matched-coverage random
abstention. Both bars were fixed before any endpoint was computed.

**12. The comparison that isolates the mechanism is reported beside the one the brief asks for.**
Criterion 1 compares against SGV5's published deployment, which is what the brief says. But that
comparison confounds two changes -- a new cut rule and a new veto -- so the identity arm under
SGV10's own cut rule is carried as a full arm (ablation A), exactly as SGV9 carried it, and every
table shows both.

    --boundary     rebuild the frozen model, fit the bands and the vetoes, write the per-row table
    --analysis     what the boundary actually is on each engine: size, harm, transfer
    --tail         tail harm among accepted, at 1%, 5% and 10%
    --coverage     the risk-coverage table: every arm, every cut rule, every epsilon
    --abstention   what abstention removes and what it costs
    --oracle       the gap to the two oracles and to both frontiers
    --calibration  boundary calibration against global calibration, and against SGV9
    --ablation     the brief's six ablations
    --negative     the five negative tests the brief requires
    --figures      the required figures
    --decide       the machine-readable finding
    --record       provenance for every artifact

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; no band, no veto, no threshold and no
selection reads a label from the held-out engine's DEVELOPMENT rows.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
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
import sgv5_candidate_reliability as c5
import sgv6_target_engine_adaptation as s6
import sgv9_target_calibration as s9
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.calibration import calibration_report
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.metrics.selective import aurc, risk_coverage_curve
from ocr_risk.risk.controller import select_threshold

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv10"

SCORES = OUT / "boundary_scores.parquet"
DESIGN_RECORD = OUT / "design_record.json"
BOUNDARY_ANALYSIS = OUT / "boundary_analysis.json"
TAIL_RISK_RESULTS = OUT / "tail_risk_results.json"
RISK_COVERAGE = OUT / "risk_coverage_table.json"
ABSTENTION_ANALYSIS = OUT / "abstention_analysis.json"
ORACLE_GAP = OUT / "oracle_gap.json"
CALIBRATION_COMPARISON = OUT / "calibration_comparison.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
DECISION = OUT / "research_decision.json"
PROVENANCE = OUT / "provenance_manifest.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# The endpoint, its frontier, its interval and the whole fold machinery have one implementation
# in this repository. Ten stages now import them rather than restating them, and
# `tests/leakage/test_sgv10_tail_risk_control.py` asserts the identity of every alias below.
achievable_repair_recall = rl.achievable_repair_recall
deployed_point = c5._deployed
build_fold = rl.build_fold
inner_fold = s9.inner_fold
Block = s9.Block
prefix_curves = s9.prefix_curves
harm_upper_bound = s9.harm_upper_bound
resample_curves = s9.resample_curves
resampled_deployment = s9.resampled_deployment
document_resamples = s9.document_resamples
_deployed_at = s9._deployed_at
_deepest = s9._deepest
_interval = s9._interval
_stable_seed = s6._stable_seed
_logit = s6._logit

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
PRIMARY_KEY = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = pilot.BOOTSTRAP_SEED
CROSS_ENGINE = s6.CROSS_ENGINE
PROB_FLOOR = s6.PROB_FLOOR
N_BINS = s9.N_BINS
CONFORMAL_COVERAGE = s9.CONFORMAL_COVERAGE
CONFORMAL_ALPHA = s9.CONFORMAL_ALPHA


class PhaseError(RuntimeError):
    """A freeze, band, veto, role, split or selection invariant failed."""


# ------------------------------------------------------------------ the arm registry

SGV5_PUBLISHED = "b0_sgv5_published"
RULE_ONLY = "aA_rule_only"
GLOBAL_CALIBRATION = "aB_global_calibration"
BOUNDARY_VETO = "mA_boundary_veto"
TAIL_CONDITIONAL = "mB_tail_conditional"
HYBRID = "mE_hybrid"
NO_BOUNDARY = "aC_no_boundary"
RANDOM_ABSTENTION = "aD_random_abstention"
BOUNDARY_FEATURES = "aE_boundary_features"
ORACLE_POOL = "oracle_pool"
ORACLE_EVAL = "oracle_eval"

PRIMARY_ARM = HYBRID

# The pre-specified family. Criterion 2 counts engines within it and nothing outside it is
# eligible to be the finding; the multiplicity family in `--coverage` is exactly this tuple
# crossed with the four engines.
PROPOSED = (BOUNDARY_VETO, TAIL_CONDITIONAL, HYBRID)
BASELINES = (SGV5_PUBLISHED, RULE_ONLY, GLOBAL_CALIBRATION)
CONTROLS = (NO_BOUNDARY, RANDOM_ABSTENTION, BOUNDARY_FEATURES)
ORACLES = (ORACLE_POOL, ORACLE_EVAL)
ALL_ARMS = (*BASELINES, *PROPOSED, *CONTROLS, *ORACLES)

# Which arms reorder inside the band, and with whose veto. An arm absent from this table keeps
# SGV5's order everywhere, which is what makes "the score changed" and "the estimate changed"
# separable in every table below.
VETO_OF = {
    BOUNDARY_VETO: "source",
    HYBRID: "source",
    NO_BOUNDARY: "source_unconfined",
    RANDOM_ABSTENTION: "random",
    BOUNDARY_FEATURES: "boundary_only",
    ORACLE_POOL: "pool_labels",
    ORACLE_EVAL: "evaluation_labels",
}

# Which arms replace SGV5's harm probability with a boundary-local one. `aB_global_calibration`
# replaces it with SGV9's primary map, which is the whole point of that ablation.
ESTIMATE_OF = {
    SGV5_PUBLISHED: "frozen",
    RULE_ONLY: "frozen",
    GLOBAL_CALIBRATION: "sgv9_quantile_transport",
    BOUNDARY_VETO: "frozen",
    TAIL_CONDITIONAL: "boundary_local",
    HYBRID: "boundary_local",
    NO_BOUNDARY: "boundary_local",
    RANDOM_ABSTENTION: "boundary_local",
    BOUNDARY_FEATURES: "boundary_local",
    ORACLE_POOL: "pool_labels",
    ORACLE_EVAL: "evaluation_labels",
}

ARM_NOTES = {
    SGV5_PUBLISHED: (
        "Baseline 0. SGV5's frozen deployment, unmodified: its decision score, its harm "
        "probability and the threshold its own calibration rows chose. Reported under the "
        "source-threshold rule only, because that is the rule it was published with."
    ),
    RULE_ONLY: (
        "Ablation A. SGV5's decision score and SGV5's harm probability, under SGV10's cut "
        "rules. Isolates what the rule change buys or costs before any veto exists; a gain "
        "visible here is not a gain from tail-risk control."
    ),
    GLOBAL_CALIBRATION: (
        "Ablation B. SGV5's decision score with SGV9's primary calibration map (quantile risk "
        "transport) supplying the harm estimate, under SGV10's cut rules. This is the direct "
        "comparison against the previous stage that the brief asks for."
    ),
    BOUNDARY_VETO: (
        "Method A. A secondary classifier fitted on the SOURCE engines' boundary rows reorders "
        "the band; the harm estimate stays SGV5's. Isolates reordering from re-estimation."
    ),
    TAIL_CONDITIONAL: (
        "Method B. SGV5's order everywhere, with the harm estimate inside the band replaced by "
        "an isotonic fit of the realised harm on the SOURCE band rows. Isolates re-estimation "
        "from reordering, and is the arm the boundary-versus-global calibration comparison is "
        "read from."
    ),
    HYBRID: (
        "Method E, the main proposed framework: SGV5's ranking outside the band, the veto "
        "model's ranking inside it, and the boundary-local harm estimate placing the cut."
    ),
    NO_BOUNDARY: (
        "Ablation C. The same veto, unconfined -- it reorders every row, not just the band. "
        "Removes the tail-risk component's defining restriction, which makes it a re-ranking "
        "of the whole pool rather than a boundary method."
    ),
    RANDOM_ABSTENTION: (
        "Ablation D. The band is reordered by a content-derived pseudo-random permutation "
        "instead of by the veto. Matched band, matched coverage, no information: any gain the "
        "proposed arms show over this one is a gain from the veto and not from abstaining."
    ),
    BOUNDARY_FEATURES: (
        "Ablation E. The veto refitted on the boundary-local features alone -- the frozen "
        "model's own outputs and the site and page aggregates -- with SGV5's representation "
        "columns removed. Tests whether the global features are necessary."
    ),
    ORACLE_POOL: (
        "Ablation F, first form. An ORACLE: the veto and the harm estimate are fitted on the "
        "held-out engine's own TRAIN labels. The ceiling of what target labels could buy without "
        "touching an evaluation page. Not a method; excluded from every selection."
    ),
    ORACLE_EVAL: (
        "Ablation F, second form. An ORACLE: the veto and the harm estimate are fitted on the "
        "evaluation block itself. The ceiling of the veto class outright. Not a method; excluded "
        "from every selection."
    ),
}


# ------------------------------------------------------------------ the pre-registration

PRE_REGISTRATION = {
    "claim": (
        "SGV10-T1: a secondary veto model, fitted on the SOURCE engines' boundary rows and "
        "applied only inside a boundary band of the frozen SGV5 ranking, raises repair recall "
        "on an unseen OCR engine while the REALISED harm rate of what it accepts stays within "
        "epsilon -- and it does so by removing harmful candidates from the accept region "
        "rather than by shrinking it."
    ),
    "sub_claims": {
        "T1a": "repair recall improves at a fixed harm bound (the brief's H1)",
        "T1b": (
            "the improvement comes from removing harmful candidates near the boundary, not "
            "from a better global ranking (the brief's H2)"
        ),
        "T1c": (
            "abstention is selective: useful corrections are preserved rather than traded "
            "away wholesale (the brief's H3)"
        ),
    },
    "primary_arm": PRIMARY_ARM,
    "primary_endpoint": "repair recall at the deployed cut, with realised harm within epsilon",
    "criteria": {
        "1": (
            "repair recall improves on SGV5's published deployment AND the realised harm rate "
            "of the arm's accepted set is within epsilon"
        ),
        "2": "criterion 1 holds on at least three of the four unseen engines",
        "3": (
            "tail harm decreases: the harm rate of the marginal 1%, 5% and 10% of the accepted "
            "set falls against SGV5's published deployment"
        ),
        "4": (
            "the arm accepts at least half as many candidates as SGV5's published deployment "
            "and beats matched-coverage random abstention"
        ),
        "5": "boundary calibration error improves against the frozen harm probability",
    },
    "verdict_rule": (
        "SUPPORTED only if all five criteria are met by the primary arm; otherwise NOT "
        "SUPPORTED, with the criteria that failed named"
    ),
}

DECISIONS = [
    {
        "id": 1,
        "decision": "Nothing upstream moves, and the check is bit-exact against two stages.",
        "prediction": (
            "the rebuilt SGV5 decision score equals the published vector exactly on all four "
            "folds, and the rebuilt blocks equal SGV9's published per-row table exactly"
        ),
        "pre_registered": True,
    },
    {
        "id": 2,
        "decision": (
            "The boundary is a band of the DECISION score, carried as two bounds; the brief's "
            "literal harm-probability band is fitted and reported beside it."
        ),
        "prediction": (
            "the two bands select different rows, because SGV9's limitation 6 recorded that "
            "the frozen decision score is nearly binary"
        ),
        "pre_registered": True,
        "informed_by": "SGV9 limitation 6, published",
    },
    {
        "id": 3,
        "decision": "The veto acts only inside the band; ablation C removes the confinement.",
        "prediction": (
            "the unconfined veto changes more rows and is not obviously safer, because it is a "
            "re-ranking of the whole pool rather than a boundary method"
        ),
        "pre_registered": True,
    },
    {
        "id": 4,
        "decision": (
            "The veto is fitted on the source CALIBRATION rows, where the frozen model's own "
            "outputs are out of sample, and its model class is chosen by an inner "
            "leave-one-source-engine-out replay inside those rows."
        ),
        "prediction": "the inner selection is stable across folds",
        "pre_registered": True,
    },
    {
        "id": 5,
        "decision": (
            "Every veto feature is computed within a site or within a page; no statistic "
            "crosses a document boundary."
        ),
        "prediction": (
            "a page's feature vector is identical whether it is scored alone or inside a batch"
        ),
        "pre_registered": True,
    },
    {
        "id": 6,
        "decision": (
            "The cut is placed on the unlabelled target pool, never on the evaluation block."
        ),
        "prediction": (
            "the pool and the evaluation block differ in harm prevalence, and that difference "
            "is a floor on what any pool-fitted quantity can achieve"
        ),
        "pre_registered": True,
        "informed_by": "SGV9 decision 3, published",
    },
    {
        "id": 7,
        "decision": (
            "Repair recall bought by breaking the bound is not credited: criterion 1 requires "
            "the realised harm rate of the accepted set to be within epsilon."
        ),
        "prediction": (
            "some arms exceed the frozen ranking's achievable frontier legitimately, because "
            "reordering is the mechanism this stage adds"
        ),
        "pre_registered": True,
    },
    {
        "id": 8,
        "decision": (
            "CVaR is taken over DOCUMENTS, unweighted, because a candidate's harm is Bernoulli."
        ),
        "prediction": (
            "the CVaR rule cuts shallower than the mean rule at the same epsilon, because it "
            "is bounding a tail rather than an average"
        ),
        "pre_registered": True,
    },
    {
        "id": 9,
        "decision": (
            "Method C bounds and does not sharpen; the inflation is the maximum of three "
            "leave-one-source-engine-out replays on SGV9's inner folds."
        ),
        "prediction": (
            "the inflation is of the same order as SGV9's, because it prices the same "
            "engine-to-engine variability"
        ),
        "pre_registered": True,
        "informed_by": "SGV9 decisions 5 and 6, published",
    },
    {
        "id": 10,
        "decision": (
            "Four references: the frozen ranking's frontier, each arm's own frontier, a veto "
            "fitted on the target's TRAIN labels, and a veto fitted on the evaluation block."
        ),
        "prediction": "the two oracles bracket what any target-free veto can reach",
        "pre_registered": True,
    },
    {
        "id": 11,
        "decision": (
            "Criterion 4 is a floor: half of SGV5's accepted count, and a win over "
            "matched-coverage random abstention."
        ),
        "prediction": "an arm that wins by refusing is caught by the floor rather than by prose",
        "pre_registered": True,
    },
    {
        "id": 12,
        "decision": (
            "The identity arm under SGV10's own cut rule is carried as ablation A, so a gain "
            "from the rule change is never read as a gain from tail-risk control."
        ),
        "prediction": ("the rule change alone moves the deployed point, as it did in SGV9"),
        "pre_registered": True,
        "informed_by": "SGV9 decision 4, published",
    },
]

# ------------------------------------------------------------------ the cut rules

SOURCE_THRESHOLD = "source_threshold"
POINT_CUT = "predicted_harm_point"
BOUND_CUT = "predicted_harm_bound"
CVAR_ALPHAS = (0.90, 0.95, 0.99)
CVAR_RULES = tuple(f"cvar_{int(a * 100)}" for a in CVAR_ALPHAS)
RULES = (SOURCE_THRESHOLD, POINT_CUT, BOUND_CUT, *CVAR_RULES)

# The recall endpoint is read under the point rule; the safety claim is read under the bound
# rule and nowhere else. Both were fixed before any endpoint was computed.
PRIMARY_RULE = POINT_CUT
SAFETY_RULE = BOUND_CUT

RULE_NOTES = {
    SOURCE_THRESHOLD: (
        "The threshold SGV5's own source calibration rows chose for this epsilon, carried to "
        "the unseen engine. Reads no target row at all."
    ),
    POINT_CUT: (
        "The deepest prefix of the arm's own ordering whose mean CALIBRATED harm estimate over "
        "the unlabelled target pool is within epsilon. A point estimate, not a bound."
    ),
    BOUND_CUT: (
        "Method C. The deepest prefix whose calibrated estimate, plus the transfer inflation "
        "measured by a leave-one-source-engine-out replay, plus a one-sided normal allowance "
        "at the prefix's effective DOCUMENT count, is within epsilon."
    ),
    **{
        f"cvar_{int(a * 100)}": (
            f"Method D at alpha = {a}. The deepest prefix whose conditional value at risk over "
            "PAGES -- the mean estimated harm rate of the worst "
            f"{round((1 - a) * 100)}% of accepted pages -- is within epsilon. Searched on "
            "a rank grid, not at every rank."
        )
        for a in CVAR_ALPHAS
    },
}


# ------------------------------------------------------------------ the band

# A rank half-width, as a share of the block the band is fitted on. 0.10 was fixed before any
# SGV10 endpoint was computed; negative test 3 sweeps it and reports the primary endpoint under
# each width, so the choice is visible rather than load-bearing.
BAND_FRACTION = 0.10
BAND_SWEEP = (0.05, 0.10, 0.25)
# The brief's literal band, on the harm-probability scale.
HARM_DELTA = 0.05
HARM_DELTA_SWEEP = (0.02, 0.05, 0.10)
# A band that carries fewer rows than this cannot support a fitted veto, and a fold where that
# happens is recorded rather than silently fitted on a handful of rows.
MIN_BAND_ROWS = 100

RANK_BAND = "rank"
HARM_BAND = "harm_probability"
BAND_KINDS = (RANK_BAND, HARM_BAND)
PRIMARY_BAND = RANK_BAND

TAIL_SHARES = (0.01, 0.05, 0.10)
COVERAGE_FLOOR = 0.50
CVAR_GRID = 256
VETO_KINDS = ("logistic", "boosted")
VETO_SEED = 20260904


@dataclass(frozen=True, slots=True)
class Band:
    """A pair of decision-score bounds, frozen on one block and carried to another.

    A band is transferred exactly the way a threshold is transferred: as two scalars. It is
    NOT re-derived on the block it is applied to, because re-deriving it there would let the
    target's own rank distribution decide which rows are marginal, and the whole question is
    whether a boundary fitted where labels exist still lands on the boundary where they do not.
    How wide the carried band turns out to be on the target is measured, in
    `boundary_analysis.json`, and is one of this stage's findings rather than one of its inputs.
    """

    kind: str
    lower: float
    upper: float
    centre: float
    fitted_on: str
    half_width_rows: int
    fitted_rows: int
    note: str

    def contains(self, block: Block) -> np.ndarray:
        value = block.utility if self.kind == RANK_BAND else block.p_harm
        return (value >= self.lower) & (value <= self.upper)


def fit_rank_band(block: Block, tau: float, fraction: float, fitted_on: str) -> Band:
    """A band of ranks around the cut, expressed as two decision-score bounds.

    The half-width is a share of the block, so a band is the same *fraction* of the ranking on
    every fold rather than the same number of rows -- the four folds differ in size by a factor
    of two. The centre is the cut itself; the bounds are the decision scores at the two ends of
    the rank window, so the band survives the ties that fill this score instead of splitting a
    tie group in half. How few distinct values there are is recorded per block in
    `boundary_analysis.json` as `distinct_decision_scores`.
    """
    order = np.argsort(-block.utility, kind="stable")
    ordered = block.utility[order]
    rank = int(np.count_nonzero(block.utility >= tau)) if np.isfinite(tau) else block.size
    rank = int(np.clip(rank, 0, block.size))
    half = round(fraction * block.size)
    lo_rank = int(max(rank - half, 0))
    hi_rank = int(min(rank + half, block.size))
    if hi_rank <= lo_rank:
        return Band(
            kind=RANK_BAND,
            lower=float("inf"),
            upper=float("-inf"),
            centre=float(tau),
            fitted_on=fitted_on,
            half_width_rows=half,
            fitted_rows=0,
            note="the rank window is empty on the block it was fitted on",
        )
    upper = float(ordered[lo_rank])
    lower = float(ordered[hi_rank - 1])
    return Band(
        kind=RANK_BAND,
        lower=lower,
        upper=upper,
        centre=float(tau),
        fitted_on=fitted_on,
        half_width_rows=half,
        fitted_rows=int(np.count_nonzero((block.utility >= lower) & (block.utility <= upper))),
        note=(
            f"ranks [{lo_rank}, {hi_rank}) of {block.size} around the cut at rank {rank}, "
            "carried as two decision-score bounds"
        ),
    )


def fit_harm_band(block: Block, tau: float, delta: float, fitted_on: str) -> Band:
    """The brief's literal band: `|P(harm) - tau_h| < delta`, on the harm-probability scale.

    `tau_h` is the harm probability of the marginal candidate -- the last one the cut accepts --
    which is what the brief's `tau` has to mean once the ordering is by `benefit - lambda *
    harm` rather than by harm alone.
    """
    accepted = np.isfinite(block.utility) & (block.utility >= tau)
    centre = float(block.p_harm[accepted].min()) if accepted.any() else float(block.p_harm.max())
    lower, upper = max(centre - delta, 0.0), min(centre + delta, 1.0)
    return Band(
        kind=HARM_BAND,
        lower=lower,
        upper=upper,
        centre=centre,
        fitted_on=fitted_on,
        half_width_rows=-1,
        fitted_rows=int(np.count_nonzero((block.p_harm >= lower) & (block.p_harm <= upper))),
        note=(
            f"|p_harm - {centre:.6f}| <= {delta}, where the centre is the harm probability of "
            "the marginal accepted candidate"
        ),
    )


# ------------------------------------------------------------------ the veto's features

# Every one of these is computed within a site or within a page. None pools across the block, so
# a page's feature vector does not depend on which other pages are in the batch -- which is what
# makes the target-side features honest, and is asserted directly in the leakage suite.
LOCAL_FEATURES = (
    "loc_p_harm",
    "loc_p_benefit",
    "loc_logit_p_harm",
    "loc_utility",
    "loc_site_candidates",
    "loc_site_harm_rank",
    "loc_site_harm_min",
    "loc_site_harm_spread",
    "loc_site_utility_gap",
    "loc_site_is_best",
    "loc_page_candidates",
    "loc_page_mean_harm",
    "loc_page_median_harm",
    "loc_page_harm_spread",
    "loc_page_mean_utility",
    "loc_page_utility_rank",
)


def _group_stats(keys: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    """Per-group count, mean, min, spread and within-group rank, without a Python loop.

    `pd.factorize` gives dense group codes; `np.bincount` then does the count and the sum in
    one pass each. The rank is the position of the row inside its own group when the group is
    sorted by `values`, computed by the same occurrence-index device `prefix_curves` uses for
    the Kish denominator.
    """
    codes = pd.factorize(keys)[0]
    n_groups = int(codes.max()) + 1 if codes.size else 0
    if n_groups == 0:
        empty = np.zeros(0, dtype=float)
        return dict.fromkeys(("count", "mean", "median", "min", "spread", "rank"), empty)
    count = np.bincount(codes, minlength=n_groups).astype(float)
    total = np.bincount(codes, weights=values, minlength=n_groups)
    mean = total / np.maximum(count, 1.0)
    squares = np.bincount(codes, weights=values**2, minlength=n_groups)
    variance = np.maximum(squares / np.maximum(count, 1.0) - mean**2, 0.0)
    minimum = np.full(n_groups, np.inf)
    np.minimum.at(minimum, codes, values)
    frame = pd.DataFrame({"g": codes, "v": values})
    median = frame.groupby("g", sort=True)["v"].median().to_numpy()
    order = np.lexsort((values, codes))
    rank = np.empty(codes.size, dtype=float)
    ordered_codes = codes[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered_codes[1:] != ordered_codes[:-1]]))
    lengths = np.diff(np.concatenate([starts, [ordered_codes.size]]))
    within = np.arange(ordered_codes.size, dtype=float) - np.repeat(starts, lengths)
    rank[order] = within
    return {
        "count": count[codes],
        "mean": mean[codes],
        "median": median[codes],
        "min": minimum[codes],
        "spread": np.sqrt(variance)[codes],
        "rank": rank / np.maximum(count[codes] - 1.0, 1.0),
    }


def _assert_sites_are_page_local(sites: np.ndarray, documents: np.ndarray) -> None:
    """A site must lie on exactly one page, or the site aggregates would cross a document.

    Decision 5 claims no veto feature pools across a document boundary. That claim rests on
    `site_id` being a location on one page, which it is in this corpus -- but resting on it
    silently would mean a corpus where it stopped being true produced features that depended on
    which other pages were in the batch, with nothing to say so.
    """
    spanning = pd.DataFrame({"site": sites, "document": documents}).groupby("site")["document"]
    offenders = spanning.nunique()
    bad = offenders[offenders > 1]
    if len(bad):
        raise PhaseError(
            f"{len(bad)} site(s) span more than one document, the first being {bad.index[0]!r}; "
            "the site aggregates would then cross a page boundary"
        )


def local_features(block: Block, sites: np.ndarray) -> np.ndarray:
    """The site- and page-local view of one candidate, from label-free quantities only."""
    _assert_sites_are_page_local(sites, block.documents)
    harm = np.asarray(block.p_harm, dtype=float)
    utility = np.asarray(block.utility, dtype=float)
    site = _group_stats(sites, harm)
    page = _group_stats(block.documents, harm)
    page_utility = _group_stats(block.documents, utility)
    site_max = -_group_stats(sites, -utility)["min"]
    return np.column_stack(
        [
            harm,
            np.asarray(block.p_benefit, dtype=float),
            _logit(harm),
            utility,
            site["count"],
            site["rank"],
            site["min"],
            site["spread"],
            site_max - utility,
            (utility >= site_max).astype(float),
            page["count"],
            page["mean"],
            page["median"],
            page["spread"],
            page_utility["mean"],
            page_utility["rank"],
        ]
    )


# ------------------------------------------------------------------ what each fitter may read


@dataclass(frozen=True, slots=True)
class SourceBoundary:
    """What a veto may read from the source engines: features, and their labels."""

    features: np.ndarray
    harmful: np.ndarray
    documents: np.ndarray
    engines: np.ndarray
    p_harm: np.ndarray


@dataclass(frozen=True, slots=True)
class TargetBoundary:
    """What a veto may read from the unseen engine: features, and nothing else.

    There is deliberately no `harmful` field. A veto that wanted a target label could not
    reach one through this object, and the leakage suite walks every fitting function's syntax
    tree to confirm none of them names another source of rows.
    """

    features: np.ndarray
    p_harm: np.ndarray


@dataclass(slots=True)
class FoldState:
    """One fold's frozen SGV5 model, the three blocks, the feature matrix and the bands."""

    held_out: str
    selected: dict[str, Any]
    lambda_: float
    calibration: Block
    pool: Block
    evaluation: Block
    features: dict[str, np.ndarray]
    feature_names: tuple[str, ...]
    global_columns: tuple[int, ...]
    local_columns: tuple[int, ...]
    source_thresholds: dict[str, float]
    source_threshold_feasible: dict[str, bool]
    source_engines: np.ndarray
    bands: dict[str, Band]
    identity: dict[str, Any]
    diagnostics: dict[str, Any]

    def block(self, name: str) -> Block:
        return {"calibration": self.calibration, "pool": self.pool, "evaluation": self.evaluation}[
            name
        ]


def _feature_matrix(
    design: dg.Design, block: Block, columns: list[int], sites: np.ndarray
) -> np.ndarray:
    """SGV5's representation for these rows, plus the site- and page-local block."""
    return np.hstack(
        [
            np.asarray(design.matrix[np.ix_(block.index, columns)], dtype=float),
            local_features(block, sites[block.index]),
        ]
    )


def build_state(
    base: dg.Design,
    fold: rl.Fold,
    selection: dict[str, Any],
    signatures: np.ndarray,
    retrieval_columns: list[int],
    held_out: str,
    frozen: pd.DataFrame | None,
) -> FoldState:
    """Rebuild SGV5's selected model on one fold, materialise the blocks, build the features.

    The blocks, the stratification and the identity check are SGV9's, imported rather than
    restated. What is new here is the feature matrix the veto reads and the bands, both of
    which are derived from quantities the frozen model already produces.
    """
    extended_raw, retrieval = c5.extend_fold(base, fold, signatures, retrieval_columns)
    extended = s6._fill_target_block(
        base, extended_raw, retrieval, fold, signatures, retrieval_columns
    )
    design = extended.design
    representation, model_kind = selection["representation"], selection["model"]
    if model_kind == "ranker":  # pragma: no cover - no fold selected the ranker
        raise PhaseError(
            "the veto reads the frozen model's calibrated harm probability and the ranker "
            "exposes none; a fold that selected it would need its own construction"
        )
    lambda_ = float(selection["lambda"])
    model = c5.fit_reliability(
        design, fold, extended.columns(representation), model_kind, representation
    )
    strata = s9.fit_strata(design.matrix, design.names, fold.source_cal)

    calibration = s9._blocks_from(design, extended, model, lambda_, strata, fold.source_cal)
    pool = s9._blocks_from(design, extended, model, lambda_, strata, fold.unlabeled_target)
    evaluation = s9._blocks_from(design, extended, model, lambda_, strata, fold.eval)

    identity: dict[str, Any] = {"checked": False}
    if frozen is not None:
        identifiers = design.meta["candidate_id"].astype(str).to_numpy()
        reference = frozen[frozen["held_out_engine"] == held_out].set_index("candidate_id")
        published = reference.loc[identifiers[fold.eval], s6.SGV5_ARM].to_numpy(dtype=float)
        difference = float(np.abs(evaluation.utility - published).max())
        if difference != 0.0:
            raise PhaseError(
                f"{held_out}: the rebuilt SGV5 decision score differs from the published one by "
                f"{difference:g}; every SGV10 number is measured against that score and a "
                "drifted baseline invalidates all of them"
            )
        identity = {"checked": True, "max_abs_difference": difference, "rows": int(fold.eval.size)}

    evaluation_documents = set(evaluation.documents.tolist())
    for name, block in (("calibration", calibration), ("pool", pool)):
        if set(block.documents.tolist()) & evaluation_documents:
            raise PhaseError(f"{held_out}: the {name} block shares pages with the evaluation")

    thresholds: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        decision = select_threshold(
            calibration.utility,
            calibration.harmful,
            epsilon,
            delta=DELTA,
            controller="empirical",
        )
        feasible[key] = bool(decision.feasible)
        thresholds[key] = float(decision.tau) if decision.feasible else float(np.inf)

    sites = design.meta["site_id"].astype(str).to_numpy()
    columns = extended.columns(representation)
    features = {
        name: _feature_matrix(design, block, columns, sites)
        for name, block in (
            ("calibration", calibration),
            ("pool", pool),
            ("evaluation", evaluation),
        )
    }
    feature_names = tuple(design.names[i] for i in columns) + LOCAL_FEATURES
    global_columns = tuple(range(len(columns)))
    local_columns = tuple(range(len(columns), len(feature_names)))

    # The bands are fitted where the labels are -- on the source calibration rows for the
    # source-side band the veto trains inside, and on the unlabelled target pool for the band
    # that is actually deployed. Both are carried as bounds, never re-derived on the block they
    # are applied to.
    bands: dict[str, Band] = {}
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        bands[f"source|{key}"] = fit_rank_band(
            calibration, thresholds[key], BAND_FRACTION, "source calibration rows"
        )
        bands[f"source_harm|{key}"] = fit_harm_band(
            calibration, thresholds[key], HARM_DELTA, "source calibration rows"
        )

    diagnostics = {
        "rows": {
            "source_calibration": calibration.size,
            "adaptation_pool": pool.size,
            "evaluation": evaluation.size,
        },
        "documents": {
            "source_calibration": len(set(calibration.documents.tolist())),
            "adaptation_pool": len(set(pool.documents.tolist())),
            "evaluation": len(evaluation_documents),
        },
        "harm_prevalence": {
            "source_calibration": float(calibration.harmful.mean()),
            "adaptation_pool": float(pool.harmful.mean()),
            "evaluation": float(evaluation.harmful.mean()),
        },
        "beneficial_rows": {
            "source_calibration": int(calibration.beneficial.sum()),
            "adaptation_pool": int(pool.beneficial.sum()),
            "evaluation": int(evaluation.beneficial.sum()),
        },
        "features": {
            "sgv5_representation_columns": len(columns),
            "local_columns": len(LOCAL_FEATURES),
            "total": len(feature_names),
            "note": (
                "every local column is an aggregate over the other candidates at the same site "
                "or on the same page. No statistic crosses a document boundary, so a page's "
                "feature vector is the same whether it is scored alone or in a batch."
            ),
        },
    }
    return FoldState(
        held_out=held_out,
        selected=dict(selection),
        lambda_=lambda_,
        calibration=calibration,
        pool=pool,
        evaluation=evaluation,
        features=features,
        feature_names=feature_names,
        global_columns=global_columns,
        local_columns=local_columns,
        source_thresholds=thresholds,
        source_threshold_feasible=feasible,
        source_engines=design.meta["engine_id"].astype(str).to_numpy()[fold.source_cal],
        bands=bands,
        identity=identity,
        diagnostics=diagnostics,
    )


# ------------------------------------------------------------------ the veto model


@dataclass(slots=True)
class Veto:
    """One fitted boundary veto: how to score a row, and enough of it to publish.

    `columns` is which of the feature matrix's columns the veto was allowed to see, so ablation
    E is a different column list and not a different code path. `apply` returns a HARM
    probability -- higher is worse -- and the ordering inside the band is by its complement.
    """

    name: str
    kind: str
    columns: tuple[int, ...]
    scaler: Any
    model: Any
    fitted_on: str
    n_rows: int
    n_harmful: int
    diagnostics: dict[str, Any]

    def apply(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(features.shape[0], dtype=float)
        block = np.asarray(features[:, list(self.columns)], dtype=float)
        scaled = np.asarray(self.scaler.transform(block), dtype=float)
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.zeros(len(scaled), dtype=float)
        return np.asarray(self.model.predict_proba(scaled)[:, classes.index(1)], dtype=float)


def _make_veto_model(kind: str) -> Any:
    """The two model classes SGV5 froze, with SGV5's own settings and SGV5's own seed."""
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
        )
    if kind == "boosted":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=c5.BOOSTING_ITERATIONS,
            class_weight="balanced",
            random_state=pilot.FIT_SEED,
        )
    raise PhaseError(f"unknown veto model {kind!r}")


def fit_veto(
    name: str,
    source: SourceBoundary,
    kind: str,
    columns: tuple[int, ...],
    fitted_on: str,
) -> Veto:
    """Fit the veto on labelled boundary rows. The scaler is fitted here and frozen here."""
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(source.harmful, dtype=int)
    if labels.size < MIN_BAND_ROWS or labels.min() == labels.max():
        return Veto(
            name=name,
            kind=kind,
            columns=columns,
            scaler=None,
            model=None,
            fitted_on=fitted_on,
            n_rows=int(labels.size),
            n_harmful=int(labels.sum()),
            diagnostics={
                "degenerate": True,
                "reason": (
                    f"{labels.size} labelled band rows with {int(labels.sum())} harmful; a veto "
                    f"needs at least {MIN_BAND_ROWS} rows and both classes"
                ),
            },
        )
    matrix = np.asarray(source.features[:, list(columns)], dtype=float)
    scaler = StandardScaler().fit(matrix)
    scaled = np.asarray(scaler.transform(matrix), dtype=float)
    model = _make_veto_model(kind).fit(scaled, labels)
    fitted = Veto(
        name=name,
        kind=kind,
        columns=columns,
        scaler=scaler,
        model=model,
        fitted_on=fitted_on,
        n_rows=int(labels.size),
        n_harmful=int(labels.sum()),
        diagnostics={"degenerate": False},
    )
    in_sample = fitted.apply(source.features)
    fitted.diagnostics.update(
        {
            "in_sample_auroc": float(roc_auc(in_sample, labels.astype(bool))),
            "band_harm_prevalence": float(labels.mean()),
            "n_features": len(columns),
        }
    )
    return fitted


def select_veto_kind(source: SourceBoundary, columns: tuple[int, ...]) -> dict[str, Any]:
    """Choose the veto's model class by leave-one-SOURCE-engine-out inside the source band.

    The held-out engine appears nowhere in this selection, and neither does any target row:
    each source engine takes a turn as a pseudo-target, the veto is fitted on the remaining
    two, and the winner is the class with the higher mean AUROC on the engines it did not see.
    `selection_scope` in the run record names exactly this, as the leakage rules require.
    """
    engines = sorted(set(source.engines.tolist()))
    scores: dict[str, list[float]] = {kind: [] for kind in VETO_KINDS}
    per_engine: dict[str, dict[str, float]] = {kind: {} for kind in VETO_KINDS}
    for pseudo_target in engines:
        train = source.engines != pseudo_target
        test = ~train
        if not test.any() or np.unique(source.harmful[train]).size < 2:
            continue
        inner = SourceBoundary(
            features=source.features[train],
            harmful=source.harmful[train],
            documents=source.documents[train],
            engines=source.engines[train],
            p_harm=source.p_harm[train],
        )
        for kind in VETO_KINDS:
            fitted = fit_veto(f"inner|{kind}|{pseudo_target}", inner, kind, columns, "inner fit")
            value = float(roc_auc(fitted.apply(source.features[test]), source.harmful[test]))
            scores[kind].append(value)
            per_engine[kind][pseudo_target] = value
    means = {
        kind: float(np.mean(values)) if values else float("nan") for kind, values in scores.items()
    }
    usable = {kind: value for kind, value in means.items() if np.isfinite(value)}
    chosen = max(sorted(usable), key=lambda k: usable[k]) if usable else VETO_KINDS[0]
    return {
        "chosen": chosen,
        "mean_auroc": means,
        "per_pseudo_target": per_engine,
        "selection_scope": (
            "leave-one-source-engine-out inside the source calibration band; the held-out "
            "engine and every target row are absent"
        ),
    }


def random_veto_scores(block: Block, band: np.ndarray, engine: str, tag: str) -> np.ndarray:
    """Ablation D. A content-derived permutation of the band, with no information in it.

    Derived from sha256 of the engine and the block name, never from `hash()`, which
    PYTHONHASHSEED moves between processes.
    """
    rng = np.random.default_rng(_stable_seed("sgv10-random-abstention", engine, tag))
    scores = np.zeros(block.size, dtype=float)
    scores[band] = rng.permutation(np.linspace(0.0, 1.0, int(band.sum())))
    return scores


# ------------------------------------------------------------------ the boundary-local estimate


@dataclass(slots=True)
class LocalEstimate:
    """Method B. An isotonic fit of realised harm on the frozen probability, inside the band.

    Outside the band the estimate is SGV5's own, untouched. That is what makes this a boundary
    method and what makes the comparison against SGV9 -- whose map acted on the whole score
    range -- a comparison of where the recalibration is applied rather than of two different
    recalibration families.
    """

    name: str
    curve: Any
    fitted_on: str
    n_rows: int
    band: Band
    diagnostics: dict[str, Any]

    def apply(self, block: Block, band: np.ndarray) -> np.ndarray:
        out = np.asarray(block.p_harm, dtype=float).copy()
        if self.curve is None or not band.any():
            return out
        out[band] = np.clip(
            np.asarray(self.curve.transform(block.p_harm[band]), dtype=float), 0.0, 1.0
        )
        return out


def fit_local_estimate(
    name: str,
    p_harm: np.ndarray,
    harmful: np.ndarray,
    band: Band,
    fitted_on: str,
) -> LocalEstimate:
    """Isotonic in the frozen probability, so the estimate is monotone where it is applied."""
    from ocr_risk.calibrate.calibrators import build_calibrator

    values = np.asarray(p_harm, dtype=float)
    labels = np.asarray(harmful, dtype=float)
    if values.size < MIN_BAND_ROWS or np.unique(labels).size < 2:
        return LocalEstimate(
            name=name,
            curve=None,
            fitted_on=fitted_on,
            n_rows=int(values.size),
            band=band,
            diagnostics={
                "degenerate": True,
                "reason": (
                    f"{values.size} labelled band rows with {int(labels.sum())} harmful; the "
                    f"local estimate needs at least {MIN_BAND_ROWS} rows and both classes"
                ),
            },
        )
    curve = build_calibrator(pilot.CALIBRATION_METHOD)
    curve.fit(values, labels)
    fitted = np.clip(np.asarray(curve.transform(values), dtype=float), 0.0, 1.0)
    return LocalEstimate(
        name=name,
        curve=curve,
        fitted_on=fitted_on,
        n_rows=int(values.size),
        band=band,
        diagnostics={
            "degenerate": False,
            "band_harm_prevalence": float(labels.mean()),
            "mean_frozen_estimate": float(values.mean()),
            "mean_local_estimate": float(fitted.mean()),
            "level_correction": float(fitted.mean() - values.mean()),
        },
    )


# ------------------------------------------------------------------ the composite score


def composite_score(block: Block, band: np.ndarray, safety: np.ndarray) -> np.ndarray:
    """SGV5's order outside the band; the veto's order inside it.

    The band rows are reassigned among the DECISION-SCORE VALUES they already occupy: collect
    the positions those rows hold in SGV5's descending order, sort them by the veto's safety,
    and hand the highest of those values to the safest. Nothing else moves.

    Three properties follow, and all three are asserted in `_assert_band_ordering` rather than
    argued for. The multiset of scores is exactly SGV5's, so an arm cannot win by inflating the
    scale. Every non-band row keeps its own score, so the veto cannot reach outside the band.
    And the interleaving between band and non-band rows is untouched, so a band row can only
    overtake another band row.

    Reassigning values rather than mapping into an interval is what lets the brief's literal
    harm-probability band be deployed at all: that band is not contiguous in the decision
    ordering -- the score is `benefit - lambda * harm`, so a band in `harm` alone is scattered
    through the ranking -- and there is no interval to map it into.
    """
    out = np.asarray(block.utility, dtype=float).copy()
    inside = np.asarray(band, dtype=bool)
    count = int(inside.sum())
    if count < 2:
        return out
    order = np.argsort(-out, kind="stable")
    positions = np.flatnonzero(inside[order])
    values = out[order][positions]
    rows = order[positions]
    ranked = np.argsort(-np.asarray(safety, dtype=float)[rows], kind="stable")
    out[rows[ranked]] = values
    return out


def _assert_band_ordering(
    label: str, block: Block, band: np.ndarray, score: np.ndarray, atol: float = 0.0
) -> dict[str, Any]:
    """The three properties `composite_score` claims. A violation is a hard failure.

    `atol` is zero: the composite is built by moving existing values, never by arithmetic on
    them, so every comparison here is exact and an approximate one would hide a real bug.
    """
    inside = np.asarray(band, dtype=bool)
    utility = np.asarray(block.utility, dtype=float)
    values = np.asarray(score, dtype=float)
    outside = ~inside
    if outside.any() and not np.array_equal(values[outside], utility[outside]):
        raise PhaseError(f"{label}: the veto changed the score of a row outside the band")
    if not np.array_equal(np.sort(values), np.sort(utility)):
        raise PhaseError(f"{label}: the veto changed the multiset of decision scores")
    if inside.any() and not np.array_equal(np.sort(values[inside]), np.sort(utility[inside])):
        raise PhaseError(f"{label}: the band's own scores are not a permutation of themselves")
    del atol
    return {
        "n_band": int(inside.sum()),
        "n_outside": int(outside.sum()),
        "marginal_preserved": True,
        "outside_unchanged": True,
    }


# ------------------------------------------------------------------ Method D: CVaR over pages


def _cvar_grid(size: int, band: np.ndarray) -> np.ndarray:
    """Where the CVaR is evaluated: an even grid, the top of the ranking, and the whole band.

    The point and bound rules search every rank. The CVaR does not, because it needs a sort of
    the accepted pages at each rank; the grid is dense where the cut can actually fall and the
    approximation is recorded as a limitation rather than hidden.
    """
    if size == 0:
        return np.zeros(0, dtype=int)
    step = max(size // CVAR_GRID, 1)
    ranks = set(range(step, size + 1, step))
    ranks.update(range(1, min(50, size) + 1))
    ranks.add(size)
    if band.any():
        positions = np.flatnonzero(band)
        ranks.update(int(p) + 1 for p in positions)
    return np.array(sorted(ranks), dtype=int)


def cvar_of_prefix(documents: np.ndarray, values: np.ndarray, alpha: float) -> float:
    """The mean estimated harm rate of the worst `1 - alpha` share of accepted PAGES.

    Unweighted over pages on purpose: a page with one bad accepted correction and a page with
    fifty count the same, because the failure this is meant to catch is a page that goes wrong,
    not a page that is large. At least one page always enters the average, so the quantity is
    defined wherever anything is accepted.
    """
    if values.size == 0:
        return float("nan")
    frame = pd.DataFrame({"d": documents, "v": np.asarray(values, dtype=float)})
    per_page = frame.groupby("d", sort=True)["v"].mean().to_numpy()
    per_page = np.sort(per_page)[::-1]
    take = max(int(np.ceil((1.0 - alpha) * per_page.size)), 1)
    return float(per_page[:take].mean())


def cvar_curve(
    block: Block, order: np.ndarray, estimates: np.ndarray, alpha: float, band: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The CVaR of every prefix on the grid, in rank order.

    The page codes are factorised once and each grid point is two `bincount` calls, so the whole
    curve costs one pass per grid point rather than one groupby per grid point. It computes the
    same quantity as `cvar_of_prefix`, which is the readable reference implementation, and
    `tests/leakage` asserts the two agree on every grid point of every fold.
    """
    ranks = _cvar_grid(block.size, band[order] if band.size else band)
    codes = pd.factorize(block.documents[order])[0]
    n_pages = int(codes.max()) + 1 if codes.size else 0
    values = np.asarray(estimates, dtype=float)[order]
    out = np.empty(ranks.size, dtype=float)
    for position, k in enumerate(ranks.tolist()):
        counts = np.bincount(codes[:k], minlength=n_pages)
        totals = np.bincount(codes[:k], weights=values[:k], minlength=n_pages)
        present = counts > 0
        means = np.sort(totals[present] / counts[present])[::-1]
        take = max(int(np.ceil((1.0 - alpha) * means.size)), 1)
        out[position] = float(means[:take].mean())
    return ranks, out


def realized_cvar(block: Block, accepted: np.ndarray, alpha: float) -> float:
    """The realised page-level tail of an accept set: what Method D was trying to bound."""
    if not accepted.any():
        return float("nan")
    return cvar_of_prefix(block.documents[accepted], block.harmful[accepted].astype(float), alpha)


# ------------------------------------------------------------------ placing the cut


def place_cut(
    pool: Block,
    score: np.ndarray,
    estimates: np.ndarray,
    band: np.ndarray,
    epsilon: float,
    rule: str,
    inflation: float,
    source_tau: float,
    source_feasible: bool,
) -> dict[str, Any]:
    """Choose a deployment threshold. Reads target SCORES and target estimates, no target label.

    Every rule but the source threshold runs its prefix search over the unlabelled adaptation
    pool; only the resulting scalar leaves this function, and none of them reaches the
    evaluation block. There is exactly one implementation, used both while the arms are being
    fitted and when the saved table is read back, so a threshold cannot drift between the two.
    `tests/leakage` walks this function's syntax tree and fails if it names an outcome array,
    an evaluation block or an oracle.
    """
    if rule == SOURCE_THRESHOLD:
        return {
            "tau": float(source_tau),
            "feasible": bool(source_feasible),
            "pool_rank": -1,
            "pool_predicted_harm": float("nan"),
            "pool_criterion": float("nan"),
        }
    scored = replace(pool, utility=np.asarray(score, dtype=float))
    curves = prefix_curves(scored, estimates)
    if rule in CVAR_RULES:
        alpha = float(int(rule.split("_")[1]) / 100.0)
        ranks, values = cvar_curve(scored, curves.order, estimates, alpha, band)
        feasible = np.flatnonzero(values <= epsilon)
        if feasible.size == 0:
            return _refused()
        index = int(ranks[int(feasible[-1])]) - 1
        criterion = float(values[int(feasible[-1])])
    elif rule in (POINT_CUT, BOUND_CUT):
        criterion_curve = (
            curves.predicted if rule == POINT_CUT else harm_upper_bound(curves, inflation)
        )
        index = _deepest(criterion_curve, epsilon)
        if index < 0:
            return _refused()
        criterion = float(criterion_curve[index])
    else:
        raise PhaseError(f"unknown cut rule {rule!r}")
    return {
        "tau": float(curves.utility[index]),
        "feasible": True,
        "pool_rank": index + 1,
        "pool_predicted_harm": float(curves.predicted[index]),
        "pool_criterion": criterion,
        "pool_effective_n": float(curves.effective_n[index]),
    }


def _refused() -> dict[str, Any]:
    return {
        "tau": float(np.inf),
        "feasible": False,
        "pool_rank": 0,
        "pool_predicted_harm": float("nan"),
        "pool_criterion": float("nan"),
    }


# ------------------------------------------------------------------ Method C: the transfer bound


def _transfer_error(state: FoldState, arm: ArmFit, epsilon: float) -> dict[str, Any]:
    """Replay the whole procedure on one pseudo-target and record how wrong the estimate was.

    The quantity is `realised harm minus calibrated estimate` over the rows the PRIMARY arm
    would actually have accepted -- not a per-row residual. A per-row residual on a Bernoulli
    outcome is dominated by the outcome's own variance and says nothing about whether an accept
    set's harm rate was estimated correctly, which is the only thing a bound has to get right.
    SGV9 made the same choice for the same reason, and the inner folds here are SGV9's, so the
    two stages' inflations are measured on identical splits.
    """
    key = f"epsilon_{int(epsilon * 100)}"
    placed = place_cut(
        state.pool,
        arm.scores[key]["pool"],
        arm.estimates[key]["pool"],
        arm.bands[key].contains(state.pool),
        epsilon,
        POINT_CUT,
        0.0,
        state.source_thresholds[key],
        state.source_threshold_feasible[key],
    )
    if not placed["feasible"]:
        return {"pseudo_target": state.held_out, "error": float("nan"), "n_accepted": 0}
    outcome = state.evaluation
    score = arm.scores[key]["evaluation"]
    accepted = np.isfinite(score) & (score >= placed["tau"])
    if not accepted.any():
        return {"pseudo_target": state.held_out, "error": float("nan"), "n_accepted": 0}
    realized = float(outcome.harmful[accepted].astype(float).mean())
    predicted = float(np.asarray(arm.estimates[key]["evaluation"])[accepted].mean())
    return {
        "pseudo_target": state.held_out,
        "error": realized - predicted,
        "realized_harm_rate": realized,
        "predicted_harm_rate": predicted,
        "n_accepted": int(accepted.sum()),
        "pool_rank": placed["pool_rank"],
        "tau": placed["tau"],
    }


def transfer_inflation(
    base: dg.Design,
    held_out: str,
    selection: dict[str, Any],
    signatures: np.ndarray,
    retrieval_columns: list[int],
) -> dict[str, Any]:
    """How wrong the primary arm is on an engine it did not fit from, priced on source engines.

    Each fit engine takes a turn as a pseudo-target and the entire pipeline -- the frozen
    model, the band, the veto and the local estimate -- is refitted from the remaining fit
    engines. Three replays, so three errors per epsilon, and the inflation is the MAXIMUM of
    the three rather than a smoothed quantile: with three exchangeable calibration points the
    distribution-free coverage a quantile can support is 3/4, and taking the maximum is the
    only choice that attains it. This stage reports 3/4 and does not claim 1 - alpha.

    The replays are structurally harder than the fold they are used on -- each is fitted from
    two engines where the outer fold has three -- so the inflation is biased upward. For a
    bound that is the safe direction, and it is recorded rather than corrected.
    """
    sources = tuple(engine for engine in base.engines if engine != held_out)
    replays: dict[str, list[dict[str, Any]]] = {}
    for pseudo_target in sources:
        fold = inner_fold(base, held_out, pseudo_target)
        state = build_state(
            base, fold, selection, signatures, retrieval_columns, pseudo_target, None
        )
        arm = fit_arms(state, EPSILONS, (HYBRID,))[HYBRID]
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            replays.setdefault(key, []).append(_transfer_error(state, arm, epsilon))
    per_epsilon: dict[str, Any] = {}
    for key, rows in replays.items():
        errors = [row["error"] for row in rows if np.isfinite(row["error"])]
        per_epsilon[key] = {
            "inflation": float(max(errors)) if errors else 0.0,
            "errors": {row["pseudo_target"]: row["error"] for row in rows},
            "replays": rows,
            "usable_replays": len(errors),
        }
    pooled = [
        row["error"] for rows in replays.values() for row in rows if np.isfinite(row["error"])
    ]
    return {
        "per_epsilon": per_epsilon,
        "pooled_inflation": float(max(pooled)) if pooled else 0.0,
        "source_engines": list(sources),
        "distribution_free_coverage": CONFORMAL_COVERAGE,
        "nominal_alpha": CONFORMAL_ALPHA,
        "note": (
            "the inflation is the largest error the SGV10 pipeline made across a leave-one-"
            "source-engine-out replay on SGV9's inner folds. With three exchangeable "
            "calibration engines the guaranteed coverage is 3/4, not 1 - alpha."
        ),
    }


def risk_sets(estimates: np.ndarray, epsilon: float, inflation: float) -> dict[str, Any]:
    """Method C's SAFE / UNCERTAIN / UNSAFE partition, on the inflated interval.

    A row is SAFE when even the inflated estimate is within epsilon, UNSAFE when even the
    deflated estimate is above it, and UNCERTAIN when the interval straddles epsilon. The
    interval is the one the bound cut uses, so the partition and the deployment decision
    cannot disagree.
    """
    upper = np.clip(np.asarray(estimates, dtype=float) + inflation, 0.0, 1.0)
    lower = np.clip(np.asarray(estimates, dtype=float) - inflation, 0.0, 1.0)
    safe = upper <= epsilon
    unsafe = lower > epsilon
    uncertain = ~(safe | unsafe)
    return {
        "safe": int(safe.sum()),
        "uncertain": int(uncertain.sum()),
        "unsafe": int(unsafe.sum()),
        "share_safe": float(safe.mean()) if safe.size else float("nan"),
        "share_uncertain": float(uncertain.mean()) if uncertain.size else float("nan"),
        "share_unsafe": float(unsafe.mean()) if unsafe.size else float("nan"),
        "inflation": float(inflation),
    }


# ------------------------------------------------------------------ fitting the whole family


@dataclass(slots=True)
class ArmFit:
    """One arm's score and harm estimate on every block, at every epsilon."""

    name: str
    scores: dict[str, dict[str, np.ndarray]]
    estimates: dict[str, dict[str, np.ndarray]]
    bands: dict[str, Band]
    diagnostics: dict[str, Any]


BLOCK_NAMES = ("calibration", "pool", "evaluation")


def _source_boundary(state: FoldState, band: Band | None) -> SourceBoundary:
    """The labelled source rows a veto is allowed to fit on. `None` means every source row."""
    block = state.calibration
    inside = np.ones(block.size, dtype=bool) if band is None else band.contains(block)
    return SourceBoundary(
        features=state.features["calibration"][inside],
        harmful=block.harmful[inside],
        documents=block.documents[inside],
        engines=state.source_engines[inside],
        p_harm=block.p_harm[inside],
    )


def _target_boundary(state: FoldState, name: str, band: Band) -> TargetBoundary:
    """The unlabelled target rows a veto is applied to. Carries no outcome field at all."""
    block = state.block(name)
    inside = band.contains(block)
    return TargetBoundary(features=state.features[name][inside], p_harm=block.p_harm[inside])


def _source_band(state: FoldState, epsilon: float, kind: str, width: float, delta: float) -> Band:
    """The band the veto trains inside, centred on SGV5's own threshold on the source rows."""
    key = f"epsilon_{int(epsilon * 100)}"
    tau = state.source_thresholds[key]
    if kind == RANK_BAND:
        return fit_rank_band(state.calibration, tau, width, "source calibration rows")
    return fit_harm_band(state.calibration, tau, delta, "source calibration rows")


def _pool_band(state: FoldState, tau: float, kind: str, width: float, delta: float) -> Band:
    """The band that is deployed, centred on the incumbent's cut on the unlabelled pool."""
    where = "adaptation pool (target, unlabelled)"
    if kind == RANK_BAND:
        return fit_rank_band(state.pool, tau, width, where)
    return fit_harm_band(state.pool, tau, delta, where)


def _reference_cut(state: FoldState, epsilon: float) -> dict[str, Any]:
    """Where the incumbent's cut falls on the unlabelled pool. The band is centred here.

    The band has to be centred on something, and it cannot be centred on the arm's own cut
    without a circularity -- the cut depends on the composite score, which depends on the band.
    It is centred on the operating point SGV5 alone would reach on the same pool under the
    point rule, which is the boundary an operator who already runs SGV5 actually has. One band
    per epsilon, shared by every arm and every cut rule, so a difference between two arms is
    never a difference between two bands.
    """
    key = f"epsilon_{int(epsilon * 100)}"
    return place_cut(
        state.pool,
        state.pool.utility,
        state.pool.p_harm,
        np.zeros(state.pool.size, dtype=bool),
        epsilon,
        POINT_CUT,
        0.0,
        state.source_thresholds[key],
        state.source_threshold_feasible[key],
    )


def fit_arms(
    state: FoldState,
    epsilons: tuple[float, ...] = EPSILONS,
    arms: tuple[str, ...] = ALL_ARMS,
    sgv9: dict[str, np.ndarray] | None = None,
    band_kind: str = PRIMARY_BAND,
    band_width: float = BAND_FRACTION,
    harm_delta: float = HARM_DELTA,
) -> dict[str, ArmFit]:
    """Every arm's score and estimate on every block, fitted in one place.

    The label-free arms see `SourceBoundary` and `TargetBoundary` and nothing else. The two
    oracles are constructed here as well, from labels this stage is explicitly allowed to read
    for an upper bound, and they live in the same dictionary so that a phase which forgets to
    exclude them fails a test rather than publishing a number.
    """
    requested = tuple(arms)
    fitted: dict[str, ArmFit] = {
        name: ArmFit(name=name, scores={}, estimates={}, bands={}, diagnostics={})
        for name in requested
    }
    all_columns = tuple(range(len(state.feature_names)))
    # The model class is chosen ONCE per fold, on the source band at the primary epsilon, and
    # reused at every epsilon. It is a hyperparameter of the veto, not a per-operating-point
    # decision, and re-selecting it at each epsilon would multiply the inner replays by three
    # without changing what is being chosen from.
    selection = select_veto_kind(
        _source_boundary(
            state, _source_band(state, PRIMARY_EPSILON, band_kind, band_width, harm_delta)
        ),
        all_columns,
    )
    for epsilon in epsilons:
        key = f"epsilon_{int(epsilon * 100)}"
        reference = _reference_cut(state, epsilon)
        pool_band = _pool_band(state, reference["tau"], band_kind, band_width, harm_delta)
        source_band = _source_band(state, epsilon, band_kind, band_width, harm_delta)
        membership = {name: pool_band.contains(state.block(name)) for name in BLOCK_NAMES}
        everything = {name: np.ones(state.block(name).size, dtype=bool) for name in BLOCK_NAMES}

        source_view = _source_boundary(state, source_band)
        kind = selection["chosen"]

        vetoes: dict[str, Veto] = {}
        if any(VETO_OF.get(name) == "source" for name in requested):
            vetoes["source"] = fit_veto(
                "source", source_view, kind, all_columns, "source calibration band rows"
            )
        if any(VETO_OF.get(name) == "boundary_only" for name in requested):
            vetoes["boundary_only"] = fit_veto(
                "boundary_only",
                source_view,
                kind,
                state.local_columns,
                "source calibration band rows, local features only",
            )
        if any(VETO_OF.get(name) == "source_unconfined" for name in requested):
            vetoes["source_unconfined"] = fit_veto(
                "source_unconfined",
                _source_boundary(state, None),
                kind,
                all_columns,
                "every source calibration row",
            )
        for tag, block_name in (("pool_labels", "pool"), ("evaluation_labels", "evaluation")):
            if any(VETO_OF.get(name) == tag for name in requested):
                block = state.block(block_name)
                inside = membership[block_name]
                vetoes[tag] = fit_veto(
                    tag,
                    SourceBoundary(
                        features=state.features[block_name][inside],
                        harmful=block.harmful[inside],
                        documents=block.documents[inside],
                        engines=np.full(int(inside.sum()), state.held_out),
                        p_harm=block.p_harm[inside],
                    ),
                    kind,
                    all_columns,
                    f"{block_name} band rows, target labels -- ORACLE",
                )

        local = fit_local_estimate(
            "boundary_local",
            source_view.p_harm,
            source_view.harmful,
            source_band,
            "source calibration band rows",
        )
        oracles: dict[str, LocalEstimate] = {}
        for tag, block_name in (("pool_labels", "pool"), ("evaluation_labels", "evaluation")):
            if any(ESTIMATE_OF.get(name) == tag for name in requested):
                block = state.block(block_name)
                inside = membership[block_name]
                oracles[tag] = fit_local_estimate(
                    tag,
                    block.p_harm[inside],
                    block.harmful[inside],
                    pool_band,
                    f"{block_name} band rows, target labels -- ORACLE",
                )

        for name in requested:
            veto_tag = VETO_OF.get(name)
            unconfined = veto_tag == "source_unconfined"
            bands = everything if unconfined else membership
            scores: dict[str, np.ndarray] = {}
            estimates: dict[str, np.ndarray] = {}
            for block_name in BLOCK_NAMES:
                block = state.block(block_name)
                inside = bands[block_name]
                if veto_tag is None:
                    score = np.asarray(block.utility, dtype=float).copy()
                elif veto_tag == "random":
                    tag = f"{key}|{block_name}"
                    score = composite_score(
                        block, inside, -random_veto_scores(block, inside, state.held_out, tag)
                    )
                else:
                    risk = np.zeros(block.size, dtype=float)
                    if inside.any():
                        rows = state.features[block_name][inside]
                        risk[inside] = vetoes[veto_tag].apply(rows)
                    score = composite_score(block, inside, -risk)
                _assert_band_ordering(
                    f"{state.held_out}|{name}|{key}|{block_name}", block, inside, score
                )
                estimate_tag = ESTIMATE_OF[name]
                if estimate_tag == "frozen":
                    estimate = np.asarray(block.p_harm, dtype=float).copy()
                elif estimate_tag == "boundary_local":
                    estimate = local.apply(block, membership[block_name])
                elif estimate_tag == "sgv9_quantile_transport":
                    if sgv9 is None:
                        raise PhaseError(
                            f"{name} needs SGV9's published estimates and none were supplied"
                        )
                    estimate = np.asarray(sgv9[block_name], dtype=float).copy()
                else:
                    estimate = oracles[estimate_tag].apply(block, membership[block_name])
                scores[block_name] = score
                estimates[block_name] = estimate
            fitted[name].scores[key] = scores
            fitted[name].estimates[key] = estimates
            fitted[name].bands[key] = pool_band
            fitted[name].diagnostics[key] = {
                "veto": veto_tag,
                "veto_selection": selection if veto_tag in (None, "source") else None,
                "veto_diagnostics": (vetoes[veto_tag].diagnostics if veto_tag in vetoes else None),
                "estimate": ESTIMATE_OF[name],
                "local_estimate": (
                    local.diagnostics if ESTIMATE_OF[name] == "boundary_local" else None
                ),
                "reference_cut": reference,
                "pool_band": {
                    "lower": pool_band.lower,
                    "upper": pool_band.upper,
                    "centre": pool_band.centre,
                    "rows": {k: int(v.sum()) for k, v in membership.items()},
                    "note": pool_band.note,
                },
                "source_band": {
                    "lower": source_band.lower,
                    "upper": source_band.upper,
                    "centre": source_band.centre,
                    "rows": source_band.fitted_rows,
                    "note": source_band.note,
                },
            }
    return fitted


# ------------------------------------------------------------------ the five metrics


def tail_harm(block: Block, score: np.ndarray, estimates: np.ndarray, tau: float) -> dict[str, Any]:
    """Metric 2. The harm rate in the riskiest part of what an arm accepts, read two ways.

    `marginal` is the last share of the accept set by the arm's own DECISION score -- the
    candidates it would drop first, and the only tail that is defined identically for every arm
    whatever its risk estimate looks like. `by_estimate` is the top share by the arm's own harm
    estimate, which asks the different question of whether an arm knows which of its accepts are
    its worst. Criterion 3 reads `marginal`, and that was fixed before any number was computed.
    """
    accepted = np.flatnonzero(np.isfinite(score) & (score >= tau))
    if accepted.size == 0:
        return {"n_accepted": 0, "marginal": {}, "by_estimate": {}, "accepts_nothing": True}
    order = accepted[np.argsort(-np.asarray(score, dtype=float)[accepted], kind="stable")]
    by_estimate = accepted[np.argsort(-np.asarray(estimates, dtype=float)[accepted], kind="stable")]
    harmful = np.asarray(block.harmful, dtype=float)
    marginal: dict[str, Any] = {}
    estimated: dict[str, Any] = {}
    for share in TAIL_SHARES:
        take = max(int(np.ceil(share * accepted.size)), 1)
        marginal[f"share_{int(share * 100)}"] = {
            "n": take,
            "harm_rate": float(harmful[order[-take:]].mean()),
            "predicted_harm_rate": float(np.asarray(estimates, dtype=float)[order[-take:]].mean()),
        }
        estimated[f"share_{int(share * 100)}"] = {
            "n": take,
            "harm_rate": float(harmful[by_estimate[:take]].mean()),
            "predicted_harm_rate": float(
                np.asarray(estimates, dtype=float)[by_estimate[:take]].mean()
            ),
        }
    return {
        "n_accepted": int(accepted.size),
        "overall_harm_rate": float(harmful[accepted].mean()),
        "marginal": marginal,
        "by_estimate": estimated,
        "accepts_nothing": False,
    }


def abstention_analysis(
    block: Block, score: np.ndarray, tau: float, baseline_score: np.ndarray, baseline_tau: float
) -> dict[str, Any]:
    """Metric 4. What abstention removed, what it cost, and what it let in instead.

    An arm that reorders does not only reject: it rejects some of the incumbent's accepts and
    accepts some of its rejects. Reporting only the first half would let an arm look efficient
    while quietly swapping one harmful candidate for another, so the whole two-by-two is
    recorded and the headline ratio is stated against the denominator it actually has.
    """
    accepted = np.isfinite(score) & (score >= tau)
    baseline = np.isfinite(baseline_score) & (baseline_score >= baseline_tau)
    harmful = np.asarray(block.harmful, dtype=bool)
    beneficial = np.asarray(block.beneficial, dtype=bool)
    removed = baseline & ~accepted
    added = accepted & ~baseline
    total = int(beneficial.sum())
    n_removed, n_added = int(removed.sum()), int(added.sum())
    return {
        "n_baseline_accepted": int(baseline.sum()),
        "n_arm_accepted": int(accepted.sum()),
        "n_removed": n_removed,
        "n_added": n_added,
        "harmful_removed": int((removed & harmful).sum()),
        "harmful_added": int((added & harmful).sum()),
        "beneficial_removed": int((removed & beneficial).sum()),
        "beneficial_added": int((added & beneficial).sum()),
        "harm_removed_per_rejection": (
            float((removed & harmful).sum() / n_removed) if n_removed else float("nan")
        ),
        "harm_added_per_acceptance": (
            float((added & harmful).sum() / n_added) if n_added else float("nan")
        ),
        "net_harmful_change": int((added & harmful).sum()) - int((removed & harmful).sum()),
        "net_repair_recall_change": (
            float((int((added & beneficial).sum()) - int((removed & beneficial).sum())) / total)
            if total
            else float("nan")
        ),
        "beneficial_lost_per_harmful_removed": (
            float((removed & beneficial).sum() / max(int((removed & harmful).sum()), 1))
            if n_removed
            else float("nan")
        ),
    }


def boundary_calibration(block: Block, estimates: np.ndarray, band: np.ndarray) -> dict[str, Any]:
    """Metric 3. Calibration error over the whole block, and over the band alone.

    Both come from `ocr_risk.metrics.calibration.calibration_report`, which is the single
    definition of Brier and ECE in this repository. Only the row subset differs, which is the
    entire point: SGV9 measured the first and found almost nothing to fix on three engines.
    """
    outcomes = np.asarray(block.harmful, dtype=float)
    values = np.asarray(estimates, dtype=float)
    inside = np.asarray(band, dtype=bool)
    report = {
        "global": calibration_report(values, outcomes, n_bins=N_BINS).as_dict(),
        "n_band": int(inside.sum()),
    }
    if int(inside.sum()) >= N_BINS:
        report["boundary"] = calibration_report(
            values[inside], outcomes[inside], n_bins=N_BINS
        ).as_dict()
    else:
        report["boundary"] = None
        report["boundary_note"] = (
            f"{int(inside.sum())} band rows is fewer than the {N_BINS} bins the report uses; "
            "no boundary calibration is computed for this cell"
        )
    return report


def deployment(
    block: Block,
    score: np.ndarray,
    estimates: np.ndarray,
    band: np.ndarray,
    placed: dict[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    """What one placed threshold does on the evaluation block, with everything it implies."""
    scored = replace(block, utility=np.asarray(score, dtype=float))
    if not placed["feasible"]:
        point = _deployed_at(scored, float(np.inf), epsilon)
    else:
        point = _deployed_at(scored, float(placed["tau"]), epsilon)
    accepted = np.isfinite(score) & (score >= float(placed["tau"]))
    point["predicted_harm_rate"] = (
        float(np.asarray(estimates, dtype=float)[accepted].mean())
        if accepted.any()
        else float("nan")
    )
    if placed["feasible"] and accepted.any():
        marginal = int(np.flatnonzero(accepted)[np.argmin(np.asarray(score)[accepted])])
        point["cut_inside_band"] = bool(band[marginal])
    else:
        point["cut_inside_band"] = False
    point["n_band_accepted"] = int((accepted & band).sum())
    point["feasible"] = bool(placed["feasible"])
    point["pool_rank"] = int(placed["pool_rank"])
    point["pool_predicted_harm"] = float(placed["pool_predicted_harm"])
    point["pool_criterion"] = float(placed["pool_criterion"])
    point["realized_cvar"] = {
        f"alpha_{int(a * 100)}": realized_cvar(block, accepted, a) for a in CVAR_ALPHAS
    }
    point["own_frontier"] = achievable_repair_recall(
        np.asarray(score, dtype=float), block.harmful, block.beneficial
    )[f"epsilon_{int(epsilon * 100)}"]
    return point


# ------------------------------------------------------------------ SGV9's published estimates


def load_sgv9_estimates() -> dict[str, dict[str, dict[str, float]]]:
    """Ablation B's input: SGV9's primary calibration map, read from SGV9's published table.

    Refitting SGV9's map here would reproduce a number that is already frozen, hashed and
    published, and a drift between the two stages would be invisible. Reading it is also the
    stricter option: the file is verified against the hash SGV9's own provenance manifest
    carries before a single value is used.
    """
    manifest = s9.PROVENANCE
    if not manifest.is_file() or not s9.SCORES.is_file():
        raise PhaseError(
            "SGV9's calibrated_scores.parquet and provenance_manifest.json are required for "
            f"{GLOBAL_CALIBRATION}; run SGV9 --calibrate ... --record first"
        )
    published = dict(cc._read_json(manifest)["artifacts"])
    relative = cc._relative(s9.SCORES)
    if relative not in published:
        raise PhaseError(f"{relative} is not in SGV9's provenance record")
    if file_sha256(s9.SCORES) != published[relative]:
        raise PhaseError(f"{relative} does not match the hash SGV9 recorded for it")
    frame = pd.read_parquet(
        s9.SCORES,
        columns=["held_out_engine", "block", "candidate_id", f"est__{s9.PRIMARY_MAP}"],
    )
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (engine, block), group in frame.groupby(["held_out_engine", "block"], sort=True):
        out.setdefault(str(engine), {})[str(block)] = dict(
            zip(
                group["candidate_id"].astype(str),
                group[f"est__{s9.PRIMARY_MAP}"].astype(float),
                strict=True,
            )
        )
    return out


def _sgv9_for_fold(
    state: FoldState, published: dict[str, dict[str, dict[str, float]]]
) -> dict[str, np.ndarray]:
    """Align SGV9's per-row estimates to this fold's blocks, and refuse a partial join.

    SGV9's blocks and SGV10's blocks are the same rows -- both stages rebuild the same frozen
    model on the same folds -- so a missing candidate means the two stages diverged and every
    comparison below would be between different row sets.
    """
    engine = published.get(state.held_out)
    if engine is None:
        raise PhaseError(f"SGV9 published no rows for the {state.held_out} fold")
    out: dict[str, np.ndarray] = {}
    for name in BLOCK_NAMES:
        lookup = engine.get(name)
        if lookup is None:
            raise PhaseError(f"SGV9 published no {name} block for {state.held_out}")
        identifiers = state.block(name).candidate_id
        missing = [c for c in identifiers.tolist() if c not in lookup]
        if missing:
            raise PhaseError(
                f"{state.held_out}/{name}: {len(missing)} rows are absent from SGV9's table; "
                "the two stages are not reading the same block"
            )
        out[name] = np.array([lookup[c] for c in identifiers.tolist()], dtype=float)
    return out


# ------------------------------------------------------------------ phase: --boundary


def _block_frame(
    engine: str, block_name: str, block: Block, arms: dict[str, ArmFit], state: FoldState
) -> pd.DataFrame:
    """One block's rows, with every arm's score and estimate and the band it was measured in."""
    frame = pd.DataFrame(
        {
            "held_out_engine": engine,
            "block": block_name,
            "candidate_id": block.candidate_id,
            "document_id": block.documents,
            "utility": block.utility,
            "p_harm": block.p_harm,
            "p_benefit": block.p_benefit,
            "is_harmful": block.harmful,
            "beneficial": block.beneficial,
        }
    )
    for epsilon in EPSILONS:
        key = f"epsilon_{int(epsilon * 100)}"
        band = arms[PRIMARY_ARM].bands[key]
        frame[f"band__{key}"] = band.contains(block)
        frame[f"harm_band__{key}"] = state.bands[f"source_harm|{key}"].contains(block)
        for name, arm in arms.items():
            frame[f"score__{name}__{key}"] = arm.scores[key][block_name]
            frame[f"est__{name}__{key}"] = arm.estimates[key][block_name]
    return frame


def run_boundary() -> int:
    """Rebuild the frozen model per fold, fit every band and every veto, write the row table.

    This is the only phase that touches a model. Everything downstream reads
    `boundary_scores.parquet`, which is what makes a re-analysis cheap and what makes "every
    number traces to an artifact" checkable rather than aspirational.
    """
    started = time.monotonic()
    base, signatures, retrieval_columns = s6.load_base()
    selection = s6.frozen_selection()
    frozen = s6.load_frozen_scores()
    sgv9_published = load_sgv9_estimates()
    engines = sorted(base.engines)

    frames: list[pd.DataFrame] = []
    folds: dict[str, Any] = {}
    for engine in engines:
        fold = build_fold(base, engine)
        inflation = transfer_inflation(
            base, engine, selection[engine], signatures, retrieval_columns
        )
        state = build_state(
            base, fold, selection[engine], signatures, retrieval_columns, engine, frozen
        )
        arms = fit_arms(state, EPSILONS, ALL_ARMS, _sgv9_for_fold(state, sgv9_published))
        for block_name in BLOCK_NAMES:
            frames.append(_block_frame(engine, block_name, state.block(block_name), arms, state))
        folds[engine] = {
            "identity_against_published_sgv5": state.identity,
            "selected_by_sgv5": dict(state.selected),
            "source_thresholds": state.source_thresholds,
            "source_threshold_feasible": state.source_threshold_feasible,
            "transfer_inflation": inflation,
            "diagnostics": state.diagnostics,
            "feature_names": list(state.feature_names),
            "arms": {
                name: {key: arm.diagnostics[key] for key in sorted(arm.diagnostics)}
                for name, arm in sorted(arms.items())
            },
            "risk_sets": {
                f"epsilon_{int(epsilon * 100)}": risk_sets(
                    arms[PRIMARY_ARM].estimates[f"epsilon_{int(epsilon * 100)}"]["pool"],
                    epsilon,
                    float(inflation["per_epsilon"][f"epsilon_{int(epsilon * 100)}"]["inflation"]),
                )
                for epsilon in EPSILONS
            },
        }
        band = arms[PRIMARY_ARM].bands[PRIMARY_KEY]
        print(
            f"  {engine}: {state.evaluation.size} evaluation rows, {state.pool.size} pool rows, "
            f"band [{band.lower:.4f}, {band.upper:.4f}] holding "
            f"{int(band.contains(state.evaluation).sum())} evaluation rows, inflation "
            f"{inflation['per_epsilon'][PRIMARY_KEY]['inflation']:+.4f} at "
            f"epsilon={PRIMARY_EPSILON}"
        )

    table = pd.concat(frames, ignore_index=True)
    pilot._write_parquet_once(SCORES, table)
    cc._write_json_once(
        DESIGN_RECORD,
        {
            "schema_version": "sgv10-design-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV10-T1",
            "stage": "SGV10 -- tail-risk controlled selective correction",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "pre_registration": PRE_REGISTRATION,
            "decisions": DECISIONS,
            "arms": {name: ARM_NOTES[name] for name in ALL_ARMS},
            "rules": dict(RULE_NOTES),
            "constants": {
                "band_fraction": BAND_FRACTION,
                "band_sweep": list(BAND_SWEEP),
                "harm_delta": HARM_DELTA,
                "harm_delta_sweep": list(HARM_DELTA_SWEEP),
                "min_band_rows": MIN_BAND_ROWS,
                "epsilons": list(EPSILONS),
                "cvar_alphas": list(CVAR_ALPHAS),
                "cvar_grid": CVAR_GRID,
                "tail_shares": list(TAIL_SHARES),
                "coverage_floor": COVERAGE_FLOOR,
                "veto_kinds": list(VETO_KINDS),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "n_bins": N_BINS,
                "delta": DELTA,
            },
            "folds": folds,
        },
    )
    elapsed = time.monotonic() - started
    print(f"boundary: {len(engines)} folds, {len(table)} rows -> {SCORES} ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ reading the artifacts back


@dataclass(slots=True)
class LoadedFold:
    """One fold, read back from the write-once row table. No model is rebuilt to read it."""

    held_out: str
    blocks: dict[str, Block]
    scores: dict[str, dict[str, dict[str, np.ndarray]]]
    estimates: dict[str, dict[str, dict[str, np.ndarray]]]
    bands: dict[str, dict[str, np.ndarray]]
    harm_bands: dict[str, dict[str, np.ndarray]]
    record: dict[str, Any]

    def inflation(self, key: str) -> float:
        return float(self.record["transfer_inflation"]["per_epsilon"][key]["inflation"])

    def source_threshold(self, key: str) -> float:
        return float(self.record["source_thresholds"][key])


def _block_from_frame(frame: pd.DataFrame) -> Block:
    """The stratum column is SGV9's and is unused here, so it is filled with zeros rather than
    carried; nothing in SGV10 reads it and a copied column would invite the belief that it does.
    """
    return Block(
        index=np.arange(len(frame), dtype=int),
        candidate_id=frame["candidate_id"].astype(str).to_numpy(),
        documents=frame["document_id"].astype(str).to_numpy(),
        p_harm=frame["p_harm"].to_numpy(dtype=float),
        p_benefit=frame["p_benefit"].to_numpy(dtype=float),
        utility=frame["utility"].to_numpy(dtype=float),
        stratum=np.zeros(len(frame), dtype=int),
        harmful=frame["is_harmful"].to_numpy(dtype=bool),
        beneficial=frame["beneficial"].to_numpy(dtype=bool),
    )


def load_folds() -> tuple[dict[str, LoadedFold], dict[str, Any]]:
    if not SCORES.is_file() or not DESIGN_RECORD.is_file():
        raise PhaseError("run --boundary before any phase that reads its output")
    table = pd.read_parquet(SCORES)
    record = cc._read_json(DESIGN_RECORD)
    folds: dict[str, LoadedFold] = {}
    for engine, group in table.groupby("held_out_engine", sort=True):
        blocks: dict[str, Block] = {}
        scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        estimates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        bands: dict[str, dict[str, np.ndarray]] = {}
        harm_bands: dict[str, dict[str, np.ndarray]] = {}
        for block_name, block_group in group.groupby("block", sort=True):
            frame = block_group.reset_index(drop=True)
            blocks[str(block_name)] = _block_from_frame(frame)
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                bands.setdefault(key, {})[str(block_name)] = frame[f"band__{key}"].to_numpy(bool)
                harm_bands.setdefault(key, {})[str(block_name)] = frame[
                    f"harm_band__{key}"
                ].to_numpy(bool)
                for name in ALL_ARMS:
                    scores.setdefault(key, {}).setdefault(name, {})[str(block_name)] = frame[
                        f"score__{name}__{key}"
                    ].to_numpy(dtype=float)
                    estimates.setdefault(key, {}).setdefault(name, {})[str(block_name)] = frame[
                        f"est__{name}__{key}"
                    ].to_numpy(dtype=float)
        if set(blocks) != set(BLOCK_NAMES):
            raise PhaseError(f"{engine}: the row table is missing a block ({sorted(blocks)})")
        folds[str(engine)] = LoadedFold(
            held_out=str(engine),
            blocks=blocks,
            scores=scores,
            estimates=estimates,
            bands=bands,
            harm_bands=harm_bands,
            record=record["folds"][str(engine)],
        )
    if not folds:
        raise PhaseError("the row table is empty")
    return folds, record


def _write(path: Path, payload: dict[str, Any], schema: str) -> None:
    cc._write_json_once(
        path,
        {
            "schema_version": schema,
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV10-T1",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            **payload,
        },
    )


def place_and_measure(fold: LoadedFold, arm: str, rule: str, epsilon: float) -> dict[str, Any]:
    """Place this arm's cut on the pool under this rule, then meet the evaluation block once.

    Every deployed number in this stage goes through here, so the pool-only threshold selection
    and the evaluation-only measurement cannot drift apart between phases.
    """
    key = f"epsilon_{int(epsilon * 100)}"
    pool, evaluation = fold.blocks["pool"], fold.blocks["evaluation"]
    if arm == SGV5_PUBLISHED and rule != SOURCE_THRESHOLD:
        raise PhaseError(
            "SGV5's published deployment is defined by its own threshold rule; asking it for "
            f"{rule} would report a baseline that was never published"
        )
    placed = place_cut(
        pool,
        fold.scores[key][arm]["pool"],
        fold.estimates[key][arm]["pool"],
        fold.bands[key]["pool"],
        epsilon,
        rule,
        fold.inflation(key),
        fold.source_threshold(key),
        bool(fold.record["source_threshold_feasible"][key]),
    )
    measured = deployment(
        evaluation,
        fold.scores[key][arm]["evaluation"],
        fold.estimates[key][arm]["evaluation"],
        fold.bands[key]["evaluation"],
        placed,
        epsilon,
    )
    measured["arm"] = arm
    measured["rule"] = rule
    measured["epsilon"] = float(epsilon)
    measured["held_out_engine"] = fold.held_out
    measured["pool_rows"] = pool.size
    return measured


# ------------------------------------------------------------------ phase: --analysis


def run_analysis() -> int:
    """What the boundary actually is on each engine, before any arm is judged on it.

    Section 3 of the report is written from this file. The band is a pair of scalars fitted on
    the unlabelled pool; how many rows it catches on the evaluation block, how harmful they are,
    and whether the veto's ordering of them survives the engine change are all measurements, and
    all three are the things that decide whether the rest of the stage can work at all.
    """
    started = time.monotonic()
    folds, _record = load_folds()
    out: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            cell: dict[str, Any] = {"epsilon": float(epsilon)}
            for block_name in BLOCK_NAMES:
                block = fold.blocks[block_name]
                band = fold.bands[key][block_name]
                harm_band = fold.harm_bands[key][block_name]
                inside, outside = band, ~band
                cell[block_name] = {
                    "rows": block.size,
                    "band_rows": int(band.sum()),
                    "band_share": float(band.mean()),
                    "harm_band_rows": int(harm_band.sum()),
                    "harm_band_share": float(harm_band.mean()),
                    "band_overlap_rows": int((band & harm_band).sum()),
                    "harm_prevalence_in_band": (
                        float(block.harmful[inside].mean()) if inside.any() else float("nan")
                    ),
                    "harm_prevalence_out_of_band": (
                        float(block.harmful[outside].mean()) if outside.any() else float("nan")
                    ),
                    "beneficial_in_band": int(block.beneficial[inside].sum()),
                    "beneficial_out_of_band": int(block.beneficial[outside].sum()),
                    "mean_p_harm_in_band": (
                        float(block.p_harm[inside].mean()) if inside.any() else float("nan")
                    ),
                    # The frozen decision score is heavily tied, and a band carried as two
                    # bounds must contain whole tie plateaus -- which is why a 10% rank
                    # half-width does not produce a 20% band. Limitation 3 is read from here.
                    "distinct_decision_scores": int(np.unique(block.utility).size),
                }
            evaluation = fold.blocks["evaluation"]
            band = fold.bands[key]["evaluation"]
            order = np.argsort(-evaluation.utility, kind="stable")
            positions = np.flatnonzero(band[order])
            # The composite score is a strictly increasing function of the veto's safety inside
            # the band, so its AUROC against realised harm on the band rows IS the veto's, and
            # no separate copy of the veto's output has to be carried in the row table.
            veto_auroc: dict[str, float] = {}
            for name in ALL_ARMS:
                if VETO_OF.get(name) is None or not band.any():
                    continue
                veto_auroc[name] = float(
                    roc_auc(
                        -fold.scores[key][name]["evaluation"][band],
                        evaluation.harmful[band],
                    )
                )
            reference = fold.record["arms"][PRIMARY_ARM][key]
            per_epsilon[key] = {
                **cell,
                "band": reference["pool_band"],
                "source_band": reference["source_band"],
                "reference_cut": reference["reference_cut"],
                "veto_selection": fold.record["arms"][PRIMARY_ARM][key]["veto_selection"],
                "veto_fit_diagnostics": reference["veto_diagnostics"],
                "local_estimate": reference["local_estimate"],
                "band_rank_window_on_evaluation": (
                    [int(positions.min()), int(positions.max()) + 1] if positions.size else [0, 0]
                ),
                "band_is_contiguous_in_rank": bool(
                    positions.size == 0 or positions.size == positions.max() - positions.min() + 1
                ),
                "veto_transfer_auroc_on_band": veto_auroc,
                "region_sizes_on_evaluation": {
                    "A_confident_accept": int(
                        (evaluation.utility > reference["pool_band"]["upper"]).sum()
                    ),
                    "B_boundary": int(band.sum()),
                    "C_confident_reject": int(
                        (evaluation.utility < reference["pool_band"]["lower"]).sum()
                    ),
                },
                "transfer_inflation": fold.inflation(key),
            }
        out[engine] = {
            "per_epsilon": per_epsilon,
            "diagnostics": fold.record["diagnostics"],
            "identity_against_published_sgv5": fold.record["identity_against_published_sgv5"],
        }
    _write(
        BOUNDARY_ANALYSIS,
        {
            "stage": "SGV10 -- what the deployment boundary is",
            "band_definition": {
                "primary": (
                    f"a rank half-width of {BAND_FRACTION:.0%} of the unlabelled adaptation pool "
                    "around the cut SGV5's own score and estimate reach on that pool under the "
                    "point rule, carried to the evaluation block as two decision-score bounds"
                ),
                "brief_literal": (
                    f"|p_harm - tau_h| <= {HARM_DELTA}, where tau_h is the harm probability of "
                    "the marginal accepted candidate; reported beside the primary band"
                ),
            },
            "engines": out,
            "sgv9_comparison_note": (
                "SGV9's transfer inflation on the same inner folds was +0.2670 (docTR), +0.1648 "
                "(EasyOCR), +0.2107 (PaddleOCR) and +0.2054 (Tesseract) at epsilon = 0.10, read "
                "from results/generated/sgv9/calibration_mapping.json. The SGV10 values above "
                "are the same construction applied to this stage's pipeline."
            ),
        },
        "sgv10-boundary-analysis-v1",
    )
    print(
        f"analysis: {len(folds)} folds x {len(EPSILONS)} epsilons "
        f"({time.monotonic() - started:.0f}s)"
    )
    return 0


# ------------------------------------------------------------------ phase: --tail


def run_tail() -> int:
    """Metric 2. The harm rate in the riskiest part of what each arm accepts."""
    started = time.monotonic()
    folds, _ = load_folds()
    out: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        evaluation = fold.blocks["evaluation"]
        cells: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            for arm in ALL_ARMS:
                rules = (SOURCE_THRESHOLD,) if arm == SGV5_PUBLISHED else RULES
                for rule in rules:
                    measured = place_and_measure(fold, arm, rule, epsilon)
                    cells[f"{arm}|{rule}|{key}"] = {
                        "arm": arm,
                        "rule": rule,
                        "epsilon": float(epsilon),
                        "tail": tail_harm(
                            evaluation,
                            fold.scores[key][arm]["evaluation"],
                            fold.estimates[key][arm]["evaluation"],
                            float(measured["tau"]),
                        ),
                        "realized_cvar": measured["realized_cvar"],
                        "overall": {
                            "repair_recall": measured["repair_recall"],
                            "realized_harm_rate": measured["realized_harm_rate"],
                            "coverage": measured["coverage"],
                            "holds_bound": measured["holds_bound"],
                            "accepts_nothing": measured["accepts_nothing"],
                        },
                    }
        out[engine] = cells
    _write(
        TAIL_RISK_RESULTS,
        {
            "stage": "SGV10 -- tail harm among accepted corrections",
            "tail_shares": list(TAIL_SHARES),
            "criterion_3_reads": (
                "tail.marginal -- the last share of the accept set by the arm's own decision "
                "score. Fixed before any number was computed, because it is the only tail "
                "defined identically for every arm whatever its risk estimate looks like."
            ),
            "engines": out,
        },
        "sgv10-tail-risk-v1",
    )
    print(f"tail: {len(folds)} folds x {len(ALL_ARMS)} arms ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --coverage

# The multiplicity family, fixed before any endpoint was computed: the three proposed arms
# against SGV5's published deployment, on the primary endpoint, at the primary epsilon and under
# the primary rule, on each of the four unseen engines. Twelve tests, Holm-adjusted.
COVERAGE_FAMILY = "3 proposed arms x 4 engines, paired repair-recall delta against SGV5"


def _recall_draws(draws: dict[str, Any], n_resamples: int) -> np.ndarray:
    """The resample distribution of repair recall, with the accept-nothing case made explicit.

    `resampled_deployment` omits the draws entirely when the threshold accepts nothing, because
    there is no operating point to describe. For a PAIRED difference the arm still has to
    contribute a value in every resample, and that value is zero: an arm that accepts nothing
    repairs nothing, in this batch of pages and in every other.
    """
    if "recall_draws" not in draws:
        return np.zeros(n_resamples, dtype=float)
    return np.asarray(draws["recall_draws"], dtype=float)


def _criterion_one(measured: dict[str, Any], baseline: dict[str, Any], epsilon: float) -> bool:
    """Recall improves AND the realised harm of what the arm accepts is within epsilon.

    An arm that accepts nothing earns no credit: its repair recall is zero, which cannot beat a
    baseline that accepts anything, and its harm rate is undefined rather than zero.
    """
    if measured["accepts_nothing"]:
        return False
    return bool(
        measured["repair_recall"] > baseline["repair_recall"]
        and measured["realized_harm_rate"] <= epsilon
    )


def run_coverage() -> int:
    """Metric 1. Every arm, every cut rule, every epsilon, with the paired interval and Holm.

    The threshold is placed on the unlabelled pool and then held fixed inside every resample:
    an operator carries one number to the next batch of pages, so the question a violation rate
    has to answer is what THAT number does on a different sample of the same engine, not what a
    freshly re-placed one would do.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    engines = sorted(folds)
    cells: dict[str, Any] = {}
    family: dict[str, float] = {}
    for engine in engines:
        fold = folds[engine]
        evaluation = fold.blocks["evaluation"]
        frozen_frontier = achievable_repair_recall(
            evaluation.utility, evaluation.harmful, evaluation.beneficial
        )
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            baseline_measured = place_and_measure(fold, SGV5_PUBLISHED, SOURCE_THRESHOLD, epsilon)
            baseline_curves = resample_curves(
                replace(evaluation, utility=fold.scores[key][SGV5_PUBLISHED]["evaluation"]),
                BOOTSTRAP_SEED,
                BOOTSTRAP_RESAMPLES,
            )
            baseline_draws = resampled_deployment(
                baseline_curves, float(baseline_measured["tau"]), epsilon
            )
            arms: dict[str, Any] = {}
            for arm in ALL_ARMS:
                curves = resample_curves(
                    replace(evaluation, utility=fold.scores[key][arm]["evaluation"]),
                    BOOTSTRAP_SEED,
                    BOOTSTRAP_RESAMPLES,
                )
                rules = (SOURCE_THRESHOLD,) if arm == SGV5_PUBLISHED else RULES
                for rule in rules:
                    measured = place_and_measure(fold, arm, rule, epsilon)
                    draws = resampled_deployment(curves, float(measured["tau"]), epsilon)
                    delta = _interval(
                        _recall_draws(draws, BOOTSTRAP_RESAMPLES)
                        - _recall_draws(baseline_draws, BOOTSTRAP_RESAMPLES)
                    )
                    resampled = {k: v for k, v in draws.items() if k != "recall_draws"}
                    arms[f"{arm}|{rule}"] = {
                        **measured,
                        "resampled": resampled,
                        "paired_delta_vs_sgv5": delta,
                        "meets_criterion_1": _criterion_one(measured, baseline_measured, epsilon),
                        "coverage_floor_met": bool(
                            measured["n_accepted"]
                            >= COVERAGE_FLOOR * max(baseline_measured["n_accepted"], 1)
                        ),
                        "frozen_ranking_frontier": frozen_frontier[key]["repair_recall"],
                    }
                    if (
                        arm in PROPOSED
                        and rule == PRIMARY_RULE
                        and abs(epsilon - PRIMARY_EPSILON) < 1e-12
                    ):
                        family[f"{engine}|{arm}"] = float(delta["p_value"])
            per_epsilon[key] = {
                "epsilon": float(epsilon),
                "coverage_risk_tradeoff": _tradeoff(fold, key, epsilon),
                "baseline": baseline_measured,
                "baseline_resampled": {
                    k: v for k, v in baseline_draws.items() if k != "recall_draws"
                },
                "frozen_ranking_frontier": frozen_frontier[key],
                "max_inflation_that_still_deploys": _max_affordable(fold, key, epsilon),
                "estimated_inflation": fold.inflation(key),
                "arms": arms,
            }
        cells[engine] = per_epsilon
    from ocr_risk.stats.multiplicity import holm_bonferroni

    adjusted = holm_bonferroni(family) if family else []
    _write(
        RISK_COVERAGE,
        {
            "stage": "SGV10 -- risk-controlled repair recall",
            "family": COVERAGE_FAMILY,
            "primary_arm": PRIMARY_ARM,
            "primary_rule": PRIMARY_RULE,
            "safety_rule": SAFETY_RULE,
            "criterion_1": (
                "repair recall exceeds SGV5's published deployment AND the realised harm rate "
                "of the arm's accepted set is within epsilon. An arm that accepts nothing earns "
                "no credit."
            ),
            "coverage_floor": COVERAGE_FLOOR,
            "bootstrap": {
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": "document",
                "threshold": "held fixed inside every resample",
            },
            "holm": [
                {
                    "label": test.label,
                    "p_value": test.p_value,
                    "adjusted_p_value": test.adjusted_p_value,
                    "significant": test.significant,
                }
                for test in adjusted
            ],
            "engines": cells,
        },
        "sgv10-risk-coverage-v1",
    )
    print(
        f"coverage: {len(engines)} folds x {len(ALL_ARMS)} arms x {len(RULES)} rules x "
        f"{len(EPSILONS)} epsilons ({time.monotonic() - started:.0f}s)"
    )
    return 0


def _tradeoff(fold: LoadedFold, key: str, epsilon: float) -> dict[str, Any]:
    """Method C's coverage-risk tradeoff, from `ocr_risk.metrics.selective`.

    The brief requires Method C to expose a tradeoff rather than a single point. The curve is
    the arm's whole frontier on the evaluation block -- a measurement, with hindsight, of what
    every threshold on that ordering would have done -- and the deployed points are marked on
    it so a reader can see how far the label-free rules land from the frontier they are aiming
    at. The curve itself is never used to choose a threshold.
    """
    evaluation = fold.blocks["evaluation"]
    out: dict[str, Any] = {}
    for arm in (RULE_ONLY, PRIMARY_ARM, NO_BOUNDARY):
        curve = risk_coverage_curve(fold.scores[key][arm]["evaluation"], evaluation.harmful)
        deployed = {
            rule: place_and_measure(fold, arm, rule, epsilon) for rule in (POINT_CUT, BOUND_CUT)
        }
        out[arm] = {
            "aurc": float(aurc(curve)),
            "curve": curve.as_dict(),
            "deployed": {
                rule: {
                    "coverage": point["coverage"],
                    "realized_harm_rate": point["realized_harm_rate"],
                    "repair_recall": point["repair_recall"],
                    "accepts_nothing": point["accepts_nothing"],
                }
                for rule, point in deployed.items()
            },
            "risk_sets": risk_sets(fold.estimates[key][arm]["pool"], epsilon, fold.inflation(key)),
        }
    return out


def _max_affordable(fold: LoadedFold, key: str, epsilon: float) -> float:
    """The largest transfer inflation the bound rule could absorb and still accept one row.

    Reported beside the inflation this stage actually estimated, so "the bound refuses" can be
    read as a magnitude rather than as a verdict. SGV9 reported the same pair and found the
    estimate 1.0 to 6.9 times the affordable value.
    """
    pool = fold.blocks["pool"]
    scored = replace(pool, utility=fold.scores[key][PRIMARY_ARM]["pool"])
    curves = prefix_curves(scored, fold.estimates[key][PRIMARY_ARM]["pool"])
    slack = epsilon - harm_upper_bound(curves, 0.0)
    return float(slack.max()) if slack.size else float("nan")


# ------------------------------------------------------------------ phase: --abstention


def run_abstention() -> int:
    """Metric 4. What abstention removed, what it cost, and what it let in instead.

    Measured against SGV5's published accept set, which is the set an operator would actually
    have. The matched-coverage random control is reported in the same cell, because "abstaining
    helps" and "abstaining on the right rows helps" are different claims and only the second is
    the stage's.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    out: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        evaluation = fold.blocks["evaluation"]
        cells: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            baseline = place_and_measure(fold, SGV5_PUBLISHED, SOURCE_THRESHOLD, epsilon)
            baseline_score = fold.scores[key][SGV5_PUBLISHED]["evaluation"]
            for arm in ALL_ARMS:
                if arm == SGV5_PUBLISHED:
                    continue
                for rule in RULES:
                    measured = place_and_measure(fold, arm, rule, epsilon)
                    cells[f"{arm}|{rule}|{key}"] = {
                        "arm": arm,
                        "rule": rule,
                        "epsilon": float(epsilon),
                        **abstention_analysis(
                            evaluation,
                            fold.scores[key][arm]["evaluation"],
                            float(measured["tau"]),
                            baseline_score,
                            float(baseline["tau"]),
                        ),
                        "repair_recall": measured["repair_recall"],
                        "realized_harm_rate": measured["realized_harm_rate"],
                        "holds_bound": measured["holds_bound"],
                        "accepts_nothing": measured["accepts_nothing"],
                    }
            cells[f"baseline|{key}"] = {
                "n_accepted": baseline["n_accepted"],
                "repair_recall": baseline["repair_recall"],
                "realized_harm_rate": baseline["realized_harm_rate"],
                "holds_bound": baseline["holds_bound"],
            }
        out[engine] = cells
    _write(
        ABSTENTION_ANALYSIS,
        {
            "stage": "SGV10 -- abstention efficiency",
            "reference_set": (
                "SGV5's published deployment at the same epsilon. An arm that reorders both "
                "rejects some of the incumbent's accepts and accepts some of its rejects, so "
                "the whole two-by-two is recorded rather than the rejection half alone."
            ),
            "headline_ratio": "harm_removed_per_rejection = harmful removed / rows removed",
            "engines": out,
        },
        "sgv10-abstention-v1",
    )
    elapsed = time.monotonic() - started
    arms = len(ALL_ARMS) - 1
    print(f"abstention: {len(folds)} folds x {arms} arms x {len(RULES)} rules ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --oracle


def run_oracle() -> int:
    """Metric 5. The gap to the two oracles, to each arm's own frontier, and to the frozen one.

    Four references. The perfect hindsight accept set -- every beneficial candidate and no
    harmful one -- is repair recall 1.000 at harm 0.000 on every fold by definition; it is
    stated once, as arithmetic, and is not used as a gap anyone could close.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    out: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        evaluation = fold.blocks["evaluation"]
        frozen = achievable_repair_recall(
            evaluation.utility, evaluation.harmful, evaluation.beneficial
        )
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            frontiers = {
                arm: achievable_repair_recall(
                    fold.scores[key][arm]["evaluation"], evaluation.harmful, evaluation.beneficial
                )[key]
                for arm in ALL_ARMS
            }
            deployed = {
                f"{arm}|{rule}": place_and_measure(fold, arm, rule, epsilon)
                for arm in ALL_ARMS
                for rule in ((SOURCE_THRESHOLD,) if arm == SGV5_PUBLISHED else RULES)
            }
            gaps: dict[str, Any] = {}
            for name, measured in deployed.items():
                arm = name.split("|")[0]
                gaps[name] = {
                    "repair_recall": measured["repair_recall"],
                    "holds_bound": measured["holds_bound"],
                    "accepts_nothing": measured["accepts_nothing"],
                    "gap_to_own_frontier": (
                        frontiers[arm]["repair_recall"] - measured["repair_recall"]
                    ),
                    "gap_to_frozen_frontier": (
                        frozen[key]["repair_recall"] - measured["repair_recall"]
                    ),
                    "gap_to_oracle_pool": (
                        frontiers[ORACLE_POOL]["repair_recall"] - measured["repair_recall"]
                    ),
                    "gap_to_oracle_eval": (
                        frontiers[ORACLE_EVAL]["repair_recall"] - measured["repair_recall"]
                    ),
                    "gap_to_perfect_hindsight": 1.0 - measured["repair_recall"],
                }
            per_epsilon[key] = {
                "epsilon": float(epsilon),
                "frozen_ranking_frontier": frozen[key],
                "arm_frontiers": frontiers,
                "deployed": gaps,
                "perfect_hindsight": {
                    "repair_recall": 1.0,
                    "realized_harm_rate": 0.0,
                    "note": (
                        "accept every beneficial candidate and no harmful one. True on every "
                        "fold by definition and reachable by no procedure; reported so that a "
                        "reader can see which gaps are informative and which are arithmetic."
                    ),
                },
            }
        out[engine] = per_epsilon
    _write(
        ORACLE_GAP,
        {
            "stage": "SGV10 -- the oracle gap",
            "references": {
                "frozen_ranking_frontier": (
                    "the deepest prefix of SGV5's own ordering whose realised harm holds "
                    "epsilon. The ceiling SGV9 lived under, and the one a reordering method can "
                    "legitimately pass."
                ),
                "own_frontier": (
                    "the deepest prefix of THIS arm's ordering whose realised harm holds "
                    "epsilon. What the arm's reordering makes available with a perfect cut."
                ),
                "oracle_pool": "a veto fitted on the held-out engine's own TRAIN labels",
                "oracle_eval": "a veto fitted on the evaluation block itself",
            },
            "engines": out,
        },
        "sgv10-oracle-gap-v1",
    )
    print(f"oracle: {len(folds)} folds ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --calibration


def run_calibration() -> int:
    """Metric 3. Calibration error near the deployment threshold, against the global number.

    The comparison the brief asks for -- global ECE against boundary ECE -- and the comparison
    against SGV9, which is the same evaluation rows scored by SGV9's published primary map.
    """
    started = time.monotonic()
    folds, _ = load_folds()
    out: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        evaluation = fold.blocks["evaluation"]
        per_epsilon: dict[str, Any] = {}
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            band = fold.bands[key]["evaluation"]
            harm_band = fold.harm_bands[key]["evaluation"]
            arms: dict[str, Any] = {}
            for arm in ALL_ARMS:
                estimates = fold.estimates[key][arm]["evaluation"]
                report = boundary_calibration(evaluation, estimates, band)
                report["harm_scale_band"] = boundary_calibration(evaluation, estimates, harm_band)[
                    "boundary"
                ]
                report["estimate_source"] = ESTIMATE_OF[arm]
                arms[arm] = report
            per_epsilon[key] = {"epsilon": float(epsilon), "arms": arms}
        out[engine] = per_epsilon
    _write(
        CALIBRATION_COMPARISON,
        {
            "stage": "SGV10 -- boundary calibration against global calibration",
            "definition": (
                "the same calibration report, computed over every evaluation row and then over "
                "the band rows alone. Brier is primary; ECE is reported under both equal-width "
                "and equal-mass binning, as `.claude/rules/metrics.md` requires."
            ),
            "sgv9_arm": GLOBAL_CALIBRATION,
            "sgv9_note": (
                "`aB_global_calibration` carries SGV9's published primary map "
                f"(`{s9.PRIMARY_MAP}`) as its estimate, read from SGV9's own table and verified "
                "against the hash SGV9's provenance manifest records. The comparison against it "
                "is therefore a comparison against the previous stage's published result, not "
                "against a re-implementation."
            ),
            "n_bins": N_BINS,
            "engines": out,
        },
        "sgv10-calibration-comparison-v1",
    )
    elapsed = time.monotonic() - started
    print(f"calibration: {len(folds)} folds x {len(ALL_ARMS)} arms ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --ablation

ABLATIONS = {
    "A_sgv5_only": {
        "arm": RULE_ONLY,
        "removes": "the veto and the boundary-local estimate; SGV5's score and SGV5's estimate",
        "question": "what does the cut rule alone buy or cost, before any tail-risk control?",
    },
    "B_global_calibration": {
        "arm": GLOBAL_CALIBRATION,
        "removes": "the boundary-local estimate, replaced by SGV9's published global map",
        "question": "does recalibrating globally, as SGV9 did, do what recalibrating locally does?",
    },
    "C_no_boundary_filtering": {
        "arm": NO_BOUNDARY,
        "removes": "the band, so the veto reorders every row rather than the boundary alone",
        "question": "is confining the veto to the boundary what makes it work?",
    },
    "D_random_abstention": {
        "arm": RANDOM_ABSTENTION,
        "removes": "the veto's information, keeping its band and its coverage",
        "question": "is the gain from abstaining, or from abstaining on the right rows?",
    },
    "E_boundary_features_only": {
        "arm": BOUNDARY_FEATURES,
        "removes": "SGV5's representation columns, leaving the local features alone",
        "question": "are the global features necessary for the veto?",
    },
    "F_oracle_boundary_model": {
        "arm": ORACLE_POOL,
        "removes": "nothing; it ADDS the target engine's own labels",
        "question": "what is the ceiling of the veto class on this engine?",
    },
    "F_oracle_evaluation": {
        "arm": ORACLE_EVAL,
        "removes": "nothing; it ADDS the evaluation block's own labels",
        "question": "what is the ceiling of the veto class outright?",
    },
}


def run_ablation() -> int:
    """The brief's six ablations, each read against the primary arm on the same rows."""
    started = time.monotonic()
    folds, _ = load_folds()
    out: dict[str, Any] = {}
    for name, spec in ABLATIONS.items():
        arm = str(spec["arm"])
        engines: dict[str, Any] = {}
        for engine, fold in sorted(folds.items()):
            evaluation = fold.blocks["evaluation"]
            per_epsilon: dict[str, Any] = {}
            for epsilon in EPSILONS:
                key = f"epsilon_{int(epsilon * 100)}"
                band = fold.bands[key]["evaluation"]
                primary = place_and_measure(fold, PRIMARY_ARM, PRIMARY_RULE, epsilon)
                measured = place_and_measure(fold, arm, PRIMARY_RULE, epsilon)
                primary_cal = boundary_calibration(
                    evaluation, fold.estimates[key][PRIMARY_ARM]["evaluation"], band
                )
                cal = boundary_calibration(evaluation, fold.estimates[key][arm]["evaluation"], band)
                changed = int(
                    np.count_nonzero(
                        fold.scores[key][arm]["evaluation"]
                        != fold.scores[key][PRIMARY_ARM]["evaluation"]
                    )
                )
                per_epsilon[key] = {
                    "epsilon": float(epsilon),
                    "repair_recall": measured["repair_recall"],
                    "realized_harm_rate": measured["realized_harm_rate"],
                    "holds_bound": measured["holds_bound"],
                    "accepts_nothing": measured["accepts_nothing"],
                    "delta_repair_recall_vs_primary": (
                        measured["repair_recall"] - primary["repair_recall"]
                    ),
                    "delta_band_brier_vs_primary": (
                        cal["boundary"]["brier"] - primary_cal["boundary"]["brier"]
                        if cal["boundary"] and primary_cal["boundary"]
                        else float("nan")
                    ),
                    "delta_global_brier_vs_primary": (
                        cal["global"]["brier"] - primary_cal["global"]["brier"]
                    ),
                    "band_auroc": (
                        float(
                            roc_auc(
                                -fold.scores[key][arm]["evaluation"][band], evaluation.harmful[band]
                            )
                        )
                        if band.any()
                        else float("nan")
                    ),
                    "rows_scored_differently_from_primary": changed,
                    "own_frontier": achievable_repair_recall(
                        fold.scores[key][arm]["evaluation"],
                        evaluation.harmful,
                        evaluation.beneficial,
                    )[key]["repair_recall"],
                }
            engines[engine] = per_epsilon
        out[name] = {**spec, "engines": engines}
    _write(
        ABLATION_RESULTS,
        {
            "stage": "SGV10 -- ablations",
            "read_against": f"{PRIMARY_ARM} under {PRIMARY_RULE}, on the same evaluation rows",
            "ablations": out,
        },
        "sgv10-ablation-v1",
    )
    elapsed = time.monotonic() - started
    print(f"ablation: {len(ABLATIONS)} ablations x {len(folds)} folds ({elapsed:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --negative


def _matched_coverage(
    block: Block, score: np.ndarray, n_accepted: int, epsilon: float
) -> dict[str, Any]:
    """What this arm's ordering delivers when it is forced to accept the same NUMBER of rows.

    Negative test 2 asks whether an improvement is only the effect of accepting less. Holding
    the accepted count equal to the incumbent's removes that channel entirely: whatever is left
    is the ordering, and nothing else.
    """
    if n_accepted <= 0:
        return {"n_accepted": 0, "repair_recall": 0.0, "realized_harm_rate": float("nan")}
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:n_accepted]
    total = int(block.beneficial.sum())
    return {
        "n_accepted": int(n_accepted),
        "repair_recall": float(block.beneficial[order].sum() / max(total, 1)),
        "realized_harm_rate": float(block.harmful[order].mean()),
        "holds_bound": bool(block.harmful[order].mean() <= epsilon),
    }


def _band_sweep() -> dict[str, Any]:
    """Negative test 3. Refit the band, the veto and the estimate at other boundary definitions.

    This is the ONE phase downstream of `--boundary` that touches a model, because a different
    band is a different training set for the veto and cannot be recovered from the saved table.
    The primary arm, the incumbent and the random control are refitted at each definition; the
    rest of the family is not, because the question is whether the METHOD survives a change of
    boundary, not whether every ablation does.
    """
    base, signatures, retrieval_columns = s6.load_base()
    selection = s6.frozen_selection()
    frozen = s6.load_frozen_scores()
    swept: dict[str, Any] = {}
    definitions = [(RANK_BAND, width, HARM_DELTA) for width in BAND_SWEEP]
    definitions += [(HARM_BAND, BAND_FRACTION, delta) for delta in HARM_DELTA_SWEEP]
    for engine in sorted(base.engines):
        fold = build_fold(base, engine)
        state = build_state(
            base, fold, selection[engine], signatures, retrieval_columns, engine, frozen
        )
        evaluation, pool = state.evaluation, state.pool
        for kind, width, delta in definitions:
            label = f"{kind}|{width}|{delta}" if kind == RANK_BAND else f"{kind}|{delta}"
            arms = fit_arms(
                state,
                (PRIMARY_EPSILON,),
                (PRIMARY_ARM, RULE_ONLY, RANDOM_ABSTENTION),
                None,
                kind,
                width,
                delta,
            )
            cell: dict[str, Any] = {
                "band_kind": kind,
                "rank_half_width": width,
                "harm_delta": delta,
            }
            for name, arm in arms.items():
                band = arm.bands[PRIMARY_KEY]
                membership = band.contains(evaluation)
                placed = place_cut(
                    pool,
                    arm.scores[PRIMARY_KEY]["pool"],
                    arm.estimates[PRIMARY_KEY]["pool"],
                    band.contains(pool),
                    PRIMARY_EPSILON,
                    PRIMARY_RULE,
                    0.0,
                    state.source_thresholds[PRIMARY_KEY],
                    state.source_threshold_feasible[PRIMARY_KEY],
                )
                measured = deployment(
                    evaluation,
                    arm.scores[PRIMARY_KEY]["evaluation"],
                    arm.estimates[PRIMARY_KEY]["evaluation"],
                    membership,
                    placed,
                    PRIMARY_EPSILON,
                )
                cell[name] = {
                    "band_rows_on_evaluation": int(membership.sum()),
                    "repair_recall": measured["repair_recall"],
                    "realized_harm_rate": measured["realized_harm_rate"],
                    "holds_bound": measured["holds_bound"],
                    "accepts_nothing": measured["accepts_nothing"],
                    "own_frontier": measured["own_frontier"]["repair_recall"],
                    "band_auroc": (
                        float(
                            roc_auc(
                                -arm.scores[PRIMARY_KEY]["evaluation"][membership],
                                evaluation.harmful[membership],
                            )
                        )
                        if membership.any()
                        else float("nan")
                    ),
                }
            swept.setdefault(engine, {})[label] = cell
    return swept


def run_negative() -> int:
    """The five negative tests the brief requires, each with the artifact it is read from."""
    started = time.monotonic()
    folds, _ = load_folds()
    engines = sorted(folds)

    refuses: dict[str, Any] = {}
    lower_coverage: dict[str, Any] = {}
    transfers: dict[str, Any] = {}
    metrics_not_harm: dict[str, Any] = {}
    for engine in engines:
        fold = folds[engine]
        evaluation = fold.blocks["evaluation"]
        for epsilon in EPSILONS:
            key = f"epsilon_{int(epsilon * 100)}"
            band = fold.bands[key]["evaluation"]
            baseline = place_and_measure(fold, SGV5_PUBLISHED, SOURCE_THRESHOLD, epsilon)
            frozen_cal = boundary_calibration(
                evaluation, fold.estimates[key][RULE_ONLY]["evaluation"], band
            )
            for arm in ALL_ARMS:
                if arm == SGV5_PUBLISHED:
                    continue
                for rule in RULES:
                    measured = place_and_measure(fold, arm, rule, epsilon)
                    label = f"{engine}|{arm}|{rule}|{key}"
                    refuses[label] = {
                        "n_accepted": measured["n_accepted"],
                        "coverage": measured["coverage"],
                        "accepts_nothing": measured["accepts_nothing"],
                        "coverage_share_of_baseline": (
                            float(measured["n_accepted"] / max(baseline["n_accepted"], 1))
                        ),
                        "coverage_floor_met": bool(
                            measured["n_accepted"]
                            >= COVERAGE_FLOOR * max(baseline["n_accepted"], 1)
                        ),
                    }
                    lower_coverage[label] = {
                        "deployed": {
                            "n_accepted": measured["n_accepted"],
                            "repair_recall": measured["repair_recall"],
                            "realized_harm_rate": measured["realized_harm_rate"],
                        },
                        "baseline": {
                            "n_accepted": baseline["n_accepted"],
                            "repair_recall": baseline["repair_recall"],
                            "realized_harm_rate": baseline["realized_harm_rate"],
                        },
                        "matched_coverage": _matched_coverage(
                            evaluation,
                            fold.scores[key][arm]["evaluation"],
                            int(baseline["n_accepted"]),
                            epsilon,
                        ),
                    }
                    cal = boundary_calibration(
                        evaluation, fold.estimates[key][arm]["evaluation"], band
                    )
                    metrics_not_harm[label] = {
                        "delta_band_brier_vs_frozen": (
                            cal["boundary"]["brier"] - frozen_cal["boundary"]["brier"]
                            if cal["boundary"] and frozen_cal["boundary"]
                            else float("nan")
                        ),
                        "delta_global_brier_vs_frozen": (
                            cal["global"]["brier"] - frozen_cal["global"]["brier"]
                        ),
                        "realized_harm_rate": measured["realized_harm_rate"],
                        "baseline_realized_harm_rate": baseline["realized_harm_rate"],
                        "harm_improved": bool(
                            not measured["accepts_nothing"]
                            and measured["realized_harm_rate"] < baseline["realized_harm_rate"]
                        ),
                    }
            record = fold.record["arms"][PRIMARY_ARM][key]
            transfers[f"{engine}|{key}"] = {
                "in_domain_mean_auroc": record["veto_selection"]["mean_auroc"],
                "in_domain_per_pseudo_target": record["veto_selection"]["per_pseudo_target"],
                "unseen_engine_band_auroc": (
                    float(
                        roc_auc(
                            -fold.scores[key][PRIMARY_ARM]["evaluation"][band],
                            evaluation.harmful[band],
                        )
                    )
                    if band.any()
                    else float("nan")
                ),
                "random_control_band_auroc": (
                    float(
                        roc_auc(
                            -fold.scores[key][RANDOM_ABSTENTION]["evaluation"][band],
                            evaluation.harmful[band],
                        )
                    )
                    if band.any()
                    else float("nan")
                ),
            }

    _write(
        NEGATIVE_TESTS,
        {
            "stage": "SGV10 -- the five negative tests",
            "test_1_rejects_everything": {
                "question": "does the method simply reject everything?",
                "reads": "the accepted count and its share of SGV5's, in every cell",
                "cells_accepting_nothing": sorted(
                    label for label, cell in refuses.items() if cell["accepts_nothing"]
                ),
                "n_cells": len(refuses),
                "per_cell": refuses,
            },
            "test_2_only_lower_coverage": {
                "question": "does the improvement come only from lower coverage?",
                "reads": (
                    "the same arm's ordering forced to accept the SAME number of rows as SGV5's "
                    "published deployment. Whatever survives that is the ordering and nothing "
                    "else."
                ),
                "per_cell": lower_coverage,
            },
            "test_3_boundary_region_changes": {
                "question": "does it work when the boundary region changes?",
                "reads": (
                    "the band, the veto and the local estimate refitted at every boundary "
                    f"definition: rank half-widths {list(BAND_SWEEP)} and the brief's literal "
                    f"harm-probability band at deltas {list(HARM_DELTA_SWEEP)}"
                ),
                "per_engine": _band_sweep(),
            },
            "test_4_transfers_to_unseen_engines": {
                "question": "does the veto transfer to unseen engines?",
                "reads": (
                    "the veto's AUROC on the held-out engine's band rows, against its AUROC on "
                    "the source engines it did not fit from, and against the random control"
                ),
                "per_cell": transfers,
            },
            "test_5_metrics_not_harm": {
                "question": "does it improve actual harm rather than only calibration metrics?",
                "reads": (
                    "the change in boundary Brier beside the change in the realised harm rate "
                    "of what is deployed, in the same cell"
                ),
                "per_cell": metrics_not_harm,
            },
        },
        "sgv10-negative-tests-v1",
    )
    print(f"negative: 5 tests written ({time.monotonic() - started:.0f}s)")
    return 0


# ------------------------------------------------------------------ phase: --figures


def run_figures() -> int:
    """Four figures, each carrying the development-only annotation the project's rules require."""
    started = time.monotonic()
    analysis = cc._read_json(BOUNDARY_ANALYSIS)
    coverage = cc._read_json(RISK_COVERAGE)
    calibration = cc._read_json(CALIBRATION_COMPARISON)
    tail = cc._read_json(TAIL_RISK_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV10 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(coverage["engines"])
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    shown = {
        RULE_ONLY: ("#444", "x", "A: SGV5 score, SGV10 rule"),
        GLOBAL_CALIBRATION: ("#c87a2b", "v", "B: SGV9 global calibration"),
        BOUNDARY_VETO: ("#3a5f9e", "o", "Method A: boundary veto"),
        TAIL_CONDITIONAL: ("#2e7d5b", "^", "Method B: tail conditional"),
        HYBRID: ("#a33", "D", "Method E: hybrid (primary)"),
        NO_BOUNDARY: ("#8a5fa3", "s", "C: veto, unconfined"),
        RANDOM_ABSTENTION: ("#999", "+", "D: random abstention"),
        ORACLE_EVAL: ("#777", "*", "oracle (evaluation labels)"),
    }

    # --- risk_coverage.png -------------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.2))
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        cell = coverage["engines"][engine][PRIMARY_KEY]
        panel.axhline(PRIMARY_EPSILON, color="#bbb", linestyle="--", linewidth=1)
        for arm, (colour, marker, label) in shown.items():
            point = cell["arms"].get(f"{arm}|{PRIMARY_RULE}")
            if point is None or point["accepts_nothing"]:
                continue
            panel.scatter(
                point["repair_recall"],
                point["realized_harm_rate"],
                color=colour,
                marker=marker,
                s=46,
                label=label,
            )
        base = cell["baseline"]
        panel.scatter(
            base["repair_recall"],
            base["realized_harm_rate"],
            color="#000",
            marker="P",
            s=60,
            label="baseline 0: SGV5 published",
        )
        panel.axvline(
            cell["frozen_ranking_frontier"]["repair_recall"],
            color="#7a9",
            linestyle=":",
            linewidth=1.2,
        )
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("repair recall at the deployed cut")
        panel.set_ylabel(f"realised harm rate (epsilon = {PRIMARY_EPSILON})")
        panel.set_xlim(-0.02, 1.02)
    np.atleast_1d(panels)[0].legend(fontsize=6, loc="upper left")
    finish(
        figure,
        FIGURE_DIR / "risk_coverage.png",
        "Deployed operating points. The dotted vertical line is the frozen ranking's "
        "achievable frontier, which a reordering arm may legitimately pass.",
    )

    # --- boundary_calibration.png ------------------------------------------------------------
    figure, panels = plt.subplots(1, 2, figsize=(10.5, 4.4))
    width = 0.35
    positions = np.arange(len(engines), dtype=float)
    for index, arm in enumerate((RULE_ONLY, HYBRID)):
        globals_ = [
            calibration["engines"][e][PRIMARY_KEY]["arms"][arm]["global"]["brier"] for e in engines
        ]
        bands = [
            calibration["engines"][e][PRIMARY_KEY]["arms"][arm]["boundary"]["brier"]
            for e in engines
        ]
        panels[0].bar(positions + index * width, globals_, width, label=shown[arm][2])
        panels[1].bar(positions + index * width, bands, width, label=shown[arm][2])
    for panel, title in zip(
        panels, ("every evaluation row", "the boundary band alone"), strict=True
    ):
        panel.set_xticks(positions + width / 2)
        panel.set_xticklabels(engines, fontsize=8)
        panel.set_ylabel("Brier score")
        panel.set_title(title, fontsize=10)
        panel.legend(fontsize=7)
    finish(
        figure,
        FIGURE_DIR / "boundary_calibration.png",
        "Calibration error is two to four times larger inside the boundary band than over the "
        "whole block, on every engine.",
    )

    # --- boundary_composition.png ------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.0))
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        regions = analysis["engines"][engine]["per_epsilon"][PRIMARY_KEY][
            "region_sizes_on_evaluation"
        ]
        labels = ["A: confident\naccept", "B: boundary", "C: confident\nreject"]
        counts = [
            regions["A_confident_accept"],
            regions["B_boundary"],
            regions["C_confident_reject"],
        ]
        panel.bar(labels, counts, color=["#2e7d5b", "#a33", "#444"])
        block = analysis["engines"][engine]["per_epsilon"][PRIMARY_KEY]["evaluation"]
        panel.set_title(
            f"held out: {engine}\nharm in band {block['harm_prevalence_in_band']:.2f}, "
            f"outside {block['harm_prevalence_out_of_band']:.2f}",
            fontsize=9,
        )
        panel.set_ylabel("evaluation rows")
    finish(
        figure,
        FIGURE_DIR / "boundary_composition.png",
        "The three regions the band induces on the unseen engine, and how harmful each is.",
    )

    # --- tail_harm.png -----------------------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.0))
    shares = [f"share_{int(s * 100)}" for s in TAIL_SHARES]
    for panel, engine in zip(np.atleast_1d(panels), engines, strict=True):
        cells = tail["engines"][engine]
        base = cells[f"{SGV5_PUBLISHED}|{SOURCE_THRESHOLD}|{PRIMARY_KEY}"]["tail"]
        for arm, (colour, marker, label) in shown.items():
            cell = cells.get(f"{arm}|{PRIMARY_RULE}|{PRIMARY_KEY}")
            if cell is None or cell["tail"]["accepts_nothing"]:
                continue
            panel.plot(
                [s * 100 for s in TAIL_SHARES],
                [cell["tail"]["marginal"][s]["harm_rate"] for s in shares],
                marker=marker,
                color=colour,
                linewidth=1.2,
                markersize=4,
                label=label,
            )
        if not base["accepts_nothing"]:
            panel.plot(
                [s * 100 for s in TAIL_SHARES],
                [base["marginal"][s]["harm_rate"] for s in shares],
                marker="P",
                color="#000",
                linewidth=1.4,
                markersize=5,
                label="baseline 0: SGV5 published",
            )
        panel.axhline(PRIMARY_EPSILON, color="#bbb", linestyle="--", linewidth=1)
        panel.set_title(f"held out: {engine}", fontsize=10)
        panel.set_xlabel("marginal share of the accept set (%)")
        panel.set_ylabel("realised harm rate in that share")
    np.atleast_1d(panels)[0].legend(fontsize=6, loc="upper right")
    finish(
        figure,
        FIGURE_DIR / "tail_harm.png",
        "Metric 2: the harm rate among the candidates each arm would drop first.",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv10-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV10-T1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (
                    BOUNDARY_ANALYSIS,
                    RISK_COVERAGE,
                    CALIBRATION_COMPARISON,
                    TAIL_RISK_RESULTS,
                )
                if path.is_file()
            },
            "figures": {cc._relative(path): file_sha256(path) for path in written},
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"figures: {len(written)} -> {cc._relative(FIGURE_DIR)}")
    return 0


# ------------------------------------------------------------------ phase: --decide

# How many of the four unseen engines a criterion has to hold on. The brief fixes this for
# criterion 2 and says nothing about the rest; three of four is applied to every criterion that
# is read per engine, so the bar is the same everywhere and was fixed before any number was read.
ENGINE_BAR = 3


def _criteria_for(arm: str, folds: dict[str, LoadedFold], epsilon: float) -> dict[str, Any]:
    """The five criteria for one arm at one epsilon, each with the cells it was read from."""
    key = f"epsilon_{int(epsilon * 100)}"
    shares = [f"share_{int(s * 100)}" for s in TAIL_SHARES]
    per_engine: dict[str, Any] = {}
    for engine, fold in sorted(folds.items()):
        evaluation = fold.blocks["evaluation"]
        band = fold.bands[key]["evaluation"]
        baseline = place_and_measure(fold, SGV5_PUBLISHED, SOURCE_THRESHOLD, epsilon)
        measured = place_and_measure(fold, arm, PRIMARY_RULE, epsilon)
        control = place_and_measure(fold, RANDOM_ABSTENTION, PRIMARY_RULE, epsilon)
        base_tail = tail_harm(
            evaluation,
            fold.scores[key][SGV5_PUBLISHED]["evaluation"],
            fold.estimates[key][SGV5_PUBLISHED]["evaluation"],
            float(baseline["tau"]),
        )
        arm_tail = tail_harm(
            evaluation,
            fold.scores[key][arm]["evaluation"],
            fold.estimates[key][arm]["evaluation"],
            float(measured["tau"]),
        )
        frozen_cal = boundary_calibration(
            evaluation, fold.estimates[key][RULE_ONLY]["evaluation"], band
        )
        arm_cal = boundary_calibration(evaluation, fold.estimates[key][arm]["evaluation"], band)
        tail_falls = bool(
            not arm_tail["accepts_nothing"]
            and not base_tail["accepts_nothing"]
            and all(
                arm_tail["marginal"][s]["harm_rate"] < base_tail["marginal"][s]["harm_rate"]
                for s in shares
            )
        )
        floor = bool(measured["n_accepted"] >= COVERAGE_FLOOR * max(baseline["n_accepted"], 1))
        beats_random = bool(
            not measured["accepts_nothing"] and measured["repair_recall"] > control["repair_recall"]
        )
        boundary_improves = bool(
            arm_cal["boundary"] is not None
            and frozen_cal["boundary"] is not None
            and arm_cal["boundary"]["brier"] < frozen_cal["boundary"]["brier"]
        )
        per_engine[engine] = {
            "criterion_1": _criterion_one(measured, baseline, epsilon),
            "criterion_3_tail_falls": tail_falls,
            "criterion_4_coverage_floor": floor,
            "criterion_4_beats_random": beats_random,
            "criterion_5_boundary_calibration": boundary_improves,
            "arm": {
                "repair_recall": measured["repair_recall"],
                "realized_harm_rate": measured["realized_harm_rate"],
                "n_accepted": measured["n_accepted"],
                "accepts_nothing": measured["accepts_nothing"],
                "boundary_brier": (
                    arm_cal["boundary"]["brier"] if arm_cal["boundary"] else float("nan")
                ),
                "tail_marginal": {s: arm_tail["marginal"].get(s) for s in shares},
            },
            "baseline": {
                "repair_recall": baseline["repair_recall"],
                "realized_harm_rate": baseline["realized_harm_rate"],
                "n_accepted": baseline["n_accepted"],
                "holds_bound": baseline["holds_bound"],
                "boundary_brier": (
                    frozen_cal["boundary"]["brier"] if frozen_cal["boundary"] else float("nan")
                ),
                "tail_marginal": {s: base_tail["marginal"].get(s) for s in shares},
            },
            "random_control_repair_recall": control["repair_recall"],
        }
    counts = {
        name: sum(1 for cell in per_engine.values() if cell[name])
        for name in (
            "criterion_1",
            "criterion_3_tail_falls",
            "criterion_4_coverage_floor",
            "criterion_4_beats_random",
            "criterion_5_boundary_calibration",
        )
    }
    met = {
        "1": counts["criterion_1"] >= 1,
        "2": counts["criterion_1"] >= ENGINE_BAR,
        "3": counts["criterion_3_tail_falls"] >= ENGINE_BAR,
        "4": (
            counts["criterion_4_coverage_floor"] >= ENGINE_BAR
            and counts["criterion_4_beats_random"] >= ENGINE_BAR
        ),
        "5": counts["criterion_5_boundary_calibration"] >= ENGINE_BAR,
    }
    return {
        "epsilon": float(epsilon),
        "arm": arm,
        "rule": PRIMARY_RULE,
        "per_engine": per_engine,
        "counts": counts,
        "criteria_met": met,
        "n_met": sum(1 for value in met.values() if value),
        "supported": all(met.values()),
    }


def run_decide() -> int:
    """The machine-readable finding. Nothing here chooses an arm; the primary was fixed first."""
    started = time.monotonic()
    folds, _ = load_folds()
    primary = {f"epsilon_{int(e * 100)}": _criteria_for(PRIMARY_ARM, folds, e) for e in EPSILONS}
    family = {
        arm: {f"epsilon_{int(e * 100)}": _criteria_for(arm, folds, e) for e in EPSILONS}
        for arm in PROPOSED
    }
    headline = primary[PRIMARY_KEY]
    verdict = "SUPPORTED" if headline["supported"] else "NOT SUPPORTED"
    failed = [name for name, value in headline["criteria_met"].items() if not value]
    _write(
        DECISION,
        {
            "stage": "SGV10 -- tail-risk controlled selective correction",
            "verdict": verdict,
            "hypothesis": PRE_REGISTRATION["claim"],
            "primary_arm": PRIMARY_ARM,
            "primary_rule": PRIMARY_RULE,
            "primary_epsilon": PRIMARY_EPSILON,
            "criteria_met": headline["criteria_met"],
            "criteria_failed": failed,
            "n_criteria_met": headline["n_met"],
            "engine_bar": ENGINE_BAR,
            "per_epsilon": primary,
            "pre_specified_family": family,
            "claims_not_made": [
                "no confirmatory claim: the CONFIRMATORY reserve is locked and was not accessed",
                "no claim of novelty: boundary-aware selective prediction, conformal risk "
                "control, CVaR objectives and abstention are established prior art and none of "
                "them is claimed here",
                "no claim beyond four OCR engines and one receipt corpus",
                "the bound rule's distribution-free coverage is 3/4, not 1 - alpha, because it "
                "rests on three exchangeable calibration engines",
                "an arm that accepts nothing is reported as a refusal, never as a safe result",
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
        "sgv10-decision-v1",
    )
    print(f"decide: {verdict} ({headline['n_met']}/5 criteria)")
    return 0


# ------------------------------------------------------------------ phase: --record


def run_record() -> int:
    started = time.monotonic()
    produced = [
        path
        for path in (
            SCORES,
            DESIGN_RECORD,
            BOUNDARY_ANALYSIS,
            TAIL_RISK_RESULTS,
            RISK_COVERAGE,
            ABSTENTION_ANALYSIS,
            ORACLE_GAP,
            CALIBRATION_COMPARISON,
            ABLATION_RESULTS,
            NEGATIVE_TESTS,
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
        s9.SCORES,
        s9.PROVENANCE,
    ]
    cc._write_json_once(
        PROVENANCE,
        {
            "schema_version": "sgv10-provenance-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV10-T1",
            "stage": "SGV10 -- tail-risk controlled selective correction",
            "development_only": True,
            "synthetic": False,
            "confirmatory_accessed": False,
            "inputs": {cc._relative(p): file_sha256(p) for p in inputs if p.is_file()},
            "artifacts": {cc._relative(p): file_sha256(p) for p in (*produced, *figures)},
            "arms": ARM_NOTES,
            "cut_rules": RULE_NOTES,
            "ablations": {name: spec["question"] for name, spec in ABLATIONS.items()},
            "regeneration": (
                "uv run python scripts/sgv10_tail_risk_control.py --boundary, then --analysis "
                "--tail --coverage --abstention --oracle --calibration --ablation --negative "
                "--figures --decide --record"
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"record: {len(produced) + len(figures)} artifacts -> {PROVENANCE}")
    return 0


# ------------------------------------------------------------------ entry point

PHASES = (
    ("boundary", run_boundary),
    ("analysis", run_analysis),
    ("tail", run_tail),
    ("coverage", run_coverage),
    ("abstention", run_abstention),
    ("oracle", run_oracle),
    ("calibration", run_calibration),
    ("ablation", run_ablation),
    ("negative", run_negative),
    ("figures", run_figures),
    ("decide", run_decide),
    ("record", run_record),
)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name, _run in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    parsed = parser.parse_args(argv)
    requested = [(name, run) for name, run in PHASES if getattr(parsed, name)]
    if not requested:
        parser.error("choose at least one phase")
    OUT.mkdir(parents=True, exist_ok=True)
    for _name, run in requested:
        code = run()
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

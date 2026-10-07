#!/usr/bin/env python3
"""SGV5: does richer candidate-level evidence move the bounded-risk frontier?

**The brief's premise needs one correction before anything is measured, and stating it here
is what keeps the rest of this stage honest.** The brief motivates SGV5 as "SGV1-SGV4 failed,
so try candidate-level reliability". SGV1 phase 3 already IS a candidate-level reliability
model: its representation is 96 per-candidate features and its `harm_aware` arm is
`P(benefit) - lambda * P(harm)` estimated per candidate. That arm is the strongest deployable
baseline in every stage since. What failed in SGV1 phase 6 was not the granularity, it was the
transfer to an unseen engine; SGV2, SGV3 and SGV4 then failed to repair that transfer with
shift detection, per-sample uncertainty and batch-level applicability respectively.

So the question this stage can actually answer is narrower and better posed than "is
candidate-level better than environment-level":

    SGV5-C1: candidate evidence that the existing representation does not contain --
    structural validation of the proposed edit against the document's own arithmetic and
    format, and retrieval evidence from previously observed corrections -- together with
    model classes the project has not used (gradient boosting, pairwise ranking), improves
    bounded-risk repair coverage on an unseen OCR engine over the candidate-level logistic
    model SGV1 already established.

**The hypothesis is recorded as SGV5-C1, not H1.** `docs/sgv1/protocol.md` binds SGV1-H1
through SGV1-H4 and says a frozen ID is never reused for a different claim; SGV2, SGV3 and
SGV4 restarted their own families for the same reason.

Five things decide what the numbers below can mean.

**1. The comparison that matters is against SGV1, not against SGV4.** Experiment 3 asks
whether matching the decision granularity helps. It cannot be answered by comparing SGV5 to
SGV4, because SGV1's arm already has the candidate granularity and SGV4's arm is SGV1's arm
plus a per-environment offset. Comparing SGV5 to SGV4 would measure the offset, not the
granularity. The granularity comparison is therefore reported as it actually decomposes:
SGV1 (candidate) against SGV4 (candidate + environment offset) against SGV5 (candidate with
more evidence), with the SGV1-to-SGV4 step already measured and null.

**2. Two new things are introduced at once, so they are separated by construction.** Richer
evidence and a stronger model class are different claims. Every fold therefore fits the
2x2 of {phase-3 representation, extended representation} x {logistic, gradient boosting}, so
"the features did it" and "the model did it" are distinguishable rather than confounded.

**3. Retrieval evidence reads fit labels, which makes it the highest-risk block here.** A
nearest-neighbour feature that includes the query row's own label is a label leak that looks
exactly like a strong feature. The retrieval index is built from fit rows only, and fit rows
are scored leave-one-out against it. `tests/leakage` constructs a case where a leak would be
visible and asserts it is not there.

**4. A page-level aggregate is not an environment-level aggregate.** Block D evaluates the
proposed edit against the arithmetic and format of the page it sits on, by substituting the
candidate into that page's own token list. The unit is still one candidate: two candidates on
the same page get different values. This is the same discipline the existing `ctx_` block
already uses, and it is not the batch-level aggregation SGV4 tested and the brief rules out.

**5. The oracle has to be decomposed or it says nothing.** Perfect knowledge of the label
gives repair recall 1.0 by construction, which is not a finding. The gap is split into a
transfer cost (the same model fitted on the held-out engine's OWN labels, document holdout
intact) and a residual, and the residual is probed for irreducible ambiguity by measuring how
often rows that are neighbours in the representation carry opposite labels.

    --features     stage 1: the two new evidence blocks and what they are worth alone
    --scores       the row-level table every later stage reads
    --curves       experiment 1: the risk-coverage frontier against every baseline
    --ablation     experiment 2: remove each evidence family
    --transfer     experiments 3 and 4: granularity, and the unseen engine
    --oracle       experiment 5: transfer cost, representation limit, irreducible ambiguity
    --figures      the three required figures
    --decide       the machine-readable finding
    --record       provenance for every artifact

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked and is absent from every artifact.
Every fold holds out an engine AND holds out documents; neither axis is relaxed anywhere.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
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
import sgv4_environment_aware as ea
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.risk.controller import CONTROLLERS, select_threshold

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv5_candidate_reliability"
FEATURES = OUT / "candidate_features.parquet"
FEATURE_SPEC = OUT / "candidate_features.json"
PREDICTIONS = OUT / "model_predictions.parquet"
POLICY_SELECTION = OUT / "policy_selection.json"
CURVE_RESULTS = OUT / "risk_coverage_results.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
TRANSFER_RESULTS = OUT / "cross_engine_results.json"
ORACLE_RESULTS = OUT / "oracle_ceiling.json"
DECISION = OUT / "research_decision.json"
FIT_RECORD = OUT / "fit_record.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# The endpoint, the restriction that keeps a rejected row rejected inside it, the interval
# and the paired delta all have exactly one implementation in this repository. Five stages
# now import them rather than restating them; `tests/leakage` asserts the identity.
achievable_repair_recall = rl.achievable_repair_recall
restricted_frontier = sa._restricted_frontier
restricted_interval = sa._restricted_interval
arm_summary = sa._arm_summary
curve = sa._curve
paired_repair_recall_delta = sa._paired_repair_recall_delta
harm_at_matched_coverage = rl.harm_at_matched_coverage
build_fold = rl.build_fold
Fold = rl.Fold

EPSILONS = policy.EPSILONS
PRIMARY_EPSILON = dg.PRIMARY_EPSILON
DELTA = dg.DELTA
BOOTSTRAP_RESAMPLES = dg.BOOTSTRAP_RESAMPLES
MATCHED_COVERAGES = rl.MATCHED_COVERAGES
CROSS_ENGINE = sa.CROSS_ENGINE
IN_DOMAIN = sa.IN_DOMAIN
SOURCE_CALIBRATION = sa.SOURCE_CALIBRATION

LAMBDA_GRID = policy.LAMBDAS
RETRIEVAL_NEIGHBOURS = 25
RETRIEVAL_COMPONENTS = 16
RETRIEVAL_MINIMUM_SUPPORT = 5
# Phase 3's own gradient-boosting configuration, reproduced exactly rather than tuned.
# `scripts/sgv1_candidate_conditioned.py` registers arm `R1_gb` as
# `HistGradientBoostingClassifier(max_iter=200, random_state=...)` on sklearn defaults, and
# `docs/sgv1/candidate_conditioned_representation.md` reports it beating the logistic arm on
# Brier, AURC and coverage-at-risk -- POOLED, with every engine in every role, which that
# document's limitation 4 flags as leaving the cross-engine question untested. Matching the
# configuration means the advantage measured here cannot be a tuning artifact of this stage.
# It is the model CLASS and its hyperparameters that are Phase 3's; the two-head construction
# around it is SGV1's, the one every stage since has used, so this arm is not literally the
# R1_gb object and is not described as one.
BOOSTING_ITERATIONS = 200
# The logistic arm carries SGV1's class_weight="balanced" and the boosted arm carries Phase
# 3's default of none, because each reproduces the arm the project published. The difference
# is a confound inside the model contrast and is measured directly by the balanced variant.
BOOSTED_BALANCED = "boosted_balanced"
RANKER_PAIRS = 60000
RANKER_STEPS = 300
RANKER_LEARNING_RATE = 0.5
AMBIGUITY_NEIGHBOURS = 10


class PhaseError(RuntimeError):
    """A freeze, role, split, feature or selection invariant failed."""


# ------------------------------------------------------------------ stage 1: new evidence


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One new candidate feature, with where it comes from and why it is there.

    The brief requires the representation to be documented and requires that no feature use
    ground truth or post-correction information. Holding the documentation in the object that
    names the number is the arrangement in which the two cannot drift apart: `--features`
    writes this table straight into the artifact.
    """

    name: str
    block: str
    source: str
    definition: str
    rationale: str


def _spec(name: str, block: str, source: str, definition: str, rationale: str) -> FeatureSpec:
    return FeatureSpec(
        name=name, block=block, source=source, definition=definition, rationale=rationale
    )


# The amount grammar this corpus actually uses. CORD receipts are Indonesian: `75,000`,
# `125.000`, occasionally `1.500,50`. The pattern accepts a leading group of 1-3 digits,
# any number of separated 3-digit groups, and an optional 1-2 digit tail.
AMOUNT = re.compile(r"^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?$")
# A strict thousands reading: every separated group is exactly three digits and there is no
# decimal tail. This corpus's amounts are `75,000`, so `1,50` is a two-digit group and is
# anomalous here even though the amount grammar accepts it as one and a half.
GROUPED = re.compile(r"^\d{1,3}(?:[.,]\d{3})*$")
DATE_TIME = re.compile(r"^(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}(?::\d{2})?)$")
# Bounded so a page with an unusual number of amounts cannot make the pairwise identity
# quadratically expensive. The cap is recorded; pages above it are flagged in the artifact.
MAX_PAGE_AMOUNTS = 48

STRUCTURAL_SPECS = (
    _spec(
        "struct_amount_valid_o",
        "structural",
        "original_ocr against the corpus amount grammar",
        "1 when the OCR token parses as an amount",
        "The precondition for every other structural test. Reported separately from the gain "
        "so that 'this site is not about money' is distinguishable from 'the edit changed "
        "nothing about its validity'.",
    ),
    _spec(
        "struct_amount_valid_y",
        "structural",
        "candidate_text against the corpus amount grammar",
        "1 when the proposed correction parses as an amount",
        "See struct_amount_valid_o.",
    ),
    _spec(
        "struct_amount_gain",
        "structural",
        "the two above",
        "valid(candidate) - valid(original)",
        "The direction of the edit with respect to the grammar: +1 repairs an unparseable "
        "amount, -1 destroys a parseable one, which is the harm shape this corpus punishes "
        "hardest.",
    ),
    _spec(
        "struct_separator_match_o",
        "structural",
        "original_ocr against the page's modal separator shape",
        "1 when the token's thousands/decimal separators match the shape most amounts on "
        "this page use",
        "A receipt is internally consistent about its separators. An amount that disagrees "
        "with its own page is more likely an OCR error than a real value, and the reference "
        "is the page's own OCR output, never ground truth.",
    ),
    _spec(
        "struct_separator_match_y",
        "structural",
        "candidate_text against the page's modal separator shape",
        "1 when the proposed correction matches the page's modal separator shape",
        "See struct_separator_match_o.",
    ),
    _spec(
        "struct_separator_gain",
        "structural",
        "the two above",
        "match(candidate) - match(original)",
        "The mixed shapes in this corpus -- 169 tokens with `,` then `.`, 143 with `.` then "
        "`,` -- are almost all recognition errors, so an edit that resolves one is doing the "
        "thing the corpus needs.",
    ),
    _spec(
        "struct_group_valid_o",
        "structural",
        "original_ocr digit grouping",
        "1 when every separated group after the first has exactly three digits",
        "Catches the error the separator test misses: `1,50` has the right separator and the "
        "wrong grouping, and it is a different digit-loss failure from `1.500,50`.",
    ),
    _spec(
        "struct_group_valid_y",
        "structural",
        "candidate_text digit grouping",
        "1 when the proposed correction groups correctly",
        "See struct_group_valid_o.",
    ),
    _spec(
        "struct_group_gain",
        "structural",
        "the two above",
        "grouping validity of the candidate minus the original",
        "See struct_group_valid_o.",
    ),
    _spec(
        "struct_magnitude_gap_o",
        "structural",
        "original_ocr value against the page's amount scale",
        "|log10(value) - median log10(page amounts)|, 0 when the token is not an amount",
        "A dropped or duplicated digit moves an amount by an order of magnitude, which is "
        "invisible to every character-level feature and obvious against the page's own scale.",
    ),
    _spec(
        "struct_magnitude_gap_y",
        "structural",
        "candidate_text value against the page's amount scale",
        "the same gap for the proposed correction",
        "See struct_magnitude_gap_o.",
    ),
    _spec(
        "struct_magnitude_gain",
        "structural",
        "the two above",
        "gap(original) - gap(candidate); positive means the edit moves the value toward the "
        "scale the rest of the page uses",
        "The signed form is the actionable one: the model needs to know the direction, and a "
        "magnitude alone cannot express it.",
    ),
    _spec(
        "struct_magnitude_defined",
        "structural",
        "the two grammars",
        "1 when the OCR token and the correction both parse as amounts",
        "The gain above is only meaningful when both sides are amounts. Without this "
        "indicator a zero gain would mean two different things -- 'the edit did not move the "
        "value' and 'one side is not a number' -- and the model could not separate them.",
    ),
    _spec(
        "struct_reconcile_residual_o",
        "structural",
        "the page's amount multiset, from canonical OCR spans",
        "min over page amounts t of |t - sum(amounts below t)| / t",
        "The receipt-total identity. Measured on this corpus at median residual 0.0001 with "
        "60.7% of pages under 1%, so it is a real constraint here rather than an assumed one.",
    ),
    _spec(
        "struct_reconcile_residual_y",
        "structural",
        "the same multiset with the candidate substituted for the original",
        "the same residual after applying the proposed edit to the page",
        "This is what makes the feature candidate-level rather than document-level: two "
        "candidates on the same page get different values because each is substituted in.",
    ),
    _spec(
        "struct_reconcile_gain",
        "structural",
        "the two above",
        "residual(original page) - residual(edited page)",
        "Positive means applying this edit makes the receipt add up better. It is the "
        "closest thing this corpus has to the balance-equation check the brief asks for.",
    ),
    _spec(
        "struct_pair_residual_o",
        "structural",
        "the page's amount multiset",
        "min over pairs (x, y) of the distance from x + y to the nearest other page amount, "
        "relative to that amount",
        "The subtotal + tax = total shape. Measured at median residual 0.0000 with 76.4% of "
        "pages under 1%, so it holds more often than the total identity and fails "
        "differently, which is why both are kept.",
    ),
    _spec(
        "struct_pair_residual_y",
        "structural",
        "the same multiset with the candidate substituted",
        "the same residual after applying the proposed edit",
        "See struct_reconcile_residual_y.",
    ),
    _spec(
        "struct_pair_gain",
        "structural",
        "the two above",
        "residual(original page) - residual(edited page)",
        "See struct_reconcile_gain.",
    ),
    _spec(
        "struct_date_valid_o",
        "structural",
        "original_ocr against a date/time grammar",
        "1 when the OCR token parses as a date or a time",
        "The brief asks for date-format validity. It is computed and reported even though "
        "this corpus turns out to place no correction site on a date token, because a "
        "feature that is dead on one corpus is a fact about the corpus worth recording.",
    ),
    _spec(
        "struct_date_valid_y",
        "structural",
        "candidate_text against a date/time grammar",
        "1 when the proposed correction parses as a date or a time",
        "See struct_date_valid_o.",
    ),
    _spec(
        "struct_line_has_amount",
        "structural",
        "context_before and context_after around the site",
        "1 when the line the site sits on contains at least one amount token",
        "Field-schema validity at the resolution this corpus supports: a token sharing a line "
        "with a price is in a line item, and line items are where digit errors are expensive.",
    ),
    _spec(
        "struct_page_amount_count",
        "structural",
        "the page's canonical OCR spans",
        "how many amounts the page contains, capped at the recorded maximum",
        "The two residuals are meaningless on a page with one amount and informative on a "
        "page with twenty. Reporting the count lets the model condition on that rather than "
        "treating a degenerate residual as evidence.",
    ),
    _spec(
        "struct_absent",
        "structural",
        "the two grammars",
        "1 when neither the OCR token nor the correction parses as an amount or a date",
        "The explicit missing indicator that `.claude/rules/data-provenance.md` requires "
        "instead of a sentinel: every structural coordinate above is zero on such a row, and "
        "zero has to be distinguishable from a real measurement of zero.",
    ),
)

RETRIEVAL_SPECS = (
    _spec(
        "retr_signature_support",
        "retrieval",
        "fit rows sharing the candidate's edit signature",
        "log1p of how many fit candidates propose the same normalised character edit",
        "The classic OCR-confusion memory: `0 -> O` has been seen thousands of times and "
        "`Q -> 2` has not, and how often a pattern has been seen is prior to what happened "
        "when it was applied.",
    ),
    _spec(
        "retr_signature_beneficial_rate",
        "retrieval",
        "labels of fit rows sharing the signature, leave-one-out",
        "fraction of those fit candidates whose edit was beneficial",
        "The success frequency the brief asks for. It reads FIT labels only, and a fit row "
        "scoring itself is excluded, which is the single most likely way this block could "
        "become a label leak.",
    ),
    _spec(
        "retr_signature_harmful_rate",
        "retrieval",
        "labels of fit rows sharing the signature, leave-one-out",
        "fraction of those fit candidates whose edit was harmful",
        "Reported beside the beneficial rate rather than as one net number, because the "
        "neutral class is large here and a difference would hide it.",
    ),
    _spec(
        "retr_signature_seen",
        "retrieval",
        "the support count",
        "1 when the signature has at least the recorded minimum support in the fit set",
        "Below that support the two rates are estimated from a handful of rows and are noise; "
        "the indicator lets the model discount them instead of trusting a rate of 1.0 built "
        "from one observation.",
    ),
    _spec(
        "retr_knn_beneficial_rate",
        "retrieval",
        "labels of the nearest fit candidates in the frozen embedding, leave-one-out",
        "fraction beneficial among the k nearest fit candidates",
        "The signature is exact-match and sparse; the neighbourhood is dense and covers edits "
        "no fit row proposed verbatim. The pair answers the same question at two resolutions.",
    ),
    _spec(
        "retr_knn_harmful_rate",
        "retrieval",
        "labels of the nearest fit candidates, leave-one-out",
        "fraction harmful among the k nearest fit candidates",
        "See retr_knn_beneficial_rate.",
    ),
    _spec(
        "retr_knn_distance",
        "retrieval",
        "the frozen embedding",
        "mean distance to the k nearest fit candidates",
        "How far the neighbourhood had to reach. A rate estimated from neighbours that are all "
        "far away is a different object from one estimated from near neighbours, and without "
        "this the model cannot tell them apart.",
    ),
    _spec(
        "retr_knn_agreement",
        "retrieval",
        "the neighbourhood label distribution",
        "the largest of the beneficial, harmful and neutral fractions among the k nearest",
        "How decided the neighbourhood is. This is the local-ambiguity measurement experiment "
        "5 needs, computed as a feature so the model can also use it.",
    ),
)

NEW_SPECS = STRUCTURAL_SPECS + RETRIEVAL_SPECS
STRUCTURAL_NAMES = tuple(s.name for s in STRUCTURAL_SPECS)
RETRIEVAL_NAMES = tuple(s.name for s in RETRIEVAL_SPECS)


def _amount(text: str) -> float | None:
    """The numeric value of an amount token, or None when it is not one.

    This corpus uses `.` and `,` interchangeably for thousands, so which separator is which
    cannot be decided from the character. What CAN be decided is the group width: a trailing
    group of three digits is a thousands group, and a trailing group of one or two digits is a
    fractional part. Stripping every separator instead would read `181.500.00` as 18,150,000
    rather than 181,500.00, and both page identities would then be comparing magnitudes that
    are wrong by two orders of magnitude on exactly the tokens that carry cents.
    """
    token = str(text).strip()
    if not AMOUNT.match(token):
        return None
    if GROUPED.match(token):
        digits = re.sub(r"[^0-9]", "", token)
        return float(digits) if digits else None
    head, _, fraction = token.rpartition(token[max(token.rfind("."), token.rfind(",")) :][0])
    whole = re.sub(r"[^0-9]", "", head)
    return float(f"{whole or 0}.{fraction}")


def _separator_shape(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[.,]", str(text).strip()))


def _group_valid(text: str) -> bool:
    return bool(GROUPED.match(str(text).strip()))


def _reconcile_residual(amounts: np.ndarray) -> float:
    """min over t of |t - sum(amounts below t)| / t. 1.0 when the page cannot be tested."""
    if amounts.size < 3:
        return 1.0
    below = np.cumsum(amounts) - amounts
    residual = np.abs(amounts - below) / np.maximum(amounts, 1e-9)
    return float(np.min(residual[1:]))


def _pair_residual(amounts: np.ndarray) -> float:
    """min over pairs of the relative distance from x + y to the nearest other page amount.

    No masking is needed to stop a pair matching one of its own members: every amount is
    strictly positive, so x + y is strictly greater than both.
    """
    if amounts.size < 3:
        return 1.0
    sums = (amounts[:, None] + amounts[None, :])[np.triu_indices(amounts.size, k=1)]
    position = np.searchsorted(amounts, sums)
    best = np.full(sums.shape, np.inf)
    for offset in (-1, 0):
        index = np.clip(position + offset, 0, amounts.size - 1)
        target = amounts[index]
        best = np.minimum(best, np.abs(target - sums) / np.maximum(target, 1e-9))
    return float(np.min(best))


@dataclass(slots=True)
class Page:
    """Everything Block D needs about one (document, engine) page, computed once.

    Built from canonical OCR spans -- the engine's own output -- so nothing here reads ground
    truth. The per-candidate features are the *change* to these quantities when the proposed
    edit is applied, which is what keeps the block candidate-level.
    """

    amounts: np.ndarray
    modal_shape: tuple[str, ...]
    median_log: float
    reconcile: float
    pair: float
    truncated: bool


def build_pages() -> dict[str, Page]:
    spans = pd.read_parquet(pilot.OCR_SPANS, columns=["document_id", "engine_id", "text"])
    pages: dict[str, Page] = {}
    for (document, engine), group in spans.groupby(["document_id", "engine_id"], sort=True):
        values: list[float] = []
        shapes: Counter[tuple[str, ...]] = Counter()
        for text in group["text"]:
            value = _amount(text)
            if value is not None and value > 0:
                values.append(value)
                shapes[_separator_shape(text)] += 1
        truncated = len(values) > MAX_PAGE_AMOUNTS
        amounts = np.sort(np.array(sorted(values, reverse=True)[:MAX_PAGE_AMOUNTS]))
        pages[f"{document}:{engine}"] = Page(
            amounts=amounts,
            modal_shape=shapes.most_common(1)[0][0] if shapes else (),
            median_log=float(np.median(np.log10(amounts))) if amounts.size else 0.0,
            reconcile=_reconcile_residual(amounts),
            pair=_pair_residual(amounts),
            truncated=truncated,
        )
    return pages


def _substituted(page: Page, original: float | None, candidate: float | None) -> np.ndarray:
    """The page's amount multiset as it would be if this one edit were applied."""
    values = page.amounts.tolist()
    if original is not None:
        position = int(np.argmin(np.abs(np.array(values) - original))) if values else -1
        if position >= 0 and abs(values[position] - original) < 1e-6:
            values.pop(position)
    if candidate is not None and candidate > 0:
        values.append(candidate)
    return np.sort(np.array(values[:MAX_PAGE_AMOUNTS]))


def structural_features(pool: pd.DataFrame, pages: dict[str, Page]) -> pd.DataFrame:
    """Block D for every candidate. Ground-truth-blind; `tests/leakage` asserts it."""
    rows = np.zeros((len(pool), len(STRUCTURAL_NAMES)))
    index = {name: position for position, name in enumerate(STRUCTURAL_NAMES)}
    line_pattern = re.compile(r"[^\n]*$")
    for position, row in enumerate(pool.itertuples()):
        original, candidate = str(row.original_ocr), str(row.candidate_text)
        page = pages.get(f"{row.document_id}:{row.engine_id}")
        value_o, value_y = _amount(original), _amount(candidate)
        valid_o, valid_y = float(value_o is not None), float(value_y is not None)
        cell = rows[position]
        cell[index["struct_amount_valid_o"]] = valid_o
        cell[index["struct_amount_valid_y"]] = valid_y
        cell[index["struct_amount_gain"]] = valid_y - valid_o
        cell[index["struct_date_valid_o"]] = float(bool(DATE_TIME.match(original.strip())))
        cell[index["struct_date_valid_y"]] = float(bool(DATE_TIME.match(candidate.strip())))
        cell[index["struct_absent"]] = float(
            value_o is None
            and value_y is None
            and not DATE_TIME.match(original.strip())
            and not DATE_TIME.match(candidate.strip())
        )
        tail = line_pattern.search(str(row.context_before))
        head = str(row.context_after).split("\n", 1)[0]
        line = f"{tail.group(0) if tail else ''} {original} {head}"
        cell[index["struct_line_has_amount"]] = float(
            any(_amount(token) is not None for token in line.split())
        )
        if page is None:
            continue
        cell[index["struct_page_amount_count"]] = float(page.amounts.size)
        if page.modal_shape or valid_o or valid_y:
            match_o = float(valid_o == 1.0 and _separator_shape(original) == page.modal_shape)
            match_y = float(valid_y == 1.0 and _separator_shape(candidate) == page.modal_shape)
            cell[index["struct_separator_match_o"]] = match_o
            cell[index["struct_separator_match_y"]] = match_y
            cell[index["struct_separator_gain"]] = match_y - match_o
        group_o = float(valid_o == 1.0 and _group_valid(original))
        group_y = float(valid_y == 1.0 and _group_valid(candidate))
        cell[index["struct_group_valid_o"]] = group_o
        cell[index["struct_group_valid_y"]] = group_y
        cell[index["struct_group_gain"]] = group_y - group_o
        if page.amounts.size:
            gap_o = abs(np.log10(value_o) - page.median_log) if value_o else 0.0
            gap_y = abs(np.log10(value_y) - page.median_log) if value_y else 0.0
            cell[index["struct_magnitude_gap_o"]] = float(gap_o)
            cell[index["struct_magnitude_gap_y"]] = float(gap_y)
            # Only defined when BOTH sides are amounts. A token that is not an amount has a
            # gap of zero by convention, and differencing against that would read "this edit
            # created an amount" as "this edit moved the value off the page's scale".
            defined = float(value_o is not None and value_y is not None)
            cell[index["struct_magnitude_defined"]] = defined
            cell[index["struct_magnitude_gain"]] = float(gap_o - gap_y) * defined
        if value_o is None and value_y is None:
            cell[index["struct_reconcile_residual_o"]] = page.reconcile
            cell[index["struct_reconcile_residual_y"]] = page.reconcile
            cell[index["struct_pair_residual_o"]] = page.pair
            cell[index["struct_pair_residual_y"]] = page.pair
            continue
        edited = _substituted(page, value_o, value_y)
        cell[index["struct_reconcile_residual_o"]] = page.reconcile
        cell[index["struct_reconcile_residual_y"]] = _reconcile_residual(edited)
        cell[index["struct_reconcile_gain"]] = (
            page.reconcile - cell[index["struct_reconcile_residual_y"]]
        )
        cell[index["struct_pair_residual_o"]] = page.pair
        cell[index["struct_pair_residual_y"]] = _pair_residual(edited)
        cell[index["struct_pair_gain"]] = page.pair - cell[index["struct_pair_residual_y"]]
    return pd.DataFrame(rows, columns=list(STRUCTURAL_NAMES))


# ------------------------------------------------------------------ block E: retrieval


def _shape(text: str) -> str:
    """Character-class skeleton: digits to `d`, letters to `a`, everything else verbatim."""
    return "".join("d" if c.isdigit() else "a" if c.isalpha() else c for c in text)


def edit_signature(original: str, candidate: str, operation: str) -> str:
    """A normalised name for the character edit this candidate proposes.

    Same-length edits become the sorted set of changed character pairs, so `0 -> O` in two
    different receipts is one signature and can accumulate support. Everything else becomes
    the operation, the length change and the character-class skeleton of what moved, which
    keeps the support high enough to estimate a rate from.
    """
    o, y = str(original).strip(), str(candidate).strip()
    if o and len(o) == len(y):
        pairs = sorted({f"{a}>{b}" for a, b in zip(o, y, strict=True) if a != b})
        if pairs:
            return "sub|" + ",".join(pairs)
    return f"{operation}|{len(y) - len(o):+d}|{_shape(o)[:6]}>{_shape(y)[:6]}"


@dataclass(slots=True)
class Retrieval:
    """Block E, fitted on the fold's fit rows and frozen.

    Two indexes over the same population. The signature table is exact-match and sparse; the
    neighbourhood is dense and covers edits no fit row proposed verbatim. Both read FIT labels
    and nothing else, and both are scored leave-one-out when the query IS a fit row -- without
    that, a fit row would carry its own outcome as a feature and the model would learn to read
    the label it is supposed to predict.
    """

    signature_total: dict[str, int]
    signature_benefit: dict[str, int]
    signature_harm: dict[str, int]
    scaler: Any
    projection: Any
    neighbours: Any
    fit_beneficial: np.ndarray
    fit_harmful: np.ndarray
    fit_row_of: dict[int, int]

    def _embed(self, design: dg.Design, index: np.ndarray, columns: list[int]) -> np.ndarray:
        block = self.scaler.transform(design.matrix[np.ix_(index, columns)])
        return np.asarray(self.projection.transform(block), dtype=float)

    def features(
        self,
        design: dg.Design,
        index: np.ndarray,
        columns: list[int],
        signatures: np.ndarray,
        *,
        leave_one_out: bool,
    ) -> np.ndarray:
        out = np.zeros((index.size, len(RETRIEVAL_NAMES)))
        position = {name: i for i, name in enumerate(RETRIEVAL_NAMES)}
        beneficial = design.beneficial
        harmful = design.harmful

        for i, row in enumerate(index.tolist()):
            signature = signatures[row]
            total = self.signature_total.get(signature, 0)
            benefit = self.signature_benefit.get(signature, 0)
            harm = self.signature_harm.get(signature, 0)
            if leave_one_out and row in self.fit_row_of:
                total -= 1
                benefit -= int(beneficial[row])
                harm -= int(harmful[row])
            out[i, position["retr_signature_support"]] = float(np.log1p(max(total, 0)))
            if total > 0:
                out[i, position["retr_signature_beneficial_rate"]] = benefit / total
                out[i, position["retr_signature_harmful_rate"]] = harm / total
            out[i, position["retr_signature_seen"]] = float(total >= RETRIEVAL_MINIMUM_SUPPORT)

        wanted = RETRIEVAL_NEIGHBOURS + (1 if leave_one_out else 0)
        distances, found = self.neighbours.kneighbors(
            self._embed(design, index, columns), n_neighbors=wanted
        )
        for i, row in enumerate(index.tolist()):
            neighbour = found[i]
            gap = distances[i]
            if leave_one_out:
                self_position = self.fit_row_of.get(row)
                keep = (
                    neighbour != self_position
                    if self_position is not None
                    else np.ones(neighbour.size, dtype=bool)
                )
                neighbour, gap = (
                    neighbour[keep][:RETRIEVAL_NEIGHBOURS],
                    gap[keep][:RETRIEVAL_NEIGHBOURS],
                )
            benefit_rate = float(self.fit_beneficial[neighbour].mean())
            harm_rate = float(self.fit_harmful[neighbour].mean())
            out[i, position["retr_knn_beneficial_rate"]] = benefit_rate
            out[i, position["retr_knn_harmful_rate"]] = harm_rate
            out[i, position["retr_knn_distance"]] = float(gap.mean())
            out[i, position["retr_knn_agreement"]] = max(
                benefit_rate, harm_rate, 1.0 - benefit_rate - harm_rate
            )
        return out


def fit_retrieval(
    design: dg.Design, fold: Fold, columns: list[int], signatures: np.ndarray
) -> Retrieval:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    block = design.matrix[np.ix_(fold.fit, columns)]
    scaler = StandardScaler().fit(block)
    scaled = np.asarray(scaler.transform(block), dtype=float)
    components = min(RETRIEVAL_COMPONENTS, scaled.shape[1], scaled.shape[0])
    projection = PCA(n_components=components, random_state=pilot.FIT_SEED).fit(scaled)
    embedding = np.asarray(projection.transform(scaled), dtype=float)
    neighbours = NearestNeighbors(n_neighbors=RETRIEVAL_NEIGHBOURS + 1, algorithm="brute").fit(
        embedding
    )

    total: dict[str, int] = {}
    benefit: dict[str, int] = {}
    harm: dict[str, int] = {}
    for row in fold.fit.tolist():
        signature = signatures[row]
        total[signature] = total.get(signature, 0) + 1
        benefit[signature] = benefit.get(signature, 0) + int(design.beneficial[row])
        harm[signature] = harm.get(signature, 0) + int(design.harmful[row])
    return Retrieval(
        signature_total=total,
        signature_benefit=benefit,
        signature_harm=harm,
        scaler=scaler,
        projection=projection,
        neighbours=neighbours,
        fit_beneficial=design.beneficial[fold.fit].astype(float),
        fit_harmful=design.harmful[fold.fit].astype(float),
        fit_row_of={row: i for i, row in enumerate(fold.fit.tolist())},
    )


# ------------------------------------------------------------------ representations

PHASE3 = "phase3"
EXTENDED = "extended"
REPRESENTATIONS = (PHASE3, EXTENDED)

# Which prefixes each ablation removes. The five families are the brief's, mapped onto the
# prefixes the representation actually uses, so "no text features" is a checkable statement
# about columns rather than an intention.
ABLATION_PREFIXES = {
    "no_text": ("text_", "plaus_"),
    "no_pattern": ("edit_", "conf_", "prov_operation_", "prov_site_type_"),
    "no_visual": ("vis_",),
    "no_structural": ("struct_",),
    "no_retrieval": ("retr_",),
}
ABLATION_LABELS = {
    "no_text": "1 -- no textual plausibility features",
    "no_pattern": "2 -- no OCR error-pattern features",
    "no_visual": "3 -- no visual consistency features",
    "no_structural": "4 -- no structural validation features (new in SGV5)",
    "no_retrieval": "5 -- no retrieval evidence features (new in SGV5)",
}

# `boosted` is Phase 3's configuration exactly; `boosted_balanced` adds the class weighting
# the logistic arm already carries, so the model contrast can be read without the weighting
# confound. Both are selectable: the extended representation is new either way, so pinning
# the grid to previously published configurations would not have bought attribution purity,
# and it would have left a measured gain unselected for a reason that is about bookkeeping.
MODELS = ("logistic", "boosted", BOOSTED_BALANCED, "ranker")


def base_design(design: dg.Design, structural: pd.DataFrame) -> dg.Design:
    """Phase 3's matrix with block D appended. Fold-invariant, so it is built once."""
    if len(structural) != len(design.meta):
        raise PhaseError("the structural block does not align with the design matrix")
    return dg.Design(
        matrix=np.hstack([design.matrix, structural.to_numpy(dtype=np.float64)]),
        names=tuple(design.names) + STRUCTURAL_NAMES,
        meta=design.meta,
    )


@dataclass(slots=True)
class FoldDesign:
    """One fold's full representation: block D already in, block E computed for this fold.

    Block E is fold-dependent by construction -- it reads the fit set's labels -- so unlike
    every other block it cannot live in a corpus-wide feature table. Rows the fold never
    touches are left unfilled and `columns_for` refuses to hand out a matrix that includes
    one, so an unfilled row cannot silently reach a model as a zero.
    """

    design: dg.Design
    filled: np.ndarray
    names: tuple[str, ...]

    def columns(self, representation: str, drop: tuple[str, ...] = ()) -> list[int]:
        if representation == PHASE3:
            keep = [
                i for i, name in enumerate(self.names) if not name.startswith(("struct_", "retr_"))
            ]
        else:
            keep = list(range(len(self.names)))
        if drop:
            keep = [i for i in keep if not self.names[i].startswith(drop)]
        return keep

    def block(self, index: np.ndarray) -> np.ndarray:
        if not self.filled[index].all():
            raise PhaseError("a row without block E features reached a model")
        return index


def extend_fold(
    base: dg.Design,
    fold: Fold,
    signatures: np.ndarray,
    retrieval_columns: list[int],
) -> tuple[FoldDesign, Retrieval]:
    """Fit block E on this fold and materialise the rows the fold will actually score."""
    retrieval = fit_retrieval(base, fold, retrieval_columns, signatures)
    matrix = np.full((base.matrix.shape[0], len(RETRIEVAL_NAMES)), np.nan)
    filled = np.zeros(base.matrix.shape[0], dtype=bool)
    for index, leave_one_out in (
        (fold.fit, True),
        (fold.source_cal, False),
        (fold.seen_eval, False),
        (fold.eval, False),
    ):
        if index.size == 0:
            continue
        matrix[index] = retrieval.features(
            base, index, retrieval_columns, signatures, leave_one_out=leave_one_out
        )
        filled[index] = True
    extended = dg.Design(
        matrix=np.hstack([base.matrix, np.nan_to_num(matrix, nan=0.0)]),
        names=tuple(base.names) + RETRIEVAL_NAMES,
        meta=base.meta,
    )
    return FoldDesign(design=extended, filled=filled, names=extended.names), retrieval


# ------------------------------------------------------------------ the models


@dataclass(slots=True)
class Reliability:
    """One fitted candidate reliability model: two heads, or one ranking score."""

    kind: str
    representation: str
    columns: tuple[int, ...]
    scaler: Any
    harm: Any
    outcome: Any
    harm_calibrator: Any
    benefit_calibrator: Any
    weights: np.ndarray | None

    def _scaled(self, design: dg.Design, index: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.scaler.transform(design.matrix[np.ix_(index, list(self.columns))]), dtype=float
        )

    @staticmethod
    def _column(model: Any, scaled: np.ndarray, klass: int) -> np.ndarray:
        classes = list(model.classes_)
        if klass not in classes:
            return np.zeros(len(scaled))
        return np.asarray(model.predict_proba(scaled)[:, classes.index(klass)], dtype=float)

    def raw(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        scaled = self._scaled(design, index)
        if self.kind == "ranker":
            assert self.weights is not None
            score = scaled @ self.weights
            return {"scaled": scaled, "rank_score": score}
        return {
            "scaled": scaled,
            "harm_raw": self._column(self.harm, scaled, 1),
            "benefit_raw": self._column(self.outcome, scaled, policy.CLASS_BENEFIT),
        }

    def utility(self, design: dg.Design, index: np.ndarray, lambda_: float) -> np.ndarray:
        """The decision score. For the ranker this is the learned score itself.

        The ranker is deliberately NOT converted into two probabilities and recombined: its
        training objective is already "rank a beneficial edit above a harmful one", which is
        the same objective `P(benefit) - lambda * P(harm)` approximates. Wrapping it would
        add a lambda it was never fitted with and make the comparison to models A and B a
        comparison of two different things.
        """
        raw = self.raw(design, index)
        if self.kind == "ranker":
            return np.asarray(raw["rank_score"], dtype=float)
        harm = np.asarray(self.harm_calibrator.transform(raw["harm_raw"]), dtype=float)
        benefit = np.asarray(self.benefit_calibrator.transform(raw["benefit_raw"]), dtype=float)
        return benefit - lambda_ * harm

    def probabilities(self, design: dg.Design, index: np.ndarray) -> dict[str, np.ndarray]:
        raw = self.raw(design, index)
        if self.kind == "ranker":
            return {"harm": np.full(index.size, np.nan), "benefit": np.full(index.size, np.nan)}
        return {
            "harm": np.asarray(self.harm_calibrator.transform(raw["harm_raw"]), dtype=float),
            "benefit": np.asarray(
                self.benefit_calibrator.transform(raw["benefit_raw"]), dtype=float
            ),
        }


def _pair_sample(design: dg.Design, fit: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Beneficial/harmful pairs for the ranking objective, within a site where possible.

    A within-site pair is the question an operator actually faces -- of the corrections
    proposed at THIS site, which should be trusted -- but this corpus offers only 2,244 sites
    with both a beneficial and a harmful candidate, so within-site pairs alone would fit a
    ranker on a few thousand comparisons. Cross-site pairs fill the rest, and how many of each
    were used is recorded rather than averaged away.
    """
    rng = np.random.default_rng(seed)
    beneficial = fit[design.beneficial[fit]]
    harmful = fit[design.harmful[fit]]
    if beneficial.size == 0 or harmful.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    sites = design.meta["site_id"].to_numpy(str)
    by_site: dict[str, list[list[int]]] = {}
    for row in beneficial.tolist():
        by_site.setdefault(sites[row], [[], []])[0].append(row)
    for row in harmful.tolist():
        by_site.setdefault(sites[row], [[], []])[1].append(row)
    within_positive: list[int] = []
    within_negative: list[int] = []
    for good, bad in by_site.values():
        for a in good:
            for b in bad:
                within_positive.append(a)
                within_negative.append(b)

    remaining = max(RANKER_PAIRS - len(within_positive), 0)
    positive = np.concatenate(
        [np.array(within_positive, dtype=int), rng.choice(beneficial, size=remaining, replace=True)]
    )
    negative = np.concatenate(
        [np.array(within_negative, dtype=int), rng.choice(harmful, size=remaining, replace=True)]
    )
    return positive, negative


def fit_reliability(
    design: dg.Design, fold: Fold, columns: list[int], kind: str, representation: str
) -> Reliability:
    """Fit on `fold.fit`; calibrate on `fold.source_cal`. Nothing else is ever seen."""
    from sklearn.preprocessing import StandardScaler

    from ocr_risk.calibrate.calibrators import build_calibrator

    block = design.matrix[np.ix_(fold.fit, columns)]
    scaler = StandardScaler().fit(block)
    scaled = np.asarray(scaler.transform(block), dtype=float)
    harmful = design.harmful[fold.fit].astype(int)
    outcome = np.full(fold.fit.size, policy.CLASS_NEUTRAL, dtype=np.int64)
    outcome[design.harmful[fold.fit]] = policy.CLASS_HARM
    outcome[design.beneficial[fold.fit]] = policy.CLASS_BENEFIT

    if kind == "ranker":
        from sklearn.linear_model import LogisticRegression

        positive, negative = _pair_sample(design, fold.fit, pilot.FIT_SEED)
        position = {row: i for i, row in enumerate(fold.fit.tolist())}
        difference = (
            scaled[[position[r] for r in positive.tolist()]]
            - scaled[[position[r] for r in negative.tolist()]]
        )
        # RankNet with a linear scorer is exactly logistic regression without an intercept on
        # the difference vectors, mirrored so the loss is symmetric. Solved rather than
        # descended, so the result cannot depend on a step size or a stopping rule.
        model = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(
            np.vstack([difference, -difference]),
            np.concatenate([np.ones(len(difference)), np.zeros(len(difference))]),
        )
        return Reliability(
            kind=kind,
            representation=representation,
            columns=tuple(columns),
            scaler=scaler,
            harm=None,
            outcome=None,
            harm_calibrator=None,
            benefit_calibrator=None,
            weights=np.asarray(model.coef_[0], dtype=float),
        )

    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        def make() -> Any:
            return LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
            )

    elif kind in ("boosted", BOOSTED_BALANCED):
        from sklearn.ensemble import HistGradientBoostingClassifier

        weight = "balanced" if kind == BOOSTED_BALANCED else None

        def make() -> Any:
            return HistGradientBoostingClassifier(
                max_iter=BOOSTING_ITERATIONS,
                class_weight=weight,
                random_state=pilot.FIT_SEED,
            )

    else:  # pragma: no cover - the model table is closed and tested
        raise PhaseError(f"unknown reliability model {kind!r}")

    harm_model = make().fit(scaled, harmful)
    outcome_model = make().fit(scaled, outcome)
    fitted = Reliability(
        kind=kind,
        representation=representation,
        columns=tuple(columns),
        scaler=scaler,
        harm=harm_model,
        outcome=outcome_model,
        harm_calibrator=None,
        benefit_calibrator=None,
        weights=None,
    )
    calibration = fitted.raw(design, fold.source_cal)
    harm_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    harm_calibrator.fit(calibration["harm_raw"], design.harmful[fold.source_cal].astype(float))
    benefit_calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
    benefit_calibrator.fit(
        calibration["benefit_raw"], design.beneficial[fold.source_cal].astype(float)
    )
    fitted.harm_calibrator = harm_calibrator
    fitted.benefit_calibrator = benefit_calibrator
    return fitted


# ------------------------------------------------------------------ selection


@dataclass(slots=True)
class Selection:
    """Which representation, which model class and how much harm aversion."""

    representation: str
    model: str
    lambda_: float
    per_configuration: dict[str, float]
    inner: dict[str, Any]


def _configurations() -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for representation in REPRESENTATIONS:
        for model in MODELS:
            # The ranker's score is not a utility and takes no lambda: its objective already
            # is "rank a beneficial edit above a harmful one". Giving it a lambda grid would
            # multiply its entries in the selection for no reason and bias the tie-break.
            for lambda_ in LAMBDA_GRID if model != "ranker" else (0.0,):
                out.append((representation, model, lambda_))
    return out


def _name(representation: str, model: str, lambda_: float) -> str:
    return f"{representation}|{model}|l_{lambda_:g}"


def select_policy(
    base: dg.Design, outer: Fold, signatures: np.ndarray, retrieval_columns: list[int]
) -> Selection:
    """Pick (representation, model, lambda) by an inner leave-one-engine-out over fit engines.

    The outer held-out engine appears in no block of any inner fold, so nothing chosen here
    can have been chosen for performing well on the engine the result is read from. Both new
    blocks are refitted inside each inner fold: block E reads fit labels, and reusing an
    outer-fold retrieval index inside the selection would let the inner evaluation rows be
    scored against neighbours drawn from an engine the inner fold is holding out.
    """
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    grid: dict[str, list[float]] = {}
    notes: list[dict[str, Any]] = []
    for inner_held in outer.train_engines:
        fold = rl._inner_fold(base, outer, inner_held)
        extended, retrieval = extend_fold(base, fold, signatures, retrieval_columns)
        harmful = base.harmful[fold.eval]
        beneficial = base.beneficial[fold.eval]
        fitted: dict[tuple[str, str], Reliability] = {}
        for representation in REPRESENTATIONS:
            columns = extended.columns(representation)
            for model in MODELS:
                fitted[(representation, model)] = fit_reliability(
                    extended.design, fold, columns, model, representation
                )
        for representation, model, lambda_ in _configurations():
            score = fitted[(representation, model)].utility(
                extended.design, extended.block(fold.eval), lambda_
            )
            measured = achievable_repair_recall(score, harmful, beneficial)
            grid.setdefault(_name(representation, model, lambda_), []).append(
                float(measured[key]["repair_recall"])
            )
        notes.append(
            {
                "inner_held_out": inner_held,
                "inner_train_engines": list(fold.train_engines),
                "rows": {"fit": int(fold.fit.size), "evaluate": int(fold.eval.size)},
                "retrieval_signatures": len(retrieval.signature_total),
                "retrieval_neighbours": RETRIEVAL_NEIGHBOURS,
            }
        )

    means = {name: float(np.mean(values)) for name, values in grid.items()}

    def parse(name: str) -> tuple[str, str, float]:
        representation, model, lambda_text = name.split("|")
        return representation, model, float(lambda_text[2:])

    def rank(name: str) -> tuple[float, int, float, str]:
        representation, model, lambda_ = parse(name)
        # Ties resolve towards the SIMPLER configuration: phase 3 before extended, so a new
        # block has to earn its place, and then the smaller lambda.
        return (means[name], 0 if representation == PHASE3 else -1, -lambda_, model)

    best = max(means, key=rank)
    representation, model, lambda_ = parse(best)
    return Selection(
        representation=representation,
        model=model,
        lambda_=lambda_,
        per_configuration=means,
        inner={
            "inner_engines": list(outer.train_engines),
            "inner_evaluation_role": "CALIBRATION",
            "configurations": [_name(r, m, x) for r, m, x in _configurations()],
            "lambda_grid": list(LAMBDA_GRID),
            "selected": {
                "representation": representation,
                "model": model,
                "lambda": lambda_,
                "score": means[best],
            },
            "mean_repair_recall_at_primary_epsilon": means,
            "per_inner_fold": grid,
            "inner_folds": notes,
            "tie_break": (
                "highest inner mean, then the phase-3 representation over the extended one, "
                "then the smaller lambda, then the model name. The representation tie-break "
                "runs towards the baseline so a new evidence block has to earn its selection."
            ),
            "engine_identity_used_by_selection": False,
        },
    )


# ------------------------------------------------------------------ stage 1: the artifact


def _univariate(values: np.ndarray, outcome: np.ndarray) -> float:
    if not outcome.any() or outcome.all() or np.unique(values).size < 2:
        return float("nan")
    return float(roc_auc(values, outcome.astype(float)))


def run_features() -> int:
    """Block D for every candidate, plus what each new coordinate is worth on its own."""
    started = time.monotonic()
    design = dg.load_design()
    pool = policy._pool_with_features()
    ordered = pool.set_index("candidate_id").loc[design.meta["candidate_id"].astype(str)]
    pages = build_pages()
    structural = structural_features(ordered.reset_index(), pages)

    engines = design.meta["engine_id"].to_numpy(str)
    role = design.meta["role"].to_numpy(str)
    development = role == "DEVELOPMENT"
    values = structural.to_numpy(dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for position, name in enumerate(STRUCTURAL_NAMES):
        column = values[:, position]
        diagnostics[name] = {
            "constant": bool(np.unique(column).size < 2),
            "nonzero_fraction": float(np.mean(column != 0.0)),
            "mean": float(column.mean()),
            "sd": float(column.std()),
            "auroc_vs_harmful": {
                engine: _univariate(
                    -column[development & (engines == engine)],
                    design.harmful[development & (engines == engine)],
                )
                for engine in design.engines
            },
            "auroc_vs_beneficial": {
                engine: _univariate(
                    column[development & (engines == engine)],
                    design.beneficial[development & (engines == engine)],
                )
                for engine in design.engines
            },
        }

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [design.meta[["candidate_id", "document_id", "engine_id", "role"]], structural], axis=1
    )
    cc._write_parquet_once(FEATURES, frame)
    truncated = sum(1 for page in pages.values() if page.truncated)
    testable = sum(1 for page in pages.values() if page.amounts.size >= 3)
    reconcile = np.array([p.reconcile for p in pages.values() if p.amounts.size >= 3])
    pairwise = np.array([p.pair for p in pages.values() if p.amounts.size >= 3])
    cc._write_json_once(
        FEATURE_SPEC,
        {
            "schema_version": "sgv5-features-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "synthetic": False,
            "development_only": True,
            "inherited_blocks": {
                "A_textual_plausibility": [
                    n for n in design.names if n.startswith(("text_", "plaus_"))
                ],
                "B_ocr_error_pattern": [
                    n
                    for n in design.names
                    if n.startswith(("edit_", "conf_", "prov_operation_", "prov_site_type_"))
                ],
                "C_visual_consistency": [n for n in design.names if n.startswith("vis_")],
            },
            "inherited_note": (
                "Blocks A, B and C already exist: they are Phase 3's candidate-conditioned "
                "representation, reused byte-for-byte rather than rebuilt, which is what "
                "makes the SGV1 baseline in this stage the published SGV1 baseline. Only "
                "blocks D and E are new in SGV5."
            ),
            "new_blocks": {
                "D_structural_validation": list(STRUCTURAL_NAMES),
                "E_retrieval_evidence": list(RETRIEVAL_NAMES),
            },
            "specs": [
                {
                    "name": s.name,
                    "block": s.block,
                    "source": s.source,
                    "definition": s.definition,
                    "rationale": s.rationale,
                }
                for s in NEW_SPECS
            ],
            "block_e_is_fold_dependent": (
                "Block E reads the fit set's labels, so it cannot live in a corpus-wide "
                "feature table: every fold has a different fit set and therefore different "
                "retrieval features for the same candidate. Its values are written per fold "
                "into model_predictions.parquet, and its per-fold diagnostics into "
                "policy_selection.json."
            ),
            "ground_truth_used_by_block_d": False,
            "page_structure": {
                "pages": len(pages),
                "max_amounts_per_page": MAX_PAGE_AMOUNTS,
                "pages_truncated_by_the_cap": truncated,
                "amount_grammar": AMOUNT.pattern,
                "thousands_grammar": GROUPED.pattern,
                "date_grammar": DATE_TIME.pattern,
                "median_amounts_per_page": float(
                    np.median([p.amounts.size for p in pages.values()])
                ),
                "pages_with_at_least_three_amounts": testable,
                # Whether receipt arithmetic is recoverable from OCR output at all. If these
                # residuals were large the two identity features would be measuring noise,
                # and the block would not be worth computing.
                "identity_residuals": {
                    name: {
                        "median": float(np.median(values)),
                        "fraction_below_0.01": float((values < 0.01).mean()),
                        "fraction_below_0.05": float((values < 0.05).mean()),
                    }
                    for name, values in (
                        ("reconciling_total", reconcile),
                        ("pairwise_sum", pairwise),
                    )
                },
            },
            "structural_diagnostics": diagnostics,
            "constant_coordinates": [n for n in STRUCTURAL_NAMES if diagnostics[n]["constant"]],
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    live = [n for n in STRUCTURAL_NAMES if not diagnostics[n]["constant"]]
    print(
        f"features: {len(STRUCTURAL_NAMES)} structural coordinates "
        f"({len(live)} with variance) over {len(frame)} candidates -> {cc._relative(FEATURES)}"
    )
    return 0


# ------------------------------------------------------------------ the frozen baselines

BASELINE_SOURCE = {
    "no_correction": "sgv3",
    "confidence_only": "sgv3",
    "harm_only": "sgv3",
    "harm_aware": "sgv3",
    "selfaware": "sgv3",
    "env_aware": "sgv4",
}
BASELINE_ARMS = tuple(BASELINE_SOURCE)
SUBSTANTIVE_BASELINES = ("confidence_only", "harm_only", "harm_aware", "selfaware", "env_aware")
KEYS = ["candidate_id", "held_out_engine", "evaluation_mode"]


def _verify(path: Path, record_path: Path) -> None:
    if not path.is_file() or not record_path.is_file():
        raise PhaseError(f"{cc._relative(path)} is required; run the stage that produces it")
    recorded = cc._read_json(record_path)["artifacts"].get(cc._relative(path))
    if recorded is not None and file_sha256(path) != recorded:
        raise PhaseError(f"{cc._relative(path)} has moved since its stage recorded its hash")


def load_baselines() -> pd.DataFrame:
    """Every baseline arm, read from the stage that published it, verified against its hash.

    None of the five baselines is refitted here. A refit that differed by a library version
    or a seed would surface as an SGV5 effect; reading the frozen tables makes "the baseline
    is the published baseline" a checkable property, and `tests/leakage` checks it.
    """
    _verify(sa.SCORES, sa.FIT_RECORD)
    _verify(ea.SCORES, ea.FIT_RECORD)
    left = pd.read_parquet(
        sa.SCORES,
        columns=[*KEYS, "document_id", "engine_id", "is_harmful", "beneficial"]
        + [f"arm__{n}" for n, source in BASELINE_SOURCE.items() if source == "sgv3"],
    )
    right = pd.read_parquet(
        ea.SCORES,
        columns=[*KEYS]
        + [f"arm__{n}" for n, source in BASELINE_SOURCE.items() if source == "sgv4"],
    )
    merged = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(left):
        raise PhaseError("SGV3 and SGV4 score tables do not cover the same rows")
    return merged


def in_engine_fold(design: dg.Design, held_out: str) -> Fold:
    """The oracle fold: fitted on the HELD-OUT engine's own labelled TRAIN documents.

    Not deployable -- it needs labels for the engine being evaluated, which is exactly what a
    new engine does not have. It exists to split the gap between what the deployable model
    achieves and what perfect knowledge would achieve into a transfer cost and a residual.
    The document partition is intact: TRAIN documents fit it, DEVELOPMENT documents evaluate.
    """
    role = design.meta["role"].to_numpy(str)
    engine = design.meta["engine_id"].to_numpy(str)
    fold = Fold(
        held_out=held_out,
        train_engines=(held_out,),
        fit=np.flatnonzero((role == "TRAIN") & (engine == held_out)),
        source_cal=np.flatnonzero((role == "CALIBRATION") & (engine == held_out)),
        seen_eval=np.array([], dtype=int),
        eval=np.flatnonzero((role == "DEVELOPMENT") & (engine == held_out)),
        unlabeled_target=np.array([], dtype=int),
    )
    documents = design.documents
    evaluation = set(documents[fold.eval].tolist())
    for name, block in (("fit", fold.fit), ("source_cal", fold.source_cal)):
        if set(documents[block].tolist()) & evaluation:
            raise PhaseError(f"in-engine oracle for {held_out}: {name} shares evaluation documents")
    return fold


# ------------------------------------------------------------------ the fold


@dataclass(slots=True)
class FoldModel:
    """Everything one leave-one-engine-out fold fits."""

    held_out: str
    fold: Fold
    extended: FoldDesign
    retrieval: Retrieval
    selection: Selection
    models: dict[tuple[str, str], Reliability]
    ablations: dict[str, Reliability]
    oracle: Reliability
    oracle_extended: FoldDesign
    permuted: Reliability
    permuted_extended: FoldDesign
    diagnostics: dict[str, Any]

    def arms(self, index: np.ndarray) -> dict[str, np.ndarray]:
        design = self.extended.design
        rows = self.extended.block(index)
        selection = self.selection
        scores: dict[str, np.ndarray] = {}
        for (representation, model), fitted in self.models.items():
            lambda_ = 0.0 if model == "ranker" else selection.lambda_
            scores[f"sgv5__{representation}__{model}"] = fitted.utility(design, rows, lambda_)
        scores["sgv5"] = scores[f"sgv5__{selection.representation}__{selection.model}"]
        for name, fitted in self.ablations.items():
            lambda_ = 0.0 if selection.model == "ranker" else selection.lambda_
            scores[f"sgv5_ablate__{name}"] = fitted.utility(design, rows, lambda_)
        lambda_ = 0.0 if selection.model == "ranker" else selection.lambda_
        scores[PERMUTED_ARM] = self.permuted.utility(
            self.permuted_extended.design, self.permuted_extended.block(rows), lambda_
        )
        return scores

    def oracle_scores(self, index: np.ndarray) -> np.ndarray:
        lambda_ = 0.0 if self.selection.model == "ranker" else self.selection.lambda_
        return self.oracle.utility(
            self.oracle_extended.design, self.oracle_extended.block(index), lambda_
        )

    def lambda_oracle(self, index: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """The ceiling from choosing lambda against the held-out engine's own labels."""
        design = self.extended.design
        rows = self.extended.block(index)
        harmful, beneficial = design.harmful[rows], design.beneficial[rows]
        key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
        fitted = self.models[(self.selection.representation, self.selection.model)]
        if self.selection.model == "ranker":
            score = fitted.utility(design, rows, 0.0)
            return score, {
                "note": "the ranker takes no lambda, so this ceiling coincides with the arm",
                "selected": "l_0",
            }
        grid = {
            f"l_{lambda_:g}": float(
                achievable_repair_recall(
                    fitted.utility(design, rows, lambda_), harmful, beneficial
                )[key]["repair_recall"]
            )
            for lambda_ in LAMBDA_GRID
        }
        best = max(grid, key=lambda name: grid[name])
        return fitted.utility(design, rows, float(best[2:])), {
            "note": (
                "lambda chosen on the held-out engine's own labels; a ceiling on what perfect "
                "harm-aversion tuning could deliver, never a deployable arm"
            ),
            "selected": best,
            "selected_repair_recall": grid[best],
            "deployable_selection_repair_recall": grid[f"l_{self.selection.lambda_:g}"],
            "grid": grid,
        }


def _ambiguity(design: dg.Design, index: np.ndarray, model: Reliability) -> dict[str, Any]:
    """How often rows that are neighbours in the representation carry opposite outcomes.

    Measured on the held-out engine's own evaluation rows, in the coordinates the selected
    model actually sees. Two rows that the representation cannot tell apart but that had
    opposite outcomes are a hard ceiling: no model of any complexity over these features can
    separate them, so this is the part of the residual gap that is not a model limitation.
    """
    from sklearn.neighbors import NearestNeighbors

    scaled = model._scaled(design, index)
    harmful, beneficial = design.harmful[index], design.beneficial[index]
    outcome = np.where(harmful, 0, np.where(beneficial, 2, 1))
    wanted = min(AMBIGUITY_NEIGHBOURS + 1, index.size)
    index_model = NearestNeighbors(n_neighbors=wanted, algorithm="brute").fit(scaled)
    _, found = index_model.kneighbors(scaled)
    neighbours = found[:, 1:]
    same = (outcome[neighbours] == outcome[:, None]).mean(axis=1)
    mixed = (
        (harmful[neighbours].any(axis=1) & beneficial[neighbours].any(axis=1))
        | (harmful[neighbours].any(axis=1) & beneficial)
        | (beneficial[neighbours].any(axis=1) & harmful)
    )
    return {
        "neighbours": int(wanted - 1),
        "mean_outcome_purity": float(same.mean()),
        "rows_with_a_mixed_neighbourhood": float(mixed.mean()),
        "beneficial_rows_with_a_harmful_neighbour": float(
            harmful[neighbours].any(axis=1)[beneficial].mean() if beneficial.any() else np.nan
        ),
        "note": (
            "purity is the fraction of the k nearest evaluation rows sharing the row's own "
            "outcome class. A beneficial row whose neighbourhood contains harmful rows is one "
            "the representation cannot separate, whatever model is fitted over it."
        ),
    }


def fit_fold(
    base: dg.Design, held_out: str, signatures: np.ndarray, retrieval_columns: list[int]
) -> FoldModel:
    fold = build_fold(base, held_out)
    selection = select_policy(base, fold, signatures, retrieval_columns)
    extended, retrieval = extend_fold(base, fold, signatures, retrieval_columns)

    models: dict[tuple[str, str], Reliability] = {}
    for representation in REPRESENTATIONS:
        columns = extended.columns(representation)
        for model in MODELS:
            models[(representation, model)] = fit_reliability(
                extended.design, fold, columns, model, representation
            )

    ablations = {
        name: fit_reliability(
            extended.design,
            fold,
            extended.columns(EXTENDED, drop=prefixes),
            selection.model,
            EXTENDED,
        )
        for name, prefixes in ABLATION_PREFIXES.items()
    }

    # The red-team control. If the pipeline manufactures performance -- through the
    # retrieval block, the calibrators, the endpoint, or anything else structural -- it will
    # do so with the fit labels shuffled too. A real effect must collapse here.
    permuted_meta = base.meta.copy()
    rng = np.random.default_rng(pilot.FIT_SEED)
    order = rng.permutation(fold.fit)
    permuted_meta.loc[fold.fit, "is_harmful"] = base.harmful[order]
    permuted_meta.loc[fold.fit, "beneficial"] = base.beneficial[order]
    permuted_base = dg.Design(matrix=base.matrix, names=base.names, meta=permuted_meta)
    permuted_extended, _ = extend_fold(permuted_base, fold, signatures, retrieval_columns)
    permuted = fit_reliability(
        permuted_extended.design,
        fold,
        permuted_extended.columns(selection.representation),
        selection.model,
        selection.representation,
    )

    oracle_fold = in_engine_fold(base, held_out)
    oracle_extended, _ = extend_fold(base, oracle_fold, signatures, retrieval_columns)
    oracle = fit_reliability(
        oracle_extended.design,
        oracle_fold,
        oracle_extended.columns(selection.representation),
        selection.model,
        selection.representation,
    )

    retrieval_columns_index = [
        i for i, name in enumerate(extended.names) if name.startswith("retr_")
    ]
    evaluation = extended.design.matrix[np.ix_(fold.eval, retrieval_columns_index)]
    diagnostics = {
        "rows": {
            "fit": int(fold.fit.size),
            "source_calibration": int(fold.source_cal.size),
            "in_domain_development": int(fold.seen_eval.size),
            "cross_engine_development": int(fold.eval.size),
        },
        "documents": {
            "fit": len(set(base.documents[fold.fit].tolist())),
            "source_calibration": len(set(base.documents[fold.source_cal].tolist())),
            "cross_engine_development": len(set(base.documents[fold.eval].tolist())),
        },
        "retrieval": {
            "distinct_signatures_in_fit": len(retrieval.signature_total),
            "neighbours": RETRIEVAL_NEIGHBOURS,
            "embedding_components": RETRIEVAL_COMPONENTS,
            "minimum_support": RETRIEVAL_MINIMUM_SUPPORT,
            "evaluation_rows_with_a_seen_signature": float(
                evaluation[:, RETRIEVAL_NAMES.index("retr_signature_seen")].mean()
            ),
            "evaluation_mean_neighbour_distance": float(
                evaluation[:, RETRIEVAL_NAMES.index("retr_knn_distance")].mean()
            ),
            "note": (
                "A signature seen in the fit engines' training documents is not necessarily "
                "seen on the held-out engine: the fraction above is how often the exact-match "
                "memory has anything to say about an unseen engine's proposal, and it bounds "
                "what that half of block E can contribute."
            ),
        },
        "ambiguity": _ambiguity(
            extended.design,
            fold.eval,
            models[(selection.representation, selection.model)],
        ),
        "oracle": {
            "fit_rows": int(oracle_fold.fit.size),
            "calibration_rows": int(oracle_fold.source_cal.size),
            "note": (
                "fitted on the held-out engine's own TRAIN documents with the document "
                "partition intact; not deployable, and used only to split the gap"
            ),
        },
    }
    return FoldModel(
        held_out=held_out,
        fold=fold,
        extended=extended,
        retrieval=retrieval,
        selection=selection,
        models=models,
        ablations=ablations,
        oracle=oracle,
        oracle_extended=oracle_extended,
        permuted=permuted,
        permuted_extended=permuted_extended,
        diagnostics=diagnostics,
    )


# ------------------------------------------------------------------ the score table

PERMUTED_ARM = "sgv5_permuted_labels"
ORACLE_IN_ENGINE = "oracle_in_engine"
ORACLE_LAMBDA = "oracle_lambda"
BLOCK_OF_MODE = {CROSS_ENGINE: "eval", IN_DOMAIN: "seen_eval", SOURCE_CALIBRATION: "source_cal"}


def run_scores() -> int:
    """Fit every fold once and write the row-level table every later stage reads."""
    started = time.monotonic()
    design = dg.load_design()
    if not FEATURES.is_file():
        raise PhaseError("run --features before --scores")
    structural = pd.read_parquet(FEATURES)
    if list(structural["candidate_id"].astype(str)) != list(
        design.meta["candidate_id"].astype(str)
    ):
        raise PhaseError("the structural feature table is not aligned with the design matrix")
    base = base_design(design, structural[list(STRUCTURAL_NAMES)])
    pool = policy._pool_with_features().set_index("candidate_id")
    ordered = pool.loc[design.meta["candidate_id"].astype(str)]
    signatures = np.array(
        [
            edit_signature(o, y, op)
            for o, y, op in zip(
                ordered["original_ocr"].astype(str),
                ordered["candidate_text"].astype(str),
                ordered["operation"].astype(str),
                strict=True,
            )
        ]
    )
    retrieval_columns = [i for i, name in enumerate(base.names) if not name.startswith("retr_")]
    baselines = load_baselines()

    frames: list[pd.DataFrame] = []
    record: dict[str, Any] = {}
    for held_out in design.engines:
        fitted = fit_fold(base, held_out, signatures, retrieval_columns)
        thresholds: dict[str, Any] = {}
        for mode in (CROSS_ENGINE, IN_DOMAIN, SOURCE_CALIBRATION):
            index = getattr(fitted.fold, BLOCK_OF_MODE[mode])
            if index.size == 0:
                continue
            block = baselines[
                (baselines["held_out_engine"] == held_out) & (baselines["evaluation_mode"] == mode)
            ].reset_index(drop=True)
            identifiers = design.meta["candidate_id"].astype(str).to_numpy()[index]
            block = block.set_index("candidate_id").loc[identifiers].reset_index()
            frame = pd.DataFrame(
                {
                    "candidate_id": identifiers,
                    "document_id": design.documents[index],
                    "engine_id": design.meta["engine_id"].to_numpy(str)[index],
                    "held_out_engine": held_out,
                    "evaluation_mode": mode,
                    "is_harmful": design.harmful[index],
                    "beneficial": design.beneficial[index],
                    "edit_signature": signatures[index],
                }
            )
            for name in BASELINE_ARMS:
                frame[f"arm__{name}"] = block[f"arm__{name}"].to_numpy(dtype=np.float64)
            for name, score in fitted.arms(index).items():
                frame[f"arm__{name}"] = score
            if mode == CROSS_ENGINE:
                # Both ceilings are defined only on the held-out engine's rows. The in-engine
                # oracle is fitted on that engine's own TRAIN documents, so it has block E
                # features for that engine alone; asking it to score the fit engines' rows is
                # a question it was never built to answer, and the `filled` guard says so.
                frame[f"arm__{ORACLE_IN_ENGINE}"] = fitted.oracle_scores(index)
                score, oracle_note = fitted.lambda_oracle(index)
                frame[f"arm__{ORACLE_LAMBDA}"] = score
                fitted.diagnostics["lambda_oracle"] = oracle_note
            else:
                frame[f"arm__{ORACLE_IN_ENGINE}"] = np.nan
                frame[f"arm__{ORACLE_LAMBDA}"] = np.nan
            for position, name in enumerate(RETRIEVAL_NAMES):
                frame[name] = fitted.extended.design.matrix[index, len(base.names) + position]
            probabilities = fitted.models[
                (fitted.selection.representation, fitted.selection.model)
            ].probabilities(fitted.extended.design, index)
            frame["p_harm"] = probabilities["harm"]
            frame["p_benefit"] = probabilities["benefit"]
            frames.append(frame)
            if mode == SOURCE_CALIBRATION:
                harmful = design.harmful[index]
                for name in frame.columns:
                    if not name.startswith("arm__"):
                        continue
                    score = frame[name].to_numpy(dtype=np.float64)
                    finite = np.isfinite(score)
                    if not finite.any():
                        continue
                    thresholds[name[len("arm__") :]] = {
                        controller: select_threshold(
                            score[finite],
                            harmful[finite],
                            PRIMARY_EPSILON,
                            delta=DELTA,
                            controller=controller,
                        ).as_dict()
                        for controller in CONTROLLERS
                    }

        record[held_out] = {
            "held_out_engine": held_out,
            "train_engines": list(fitted.fold.train_engines),
            "selected": {
                "representation": fitted.selection.representation,
                "model": fitted.selection.model,
                "lambda": fitted.selection.lambda_,
            },
            "inner_selection": fitted.selection.inner,
            "certified_thresholds": thresholds,
            "diagnostics": fitted.diagnostics,
        }
        print(
            f"  fold {held_out:11s} selected={fitted.selection.representation}/"
            f"{fitted.selection.model} lambda={fitted.selection.lambda_:g}  "
            f"signature coverage on the unseen engine="
            f"{fitted.diagnostics['retrieval']['evaluation_rows_with_a_seen_signature']:.3f}"
        )

    cc._write_parquet_once(PREDICTIONS, pd.concat(frames, ignore_index=True))
    cc._write_json_once(
        POLICY_SELECTION,
        {
            "schema_version": "sgv5-policy-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "selection_scope": (
                "representation, model class and lambda are chosen by an inner "
                "leave-one-engine-out over the FIT engines, evaluated on CALIBRATION "
                "documents with the calibration pages split in half so the frozen calibrators "
                "and the inner evaluation rows never share a page. Block E is refitted inside "
                "every inner fold. The held-out engine contributes nothing to any choice."
            ),
            "engine_identity_used_by_method": False,
            "baselines_source": (
                "all five baselines are copied from the tables SGV3 and SGV4 published, "
                "joined on candidate_id and verified against those stages' recorded hashes"
            ),
            "folds": record,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"scores: {sum(len(f) for f in frames)} rows -> {cc._relative(PREDICTIONS)}")
    return 0


# ------------------------------------------------------------------ reading the table


@dataclass(slots=True)
class Slice:
    """One (fold, evaluation mode) block, read back from the write-once prediction table."""

    held_out: str
    mode: str
    frame: pd.DataFrame
    harmful: np.ndarray
    beneficial: np.ndarray
    documents: np.ndarray

    def arm(self, name: str) -> np.ndarray:
        return self.frame[f"arm__{name}"].to_numpy(dtype=np.float64)

    @property
    def arm_names(self) -> list[str]:
        return [c[len("arm__") :] for c in self.frame.columns if c.startswith("arm__")]


def load_predictions() -> tuple[dict[tuple[str, str], Slice], dict[str, Any]]:
    if not PREDICTIONS.is_file() or not POLICY_SELECTION.is_file():
        raise PhaseError("run --scores before any experiment that reads them")
    table = pd.read_parquet(PREDICTIONS)
    record = cc._read_json(POLICY_SELECTION)
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
        )
    return slices, record


PROPOSED_ARM = "sgv5"
CEILING_ARMS = (ORACLE_IN_ENGINE, ORACLE_LAMBDA)
FAMILY_ARMS = tuple(
    f"sgv5__{representation}__{model}" for representation in REPRESENTATIONS for model in MODELS
)
ABLATION_ARMS = tuple(f"sgv5_ablate__{name}" for name in ABLATION_PREFIXES)
HEADLINE_ARMS = (
    *BASELINE_ARMS,
    PROPOSED_ARM,
    *FAMILY_ARMS,
    *ABLATION_ARMS,
    PERMUTED_ARM,
    *CEILING_ARMS,
)


# ------------------------------------------------------------------ experiment 1


def run_curves() -> int:
    """Experiment 1: the risk-coverage frontier, in-domain and cross-engine, for every arm."""
    started = time.monotonic()
    slices, _ = load_predictions()
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
                "beneficial": int(block.beneficial.sum()),
                "harmful": int(block.harmful.sum()),
                "arms": arms,
            }
        folds[held_out] = {"held_out_engine": held_out, "modes": modes}
        line = "  ".join(
            f"{name}={modes[CROSS_ENGINE]['arms'][name]['frontier'][key]['repair_recall']:.3f}"
            for name in ("harm_aware", "selfaware", "env_aware", PROPOSED_ARM)
        )
        print(f"  fold {held_out:11s} {line}")

    cc._write_json_once(
        CURVE_RESULTS,
        {
            "schema_version": "sgv5-risk-coverage-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "synthetic": False,
            "development_only": True,
            "primary_endpoint": (
                "repair recall while accepted harm stays within epsilon, computed on the rows "
                "an arm is willing to accept and rescaled onto all sites. The cut is chosen "
                "with hindsight on the evaluation rows, which is what makes it an ACHIEVABLE "
                "frontier and not a deployment number; the deployed threshold is reported "
                "separately in cross_engine_results.json and the two disagree."
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


# ------------------------------------------------------------------ experiment 2


def run_ablation() -> int:
    """Experiment 2: remove each evidence family and measure the change in the frontier."""
    started = time.monotonic()
    slices, record = load_predictions()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        block = slices[(held_out, CROSS_ENGINE)]
        # The reference is the full-evidence arm under the SELECTED model class, not the
        # selected arm itself. On the folds where the inner selection chose the phase-3
        # representation those two are different objects, and differencing against the
        # selected arm would report "removing the visual family" as the sum of that removal
        # and a representation change. Every ablation below therefore differs from its
        # reference in exactly one family.
        selected_model = record["folds"][held_out]["selected"]["model"]
        reference_name = f"sgv5__{EXTENDED}__{selected_model}"
        proposed = block.arm(reference_name)

        def cell(
            name: str, block: Slice = block, proposed: np.ndarray = proposed
        ) -> dict[str, Any]:
            score = block.arm(name)
            frontier = restricted_frontier(score, block.harmful, block.beneficial)
            out: dict[str, Any] = {
                "epsilon_grid": {
                    f"epsilon_{int(e * 100)}": {
                        "repair_recall": float(
                            frontier[f"epsilon_{int(e * 100)}"]["repair_recall"]
                        ),
                        "coverage": float(frontier[f"epsilon_{int(e * 100)}"]["coverage"]),
                    }
                    for e in EPSILONS
                },
                "repair_recall_at_primary_epsilon": float(frontier[key]["repair_recall"]),
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

        folds[held_out] = {
            "held_out_engine": held_out,
            "selected": record["folds"][held_out]["selected"],
            "reference_arm": reference_name,
            "reference": cell(reference_name),
            "proposed": cell(PROPOSED_ARM),
            "ablations": {
                name: {"brief_label": ABLATION_LABELS[name], **cell(f"sgv5_ablate__{name}")}
                for name in ABLATION_PREFIXES
            },
            "model_representation_grid": {
                arm: cell(arm)["repair_recall_at_primary_epsilon"] for arm in FAMILY_ARMS
            },
            "note": (
                "Every ablation refits the SELECTED model class on the extended "
                "representation minus one family, and is differenced against the same model "
                "class on the FULL extended representation, so the comparison isolates the "
                "family and neither the model nor the representation. `proposed` is the arm "
                "the inner selection actually chose and is reported for context only; where "
                "it chose the phase-3 representation it is a different object from the "
                "reference. The 2x2 separates the two things SGV5 introduces at once: down a "
                "column is the representation contribution, across a row the model class."
            ),
        }
        line = "  ".join(
            f"{n[3:14]}={v['repair_recall_at_primary_epsilon']:.3f}"
            for n, v in folds[held_out]["ablations"].items()
        )
        reference_value = folds[held_out]["reference"]["repair_recall_at_primary_epsilon"]
        print(f"  fold {held_out:11s} full={reference_value:.3f}  {line}")

    cc._write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv5-ablation-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "synthetic": False,
            "development_only": True,
            "labels": ABLATION_LABELS,
            "removed_prefixes": {k: list(v) for k, v in ABLATION_PREFIXES.items()},
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"ablation: {len(folds)} folds -> {cc._relative(ABLATION_RESULTS)}")
    return 0


# ------------------------------------------------------------------ experiments 3 and 4


def _deployed(
    score: np.ndarray, harmful: np.ndarray, beneficial: np.ndarray, tau: float
) -> dict[str, float]:
    """What happens at the threshold an operator would actually have.

    The primary endpoint chooses its cut with hindsight on the evaluation rows. This does
    not: the threshold comes from the fit engines' calibration rows and is then met with the
    unseen engine. The two answer different questions and both are reported, because an arm
    can win the first and lose the second and that difference is the deployment finding.
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
    """Experiments 3 and 4: decision granularity, and the unseen engine."""
    started = time.monotonic()
    slices, record = load_predictions()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        block = slices[(held_out, CROSS_ENGINE)]
        seen = slices[(held_out, IN_DOMAIN)]
        proposed = block.arm(PROPOSED_ARM)

        comparisons = {
            baseline: {
                "paired_repair_recall_delta": paired_repair_recall_delta(
                    proposed,
                    block.arm(baseline),
                    block.harmful,
                    block.beneficial,
                    block.documents,
                    PRIMARY_EPSILON,
                ),
                "matched_coverage_harm_reduction": rl._matched_coverage_contrast(
                    block.arm(baseline),
                    proposed,
                    block.harmful,
                    block.beneficial,
                    block.documents,
                ),
            }
            for baseline in SUBSTANTIVE_BASELINES
        }

        transfer_gap = {}
        for name in HEADLINE_ARMS:
            if name not in block.arm_names:
                continue
            cross = restricted_frontier(block.arm(name), block.harmful, block.beneficial)
            if not np.isfinite(seen.arm(name)).any():
                # The two ceilings are cross-engine only, by construction. Reporting an
                # in-domain zero for them would read as a measurement rather than as "this
                # arm does not exist here".
                transfer_gap[name] = {
                    "in_domain": float("nan"),
                    "cross_engine": float(cross[key]["repair_recall"]),
                    "gap": float("nan"),
                }
                continue
            inside = restricted_frontier(seen.arm(name), seen.harmful, seen.beneficial)
            transfer_gap[name] = {
                "in_domain": float(inside[key]["repair_recall"]),
                "cross_engine": float(cross[key]["repair_recall"]),
                "gap": float(inside[key]["repair_recall"] - cross[key]["repair_recall"]),
            }

        thresholds = record["folds"][held_out]["certified_thresholds"]
        deployment = {
            name: {
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
            for name in HEADLINE_ARMS
            if name in thresholds
        }

        # Experiment 3, as it actually decomposes. SGV1's arm is already candidate-level;
        # SGV4's is that arm plus a per-environment offset. Comparing SGV5 to SGV4 alone
        # would measure the offset, not the granularity, so the chain is reported whole.
        chain = {
            name: float(
                restricted_frontier(block.arm(name), block.harmful, block.beneficial)[key][
                    "repair_recall"
                ]
            )
            for name in ("harm_aware", "selfaware", "env_aware", PROPOSED_ARM)
        }
        granularity = {
            "chain_repair_recall_at_primary_epsilon": chain,
            "candidate_level_baseline": "harm_aware",
            "environment_level_step": float(chain["env_aware"] - chain["harm_aware"]),
            "candidate_evidence_step": float(chain[PROPOSED_ARM] - chain["harm_aware"]),
            "note": (
                "SGV1's harm-aware arm is already a candidate-level reliability model, so the "
                "granularity question is not SGV5 against SGV4 but how far each step moves "
                "from that shared candidate-level baseline. The environment-level step was "
                "measured null in SGV4 and is repeated here from the frozen table."
            ),
        }

        folds[held_out] = {
            "held_out_engine": held_out,
            "selected": record["folds"][held_out]["selected"],
            "comparisons_against_baselines": comparisons,
            "granularity": granularity,
            "transfer_gap": transfer_gap,
            "deployed_threshold": deployment,
            "per_engine": {
                "coverage_at_primary_epsilon": float(
                    restricted_frontier(proposed, block.harmful, block.beneficial)[key]["coverage"]
                ),
                "repair_recall_at_primary_epsilon": chain[PROPOSED_ARM],
                "realized_harm_at_primary_epsilon": float(
                    restricted_frontier(proposed, block.harmful, block.beneficial)[key][
                        "realized_harm_rate"
                    ]
                ),
            },
        }
        delta = comparisons["harm_aware"]["paired_repair_recall_delta"]
        print(
            f"  fold {held_out:11s} sgv5={chain[PROPOSED_ARM]:.3f} "
            f"vs SGV1={chain['harm_aware']:.3f} delta={delta['delta_repair_recall']:+.3f} "
            f"CI=[{delta['ci_lower']:+.3f},{delta['ci_upper']:+.3f}]"
        )

    cc._write_json_once(
        TRANSFER_RESULTS,
        {
            "schema_version": "sgv5-transfer-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "synthetic": False,
            "development_only": True,
            "experiment_3": (
                "decision granularity, reported as the chain from the shared candidate-level "
                "baseline rather than as SGV5 against SGV4"
            ),
            "experiment_4": (
                "cross-engine generalisation: every number is measured on the held-out engine "
                "only, with per-engine risk, coverage and recall, plus the deployed threshold"
            ),
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"transfer: {len(folds)} folds -> {cc._relative(TRANSFER_RESULTS)}")
    return 0


# ------------------------------------------------------------------ experiment 5


def run_oracle() -> int:
    """Experiment 5: split the gap into transfer cost, representation limit and ambiguity.

    Perfect knowledge of the label gives repair recall 1.0 by construction, so quoting it as
    "the oracle" would be a tautology. The informative split is:

        deployable  <=  in-engine oracle  <=  1.0

    where the in-engine oracle is the SAME model and representation fitted on the held-out
    engine's own labelled TRAIN documents. The first gap is what the engine change costs; the
    second is what the representation and the irreducible ambiguity of the task cost, and the
    neighbourhood probe estimates how much of that second gap cannot be closed by any model.
    """
    started = time.monotonic()
    slices, record = load_predictions()
    engines = sorted({key[0] for key in slices})
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    folds: dict[str, Any] = {}

    for held_out in engines:
        block = slices[(held_out, CROSS_ENGINE)]
        proposed = block.arm(PROPOSED_ARM)
        oracle = block.arm(ORACLE_IN_ENGINE)

        def recall(score: np.ndarray, block: Slice = block) -> dict[str, float]:
            frontier = restricted_frontier(score, block.harmful, block.beneficial)
            return {
                f"epsilon_{int(e * 100)}": float(
                    frontier[f"epsilon_{int(e * 100)}"]["repair_recall"]
                )
                for e in EPSILONS
            }

        deployable = recall(proposed)
        in_engine = recall(oracle)
        ambiguity = record["folds"][held_out]["diagnostics"]["ambiguity"]
        folds[held_out] = {
            "held_out_engine": held_out,
            "selected": record["folds"][held_out]["selected"],
            "deployable": deployable,
            "in_engine_oracle": in_engine,
            "label_oracle": {f"epsilon_{int(e * 100)}": 1.0 for e in EPSILONS},
            "label_oracle_note": (
                "1.0 by construction: knowing every label, accept every beneficial edit and "
                "no harmful one, so repair recall is total at zero harm for any epsilon. It "
                "is stated rather than measured because measuring it would be circular."
            ),
            "transfer_cost": {k: float(in_engine[k] - deployable[k]) for k in deployable},
            "residual_to_perfect": {k: float(1.0 - in_engine[k]) for k in in_engine},
            "paired_delta_oracle_minus_deployable": paired_repair_recall_delta(
                oracle,
                proposed,
                block.harmful,
                block.beneficial,
                block.documents,
                PRIMARY_EPSILON,
            ),
            "ambiguity_probe": ambiguity,
            "diagnosis": (
                "transfer"
                if in_engine[key] - deployable[key] > 1.0 - in_engine[key]
                else "residual"
            ),
        }
        print(
            f"  fold {held_out:11s} deployable={deployable[key]:.3f} "
            f"in-engine oracle={in_engine[key]:.3f} "
            f"transfer cost={in_engine[key] - deployable[key]:+.3f} "
            f"residual={1.0 - in_engine[key]:.3f} "
            f"mixed neighbourhoods={ambiguity['rows_with_a_mixed_neighbourhood']:.3f}"
        )

    mean_transfer = float(np.mean([folds[e]["transfer_cost"][key] for e in engines]))
    mean_residual = float(np.mean([folds[e]["residual_to_perfect"][key] for e in engines]))
    cc._write_json_once(
        ORACLE_RESULTS,
        {
            "schema_version": "sgv5-oracle-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "synthetic": False,
            "development_only": True,
            "decomposition": (
                "deployable <= in-engine oracle <= 1.0. The first gap is the cost of the "
                "engine change; the second is the representation plus whatever ambiguity is "
                "irreducible given it."
            ),
            "primary_epsilon": PRIMARY_EPSILON,
            "mean_transfer_cost": mean_transfer,
            "mean_residual_to_perfect": mean_residual,
            "dominant_limitation": "transfer" if mean_transfer > mean_residual else "residual",
            "folds": folds,
            "confirmatory_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"oracle: {len(folds)} folds -> {cc._relative(ORACLE_RESULTS)}")
    return 0


# ------------------------------------------------------------------ the decision


def _favourable(cell: dict[str, Any]) -> bool:
    """An interval that excludes zero on the favourable side, and was actually resampled."""
    return bool(
        cell.get("ci_lower", float("nan")) > 0.0 and not cell.get("degenerate_interval", False)
    )


def run_decide() -> int:
    """The machine-readable finding, under the brief's own success criteria."""
    started = time.monotonic()
    curves = cc._read_json(CURVE_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)
    oracle = cc._read_json(ORACLE_RESULTS)
    record = cc._read_json(POLICY_SELECTION)
    engines = sorted(transfer["folds"])
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"

    def frontier(engine: str, arm: str, epsilon: str = key) -> float:
        return float(
            curves["folds"][engine]["modes"][CROSS_ENGINE]["arms"][arm]["frontier"][epsilon][
                "repair_recall"
            ]
        )

    criteria: dict[str, Any] = {}
    per_baseline: dict[str, Any] = {}
    for baseline in SUBSTANTIVE_BASELINES:
        cells = {}
        for engine in engines:
            comparison = transfer["folds"][engine]["comparisons_against_baselines"][baseline]
            delta = comparison["paired_repair_recall_delta"]
            matched = comparison["matched_coverage_harm_reduction"][
                f"coverage_{int(MATCHED_COVERAGES[1] * 100)}"
            ]
            cells[engine] = {
                "delta_repair_recall": delta["delta_repair_recall"],
                "ci_lower": delta["ci_lower"],
                "ci_upper": delta["ci_upper"],
                "degenerate_interval": delta["degenerate_interval"],
                "favours_proposed": _favourable(delta),
                "harm_reduction_at_matched_coverage": matched["harm_reduction"],
                "harm_reduction_ci": [matched["ci_lower"], matched["ci_upper"]],
                "harm_significantly_worse": bool(matched["ci_upper"] < 0.0),
            }
        per_baseline[baseline] = {
            "per_engine": cells,
            "engines_favouring_proposed": [e for e in engines if cells[e]["favours_proposed"]],
            "engines_where_harm_is_worse": [
                e for e in engines if cells[e]["harm_significantly_worse"]
            ],
            "unanimous": all(cells[e]["favours_proposed"] for e in engines),
        }

    sgv1 = per_baseline["harm_aware"]
    criteria["1_improves_over_sgv1"] = {
        "per_engine_delta": {e: sgv1["per_engine"][e]["delta_repair_recall"] for e in engines},
        "passes": all(sgv1["per_engine"][e]["delta_repair_recall"] > 0.0 for e in engines),
    }
    criteria["2_survives_unseen_engine"] = {
        "note": "every number in this stage is measured on the held-out engine only",
        "engines_with_nonzero_repair_recall": [
            e for e in engines if frontier(e, PROPOSED_ARM) > 0.0
        ],
        "passes": all(frontier(e, PROPOSED_ARM) > 0.0 for e in engines),
    }
    criteria["3_holds_at_matched_harm"] = {
        "harm_reduction_at_matched_coverage": {
            e: sgv1["per_engine"][e]["harm_reduction_at_matched_coverage"] for e in engines
        },
        "engines_where_harm_is_worse": sgv1["engines_where_harm_is_worse"],
        "passes": not sgv1["engines_where_harm_is_worse"],
    }
    criteria["4_interval_excludes_zero"] = {
        "engines_favouring_proposed": sgv1["engines_favouring_proposed"],
        "passes": sgv1["unanimous"],
    }
    # Criterion 5 is not a statistic; it is the set of structural guarantees the stage can
    # point at. Each is asserted by a test in tests/leakage, and the decision records which.
    permuted = {e: frontier(e, PERMUTED_ARM) for e in engines}
    proposed = {e: frontier(e, PROPOSED_ARM) for e in engines}
    criteria["5_not_leakage_or_memorisation"] = {
        "label_permutation_control": {
            "per_engine_repair_recall": permuted,
            "proposed_per_engine": proposed,
            "collapses_on_every_engine": all(permuted[e] < proposed[e] for e in engines),
            "note": (
                "the selected configuration refitted with the fit rows' labels shuffled. A "
                "pipeline that manufactures performance structurally would do so here too, "
                "so this is the control the size of the effect calls for."
            ),
        },
        "engine_identity_is_not_a_feature": True,
        "retrieval_block_is_leave_one_out_on_fit_rows": True,
        "baselines_are_the_published_frozen_arms": True,
        "documents_are_disjoint_across_roles": True,
        "selection_never_sees_the_held_out_engine": True,
        "asserted_by": "tests/leakage/test_sgv5_candidate_reliability.py",
        "passes": all(permuted[e] < proposed[e] for e in engines),
    }

    passes = all(criteria[name]["passes"] for name in criteria)
    verdict = "SUPPORTED" if passes else "NOT SUPPORTED"

    decomposition = {
        engine: {
            "sgv1_logistic_phase3": frontier(engine, "harm_aware"),
            "phase3_representation_boosted": frontier(engine, "sgv5__phase3__boosted"),
            "extended_representation_logistic": frontier(engine, "sgv5__extended__logistic"),
            "extended_representation_boosted": frontier(engine, "sgv5__extended__boosted"),
            "selected": frontier(engine, PROPOSED_ARM),
            "model_class_step": frontier(engine, "sgv5__phase3__boosted")
            - frontier(engine, "harm_aware"),
            "representation_step": frontier(engine, "sgv5__extended__logistic")
            - frontier(engine, "harm_aware"),
        }
        for engine in engines
    }

    cc._write_json_once(
        DECISION,
        {
            "schema_version": "sgv5-decision-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "hypothesis": (
                "candidate evidence the existing representation does not contain -- "
                "structural validation and retrieval -- together with model classes the "
                "cross-engine protocol has not used, improves bounded-risk repair coverage "
                "on an unseen OCR engine over the candidate-level logistic model SGV1 "
                "established"
            ),
            "verdict": verdict,
            "synthetic": False,
            "development_only": True,
            "decision_rule": (
                "the brief's five success criteria, all required. (1) improves over the SGV1 "
                "baseline on every held-out engine; (2) reaches non-zero repair recall on "
                "every unseen engine; (3) is not significantly more harmful than SGV1 at "
                "matched site coverage on any engine; (4) the paired document-clustered "
                "interval against SGV1 excludes zero on every engine -- unanimity rather than "
                "an average, for SGV2's reason: four folds are four experiments and a mean "
                "over them hides the engine where the method broke; (5) the gain is not "
                "attributable to leakage or engine memorisation, which is a set of structural "
                "guarantees rather than a statistic."
            ),
            "success_criteria": criteria,
            "comparisons_against_baselines": per_baseline,
            "attribution": {
                "per_engine": decomposition,
                "note": (
                    "The stage introduces two things at once. `model_class_step` is what "
                    "carrying gradient boosting into the cross-engine protocol is worth on "
                    "Phase 3's own representation; `representation_step` is what the two new "
                    "evidence blocks are worth under the logistic model SGV1 used. Reading "
                    "them together says which of the two the finding actually rests on."
                ),
                "prior_art_within_this_project": (
                    "gradient boosting over the candidate-conditioned representation is not "
                    "new to this project: `docs/sgv1/candidate_conditioned_representation.md` "
                    "reports arm R1_gb beating the logistic arm on Brier, AURC and "
                    "coverage-at-risk. It was measured POOLED, with every engine in every "
                    "role, and that document's limitation 4 records the cross-engine question "
                    "as untested; no later stage carried it into the cross-engine protocol. "
                    "SGV5 uses that model class and its hyperparameters unchanged, inside "
                    "SGV1's two-head construction, so the arm here is not literally the R1_gb "
                    "object and any credit for the model class belongs to Phase 3."
                ),
            },
            "oracle_decomposition": {
                "mean_transfer_cost": oracle["mean_transfer_cost"],
                "mean_residual_to_perfect": oracle["mean_residual_to_perfect"],
                "dominant_limitation": oracle["dominant_limitation"],
                "per_engine": {
                    e: {
                        "deployable": oracle["folds"][e]["deployable"][key],
                        "in_engine_oracle": oracle["folds"][e]["in_engine_oracle"][key],
                        "mixed_neighbourhoods": oracle["folds"][e]["ambiguity_probe"][
                            "rows_with_a_mixed_neighbourhood"
                        ],
                    }
                    for e in engines
                },
            },
            "ablations": {
                name: {
                    "brief_label": ABLATION_LABELS[name],
                    "per_engine": {
                        e: ablation["folds"][e]["ablations"][name][
                            "repair_recall_at_primary_epsilon"
                        ]
                        for e in engines
                    },
                }
                for name in ABLATION_PREFIXES
            },
            "selected_per_fold": {e: record["folds"][e]["selected"] for e in engines},
            "deployed_threshold": {
                e: {
                    arm: transfer["folds"][e]["deployed_threshold"][arm]["empirical"]
                    for arm in ("harm_aware", PROPOSED_ARM)
                    if arm in transfer["folds"][e]["deployed_threshold"]
                }
                for e in engines
            },
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
    curves = cc._read_json(CURVE_RESULTS)
    ablation = cc._read_json(ABLATION_RESULTS)
    transfer = cc._read_json(TRANSFER_RESULTS)
    oracle = cc._read_json(ORACLE_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    note = "SGV5 DEVELOPMENT -- not a confirmatory result"
    engines = sorted(transfer["folds"])
    key = f"epsilon_{int(PRIMARY_EPSILON * 100)}"
    written: list[Path] = []

    def finish(figure: Any, path: Path, title: str) -> None:
        figure.suptitle(f"{title}\n{note}", fontsize=9)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    # --- risk_coverage_curve.png -----------------------------------------------------------
    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.4), sharey=True)
    drawn = ("harm_only", "harm_aware", "selfaware", "env_aware", PROPOSED_ARM, ORACLE_IN_ENGINE)
    styles = {
        "harm_only": ("#888", "-"),
        "harm_aware": ("#3a5f9e", "-"),
        "selfaware": ("#2e7d5b", "--"),
        "env_aware": ("#c87a2b", "--"),
        PROPOSED_ARM: ("#a33", "-"),
        ORACLE_IN_ENGINE: ("#a33", ":"),
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
        FIGURE_DIR / "risk_coverage_curve.png",
        "Risk against coverage on the held-out engine; the dotted line is the nominal bound "
        "and the dotted red curve is the in-engine oracle",
    )

    # --- feature_ablation.png --------------------------------------------------------------
    # Left: remove one evidence family. Right: the 2x2 that separates the two things this
    # stage introduces at once, which is the question the ablation alone cannot answer.
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    order = tuple(ABLATION_PREFIXES)
    width = 0.8 / (len(order) + 1)
    positions = np.arange(len(engines))
    left.bar(
        positions,
        [ablation["folds"][e]["reference"]["repair_recall_at_primary_epsilon"] for e in engines],
        width,
        color="#a33",
        label="all families (same model)",
    )
    for offset, name in enumerate(order, start=1):
        left.bar(
            positions + offset * width,
            [
                ablation["folds"][e]["ablations"][name]["repair_recall_at_primary_epsilon"]
                for e in engines
            ],
            width,
            label=name,
        )
    left.set_xticks(positions + 0.4 - width / 2)
    left.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    left.set_ylabel(f"repair recall at harm <= {PRIMARY_EPSILON:g}")
    left.set_title("removing one evidence family", fontsize=9)
    left.legend(fontsize=7, ncol=2)

    grid = ("harm_aware", "sgv5__phase3__boosted", "sgv5__extended__logistic", PROPOSED_ARM)
    labels = (
        "phase-3 features\nlogistic (SGV1)",
        "phase-3 features\nboosted",
        "extended features\nlogistic",
        "selected",
    )
    width = 0.8 / len(grid)
    for offset, (name, label) in enumerate(zip(grid, labels, strict=True)):
        right.bar(
            positions + offset * width,
            [
                curves["folds"][e]["modes"][CROSS_ENGINE]["arms"][name]["frontier"][key][
                    "repair_recall"
                ]
                for e in engines
            ],
            width,
            label=label,
        )
    right.set_xticks(positions + 0.4 - width / 2)
    right.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    right.set_title("which of the two changes carries it", fontsize=9)
    right.legend(fontsize=7, ncol=2)
    finish(
        figure,
        FIGURE_DIR / "feature_ablation.png",
        "What each evidence family is worth, and whether the gain is the features or the model",
    )

    # --- engine_transfer.png ---------------------------------------------------------------
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    drawn = ("harm_aware", "selfaware", "env_aware", PROPOSED_ARM)
    width = 0.8 / len(drawn)
    for offset, name in enumerate(drawn):
        left.bar(
            positions + offset * width,
            [transfer["folds"][e]["transfer_gap"][name]["in_domain"] for e in engines],
            width,
            color="#c9d6e8",
            edgecolor="#3a5f9e",
            label="in-domain" if offset == 0 else None,
        )
        left.bar(
            positions + offset * width,
            [transfer["folds"][e]["transfer_gap"][name]["cross_engine"] for e in engines],
            width * 0.55,
            color="#a33",
            label="cross-engine" if offset == 0 else None,
        )
        for position in positions:
            left.text(
                position + offset * width,
                0.02,
                name.replace("harm_aware", "SGV1")
                .replace("selfaware", "SGV3")
                .replace("env_aware", "SGV4"),
                fontsize=6,
                rotation=90,
                ha="center",
                va="bottom",
            )
    left.set_xticks(positions + 0.4 - width / 2)
    left.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    left.set_ylabel(f"repair recall at harm <= {PRIMARY_EPSILON:g}")
    left.set_title("what holding out the engine costs each arm", fontsize=9)
    left.legend(fontsize=7, loc="upper left")

    bottom = np.array([oracle["folds"][e]["deployable"][key] for e in engines])
    middle = np.array([oracle["folds"][e]["in_engine_oracle"][key] for e in engines])
    right.bar(positions, bottom, 0.55, color="#a33", label="deployable (SGV5)")
    right.bar(
        positions,
        np.maximum(middle - bottom, 0.0),
        0.55,
        bottom=bottom,
        color="#c87a2b",
        label="transfer cost (in-engine oracle)",
    )
    right.bar(
        positions,
        np.maximum(1.0 - middle, 0.0),
        0.55,
        bottom=middle,
        color="#ddd",
        label="residual to perfect knowledge",
    )
    right.set_xticks(positions)
    right.set_xticklabels([f"held out:\n{e}" for e in engines], fontsize=8)
    right.set_ylim(0.0, 1.0)
    right.set_title("experiment 5: where the remaining gap sits", fontsize=9)
    right.legend(fontsize=7, loc="lower right")
    finish(
        figure,
        FIGURE_DIR / "engine_transfer.png",
        "Cross-engine transfer per arm, and the decomposition of what is left",
    )

    cc._write_json_once(
        FIGURE_MANIFEST,
        {
            "schema_version": "sgv5-figures-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "synthetic": False,
            "development_only": True,
            "sources": {
                cc._relative(path): file_sha256(path)
                for path in (CURVE_RESULTS, ABLATION_RESULTS, TRANSFER_RESULTS, ORACLE_RESULTS)
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
    started = time.monotonic()
    design = dg.load_design()
    produced = [
        path
        for path in (
            FEATURES,
            FEATURE_SPEC,
            PREDICTIONS,
            POLICY_SELECTION,
            CURVE_RESULTS,
            ABLATION_RESULTS,
            TRANSFER_RESULTS,
            ORACLE_RESULTS,
            DECISION,
            FIGURE_MANIFEST,
        )
        if path.is_file()
    ]
    cc._write_json_once(
        FIT_RECORD,
        {
            "schema_version": "sgv5-fit-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "hypothesis_id": "SGV5-C1",
            "representation": {
                "inherited": (
                    "Phase 3's candidate-conditioned matrix, reused byte-for-byte from Phase "
                    "6, SGV2, SGV3 and SGV4, which is what makes the SGV1 baseline here the "
                    "published SGV1 baseline"
                ),
                "inherited_columns": len(design.names),
                "new_structural_columns": len(STRUCTURAL_NAMES),
                "new_retrieval_columns": len(RETRIEVAL_NAMES),
                "total_extended_columns": len(design.names)
                + len(STRUCTURAL_NAMES)
                + len(RETRIEVAL_NAMES),
            },
            "models": {
                "logistic": "sklearn LogisticRegression, SGV1's configuration",
                "boosted": (
                    f"sklearn HistGradientBoostingClassifier(max_iter={BOOSTING_ITERATIONS}), "
                    "Phase 3's arm R1_gb configuration, unchanged"
                ),
                BOOSTED_BALANCED: "the same with class_weight='balanced'",
                "ranker": (
                    "linear pairwise ranking, solved as intercept-free logistic regression on "
                    "mirrored difference vectors"
                ),
                "random_state": pilot.FIT_SEED,
                "calibration_method": pilot.CALIBRATION_METHOD,
            },
            "retrieval": {
                "neighbours": RETRIEVAL_NEIGHBOURS,
                "embedding_components": RETRIEVAL_COMPONENTS,
                "minimum_support": RETRIEVAL_MINIMUM_SUPPORT,
                "leave_one_out_on_fit_rows": True,
                "reads_labels_from": "fit rows only",
            },
            "selection": {
                "lambda_grid": list(LAMBDA_GRID),
                "representations": list(REPRESENTATIONS),
                "models": list(MODELS),
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
                "unit": "document",
            },
            "ground_truth_used_for_structural_features": False,
            "engine_identity_used_as_a_feature": False,
            "confirmatory_accessed": False,
            "inputs": {
                cc._relative(path): file_sha256(path)
                for path in (
                    pilot.CANDIDATE_TABLE,
                    pilot.LABEL_TABLE,
                    pilot.OCR_SPANS,
                    cc.FEATURES,
                    dg.DESIGN_MATRIX,
                    sa.SCORES,
                    ea.SCORES,
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
        ("features", run_features),
        ("scores", run_scores),
        ("curves", run_curves),
        ("ablation", run_ablation),
        ("transfer", run_transfer),
        ("oracle", run_oracle),
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

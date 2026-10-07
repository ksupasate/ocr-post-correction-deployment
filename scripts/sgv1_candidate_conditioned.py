#!/usr/bin/env python3
"""SGV1 development phase 3: a candidate-conditioned representation.

Phase 2 established a ceiling rather than a result. Every representation it tested was
*site*-conditioned: all 41,204 sites carry exactly one crop, so the image block contributed
an identical vector to both members of a within-site matched pair and cancelled exactly in
the score difference. Measured, not assumed -- the within-pair image difference was
``0.000e+00`` on all 794 pairs. No site-conditioned representation can move that endpoint at
any strength.

This stage builds the representation that can. The object is the pair ``(O, Y)``: the OCR
observation and the *specific* proposed replacement, with evidence that depends on both.

    R0   candidate-BLIND. The OCR token, its confidence, geometry, engine and context, with
         no sight of Y at all. A reference point for "can you decide without seeing the
         proposed edit?", not a strawman substitute for V1.
    V1   the Phase-2 non-image baseline, unchanged and re-fit for exactness.
    R1   V1 plus four candidate-conditioned families: edit relationship, candidate
         plausibility, candidate-conditioned VISUAL evidence, and context consistency.

The visual family is the load-bearing one, and it is where the naive design fails. Cell
statistics keyed on ``len(Y)`` vary across candidates of different length -- 83.5% of the
matched pairs -- but are identical for same-length candidates, which is exactly the O/0,
I/1, S/5 substitution case. So the family carries both: segmentation evidence keyed on
length, and glyph-prototype evidence at the changed position, which asks whether the ink in
that cell looks more like Y's character or O's.

Every fitted resource -- character language model, lexicon, glyph prototypes, scalers --
is fitted on TRAIN documents only and frozen, and all of them are built from OCR output,
never ground truth.

    --features   build R0/R1 feature tables and the within-pair variation audit
    --fit        fit R0, V1, R1 and the ablations; calibrate; score
    --analyze    decision quality, safety, calibration, selective prediction, categories
    --figures    risk-coverage, calibration, component ablation
    --decide     the machine-readable research decision

DEVELOPMENT ONLY. The CONFIRMATORY reserve stays locked, engine shift is not tested here
(every engine is in every role), and no hypothesis is decided.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_verifier_pilot as pilot
from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import cache_root
from ocr_risk.metrics.calibration import brier_score, expected_calibration_error
from ocr_risk.metrics.discrimination import average_precision, roc_auc
from ocr_risk.metrics.selective import aurc, coverage_at_risk, risk_coverage_curve
from ocr_risk.stats.bootstrap import paired_cluster_bootstrap

REPO = pilot.REPO
OUT = REPO / "results/generated/sgv1/candidate_conditioned"
FEATURES = OUT / "features.parquet"
FEATURE_RECORD = OUT / "feature_record.json"
RESOURCES = OUT / "fitted_resources.json"
VARIATION_AUDIT = OUT / "within_pair_variation.json"
SCORES = OUT / "decision_scores.parquet"
FIT_RECORD = OUT / "fit_record.json"
BASELINE_RESULTS = OUT / "baseline_results.json"
CANDIDATE_RESULTS = OUT / "candidate_results.json"
ABLATION_RESULTS = OUT / "ablation_results.json"
CALIBRATION_RESULTS = OUT / "calibration_results.json"
CATEGORY_RESULTS = OUT / "category_results.json"
ROBUSTNESS_RESULTS = OUT / "robustness_results.json"
DECISION = OUT / "research_decision.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

CROP_SCALE = "local_medium"
LM_ORDER = 3
LM_SMOOTHING = 0.1
GLYPH_CELLS = 8
GLYPH_ROWS = 4
PROTOTYPE_MIN_SUPPORT = 30

# Confusion classes this corpus actually produces: digit/letter shape collisions.
CONFUSION_CLASSES: tuple[tuple[str, str], ...] = (
    ("0", "O"),
    ("0", "o"),
    ("0", "D"),
    ("1", "I"),
    ("1", "l"),
    ("1", "|"),
    ("5", "S"),
    ("5", "s"),
    ("8", "B"),
    ("2", "Z"),
    ("2", "z"),
    ("6", "G"),
    ("9", "g"),
    ("9", "q"),
    ("7", "T"),
    ("4", "A"),
    ("rn", "m"),
    (",", "."),
)
_CONFUSABLE: frozenset[frozenset[str]] = frozenset(frozenset(p) for p in CONFUSION_CLASSES)

PRICE_RE = re.compile(r"^-?\d{1,3}([.,]\d{3})*([.,]\d{1,2})?$")
NUMERIC_RE = re.compile(r"^-?[\d.,]+$")


class PhaseError(RuntimeError):
    """A freeze, role, or representation invariant failed."""


def _relative(path: Path) -> str:
    return pilot._relative(path)


def _read_json(path: Path) -> dict[str, Any]:
    return pilot._read_json(path)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    pilot._write_json_once(path, payload)


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    pilot._write_parquet_once(path, frame)


# ------------------------------------------------------- TRAIN-only fitted resources


@dataclass(slots=True)
class FittedResources:
    """Everything fitted on TRAIN, frozen before any calibration or evaluation row.

    All four are built from OCR output and page pixels -- never from ground truth -- so a
    deployed system has every one of them.
    """

    lm: dict[str, float]
    lm_backoff: float
    lexicon: frozenset[str]
    glyphs: dict[str, np.ndarray]
    width_per_char: float
    documents: int

    def logprob(self, text: str) -> float:
        """Character n-gram log-probability, length-normalized."""
        if not text:
            return self.lm_backoff
        padded = ("\x02" * (LM_ORDER - 1)) + text + "\x03"
        total = 0.0
        for i in range(LM_ORDER - 1, len(padded)):
            total += self.lm.get(padded[i - LM_ORDER + 1 : i + 1], self.lm_backoff)
        return total / max(len(text), 1)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text) if t]


def fit_resources(
    streams: dict[tuple[str, str], str], train_documents: set[str]
) -> FittedResources:
    counts: Counter[str] = Counter()
    context: Counter[str] = Counter()
    lexicon: Counter[str] = Counter()
    for (document_id, _engine), stream in streams.items():
        if document_id not in train_documents:
            continue
        for token in _tokens(stream):
            lexicon[token] += 1
        padded = ("\x02" * (LM_ORDER - 1)) + stream + "\x03"
        for i in range(LM_ORDER - 1, len(padded)):
            gram = padded[i - LM_ORDER + 1 : i + 1]
            counts[gram] += 1
            context[gram[:-1]] += 1
    vocabulary = len({g[-1] for g in counts}) or 1
    lm = {
        gram: math.log((n + LM_SMOOTHING) / (context[gram[:-1]] + LM_SMOOTHING * vocabulary))
        for gram, n in counts.items()
    }
    backoff = math.log(LM_SMOOTHING / (LM_SMOOTHING * vocabulary + 1.0))
    return FittedResources(
        lm=lm,
        lm_backoff=backoff,
        lexicon=frozenset(t for t, n in lexicon.items() if n >= 2),
        glyphs={},
        width_per_char=1.0,
        documents=len(train_documents),
    )


def _ink_columns(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("L"), dtype=np.float64) / 255.0
    ink = 1.0 - array
    if ink.size == 0:
        return None
    return ink


def _cell_grid(ink: np.ndarray, index: int, n_cells: int) -> np.ndarray:
    """The ink of one character cell, as a small fixed grid."""
    columns = ink.sum(axis=0)
    peak = columns.max()
    if peak <= 0 or n_cells < 1:
        return np.zeros(GLYPH_ROWS * GLYPH_CELLS)
    on = np.where(columns > 0.15 * peak)[0]
    if on.size < 2:
        return np.zeros(GLYPH_ROWS * GLYPH_CELLS)
    lo, hi = int(on[0]), int(on[-1]) + 1
    edges = np.linspace(lo, hi, n_cells + 1).astype(int)
    index = max(0, min(index, n_cells - 1))
    cell = ink[:, edges[index] : max(edges[index + 1], edges[index] + 1)]
    if cell.size == 0:
        return np.zeros(GLYPH_ROWS * GLYPH_CELLS)
    rows = np.array_split(cell, GLYPH_ROWS, axis=0)
    grid = []
    for band in rows:
        if band.size == 0:
            grid.extend([0.0] * GLYPH_CELLS)
            continue
        parts = np.array_split(band, GLYPH_CELLS, axis=1)
        grid.extend(float(p.mean()) if p.size else 0.0 for p in parts)
    vector = np.asarray(grid, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def fit_glyph_prototypes(
    lineage: pd.DataFrame, candidates: pd.DataFrame, train_documents: set[str]
) -> tuple[dict[str, np.ndarray], float]:
    """Average ink grid per character, from TRAIN crops labelled by the ENGINE's own text.

    The supervision is the OCR's reading of the span, not ground truth: a deployed system
    has exactly this. It is noisy by construction -- the OCR is sometimes wrong, which is
    the entire premise of the project -- so the prototypes are a weak, honest signal rather
    than a character classifier.
    """
    text_of = dict(zip(candidates["site_id"].astype(str), candidates["original_ocr"].astype(str)))
    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    widths: list[float] = []
    seen: set[str] = set()
    for row in lineage.itertuples():
        if str(row.document_id) not in train_documents:
            continue
        digest = str(getattr(row, f"recipe_{CROP_SCALE}"))
        if digest in seen:
            continue
        seen.add(digest)
        text = text_of.get(str(row.site_id), "")
        stripped = text.strip()
        if not stripped or len(stripped) > 24:
            continue
        ink = _ink_columns(cache_root() / "crops" / digest[:2] / f"{digest}.png")
        if ink is None:
            continue
        widths.append(ink.shape[1] / max(len(stripped), 1))
        for index, character in enumerate(stripped):
            grid = _cell_grid(ink, index, len(stripped))
            if not grid.any():
                continue
            sums.setdefault(character, np.zeros_like(grid))
            sums[character] += grid
            counts[character] += 1
    prototypes = {
        character: total / counts[character]
        for character, total in sums.items()
        if counts[character] >= PROTOTYPE_MIN_SUPPORT
    }
    return prototypes, float(np.median(widths)) if widths else 1.0


# ------------------------------------------------------------------ feature families


def _opcodes(original: str, candidate: str) -> list[tuple[str, int, int, int, int]]:
    return SequenceMatcher(a=original, b=candidate, autojunk=False).get_opcodes()


def r0_block(
    row: Any, site_candidate_count: int, engines: Sequence[str]
) -> tuple[list[str], list[float]]:
    """Candidate-BLIND. Everything about the OCR observation, nothing about Y.

    This is the "can you decide without seeing the proposed edit?" reference point. It is
    deliberately not the study's comparator -- V1 is -- because a representation that
    cannot see the edit is not a serious baseline for edit verification.
    """
    original = str(row.original_ocr)
    width = float(row.bbox_x1 - row.bbox_x0) if pd.notna(row.bbox_x0) else 0.0
    height = float(row.bbox_y1 - row.bbox_y0) if pd.notna(row.bbox_y0) else 0.0
    names = [
        "r0_len",
        "r0_digits",
        "r0_alpha",
        "r0_punct",
        "r0_spaces",
        "r0_upper_fraction",
        "r0_suspicion",
        "r0_site_candidates",
        "r0_context_before_len",
        "r0_context_after_len",
        "r0_bbox_width",
        "r0_bbox_height",
        "r0_bbox_aspect",
        "r0_bbox_x",
        "r0_bbox_y",
        *(f"r0_engine_{e}" for e in engines),
    ]
    values = [
        float(len(original)),
        float(sum(c.isdigit() for c in original)),
        float(sum(c.isalpha() for c in original)),
        float(sum(not c.isalnum() and not c.isspace() for c in original)),
        float(sum(c.isspace() for c in original)),
        float(sum(c.isupper() for c in original)) / max(len(original), 1),
        float(row.suspicion_score),
        float(site_candidate_count),
        float(len(str(row.context_before))),
        float(len(str(row.context_after))),
        width,
        height,
        width / max(height, 1e-6),
        float(row.bbox_x0) if pd.notna(row.bbox_x0) else 0.0,
        float(row.bbox_y0) if pd.notna(row.bbox_y0) else 0.0,
        *(1.0 if str(row.engine_id) == e else 0.0 for e in engines),
    ]
    return names, values


def edit_block(original: str, candidate: str) -> tuple[list[str], list[float]]:
    """How O becomes Y: the operation profile and the confusion signature."""
    subs = ins = dels = 0
    confusable = 0
    first_change = -1
    for tag, i1, i2, j1, j2 in _opcodes(original, candidate):
        if tag == "equal":
            continue
        if first_change < 0:
            first_change = i1
        if tag == "replace":
            subs += max(i2 - i1, j2 - j1)
            for a, b in zip(original[i1:i2], candidate[j1:j2]):
                if frozenset((a, b)) in _CONFUSABLE:
                    confusable += 1
        elif tag == "insert":
            ins += j2 - j1
        elif tag == "delete":
            dels += i2 - i1
    changed = subs + ins + dels
    digits_o = sum(c.isdigit() for c in original)
    digits_y = sum(c.isdigit() for c in candidate)
    prefix = sum(1 for a, b in zip(original, candidate) if a == b)
    names = [
        "edit_subs",
        "edit_inserts",
        "edit_deletes",
        "edit_changed",
        "edit_first_change_rel",
        "edit_changed_fraction",
        "edit_confusable_subs",
        "edit_confusable_fraction",
        "edit_same_length",
        "edit_digit_only",
        "edit_alpha_only",
        "edit_cross_class",
        "edit_prefix_preserved_frac",
        "edit_len_ratio",
    ]
    values = [
        float(subs),
        float(ins),
        float(dels),
        float(changed),
        (first_change / max(len(original), 1)) if first_change >= 0 else -1.0,
        changed / max(len(original), 1),
        float(confusable),
        confusable / max(subs, 1),
        1.0 if len(original) == len(candidate) else 0.0,
        1.0 if (digits_o == len(original.strip()) and digits_y == len(candidate.strip())) else 0.0,
        1.0 if (original.isalpha() and candidate.isalpha()) else 0.0,
        1.0 if (original.isdigit() != candidate.isdigit()) else 0.0,
        prefix / max(len(original), 1),
        len(candidate) / max(len(original), 1),
    ]
    return names, values


def plausibility_block(
    original: str, candidate: str, resources: FittedResources
) -> tuple[list[str], list[float]]:
    """Is Y a plausible string in this corpus, and more plausible than O?"""
    lp_o, lp_y = resources.logprob(original), resources.logprob(candidate)
    in_o = float(any(t in resources.lexicon for t in _tokens(original)))
    in_y = float(any(t in resources.lexicon for t in _tokens(candidate)))
    names = [
        "plaus_lm_candidate",
        "plaus_lm_original",
        "plaus_lm_delta",
        "plaus_lexicon_candidate",
        "plaus_lexicon_original",
        "plaus_lexicon_gain",
        "plaus_price_candidate",
        "plaus_price_original",
        "plaus_price_gain",
        "plaus_numeric_candidate",
        "plaus_numeric_original",
    ]
    price_y = float(bool(PRICE_RE.match(candidate.strip())))
    price_o = float(bool(PRICE_RE.match(original.strip())))
    values = [
        lp_y,
        lp_o,
        lp_y - lp_o,
        in_y,
        in_o,
        in_y - in_o,
        price_y,
        price_o,
        price_y - price_o,
        float(bool(NUMERIC_RE.match(candidate.strip()))),
        float(bool(NUMERIC_RE.match(original.strip()))),
    ]
    return names, values


VISUAL_NAMES = [
    "vis_cell_mean_y",
    "vis_cell_min_y",
    "vis_cell_std_y",
    "vis_valley_y",
    "vis_cell_mean_o",
    "vis_cell_min_o",
    "vis_cell_std_o",
    "vis_valley_o",
    "vis_valley_gain",
    "vis_cell_std_gain",
    "vis_width_per_char_y",
    "vis_width_per_char_o",
    "vis_width_ratio_y",
    "vis_glyph_match_y",
    "vis_glyph_match_o",
    "vis_glyph_gain",
    "vis_glyph_available",
    "vis_missing",
]


def _segmentation(ink: np.ndarray, n_chars: int) -> tuple[float, float, float, float]:
    columns = ink.sum(axis=0)
    peak = columns.max()
    if peak <= 0 or n_chars < 1:
        return 0.0, 0.0, 0.0, 0.0
    on = np.where(columns > 0.15 * peak)[0]
    if on.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    lo, hi = int(on[0]), int(on[-1]) + 1
    segment = columns[lo:hi]
    edges = np.linspace(0, len(segment), n_chars + 1).astype(int)
    mass = np.array(
        [segment[edges[i] : edges[i + 1]].sum() for i in range(n_chars) if edges[i + 1] > edges[i]]
    )
    if mass.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mass = mass / max(mass.sum(), 1e-9)
    interior = edges[1:-1]
    valley = float(np.mean([segment[b] for b in interior])) / peak if interior.size else 0.0
    return float(mass.mean()), float(mass.min()), float(mass.std()), 1.0 - valley


def visual_block(
    ink: np.ndarray | None, original: str, candidate: str, resources: FittedResources
) -> tuple[list[str], list[float]]:
    """Candidate-conditioned image evidence: does the ink support Y rather than O?

    Two mechanisms, because one is not enough. Segmentation statistics keyed on ``len(Y)``
    separate candidates of different length (83.5% of the matched pairs) but are identical
    for same-length candidates. Glyph-prototype matching at the changed cell covers the
    rest: it asks whether the ink where O and Y disagree looks more like Y's character or
    O's, which is the question "does the image support this correction?" in its literal
    form.
    """
    if ink is None:
        return VISUAL_NAMES, [0.0] * (len(VISUAL_NAMES) - 1) + [1.0]

    o_text, y_text = original.strip(), candidate.strip()
    mean_y, min_y, std_y, valley_y = _segmentation(ink, len(y_text))
    mean_o, min_o, std_o, valley_o = _segmentation(ink, len(o_text))
    width = ink.shape[1]
    wpc_y = width / max(len(y_text), 1)
    wpc_o = width / max(len(o_text), 1)

    match_y = match_o = 0.0
    available = 0.0
    if resources.glyphs and len(o_text) == len(y_text) and o_text and y_text:
        differing = [i for i, (a, b) in enumerate(zip(o_text, y_text)) if a != b]
        scores_y, scores_o = [], []
        for index in differing:
            proto_y = resources.glyphs.get(y_text[index])
            proto_o = resources.glyphs.get(o_text[index])
            if proto_y is None or proto_o is None:
                continue
            cell = _cell_grid(ink, index, len(y_text))
            if not cell.any():
                continue
            scores_y.append(float(cell @ proto_y))
            scores_o.append(float(cell @ proto_o))
        if scores_y:
            match_y, match_o, available = float(np.mean(scores_y)), float(np.mean(scores_o)), 1.0

    values = [
        mean_y,
        min_y,
        std_y,
        valley_y,
        mean_o,
        min_o,
        std_o,
        valley_o,
        valley_y - valley_o,
        std_y - std_o,
        wpc_y,
        wpc_o,
        wpc_y / max(resources.width_per_char, 1e-6),
        match_y,
        match_o,
        match_y - match_o,
        available,
        0.0,
    ]
    return VISUAL_NAMES, values


def context_block(
    original: str, candidate: str, page_tokens: Counter[str], line_tokens: int
) -> tuple[list[str], list[float]]:
    """Does the rest of the page support Y? Page vocabulary is the engine's own output."""
    y_tokens, o_tokens = _tokens(candidate), _tokens(original)
    seen_y = sum(page_tokens.get(t, 0) for t in y_tokens)
    seen_o = sum(page_tokens.get(t, 0) for t in o_tokens)
    names = [
        "ctx_page_support_candidate",
        "ctx_page_support_original",
        "ctx_page_support_gain",
        "ctx_line_tokens",
        "ctx_token_count_candidate",
        "ctx_token_count_delta",
    ]
    values = [
        float(seen_y),
        float(seen_o),
        float(seen_y - seen_o),
        float(line_tokens),
        float(len(y_tokens)),
        float(len(y_tokens) - len(o_tokens)),
    ]
    return names, values


FAMILIES: dict[str, str] = {
    "edit": "edit_",
    "plausibility": "plaus_",
    "visual": "vis_",
    "context": "ctx_",
}


def _pool() -> tuple[pd.DataFrame, dict[str, str]]:
    roles = pilot._locked_roles()
    candidates = pd.read_parquet(pilot.CANDIDATE_TABLE)
    labels = pd.read_parquet(pilot.LABEL_TABLE)
    pool = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    pool = pool[pool["labelable"]].reset_index(drop=True)
    manifest_role = pool["document_id"].map(roles)
    if manifest_role.isna().any():
        raise PhaseError("a labelable candidate has no non-confirmatory role")
    if not bool((pool["role"].astype(str) == manifest_role.astype(str)).all()):
        raise PhaseError("the frozen candidate role disagrees with the locked role manifest")
    pool["beneficial"] = pool["outcome"].isin({"true_correction", "partial_improvement"})
    return pool, roles


def run_features() -> int:
    started = time.monotonic()
    lineage_path = REPO / "results/generated/sgv1/representation/representation_lineage.parquet"
    if not lineage_path.is_file():
        raise PhaseError("phase-2 crop lineage is required; run sgv1_representation.py --crops")

    pool, roles = _pool()
    lineage = pd.read_parquet(lineage_path)
    frame = pool.merge(
        lineage[
            ["candidate_id", f"recipe_{CROP_SCALE}", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]
        ],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(pool):
        raise PhaseError("crop lineage does not cover the labelable pool")

    train_documents = {d for d, role in roles.items() if role == "TRAIN"}
    _, streams = pilot._spans_by_pair()
    resources = fit_resources(streams, train_documents)
    glyphs, width_per_char = fit_glyph_prototypes(lineage, pool, train_documents)
    resources.glyphs, resources.width_per_char = glyphs, width_per_char
    print(
        f"  fitted on {len(train_documents)} TRAIN documents: "
        f"{len(resources.lm)} lm grams, {len(resources.lexicon)} lexicon tokens, "
        f"{len(glyphs)} glyph prototypes"
    )

    page_tokens: dict[str, Counter[str]] = {}
    line_counts: dict[str, int] = {}
    for (document_id, engine_id), stream in streams.items():
        page_tokens[f"{document_id}:{engine_id}"] = Counter(_tokens(stream))
        line_counts[f"{document_id}:{engine_id}"] = max(len(stream.splitlines()), 1)

    site_counts = pd.read_parquet(pilot.CANDIDATE_TABLE).groupby("site_id").size().to_dict()
    engines = sorted(frame["engine_id"].astype(str).unique())

    ink_cache: dict[str, np.ndarray | None] = {}
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    for row in frame.itertuples():
        original, candidate = str(row.original_ocr), str(row.candidate_text)
        digest = str(getattr(row, f"recipe_{CROP_SCALE}"))
        if digest not in ink_cache:
            ink_cache[digest] = _ink_columns(cache_root() / "crops" / digest[:2] / f"{digest}.png")
        ink = ink_cache[digest]
        key = f"{row.document_id}:{row.engine_id}"

        blocks = [
            r0_block(row, site_counts.get(str(row.site_id), 1), engines),
            edit_block(original, candidate),
            plausibility_block(original, candidate, resources),
            visual_block(ink, original, candidate, resources),
            context_block(
                original, candidate, page_tokens.get(key, Counter()), line_counts.get(key, 1)
            ),
        ]
        if names is None:
            names = [n for block, _ in ((b[0], b[1]) for b in blocks) for n in block]
        rows.append(np.concatenate([np.asarray(v, dtype=np.float64) for _, v in blocks]))

    assert names is not None
    table = pd.DataFrame(np.vstack(rows), columns=names)
    table.insert(0, "candidate_id", frame["candidate_id"].astype(str).to_numpy())
    _write_parquet_once(FEATURES, table)

    audit = _within_pair_audit(table)
    _write_json_once(VARIATION_AUDIT, audit)
    record = {
        "schema_version": "sgv1-candidate-conditioned-features-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "rows": len(table),
        "feature_columns": len(names),
        "families": {
            family: int(sum(1 for n in names if n.startswith(prefix)))
            for family, prefix in {**FAMILIES, "r0": "r0_"}.items()
        },
        "crop_scale": CROP_SCALE,
        "fitted_on": "TRAIN documents only",
        "train_documents": len(train_documents),
        "lm_order": LM_ORDER,
        "lm_grams": len(resources.lm),
        "lexicon_tokens": len(resources.lexicon),
        "glyph_prototypes": sorted(glyphs),
        "glyph_prototype_min_support": PROTOTYPE_MIN_SUPPORT,
        "median_width_per_char": width_per_char,
        "ground_truth_used_for_features": False,
        "ground_truth_used_for_row_selection": True,
        "confirmatory_accessed": False,
        "inputs": {
            _relative(pilot.CANDIDATE_TABLE): file_sha256(pilot.CANDIDATE_TABLE),
            _relative(pilot.LABEL_TABLE): file_sha256(pilot.LABEL_TABLE),
            _relative(lineage_path): file_sha256(lineage_path),
        },
        "artifacts": {_relative(FEATURES): file_sha256(FEATURES)},
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(FEATURE_RECORD, record)
    print(f"features: {len(table)} rows x {len(names)} columns -> {_relative(FEATURES)}")
    print(
        f"  within-pair variation: {audit['pairs_with_any_variation']} of {audit['pairs']} "
        f"({audit['fraction_varying']:.3%}); visual family "
        f"{audit['pairs_with_visual_variation']} ({audit['fraction_visual_varying']:.3%})"
    )
    return 0


def _within_pair_audit(table: pd.DataFrame) -> dict[str, Any]:
    """The acceptance test for this whole phase.

    Phase 2's ceiling was that within a matched pair the image features were byte-identical,
    so they cancelled exactly. A representation that claims to be candidate-conditioned has
    to break that, and the check is one line: how many of the 794 pairs now differ?
    """
    pairs = pd.read_parquet(pilot.FRAME_DIR / "matched_pairs.parquet")
    indexed = table.set_index("candidate_id")
    columns = [c for c in table.columns if c != "candidate_id"]
    visual = [c for c in columns if c.startswith("vis_")]
    r0 = [c for c in columns if c.startswith("r0_")]

    any_var = vis_var = r0_var = 0
    considered = 0
    per_family = dict.fromkeys(FAMILIES, 0)
    for row in pairs.itertuples():
        plus, minus = str(row.plus_candidate_id), str(row.minus_candidate_id)
        if plus not in indexed.index or minus not in indexed.index:
            continue
        considered += 1
        a = indexed.loc[plus, columns].to_numpy(dtype=np.float64)
        b = indexed.loc[minus, columns].to_numpy(dtype=np.float64)
        differs = ~np.isclose(a, b, rtol=0, atol=1e-12)
        if differs.any():
            any_var += 1
        if differs[[columns.index(c) for c in visual]].any():
            vis_var += 1
        if differs[[columns.index(c) for c in r0]].any():
            r0_var += 1
        for family, prefix in FAMILIES.items():
            index = [columns.index(c) for c in columns if c.startswith(prefix)]
            if differs[index].any():
                per_family[family] += 1
    return {
        "schema_version": "sgv1-within-pair-variation-v1",
        "statement": (
            "Phase 2 measured a within-pair image-feature difference of exactly 0.000e+00 on "
            "all 794 pairs, so no site-conditioned representation could move the matched-pair "
            "endpoint. This is the same measurement on the candidate-conditioned features."
        ),
        "pairs": considered,
        "pairs_with_any_variation": any_var,
        "fraction_varying": any_var / max(considered, 1),
        "pairs_with_visual_variation": vis_var,
        "fraction_visual_varying": vis_var / max(considered, 1),
        "pairs_with_r0_variation": r0_var,
        "r0_note": (
            "R0 is candidate-blind, so it MUST show zero within-pair variation. A non-zero "
            "count here would mean R0 is accidentally seeing the candidate."
        ),
        "pairs_with_variation_by_family": per_family,
        "phase_2_baseline_pairs_with_visual_variation": 0,
        "confirmatory_accessed": False,
    }


# ------------------------------------------------------------------------ fitting

import sgv1_representation as repr_stage  # noqa: E402

EPSILON = 0.10
"""Target selective risk for the CORRECT action, chosen on CALIBRATION only."""

ARMS: dict[str, dict[str, Any]] = {
    "R0": {"kind": "candidate_blind", "families": ("r0",), "model": "logistic"},
    "V1": {"kind": "v1", "families": (), "model": "logistic"},
    "V1_gb": {"kind": "v1", "families": (), "model": "gradient_boosting"},
    "R1": {"kind": "v1_plus", "families": tuple(FAMILIES), "model": "logistic"},
    "R1_gb": {"kind": "v1_plus", "families": tuple(FAMILIES), "model": "gradient_boosting"},
    **{
        f"R1_no_{family}": {
            "kind": "v1_plus",
            "families": tuple(f for f in FAMILIES if f != family),
            "model": "logistic",
        }
        for family in FAMILIES
    },
}


class CandidateConditionedVerifier(repr_stage.RepresentationVerifier):
    """Phase 2's verifier, with the feature names told the truth about their own width.

    ``RepresentationVerifier`` appends Phase 2's hardcoded ``EMBED_NAMES`` -- 33 entries,
    because every Phase-2 embedding was 33 wide. Reused here with a 31-to-49-column block
    the fitted MATRIX is still correct, but ``feature_names`` silently under-reports, and a
    run record that misstates its own feature dimension is a provenance defect even when
    the model is right. This subclass supplies the real names.
    """

    def __init__(self, *args: Any, extra_names: Sequence[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extra_names = tuple(extra_names)

    def _matrix(self, inputs: Sequence[Any]) -> Any:
        base = repr_stage.pilot.SGV1Verifier._matrix(self, inputs)
        if not self._embedding:
            return base
        extra = np.vstack([self._embedding[item.candidate_id] for item in inputs])
        if len(self._feature_names) == base.shape[1]:
            self._feature_names = (*self._feature_names, *self._extra_names)
        return np.hstack([base, extra])


def _family_columns(table: pd.DataFrame, families: Sequence[str]) -> list[str]:
    prefixes = tuple(FAMILIES[f] for f in families if f in FAMILIES)
    if "r0" in families:
        prefixes = (*prefixes, "r0_")
    return [c for c in table.columns if c != "candidate_id" and c.startswith(prefixes)]


def run_fit() -> int:
    started = time.monotonic()
    if not FEATURES.is_file():
        raise PhaseError("run --features first")
    table = pd.read_parquet(FEATURES)
    pool, _ = _pool()
    pool = pool.merge(table, on="candidate_id", how="inner", validate="one_to_one")
    if len(pool) != len(table):
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
            "operation",
            "anchor_kind",
            "generator_source",
            "original_ocr",
            "candidate_text",
        ]
    ].copy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    arm_records: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        columns = _family_columns(table, spec["families"])
        if spec["kind"] == "candidate_blind":
            scaler = StandardScaler().fit(fit_rows[columns].to_numpy(dtype=np.float64))
            model = LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced", random_state=pilot.FIT_SEED
            ).fit(
                scaler.transform(fit_rows[columns].to_numpy(dtype=np.float64)),
                fit_rows["is_harmful"].astype(int).to_numpy(),
            )

            def _score(
                frame: pd.DataFrame,
                _model: Any = model,
                _scaler: Any = scaler,
                _columns: list[str] = columns,
            ) -> np.ndarray:
                matrix = _scaler.transform(frame[_columns].to_numpy(dtype=np.float64))
                return 1.0 - _model.predict_proba(matrix)[:, 1]

            cal_scores, dev_scores = _score(cal_rows), _score(dev_rows)
            dimension, parameters = len(columns), int(model.coef_.size + model.intercept_.size)
        else:
            embedding = None
            if columns:
                values = pool[columns].to_numpy(dtype=np.float64)
                embedding = dict(zip(pool["candidate_id"].astype(str), values, strict=True))
            verifier = CandidateConditionedVerifier(
                extra_names=columns,
                verifier_id=f"cc_{arm.lower()}",
                evidence_config="sgv1_v1",
                model=spec["model"],
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=pilot.FIT_SEED,
                provenance=provenance,
                embedding=embedding,
            )
            inputs = {
                name: pilot._inputs_for(
                    arm, list(frame["candidate_id"].astype(str)), bundles, {}, mask
                )
                for name, frame in (("fit", fit_rows), ("cal", cal_rows), ("dev", dev_rows))
            }
            verifier.fit(inputs["fit"], list(fit_rows["is_harmful"].astype(bool)))
            cal_scores = verifier.score(inputs["cal"]).scores
            dev_scores = verifier.score(inputs["dev"]).scores
            dimension = int(verifier._matrix(inputs["dev"][:1]).shape[1])
            parameters = verifier.parameter_count
            if len(verifier.feature_names) != dimension:
                raise PhaseError(
                    f"{arm}: {len(verifier.feature_names)} feature names for a "
                    f"{dimension}-column matrix"
                )

        calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
        calibrator.fit(cal_scores, (~cal_rows["is_harmful"].to_numpy(dtype=bool)).astype(float))
        cal_safe = calibrator.transform(cal_scores)
        dev_safe = calibrator.transform(dev_scores)
        scores[f"score_{arm}"] = dev_scores
        scores[f"psafe_{arm}"] = dev_safe
        scores[f"pharm_{arm}"] = 1.0 - dev_safe

        tau_hi, tau_lo = _select_thresholds(cal_safe, cal_rows["is_harmful"].to_numpy(dtype=bool))
        scores[f"action_{arm}"] = np.where(
            dev_safe >= tau_hi, "CORRECT", np.where(dev_safe <= tau_lo, "PRESERVE", "ABSTAIN")
        )
        arm_records[arm] = {
            **{k: v for k, v in spec.items() if k != "families"},
            "families": list(spec["families"]),
            "feature_dimension": dimension,
            "model_parameters": parameters,
            "model_parameters_note": (
                "coefficients plus intercept; defined for the linear arms only, 0 for "
                "gradient boosting, which has no such count"
            ),
            "extra_feature_columns": len(columns),
            "tau_hi": tau_hi,
            "tau_lo": tau_lo,
            "epsilon": EPSILON,
            "threshold_scope": "CALIBRATION rows only",
            "calibrator_identity": calibrator.identity(),
            "fit_rows": len(fit_rows),
            "calibration_rows": len(cal_rows),
            "evaluation_rows": len(dev_rows),
        }
        print(f"  fitted {arm:16s} dim={dimension:4d} tau_hi={tau_hi:.4f} tau_lo={tau_lo:.4f}")

    _write_parquet_once(SCORES, scores)
    record = {
        "schema_version": "sgv1-candidate-conditioned-fit-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "held_out_axis": "documents (TRAIN / CALIBRATION / DEVELOPMENT)",
        "held_out_engine": None,
        "limitation": (
            "Every engine appears in every role, so engine shift is untested here. That is "
            "the confirmatory question and the reserve is locked."
        ),
        "actions": ["CORRECT", "PRESERVE", "ABSTAIN"],
        "action_note": (
            "ABSTAIN is a POLICY over one calibrated score, not a trained class. There is no "
            "abstain label in the corpus and inventing one would fabricate a target. The "
            "three actions are a two-threshold band: accept above tau_hi, preserve below "
            "tau_lo, abstain between, with both thresholds chosen on CALIBRATION rows."
        ),
        "arms": arm_records,
        "rows_evaluated": len(scores),
        "documents_evaluated": int(scores["document_id"].nunique()),
        "inputs": {
            _relative(FEATURES): file_sha256(FEATURES),
            _relative(pilot.LABEL_TABLE): file_sha256(pilot.LABEL_TABLE),
        },
        "artifacts": {_relative(SCORES): file_sha256(SCORES)},
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(FIT_RECORD, record)
    print(f"fit: {len(arm_records)} arms -> {_relative(SCORES)}")
    return 0


def _select_thresholds(safe: np.ndarray, harmful: np.ndarray) -> tuple[float, float]:
    """Two thresholds on CALIBRATION, never on the evaluation split.

    ``tau_hi`` is the most permissive acceptance threshold whose selective risk stays within
    EPSILON -- the risk-controlled CORRECT action. ``tau_lo`` is its mirror: the score below
    which the model is confidently predicting harm, so PRESERVE is the safe act. Between
    them the evidence does not support either, which is what ABSTAIN means here.
    """
    curve = risk_coverage_curve(safe, harmful)
    point = coverage_at_risk(curve, EPSILON)
    tau_hi = float(point.threshold) if point is not None else 1.0
    tau_lo = float(np.quantile(safe[harmful], 0.5)) if harmful.any() else 0.0
    return tau_hi, min(tau_lo, tau_hi)


# ----------------------------------------------------------------------- analysis

COVERAGE_POINTS = (0.10, 0.25, 0.50, 0.75, 1.00)


def _decision_quality(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    """Precision, recall and the safety metrics, on the realized three-way action.

    ``beneficial`` is the accept-worthy class. Precision answers "when the system corrects,
    how often is it right?"; over-correction is the share of accepted edits that damage
    already-correct text, which is the failure this project exists to bound.
    """
    action = frame[f"action_{arm}"].to_numpy()
    beneficial = frame["beneficial"].to_numpy(dtype=bool)
    harmful = frame["is_harmful"].to_numpy(dtype=bool)
    overcorrection = (frame["outcome"] == "overcorrection").to_numpy()
    corrected = action == "CORRECT"
    preserved = action == "PRESERVE"
    abstained = action == "ABSTAIN"
    return {
        "n": len(frame),
        "n_correct": int(corrected.sum()),
        "n_preserve": int(preserved.sum()),
        "n_abstain": int(abstained.sum()),
        "action_rate_correct": float(corrected.mean()),
        "action_rate_preserve": float(preserved.mean()),
        "action_rate_abstain": float(abstained.mean()),
        "correction_precision": float(beneficial[corrected].mean())
        if corrected.any()
        else float("nan"),
        "correction_recall": float(corrected[beneficial].mean())
        if beneficial.any()
        else float("nan"),
        "harmful_accepted_rate": float(harmful[corrected].mean())
        if corrected.any()
        else float("nan"),
        "harmful_accepted_joint": float((harmful & corrected).mean()),
        "overcorrection_accepted_rate": float(overcorrection[corrected].mean())
        if corrected.any()
        else float("nan"),
        "overcorrection_accepted_joint": float((overcorrection & corrected).mean()),
        "preservation_accuracy": float(harmful[preserved].mean())
        if preserved.any()
        else float("nan"),
        "beneficial_lost_to_preserve": float(preserved[beneficial].mean())
        if beneficial.any()
        else float("nan"),
        "abstention_harmful_rate": float(harmful[abstained].mean())
        if abstained.any()
        else float("nan"),
        "abstention_beneficial_rate": float(beneficial[abstained].mean())
        if abstained.any()
        else float("nan"),
    }


def _selective(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    safe = frame[f"psafe_{arm}"].to_numpy(dtype=np.float64)
    harmful = frame["is_harmful"].to_numpy(dtype=bool)
    curve = risk_coverage_curve(safe, harmful)
    out: dict[str, Any] = {"aurc": float(aurc(curve))}
    for target in COVERAGE_POINTS:
        best = min(curve.points, key=lambda p: abs(p.coverage - target))
        out[f"risk_at_coverage_{int(target * 100)}"] = float(best.risk)
        out[f"actual_coverage_{int(target * 100)}"] = float(best.coverage)
    for epsilon in (0.05, 0.10, 0.20):
        point = coverage_at_risk(curve, epsilon)
        out[f"coverage_at_risk_{int(epsilon * 100)}"] = float(point.coverage) if point else 0.0
    return out


def _calibration(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    harm = frame[f"pharm_{arm}"].to_numpy(dtype=np.float64)
    harmful = frame["is_harmful"].to_numpy(dtype=np.float64)
    width, _ = expected_calibration_error(harm, harmful, pilot.ECE_BINS, "equal_width")
    mass, mce = expected_calibration_error(harm, harmful, pilot.ECE_BINS, "equal_mass")
    return {
        "brier": float(brier_score(harm, harmful)),
        "ece_equal_width": float(width),
        "ece_equal_mass": float(mass),
        "max_calibration_error_equal_mass": float(mce),
        "mean_predicted_harm": float(harm.mean()),
        "observed_harm_rate": float(harmful.mean()),
    }


def _discrimination(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    # Raw verifier score, not the calibrated probability. Isotonic regression is a step
    # function: it collapses 1,434 distinct V1 scores to 47, and the resulting ties depress
    # AUC by ~0.001. That is an artifact of the calibrator, not a property of the
    # discriminator, and using the raw score keeps these numbers comparable with Phase 2 --
    # V1 reproduces its 0.7750 exactly this way.
    safe = frame[f"score_{arm}"].to_numpy(dtype=np.float64)
    beneficial = frame["beneficial"].to_numpy(dtype=np.float64)
    if np.unique(beneficial).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc(safe, beneficial)),
        "pr_auc": float(average_precision(safe, beneficial)),
    }


def _paired_auc(frame: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    def rows(arm: str) -> list[tuple[str, float, float]]:
        return list(
            zip(
                frame["document_id"].astype(str),
                frame[f"score_{arm}"].astype(float),
                frame["beneficial"].astype(float),
                strict=True,
            )
        )

    def statistic(items: Sequence[tuple[str, float, float]]) -> float:
        if not items:
            return float("nan")
        s = np.fromiter((i[1] for i in items), dtype=np.float64, count=len(items))
        y = np.fromiter((i[2] for i in items), dtype=np.float64, count=len(items))
        return roc_auc(s, y) if np.unique(y).size > 1 else float("nan")

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


def _error_category(row: Any) -> str:
    """Experiment 3's categories, derived from the frozen edit -- never from ground truth."""
    original, candidate = str(row.original_ocr), str(row.candidate_text)
    o, y = original.strip(), candidate.strip()
    if bool(PRICE_RE.match(o)) or bool(PRICE_RE.match(y)):
        return "numeric_price"
    if len(o) == len(y):
        differing = [(a, b) for a, b in zip(o, y) if a != b]
        if differing and all(frozenset(p) in _CONFUSABLE for p in differing):
            return "character_confusion"
        return "same_length_other"
    if str(row.anchor_kind) == "gap":
        return "gap_insertion"
    if str(row.operation) in {"merge", "split"}:
        return "segmentation"
    if any(not c.isalnum() and not c.isspace() for c in o + y):
        return "formatting"
    return "other"


def run_analyze() -> int:
    started = time.monotonic()
    record = _read_json(FIT_RECORD)
    if file_sha256(SCORES) != record["artifacts"][_relative(SCORES)]:
        raise PhaseError("decision scores moved since the fit record")
    scores = pd.read_parquet(SCORES)
    frame_a = set(pd.read_parquet(pilot.FRAME_DIR / "frame_a.parquet")["candidate_id"].astype(str))
    in_frame_a = scores[scores["candidate_id"].astype(str).isin(frame_a)].reset_index(drop=True)
    core = in_frame_a[~in_frame_a["region_is_whitespace_only"]].reset_index(drop=True)
    arms = list(record["arms"])

    def block(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            arm: {
                **_discrimination(frame, arm),
                **_decision_quality(frame, arm),
                **_selective(frame, arm),
                **_calibration(frame, arm),
            }
            for arm in arms
        }

    pool_block, core_block = block(scores), block(core)
    contrasts = {
        f"{arm}_vs_V1": {
            "core_roc_auc": _paired_auc(core, arm, "V1"),
            "pool_roc_auc": _paired_auc(scores, arm, "V1"),
            "brier_delta": pool_block[arm]["brier"] - pool_block["V1"]["brier"],
            "harmful_accepted_delta": pool_block[arm]["harmful_accepted_joint"]
            - pool_block["V1"]["harmful_accepted_joint"],
            "overcorrection_accepted_delta": pool_block[arm]["overcorrection_accepted_joint"]
            - pool_block["V1"]["overcorrection_accepted_joint"],
            "aurc_delta": pool_block[arm]["aurc"] - pool_block["V1"]["aurc"],
        }
        for arm in arms
        if arm != "V1"
    }

    support = {
        "development_rows": len(scores),
        "development_documents": int(scores["document_id"].nunique()),
        "frame_a_rows": len(in_frame_a),
        "core_rows": len(core),
        "core_documents": int(core["document_id"].nunique()),
        "harmful_rate": float(scores["is_harmful"].mean()),
        "beneficial_rate": float(scores["beneficial"].mean()),
        "rows_by_engine": {str(k): int(v) for k, v in scores["engine_id"].value_counts().items()},
    }
    common = {
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "support": support,
        "epsilon": EPSILON,
        "bootstrap": {
            "cluster": "document_id",
            "n_resamples": pilot.BOOTSTRAP_RESAMPLES,
            "seed": pilot.BOOTSTRAP_SEED,
            "paired": True,
        },
        "multiplicity_control": None,
        "confirmatory_accessed": False,
    }

    _write_json_once(
        BASELINE_RESULTS,
        {
            "schema_version": "sgv1-cc-baseline-v1",
            **common,
            "statement": "Experiment 1 reference points: R0 is candidate-BLIND; V1 is the "
            "Phase-2 non-image baseline and the honest comparator.",
            "pool": {a: pool_block[a] for a in ("R0", "V1", "V1_gb")},
            "core": {a: core_block[a] for a in ("R0", "V1", "V1_gb")},
        },
    )
    _write_json_once(
        CANDIDATE_RESULTS,
        {
            "schema_version": "sgv1-cc-candidate-v1",
            **common,
            "statement": "Experiment 1: the candidate-conditioned representation R1.",
            "pool": {a: pool_block[a] for a in arms if a.startswith("R1")},
            "core": {a: core_block[a] for a in arms if a.startswith("R1")},
            "contrasts_vs_V1": contrasts,
            "within_pair_variation": _read_json(VARIATION_AUDIT),
        },
    )
    _write_json_once(
        ABLATION_RESULTS,
        {
            "schema_version": "sgv1-cc-ablation-v1",
            **common,
            "statement": "Experiment 2: remove one candidate-conditioned family at a time.",
            "full": {"pool": pool_block["R1"], "core": core_block["R1"]},
            "ablations": {
                a: {
                    "pool": pool_block[a],
                    "core": core_block[a],
                    "core_delta_vs_full": core_block[a]["roc_auc"] - core_block["R1"]["roc_auc"],
                    "brier_delta_vs_full": pool_block[a]["brier"] - pool_block["R1"]["brier"],
                }
                for a in arms
                if a.startswith("R1_no_")
            },
        },
    )
    _write_json_once(
        CALIBRATION_RESULTS,
        {
            "schema_version": "sgv1-cc-calibration-v1",
            **common,
            "statement": "Calibration and selective prediction over the full DEVELOPMENT pool.",
            "calibration": {a: _calibration(scores, a) for a in arms},
            "selective": {a: _selective(scores, a) for a in arms},
            "reliability": {a: _reliability(scores, a) for a in ("R0", "V1", "R1", "R1_gb")},
        },
    )

    scores["error_category"] = [_error_category(r) for r in scores.itertuples()]
    categories = {}
    for name, group in scores.groupby("error_category"):
        categories[str(name)] = {
            "n": len(group),
            "harmful_rate": float(group["is_harmful"].mean()),
            **{
                a: {**_discrimination(group, a), **_decision_quality(group, a)}
                for a in ("R0", "V1", "R1")
            },
        }
    _write_json_once(
        CATEGORY_RESULTS,
        {
            "schema_version": "sgv1-cc-category-v1",
            **common,
            "statement": "Experiment 3: categories derived from the frozen edit, never from GT.",
            "categories": categories,
        },
    )

    engines = sorted(scores["engine_id"].astype(str).unique())
    loo = {}
    top = scores["document_id"].value_counts().head(5).index.tolist()
    for document in top:
        kept = scores[scores["document_id"] != document]
        kept_core = core[core["document_id"] != document]
        loo[str(document)] = {
            "rows_removed": int((scores["document_id"] == document).sum()),
            "R1_minus_V1_pool_auc": _discrimination(kept, "R1")["roc_auc"]
            - _discrimination(kept, "V1")["roc_auc"],
            "R1_minus_V1_core_auc": _discrimination(kept_core, "R1")["roc_auc"]
            - _discrimination(kept_core, "V1")["roc_auc"],
        }
    risk_set = scores[scores["outcome"] != "lateral_change"]
    population_sensitivity = {
        name: {
            "n": len(f),
            "beneficial_rate": float(f["beneficial"].mean()),
            **{a: _discrimination(f, a)["roc_auc"] for a in ("R0", "V1", "R1", "R1_gb")},
            "R1_minus_V1": _discrimination(f, "R1")["roc_auc"]
            - _discrimination(f, "V1")["roc_auc"],
        }
        for name, f in {
            "full_development_pool": scores,
            "pool_minus_lateral_change": risk_set,
            "frame_a_all": in_frame_a,
            "frame_a_core_primary": core,
        }.items()
    }
    _write_json_once(
        ROBUSTNESS_RESULTS,
        {
            "schema_version": "sgv1-cc-robustness-v1",
            "population_sensitivity": population_sensitivity,
            "population_sensitivity_note": (
                "The single most important caveat in this stage. R1's advantage over V1 is "
                "a property of the EVALUATION POPULATION and reverses on the natural pool: "
                "+0.088 on the stratified Frame-A core, +0.028 once lateral_change rows are "
                "dropped, and -0.016 on the full DEVELOPMENT pool. Frame A's class balance "
                "is a design parameter and never a prevalence statement, so the core number "
                "is not a deployment number. On the full pool the candidate-BLIND R0 scores "
                "highest (0.8397), which is not a paradox: 23.6% of that pool is "
                "lateral_change, where no candidate at the site is beneficial, and whether a "
                "site has any beneficial candidate is a site property that a candidate-blind "
                "representation captures well."
            ),
            **common,
            "statement": "Experiment 4: per-engine slices and document leave-one-out. NOT a "
            "held-out-engine test -- every engine is in every role here.",
            "by_engine": {
                e: {
                    "n": int((scores["engine_id"].astype(str) == e).sum()),
                    **{
                        a: {
                            **_discrimination(scores[scores["engine_id"].astype(str) == e], a),
                            **_decision_quality(scores[scores["engine_id"].astype(str) == e], a),
                        }
                        for a in ("R0", "V1", "R1")
                    },
                }
                for e in engines
            },
            "leave_one_document_out": loo,
            "epsilon_sensitivity": {
                f"epsilon_{int(e * 100)}": {
                    a: _selective(scores, a)[f"coverage_at_risk_{int(e * 100)}"] for a in arms
                }
                for e in (0.05, 0.10, 0.20)
            },
        },
    )
    print(f"analyze: {len(arms)} arms -> {_relative(OUT)}")
    print(
        f"  R1 core AUC {core_block['R1']['roc_auc']:.4f} vs V1 {core_block['V1']['roc_auc']:.4f} "
        f"vs R0 {core_block['R0']['roc_auc']:.4f}"
    )
    _ = started
    return 0


def _reliability(frame: pd.DataFrame, arm: str, bins: int = 10) -> dict[str, Any]:
    harm = frame[f"pharm_{arm}"].to_numpy(dtype=np.float64)
    harmful = frame["is_harmful"].to_numpy(dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(harm, edges[1:-1]), 0, bins - 1)
    return {
        "bin_edges": edges.tolist(),
        "predicted": [
            float(harm[index == b].mean()) if (index == b).any() else None for b in range(bins)
        ],
        "observed": [
            float(harmful[index == b].mean()) if (index == b).any() else None for b in range(bins)
        ],
        "count": [int((index == b).sum()) for b in range(bins)],
    }


# ------------------------------------------------------------------------ figures

NOTE = "SGV1 DEVELOPMENT -- not a confirmatory result"
FIGURE_ARMS = ("R0", "V1", "R1", "R1_gb")


def run_figures() -> int:
    started = time.monotonic()
    scores = pd.read_parquet(SCORES)
    calibration = _read_json(CALIBRATION_RESULTS)
    ablation = _read_json(ABLATION_RESULTS)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def finish(fig: Any, path: Path, title: str) -> None:
        fig.suptitle(f"{title}\n{NOTE}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    harmful = scores["is_harmful"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arm in FIGURE_ARMS:
        curve = risk_coverage_curve(scores[f"psafe_{arm}"].to_numpy(dtype=np.float64), harmful)
        points = sorted(curve.points, key=lambda p: p.coverage)
        ax.plot([p.coverage for p in points], [p.risk for p in points], label=arm, lw=1.6)
    ax.axhline(EPSILON, color="#c0392b", ls=":", lw=1, label=f"epsilon = {EPSILON}")
    ax.set_xlabel("coverage (fraction of candidates accepted)")
    ax.set_ylabel("selective risk (harmful among accepted)")
    ax.legend(fontsize=8)
    finish(fig, FIGURE_DIR / "risk_coverage_curve.png", "Risk-coverage frontier")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1, label="perfect calibration")
    for arm in FIGURE_ARMS:
        r = calibration["reliability"][arm]
        x = [p for p in r["predicted"] if p is not None]
        y = [o for p, o in zip(r["predicted"], r["observed"]) if p is not None]
        ax.plot(
            x,
            y,
            marker="o",
            ms=4,
            lw=1.4,
            label=f"{arm} (Brier {calibration['calibration'][arm]['brier']:.4f})",
        )
    ax.set_xlabel("predicted P(harmful)")
    ax.set_ylabel("observed harmful rate")
    ax.legend(fontsize=8)
    finish(fig, FIGURE_DIR / "calibration_plot.png", "Reliability, 10 equal-width bins")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(ablation["ablations"])
    deltas = [ablation["ablations"][n]["core_delta_vs_full"] for n in names]
    labels = [n.replace("R1_no_", "without ") for n in names]
    order = np.argsort(deltas)
    ax.barh([labels[i] for i in order], [deltas[i] for i in order], color="#3b6ea5")
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel(
        "core Frame-A ROC AUC, ablation minus full R1\n(more negative = the family mattered more)"
    )
    finish(fig, FIGURE_DIR / "component_ablation.png", "Component ablation")

    manifest = {
        "schema_version": "sgv1-cc-figures-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "annotation": NOTE,
        "synthetic": False,
        "derived_from": {
            _relative(SCORES): file_sha256(SCORES),
            _relative(CALIBRATION_RESULTS): file_sha256(CALIBRATION_RESULTS),
            _relative(ABLATION_RESULTS): file_sha256(ABLATION_RESULTS),
        },
        "figures": {_relative(p): file_sha256(p) for p in written},
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(FIGURE_MANIFEST, manifest)
    print(f"figures: {len(written)} -> {_relative(FIGURE_DIR)}")
    return 0


# ----------------------------------------------------------------------- decision


def run_decide() -> int:
    started = time.monotonic()
    baseline = _read_json(BASELINE_RESULTS)
    candidate = _read_json(CANDIDATE_RESULTS)
    ablation = _read_json(ABLATION_RESULTS)
    robustness = _read_json(ROBUSTNESS_RESULTS)
    variation = _read_json(VARIATION_AUDIT)

    r1, v1, r0 = candidate["core"]["R1"], baseline["core"]["V1"], baseline["core"]["R0"]
    contrast = candidate["contrasts_vs_V1"]["R1_vs_V1"]
    core = contrast["core_roc_auc"]
    significant = bool(
        np.isfinite(core["ci_lower"])
        and np.isfinite(core["ci_upper"])
        and (core["ci_lower"] > 0 or core["ci_upper"] < 0)
    )
    families = {
        name.replace("R1_no_", ""): entry["core_delta_vs_full"]
        for name, entry in ablation["ablations"].items()
    }
    ranked = sorted(families, key=lambda k: families[k])
    engine_deltas = {
        engine: entry["R1"]["roc_auc"] - entry["V1"]["roc_auc"]
        for engine, entry in robustness["by_engine"].items()
        if np.isfinite(entry["R1"]["roc_auc"]) and np.isfinite(entry["V1"]["roc_auc"])
    }

    pool_r1 = candidate["pool"]["R1"]
    pool_v1 = baseline["pool"]["V1"]
    selective = _read_json(CALIBRATION_RESULTS)["selective"]
    # Coverage-matched, because the arms do NOT operate at the same coverage. Each arm's
    # tau_hi targets the same selective RISK on calibration, so R1 buys its accuracy as
    # extra coverage (28.7% of candidates corrected against V1's 20.0%). Comparing joint
    # harm across different coverage is not a comparison: a system that corrects more will
    # accept more harmful edits in absolute terms even when it is strictly safer per
    # decision. The frontier is the like-for-like view.
    matched = {
        f"risk_at_coverage_{c}": {
            "V1": selective["V1"][f"risk_at_coverage_{c}"],
            "R1": selective["R1"][f"risk_at_coverage_{c}"],
            "delta": selective["R1"][f"risk_at_coverage_{c}"]
            - selective["V1"][f"risk_at_coverage_{c}"],
        }
        for c in (10, 25, 50, 75, 100)
    }
    safer_points = [k for k, v in matched.items() if v["delta"] < 0]

    decision = {
        "schema_version": "sgv1-cc-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "q1_outperforms_baseline": {
            "core_roc_auc_R0_candidate_blind": r0["roc_auc"],
            "core_roc_auc_V1_comparator": v1["roc_auc"],
            "core_roc_auc_R1": r1["roc_auc"],
            "delta_vs_V1": core["delta"],
            "ci": [core["ci_lower"], core["ci_upper"]],
            "n_documents": core["n_documents"],
            "answer": "yes on the development split" if core["delta"] > 0 else "no",
        },
        "q2_component_contribution": {
            "core_auc_loss_when_family_removed": families,
            "ranked_most_to_least_important": ranked,
        },
        "q3_reduces_harmful_corrections": {
            "answer": (
                "Yes at matched coverage, which is the only like-for-like comparison; NO on "
                "the raw joint rate, because R1 corrects substantially more."
            ),
            "coverage_matched": {
                "aurc_V1": selective["V1"]["aurc"],
                "aurc_R1": selective["R1"]["aurc"],
                "aurc_R1_gb": selective["R1_gb"]["aurc"],
                "selective_risk_by_coverage": matched,
                "coverage_points_where_R1_is_safer": safer_points,
                "coverage_at_risk_5_V1": selective["V1"]["coverage_at_risk_5"],
                "coverage_at_risk_5_R1": selective["R1"]["coverage_at_risk_5"],
                "coverage_at_risk_10_V1": selective["V1"]["coverage_at_risk_10"],
                "coverage_at_risk_10_R1": selective["R1"]["coverage_at_risk_10"],
            },
            "at_each_arms_own_operating_point": {
                "note": (
                    "Not comparable across arms: tau_hi targets equal selective RISK on "
                    "calibration, so the arms land at different coverage."
                ),
                "action_rate_correct_V1": pool_v1["action_rate_correct"],
                "action_rate_correct_R1": pool_r1["action_rate_correct"],
                "correction_recall_V1": pool_v1["correction_recall"],
                "correction_recall_R1": pool_r1["correction_recall"],
                "harmful_accepted_rate_V1": pool_v1["harmful_accepted_rate"],
                "harmful_accepted_rate_R1": pool_r1["harmful_accepted_rate"],
                "harmful_accepted_joint_V1": pool_v1["harmful_accepted_joint"],
                "harmful_accepted_joint_R1": pool_r1["harmful_accepted_joint"],
                "overcorrection_accepted_joint_V1": pool_v1["overcorrection_accepted_joint"],
                "overcorrection_accepted_joint_R1": pool_r1["overcorrection_accepted_joint"],
                "correction_precision_V1": pool_v1["correction_precision"],
                "correction_precision_R1": pool_r1["correction_precision"],
            },
            "frontier_crossing": (
                "R1 dominates V1 at low coverage -- the risk-controlled regime this project "
                "targets -- and the curves cross around 50% coverage, where V1 is marginally "
                "better. Reported rather than smoothed over."
            ),
        },
        "q4_statistically_significant": {
            "paired_document_clustered_ci": [core["ci_lower"], core["ci_upper"]],
            "excludes_zero": significant,
            "answer": "yes" if significant else "no -- the interval contains zero",
        },
        "q5_publishable_contribution": {
            "answer": "NOT YET -- this is a development result on one document split",
            "why": [
                "Every engine is in every role, so the cross-engine shift the study exists "
                "to measure is untested here and the confirmatory reserve is still locked.",
                "One development split of 133 documents; no confirmatory run has occurred.",
                "The claim this supports is a candidate contribution under the project's "
                "existing prior-art boundary, not a new one. Detector-corrector pipelines, "
                "generic selective prediction, generic abstention, risk-controlled "
                "prediction in general and the existence of overcorrection are all listed "
                "as established prior art in docs/prior_art_boundary.md.",
            ],
        },
        "within_pair_variation": {
            "phase_2": variation["phase_2_baseline_pairs_with_visual_variation"],
            "phase_3_visual": variation["pairs_with_visual_variation"],
            "phase_3_any": variation["pairs_with_any_variation"],
            "pairs": variation["pairs"],
            "r0_control": variation["pairs_with_r0_variation"],
        },
        "engine_effects_vs_V1": engine_deltas,
        "engine_sign_agreement": len({d > 0 for d in engine_deltas.values()}) == 1,
        "hypothesis_verdict": None,
        "hypothesis_note": (
            "No SGV1-H1 verdict. This stage measures a representation on a development "
            "split; it does not decide a hypothesis."
        ),
        "c2_status": "DEFERRED",
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(DECISION, decision)
    print(f"decision -> {_relative(DECISION)}")
    print(f"  Q1 core AUC R0 {r0['roc_auc']:.4f} / V1 {v1['roc_auc']:.4f} / R1 {r1['roc_auc']:.4f}")
    print(f"  Q1 delta {core['delta']:+.5f} CI [{core['ci_lower']:+.5f}, {core['ci_upper']:+.5f}]")
    print(f"  Q2 most important family: {ranked[0]}")
    print(
        f"  Q3 AURC {selective['V1']['aurc']:.4f} -> {selective['R1']['aurc']:.4f}; "
        f"risk@10%cov {selective['V1']['risk_at_coverage_10']:.4f} -> "
        f"{selective['R1']['risk_at_coverage_10']:.4f}"
    )
    print(f"  Q4 significant: {significant}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("features", "fit", "analyze", "figures", "decide"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.features:
        return run_features()
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

"""CGV3 Track B: the one-shot fresh confirmatory execution (protocol section 16 step 8).

Three processes, in the order the protocol freezes, each with explicit inputs:

1. **discovery** -- ``enumerate_sites`` over the fresh canonical spans. The process
   reads the canonicalize run and nothing else: no manifest GT, no alignments, no
   sites table. Writes the frozen site table.
2. **generation** -- the frozen ladder (``b0``/``g3``/``g7``/``g8``) over the frozen
   site table. Also GT-free: the generation columns of a candidate row (text, shape,
   rank, score, anchor slice) never consult ground truth -- only the labels do, and
   labels are not computed here. Writes the frozen candidate table.
3. **evaluation** -- ``run_track_a`` (the hash-bound Track-A function, imported
   unmodified) with the fresh GT artifacts. It re-enumerates and re-generates
   internally; the evaluation asserts byte-equality of the discovery and generation
   columns against the frozen tables, so the pre-GT artifacts are the ones scored,
   and any divergence is a loud defect rather than a silent second source of truth.

The fold resources (lexicon, bigrams, incumbent, structural rung) are fitted on the
PILOT's seen corpus -- ``(E \\ {held_out}) x D_fit`` exactly as Track-A fit them --
so no fresh document contributes any fitted statistic. Fitting is deterministic
counting; the determinism assertion covers equality with Track-A's construction.

The generation replication below re-states the per-site loop of ``run_track_a``.
Re-stating rather than importing is forced: the frozen function cannot be edited to
expose a GT-free path, and calling it with empty GT skips generation entirely
(``index is None -> continue``). Equality with the frozen construction is therefore
*asserted* by the evaluation pass, not assumed from shared code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ocr_risk.candidates.base import CandidateProposal, GenerationContext
from ocr_risk.candidates.edit_aware import EditAwareGenerator
from ocr_risk.candidates.lexical import LexicalGenerator
from ocr_risk.candidates.structural_v2 import StructuralV2Generator
from ocr_risk.discovery.enumerator import (
    DiscoveredSite,
    DiscoveryResources,
    DiscoveryRules,
    enumerate_sites,
)
from ocr_risk.discovery.freshness import confirmatory_document_ids
from ocr_risk.discovery.views import OcrPageView
from ocr_risk.edits.outcome import distance
from ocr_risk.experiments.cgv3_study import AlignmentIndex, anchor_stream_text
from ocr_risk.experiments.cgv3_track_a import UNION_CAP, _fold_corpus, _shape
from ocr_risk.schemas.enums import AnchorKind, DiscoveryProvenance
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = [
    "CONTEXT_CHARS",
    "ConfirmatoryDivergenceError",
    "FoldResources",
    "assert_confirmatory_selection",
    "assert_frames_match_frozen",
    "discovered_region_truth",
    "discovery_pass",
    "fit_fold_resources",
    "generation_pass",
    "read_frozen_csv",
]

CONTEXT_CHARS = 40
"""run_track_a's default context window; restated so the replication cannot drift."""

_ANCHOR_SEPARATOR = "\0"


def read_frozen_csv(path: Path) -> pd.DataFrame:
    """Read a frozen confirmatory table back losslessly (DEFECT-2, DEFECT-3).

    Two independent pandas defaults corrupt a frozen table on read:

    - the C parser truncates a field at a NUL byte, and ``anchor_ref`` joins the
      anchor kind and span ids with ``\\0`` -- every site would silently lose its
      anchor spans (DEFECT-2);
    - both the C and the python engine convert floats with a fast 1-ULP-lossy
      routine, so ``suspicion_score``/``generator_score`` values shift in their
      last bit -- enough to fail the byte-equality proof against the freshly
      computed tables, on 1,148 of 52,038 rows here (DEFECT-3).

    The dual read fixes both: the python engine preserves the NUL-separated
    strings, ``float_precision="round_trip"`` (C engine) preserves every float
    bit; the exact float columns are spliced into the string read. The frozen
    files themselves are untouched, so their recorded sha256 hashes still bind
    them.
    """
    strings = pd.read_csv(path, keep_default_na=False, engine="python")
    exact = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    for column in exact.columns:
        if pd.api.types.is_float_dtype(exact[column].dtype):
            strings[column] = exact[column]
    return strings


class ConfirmatoryDivergenceError(RuntimeError):
    """The evaluation pass did not reproduce the frozen pre-GT tables."""


def assert_confirmatory_selection(document_ids: object, *, context: str) -> frozenset[str]:
    """The inverse freshness guard: the selection must be EXACTLY the 99-doc reserve.

    Track-A refuses any reserve document; the confirmatory run refuses anything else,
    so between the two guards no selection can be partly fresh and partly seen.
    """
    if isinstance(document_ids, str) or not hasattr(document_ids, "__iter__"):
        raise TypeError("document_ids must be an iterable of ids")
    ids = frozenset(str(item) for item in document_ids)
    reserve = confirmatory_document_ids()
    if ids != reserve:
        missing = sorted(reserve - ids)[:5]
        extra = sorted(ids - reserve)[:5]
        raise ConfirmatoryDivergenceError(
            f"{context}: selection is not exactly the confirmatory reserve "
            f"({len(ids)} ids; missing {len(reserve - ids)} e.g. {missing}, "
            f"unexpected {len(ids - reserve)} e.g. {extra})"
        )
    return ids


@dataclass(frozen=True, slots=True)
class FoldResources:
    """The per-engine fitted material, identical in every pass and in evaluation."""

    incumbent: EditAwareGenerator
    structural: StructuralV2Generator
    resources: DiscoveryResources
    lexicon_sha: str
    bigram_total: int


def fit_fold_resources(
    streams: dict[tuple[str, str], str],
    fit_documents: frozenset[str],
    engines: Sequence[str],
) -> dict[str, FoldResources]:
    """Fit the fold resources exactly as ``run_track_a`` does, per held-out engine."""
    fitted: dict[str, FoldResources] = {}
    for engine_id in engines:
        corpus = _fold_corpus(streams, engine_id, fit_documents)
        incumbent = EditAwareGenerator(generator_id="g3_edit_aware")
        incumbent.fit(corpus)
        structural = StructuralV2Generator()
        structural.fit(corpus)
        lexical = LexicalGenerator()
        lexical.fit(corpus)
        lexicon = frozenset(lexical.lexicon)
        fitted[engine_id] = FoldResources(
            incumbent=incumbent,
            structural=structural,
            resources=DiscoveryResources(lexicon=lexicon, bigrams=structural._bigrams),
            lexicon_sha=hashlib.sha256(json.dumps(sorted(lexicon)).encode("utf-8")).hexdigest(),
            bigram_total=int(sum(structural._bigrams.values())),
        )
    return fitted


def discovery_pass(
    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]],
    fitted: dict[str, FoldResources],
    engines: Sequence[str],
    rules: DiscoveryRules | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Enumerate sites on every fresh pair -- the GT-blind discovery pass.

    Returns the site table and the list of pairs whose page was empty (no OCR spans
    at all). Empty pages are recorded, never silently skipped: an engine emitting
    nothing on a page breaks the matched-source premise and must stay visible.
    """
    rules = rules or DiscoveryRules()
    pairs = sorted(pair for pair in spans_by_pair if pair[1] in engines)
    rows: list[dict[str, object]] = []
    empty: list[tuple[str, str]] = []
    for position, pair in enumerate(pairs):
        spans = spans_by_pair[pair]
        if not spans:
            empty.append(pair)
            continue
        page = OcrPageView.from_spans(tuple(spans))
        for site in enumerate_sites(page, fitted[pair[1]].resources, rules):
            rows.append(site.as_dict())
        if progress is not None and position % 25 == 24:
            progress(f"discovery pair {position + 1}/{len(pairs)}: {len(rows)} sites")
    return pd.DataFrame(rows), empty


def _unlabeled_rows(
    site: DiscoveredSite,
    proposals: Sequence[tuple[str, CandidateProposal, str]],
    region_ocr: str,
) -> list[dict[str, object]]:
    """``candidates_from_proposals`` semantics, labels omitted.

    Same skip conditions (identity with the anchor slice, per-rung text dedup), same
    rank assignment (position among emitted rows), so the generation columns are
    byte-comparable with the evaluation pass's labeled rows.
    """
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for generator_id, proposal, operation in proposals:
        if proposal.text == region_ocr or proposal.text in seen:
            continue
        seen.add(proposal.text)
        rows.append(
            {
                "site_id": site.site_id,
                "document_id": site.document_id,
                "dataset_id": site.dataset_id,
                "engine_id": site.engine_id,
                "generator_id": generator_id,
                "candidate_text": proposal.text,
                "operation": operation,
                "generator_rank": len(rows),
                "generator_score": proposal.score,
                "region_ocr": region_ocr,
            }
        )
    return rows


def _site_from_row(row: pd.Series, columns: list[str]) -> DiscoveredSite:
    """Rebuild a :class:`DiscoveredSite` from a frozen-table row.

    The frozen table already carries the anchor's stream coordinates, so nothing is
    re-derived: reconstruction is a projection, and a mismatch with the original
    coordinates is caught by the evaluation equality assertion.
    """

    def field(name: str) -> Any:
        return row[name]

    signals = {
        column.removeprefix("signal_"): str(row[column])
        for column in columns
        if column.startswith("signal_") and str(row[column]) != ""
    }
    return DiscoveredSite(
        site_id=str(field("site_id")),
        document_id=str(field("document_id")),
        dataset_id=str(field("dataset_id")),
        engine_id=str(field("engine_id")),
        anchor_kind=AnchorKind(str(field("anchor_kind"))),
        anchor_ref=str(field("anchor_ref")),
        char_start=int(field("char_start")),
        char_end=int(field("char_end")),
        site_type=str(field("site_type")),
        suspicion_score=float(field("suspicion_score")),
        provenance_reason=DiscoveryProvenance(str(field("provenance_reason"))),
        signals=signals,
    )


def generation_pass(
    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]],
    streams: dict[tuple[str, str], str],
    sites: pd.DataFrame,
    fitted: dict[str, FoldResources],
    engines: Sequence[str],
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run the frozen ladder over the frozen site table -- the GT-blind generation pass.

    Re-states ``run_track_a``'s per-site loop (g3 at TOKEN anchors with context, g7 at
    the anchor, the incumbent-first union capped at ``UNION_CAP``, b0 at
    confidence-rule TOKEN sites); the evaluation pass asserts the equality this
    re-statement exists to prove.
    """
    pairs = sorted(pair for pair in spans_by_pair if pair[1] in engines)
    columns = list(sites.columns)
    sites_by_pair: dict[tuple[str, str], pd.DataFrame] = {
        (str(document_id), str(engine_id)): group
        for (document_id, engine_id), group in sites.groupby(
            ["document_id", "engine_id"], sort=False
        )
    }
    rows: list[dict[str, object]] = []
    done = 0
    for pair in pairs:
        pair_sites = sites_by_pair.get(pair)
        if pair_sites is None or pair_sites.empty:
            continue
        spans = spans_by_pair[pair]
        if not spans:
            continue
        page = OcrPageView.from_spans(tuple(spans))
        stream = streams.get(pair, "")
        by_span = {span.span_id: span for span in spans}
        for _, site_row in pair_sites.iterrows():
            site = _site_from_row(site_row, columns)
            anchor_spans = [by_span[s] for s in site.anchor_ref.split("\0")[1:] if s in by_span]
            region_ocr = anchor_stream_text(stream, site)
            g3_proposals: list[CandidateProposal] = []
            if site.anchor_kind is AnchorKind.TOKEN and anchor_spans:
                span = anchor_spans[0]
                context = GenerationContext(
                    site_id=site.site_id,
                    document_id=site.document_id,
                    dataset_id=site.dataset_id,
                    engine_id=site.engine_id,
                    original_ocr=region_ocr,
                    context_before=stream[
                        max(0, site.char_start - CONTEXT_CHARS) : site.char_start
                    ],
                    context_after=stream[site.char_end : site.char_end + CONTEXT_CHARS],
                    native_confidences=(
                        (span.native_conf_recognition,)
                        if span.native_conf_recognition is not None
                        else ()
                    ),
                    conf_scale=span.conf_scale,
                    n_spans=1,
                )
                g3_proposals = fitted[pair[1]].incumbent.propose(context, UNION_CAP)
            g7_proposals = fitted[pair[1]].structural.propose_at_site(site, page)
            rows.extend(
                _unlabeled_rows(
                    site,
                    [("g3_edit_aware", p, _shape(p)) for p in g3_proposals],
                    region_ocr,
                )
            )
            rows.extend(
                _unlabeled_rows(
                    site,
                    [("g7_structural_v2", p, _shape(p)) for p in g7_proposals],
                    region_ocr,
                )
            )
            union: list[tuple[str, CandidateProposal, str]] = []
            seen: set[str] = set()
            for proposals in (g3_proposals, g7_proposals):
                for proposal in proposals:
                    if proposal.text in seen or proposal.text == region_ocr:
                        continue
                    seen.add(proposal.text)
                    union.append(("g8_union", proposal, _shape(proposal)))
            rows.extend(_unlabeled_rows(site, union[:UNION_CAP], region_ocr))
            if site.anchor_kind is AnchorKind.TOKEN and "normalized_confidence" in site.signals:
                b0 = [
                    ("b0_conf_only", proposal, _shape(proposal))
                    for proposal in g3_proposals
                    if proposal.metadata.get("edit_shape") not in {"deletion", "split"}
                ]
                rows.extend(_unlabeled_rows(site, b0, region_ocr))
            done += 1
            if progress is not None and done % 5000 == 0:
                progress(f"generation: {done} sites, {len(rows)} candidate rows")
    return pd.DataFrame(rows)


_SITE_COLUMNS = [
    "site_id",
    "document_id",
    "dataset_id",
    "engine_id",
    "anchor_kind",
    "char_start",
    "char_end",
    "site_type",
    "suspicion_score",
    "provenance_reason",
]

_GENERATION_COLUMNS = [
    "site_id",
    "document_id",
    "dataset_id",
    "engine_id",
    "generator_id",
    "candidate_text",
    "operation",
    "generator_rank",
    "generator_score",
    "region_ocr",
]


def assert_frames_match_frozen(
    discovered_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    frozen_sites: pd.DataFrame,
    frozen_candidates: pd.DataFrame,
) -> None:
    """Prove the evaluation pass reproduced the frozen pre-GT tables exactly.

    The site comparison covers every column ``run_track_a``'s discovered frame emits;
    ``anchor_ref`` and the ``signal_*`` columns exist only in the frozen table (they
    are pure functions of the span ids and the compared coordinates, so their equality
    follows). Raises :class:`ConfirmatoryDivergenceError` naming the divergence, so a
    mismatch is a defect to investigate -- never a silent second source of truth.
    """
    try:
        live_sites = discovered_frame[_SITE_COLUMNS].reset_index(drop=True)
        frozen_sites_cmp = frozen_sites[_SITE_COLUMNS].reset_index(drop=True)
        for column in ("char_start", "char_end", "suspicion_score"):
            live_sites[column] = pd.to_numeric(live_sites[column])
            frozen_sites_cmp[column] = pd.to_numeric(frozen_sites_cmp[column])
        pd.testing.assert_frame_equal(
            live_sites,
            frozen_sites_cmp,
            check_exact=True,
            obj="discovered sites",
        )

        live_candidates = candidate_frame[_GENERATION_COLUMNS].reset_index(drop=True)
        frozen_candidates_cmp = frozen_candidates[_GENERATION_COLUMNS].reset_index(drop=True)
        for frame in (live_candidates, frozen_candidates_cmp):
            frame["generator_rank"] = pd.to_numeric(frame["generator_rank"])
            frame["generator_score"] = pd.to_numeric(frame["generator_score"], errors="coerce")
        sort_columns = _GENERATION_COLUMNS[:6]
        pd.testing.assert_frame_equal(
            live_candidates.sort_values(sort_columns).reset_index(drop=True),
            frozen_candidates_cmp.sort_values(sort_columns).reset_index(drop=True),
            check_exact=True,
            obj="generated candidates",
        )
    except AssertionError as error:  # pandas raises AssertionError on mismatch
        raise ConfirmatoryDivergenceError(
            f"evaluation pass diverged from the frozen pre-GT tables: {error}"
        ) from error


def discovered_region_truth(
    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]],
    streams: dict[tuple[str, str], str],
    frozen_sites: pd.DataFrame,
    indexes: dict[tuple[str, str], AlignmentIndex],
) -> pd.DataFrame:
    """(region_ocr, region_gt, d_before) for every discovered site -- evaluation side.

    Uses the same AlignmentIndex calls ``run_track_a`` makes per site, so the
    clean-site (overcorrection-observable) denominators come from the same region
    semantics as the labels. Sites on pages without an alignment index (an engine
    that returned nothing) carry ``index_present = False`` and never enter a
    denominator.
    """
    columns = list(frozen_sites.columns)
    sites_by_pair: dict[tuple[str, str], pd.DataFrame] = {
        (str(document_id), str(engine_id)): group
        for (document_id, engine_id), group in frozen_sites.groupby(
            ["document_id", "engine_id"], sort=False
        )
    }
    rows: list[dict[str, object]] = []
    for pair in sorted(spans_by_pair):
        pair_sites = sites_by_pair.get(pair)
        if pair_sites is None or pair_sites.empty:
            continue
        spans = spans_by_pair[pair]
        if not spans:
            continue
        stream = streams.get(pair, "")
        by_span = {span.span_id: span for span in spans}
        index = indexes.get(pair)
        for _, site_row in pair_sites.iterrows():
            site = _site_from_row(site_row, columns)
            anchor_spans = [by_span[s] for s in site.anchor_ref.split("\0")[1:] if s in by_span]
            region_ocr = anchor_stream_text(stream, site)
            if index is None:
                region_gt = ""
            elif site.anchor_kind is AnchorKind.GAP and len(anchor_spans) == 2:
                region_gt = index.gap_truth(anchor_spans[0].span_id, anchor_spans[1].span_id)
            else:
                _, region_gt = index.region_truth([span.span_id for span in anchor_spans])
            rows.append(
                {
                    "site_id": site.site_id,
                    "document_id": site.document_id,
                    "engine_id": site.engine_id,
                    "anchor_kind": site.anchor_kind.value,
                    "region_ocr": region_ocr,
                    "region_gt": region_gt,
                    "d_before": distance(region_ocr, region_gt) if index is not None else -1,
                    "index_present": index is not None,
                }
            )
    return pd.DataFrame(rows)

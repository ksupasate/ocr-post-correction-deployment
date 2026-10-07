"""The CGV3 Track-A run: enumerate, label offline, generate, decompose, measure.

Everything here is development-only (``analysis_role = exploratory``). The discovery
pass consumes page views; the GT-bearing AlignmentIndex is built and consulted only
afterwards, for true-region flanks, region ground truth, and labels -- the stage
separation the protocol freezes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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
from ocr_risk.discovery.views import OcrPageView
from ocr_risk.experiments.cgv2_gates import (
    _H2_ACCEPTED_EDITS_FLOOR,
    _H2_SAFE_COVERAGE_FLOOR,
)
from ocr_risk.experiments.cgv3_study import (
    AlignmentIndex,
    CandidateRow,
    anchor_stream_text,
    candidates_from_proposals,
    coverage_table,
    true_region_rows,
)
from ocr_risk.schemas.enums import AnchorKind
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["TrackAFrames", "run_track_a"]

K_GRID = (1, 2, 4, 8)
UNION_CAP = 4
# Outcome enum values are lowercase; every metric below compares against these sets.
_BENEFICIAL = ("true_correction", "partial_improvement")
_HARMFUL = ("miscorrection", "overcorrection")
_RUNGS = ("b0_conf_only", "g3_edit_aware", "g7_structural_v2", "g8_union")


@dataclass(slots=True)
class TrackAFrames:
    discovered: pd.DataFrame = field(default_factory=pd.DataFrame)
    true_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    site_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    rule_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    conditional: pd.DataFrame = field(default_factory=pd.DataFrame)
    end_to_end: pd.DataFrame = field(default_factory=pd.DataFrame)
    harm: pd.DataFrame = field(default_factory=pd.DataFrame)
    k_grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    failure: pd.DataFrame = field(default_factory=pd.DataFrame)
    oracle: pd.DataFrame = field(default_factory=pd.DataFrame)
    page_counts: pd.DataFrame = field(default_factory=pd.DataFrame)


def _fold_corpus(
    streams: dict[tuple[str, str], str], held_out: str, fit_documents: frozenset[str]
) -> list[str]:
    """(E \\ {held_out}) x D_fit OCR stream texts -- the only material resources see."""
    return [
        streams[pair]
        for pair in sorted(streams)
        if pair[1] != held_out and pair[0] in fit_documents
    ]


def run_track_a(
    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]],
    streams: dict[tuple[str, str], str],
    alignments: pd.DataFrame,
    gt_tokens: pd.DataFrame,
    sites: pd.DataFrame,
    document_ids: frozenset[str],
    fit_documents: frozenset[str],
    engines: Sequence[str],
    rules: DiscoveryRules | None = None,
    audit_states: pd.DataFrame | None = None,
    context_chars: int = 40,
    progress: Callable[[str], None] | None = None,
) -> TrackAFrames:
    """One Track-A pass over the selected seen documents."""
    rules = rules or DiscoveryRules()
    indexes = AlignmentIndex.from_frames(alignments, gt_tokens)
    track_pairs = [
        pair for pair in sorted(spans_by_pair) if pair[0] in document_ids and pair[1] in engines
    ]
    sites_limited = sites[
        sites["document_id"].astype(str).isin(document_ids)
        & sites["engine_id"].astype(str).isin(set(engines))
    ]
    true_regions = true_region_rows(sites_limited, indexes)

    incumbent: dict[str, EditAwareGenerator] = {}
    structural: dict[str, StructuralV2Generator] = {}
    resources: dict[str, DiscoveryResources] = {}
    for engine_id in engines:
        corpus = _fold_corpus(streams, engine_id, fit_documents)
        generator = EditAwareGenerator(generator_id="g3_edit_aware")
        generator.fit(corpus)
        incumbent[engine_id] = generator
        rung = StructuralV2Generator()
        rung.fit(corpus)
        structural[engine_id] = rung
        lexical = LexicalGenerator()
        lexical.fit(corpus)
        resources[engine_id] = DiscoveryResources(
            lexicon=frozenset(lexical.lexicon), bigrams=rung._bigrams
        )

    all_sites: list[DiscoveredSite] = []
    page_token_counts: list[dict[str, object]] = []
    candidate_rows: list[CandidateRow] = []
    for position, pair in enumerate(track_pairs):
        page = OcrPageView.from_spans(tuple(spans_by_pair[pair]))
        stream = streams.get(pair, "")
        discovered = enumerate_sites(page, resources[pair[1]], rules)
        all_sites.extend(discovered)
        page_token_counts.append(
            {
                "document_id": pair[0],
                "engine_id": pair[1],
                "n_tokens": len(page.tokens),
                "n_discovered": len(discovered),
                "stream_chars": len(stream),
            }
        )
        index = indexes.get(pair)
        if index is None:
            continue
        by_span = {span.span_id: span for span in spans_by_pair[pair]}
        for site in discovered:
            anchor_spans = [by_span[s] for s in site.anchor_ref.split("\0")[1:] if s in by_span]
            region_ocr = anchor_stream_text(stream, site)
            if site.anchor_kind is AnchorKind.GAP and len(anchor_spans) == 2:
                region_gt = index.gap_truth(anchor_spans[0].span_id, anchor_spans[1].span_id)
            else:
                _, region_gt = index.region_truth([span.span_id for span in anchor_spans])
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
                        max(0, site.char_start - context_chars) : site.char_start
                    ],
                    context_after=stream[site.char_end : site.char_end + context_chars],
                    native_confidences=(
                        (span.native_conf_recognition,)
                        if span.native_conf_recognition is not None
                        else ()
                    ),
                    conf_scale=span.conf_scale,
                    n_spans=1,
                )
                g3_proposals = incumbent[site.engine_id].propose(context, UNION_CAP)
            g7_proposals = structural[site.engine_id].propose_at_site(site, page)
            candidate_rows.extend(
                candidates_from_proposals(
                    site,
                    page,
                    region_ocr,
                    region_gt,
                    [("g3_edit_aware", p, _shape(p)) for p in g3_proposals],
                )
            )
            candidate_rows.extend(
                candidates_from_proposals(
                    site,
                    page,
                    region_ocr,
                    region_gt,
                    [("g7_structural_v2", p, _shape(p)) for p in g7_proposals],
                )
            )
            # g8_union: the incumbent first, then the structural rung, deduplicated by
            # text and capped -- its own rung, with member shapes retained per row.
            union: list[tuple[str, CandidateProposal, str]] = []
            seen: set[str] = set()
            for proposals in (g3_proposals, g7_proposals):
                for proposal in proposals:
                    if proposal.text in seen or proposal.text == region_ocr:
                        continue
                    seen.add(proposal.text)
                    union.append(("g8_union", proposal, _shape(proposal)))
            candidate_rows.extend(
                candidates_from_proposals(site, page, region_ocr, region_gt, union[:UNION_CAP])
            )
            # b0 isolates the confidence signal: the incumbent's lexical proposals,
            # gated to the sites the confidence rule discovered.
            if site.anchor_kind is AnchorKind.TOKEN and "normalized_confidence" in site.signals:
                b0 = [
                    ("b0_conf_only", proposal, _shape(proposal))
                    for proposal in g3_proposals
                    if proposal.metadata.get("edit_shape") not in {"deletion", "split"}
                ]
                candidate_rows.extend(
                    candidates_from_proposals(site, page, region_ocr, region_gt, b0)
                )
        if progress is not None and position % 25 == 0:
            progress(f"track-a pair {position + 1}/{len(track_pairs)}: {len(all_sites)} sites")

    true_coverage, discovered_frame = coverage_table(all_sites, true_regions, audit_states)
    frames = TrackAFrames(
        discovered=discovered_frame,
        true_coverage=true_coverage,
        candidates=pd.DataFrame([row.as_dict() for row in candidate_rows]),
        page_counts=pd.DataFrame(page_token_counts),
    )
    frames.site_metrics = _site_metrics(frames)
    frames.rule_metrics = _rule_metrics(frames)
    frames.conditional, frames.end_to_end = _conditional_metrics(frames)
    frames.harm = _harm_table(frames)
    frames.k_grid = _k_grid_table(frames)
    frames.failure = _failure_table(frames)
    frames.oracle = _oracle_table(frames)
    return frames


def _shape(proposal: CandidateProposal) -> str:
    return str(proposal.metadata.get("edit_shape", "substitution"))


# --------------------------------------------------------------------------- metrics


def _site_metrics(frames: TrackAFrames) -> pd.DataFrame:
    coverage = frames.true_coverage
    pages = frames.page_counts
    rows: list[dict[str, object]] = []
    for engine_id, group in coverage.groupby("engine_id", sort=True):
        eligible = group[group["audit_state"] == "eligible"]
        discovered = int((eligible["disposition"] == "discovered").sum())
        page_group = pages[pages["engine_id"] == engine_id]
        tokens = int(page_group["n_tokens"].sum()) if not page_group.empty else 0
        site_group = frames.discovered[frames.discovered["engine_id"] == engine_id]
        rows.append(
            {
                "engine_id": engine_id,
                "n_true_eligible": len(eligible),
                "n_discovered": discovered,
                "site_recall": discovered / len(eligible) if len(eligible) else float("nan"),
                "n_boundary_mismatch": int((eligible["disposition"] == "boundary_mismatch").sum()),
                "n_not_discovered": int((eligible["disposition"] == "not_discovered").sum()),
                "n_excluded_ambiguous": int((group["disposition"] == "excluded_ambiguous").sum()),
                "n_excluded_unresolved": int((group["disposition"] == "excluded_unresolved").sum()),
                "n_discovered_sites": len(site_group),
                "sites_per_document": len(site_group) / page_group["document_id"].nunique()
                if not page_group.empty
                else float("nan"),
                "sites_per_1k_tokens": len(site_group) / (tokens / 1000)
                if tokens
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _rule_metrics(frames: TrackAFrames) -> pd.DataFrame:
    sites = frames.discovered
    if sites.empty:
        return pd.DataFrame(columns=["engine_id", "provenance_reason", "n_sites", "precision"])
    rows = [
        {
            "engine_id": engine_id,
            "provenance_reason": reason,
            "n_sites": len(group),
            "precision": float(group["covers_true_region"].mean()),
        }
        for (engine_id, reason), group in sites.groupby(
            ["engine_id", "provenance_reason"], sort=True
        )
    ]
    return pd.DataFrame(rows)


def _conditional_metrics(frames: TrackAFrames) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P(exact | covered) and P(beneficial | covered) per rung; end-to-end per engine.

    Amendment A5: both conditional quantities count REGIONS whose matched site holds the
    candidate, with covered REGIONS in the denominator. A shared anchor (one gap anchor
    covering consecutive missing GT tokens) is one site but several regions; mixing a
    site-unit numerator with the region-unit denominator produced a quantity that was
    neither and made this table disagree with the end-to-end table about the same
    numerator. Region units on both sides also restore the H4 factorization exactly:
    site_recall x conditional = end-to-end.
    """
    matched = frames.true_coverage[frames.true_coverage["disposition"] == "discovered"]
    candidates = frames.candidates
    conditional: list[dict[str, object]] = []
    end_to_end: list[dict[str, object]] = []
    for rung in _RUNGS:
        rung_rows = candidates[candidates["generator_id"] == rung]
        beneficial_site_ids = set(
            rung_rows[rung_rows["outcome"].isin(_BENEFICIAL)]["site_id"].astype(str)
        )
        exact_site_ids = set(
            rung_rows[rung_rows["outcome"] == "true_correction"]["site_id"].astype(str)
        )
        for engine_id, group in matched.groupby("engine_id", sort=True):
            n_covered = len(group)
            exact_regions = int(group["matched_site_id"].astype(str).isin(exact_site_ids).sum())
            beneficial_regions = int(
                group["matched_site_id"].astype(str).isin(beneficial_site_ids).sum()
            )
            conditional.append(
                {
                    "engine_id": engine_id,
                    "generator_id": rung,
                    "n_covered_regions": n_covered,
                    "n_regions_with_exact": exact_regions,
                    "n_regions_with_beneficial": beneficial_regions,
                    "exact_recall_given_discovery": exact_regions / n_covered
                    if n_covered
                    else float("nan"),
                    "beneficial_recall_given_discovery": beneficial_regions / n_covered
                    if n_covered
                    else float("nan"),
                }
            )
        for engine_id, group in matched.groupby("engine_id", sort=True):
            eligible = frames.true_coverage[
                (frames.true_coverage["engine_id"] == engine_id)
                & (frames.true_coverage["audit_state"] == "eligible")
            ]
            hit = group["matched_site_id"].astype(str).isin(beneficial_site_ids)
            end_to_end.append(
                {
                    "engine_id": engine_id,
                    "generator_id": rung,
                    "n_eligible_regions": len(eligible),
                    "n_discovered": len(group),
                    "n_end_to_end_beneficial": int(hit.sum()),
                    "end_to_end_beneficial_opportunity": float(hit.sum()) / len(eligible)
                    if len(eligible)
                    else float("nan"),
                }
            )
    return pd.DataFrame(conditional), pd.DataFrame(end_to_end)


def _harm_table(frames: TrackAFrames) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (rung, engine_id), group in frames.candidates.groupby(
        ["generator_id", "engine_id"], sort=True
    ):
        beneficial = int(group["outcome"].isin(_BENEFICIAL).sum())
        harmful = int(group["outcome"].isin(_HARMFUL).sum())
        rows.append(
            {
                "engine_id": engine_id,
                "generator_id": rung,
                "n_candidates": len(group),
                "n_beneficial": beneficial,
                "n_harmful": harmful,
                "n_neutral": len(group) - beneficial - harmful,
                "harmful_beneficial_ratio": harmful / beneficial if beneficial else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def _k_grid_table(frames: TrackAFrames) -> pd.DataFrame:
    matched = frames.true_coverage[frames.true_coverage["disposition"] == "discovered"]
    site_to_engine = {
        str(row.matched_site_id): str(row.engine_id) for row in matched.itertuples(index=False)
    }
    union = frames.candidates[frames.candidates["generator_id"] == "g8_union"]
    rows: list[dict[str, object]] = []
    for k in K_GRID:
        within = union[union["generator_rank"] < k]
        for engine_id, group in within.groupby("engine_id", sort=True):
            engine_sites = {site for site, engine in site_to_engine.items() if engine == engine_id}
            covered_group = group[group["site_id"].astype(str).isin(engine_sites)]
            rows.append(
                {
                    "k": k,
                    "engine_id": engine_id,
                    "n_candidates": len(group),
                    "candidates_per_covered_site": len(group) / max(len(engine_sites), 1),
                    "harmful_burden": float(group["outcome"].isin(_HARMFUL).mean())
                    if len(group)
                    else float("nan"),
                    "beneficial_fraction": float(covered_group["outcome"].isin(_BENEFICIAL).mean())
                    if len(covered_group)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _failure_table(frames: TrackAFrames) -> pd.DataFrame:
    """First failed stage per eligible true region without a beneficial g8 candidate."""
    candidates = frames.candidates
    beneficial_sites = set(
        candidates[
            (candidates["generator_id"] == "g8_union") & candidates["outcome"].isin(_BENEFICIAL)
        ]["site_id"].astype(str)
    )
    rows: list[dict[str, object]] = []
    for region in frames.true_coverage.itertuples(index=False):
        if region.audit_state != "eligible":
            stage, failure_class = "evaluation", "alignment_" + str(region.audit_state)
        elif region.disposition == "discovered":
            matched = str(region.matched_site_id)
            if matched in beneficial_sites:
                continue
            rung_rows = candidates[
                (candidates["generator_id"] == "g8_union")
                & (candidates["site_id"].astype(str) == matched)
            ]
            stage = "generation"
            failure_class = (
                "expressible_not_generated" if rung_rows.empty else "no_beneficial_candidate"
            )
        elif region.disposition == "boundary_mismatch":
            stage, failure_class = "discovery", "wrong_boundary"
        else:
            stage, failure_class = "discovery", "not_discovered"
        rows.append(
            {
                "true_site_id": str(region.true_site_id),
                "engine_id": str(region.engine_id),
                "site_kind": str(region.site_kind),
                "stage": stage,
                "failure_class": failure_class,
            }
        )
    return pd.DataFrame(rows)


def _oracle_table(frames: TrackAFrames) -> pd.DataFrame:
    """Exploratory H2-identifiability diagnostic on the Track-A pool."""
    union = frames.candidates[frames.candidates["generator_id"] == "g8_union"]
    beneficial_sites = set(union[union["outcome"].isin(_BENEFICIAL)]["site_id"].astype(str))
    rows: list[dict[str, object]] = []
    for engine_id, group in frames.true_coverage.groupby("engine_id", sort=True):
        eligible = group[group["audit_state"] == "eligible"]
        discovered = eligible[eligible["disposition"] == "discovered"]
        n_beneficial = int(discovered["matched_site_id"].astype(str).isin(beneficial_sites).sum())
        rows.append(
            {
                "engine_id": engine_id,
                "n_eligible_regions": len(eligible),
                "oracle_accepted_edits": n_beneficial,
                "oracle_safe_coverage": n_beneficial / len(eligible)
                if len(eligible)
                else float("nan"),
                # The frozen CGV2 H2 floors, imported from their single definition so the
                # two gates can never drift apart (reviewer finding, single-source rule).
                "h2_floor_edits": _H2_ACCEPTED_EDITS_FLOOR,
                "h2_floor_coverage": _H2_SAFE_COVERAGE_FLOOR,
                "analysis_role": "exploratory",
            }
        )
    return pd.DataFrame(rows)

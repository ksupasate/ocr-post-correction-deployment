"""The CGV2 structural candidate-generation study (``docs/cgv2/protocol.md``).

Three things the recovery-phase study did not do, all frozen in the CGV2 protocol before
any ``evaluate``-role run:

1. **Regions** — candidates that rewrite two adjacent sites jointly, labeled with the
   same string-pure outcome functions as site candidates, stored beside (never inside)
   the frozen candidate tables.
2. **Fold certificates** — per held-out engine, per rung: the actually-fitted lexicon
   contains none of the vocabulary only that engine contributed to the all-engines fit
   corpus. Asserted on the fitted objects, not on the construction's intentions.
3. **A site census** — every evaluable scored site with its kind and ``d_before``, so
   every downstream table has its denominator from the same canonical artifact instead
   of reconstructing it from proposals.

The site-level pass reuses :func:`run_generator_study` unchanged at a raised generation
cap (protocol §12); the K-grid tables truncate by ``generator_rank`` afterwards.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter

from ocr_risk.candidates.base import CandidateGenerator
from ocr_risk.candidates.lexical import LexicalGenerator
from ocr_risk.candidates.pipeline import SiteContext
from ocr_risk.candidates.registry import build_generator
from ocr_risk.candidates.structural import StructuralGenerator
from ocr_risk.edits.outcome import classify_accepted, distance
from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER
from ocr_risk.experiments.generator_study import (
    GeneratorRun,
    GeneratorSpec,
    ProposalRecord,
    run_generator_study,
)
from ocr_risk.io.hashing import canonical_hash, stable_string_set_hash
from ocr_risk.metrics.generator import SiteProposals
from ocr_risk.schemas.enums import CandidatePool, HarmPolicy, OutcomeIfAccepted, SplitRole
from ocr_risk.splits.document_partition import DocumentPartition

__all__ = [
    "CGV2_RUNGS",
    "GENERATION_CAP",
    "HEADLINE_EXCLUDED_DATASETS",
    "K_GRID",
    "PROTOCOL_AMENDMENT_COMMIT",
    "PROTOCOL_FREEZE_COMMIT",
    "STRUCTURAL_SITE_KINDS",
    "Cgv2Run",
    "RegionRecord",
    "SiteCensusRecord",
    "build_site_census",
    "engine_contrast",
    "region_records_from_csv",
    "run_cgv2_study",
]

PROTOCOL_FREEZE_COMMIT = "d0fdf8d"
"""The commit that froze ``docs/cgv2/protocol.md``; evaluate-role runs cite it."""

PROTOCOL_AMENDMENT_COMMIT = "e61b412"
"""Latest numbered pre-evaluation amendment (A5); no evaluate-role output preceded it."""

HEADLINE_EXCLUDED_DATASETS = frozenset({"ocrd_sbb"})
"""Historical/model-availability stress tracks excluded from headline aggregates.

The records remain in the canonical study artifacts and are reported by dataset. This
constant implements protocol section 4 and ``docs/benchmark_design.md``; it is not a
post-hoc result filter.
"""

CGV2_RUNGS: tuple[str, ...] = (
    "g0_lexical",
    "g1_error_gated",
    "g2_byt5",
    "g2_byt5_ctx0",
    "g3_edit_aware",
    "g4_union",
    "g5_structural",
    "g6_union",
)
GENERATION_CAP = 8
K_GRID: tuple[int, ...] = (1, 2, 4, 8)
STRUCTURAL_SITE_KINDS = frozenset({"deletion", "insertion", "segmentation"})
REGION_CANDIDATE_CAP = 2
"""The region generator's frozen per-region budget (StructuralGenerator.max_region_candidates)."""


@dataclass(frozen=True, slots=True)
class SiteCensusRecord:
    """One evaluable scored site: the denominator row for every CGV2 table."""

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    site_kind: str
    d_before: int
    gt_text: str
    char_start: int
    char_end: int
    n_spans: int
    ocr_text: str
    min_align_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "engine_id": self.engine_id,
            "site_kind": self.site_kind,
            "d_before": self.d_before,
            "gt_text": self.gt_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "n_spans": self.n_spans,
            "ocr_text": self.ocr_text,
            "min_align_confidence": self.min_align_confidence,
        }


@dataclass(frozen=True, slots=True)
class RegionRecord:
    """One region proposal, labeled. Ground truth is present for offline analysis only."""

    generator_id: str
    region_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    role: str
    left_site_id: str
    right_site_id: str
    gap_chars: int
    region_char_start: int
    region_char_end: int
    left_kind: str
    right_kind: str
    region_ocr: str
    region_gt: str
    candidate_text: str
    d_before: int
    d_after: int
    delta: int
    outcome: OutcomeIfAccepted
    generator_rank: int
    generator_meta: dict[str, str] = field(default_factory=dict)
    pool: str = "natural"
    generator_version: str = "unknown"
    generator_score: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "generator_id": self.generator_id,
            "region_id": self.region_id,
            "pool": self.pool,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "engine_id": self.engine_id,
            "role": self.role,
            "left_site_id": self.left_site_id,
            "right_site_id": self.right_site_id,
            "gap_chars": self.gap_chars,
            "region_char_start": self.region_char_start,
            "region_char_end": self.region_char_end,
            "left_kind": self.left_kind,
            "right_kind": self.right_kind,
            "region_ocr": self.region_ocr,
            "region_gt": self.region_gt,
            "candidate_text": self.candidate_text,
            "d_before": self.d_before,
            "d_after": self.d_after,
            "delta": self.delta,
            "outcome": self.outcome.value,
            "generator_rank": self.generator_rank,
            "generator_version": self.generator_version,
            "generator_score": self.generator_score,
            **{f"meta_{k}": v for k, v in self.generator_meta.items()},
        }


@dataclass(slots=True)
class Cgv2Run:
    """Everything one CGV2 study produced, site level and region level."""

    site_run: GeneratorRun
    census: list[SiteCensusRecord] = field(default_factory=list)
    region_records: list[RegionRecord] = field(default_factory=list)
    region_census: dict[str, int] = field(default_factory=dict)
    certificates: dict[str, dict[str, object]] = field(default_factory=dict)
    fold_tokens: dict[str, list[str]] = field(default_factory=dict)
    role: str = ""
    k_grid: tuple[int, ...] = K_GRID
    runtime_seconds: dict[str, float] = field(default_factory=dict)


def build_site_census(
    contexts: Sequence[SiteContext],
    partition: DocumentPartition,
    role: SplitRole,
    scored_document_ids: frozenset[str] | None = None,
    scored_engines: frozenset[str] | None = None,
) -> list[SiteCensusRecord]:
    """Every evaluable site on the scored role's documents — proposed-at or not."""
    role_documents = partition.documents(role)
    scored = role_documents if scored_document_ids is None else scored_document_ids
    outside_role = scored - role_documents
    if outside_role:
        msg = (
            f"scored_document_ids contains {len(outside_role)} document(s) outside the "
            f"{role.value} role: {sorted(outside_role)[:5]}"
        )
        raise ValueError(msg)
    census: list[SiteCensusRecord] = []
    for context in contexts:
        site = context.site
        if (
            site.document_id not in scored
            or not site.evaluable
            or (scored_engines is not None and site.engine_id not in scored_engines)
        ):
            continue
        census.append(
            SiteCensusRecord(
                site_id=site.site_id,
                document_id=site.document_id,
                dataset_id=site.dataset_id,
                engine_id=site.engine_id,
                site_kind=site.site_kind.value,
                d_before=site.d_before,
                gt_text=site.gt_text,
                char_start=site.char_start,
                char_end=site.char_end,
                n_spans=len(site.ocr_span_ids),
                ocr_text=site.ocr_text,
                min_align_confidence=site.min_align_confidence,
            )
        )
    return census


@dataclass(frozen=True, slots=True)
class _Region:
    region_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    left: SiteCensusRecord
    right: SiteCensusRecord
    gap_chars: int
    region_char_start: int
    region_char_end: int
    region_ocr: str
    region_gt: str


def classify_site_pair(
    left: SiteCensusRecord, right: SiteCensusRecord, stream: str
) -> tuple[str, str, int]:
    """Bucket one adjacent pair against the exact stream slice between them.

    Returns ``(bucket, region_ocr, gap_chars)``. Eligibility is defined on the slice
    itself, never on offsets alone: the pair is eligible only when the slice is exactly
    the left site's text, pure same-line whitespace, and the right site's text. Anything
    else is named for what the slice actually contains -- a newline (a cross-line pair,
    which no word-merge semantics can join), an intervening excluded site's characters,
    or a text/range mismatch from non-contiguous alignment components.
    """
    if right.char_start < left.char_end:
        return "non_adjacent", "", 0
    if not left.ocr_text.strip() or not right.ocr_text.strip():
        return "degenerate", "", 0
    if len(stream) < right.char_end:
        return "no_stream", "", 0
    region_ocr = stream[left.char_start : right.char_end]
    if not region_ocr.startswith(left.ocr_text) or not region_ocr.endswith(right.ocr_text):
        return "text_mismatch", region_ocr, 0
    middle = region_ocr[len(left.ocr_text) : len(region_ocr) - len(right.ocr_text)]
    if not middle:
        return "non_adjacent", region_ocr, 0
    if middle.strip():
        return "intervening_text", region_ocr, 0
    if "\n" in middle or "\r" in middle:
        return "cross_line", region_ocr, len(middle)
    return "eligible", region_ocr, len(middle)


def build_regions(
    census: Sequence[SiteCensusRecord],
    streams: dict[tuple[str, str], str] | None = None,
) -> tuple[list[_Region], dict[str, int]]:
    """Adjacent eligible site pairs per (document, engine), in linearized-stream order.

    Requires the linearized streams (amendment A3): a region's OCR text is the exact
    stream slice ``stream[left.char_start : right.char_end]``, so a newline separator or
    a swallowed excluded site shows up in the slice instead of being silently rewritten
    as spaces. Pairs with an empty-OCR member are degenerate (amendment A1).
    """
    counts = {
        "eligible": 0,
        "cross_line": 0,
        "intervening_text": 0,
        "degenerate": 0,
        "text_mismatch": 0,
        "non_adjacent": 0,
        "no_stream": 0,
    }
    by_pair: dict[tuple[str, str], list[SiteCensusRecord]] = {}
    for record in census:
        by_pair.setdefault((record.document_id, record.engine_id), []).append(record)

    regions: list[_Region] = []
    for (document_id, engine_id), sites in by_pair.items():
        stream = (streams or {}).get((document_id, engine_id), "")
        sites.sort(key=lambda s: (s.char_start, s.site_id))
        for index, (left, right) in enumerate(pairwise(sites)):
            bucket, region_ocr, gap_chars = classify_site_pair(left, right, stream)
            counts[bucket] += 1
            if bucket != "eligible":
                continue
            regions.append(
                _Region(
                    region_id=f"{document_id}:{engine_id}:region:{index:05d}",
                    document_id=document_id,
                    dataset_id=left.dataset_id,
                    engine_id=engine_id,
                    left=left,
                    right=right,
                    gap_chars=gap_chars,
                    region_char_start=left.char_start,
                    region_char_end=right.char_end,
                    region_ocr=region_ocr,
                    region_gt=" ".join(t for t in (left.gt_text, right.gt_text) if t),
                )
            )
    return regions, counts


def _lexicon_tokens(generator: CandidateGenerator) -> set[str]:
    """Every token a generator actually fitted, whatever attribute it lives on.

    A rung with several fitted states must have all of them certified or the
    certificate asserts less than the generator uses: the structural rung's insertion
    vocabulary and bigram evidence are separate fitted states, and either could carry
    forbidden vocabulary even when the lexicon does not.
    """
    tokens: set[str] = set()
    lexicon = getattr(generator, "lexicon", None)
    if lexicon:
        tokens |= set(lexicon)
    for attribute in ("_vocabulary", "_insertion_vocab"):
        vocabulary = getattr(generator, attribute, None)
        if vocabulary:
            tokens |= set(vocabulary)
    bigrams = getattr(generator, "_bigrams", None)
    if bigrams:
        for left, right in bigrams:
            tokens.add(left)
            tokens.add(right)
    members = getattr(generator, "_members", None) or []
    for member in members:
        tokens |= _lexicon_tokens(member)
    return tokens


def fold_certificates(
    contexts: Sequence[SiteContext],
    partition: DocumentPartition,
    specs: Sequence[GeneratorSpec],
    engines: Sequence[str],
    lexicon_role: SplitRole = SplitRole.FIT,
    scored_role: SplitRole = SplitRole.CALIBRATE,
    scored_document_ids: frozenset[str] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    """Per held-out engine: certify every fitted token source on both split axes.

    The forbidden set is what the held-out engine uniquely contributed to the all-engines
    fit corpus. Each rung is fitted on the leave-this-engine-out corpus and every token in
    its *actual fitted state* (lexicon, insertion vocabulary, bigrams, and composite
    members) is intersected with that forbidden set. The certificate also hashes the
    exact source records and all three document roles, so target-engine isolation and
    held-out-document isolation are independently checkable without trusting this loop.

    The fold's lexical token list is returned beside the certificates so downstream
    tables can ask "was the ground truth in this fold's lexicon?" without refitting.
    """
    fit_documents = partition.documents(lexicon_role)
    calibrate_documents = partition.documents(SplitRole.CALIBRATE)
    evaluate_documents = partition.documents(SplitRole.EVALUATE)
    scored_role_documents = partition.documents(scored_role)
    scored_documents = scored_role_documents if scored_document_ids is None else scored_document_ids
    outside_role = scored_documents - scored_role_documents
    if outside_role:
        msg = (
            f"scored_document_ids contains {len(outside_role)} document(s) outside the "
            f"{scored_role.value} role: {sorted(outside_role)[:5]}"
        )
        raise ValueError(msg)
    all_engines = sorted({context.site.engine_id for context in contexts})
    corpus_all = [c.site.ocr_text for c in contexts if c.site.document_id in fit_documents]
    certificates: dict[str, dict[str, object]] = {}
    fold_tokens: dict[str, list[str]] = {}

    def _token_counts(corpus: list[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for text in corpus:
            counts.update(text.split())
        return counts

    # The forbidden set is FULL-token (length >= 1, frequency >= 2): the insertion
    # vocabulary is the full fold token distribution since A2, so certifying against a
    # min-length-3 lexicon rule would leave exactly the short tokens unauditable.
    tokens_all = {t for t, n in _token_counts(corpus_all).items() if n >= 2}

    for engine_id in engines:
        source_contexts = [
            context
            for context in contexts
            if context.site.document_id in fit_documents and context.site.engine_id != engine_id
        ]
        corpus_without = [context.site.ocr_text for context in source_contexts]
        lexicon_without = LexicalGenerator(max_edit_distance=2, min_lexicon_frequency=2)
        lexicon_without.fit(corpus_without)
        counts_without = _token_counts(corpus_without)
        # "Engine-only" means the allowed corpus contains no occurrence. A token seen
        # once in an allowed engine and once in the held-out engine crosses a frequency
        # threshold in the pooled corpus, but its presence in a fitted bigram state is
        # still sourced from allowed material and is not target-engine leakage.
        forbidden = {token for token in tokens_all if counts_without[token] == 0}

        source_records = sorted(
            [
                {
                    "document_id": context.site.document_id,
                    "engine_id": context.site.engine_id,
                    "site_id": context.site.site_id,
                    "ocr_text": context.site.ocr_text,
                }
                for context in source_contexts
            ],
            key=lambda record: (
                str(record["document_id"]),
                str(record["engine_id"]),
                str(record["site_id"]),
            ),
        )
        source_record_ids = [
            f"{record['document_id']}\0{record['engine_id']}\0{record['site_id']}"
            for record in source_records
        ]
        source_checks = {
            "target_engine_records_selected": sum(
                context.site.engine_id == engine_id for context in source_contexts
            ),
            "non_fit_document_records_selected": sum(
                context.site.document_id not in fit_documents for context in source_contexts
            ),
            "calibrate_document_records_selected": sum(
                context.site.document_id in calibrate_documents for context in source_contexts
            ),
            "evaluate_document_records_selected": sum(
                context.site.document_id in evaluate_documents for context in source_contexts
            ),
            "scored_document_records_selected": sum(
                context.site.document_id in scored_documents for context in source_contexts
            ),
        }
        sources_clean = not any(source_checks.values())

        rungs: dict[str, dict[str, object]] = {}
        for spec in specs:
            entry: dict[str, object] = {
                "generator_kind": spec.kind,
                "generator_config_sha256": canonical_hash(
                    {
                        "id": spec.id,
                        "kind": spec.kind,
                        "params": spec.params,
                        "max_candidates": spec.max_candidates,
                        "datasets": spec.datasets,
                    }
                ),
                "lexicon_size": None,
                "fitted_token_count": 0,
                "fitted_tokens_sha256": canonical_hash([]),
                "forbidden_intersection_count": 0,
                "forbidden_intersection_sha256": canonical_hash([]),
                # Retained for compatibility with the committed calibration artifact.
                "forbidden_in_lexicon": [],
            }
            generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
            entry["generator_version"] = generator.version
            if not generator.available():
                entry["unavailable"] = True
                entry["clean"] = False
                rungs[spec.id] = entry
                continue
            generator.fit(corpus_without)
            tokens = _lexicon_tokens(generator)
            leaked = sorted(tokens & forbidden)
            lexicon_size = getattr(generator, "lexicon_size", None)
            entry["lexicon_size"] = int(lexicon_size) if lexicon_size is not None else None
            entry["fitted_token_count"] = len(tokens)
            entry["fitted_tokens_sha256"] = canonical_hash(sorted(tokens))
            entry["forbidden_intersection_count"] = len(leaked)
            entry["forbidden_intersection_sha256"] = canonical_hash(leaked)
            entry["forbidden_in_lexicon"] = leaked
            entry["clean"] = not leaked
            rungs[spec.id] = entry
        fold_tokens[engine_id] = sorted(lexicon_without.lexicon)
        rungs_clean = all(bool(entry.get("clean")) for entry in rungs.values())
        certificates[engine_id] = {
            "schema_version": "cgv2-leakage-certificate-v1",
            "fold_id": f"loeo_zero_shot:held_out={engine_id}",
            "held_out_engine": engine_id,
            "training_engines": [engine for engine in all_engines if engine != engine_id],
            "scored_role": scored_role.value,
            "lexicon_role": lexicon_role.value,
            "partition_sha256": partition.partition_sha256,
            "fit_documents": len(fit_documents),
            "fit_documents_sha256": stable_string_set_hash(fit_documents),
            "calibrate_documents": len(calibrate_documents),
            "calibrate_documents_sha256": stable_string_set_hash(calibrate_documents),
            "evaluate_documents": len(evaluate_documents),
            "evaluate_documents_sha256": stable_string_set_hash(evaluate_documents),
            "scored_documents": len(scored_documents),
            "scored_documents_sha256": stable_string_set_hash(scored_documents),
            "lexicon_source_records": len(source_records),
            "lexicon_source_record_ids_sha256": stable_string_set_hash(source_record_ids),
            "lexicon_source_records_sha256": canonical_hash(source_records),
            "source_checks": source_checks,
            "sources_clean": sources_clean,
            "all_engines_lexicon": len(tokens_all),
            "engine_only_vocabulary": len(forbidden),
            "engine_only_vocabulary_sha256": canonical_hash(sorted(forbidden)),
            "rungs": rungs,
            "pass": sources_clean and rungs_clean,
        }
    return certificates, fold_tokens


def neighbor_contexts(
    contexts: Sequence[SiteContext],
    alignments_by_pair: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Real before/after text for empty-OCR sites, from their alignment neighbours.

    A span-less site pins to char range ``[0, 0]``, so the stream window around it is
    empty by construction and every context-based trigger at those sites sees nothing.
    The alignment layer still knows where the missing token sits: between its component's
    neighbours. ``alignments_by_pair`` maps (document, engine) to that document's
    components in alignment order as ``(alignment_id, ocr_text)``; the neighbours' OCR
    text is the context. Ground-truth-blind -- it is OCR text and ordering, nothing else.
    """
    if not alignments_by_pair:
        return {}
    contexts_by_alignment = {
        alignment_id: context for context in contexts for alignment_id in context.site.alignment_ids
    }
    neighbours: dict[str, tuple[str, str]] = {}
    for (_, _), components in alignments_by_pair.items():
        # The empty components keep their positions: the whole point is to give THEM
        # neighbours, so filtering them out first (an earlier version did) makes the
        # lookup miss exactly the sites it exists for.
        for position, (alignment_id, text) in enumerate(components):
            if text.strip() or alignment_id not in contexts_by_alignment:
                continue
            before = next((t for _a, t in reversed(components[:position]) if t.strip()), "")
            after = next((t for _a, t in components[position + 1 :] if t.strip()), "")
            if before and after:
                neighbours[alignment_id] = (before, after)
    return neighbours


def run_cgv2_study(
    contexts: Sequence[SiteContext],
    partition: DocumentPartition,
    policy: HarmPolicy,
    role: SplitRole = SplitRole.CALIBRATE,
    rungs: Sequence[str] = CGV2_RUNGS,
    streams: dict[tuple[str, str], str] | None = None,
    alignments_by_pair: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
    scored_document_ids: frozenset[str] | None = None,
    scored_engines: frozenset[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Cgv2Run:
    """Site-level ladder at the raised cap, then regions, census, and certificates."""
    study_started = perf_counter()
    specs = []
    for rung_id in rungs:
        if rung_id not in GENERATOR_LADDER:
            msg = f"unknown rung {rung_id!r}; known: {', '.join(sorted(GENERATOR_LADDER))}"
            raise ValueError(msg)
        specs.append(replace(GENERATOR_LADDER[rung_id], max_candidates=GENERATION_CAP))

    # Empty-OCR sites carry no stream window of their own (their char range is [0, 0]);
    # give them the text of their alignment neighbours so context-based triggers can
    # fire there at all. Every rung sees the same corrected context.
    neighbours = neighbor_contexts(contexts, alignments_by_pair)
    augmented: list[SiteContext] = []
    for context in contexts:
        if not context.site.ocr_text.strip() and context.site.alignment_ids:
            pair = next(
                (neighbours[a] for a in context.site.alignment_ids if a in neighbours), None
            )
            if pair is not None:
                augmented.append(replace(context, context_before=pair[0], context_after=pair[1]))
                continue
        augmented.append(context)

    site_run = run_generator_study(
        augmented,
        partition,
        specs,
        policy,
        role=role,
        scored_document_ids=scored_document_ids,
        scored_engines=scored_engines,
        progress=progress,
    )
    census = build_site_census(
        contexts,
        partition,
        role,
        scored_document_ids=scored_document_ids,
        scored_engines=scored_engines,
    )

    engines = sorted({record.engine_id for record in census})
    fit_documents = partition.documents(SplitRole.FIT)

    region_started = perf_counter()
    regions, region_census = build_regions(census, streams)
    region_records: list[RegionRecord] = []
    for engine_id in engines:
        corpus_without = [
            c.site.ocr_text
            for c in contexts
            if c.site.document_id in fit_documents and c.site.engine_id != engine_id
        ]
        structural = StructuralGenerator(generator_id="g5_structural")
        structural.fit(corpus_without)
        engine_regions = [r for r in regions if r.engine_id == engine_id]
        if progress is not None and engine_regions:
            progress(f"g5_structural regions x {engine_id}: {len(engine_regions)} pairs")
        for region in engine_regions:
            proposals = structural.propose_region(region.left.ocr_text, region.right.ocr_text, None)
            seen: set[str] = set()
            for rank, proposal in enumerate(proposals):
                if proposal.pool is not CandidatePool.NATURAL:
                    raise ValueError(
                        f"g5_structural emitted a {proposal.pool.value}-pool region candidate "
                        f"at {region.region_id}; the study admits natural candidates only"
                    )
                if proposal.text == region.region_ocr or proposal.text in seen:
                    continue
                seen.add(proposal.text)
                d_after = distance(proposal.text, region.region_gt)
                region_records.append(
                    RegionRecord(
                        generator_id="g5_structural",
                        region_id=region.region_id,
                        document_id=region.document_id,
                        dataset_id=region.dataset_id,
                        engine_id=region.engine_id,
                        role=role.value,
                        left_site_id=region.left.site_id,
                        right_site_id=region.right.site_id,
                        gap_chars=region.gap_chars,
                        region_char_start=region.region_char_start,
                        region_char_end=region.region_char_end,
                        left_kind=region.left.site_kind,
                        right_kind=region.right.site_kind,
                        region_ocr=region.region_ocr,
                        region_gt=region.region_gt,
                        candidate_text=proposal.text,
                        d_before=distance(region.region_ocr, region.region_gt),
                        d_after=d_after,
                        delta=distance(region.region_ocr, region.region_gt) - d_after,
                        outcome=classify_accepted(
                            region.region_ocr, proposal.text, region.region_gt
                        ),
                        generator_rank=rank,
                        generator_meta=dict(proposal.metadata),
                        generator_version=structural.version,
                        generator_score=proposal.score,
                        pool=proposal.pool.value,
                    )
                )

    region_seconds = perf_counter() - region_started
    certificate_started = perf_counter()
    certificates, fold_tokens = fold_certificates(
        contexts,
        partition,
        specs,
        engines,
        scored_role=role,
        scored_document_ids=scored_document_ids,
    )
    failed_folds = [
        engine_id
        for engine_id, certificate in certificates.items()
        if not bool(certificate.get("pass"))
    ]
    if failed_folds:
        msg = f"CGV2 leakage certificate failed for folds: {', '.join(sorted(failed_folds))}"
        raise ValueError(msg)
    # The certificate refits its own generators; assert the refit matches what scored.
    # Without the check the certificate verifies the rule as coded twice, not the state
    # that produced the proposals.
    for engine_id, certificate in certificates.items():
        certified_rungs = certificate["rungs"]
        assert isinstance(certified_rungs, dict)
        for spec in specs:
            recorded = site_run.lexicon_sizes.get(f"{spec.id}:{engine_id}")
            certified = certified_rungs.get(spec.id, {}).get("lexicon_size")
            if recorded is not None and certified is not None and recorded != certified:
                msg = (
                    f"fold certificate mismatch for {spec.id} on {engine_id}: scoring loop "
                    f"fitted {recorded} tokens, certificate refit {certified}"
                )
                raise ValueError(msg)
    # Document-axis runtime guard: the scored role's documents must be disjoint from the
    # fit corpus the certificates and lexicons were built on.
    scored_document_set = set(partition.documents(role))
    fit_document_set = set(partition.documents(SplitRole.FIT))
    overlap = scored_document_set & fit_document_set
    if overlap:
        msg = f"scored role documents overlap the fit documents: {sorted(overlap)[:5]}"
        raise ValueError(msg)
    runtime_seconds = {
        **site_run.runtime_seconds,
        "regions": region_seconds,
        "certificates": perf_counter() - certificate_started,
        "total": perf_counter() - study_started,
    }
    return Cgv2Run(
        site_run=site_run,
        census=census,
        region_records=region_records,
        region_census=region_census,
        certificates=certificates,
        fold_tokens=fold_tokens,
        role=role.value,
        runtime_seconds=runtime_seconds,
    )


@dataclass(frozen=True, slots=True)
class EngineContrast:
    """One rung against another on one engine, site-level availability, paired by document."""

    engine_id: str
    challenger: str
    baseline: str
    metric: str
    stratum: str
    challenger_value: float
    baseline_value: float
    delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    standard_error: float
    minimum_detectable_effect: float
    degenerate_interval: bool
    challenger_native_cap: int
    baseline_native_cap: int
    n_documents: int
    n_sites: int

    def as_dict(self) -> dict[str, object]:
        return {
            "engine_id": self.engine_id,
            "challenger": self.challenger,
            "baseline": self.baseline,
            "metric": self.metric,
            "stratum": self.stratum,
            "challenger_value": self.challenger_value,
            "baseline_value": self.baseline_value,
            "delta": self.delta,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "standard_error": self.standard_error,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "degenerate_interval": self.degenerate_interval,
            "challenger_native_cap": self.challenger_native_cap,
            "baseline_native_cap": self.baseline_native_cap,
            "n_documents": self.n_documents,
            "n_sites": self.n_sites,
        }


def _repairable_sites(
    proposals: Sequence[ProposalRecord],
    generator_id: str,
    policy: HarmPolicy,
    *,
    exact_only: bool = False,
) -> set[str]:
    """Site ids where ``generator_id`` has at least one repair-counting proposal."""
    if generator_id not in GENERATOR_LADDER:
        raise ValueError(
            f"unknown rung {generator_id!r}; its frozen candidate cap cannot be applied"
        )
    repairable: set[str] = set()
    for proposal in proposals:
        if proposal.generator_id != generator_id:
            continue
        if proposal.generator_rank >= GENERATOR_LADDER[generator_id].max_candidates:
            continue
        if exact_only and proposal.outcome is not OutcomeIfAccepted.TRUE_CORRECTION:
            continue
        site = SiteProposals(
            site_id=proposal.site_id,
            d_before=proposal.d_before,
            outcomes=(proposal.outcome,),
            deltas=(proposal.delta,),
            needs_deletion=not proposal.gt_text,
        )
        if site.repairable(policy):
            repairable.add(proposal.site_id)
    return repairable


def engine_contrast(
    census: Sequence[SiteCensusRecord],
    proposals: Sequence[ProposalRecord],
    challenger: str,
    baseline: str,
    policy: HarmPolicy,
    *,
    kinds: frozenset[str] | None = None,
    only_error_sites: bool = True,
    exact_only: bool = False,
    metric_name: str = "site_availability",
    n_resamples: int = 10_000,
    seed: int = 7,
    ci_level: float = 0.95,
) -> list[EngineContrast]:
    """Per-engine paired document-cluster bootstrap on site availability.

    One contrast per engine (CGV2 reports per engine and never averages), documents drawn
    once for both rungs so the pair is matched. ``kinds`` restricts the stratum;
    ``only_error_sites`` restricts to error sites, which is the denominator
    ``error_repair_opportunity`` uses; ``exact_only`` is the pre-registered sensitivity
    that counts exact repairs alone.
    """
    from ocr_risk.stats import cluster_bootstrap, minimum_detectable_effect

    repairable = {
        rung: _repairable_sites(proposals, rung, policy, exact_only=exact_only)
        for rung in (challenger, baseline)
    }
    engines = sorted({record.engine_id for record in census})
    contrasts: list[EngineContrast] = []

    for engine_id in engines:
        rows = [
            record
            for record in census
            if record.engine_id == engine_id
            and (kinds is None or record.site_kind in kinds)
            and (not only_error_sites or record.d_before > 0)
        ]
        if not rows:
            continue
        by_document: dict[str, list[SiteCensusRecord]] = {}
        for record in rows:
            by_document.setdefault(record.document_id, []).append(record)
        items = [
            (
                document_id,
                len(sites),
                sum(1 for s in sites if s.site_id in repairable[challenger]),
                sum(1 for s in sites if s.site_id in repairable[baseline]),
            )
            for document_id, sites in sorted(by_document.items())
        ]

        def _delta(sample: Sequence[tuple[str, int, int, int]]) -> float:
            totals = sum(row[1] for row in sample)
            if not totals:
                return float("nan")
            hit_c = sum(row[2] for row in sample)
            hit_b = sum(row[3] for row in sample)
            return hit_c / totals - hit_b / totals

        result = cluster_bootstrap(
            items,
            lambda item: item[0],
            _delta,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
            bounds=None,
        )
        totals = sum(row[1] for row in items)
        contrasts.append(
            EngineContrast(
                engine_id=engine_id,
                challenger=challenger,
                baseline=baseline,
                metric=metric_name,
                stratum=",".join(sorted(kinds)) if kinds else "all_error_sites",
                challenger_value=sum(row[2] for row in items) / totals,
                baseline_value=sum(row[3] for row in items) / totals,
                delta=result.estimate,
                ci_lower=result.lower,
                ci_upper=result.upper,
                p_value=result.p_value_two_sided,
                standard_error=result.standard_error,
                minimum_detectable_effect=minimum_detectable_effect(
                    result.standard_error, n_tests=4
                ),
                degenerate_interval=result.degenerate_interval,
                challenger_native_cap=GENERATOR_LADDER[challenger].max_candidates,
                baseline_native_cap=GENERATOR_LADDER[baseline].max_candidates,
                n_documents=len(items),
                n_sites=totals,
            )
        )
    return contrasts


def augmented_oracle(
    census: Sequence[SiteCensusRecord],
    proposals: Sequence[ProposalRecord],
    region_records: Sequence[RegionRecord],
    site_rung: str,
    policy: HarmPolicy,
) -> dict[str, dict[str, float | int]]:
    """Oracle opportunity with regions credited to their member sites (protocol §14).

    A site is oracle-accepted when a site-level acceptance repairs it or a region-level
    acceptance whose members include it repairs the region. The unit stays the site, so
    the counts are directly comparable to the historical criterion A.
    """
    non_natural_sites = [record for record in proposals if record.pool != "natural"]
    non_natural_regions = [record for record in region_records if record.pool != "natural"]
    if non_natural_sites or non_natural_regions:
        msg = (
            "CGV2 oracle accepts only the natural pool; received "
            f"{len(non_natural_sites)} site and {len(non_natural_regions)} region "
            "challenge/diagnostic candidate(s)"
        )
        raise ValueError(msg)
    accepted: dict[str, set[str]] = {}
    engine_sites = {
        engine_id: {record.site_id for record in census if record.engine_id == engine_id}
        for engine_id in sorted({record.engine_id for record in census})
    }
    all_repairable = _repairable_sites(proposals, site_rung, policy)
    for engine_id, sites in engine_sites.items():
        accepted[engine_id] = all_repairable & sites
    for record in region_records:
        if record.engine_id not in engine_sites:
            continue
        if record.generator_rank >= REGION_CANDIDATE_CAP:
            # The region generator's frozen cap: ranks beyond it are K-grid diagnostics,
            # not pool members, exactly as site ranks beyond a rung's cap are.
            continue
        if not {record.left_site_id, record.right_site_id} <= engine_sites[record.engine_id]:
            continue
        # A1: a region with any insertion-kind member needs an exact repair -- edit
        # distance would otherwise score a merged hallucination as a partial improvement,
        # the region-level form of the artifact amendment A6 closed at site level.
        site = SiteProposals(
            site_id=record.region_id,
            d_before=record.d_before,
            outcomes=(record.outcome,),
            deltas=(record.delta,),
            needs_deletion="insertion" in (record.left_kind, record.right_kind),
        )
        if not site.repairable(policy):
            continue
        accepted.setdefault(record.engine_id, set()).update(
            (record.left_site_id, record.right_site_id)
        )

    result: dict[str, dict[str, float | int]] = {}
    for engine_id, sites in accepted.items():
        n_evaluable = sum(1 for record in census if record.engine_id == engine_id)
        n_accepted = len(sites)
        via_site = all_repairable & engine_sites.get(engine_id, set())
        result[engine_id] = {
            "n_evaluable_sites": n_evaluable,
            "oracle_accepted_edits": n_accepted,
            "oracle_safe_coverage": n_accepted / n_evaluable if n_evaluable else float("nan"),
            # The credit-both rule of section 14 means one region acceptance marks two
            # sites; this column separates site-level repairs from region-credited ones
            # so a reader can see how much of A rides on the region layer.
            "sites_via_region": n_accepted - len(via_site),
        }
    return result


def proposals_from_csv(path: Path) -> list[ProposalRecord]:
    """Rebuild typed proposal records from a canonical CSV, for sensitivity contrasts.

    Reading the artifact back -- rather than keeping the in-memory records alive past the
    run -- is what makes the sensitivity a re-derivation from canonical artifacts instead
    of a second execution of the study.
    """
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False)
    if "pool" not in frame.columns:
        raise ValueError(f"{path} is missing the required candidate-pool provenance")
    records: list[ProposalRecord] = []
    for row in frame.to_dict(orient="records"):
        confidence = str(row.get("normalized_confidence", "") or "")
        score = str(row.get("generator_score", "") or "")
        metadata = {
            str(key)[5:]: str(value)
            for key, value in row.items()
            if str(key).startswith("meta_") and str(value)
        }
        records.append(
            ProposalRecord(
                generator_id=str(row["generator_id"]),
                site_id=str(row["site_id"]),
                document_id=str(row["document_id"]),
                dataset_id=str(row["dataset_id"]),
                engine_id=str(row["engine_id"]),
                role=str(row["role"]),
                original_ocr=str(row["original_ocr"]),
                candidate_text=str(row["candidate_text"]),
                gt_text=str(row["gt_text"]),
                d_before=int(str(row["d_before"])),
                d_after=int(str(row["d_after"])),
                delta=int(str(row["delta"])),
                outcome=OutcomeIfAccepted(str(row["outcome"])),
                site_kind=str(row["site_kind"]),
                normalized_confidence=float(confidence) if confidence else None,
                generator_rank=int(str(row.get("generator_rank", 0) or 0)),
                generator_meta=metadata,
                generator_version=str(row.get("generator_version", "unknown") or "unknown"),
                generator_score=float(score) if score else None,
                pool=str(row["pool"]),
            )
        )
    return records


def region_records_from_csv(path: Path) -> list[RegionRecord]:
    """Rebuild typed region records from their canonical CSV.

    Alignment and harm-policy sensitivities re-derive the augmented oracle from saved
    artifacts. Keeping this parser beside :func:`proposals_from_csv` prevents a second,
    subtly different representation of a region proposal in the CLI layer.
    """
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False)
    if "pool" not in frame.columns:
        raise ValueError(f"{path} is missing the required candidate-pool provenance")
    records: list[RegionRecord] = []
    for row in frame.to_dict(orient="records"):
        score = str(row.get("generator_score", "") or "")
        metadata = {
            str(key)[5:]: str(value)
            for key, value in row.items()
            if str(key).startswith("meta_") and str(value)
        }
        records.append(
            RegionRecord(
                generator_id=str(row["generator_id"]),
                region_id=str(row["region_id"]),
                document_id=str(row["document_id"]),
                dataset_id=str(row["dataset_id"]),
                engine_id=str(row["engine_id"]),
                role=str(row["role"]),
                left_site_id=str(row["left_site_id"]),
                right_site_id=str(row["right_site_id"]),
                gap_chars=int(str(row["gap_chars"])),
                region_char_start=int(str(row["region_char_start"])),
                region_char_end=int(str(row["region_char_end"])),
                left_kind=str(row["left_kind"]),
                right_kind=str(row["right_kind"]),
                region_ocr=str(row["region_ocr"]),
                region_gt=str(row["region_gt"]),
                candidate_text=str(row["candidate_text"]),
                d_before=int(str(row["d_before"])),
                d_after=int(str(row["d_after"])),
                delta=int(str(row["delta"])),
                outcome=OutcomeIfAccepted(str(row["outcome"])),
                generator_rank=int(str(row.get("generator_rank", 0) or 0)),
                generator_meta=metadata,
                pool=str(row["pool"]),
                generator_version=str(row.get("generator_version", "unknown") or "unknown"),
                generator_score=float(score) if score else None,
            )
        )
    return records

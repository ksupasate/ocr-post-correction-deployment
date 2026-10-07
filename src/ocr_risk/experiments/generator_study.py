"""The candidate-generator readiness study: measuring Q1 on its own.

The H1 pilot could not tell three failures apart — a generator that proposes nothing worth
accepting, a verifier that cannot rank, and a controller that cannot hit a harm target. It
observed zero coverage and had one number for all three. This module measures the first in
isolation: no verifier, no calibrator, no threshold. Just what the generator proposed and
what would have happened if each proposal were accepted.

**The lexicon is built without the engine whose sites are being scored.** The shipped
pipeline built one lexicon over the fit documents of *every* engine and shared it across
folds, which its own comment forbids: under leave-one-engine-out the held-out engine
contributed vocabulary that then gated candidate generation on its own sites. Measured on
the pilot corpus, that suppressed 4.9%-8.6% of each engine's evaluable sites — the ones
where its characteristic misreading recurred, which is exactly where correction matters.
Here the corpus for engine *e*'s lexicon excludes engine *e* entirely.

**Evaluation documents are disjoint from the lexicon corpus.** The lexicon comes from
``D_fit``; the study scores ``D_cal`` for development and ``D_test`` only once, for the
frozen generator. A generator scored on the pages its own vocabulary came from is being
asked an easier question than deployment asks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from ocr_risk.candidates.base import CandidateGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.pipeline import SiteContext
from ocr_risk.candidates.registry import build_generator
from ocr_risk.edits.outcome import classify_accepted, distance
from ocr_risk.metrics.generator import GeneratorQuality, SiteProposals, generator_quality
from ocr_risk.schemas.enums import CandidatePool, HarmPolicy, OutcomeIfAccepted, SplitRole
from ocr_risk.splits.document_partition import DocumentPartition

__all__ = [
    "GeneratorRun",
    "GeneratorSpec",
    "ProposalRecord",
    "SelectionContrast",
    "StudyCell",
    "run_generator_study",
    "selection_contrast",
]


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    """One generator to evaluate, with the scope it is allowed to claim."""

    id: str
    kind: str
    params: dict[str, object] = field(default_factory=dict)
    max_candidates: int = 4
    datasets: tuple[str, ...] = ()
    """Corpora this generator may be scored on. Empty means all of them.

    Not a convenience. ``byt5`` is fine-tuned on English; running it on German Fraktur or
    Indonesian receipts would measure a language mismatch and report it as generator
    quality, and a reader comparing that number against a language-agnostic baseline would
    be comparing two different things.
    """

    def applies_to(self, dataset_id: str) -> bool:
        return not self.datasets or dataset_id in self.datasets


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """One proposed edit, with everything needed to classify and stratify it.

    Ground truth is present because this is **offline analysis**, downstream of generation.
    The generator saw a :class:`GenerationContext`, which has no ground-truth field.
    """

    generator_id: str
    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    role: str
    original_ocr: str
    candidate_text: str
    gt_text: str
    d_before: int
    d_after: int
    delta: int
    outcome: OutcomeIfAccepted
    site_kind: str
    normalized_confidence: float | None
    generator_rank: int = 0
    """Rank within the generator's distinct non-identity proposals at this site.

    Zero-based, so the CGV2 K-grid truncation is ``generator_rank < K`` (protocol §12).
    """
    generator_meta: dict[str, str] = field(default_factory=dict)
    generator_version: str = "unknown"
    generator_score: float | None = None
    pool: str = "natural"

    def as_dict(self) -> dict[str, object]:
        return {
            "generator_id": self.generator_id,
            "site_id": self.site_id,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "engine_id": self.engine_id,
            "role": self.role,
            "original_ocr": self.original_ocr,
            "candidate_text": self.candidate_text,
            "gt_text": self.gt_text,
            "d_before": self.d_before,
            "d_after": self.d_after,
            "delta": self.delta,
            "outcome": self.outcome.value,
            "site_kind": self.site_kind,
            "normalized_confidence": self.normalized_confidence,
            "generator_rank": self.generator_rank,
            "generator_version": self.generator_version,
            "generator_score": self.generator_score,
            "pool": self.pool,
            **{f"meta_{k}": v for k, v in self.generator_meta.items()},
        }


@dataclass(frozen=True, slots=True)
class StudyCell:
    """Generator quality for one (generator, engine, dataset) cell."""

    generator_id: str
    engine_id: str
    dataset_id: str
    role: str
    quality: GeneratorQuality

    def as_dict(self) -> dict[str, object]:
        return {
            "generator_id": self.generator_id,
            "engine_id": self.engine_id,
            "dataset_id": self.dataset_id,
            "role": self.role,
            **self.quality.as_dict(),
        }


@dataclass(slots=True)
class GeneratorRun:
    """Everything one study produced."""

    proposals: list[ProposalRecord] = field(default_factory=list)
    cells: list[StudyCell] = field(default_factory=list)
    lexicon_sizes: dict[str, int] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    n_sites_scored: int = 0
    runtime_seconds: dict[str, float] = field(default_factory=dict)
    """Wall time per ``generator_id:held_out_engine`` cell, including load and fitting."""


def _normalized_confidence(context: SiteContext) -> float | None:
    from ocr_risk.canonical.confidence import normalize_confidence

    values = [
        v
        for v in (normalize_confidence(c, context.conf_scale) for c in context.native_confidences)
        if v is not None
    ]
    return min(values) if values else None


def _propose_all(
    generator: CandidateGenerator,
    contexts: Sequence[GenerationContext],
    max_candidates: int,
) -> list[list[CandidateProposal]]:
    """Use a batched interface when the generator offers one.

    A byte model tokenizes one token per byte; calling it per site leaves the device idle
    between spans and turns a 15-minute study into a 3-hour one.
    """
    batched = getattr(generator, "propose_batch", None)
    if callable(batched):
        result: list[list[CandidateProposal]] = batched(contexts, max_candidates)
        return result
    return [generator.propose(context, max_candidates) for context in contexts]


def run_generator_study(
    contexts: Sequence[SiteContext],
    partition: DocumentPartition,
    generators: Sequence[GeneratorSpec],
    policy: HarmPolicy,
    role: SplitRole = SplitRole.CALIBRATE,
    lexicon_role: SplitRole = SplitRole.FIT,
    scored_document_ids: frozenset[str] | None = None,
    scored_engines: frozenset[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> GeneratorRun:
    """Score every generator on every engine, on documents its lexicon never saw.

    ``role`` selects the documents being scored — ``calibrate`` for development, ``evaluate``
    only once, for the generator that has already been frozen. Keeping the two apart is what
    stops the generator from being chosen on the data it is then reported on.
    """
    run = GeneratorRun()
    lexicon_documents = partition.documents(lexicon_role)
    role_documents = partition.documents(role)
    scored_documents = role_documents if scored_document_ids is None else scored_document_ids
    outside_role = scored_documents - role_documents
    if outside_role:
        msg = (
            f"scored_document_ids contains {len(outside_role)} document(s) outside the "
            f"{role.value} role: {sorted(outside_role)[:5]}"
        )
        raise ValueError(msg)

    by_engine: dict[str, list[SiteContext]] = {}
    for context in contexts:
        if (
            context.site.document_id in scored_documents
            and context.site.evaluable
            and (scored_engines is None or context.site.engine_id in scored_engines)
        ):
            by_engine.setdefault(context.site.engine_id, []).append(context)
    run.n_sites_scored = sum(len(v) for v in by_engine.values())

    for engine_id in sorted(by_engine):
        # Leave-this-engine-out lexicon corpus, from the lexicon-role documents only.
        corpus = [
            c.site.ocr_text
            for c in contexts
            if c.site.document_id in lexicon_documents and c.site.engine_id != engine_id
        ]
        engine_contexts = by_engine[engine_id]

        for spec in generators:
            eligible = [c for c in engine_contexts if spec.applies_to(c.site.dataset_id)]
            if not eligible:
                continue
            cell_started = perf_counter()
            generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
            if not generator.available():
                reason = str(getattr(generator, "load_error", "") or "reported unavailable")
                run.unavailable[spec.id] = reason
                run.runtime_seconds[f"{spec.id}:{engine_id}"] = perf_counter() - cell_started
                continue
            generator.fit(corpus)
            # Only when the generator has one. The fallback used to be
            # `len(set(corpus))`, so three of four rungs recorded 21 293 -- the count of
            # distinct corpus strings -- and a reader would conclude they had four times
            # the lexicon exposure of the one rung that reported honestly. ByT5 has no
            # lexicon at all and now records none.
            lexicon_size = getattr(generator, "lexicon_size", None)
            if lexicon_size is not None:
                run.lexicon_sizes[f"{spec.id}:{engine_id}"] = int(lexicon_size)
            if progress is not None:
                progress(f"{spec.id} x {engine_id}: {len(eligible)} sites")

            generation_contexts = [c.generation_context() for c in eligible]
            proposals = _propose_all(generator, generation_contexts, spec.max_candidates)

            per_site: dict[tuple[str, str], SiteProposals] = {}
            for context, site_proposals in zip(eligible, proposals, strict=True):
                site = context.site
                key = (site.dataset_id, site.site_id)
                outcomes: list[OutcomeIfAccepted] = []
                deltas: list[int] = []
                seen: set[str] = set()
                for proposal in site_proposals:
                    if proposal.pool is not CandidatePool.NATURAL:
                        # The readiness study measures the natural pool; a challenge row
                        # here would corrupt every denominator downstream of it.
                        raise ValueError(
                            f"{spec.id} emitted a {proposal.pool.value}-pool candidate at "
                            f"{site.site_id}; the study admits natural candidates only"
                        )
                    if proposal.text == site.ocr_text or proposal.text in seen:
                        continue  # identity, or the same edit twice
                    seen.add(proposal.text)
                    d_after = distance(proposal.text, site.gt_text)
                    outcome = classify_accepted(site.ocr_text, proposal.text, site.gt_text)
                    outcomes.append(outcome)
                    deltas.append(site.d_before - d_after)
                    run.proposals.append(
                        ProposalRecord(
                            generator_id=spec.id,
                            site_id=site.site_id,
                            document_id=site.document_id,
                            dataset_id=site.dataset_id,
                            engine_id=site.engine_id,
                            role=role.value,
                            original_ocr=site.ocr_text,
                            candidate_text=proposal.text,
                            gt_text=site.gt_text,
                            d_before=site.d_before,
                            d_after=d_after,
                            delta=site.d_before - d_after,
                            outcome=outcome,
                            site_kind=site.site_kind.value,
                            normalized_confidence=_normalized_confidence(context),
                            generator_rank=len(seen) - 1,
                            generator_meta=dict(proposal.metadata),
                            generator_version=generator.version,
                            generator_score=proposal.score,
                            pool=proposal.pool.value,
                        )
                    )
                per_site[key] = SiteProposals(
                    site_id=site.site_id,
                    d_before=site.d_before,
                    outcomes=tuple(outcomes),
                    deltas=tuple(deltas),
                    # The ground truth is empty: the OCR invented this span and the only
                    # correct edit removes it. Without the flag, a shorter hallucination
                    # scores as a partial repair in every oracle ceiling.
                    needs_deletion=not site.gt_text,
                )

            by_dataset: dict[str, list[SiteProposals]] = {}
            for (dataset_id, _), site_proposal in per_site.items():
                by_dataset.setdefault(dataset_id, []).append(site_proposal)
            for dataset_id, sites in sorted(by_dataset.items()):
                run.cells.append(
                    StudyCell(
                        generator_id=spec.id,
                        engine_id=engine_id,
                        dataset_id=dataset_id,
                        role=role.value,
                        quality=generator_quality(sites, policy),
                    )
                )
            # The pooled cell as well: a per-corpus rate cannot be averaged into a corpus
            # total, because the corpora differ in size by more than a factor of two.
            #
            # Labelled with its SCOPE, not "ALL". A corpus-restricted generator's pooled
            # cell covers only the corpora it was allowed to run on -- g2_byt5's read
            # `ALL` while holding 2 151-2 742 FUNSD sites against the others' 4 101-4 890,
            # and the readiness floor it was then judged against was derived from the
            # larger denominator.
            covered = sorted({dataset_id for dataset_id, _ in per_site})
            run.cells.append(
                StudyCell(
                    generator_id=spec.id,
                    engine_id=engine_id,
                    dataset_id="ALL" if not spec.datasets else f"ALL:{'+'.join(covered)}",
                    role=role.value,
                    quality=generator_quality(list(per_site.values()), policy),
                )
            )
            run.runtime_seconds[f"{spec.id}:{engine_id}"] = perf_counter() - cell_started
    return run


# (document, engine index, sites, repairable-for-challenger, repairable-for-baseline).
_SelectionRow = tuple[str, int, int, int, int]


@dataclass(frozen=True, slots=True)
class SelectionContrast:
    """One generator against another on the selection metric, with a paired interval.

    The protocol's selection rule is a mean over engines and was applied as a point
    estimate. `.claude/rules/metrics.md` requires a paired bootstrap and an effect size
    with its CI for a comparison on the same documents, and the margin turned out to be
    inside sampling noise — which is a fact about the selection that a reader needs and a
    point estimate cannot carry.
    """

    challenger: str
    baseline: str
    metric: str
    challenger_value: float
    baseline_value: float
    delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_documents: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "challenger": self.challenger,
            "baseline": self.baseline,
            "metric": self.metric,
            "challenger_value": self.challenger_value,
            "baseline_value": self.baseline_value,
            "delta": self.delta,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "n_documents": self.n_documents,
        }


def selection_contrast(
    run: GeneratorRun,
    site_totals: dict[tuple[str, str], int],
    challenger: str,
    baseline: str,
    policy: HarmPolicy,
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 7,
) -> SelectionContrast:
    """Paired document-level bootstrap on the mean-over-engines oracle safe coverage.

    ``site_totals`` maps ``(engine_id, document_id)`` to the number of evaluable sites,
    which is the denominator; the proposals alone cannot supply it, because a site with no
    proposal appears nowhere in them and is exactly what the denominator is for. Documents
    are the resampling unit, drawn once for both generators so the contrast is paired.
    """
    from ocr_risk.stats import cluster_bootstrap

    repairable: dict[str, set[tuple[str, str, str]]] = {challenger: set(), baseline: set()}
    for proposal in run.proposals:
        if proposal.generator_id not in repairable:
            continue
        site = SiteProposals(
            site_id=proposal.site_id,
            d_before=proposal.d_before,
            outcomes=(proposal.outcome,),
            deltas=(proposal.delta,),
            needs_deletion=not proposal.gt_text,
        )
        if site.repairable(policy):
            repairable[proposal.generator_id].add(
                (proposal.engine_id, proposal.document_id, proposal.site_id)
            )

    engines = sorted({engine for engine, _ in site_totals})
    engine_index = {engine: position for position, engine in enumerate(engines)}

    # Per (engine, document): sites, and repairable sites for each generator. Precomputed
    # so a resample is a sum over its rows rather than a scan of the hit set -- the naive
    # form is O(|hits| x |documents|) per engine per resample, which at 10 000 resamples is
    # billions of comparisons and does not finish.
    hits: dict[str, dict[tuple[str, str], int]] = {challenger: {}, baseline: {}}
    for generator_id, sites in repairable.items():
        for engine_id, document_id, _site_id in sites:
            key = (engine_id, document_id)
            hits[generator_id][key] = hits[generator_id].get(key, 0) + 1

    items: list[_SelectionRow] = [
        (
            document_id,
            engine_index[engine_id],
            count,
            hits[challenger].get((engine_id, document_id), 0),
            hits[baseline].get((engine_id, document_id), 0),
        )
        for (engine_id, document_id), count in site_totals.items()
    ]

    def _means(sample: Sequence[_SelectionRow]) -> tuple[float, float]:
        n_engines = len(engines)
        totals = [0] * n_engines
        covered_a = [0] * n_engines
        covered_b = [0] * n_engines
        for _document, engine, count, hit_a, hit_b in sample:
            totals[engine] += count
            covered_a[engine] += hit_a
            covered_b[engine] += hit_b
        # Equal weight per engine, as the protocol's rule specifies: four fixed
        # environments, not four draws from a population of engines.
        rates_a = [covered_a[i] / totals[i] for i in range(n_engines) if totals[i]]
        rates_b = [covered_b[i] / totals[i] for i in range(n_engines) if totals[i]]
        if not rates_a:
            return float("nan"), float("nan")
        return float(np.mean(rates_a)), float(np.mean(rates_b))

    def paired(sample: Sequence[_SelectionRow]) -> float:
        left, right = _means(sample)
        return left - right

    result = cluster_bootstrap(
        items,
        lambda item: item[0],
        paired,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        bounds=None,
    )
    challenger_value, baseline_value = _means(items)
    return SelectionContrast(
        challenger=challenger,
        baseline=baseline,
        metric="mean_oracle_safe_coverage",
        challenger_value=challenger_value,
        baseline_value=baseline_value,
        delta=result.estimate,
        ci_lower=result.lower,
        ci_upper=result.upper,
        p_value=result.p_value_two_sided,
        n_documents=result.n_clusters,
    )

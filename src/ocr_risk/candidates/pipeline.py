"""Turning sites into candidates, labels, and evidence.

Three stages that must stay in this order and this separation:

1. **candidates** — generators see a ground-truth-free projection of each site;
2. **labels** — ground truth enters *here and nowhere else* downstream of alignment;
3. **evidence** — built from an ``ObservationView`` that structurally has no labels.

Running labelling between generation and evidence is deliberate: it keeps the only
GT-touching step in one place, small enough to audit by reading it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ocr_risk.candidates.base import CandidateGenerator, GenerationContext
from ocr_risk.config.models import CandidateConfig
from ocr_risk.edits.outcome import classify_accepted, classify_rejected, distance
from ocr_risk.schemas.candidates import Candidate, CandidateLabel
from ocr_risk.schemas.enums import OutcomeIfAccepted
from ocr_risk.schemas.evidence import ObservationView
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan

__all__ = ["SiteContext", "build_labels", "build_observation_views", "generate_candidates"]


@dataclass(frozen=True, slots=True)
class SiteContext:
    """Per-site information the generation and evidence stages both need."""

    site: CorrectionSite
    context_before: str
    context_after: str
    native_confidences: tuple[float | None, ...]
    conf_scale: str | None
    image_sha256: str
    image_width: int
    image_height: int

    def generation_context(self) -> GenerationContext:
        """Project to the generator's view, dropping ground truth."""
        return GenerationContext.from_site(
            self.site,
            self.context_before,
            self.context_after,
            self.native_confidences,
            self.conf_scale,
        )


def generate_candidates(
    contexts: Sequence[SiteContext],
    generators: Sequence[tuple[CandidateGenerator, int]],
    config: CandidateConfig,
) -> list[Candidate]:
    """Run every generator over every site.

    Identity proposals are dropped when configured: an edit that changes nothing is not a
    decision the verifier should be scored on, and leaving them in would inflate coverage
    with free "accepts".
    """
    candidates: list[Candidate] = []
    for site_context in contexts:
        site = site_context.site
        generation_context = site_context.generation_context()
        seen: set[str] = set()
        per_site = 0

        for generator, max_candidates in generators:
            if not generator.available():
                continue
            for rank, proposal in enumerate(generator.propose(generation_context, max_candidates)):
                if per_site >= config.max_candidates_per_site:
                    break
                if config.drop_identity_candidates and proposal.text == site.ocr_text:
                    continue
                if proposal.text in seen:
                    # The same text from two generators is one edit; keeping both would
                    # double-count it in coverage and in harm.
                    continue
                seen.add(proposal.text)
                candidates.append(
                    Candidate(
                        candidate_id=f"{site.site_id}:{generator.generator_id}:{rank}",
                        site_id=site.site_id,
                        document_id=site.document_id,
                        dataset_id=site.dataset_id,
                        engine_id=site.engine_id,
                        candidate_text=proposal.text,
                        generator_id=generator.generator_id,
                        generator_version=generator.version,
                        generator_rank=rank,
                        generator_score=proposal.score,
                        is_synthetic_hard_negative=proposal.is_synthetic_hard_negative,
                        hard_negative_family=proposal.hard_negative_family,
                        metadata=proposal.metadata,
                    )
                )
                per_site += 1
    return candidates


def build_labels(
    candidates: Sequence[Candidate], sites: Sequence[CorrectionSite]
) -> list[CandidateLabel]:
    """Compute the counterfactual outcome of every candidate.

    **This is the only place ground truth enters after alignment.** Keeping it in one
    small function is what makes the claim auditable by reading rather than by trust.
    """
    site_by_id = {site.site_id: site for site in sites}
    labels: list[CandidateLabel] = []

    for candidate in candidates:
        site = site_by_id.get(candidate.site_id)
        if site is None:
            msg = f"candidate {candidate.candidate_id} references unknown site {candidate.site_id}"
            raise KeyError(msg)

        d_after = distance(candidate.candidate_text, site.gt_text)
        outcome = classify_accepted(site.ocr_text, candidate.candidate_text, site.gt_text)
        labels.append(
            CandidateLabel(
                candidate_id=candidate.candidate_id,
                site_id=site.site_id,
                document_id=site.document_id,
                dataset_id=site.dataset_id,
                engine_id=site.engine_id,
                d_before=site.d_before,
                d_after=d_after,
                delta=site.d_before - d_after,
                outcome_if_accepted=outcome,
                outcome_if_rejected=classify_rejected(site.ocr_text, site.gt_text),
                gt_text=site.gt_text,
            )
        )
    return labels


def build_observation_views(
    candidates: Sequence[Candidate], contexts: Sequence[SiteContext]
) -> list[tuple[ObservationView, SiteContext]]:
    """Project each candidate into the label-free view the evidence builder accepts."""
    context_by_site = {c.site.site_id: c for c in contexts}
    views: list[tuple[ObservationView, SiteContext]] = []

    for candidate in candidates:
        site_context = context_by_site.get(candidate.site_id)
        if site_context is None:
            continue
        site = site_context.site
        views.append(
            (
                ObservationView(
                    site_id=site.site_id,
                    candidate_id=candidate.candidate_id,
                    document_id=site.document_id,
                    dataset_id=site.dataset_id,
                    engine_id=site.engine_id,
                    original_ocr=site.ocr_text,
                    candidate_text=candidate.candidate_text,
                    ocr_span_ids=site.ocr_span_ids,
                    image_sha256=site_context.image_sha256,
                    image_width=site_context.image_width,
                    image_height=site_context.image_height,
                    bbox=site.bbox,
                ),
                site_context,
            )
        )
    return views


def confidences_for_site(
    site: CorrectionSite, spans_by_id: dict[str, CanonicalSpan]
) -> tuple[tuple[float | None, ...], str | None]:
    """Collect the native confidences of a site's spans, with their shared scale."""
    spans = [spans_by_id[sid] for sid in site.ocr_span_ids if sid in spans_by_id]
    scales = {s.conf_scale for s in spans if s.conf_scale}
    if len(scales) > 1:
        msg = f"site {site.site_id} mixes confidence scales {sorted(scales)}"
        raise ValueError(msg)
    return tuple(s.native_conf_recognition for s in spans), next(iter(scales), None)


def identity_outcomes(labels: Sequence[CandidateLabel]) -> int:
    """How many labels are identity edits — should be zero once filtering is on."""
    return sum(1 for label in labels if label.outcome_if_accepted is OutcomeIfAccepted.IDENTITY)

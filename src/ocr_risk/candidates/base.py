"""The candidate generator contract.

Candidate generation is a **baseline subsystem, not the proposed mechanism**. It exists so
the verifier has proposed edits to judge; the research question is about the judging. That
is why the interface is deliberately thin and why generators are interchangeable: the same
verifier must be evaluable across every generator, and a new generator must never require
a change to verification or evaluation code.

Generators are ground-truth-blind by construction. They receive a
:class:`GenerationContext` that carries the OCR span, its neighbours, and the page — and
no ground truth at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ocr_risk.schemas.enums import CandidatePool
from ocr_risk.schemas.sites import CorrectionSite

__all__ = ["CandidateGenerator", "CandidateProposal", "GenerationContext"]


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """What a generator may see. Deliberately contains no ground truth.

    ``site.gt_text`` exists on the site record, so generators never receive the site
    itself — only this projection of it. The distinction is enforced by
    ``tests/leakage/test_generator_is_gt_blind.py``.
    """

    site_id: str
    document_id: str
    dataset_id: str
    engine_id: str
    original_ocr: str
    context_before: str
    context_after: str
    native_confidences: tuple[float | None, ...] = ()
    conf_scale: str | None = None
    n_spans: int = 1

    @classmethod
    def from_site(
        cls,
        site: CorrectionSite,
        context_before: str,
        context_after: str,
        native_confidences: Sequence[float | None] = (),
        conf_scale: str | None = None,
    ) -> GenerationContext:
        """Project a site into the generator's view, dropping ground truth."""
        return cls(
            site_id=site.site_id,
            document_id=site.document_id,
            dataset_id=site.dataset_id,
            engine_id=site.engine_id,
            original_ocr=site.ocr_text,
            context_before=context_before,
            context_after=context_after,
            native_confidences=tuple(native_confidences),
            conf_scale=conf_scale,
            n_spans=len(site.ocr_span_ids),
        )


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """One proposed replacement, before it is assigned an id and persisted."""

    text: str
    score: float | None = None
    """The generator's own score, on its own scale. Not comparable across generators and
    never used as a verifier score."""
    is_synthetic_hard_negative: bool = False
    hard_negative_family: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def pool(self) -> CandidatePool:
        """Pool membership, derived from the one stored fact.

        The same rule the ``Candidate`` schema enforces: ``challenge`` if and only if the
        candidate was synthesized as a hard negative. Keeping the derivation here means
        callers cannot write a pool label that disagrees with its cause.
        """
        return CandidatePool.CHALLENGE if self.is_synthetic_hard_negative else CandidatePool.NATURAL


@runtime_checkable
class CandidateGenerator(Protocol):
    """Contract every candidate generator satisfies."""

    generator_id: str
    version: str

    def available(self) -> bool:
        """Whether this generator can run here (models present, keys configured)."""
        ...

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        """Propose replacements for one site, best first.

        Must be a pure function of ``context``. A generator that consulted ground truth,
        or that varied run to run, would make the verifier's job either trivial or
        irreproducible.
        """
        ...

    def fit(self, corpus: Sequence[str]) -> None:
        """Optionally learn from in-scope text (e.g. build a lexicon).

        ``corpus`` must come from the **fit split only**. Building a lexicon over every
        document would leak test-document vocabulary into the generator (leakage vector
        L5), which is why the caller passes an explicit corpus rather than the generator
        reading whatever it likes.
        """
        ...


class BaseGenerator:
    """Convenience base with the no-op defaults most generators want."""

    generator_id: str = "base"
    version: str = "1"

    def available(self) -> bool:
        return True

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def propose(
        self, context: GenerationContext, max_candidates: int
    ) -> list[CandidateProposal]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _meta(self, **values: Any) -> dict[str, str]:
        return {k: str(v) for k, v in values.items()}

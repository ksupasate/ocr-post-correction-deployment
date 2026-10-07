"""Declared-but-unimplemented generators.

These exist so the interface is demonstrably wide enough for the generators the roadmap
names, and so a config can reference one and get an honest "not implemented here" instead
of a mysterious registry miss.

They deliberately do **not** ship a half-working implementation. An LLM generator that
silently returned nothing would look like a null result rather than an absent component,
which is exactly the confusion this project's integrity rules exist to prevent.

``byt5`` used to live here. It is now real, in :mod:`ocr_risk.candidates.byt5`, against a
named and revision-pinned public checkpoint.

The direct multimodal rewriter is listed here as a **baseline**, not as the proposed
mechanism: a model that receives OCR plus the image and emits corrected text is the thing
H3 compares generate-then-verify *against*.
"""

from __future__ import annotations

import os

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import register_generator

__all__ = ["LlmGenerator", "MultimodalRewriterBaseline"]


class _NotYetImplemented(BaseGenerator):
    """Shared behaviour: report unavailable, and refuse rather than return nothing."""

    milestone: str = "roadmap"
    extra: str = ""

    def available(self) -> bool:
        return False

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        msg = (
            f"{self.generator_id!r} is declared but not implemented in this bootstrap "
            f"({self.milestone}). Returning no candidates would be indistinguishable from "
            "a model that found nothing, so it raises instead."
            + (f" Install extra: {self.extra}." if self.extra else "")
        )
        raise NotImplementedError(msg)


@register_generator("llm")
class LlmGenerator(_NotYetImplemented):
    """LLM post-correction. Requires an API key and is never required by core or CI."""

    generator_id = "llm"
    version = "0"
    milestone = "next milestone: learned generators"
    extra = "uv sync --extra llm, plus ANTHROPIC_API_KEY or OPENAI_API_KEY"

    def __init__(self, generator_id: str = "llm", model: str = "", **_: object) -> None:
        self.generator_id = generator_id
        self.model = model

    def available(self) -> bool:
        # Report honestly on both counts: the code is absent *and* so may be the key.
        return False

    @property
    def has_credentials(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@register_generator("multimodal_rewriter")
class MultimodalRewriterBaseline(_NotYetImplemented):
    """Direct image+OCR -> corrected text.

    This is the **baseline** in hypothesis H3, not the proposed method. The comparison it
    supports is whether generating a candidate and verifying it independently beats
    rewriting directly.
    """

    generator_id = "multimodal_rewriter"
    version = "0"
    milestone = "next milestone: H3 baseline"
    extra = "uv sync --extra torch"

    def __init__(self, generator_id: str = "multimodal_rewriter", **_: object) -> None:
        self.generator_id = generator_id

"""Identity generator: propose no change.

Emits ``O -> O``. The pipeline normally filters identity candidates out of the scored pool
(``candidates.drop_identity_candidates``), but the generator is kept because the
no-correction baseline is a real reference point: it is the policy that achieves zero
harmful edits and zero repair, and every risk-coverage curve should be readable against it.
"""

from __future__ import annotations

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import register_generator

__all__ = ["IdentityGenerator"]


@register_generator("identity")
class IdentityGenerator(BaseGenerator):
    generator_id = "identity"
    version = "1"

    def __init__(self, generator_id: str = "identity") -> None:
        self.generator_id = generator_id

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        if max_candidates < 1:
            return []
        return [
            CandidateProposal(
                text=context.original_ocr,
                score=0.0,
                metadata=self._meta(policy="no_correction_baseline"),
            )
        ]

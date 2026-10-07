"""Several generators behind one identity, in the order the pipeline would run them.

Not a new proposal mechanism. ``candidates/pipeline.py`` already loops over every enabled
generator and de-duplicates the result, so a configuration listing two correctors produces
their union — and the readiness study was scoring the rungs *individually*, which is not
the configuration the pipeline runs. This closes that gap: the union is measurable on the
same footing as the rungs it is made of.

Composition is not free. The union inherits the worst clean-span proposal rate of its
members, because a site either member touches is a site the union touches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import build_generator, register_generator

__all__ = ["CompositeGenerator"]


@register_generator("composite")
class CompositeGenerator(BaseGenerator):
    """Run each member in order and merge their proposals, first occurrence winning."""

    generator_id = "composite"
    version = "1"

    def __init__(
        self,
        generator_id: str = "composite",
        members: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.generator_id = generator_id
        self._members = [
            build_generator(
                str(member["kind"]),
                generator_id=f"{generator_id}:{member.get('id', member['kind'])}",
                **dict(member.get("params", {})),
            )
            for member in members
        ]
        if not self._members:
            msg = "a composite generator needs at least one member"
            raise ValueError(msg)

    def available(self) -> bool:
        return all(member.available() for member in self._members)

    @property
    def lexicon_size(self) -> int:
        sizes = [
            int(size)
            for member in self._members
            if (size := getattr(member, "lexicon_size", None)) is not None
        ]
        return max(sizes, default=0)

    def fit(self, corpus: Sequence[str]) -> None:
        for member in self._members:
            member.fit(corpus)

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        merged: list[CandidateProposal] = []
        seen: set[str] = set()
        for member in self._members:
            # Each member gets the FULL budget, and the merge is truncated afterwards.
            # Splitting the budget between members would make the union's per-member reach
            # depend on how many members there are, which is not a property of either.
            for proposal in member.propose(context, max_candidates):
                if proposal.text in seen:
                    continue
                seen.add(proposal.text)
                metadata = {
                    **proposal.metadata,
                    "member_generator_id": member.generator_id,
                    "member_generator_version": member.version,
                }
                merged.append(replace(proposal, metadata=metadata))
        return merged[:max_candidates]

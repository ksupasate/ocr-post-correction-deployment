"""Alignment coverage statistics.

These are not diagnostics in the "nice to have" sense. Per-engine ambiguity rate is a
**confound for the entire cross-engine comparison**: if one engine's output is harder to
align, it yields fewer evaluated sites and a differently-filtered population, which can
move risk and coverage on its own. So the rate is reported next to the results it could
explain, not buried in a log.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.enums import AlignmentRelation, AlignmentStatus

__all__ = ["AlignmentStats", "summarize"]


@dataclass(slots=True)
class AlignmentStats:
    """Coverage and quality of alignment for one engine."""

    engine_id: str
    n_components: int = 0
    n_documents: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    by_relation: dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0
    mean_char_agreement: float = 0.0
    geometry_available: bool = False

    @property
    def resolved_rate(self) -> float:
        return self.by_status.get(AlignmentStatus.RESOLVED.value, 0) / max(self.n_components, 1)

    @property
    def ambiguity_rate(self) -> float:
        ambiguous = self.by_status.get(AlignmentStatus.AMBIGUOUS.value, 0)
        unresolved = self.by_status.get(AlignmentStatus.UNRESOLVED.value, 0)
        return (ambiguous + unresolved) / max(self.n_components, 1)

    @property
    def segmentation_rate(self) -> float:
        """Share of components where the engine split or merged tokens."""
        splits = self.by_relation.get(AlignmentRelation.SPLIT.value, 0)
        merges = self.by_relation.get(AlignmentRelation.MERGE.value, 0)
        many = self.by_relation.get(AlignmentRelation.MANY_TO_MANY.value, 0)
        return (splits + merges + many) / max(self.n_components, 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "engine_id": self.engine_id,
            "n_components": self.n_components,
            "n_documents": self.n_documents,
            "by_status": dict(sorted(self.by_status.items())),
            "by_relation": dict(sorted(self.by_relation.items())),
            "mean_confidence": round(self.mean_confidence, 6),
            "mean_char_agreement": round(self.mean_char_agreement, 6),
            "resolved_rate": round(self.resolved_rate, 6),
            "ambiguity_rate": round(self.ambiguity_rate, 6),
            "segmentation_rate": round(self.segmentation_rate, 6),
            "geometry_available": self.geometry_available,
        }


def summarize(records: Sequence[AlignmentRecord]) -> dict[str, AlignmentStats]:
    """Per-engine alignment statistics over a set of records."""
    by_engine: dict[str, list[AlignmentRecord]] = {}
    for record in records:
        by_engine.setdefault(record.engine_id, []).append(record)

    stats: dict[str, AlignmentStats] = {}
    for engine_id, engine_records in sorted(by_engine.items()):
        n = len(engine_records)
        stats[engine_id] = AlignmentStats(
            engine_id=engine_id,
            n_components=n,
            n_documents=len({r.document_id for r in engine_records}),
            by_status=dict(Counter(r.status.value for r in engine_records)),
            by_relation=dict(Counter(r.relation.value for r in engine_records)),
            mean_confidence=sum(r.align_confidence for r in engine_records) / n if n else 0.0,
            mean_char_agreement=sum(r.char_agreement for r in engine_records) / n if n else 0.0,
            geometry_available=any(r.geom_score is not None for r in engine_records),
        )
    return stats

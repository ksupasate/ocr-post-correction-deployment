"""The dataset adapter contract.

A dataset adapter turns a corpus on disk into :class:`DocumentBundle` objects. It must not
know anything about OCR engines, alignment, or the experiment protocol — adding a corpus
should never require touching the scientific code.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ocr_risk.schemas.documents import DocumentBundle
from ocr_risk.schemas.enums import RedistributionPolicy

__all__ = ["DatasetAdapter", "LicenseSpec", "PreflightReport"]


@dataclass(frozen=True, slots=True)
class LicenseSpec:
    """A dataset's licensing facts, recorded rather than assumed.

    ``redistribution`` gates the license audit. ``UNCLEAR`` is treated as prohibited, but
    stored distinctly: "we could not determine the terms" is a different statement from
    "the terms forbid it", and conflating them would misrepresent the corpus.
    """

    license_id: str
    name: str
    url: str
    redistribution: RedistributionPolicy
    attribution_required: bool
    notes: str = ""

    @property
    def may_redistribute(self) -> bool:
        return self.redistribution is RedistributionPolicy.ALLOWED


@dataclass(slots=True)
class PreflightReport:
    """Whether a dataset is present, verified, and usable."""

    dataset_id: str
    available: bool
    n_documents: int = 0
    root: Path | None = None
    checksums_verified: bool = False
    problems: list[str] = field(default_factory=list)
    remediation: str = ""

    def raise_if_unavailable(self) -> None:
        if not self.available:
            detail = "; ".join(self.problems) or "dataset not found"
            hint = f"\n  remediation: {self.remediation}" if self.remediation else ""
            msg = f"dataset {self.dataset_id!r} is not usable: {detail}{hint}"
            raise FileNotFoundError(msg)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Contract every corpus adapter satisfies."""

    dataset_id: str
    license: LicenseSpec
    gt_policy: str
    """Identifier of the transcription convention this corpus follows. Corpora with
    different policies must not be merged into one benchmark track without saying so."""

    def preflight(self) -> PreflightReport:
        """Check presence and integrity without loading the corpus."""
        ...

    def documents(self, limit: int | None = None) -> Iterator[DocumentBundle]:
        """Yield documents in a deterministic order.

        Order must be stable across runs and machines: the split partition is derived from
        document identity, but a wobbling iteration order would still make artifacts
        byte-unstable for no reason.
        """
        ...

"""The seam where a learned multimodal verifier will attach.

Declared, registered, and raising. The shallow feature verifier proves the architecture
end to end on CPU; this is where a trained encoder replaces
:func:`~ocr_risk.verify.featurizers.image_block` and the text features, with nothing else
in the pipeline changing — the mask, the split plan, the calibrator, the risk controller,
and every metric stay exactly as they are.

It raises rather than returning constant scores. A stub that silently scored 0.5 would
produce a complete, plausible-looking risk-coverage curve for a model that does not exist,
which is precisely the kind of result this repository's integrity rules exist to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence

from ocr_risk.evidence.fields import get_evidence_config
from ocr_risk.verify.base import ScoredBatch, VerificationInput
from ocr_risk.verify.registry import register_verifier

__all__ = ["TorchVerifier"]


@register_verifier("torch_multimodal")
class TorchVerifier:
    """Placeholder for the learned source-grounded verifier."""

    def __init__(
        self,
        verifier_id: str = "torch_multimodal",
        evidence_config: str = "v6",
        checkpoint: str = "",
        **_: object,
    ) -> None:
        self.verifier_id = verifier_id
        self.evidence_config = evidence_config
        self.required_evidence = get_evidence_config(evidence_config).fields
        self.checkpoint = checkpoint

    def _unavailable(self) -> None:
        msg = (
            f"verifier {self.verifier_id!r} is not implemented in this bootstrap. Training "
            "the learned verifier is the next milestone, deliberately excluded here so the "
            "infrastructure could be validated first. Returning constant scores would "
            "produce a complete-looking risk-coverage curve for a model that does not "
            "exist. Install with: uv sync --extra torch."
        )
        raise NotImplementedError(msg)

    def fit(self, inputs: Sequence[VerificationInput], harmful: Sequence[bool]) -> None:
        self._unavailable()

    def score(self, inputs: Sequence[VerificationInput]) -> ScoredBatch:
        self._unavailable()
        raise AssertionError("unreachable")  # pragma: no cover

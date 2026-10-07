"""OCR-only site discovery (CGV3): where might a repair be needed, without the answer?

The package exists to close R-65. Its input type carries no ground truth, its fitted
resources are fold-scoped lexical statistics passed in by the caller, and the layering
test forbids imports of ``align``, ``edits``, and ``datasets`` -- the layers that carry
ground truth -- so the GT-informed topology of earlier phases cannot reach discovery
even indirectly.
"""

from ocr_risk.discovery.enumerator import (
    DiscoveredSite,
    DiscoveryResources,
    DiscoveryRules,
    enumerate_sites,
)
from ocr_risk.discovery.freshness import (
    DevelopmentDocumentError,
    assert_development_documents,
    confirmatory_document_ids,
)
from ocr_risk.discovery.views import OcrPageView, OcrTokenView

__all__ = [
    "DevelopmentDocumentError",
    "DiscoveredSite",
    "DiscoveryResources",
    "DiscoveryRules",
    "OcrPageView",
    "OcrTokenView",
    "assert_development_documents",
    "confirmatory_document_ids",
    "enumerate_sites",
]

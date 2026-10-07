"""Document-level cluster bootstrap, paired comparisons, effect sizes, multiplicity.

Layer 8. The assumptions behind every procedure here are documented in
``docs/statistics.md``; the one that matters most is that the resampling unit is the
document, never the individual edit.
"""

from __future__ import annotations

from ocr_risk.stats.bootstrap import (
    BootstrapResult,
    cluster_bootstrap,
    cluster_bootstrap_indices,
    group_by_cluster,
    paired_cluster_bootstrap,
    paired_cluster_bootstrap_multi,
)
from ocr_risk.stats.multiplicity import (
    AdjustedTest,
    benjamini_hochberg,
    cliffs_delta,
    cohens_h,
    holm_bonferroni,
    minimum_detectable_effect,
)

__all__ = [
    "AdjustedTest",
    "BootstrapResult",
    "benjamini_hochberg",
    "cliffs_delta",
    "cluster_bootstrap",
    "cluster_bootstrap_indices",
    "cohens_h",
    "group_by_cluster",
    "holm_bonferroni",
    "minimum_detectable_effect",
    "paired_cluster_bootstrap",
    "paired_cluster_bootstrap_multi",
]

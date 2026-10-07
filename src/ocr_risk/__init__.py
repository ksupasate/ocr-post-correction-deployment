"""Source-grounded risk control for OCR post-correction under cross-engine shift.

The research object of this package is a *proposed edit* ``O -> Y``, not a transcription.
Layering (never import upward)::

    schemas -> io/config/provenance -> datasets/engines -> canonical -> align -> edits
            -> candidates/evidence -> verify/calibrate/risk/splits -> metrics/stats
            -> experiments/analysis/cli

Enforced by ``tests/architecture/test_layering.py``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

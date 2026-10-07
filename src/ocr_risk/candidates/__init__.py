"""Pluggable candidate generators. A baseline subsystem, not the core mechanism.

Layer 6. Generation and verification are kept separable so the same verifier can be
evaluated across generators, and so a new generator never requires a change to
verification or evaluation code.
"""

from __future__ import annotations

from ocr_risk.candidates import byt5 as _byt5  # noqa: F401  (registration)
from ocr_risk.candidates import composite as _composite  # noqa: F401
from ocr_risk.candidates import detector as _detector  # noqa: F401
from ocr_risk.candidates import edit_aware as _edit_aware  # noqa: F401
from ocr_risk.candidates import hard_negative as _hard_negative  # noqa: F401
from ocr_risk.candidates import identity as _identity  # noqa: F401
from ocr_risk.candidates import lexical as _lexical  # noqa: F401
from ocr_risk.candidates import structural as _structural  # noqa: F401
from ocr_risk.candidates import structural_v2 as _structural_v2  # noqa: F401
from ocr_risk.candidates import stubs as _stubs  # noqa: F401
from ocr_risk.candidates.base import (
    BaseGenerator,
    CandidateGenerator,
    CandidateProposal,
    GenerationContext,
)
from ocr_risk.candidates.byt5 import DEFAULT_BYT5, Byt5Generator, Byt5Spec
from ocr_risk.candidates.composite import CompositeGenerator
from ocr_risk.candidates.detector import ErrorGatedGenerator, ErrorSignals, error_signals
from ocr_risk.candidates.edit_aware import EditAwareGenerator
from ocr_risk.candidates.hard_negative import HARD_NEGATIVE_FAMILIES, HardNegativeGenerator
from ocr_risk.candidates.identity import IdentityGenerator
from ocr_risk.candidates.lexical import LexicalGenerator
from ocr_risk.candidates.registry import available_generators, build_generator, register_generator

__all__ = [
    "DEFAULT_BYT5",
    "HARD_NEGATIVE_FAMILIES",
    "BaseGenerator",
    "Byt5Generator",
    "Byt5Spec",
    "CandidateGenerator",
    "CandidateProposal",
    "CompositeGenerator",
    "EditAwareGenerator",
    "ErrorGatedGenerator",
    "ErrorSignals",
    "GenerationContext",
    "HardNegativeGenerator",
    "IdentityGenerator",
    "LexicalGenerator",
    "available_generators",
    "build_generator",
    "error_signals",
    "register_generator",
]

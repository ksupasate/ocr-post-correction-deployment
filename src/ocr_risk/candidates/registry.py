"""Candidate generator registry.

Adding a generator means writing a class and registering it. Verification and evaluation
code never changes — which is the property that lets the same verifier be compared across
generators.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ocr_risk.candidates.base import CandidateGenerator

__all__ = ["available_generators", "build_generator", "register_generator"]

_REGISTRY: dict[str, Callable[..., CandidateGenerator]] = {}


def register_generator(
    kind: str,
) -> Callable[[Callable[..., CandidateGenerator]], Callable[..., CandidateGenerator]]:
    def decorator(factory: Callable[..., CandidateGenerator]) -> Callable[..., CandidateGenerator]:
        if kind in _REGISTRY:
            msg = f"candidate generator {kind!r} is already registered"
            raise ValueError(msg)
        _REGISTRY[kind] = factory
        return factory

    return decorator


def build_generator(kind: str, generator_id: str, **params: Any) -> CandidateGenerator:
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        msg = f"unknown candidate generator {kind!r}; registered: {known}"
        raise KeyError(msg) from None
    return factory(generator_id=generator_id, **params)


def available_generators() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))

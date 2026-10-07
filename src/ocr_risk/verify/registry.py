"""Verifier registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["available_verifiers", "build_verifier", "register_verifier"]

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_verifier(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if kind in _REGISTRY:
            msg = f"verifier {kind!r} is already registered"
            raise ValueError(msg)
        _REGISTRY[kind] = factory
        return factory

    return decorator


def build_verifier(kind: str, verifier_id: str, evidence_config: str, **params: Any) -> Any:
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        msg = f"unknown verifier {kind!r}; registered: {known}"
        raise KeyError(msg) from None
    return factory(verifier_id=verifier_id, evidence_config=evidence_config, **params)


def available_verifiers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))

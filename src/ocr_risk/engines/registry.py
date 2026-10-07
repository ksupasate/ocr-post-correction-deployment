"""OCR engine adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ocr_risk.engines.base import Availability, OCREngineAdapter

__all__ = ["available_engines", "build_engine", "engine_availability", "register_engine"]

_REGISTRY: dict[str, Callable[..., OCREngineAdapter]] = {}


def register_engine(
    adapter_name: str,
) -> Callable[[Callable[..., OCREngineAdapter]], Callable[..., OCREngineAdapter]]:
    """Register an engine adapter factory under ``adapter_name``."""

    def decorator(factory: Callable[..., OCREngineAdapter]) -> Callable[..., OCREngineAdapter]:
        if adapter_name in _REGISTRY:
            msg = f"engine adapter {adapter_name!r} is already registered"
            raise ValueError(msg)
        _REGISTRY[adapter_name] = factory
        return factory

    return decorator


def build_engine(adapter_name: str, engine_id: str, **params: Any) -> OCREngineAdapter:
    try:
        factory = _REGISTRY[adapter_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        msg = f"unknown engine adapter {adapter_name!r}; registered: {known}"
        raise KeyError(msg) from None
    return factory(engine_id=engine_id, **params)


def available_engines() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def engine_availability() -> dict[str, Availability]:
    """Probe every registered adapter.

    Heavy backends import lazily inside their own ``availability()``, so this is safe to
    call with none of them installed.
    """
    report: dict[str, Availability] = {}
    for name, factory in sorted(_REGISTRY.items()):
        try:
            report[name] = factory(engine_id=name).availability()
        except Exception as exc:
            # take down the whole availability probe; report it instead.
            report[name] = Availability(
                engine_id=name,
                installed=False,
                detail=f"adapter failed to construct: {exc}",
                remediation=f"install the optional extra for {name}",
            )
    return report

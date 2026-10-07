"""Dataset adapter registry.

Adding a corpus means writing an adapter and registering it here. Nothing in the
experiment, alignment, or evaluation code changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ocr_risk.datasets.base import DatasetAdapter

__all__ = ["available_datasets", "build_dataset", "register_dataset"]

_REGISTRY: dict[str, Callable[..., DatasetAdapter]] = {}


def register_dataset(
    dataset_id: str,
) -> Callable[[Callable[..., DatasetAdapter]], Callable[..., DatasetAdapter]]:
    """Register a dataset adapter factory under ``dataset_id``."""

    def decorator(factory: Callable[..., DatasetAdapter]) -> Callable[..., DatasetAdapter]:
        if dataset_id in _REGISTRY:
            msg = f"dataset {dataset_id!r} is already registered"
            raise ValueError(msg)
        _REGISTRY[dataset_id] = factory
        return factory

    return decorator


def build_dataset(dataset_id: str, **params: Any) -> DatasetAdapter:
    try:
        factory = _REGISTRY[dataset_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        msg = f"unknown dataset {dataset_id!r}; registered: {known}"
        raise KeyError(msg) from None
    return factory(**params)


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))

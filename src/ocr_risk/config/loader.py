"""YAML composition, CLI overrides, and canonical config hashing.

A deliberately small include-DAG resolver rather than Hydra (ADR-004). What this buys is
the thing reproducibility auditing actually needs: one fully resolved mapping, with no
global state or working-directory rewriting between it and the run, that hashes to a
stable value.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ocr_risk.config.models import ExperimentConfig
from ocr_risk.io.hashing import canonical_hash
from ocr_risk.io.paths import project_root

__all__ = [
    "ConfigCycleError",
    "ResolvedConfig",
    "apply_overrides",
    "config_hash",
    "deep_merge",
    "load_config",
    "load_mapping",
    "resolve_includes",
]

EXTENDS_KEY = "extends"


class ConfigCycleError(ValueError):
    """Raised when ``extends:`` forms a cycle."""


class ResolvedConfig:
    """A validated config together with the exact mapping and hash that produced it."""

    __slots__ = ("config", "mapping", "path", "sha256")

    def __init__(self, config: ExperimentConfig, mapping: dict[str, Any], path: Path) -> None:
        self.config = config
        self.mapping = mapping
        self.path = path
        self.sha256 = canonical_hash(mapping)

    def __repr__(self) -> str:
        return f"ResolvedConfig(name={self.config.name!r}, sha256={self.sha256[:8]})"


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``.

    Mappings merge key-wise; every other type (including lists) is replaced wholesale.
    Element-wise list merging would make it impossible to *remove* an entry in a child
    config, which is exactly what ablation configs need to do.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML file as a mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        raise TypeError(msg)
    return raw


def _config_dir() -> Path:
    return project_root() / "configs"


def _resolve_path(reference: str, relative_to: Path) -> Path:
    """Resolve an ``extends`` entry against the including file, then ``configs/``."""
    candidate = (relative_to.parent / reference).resolve()
    if candidate.is_file():
        return candidate
    fallback = (_config_dir() / reference).resolve()
    if fallback.is_file():
        return fallback
    msg = f"config {reference!r} (referenced from {relative_to}) not found"
    raise FileNotFoundError(msg)


def resolve_includes(path: Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Depth-first resolution of the ``extends`` DAG into one mapping.

    Parents are merged left to right, then the including file is merged on top, so a child
    always wins over what it extends.
    """
    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(p.name for p in (*_stack, path))
        msg = f"circular config inheritance: {chain}"
        raise ConfigCycleError(msg)

    mapping = load_mapping(path)
    parents = mapping.pop(EXTENDS_KEY, [])
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = _resolve_path(str(parent), path)
        merged = deep_merge(merged, resolve_includes(parent_path, (*_stack, path)))
    return deep_merge(merged, mapping)


def _coerce_scalar(text: str) -> Any:
    """Parse an override value using YAML rules, so types survive the command line."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(mapping: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides.

    Only existing branches may be extended with new leaves; a typo'd path that would
    create an entirely new top-level section is caught by the model's ``extra="forbid"``
    at validation, so a silent no-op override is not possible.
    """
    if not overrides:
        return mapping
    result = copy.deepcopy(mapping)
    for override in overrides:
        if "=" not in override:
            msg = f"malformed override {override!r}; expected key.path=value"
            raise ValueError(msg)
        dotted, _, raw_value = override.partition("=")
        keys = dotted.strip().split(".")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _coerce_scalar(raw_value.strip())
    return result


def config_hash(mapping: dict[str, Any]) -> str:
    return canonical_hash(mapping)


def load_config(path: Path | str, overrides: list[str] | None = None) -> ResolvedConfig:
    """Resolve, override, and validate a config file into a typed experiment."""
    resolved_path = Path(path)
    if not resolved_path.is_file():
        resolved_path = (_config_dir() / str(path)).resolve()
    mapping = apply_overrides(resolve_includes(resolved_path), overrides)
    config = ExperimentConfig.model_validate(mapping)
    # Re-dump through the model so the hashed mapping includes every default, making the
    # hash a function of the effective settings rather than of what was typed.
    effective = config.model_dump(mode="json")
    return ResolvedConfig(config=config, mapping=effective, path=resolved_path)

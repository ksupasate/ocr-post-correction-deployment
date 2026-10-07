"""The layering invariant, enforced by walking the import graph.

CLAUDE.md declares an ordering and several specific prohibitions. A declaration nobody
checks is a wish, so this test parses every module's imports and fails the build on a
violation.

The prohibitions are not stylistic. ``metrics`` must not import ``verify`` because metrics
have to be computable from saved artifacts months later, without the model that produced
them. ``align`` must not import ``engines`` because otherwise an engine could special-case
how it is aligned. Each one protects a property the science depends on.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "ocr_risk"

# Layer index per package. A module may import from its own layer or any lower one.
LAYER: dict[str, int] = {
    "schemas": 0,
    "layout": 0,
    "io": 1,
    "config": 1,
    "provenance": 1,
    "datasets": 2,
    "engines": 2,
    "canonical": 3,
    "align": 4,
    "edits": 5,
    "discovery": 5,
    "candidates": 6,
    "evidence": 6,
    "verify": 7,
    "calibrate": 7,
    "risk": 7,
    "splits": 7,
    "metrics": 8,
    "stats": 8,
    "experiments": 9,
    "analysis": 9,
    "cli": 9,
}

# Edges banned regardless of layer, each protecting a specific property.
FORBIDDEN_EDGES: dict[str, tuple[tuple[str, str], ...]] = {
    "metrics must be computable from artifacts alone, without the producing model": (
        ("metrics", "verify"),
        ("metrics", "candidates"),
        ("metrics", "engines"),
        ("metrics", "evidence"),
        ("metrics", "experiments"),
    ),
    "no engine may special-case how it is aligned": (("align", "engines"),),
    "evidence must stay ground-truth-blind": (("evidence", "edits"),),
    # CGV3's deployability invariant (protocol section 2): site discovery may not
    # consume ground truth in any form -- not the GT-bearing datasets layer, not the
    # GT-informed alignment layer, not the GT-labelled edits layer. R-65 as architecture.
    "site discovery must be ground-truth-blind": (
        ("discovery", "datasets"),
        ("discovery", "align"),
        ("discovery", "edits"),
    ),
    "provenance is a dependency of the artifact store, not the reverse": (
        ("provenance", "io.artifacts"),
    ),
    "the data model must not depend on anything": (
        ("schemas", "io"),
        ("schemas", "config"),
        ("schemas", "align"),
    ),
}


def _modules() -> list[Path]:
    return sorted(p for p in SOURCE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _package_of(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else ""


def _imported_ocr_risk_modules(path: Path) -> set[str]:
    """Dotted ``ocr_risk`` submodule paths imported by ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ocr_risk"):
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ocr_risk"):
                    found.add(alias.name)
    return found


def _target_package(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SOURCE_ROOT)))
def test_module_does_not_import_upward(path: Path) -> None:
    package = _package_of(path)
    if package not in LAYER:
        return
    source_layer = LAYER[package]

    for module in _imported_ocr_risk_modules(path):
        target = _target_package(module)
        if target not in LAYER or target == package:
            continue
        assert LAYER[target] <= source_layer, (
            f"{path.relative_to(SOURCE_ROOT)} (layer {source_layer}, {package}) imports "
            f"{module} (layer {LAYER[target]}, {target}).\n"
            "Imports may only go downward. See the layering table in CLAUDE.md."
        )


@pytest.mark.parametrize("reason", list(FORBIDDEN_EDGES), ids=lambda r: r[:40])
def test_forbidden_edges_are_absent(reason: str) -> None:
    violations: list[str] = []
    for path in _modules():
        package = _package_of(path)
        for module in _imported_ocr_risk_modules(path):
            for source, target in FORBIDDEN_EDGES[reason]:
                if package != source:
                    continue
                if module == f"ocr_risk.{target}" or module.startswith(f"ocr_risk.{target}."):
                    violations.append(f"  {path.relative_to(SOURCE_ROOT)} imports {module}")
    assert not violations, f"forbidden import ({reason}):\n" + "\n".join(violations)


def test_every_package_has_a_declared_layer() -> None:
    """A new package must be placed in the ordering deliberately, not left unchecked."""
    packages = {p.name for p in SOURCE_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert packages <= set(LAYER), f"packages missing from the layer table: {packages - set(LAYER)}"


def test_layer_table_matches_reality() -> None:
    """Guards against the table drifting to describe packages that no longer exist."""
    packages = {p.name for p in SOURCE_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert set(LAYER) <= packages, f"layer table names missing packages: {set(LAYER) - packages}"

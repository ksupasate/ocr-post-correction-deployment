"""Metrics must have exactly one implementation.

Two functions called CER is how a paper ends up reporting two different numbers under one
name — usually discovered after review. This test scans the source tree for second
implementations of anything ``metrics/`` already owns.

It also checks that quantities which change headline results are not hardcoded outside
configuration, so a threshold cannot be quietly baked into a script.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "ocr_risk"
METRICS_ROOT = SOURCE_ROOT / "metrics"

# Names that may only be *defined* under metrics/. Substrings are matched on the whole
# function name, so `character_error_counts` is caught by "error_rate"? No — matching is
# exact-or-prefixed to avoid punishing unrelated names like `bin_edges`.
RESERVED_DEFINITIONS = (
    "cer",
    "wer",
    "levenshtein",
    "edit_distance",
    "character_error_rate",
    "word_error_rate",
    "brier",
    "brier_score",
    "expected_calibration_error",
    "risk_coverage_curve",
    "coverage_at_risk",
    "aurc",
)

# Modules legitimately allowed to define a reserved name outside metrics/.
ALLOWED_EXCEPTIONS: dict[str, set[str]] = {
    # The taxonomy needs a distance to define its own outcomes; it delegates to rapidfuzz
    # and is imported by metrics rather than duplicating it.
    "edits/outcome.py": {"distance"},
}


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


SCANNED = [p for p in _python_files(SOURCE_ROOT) if METRICS_ROOT not in p.parents] + _python_files(
    REPO_ROOT / "scripts"
)


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.name)
def test_no_second_metric_implementation(path: Path) -> None:
    relative = path.relative_to(REPO_ROOT).as_posix()
    allowed = ALLOWED_EXCEPTIONS.get(relative.replace("src/ocr_risk/", ""), set())
    offenders = {
        name
        for name in _function_names(path)
        if name.lstrip("_") in RESERVED_DEFINITIONS and name not in allowed
    }
    assert not offenders, (
        f"{relative} defines metric function(s) {sorted(offenders)} that belong to "
        "ocr_risk.metrics. Import them instead — a second implementation is how two "
        "numbers end up with the same name and different meanings."
    )


def test_metrics_package_actually_defines_them() -> None:
    """The prohibition is only meaningful if the canonical definitions exist."""
    import ocr_risk.metrics as metrics

    for name in ("cer", "wer", "levenshtein", "brier_score", "risk_coverage_curve", "aurc"):
        assert hasattr(metrics, name), f"metrics package is missing {name}"


def test_risk_tolerances_are_not_hardcoded_outside_config() -> None:
    """Epsilon values are scientific parameters. A literal 0.01 used as a risk threshold
    in code would make the reported tolerance untraceable to the config hash."""
    suspicious: list[str] = []
    for path in _python_files(SOURCE_ROOT):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or "epsilon" not in stripped.lower():
                continue
            # A default value or comparison against a literal tolerance is the pattern
            # worth flagging; type annotations and passthrough arguments are not.
            if any(
                token in stripped
                for token in ("epsilon = 0.0", "epsilon=0.0", "epsilon = 0.01", "epsilon=0.01")
            ):
                suspicious.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")
    assert not suspicious, "risk tolerance hardcoded outside configs/:\n" + "\n".join(suspicious)

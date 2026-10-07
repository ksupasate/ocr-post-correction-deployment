"""Guard tests for the ``cgv3`` command surface: refuse loudly, never silently.

The two refusal paths are the ones the pre-confirmatory review made fail-closed: the
spent evaluate role (R-66) and missing prerequisite runs. Both must reject before any
document is read.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from ocr_risk.cli.cmd_cgv3 import app

runner = CliRunner()


def _panel_text(output: str) -> str:
    """The error panel as plain words: rich wraps to the terminal width and interleaves
    box-drawing characters between the wrapped words, so strip everything but words."""
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", output).split())


def test_the_spent_evaluate_role_is_refused_before_anything_runs() -> None:
    result = runner.invoke(app, ["experiments/smoke_synthetic.yaml", "--role", "evaluate"])
    assert result.exit_code != 0
    assert "the old evaluate role is spent" in _panel_text(result.output)


def test_an_unknown_role_is_refused() -> None:
    result = runner.invoke(app, ["experiments/smoke_synthetic.yaml", "--role", "test"])
    assert result.exit_code != 0


def test_missing_prerequisite_runs_are_refused(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from ocr_risk.io.artifacts import ArtifactStore

    monkeypatch.setattr(ArtifactStore, "latest", lambda self, stage, name: None)
    result = runner.invoke(app, ["experiments/smoke_synthetic.yaml"])
    assert result.exit_code != 0
    assert "is missing runs" in _panel_text(result.output)

"""Package-level smoke checks: the distribution installs and the CLI is reachable."""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

import ocr_risk
from ocr_risk.cli.app import app

SUBPACKAGES = [
    "schemas",
    "io",
    "config",
    "provenance",
    "datasets",
    "engines",
    "canonical",
    "align",
    "edits",
    "candidates",
    "evidence",
    "verify",
    "calibrate",
    "risk",
    "splits",
    "metrics",
    "stats",
    "experiments",
    "analysis",
    "cli",
]


def test_version_is_exposed() -> None:
    assert ocr_risk.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(f"ocr_risk.{name}") is not None


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ocr-risk" in result.output


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert ocr_risk.__version__ in result.output

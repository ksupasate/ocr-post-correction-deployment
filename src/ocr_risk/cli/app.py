"""Typer application root. One subcommand group per pipeline stage."""

from __future__ import annotations

import typer
from rich.console import Console

from ocr_risk import __version__
from ocr_risk.cli import (
    cmd_align,
    cmd_analyze,
    cmd_audit,
    cmd_candidates,
    cmd_cgv2,
    cmd_cgv3,
    cmd_config,
    cmd_data,
    cmd_experiment,
    cmd_ocr,
    cmd_provenance,
    cmd_sites,
)

app = typer.Typer(
    name="ocr-risk",
    help=(
        "Source-grounded risk control for OCR post-correction under cross-engine "
        "distribution shift. Research object: a proposed edit O -> Y."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


# An explicit callback keeps `app` in multi-command mode; without it Typer collapses a
# single-command application into that command.
@app.callback()
def _root() -> None:
    """Research CLI. Every pipeline stage writes an immutable, provenance-tracked run."""


app.add_typer(cmd_align.app)
app.add_typer(cmd_analyze.app)
app.add_typer(cmd_analyze.gate_app)
app.add_typer(cmd_audit.app)
app.add_typer(cmd_candidates.app)
app.add_typer(cmd_cgv2.app)
app.add_typer(cmd_cgv3.app)
app.add_typer(cmd_config.app)
app.add_typer(cmd_data.app)
app.add_typer(cmd_experiment.app)
app.add_typer(cmd_ocr.app)
app.add_typer(cmd_provenance.app)
app.add_typer(cmd_sites.app)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()

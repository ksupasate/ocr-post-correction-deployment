"""``ocr-risk audit`` — leakage, licensing, and documentation integrity."""

from __future__ import annotations

import re
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ocr_risk.datasets import available_datasets, build_dataset
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.paths import project_root
from ocr_risk.schemas.enums import RedistributionPolicy
from ocr_risk.splits import audit_records

app = typer.Typer(
    name="audit", help="Leakage, licence, and integrity audits.", no_args_is_help=True
)
console = Console()

# Phrases the project has committed not to use. See docs/prior_art_boundary.md.
_FORBIDDEN_CLAIMS = (
    # "the first" only counts as a novelty claim in a research context. A bare
    # \bthe first\b also matches "the first crossing" and "the first fold", and a check
    # that cries wolf is a check people learn to silence.
    r"\bthe first (?:\w+ )?(?:to\b|method|system|approach|work|study|paper|framework|"
    r"attempt|benchmark|dataset|model)\b",
    r"\bfirst to (?:show|demonstrate|propose|introduce|formulate|report|study)\b",
    r"\bnovel\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bSOTA\b",
    r"\bunprecedented\b",
    r"\bwe are the only\b",
    r"\bbreakthrough\b",
)
# Where the prohibition is *described* rather than violated.
_ALLOWED_FILES = {
    "docs/prior_art_boundary.md",
    "docs/research_integrity.md",
    ".claude/rules/research-integrity.md",
    ".claude/skills/research-integrity-check/SKILL.md",
    ".claude/agents/research-integrity-reviewer.md",
    "CLAUDE.md",
    "AGENTS.md",
    "README.md",
    "src/ocr_risk/cli/cmd_audit.py",
}


@app.command("leakage")
def leakage(
    experiment: Annotated[str | None, typer.Option("--experiment", "-e")] = None,
) -> None:
    """Re-derive the split scopes from finished run records.

    Independent of the runtime guard on purpose: a bug in the guard would pass the guard's
    own tests, and this reads only what the runs actually committed to.
    """
    store = ArtifactStore()
    records = [r for r in store.iter_records() if experiment is None or r.experiment == experiment]
    audit = audit_records(records)

    if audit.folds_checked == 0:
        console.print("[yellow]no runs with a split descriptor were found[/yellow]")
        return

    console.print(
        f"audited {audit.folds_checked} fold(s); "
        f"{len(audit.partition_hashes)} distinct document partition(s)"
    )
    for finding in audit.findings:
        colour = "red" if finding.severity == "HIGH" else "yellow"
        console.print(f"  [{colour}]{finding}[/{colour}]")

    if audit.passed:
        console.print("[green]leakage audit passed[/green]")
    else:
        console.print("[red]leakage audit FAILED[/red]")
        raise typer.Exit(code=1)


@app.command("licenses")
def licenses() -> None:
    """Confirm no non-redistributable corpus has files tracked in the repository."""
    table = Table("dataset", "license", "redistribution", "tracked files", "status")
    failures = 0

    for dataset_id in available_datasets():
        adapter = build_dataset(dataset_id)
        spec = adapter.license
        # Only files under the repo count; downloaded corpora live under data/raw, which
        # is gitignored and therefore not redistributed by this repository.
        tracked = sorted((project_root() / "data" / "committed" / dataset_id).glob("**/*"))
        offending = [p for p in tracked if p.is_file()]
        ok = spec.may_redistribute or not offending
        failures += 0 if ok else 1
        table.add_row(
            dataset_id,
            spec.license_id,
            spec.redistribution.value,
            str(len(offending)),
            "[green]ok[/green]" if ok else "[red]VIOLATION[/red]",
        )
        if spec.redistribution is RedistributionPolicy.UNCLEAR:
            console.print(
                f"[yellow]{dataset_id}: redistribution terms unclear; treated as "
                "prohibited until determined[/yellow]"
            )

    console.print(table)
    registry = project_root() / "manifests" / "licenses" / "registry.yaml"
    if registry.is_file():
        entries = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
        console.print(f"  licence registry: {len(entries.get('datasets', {}))} entries")
    else:
        console.print("[yellow]  manifests/licenses/registry.yaml is missing[/yellow]")

    if failures:
        raise typer.Exit(code=1)
    console.print("[green]licence audit passed[/green]")


@app.command("docs")
def docs() -> None:
    """Scan documentation for prohibited novelty claims and broken internal links."""
    root = project_root()
    targets = [
        p
        for p in sorted(root.rglob("*.md"))
        if ".venv" not in p.parts
        and "node_modules" not in p.parts
        and p.relative_to(root).as_posix() not in _ALLOWED_FILES
    ]

    claim_findings: list[str] = []
    link_findings: list[str] = []

    for path in targets:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in _FORBIDDEN_CLAIMS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    claim_findings.append(f"  {relative}:{lineno}: {line.strip()[:90]}")

        for match in re.finditer(r"\[[^\]]+\]\(([^)#]+\.md)(#[^)]*)?\)", text):
            target = (path.parent / match.group(1)).resolve()
            if not target.is_file():
                link_findings.append(f"  {relative}: broken link -> {match.group(1)}")

    for finding in claim_findings:
        console.print(f"[red]novelty claim[/red] {finding}")
    for finding in link_findings:
        console.print(f"[yellow]{finding}[/yellow]")

    if claim_findings:
        console.print(
            "[red]docs audit FAILED: use 'candidate contribution' or 'research "
            "hypothesis' instead. See docs/prior_art_boundary.md[/red]"
        )
        raise typer.Exit(code=1)
    if link_findings:
        console.print("[red]docs audit FAILED: broken internal links[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]docs audit passed[/green] ({len(targets)} files scanned)")


@app.command("determinism")
def determinism(
    experiment: Annotated[str, typer.Option("--experiment", "-e")],
) -> None:
    """Compare the output digests of repeated runs of each stage, **like with like**.

    Timestamps are excluded from the digest by construction, so a difference *within a
    group* is real nondeterminism: unseeded RNG, set or dict iteration order, or thread
    scheduling.

    Runs are grouped by ``(config_sha256, code_sha256)``. Comparing across those was the
    original behaviour and it was useless in an active repository: any stage re-run after
    an edit came back NONDETERMINISTIC, so the check was red almost always and nobody could
    believe it. Two runs of different code producing different output is not a defect; it
    is the edit.
    """
    store = ArtifactStore()
    digests_by_group: dict[tuple[str, str, str], set[str]] = {}
    runs_by_group: dict[tuple[str, str, str], int] = {}
    for record in store.iter_records():
        if record.experiment != experiment or record.outputs_digest is None:
            continue
        key = (record.stage.value, record.config_sha256, record.code_sha256)
        digests_by_group.setdefault(key, set()).add(record.outputs_digest)
        runs_by_group[key] = runs_by_group.get(key, 0) + 1

    if not digests_by_group:
        console.print(f"[yellow]no runs found for {experiment!r}[/yellow]")
        raise typer.Exit(code=1)

    table = Table("stage", "config", "code", "runs", "distinct digests", "status")
    unstable = 0
    uncompared = 0
    for (stage, config_sha, code_sha), digests in sorted(digests_by_group.items()):
        n_runs = runs_by_group[(stage, config_sha, code_sha)]
        if n_runs < 2:
            # A single run compares nothing. Saying "deterministic" here would be a
            # vacuous pass, which is worse than reporting that the check did not run.
            uncompared += 1
            status = "[yellow]NOT COMPARED (1 run)[/yellow]"
        elif len(digests) == 1:
            status = "[green]deterministic[/green]"
        else:
            unstable += 1
            status = "[red]NONDETERMINISTIC[/red]"
        table.add_row(stage, config_sha[:10], code_sha[:10], str(n_runs), str(len(digests)), status)
    console.print(table)

    if unstable:
        console.print(
            f"[red]{unstable} stage/config/code group(s) produced different outputs across "
            "runs of the SAME code and config[/red]"
        )
        raise typer.Exit(code=1)
    if uncompared:
        console.print(
            f"[yellow]{uncompared} group(s) have only one run, so determinism was not "
            "actually tested. Run the experiment twice before trusting this.[/yellow]"
        )
        raise typer.Exit(code=1)
    console.print("[green]all repeated stages produced identical outputs[/green]")

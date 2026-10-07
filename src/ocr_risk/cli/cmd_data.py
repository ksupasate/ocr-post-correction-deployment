"""``ocr-risk data`` — corpus materialization and the document manifest stage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.config import load_config
from ocr_risk.datasets import available_datasets, build_dataset
from ocr_risk.datasets.synthetic import SyntheticDataset, noise_level_of
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.hashing import canonical_hash
from ocr_risk.provenance import capture_git_state
from ocr_risk.schemas.documents import GtToken, SourceDocument
from ocr_risk.schemas.enums import StageName
from ocr_risk.splits import duplicate_groups

app = typer.Typer(
    name="data", help="Acquire corpora and build document manifests.", no_args_is_help=True
)
console = Console()


@app.command("list")
def list_datasets() -> None:
    """Show registered dataset adapters and whether their data is present."""
    table = Table("dataset", "available", "documents", "license", "redistribution")
    for dataset_id in available_datasets():
        adapter = build_dataset(dataset_id)
        report = adapter.preflight()
        table.add_row(
            dataset_id,
            "yes" if report.available else "no",
            str(report.n_documents),
            adapter.license.license_id,
            adapter.license.redistribution.value,
        )
    console.print(table)


@app.command("synth")
def synth(
    n_documents: Annotated[int, typer.Option("--pages", help="Number of pages to render")] = 48,
    seed: Annotated[int, typer.Option("--seed")] = 20260817,
    force: Annotated[bool, typer.Option("--force", help="Re-render existing pages")] = False,
) -> None:
    """Render the synthetic corpus into the write-once raw layer.

    Deterministic: re-running writes byte-identical pages, which the raw store accepts as
    a no-op. That is the determinism check, not a convenience.
    """
    adapter = SyntheticDataset(n_documents=n_documents, seed=seed)
    written = adapter.materialize(force=force)
    report = adapter.preflight()
    console.print(
        f"[green]synthetic corpus ready[/green]  rendered={written}  "
        f"total={report.n_documents}  root={report.root}"
    )
    console.print("[yellow]SYNTHETIC — not a research corpus[/yellow]")


@app.command("manifest")
def manifest(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Build the ``documents`` and ``gt_tokens`` tables for every enabled dataset."""
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()

    documents: list[SourceDocument] = []
    gt_tokens: list[GtToken] = []
    noise_by_document: dict[str, str] = {}

    for dataset_cfg in cfg.datasets:
        if not dataset_cfg.enabled:
            continue
        adapter = build_dataset(dataset_cfg.id, **dataset_cfg.params)
        adapter.preflight().raise_if_unavailable()
        for bundle in adapter.documents(limit=dataset_cfg.max_documents):
            documents.append(bundle.document)
            gt_tokens.extend(bundle.gt_tokens)
            if dataset_cfg.id == "synthetic":
                noise_by_document[bundle.document.document_id] = noise_level_of(
                    bundle.document.document_id
                )

    duplicates = duplicate_groups(documents)

    with store.begin(
        stage=StageName.MANIFEST,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as run:
        run.write_records("documents", documents)
        run.write_records("gt_tokens", gt_tokens)
        run.write_json(
            "manifest_stats.json",
            {
                "n_documents": len(documents),
                "n_gt_tokens": len(gt_tokens),
                "datasets": sorted({d.dataset_id for d in documents}),
                "duplicate_groups": duplicates,
                "noise_levels": noise_by_document,
                "synthetic": cfg.synthetic,
            },
        )
        run.set_datasets(
            cfg.enabled_dataset_ids, canonical_hash([d.image_sha256 for d in documents])
        )
        run.mark_synthetic(cfg.synthetic)

    console.print(
        f"[green]manifest[/green] {len(documents)} documents, {len(gt_tokens)} GT tokens "
        f"-> {run.run_id}"
    )
    if duplicates:
        console.print(
            f"[yellow]{len(duplicates)} near-duplicate group(s) detected; they will be "
            "forced into the same split bucket[/yellow]"
        )

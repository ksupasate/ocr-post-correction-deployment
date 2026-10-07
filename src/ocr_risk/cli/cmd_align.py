"""``ocr-risk align`` — align canonical OCR spans to ground truth, and report coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.align import align_document, summarize
from ocr_risk.config import load_config
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.provenance import capture_git_state
from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.documents import GtToken
from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.spans import CanonicalSpan

app = typer.Typer(name="align", help="Align OCR spans to ground truth.", no_args_is_help=True)
console = Console()


@app.command("run")
def run(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Produce the ``alignments`` table for every (document, engine) pair."""
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()

    manifest_run = store.latest(StageName.MANIFEST, cfg.name)
    canonical_run = store.latest(StageName.CANONICALIZE, cfg.name)
    if manifest_run is None or canonical_run is None:
        msg = f"experiment {cfg.name!r} needs manifest and canonicalize runs first"
        raise typer.BadParameter(msg)

    documents = {d.document_id: d for d in store.read_records(manifest_run, "documents")}
    tokens_by_document: dict[str, list[GtToken]] = {}
    for token in store.read_records(manifest_run, "gt_tokens"):
        tokens_by_document.setdefault(token.document_id, []).append(token)

    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for span in store.read_records(canonical_run, "spans"):
        spans_by_pair.setdefault((span.document_id, span.engine_id), []).append(span)

    records: list[AlignmentRecord] = []
    n_unresolved = 0
    for (document_id, _engine_id), spans in sorted(spans_by_pair.items()):
        outcome = align_document(
            documents[document_id],
            spans,
            tokens_by_document.get(document_id, []),
            cfg.alignment,
        )
        records.extend(outcome.records)
        n_unresolved += outcome.n_unresolved_blocks

    stats = {engine: s.as_dict() for engine, s in summarize(records).items()}

    with store.begin(
        stage=StageName.ALIGN,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as run_writer:
        run_writer.inherit_from(canonical_run)
        run_writer.write_records("alignments", records)
        run_writer.write_json(
            "alignment_stats.json",
            {
                "per_engine": stats,
                "n_components": len(records),
                "n_unresolved_blocks": n_unresolved,
                "synthetic": cfg.synthetic,
            },
        )
        run_writer.set_datasets(cfg.enabled_dataset_ids)
        run_writer.mark_synthetic(cfg.synthetic)

    console.print(f"[green]aligned[/green] {len(records)} components -> {run_writer.run_id}")
    _print_stats(stats)


@app.command("stats")
def stats(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Show per-engine alignment coverage from the latest align run.

    Ambiguity rate is reported per engine on purpose: if it differs across engines it
    partially explains any cross-engine difference in risk or coverage, so it belongs
    beside those results rather than in a log.
    """
    resolved = load_config(config_path, overrides)
    store = ArtifactStore()
    align_run = store.latest(StageName.ALIGN, resolved.config.name)
    if align_run is None:
        raise typer.BadParameter(f"no align run for experiment {resolved.config.name!r}")
    payload = store.read_json(align_run, "alignment_stats.json")
    _print_stats(payload["per_engine"])
    if payload.get("n_unresolved_blocks"):
        console.print(
            f"[yellow]{payload['n_unresolved_blocks']} block(s) exceeded the DP limit and "
            "were marked UNRESOLVED rather than aligned approximately[/yellow]"
        )


def _print_stats(per_engine: dict[str, dict[str, Any]]) -> None:
    table = Table(
        "engine",
        "components",
        "resolved",
        "ambiguity",
        "segmentation",
        "char agree",
        "geometry",
        title="alignment coverage (ambiguity rate is a cross-engine confound)",
    )
    for engine, s in sorted(per_engine.items()):
        table.add_row(
            engine,
            str(s["n_components"]),
            f"{s['resolved_rate']:.3f}",
            f"{s['ambiguity_rate']:.3f}",
            f"{s['segmentation_rate']:.3f}",
            f"{s['mean_char_agreement']:.3f}",
            "yes" if s["geometry_available"] else "no",
        )
    console.print(table)

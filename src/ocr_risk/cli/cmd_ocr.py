"""``ocr-risk ocr`` — run engines and canonicalize their preserved output."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response
from ocr_risk.config import load_config
from ocr_risk.engines import PageInput, build_engine, engine_availability
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.paths import data_root
from ocr_risk.io.raw_store import RawStore
from ocr_risk.provenance import capture_git_state
from ocr_risk.schemas.documents import SourceDocument
from ocr_risk.schemas.enums import StageName

app = typer.Typer(name="ocr", help="Run OCR engines and canonicalize spans.", no_args_is_help=True)
console = Console()


@app.command("engines")
def engines() -> None:
    """Show which OCR backends are installed here."""
    table = Table("adapter", "installed", "version", "detail / remediation")
    for name, availability in engine_availability().items():
        table.add_row(
            name,
            "[green]yes[/green]" if availability.installed else "[yellow]no[/yellow]",
            availability.version or "-",
            availability.detail or availability.remediation or "",
        )
    console.print(table)


@app.command("run")
def run(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    engine: Annotated[
        str | None, typer.Option("--engine", help="Comma-separated engine ids")
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Recognize only the first N pages per dataset"),
    ] = None,
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Recognize every manifested document with the selected engines.

    Raw responses go to the write-once ``data/raw/ocr`` layer, keyed by engine fingerprint.
    A document already recognized under the same fingerprint is skipped, so re-running is
    cheap and never rewrites evidence.
    """
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()
    raw_store = RawStore()

    manifest_run = store.latest(StageName.MANIFEST, cfg.name)
    if manifest_run is None:
        msg = f"no manifest run for experiment {cfg.name!r}; run `ocr-risk data manifest` first"
        raise typer.BadParameter(msg)
    documents = store.read_records(manifest_run, "documents")

    wanted = set(engine.split(",")) if engine else set(cfg.enabled_engine_ids)
    selected = [e for e in cfg.engines if e.enabled and e.id in wanted]
    if not selected:
        raise typer.BadParameter(f"no enabled engines match {sorted(wanted)}")

    by_dataset: dict[str, list[SourceDocument]] = {}
    for document in documents:
        by_dataset.setdefault(document.dataset_id, []).append(document)

    for engine_cfg in selected:
        # One adapter per (engine, dataset): recognition language is a property of the
        # corpus, and reading German Fraktur with an English model produces
        # plausible-looking garbage that is worse than a failure.
        for dataset_id, dataset_documents in sorted(by_dataset.items()):
            params = engine_cfg.params_for(dataset_id)
            adapter = build_engine(engine_cfg.adapter, engine_id=engine_cfg.id, **params)
            adapter.availability().require()
            fingerprint = adapter.fingerprint()
            recognized = skipped = failed = 0
            elapsed = 0.0

            pages = dataset_documents if limit is None else dataset_documents[:limit]
            for document in pages:
                if raw_store.has_engine_response(
                    dataset_id, engine_cfg.id, fingerprint.fingerprint, document.document_id
                ):
                    skipped += 1
                    continue
                page = PageInput(
                    document_id=document.document_id,
                    dataset_id=document.dataset_id,
                    image_path=data_root() / document.image_path,
                    image_sha256=document.image_sha256,
                    width=document.width,
                    height=document.height,
                )
                started = time.perf_counter()
                try:
                    raw_store.write_engine_response(adapter.recognize(page))
                except Exception as error:
                    # One unreadable page must not abandon a corpus mid-matrix. The
                    # failure is counted and reported; a page missing from one engine
                    # breaks the matched-source premise, so it has to be visible.
                    failed += 1
                    console.print(
                        f"  [red]FAILED[/red] {engine_cfg.id}/{document.document_id}: "
                        f"{type(error).__name__}: {str(error)[:120]}"
                    )
                    continue
                elapsed += time.perf_counter() - started
                recognized += 1

            if recognized or skipped:
                raw_store.record_fingerprint(
                    dataset_id,
                    engine_cfg.id,
                    {"fingerprint": fingerprint.fingerprint, "params": params},
                )
            rate = f"{elapsed / recognized:.2f}s/page" if recognized else "-"
            console.print(
                f"[green]{engine_cfg.id}[/green] {dataset_id}: recognized={recognized} "
                f"cached={skipped} failed={failed} {rate} "
                f"fingerprint={fingerprint.fingerprint[:8]}"
            )
    return None


@app.command("canonicalize")
def canonicalize(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Derive canonical spans from preserved raw responses."""
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()
    raw_store = RawStore()

    manifest_run = store.latest(StageName.MANIFEST, cfg.name)
    if manifest_run is None:
        raise typer.BadParameter(f"no manifest run for experiment {cfg.name!r}")
    documents = store.read_records(manifest_run, "documents")

    policy = CanonicalizationPolicy(unicode_policy=cfg.alignment.unicode_policy)
    spans = []
    fingerprints: dict[str, str] = {}

    for engine_cfg in cfg.engines:
        if not engine_cfg.enabled:
            continue
        adapter = build_engine(engine_cfg.adapter, engine_id=engine_cfg.id, **engine_cfg.params)
        fingerprint = adapter.fingerprint()
        fingerprints[engine_cfg.id] = fingerprint.fingerprint
        scale_name = adapter.confidence_scale.name

        for document in documents:
            path = raw_store.engine_response_path(
                document.dataset_id, engine_cfg.id, fingerprint.fingerprint, document.document_id
            )
            if not path.exists():
                continue
            raw = raw_store.read_engine_response(path)
            spans.extend(
                canonicalize_response(
                    raw=raw,
                    parsed=adapter.parse(raw),
                    policy=policy,
                    conf_scale_name=scale_name,
                    raw_ref=path.relative_to(data_root()).as_posix(),
                )
            )

    with store.begin(
        stage=StageName.CANONICALIZE,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as run:
        run.inherit_from(manifest_run)
        run.write_records("spans", spans)
        run.set_engine_fingerprints(fingerprints)
        run.set_datasets(cfg.enabled_dataset_ids)
        run.mark_synthetic(cfg.synthetic)

    by_engine: dict[str, int] = {}
    for span in spans:
        by_engine[span.engine_id] = by_engine.get(span.engine_id, 0) + 1
    console.print(f"[green]canonicalized[/green] {len(spans)} spans -> {run.run_id}")
    for engine_id, count in sorted(by_engine.items()):
        console.print(f"  {engine_id}: {count} spans")

"""``ocr-risk sites`` — build correction sites and report raw OCR quality."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.canonical import rebuild_stream
from ocr_risk.config import load_config
from ocr_risk.edits import build_sites
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.paths import project_root
from ocr_risk.metrics import corpus_cer, corpus_wer, exact_match_rate, per_document_cer
from ocr_risk.provenance import capture_git_state
from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan

app = typer.Typer(name="sites", help="Build correction sites.", no_args_is_help=True)
console = Console()


@app.command("build")
def build(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Produce the ``sites`` table: the decision points of the experiment."""
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()

    align_run = store.latest(StageName.ALIGN, cfg.name)
    canonical_run = store.latest(StageName.CANONICALIZE, cfg.name)
    if align_run is None or canonical_run is None:
        raise typer.BadParameter(f"experiment {cfg.name!r} needs canonicalize and align runs")

    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for span in store.read_records(canonical_run, "spans"):
        spans_by_pair.setdefault((span.document_id, span.engine_id), []).append(span)

    alignments_by_pair: dict[tuple[str, str], list[AlignmentRecord]] = {}
    for record in store.read_records(align_run, "alignments"):
        alignments_by_pair.setdefault((record.document_id, record.engine_id), []).append(record)

    sites: list[CorrectionSite] = []
    for key, records in sorted(alignments_by_pair.items()):
        sites.extend(build_sites(records, spans_by_pair.get(key, []), cfg.sites))

    stats = _site_stats(sites)

    with store.begin(
        stage=StageName.SITES,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as run:
        run.inherit_from(align_run)
        run.write_records("sites", sites)
        run.write_json("site_stats.json", {"per_engine": stats, "synthetic": cfg.synthetic})
        run.set_datasets(cfg.enabled_dataset_ids)
        run.mark_synthetic(cfg.synthetic)

    console.print(f"[green]built[/green] {len(sites)} correction sites -> {run.run_id}")
    _print_site_stats(stats)


@app.command("baseline")
def baseline(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Report raw OCR quality per engine: CER, WER, and exact match, before any correction.

    This is the reference every later claim is measured against. Corpus-level aggregation
    (total edits over total reference length) rather than the mean of per-document rates,
    which would let a short page outvote a long one.
    """
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()

    manifest_run = store.latest(StageName.MANIFEST, cfg.name)
    canonical_run = store.latest(StageName.CANONICALIZE, cfg.name)
    if manifest_run is None or canonical_run is None:
        raise typer.BadParameter(f"experiment {cfg.name!r} needs manifest and canonicalize runs")

    documents = store.read_records(manifest_run, "documents")
    gt_of = {d.document_id: d.gt_text for d in documents}
    dataset_of = {d.document_id: d.dataset_id for d in documents}
    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for span in store.read_records(canonical_run, "spans"):
        spans_by_pair.setdefault((span.document_id, span.engine_id), []).append(span)

    # Keyed by (engine, corpus) as well as by engine. The three corpora use different
    # transcription conventions and differ in difficulty by more than the engines do, so
    # a single per-engine number would mostly report the corpus mix each engine happened
    # to cover -- and on a matched benchmark they all cover the same mix, which makes the
    # aggregate hide exactly the variation worth seeing.
    pairs_by_cell: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (document_id, engine_id), spans in spans_by_pair.items():
        pair = (gt_of[document_id], rebuild_stream(spans))
        pairs_by_cell.setdefault((engine_id, dataset_of[document_id]), []).append(pair)
        pairs_by_cell.setdefault((engine_id, "ALL"), []).append(pair)

    # Pages an engine read as EMPTY. They vanish from spans_by_pair, so a bare document
    # count silently shrinks and the corpus stops being matched without anything saying
    # so -- Tesseract reads nothing at all on two CORD receipts. An engine that recognizes
    # no text on a page has failed on that page, and the failure belongs in the table
    # beside the error rates rather than in the difference between two row counts.
    engines = {engine_id for _, engine_id in spans_by_pair}
    recognized: dict[tuple[str, str], set[str]] = {}
    for document_id, engine_id in spans_by_pair:
        recognized.setdefault((engine_id, dataset_of[document_id]), set()).add(document_id)
        recognized.setdefault((engine_id, "ALL"), set()).add(document_id)
    expected: dict[str, int] = {"ALL": len(documents)}
    for document in documents:
        expected[document.dataset_id] = expected.get(document.dataset_id, 0) + 1
    empty_pages = {
        (engine_id, dataset_id): count - len(recognized.get((engine_id, dataset_id), set()))
        for engine_id in engines
        for dataset_id, count in expected.items()
    }

    rows: list[dict[str, object]] = []
    table = Table(
        "engine", "corpus", "pages", "empty", "CER", "WER", "exact match", "median doc CER"
    )
    for (engine_id, dataset_id), pairs in sorted(pairs_by_cell.items()):
        document_rates = sorted(per_document_cer(pairs))
        median = document_rates[len(document_rates) // 2] if document_rates else 0.0
        rows.append(
            {
                "engine_id": engine_id,
                "dataset_id": dataset_id,
                "n_documents_expected": expected.get(dataset_id, len(pairs)),
                "n_documents_with_text": len(pairs),
                "n_documents_read_empty": empty_pages.get((engine_id, dataset_id), 0),
                "cer": corpus_cer(pairs).rate,
                "wer": corpus_wer(pairs).rate,
                "exact_match": exact_match_rate(pairs),
                "median_document_cer": median,
            }
        )
        table.add_row(
            engine_id,
            dataset_id,
            f"{len(pairs)}/{expected.get(dataset_id, len(pairs))}",
            str(empty_pages.get((engine_id, dataset_id), 0)),
            f"{corpus_cer(pairs).rate:.4f}",
            f"{corpus_wer(pairs).rate:.4f}",
            f"{exact_match_rate(pairs):.4f}",
            f"{median:.4f}",
        )
    console.print(table)

    output = project_root() / "results" / "generated" / "raw_ocr_baseline.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    console.print(f"wrote {output}")
    if cfg.synthetic:
        console.print("[yellow]SYNTHETIC — not a research result[/yellow]")


def _site_stats(sites: list[CorrectionSite]) -> dict[str, dict[str, float | int]]:
    by_engine: dict[str, list[CorrectionSite]] = {}
    for site in sites:
        by_engine.setdefault(site.engine_id, []).append(site)

    stats: dict[str, dict[str, float | int]] = {}
    for engine_id, engine_sites in sorted(by_engine.items()):
        evaluable = [s for s in engine_sites if s.evaluable]
        with_error = [s for s in evaluable if s.d_before > 0]
        kinds = Counter(s.site_kind.value for s in engine_sites)
        stats[engine_id] = {
            "n_sites": len(engine_sites),
            "n_evaluable": len(evaluable),
            "n_clean": len(evaluable) - len(with_error),
            "n_with_error": len(with_error),
            "error_site_rate": len(with_error) / len(evaluable) if evaluable else 0.0,
            "total_error_characters": sum(s.d_before for s in evaluable),
            **{f"kind_{name}": count for name, count in sorted(kinds.items())},
        }
    return stats


def _print_site_stats(stats: dict[str, dict[str, float | int]]) -> None:
    table = Table(
        "engine",
        "sites",
        "evaluable",
        "clean",
        "with error",
        "error rate",
        "error chars",
        title="correction sites (clean sites kept: overcorrection is unobservable without them)",
    )
    for engine_id, s in sorted(stats.items()):
        table.add_row(
            engine_id,
            str(s["n_sites"]),
            str(s["n_evaluable"]),
            str(s["n_clean"]),
            str(s["n_with_error"]),
            f"{float(s['error_site_rate']):.4f}",
            str(s["total_error_characters"]),
        )
    console.print(table)

"""``ocr-risk cgv3`` — OCR-only site discovery and its Track-A development runs.

One command per canonical stage, mirroring the CGV2 discipline: ``track-a`` is the
exploratory development study over already-seen documents; every artifact it writes
carries ``analysis_role = exploratory``. The confirmatory reserve is refused loudly by
the freshness guard at selection time -- no CGV3 command in this phase can read a fresh
document even by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from ocr_risk.canonical import rebuild_stream
from ocr_risk.discovery.enumerator import DiscoveryRules
from ocr_risk.discovery.freshness import assert_development_documents
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.hashing import file_sha256
from ocr_risk.schemas.enums import SplitRole, StageName
from ocr_risk.schemas.spans import CanonicalSpan
from ocr_risk.splits.document_partition import DocumentPartition

app = typer.Typer(
    name="cgv3",
    help="CGV3: OCR-only structural candidate discovery.",
    no_args_is_help=True,
)
console = Console()

_FRAME_FILES = (
    ("discovered", "cgv3_discovered_sites.csv"),
    ("true_coverage", "cgv3_true_coverage.csv"),
    ("candidates", "cgv3_candidates.csv"),
    ("page_counts", "cgv3_page_counts.csv"),
    ("site_metrics", "cgv3_site_metrics.csv"),
    ("rule_metrics", "cgv3_rule_metrics.csv"),
    ("conditional", "cgv3_conditional.csv"),
    ("end_to_end", "cgv3_end_to_end.csv"),
    ("harm", "cgv3_harm.csv"),
    ("k_grid", "cgv3_k_grid.csv"),
    ("failure", "cgv3_failure.csv"),
    ("oracle", "cgv3_oracle.csv"),
)


@app.command("track-a")
def track_a(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    role: Annotated[
        str,
        typer.Option("--role", help="Seen-partition role to develop on (calibrate or fit)"),
    ] = "calibrate",
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="First N role documents (pilot mode)"),
    ] = 8,
    engines: Annotated[
        str,
        typer.Option("--engines", help="Comma-separated engine ids (default: all four)"),
    ] = "",
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated/cgv3/track_a"),
) -> None:
    """Run the exploratory Track-A development study over already-seen documents."""
    from ocr_risk.config import load_config
    from ocr_risk.experiments.cgv3_track_a import run_track_a

    if role not in {"calibrate", "fit"}:
        raise typer.BadParameter(
            "--role must be calibrate or fit: the old evaluate role is spent (R-66) and "
            "no fresh document may enter Track-A"
        )
    resolved = load_config(config_path)
    cfg = resolved.config
    store = ArtifactStore()
    canonical_run = store.latest(StageName.CANONICALIZE, cfg.name)
    align_run = store.latest(StageName.ALIGN, cfg.name)
    sites_run = store.latest(StageName.SITES, cfg.name)
    manifest_run = store.latest(StageName.MANIFEST, cfg.name)
    missing = [
        name
        for name, record in (
            ("canonicalize", canonical_run),
            ("align", align_run),
            ("sites", sites_run),
            ("manifest", manifest_run),
        )
        if record is None
    ]
    if missing:
        raise typer.BadParameter(f"experiment {cfg.name!r} is missing runs: {', '.join(missing)}")
    assert canonical_run and align_run and sites_run and manifest_run

    candidates_run = store.latest(StageName.CANDIDATES, cfg.name)
    if candidates_run is None:
        raise typer.BadParameter(f"experiment {cfg.name!r} has no partition to read roles from")
    partition = DocumentPartition.load(store.run_dir(candidates_run) / "document_partition.json")
    role_documents = sorted(partition.documents(SplitRole(role)))
    selected = role_documents[:limit]
    fit_documents = frozenset(partition.documents(SplitRole.FIT))
    # The fresh partition guard: loud refusal, never silent inclusion. The fit corpus is
    # guarded too -- it reaches the fold resources, so a reserve id here would leak even
    # without ever being scored (reviewer-B hardening finding).
    assert_development_documents(selected, context="cgv3 track-a document selection")
    assert_development_documents(fit_documents, context="cgv3 track-a fold corpus")

    engine_ids = tuple(e.strip() for e in engines.split(",") if e.strip()) or tuple(
        sorted({span.engine_id for span in store.read_records(canonical_run, "spans")})
    )

    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]] = {}
    selected_set = set(selected)
    for span in store.read_records(canonical_run, "spans"):
        if span.document_id in selected_set or span.document_id in partition.documents(
            SplitRole.FIT
        ):
            spans_by_pair.setdefault((span.document_id, span.engine_id), []).append(span)
    track_pairs = [pair for pair in spans_by_pair if pair[0] in selected_set]
    streams = {pair: rebuild_stream(spans) for pair, spans in spans_by_pair.items()}
    console.print(
        f"developing on {len(selected)} {role}-role documents x {len(engine_ids)} engines "
        f"({len(track_pairs)} scored pages; fit corpus {len(streams) - len(track_pairs)} pages)"
    )

    alignments = store.read_table(align_run, "alignments").to_pandas()
    sites = store.read_table(sites_run, "sites").to_pandas()
    gt_tokens = pd.DataFrame(
        [record.model_dump(mode="json") for record in store.read_records(manifest_run, "gt_tokens")]
    )
    audit_path = Path("results/generated/cgv2/audits") / role / "alignment_site_audit.csv"
    # Fail closed (reviewer-C latent finding): without the audit states, coverage_table
    # would default every region to eligible and quietly admit ambiguous pairings into
    # the confirmatory denominator.
    if not audit_path.exists():
        raise typer.BadParameter(
            f"alignment site audit for role {role!r} is missing ({audit_path}); refusing "
            "to run with every region defaulted eligible"
        )
    audit_states = pd.read_csv(audit_path, keep_default_na=False)

    started = perf_counter()
    frames = run_track_a(
        spans_by_pair={
            pair: spans for pair, spans in spans_by_pair.items() if pair in set(track_pairs)
        },
        streams={pair: streams[pair] for pair, spans in spans_by_pair.items()},
        alignments=alignments,
        gt_tokens=gt_tokens,
        sites=sites,
        document_ids=frozenset(selected),
        fit_documents=fit_documents,
        engines=engine_ids,
        rules=DiscoveryRules(),
        audit_states=audit_states,
        progress=lambda message: console.print(f"  {message}"),
    )
    duration = perf_counter() - started

    out.mkdir(parents=True, exist_ok=True)
    for attribute, filename in _FRAME_FILES:
        frame = getattr(frames, attribute)
        frame.to_csv(out / filename, index=False)
        console.print(f"  wrote {filename} ({len(frame)} rows)")

    record = {
        "schema_version": "cgv3-track-a-v1",
        "analysis_role": "exploratory",
        "experiment": cfg.name,
        "role": role,
        "documents": selected,
        "engines": list(engine_ids),
        "input_runs": {
            "canonicalize": canonical_run.run_id,
            "align": align_run.run_id,
            "sites": sites_run.run_id,
            "manifest": manifest_run.run_id,
            "candidates": candidates_run.run_id,
        },
        "audit_states_file": str(audit_path) if audit_path.exists() else None,
        "freshness": {
            "confirmatory_reserve_size": 99,
            "reserve_intersect_selected": [],
            "guard": "assert_development_documents",
        },
        "duration_seconds": duration,
        "outputs": {
            filename: {
                "rows": len(getattr(frames, attribute)),
                "sha256": file_sha256(out / filename),
            }
            for attribute, filename in _FRAME_FILES
        },
    }
    (out / "cgv3_track_a_run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console.print(
        f"[green]track-a pass complete in {duration:.1f}s[/green] "
        f"(analysis_role=exploratory; run record -> cgv3_track_a_run.json)"
    )

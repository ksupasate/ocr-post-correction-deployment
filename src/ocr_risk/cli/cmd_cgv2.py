"""``ocr-risk cgv2`` — the Candidate Generation Study v2 (docs/cgv2/protocol.md).

One producer per canonical artifact: the study command writes the census, the proposals,
the region proposals, the fold lexicons, the contrasts, and the oracle table; everything
downstream (tables, figures, gates) reads those files and never recomputes upstream.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

import pandas as pd
import typer
from rich.console import Console

from ocr_risk.candidates.byt5 import DEFAULT_BYT5
from ocr_risk.candidates.pipeline import SiteContext
from ocr_risk.cli.cmd_candidates import _site_contexts_and_streams
from ocr_risk.config import load_config
from ocr_risk.evidence.crops import CropPolicy, build_recipe
from ocr_risk.experiments.cgv2_study import (
    CGV2_RUNGS,
    GENERATION_CAP,
    HEADLINE_EXCLUDED_DATASETS,
    K_GRID,
    PROTOCOL_AMENDMENT_COMMIT,
    PROTOCOL_FREEZE_COMMIT,
    STRUCTURAL_SITE_KINDS,
    Cgv2Run,
    RegionRecord,
    SiteCensusRecord,
    augmented_oracle,
    engine_contrast,
    region_records_from_csv,
    run_cgv2_study,
)
from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER
from ocr_risk.experiments.generator_study import ProposalRecord
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.hashing import canonical_hash, file_sha256, stable_string_set_hash
from ocr_risk.io.paths import project_root
from ocr_risk.provenance.envcapture import capture_environment, code_sha256
from ocr_risk.provenance.gitstate import capture_git_state
from ocr_risk.schemas.enums import HarmPolicy, SiteKind, SplitRole, StageName
from ocr_risk.splits.document_partition import DocumentPartition
from ocr_risk.stats import holm_bonferroni

app = typer.Typer(
    name="cgv2",
    help="Candidate Generation Study v2: structural edit coverage.",
    no_args_is_help=True,
)
console = Console()

_PRIMARY_PAIR = ("g6_union", "g3_edit_aware")


def _json_cell(value: object) -> str:
    """Stable JSON for structured values stored in the canonical CSVs."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _bbox_payload(context: SiteContext) -> dict[str, float] | None:
    box = context.site.bbox
    if box is None:
        return None
    return {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}


def _normalized_bbox_payload(context: SiteContext) -> dict[str, float] | None:
    box = context.site.bbox
    if box is None or context.image_width <= 0 or context.image_height <= 0:
        return None
    return {
        "x0": box.x0 / context.image_width,
        "y0": box.y0 / context.image_height,
        "x1": box.x1 / context.image_width,
        "y1": box.y1 / context.image_height,
    }


def _operation_type(
    original_ocr: str,
    candidate_text: str,
    site_kind: str,
    n_spans: int,
    metadata: Mapping[str, str],
) -> str:
    """Name the proposed edit from observable strings, never from ground truth."""
    shape = metadata.get("edit_shape", "")
    if shape in {"insertion", "deletion", "split", "merge"}:
        return shape
    if not original_ocr and candidate_text:
        return "insertion"
    if original_ocr and not candidate_text:
        return "deletion"
    if site_kind == SiteKind.SEGMENTATION.value or n_spans > 1:
        source_parts = len(original_ocr.split())
        target_parts = len(candidate_text.split())
        if target_parts > source_parts:
            return "split"
        if target_parts < source_parts or n_spans > 1:
            return "merge"
        return "variable_span_replacement"
    return "substitution"


def _region_operation_type(record: RegionRecord) -> str:
    shape = str(record.generator_meta.get("edit_shape", ""))
    source_parts = len(record.region_ocr.split())
    target_parts = len(record.candidate_text.split())
    if shape == "region_join" or target_parts < source_parts:
        return "merge"
    if target_parts > source_parts:
        return "split"
    return "many_to_many"


def _rung_certificate(run: Cgv2Run, engine_id: str, generator_id: str) -> dict[str, object]:
    fold = run.certificates.get(engine_id, {})
    rungs = fold.get("rungs", {})
    if not isinstance(rungs, dict):
        return {}
    entry = rungs.get(generator_id, {})
    return entry if isinstance(entry, dict) else {}


def _source_geometry_rows(
    run: Cgv2Run,
    contexts: Sequence[SiteContext],
    span_geometry: Mapping[str, dict[str, object]],
    crop_policy: CropPolicy,
    geometry_source_run: str,
) -> list[dict[str, object]]:
    """Store geometry once per scored site; proposals refer to it by stable id.

    Full source polygons remain in this table instead of being copied onto every rung's
    proposal. This keeps the candidate pool bounded in size while retaining an exact
    path from a candidate to the canonical span geometry and crop recipe.
    """
    scored = {record.site_id for record in run.census}
    rows: list[dict[str, object]] = []
    for context in sorted(contexts, key=lambda item: item.site.site_id):
        site = context.site
        if site.site_id not in scored:
            continue
        geometry = [
            span_geometry[span_id] for span_id in site.ocr_span_ids if span_id in span_geometry
        ]
        recipe = (
            build_recipe(context.image_sha256, site.bbox, crop_policy).model_dump(mode="json")
            if site.bbox is not None
            else None
        )
        rows.append(
            {
                "source_geometry_id": f"site:{site.site_id}",
                "site_id": site.site_id,
                "document_id": site.document_id,
                "dataset_id": site.dataset_id,
                "engine_id": site.engine_id,
                "image_sha256": context.image_sha256,
                "image_width": context.image_width,
                "image_height": context.image_height,
                "source_span_ids": _json_cell(list(site.ocr_span_ids)),
                "alignment_component_ids": _json_cell(list(site.alignment_ids)),
                "source_bbox": _json_cell(_bbox_payload(context)),
                "source_bbox_normalized": _json_cell(_normalized_bbox_payload(context)),
                "source_span_geometry": _json_cell(geometry),
                "crop_recipe": _json_cell(recipe),
                "geometry_source_run": geometry_source_run,
            }
        )
    return rows


def _site_proposal_rows(
    run: Cgv2Run, contexts: Sequence[SiteContext], geometry_source_run: str
) -> list[dict[str, object]]:
    contexts_by_site = {context.site.site_id: context for context in contexts}
    rows: list[dict[str, object]] = []
    for proposal in run.site_run.proposals:
        context = contexts_by_site[proposal.site_id]
        site = context.site
        certificate = _rung_certificate(run, proposal.engine_id, proposal.generator_id)
        rows.append(
            {
                **proposal.as_dict(),
                "candidate_id": (
                    f"{proposal.site_id}:{proposal.generator_id}:{proposal.generator_rank}"
                ),
                "operation_type": _operation_type(
                    proposal.original_ocr,
                    proposal.candidate_text,
                    proposal.site_kind,
                    len(site.ocr_span_ids),
                    proposal.generator_meta,
                ),
                "source_char_start": site.char_start,
                "source_char_end": site.char_end,
                "source_span_ids": _json_cell(list(site.ocr_span_ids)),
                "alignment_component_ids": _json_cell(list(site.alignment_ids)),
                "source_geometry_id": f"site:{site.site_id}",
                "geometry_source_run": geometry_source_run,
                "fold_id": f"loeo_zero_shot:held_out={proposal.engine_id}",
                "generator_config_sha256": certificate.get("generator_config_sha256", ""),
                "candidate_budget_k": GENERATION_CAP,
                "generator_native_cap": GENERATOR_LADDER[proposal.generator_id].max_candidates,
                "ambiguity_status": (
                    "eligible_low_confidence_sensitivity"
                    if site.min_align_confidence <= 0.30
                    else "eligible"
                ),
                "member_generator_id": proposal.generator_meta.get("member_generator_id", ""),
                "member_generator_version": proposal.generator_meta.get(
                    "member_generator_version", ""
                ),
                "generator_provenance": _json_cell(proposal.generator_meta),
            }
        )
    return rows


def _region_proposal_rows(
    run: Cgv2Run, contexts: Sequence[SiteContext], geometry_source_run: str
) -> list[dict[str, object]]:
    contexts_by_site = {context.site.site_id: context for context in contexts}
    rows: list[dict[str, object]] = []
    for record in run.region_records:
        left = contexts_by_site[record.left_site_id].site
        right = contexts_by_site[record.right_site_id].site
        source_span_ids = _ordered_unique([*left.ocr_span_ids, *right.ocr_span_ids])
        alignment_ids = _ordered_unique([*left.alignment_ids, *right.alignment_ids])
        certificate = _rung_certificate(run, record.engine_id, record.generator_id)
        rows.append(
            {
                **record.as_dict(),
                "candidate_id": (
                    f"{record.region_id}:{record.generator_id}:{record.generator_rank}"
                ),
                "operation_type": _region_operation_type(record),
                "source_span_ids": _json_cell(source_span_ids),
                "alignment_component_ids": _json_cell(alignment_ids),
                "left_source_span_ids": _json_cell(list(left.ocr_span_ids)),
                "right_source_span_ids": _json_cell(list(right.ocr_span_ids)),
                "left_alignment_component_ids": _json_cell(list(left.alignment_ids)),
                "right_alignment_component_ids": _json_cell(list(right.alignment_ids)),
                "source_geometry_ids": _json_cell(
                    [f"site:{record.left_site_id}", f"site:{record.right_site_id}"]
                ),
                "geometry_source_run": geometry_source_run,
                "fold_id": f"loeo_zero_shot:held_out={record.engine_id}",
                "generator_config_sha256": certificate.get("generator_config_sha256", ""),
                "candidate_budget_k": GENERATION_CAP,
                "generator_native_cap": 2,
                "ambiguity_status": "eligible",
                "pool_rung": "g6_union",
                "generator_provenance": _json_cell(record.generator_meta),
            }
        )
    return rows


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _peak_rss_bytes() -> int | None:
    """Process peak RSS in bytes (``ru_maxrss`` has platform-specific units)."""
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def _pilot_documents(
    contexts: list[SiteContext],
    partition: DocumentPartition,
    role: SplitRole,
    engine_id: str,
    documents_per_dataset: int,
) -> frozenset[str]:
    """Deterministic real-data pilot: first N role documents in each dataset."""
    if documents_per_dataset < 1:
        raise ValueError("documents_per_dataset must be positive")
    role_documents = partition.documents(role)
    by_dataset: dict[str, set[str]] = {}
    for context in contexts:
        site = context.site
        if site.engine_id == engine_id and site.document_id in role_documents:
            by_dataset.setdefault(site.dataset_id, set()).add(site.document_id)
    if not by_dataset:
        msg = f"pilot engine {engine_id!r} has no {role.value}-role documents"
        raise ValueError(msg)
    chosen = {
        document_id
        for documents in by_dataset.values()
        for document_id in sorted(documents)[:documents_per_dataset]
    }
    return frozenset(chosen)


def _completion_matches(path: Path, resume_key: str, out: Path) -> bool:
    """A completion marker is reusable only when every named output still hashes."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "complete" or payload.get("resume_key") != resume_key:
        return False
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return False
    for name, metadata in outputs.items():
        candidate = out / str(name)
        if not candidate.is_file() or not isinstance(metadata, dict):
            return False
        if file_sha256(candidate) != metadata.get("sha256"):
            return False
    return True


def _alignments_by_pair(
    store: ArtifactStore, cfg: object
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Alignment components per (document, engine), in alignment order.

    The alignment order is the reading order the component list was built in; a
    span-less site's neighbours come from this sequence, because its own char range is
    [0, 0] and the linearized stream cannot place it.
    """
    align_run = store.latest(StageName.ALIGN, cfg.name)  # type: ignore[attr-defined]
    if align_run is None:
        return {}
    frame = store.read_table(align_run, "alignments").to_pandas()
    by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.document_id), str(row.engine_id))
        by_pair.setdefault(key, []).append((str(row.alignment_id), str(row.ocr_text or "")))
    return by_pair


@app.command("study")
def study(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    role: Annotated[
        str, typer.Option("--role", help="Documents to score: calibrate (dev) or evaluate")
    ] = "calibrate",
    rungs: Annotated[str, typer.Option("--rungs", help="Comma-separated rung ids")] = ",".join(
        CGV2_RUNGS
    ),
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated/cgv2"),
    pilot_engine: Annotated[
        str | None,
        typer.Option(
            "--pilot-engine",
            help="Run a calibration-only pilot on one engine; never names canonical outputs.",
        ),
    ] = None,
    pilot_documents_per_dataset: Annotated[
        int,
        typer.Option(
            "--pilot-documents-per-dataset",
            min=0,
            help="Deterministic documents per dataset for --pilot-engine (0 disables pilot).",
        ),
    ] = 0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse a matching complete run after verifying every output hash.",
        ),
    ] = True,
) -> None:
    """Run the CGV2 study: ladder at cap 8, regions, census, certificates, contrasts.

    ``--role evaluate`` touches the held-out documents. Legitimate only after the
    protocol commit named in the run record; the command prints that name on every run.
    """
    command_started_at = _utc_now()
    command_started = perf_counter()
    resolved = load_config(config_path)
    cfg = resolved.config
    _validate_contrast_method(cfg.stats.bootstrap_unit, cfg.stats.multiplicity)
    store = ArtifactStore()
    split_role = SplitRole(role)
    run_rungs = tuple(rung.strip() for rung in rungs.split(",") if rung.strip())
    if not run_rungs:
        raise typer.BadParameter("--rungs must name at least one generator")

    candidates_run = store.latest(StageName.CANDIDATES, cfg.name)
    if candidates_run is None:
        raise typer.BadParameter(
            f"experiment {cfg.name!r} has no candidates run to read a partition from"
        )
    partition = DocumentPartition.load(store.run_dir(candidates_run) / "document_partition.json")
    contexts, streams = _site_contexts_and_streams(store, cfg, cfg.evidence.context_window_chars)
    alignments_by_pair = _alignments_by_pair(store, cfg)
    policy = cfg.risk.harm_policy

    scored_document_ids: frozenset[str] | None = None
    scored_engines: frozenset[str] | None = None
    if pilot_engine is not None:
        if split_role is not SplitRole.CALIBRATE:
            raise typer.BadParameter("--pilot-engine is calibration-only")
        if pilot_documents_per_dataset < 1:
            raise typer.BadParameter("--pilot-engine requires --pilot-documents-per-dataset >= 1")
        try:
            scored_document_ids = _pilot_documents(
                contexts,
                partition,
                split_role,
                pilot_engine,
                pilot_documents_per_dataset,
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        scored_engines = frozenset({pilot_engine})
        suffix = f"__pilot_{pilot_engine}"
    else:
        if pilot_documents_per_dataset:
            raise typer.BadParameter("--pilot-documents-per-dataset requires --pilot-engine")
        suffix = "" if split_role is SplitRole.CALIBRATE else f"__{split_role.value}"

    out.mkdir(parents=True, exist_ok=True)
    canonical_run = store.latest(StageName.CANONICALIZE, cfg.name)
    align_run = store.latest(StageName.ALIGN, cfg.name)
    input_runs = {
        stage.value: stage_run.run_id
        for stage in (
            StageName.MANIFEST,
            StageName.CANONICALIZE,
            StageName.ALIGN,
            StageName.SITES,
            StageName.CANDIDATES,
        )
        if (stage_run := store.latest(stage, cfg.name)) is not None
    }
    source_code_sha256 = code_sha256()
    protocol_path = project_root() / "docs" / "cgv2" / "protocol.md"
    protocol_sha256 = file_sha256(protocol_path)
    resume_key = canonical_hash(
        {
            "schema": "cgv2-study-resume-v1",
            "experiment": cfg.name,
            "config_sha256": resolved.sha256,
            "code_sha256": source_code_sha256,
            "protocol_frozen_commit": PROTOCOL_FREEZE_COMMIT,
            "protocol_amendment_commit": PROTOCOL_AMENDMENT_COMMIT,
            "protocol_sha256": protocol_sha256,
            "partition_sha256": partition.partition_sha256,
            "role": split_role.value,
            "rungs": run_rungs,
            "scored_document_ids": scored_document_ids,
            "scored_engines": scored_engines,
            "input_runs": input_runs,
        }
    )
    completion_path = out / f"cgv2_completion{suffix}.json"
    if resume and _completion_matches(completion_path, resume_key, out):
        console.print(f"[green]verified complete CGV2 run; reusing {completion_path.name}[/green]")
        return
    study_path = out / f"cgv2_study{suffix}.json"
    if split_role is SplitRole.EVALUATE and resume and study_path.exists():
        raise typer.BadParameter(
            f"{study_path} exists without a matching completion marker. Inspect the partial "
            "confirmatory run before explicitly continuing with --no-resume."
        )

    console.print(
        f"[dim]protocol frozen at {PROTOCOL_FREEZE_COMMIT[:7]}, amended through "
        f"{PROTOCOL_AMENDMENT_COMMIT[:7]}; "
        f"role={split_role.value} policy={policy.value}[/dim]"
    )
    if split_role is SplitRole.EVALUATE:
        console.print(
            "[yellow]scoring the HELD-OUT documents -- the confirmatory pass. "
            "One run, no redesign after it.[/yellow]"
        )
    if pilot_engine is not None:
        n_pilot_documents = len(scored_document_ids or ())
        console.print(
            f"[cyan]pilot scope: engine={pilot_engine}; documents={n_pilot_documents}; "
            f"{pilot_documents_per_dataset}/dataset[/cyan]"
        )

    run = run_cgv2_study(
        contexts,
        partition,
        policy,
        role=split_role,
        rungs=run_rungs,
        streams=streams,
        alignments_by_pair=alignments_by_pair,
        scored_document_ids=scored_document_ids,
        scored_engines=scored_engines,
        progress=lambda line: console.print(f"  [dim]{line}[/dim]"),
    )

    assert canonical_run is not None
    span_geometry = {
        span.span_id: {
            "span_id": span.span_id,
            "char_start": span.char_start,
            "char_end": span.char_end,
            "line_id": span.line_id,
            "block_id": span.block_id,
            "bbox": span.bbox.model_dump(mode="json") if span.bbox is not None else None,
            "polygon": span.polygon.model_dump(mode="json") if span.polygon is not None else None,
            "raw_ref": span.raw_ref,
            "raw_index": span.raw_index,
        }
        for span in store.read_records(canonical_run, "spans")
    }
    crop_policy = CropPolicy(
        padding_ratio=cfg.evidence.crop_padding_ratio,
        padding_min_px=cfg.evidence.crop_padding_min_px,
        target_height=cfg.evidence.crop_target_height,
        grayscale=cfg.evidence.crop_grayscale,
    )
    source_geometry_rows = _source_geometry_rows(
        run, contexts, span_geometry, crop_policy, canonical_run.run_id
    )

    pd.DataFrame([record.as_dict() for record in run.census]).to_csv(
        out / f"cgv2_site_census{suffix}.csv", index=False
    )
    pd.DataFrame(_site_proposal_rows(run, contexts, canonical_run.run_id)).to_csv(
        out / f"cgv2_proposals{suffix}.csv", index=False
    )
    pd.DataFrame(_region_proposal_rows(run, contexts, canonical_run.run_id)).to_csv(
        out / f"cgv2_region_proposals{suffix}.csv", index=False
    )
    pd.DataFrame(source_geometry_rows).to_csv(
        out / f"cgv2_source_geometry{suffix}.csv", index=False
    )
    lexicon_rows = [
        {"engine_id": engine_id, "token": token}
        for engine_id, tokens in sorted(run.fold_tokens.items())
        for token in tokens
    ]
    pd.DataFrame(lexicon_rows).to_csv(out / f"cgv2_fold_lexicons{suffix}.csv", index=False)

    _write_contrasts(
        run,
        policy,
        out,
        suffix,
        n_resamples=cfg.stats.n_bootstrap,
        seed=cfg.stats.bootstrap_seed,
        ci_level=cfg.stats.ci_level,
        bootstrap_unit=cfg.stats.bootstrap_unit,
        multiplicity=cfg.stats.multiplicity,
    )
    _write_oracle(run, policy, out, suffix)

    certificate_path = out / f"leakage_certificates{suffix}.json"
    certificate_path.write_text(
        json.dumps(
            {
                "schema_version": "cgv2-leakage-certificates-v1",
                "experiment": cfg.name,
                "role_scored": split_role.value,
                "partition_sha256": partition.partition_sha256,
                "selection_scope": cfg.splits.selection_scope,
                "overall_pass": all(
                    bool(certificate.get("pass")) for certificate in run.certificates.values()
                ),
                "folds": run.certificates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    runtime_by_generator: dict[str, float] = {}
    for cell, seconds in run.site_run.runtime_seconds.items():
        generator_id = cell.rsplit(":", maxsplit=1)[0]
        runtime_by_generator[generator_id] = runtime_by_generator.get(generator_id, 0.0) + seconds
    scored_documents = sorted({record.document_id for record in run.census})
    dataset_ids = sorted({record.dataset_id for record in run.census})
    engine_ids = sorted({record.engine_id for record in run.census})
    headline_census = _headline_census(run.census)
    headline_proposals = _headline_proposals(run.site_run.proposals)
    headline_regions = _headline_regions(run.region_records)
    model_ids = (
        {
            rung: {
                "repo_id": DEFAULT_BYT5.repo_id,
                "revision": DEFAULT_BYT5.revision,
                "identity": DEFAULT_BYT5.identity,
            }
            for rung in run_rungs
            if rung in {"g2_byt5", "g2_byt5_ctx0"}
        }
        if any(rung in {"g2_byt5", "g2_byt5_ctx0"} for rung in run_rungs)
        else {}
    )
    command_finished_at = _utc_now()
    command_duration_seconds = perf_counter() - command_started
    git_state = capture_git_state()
    environment = capture_environment()
    study_path.write_text(
        json.dumps(
            {
                "schema_version": "cgv2-study-v2",
                "run_id": f"cgv2-{command_started_at.replace(':', '').replace('-', '')}",
                "experiment": cfg.name,
                "align_run": align_run.run_id if align_run else None,
                "input_runs": input_runs,
                "config_sha256": resolved.sha256,
                "code_sha256": source_code_sha256,
                "git": git_state.model_dump(mode="json"),
                "environment": environment.model_dump(mode="json"),
                "protocol_frozen_commit": PROTOCOL_FREEZE_COMMIT,
                "protocol_amendment_commit": PROTOCOL_AMENDMENT_COMMIT,
                "protocol_sha256": protocol_sha256,
                "partition_sha256": partition.partition_sha256,
                "harm_policy": policy.value,
                "bootstrap": {
                    "n_resamples": cfg.stats.n_bootstrap,
                    "seed": cfg.stats.bootstrap_seed,
                    "ci_level": cfg.stats.ci_level,
                    "unit": cfg.stats.bootstrap_unit,
                    "multiplicity": cfg.stats.multiplicity,
                },
                "role_scored": split_role.value,
                "pilot": pilot_engine is not None,
                "pilot_engine": pilot_engine,
                "pilot_documents_per_dataset": pilot_documents_per_dataset,
                "scored_documents": len(scored_documents),
                "scored_documents_sha256": stable_string_set_hash(scored_documents),
                "datasets": dataset_ids,
                "headline_datasets": sorted(
                    set(dataset_ids).difference(HEADLINE_EXCLUDED_DATASETS)
                ),
                "stress_track_datasets": sorted(
                    set(dataset_ids).intersection(HEADLINE_EXCLUDED_DATASETS)
                ),
                "engines": engine_ids,
                "rungs": list(run_rungs),
                "generation_cap": 8,
                "k_grid": list(K_GRID),
                "n_sites_scored": run.site_run.n_sites_scored,
                "site_census_rows": len(run.census),
                "headline_site_census_rows": len(headline_census),
                "region_census": dict(run.region_census),
                "region_proposals": len(run.region_records),
                "site_proposals": len(run.site_run.proposals),
                "total_candidates": len(run.site_run.proposals) + len(run.region_records),
                "headline_region_proposals": len(headline_regions),
                "headline_site_proposals": len(headline_proposals),
                "headline_total_candidates": len(headline_proposals) + len(headline_regions),
                "source_geometry_rows": len(source_geometry_rows),
                "source_geometry_file": f"cgv2_source_geometry{suffix}.csv",
                "lexicon_sizes": run.site_run.lexicon_sizes,
                "unavailable": run.site_run.unavailable,
                "model_ids": model_ids,
                "certificates": run.certificates,
                "leakage_certificates_file": certificate_path.name,
                "leakage_certificates_sha256": file_sha256(certificate_path),
                "started_at_utc": command_started_at,
                "finished_at_utc": command_finished_at,
                "duration_seconds": command_duration_seconds,
                "peak_rss_bytes": _peak_rss_bytes(),
                "runtime_seconds": run.runtime_seconds,
                "runtime_by_generator_seconds": runtime_by_generator,
                "pid": os.getpid(),
                "command": sys.argv,
                "resume_key": resume_key,
                "completion_marker": completion_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_names = (
        f"cgv2_site_census{suffix}.csv",
        f"cgv2_proposals{suffix}.csv",
        f"cgv2_region_proposals{suffix}.csv",
        f"cgv2_source_geometry{suffix}.csv",
        f"cgv2_fold_lexicons{suffix}.csv",
        f"cgv2_contrasts{suffix}.csv",
        f"cgv2_oracle{suffix}.csv",
        certificate_path.name,
        study_path.name,
    )
    output_manifest = {
        name: {"sha256": file_sha256(out / name), "size_bytes": (out / name).stat().st_size}
        for name in output_names
    }
    completion_path.write_text(
        json.dumps(
            {
                "schema_version": "cgv2-completion-v1",
                "status": "complete",
                "exit_code": 0,
                "resume_key": resume_key,
                "experiment": cfg.name,
                "role_scored": split_role.value,
                "pilot": pilot_engine is not None,
                "pid": os.getpid(),
                "started_at_utc": command_started_at,
                "finished_at_utc": command_finished_at,
                "duration_seconds": command_duration_seconds,
                "expected_artifact_count": len(output_manifest),
                "outputs": output_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]wrote {out}/cgv2_*{suffix or '(calibrate)'} artifacts[/green]")


def _holm_column(rows: list[dict[str, object]]) -> None:
    """Adjust the four per-engine p-values of one contrast family (protocol §11)."""
    family = {str(row["engine_id"]): float(str(row["p_value"])) for row in rows}
    adjusted = {test.label: test for test in holm_bonferroni(family)}
    for row in rows:
        test = adjusted[str(row["engine_id"])]
        row["p_holm"] = test.adjusted_p_value
        row["reject_holm"] = test.significant


def _headline_census(records: Sequence[SiteCensusRecord]) -> list[SiteCensusRecord]:
    """Apply the frozen OCR-D stress-track exclusion to headline site records."""
    return [record for record in records if record.dataset_id not in HEADLINE_EXCLUDED_DATASETS]


def _headline_proposals(records: Sequence[ProposalRecord]) -> list[ProposalRecord]:
    return [record for record in records if record.dataset_id not in HEADLINE_EXCLUDED_DATASETS]


def _headline_regions(records: Sequence[RegionRecord]) -> list[RegionRecord]:
    return [record for record in records if record.dataset_id not in HEADLINE_EXCLUDED_DATASETS]


def _write_contrasts(
    run: Cgv2Run,
    policy: HarmPolicy,
    out: Path,
    suffix: str,
    *,
    n_resamples: int,
    seed: int,
    ci_level: float,
    bootstrap_unit: str,
    multiplicity: str,
) -> None:
    rows = _contrast_rows(
        _headline_census(run.census),
        _headline_proposals(run.site_run.proposals),
        policy,
        n_resamples=n_resamples,
        seed=seed,
        ci_level=ci_level,
        bootstrap_unit=bootstrap_unit,
        multiplicity=multiplicity,
    )
    pd.DataFrame(rows).to_csv(out / f"cgv2_contrasts{suffix}.csv", index=False)


def _contrast_rows(
    census: Sequence[SiteCensusRecord],
    proposals: Sequence[ProposalRecord],
    policy: HarmPolicy,
    *,
    n_resamples: int = 10_000,
    seed: int = 7,
    ci_level: float = 0.95,
    bootstrap_unit: str = "document",
    multiplicity: str = "holm",
) -> list[dict[str, object]]:
    """All frozen contrast families, shared by headline and R-37 sensitivity runs."""
    _validate_contrast_method(bootstrap_unit, multiplicity)
    challenger, baseline = _PRIMARY_PAIR
    rows: list[dict[str, object]] = []

    primary = [
        c.as_dict()
        for c in engine_contrast(
            census,
            proposals,
            challenger,
            baseline,
            policy,
            n_resamples=n_resamples,
            seed=seed,
            ci_level=ci_level,
        )
    ]
    _holm_column(primary)
    rows.extend(primary)

    structural = [
        c.as_dict()
        for c in engine_contrast(
            census,
            proposals,
            challenger,
            baseline,
            policy,
            kinds=STRUCTURAL_SITE_KINDS,
            metric_name="structural_stratum_availability",
            n_resamples=n_resamples,
            seed=seed,
            ci_level=ci_level,
        )
    ]
    _holm_column(structural)
    rows.extend(structural)

    for label, kinds in (
        ("insertion_stratum", frozenset({SiteKind.DELETION.value})),
        ("merge_split_stratum", frozenset({SiteKind.SEGMENTATION.value})),
        ("deletion_stratum", frozenset({SiteKind.INSERTION.value})),
    ):
        rows.extend(
            c.as_dict()
            for c in engine_contrast(
                census,
                proposals,
                challenger,
                baseline,
                policy,
                kinds=kinds,
                metric_name=label,
                n_resamples=n_resamples,
                seed=seed,
                ci_level=ci_level,
            )
        )

    exact = [
        c.as_dict()
        for c in engine_contrast(
            census,
            proposals,
            challenger,
            baseline,
            policy,
            exact_only=True,
            metric_name="exact_only_availability",
            n_resamples=n_resamples,
            seed=seed,
            ci_level=ci_level,
        )
    ]
    _holm_column(exact)
    rows.extend(exact)
    for row in rows:
        row.update(
            {
                "harm_policy": policy.value,
                "n_resamples": n_resamples,
                "bootstrap_seed": seed,
                "ci_level": ci_level,
                "bootstrap_unit": bootstrap_unit,
                "multiplicity": multiplicity,
            }
        )
    return rows


def _validate_contrast_method(bootstrap_unit: str, multiplicity: str) -> None:
    """Fail closed when config names a method the frozen CGV2 code does not implement."""
    if bootstrap_unit != "document":
        raise typer.BadParameter(
            f"CGV2 contrasts require stats.bootstrap_unit=document; received {bootstrap_unit!r}"
        )
    if multiplicity != "holm":
        raise typer.BadParameter(
            f"CGV2 contrasts require stats.multiplicity=holm; received {multiplicity!r}"
        )


def _oracle_rows(
    census: Sequence[SiteCensusRecord],
    proposals: Sequence[ProposalRecord],
    regions: Sequence[RegionRecord],
    policy: HarmPolicy,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rung in ("g3_edit_aware", "g6_union"):
        rung_regions = regions if rung == "g6_union" else []
        augmented = augmented_oracle(census, proposals, rung_regions, rung, policy)
        for engine_id, values in augmented.items():
            rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "variant": "site_plus_regions",
                    "harm_policy": policy.value,
                    **values,
                }
            )
        site_only = augmented_oracle(census, proposals, [], rung, policy)
        for engine_id, values in site_only.items():
            rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "variant": "site_only",
                    "harm_policy": policy.value,
                    **values,
                }
            )
    return rows


def _write_oracle(run: Cgv2Run, policy: HarmPolicy, out: Path, suffix: str) -> None:
    rows = _oracle_rows(
        _headline_census(run.census),
        _headline_proposals(run.site_run.proposals),
        _headline_regions(run.region_records),
        policy,
    )
    pd.DataFrame(rows).to_csv(out / f"cgv2_oracle{suffix}.csv", index=False)


@app.command("tables")
def tables(
    source: Annotated[
        Path, typer.Option("--source", help="Directory holding the cgv2 study CSVs")
    ] = Path("results/generated/cgv2"),
    role: Annotated[str, typer.Option("--role", help="Which study pass to read")] = "evaluate",
    sensitivity: Annotated[
        str,
        typer.Option("--sensitivity", help="'full' or 'no_twin' (excludes R-37 signature sites)"),
    ] = "full",
    include_stress_track: Annotated[
        bool,
        typer.Option(
            "--include-stress-track",
            help="Include OCR-D in aggregate sensitivity tables (never the headline default)",
        ),
    ] = False,
    harm_policy: Annotated[
        str,
        typer.Option(
            "--harm-policy",
            help="strict_worsening, non_improving, or exact_only",
        ),
    ] = HarmPolicy.STRICT_WORSENING.value,
) -> None:
    """Derive the protocol section-10 tables from the canonical study CSVs.

    The default aggregate excludes the frozen OCR-D historical/model-availability stress
    track. ``--include-stress-track`` produces the pre-registered with-OCR-D sensitivity.
    ``--sensitivity no_twin`` excludes R-37 signature sites and re-runs every contrast.
    Harm-policy variants are derived from the same immutable proposal outcomes.
    """
    from ocr_risk.analysis import cgv2_tables
    from ocr_risk.experiments.cgv2_study import proposals_from_csv

    if sensitivity not in {"full", "no_twin"}:
        raise typer.BadParameter("--sensitivity must be 'full' or 'no_twin'")
    try:
        policy = HarmPolicy(harm_policy)
    except ValueError as error:
        choices = ", ".join(item.value for item in HarmPolicy)
        raise typer.BadParameter(f"--harm-policy must be one of: {choices}") from error

    suffix = "" if role == "calibrate" else f"__{role}"
    required = [
        f"cgv2_site_census{suffix}.csv",
        f"cgv2_proposals{suffix}.csv",
        f"cgv2_fold_lexicons{suffix}.csv",
        f"cgv2_study{suffix}.json",
    ]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise typer.BadParameter(f"missing study artifacts in {source}: {', '.join(missing)}")

    study_record = json.loads((source / f"cgv2_study{suffix}.json").read_text())
    bootstrap = study_record.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise typer.BadParameter(
            f"cgv2_study{suffix}.json lacks frozen bootstrap provenance; rerun the study"
        )
    n_resamples = int(str(bootstrap["n_resamples"]))
    bootstrap_seed = int(str(bootstrap["seed"]))
    ci_level = float(str(bootstrap["ci_level"]))
    bootstrap_unit = str(bootstrap["unit"])
    multiplicity = str(bootstrap["multiplicity"])
    _validate_contrast_method(bootstrap_unit, multiplicity)

    canonical_census = pd.read_csv(source / f"cgv2_site_census{suffix}.csv", keep_default_na=False)
    canonical_proposals = pd.read_csv(source / f"cgv2_proposals{suffix}.csv", keep_default_na=False)
    lexicons = pd.read_csv(source / f"cgv2_fold_lexicons{suffix}.csv", keep_default_na=False)
    region_path = source / f"cgv2_region_proposals{suffix}.csv"
    canonical_regions = (
        pd.read_csv(region_path, keep_default_na=False) if region_path.exists() else None
    )
    audit_dir = source / "audits" / role
    required_audit_names = (
        "alignment_audit_gate.json",
        "alignment_twin_sites.csv",
        "alignment_site_audit.csv",
        "region_pair_audit.csv",
        "site_projection_audit.csv",
        "region_projection_audit.csv",
    )
    missing_audits = [name for name in required_audit_names if not (audit_dir / name).exists()]
    if missing_audits:
        raise typer.BadParameter(
            f"missing role-scoped {role} audit artifacts in {audit_dir}: "
            + ", ".join(missing_audits)
        )
    twin_path = audit_dir / "alignment_twin_sites.csv"
    twin_sites = pd.read_csv(twin_path, keep_default_na=False) if twin_path.exists() else None
    region_pair_path = audit_dir / "region_pair_audit.csv"
    region_pairs = (
        pd.read_csv(region_pair_path, keep_default_na=False) if region_pair_path.exists() else None
    )
    site_audit = pd.read_csv(audit_dir / "alignment_site_audit.csv", keep_default_na=False)
    site_projection = pd.read_csv(audit_dir / "site_projection_audit.csv", keep_default_na=False)
    region_projection = pd.read_csv(
        audit_dir / "region_projection_audit.csv", keep_default_na=False
    )

    twin_ids: set[str] = set()
    if sensitivity == "no_twin" and (twin_sites is None or twin_sites.empty):
        raise typer.BadParameter(
            "--sensitivity no_twin needs alignment_twin_sites.csv; run "
            "`ocr-risk analyze alignment-twin` first. Falling back to the full analysis "
            "silently would report a sensitivity that was never computed."
        )
    if (
        canonical_regions is not None
        and not canonical_regions.empty
        and (region_pairs is None or region_pairs.empty)
    ):
        raise typer.BadParameter(
            "CGV2 region tables need region_pair_audit.csv so headline/stress and R-37 "
            "denominators can be re-derived; run `ocr-risk analyze alignment-twin` first"
        )

    (
        audit_census,
        audit_proposals,
        audit_regions,
        audit_exclusion_counts,
    ) = cgv2_tables.audit_clean_pool(
        canonical_census,
        canonical_proposals,
        canonical_regions,
        site_audit,
        site_projection,
        region_projection,
    )
    dataset_census = audit_census.copy()
    dataset_proposals = audit_proposals.copy()
    dataset_regions = audit_regions.copy() if audit_regions is not None else None
    dataset_pairs = region_pairs.copy() if region_pairs is not None else None
    if twin_sites is not None and not twin_sites.empty and sensitivity == "no_twin":
        all_twin_ids = set(
            twin_sites[twin_sites["twin_signature"].astype(bool)]["site_id"].astype(str)
        )
        twin_ids = all_twin_ids & set(dataset_census["site_id"].astype(str))
        dataset_census = dataset_census[~dataset_census["site_id"].astype(str).isin(twin_ids)]
        dataset_proposals = dataset_proposals[
            ~dataset_proposals["site_id"].astype(str).isin(twin_ids)
        ]
        if dataset_regions is not None and not dataset_regions.empty:
            keep = ~dataset_regions["left_site_id"].astype(str).isin(twin_ids) & ~dataset_regions[
                "right_site_id"
            ].astype(str).isin(twin_ids)
            dataset_regions = dataset_regions[keep]
        if dataset_pairs is not None and not dataset_pairs.empty:
            pair_keep = ~dataset_pairs["left_site_id"].astype(str).isin(twin_ids) & ~dataset_pairs[
                "right_site_id"
            ].astype(str).isin(twin_ids)
            dataset_pairs = dataset_pairs[pair_keep]
        console.print(
            f"  R-37 sensitivity: excluded {len(twin_ids)} role/audit-eligible "
            f"twin-signature sites ({len(all_twin_ids)} in the all-role source map)"
        )

    def _without_stress(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None or frame.empty or "dataset_id" not in frame.columns:
            return frame
        return frame[~frame["dataset_id"].astype(str).isin(HEADLINE_EXCLUDED_DATASETS)]

    if include_stress_track:
        census = dataset_census.copy()
        proposals = dataset_proposals.copy()
        region_proposals = dataset_regions.copy() if dataset_regions is not None else None
    else:
        filtered_census = _without_stress(dataset_census)
        filtered_proposals = _without_stress(dataset_proposals)
        assert filtered_census is not None and filtered_proposals is not None
        census = filtered_census
        proposals = filtered_proposals
        region_proposals = _without_stress(dataset_regions)

    analysis_site_ids = set(census["site_id"].astype(str))
    analysis_pairs = dataset_pairs
    if analysis_pairs is not None and not analysis_pairs.empty:
        if "population" in analysis_pairs.columns:
            analysis_pairs = analysis_pairs[
                analysis_pairs["population"] == "study_evaluable_adjacency"
            ]
        analysis_pairs = analysis_pairs[
            analysis_pairs["left_site_id"].astype(str).isin(analysis_site_ids)
            & analysis_pairs["right_site_id"].astype(str).isin(analysis_site_ids)
        ]

    outputs = {
        "cgv2_opportunity": cgv2_tables.opportunity_table(census, proposals, policy=policy),
        "cgv2_opportunity_by_dataset": cgv2_tables.opportunity_by_dataset_table(
            dataset_census, dataset_proposals, policy=policy
        ),
        "cgv2_quality": cgv2_tables.candidate_quality_table(
            census, proposals, region_proposals, policy=policy
        ),
        "cgv2_quality_by_dataset": cgv2_tables.candidate_quality_by_dataset_table(
            dataset_census, dataset_proposals, dataset_regions, policy=policy
        ),
        "cgv2_structural_coverage": cgv2_tables.structural_coverage_table(
            census, proposals, policy=policy
        ),
        "cgv2_structural_operations": cgv2_tables.structural_operation_table(
            census,
            proposals,
            policy=policy,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            ci_level=ci_level,
        ),
        "cgv2_structural_operations_by_dataset": (
            cgv2_tables.structural_operation_by_dataset_table(
                dataset_census,
                dataset_proposals,
                policy=policy,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
                ci_level=ci_level,
            )
        ),
        "cgv2_region_opportunity": cgv2_tables.region_opportunity_table(
            region_proposals,
            analysis_pairs,
            policy=policy,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            ci_level=ci_level,
        ),
        "cgv2_preservation": cgv2_tables.preservation_table(census, proposals, region_proposals),
        "cgv2_budget": cgv2_tables.budget_table(census, proposals, policy=policy),
        "cgv2_failure_taxonomy": cgv2_tables.failure_taxonomy_table(
            census, proposals, lexicons, twin_sites=twin_sites, policy=policy
        ),
        "cgv2_failure_taxonomy_by_dataset": cgv2_tables.failure_taxonomy_by_dataset_table(
            dataset_census,
            dataset_proposals,
            lexicons,
            twin_sites=twin_sites,
            policy=policy,
        ),
    }

    modifiers: list[str] = []
    if sensitivity == "no_twin":
        modifiers.append("no_twin")
    if include_stress_track:
        modifiers.append("with_ocrd")
    if policy is not HarmPolicy.STRICT_WORSENING:
        modifiers.append(policy.value)
    out_suffix = suffix + "".join(f"__{modifier}" for modifier in modifiers)

    eligible_site_record_keys = {
        (
            str(row.site_id),
            str(row.generator_id),
            int(str(row.generator_rank)),
            str(row.candidate_text),
        )
        for row in proposals.itertuples(index=False)
    }
    site_records = [
        record
        for record in proposals_from_csv(source / f"cgv2_proposals{suffix}.csv")
        if (
            record.site_id,
            record.generator_id,
            record.generator_rank,
            record.candidate_text,
        )
        in eligible_site_record_keys
    ]
    eligible_region_record_keys = (
        {
            (
                str(row.region_id),
                str(row.generator_id),
                int(str(row.generator_rank)),
                str(row.candidate_text),
            )
            for row in region_proposals.itertuples(index=False)
        }
        if region_proposals is not None
        else set()
    )
    region_records = (
        [
            record
            for record in region_records_from_csv(source / f"cgv2_region_proposals{suffix}.csv")
            if (
                record.region_id,
                record.generator_id,
                record.generator_rank,
                record.candidate_text,
            )
            in eligible_region_record_keys
        ]
        if region_path.exists()
        else []
    )
    oracle_rows = _oracle_rows(_census_records(census), site_records, region_records, policy)
    outputs["cgv2_oracle"] = pd.DataFrame(oracle_rows)

    oracle_by_dataset: list[dict[str, object]] = []
    dataset_site_record_keys = {
        (
            str(row.site_id),
            str(row.generator_id),
            int(str(row.generator_rank)),
            str(row.candidate_text),
        )
        for row in dataset_proposals.itertuples(index=False)
    }
    canonical_site_records = [
        record
        for record in proposals_from_csv(source / f"cgv2_proposals{suffix}.csv")
        if (
            record.site_id,
            record.generator_id,
            record.generator_rank,
            record.candidate_text,
        )
        in dataset_site_record_keys
    ]
    dataset_region_record_keys = (
        {
            (
                str(row.region_id),
                str(row.generator_id),
                int(str(row.generator_rank)),
                str(row.candidate_text),
            )
            for row in dataset_regions.itertuples(index=False)
        }
        if dataset_regions is not None
        else set()
    )
    canonical_region_records = (
        [
            record
            for record in region_records_from_csv(region_path)
            if (
                record.region_id,
                record.generator_id,
                record.generator_rank,
                record.candidate_text,
            )
            in dataset_region_record_keys
        ]
        if region_path.exists()
        else []
    )
    for dataset_id, dataset_group in dataset_census.groupby("dataset_id", sort=True):
        ids = set(dataset_group["site_id"].astype(str))
        dataset_site_records = [
            record for record in canonical_site_records if record.site_id in ids
        ]
        dataset_region_records = [
            record
            for record in canonical_region_records
            if record.left_site_id in ids and record.right_site_id in ids
        ]
        if twin_ids:
            dataset_site_records = [
                record for record in dataset_site_records if record.site_id not in twin_ids
            ]
            dataset_region_records = [
                record
                for record in dataset_region_records
                if record.left_site_id not in twin_ids and record.right_site_id not in twin_ids
            ]
        oracle_by_dataset.extend(
            {**row, "dataset_id": dataset_id}
            for row in _oracle_rows(
                _census_records(dataset_group),
                dataset_site_records,
                dataset_region_records,
                policy,
            )
        )
    outputs["cgv2_oracle_by_dataset"] = pd.DataFrame(oracle_by_dataset)

    for name, table in outputs.items():
        table.to_csv(source / f"{name}{out_suffix}.csv", index=False)
        console.print(f"  wrote {name}{out_suffix}.csv ({len(table)} rows)")

    rows = _contrast_rows(
        _census_records(census),
        site_records,
        policy,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
        ci_level=ci_level,
        bootstrap_unit=bootstrap_unit,
        multiplicity=multiplicity,
    )
    pd.DataFrame(rows).to_csv(source / f"cgv2_contrasts{out_suffix}.csv", index=False)
    console.print(f"  wrote cgv2_contrasts{out_suffix}.csv ({len(rows)} rows)")

    output_paths = {
        **{name: source / f"{name}{out_suffix}.csv" for name in outputs},
        "cgv2_contrasts": source / f"cgv2_contrasts{out_suffix}.csv",
    }
    analysis_manifest = {
        "schema_version": "cgv2-analysis-manifest-v1",
        "role": role,
        "harm_policy": policy.value,
        "bootstrap": bootstrap,
        "r37_sensitivity": sensitivity,
        "headline_excluded_datasets": sorted(HEADLINE_EXCLUDED_DATASETS),
        "stress_track_included_in_aggregates": include_stress_track,
        "aggregate_datasets": sorted(set(census["dataset_id"].astype(str))),
        "per_dataset_tables_include": sorted(set(dataset_census["dataset_id"].astype(str))),
        "n_aggregate_sites": len(census),
        "n_aggregate_site_candidates": len(proposals),
        "n_aggregate_region_candidates": len(region_proposals)
        if region_proposals is not None
        else 0,
        "audit_validity_overlay": audit_exclusion_counts,
        "n_twin_sites_excluded": len(twin_ids),
        "inputs": {
            path.name: file_sha256(path)
            for path in (
                source / f"cgv2_site_census{suffix}.csv",
                source / f"cgv2_proposals{suffix}.csv",
                source / f"cgv2_region_proposals{suffix}.csv",
                source / f"cgv2_fold_lexicons{suffix}.csv",
                source / f"cgv2_study{suffix}.json",
                source / f"cgv2_completion{suffix}.json",
                source / f"leakage_certificates{suffix}.json",
                audit_dir / "alignment_audit_gate.json",
                audit_dir / "alignment_twin_sites.csv",
                audit_dir / "alignment_site_audit.csv",
                audit_dir / "region_pair_audit.csv",
                audit_dir / "site_projection_audit.csv",
                audit_dir / "region_projection_audit.csv",
            )
            if path.exists()
        },
        "outputs": {
            path.name: {"rows": len(pd.read_csv(path)), "sha256": file_sha256(path)}
            for path in output_paths.values()
            if path.exists()
        },
    }
    manifest_path = source / f"cgv2_analysis_manifest{out_suffix}.json"
    manifest_path.write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console.print(f"  wrote {manifest_path.name}")
    console.print("[green]all section-10 tables derived from canonical artifacts[/green]")


def _census_records(census: pd.DataFrame) -> list[SiteCensusRecord]:
    return [
        SiteCensusRecord(
            site_id=str(row.site_id),
            document_id=str(row.document_id),
            dataset_id=str(row.dataset_id),
            engine_id=str(row.engine_id),
            site_kind=str(row.site_kind),
            d_before=int(str(row.d_before)),
            gt_text=str(row.gt_text),
            char_start=int(str(row.char_start)),
            char_end=int(str(row.char_end)),
            n_spans=int(str(row.n_spans)),
            ocr_text=str(row.ocr_text),
            min_align_confidence=float(str(row.min_align_confidence)),
        )
        for row in census.itertuples(index=False)
    ]


@app.command("figures")
def figures(
    source: Annotated[
        Path, typer.Option("--source", help="Directory holding the cgv2 tables")
    ] = Path("results/generated/cgv2"),
    role: Annotated[str, typer.Option("--role")] = "evaluate",
    manifest: Annotated[Path, typer.Option("--manifest")] = Path(
        "results/generated/figure_manifest.json"
    ),
) -> None:
    """Render the protocol section-17 figures from the canonical tables."""
    import hashlib

    from ocr_risk.analysis import cgv2_figures
    from ocr_risk.analysis.figures import FigureSpec

    suffix = "" if role == "calibrate" else f"__{role}"

    def _read(name: str) -> pd.DataFrame:
        path = source / f"{name}{suffix}.csv"
        if not path.exists():
            raise typer.BadParameter(f"missing {path}; run `ocr-risk cgv2 tables` first")
        return pd.read_csv(path, keep_default_na=False)

    study = json.loads((source / f"cgv2_study{suffix}.json").read_text())
    engines = tuple(sorted({key.split(":")[1] for key in study["lexicon_sizes"] if ":" in key}))
    config_sha = study.get("config_sha256", "0" * 64)

    input_names = (
        "cgv2_budget",
        "cgv2_quality",
        "cgv2_structural_coverage",
        "cgv2_opportunity",
        "cgv2_oracle",
    )
    budget, quality, coverage, opportunity, oracle = (_read(name) for name in input_names)

    rendered = []
    for name, table, plot in (
        ("A_budget_recall", budget, cgv2_figures.plot_budget_recall),
        ("B_candidate_composition", quality, cgv2_figures.plot_candidate_composition),
        ("C_structural_coverage", coverage, cgv2_figures.plot_structural_coverage),
        ("D_per_engine_availability", opportunity, cgv2_figures.plot_per_engine_availability),
        ("E_oracle_opportunity", oracle, cgv2_figures.plot_oracle_opportunity),
    ):
        spec = FigureSpec(
            path=source / f"cgv2_{name}{suffix}.png",
            title=name,
            caption=f"CGV2 {name}, role={role}",
            synthetic=False,
            source_runs=(study.get("protocol_frozen_commit", ""),),
            config_sha256=config_sha,
        )
        rendered.append(plot(table, engines, spec))
        console.print(f"  wrote {rendered[-1].name}")

    if manifest.parent.exists() or manifest == Path("results/generated/figure_manifest.json"):
        manifest.parent.mkdir(parents=True, exist_ok=True)
        record = json.loads(manifest.read_text()) if manifest.exists() else {"outputs": {}}
        inputs = record.setdefault("inputs", {})
        for name in input_names:
            path = source / f"{name}{suffix}.csv"
            inputs[path.name] = {"kind": "table", "sha256": file_sha256(path)}
        study_path = source / f"cgv2_study{suffix}.json"
        inputs[study_path.name] = {"kind": "study", "sha256": file_sha256(study_path)}
        outputs = record.setdefault("outputs", {})
        for produced in rendered:
            outputs[produced.name] = {
                "kind": "figure",
                "name": produced.stem,
                "sha256": hashlib.sha256(produced.read_bytes()).hexdigest(),
            }
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        console.print(f"registered {len(rendered)} figure(s) in {manifest}")


@app.command("gates")
def gates(
    source: Annotated[
        Path, typer.Option("--source", help="Directory holding canonical CGV2 analyses")
    ] = Path("results/generated/cgv2"),
    role: Annotated[str, typer.Option("--role", help="Which study pass to judge")] = "evaluate",
) -> None:
    """Write the two frozen CGV2 gate verdicts as separate machine-readable artifacts."""
    from ocr_risk.experiments.cgv2_gates import audit_lineage_mismatches, evaluate_cgv2_gates
    from ocr_risk.experiments.cgv2_integrity import evaluate_cgv2_integrity
    from ocr_risk.io.artifacts import ArtifactStore
    from ocr_risk.schemas.enums import StageName

    integrity_path = source / "cgv2_integrity_gate.json"
    if not integrity_path.exists():
        raise typer.BadParameter(
            f"{integrity_path} is missing; run `ocr-risk cgv2 integrity` first. The "
            "verdicts must carry the validity review's outcome, not fall back to numbers."
        )
    suffix = "" if role == "calibrate" else f"__{role}"
    study_record = json.loads((source / f"cgv2_study{suffix}.json").read_text(encoding="utf-8"))
    experiment = str(study_record.get("experiment", ""))

    # Re-derive the integrity gate and refuse a hand-edited or stale copy: the verdicts
    # read its overall_pass, so it must reproduce from the review file it names plus the
    # immutable artifacts.
    recorded_integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    recorded_inputs = recorded_integrity.get("inputs", {})
    findings_path = (
        Path(str(recorded_inputs.get("findings_path")))
        if isinstance(recorded_inputs, dict) and recorded_inputs.get("findings_path")
        else None
    )
    prior_path = (
        Path(str(recorded_inputs.get("prior_evaluate_study")))
        if isinstance(recorded_inputs, dict) and recorded_inputs.get("prior_evaluate_study")
        else None
    )
    recomputed = evaluate_cgv2_integrity(
        source,
        role=role,
        findings_path=findings_path,
        prior_evaluate_study=prior_path,
    )
    if recomputed["overall_pass"] != recorded_integrity.get("overall_pass"):
        raise typer.BadParameter(
            f"{integrity_path} disagrees with a fresh derivation from its recorded review "
            "file; re-run `ocr-risk cgv2 integrity`. A hand-edited validity verdict is "
            "not evidence."
        )

    # The alignment audit licensed this pool against specific upstream tables; verify the
    # store's current runs still are those tables (review finding: proposals bytes alone
    # cannot detect a re-run align chain with changed confidence knobs).
    audit_gate_path = source / "audits" / role / "alignment_audit_gate.json"
    audit_gate = json.loads(audit_gate_path.read_text(encoding="utf-8"))
    store = ArtifactStore()
    lineage_stages = {
        "align": StageName.ALIGN,
        "sites": StageName.SITES,
        "canonicalize": StageName.CANONICALIZE,
        "manifest": StageName.MANIFEST,
    }
    lineage_tables = {
        "align": "alignments",
        "sites": "sites",
        "canonicalize": "spans",
        "manifest": "gt_tokens",
    }
    current: dict[str, tuple[str | None, str | None]] = {}
    for stage, stage_enum in lineage_stages.items():
        table = lineage_tables[stage]
        run = store.latest(stage_enum, experiment)
        if run is None:
            current[stage] = (None, None)
        else:
            table_sha = file_sha256(store.path_of(run, store.output_ref(run, f"{table}.parquet")))
            current[stage] = (run.run_id, table_sha)
    lineage = audit_lineage_mismatches(audit_gate, current)
    if lineage:
        raise typer.BadParameter(
            "the alignment audit is stale against the current artifact store: "
            + "; ".join(lineage)
            + f". Re-run `ocr-risk analyze alignment-twin --role {role}` before gating."
        )

    result = evaluate_cgv2_gates(source, role=role)
    h2_result = result["h2_readiness"]
    if not isinstance(h2_result, dict):
        raise RuntimeError("CGV2 gate returned a malformed H2 payload")
    h2_payload = dict(h2_result)
    h2_payload["role"] = role
    h2_payload["inputs"] = result.get("inputs", {})
    h2_payload["notes"] = result.get("notes", [])

    generator_payload = {key: value for key, value in result.items() if key != "h2_readiness"}
    output_suffix = "" if role == "evaluate" else f"__{role}"
    generator_path = source / f"generator_gate{output_suffix}.json"
    h2_path = source / f"h2_readiness_gate{output_suffix}.json"
    generator_path.write_text(
        json.dumps(generator_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    h2_path.write_text(json.dumps(h2_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"[bold]{generator_payload['verdict']}[/bold] -> {generator_path}")
    console.print(f"[bold]{h2_payload['verdict']}[/bold] -> {h2_path}")


@app.command("integrity")
def integrity(
    source: Annotated[
        Path, typer.Option("--source", help="Directory holding the immutable CGV2 run")
    ] = Path("results/generated/cgv2"),
    role: Annotated[str, typer.Option("--role", help="Which study pass to certify")] = "evaluate",
    findings: Annotated[
        Path,
        typer.Option("--findings", help="Tracked independent-review findings JSON"),
    ] = Path("docs/cgv2/integrity_review.json"),
    prior_evaluate_study: Annotated[
        Path,
        typer.Option(
            "--prior-evaluate-study",
            help="A pre-CGV2 evaluate-role study record, proving partition (non-)freshness",
        ),
    ] = Path("results/generated/generators/generator_study__evaluate.json"),
) -> None:
    """Derive the machine-readable integrity gate from artifacts plus the review file."""
    from ocr_risk.experiments.cgv2_integrity import evaluate_cgv2_integrity

    payload = evaluate_cgv2_integrity(
        source,
        role=role,
        findings_path=findings,
        prior_evaluate_study=prior_evaluate_study,
    )
    output_path = source / "cgv2_integrity_gate.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = cast(dict[str, object], payload["checks"])
    failed = [name for name, value in checks.items() if value is not True]
    blocking = cast(list[dict[str, str]], payload["blocking_findings"])
    console.print(
        f"overall_pass={payload['overall_pass']}; "
        f"confirmatory_status={payload['confirmatory_status']}"
    )
    if failed:
        console.print(f"[red]checks not passing: {', '.join(failed)}[/red]")
    for finding in blocking:
        console.print(f"[red]{finding['severity']}: {finding['id']} {finding['summary']}")
    console.print(f"wrote {output_path}")

"""``ocr-risk candidates`` — generate proposed edits, label them, and build evidence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.candidates import build_generator
from ocr_risk.candidates.pipeline import (
    SiteContext,
    build_labels,
    build_observation_views,
    confidences_for_site,
    generate_candidates,
    identity_outcomes,
)
from ocr_risk.canonical import rebuild_stream
from ocr_risk.config import load_config
from ocr_risk.evidence import ConfidenceFeaturizer, CropPolicy, EvidenceBuilder
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.provenance import capture_git_state
from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan

app = typer.Typer(
    name="candidates", help="Generate, label, and build evidence for edits.", no_args_is_help=True
)
console = Console()


def _site_contexts_and_streams(
    store: ArtifactStore, cfg: object, context_chars: int
) -> tuple[list[SiteContext], dict[tuple[str, str], str]]:
    """Assemble per-site context, and the linearized streams the contexts were cut from.

    The streams are returned because a context window cannot answer questions *between*
    sites: the CGV2 region layer reconstructs its candidates from exact stream slices, and
    anything reconstructed from site text plus assumed separators is wrong wherever the
    stream's separator is a newline or an excluded site sits in the gap.
    """
    name = cfg.name  # type: ignore[attr-defined]
    manifest_run = store.latest(StageName.MANIFEST, name)
    canonical_run = store.latest(StageName.CANONICALIZE, name)
    sites_run = store.latest(StageName.SITES, name)
    if manifest_run is None or canonical_run is None or sites_run is None:
        raise typer.BadParameter(
            f"experiment {name!r} needs manifest, canonicalize, and sites runs"
        )

    documents = {d.document_id: d for d in store.read_records(manifest_run, "documents")}
    spans = store.read_records(canonical_run, "spans")
    spans_by_id = {s.span_id: s for s in spans}

    streams: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for span in spans:
        grouped.setdefault((span.document_id, span.engine_id), []).append(span)
    for key, group in grouped.items():
        streams[key] = rebuild_stream(group)

    contexts: list[SiteContext] = []
    for site in store.read_records(sites_run, "sites"):
        if not site.evaluable:
            continue  # ambiguous alignments produce no candidates and no metrics
        document = documents[site.document_id]
        stream = streams.get((site.document_id, site.engine_id), "")
        confidences, scale = confidences_for_site(site, spans_by_id)
        contexts.append(
            SiteContext(
                site=site,
                context_before=stream[max(0, site.char_start - context_chars) : site.char_start],
                context_after=stream[site.char_end : site.char_end + context_chars],
                native_confidences=confidences,
                conf_scale=scale,
                image_sha256=document.image_sha256,
                image_width=document.width,
                image_height=document.height,
            )
        )
    return contexts, streams


def _site_contexts(store: ArtifactStore, cfg: object, context_chars: int) -> list[SiteContext]:
    return _site_contexts_and_streams(store, cfg, context_chars)[0]


@app.command("generate")
def generate(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Produce ``candidates``, ``labels``, and ``evidence`` tables.

    The lexicon of the lexical generator is fitted on the OCR text of the sites being
    processed. For a real experiment that corpus must be restricted to the fit split; the
    split-aware runner does that, and this ad-hoc command is for inspection only.
    """
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()

    contexts = _site_contexts(store, cfg, cfg.evidence.context_window_chars)
    if not contexts:
        raise typer.BadParameter("no evaluable sites; run `ocr-risk sites build` first")

    generators = []
    for spec in cfg.candidates.generators:
        if not spec.enabled:
            continue
        generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
        generator.fit([c.site.ocr_text for c in contexts])
        generators.append((generator, spec.max_candidates))

    candidates = generate_candidates(contexts, generators, cfg.candidates)
    sites: list[CorrectionSite] = [c.site for c in contexts]
    labels = build_labels(candidates, sites)

    featurizer = ConfidenceFeaturizer()
    featurizer.fit(
        [
            (c.site.engine_id, value, c.conf_scale)
            for c in contexts
            for value in c.native_confidences
        ]
    )
    builder = EvidenceBuilder(
        crop_policy=CropPolicy(
            padding_ratio=cfg.evidence.crop_padding_ratio,
            padding_min_px=cfg.evidence.crop_padding_min_px,
            target_height=cfg.evidence.crop_target_height,
            grayscale=cfg.evidence.crop_grayscale,
        ),
        context_chars=cfg.evidence.context_window_chars,
        confidence_featurizer=featurizer,
    )

    # The site's local window is already exactly (before + O + after), so the builder's
    # offsets into it are known without re-deriving the page stream.
    bundles = []
    for view, site_context in build_observation_views(candidates, contexts):
        site = site_context.site
        start = len(site_context.context_before)
        bundles.append(
            builder.build(
                view,
                ocr_stream=site_context.context_before + site.ocr_text + site_context.context_after,
                char_start=start,
                char_end=start + len(site.ocr_text),
                native_confidences=site_context.native_confidences,
                conf_scale=site_context.conf_scale,
            )
        )

    sites_run = store.latest(StageName.SITES, cfg.name)
    assert sites_run is not None
    stats = _candidate_stats(candidates, labels)

    with store.begin(
        stage=StageName.CANDIDATES,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as run:
        run.inherit_from(sites_run)
        run.write_records("candidates", candidates)
        run.write_records("labels", labels)
        run.write_records("evidence", bundles)
        run.write_json("candidate_stats.json", {**stats, "synthetic": cfg.synthetic})
        run.set_datasets(cfg.enabled_dataset_ids)
        run.mark_synthetic(cfg.synthetic)

    console.print(
        f"[green]generated[/green] {len(candidates)} candidates, {len(labels)} labels, "
        f"{len(bundles)} evidence bundles -> {run.run_id}"
    )
    _print_candidate_stats(stats)
    if identity_outcomes(labels):
        console.print("[red]identity candidates leaked into the scored pool[/red]")


def _candidate_stats(candidates: list, labels: list) -> dict[str, object]:  # type: ignore[type-arg]
    by_generator = Counter(c.generator_id for c in candidates)
    by_outcome = Counter(label.outcome_if_accepted.value for label in labels)
    by_family = Counter(
        c.hard_negative_family for c in candidates if c.hard_negative_family is not None
    )
    return {
        "n_candidates": len(candidates),
        "n_hard_negatives": sum(1 for c in candidates if c.is_synthetic_hard_negative),
        "by_generator": dict(sorted(by_generator.items())),
        "by_outcome_if_accepted": dict(sorted(by_outcome.items())),
        "by_hard_negative_family": dict(sorted(by_family.items())),
    }


def _print_candidate_stats(stats: dict[str, object]) -> None:
    table = Table("outcome if accepted", "count", title="candidate pool")
    outcomes: dict[str, int] = stats["by_outcome_if_accepted"]  # type: ignore[assignment]
    for name, count in outcomes.items():
        table.add_row(name, str(count))
    console.print(table)
    console.print(f"  hard negatives: {stats['n_hard_negatives']} of {stats['n_candidates']}")


@app.command("study")
def study(
    config_path: Annotated[Path, typer.Argument(help="Experiment config")],
    generators: Annotated[
        str, typer.Option("--generators", help="Comma-separated generator study ids")
    ] = "g0_lexical,g1_error_gated,g3_edit_aware",
    role: Annotated[
        str, typer.Option("--role", help="Documents to score: calibrate (dev) or evaluate")
    ] = "calibrate",
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated/generators"),
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Score candidate generators on their own, with no verifier in the loop.

    ``--role calibrate`` is the development view and is where a generator may be chosen.
    ``--role evaluate`` touches the held-out documents and is for the frozen generator
    only; the command says so on every run, because the discipline is the whole point.
    """
    import json

    import pandas as pd

    from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER
    from ocr_risk.experiments.generator_study import (
        run_generator_study,
        selection_contrast,
    )
    from ocr_risk.schemas.enums import SplitRole
    from ocr_risk.splits.document_partition import DocumentPartition

    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()
    requested = [g.strip() for g in generators.split(",") if g.strip()]
    unknown = [g for g in requested if g not in GENERATOR_LADDER]
    if unknown:
        raise typer.BadParameter(
            f"unknown generator(s) {unknown}; known: {', '.join(sorted(GENERATOR_LADDER))}"
        )

    candidates_run = store.latest(StageName.CANDIDATES, cfg.name)
    if candidates_run is None:
        raise typer.BadParameter(
            f"experiment {cfg.name!r} has no candidates run to read a partition from"
        )
    partition_path = store.run_dir(candidates_run) / "document_partition.json"
    partition = DocumentPartition.load(partition_path)

    contexts = _site_contexts(store, cfg, cfg.evidence.context_window_chars)
    split_role = SplitRole(role)
    if split_role is SplitRole.EVALUATE:
        console.print(
            "[yellow]scoring the HELD-OUT documents. Legitimate only for a generator "
            "already frozen by the pre-registered criteria.[/yellow]"
        )

    run = run_generator_study(
        contexts,
        partition,
        [GENERATOR_LADDER[g] for g in requested],
        policy=cfg.risk.harm_policy,
        role=split_role,
        progress=lambda line: console.print(f"  [dim]{line}[/dim]"),
    )

    # Evaluable sites per (engine, document): the denominator the selection contrast needs
    # and the proposals alone cannot supply, since a site nobody proposed at appears in
    # them nowhere and is exactly what the denominator is for.
    scored_documents = partition.documents(split_role)
    site_totals: dict[tuple[str, str], int] = {}
    for context in contexts:
        site = context.site
        if site.document_id in scored_documents and site.evaluable:
            key = (site.engine_id, site.document_id)
            site_totals[key] = site_totals.get(key, 0) + 1

    out.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([c.as_dict() for c in run.cells])
    proposals = pd.DataFrame([p.as_dict() for p in run.proposals])
    suffix = "" if split_role is SplitRole.CALIBRATE else f"__{split_role.value}"
    cells.to_csv(out / f"generator_quality{suffix}.csv", index=False)
    proposals.to_csv(out / f"generator_proposals{suffix}.csv", index=False)
    (out / f"generator_study{suffix}.json").write_text(
        json.dumps(
            {
                "experiment": cfg.name,
                "config_sha256": resolved.sha256,
                "harm_policy": cfg.risk.harm_policy.value,
                "role_scored": split_role.value,
                "n_sites_scored": run.n_sites_scored,
                "lexicon_sizes": run.lexicon_sizes,
                "unavailable": run.unavailable,
                "generators": requested,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # The selection rule's margin, with the paired interval the metrics rule requires. The
    # rule was applied as a point estimate, and the margin sits inside sampling noise --
    # which a reader needs and a point estimate cannot carry.
    contrasts: list[dict[str, float | int | str]] = []
    corpus_wide = [g for g in requested if not GENERATOR_LADDER[g].datasets]
    if len(corpus_wide) > 1:
        baseline = corpus_wide[0]
        for challenger in corpus_wide[1:]:
            contrasts.append(
                selection_contrast(
                    run,
                    site_totals,
                    challenger,
                    baseline,
                    cfg.risk.harm_policy,
                    n_resamples=cfg.stats.n_bootstrap,
                    ci_level=cfg.stats.ci_level,
                    seed=cfg.stats.bootstrap_seed,
                ).as_dict()
            )

        # The margin between the TOP TWO rungs is what a no-reselection defense rests on,
        # and contrasting every rung against the baseline alone leaves it quotable only as
        # a difference of point estimates. Rank on the selection metric and contrast the
        # leaders directly, so the interval exists in an artifact.
        def mean_coverage(generator_id: str) -> float:
            rates = [
                cell.quality.oracle_safe_coverage
                for cell in run.cells
                if cell.generator_id == generator_id and cell.dataset_id == "ALL"
            ]
            return sum(rates) / len(rates) if rates else float("nan")

        means = {generator_id: mean_coverage(generator_id) for generator_id in corpus_wide}
        ranked = sorted(corpus_wide, key=lambda g: means[g], reverse=True)
        if len(ranked) >= 2 and not {ranked[0], ranked[1]} & {baseline}:
            contrasts.append(
                selection_contrast(
                    run,
                    site_totals,
                    ranked[0],
                    ranked[1],
                    cfg.risk.harm_policy,
                    n_resamples=cfg.stats.n_bootstrap,
                    ci_level=cfg.stats.ci_level,
                    seed=cfg.stats.bootstrap_seed,
                ).as_dict()
            )
        pd.DataFrame(contrasts).to_csv(
            out / f"generator_selection_contrast{suffix}.csv", index=False
        )
        margins = Table(
            "challenger",
            "baseline",
            "value",
            "vs",
            "delta",
            "95% CI",
            "p",
            title="selection metric: mean oracle safe coverage, paired over documents",
        )
        for row in contrasts:
            margins.add_row(
                str(row["challenger"]),
                str(row["baseline"]),
                f"{float(row['challenger_value']):.4f}",
                f"{float(row['baseline_value']):.4f}",
                f"{float(row['delta']):+.4f}",
                f"[{float(row['ci_lower']):+.4f}, {float(row['ci_upper']):+.4f}]",
                f"{float(row['p_value']):.3f}",
            )
        console.print(margins)

    table = Table(
        "generator",
        "engine",
        "corpus",
        "sites",
        "props",
        "BCR",
        "HCR",
        "no-chg",
        "ERO",
        "CSPR",
        "oracle cov",
        title=f"candidate generator quality ({split_role.value} documents)",
    )
    for cell in run.cells:
        if cell.dataset_id != "ALL":
            continue
        q = cell.quality
        table.add_row(
            cell.generator_id,
            cell.engine_id,
            cell.dataset_id,
            str(q.n_sites),
            str(q.n_nonidentity_candidates),
            f"{q.beneficial_rate:.3f}",
            f"{q.harmful_rate:.3f}",
            f"{q.no_change_rate:.3f}",
            f"{q.error_repair_opportunity:.3f}",
            f"{q.clean_span_proposal_rate:.3f}",
            f"{q.oracle_safe_coverage:.3f}",
        )
    console.print(table)
    console.print(
        "[yellow]oracle cov is a GROUND-TRUTH ORACLE upper bound and is NOT deployable[/yellow]"
    )
    for generator_id, reason in sorted(run.unavailable.items()):
        console.print(f"[red]{generator_id} unavailable:[/red] {reason}")
    console.print(f"wrote {out}")

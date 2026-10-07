"""Publication figures, generated from artifacts and labelled with their provenance.

Two rules are structural here, not stylistic:

**Synthetic output is watermarked.** Any figure derived from simulated documents or
engines carries a visible ``SYNTHETIC - NOT A RESEARCH RESULT`` band. A plot that looks
like a finding will eventually be pasted into a slide by someone who did not run it.

**Every figure records its sources.** The caption carries the config hash and the run ids
it came from, and :mod:`ocr_risk.analysis.figure_manifest` records the content hashes, so
"where did this number come from?" is answerable from the file itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are produced in CI and on servers
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ocr_risk.metrics import RiskCoverageCurve

__all__ = [
    "FigureSpec",
    "plot_confidence_reliability",
    "plot_coverage_at_risk",
    "plot_reliability",
    "plot_risk_coverage",
    "plot_transfer_gap",
]

_WATERMARK = "SYNTHETIC - NOT A RESEARCH RESULT"
_PALETTE = ("#2f6f9f", "#c0603a", "#4f8f52", "#8a5fa8", "#b8933a", "#5f8f95", "#96566b")


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """What a figure is, and where it came from."""

    path: Path
    title: str
    caption: str
    synthetic: bool
    source_runs: tuple[str, ...]
    config_sha256: str


def _finish(fig: Figure, spec: FigureSpec) -> Path:
    """Apply the provenance footer, the synthetic watermark, and save."""
    footer = (
        f"config {spec.config_sha256[:12]} | runs {', '.join(r[:24] for r in spec.source_runs)}"
    )
    fig.text(0.01, 0.01, footer, fontsize=6, color="#666666", ha="left", va="bottom")

    if spec.synthetic:
        fig.text(
            0.5,
            0.5,
            _WATERMARK,
            fontsize=22,
            color="#d0d0d0",
            ha="center",
            va="center",
            rotation=24,
            alpha=0.45,
            zorder=0,
        )
        fig.text(
            0.5,
            0.965,
            _WATERMARK,
            fontsize=9,
            color="#a03020",
            ha="center",
            va="top",
            weight="bold",
        )

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spec.path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return spec.path


def plot_risk_coverage(
    curves: dict[str, RiskCoverageCurve], spec: FigureSpec, epsilons: tuple[float, ...] = ()
) -> Path:
    """Risk against coverage, one small multiple per fold and one line per method.

    Faceted rather than overlaid. Keys arrive as ``"<fold> | <verifier>"`` because per-fold
    reporting is mandatory -- pooling folds would count every test document once per
    engine -- and with four engines and seven evidence configurations that is 28 curves.
    Overlaying them produced a legend that covered the plot and a figure nobody could read
    a value off. The facet keeps every curve and makes the comparison the eye actually
    needs (methods within an engine) the one that is adjacent.
    """
    panels: dict[str, dict[str, RiskCoverageCurve]] = {}
    for label, curve in curves.items():
        fold, _, method = label.partition(" | ")
        panels.setdefault(fold if method else "", {})[method or fold] = curve

    n_panels = max(len(panels), 1)
    n_columns = 1 if n_panels == 1 else 2
    n_rows = (n_panels + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.4 if n_columns > 1 else 7.0, 2.6 * n_rows + 1.4),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat = [axes[r][c] for r in range(n_rows) for c in range(n_columns)]

    for panel_index, (fold, methods) in enumerate(sorted(panels.items())):
        ax = flat[panel_index]
        for index, (label, curve) in enumerate(sorted(methods.items())):
            points = [p for p in curve.points if p.n_accepted > 0]
            if not points:
                continue
            points.sort(key=lambda p: p.coverage)
            ax.plot(
                [p.coverage for p in points],
                [p.risk for p in points],
                label=label,
                color=_PALETTE[index % len(_PALETTE)],
                linewidth=1.4,
                marker="o",
                markersize=2.0,
            )
        for epsilon in epsilons:
            ax.axhline(epsilon, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_title(fold or spec.title, fontsize=8.5)
        ax.grid(alpha=0.25, linewidth=0.6)

    for ax in flat[len(panels) :]:
        ax.set_axis_off()

    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=7,
            loc="lower center",
            ncols=min(len(labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.055),
        )
    subtitle = spec.title
    if epsilons:
        subtitle += "   (dashed: " + ", ".join(f"eps={e:g}" for e in epsilons) + ")"
    fig.suptitle(subtitle, fontsize=11)
    fig.supxlabel("coverage (accepted edits / edits considered)", fontsize=9, y=0.135)
    fig.supylabel("accepted-edit risk (harmful / accepted)", fontsize=9)
    fig.text(0.01, 0.032, spec.caption, fontsize=7, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(bottom=0.22 if n_rows > 1 else 0.36, top=0.90)
    return _finish(fig, spec)


def plot_coverage_at_risk(table: pd.DataFrame, spec: FigureSpec) -> Path:
    """Coverage attainable at each tolerance, per held-out engine, with document-level CIs.

    Faceted by engine for the same reason the other two headline figures are: per-fold
    reporting is mandatory, and four engines times seven evidence configurations is 28
    labelled groups on one axis.

    Intervals are the per-fold percentile intervals, never an average of interval
    endpoints across folds -- that would keep within-fold sampling noise and discard the
    between-engine variance while looking like a tightened estimate.
    """
    if table.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.text(0.5, 0.5, "no feasible operating points", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return _finish(fig, spec)

    frame = table.assign(engine=table["fold_id"].astype(str).str.rsplit(":", n=1).str[-1])
    engines = sorted(frame["engine"].unique())
    epsilons = sorted(frame["epsilon"].unique())
    n_columns = 1 if len(engines) == 1 else 2
    n_rows = (len(engines) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.6 if n_columns > 1 else 7.0, 2.5 * n_rows + 1.5),
        squeeze=False,
        sharey=True,
    )
    flat = [axes[r][c] for r in range(n_rows) for c in range(n_columns)]
    width = 0.8 / max(len(epsilons), 1)

    for panel_index, engine in enumerate(engines):
        ax = flat[panel_index]
        panel = frame[frame["engine"] == engine]
        verifiers = sorted(panel["verifier_id"].unique())
        for index, epsilon in enumerate(epsilons):
            subset = panel[panel["epsilon"] == epsilon].set_index("verifier_id").reindex(verifiers)
            positions = [i + index * width - 0.4 + width / 2 for i in range(len(verifiers))]
            errors = [
                (subset["coverage"] - subset["coverage_ci_lower"])
                .clip(lower=0)
                .fillna(0.0)
                .to_numpy(),
                (subset["coverage_ci_upper"] - subset["coverage"])
                .clip(lower=0)
                .fillna(0.0)
                .to_numpy(),
            ]
            ax.bar(
                positions,
                subset["coverage"].fillna(0.0).to_numpy(),
                width=width * 0.92,
                label=f"eps={epsilon:g}",
                color=_PALETTE[index % len(_PALETTE)],
                yerr=errors,
                capsize=1.8,
                error_kw={"linewidth": 0.7, "ecolor": "#555555"},
            )
        ax.set_xticks(range(len(verifiers)))
        ax.set_xticklabels(verifiers, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"held out: {engine}", fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    for ax in flat[len(engines) :]:
        ax.set_axis_off()

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        ncols=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.supylabel("coverage at tolerance", fontsize=9)
    fig.suptitle(spec.title, fontsize=11)
    fig.text(0.01, 0.032, spec.caption, fontsize=7, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(bottom=0.19 if n_rows > 1 else 0.34, top=0.90, hspace=0.7)
    return _finish(fig, spec)


def plot_reliability(table: pd.DataFrame, spec: FigureSpec) -> Path:
    """Brier and ECE per fold and method, faceted by held-out engine.

    Per (fold, verifier) and never averaged: a mean over four fixed engines collapses the
    cross-engine variation H1 is about, and an error bar on such a mean invites reading
    four fixed environments as a sample from a population of engines.

    Faceted for the same reason the risk-coverage figure is. Four engines times seven
    evidence configurations is 28 bars, and 28 long labels on one axis overlap into
    illegibility -- a headline figure nobody can read a value off is a defect, not a
    cosmetic issue.
    """
    if table.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.text(0.5, 0.5, "no calibration data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return _finish(fig, spec)

    frame = table.assign(engine=table["fold_id"].astype(str).str.rsplit(":", n=1).str[-1])
    engines = sorted(frame["engine"].unique())
    n_columns = 1 if len(engines) == 1 else 2
    n_rows = (len(engines) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.6 if n_columns > 1 else 7.0, 2.5 * n_rows + 1.5),
        squeeze=False,
        sharey=True,
    )
    flat = [axes[r][c] for r in range(n_rows) for c in range(n_columns)]

    for panel_index, engine in enumerate(engines):
        ax = flat[panel_index]
        group = frame[frame["engine"] == engine].sort_values("verifier_id")
        positions = range(len(group))
        ax.bar(
            [p - 0.2 for p in positions],
            group["brier"].to_numpy(),
            width=0.38,
            label="Brier (primary)",
            color=_PALETTE[0],
        )
        ax.bar(
            [p + 0.2 for p in positions],
            group["ece_equal_mass"].to_numpy(),
            width=0.38,
            label="ECE (equal-mass)",
            color=_PALETTE[1],
        )
        ax.set_xticks(list(positions))
        ax.set_xticklabels(group["verifier_id"], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"held out: {engine}", fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    for ax in flat[len(engines) :]:
        ax.set_axis_off()

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.supylabel("error (lower is better)", fontsize=9)
    fig.suptitle(spec.title, fontsize=11)
    fig.text(0.01, 0.032, spec.caption, fontsize=7, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(bottom=0.19 if n_rows > 1 else 0.34, top=0.90, hspace=0.7)
    return _finish(fig, spec)


def plot_transfer_gap(table: pd.DataFrame, spec: FigureSpec, metric: str) -> Path:
    """The H1 view: the zero-shot minus oracle gap, per target engine, with paired CIs.

    A dot-and-interval per (engine, verifier) rather than bars over a mean. Four OCR
    engines are four fixed environments, so a mean across them summarizes four points and
    hides the heterogeneity that is the most useful thing four engines can show; and an
    error bar drawn on such a mean invites reading them as a sample from a population.

    Positive is worse: the gap is measured so that transfer degrading the metric moves the
    point to the right of zero.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    subset = table[table["metric"] == metric] if "metric" in table else pd.DataFrame()
    if subset.empty:
        ax.text(0.5, 0.5, f"no paired {metric} comparison", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return _finish(fig, spec)

    subset = subset.sort_values(["held_out_engine", "verifier_id"]).reset_index(drop=True)
    labels = [f"{r.held_out_engine} | {r.verifier_id}" for r in subset.itertuples()]
    positions = list(range(len(subset)))
    lower = (subset["delta"] - subset["delta_ci_lower"]).clip(lower=0).to_numpy()
    upper = (subset["delta_ci_upper"] - subset["delta"]).clip(lower=0).to_numpy()

    colours = [
        _PALETTE[1] if degraded else _PALETTE[0]
        for degraded in subset.get("degraded_after_holm", pd.Series([False] * len(subset)))
    ]
    ax.errorbar(
        subset["delta"].to_numpy(),
        positions,
        xerr=[lower, upper],
        fmt="none",
        ecolor="#777777",
        elinewidth=1.0,
        capsize=2.5,
    )
    ax.scatter(subset["delta"].to_numpy(), positions, c=colours, s=34, zorder=3)
    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric}:  loeo_zero_shot - in_engine_oracle   (positive = transfer is worse)")
    ax.set_title(spec.title, fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.text(0.01, 0.045, spec.caption, fontsize=7, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(left=0.34, bottom=0.22)
    return _finish(fig, spec)


def plot_confidence_reliability(table: pd.DataFrame, spec: FigureSpec) -> Path:
    """Native OCR confidence against empirical correctness, per engine.

    Descriptive, and independent of any verifier: it says whether an engine's own
    confidence means the same thing across engines. If the curves differ in shape, a
    verifier that learned one engine's confidence semantics has to relearn them for
    another — which is a mechanism by which H1 could be true, visible before any model is
    fitted.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    if table.empty:
        ax.text(0.5, 0.5, "no confidence data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return _finish(fig, spec)

    for index, (engine_id, group) in enumerate(table.groupby("engine_id", sort=True)):
        group = group.sort_values("confidence_bin")
        ax.plot(
            group["confidence_bin"].to_numpy(),
            group["empirical_correct"].to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=_PALETTE[index % len(_PALETTE)],
            label=f"{engine_id} (n={int(group['n'].sum())})",
        )
    ax.plot([0, 1], [0, 1], linestyle=":", color="#666666", linewidth=1.0, label="perfect")
    ax.set_xlabel("normalized native confidence")
    ax.set_ylabel("empirical fraction of spans read correctly")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(spec.title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.text(0.01, 0.045, spec.caption, fontsize=7, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(bottom=0.22)
    return _finish(fig, spec)

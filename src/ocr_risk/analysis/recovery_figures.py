"""Recovery-phase figures: candidate quality, oracle potential, and matched discrimination.

Recovery-phase **evidence**, not manuscript figures. They exist to make four questions
legible that the H1 pilot could only answer with one number between them: what the
generator proposes, what a perfect verifier could do with it, whether transfer costs
ranking ability, and whether the scores separate the two classes at all.

Every ground-truth oracle quantity is labelled as one on the figure itself. A ceiling that
looks like a result will eventually be pasted into a slide by someone who did not run it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ocr_risk.analysis.figures import FigureSpec, _finish

__all__ = [
    "plot_candidate_quality",
    "plot_matched_discrimination",
    "plot_oracle_potential",
    "plot_score_distributions",
]

_ORACLE_BANNER = "GROUND-TRUTH ORACLE — NOT DEPLOYABLE"
_OUTCOME_COLOURS = {
    "beneficial": "#4f8f52",
    "no_change": "#b8933a",
    "harmful": "#c0603a",
}


def _layout(fig: Figure, spec: FigureSpec, *, legend_columns: int, oracle: bool) -> None:
    """One layout for every recovery figure: banner, title, legend, caption, in that order.

    Placed in explicit figure coordinates with reserved space rather than by
    ``tight_layout`` alone, because the banner and the caption are the two elements a
    reader must not miss and they are exactly the ones that collide with the title and the
    legend when the axes are allowed to expand into them.
    """
    top = 0.90 if oracle else 0.93
    fig.tight_layout(rect=(0.0, 0.16, 1.0, top))
    if oracle:
        fig.text(
            0.5,
            0.995,
            _ORACLE_BANNER,
            fontsize=9,
            color="#a03020",
            ha="center",
            va="top",
            weight="bold",
        )
    fig.suptitle(spec.title, fontsize=11, y=top + (0.045 if oracle else 0.05))
    if legend_columns:
        fig.legend(
            loc="lower center",
            ncol=legend_columns,
            frameon=False,
            fontsize=7.5,
            bbox_to_anchor=(0.5, 0.075),
        )
    fig.text(
        0.5, 0.055, spec.caption, ha="center", va="top", fontsize=7, color="#444444", wrap=True
    )


def plot_candidate_quality(quality: pd.DataFrame, spec: FigureSpec) -> Path:
    """R1 — beneficial, no-change and harmful shares per generator and engine.

    Stacked to full height because the three shares partition the non-identity pool: a
    grouped bar would let a reader compare beneficial rates without seeing that one
    generator proposed 80 times fewer edits to take that share of.
    """
    frame = quality[quality["dataset_id"] == "ALL"].copy()
    generators = sorted(frame["generator_id"].unique())
    engines = sorted(frame["engine_id"].unique())

    fig, axes = plt.subplots(
        1, len(engines), figsize=(2.6 * len(engines) + 1.6, 5.2), squeeze=False, sharey=True
    )
    for column, engine in enumerate(engines):
        ax = axes[0][column]
        subset = frame[frame["engine_id"] == engine].set_index("generator_id")
        positions = np.arange(len(generators))
        bottom = np.zeros(len(generators))
        for name, key in (
            ("beneficial", "beneficial_rate"),
            ("no_change", "no_change_rate"),
            ("harmful", "harmful_rate"),
        ):
            values = np.array([float(subset[key].get(g, np.nan)) for g in generators], dtype=float)
            values = np.nan_to_num(values)
            ax.bar(
                positions,
                values,
                bottom=bottom,
                color=_OUTCOME_COLOURS[name],
                label=name if column == 0 else None,
                width=0.68,
            )
            bottom += values
        for index, generator in enumerate(generators):
            count = subset["n_nonidentity_candidates"].get(generator, 0)
            ax.text(
                index,
                1.02,
                f"n={int(count)}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#555555",
            )
        ax.set_title(engine, fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(generators, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0][0].set_ylabel("share of non-identity candidates")
    _layout(fig, spec, legend_columns=3, oracle=False)
    return _finish(fig, spec)


def plot_oracle_potential(quality: pd.DataFrame, spec: FigureSpec, threshold: float) -> Path:
    """R2 — what a perfect verifier could reach, against the pre-registered floor."""
    frame = quality[quality["dataset_id"] == "ALL"].copy()
    generators = sorted(frame["generator_id"].unique())
    engines = sorted(frame["engine_id"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), squeeze=False)
    for panel, (key, label) in enumerate(
        (
            ("oracle_safe_coverage", "oracle safe coverage"),
            ("oracle_repair_recall", "oracle repair recall"),
        )
    ):
        ax = axes[0][panel]
        width = 0.8 / max(len(generators), 1)
        for index, generator in enumerate(generators):
            subset = frame[frame["generator_id"] == generator].set_index("engine_id")
            values = [float(subset[key].get(e, np.nan)) for e in engines]
            ax.bar(
                np.arange(len(engines)) + index * width,
                np.nan_to_num(values),
                width=width,
                label=generator if panel == 0 else None,
            )
        if key == "oracle_safe_coverage":
            ax.axhline(
                threshold,
                color="#a03020",
                linestyle="--",
                linewidth=1.0,
                label="H2 readiness floor" if panel == 0 else None,
            )
        ax.set_xticks(np.arange(len(engines)) + 0.4 - width / 2)
        ax.set_xticklabels(engines, rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0][0].set_ylabel("fraction")
    _layout(fig, spec, legend_columns=5, oracle=True)
    return _finish(fig, spec)


def plot_matched_discrimination(
    table: pd.DataFrame, spec: FigureSpec, metric: str, verifier: str | None = None
) -> Path:
    """R3 — zero-shot minus matched reference, per target engine.

    The **pooled** interval is what the verdict is read from, and the three donor
    substitutions are drawn beside it. Showing only the pooled row would hide how much of
    it rests on one substitution; showing only the donors would leave a reader to average
    three overlapping intervals by eye. Negative means transfer ranked worse.

    ``verifier`` filters to one verifier's rows before anything is drawn. The table holds
    three verifiers; without the filter the pooled diamond falls to the first row in CSV
    order -- a confidence-free control -- under the primary verifier's caption.
    """
    frame = table[table["metric"] == metric]
    if verifier is not None and "verifier_id" in frame:
        frame = frame[frame["verifier_id"] == verifier]
    # A table without the aggregation column predates the pooled row: every row is a
    # per-donor diagnostic and none is a verdict row. Inheriting the whole frame as
    # "pooled" drew a verdict diamond over the first donor -- so the legacy path renders
    # donor points only.
    pooled = frame[frame["aggregation"] == "pooled"] if "aggregation" in frame else frame.iloc[0:0]
    per_donor = frame[frame["aggregation"] == "per_donor"] if "aggregation" in frame else frame
    engines = sorted(pooled["held_out_engine"].unique()) or sorted(
        frame["held_out_engine"].unique()
    )

    fig, ax = plt.subplots(figsize=(7.8, 0.9 * max(len(engines), 1) + 3.2))
    for row_index, engine in enumerate(engines):
        donors = per_donor[per_donor["held_out_engine"] == engine]
        for offset, row in zip(
            np.linspace(-0.22, 0.22, max(len(donors), 1)), donors.itertuples(), strict=False
        ):
            ax.plot(
                [float(row.delta_ci_lower), float(row.delta_ci_upper)],
                [row_index + offset] * 2,
                color="#9aa7b1",
                linewidth=0.9,
                zorder=1,
            )
            ax.plot(
                float(row.delta),
                row_index + offset,
                "o",
                markersize=3.0,
                color="#5f7480",
                zorder=2,
            )
        summary = pooled[pooled["held_out_engine"] == engine]
        if summary.empty:
            continue
        if len(summary) > 1:
            msg = (
                f"{len(summary)} pooled {metric} rows for {engine} survive the filters; "
                "refusing to plot the first of them under one caption"
            )
            raise ValueError(msg)
        row = summary.iloc[0]
        significant = bool(row["degraded_after_holm"])
        ax.errorbar(
            float(row["delta"]),
            row_index,
            xerr=[
                [float(row["delta"]) - float(row["delta_ci_lower"])],
                [float(row["delta_ci_upper"]) - float(row["delta"])],
            ],
            fmt="D",
            markersize=6.5,
            capsize=4,
            elinewidth=2.0,
            color="#c0603a" if significant else "#2f6f9f",
            zorder=3,
        )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(range(len(engines)))
    ax.set_yticklabels(engines, fontsize=9)
    ax.set_ylim(-0.6, len(engines) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric}: zero-shot minus matched reference (negative = transfer ranks worse)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    _layout(fig, spec, legend_columns=0, oracle=False)
    return _finish(fig, spec)


def plot_score_distributions(
    predictions: pd.DataFrame, spec: FigureSpec, arms: tuple[str, str]
) -> Path:
    """R4 — beneficial versus harmful score distributions, per held-out engine and arm.

    The picture behind every ranking statistic. Two arms whose AUCs differ by a hundredth
    can look very different here, and an arm whose scores have collapsed onto one value
    shows it immediately — which no summary number does.
    """
    engines = sorted(predictions["held_out_engine"].unique())
    fig, axes = plt.subplots(
        len(engines), 2, figsize=(8.6, 2.1 * len(engines) + 2.4), squeeze=False, sharex=True
    )
    bins = np.linspace(0.0, 1.0, 31)
    for row, engine in enumerate(engines):
        for column, arm in enumerate(arms):
            ax = axes[row][column]
            subset = predictions[
                (predictions["held_out_engine"] == engine) & (predictions["arm"] == arm)
            ]
            for label, colour in (("safe", "#4f8f52"), ("harmful", "#c0603a")):
                values = subset[subset["class"] == label]["score"]
                if values.empty:
                    continue
                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    alpha=0.5,
                    color=colour,
                    label=label if row == 0 and column == 0 else None,
                )
            ax.set_title(f"{engine} — {arm}", fontsize=8)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    axes[-1][0].set_xlabel("calibrated score")
    axes[-1][1].set_xlabel("calibrated score")
    _layout(fig, spec, legend_columns=2, oracle=False)
    return _finish(fig, spec)

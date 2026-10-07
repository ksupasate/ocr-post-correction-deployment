"""CGV2 figures A-E (protocol §17), regenerated from the canonical tables.

Research evidence figures, not manuscript figures. Each one renders exactly one canonical
CSV, names the file it read in the provenance footer, and labels every oracle quantity as
an oracle on the figure itself — a ceiling that looks like a result will eventually be
pasted into a slide by someone who did not run it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ocr_risk.analysis.figures import FigureSpec, _finish

__all__ = [
    "plot_budget_recall",
    "plot_candidate_composition",
    "plot_oracle_opportunity",
    "plot_per_engine_availability",
    "plot_structural_coverage",
]

_ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")
_H2_EDITS_FLOOR = 245
_H2_COVERAGE_FLOOR = 0.05
_STRUCTURAL_SITE_LABELS = {
    "deletion": "OCR deletion\n(repair insertion)",
    "insertion": "OCR insertion\n(repair deletion)",
    "segmentation": "segmentation",
}


def _grid(columns: int, engines: tuple[str, ...] = _ENGINES) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(4.2 * columns, 3.6),
        squeeze=False,
    )
    return fig, list(axes[0])


def plot_budget_recall(budget: pd.DataFrame, engines: tuple[str, ...], spec: FigureSpec) -> Path:
    """Figure A: exact candidate recall against the candidate budget K."""
    fig, axes = _grid(len(engines), engines)
    for axis, engine_id in zip(axes, engines, strict=True):
        frame = budget[(budget["engine_id"] == engine_id) & (budget["generator_id"] != "g2_byt5")]
        for rung, group in frame.groupby("generator_id", sort=True):
            ordered = group.sort_values("k")
            axis.plot(ordered["k"], ordered["exact_recall"], marker="o", label=str(rung))
        axis.set_title(engine_id)
        axis.set_xlabel("candidate budget K")
        axis.set_ylabel("exact recall (error sites)")
        axis.set_xscale("log", base=2)
    axes[0].legend(fontsize=6)
    fig.suptitle("Candidate recall vs budget K, by rung and held-out engine", fontsize=10)
    return _finish(fig, spec)


def plot_candidate_composition(
    quality: pd.DataFrame, engines: tuple[str, ...], spec: FigureSpec
) -> Path:
    """Figure B: beneficial against harmful candidate share, per rung and engine."""
    fig, axes = _grid(len(engines), engines)
    for axis, engine_id in zip(axes, engines, strict=True):
        frame = quality[quality["engine_id"] == engine_id].sort_values("generator_id")
        positions = range(len(frame))
        axis.bar(positions, frame["beneficial_rate"], label="beneficial", color="#4f8f52")
        axis.bar(
            positions,
            frame["harmful_rate"],
            bottom=frame["beneficial_rate"],
            label="harmful",
            color="#c0603a",
        )
        axis.set_xticks(list(positions))
        axis.set_xticklabels(frame["generator_id"], rotation=60, ha="right", fontsize=6)
        axis.set_title(engine_id)
        axis.set_ylabel("share of non-identity candidates")
    axes[0].legend(fontsize=7)
    fig.suptitle("Beneficial vs harmful candidate composition by generator", fontsize=10)
    return _finish(fig, spec)


def plot_structural_coverage(
    coverage: pd.DataFrame, engines: tuple[str, ...], spec: FigureSpec
) -> Path:
    """Figure C: availability on each structural stratum (K = 4), g6 against g3."""
    fig, axes = _grid(len(engines), engines)
    strata = sorted(set(coverage["stratum"]))
    positions = list(range(len(strata)))
    width = 0.36
    for axis, engine_id in zip(axes, engines, strict=True):
        for rung_index, (rung, color) in enumerate(
            (("g3_edit_aware", "#b0b0b0"), ("g6_union", "#2f6f9f"))
        ):
            values: list[float] = []
            for stratum in strata:
                rows = coverage[
                    (coverage["engine_id"] == engine_id)
                    & (coverage["stratum"] == stratum)
                    & (coverage["generator_id"] == rung)
                ]
                values.append(float(rows["availability"].iloc[0]) if not rows.empty else 0.0)
            offsets = [position + (rung_index - 0.5) * width for position in positions]
            axis.bar(offsets, values, width=width, color=color, label=rung)
        axis.set_title(engine_id)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [_STRUCTURAL_SITE_LABELS.get(stratum, stratum) for stratum in strata],
            rotation=25,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel("availability (error sites)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=2)
    fig.suptitle(
        "Structural availability by OCR-side alignment kind",
        fontsize=10,
    )
    return _finish(fig, spec)


def plot_per_engine_availability(
    opportunity: pd.DataFrame, engines: tuple[str, ...], spec: FigureSpec
) -> Path:
    """Figure D: site-level availability at K = 4, every rung, every engine."""
    frame = opportunity[(opportunity["k"] == 4) & (opportunity["stratum"] == "all")]
    rungs = sorted(set(frame["generator_id"]))
    fig, axis = plt.subplots(figsize=(1.6 * len(rungs) + 2, 3.8))
    positions = list(range(len(rungs)))
    width = 0.8 / len(engines)
    for engine_index, engine_id in enumerate(engines):
        values = [
            float(
                frame[(frame["generator_id"] == rung) & (frame["engine_id"] == engine_id)][
                    "availability"
                ].iloc[0]
            )
            if not frame[(frame["generator_id"] == rung) & (frame["engine_id"] == engine_id)].empty
            else 0.0
            for rung in rungs
        ]
        offsets = [
            position + (engine_index - (len(engines) - 1) / 2) * width for position in positions
        ]
        axis.bar(
            offsets,
            values,
            width=width,
            label=engine_id,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(rungs, rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("availability (error sites, K=4)")
    axis.legend(fontsize=7, ncol=len(engines))
    fig.suptitle("Per-engine candidate availability by rung", fontsize=10)
    return _finish(fig, spec)


def plot_oracle_opportunity(
    oracle: pd.DataFrame, engines: tuple[str, ...], spec: FigureSpec
) -> Path:
    """Figure E: ORACLE safe coverage and accepted edits against the frozen H2 floors.

    Every quantity on this figure is a ground-truth oracle ceiling — what a perfect
    verifier could do, not what any method did.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
    positions = list(range(len(engines)))
    width = 0.36
    for rung_index, (rung, color) in enumerate(
        (("g3_edit_aware", "#b0b0b0"), ("g6_union", "#2f6f9f"))
    ):
        frame = oracle[
            (oracle["generator_id"] == rung) & (oracle["variant"] == "site_plus_regions")
        ]
        values = {
            str(row.engine_id): row
            for row in frame[frame["engine_id"].isin(engines)].itertuples(index=False)
        }
        offsets = [position + (rung_index - 0.5) * width for position in positions]
        left.bar(
            offsets,
            [float(str(values[engine].oracle_accepted_edits)) for engine in engines],
            width=width,
            color=color,
            label=rung,
        )
        right.bar(
            offsets,
            [float(str(values[engine].oracle_safe_coverage)) for engine in engines],
            width=width,
            color=color,
            label=rung,
        )
    left.axhline(_H2_EDITS_FLOOR, color="#a03020", linestyle="--", linewidth=1)
    left.text(
        0.02,
        _H2_EDITS_FLOOR,
        " H2 floor: 245 accepted edits",
        fontsize=6,
        color="#a03020",
        va="bottom",
    )
    right.axhline(_H2_COVERAGE_FLOOR, color="#a03020", linestyle="--", linewidth=1)
    right.text(
        0.02,
        _H2_COVERAGE_FLOOR,
        " H2 floor: 0.05 coverage",
        fontsize=6,
        color="#a03020",
        va="bottom",
    )
    left.set_ylabel("ORACLE accepted edits (perfect verifier)")
    left.set_title("Oracle repair opportunity vs frozen floor A")
    right.set_ylabel("ORACLE safe coverage (perfect verifier)")
    right.set_title("Oracle safe coverage vs frozen floor B")
    for axis in (left, right):
        axis.set_xticks(positions)
        axis.set_xticklabels(engines, rotation=20, ha="right", fontsize=7)
        axis.legend(fontsize=7)
    fig.suptitle("ORACLE opportunity — a ceiling, not a result", fontsize=10)
    return _finish(fig, spec)

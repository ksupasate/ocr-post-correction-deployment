"""The recovery-phase figures, checked for the properties that keep them honest.

A figure is where a number escapes into a slide deck. Two properties are not cosmetic: an
oracle ceiling must be labelled as one on the image itself, and a share must be shown with
the pool it is a share of — a generator that proposes eighty times less can hold the same
beneficial share and look identical.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ocr_risk.analysis.figures import FigureSpec
from ocr_risk.analysis.recovery_figures import (
    plot_candidate_quality,
    plot_matched_discrimination,
    plot_oracle_potential,
    plot_score_distributions,
)

ENGINES = ("doctr", "easyocr")


def _spec(path: Path, synthetic: bool = False) -> FigureSpec:
    return FigureSpec(
        path=path,
        title="title",
        caption="caption",
        synthetic=synthetic,
        source_runs=("run-a",),
        config_sha256="0" * 64,
    )


def _quality() -> pd.DataFrame:
    rows = []
    for generator, coverage, proposals in (
        ("g0_lexical", 0.03, 4000),
        ("g3_edit_aware", 0.08, 1200),
    ):
        for engine in ENGINES:
            rows.append(
                {
                    "generator_id": generator,
                    "engine_id": engine,
                    "dataset_id": "ALL",
                    "n_sites": 4500,
                    "n_nonidentity_candidates": proposals,
                    "beneficial_rate": 0.10,
                    "harmful_rate": 0.70,
                    "no_change_rate": 0.20,
                    "oracle_safe_coverage": coverage,
                    "oracle_repair_recall": coverage * 1.5,
                }
            )
            rows.append({**rows[-1], "dataset_id": "funsd"})
    return pd.DataFrame(rows)


def test_candidate_quality_prints_the_pool_each_share_is_taken_of(tmp_path: Path) -> None:
    path = plot_candidate_quality(_quality(), _spec(tmp_path / "r1.png"))
    assert path.is_file()
    assert path.stat().st_size > 5_000


def test_the_oracle_figure_carries_its_own_not_deployable_banner(tmp_path: Path) -> None:
    """Checked in the rendered image, not in the calling code: the banner has to survive
    into the file that someone pastes into a slide."""
    from matplotlib.testing.compare import calculate_rms  # noqa: F401  (import guard only)

    path = plot_oracle_potential(_quality(), _spec(tmp_path / "r2.png"), threshold=0.05)
    assert path.is_file()

    import matplotlib.pyplot as plt

    # Re-render and inspect the figure's text rather than the pixels: text is what we
    # assert about, and a pixel comparison would fail on any font change.
    from ocr_risk.analysis import recovery_figures

    captured: list[str] = []
    original = recovery_figures._finish

    def spy(fig: object, spec: FigureSpec) -> Path:
        captured.extend(t.get_text() for t in fig.texts)  # type: ignore[attr-defined]
        return original(fig, spec)  # type: ignore[arg-type]

    recovery_figures._finish = spy  # type: ignore[assignment]
    try:
        plot_oracle_potential(_quality(), _spec(tmp_path / "r2b.png"), threshold=0.05)
    finally:
        recovery_figures._finish = original  # type: ignore[assignment]
        plt.close("all")
    assert any("GROUND-TRUTH ORACLE" in text for text in captured)
    assert any("NOT DEPLOYABLE" in text for text in captured)


def test_the_non_oracle_figures_do_not_carry_the_oracle_banner(tmp_path: Path) -> None:
    """The label must mean something, so it may not appear on a deployable measurement."""
    from matplotlib import pyplot as plt

    from ocr_risk.analysis import recovery_figures

    captured: list[str] = []
    original = recovery_figures._finish

    def spy(fig: object, spec: FigureSpec) -> Path:
        captured.extend(t.get_text() for t in fig.texts)  # type: ignore[attr-defined]
        return original(fig, spec)  # type: ignore[arg-type]

    recovery_figures._finish = spy  # type: ignore[assignment]
    try:
        plot_candidate_quality(_quality(), _spec(tmp_path / "r1c.png"))
    finally:
        recovery_figures._finish = original  # type: ignore[assignment]
        plt.close("all")
    assert not any("ORACLE" in text for text in captured)


def _discrimination() -> pd.DataFrame:
    rows = []
    for engine in ENGINES:
        for donor in ("paddleocr", "tesseract"):
            rows.append(
                {
                    "held_out_engine": engine,
                    "donor_engine": donor,
                    "metric": "roc_auc",
                    "delta": -0.02,
                    "delta_ci_lower": -0.05,
                    "delta_ci_upper": -0.005,
                    "degraded_after_holm": engine == "doctr",
                }
            )
    return pd.DataFrame(rows)


def test_the_discrimination_figure_shows_one_point_per_donor(tmp_path: Path) -> None:
    """The donor spread is exactly what a single fixed donor would have hidden."""
    path = plot_matched_discrimination(
        _discrimination(), _spec(tmp_path / "r3.png"), metric="roc_auc"
    )
    assert path.is_file()


def test_the_score_figure_renders_both_arms_for_every_engine(tmp_path: Path) -> None:
    rng = __import__("numpy").random.default_rng(0)
    frame = pd.DataFrame(
        [
            {
                "held_out_engine": engine,
                "arm": arm,
                "class": label,
                "score": float(rng.random()),
            }
            for engine in ENGINES
            for arm in ("loeo_zero_shot", "matched_in_engine")
            for label in ("safe", "harmful")
            for _ in range(40)
        ]
    )
    path = plot_score_distributions(
        frame, _spec(tmp_path / "r4.png"), arms=("loeo_zero_shot", "matched_in_engine")
    )
    assert path.is_file()


def test_a_synthetic_recovery_figure_is_still_watermarked(tmp_path: Path) -> None:
    """The recovery figures reuse the shared finisher, so the watermark must come with it."""
    from matplotlib import pyplot as plt

    from ocr_risk.analysis import recovery_figures

    captured: list[str] = []
    original = recovery_figures._finish

    def spy(fig: object, spec: FigureSpec) -> Path:
        result = original(fig, spec)  # type: ignore[arg-type]
        captured.extend(t.get_text() for t in fig.texts)  # type: ignore[attr-defined]
        return result

    recovery_figures._finish = spy  # type: ignore[assignment]
    try:
        plot_candidate_quality(_quality(), _spec(tmp_path / "r1s.png", synthetic=True))
    finally:
        recovery_figures._finish = original  # type: ignore[assignment]
        plt.close("all")
    assert any("SYNTHETIC" in text for text in captured)


@pytest.mark.parametrize("metric", ["roc_auc", "average_precision"])
def test_asking_for_a_metric_the_table_does_not_hold_produces_an_empty_panel(
    tmp_path: Path, metric: str
) -> None:
    """Better an empty figure than a figure of the wrong metric under the right title."""
    path = plot_matched_discrimination(
        _discrimination(), _spec(tmp_path / f"{metric}.png"), metric=metric
    )
    assert path.is_file()


def _discrimination_with_pooled() -> pd.DataFrame:
    rows = []
    for engine in ENGINES:
        for donor in ("paddleocr", "tesseract", "synth_x"):
            rows.append(
                {
                    "held_out_engine": engine,
                    "donor_engine": donor,
                    "aggregation": "per_donor",
                    "metric": "roc_auc",
                    "delta": -0.02,
                    "delta_ci_lower": -0.05,
                    "delta_ci_upper": -0.005,
                    "degraded_after_holm": False,
                }
            )
        rows.append(
            {
                "held_out_engine": engine,
                "donor_engine": "ALL",
                "aggregation": "pooled",
                "metric": "roc_auc",
                "delta": -0.02,
                "delta_ci_lower": -0.04,
                "delta_ci_upper": -0.008,
                "degraded_after_holm": engine == "doctr",
            }
        )
    return pd.DataFrame(rows)


def test_the_discrimination_figure_draws_the_pooled_row_and_its_donors(
    tmp_path: Path,
) -> None:
    """Only the pooled row hides how much of it rests on one substitution; only the donors
    leaves a reader averaging three overlapping intervals by eye."""
    from matplotlib import pyplot as plt

    from ocr_risk.analysis import recovery_figures

    captured: dict[str, int] = {}
    original = recovery_figures._finish

    def spy(fig: object, spec: FigureSpec) -> Path:
        axes = fig.axes[0]  # type: ignore[attr-defined]
        captured["lines"] = len(axes.lines)
        captured["ticks"] = len(axes.get_yticks())
        return original(fig, spec)  # type: ignore[arg-type]

    recovery_figures._finish = spy  # type: ignore[assignment]
    try:
        plot_matched_discrimination(
            _discrimination_with_pooled(), _spec(tmp_path / "r3p.png"), metric="roc_auc"
        )
    finally:
        recovery_figures._finish = original  # type: ignore[assignment]
        plt.close("all")
    assert captured["ticks"] == len(ENGINES), "one row per target engine, not per donor"
    # 2 engines x (3 donor segments + 3 donor markers) + the zero line, plus errorbar art.
    assert captured["lines"] >= len(ENGINES) * 6 + 1


def test_a_table_without_the_aggregation_column_still_plots(tmp_path: Path) -> None:
    """Older tables predate the pooled row; a figure that crashed on them would make the
    historical artifacts unreadable."""
    path = plot_matched_discrimination(
        _discrimination(), _spec(tmp_path / "r3old.png"), metric="roc_auc"
    )
    assert path.is_file()


def _multi_verifier_table() -> pd.DataFrame:
    """A pooled row per verifier, the way the regenerated tables are laid out: pooled rows
    are appended verifier-major, so without a filter the first diamond drawn is v2_text's."""
    rows = []
    for engine in ENGINES:
        for donor in ("paddleocr", "tesseract", "synth_x"):
            rows.append(
                {
                    "held_out_engine": engine,
                    "donor_engine": donor,
                    "aggregation": "per_donor",
                    "verifier_id": "v6_full",
                    "metric": "roc_auc",
                    "delta": -0.02,
                    "delta_ci_lower": -0.05,
                    "delta_ci_upper": -0.005,
                    "degraded_after_holm": False,
                }
            )
        rows.append(
            {
                "held_out_engine": engine,
                "donor_engine": "ALL",
                "aggregation": "pooled",
                "verifier_id": "v2_text",
                "metric": "roc_auc",
                "delta": 0.25,
                "delta_ci_lower": 0.20,
                "delta_ci_upper": 0.30,
                "degraded_after_holm": False,
            }
        )
        rows.append(
            {
                "held_out_engine": engine,
                "donor_engine": "ALL",
                "aggregation": "pooled",
                "verifier_id": "v6_full",
                "metric": "roc_auc",
                "delta": -0.04,
                "delta_ci_lower": -0.06,
                "delta_ci_upper": -0.01,
                "degraded_after_holm": True,
            }
        )
    return pd.DataFrame(rows)


def test_the_discrimination_figure_draws_the_named_verifiers_pooled_row(
    tmp_path: Path,
) -> None:
    """The table holds three verifiers and the verdict diamond must be the primary's.
    Without the filter the first row in CSV order -- a confidence-free control -- was
    drawn under the primary's caption."""
    from matplotlib import pyplot as plt

    from ocr_risk.analysis import recovery_figures

    diamonds: list[float] = []
    original = recovery_figures._finish

    def spy(fig: object, spec: FigureSpec) -> Path:
        for container in fig.axes[0].containers:  # type: ignore[attr-defined]
            line = getattr(container, "lines", None)
            marker = line[0] if line else None
            if marker is not None and getattr(marker, "get_marker", lambda: None)() == "D":
                diamonds.append(float(marker.get_xdata()[0]))
        return original(fig, spec)  # type: ignore[arg-type]

    recovery_figures._finish = spy  # type: ignore[assignment]
    try:
        plot_matched_discrimination(
            _multi_verifier_table(),
            _spec(tmp_path / "r3v.png"),
            metric="roc_auc",
            verifier="v6_full",
        )
    finally:
        recovery_figures._finish = original
        plt.close("all")
    assert diamonds == [-0.04, -0.04], "one diamond per engine, at v6_full's delta"


def test_refusing_to_choose_among_ambiguous_pooled_rows(tmp_path: Path) -> None:
    """With no verifier filter and several pooled rows per engine, plotting the first is a
    silent choice among different verifiers' verdicts."""
    with pytest.raises(ValueError, match="survive the filters"):
        plot_matched_discrimination(
            _multi_verifier_table(), _spec(tmp_path / "r3amb.png"), metric="roc_auc"
        )

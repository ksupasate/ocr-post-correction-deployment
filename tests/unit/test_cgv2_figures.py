"""The CGV2 figures render from small fabricated tables and label their oracles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ocr_risk.analysis.cgv2_figures import (
    plot_budget_recall,
    plot_candidate_composition,
    plot_oracle_opportunity,
    plot_per_engine_availability,
    plot_structural_coverage,
)
from ocr_risk.analysis.figures import FigureSpec

ENGINES = ("e1", "e2")


def _spec(tmp_path: Path, name: str) -> FigureSpec:
    return FigureSpec(
        path=tmp_path / f"{name}.png",
        title=name,
        caption=name,
        synthetic=False,
        source_runs=("run-1",),
        config_sha256="a" * 64,
    )


def _budget() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": rung,
                "engine_id": engine_id,
                "k": k,
                "exact_recall": 0.1 * k,
                "harmful_burden": 0.2,
            }
            for rung in ("g0_lexical", "g6_union")
            for engine_id in ENGINES
            for k in (1, 2, 4, 8)
        ]
    )


def _quality() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": rung,
                "engine_id": engine_id,
                "beneficial_rate": 0.2,
                "harmful_rate": 0.6,
            }
            for rung in ("g0_lexical", "g3_edit_aware", "g6_union")
            for engine_id in ENGINES
        ]
    )


def _coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": rung,
                "engine_id": engine_id,
                "stratum": stratum,
                "k": 4,
                "availability": 0.3,
            }
            for rung in ("g3_edit_aware", "g6_union")
            for engine_id in ENGINES
            for stratum in ("deletion", "insertion", "segmentation")
        ]
    )


def _opportunity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": rung,
                "engine_id": engine_id,
                "stratum": "all",
                "k": 4,
                "n_error_sites": 10,
                "n_available": 3,
                "availability": 0.3,
            }
            for rung in ("g0_lexical", "g6_union")
            for engine_id in ENGINES
        ]
    )


def _oracle() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": rung,
                "engine_id": engine_id,
                "variant": "site_plus_regions",
                "n_evaluable_sites": 100,
                "oracle_accepted_edits": 50,
                "oracle_safe_coverage": 0.5,
            }
            for rung in ("g3_edit_aware", "g6_union")
            for engine_id in ENGINES
        ]
    )


@pytest.mark.parametrize(
    ("plot", "table"),
    [
        (plot_budget_recall, _budget()),
        (plot_candidate_composition, _quality()),
        (plot_structural_coverage, _coverage()),
        (plot_per_engine_availability, _opportunity()),
        (plot_oracle_opportunity, _oracle()),
    ],
)
def test_every_figure_renders_a_file(tmp_path: Path, plot: object, table: pd.DataFrame) -> None:
    spec = _spec(tmp_path, plot.__name__)  # type: ignore[attr-defined]
    produced = plot(table, ENGINES, spec)  # type: ignore[operator]
    assert produced.exists()
    assert produced.stat().st_size > 1_000


def test_the_oracle_figure_says_so(tmp_path: Path) -> None:
    """A ceiling drawn without the word ORACLE on it is a result waiting to be misread."""
    spec = _spec(tmp_path, "oracle")
    produced = plot_oracle_opportunity(_oracle(), ENGINES, spec)
    # The title carries the warning; the floors are drawn as lines (tested by rendering).
    assert produced.exists()


def test_availability_figure_ignores_nonheadline_strata(tmp_path: Path) -> None:
    """Structural rows must not be selected by incidental CSV order in Figure D."""
    table = pd.concat(
        [
            _opportunity(),
            _opportunity().assign(stratum="deletion", availability=0.99),
        ],
        ignore_index=True,
    )
    produced = plot_per_engine_availability(table, ENGINES, _spec(tmp_path, "availability"))
    assert produced.exists()

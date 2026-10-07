"""The RH1 gate CLI, on fabricated table families.

The gate's verdict rules live behind a command that reads whatever CSVs happen to be on
disk beside the primary table. A mixed-freshness window -- fresh primary, stale
per-donor-only sensitivity tables -- was read as a measured policy disagreement, and the
disagreement downgrade itself was a no-op from SUPPORTED. Both are only reachable through
the command, so both are tested through the command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ocr_risk.cli.cmd_analyze import rh1_gate

ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")
DONORS = ("doctr", "easyocr", "paddleocr", "tesseract")


def _family(
    directory: Path,
    degraded_by_policy: dict[str, set[str]],
    pooled: bool = True,
) -> Path:
    """Write a primary table plus one sensitivity table per non-primary policy.

    ``degraded_by_policy`` maps ``{"": {engines}}`` for the primary and policy names for
    the sensitivity tables. Per-donor rows are always written not-degraded, which is what
    the real tables carry.
    """
    rows: list[dict[str, object]] = []
    for policy, degraded in degraded_by_policy.items():
        suffix = "" if policy == "" else f"__{policy}"
        for engine in ENGINES:
            for donor in DONORS:
                if donor == engine:
                    continue
                rows.append(
                    {
                        "held_out_engine": engine,
                        "donor_engine": donor,
                        "aggregation": "per_donor",
                        "verifier_id": "v6_full",
                        "metric": "roc_auc",
                        "loeo_zero_shot": 0.7,
                        "matched_in_engine": 0.71,
                        "delta": -0.01,
                        "delta_ci_lower": -0.03,
                        "delta_ci_upper": 0.01,
                        "holm_adjusted_p": float("nan"),
                        "degraded_after_holm": False,
                        "minimum_detectable_effect": 0.02,
                        "n_candidates": 100,
                        "n_documents": 50,
                        "base_rate_safe": 0.14,
                        "degenerate_interval": False,
                    }
                )
            if pooled:
                rows.append(
                    {
                        "held_out_engine": engine,
                        "donor_engine": "ALL",
                        "aggregation": "pooled",
                        "verifier_id": "v6_full",
                        "metric": "roc_auc",
                        "loeo_zero_shot": 0.7,
                        "matched_in_engine": 0.71,
                        "delta": -0.02,
                        "delta_ci_lower": -0.04,
                        "delta_ci_upper": -0.005,
                        "holm_adjusted_p": 0.01,
                        "degraded_after_holm": engine in degraded,
                        "minimum_detectable_effect": 0.02,
                        "n_candidates": 100,
                        "n_documents": 50,
                        "base_rate_safe": 0.14,
                        "degenerate_interval": False,
                    }
                )
        pd.DataFrame(rows).to_csv(directory / f"rh1_discrimination{suffix}.csv", index=False)
        rows.clear()
    return directory / "rh1_discrimination.csv"


def _verdict(directory: Path) -> dict[str, object]:
    payload = json.loads((directory / "rh1_gate.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_disagreement_from_supported_lands_in_partially(tmp_path: Path) -> None:
    """All four engines degraded on the primary, three on the sensitivity: the
    pre-committed rule must not leave a SUPPORTED verdict standing on disagreeing
    policies."""
    primary = _family(
        tmp_path,
        {"": set(ENGINES), "non_improving": {"doctr", "easyocr", "paddleocr"}},
    )
    rh1_gate(table_path=primary, verifier="v6_full", out=tmp_path)
    payload = _verdict(tmp_path)
    assert payload["verdict"] == "PARTIALLY SUPPORTED"
    assert payload["context"]["harm_policy_disagreement"] is True
    # The criterion reports the true count, not a fabricated one.
    assert payload["criteria"][0]["observed"] == "4 of 4 engines"
    assert any("harm policies disagree" in note for note in payload["notes"])


def test_agreement_keeps_the_count_verdict(tmp_path: Path) -> None:
    primary = _family(
        tmp_path,
        {"": set(ENGINES), "non_improving": set(ENGINES), "exact_only": set(ENGINES)},
    )
    rh1_gate(table_path=primary, verifier="v6_full", out=tmp_path)
    assert _verdict(tmp_path)["verdict"] == "DISCRIMINATION DEGRADATION SUPPORTED"


def test_a_sensitivity_table_without_pooled_rows_is_unmeasurable_not_zero(tmp_path: Path):
    """During the regeneration window a stale sensitivity table holds only per-donor
    rows. Reading that as '0 engines degraded' manufactured a disagreement and could
    wrongly downgrade a NOT SUPPORTED verdict."""
    primary = _family(tmp_path, {"": set(), "non_improving": set()}, pooled=True)
    # Overwrite one sensitivity table with a per-donor-only frame: the mixed-freshness
    # state the disk can genuinely be in.
    stale = pd.read_csv(primary)
    pd.DataFrame(stale[stale["aggregation"] == "per_donor"]).to_csv(
        tmp_path / "rh1_discrimination__non_improving.csv", index=False
    )
    rh1_gate(table_path=primary, verifier="v6_full", out=tmp_path)
    payload = _verdict(tmp_path)
    assert payload["verdict"] == "INCONCLUSIVE"
    agreement = next(c for c in payload["criteria"] if c["key"] == "harm_policy_agreement")
    assert agreement["measurable"] is False
    assert "no pooled rows in: non_improving" in agreement["observed"]


def test_a_degenerate_pooled_interval_makes_the_engine_unmeasurable(tmp_path: Path) -> None:
    primary = _family(tmp_path, {"": {"doctr"}, "exact_only": {"doctr"}})
    table = pd.read_csv(primary)
    table.loc[
        (table["aggregation"] == "pooled") & (table["held_out_engine"] == "easyocr"),
        "degenerate_interval",
    ] = True
    table.to_csv(primary, index=False)
    rh1_gate(table_path=primary, verifier="v6_full", out=tmp_path)
    payload = _verdict(tmp_path)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["context"]["engines_unmeasurable"] == ["easyocr"]


def test_secondary_criteria_count_pooled_rows_only(tmp_path: Path) -> None:
    """The per-donor rows carry degraded_after_holm=False by construction; folding them
    into an `.all()` forced every secondary endpoint to read '0 of 4 engines' whatever
    the data said."""
    primary = _family(tmp_path, {"": set(), "non_improving": set()})
    table = pd.read_csv(primary)
    extra = table[(table["aggregation"] == "pooled")].assign(
        metric="average_precision", degraded_after_holm=True
    )
    pd.concat([table, extra], ignore_index=True).to_csv(primary, index=False)
    rh1_gate(table_path=primary, verifier="v6_full", out=tmp_path)
    payload = _verdict(tmp_path)
    secondary = next(c for c in payload["criteria"] if c["key"] == "secondary_average_precision")
    assert secondary["observed"] == "4 of 4 engines"

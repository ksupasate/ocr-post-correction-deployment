#!/usr/bin/env python3
"""Replay publication Figures 2-4 from registered VF2 metrics, without evaluation.

No models, candidate tables, outcome labels, or experiment modules are loaded.
The only aggregation is the frozen median of already registered per-draw metrics.
Missing bootstrap intervals remain missing. Any source mismatch stops rendering.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOC = Path("docs/paper3/journal_track_2027/validity_recovery")
RUN = Path("results/paper3_vf2/run1")
REGISTRY = DOC / "P3_VF2_RESULT_REGISTRY.json"
DIRECTIONS = {"current_to_qwen": "TXT→VLM", "qwen_to_current": "VLM→TXT"}
BUDGETS = ("0", "5", "10", "25", "50", "100", "pool")
PRIMARY = "strict_worsening"
STRICT = "non_improving"
# Mathematical reference, not an observed scientific result.
CHANCE_AUROC = 0.5
POLICIES = {
    "plug_in": ("Plug-in rule", "o", "#0072B2"),
    "conservative": ("Pooled conservative", "s", "#444444"),
    "navarro_adapted": ("Adapted Navarro-Cerdán", "^", "#D55E00"),
    "conservative::C1": ("Corpus-specific", "D", "#667733"),
    "conservative::C3@prior=5": ("Partially pooled", "P", "#885588"),
}
VERDICTS = {
    "USEFUL": ("Useful", "#dce7ea"),
    "VIOLATING": ("Violating", "#f1ddd0"),
    "NOT_WORKING": ("Not working", "#eeeeee"),
    "DEPLOYABLE_NOT_USEFUL": ("Deployable but\nnot useful", "#e9e4d5"),
    "NOT_ESTIMABLE": ("Not estimable", "#ffffff"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"VF2 FIGURE GATE FAILED: {message}")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    """Archives lack Git metadata; file hashes remain the provenance authority."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read(path: Path) -> Any:
    return json.loads((ROOT / path).read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def arm(budget: str) -> str:
    return "a0_zero_shot" if budget == "0" else "m1_full_adaptation"


class Evidence:
    """Resolve actual registered records and retain exact JSON locations."""

    def __init__(self) -> None:
        self.delivery = read(Path("results/paper3_vf2/P3_VF2_FINAL_DELIVERY_MANIFEST.json"))
        self.freeze = read(DOC / "P3_VF2_EXECUTION_FREEZE.json")
        self.result_freeze = read(RUN / "manifests/result_freeze.json")
        self.sources: dict[str, str] = {}
        paths = [
            REGISTRY,
            DOC / "P3_VF2_CONFIG.json",
            DOC / "P3_VF2_PROTOCOL.md",
            DOC / "P3_VF2_EXECUTION_REGISTRY.json",
            DOC / "P3_VF2_DISCLOSURE_RECORD.md",
            DOC / "P3_VF2_RESULTS.md",
            DOC / "P3_VF2_RESULT_TABLES.md",
            DOC / "P3_VF2_CLAIM_AUDIT.md",
            DOC / "P3_VF2_EXECUTION_FREEZE.json",
            RUN / "tables/scientific_results.json",
            RUN / "tables/ranking_rows.csv",
            RUN / "tables/boundary_rows.csv",
            RUN / "tables/support_rows.csv",
            RUN / "policies/policy_rows.csv",
            RUN / "manifests/model_registry.json",
            RUN / "manifests/resource_manifest.json",
            RUN / "manifests/postrun_integrity.json",
            Path("scripts/paper3_vf2_analysis.py"),
            Path("scripts/paper3_vf2_clean_execution.py"),
            Path("scripts/paper3_rc1_reviewer_closure.py"),
        ]
        expected = (
            self.freeze["files_sha256"]
            | self.result_freeze["files_sha256"]
            | self.delivery["docs_sha256"]
        )
        for path in paths:
            require((ROOT / path).is_file(), f"missing {path}")
            digest = sha(ROOT / path)
            if str(path) in expected:
                require(digest == expected[str(path)], f"frozen hash mismatch: {path}")
            else:
                require(path.name in {"postrun_integrity.json"}, f"no authoritative hash: {path}")
            self.sources[str(path)] = digest
        self.registry = read(REGISTRY)
        self.entries = {x["key"]: (i, x) for i, x in enumerate(self.registry["entries"])}
        require(len(self.entries) == len(self.registry["entries"]), "duplicate aggregate keys")
        self.draws = self.registry["draw_records"]
        self.science = read(RUN / "tables/scientific_results.json")
        self.config = read(DOC / "P3_VF2_CONFIG.json")
        self.execution = read(DOC / "P3_VF2_EXECUTION_REGISTRY.json")
        self.models = {
            m["model_id"]: m for m in read(RUN / "manifests/model_registry.json")["models"]
        }
        self.audit: list[dict[str, Any]] = []
        self.checks: dict[str, Any] = {}
        self.serialization_differences: list[dict[str, Any]] = []
        # Independently verify the registered draw values against their frozen CSVs.
        self.csv_records: dict[str, dict[tuple[Any, ...], dict[str, str]]] = {}
        for path, tail in [
            (RUN / "tables/ranking_rows.csv", "unit"),
            (RUN / "tables/boundary_rows.csv", None),
        ]:
            with (ROOT / path).open(newline="") as f:
                rows = list(csv.DictReader(f))
            fields = ("model_id", "harm_definition", "population") + ((tail,) if tail else ())
            self.csv_records[str(path)] = {tuple(row[k] for k in fields): row for row in rows}

    def prefix(self, direction: str, budget: str, harm: str, pop: str, policy: str) -> str:
        return "|".join((direction, arm(budget), budget, harm, pop, policy))

    def entry(self, prefix: str, metric: str) -> tuple[Any, str, str]:
        key = prefix + "|" + metric
        require(key in self.entries, f"aggregate record not found: {key}")
        i, row = self.entries[key]
        require(row["source_artifact"].startswith(str(RUN) + "/"), f"non-VF2 source: {key}")
        return row["value"], key, f"$.entries[{i}].value"

    def add(
        self,
        figure: int,
        panel: str,
        direction: str,
        budget: str,
        harm: str,
        pop: str,
        level: str,
        policy: str,
        metric: str,
        value: Any,
        key: str | list[str],
        json_path: str | list[str],
        *,
        ci_low: float | None = None,
        ci_high: float | None = None,
        interval_keys: list[str] | None = None,
        plot_status: str = "plotted",
        aggregation: str = "registered aggregate",
        notes: str = "",
        model_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        row = {
            "figure": f"Fig{figure}",
            "panel": panel,
            "direction": direction,
            "budget": budget,
            "population": pop,
            "harm_definition": harm,
            "ranking_level": level,
            "policy": policy,
            "metric": metric,
            "value": value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "source": str(REGISTRY),
            "key": key,
            "json_path": json_path,
            "interval_keys": interval_keys or [],
            "plot_status": plot_status,
            "aggregation": aggregation,
            "notes": notes,
            "model_ids": model_ids or [],
        }
        self.audit.append(row)
        return row

    def draw_values(
        self, direction: str, budget: str, harm: str, pop: str, unit: str, field: str
    ) -> tuple[float, list[str], list[str], list[str]]:
        selected = [
            (i, x)
            for i, x in enumerate(self.draws)
            if x["direction"] == direction
            and str(x["budget"]) == budget
            and x["harm_definition"] == harm
            and x["population"] == pop
            and x["value"].get("arm") == arm(budget)
            and (
                x["value"].get("unit") == unit
                if unit != "boundary"
                else x["value"].get("analysis_only") is True
            )
        ]
        selected.sort(key=lambda ix: ix[1]["draw"])
        count = 1 if budget == "0" else self.config["study_defined_operating_criteria"]["draws"]
        require(
            [x["draw"] for _, x in selected] == list(range(count)),
            f"incomplete predetermined draws: {direction}, {budget}, {harm}, {pop}, {unit}",
        )
        values, keys, locations, models = [], [], [], []
        for i, x in selected:
            value = x["value"][field]
            require(
                value is not None and math.isfinite(value), f"unavailable draw value: {x['key']}"
            )
            source = x["source_artifact"]
            require(source in self.csv_records, f"unexpected draw source: {source}")
            csv_key = (x["model_id"], harm, pop) + ((unit,) if unit != "boundary" else ())
            csv_value = float(self.csv_records[source][csv_key][field])
            require(
                math.isclose(csv_value, value, rel_tol=0, abs_tol=1e-15),
                f"CSV/registry conflict: {x['key']}.{field}",
            )
            if csv_value != value:
                self.serialization_differences.append(
                    {
                        "key": x["key"],
                        "field": field,
                        "registry_value": value,
                        "csv_value": csv_value,
                        "absolute_difference": abs(value - csv_value),
                    }
                )
            values.append(value)
            keys.append(x["key"])
            locations.append(f"$.draw_records[{i}].value.{field}")
            models.append(x["model_id"])
        return statistics.median(values), keys, locations, models

    def ranking(
        self, figure: int, panel: str, d: str, b: str, h: str, pop: str, unit: str
    ) -> dict[str, Any]:
        median, draw_keys, locations, model_ids = self.draw_values(d, b, h, pop, unit, "harm_auroc")
        p = self.prefix(d, b, h, pop, unit)
        if p + "|median" in self.entries:
            value, key, loc = self.entry(p, "median")
            require(
                math.isclose(value, median, rel_tol=0, abs_tol=1e-15),
                f"registered/per-draw median conflict: {p}",
            )
            low, lk, _ = self.entry(p, "ci_low")
            high, hk, _ = self.entry(p, "ci_high")
            require(
                self.science["auroc_summaries"][p]
                == {"median": value, "ci_low": low, "ci_high": high},
                f"native summary/registry conflict: {p}",
            )
            return self.add(
                figure,
                panel,
                d,
                b,
                h,
                pop,
                unit,
                "",
                "harm_auroc",
                value,
                key,
                loc,
                ci_low=low,
                ci_high=high,
                interval_keys=[lk, hk],
                model_ids=model_ids,
            )
        return self.add(
            figure,
            panel,
            d,
            b,
            h,
            pop,
            unit,
            "",
            "harm_auroc",
            median,
            draw_keys,
            locations,
            aggregation="median of registered per-draw AUROCs",
            notes="Grouped interval was not computed for this budget; omitted.",
            model_ids=model_ids,
        )

    def policy(
        self,
        figure: int,
        panel: str,
        d: str,
        h: str,
        pop: str,
        rule: str,
        metric: str,
        status: str = "plotted",
    ) -> dict[str, Any]:
        p = self.prefix(d, "pool", h, pop, rule)
        value, key, loc = self.entry(p, metric)
        summary = self.science["policy_summaries"][p]
        require(summary[metric] == value, f"policy/native registry conflict: {key}")
        low = high = None
        interval_keys: list[str] = []
        ci_name = {
            "median_acceptance_coverage": "median_coverage",
            "median_harm": "median_harm",
        }.get(metric)
        if ci_name and "intervals" in summary:
            low, lk, _ = self.entry(p, f"intervals.{ci_name}_ci_low")
            high, hk, _ = self.entry(p, f"intervals.{ci_name}_ci_high")
            require(
                low == summary["intervals"][ci_name + "_ci_low"]
                and high == summary["intervals"][ci_name + "_ci_high"],
                f"policy interval conflict: {p}",
            )
            interval_keys = [lk, hk]
        notes = (
            "Harm median/interval conditional on draws applying edits; "
            "coverage median includes all draws."
            if metric == "median_harm"
            else ""
        )
        return self.add(
            figure,
            panel,
            d,
            "pool",
            h,
            pop,
            "winner",
            rule,
            metric,
            value,
            key,
            loc,
            ci_low=low,
            ci_high=high,
            interval_keys=interval_keys,
            plot_status=status,
            notes=notes,
            model_ids=self.entries[key][1]["model_IDs"],
        )

    def load_figure_data(self) -> None:
        for panel, d in zip(("a", "b"), DIRECTIONS, strict=True):
            for b in BUDGETS:
                for unit in ("all", "winner"):
                    self.ranking(2, panel, d, b, PRIMARY, "original", unit)
            for rule in POLICIES:
                cov = self.policy(
                    3, panel, d, PRIMARY, "original", rule, "median_acceptance_coverage"
                )
                harm = self.policy(3, panel, d, PRIMARY, "original", rule, "median_harm")
                no_action = cov["value"] == 0 or harm["value"] is None
                if no_action:
                    cov["plot_status"] = harm["plot_status"] = "margin_no_action_at_median"
                self.policy(
                    3,
                    panel,
                    d,
                    PRIMARY,
                    "original",
                    rule,
                    "accepting_draws",
                    "margin" if no_action else "support",
                )
                self.policy(3, panel, d, PRIMARY, "original", rule, "draws", "support")
                for metric in ("verdict", "median_recall", "median_credited_recall"):
                    self.policy(3, panel, d, PRIMARY, "original", rule, metric, "support")
            for metric in ("coverage", "harm"):
                val, keys, locations, models = self.draw_values(
                    d, "pool", PRIMARY, "original", "boundary", metric
                )
                self.add(
                    3,
                    panel,
                    d,
                    "pool",
                    PRIMARY,
                    "original",
                    "winner",
                    "boundary",
                    metric,
                    val,
                    keys,
                    locations,
                    aggregation="median of registered boundary draw metrics",
                    notes="Analysis only; no grouped boundary interval was computed.",
                    model_ids=models,
                )
            for harm, pop in ((PRIMARY, "original"), (STRICT, "original"), (PRIMARY, "matched")):
                self.ranking(4, panel, d, "pool", harm, pop, "winner")
                bottom_panel = "c" if panel == "a" else "d"
                for rule in ("plug_in", "conservative", "navarro_adapted"):
                    self.policy(4, bottom_panel, d, harm, pop, rule, "verdict")
        eps = self.config["study_defined_operating_criteria"]["epsilon"]
        self.audit.append(
            {
                "figure": "Fig3",
                "panel": "a,b",
                "direction": "both",
                "budget": "pool",
                "population": "original",
                "harm_definition": PRIMARY,
                "ranking_level": "winner",
                "policy": "study criterion",
                "metric": "epsilon",
                "value": eps,
                "ci_low": None,
                "ci_high": None,
                "source": str(DOC / "P3_VF2_CONFIG.json"),
                "key": "study_defined_operating_criteria.epsilon",
                "json_path": "$.study_defined_operating_criteria.epsilon",
                "plot_status": "plotted",
            }
        )
        for d in DIRECTIONS:
            for role, metric in (("target_pool", "pages"), ("test", "pages"), ("test", "groups")):
                self.audit.append(
                    {
                        "figure": "all",
                        "panel": "support",
                        "direction": d,
                        "budget": "pool",
                        "population": "original",
                        "harm_definition": PRIMARY,
                        "ranking_level": "",
                        "policy": "",
                        "metric": f"{role}_{metric}",
                        "value": self.execution["support"][d][role][metric],
                        "ci_low": None,
                        "ci_high": None,
                        "source": str(DOC / "P3_VF2_EXECUTION_REGISTRY.json"),
                        "key": f"support.{d}.{role}.{metric}",
                        "json_path": f"$.support.{d}.{role}.{metric}",
                        "plot_status": "support",
                    }
                )

        for figure in ("Fig2", "Fig4"):
            self.audit.append(
                {
                    "figure": figure,
                    "panel": "a,b",
                    "direction": "both",
                    "budget": "all",
                    "population": "reference",
                    "harm_definition": "reference",
                    "ranking_level": "reference",
                    "policy": "mathematical reference",
                    "metric": "chance_auroc",
                    "value": CHANCE_AUROC,
                    "ci_low": None,
                    "ci_high": None,
                    "source": "scripts/plot_paper3_vf2_figures.py",
                    "key": "CHANCE_AUROC",
                    "json_path": "code: CHANCE_AUROC",
                    "plot_status": "plotted",
                    "notes": "Mathematical chance reference; not an estimated outcome.",
                }
            )

    def validate(self) -> None:
        for d, pages in zip(DIRECTIONS, (134, 131), strict=True):
            support = self.execution["support"][d]
            require(
                support["target_pool"]["pages"] == pages, f"incorrect corrected full pool for {d}"
            )
            require(
                support["test"]["pages"] == 66 and support["test"]["groups"] == 59,
                f"incorrect test support: {d}",
            )
        self.checks["corrected_pools_and_test_support"] = "PASS"
        require(self.execution["reserve_outcomes_inspected"] is False, "reserve exposure")
        post = read(RUN / "manifests/postrun_integrity.json")
        require(
            post["reserve_outcomes_read"] is False
            and post["test_group_overlap"] == 0
            and post["calibration_IN_FIT_fraction"] == 0,
            "postrun integrity conflict",
        )
        cells = {c["model_id"]: c for c in self.execution["cells"]}
        for model_id in sorted({m for row in self.audit for m in row.get("model_ids", [])}):
            c, m = cells[model_id], self.models[model_id]
            test, reserve = set(c["test_groups"]), set(c["reserve_confirmation_groups"])
            fit = set(m["source_fit_groups"]) | set(m["target_adapt_groups"])
            cal = set(m["calibration_groups"])
            resources = set(c["resource_fit_groups"])
            require(not test & (fit | cal | resources), f"test group crossing: {model_id}")
            require(not fit & cal, f"in-fit cutoff evidence: {model_id}")
            require(not reserve & (fit | cal | resources), f"reserve crossing: {model_id}")
            require(
                not set(c["target_adapt_groups"]) & set(c["target_calibrate_groups"]),
                "target role crossing",
            )
            require(
                m["calibration_OUT_OF_FIT_fraction"] == 1 and m["test_group_overlap"] == 0,
                f"model provenance invalid: {model_id}",
            )
            require(
                c["arm"] in {"a0_zero_shot", "m1_full_adaptation"},
                "legacy/comparator model enters figures",
            )
            require(
                sha(ROOT / m["score_path"]) == m["score_sha256"], f"modified scores: {model_id}"
            )
            self.sources[m["score_path"]] = m["score_sha256"]
            if str(c["budget"]) == "pool":
                require(
                    c["target_label_pages"]
                    == len(c["target_adapt_pages"]) + len(c["target_calibrate_pages"]),
                    f"budget accounting: {model_id}",
                )
        resource_checks = read(RUN / "manifests/resource_manifest.json")["checks"]
        test = set(next(iter(cells.values()))["test_groups"])
        reserve = set(next(iter(cells.values()))["reserve_confirmation_groups"])
        for resource in resource_checks:
            require(
                not set(resource["groups"]) & (test | reserve),
                f"resource crossing: {resource['environment']}",
            )
        self.checks["scientific_group_isolation_and_out_of_fit_cutoff_evidence"] = "PASS"
        # The frozen analysis filters BOTH winner populations before score_rule.
        # Confirm the dataflow from the actual hash-verified AST, without executing it.
        tree = ast.parse((ROOT / "scripts/paper3_vf2_analysis.py").read_text())
        evaluate = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate"
        )
        text = ast.get_source_segment(
            (ROOT / "scripts/paper3_vf2_analysis.py").read_text(), evaluate
        )
        require(
            "current_cal[current_cal.site_key.isin(matched)]" in text
            and "current_winners[current_winners.site_key.isin(matched)]" in text,
            "matched population lacks upstream cutoff restriction",
        )
        require(
            text.index("current_cal[current_cal.site_key.isin(matched)]")
            < text.index("rc.score_rule("),
            "matching occurs after cutoff selection",
        )
        require(
            not any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"fit", "fit_predict", "predict", "predict_proba"}
                for n in ast.walk(evaluate)
            ),
            "sensitivity refits or rescores models",
        )
        selected_draws = [
            x
            for x in self.draws
            if x["value"].get("arm") == "m1_full_adaptation"
            and str(x["budget"]) == "pool"
            and x["value"].get("unit") == "winner"
        ]
        for x in selected_draws:
            require(x["model_id"] == x["value"]["model_id"], "sensitivity changes model ID")
            d = x["direction"]
            model_id = x["model_id"]
            require(
                model_id in self.models and self.models[model_id]["direction"] == d,
                "unknown score model",
            )
            original = [
                y
                for y in selected_draws
                if y["model_id"] == model_id
                and y["harm_definition"] == PRIMARY
                and y["population"] == "original"
            ]
            require(len(original) == 1, "sensitivity not tied to original model/score artifact")
        self.checks["matched_cutoff_restriction_and_identical_models_scores"] = "PASS"
        self.checks["strict_harm_no_refitting_no_rescoring"] = "PASS"
        for row in self.audit:
            if row["metric"] == "verdict":
                require(row["value"] in VERDICTS, f"unknown verdict: {row['value']}")
            if row["metric"] == "harm_auroc":
                require(0 <= row["value"] <= 1, "invalid AUROC")
            if (
                row["figure"] == "Fig3"
                and row.get("plot_status") == "plotted"
                and row["metric"] == "median_harm"
            ):
                cov = next(
                    x
                    for x in self.audit
                    if x["figure"] == "Fig3"
                    and x["direction"] == row["direction"]
                    and x["policy"] == row["policy"]
                    and x["metric"] == "median_acceptance_coverage"
                )
                require(
                    cov["value"] > 0 and row["value"] is not None, "no action plotted as zero harm"
                )
            if row["source"] == str(REGISTRY):
                keys = row["key"] if isinstance(row["key"], list) else [row["key"]]
                require(
                    all("|a0_zero_shot|" in k or "|m1_full_adaptation|" in k for k in keys),
                    "unregistered legacy arm enters figures",
                )
        self.checks["no_action_has_no_risk_coordinate"] = "PASS"
        self.checks["only_final_registered_primary_models_no_legacy_outcomes"] = "PASS"
        # User-supplied rounded checks are validation targets, NEVER plot inputs.
        expected = {
            ("current_to_qwen", "all"): 0.9086,
            ("current_to_qwen", "winner"): 0.9086,
            ("qwen_to_current", "all"): 0.9190,
            ("qwen_to_current", "winner"): 0.9518,
        }
        for (d, unit), value in expected.items():
            require(
                abs(
                    self.entry(self.prefix(d, "pool", PRIMARY, "original", unit), "median")[0]
                    - value
                )
                < 0.00005,
                f"headline check conflict: {d}/{unit}",
            )
        for d, cov, harm in [
            ("current_to_qwen", 0.4227, 0.1399),
            ("qwen_to_current", 0.3680, 0.0656),
        ]:
            p = self.prefix(d, "pool", PRIMARY, "original", "plug_in")
            require(
                abs(self.entry(p, "median_acceptance_coverage")[0] - cov) < 0.00005
                and abs(self.entry(p, "median_harm")[0] - harm) < 0.00005,
                "plug-in check conflict",
            )
        p = self.prefix("qwen_to_current", "pool", PRIMARY, "original", "plug_in")
        eps = self.config["study_defined_operating_criteria"]["epsilon"]
        require(
            self.entry(p, "intervals.median_harm_ci_low")[0]
            < eps
            < self.entry(p, "intervals.median_harm_ci_high")[0],
            "VLM→TXT harm interval must cross epsilon",
        )
        self.checks["headline_checks_and_grouped_interval_crossing_epsilon"] = "PASS"
        for d, h, pop, target in [
            ("qwen_to_current", STRICT, "original", 0.7522),
            ("current_to_qwen", PRIMARY, "matched", 0.8682),
            ("qwen_to_current", PRIMARY, "matched", 0.9120),
        ]:
            p = self.prefix(d, "pool", h, pop, "winner")
            require(
                abs(self.entry(p, "median")[0] - target) < 0.00005,
                f"sensitivity check conflict: {p}",
            )
        self.checks["sensitivity_headline_checks"] = "PASS"


def select(data: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    rows = [row for row in data if all(row.get(k) == v for k, v in fields.items())]
    require(len(rows) == 1, f"plot record not unique: {fields}")
    return rows[0]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper3-final-vf2-figures",
            "savefig.facecolor": "white",
        }
    )


def axes_style(ax: Any, title: str) -> None:
    ax.set_title(title, loc="left", pad=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=3)
    ax.set_axisbelow(True)


def point_interval(ax: Any, x: float, row: dict[str, Any], color: str, marker: str) -> None:
    if row["ci_low"] is not None:
        ax.vlines(x, row["ci_low"], row["ci_high"], colors=color, lw=0.8, zorder=2)
        ax.hlines([row["ci_low"], row["ci_high"]], x - 0.04, x + 0.04, colors=color, lw=0.8)
    ax.plot(x, row["value"], marker=marker, color=color, ms=4.3, linestyle="none", zorder=5)


def figure2(data: list[dict[str, Any]]) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 78 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.24, top=0.85, wspace=0.15)
    for panel, (d, title), ax in zip(("a", "b"), DIRECTIONS.items(), axes, strict=True):
        axes_style(ax, f"({panel}) {title}")
        ax.axhline(
            select(data, figure="Fig2", metric="chance_auroc")["value"],
            color=".65",
            lw=0.7,
            ls=(0, (3, 3)),
        )
        for unit, color, marker, ls, size in [
            ("all", "#333333", "o", "-", 5.2),
            ("winner", "#0072B2", "D", "--", 3.2),
        ]:
            rows = [
                select(
                    data,
                    figure="Fig2",
                    direction=d,
                    budget=b,
                    ranking_level=unit,
                    metric="harm_auroc",
                )
                for b in BUDGETS
            ]
            ax.plot(
                range(len(rows)),
                [x["value"] for x in rows],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=size,
                markerfacecolor="white" if unit == "all" else color,
                markeredgewidth=0.9,
                zorder=4 if unit == "winner" else 3,
            )
            for x, row in enumerate(rows):
                if row["ci_low"] is not None:
                    # Coincident curves keep their true coordinates; no numerical jitter.
                    ax.vlines(x, row["ci_low"], row["ci_high"], color=color, lw=0.8, alpha=0.85)
                    ax.hlines(
                        [row["ci_low"], row["ci_high"]], x - 0.05, x + 0.05, color=color, lw=0.8
                    )
        ax.set_xticks(range(len(BUDGETS)), ["Full" if b == "pool" else b for b in BUDGETS])
        ax.set_xlim(-0.25, len(BUDGETS) - 0.75)
        ax.set_ylim(0.48, 1)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xlabel("Target-label budget")
    axes[0].set_ylabel("Harm AUROC")
    handles = [
        Line2D([], [], color="#333333", marker="o", mfc="white", ms=5, label="All-candidate AUROC"),
        Line2D([], [], color="#0072B2", marker="D", ls="--", ms=3.5, label="Site-winner AUROC"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        handlelength=2.7,
        columnspacing=2.3,
    )
    return fig


def figure3(data: list[dict[str, Any]]) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 115 / 25.4))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.45, top=0.88, wspace=0.18)
    eps = select(data, figure="Fig3", metric="epsilon")["value"]
    for panel, (d, title), ax in zip(("a", "b"), DIRECTIONS.items(), axes, strict=True):
        axes_style(ax, f"({panel}) {title}")
        ax.axhline(eps, color=".5", linestyle=(0, (4, 3)), lw=0.85, zorder=1)
        ax.text(
            0.985,
            eps + 0.004,
            rf"$\varepsilon={eps:.2f}$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=".4",
            fontsize=7,
        )
        margin_rows = []
        for rule, (name, marker, color) in POLICIES.items():
            cov = select(
                data, figure="Fig3", direction=d, policy=rule, metric="median_acceptance_coverage"
            )
            harm = select(data, figure="Fig3", direction=d, policy=rule, metric="median_harm")
            if cov["plot_status"] == "margin_no_action_at_median":
                n = select(data, figure="Fig3", direction=d, policy=rule, metric="accepting_draws")[
                    "value"
                ]
                draws = select(data, figure="Fig3", direction=d, policy=rule, metric="draws")[
                    "value"
                ]
                margin_rows.append((name, marker, color, n, draws))
                continue
            # Absolute endpoints avoid assuming symmetry or intervals centred on medians.
            if cov["ci_low"] is not None and harm["ci_low"] is not None:
                interval_style = (0, (2, 2)) if rule == "conservative::C1" else "solid"
                ax.hlines(
                    harm["value"],
                    cov["ci_low"],
                    cov["ci_high"],
                    color=color,
                    lw=0.7,
                    alpha=0.65,
                    linestyles=interval_style,
                )
                ax.vlines(
                    cov["value"],
                    harm["ci_low"],
                    harm["ci_high"],
                    color=color,
                    lw=0.7,
                    alpha=0.65,
                    linestyles=interval_style,
                )
            ax.plot(
                cov["value"],
                harm["value"],
                marker=marker,
                color=color,
                ms=7 if rule == "conservative::C1" else 5,
                markerfacecolor="white" if rule == "conservative::C1" else color,
                markeredgewidth=0.8,
                linestyle="none",
                zorder=5,
            )
        bx = select(data, figure="Fig3", direction=d, policy="boundary", metric="coverage")
        by = select(data, figure="Fig3", direction=d, policy="boundary", metric="harm")
        ax.plot(
            bx["value"],
            by["value"],
            marker="h",
            ms=7,
            color=".15",
            mfc="white",
            linestyle="none",
            markeredgewidth=1,
            zorder=6,
        )
        ax.set_xlim(0, 0.60)
        ax.set_ylim(0, 0.20)
        ax.set_xticks([0, 0.15, 0.30, 0.45, 0.60])
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Application coverage")
        if panel == "a":
            ax.set_ylabel("Harmful fraction among applied edits")
        else:
            ax.tick_params(labelleft=False)
        # Separate strip: zero-median policies NEVER receive a risk coordinate.
        pos = ax.get_position()
        strip = fig.add_axes([pos.x0, 0.205, pos.width, 0.135])
        strip.set_axis_off()
        strip.text(
            0,
            1,
            "No automatic action at median allocation",
            fontsize=7.1,
            fontweight="bold",
            va="top",
        )
        for i, (name, marker, color, n, draws) in enumerate(margin_rows):
            y = 0.64 - i * 0.28
            strip.plot(
                0.025,
                y,
                marker=marker,
                ms=4.2,
                color=color,
                transform=strip.transAxes,
                clip_on=False,
            )
            strip.text(
                0.07,
                y,
                f"{name} ({n}/{draws} allocations apply edits)",
                fontsize=6.8,
                va="center",
                transform=strip.transAxes,
            )
        strip.set_xlim(0, 1)
        strip.set_ylim(0, 1)
    handles = [
        Line2D(
            [],
            [],
            color=c,
            marker=m,
            ls="none",
            ms=7 if rule == "conservative::C1" else 5,
            mfc="white" if rule == "conservative::C1" else c,
            label=name,
        )
        for rule, (name, m, c) in POLICIES.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=".15",
            marker="h",
            mfc="white",
            ls="none",
            ms=7,
            label="Test-label boundary (analysis only)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.012),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handletextpad=0.65,
        labelspacing=0.8,
    )
    return fig


def figure4(data: list[dict[str, Any]]) -> Any:
    fig = plt.figure(figsize=(174 / 25.4, 127 / 25.4))
    columns = [(PRIMARY, "original"), (STRICT, "original"), (PRIMARY, "matched")]
    labels = ["Primary", "Stricter harm", "Matched sites"]
    for i, (d, title) in enumerate(DIRECTIONS.items()):
        x0 = 0.10 + i * 0.475
        ax = fig.add_axes([x0, 0.61, 0.405, 0.27])
        axes_style(ax, f"({'a' if i == 0 else 'b'}) {title}")
        ax.axhline(
            select(data, figure="Fig4", metric="chance_auroc")["value"],
            color=".7",
            lw=0.6,
            ls=(0, (3, 3)),
        )
        for j, (h, pop) in enumerate(columns):
            row = select(
                data,
                figure="Fig4",
                direction=d,
                harm_definition=h,
                population=pop,
                metric="harm_auroc",
            )
            point_interval(ax, j, row, "#333333", ["o", "s", "D"][j])
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(0.48, 1)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xticks(range(3), labels)
        if i == 0:
            ax.set_ylabel("Site-winner harm AUROC")
        else:
            ax.tick_params(labelleft=False)
        matrix = fig.add_axes([x0, 0.11, 0.405, 0.31])
        matrix.set_title(
            f"({'c' if i == 0 else 'd'}) {title}: policy outcome",
            loc="left",
            fontweight="bold",
            fontsize=8.5,
            pad=37,
        )
        for y, rule in enumerate(("plug_in", "conservative", "navarro_adapted")):
            for x, (h, pop) in enumerate(columns):
                record = select(
                    data,
                    figure="Fig4",
                    direction=d,
                    harm_definition=h,
                    population=pop,
                    policy=rule,
                    metric="verdict",
                )
                text, color = VERDICTS[record["value"]]
                matrix.add_patch(Rectangle((x, y), 1, 1, fc=color, ec="white", lw=1.5))
                matrix.text(
                    x + 0.5, y + 0.5, text, fontsize=7, ha="center", va="center", color=".15"
                )
        # Row names inside a narrow band above each row avoid an enormous external gutter.
        # Full names are written in the left margin of each panel, above the corresponding row.
        matrix.set_xlim(0, 3)
        matrix.set_ylim(3, 0)
        matrix.set_xticks([0.5, 1.5, 2.5], labels)
        matrix.xaxis.tick_top()
        matrix.tick_params(axis="x", length=0, pad=5, labelsize=7)
        matrix.set_yticks([])
        for spine in matrix.spines.values():
            spine.set_visible(False)
        # Three row captions sit in reserved gaps, with the matrix itself horizontally aligned.
        for y, name in enumerate(["Plug-in", "Pooled conservative", "Adapted Navarro-Cerdán"]):
            matrix.text(
                0, y + 0.05, name, ha="left", va="top", fontsize=7, color=".25", fontweight="bold"
            )
    category_handles = [
        Patch(facecolor=color, edgecolor=".8", label=label.replace("\n", " "))
        for key, (label, color) in VERDICTS.items()
        if key != "NOT_ESTIMABLE"
    ]
    fig.legend(
        handles=category_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.027),
        ncol=4,
        frameon=False,
        fontsize=7,
        columnspacing=1.4,
        handlelength=1,
    )
    fig.text(
        0.52, 0.012, "Study-defined classifications; no formal guarantee.", fontsize=7, ha="center"
    )
    return fig


def export(fig: Any, name: str, output: Path) -> None:
    # Detect clipped visible text before saving. Hidden tick labels are excluded.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            extent = text.get_window_extent(renderer)
            require(
                extent.x0 >= box.x0 - 1
                and extent.y0 >= box.y0 - 1
                and extent.x1 <= box.x1 + 1
                and extent.y1 <= box.y1 + 1,
                f"clipped text in {name}: {text.get_text()}",
            )
            banned = re.compile(
                r"\b(zero-shot|safe|unsafe|certified|oracle|held-out|generator effect)\b"
                r"|calibration pages|preregistered ceiling|hierarchical calibration",
                re.I,
            )
            require(not banned.search(text.get_text()), f"prohibited display term in {name}")
    fig.savefig(
        output / f"{name}.pdf",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "VF2 figure replay",
            "Title": name,
        },
    )
    fig.savefig(
        output / f"{name}.svg",
        metadata={"Date": None, "Creator": "VF2 figure replay", "Title": name},
    )
    fig.savefig(output / f"{name}.png", dpi=600, metadata={"Software": "VF2 figure replay"})
    with Image.open(output / f"{name}.png") as image:
        require(min(image.info["dpi"]) >= 599.9, f"PNG resolution below 600 dpi: {name}")
    svg = (output / f"{name}.svg").read_text()
    require("<image" not in svg and "<path" in svg, f"non-vector SVG: {name}")
    plt.close(fig)


def captions(e: Evidence) -> str:
    a, b = (e.execution["support"][d]["target_pool"]["pages"] for d in DIRECTIONS)
    return rf"""% Generated from final corrected VF2 records; use with the three new assets.
\newcommand{{\VFtwoCaption}}{{Harm discrimination across target-label budgets on the corrected
test block. (a) TXT$\rightarrow$VLM. (b) VLM$\rightarrow$TXT. Curves show all-candidate and
site-winner AUROC before cutoff selection. Budget $B=0$ denotes no additional target-generator
labels; Full denotes the full target pool ({a} and {b} pages, respectively). Values are medians
over predetermined allocations. Where available, error bars show 95\% scientific-group
percentile intervals; these were computed only at $B=0$ and Full.}}

\newcommand{{\VFthreeCaption}}{{Full-pool deployment under the primary harm definition. (a)
TXT$\rightarrow$VLM. (b) VLM$\rightarrow$TXT. Markers show median application coverage and
median harmful fraction among applied edits; harm summaries condition on allocations applying
edits. Horizontal and vertical intervals are 95\% scientific-group percentile intervals for the
deployable rules. The dashed line marks $\varepsilon=0.10$. Rules with zero median coverage are
listed separately because no-action harm is undefined; counts identify allocations that apply
edits. Corpus-specific and partially pooled variants use conservative cutoff selection.
The hollow test-label boundary is an analysis-only reference, with no grouped interval
reported, and is not a deployable policy.}}

\newcommand{{\VFfourCaption}}{{Sensitivity of ranking and deployment conclusions at the full
target pool. (a, b) Site-winner harm AUROC under primary evaluation, the stricter harm
definition, and matched-site analysis, with 95\% scientific-group percentile intervals. (c, d)
Study-defined operating classifications for the plug-in, pooled conservative, and adapted
Navarro-Cerd\'an rules. Stricter harm keeps models, scores, and winners fixed while recomputing
harm-dependent cutoffs and metrics. Matched-site analysis restricts cutoff selection and test
evaluation to sites shared by both generators with models and scores fixed. It is a population
sensitivity and does not isolate generator identity causally.}}
"""


def legacy_report() -> str:
    pattern = re.compile(
        r"pf1|rk4|cal2|known_generator|small_pool|stratified_acquisition|holm|sign_test", re.I
    )
    lines = [
        "# Legacy-reference detection",
        "",
        "The plotting script imports no experiment modules and opens only final VF2 registries, "
        "manifests, and frozen VF2 metric tables. "
        "No historical outcome record supplies a plotted value.",
        "",
        "The following identifiers occur in the hash-verified *producing* "
        "VF2 analysis implementation. "
        "These are inherited implementation/resource names, not plotting data sources. "
        "Every plotted outcome is registered again in final VF2 evidence.",
        "",
        "| Producing code location | Reference |",
        "|---|---|",
    ]
    for p in [Path("scripts/paper3_vf2_analysis.py")]:
        for n, text in enumerate((ROOT / p).read_text().splitlines(), 1):
            if pattern.search(text):
                lines.append(f"| {p}:{n} | `{text.strip().replace('|', '/').replace('`', '')}` |")
    lines.extend(
        [
            "",
            "Validation rejects any plotted record outside the final A0/M1 VF2 arms. "
            "Internal A0 identifiers retain historical naming, but display text uses "
            "'no additional target-generator labels'. Target-only, small-pool, known-reference, "
            "acquisition, Holm, and sign-test outcomes are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


def metadata(e: Evidence, output: Path) -> None:
    files = {
        p.name: sha(p) for p in sorted(output.iterdir()) if p.suffix in {".pdf", ".svg", ".png"}
    }
    write_json(
        output / "validation_report.json",
        {
            "status": "PASS",
            "checks": e.checks,
            "audit_records": len(e.audit),
            "scientific_evaluation_performed": False,
            "model_score_provenance": {
                model_id: {
                    key: e.models[model_id][key]
                    for key in ("score_path", "score_sha256", "feature_sha256", "model_sha256")
                }
                for model_id in sorted({m for row in e.audit for m in row.get("model_ids", [])})
            },
            "manuscript_assets_overwritten": False,
            "missing_intervals": [
                "Fig2: budgets 5, 10, 25, 50, 100",
                "Fig3: analysis-only test-label boundary",
            ],
            "conditional_harm_note": "Zero median coverage does not mean every allocation abstains;"
            " "
            "accepting-draw counts are shown separately.",
            "float_serialization_comparison_absolute_tolerance": 1e-15,
            "float_serialization_differences": e.serialization_differences,
            "float_authority": "Registry values always take precedence; tolerance checks "
            "serialization only, not scientific disagreement.",
        },
    )
    write_json(
        output / "provenance.json",
        {
            "study": "P3-VF2",
            "status": "PROSPECTIVELY FROZEN CORRECTIVE RE-ANALYSIS",
            "numerical_authority": str(REGISTRY),
            "source_sha256": e.sources,
            "plotting_script": "scripts/plot_paper3_vf2_figures.py",
            "plotting_script_sha256": sha(Path(__file__)),
            "git_HEAD": git_head(),
            "artifact_sha256": files,
            "scientific_evaluation_performed": False,
            "aggregation": "Registered summaries, or frozen median of existing registered draw "
            "metrics. No AUROC, policy, labels, or bootstrap is recomputed.",
            "intervals": "Existing 95% scientific-group percentile intervals; "
            "no missing interval estimated.",
            "corpus_specific_policy": "conservative::C1",
            "partially_pooled_policy": "conservative::C3@prior=5 "
            "(fixed native main setting; 2/10 are sensitivity settings)",
            "exports": {
                "width_mm": 174,
                "png_dpi": 600,
                "matplotlib": matplotlib.__version__,
                "font": "DejaVu Sans",
                "pdf": "embedded TrueType, vector",
                "svg": "vector, live text",
            },
        },
    )
    # The repository documentation rule also requires this conventional name.
    write_json(
        output / "figure_manifest.json", {"source_sha256": e.sources, "artifact_sha256": files}
    )
    (output / "legacy_reference_report.md").write_text(legacy_report())
    (output / "README.md").write_text("""# Final corrected VF2 manuscript figures

Reproduce from repository root:

```sh
uv run --locked python scripts/plot_paper3_vf2_figures.py
```

The frozen corrected artifacts must be present; a code-only checkout cannot reconstruct the
evidence. The script rejects missing or changed sources and never falls back to historical data.
It reads immutable registered metrics, not models, candidate labels, or raw datasets. It
performs presentation medians only, using the protocol's predetermined draw aggregation.

## Outputs and interpretation

- Fig2: A0/M1 all-candidate versus site-winner harm AUROC across the seven budgets. Full pools
are read from the corrected execution registry. Intermediate-budget grouped intervals are
unavailable and omitted.
- Fig3: primary full-pool operating points. All nonzero-median policies have existing grouped
coverage/harm intervals. The analysis-only boundary has no grouped interval and is hollow.
Zero-median policies appear in a separate strip; their conditional harm medians are retained in
the audit, never used as no-action coordinates. Coverage includes all allocations, whereas harm
conditions on applying edits. Crosses are marginal intervals, not joint confidence regions.
Corpus-specific C1 and partially pooled C3 (native strength 5) are empirical procedures, without
formal guarantees.
- Fig4: winner AUROC and registered aggregate verdicts, without deriving classifications from
median harm alone. This matters because a verdict depends on draw-level behaviour. Strict harm
fixes models/scores/winners; matched sites restrict BOTH cutoff-selection and test winners
before threshold selection.

Every plotted value is in `figure_data_audit.json` and `.csv`, including exact registry
keys/JSON locations, interval keys and model IDs. Audit files are written before rendering. The
final registry takes precedence over earlier manuscript text. `validation_report.json`,
`provenance.json`, `figure_manifest.json`, and `legacy_reference_report.md` document validation
and hashes. No legacy manuscript figure or historical outcome supplies a plotted number.

## Manuscript integration

Legacy manuscript figures are unchanged. From `paper3_icdar_ijdar_2027/`, use:

```tex
\\includegraphics[width=\\textwidth]{../figures/final_vf2/Fig2.pdf}
\\includegraphics[width=\\textwidth]{../figures/final_vf2/Fig3.pdf}
\\includegraphics[width=\\textwidth]{../figures/final_vf2/Fig4.pdf}
```

`CAPTIONS.tex` defines `\\VFtwoCaption`, `\\VFthreeCaption`, and `\\VFfourCaption`. These
figures describe a prospectively frozen corrective re-analysis on a previously observed test
block, not an independent confirmation. Current legacy manuscript text must be revised to final
VF2 evidence; this replay does not edit it.

## Deliberate replacements

Fig2 replaces the legacy joint/target-only comparison with all-candidate/site-winner
discrimination. Fig3 replaces the legacy coverage/recall bars with operating-point coordinates
and uncertainty. Fig4 replaces budget trajectories with harm-definition and population
sensitivities. Old pool sizes, historical AUROCs, known-generator comparisons, and significance
annotations are excluded rather than manually relabelled.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/final_vf2")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(
        output != (ROOT / "paper3_icdar_ijdar_2027/figures").resolve(),
        "refusing to overwrite legacy assets",
    )
    evidence = Evidence()
    evidence.load_figure_data()
    evidence.validate()
    output.mkdir(parents=True, exist_ok=True)
    # The numerical audit is persisted BEFORE any figure is rendered.
    write_json(output / "figure_data_audit.json", evidence.audit)
    fields = [
        "figure",
        "panel",
        "direction",
        "budget",
        "population",
        "harm_definition",
        "ranking_level",
        "policy",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "source",
        "key",
        "json_path",
        "interval_keys",
        "plot_status",
        "aggregation",
        "notes",
        "model_ids",
    ]
    with (output / "figure_data_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in evidence.audit:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )
    (output / "CAPTIONS.tex").write_text(captions(evidence))
    style()
    for name, plot in [("Fig2", figure2), ("Fig3", figure3), ("Fig4", figure4)]:
        export(plot(evidence.audit), name, output)
    require(
        all(sha(ROOT / Path(p)) == digest for p, digest in evidence.sources.items()),
        "scientific source changed during rendering",
    )
    evidence.checks["scientific_sources_unchanged"] = "PASS"
    evidence.checks["no_clipped_text_or_prohibited_display_terms"] = "PASS"
    evidence.checks["vector_svg_and_600_dpi_png"] = "PASS"
    metadata(evidence, output)
    (output / "OUTPUT_SHA256SUMS.txt").write_text(
        "".join(
            f"{sha(p)}  {p.name}\n"
            for p in sorted(output.iterdir())
            if p.suffix in {".pdf", ".svg", ".png"}
        )
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "audit_records": len(evidence.audit),
                "exports": 9,
                "scientific_evaluation_performed": False,
            }
        )
    )


if __name__ == "__main__":
    main()

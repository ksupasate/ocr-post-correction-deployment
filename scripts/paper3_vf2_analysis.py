#!/usr/bin/env python3
"""Analysis of immutable clean scores; no upstream rebuilding or fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import paper3_rc1_reviewer_closure as rc
import paper3_vf2_clean_execution as ex
import sgv_cal2_domain_stratified_calibration as cal2

pf1 = ex.pf1
rk3 = ex.rk3


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def context(registry: dict[str, Any]) -> rc.Context:
    pages = set(registry["cells"][0]["test_pages"])
    errors = pf1.error_population()
    errors = errors[errors.document_id.isin(pages)].reset_index(drop=True)
    assert set(errors.document_id).issubset(pages)
    links = pd.read_parquet(pf1.PROPOSAL_ERROR_LINKS)
    return set(errors.error_key.astype(str)), pf1.site_to_errors(errors, links), errors


def sampling(registry: dict[str, Any], settings: dict[str, Any]) -> rc.Resampling:
    page_to_group = registry["page_to_group"]
    page_to_corpus = registry["page_to_corpus"]
    pairs = sorted(
        {(page_to_group[p], page_to_corpus[p]) for p in registry["cells"][0]["test_pages"]}
    )
    groups, corpora = zip(*pairs, strict=True)
    assert len(groups) == 59
    weights = cal2.bootstrap_weights(
        corpora, settings["uncertainty"]["resamples"], cal2.BOOTSTRAP_SEED
    )
    return rc.Resampling(groups, corpora, {g: i for i, g in enumerate(groups)}, weights)


def score_frames(
    cell: dict[str, Any], manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_parquet(ex.ROOT / manifest["score_path"])
    labels = ex.label_rows(scores.candidate_id.tolist()).set_index("candidate_id")
    frame = scores.join(labels, on="candidate_id", validate="many_to_one")
    frame = frame.assign(safety=frame.score, rank_score=frame.score)
    cal = frame[frame.role == "calibration"].reset_index(drop=True)
    test = frame[frame.role == "test"].reset_index(drop=True)
    assert set(cal.candidate_id) == set(cell["calibration_candidate_ids"])
    assert set(test.candidate_id) == set(cell["test_candidate_ids"])
    return cal, test


def reusable_corpus_rule(
    rule: str, calibration: pd.DataFrame, test: pd.DataFrame, ctx: rc.Context, usable: bool
) -> rc.Scored:
    """Original C1/C3 selectors; they receive only blind test score/corpus columns."""
    base_rule, family = rule.split("::", 1)
    policy = rc.FROZEN_POLICY[base_rule]
    blind = test[["candidate_id", "site_key", "document_id", "corpus", "safety"]]
    calibration = cal2.with_clusters(calibration)
    if family == "C1":
        bands = cal2.corpus_bands(policy, calibration, blind, usable)
    else:
        prior = float(family.split("=")[1])
        bands = cal2.hierarchical_bands(policy, calibration, blind, usable, prior)
    cctx = cal2.build_context(ctx)
    row, _ = cal2.evaluate(policy, cal2.test_view(test, cctx), bands, cctx)
    return rc.Scored(row, bands.accept, {"cutoff_records": bands.records})


def boundary_row(test: pd.DataFrame, ctx: rc.Context, usable: bool) -> dict[str, Any]:
    accepted = pf1.rk2.oracle_accepted(test, pf1.EPSILON) if usable else test.head(0)
    cutoff = pf1.rk2.rk1.choose_threshold(test, pf1.EPSILON) if usable else None
    return {
        "analysis_only": True,
        "cutoff": cutoff,
        "coverage": len(accepted) / max(len(test), 1),
        "harm": float(accepted.is_harmful.mean()) if len(accepted) else None,
        "recall": len(pf1.repaired_sites(accepted, ctx)) / max(len(ctx[0]), 1),
    }


def evaluate(run: int) -> None:
    print(json.dumps(ex.hard_gate()), flush=True)
    r = ex.read(ex.REGISTRY)
    config = ex.read(ex.CONFIG)
    settings = config["scientific_settings"]
    rd = ex.run_dir(run)
    model_registry = ex.read(rd / "manifests/model_registry.json")
    manifests = {m["model_id"]: m for m in model_registry["models"]}
    ctx = context(r)
    # Generator/site membership only; no labels read by matching function.
    blind = []
    for path in (ex.rk1.RANKING_POPULATION, pf1.POPULATION):
        cols = ["candidate_id", "site_key", "corrector_sources"]
        if path == ex.rk1.RANKING_POPULATION:
            blind.append(
                pd.read_parquet(path, columns=cols, filters=[("population", "==", ex.rk1.POP_U5)])
            )
        else:
            blind.append(pd.read_parquet(path, columns=cols))
    matched = rc.matched_sites(pd.concat(blind).drop_duplicates("candidate_id"))
    mctx = rc.restricted_context(ctx, matched)
    contexts = {rc.ORIGINAL: ctx, rc.MATCHED: mctx}
    samp = sampling(r, settings)
    clusters = {p: rc.error_clusters(c, samp) for p, c in contexts.items()}
    ranking_rows = []
    policy_rows = []
    boundary_rows = []
    support_rows = []
    risk_rows = []
    grouped_counts: dict[str, list[dict[str, np.ndarray]]] = {}
    policy_groups: dict[str, list[dict[str, Any]]] = {}
    auroc_inputs: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    auroc_keys: dict[str, dict[str, Any]] = {}
    unavailable = []
    curve_grid = np.round(np.arange(0.01, 1.000001, 0.01), 10)
    key_fields = ["direction", "arm", "budget", "harm_definition", "population", "rule"]

    def key(base: dict[str, Any], last: str) -> str:
        return "|".join(str(base[x]) for x in key_fields[:-1]) + "|" + last

    for index, cell in enumerate(r["cells"]):
        m = manifests[cell["model_id"]]
        assert ex.sha(ex.ROOT / m["score_path"]) == m["score_sha256"]
        cal_candidates, candidates = score_frames(cell, m)
        assert all(
            rc.harm_label(o, rc.SW) == bool(h)
            for o, h in zip(
                pd.concat([candidates, cal_candidates]).outcome,
                pd.concat([candidates, cal_candidates]).is_harmful,
                strict=True,
            )
        )
        usable = m["status"] == "ESTIMABLE"
        cal = pf1.rk4.decisions(cal_candidates)
        winners = pf1.rk4.decisions(candidates)
        is_comparator = cell["arm"] == pf1.B2
        is_control = cell["arm"] == "M1_TARGET_LABEL_PERMUTATION"
        definitions = (rc.SW,) if is_comparator or is_control else (rc.SW, rc.NI)
        for definition in definitions:
            labels = {
                i: rc.harm_label(o, definition)
                for i, o in zip(
                    pd.concat([candidates, cal_candidates]).candidate_id,
                    pd.concat([candidates, cal_candidates]).outcome,
                    strict=True,
                )
            }
            current_cal = rc.with_harm(cal, labels)
            current_winners = rc.with_harm(winners, labels)
            current_candidates = rc.with_harm(candidates, labels)
            populations = [(rc.ORIGINAL, current_winners, current_cal)]
            if cell["budget"] == "pool" and not is_comparator and not is_control:
                populations.append(
                    (
                        rc.MATCHED,
                        current_winners[current_winners.site_key.isin(matched)].reset_index(
                            drop=True
                        ),
                        current_cal[current_cal.site_key.isin(matched)].reset_index(drop=True),
                    )
                )
            for population, test, calibration in populations:
                base = {
                    k: cell[k] for k in ["model_id", "direction", "scheme", "arm", "budget", "draw"]
                }
                base.update(
                    harm_definition=definition, population=population, model_status=m["status"]
                )
                own_ctx = contexts[population]
                support_rows.append(
                    base
                    | {
                        "calibration_decisions": len(calibration),
                        "test_decisions": len(test),
                        "calibration_pages": calibration.document_id.nunique(),
                        "test_pages": test.document_id.nunique(),
                        "calibration_groups": len(
                            {r["page_to_group"][p] for p in calibration.document_id}
                        ),
                        "test_groups": len({r["page_to_group"][p] for p in test.document_id}),
                        "calibration_corpus": calibration.corpus.value_counts().to_dict(),
                        "test_corpus": test.corpus.value_counts().to_dict(),
                    }
                )
                frames = [("winner", test)]
                if population == rc.ORIGINAL:
                    frames.append(("all", current_candidates))
                for unit, frame in frames:
                    auc = (
                        rc._auroc(frame.safety.to_numpy(float), frame.is_harmful.to_numpy(bool))
                        if usable and len(frame)
                        else None
                    )
                    ranking_rows.append(
                        base
                        | {
                            "unit": unit,
                            "harm_auroc": auc,
                            "candidate_rows": len(frame),
                            "scientific_groups": len(
                                {r["page_to_group"][p] for p in frame.document_id}
                            ),
                            "harmful_prevalence": float(frame.is_harmful.mean())
                            if len(frame)
                            else None,
                        }
                    )
                    if cell["budget"] in ("0", "pool") and not is_control and usable and len(frame):
                        ak = key(base, unit)
                        auroc_inputs.setdefault(ak, []).append(
                            (
                                frame.safety.to_numpy(float),
                                frame.is_harmful.to_numpy(bool),
                                rc.cluster_index(frame, samp),
                            )
                        )
                        auroc_keys[ak] = base | {"unit": unit}
                if is_comparator:
                    continue
                boundary_rows.append(base | boundary_row(test, own_ctx, usable))
                rules = (
                    [rc.PLUG_IN, rc.CONSERVATIVE]
                    if is_control
                    else [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]
                )
                if population == rc.ORIGINAL and not is_control:
                    rules += [
                        rc.TRIAGE,
                        rc.TRIAGE_COST,
                        f"{rc.NAVARRO}@w=0.025",
                        f"{rc.NAVARRO}@w=0.1",
                    ]
                    for family in ["C1", "C3@prior=2", "C3@prior=5", "C3@prior=10"]:
                        rules += [
                            f"{rule}::{family}"
                            for rule in [rc.PLUG_IN, rc.CONSERVATIVE, rc.TRIAGE, rc.TRIAGE_COST]
                        ]
                for rule in rules:
                    if "::" in rule:
                        scored = reusable_corpus_rule(rule, calibration, test, own_ctx, usable)
                    else:
                        native, sep, width = rule.partition("@w=")
                        scored = rc.score_rule(
                            native,
                            calibration,
                            test,
                            own_ctx,
                            usable,
                            settings,
                            float(width) if sep else None,
                        )
                    row = base | {
                        "rule": rule,
                        **{k: v for k, v in scored.row.items() if k != "policy"},
                        **scored.extra,
                    }
                    row["calibration_model_group_overlap"] = 0
                    if calibration.empty:
                        row["policy_status"] = "NOT ESTIMABLE: no calibration decisions"
                    elif "conservative" in rule and calibration.document_id.nunique() < 2:
                        row["policy_status"] = (
                            "NOT ESTIMABLE BY DESIGN: fewer than two calibration pages"
                        )
                    elif not usable:
                        row["policy_status"] = "NOT ESTIMABLE: score model"
                    else:
                        row["policy_status"] = "ESTIMABLE"
                    policy_rows.append(row)
                    pk = key(base, rule)
                    policy_groups.setdefault(pk, []).append(scored.row)
                    if not is_control:
                        grouped_counts.setdefault(pk, []).append(
                            rc.cluster_counts(
                                test, scored.accept, own_ctx, samp, clusters[population][0]
                            )
                        )
                unavailable.append(
                    base | {"rule": "M10", "status": "NOT ESTIMABLE", "reason": r["formal_M10"]}
                )
                if cell["budget"] == "pool" and population == rc.ORIGINAL and not is_control:
                    risk_rows.extend(
                        base | point for point in rc.winner_curve(test, own_ctx, curve_grid)
                    )
                # Original retained seeded random-score diagnostics; no additional model.
                if population == rc.ORIGINAL and not is_control and cell["budget"] != "0":
                    rng = pf1._random_generator(cell["direction"], cell["budget"], cell["draw"])
                    random_test = test.assign(safety=rng.random(len(test)))
                    random_cal = calibration.assign(safety=rng.random(len(calibration)))
                    for rule in [rc.PLUG_IN, rc.CONSERVATIVE, rc.TRIAGE, rc.TRIAGE_COST]:
                        scored = rc.score_rule(
                            rule, random_cal, random_test, own_ctx, True, settings
                        )
                        control_base = base | {
                            "arm": "RANDOM_SCORE_CONTROL",
                            "model_id": "NO_LEARNED_MODEL",
                        }
                        policy_rows.append(
                            control_base
                            | {
                                "rule": rule,
                                **{k: v for k, v in scored.row.items() if k != "policy"},
                            }
                        )
                        policy_groups.setdefault(key(control_base, rule), []).append(scored.row)
            print(f"evaluate run{run}: {index + 1}/492 {cell['model_id']} {definition}", flush=True)
    policy_summary = {}
    for pk, rows in policy_groups.items():
        rule = pk.split("|")[-1]
        native = rule.split("::")[0].split("@")[0]
        summary = rc.summarize_rule(native, rows)
        if pk in grouped_counts:
            stack = {
                field: np.vstack([c[field] for c in grouped_counts[pk]])
                for field in ["decisions", "accepted", "harmful_accepted", "auto_repaired"]
            }
            population = pk.split("|")[-2]
            summary["intervals"] = rc.policy_intervals(stack, clusters[population][1], samp.weights)
        policy_summary[pk] = summary
    print(
        f"run{run}: policies complete; grouped AUROC resampling starts ({len(auroc_inputs)} tasks)",
        flush=True,
    )
    resampled = rc.auroc_resamples(auroc_inputs, samp.weights, config["workers"])
    auroc_summary = {}
    for ak, matrix in resampled.items():
        median = float(np.median([rc._auroc(s, h) for s, h, _ in auroc_inputs[ak]]))
        low, high = rc.percentile_interval(np.nanmedian(matrix, axis=1))
        auroc_summary[ak] = {"median": median, "ci_low": low, "ci_high": high}
        base = auroc_keys[ak]
        if base["unit"] == "winner" and base["population"] == rc.ORIGINAL:
            other = ak.rsplit("|", 1)[0] + "|all"
            delta = matrix - resampled[other]
            point = np.median(
                [
                    rc._auroc(s, h) - rc._auroc(s2, h2)
                    for (s, h, _), (s2, h2, _) in zip(
                        auroc_inputs[ak], auroc_inputs[other], strict=True
                    )
                ]
            )
            low, high = rc.percentile_interval(np.nanmedian(delta, axis=1))
            auroc_summary[ak.rsplit("|", 1)[0] + "|winner_minus_all"] = {
                "median": float(point),
                "ci_low": low,
                "ci_high": high,
            }
    csv(rd / "tables/ranking_rows.csv", ranking_rows)
    csv(rd / "policies/policy_rows.csv", policy_rows)
    ex.write(rd / "policies/policy_rows.json", plain(policy_rows))
    csv(rd / "tables/boundary_rows.csv", boundary_rows)
    csv(rd / "tables/support_rows.csv", support_rows)
    csv(rd / "tables/winner_risk_coverage.csv", risk_rows)
    csv(rd / "tables/not_estimable.csv", unavailable)
    np.savez_compressed(rd / "tables/auroc_resampling.npz", **resampled)
    ex.write(
        rd / "tables/scientific_results.json",
        plain(
            {
                "study": "P3-VF2",
                "analysis_status": ex.STATUS,
                "run": run,
                "auroc_summaries": auroc_summary,
                "policy_summaries": policy_summary,
                "error_site_denominators": {p: len(c[0]) for p, c in contexts.items()},
                "matched_sites": sorted(matched),
                "model_counts": {
                    k: model_registry[k] for k in ["requested", "completed", "non_estimable"]
                },
                "execution_freeze_sha256": ex.sha(ex.FREEZE),
                "grouped_bootstrap": {
                    "groups": list(samp.clusters),
                    "seed": cal2.BOOTSTRAP_SEED,
                    "resamples": len(samp.weights),
                },
            }
        ),
    )
    figures(run)
    result_registry(run)
    print(f"run{run} CORRECTED RESULTS FROZEN", flush=True)


def result_registry(run: int) -> None:
    rd = ex.run_dir(run)
    data = ex.read(rd / "tables/scientific_results.json")
    models = ex.read(rd / "manifests/model_registry.json")["models"]
    resources = ex.read(rd / "manifests/resource_manifest.json")
    entries = []
    for family, source in [
        ("auroc_summaries", "tables/ranking_rows.csv"),
        ("policy_summaries", "policies/policy_rows.csv"),
    ]:
        for k, metrics in data[family].items():
            direction, arm, budget, harm, population, policy = k.split("|")
            selected = [
                m
                for m in models
                if m["direction"] == direction and m["arm"] == arm and m["budget"] == budget
            ]

            def add(
                value: Any,
                trail: list[str],
                k=k,
                direction=direction,
                budget=budget,
                population=population,
                harm=harm,
                policy=policy,
                metrics=metrics,
                selected=selected,
                source=source,
            ) -> None:
                if isinstance(value, dict):
                    for name, item in value.items():
                        add(item, [*trail, name])
                else:
                    entries.append(
                        {
                            "key": k + "|" + ".".join(trail),
                            "value": value,
                            "direction": direction,
                            "scheme": "conditional_honest_group_allocation",
                            "budget": budget,
                            "draw": "aggregate",
                            "population": population,
                            "harm_definition": harm,
                            "policy": policy,
                            "metric": ".".join(trail),
                            "uncertainty": metrics.get(
                                "intervals", {n: metrics.get(n) for n in ["ci_low", "ci_high"]}
                            ),
                            "group_support": 59
                            if population == rc.ORIGINAL
                            else "see support_rows.csv",
                            "model_IDs": [m["model_id"] for m in selected],
                            "resource_hashes": {
                                "resource_file_sha256": resources["resource_file_sha256"],
                                "environment_record_sha256": {
                                    c["environment"]: c["record_sha256"]
                                    for c in resources["checks"]
                                },
                            },
                            "source_artifact": str((rd / source).relative_to(ex.ROOT)),
                            "generating_script": "scripts/paper3_vf2_analysis.py",
                            "code_sha256": ex.sha(Path(__file__)),
                            "execution_freeze_sha256": ex.sha(ex.FREEZE),
                        }
                    )

            add(metrics, [])
    # Bind all individual draw metrics, not only aggregate headline summaries.
    for relative in [
        "policies/policy_rows.csv",
        "tables/ranking_rows.csv",
        "tables/boundary_rows.csv",
        "tables/support_rows.csv",
        "tables/winner_risk_coverage.csv",
        "tables/not_estimable.csv",
    ]:
        entries.append(
            {
                "key": "all_draw_records|" + relative,
                "value": "every row retained in immutable table",
                "source_artifact": str((rd / relative).relative_to(ex.ROOT)),
                "sha256": ex.sha(rd / relative),
                "metric": "table_row_registry",
                "generating_script": "scripts/paper3_vf2_analysis.py",
            }
        )
    support = pd.read_csv(rd / "tables/support_rows.csv")
    draw_records = []
    for relative in [
        "policies/policy_rows.csv",
        "tables/ranking_rows.csv",
        "tables/boundary_rows.csv",
    ]:
        table = pd.read_csv(rd / relative)
        for row in table.to_dict("records"):
            own = support[
                (support.model_id == row["model_id"])
                & (support.harm_definition == row["harm_definition"])
                & (support.population == row["population"])
            ]
            draw_records.append(
                {
                    "key": row["model_id"]
                    + "|"
                    + row["harm_definition"]
                    + "|"
                    + row["population"]
                    + "|"
                    + str(row.get("rule", row.get("unit", "boundary"))),
                    "value": row,
                    "direction": row["direction"],
                    "scheme": row["scheme"],
                    "budget": row["budget"],
                    "draw": row["draw"],
                    "population": row["population"],
                    "harm_definition": row["harm_definition"],
                    "policy": row.get("rule"),
                    "metric": "complete_draw_record",
                    "uncertainty": "grouped aggregate interval in entries; fixed calibration",
                    "group_support": plain(own.to_dict("records")),
                    "model_id": row["model_id"],
                    "resource_hashes": resources["resource_file_sha256"],
                    "source_artifact": str((rd / relative).relative_to(ex.ROOT)),
                    "generating_script": "scripts/paper3_vf2_analysis.py",
                }
            )
    ex.write(
        rd / "manifests/result_registry.json",
        plain({"entries": entries, "draw_records": draw_records, "analysis_status": ex.STATUS}),
    )
    files = [
        p
        for folder in [
            "resources",
            "features",
            "models",
            "scores",
            "policies",
            "tables",
            "figures",
            "manifests",
        ]
        for p in (rd / folder).rglob("*")
        if p.is_file()
    ]
    ex.write(
        rd / "manifests/result_freeze.json",
        {
            "study": "P3-VF2",
            "run": run,
            "files_sha256": {str(p.relative_to(ex.ROOT)): ex.sha(p) for p in sorted(files)},
            "gates": ex.hard_gate(),
            "original_outcomes_compared": False,
        },
    )


def figures(run: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "P3-VF2"
    import matplotlib.pyplot as plt

    rd = ex.run_dir(run)
    data = ex.read(rd / "tables/scientific_results.json")
    policies = data["policy_summaries"]
    auc = data["auroc_summaries"]
    directions = list(pf1.DIRECTIONS)

    def pkey(d: str, h: str, pop: str, rule: str) -> str:
        return f"{d}|{pf1.M1}|pool|{h}|{pop}|{rule}"

    labels = ["TXT→VLM", "VLM→TXT"]
    note = "Analysis only · prospectively frozen corrective re-analysis"

    def save(fig: Any, number: int, name: str, sources: list[str]) -> None:
        fig.text(0.5, 0.01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        for ext in ["png", "pdf", "svg"]:
            metadata = (
                {"Creator": "P3-VF2", "CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else ({"Date": None} if ext == "svg" else {})
            )
            fig.savefig(rd / "figures" / f"VF2-{number}_{name}.{ext}", dpi=180, metadata=metadata)
        plt.close(fig)
        ex.write(
            rd / "figures" / f"VF2-{number}_manifest.json",
            {
                "source_sha256": {s: ex.sha(rd / s) for s in sources},
                "analysis_only": True,
                "note": note,
            },
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, unit in enumerate(["all", "winner"]):
        values = [auc[pkey(d, rc.SW, rc.ORIGINAL, unit)]["median"] for d in directions]
        ax.bar(np.arange(2) + (i - 0.5) * 0.3, values, width=0.3, label=unit)
    ax.set(xticks=np.arange(2), xticklabels=labels, ylabel="Harm AUROC", ylim=(0.5, 1))
    ax.legend()
    save(fig, 1, "all_winner_AUROC", ["tables/scientific_results.json"])
    curves = pd.read_csv(rd / "tables/winner_risk_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, label in zip(axes, directions, labels, strict=True):
        for h in [rc.SW, rc.NI]:
            block = (
                curves[(curves.direction == d) & (curves.harm_definition == h)]
                .groupby("coverage_target")[["coverage", "harmful_share"]]
                .median()
            )
            ax.plot(block.coverage, block.harmful_share, label=h)
        ax.axhline(0.1, color="gray", ls="--")
        ax.set(title=label, xlabel="Winner coverage", ylabel="Harmful share")
        ax.legend(fontsize=8)
    save(fig, 2, "winner_risk_coverage", ["tables/winner_risk_coverage.csv"])
    for number, name, comparison in [
        (3, "primary_strict_harm", [rc.SW, rc.NI]),
        (4, "original_matched", [rc.ORIGINAL, rc.MATCHED]),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, d, label in zip(axes, directions, labels, strict=True):
            for i, c in enumerate(comparison):
                h = c if number == 3 else rc.SW
                pop = rc.ORIGINAL if number == 3 else c
                vals = [
                    policies[pkey(d, h, pop, rule)]["median_acceptance_coverage"]
                    for rule in [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]
                ]
                ax.bar(np.arange(3) + (i - 0.5) * 0.35, vals, width=0.35, label=c)
            ax.set(
                title=label,
                xticks=np.arange(3),
                xticklabels=["Plug-in", "Conservative", "Navarro"],
                ylabel="Coverage",
            )
            ax.legend(fontsize=7)
        save(fig, number, name, ["tables/scientific_results.json"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, label in zip(axes, directions, labels, strict=True):
        for rule in [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]:
            s = policies[pkey(d, rc.SW, rc.ORIGINAL, rule)]
            ax.scatter(
                s["median_acceptance_coverage"],
                s["median_harm"] if s["median_harm"] is not None else 0,
                label=rule,
            )
        ax.axhline(0.1, color="gray", ls="--")
        ax.axvline(0.05, color="gray", ls=":")
        ax.set(title=label, xlabel="Coverage", ylabel="Harmful share among accepting draws")
        ax.legend(fontsize=7)
    save(fig, 5, "policy_comparison", ["tables/scientific_results.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=[1, 2], required=True)
    evaluate(parser.parse_args().run)

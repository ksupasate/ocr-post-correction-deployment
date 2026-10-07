#!/usr/bin/env python3
"""Finalize frozen corrective results, then inspect invalidated history for disclosure."""
# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import paper3_rc1_reviewer_closure as rc
import paper3_vf2_analysis as analysis
import paper3_vf2_clean_execution as ex

pf1 = ex.pf1


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k != "run"}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return value.replace("paper3_vf2/run2/", "paper3_vf2/run1/")
    return value


def verify_determinism() -> dict[str, Any]:
    config = ex.read(ex.CONFIG)
    report = {"checks": [], "atol": config["determinism_atol"], "rtol": config["determinism_rtol"]}
    rd1, rd2 = ex.run_dir(1), ex.run_dir(2)
    scientific = [
        p
        for d in [
            "resources",
            "features",
            "models",
            "scores",
            "policies",
            "tables",
            "figures",
            "manifests",
        ]
        for p in (rd1 / d).rglob("*")
        if p.is_file() and p.name != "result_freeze.json"
    ]
    for p in sorted(scientific):
        relative = p.relative_to(rd1)
        q = rd2 / relative
        assert q.exists(), str(relative)
        same = ex.sha(p) == ex.sha(q)
        kind = "BIT-IDENTICAL" if same else None
        if not same and p.suffix == ".json" and normalize(ex.read(p)) == normalize(ex.read(q)):
            kind = "IDENTICAL AFTER RUN-PATH NORMALIZATION"
        if not same and p.suffix == ".npz":
            with np.load(p) as a, np.load(q) as b:
                assert set(a.files) == set(b.files)
                if all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files):
                    kind = "BIT-IDENTICAL ARRAYS; CONTAINER METADATA DIFFERS"
                elif all(
                    np.allclose(
                        a[k], b[k], atol=report["atol"], rtol=report["rtol"], equal_nan=True
                    )
                    for k in a.files
                ):
                    kind = "NUMERICALLY EQUIVALENT"
        assert kind is not None, f"Determinism failure: {relative}"
        report["checks"].append(
            {
                "artifact": str(relative),
                "result": kind,
                "run1_sha256": ex.sha(p),
                "run2_sha256": ex.sha(q),
            }
        )
    report["status"] = (
        "NUMERICALLY EQUIVALENT"
        if any(c["result"] == "NUMERICALLY EQUIVALENT" for c in report["checks"])
        else "BIT-IDENTICAL"
    )
    report["normalization_note"] = (
        "Run ID/path fields and NPZ container timestamps are administrative; prediction/threshold/metric/bootstrap arrays and figure underlying data are compared directly. No averaging."
    )
    ex.write(ex.OUT / "determinism.json", report)
    return report


def key(d: str, harm: str, population: str, metric: str, budget: str = "pool") -> str:
    return f"{d}|{pf1.M1 if budget != '0' else pf1.A0}|{budget}|{harm}|{population}|{metric}"


def classify(data: dict[str, Any], rd: Path) -> dict[str, Any]:
    auc, policies = data["auroc_summaries"], data["policy_summaries"]
    boundaries = pd.read_csv(rd / "tables/boundary_rows.csv")
    support = pd.read_csv(rd / "tables/support_rows.csv")
    config = ex.read(ex.CONFIG)
    directions = list(pf1.DIRECTIONS)

    def signature(d: str, h: str, pop: str = rc.ORIGINAL) -> list[bool]:
        useful = [
            policies[key(d, h, pop, p)]["verdict"] == "USEFUL"
            for p in [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]
        ]
        b = boundaries[
            (boundaries.direction == d)
            & (boundaries.arm == pf1.M1)
            & (boundaries.budget == "pool")
            & (boundaries.harm_definition == h)
            & (boundaries.population == pop)
        ]
        return [
            *useful,
            float(b.coverage.median()) >= 0.05,
            auc[key(d, h, pop, "winner")]["median"] >= 0.85,
        ]

    result = {"winner": {}, "strict": {}, "matched": {}, "primary_policy_verdicts": {}}
    changes = []
    for d in directions:
        w = auc[key(d, rc.SW, rc.ORIGINAL, "winner")]["median"]
        delta = auc[key(d, rc.SW, rc.ORIGINAL, "winner_minus_all")]["median"]
        result["winner"][d] = (
            "NOT SUPPORTED"
            if w < 0.8
            else (
                "ATTENUATED"
                if w < 0.85 or delta < -0.02
                else ("STRENGTHENED" if delta > 0.02 else "PRESERVED")
            )
        )
        primary, strict = signature(d, rc.SW), signature(d, rc.NI)
        changes.append(primary != strict)
        result["strict"][d] = {
            "primary_signature": primary,
            "strict_signature": strict,
            "primary_verdicts": {
                p: policies[key(d, rc.SW, rc.ORIGINAL, p)]["verdict"]
                for p in [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]
            },
            "strict_verdicts": {
                p: policies[key(d, rc.NI, rc.ORIGINAL, p)]["verdict"]
                for p in [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]
            },
        }
        result["primary_policy_verdicts"][d] = result["strict"][d]["primary_verdicts"]
        s = support[
            (support.direction == d)
            & (support.arm == pf1.M1)
            & (support.budget == "pool")
            & (support.harm_definition == rc.SW)
            & (support.population == rc.MATCHED)
        ]
        m = auc[key(d, rc.SW, rc.MATCHED, "winner")]["median"]
        floors = config["scientific_settings"]["rc3"]["support_rule"]
        low = (
            s.test_decisions.min() < floors["min_matched_test_decisions"]
            or s.test_groups.min() < floors["min_matched_test_groups"]
            or s.calibration_decisions.median() < floors["min_median_matched_calibration_decisions"]
            or s.calibration_pages.median() < floors["min_median_matched_calibration_pages"]
        )
        result["matched"][d] = (
            "LOW SUPPORT / NOT ESTIMABLE"
            if low
            else (
                "materially changes"
                if signature(d, rc.SW, rc.MATCHED)[:3] != primary[:3] or m < 0.8
                else ("attenuates" if m < 0.85 or m - w < -0.02 else "qualitative result persists")
            )
        )
    primary_set = {d for d in directions if any(result["strict"][d]["primary_signature"][:3])}
    strict_set = {d for d in directions if any(result["strict"][d]["strict_signature"][:3])}
    reversed_case = (
        len(primary_set) == 1 and len(strict_set) == 1 and primary_set.isdisjoint(strict_set)
    ) or all(auc[key(d, rc.NI, rc.ORIGINAL, "winner")]["median"] < 0.8 for d in directions)
    result["strict_overall"] = (
        "REVERSED"
        if reversed_case
        else (
            "ROBUST"
            if not any(changes)
            else ("DIRECTION-DEPENDENT" if sum(changes) == 1 else "ATTENUATED")
        )
    )
    weak = any(auc[key(d, rc.SW, rc.ORIGINAL, "winner")]["median"] < 0.8 for d in directions)
    strong = all(auc[key(d, rc.SW, rc.ORIGINAL, "winner")]["median"] >= 0.85 for d in directions)
    navarro_both = all(
        result["primary_policy_verdicts"][d][rc.NAVARRO] == "USEFUL" for d in directions
    )
    a_missing = any(
        not any(v == "USEFUL" for v in result["primary_policy_verdicts"][d].values())
        for d in directions
    )
    result["added_policy_recovers_both"] = navarro_both
    result["thesis"] = (
        "NOT SUPPORTED"
        if weak
        else (
            "SUBSTANTIALLY REFRAMED"
            if navarro_both
            else (
                "PRESERVED WITH NARROWER CLAIMS"
                if strong and a_missing
                else "SUBSTANTIALLY REFRAMED"
            )
        )
    )
    result["population_sensitivity"] = (
        "LARGE"
        if all(v == "materially changes" for v in result["matched"].values())
        else (
            "MATERIAL"
            if any(
                v in ("materially changes", "attenuates", "LOW SUPPORT / NOT ESTIMABLE")
                for v in result["matched"].values()
            )
            else "LOW"
        )
    )
    result["decision"] = (
        "CENTRAL PAPER-3 THESIS REQUIRES REFRAMING"
        if result["thesis"] in ("NOT SUPPORTED", "SUBSTANTIALLY REFRAMED")
        else "CLEAN CORRECTIVE RESULTS SUPPORT NARROWED MANUSCRIPT"
    )
    return result


def compare_original(data: dict[str, Any]) -> list[dict[str, Any]]:
    # This is the first access to invalidated original performance in VF2.
    for run in [1, 2]:
        assert (ex.run_dir(run) / "manifests/result_freeze.json").exists()
    original = ex.read(rc.RESULTS_OUT)
    rows = []
    for d in pf1.DIRECTIONS:
        for h in [rc.SW, rc.NI]:
            for unit in ["all", "winner", "winner_minus_all"]:
                oldkey = f"{d}|pool|{h}|{rc.ORIGINAL}|{unit}"
                old = original["auroc_summaries"][oldkey]["median"]
                new = data["auroc_summaries"][key(d, h, rc.ORIGINAL, unit)]["median"]
                difference = new - old
                change = (
                    "STABLE"
                    if abs(difference) <= 0.02
                    else ("STRENGTHENED" if difference > 0.02 else "ATTENUATED")
                )
                rows.append(
                    {
                        "direction": d,
                        "harm_definition": h,
                        "population": rc.ORIGINAL,
                        "metric": unit + "_AUROC",
                        "policy": None,
                        "invalidated_original": old,
                        "clean_corrected": new,
                        "difference": difference,
                        "classification": change,
                        "old_source": str(rc.RESULTS_OUT.relative_to(ex.ROOT)),
                        "new_source": "results/paper3_vf2/run1/tables/scientific_results.json",
                    }
                )
            for p in [rc.PLUG_IN, rc.CONSERVATIVE, rc.NAVARRO]:
                old = original["policy_summaries"][f"{d}|pool|{h}|{rc.ORIGINAL}|{p}"]
                new = data["policy_summaries"][key(d, h, rc.ORIGINAL, p)]
                for metric in [
                    "median_acceptance_coverage",
                    "median_harm",
                    "median_recall",
                    "median_credited_recall",
                    "verdict",
                ]:
                    a, b = old[metric], new[metric]
                    delta = (
                        b - a
                        if isinstance(a, (int, float)) and isinstance(b, (int, float))
                        else None
                    )
                    changed = (
                        "VERDICT CHANGED"
                        if old["verdict"] != new["verdict"]
                        else (
                            "STABLE"
                            if delta is None or abs(delta) <= 0.02
                            else "ATTENUATED"
                            if (
                                (metric == "median_harm" and delta > 0)
                                or (metric != "median_harm" and delta < 0)
                            )
                            else "STRENGTHENED"
                        )
                    )
                    rows.append(
                        {
                            "direction": d,
                            "harm_definition": h,
                            "population": rc.ORIGINAL,
                            "metric": metric,
                            "policy": p,
                            "invalidated_original": a,
                            "clean_corrected": b,
                            "difference": delta,
                            "classification": changed,
                            "old_source": str(rc.RESULTS_OUT.relative_to(ex.ROOT)),
                            "new_source": "results/paper3_vf2/run1/tables/scientific_results.json",
                        }
                    )
    analysis.csv(ex.DOC / "P3_VF2_CORRECTION_COMPARISON.csv", rows)
    return rows


def scientific_tables(rd: Path, data: dict[str, Any]) -> str:
    lines = [
        "# Corrected result tables",
        "",
        "All retained cells/draws/failures are in the immutable CSV tables. Intervals are per-quantity 95% grouped percentile intervals, calibration held fixed; they are not simultaneous.",
        "",
        "| Direction | Harm | Population | All AUROC (CI) | Winner AUROC (CI) | Paired difference (CI) |",
        "|---|---|---|---|---|---|",
    ]

    def fmt(v: Any) -> str:
        return "undefined" if v is None else f"{v:.4f}"

    def interval(v: dict[str, Any]) -> str:
        return f"{fmt(v['median'])} [{fmt(v['ci_low'])}, {fmt(v['ci_high'])}]"

    for d in pf1.DIRECTIONS:
        for h in [rc.SW, rc.NI]:
            for pop in [rc.ORIGINAL, rc.MATCHED]:
                a = data["auroc_summaries"]
                allkey = key(d, h, pop, "all")
                diffkey = key(d, h, pop, "winner_minus_all")
                lines.append(
                    f"| {d} | {h} | {pop} | {interval(a[allkey]) if allkey in a else 'not specified'} | {interval(a[key(d, h, pop, 'winner')])} | {interval(a[diffkey]) if diffkey in a else 'not specified'} |"
                )
    lines += [
        "",
        "| Direction | Harm | Population | Rule | Coverage | Harm among accepting draws | Uncredited recall | Credited recall | Verdict / practical |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for k, s in sorted(data["policy_summaries"].items()):
        d, arm, b, h, pop, rule = k.split("|")
        if b != "pool" or arm != pf1.M1:
            continue
        lines.append(
            f"| {d} | {h} | {pop} | {rule} | {fmt(s['median_acceptance_coverage'])} | {fmt(s['median_harm'])} | {fmt(s['median_recall'])} | {fmt(s['median_credited_recall'])} | {s.get('verdict', s.get('practical'))} |"
        )
    lines += [
        "",
        "Full grouped policy intervals are in scientific_results.json and result_registry.json, including all unfavorable rows. Harm medians condition on draws with acceptance; coverage and credited recall include abstaining/violating draws. Do not confuse these denominators.",
        "",
        "M10: NOT ESTIMABLE, no empirical certificate. B5 conservative: NOT ESTIMABLE BY DESIGN (one calibration page). Navarro has no formal guarantee; conservative is an approximate effective-sample procedure.",
        "",
        "Matched recall uses linked eligible error sites; secondary original denominator remains available from the frozen error-site manifest. B2 is a ranking comparator only; controls have no headline deployment role.",
    ]
    return "\n".join(lines) + "\n"


def finalize() -> None:
    ex.hard_gate()
    for run in [1, 2]:
        f = ex.read(ex.run_dir(run) / "manifests/result_freeze.json")
        for p, h in f["files_sha256"].items():
            assert ex.sha(ex.ROOT / p) == h, p
    determinism = verify_determinism()
    rd = ex.run_dir(1)
    data = ex.read(rd / "tables/scientific_results.json")
    result = classify(data, rd)
    ex.write(ex.OUT / "corrected_interpretation.json", result)
    # Establish the new scientific source of truth before accessing old values.
    registry = ex.read(rd / "manifests/result_registry.json")
    registry["corrective_interpretation"] = result
    ex.write(ex.DOC / "P3_VF2_RESULT_REGISTRY.json", registry)
    comparison = compare_original(data)
    text = scientific_tables(rd, data)
    ex.text_once(ex.DOC / "P3_VF2_RESULT_TABLES.md", text)
    ex.text_once(
        ex.DOC / "P3_VF2_RESULTS.md",
        "# Corrected scientific results\n\n"
        + ex.STATUS
        + "\n\n"
        + "The corrected registry controls final inference; original invalidated results are historical context only.\n\n"
        + json.dumps(result, indent=2)
        + "\n\nAll retained budget/draw/ranking/policy/support/controls outcomes, uncertainty and unavailable reasons are in run1 tables and policies. No favorable panel or failed cell was omitted. See P3_VF2_RESULT_TABLES.md for pool results and individual immutable tables for every budget.\n",
    )
    for run in [1, 2]:
        m = ex.read(ex.run_dir(run) / "manifests/model_registry.json")
        ex.text_once(
            ex.DOC / f"P3_VF2_RUN{run}.md",
            f"# Corrective Run {run}\n\n{ex.STATUS}\n\n"
            f"Requested {m['requested']}; completed {m['completed']}; non-estimable {m['non_estimable']}. All rows preserved. "
            "Resource, feature, model, score, policy and result manifests frozen under the isolated run namespace. "
            "No reference/legacy reintroduction or candidate generation.\n",
        )
    ex.text_once(
        ex.DOC / "P3_VF2_DETERMINISM.md",
        "# Deterministic corrective repeat\n\n"
        + f"Full Run 1 and Run 2: **{determinism['status']}** across {len(determinism['checks'])} artifact checks. "
        + determinism["normalization_note"]
        + "\n\nFrozen tolerance: 1e-12 absolute/relative. Resources, features, predictors, score arrays, thresholds, policy metrics, bootstrap arrays and figure data checked; no averaging.\n",
    )
    ex.text_once(
        ex.DOC / "P3_VF2_CORRECTION_IMPACT.md",
        "# Original invalidated versus clean corrected\n\n"
        "Both corrected runs/results were frozen before the invalidated outcome artifacts were loaded. "
        "The comparison is descriptive: honest budget allocation, support and resources changed together; it does not isolate a causal leakage effect. "
        "Classification uses the pre-frozen 0.02 difference threshold and verdict precedence, not significance. "
        "Every primary/strict pool AUROC and retained primary operating quantity appears in P3_VF2_CORRECTION_COMPARISON.csv.\n\n"
        + json.dumps(
            pd.Series([r["classification"] for r in comparison]).value_counts().to_dict(), indent=2
        )
        + "\n\n"
        + f"Clean thesis classification: {result['thesis']}.\n",
    )
    claim_audit(data, result)
    disclosure(result)
    plan(data, result)
    stage_release()
    comparison_figure(comparison)
    protection = ex.read(ex.OUT / "preparation/source_audit.json")
    for scope in ["protected_sha256", "inputs_sha256"]:
        for p, h in protection[scope].items():
            assert ex.sha(ex.ROOT / p) == h, p
    integrity = dict.fromkeys(
        [
            "new_OCR",
            "new_decoding",
            "new_generator",
            "feature_definitions_changed",
            "model_architecture_changed",
            "hyperparameters_tuned",
            "split_changed_after_outcomes",
            "draws_redrawn_after_outcomes",
            "test_block_changed",
            "reserve_outcomes_inspected",
            "historical_invalid_evidence_reintroduced",
            "test_labels_used_for_fitting",
            "test_groups_used_for_resource_fitting",
            "adaptation_calibration_overlap",
            "manuscript_edited",
            "original_registries_overwritten",
            "commit",
            "push",
            "submission",
            "DOI",
        ],
        False,
    )
    ex.write(
        ex.OUT / "integrity_record.json",
        {
            "integrity": integrity,
            "candidate_population": "unchanged frozen candidates except deterministic clean-role exclusions",
            "protected_originals_unchanged": True,
        },
    )
    ex.text_once(
        ex.DOC / "P3_VF2_SUBMISSION_READINESS.md",
        "# Corrective scientific readiness\n\n"
        + f"Status: {ex.STATUS}. Thesis: {result['thesis']}. Full repeat: {determinism['status']}. "
        + "All corrected numerical claims are bound by the new result registry; no manuscript edits or public release. "
        + "Finite calibration/OCR-D support and previously observed test history constrain interpretation. "
        + "AUTHOR INPUT REQUIRED release metadata are recorded separately and must be completed before submission/release. "
        + "Unfavorable corrected outcomes require narrowing/reframing, not another experiment. No new validity defect was found by these execution gates.\n\n"
        + json.dumps(integrity, indent=2)
        + "\n\n"
        + result["decision"]
        + "\n",
    )
    ex.write(
        ex.OUT / "final_delivery_manifest.json",
        {
            "docs_sha256": {
                str(p.relative_to(ex.ROOT)): ex.sha(p) for p in sorted(ex.DOC.glob("P3_VF2_*"))
            },
            "run_freezes_sha256": {
                str(ex.run_dir(i).relative_to(ex.ROOT)): ex.sha(
                    ex.run_dir(i) / "manifests/result_freeze.json"
                )
                for i in [1, 2]
            },
            "decision": result["decision"],
            "determinism": determinism["status"],
            "integrity": integrity,
        },
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "thesis": result["thesis"],
                "determinism": determinism["status"],
            }
        ),
        flush=True,
    )


def claim_audit(data: dict[str, Any], result: dict[str, Any]) -> None:
    winners = result["winner"]
    claims = {
        "A ranking discrimination recovers strongly": "SUPPORTED WITH QUALIFICATION"
        if all(
            data["auroc_summaries"][key(d, rc.SW, rc.ORIGINAL, "all")]["median"] >= 0.85
            for d in pf1.DIRECTIONS
        )
        else "NOT SUPPORTED",
        "B winner discrimination remains strong": "SUPPORTED WITH QUALIFICATION"
        if all(v not in ("NOT SUPPORTED", "ATTENUATED") for v in winners.values())
        else "NOT SUPPORTED",
        "C plug-in direction dependence": "SUPPORTED WITH QUALIFICATION"
        if len({result["primary_policy_verdicts"][d][rc.PLUG_IN] for d in pf1.DIRECTIONS}) > 1
        else "NOT SUPPORTED",
        "D conservative safe mainly by abstention/low usefulness": "SUPPORTED WITH QUALIFICATION"
        if all(
            result["primary_policy_verdicts"][d][rc.CONSERVATIVE] != "USEFUL"
            for d in pf1.DIRECTIONS
        )
        else "NOT SUPPORTED",
        "E strict harm changes conclusions": "SUPPORTED WITH QUALIFICATION"
        if result["strict_overall"] != "ROBUST"
        else "NOT SUPPORTED",
        "F matched population changes conclusions": "SUPPORTED WITH QUALIFICATION"
        if result["population_sensitivity"] != "LOW"
        else "NOT SUPPORTED",
        "G Navarro does not consistently recover both directions": "SUPPORTED WITH QUALIFICATION"
        if not result["added_policy_recovers_both"]
        else "NOT SUPPORTED",
        "H finite label evidence remains limiting": "SUPPORTED WITH QUALIFICATION",
        "Legacy no-shift contextual empirical comparison": "REMOVE",
        "Legacy acquisition balancing effectiveness": "REMOVE",
    }
    ex.text_once(
        ex.DOC / "P3_VF2_CLAIM_AUDIT.md",
        "# Corrected claim audit\n\n"
        + "Every status is controlled by clean results, within evaluated generators/populations/rules and finite evidence. No generator-only causal claim, general calibration failure, arbitrary-generator generalization or formal-risk-control failure. M10 was not estimable.\n\n"
        + "\n".join(f"- {k}: **{v}**" for k, v in claims.items())
        + "\n\n"
        + "Claim H refers to observed finite geometry, pointwise approximate conservative evidence and small-budget non-estimability; it does not establish sufficiency of a larger future budget. Exact quantities: P3_VF2_RESULT_REGISTRY.json.\n",
    )


def disclosure(result: dict[str, Any]) -> None:
    ex.text_once(
        ex.DOC / "P3_VF2_DISCLOSURE_RECORD.md",
        "# Factual split-defect and correction record\n\n"
        "The original internally preregistered evaluation is invalidated for final inference. Exact PF1 test pages/labels did not enter upstream fitting, but same-volume pages from alberti_pictura_1540 and estor_rechtsgelehrsamkeit02_1758 entered labelled fitting/calibration. Five other pages of glauber_opera01_1658 entered label-free fitted ranking resources. Adaptation/calibration volume groups also overlapped. Pre-submission AC1/VF1/VF1B audits found these defects. Older RL1 development fitting also included the Euler test volume; those predictors are not retained.\n\n"
        "VF1B clean group memberships, budget allocation, resource exclusion, model/policy scope and historical-reference removals were bound before this rerun. VF2 resources/features were rebuilt and checked, then all implementation/config/protocol hashes frozen BEFORE fitting. Every threshold calibrates out-of-fit group scores. The previously observed 66-page test was retained transparently. CORD reserve remains blind and unsupported; no independent confirmation is claimed. Six-volume-crossing historical reference evidence and stratified legacy comparisons were dropped without erasing history.\n\n"
        + f"Corrected thesis: {result['thesis']}; strict sensitivity: {result['strict_overall']}; population sensitivity: {result['population_sensitivity']}. "
        + "All original-versus-corrected changes are recorded, including unfavorable values. No polished manuscript prose or manuscript editing occurs here.\n",
    )


def plan(data: dict[str, Any], result: dict[str, Any]) -> None:
    sections = [
        "Title",
        "Abstract",
        "Introduction",
        "Contributions",
        "Calibration/operating-point terminology",
        "Data splits",
        "Labelled-page budgets",
        "Resource fitting",
        "Ranking evaluation",
        "Policy methods",
        "RQ1",
        "RQ2",
        "RQ3",
        "Strict-harm sensitivity",
        "Matched-site sensitivity",
        "Discussion",
        "Limitations",
        "Conclusion",
        "Table 2",
        "Main figures",
        "Supplement",
        "Reproducibility statement",
    ]
    rows = [
        "# Manuscript change plan — no manuscript edits",
        "",
        "All proposed numbers must be read from P3_VF2_RESULT_REGISTRY.json. Invalidated original results are disclosure/history only. No external preregistration implication; say internally preregistered in a time-stamped design record where true.",
        "",
        "| Section | Required scientific change | Scope/evidence |",
        "|---|---|---|",
    ]
    for s in sections:
        action = "Replace retained empirical values with corrected registry; narrow to evaluated generators, populations, policies and finite labelled-page evidence."
        if s == "Data splits":
            action = "Disclose original volume-level defect and corrective status; show clean fitting/adaptation/calibration/resource/test disjointness."
        elif s == "Labelled-page budgets":
            action = "B=adaptation+calibration unique pages; disclose source labels separately and generator-specific zero-budget source calibration."
        elif s == "Resource fitting":
            action = "Report five OCR-D records rebuilt without Glauber; exact resource lists/manifests and unchanged feature definitions."
        elif s == "Table 2":
            action = "Drop invalidated known-reference/no-shift rows; do not replace them with unrun legacy reconstructions."
        elif s == "Main figures":
            action = "Use corrected VF2-1 through VF2-5 data; audit-only VF2-6 need not enter manuscript."
        elif s == "Supplement":
            action = "Keep ten fit-label controls, random score/corpus/strict/matched sensitivity, all failures; remove stratified legacy claim."
        elif s == "Limitations":
            action = "Previously observed test, no independent confirmation, scarce OCR-D volumes, harm/population sensitivity and unlike policy guarantees."
        elif s == "Reproducibility statement":
            action = "Use staged corrected manifest; complete author/repository/licensing metadata before public release; no DOI claimed."
        rows.append(f"| {s} | {action} | {result['thesis']}; corrected registry |")
    rows += [
        "",
        "Study-defined usefulness/coverage criteria are not industry standards. Separate harm-credited and uncredited exact-repair recall. Explain page-query ranking versus site-winner decisions and global thresholds early. Avoid strong OCR-D corpus interpretation from small support. Generator identity is not causally isolated.",
    ]
    ex.text_once(ex.DOC / "P3_VF2_MANUSCRIPT_CHANGE_PLAN.md", "\n".join(rows) + "\n")


def stage_release() -> None:
    stage = ex.OUT / "release_candidate"
    archive = ex.OUT / "archive_staging"
    assert not stage.exists() and not archive.exists()
    stage.mkdir()
    archive.mkdir()
    paths = [
        ex.CONFIG,
        ex.REGISTRY,
        ex.FREEZE,
        ex.DOC / "P3_VF2_PROTOCOL.md",
        ex.DOC / "P3_VF2_RESULT_REGISTRY.json",
        ex.ROOT / "uv.lock",
        ex.ROOT / "pyproject.toml",
        *ex.implementation_files(),
    ]
    paths += [
        p
        for folder in ["resources", "features", "scores", "policies", "tables", "manifests"]
        for p in (ex.run_dir(1) / folder).rglob("*")
        if p.is_file()
    ]
    paths += list((ex.OUT / "preparation").glob("*.csv"))
    for p in paths:
        target = stage / p.relative_to(ex.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    # Candidate replay metadata/labels are frozen artifacts; images and raw response files are not copied.
    cells = ex.read(ex.REGISTRY)["cells"]
    ids = sorted(
        {
            i
            for c in cells
            for role in [
                "source_fit_candidate_ids",
                "target_adapt_candidate_ids",
                "calibration_candidate_ids",
                "test_candidate_ids",
            ]
            for i in c[role]
        }
    )
    replay = ex.label_rows(ids)
    replay.to_parquet(stage / "candidate_replay_labels.parquet", index=False)
    manifest = {
        "staged_sha256": {
            str(p.relative_to(stage)): ex.sha(p) for p in sorted(stage.rglob("*")) if p.is_file()
        },
        "raw_images_redistributed": False,
        "public_release": False,
        "DOI_created": False,
        "AUTHOR_INPUT_REQUIRED": [
            "Final public repository URL and immutable release reference",
            "Author identities/ORCIDs and archive ownership",
            "Dataset/model/candidate-text redistribution permission and attribution sign-off",
            "Archive metadata and DOI creation after author approval",
        ],
        "replay_note": "Private staging only; full source dependency/input manifests retained. No image/raw-annotation redistribution.",
    }
    ex.write(stage / "reproducibility_manifest.json", manifest)
    shutil.copytree(stage, archive / "clean_release_candidate")
    ex.text_once(
        ex.DOC / "P3_VF2_REPRODUCIBILITY_READINESS.md",
        "# Private reproducibility staging\n\n"
        "Corrected result registry, clean IDs, resources, features, scores, policy tables/config, lockfile and implementation/scripts are staged in results/paper3_vf2/release_candidate and copied to archive_staging. Candidate replay metadata/labels are private; no images/raw annotation corpus was copied. No public release, push or DOI.\n\n"
        + "AUTHOR INPUT REQUIRED:\n\n"
        + "\n".join("- " + s for s in manifest["AUTHOR_INPUT_REQUIRED"])
        + "\n\n"
        + "Scientific reproduction staging exists; public submission/release readiness remains conditional on those fields and licensing sign-off.\n",
    )


def comparison_figure(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d in zip(axes, pf1.DIRECTIONS, strict=True):
        b = frame[
            (frame.direction == d)
            & (frame.harm_definition == rc.SW)
            & frame.metric.isin(["all_AUROC", "winner_AUROC"])
        ]
        x = np.arange(len(b))
        ax.bar(
            x - 0.15, b.invalidated_original.astype(float), width=0.3, label="Invalidated original"
        )
        ax.bar(x + 0.15, b.clean_corrected.astype(float), width=0.3, label="Clean corrected")
        ax.set(title=d, xticks=x, xticklabels=b.metric, ylabel="Harm AUROC", ylim=(0.5, 1))
        ax.legend(fontsize=8)
    fig.suptitle("Audit comparison; original values are invalidated")
    fig.tight_layout()
    for run in [1, 2]:
        for ext in ["png", "pdf", "svg"]:
            meta = (
                {"CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else {"Date": None}
                if ext == "svg"
                else {}
            )
            fig.savefig(
                ex.run_dir(run) / "figures" / f"VF2-6_correction_comparison.{ext}",
                dpi=180,
                metadata=meta,
            )
        ex.write(
            ex.run_dir(run) / "figures/VF2-6_manifest.json",
            {
                "created_after_corrected_result_freeze": True,
                "source": str((ex.DOC / "P3_VF2_CORRECTION_COMPARISON.csv").relative_to(ex.ROOT)),
                "source_sha256": ex.sha(ex.DOC / "P3_VF2_CORRECTION_COMPARISON.csv"),
                "original_invalidated": True,
                "analysis_only": True,
            },
        )
    plt.close(fig)


if __name__ == "__main__":
    finalize()

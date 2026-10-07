#!/usr/bin/env python3
"""Prospectively frozen corrective re-analysis. No decoding or legacy stage runner."""
# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper3_vf1b_decontamination as audit
import sgv_pf1_page_frontier_scaling as pf1
import sgv_rk1_learning_to_rank as rk1
import sgv_rk3_risk_aware_ranking as rk3
import sgv_rl1_generator_agnostic_reliability as rl1

DOC = ROOT / "docs/paper3/journal_track_2027/validity_recovery"
OUT = ROOT / "results/paper3_vf2"
VF1B = ROOT / "results/paper3_vf1b"
CONFIG = DOC / "P3_VF2_CONFIG.json"
REGISTRY = DOC / "P3_VF2_EXECUTION_REGISTRY.json"
FREEZE = DOC / "P3_VF2_EXECUTION_FREEZE.json"
STATUS = "PROSPECTIVELY FROZEN CORRECTIVE RE-ANALYSIS"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def text_once(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def run_dir(run: int) -> Path:
    return OUT / f"run{run}"


def validate(r: dict[str, Any]) -> dict[str, Any]:
    assert len(r["cells"]) == 492
    assert all(not h["retained_as_scientific_evidence"] for h in r["historical_references"])
    cal = {p for c in r["cells"] for p in c["calibration_page_ids"]}
    for c in r["cells"]:
        audit.validate_cell(c, r["page_to_group"])
        assert set(c["test_candidate_ids"]).isdisjoint(c["calibration_candidate_ids"])
        assert set(c["target_adapt_candidate_ids"]).isdisjoint(c["calibration_candidate_ids"])
        if c["budget"] != "0":
            assert (
                len(c["target_adapt_pages"]) + len(c["target_calibrate_pages"])
                == c["target_label_pages"]
            )
    for resource in r["resource_inventory"]:
        if resource["retained"] and resource["class"] != "R0":
            audit.validate_resource(
                resource,
                r["cells"][0]["test_pages"],
                r["cells"][0]["reserve_confirmation_pages"],
                cal,
                r["page_to_group"],
            )
    assert not r["reserve_confirmation_enabled"]
    assert r["candidate_generation_reused"] and not r["candidate_regeneration_required"]
    return {
        "TEST_intersect_SOURCE_FIT": 0,
        "TEST_intersect_TARGET_ADAPT": 0,
        "TEST_intersect_TARGET_CALIBRATE": 0,
        "TEST_intersect_RESOURCE_FIT": 0,
        "TARGET_ADAPT_intersect_TARGET_CALIBRATE": 0,
        "reserve_outcomes_inspected": False,
        "historical_evidence_excluded": True,
        "candidate_generation_unchanged": True,
        "models_requested": 492,
    }


def prepare() -> None:
    delivery = read(VF1B / "P3_VF1B_FINAL_DELIVERY_MANIFEST.json")
    inputs = read(VF1B / "input_manifest.json")
    for records in (
        delivery["code_sha256"],
        delivery["deliverables_sha256"],
        inputs["inputs_sha256"],
        inputs["protected_sha256"],
    ):
        for name, expected in records.items():
            assert sha(ROOT / name) == expected, name
    r = read(VF1B / "clean_execution_registry.json")
    gates = validate(r)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == delivery["git_HEAD"]
    diff = subprocess.check_output(["git", "diff"], cwd=ROOT)
    assert hashlib.sha256(diff).hexdigest() == delivery["tracked_diff_sha256"]
    git = {
        "HEAD": head,
        "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }
    write(
        OUT / "preparation/source_audit.json",
        {
            "git": git,
            "gates": gates,
            "vf1b_delivery_sha256": sha(VF1B / "P3_VF1B_FINAL_DELIVERY_MANIFEST.json"),
            "inputs_sha256": inputs["inputs_sha256"],
            "protected_sha256": inputs["protected_sha256"],
            "corrected_outcomes_computed": False,
        },
    )
    text_once(OUT / "preparation/git_diff.txt", diff.decode())
    config = {
        "study": "P3-VF2",
        "analysis_status": STATUS,
        "workers": 6,
        "run_repetitions": 2,
        "determinism_atol": 1e-12,
        "determinism_rtol": 1e-12,
        "scientific_settings": r["frozen_scientific_settings"],
        "study_defined_operating_criteria": r["study_defined_operating_criteria"],
        "budget_semantics": r["budget_semantics"],
        "zero_budget_semantics": r["zero_budget_semantics"],
        "group_unit": {"funsd": "form/document", "ocrd_sbb": "volume"},
        "model_seeds": {
            "fit": rk3.FIT_SEED,
            "permutation": pf1.rk4.PERMUTATION_SEED,
            "random_control": pf1.CONTROL_SEED,
        },
        "learner_parameters": {
            "rounds": rk3.ROUNDS,
            "learning_rate": rk3.LEARNING_RATE,
            "max_depth": rk3.MAX_DEPTH,
            "min_samples_leaf": rk3.MIN_LEAF,
        },
        "source_inputs_sha256": inputs["inputs_sha256"],
        "candidate_regeneration": False,
        "reserve_access": False,
        "non_estimable_rule": "No fit rows, no multi-grade page query, or empty frozen tree fit => NOT ESTIMABLE; preserve cell, no redraw. Runtime exceptions stop execution.",
        "headline_scope": "M1 pool per direction; all budgets and B2/control outcomes also reported",
        "correction_change_rule": "Verdict changes take precedence; AUROC changes >0.02 classify strengthened/attenuated; coverage or credited-recall absolute change >0.02 classifies change, otherwise stable; no significance-based equivalence claim.",
        "thesis_rule": "Either primary pool winner AUROC<0.80 => NOT SUPPORTED. Navarro USEFUL in both directions => SUBSTANTIALLY REFRAMED. Otherwise strong winners>=0.85 in both directions with at least one direction lacking any stable USEFUL policy => PRESERVED WITH NARROWER CLAIMS. All other supported ranking/deployment contrasts => SUBSTANTIALLY REFRAMED; no evaluable direction => evidence too weak.",
    }
    write(CONFIG, config)
    r["study"] = "P3-VF2"
    r["analysis_status"] = STATUS
    r["execution_authorized"] = True
    r["authorization"] = (
        "Explicit user P3-VF2 instruction; no publishing, commit or manuscript permission"
    )
    write(REGISTRY, r)
    for name in [
        "clean_group_registry.csv",
        "resource_rebuild_registry.csv",
        "model_rebuild_registry.csv",
        "policy_rebuild_registry.csv",
    ]:
        path = OUT / "preparation" / name
        path.write_bytes((VF1B / name).read_bytes())
    text_once(
        DOC / "P3_VF2_SOURCE_AUDIT.md",
        "# P3-VF2 source audit\n\n" + STATUS + "\n\n"
        f"Git HEAD `{head}`. VF1B delivery, input and protected hashes verified. Existing six tracked modifications preserved. "
        "492 clean cells independently validate at exact-page and scientific-group levels. Historical references remain excluded. "
        "The 66-page/59-group test and blind unsupported CORD reserve are unchanged. "
        "Full git diff/status and membership gate: results/paper3_vf2/preparation/source_audit.json. No corrected performance inspected.\n",
    )
    text_once(DOC / "P3_VF2_PROTOCOL.md", protocol(config))
    print(json.dumps({"source_audit": "PASS", **gates}), flush=True)


def protocol(config: dict[str, Any]) -> str:
    return (
        """# P3-VF2 corrective execution protocol

PROSPECTIVELY FROZEN CORRECTIVE RE-ANALYSIS on previously observed test outcomes.
Not independent or untouched confirmation. No reserve outcomes, candidate generation, manuscript edits or release.

Exact inputs, IDs, resource exclusions, 492 retained fits and policy selectors are bound by
P3_VF2_EXECUTION_REGISTRY.json and input hashes in P3_VF2_CONFIG.json. Scientific groups are
FUNSD forms/documents and OCR-D volumes. Test remains 66 pages/59 groups. Source/target and
calibration scores must be out-of-fit at group level; source and target joint fitting is explicit.
Historical/reference and stratified-acquisition comparisons are excluded from final evidence.

Rebuild OCR-D environment resources from their original lists minus all test groups, without
backfill; reuse verified unchanged FUNSD resources. Call original RL1/RK3 feature builders with
GT-blind projected candidate inputs and clean resource bindings. Preserve context candidates on
retained pages so agreement/page features keep the original definition. Feature order and all
146 R5/93 R3 columns remain unchanged. Compare resource-independent and unaffected-environment
features exactly; explain changed resource-dependent columns before fitting. Any unexplained
difference or schema change stops execution. No OCR/generator function is invoked.

242 A0/M1 fits use frozen R5 risk-aware trees; 240 B2 fits use original R3 LambdaMART; ten existing
B100 controls permute target-adaptation grades within pages, five draws/direction. Source and
target labels are projected by exact fit candidate IDs. No validation/early stopping/tuning is
added. Empty frozen fits are NOT ESTIMABLE; preserve every draw. Runtime errors stop, never redraw.

B is unique target adaptation plus calibration pages, group-disjoint, with historical source
annotation costs separately disclosed. Numeric budgets 5/10/25/50/100 and full pool, 20 draws,
retain VF1B allocations and seed. Zero means zero additional target-generator labels with fixed
labelled source fitting and independent source calibration, not zero annotated evidence.

Thresholds use clean calibration winners only. Highest clean score wins each site; ties use
candidate ID ascending. Frozen plug-in, conservative, triage, Navarro widths (.05 primary,
.025/.1 sensitivity), C1 corpus and C3 hierarchical (2/5/10 prior pages) implementations remain
unchanged. Navarro may inspect unlabelled batch scores only. M10 is NOT ESTIMABLE and never run
as a certificate. Matched pool calibration is restricted by generator/site availability before
threshold selection; matched test is separately restricted. No ground truth chooses a winner.

Primary harm is STRICT_WORSENING; NON_IMPROVING is sensitivity only, using centralized taxonomy
and identical model scores. Test-label boundary and risk-coverage are analysis-only, isolated from
deployable selection. Exact repair recall reports both uncredited and harm-credited values and
the frozen linked error-site denominator. Deployment/usefulness/practical rules and all numeric
interpretation boundaries are copied verbatim in config; epsilon=.1, delta=.1, eta=.1.

All-candidate/winner AUROC and paired differences use median draw statistics. Uncertainty uses
2,000 corpus-stratified scientific-group bootstrap resamples with fixed CAL2 seed and calibration
held fixed. No candidate IID resampling, p-values or equivalence claims. Pool supplies headline
RC1/RC2/RC3/RC4 interpretations; all retained budgets, draws, failures and policies are reported.

Write isolated run1 and run2 resources/features/models/scores/policies/tables/figures/manifests/logs.
Freeze implementation and resource/feature manifests after integrity tests and BEFORE fitting.
Hash/freeze corrected result registries before loading invalidated outcome values for comparison.
Repeat the complete pipeline with identical seeds; exact output equality preferred, numeric
tolerance 1e-12 absolute/relative only for serialization/library rounding. Never average runs.

Figures: all/winner AUROC, winner risk-coverage, primary/strict harm, original/matched population,
policies, then original-invalidated/corrected comparison after both corrected registries freeze.
Tables: every model, score provenance, policy/draw, grouped summaries, ranking, matched support,
controls, unavailable M10, deterministic verification and correction impact. Every reported
number carries model/resource/code/artifact provenance in the corrected result registry.

Any group crossing, candidate-generation dependency defect, unexplained feature difference or
unexpected runtime failure stops. Preserve failed output; no silent overwrite. Post-freeze bugs
require a documented versioned freeze and affected rerun. Unfavorable performance alone never
justifies another experiment. The original outcome history is invalidated, preserved and disclosed.
"""
        + "\nConfig-frozen thesis and correction interpretation:\n\n"
        + json.dumps(
            {k: config[k] for k in ["thesis_rule", "correction_change_rule", "non_estimable_rule"]},
            indent=2,
        )
        + "\n"
    )


def environment(spec: dict[str, Any], permitted: set[str]) -> Any:
    """Canonical OCR/pixels only; restrict construction to explicit permitted pages."""
    import sgv14_confirmatory_validation as s14

    original = s14._document_bundles

    def bundles(corpus: str) -> dict[str, Any]:
        # Only FUNSD/OCR-D are callable here; never CORD reserve.
        assert corpus in ("funsd", "ocrd_sbb")
        return {k: v for k, v in original(corpus).items() if k in permitted}

    with pf1.rebound([(s14, "_document_bundles", bundles)]):
        return ORIGINAL_ENVIRONMENT(spec)


ORIGINAL_ENVIRONMENT = rl1._environment


def resources(run: int) -> None:
    r = read(REGISTRY)
    validate(r)
    old = read(rl1.FITTED_RESOURCES)
    records = {}
    checks = []
    for spec in rl1.environment_specs():
        name = spec["environment"]
        listed = next(
            x["proposed_fit_pages"]
            for x in r["resource_inventory"]
            if x["resource_id"].startswith(name + "::")
        )
        if listed == old["by_environment"][name]["documents"]:
            record = old["by_environment"][name]
            action = "REUSE VERIFIED CLEAN"
        else:
            env = environment(spec, set(listed))
            with pf1.rebound([(rl1, "_resource_documents", lambda e, pages=listed: pages)]):
                record = rl1.fit_environment_resources(env)
            assert set(record) == set(old["by_environment"][name])
            assert len(record["conf_quantiles"]) == len(
                old["by_environment"][name]["conf_quantiles"]
            )
            action = "REBUILT WITHOUT TEST GROUPS"
        assert record["documents"] == listed
        records[name] = record
        checks.append(
            {
                "environment": name,
                "action": action,
                "pages": listed,
                "groups": sorted({r["page_to_group"][p] for p in listed}),
                "record_sha256": hashlib.sha256(
                    json.dumps(record, sort_keys=True).encode()
                ).hexdigest(),
                "schema_matches": True,
            }
        )
        print(f"resources {run}: {name} {action} ({len(listed)} pages)", flush=True)
    write(
        run_dir(run) / "resources/fitted_resources.json",
        {"by_environment": records, "study": "P3-VF2", "fitted_from_ground_truth": False},
    )
    write(
        run_dir(run) / "manifests/resource_manifest.json",
        {
            "checks": checks,
            "gates": validate(r),
            "resource_file_sha256": sha(run_dir(run) / "resources/fitted_resources.json"),
        },
    )


BLIND_CANDIDATE_COLUMNS = (
    "candidate_id",
    "environment",
    "document_id",
    "lattice_site_id",
    "corrector_source",
    "arm",
    "original_ocr",
    "candidate_text",
    "char_start",
    "char_end",
    "generator_rank",
    "anchor_kind",
    "proposal_stratum",
)


def features(run: int) -> None:
    r = read(REGISTRY)
    validate(r)
    rd = run_dir(run)
    permitted_ids = {
        i
        for c in r["cells"]
        for k in (
            "source_fit_candidate_ids",
            "target_adapt_candidate_ids",
            "calibration_candidate_ids",
            "test_candidate_ids",
        )
        for i in c[k]
    }
    permitted_pages = {
        p
        for c in r["cells"]
        for k in (
            "source_fit_pages",
            "target_adapt_pages",
            "target_calibrate_pages",
            "source_calibration_zero_pages",
            "test_pages",
        )
        for p in c[k]
    }
    stages = [
        (
            "U5",
            rk1.RANKING_POPULATION,
            rl1.hy1.CANDIDATES,
            rl1.hy1.PROPOSAL_STRATA,
            rl1.hy1.CONTEXTS,
            rk3.FEATURE_MATRIX,
        ),
        (
            "PF1",
            pf1.POPULATION,
            pf1.CANDIDATES,
            pf1.PROPOSAL_STRATA,
            pf1.CORRECTION_CONTEXTS,
            pf1.FEATURE_MATRIX,
        ),
    ]
    results = []
    verification = []
    resource_path = rd / "resources/fitted_resources.json"
    for stage, pop_path, candidate_path, strata, contexts, old_path in stages:
        cols = list(dict.fromkeys([*rk3.OBSERVATION_COLUMNS, "corpus", "site_key", "population"]))
        cols = [x for x in cols if x in pq.read_schema(pop_path).names]
        population = pd.read_parquet(pop_path, columns=cols)
        if stage == "U5":
            population = population[population.population == rk1.POP_U5]
        population = population[population.document_id.isin(permitted_pages)].reset_index(drop=True)
        tmp_pop = rd / "features" / f"{stage}_blind_population.parquet"
        tmp_pop.parent.mkdir(parents=True, exist_ok=True)
        population.to_parquet(tmp_pop, index=False)
        ccols = [x for x in BLIND_CANDIDATE_COLUMNS if x in pq.read_schema(candidate_path).names]
        candidates = pd.read_parquet(candidate_path, columns=ccols)
        candidates = candidates[candidates.document_id.isin(permitted_pages)]
        tmp_candidates = rd / "features" / f"{stage}_blind_candidates.parquet"
        candidates.to_parquet(tmp_candidates, index=False)

        def local_env(spec: dict[str, Any], frame=population) -> Any:
            needed = set(frame.loc[frame.environment == spec["environment"], "document_id"])
            needed |= set(read(resource_path)["by_environment"][spec["environment"]]["documents"])
            return environment(spec, needed)

        with pf1.rebound(
            [(rl1, "FITTED_RESOURCES", resource_path), (rl1, "_environment", local_env)]
        ):
            matrix = pf1.build_feature_matrix(population, tmp_pop, tmp_candidates, strata, contexts)
        old = pd.read_parquet(old_path)
        old = old.set_index("candidate_id").loc[matrix.candidate_id]
        columns = rk3.columns_for(rk3.R5)
        assert columns == [c for c in matrix.columns if c not in ("candidate_id", "environment")]
        assert np.isfinite(matrix[columns].to_numpy()).all()
        new = matrix.set_index("candidate_id")
        stable_families = [
            c for c in columns if c.startswith(("text_", "edit_", "ctx_", "rx_agree_"))
        ]
        # Other families can be resource-dependent; unaffected environments must match in full.
        is_funsd = new.environment.str.startswith("funsd/").to_numpy()
        for col in columns:
            same = np.array_equal(new[col].to_numpy()[is_funsd], old[col].to_numpy()[is_funsd])
            assert same, f"Unexpected unaffected FUNSD change {stage}/{col}"
        for col in stable_families:
            assert np.array_equal(new[col].to_numpy(), old[col].to_numpy()), (
                f"Unexpected resource-independent change {stage}/{col}"
            )
        diffs = (new[columns].to_numpy() != old[columns].to_numpy()).sum(axis=0)
        verification.append(
            {
                "stage": stage,
                "rows": len(matrix),
                "feature_columns": columns,
                "unaffected_FUNSD_exact_match": True,
                "resource_independent_exact_match": True,
                "changed_column_counts": dict(zip(columns, map(int, diffs))),
                "expected_differences": "OCR-D resource-dependent families only",
                "unexpected_differences": 0,
            }
        )
        matrix = matrix[matrix.candidate_id.isin(permitted_ids)]
        results.append(matrix)
    matrix = (
        pd.concat(results, ignore_index=True)
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
    )
    assert not matrix.candidate_id.duplicated().any()
    assert set(matrix.candidate_id) == permitted_ids
    path = rd / "features/clean_feature_matrix.parquet"
    assert not path.exists()
    matrix.to_parquet(path, index=False)
    write(
        rd / "manifests/feature_manifest.json",
        {
            "candidate_ids": matrix.candidate_id.tolist(),
            "schema": list(matrix.columns),
            "resource_sha256": sha(resource_path),
            "feature_sha256": sha(path),
            "builders": ["rl1.build_features", "rk3.build_new_features"],
            "GT_feature_input_columns": [],
            "verification": verification,
        },
    )
    print(f"features {run}: {len(matrix)} rows, unchanged 146-column R5 schema", flush=True)


def implementation_files() -> list[Path]:
    local = sorted((ROOT / "scripts").glob("paper3_vf2_*.py"))
    local += sorted((ROOT / "tests/unit").glob("test_paper3_vf2*.py"))
    upstream = [ROOT / "scripts" / f"{m.__name__}.py" for m in [pf1, rk3, rl1, rk1]]
    upstream += [
        ROOT / "scripts" / n
        for n in [
            "paper3_rc1_reviewer_closure.py",
            "sgv_rk4_generator_adaptive_ranking.py",
            "sgv_dep1_deployment_frontier.py",
            "sgv_cal2_domain_stratified_calibration.py",
            "sgv_th1_threshold_calibration.py",
        ]
    ]
    # Freeze every repository-local implementation dependency, not only entrypoints.
    upstream += list((ROOT / "src/ocr_risk").rglob("*.py"))
    upstream += list((ROOT / "scripts").glob("sgv*.py"))
    return sorted(set(local + upstream))


def freeze() -> None:
    r = read(REGISTRY)
    config = read(CONFIG)
    gate = validate(r)
    for i in [1, 2]:
        assert not list((run_dir(i) / "models").glob("*.pkl")), (
            "Model artifact predates execution freeze"
        )
        assert not (run_dir(i) / "tables/scientific_results.json").exists()
    tests = read(OUT / "preparation/tests.json")
    assert tests["exit_code"] == 0
    files = [
        CONFIG,
        REGISTRY,
        DOC / "P3_VF2_PROTOCOL.md",
        ROOT / "uv.lock",
        *implementation_files(),
    ]
    files += sorted((OUT / "preparation").glob("*.csv"))
    files += sorted((run_dir(1) / "features").glob("*.parquet"))
    files += [
        run_dir(1) / "resources/fitted_resources.json",
        run_dir(1) / "manifests/resource_manifest.json",
        run_dir(1) / "manifests/feature_manifest.json",
        OUT / "preparation/resource_schema_comparison.json",
        OUT / "preparation/reconstruction_input_manifest.json",
        ROOT / "pyproject.toml",
    ]
    payload = {
        "study": "P3-VF2",
        "analysis_status": STATUS,
        "frozen_UTC": datetime.now(UTC).isoformat(),
        "git_HEAD": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git": read(OUT / "preparation/source_audit.json")["git"],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "Pillow", "matplotlib"]
        },
        "workers": config["workers"],
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "input_sha256": config["source_inputs_sha256"],
        "model_seeds": config["model_seeds"],
        "tests": tests,
        "gates": gate,
        "sequence": [
            ".venv/bin/python scripts/paper3_vf2_clean_execution.py fit --run 1",
            ".venv/bin/python scripts/paper3_vf2_analysis.py --run 1",
            ".venv/bin/python scripts/paper3_vf2_clean_execution.py resources --run 2",
            ".venv/bin/python scripts/paper3_vf2_clean_execution.py features --run 2",
            ".venv/bin/python scripts/paper3_vf2_clean_execution.py fit --run 2",
            ".venv/bin/python scripts/paper3_vf2_analysis.py --run 2",
            ".venv/bin/python scripts/paper3_vf2_reporting.py",
        ],
        "corrected_models_fitted_before_freeze": False,
    }
    write(FREEZE, payload)
    text_once(
        DOC / "P3_VF2_EXECUTION_FREEZE.md",
        f"# P3-VF2 execution freeze\n\n{STATUS}\n\n"
        "All file hashes, environment, seeds, 6 workers, membership gates and commands are bound in P3_VF2_EXECUTION_FREEZE.json. "
        "Resources/features rebuilt and checked; no corrected model has been fitted. No post-freeze scientific/code changes. "
        "Errors require preserved failure and versioned freeze.\n",
    )
    print(json.dumps(hard_gate()), flush=True)


def hard_gate() -> dict[str, Any]:
    f = read(FREEZE)
    for name, expected in f["files_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        == f["git_HEAD"]
    )
    return {
        **validate(read(REGISTRY)),
        "protocol_hash_matches": True,
        "implementation_hashes_match": True,
        "feature_schema_unchanged": True,
        "execution_freeze_sha256": sha(FREEZE),
    }


def fit_one(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    run, cell = task
    r = read(REGISTRY)
    audit.validate_cell(cell, r["page_to_group"])
    rd = run_dir(run)
    ids = cell["source_fit_candidate_ids"] + cell["target_adapt_candidate_ids"]
    ids = list(dict.fromkeys(ids))
    pop = label_rows(ids)
    pop = pop.set_index("candidate_id").loc[ids].reset_index()
    actual_fit_groups = {r["page_to_group"][p] for p in pop.document_id}
    assert actual_fit_groups.isdisjoint(cell["test_groups"])
    assert actual_fit_groups.isdisjoint(
        {r["page_to_group"][p] for p in cell["calibration_page_ids"]}
    )
    assert set(pop.document_id).issubset(set(cell["source_fit_pages"] + cell["target_adapt_pages"]))
    matrix = pd.read_parquet(rd / "features/clean_feature_matrix.parquet").set_index("candidate_id")
    cols = rk3.columns_for(cell["representation"])
    design = matrix.loc[ids, cols].to_numpy(np.float64)
    grades = pop.grade.to_numpy(np.int64).copy()
    groups = pop.document_id.astype(str).to_numpy()
    if cell["arm"] == "M1_TARGET_LABEL_PERMUTATION":
        target = set(cell["target_adapt_candidate_ids"])
        grades = pf1.rk4.shuffled_within_pages(
            grades, groups, np.flatnonzero(pop.candidate_id.isin(target).to_numpy()), cell["draw"]
        )
    if cell["arm"] == pf1.B2:
        model = rk1.fit_lambdamart(design, grades, groups)
    else:
        model = rk3.fit_boosted(
            design, rk3.utility_of(grades, rk3.RISK_UTILITY), groups, rk3.OBJ_RISK
        )
    usable = bool(model.trees)
    fileid = hashlib.sha256(cell["model_id"].encode()).hexdigest()[:20]
    model_path = rd / "models" / f"{fileid}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    assert not model_path.exists()
    with model_path.open("xb") as f:
        pickle.dump(model, f, protocol=5)
    rows = []
    for role, candidates in [
        ("calibration", cell["calibration_candidate_ids"]),
        ("test", cell["test_candidate_ids"]),
    ]:
        scores = model.score(matrix.loc[candidates, cols].to_numpy(np.float64))
        rows.extend(
            {"candidate_id": i, "score": float(s), "role": role}
            for i, s in zip(candidates, scores, strict=True)
        )
    score_path = rd / "scores" / f"{fileid}.parquet"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    assert not score_path.exists()
    pd.DataFrame(rows).to_parquet(score_path, index=False)
    manifest = {
        **{
            k: cell[k]
            for k in [
                "model_id",
                "direction",
                "scheme",
                "arm",
                "budget",
                "draw",
                "representation",
                "source_fit_groups",
                "target_adapt_groups",
                "target_calibrate_groups",
                "test_groups",
            ]
        },
        "seed": rk3.FIT_SEED,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha(model_path),
        "score_path": str(score_path.relative_to(ROOT)),
        "score_sha256": sha(score_path),
        "feature_sha256": sha(rd / "features/clean_feature_matrix.parquet"),
        "calibration_groups": sorted({r["page_to_group"][p] for p in cell["calibration_page_ids"]}),
        "calibration_page_ids": cell["calibration_page_ids"],
        "fit_candidate_ids": ids,
        "trees": len(model.trees),
        "status": "ESTIMABLE" if usable else "NOT ESTIMABLE",
        "reason": None
        if usable
        else "Frozen learner returned no tree / insufficient multi-grade page-query support",
        "calibration_OUT_OF_FIT_fraction": 1.0,
        "test_group_overlap": 0,
    }
    write(rd / "manifests/models" / f"{fileid}.json", manifest)
    return manifest


def label_rows(wanted: list[str]) -> pd.DataFrame:
    """Explicit candidate projection; no reserve table or other candidate labels are loaded."""
    columns = [
        "candidate_id",
        "document_id",
        "corpus",
        "environment",
        "site_key",
        "site_group",
        "corrector_sources",
        "grade",
        "is_harmful",
        "beneficial",
        "exact",
        "outcome",
    ]
    records = []
    for path in (rk1.RANKING_POPULATION, pf1.POPULATION):
        filters = [("candidate_id", "in", wanted)]
        if path == rk1.RANKING_POPULATION:
            filters.append(("population", "==", rk1.POP_U5))
        records.append(pd.read_parquet(path, columns=columns, filters=filters))
    rows = pd.concat(records, ignore_index=True).drop_duplicates("candidate_id")
    assert set(rows.candidate_id) == set(wanted)
    return rows


def fit(run: int) -> None:
    print(json.dumps(hard_gate()), flush=True)
    r = read(REGISTRY)
    config = read(CONFIG)
    rd = run_dir(run)
    for sub in ["models", "scores", "policies", "tables", "figures", "manifests", "logs"]:
        (rd / sub).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        manifests = []
        for m in pool.map(fit_one, [(run, c) for c in r["cells"]]):
            manifests.append(m)
            print(f"fit run{run} {len(manifests)}/492 {m['status']} {m['model_id']}", flush=True)
    write(
        rd / "manifests/model_registry.json",
        {
            "models": manifests,
            "execution_freeze_sha256": sha(FREEZE),
            "requested": 492,
            "completed": len(manifests),
            "non_estimable": sum(m["status"] != "ESTIMABLE" for m in manifests),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["prepare", "resources", "features", "freeze", "fit", "gate"]
    )
    parser.add_argument("--run", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.phase in ("prepare", "freeze"):
        globals()[args.phase]()
    elif args.phase == "gate":
        print(json.dumps(hard_gate()))
    else:
        globals()[args.phase](args.run)


if __name__ == "__main__":
    main()

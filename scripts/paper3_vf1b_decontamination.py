# ruff: noqa: E501
"""VF1B audit/design only: identifiers, provenance, and restricted source class counts.

No experiment/model/score/policy implementation is imported or executed. Reserve
tables and outcomes cannot enter the projected reader. Outputs are new VF1B paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper3_vf1b"
DOC = ROOT / "docs/paper3/journal_track_2027/validity_recovery"
IDS = frozenset(
    {
        "population",
        "candidate_id",
        "document_id",
        "corpus",
        "block",
        "fold",
        "corrector_sources",
        "site_group",
        "site_id",
        "environment",
        "page_id",
        "partition_group",
        "role",
        "split",
        "direction",
        "arm",
        "scheme",
        "budget",
        "draw",
        "evaluation_set",
    }
)
INPUTS: dict[str, str] = {}
READS: list[dict] = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def js(path):
    return json.loads(bind(path).read_text())


def ids(path, columns):
    assert columns and set(columns) <= IDS, "Forbidden value column"
    READS.append({"path": str(path), "columns": columns, "scope": "identifiers only"})
    return pd.read_parquet(bind(path), columns=columns)


def source_counts(path, candidate_ids, metadata):
    """Only already-labelled SOURCE fitting rows; never adaptation/test/reserve rows.

    Establish membership from metadata first, then push down the exact candidate
    predicate before materializing labels. Return class/query counts, no rates.
    """
    allowed = set(candidate_ids)
    view = metadata[metadata.candidate_id.isin(allowed)]
    assert len(view) == len(allowed) and set(view.block) == {"fit"}
    READS.append(
        {
            "path": path,
            "columns": ["candidate_id", "is_harmful", "grade"],
            "filters": {"candidate_id_in": sorted(allowed)},
            "scope": "SOURCE fitting only; mathematical fit feasibility",
        }
    )
    frame = pd.read_parquet(
        bind(path),
        columns=["candidate_id", "is_harmful", "grade"],
        filters=[("population", "==", "u5_dual_union"), ("candidate_id", "in", sorted(allowed))],
    )
    assert set(frame.candidate_id) == allowed
    frame = frame.merge(view[["candidate_id", "document_id"]], validate="one_to_one")
    return {
        "rows": len(frame),
        "harmful_count": int(frame.is_harmful.sum()),
        "non_harmful_count": int((~frame.is_harmful).sum()),
        "grade_counts": {str(k): int(v) for k, v in frame.grade.value_counts().items()},
        "multi_grade_queries": int((frame.groupby("document_id").grade.nunique() > 1).sum()),
        "label_read_scope": "exact source-only candidate IDs on cleaned fit pages",
    }


def groups(pages, pg):
    return sorted({pg[p] for p in pages})


def validate_cell(c, pg):
    roles = (
        "source_fit",
        "target_adapt",
        "target_calibrate",
        "source_calibration_zero",
        "test",
        "reserve_confirmation",
        "resource_fit",
    )
    p = {r: set(c.get(r + "_pages", [])) for r in roles}
    p["resource_fit"].update(c.get("external_resource_fit_pages", []))
    pairs = [
        ("target_adapt", "target_calibrate"),
        ("source_fit", "target_calibrate"),
        ("resource_fit", "target_calibrate"),
        ("test", "reserve_confirmation"),
    ]
    pairs += [
        (h, u)
        for h in ("test", "reserve_confirmation")
        for u in (
            "source_fit",
            "target_adapt",
            "target_calibrate",
            "resource_fit",
            "source_calibration_zero",
        )
    ]
    pairs += [("source_calibration_zero", u) for u in ("source_fit", "resource_fit")]
    for a, b in pairs:
        assert not p[a] & p[b], f"page crossing {a}/{b}"
        assert not set(groups(p[a], pg)) & set(groups(p[b], pg)), f"group crossing {a}/{b}"
    assert len(p["target_adapt"]) + len(p["target_calibrate"]) == c["target_label_pages"]


def validate_resource(resource, test, reserve, calibration, pg):
    assert resource["class"] != "R4", "Ambiguous retained resource"
    for held in (test, reserve, calibration):
        assert not set(resource["proposed_fit_pages"]) & set(held), "resource page crossing"
        assert not set(groups(resource["proposed_fit_pages"], pg)) & set(groups(held, pg)), (
            "resource group crossing"
        )


def validate_reference(record, pg):
    if record["retained_as_scientific_evidence"]:
        assert record["disposition"] == "REBUILD CLEANLY"
        assert record["clean_reconstruction_cells"], "Missing retained reference reconstruction"
        for c in record["clean_reconstruction_cells"]:
            validate_cell(c, pg)


def write_json(path, value):
    assert not path.exists(), f"Do not overwrite {path}"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    assert not path.exists(), f"Do not overwrite {path}"
    frame = pd.DataFrame(rows)
    for col in frame:
        frame[col] = frame[col].map(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
        )
    frame.to_csv(path, index=False)


def main():
    assert not OUT.exists() and not list(DOC.glob("P3_VF1B_*"))
    manifest = js("results/paper3_vf1/P3_VF1_FINAL_DELIVERY_MANIFEST_v2.json")
    protected = {}
    for key in ("artifacts_sha256", "code_sha256", "inputs_sha256"):
        for path, digest in manifest[key].items():
            assert sha(ROOT / path) == digest, path
            protected[path] = digest
    for p in (ROOT / "paper3_icdar_ijdar_2027").rglob("*"):
        if p.is_file():
            protected[str(p.relative_to(ROOT))] = sha(p)
    # Read all delivered VF1 documents/tables, not only their summaries.
    handoff = {}
    for p in DOC.glob("P3_VF1_*"):
        bind(p)
        if p.suffix == ".md":
            handoff[p.name] = {"characters_read": len(p.read_text())}
        elif p.suffix == ".csv":
            f = pd.read_csv(p)
            handoff[p.name] = {"rows": len(f), "columns": list(f)}
        else:
            handoff[p.name] = {"keys": list(json.loads(p.read_text()))}
    registry = js("results/paper3_vf1/proposed_clean_split_registry.json")
    pg, corpus = registry["page_to_group"], registry["page_to_corpus"]
    oldpath = "results/generated/sgv_rk1_learning_to_rank/ranking_population.parquet"
    cols = [
        "candidate_id",
        "document_id",
        "corpus",
        "block",
        "fold",
        "corrector_sources",
        "site_group",
        "environment",
    ]
    old = ids(oldpath, ["population", *cols])
    old = old[old.population == "u5_dual_union"].drop(columns="population").reset_index(drop=True)
    new = ids("results/generated/sgv_pf1/population.parquet", cols)
    pop = pd.concat([old, new], ignore_index=True)
    assert not pop.candidate_id.duplicated().any()
    page_pool = ids(
        "results/generated/sgv_pf1/page_pool.parquet",
        ["document_id", "corpus", "partition_group", "split"],
    )
    volumes = {v["volume_id"] for v in js("manifests/datasets/ocrd_sbb.json")["volumes"]}
    assert all(pg[p] == g for p, g in zip(page_pool.document_id, page_pool.partition_group))
    assert {pg[p] for p in pop[pop.corpus == "ocrd_sbb"].document_id} <= volumes
    assert all(pg[p] == p for p in pop[pop.corpus == "funsd"].document_id)
    test = sorted(set(new[new.block == "pf1_test"].document_id))
    assert len(test) == 66 and len(groups(test, pg)) == 59
    role = js("manifests/sgv1/role_manifest.json")
    lock = js("manifests/sgv1/confirmatory_reserve_lock.json")
    assert lock["status"] == "LOCKED" and lock["unlock_record"] is None
    assert lock["role_manifest_sha256"] == sha(ROOT / lock["role_manifest_path"])
    assert lock["pre_access_snapshot_sha256"] == sha(bind(lock["pre_access_snapshot_path"]))
    reserve = sorted(p for p, r in role["role_of"].items() if r == "CONFIRMATORY")
    train = sorted(p for p, r in role["role_of"].items() if r == "TRAIN")
    assert len(reserve) == 177 and len(groups(reserve, pg)) == 164
    assert not set(pop.document_id) & set(reserve)
    reserve_manifest = js(
        "docs/paper3/journal_track_2027/validity_recovery/P3_VF1_RESERVE_MANIFEST_LABEL_FREE.json"
    )
    assert registry["reserve_manifest_sha256"] == sha(
        ROOT / "docs/paper3/journal_track_2027/"
        "validity_recovery/P3_VF1_RESERVE_MANIFEST_LABEL_FREE.json"
    )
    assert reserve_manifest  # IDs/hash only, no raw reserve data.
    for c in registry["cells"]:
        validate_cell(c, pg)
    # Independently recover all native model membership from source-only rows,
    # explicit CAL2 page draws and exact score-cell identifiers; no score values.
    native = js("results/paper3_ac1/P3_AC1_MODEL_MEMBERSHIP.json")
    native_scores = ids(
        "results/generated/sgv_pf1/cell_scores.parquet",
        ["direction", "arm", "scheme", "budget", "draw", "evaluation_set", "candidate_id"],
    )
    kcols = ["direction", "arm", "scheme", "budget", "draw", "evaluation_set"]
    score_sets = {
        tuple(str(x) for x in k): set(v.candidate_id)
        for k, v in native_scores.groupby(kcols, sort=False)
    }
    draws_path = bind(
        "results/generated/sgv_cal2_domain_stratified_calibration/page_budget_draws.csv"
    )
    dc = ["direction", "scheme", "budget", "draw", "page_id", "role"]
    draw_table = pd.read_csv(draws_path, usecols=dc, dtype={"budget": str})
    READS.append({"path": str(draws_path.relative_to(ROOT)), "columns": dc, "scope": "IDs/roles"})
    drawsets = {tuple(str(x) for x in k): v for k, v in draw_table.groupby(dc[:4], sort=False)}
    source_name = {
        "current_to_qwen": "c0_frozen_generator",
        "qwen_to_current": "c1_image_corrector",
    }
    target_name = {
        d: "c1_image_corrector" if s == "c0_frozen_generator" else "c0_frozen_generator"
        for d, s in source_name.items()
    }
    membership = pop.corrector_sources.str.split("|")
    for model_id, v in native.items():
        c = v["score_cell"]
        d = c["direction"]
        key = tuple(str(c[k]) for k in kcols[:-1])
        sf = pop[
            (pop.block == "fit")
            & membership.map(lambda s, d=d: source_name[d] in s and target_name[d] not in s)
        ]
        expected_sf = set(sf.candidate_id) if c["arm"] != "b2_rk2_target_only" else set()
        assert c["arm"] in {"a0_zero_shot", "m1_full_adaptation", "b2_rk2_target_only"}
        assert expected_sf == set(v["source_label_candidate_ids"]), model_id
        for stage, field in [
            ("calibration", "calibration_candidate_ids"),
            ("test", "test_candidate_ids"),
        ]:
            assert score_sets[(*key, stage)] == set(v[field]), model_id
        if c["arm"] != "a0_zero_shot":
            dv = drawsets[(d, c["scheme"], str(c["budget"]), str(c["draw"]))]
            for rr, pagefield, idfield in [
                ("fit", "target_label_pages", "target_label_candidate_ids"),
                ("calibration", "calibration_pages", "calibration_candidate_ids"),
            ]:
                pages = set(dv[dv.role == rr].page_id)
                assert pages == set(v[pagefield]), model_id
                candidates = pop[
                    pop.document_id.isin(pages) & membership.map(lambda s, d=d: target_name[d] in s)
                ]
                assert set(candidates.candidate_id) == set(v[idfield]), model_id
    # Six older reference own-test crossings, plus its separate fit/cal overlap.
    historical_roles = {
        b: sorted(set(old[old.block == b].document_id)) for b in ("fit", "threshold", "test")
    }
    crossings = []
    for a, b in [("fit", "threshold"), ("fit", "test"), ("threshold", "test")]:
        for g in sorted(
            set(groups(historical_roles[a], pg)) & set(groups(historical_roles[b], pg))
        ):
            crossings.append(
                {
                    "group_id": g,
                    "left_role": a,
                    "right_role": b,
                    "left_pages": [p for p in historical_roles[a] if pg[p] == g],
                    "right_pages": [p for p in historical_roles[b] if pg[p] == g],
                    "origin": "RK1 U5 blocks / RK3 build_protocols(P_IN)",
                    "artifact": "results/generated/sgv_rk3/training_registry.json",
                    "downstream": "RK3 reference, RK2/RK4/TH1/TH2/DEP1 small-pool history",
                    "disposition": "DROP FROM FINAL PAPER; preserve disclosed invalidated history",
                }
            )
    crossing_test = sorted({r["group_id"] for r in crossings if r["right_role"] == "test"})
    assert len(crossing_test) == 6
    upstream_fit = set().union(
        *(set(v["source_label_pages"]) | set(v["target_label_pages"]) for v in native.values())
    )
    upstream_cal = set().union(*(set(v["calibration_pages"]) for v in native.values()))
    labelled_cross = sorted(set(groups(upstream_fit | upstream_cal, pg)) & set(groups(test, pg)))
    assert labelled_cross == ["alberti_pictura_1540", "estor_rechtsgelehrsamkeit02_1758"]
    assert not (upstream_fit | upstream_cal) & set(test)
    resources = js("results/generated/sgv_rl1_generator_agnostic_reliability/fitted_resources.json")
    original_resource_pages = sorted(
        {p for e in resources["by_environment"].values() for p in e["documents"]}
    )
    clean_resource_pages = sorted(
        p for p in original_resource_pages if pg[p] not in set(groups(test, pg))
    )
    assert clean_resource_pages == registry["resource_fit_proposed_pages"]
    assert len(original_resource_pages) == 80 and len(clean_resource_pages) == 75
    all_cal = sorted(
        {
            p
            for c in registry["cells"]
            for p in c["target_calibrate_pages"] + c["source_calibration_zero_pages"]
        }
    )
    resource_rows = []

    def resource(
        rid,
        path,
        producer,
        input_pages,
        klass,
        family,
        proposed=None,
        use="ranking",
        labels="NO",
        detail="",
    ):
        proposed = sorted(input_pages if proposed is None else proposed)
        row = {
            "resource_id": rid,
            "path": path,
            "path_sha256": sha(bind(path))
            if (ROOT / path).is_file()
            else "IN-MEMORY / CODE+MANIFEST BOUND",
            "producing_script": producer,
            "input_artifacts": [producer.split("::")[0], path],
            "input_page_ids": sorted(input_pages),
            "input_group_ids": groups(input_pages, pg),
            "class": klass,
            "family": family,
            "labels_involved": labels,
            "source_or_target": "fixed separate resource pool" if use == "ranking" else use,
            "test_group_crossings": sorted(set(groups(input_pages, pg)) & set(groups(test, pg))),
            "proposed_fit_pages": proposed,
            "proposed_fit_groups": groups(proposed, pg),
            "used_by_models": "all retained R3/R5 A0/M1/B2 and fit-only permutation controls"
            if use == "ranking"
            else use,
            "used_by_policies": "all retained score consumers"
            if use == "ranking"
            else "candidate/site supply"
            if use == "candidate pipeline"
            else use,
            "manuscript_results": "Methods features; Fig2; Table2; RQ1-RQ4; RC1 sensitivities"
            if use == "ranking"
            else use,
            "rebuildable": "YES",
            "candidate_regeneration_required": "NO",
            "detail": detail,
            "retained": True,
        }
        if klass != "R0":
            validate_resource(row, test, reserve, all_cal, pg)
        resource_rows.append(row)

    rp = "results/generated/sgv_rl1_generator_agnostic_reliability/fitted_resources.json"
    families = [
        "character_language_model",
        "lexicon_frequency",
        "glyph_prototypes",
        "glyph_width",
        "confidence_normalizer_quantiles",
        "geometry_imputations",
        "RK3_character_counts_evidence",
    ]
    for env, record in resources["by_environment"].items():
        assert record["role"] == "SGV14_ADAPTATION"
        pages = record["documents"]
        for family in families:
            resource(
                f"{env}::{family}",
                rp,
                (
                    "scripts/sgv_rk3_risk_aware_ranking.py::environment_evidence"
                    if family.startswith("RK3")
                    else "scripts/sgv_rl1_generator_agnostic_reliability.py::fit_environment_resources"
                ),
                pages,
                "R1",
                family,
                [p for p in pages if p in clean_resource_pages],
                detail="OCR text/confidences/pixels; no GT. Same formulas; explicit clean list, no cap backfill.",
            )
    # Reachable CORD generation resources; labelability eligibility is R2, not R1.
    candidate_meta = ids(
        "results/generated/sgv1/dev_candidates/candidates_pre_gt.parquet",
        ["candidate_id", "document_id", "site_id"],
    )
    assert not set(candidate_meta.document_id) & set(reserve)
    for family in [
        "per_engine_edit_aware",
        "structural_bigrams",
        "lexical_discovery",
        "candidate_character_LM_lexicon",
        "confidence_featurizer",
    ]:
        resource(
            "CORD_TRAIN::" + family,
            "manifests/sgv1/role_manifest.json",
            "scripts/sgv14_confirmatory_validation.py::build_pipeline",
            train,
            "R1",
            family,
            use="candidate pipeline",
            detail="444 allowed TRAIN pages; 412 atomic groups. Actual OCR availability bounded by TRAIN; no test/reserve/calibration resource input.",
        )
    resource(
        "CORD_TRAIN::glyph_eligibility",
        "manifests/sgv1/role_manifest.json",
        "scripts/sgv14_confirmatory_validation.py::build_pipeline -> cc.fit_glyph_prototypes",
        train,
        "R2",
        "glyph prototypes / OCR supervision / GT labelability crop eligibility",
        use="candidate pipeline",
        labels="YES: source TRAIN labelability only",
        detail="Exact per-crop realized nonzero pixel contributors not serialized; enumerated TRAIN input universe and deterministic lineage/crop code bound. No fabricated actual crop subset; entire allowed universe passes exclusion.",
    )
    # Static checkpoints and constants; publication pretraining membership is not locally knowable.
    for rid, family in [
        (
            "Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "fixed external VLM checkpoint",
        ),
        ("OCR_ENGINE_CHECKPOINTS", "five fixed OCR configurations per corpus"),
        ("FROZEN_PROMPTS_RULES", "external/fixed edit classes, crop/discovery rules and decoding"),
    ]:
        resource(
            rid,
            "results/generated/sgv_pf1/frozen_upstream_configuration.json",
            "scripts/sgv_pf1_page_frontier_scaling.py::run_freeze",
            [],
            "R0",
            family,
            use="candidate pipeline",
            detail="No local study-label fitting; external pretraining exposure is outside this bounded local audit.",
        )
    # Local context is inference input, not a globally fitted transformer.
    resource(
        "INFERENCE_LOCAL_CONTEXT",
        "scripts/sgv_rk3_risk_aware_ranking.py",
        "scripts/sgv_rk3_risk_aware_ranking.py::build_new_features",
        [],
        "R1",
        "within-site agreement and per-page OCR/candidate context",
        use="ranking",
        detail="Fit membership EMPTY. Current page may supply label-free inference input, including test images/OCR. Never a cross-page fitted statistic.",
    )
    rebuilds = []
    for row in resource_rows:
        affected = bool(row["test_group_crossings"])
        rebuilds.append(
            {
                "resource_id": row["resource_id"],
                "class": row["class"],
                "old_fit_pages": row["input_page_ids"],
                "proposed_fit_pages": row["proposed_fit_pages"],
                "proposed_fit_groups": row["proposed_fit_groups"],
                "test_group_crossings": row["test_group_crossings"],
                "repair": "RECOMPUTE_FEATURES -> REFIT_MODELS -> RECALIBRATE_POLICIES"
                if affected
                else "REUSE CLEAN RESOURCE",
                "rebuild_resource": "YES" if affected else "NO",
                "recompute_features": "YES" if affected else "NO",
                "refit_models": "YES" if affected else "NO",
                "recalibrate_policies": "YES" if affected else "NO",
                "regenerate_candidates": "NO",
                "status": "PLANNED; NOT EXECUTED" if affected else "MEMBERSHIP VERIFIED",
                "output_path": "results/paper3_vf2/resources/"
                + row["resource_id"].replace("/", "_").replace("::", "_")
                + ".json"
                if affected
                else row["path"],
            }
        )
    # Actual source feasibility, restricted labels ONLY; no target/test class counts.
    source_support = {}
    for d in source_name:
        c = next(c for c in registry["cells"] if c["direction"] == d)
        v = native[d + "|a0_zero_shot|random|0|0"]
        sf = old[
            old.candidate_id.isin(v["source_label_candidate_ids"])
            & old.document_id.isin(c["source_fit_pages"])
        ]
        assert set(sf.candidate_id) == set(c["source_fit_candidate_ids"])
        counts = source_counts(oldpath, c["source_fit_candidate_ids"], old)
        counts["status"] = (
            "FIT FEASIBLE"
            if min(counts["harmful_count"], counts["non_harmful_count"]) > 0
            and counts["multi_grade_queries"] > 0
            and counts["rows"] >= 40
            else "NOT FITTABLE"
        )
        source_support[d] = counts
    # Known reference: inspect native job metadata only, never metric fields.
    jobs = js("results/generated/sgv_rk3/training_registry.json")["jobs"]
    known = []
    for job in jobs:
        if job["protocol"] != "in_distribution":
            continue
        mid = "RK3|" + "|".join(job[k] for k in ("protocol", "model", "representation", "variant"))
        primary = job["model"].startswith("r_risk_lambdarank") and job["variant"] == "full"
        r = {
            "record_id": mid,
            "model_id": mid,
            "objective": job["objective"],
            "training_pages": historical_roles["fit"],
            "training_groups": groups(historical_roles["fit"], pg),
            "calibration_pages": historical_roles["threshold"],
            "calibration_groups": groups(historical_roles["threshold"], pg),
            "evaluation_pages": historical_roles["test"],
            "evaluation_groups": groups(historical_roles["test"], pg),
            "contaminated_groups": crossing_test,
            "importance": "H3" if primary else "H0",
            "manuscript_usage": "Table2 known-generator context; Results 4.2; Supplement S8"
            if primary
            else "No specific current scientific claim found for this native record",
            "rebuild_required_if_retained": "NO: fixed random"
            if job["objective"] == "random"
            else "YES",
            "rebuild_required_in_minimum_VF2": "NO",
            "retained_as_scientific_evidence": False,
            "disposition": "DROP FROM FINAL PAPER",
            "clean_reconstruction_cells": [],
            "history": "RETAIN ONLY AS DISCLOSED INVALIDATED DEVELOPMENT HISTORY",
        }
        validate_reference(r, pg)
        known.append(r)
    assert len(known) == 21 and sum(j["objective"] != "random" for j in known) == 20
    # Minimal complete current story: main Fig2 retains target-only comparison;
    # native permutation control has FIVE draws, not 20 (PF1 constants audited).
    planned_models, scope_rows = [], []
    clean_cells = {(c["direction"], c["budget"], c["draw"]): c for c in registry["cells"]}
    for old_id, v in native.items():
        c = v["score_cell"]
        retained = c["scheme"] == "random"
        classification = "MUST REBUILD — MAIN CLAIM" if retained else "OPTIONAL LEGACY / DROP"
        new_id = "VF2|" + "|".join(str(c[k]) for k in ("direction", "arm", "budget", "draw"))
        scope_rows.append(
            {
                "original_model_id": old_id,
                "proposed_model_id": new_id if retained else "",
                **c,
                "classification": classification,
                "retraining_required": "YES" if retained else "NO: DROP",
                "scientific_role": "A0/M1 primary ranking/deployment"
                if c["arm"] in ("a0_zero_shot", "m1_full_adaptation")
                else "B2 target-only main Fig2 comparator",
                "reason": "new resources/features and honest fit/calibration partition"
                if retained
                else "old stratified purchase adds a separate acquisition family; remove associated balanced-order claim",
            }
        )
        if not retained:
            continue
        if c["arm"] == "a0_zero_shot":
            cell = copy.deepcopy(
                next(x for x in registry["cells"] if x["direction"] == c["direction"])
            )
            cell.update(
                target_adapt_pages=[],
                target_calibrate_pages=[],
                target_label_pages=0,
                budget="0",
                draw=0,
            )
        else:
            cell = copy.deepcopy(clean_cells[(c["direction"], str(c["budget"]), c["draw"])])
        if c["arm"] not in ("a0_zero_shot", "m1_full_adaptation"):
            cell["source_fit_pages"] = []
            cell["source_fit_candidate_ids"] = []
        cell.update(
            model_id=new_id,
            original_model_id=old_id,
            arm=c["arm"],
            classification=classification,
            representation="R3" if c["arm"] not in ("a0_zero_shot", "m1_full_adaptation") else "R5",
            feature_registry_id="VF2_CLEAN_R3_R5",
            scores_reusable=False,
        )
        cell["calibration_page_ids"] = (
            cell["source_calibration_zero_pages"]
            if c["arm"] == "a0_zero_shot"
            else cell["target_calibrate_pages"]
        )
        for field in [
            "source_fit",
            "target_adapt",
            "target_calibrate",
            "test",
            "reserve_confirmation",
            "resource_fit",
        ]:
            cell[field + "_groups"] = groups(cell[field + "_pages"], pg)
        cell["target_adapt_candidate_ids"] = sorted(
            pop[
                pop.document_id.isin(cell["target_adapt_pages"])
                & membership.map(lambda s, d=c["direction"]: target_name[d] in s)
            ].candidate_id
        )
        cell["calibration_candidate_ids"] = (
            cell["source_calibration_zero_candidate_ids"]
            if c["arm"] == "a0_zero_shot"
            else sorted(
                pop[
                    pop.document_id.isin(cell["target_calibrate_pages"])
                    & membership.map(lambda s, d=c["direction"]: target_name[d] in s)
                ].candidate_id
            )
        )
        cell["test_candidate_ids"] = sorted(
            pop[
                pop.document_id.isin(test)
                & membership.map(lambda s, d=c["direction"]: target_name[d] in s)
            ].candidate_id
        )
        cell["source_label_page_cost"] = len(cell["source_fit_pages"])
        cell["source_calibration_page_cost"] = (
            len(cell["source_calibration_zero_pages"]) if c["arm"] == "a0_zero_shot" else 0
        )
        cell["distinct_labelled_page_ledger"] = sorted(
            set(cell["source_fit_pages"])
            | set(cell["target_adapt_pages"])
            | set(cell["calibration_page_ids"])
        )
        validate_cell(cell, pg)
        planned_models.append(cell)
    assert len(planned_models) == 482
    for d in source_name:
        for draw in range(5):
            base = next(
                c
                for c in planned_models
                if c["direction"] == d
                and c["budget"] == "100"
                and c["draw"] == draw
                and c["arm"] == "m1_full_adaptation"
            )
            cell = copy.deepcopy(base)
            cell.update(
                model_id=f"VF2|{d}|M1_TARGET_LABEL_PERMUTATION|100|{draw}",
                original_model_id=f"PF1_PERMUTATION|{d}|100|{draw}",
                arm="M1_TARGET_LABEL_PERMUTATION",
                classification="MUST REBUILD — REQUIRED SUPPLEMENT",
                permutation_scope="purchased target-adaptation grades within each page ONLY; source/calibration/test unchanged",
            )
            validate_cell(cell, pg)
            planned_models.append(cell)
            scope_rows.append(
                {
                    "original_model_id": cell["original_model_id"],
                    "proposed_model_id": cell["model_id"],
                    "direction": d,
                    "arm": cell["arm"],
                    "budget": "100",
                    "draw": draw,
                    "scheme": registry["cells"][0]["scheme"],
                    "classification": cell["classification"],
                    "retraining_required": "YES",
                    "scientific_role": "main supervision claim / Supplement S3 control",
                    "reason": "PF1 PERMUTATION_DRAWS=5; preserve existing control scope, not a new model family",
                }
            )
    for r in known:
        scope_rows.append(
            {
                "original_model_id": r["model_id"],
                "proposed_model_id": "",
                "classification": "INVALID / DROP",
                "retraining_required": "NO: DROP",
                "scientific_role": r["manuscript_usage"],
                "reason": "older own-test group crossings; unnecessary contextual evidence",
            }
        )
    policies = []
    specifications = [
        (
            "plugin",
            "MAIN REQUIRED",
            "calibration harm labels; empirical selective share; no certificate",
        ),
        (
            "conservative",
            "MAIN REQUIRED",
            "calibration page design effect + pointwise effective-binomial approximation; no certificate",
        ),
        (
            "navarro_adapted",
            "MAIN REQUIRED",
            "RC1 widths .05 primary, .025/.1 sensitivity; calibration labels; unlabelled test batch scores permitted",
        ),
        (
            "conservative_triage",
            "MAIN REQUIRED",
            "accept harm and preserve exact-repair cutoff; idealised review; fixed eta=.1",
        ),
        (
            "cost_aware_triage",
            "SUPPLEMENT REQUIRED",
            "existing plug-in accept/preserve rules; no new loss/threshold family",
        ),
        (
            "CAL2_corpus_stratified_C1",
            "SUPPLEMENT REQUIRED",
            "existing per-corpus rules on honest calibration; secondary exploratory replay",
        ),
        (
            "CAL2_hierarchical_C3",
            "SUPPLEMENT REQUIRED",
            "existing prior-strength registry only; empirical-Bayes counts derived from calibration alone",
        ),
        (
            "strict_NON_IMPROVING",
            "SUPPLEMENT REQUIRED",
            "same fixed ranker scores; recompute every retained label-dependent policy quantity",
        ),
        (
            "matched_site",
            "SUPPLEMENT REQUIRED",
            "calibrate matched availability-only winner decisions BEFORE matched evaluation; pool only",
        ),
        (
            "winner_and_all_candidate",
            "MAIN REQUIRED",
            "all budgets ranking; pool paired grouped intervals and analysis-only risk-coverage",
        ),
        (
            "test_label_boundary",
            "MAIN REQUIRED",
            "ANALYSIS ONLY: isolated test-label reference, never deployable cutoff or model selection",
        ),
        (
            "M10",
            "SUPPLEMENT REQUIRED",
            "NOT ESTIMABLE; mathematical applicability and support only, do not execute a certificate",
        ),
        (
            "random_score_control",
            "SUPPLEMENT REQUIRED",
            "fixed PF1 seeds, no learned model; same honest cal/test partition",
        ),
        (
            "calibration_label_permutation",
            "SUPPLEMENT REQUIRED",
            "diagnostic calibration-only permutation; original frozen control scope only",
        ),
        (
            "CAL2_balanced_purchase_C2",
            "DEVELOPMENT ONLY / DROP",
            "160 native stratified fits dropped; remove balanced-order effectiveness claim",
        ),
        (
            "legacy_RK2_RK4_TH1_TH2_DEP1_known_reference",
            "DEVELOPMENT ONLY / DROP",
            "preserve invalidated history; no final inferential comparison",
        ),
    ]
    for name, status, desc in specifications:
        policies.append(
            {
                "policy_or_analysis_id": name,
                "classification": status,
                "objective_and_assumptions": desc,
                "recalibration_required": "NO"
                if name in ("winner_and_all_candidate", "M10") or "DROP" in status
                else "YES"
                if name != "random_score_control"
                else "YES: new calibration for fixed random scores",
                "score_model_scope": "A0/M1 only for deployment; B2 main ranking comparator only",
                "population": "matched calibration/test, pool"
                if name == "matched_site"
                else "unchanged 66-page test; clean calibration",
                "harm_definitions": "strict-worsening primary and frozen NON_IMPROVING sensitivity"
                if name not in ("M10", "winner_and_all_candidate")
                else "frozen RC1 definitions",
                "execution_status": "DESIGN ONLY / NOT AUTHORIZED",
                "test_labels_for_deployable_selection": False,
            }
        )
    # Inventory cached model/score/policy resources separately from independent feature resources.
    derived = [
        {
            "resource_id": "PF1_LEARNED_MODEL_FAMILY",
            "path": "results/generated/sgv_pf1/adaptation_registry.json",
            "class": "R2",
            "family": "model-training resources",
            "fit_pages": sorted(upstream_fit),
            "test_group_crossings": sorted(set(groups(upstream_fit, pg)) & set(groups(test, pg))),
            "labels_involved": "YES",
            "retained": False,
            "replacement": "492 explicit proposed clean model cells",
        },
        {
            "resource_id": "PF1_SCORE_CACHE",
            "path": "results/generated/sgv_pf1/cell_scores.parquet",
            "class": "R3",
            "family": "score/model-dependent derived cache",
            "fit_pages": sorted(upstream_fit),
            "test_group_crossings": sorted(set(groups(upstream_fit, pg)) & set(groups(test, pg))),
            "labels_involved": "inherited model fitting",
            "retained": False,
            "replacement": "new scores from same per-cell clean frozen predictor for cal/test",
        },
        {
            "resource_id": "PF1_CAL_DEP_RC1_THRESHOLDS_PRIORS",
            "path": "results/generated/sgv_pf1/policy_registry.json",
            "class": "R2",
            "family": "label/score-dependent cutoffs, page deff, corpus priors and Navarro window error",
            "fit_pages": sorted(upstream_cal),
            "test_group_crossings": sorted(set(groups(upstream_cal, pg)) & set(groups(test, pg))),
            "labels_involved": "YES: calibration only; old in-fit provenance",
            "retained": False,
            "replacement": "new per-policy honest calibration; all prior outputs invalidated",
        },
        {
            "resource_id": "RK3_KNOWN_REFERENCE_MODELS",
            "path": "results/generated/sgv_rk3/training_registry.json",
            "class": "R2",
            "family": "historical reference models",
            "fit_pages": historical_roles["fit"],
            "test_group_crossings": sorted(
                set(groups(historical_roles["fit"], pg)) & set(groups(test, pg))
            ),
            "labels_involved": "YES",
            "retained": False,
            "replacement": "DROP older reference's own six crossing volumes; no final inference",
        },
        {
            "resource_id": "PF1_R3_R5_FEATURE_CACHE",
            "path": "results/generated/sgv_pf1/feature_matrix.parquet",
            "class": "R1",
            "family": "OCR/pixel features derived from contaminated ranking resources",
            "fit_pages": original_resource_pages,
            "test_group_crossings": ["glauber_opera01_1658"],
            "labels_involved": "NO",
            "retained": False,
            "replacement": "regenerate all candidate features with clean resources; test pages inference inputs only",
        },
    ]
    for row in derived:
        path = ROOT / row["path"]
        row.update(
            input_page_ids=row.pop("fit_pages"),
            producing_script="native PF1/RK3/CAL2/DEP1/RC1 frozen stage code",
            used_by_models="old 642 native cells / earlier reference",
            used_by_policies="all old score consumers",
            rebuildable="YES; rebuild retained path or DROP",
            candidate_regeneration_required="NO",
        )
        row["input_group_ids"] = groups(row["input_page_ids"], pg)
        row["path_sha256"] = (
            sha(bind(path))
            if path.exists()
            else "NO SERIALIZED CHECKPOINT; native cell/code provenance"
        )
        resource_rows.append(row)
    # Record all inspected source/config files and full current manuscript input hashes.
    source_paths = [
        "scripts/sgv_rl1_generator_agnostic_reliability.py",
        "scripts/sgv_rk3_risk_aware_ranking.py",
        "scripts/sgv_rk4_generator_adaptive_ranking.py",
        "scripts/sgv_rk2_fewshot_ranker_adaptation.py",
        "scripts/sgv_pf1_page_frontier_scaling.py",
        "scripts/sgv14_confirmatory_validation.py",
        "scripts/sgv1_candidate_conditioned.py",
        "scripts/sgv_lp1_image_error_site_proposal.py",
        "scripts/sgv_cg1_opportunity_ceiling.py",
        "scripts/sgv_xc1_cross_correction_opportunity.py",
        "scripts/sgv_hy1_hybrid_site_discovery.py",
        "src/ocr_risk/evidence/features_conf.py",
        "src/ocr_risk/experiments/cgv3_confirmatory.py",
        "scripts/sgv_cal2_domain_stratified_calibration.py",
        "scripts/paper3_rc1_reviewer_closure.py",
        "docs/paper3/journal_track_2027/reviewer_closure/P3_RC1_CONFIG.json",
    ]
    for p in source_paths:
        if (ROOT / p).exists():
            bind(p)
    for p in (ROOT / "paper3_icdar_ijdar_2027").rglob("*.tex"):
        bind(p)
    git = {
        "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=ROOT)
        ).hexdigest(),
    }
    summary = {
        "test_pages": test,
        "test_groups": groups(test, pg),
        "original_label_crossings": labelled_cross,
        "original_resource_test_crossings": sorted(
            set(groups(original_resource_pages, pg)) & set(groups(test, pg))
        ),
        "older_reference_test_crossings": crossing_test,
        "native_models_independently_reconciled": len(native),
        "source_fit_support": source_support,
        "original_resource_pages": original_resource_pages,
        "clean_resource_pages": clean_resource_pages,
        "clean_resource_groups": groups(clean_resource_pages, pg),
        "resource_inventory_rows": len(resource_rows),
        "contaminated_independent_resource_components": sum(
            bool(r["test_group_crossings"]) for r in rebuilds
        ),
        "primary_models": 242,
        "target_only_comparator_models": 240,
        "headline_models_total": 482,
        "supplement_permutation_models": 10,
        "total_rebuild_models": len(planned_models),
        "legacy_PF1_models_dropped": 160,
        "known_reference_learned_records_dropped": 20,
        "known_reference_fixed_random_records_dropped": 1,
        "reserve_pages": 177,
        "reserve_groups": 164,
        "reserve_status": "UNTOUCHED BUT INSUFFICIENT",
        "resource_reconstruction_executed": False,
        "performance_computed": False,
    }
    final = {
        "study": "P3-VF1B",
        "status": "PROPOSED CLEAN DESIGN; NOT EXECUTED",
        "execution_authorized": False,
        "VF2_execution_freeze_created": False,
        "git_head": git["HEAD"],
        "seed": registry["seed"],
        "analysis_status": "PROSPECTIVELY FROZEN CORRECTIVE RE-ANALYSIS (future execution only)",
        "page_to_group": pg,
        "page_to_corpus": corpus,
        "cells": planned_models,
        "resource_inventory": resource_rows,
        "resource_rebuilds": rebuilds,
        "policies": policies,
        "historical_references": known,
        "support": registry["support"],
        "summary": summary,
        "budget_labels": registry["budget_labels"],
        "draws": registry["draws"],
        "budget_semantics": "B=distinct target-generator adaptation+calibration pages; historical source labels separately costed; not B new manual annotations",
        "zero_budget_semantics": "zero added target-generator candidate labels; NOT zero labelled pages/evidence overall",
        "reserve_confirmation_enabled": False,
        "reserve_outcomes_inspected": False,
        "candidate_generation_reused": True,
        "candidate_regeneration_required": False,
        "old_scores_features_thresholds_reusable": False,
        "formal_M10": "NOT ESTIMABLE; no empirical certificate or replacement method",
        "readiness_decision": "CLEAN DESIGN VERIFIED — HISTORICAL REFERENCES MUST BE DROPPED",
    }
    OUT.mkdir()
    write_json(OUT / "clean_execution_registry.json", final)
    bind("results/paper3_vf1/proposed_clean_split_groups.csv")
    group_table = pd.read_csv(ROOT / "results/paper3_vf1/proposed_clean_split_groups.csv")
    group_table["VF1B_resource_fit_clean"] = group_table.page_id.isin(clean_resource_pages)
    group_table["historical_reference_retained"] = False
    group_table.to_csv(OUT / "clean_group_registry.csv", index=False)
    write_csv(OUT / "resource_rebuild_registry.csv", rebuilds)
    write_csv(OUT / "model_rebuild_registry.csv", scope_rows)
    write_csv(OUT / "policy_rebuild_registry.csv", policies)
    write_csv(DOC / "P3_VF1B_RESOURCE_INVENTORY.csv", resource_rows)
    write_csv(DOC / "P3_VF1B_RESOURCE_REBUILD_MATRIX.csv", rebuilds)
    write_csv(DOC / "P3_VF1B_REFERENCE_CROSSINGS.csv", crossings)
    write_csv(DOC / "P3_VF1B_KNOWN_REFERENCE_REBUILD.csv", known)
    write_csv(DOC / "P3_VF1B_MINIMUM_MODEL_REBUILD.csv", scope_rows)
    write_json(OUT / "audit_summary.json", summary)
    for p, digest in protected.items():
        assert sha(ROOT / p) == digest, p
    write_json(
        OUT / "input_manifest.json",
        {
            "study": "P3-VF1B",
            "git": git,
            "inputs_sha256": INPUTS,
            "protected_sha256": protected,
            "all_column_requests": READS,
            "handoff_files_inspected": handoff,
            "synthetic": False,
            "reserve_outcomes_inspected": False,
            "performance_computed": False,
            "new_models_trained": False,
        },
    )
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()

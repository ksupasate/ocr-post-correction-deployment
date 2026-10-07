"""Hand-computed tests for the CGV3 confirmatory gates.

Every criterion is exercised at its threshold boundary: a value exactly on the floor
passes, one below fails, and a missing evidence file forces INCONCLUSIVE (fail
closed). The verdict aggregation mirrors cgv2_gates and is red-teamed: a failed
integrity review must floor the generator verdict regardless of the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ocr_risk.experiments.cgv2_gates import (
    _H2_ACCEPTED_EDITS_FLOOR,
    _H2_SAFE_COVERAGE_FLOOR,
)
from ocr_risk.experiments.cgv3_gates import (
    evaluate_cgv3_generator_gate,
    evaluate_cgv3_h2_readiness,
    evaluate_cgv3_integrity,
)

__all__ = [
    "TestGeneratorGate",
    "TestH2Readiness",
    "TestIntegrityGate",
    "_frames_dir",
    "_freeze",
]

ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")

_APPLICABLE_AUDIT_CHECKS = {
    "component_denominators_reconcile": True,
    "site_denominators_reconcile": True,
    "region_denominators_reconcile": True,
}
# Mirrors cmd_analyze: the CGV2 proposals-projection checks exist in the gate file
# but are structurally False without a proposals layer -- DEFECT-1 made F evaluate
# only the applicable ones, so the fixture carries both to prove the split.
_NOT_APPLICABLE_AUDIT_CHECKS = {
    "site_projection_rows_accounted": False,
    "region_projection_rows_accounted": False,
    "eligible_site_projections_valid": False,
    "eligible_region_projections_valid": False,
}


def _write_audit_gate(
    source: Path,
    *,
    checks: dict[str, bool] | None = None,
    with_lineage: bool = True,
) -> None:
    audits = source / "audits" / "evaluate"
    audits.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": "cgv2-alignment-audit-v2",
        # overall_pass is False whenever a not-applicable check is present -- the
        # exact production condition DEFECT-1 documented.
        "overall_pass": False,
        "checks": {
            **_NOT_APPLICABLE_AUDIT_CHECKS,
            **_APPLICABLE_AUDIT_CHECKS,
            **(checks or {}),
        },
    }
    if with_lineage:
        payload["input_runs"] = {
            "align": "run-align",
            "sites": "run-sites",
            "canonicalize": "run-canonicalize",
            "manifest": "run-manifest",
        }
    (audits / "alignment_audit_gate.json").write_text(json.dumps(payload))


def _freeze() -> dict[str, object]:
    return {
        "sha256": "0" * 64,
        "site_recall_floors": {
            "s1_overall": 0.21,
            "s2_by_class": {
                "deletion": 0.13,
                "insertion": 0.16,
                "segmentation": 0.22,
                "substitution": 0.2,
            },
        },
        "statistics": {"n_resamples": 10000, "seed": 7},
    }


def _frames_dir(tmp_path: Path, *, with_contrasts: bool = True) -> Path:
    source = tmp_path / "confirmatory"
    source.mkdir(parents=True, exist_ok=True)
    recall_by_engine = {"doctr": 0.30, "easyocr": 0.50, "paddleocr": 0.25, "tesseract": 0.45}
    site_metrics = pd.DataFrame(
        [
            {
                "engine_id": engine,
                "site_recall": recall_by_engine[engine],
                "n_true_eligible": 100,
                "n_discovered": int(100 * recall_by_engine[engine]),
            }
            for engine in ENGINES
        ]
    )
    site_metrics.to_csv(source / "cgv3_site_metrics.csv", index=False)

    # 25 document clusters (H2 C counts >= 20 on the site census); 100 eligible
    # regions per (engine, class), 30 discovered -> class recall 0.30; discovered
    # rows name their matched site so CG3's insertion clause can find candidates.
    class_rows = []
    for engine in ENGINES:
        for kind in ("deletion", "insertion", "segmentation", "substitution"):
            for i in range(100):
                class_rows.append(
                    {
                        "true_site_id": f"r-{engine}-{kind}-{i}",
                        "document_id": f"d{i // 4}",
                        "dataset_id": "funsd",
                        "engine_id": engine,
                        "site_kind": kind,
                        "d_before": 1,
                        "audit_state": "eligible",
                        "disposition": "discovered" if i < 30 else "not_discovered",
                        "matched_site_id": f"{engine}-{i}" if i < 30 else "",
                    }
                )
    pd.DataFrame(class_rows).to_csv(source / "cgv3_true_coverage.csv", index=False)

    oracle = pd.DataFrame(
        [
            {
                "engine_id": engine,
                "generator_id": "g8_union",
                "n_eligible_regions": 1000,
                "oracle_accepted_edits": 300,
                "oracle_safe_coverage": 0.30,
            }
            for engine in ENGINES
        ]
    )
    oracle.to_csv(source / "cgv3_oracle.csv", index=False)

    # 1200 g8 rows per engine, all beneficial: 100 at distinct clean sites
    # (region ids < 400 are clean), 1100 at non-clean sites. Clean share 0.40,
    # distinct clean sites proposed 100 (the H2 D floor exactly), proposal rate
    # 100/400 = 0.25 <= 0.30 (the CG2 bound).
    candidate_rows = []
    for engine in ENGINES:
        for i in range(100):
            candidate_rows.append(
                {
                    "site_id": f"{engine}-{i}",
                    "document_id": f"d{i // 4}",
                    "engine_id": engine,
                    "generator_id": "g8_union",
                    "generator_rank": 0,
                    "outcome": "true_correction",
                }
            )
        for j in range(1100):
            site = 400 + (j % 600)
            candidate_rows.append(
                {
                    "site_id": f"{engine}-{site}",
                    "document_id": f"d{site // 40}",
                    "engine_id": engine,
                    "generator_id": "g8_union",
                    "generator_rank": j % 4,
                    "outcome": "true_correction",
                }
            )
    pd.DataFrame(candidate_rows).to_csv(source / "cgv3_candidates.csv", index=False)

    harm = pd.DataFrame(
        [
            {
                "engine_id": engine,
                "generator_id": rung,
                "harmful_beneficial_ratio": 1.0,
            }
            for engine in ENGINES
            for rung in ("g8_union", "g3_edit_aware")
        ]
    )
    harm.to_csv(source / "cgv3_harm.csv", index=False)

    # 1000 discovered sites per engine over 25 documents, the first 400 clean.
    region_truth = pd.DataFrame(
        [
            {
                "site_id": f"{engine}-{i}",
                "document_id": f"d{i // 40}",
                "engine_id": engine,
                "anchor_kind": "token",
                "region_ocr": "x",
                "region_gt": "x" if i < 400 else "y",
                "d_before": 0 if i < 400 else 1,
                "index_present": True,
            }
            for engine in ENGINES
            for i in range(1000)
        ]
    )
    region_truth.to_csv(source / "cgv3_discovered_region_truth.csv", index=False)

    conditional = pd.DataFrame(
        [
            {
                "engine_id": engine,
                "generator_id": "g8_union",
                "n_covered_regions": 100,
                "n_regions_with_exact": 30,
                "n_regions_with_beneficial": 40,
                "exact_recall_given_discovery": 0.3,
                "beneficial_recall_given_discovery": 0.4,
            }
            for engine in ENGINES
        ]
    )
    conditional.to_csv(source / "cgv3_conditional.csv", index=False)
    end_to_end = conditional.assign(
        n_eligible_regions=100,
        n_discovered=100,
        n_end_to_end_beneficial=40,
        end_to_end_beneficial_opportunity=0.4,
    )
    end_to_end.to_csv(source / "cgv3_end_to_end.csv", index=False)

    if with_contrasts:
        stats = source / "statistics"
        stats.mkdir(exist_ok=True)
        contrasts = pd.DataFrame(
            [
                {
                    "contrast": contrast,
                    "engine_id": engine,
                    "delta": 0.05,
                    "ci_lower": 0.01,
                    "ci_upper": 0.09,
                    "p_value": 0.001,
                    "p_holm": 0.004,
                    "reject_holm": True,
                    "standard_error": 0.02,
                    "degenerate_interval": False,
                }
                for contrast in (
                    "g8_union_vs_g3_edit_aware__end_to_end",
                    "g8_union_vs_g3_edit_aware__structural_strata",
                )
                for engine in ENGINES
            ]
        )
        contrasts.to_csv(stats / "cgv3_contrasts.csv", index=False)
        (stats / "statistics_record.json").write_text(
            json.dumps(
                {
                    "n_resamples": 10000,
                    "seed": 7,
                    "unit": "document",
                    "multiplicity": "holm_within_four_engine_family",
                }
            )
        )
    return source


class TestGeneratorGate:
    def test_all_criteria_met_yields_ready(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        gate = evaluate_cgv3_generator_gate(source, _freeze(), integrity_pass=True)
        by_key = {c["key"]: c for c in gate["criteria"]}
        assert by_key["S1_overall_site_recall"]["met"] is True
        assert by_key["CG1_availability_contrast"]["met"] is True
        assert by_key["CG4_h2_floor"]["met"] is True
        assert gate["verdict"] == "GENERATOR READY"

    def test_s1_below_floor_on_one_engine_fails_the_criterion(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        frame = pd.read_csv(source / "cgv3_site_metrics.csv")
        frame.loc[frame.engine_id == "paddleocr", "site_recall"] = 0.20  # floor is 0.21
        frame.to_csv(source / "cgv3_site_metrics.csv", index=False)
        gate = evaluate_cgv3_generator_gate(source, _freeze(), integrity_pass=True)
        by_key = {c["key"]: c for c in gate["criteria"]}
        assert by_key["S1_overall_site_recall"]["met"] is False
        assert gate["verdict"] != "GENERATOR READY"

    def test_s2_boundary_value_on_the_floor_passes(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        coverage = pd.read_csv(source / "cgv3_true_coverage.csv", keep_default_na=False)
        # Set segmentation recall to exactly 0.22 on every engine: on-floor passes.
        for engine in ENGINES:
            mask = (coverage.engine_id == engine) & (coverage.site_kind == "segmentation")
            coverage.loc[mask, "disposition"] = ["discovered"] * 22 + ["not_discovered"] * 78
        coverage.to_csv(source / "cgv3_true_coverage.csv", index=False)
        gate = evaluate_cgv3_generator_gate(source, _freeze(), integrity_pass=True)
        by_key = {c["key"]: c for c in gate["criteria"]}
        assert by_key["S2_structural_site_recall"]["met"] is True

    def test_missing_contrast_evidence_is_inconclusive(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path, with_contrasts=False)
        gate = evaluate_cgv3_generator_gate(source, _freeze(), integrity_pass=True)
        assert gate["verdict"] == "INCONCLUSIVE"

    def test_failed_integrity_floors_the_verdict_to_not_ready(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        gate = evaluate_cgv3_generator_gate(source, _freeze(), integrity_pass=False)
        assert gate["verdict"] == "GENERATOR NOT READY"


class TestH2Readiness:
    def test_all_abcdefg_met_yields_ready(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        _write_audit_gate(source)
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        assert gate["verdict"] == "H2 READY"

    def test_criterion_a_boundary_245_passes_and_244_fails(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        _write_audit_gate(source)
        oracle = pd.read_csv(source / "cgv3_oracle.csv")
        oracle["oracle_accepted_edits"] = _H2_ACCEPTED_EDITS_FLOOR
        oracle.to_csv(source / "cgv3_oracle.csv", index=False)
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["A_repair_opportunity"]["met"] is True

        oracle["oracle_accepted_edits"] = _H2_ACCEPTED_EDITS_FLOOR - 1
        oracle.to_csv(source / "cgv3_oracle.csv", index=False)
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["A_repair_opportunity"]["met"] is False

    def test_criterion_b_boundary(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        _write_audit_gate(source)
        oracle = pd.read_csv(source / "cgv3_oracle.csv")
        oracle["oracle_safe_coverage"] = _H2_SAFE_COVERAGE_FLOOR
        oracle.to_csv(source / "cgv3_oracle.csv", index=False)
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["B_oracle_safe_coverage"]["met"] is True

    def test_missing_audit_gate_is_inconclusive(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        assert gate["verdict"] == "H2 INCONCLUSIVE"

    def test_f_ignores_not_applicable_projection_checks(self, tmp_path: Path) -> None:
        """DEFECT-1 red team: overall_pass is False, only applicable checks count."""
        source = _frames_dir(tmp_path)
        _write_audit_gate(source)  # overall_pass False; projection checks False
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["F_no_unresolved_confound"]["met"] is True

    def test_f_fails_when_a_reconciliation_check_fails(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        _write_audit_gate(source, checks={"site_denominators_reconcile": False})
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["F_no_unresolved_confound"]["met"] is False
        assert gate["verdict"] == "H2 NOT READY"

    def test_f_fails_without_recorded_lineage(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        _write_audit_gate(source, with_lineage=False)
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["F_no_unresolved_confound"]["met"] is False
        assert gate["verdict"] == "H2 NOT READY"

    def test_missing_region_truth_is_inconclusive_not_ready(self, tmp_path: Path) -> None:
        """DEFECT-4 red team: absent region truth must not skip criterion D.

        With A/B/C/F/G all met and D silently unevaluated, the pre-fix gate
        returned H2 READY; it must return H2 INCONCLUSIVE instead.
        """
        source = _frames_dir(tmp_path)
        _write_audit_gate(source)
        (source / "cgv3_discovered_region_truth.csv").unlink()
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["D_overcorrection_observable"]["met"] is None
        assert criterion["E_natural_pool"]["met"] is True
        assert gate["verdict"] == "H2 INCONCLUSIVE"

    def test_missing_candidates_makes_d_and_e_inconclusive(self, tmp_path: Path) -> None:
        source = _frames_dir(tmp_path)
        _write_audit_gate(source)
        (source / "cgv3_candidates.csv").unlink()
        gate = evaluate_cgv3_h2_readiness(source, integrity_pass=True)
        criterion = {c["key"]: c for c in gate["criteria"]}
        assert criterion["D_overcorrection_observable"]["met"] is None
        assert criterion["E_natural_pool"]["met"] is None
        assert gate["verdict"] == "H2 INCONCLUSIVE"


class TestIntegrityGate:
    def _source(self, tmp_path: Path) -> Path:
        source = _frames_dir(tmp_path)
        import hashlib

        snapshot_path = source / "pre_access_snapshot.json"
        snapshot_path.write_text(json.dumps({"all_previously_untouched": True}))
        # DEFECT-5: the unlock record binds the snapshot it unlocked by hash, and the
        # integrity check enforces the match rather than merely reporting it.
        (source / "unlock_record.json").write_text(
            json.dumps(
                {
                    "unlocked": True,
                    "pre_access_snapshot_sha256": hashlib.sha256(
                        snapshot_path.read_bytes()
                    ).hexdigest(),
                }
            )
        )
        (source / "ocr_census.json").write_text(
            json.dumps(
                {
                    "expected_pairs": 396,
                    "present_pairs": 396,
                    "fingerprint_parity": True,
                }
            )
        )
        sites_path = source / "cgv3_fresh_sites.csv"
        sites_path.write_text("site_id\ns0\n")
        candidates_path = source / "cgv3_fresh_candidates_pre_gt.csv"
        candidates_path.write_text("site_id\ns0\n")
        (source / "execution_complete_manifest.json").write_text(
            json.dumps(
                {
                    "sites_sha256": hashlib.sha256(sites_path.read_bytes()).hexdigest(),
                    "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
                }
            )
        )
        (source / "r65_certificate.json").write_text(json.dumps({"status": "CLOSED"}))
        return source

    def _expected(self, source: Path) -> dict[str, object]:
        return {
            "sealed_sources_unchanged": True,
            "fold_corpus_reserve_intersection": [],
            "generator_ids": ["g8_union"],
            "analysis_roles": ["confirmatory"],
            "audit_lineage_mismatches": [],
            "unresolved_critical_findings": [],
        }

    def test_all_checks_pass(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        gate = evaluate_cgv3_integrity(source, self._expected(source))
        assert gate["verdict"] == "PASS"

    def test_missing_lineage_binding_fails_closed(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        expected = self._expected(source)
        del expected["audit_lineage_mismatches"]
        gate = evaluate_cgv3_integrity(source, expected)  # type: ignore[arg-type]
        assert gate["verdict"] == "FAIL"
        assert gate["checks"]["audit_lineage_matches_live_store"]["pass"] is False

    def test_stale_audit_run_fails_the_lineage_binding(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        expected = self._expected(source)
        expected["audit_lineage_mismatches"] = ["align: audit=run-old live=run-new"]
        gate = evaluate_cgv3_integrity(source, expected)  # type: ignore[arg-type]
        assert gate["verdict"] == "FAIL"
        assert gate["checks"]["audit_lineage_matches_live_store"]["pass"] is False

    def test_moved_candidate_table_fails_the_manifest_binding(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        (source / "cgv3_fresh_candidates_pre_gt.csv").write_text("site_id\ns0\ns1\n")
        gate = evaluate_cgv3_integrity(source, self._expected(source))
        assert gate["verdict"] == "FAIL"
        assert gate["checks"]["execution_manifest_binds_tables"]["pass"] is False

    def test_unresolved_critical_finding_fails(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        expected = self._expected(source)
        expected["unresolved_critical_findings"] = ["R-100"]
        gate = evaluate_cgv3_integrity(source, expected)  # type: ignore[arg-type]
        assert gate["verdict"] == "FAIL"

    def test_tampered_snapshot_fails_the_unlock_binding(self, tmp_path: Path) -> None:
        """DEFECT-5 red team: a snapshot hash mismatch must fail, not just report."""
        source = self._source(tmp_path)
        (source / "unlock_record.json").write_text(
            json.dumps({"unlocked": True, "pre_access_snapshot_sha256": "0" * 64})
        )
        gate = evaluate_cgv3_integrity(source, self._expected(source))
        assert gate["checks"]["unlock_record_present"]["pass"] is False
        assert gate["verdict"] == "FAIL"

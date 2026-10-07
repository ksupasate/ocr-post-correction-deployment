"""Gate-logic tests for CGV2 §16, on fabricated canonical artifacts.

The gates must be decidable from the CSVs alone, INCONCLUSIVE when evidence is missing,
and never silently unmet. Every threshold asserted here is hand-derived from the frozen
protocol text, not from a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ocr_risk.experiments.cgv2_gates import evaluate_cgv2_gates
from ocr_risk.io.hashing import file_sha256


def _contrast_rows(
    metric: str,
    deltas: dict[str, tuple[float, float, float, float]],
) -> list[dict[str, object]]:
    """(delta, ci_lower, ci_upper, p_holm) per engine; reject follows the CI sign."""
    return [
        {
            "engine_id": engine_id,
            "challenger": "g6_union",
            "baseline": "g3_edit_aware",
            "metric": metric,
            "stratum": "all_error_sites",
            "challenger_value": 0.2,
            "baseline_value": 0.1,
            "delta": values[0],
            "ci_lower": values[1],
            "ci_upper": values[2],
            "p_value": 0.01,
            "p_holm": values[3],
            "reject_holm": values[1] > 0,
            "n_documents": 20,
            "n_sites": 100,
        }
        for engine_id, values in deltas.items()
    ]


def _write_artifacts(
    tmp_path: object,
    *,
    deltas: dict[str, tuple[float, float, float, float]],
    structural_deltas: dict[str, tuple[float, float, float, float]] | None = None,
    accepted_edits: dict[str, int] | None = None,
    insertion_availability: dict[str, float] | None = None,
    clean_rate: float = 0.2,
    integrity_pass: bool | None = True,
) -> None:
    structural_deltas = structural_deltas or deltas
    accepted_edits = accepted_edits or {"e1": 300, "e2": 300, "e3": 300, "e4": 300}
    insertion_availability = insertion_availability or {
        "e1": 0.1,
        "e2": 0.1,
        "e3": 0.1,
        "e4": 0.1,
    }
    engines = list(accepted_edits)
    root = Path(tmp_path)

    contrasts = [
        *_contrast_rows("site_availability", deltas),
        *_contrast_rows("structural_stratum_availability", structural_deltas),
    ]
    pd.DataFrame(contrasts).to_csv(root / "cgv2_contrasts__evaluate.csv", index=False)

    quality_rows = []
    preservation_rows = []
    oracle_rows = []
    for engine_id in engines:
        for rung in ("g3_edit_aware", "g6_union"):
            quality_rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "n_candidates": 1500,
                    "n_beneficial": 300,
                    "n_harmful": 900,
                    "n_neutral": 300,
                    "beneficial_rate": 0.2,
                    "harmful_rate": 0.6,
                    "harmful_beneficial_ratio": 3.0,
                    "candidates_per_site": 1.5,
                }
            )
            preservation_rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "n_clean_sites": 400,
                    "n_clean_sites_proposed": 200,
                    "clean_span_proposal_rate": clean_rate if rung == "g6_union" else 0.15,
                }
            )
        oracle_rows.append(
            {
                "generator_id": "g6_union",
                "engine_id": engine_id,
                "variant": "site_plus_regions",
                "n_evaluable_sites": 5000,
                "oracle_accepted_edits": accepted_edits[engine_id],
                "oracle_safe_coverage": accepted_edits[engine_id] / 5000,
            }
        )
        oracle_rows.append(
            {
                "generator_id": "g6_union",
                "engine_id": engine_id,
                "variant": "site_only",
                "n_evaluable_sites": 5000,
                "oracle_accepted_edits": accepted_edits[engine_id],
                "oracle_safe_coverage": accepted_edits[engine_id] / 5000,
            }
        )
    pd.DataFrame(quality_rows).to_csv(root / "cgv2_quality__evaluate.csv", index=False)
    pd.DataFrame(preservation_rows).to_csv(root / "cgv2_preservation__evaluate.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(root / "cgv2_oracle__evaluate.csv", index=False)

    opportunity_rows = [
        {
            "generator_id": "g6_union",
            "engine_id": engine_id,
            "stratum": "deletion",
            "k": 4,
            "n_error_sites": 1000,
            "n_available": int(1000 * insertion_availability[engine_id]),
            "availability": insertion_availability[engine_id],
        }
        for engine_id in engines
    ]
    pd.DataFrame(opportunity_rows).to_csv(root / "cgv2_opportunity__evaluate.csv", index=False)

    census_rows = [
        {
            "site_id": f"{engine_id}-site-{i}",
            "document_id": f"{engine_id}-doc-{i // 100}",
            "dataset_id": "funsd",
            "engine_id": engine_id,
            "site_kind": "substitution",
            "d_before": 0 if i % 2 else 1,
            "gt_text": "x",
            "ocr_text": "y",
        }
        for engine_id in engines
        for i in range(1000)
    ]
    pd.DataFrame(census_rows).to_csv(root / "cgv2_site_census__evaluate.csv", index=False)
    # 1,499 site candidates plus one region candidate per engine: exactly 1,500 pool
    # candidates. This catches the historical defect where the gate added the region row
    # to a quality count that already included it.
    site_proposals = pd.DataFrame(
        [
            {
                "candidate_id": f"{engine_id}-cand-{i}",
                "generator_id": "g6_union",
                "generator_rank": 0,
                "generator_native_cap": 8,
                "site_id": f"{engine_id}-site-{i % 1000}",
                "document_id": f"{engine_id}-doc-{(i % 1000) // 100}",
                "dataset_id": "funsd",
                "engine_id": engine_id,
                "candidate_text": f"word-{i}",
                "pool": "natural",
            }
            for engine_id in engines
            for i in range(1499)
        ]
    )
    site_proposals.to_csv(root / "cgv2_proposals__evaluate.csv", index=False)
    region_proposals = pd.DataFrame(
        [
            {
                "candidate_id": f"{engine_id}-region-cand-0",
                "generator_rank": 0,
                "generator_native_cap": 2,
                "region_id": f"{engine_id}:r:0",
                "left_site_id": f"{engine_id}-site-0",
                "right_site_id": f"{engine_id}-site-1",
                "dataset_id": "funsd",
                "engine_id": engine_id,
                "candidate_text": "word",
                "pool": "natural",
            }
            for engine_id in engines
        ]
    )
    region_proposals.to_csv(root / "cgv2_region_proposals__evaluate.csv", index=False)

    # The role-scoped alignment audit: every site eligible, every candidate projection
    # valid, so the audit overlay keeps the full pool and the gates judge the numbers.
    audit_dir = root / "audits" / "evaluate"
    audit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"site_id": f"{engine_id}-site-{i}", "state": "eligible"}
            for engine_id in engines
            for i in range(1000)
        ]
    ).to_csv(audit_dir / "alignment_site_audit.csv", index=False)
    pd.DataFrame([{"engine_id": engine_id} for engine_id in engines]).to_csv(
        audit_dir / "alignment_site_denominators.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate_id": row["candidate_id"],
                "site_id": row["site_id"],
                "state": "eligible",
                "projection_correct": True,
                "label_valid": True,
            }
            for row in site_proposals.to_dict(orient="records")
        ]
    ).to_csv(audit_dir / "site_projection_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": f"{engine_id}-region-cand-0",
                "state": "eligible",
                "projection_correct": True,
                "label_valid": True,
            }
            for engine_id in engines
        ]
    ).to_csv(audit_dir / "region_projection_audit.csv", index=False)
    for name in (
        "alignment_component_audit.csv",
        "alignment_component_denominators.csv",
        "alignment_twin_sites.csv",
        "region_pair_audit.csv",
    ):
        (audit_dir / name).write_text("engine_id\n")

    # The no-twin sensitivity family with a hash-bound manifest, as criterion F demands.
    sensitivity_names = []
    for source_name, sensitivity_name in (
        ("cgv2_contrasts__evaluate.csv", "cgv2_contrasts__evaluate__no_twin.csv"),
        ("cgv2_opportunity__evaluate.csv", "cgv2_opportunity__evaluate__no_twin.csv"),
        ("cgv2_quality__evaluate.csv", "cgv2_quality__evaluate__no_twin.csv"),
        ("cgv2_oracle__evaluate.csv", "cgv2_oracle__evaluate__no_twin.csv"),
    ):
        (root / sensitivity_name).write_bytes((root / source_name).read_bytes())
        sensitivity_names.append(sensitivity_name)
    manifest = {
        "inputs": {
            path.name: file_sha256(path)
            for path in (
                audit_dir / "alignment_twin_sites.csv",
                audit_dir / "alignment_site_audit.csv",
                audit_dir / "region_pair_audit.csv",
                audit_dir / "site_projection_audit.csv",
                audit_dir / "region_projection_audit.csv",
            )
        },
        "outputs": {
            name: {"rows": 1, "sha256": file_sha256(root / name)} for name in sensitivity_names
        },
    }
    (root / "cgv2_analysis_manifest__evaluate__no_twin.json").write_text(
        json.dumps(manifest) + "\n"
    )

    audit_outputs = {
        path.name: {"rows": 1, "sha256": file_sha256(path)}
        for path in sorted(audit_dir.glob("*.csv"))
    }
    (audit_dir / "alignment_audit_gate.json").write_text(
        json.dumps(
            {
                "overall_pass": True,
                "inputs": {
                    "site_proposals_sha256": file_sha256(root / "cgv2_proposals__evaluate.csv"),
                    "region_proposals_sha256": file_sha256(
                        root / "cgv2_region_proposals__evaluate.csv"
                    ),
                },
                "outputs": audit_outputs,
            }
        )
        + "\n"
    )

    if integrity_pass is not None:
        (root / "cgv2_integrity_gate.json").write_text(
            json.dumps(
                {
                    "overall_pass": integrity_pass,
                    "blocking_findings": []
                    if integrity_pass
                    else [
                        {
                            "id": "R-TEST",
                            "severity": "Critical",
                            "summary": "fixture blocker",
                            "evidence": [],
                        }
                    ],
                    "confirmatory_status": "confirmatory" if integrity_pass else "exploratory_only",
                }
            )
            + "\n"
        )


_PASSING = {
    "e1": (0.05, 0.02, 0.08, 0.01),
    "e2": (0.05, 0.02, 0.08, 0.01),
    "e3": (0.05, 0.02, 0.08, 0.01),
    "e4": (0.05, 0.02, 0.08, 0.01),
}


def test_all_criteria_met_is_generator_ready(tmp_path: object) -> None:
    _write_artifacts(
        tmp_path, deltas=_PASSING, accepted_edits={"e1": 300, "e2": 300, "e3": 300, "e4": 300}
    )
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    assert result["verdict"] == "GENERATOR READY"
    assert result["h2_readiness"]["verdict"] == "H2 READY"
    criteria = {c["key"]: c["met"] for c in result["criteria"]}
    assert all(criteria.values())
    # CG1-CG3, the integrity review, and CG4.
    assert len(result["criteria"]) == 5
    assert len(result["h2_readiness"]["criteria"]) == 7


def test_h2_floor_failing_downgrades_to_partially_ready(tmp_path: object) -> None:
    _write_artifacts(
        tmp_path,
        deltas=_PASSING,
        accepted_edits={"e1": 300, "e2": 300, "e3": 100, "e4": 50},
    )
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    assert result["verdict"] == "GENERATOR PARTIALLY READY"
    assert result["h2_readiness"]["verdict"] == "H2 NOT READY"


def test_no_criterion_met_is_not_ready(tmp_path: object) -> None:
    failing = {
        "e1": (-0.02, -0.05, 0.01, 0.5),
        "e2": (-0.02, -0.05, 0.01, 0.5),
        "e3": (-0.02, -0.05, 0.01, 0.5),
        "e4": (-0.02, -0.05, 0.01, 0.5),
    }
    _write_artifacts(
        tmp_path,
        deltas=failing,
        structural_deltas=failing,
        insertion_availability={"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 0.0},
        clean_rate=0.5,
    )
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    assert result["verdict"] == "GENERATOR NOT READY"


def test_missing_artifacts_are_inconclusive(tmp_path: object) -> None:
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["h2_readiness"]["verdict"] == "H2 INCONCLUSIVE"


def test_structural_gain_required_not_just_primary(tmp_path: object) -> None:
    flat_structural = {
        "e1": (0.0, -0.01, 0.01, 0.9),
        "e2": (0.0, -0.01, 0.01, 0.9),
        "e3": (0.0, -0.01, 0.01, 0.9),
        "e4": (0.0, -0.01, 0.01, 0.9),
    }
    _write_artifacts(tmp_path, deltas=_PASSING, structural_deltas=flat_structural)
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    criteria = {c["key"]: c["met"] for c in result["criteria"]}
    assert criteria["CG1_availability"] is True
    assert criteria["CG3_structural_coverage"] is False
    assert result["verdict"] == "GENERATOR PARTIALLY READY"


def test_challenge_row_fails_the_natural_pool_gate(tmp_path: object) -> None:
    _write_artifacts(tmp_path, deltas=_PASSING)
    proposals = pd.read_csv(tmp_path / "cgv2_proposals__evaluate.csv")  # type: ignore[operator]
    proposals.loc[0, "pool"] = "challenge"
    proposals.to_csv(tmp_path / "cgv2_proposals__evaluate.csv", index=False)  # type: ignore[operator]
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    criteria = {c["key"]: c["met"] for c in result["h2_readiness"]["criteria"]}
    assert criteria["E_natural_pool"] is False
    assert result["h2_readiness"]["verdict"] == "H2 NOT READY"


def test_region_candidates_are_not_double_counted_for_sample_size(tmp_path: object) -> None:
    _write_artifacts(tmp_path, deltas=_PASSING)
    path = Path(tmp_path) / "cgv2_proposals__evaluate.csv"
    proposals = pd.read_csv(path)
    proposals = proposals.groupby("engine_id", sort=False).head(998)
    proposals.to_csv(path, index=False)
    gate_path = Path(tmp_path) / "audits" / "evaluate" / "alignment_audit_gate.json"
    audit = json.loads(gate_path.read_text())
    audit["inputs"]["site_proposals_sha256"] = file_sha256(path)
    gate_path.write_text(json.dumps(audit) + "\n")

    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    criteria = {c["key"]: c["met"] for c in result["h2_readiness"]["criteria"]}
    # 998 site + 1 region = 999, below the frozen 1,000 floor. The former gate used
    # quality's already-augmented 1,500 and then added the region again.
    assert criteria["C_sample_size"] is False


def test_alignment_gate_rejects_a_stale_proposal_hash(tmp_path: object) -> None:
    _write_artifacts(tmp_path, deltas=_PASSING)
    gate_path = Path(tmp_path) / "audits" / "evaluate" / "alignment_audit_gate.json"
    audit = json.loads(gate_path.read_text())
    audit["inputs"]["site_proposals_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(audit) + "\n")

    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    criteria = {c["key"]: c["met"] for c in result["h2_readiness"]["criteria"]}
    assert criteria["F_no_unresolved_confound"] is False


def test_a_failed_integrity_review_floors_the_verdict_at_not_ready(tmp_path: object) -> None:
    """Numbers cannot lift a pool whose evidence chain failed review (protocol section 20).

    Every numerical criterion passes here; only the integrity gate fails. The verdict
    must be NOT READY -- not READY, not PARTIALLY READY, and not INCONCLUSIVE, because
    the evidence was produced and it is invalid.
    """
    _write_artifacts(
        tmp_path,
        deltas=_PASSING,
        accepted_edits={"e1": 300, "e2": 300, "e3": 300, "e4": 300},
        integrity_pass=False,
    )
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    criteria = {c["key"]: c["met"] for c in result["criteria"]}
    assert criteria["CG1_availability"] is True
    assert criteria["VALIDITY_integrity_review"] is False
    assert result["verdict"] == "GENERATOR NOT READY"
    assert result["h2_readiness"]["verdict"] == "H2 NOT READY"


def test_a_missing_integrity_gate_is_inconclusive(tmp_path: object) -> None:
    """Without the validity review the verdict cannot be produced, only deferred."""
    _write_artifacts(tmp_path, deltas=_PASSING, integrity_pass=None)
    result = evaluate_cgv2_gates(tmp_path)  # type: ignore[arg-type]
    criteria = {c["key"]: c["met"] for c in result["criteria"]}
    assert criteria["VALIDITY_integrity_review"] is None
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["h2_readiness"]["verdict"] == "H2 INCONCLUSIVE"


def test_audit_lineage_mismatches_detect_a_stale_alignment_chain() -> None:
    """R-74's guard: the audit must be bound to the store tables it certified, not only
    to the proposal bytes, which a re-run align chain can leave identical."""
    from ocr_risk.experiments.cgv2_gates import audit_lineage_mismatches

    audit_gate = {
        "input_runs": {
            "align": "align-1",
            "sites": "sites-1",
            "canonicalize": "canonicalize-1",
            "manifest": "manifest-1",
        },
        "inputs": {
            "alignments_sha256": "a" * 64,
            "sites_sha256": "b" * 64,
            "spans_sha256": "c" * 64,
            "gt_tokens_sha256": "d" * 64,
        },
    }
    fresh = {
        "align": ("align-1", "a" * 64),
        "sites": ("sites-1", "b" * 64),
        "canonicalize": ("canonicalize-1", "c" * 64),
        "manifest": ("manifest-1", "d" * 64),
    }
    assert audit_lineage_mismatches(audit_gate, fresh) == []
    # Same run ids, but the alignments table hash moved: a re-run with a changed
    # confidence knob rewrites the table under a new run id normally; a same-id hash
    # move catches in-place rewrites.
    retuned = dict(fresh, align=("align-1", "e" * 64))
    assert audit_lineage_mismatches(audit_gate, retuned) == [
        "alignments: table hash moved since the audit (alignments_sha256)"
    ]
    replaced = dict(fresh, sites=("sites-2", "b" * 64))
    assert any("sites" in item for item in audit_lineage_mismatches(audit_gate, replaced))
    assert audit_lineage_mismatches(audit_gate, {}) == [
        f"{stage}: no current {table} run for this experiment"
        for stage, table in (
            ("align", "alignments"),
            ("sites", "sites"),
            ("canonicalize", "spans"),
            ("manifest", "gt_tokens"),
        )
    ]

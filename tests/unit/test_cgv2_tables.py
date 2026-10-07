"""Hand-computed tests for the CGV2 §10 tables."""

from __future__ import annotations

import pandas as pd
import pytest

from ocr_risk.analysis.cgv2_tables import (
    budget_table,
    candidate_quality_table,
    failure_taxonomy_by_dataset_table,
    failure_taxonomy_table,
    opportunity_by_dataset_table,
    opportunity_table,
    preservation_table,
    region_opportunity_table,
    structural_coverage_table,
    structural_operation_by_dataset_table,
    structural_operation_table,
)
from ocr_risk.schemas.enums import HarmPolicy


def _census() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_id": "s1",
                "document_id": "d1",
                "engine_id": "e1",
                "dataset_id": "funsd",
                "site_kind": "substitution",
                "d_before": 1,
                "gt_text": "word",
                "ocr_text": "wurd",
                "n_spans": 1,
            },
            {
                "site_id": "s2",
                "document_id": "d1",
                "engine_id": "e1",
                "dataset_id": "funsd",
                "site_kind": "deletion",
                "d_before": 3,
                "gt_text": "due",
                "ocr_text": "",
                "n_spans": 0,
            },
            {
                "site_id": "s3",
                "document_id": "d1",
                "engine_id": "e1",
                "dataset_id": "funsd",
                "site_kind": "clean",
                "d_before": 0,
                "gt_text": "fine",
                "ocr_text": "fine",
                "n_spans": 1,
            },
            {
                "site_id": "s4",
                "document_id": "d1",
                "engine_id": "e1",
                "dataset_id": "funsd",
                "site_kind": "substitution",
                "d_before": 9,
                "gt_text": "unknowableword",
                "ocr_text": "unknowablevvord",
                "n_spans": 1,
            },
        ]
    )


def _proposals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generator_id": "g6_union",
                "site_id": "s1",
                "engine_id": "e1",
                "outcome": "MISCORRECTION",
                "generator_rank": 0,
                "gt_text": "word",
                "candidate_text": "ward",
                "pool": "natural",
            },
            {
                "generator_id": "g6_union",
                "site_id": "s1",
                "engine_id": "e1",
                "outcome": "TRUE_CORRECTION",
                "generator_rank": 1,
                "gt_text": "word",
                "candidate_text": "word",
                "pool": "natural",
            },
            {
                "generator_id": "g6_union",
                "site_id": "s2",
                "engine_id": "e1",
                "outcome": "TRUE_CORRECTION",
                "generator_rank": 0,
                "gt_text": "due",
                "candidate_text": "due",
                "pool": "natural",
            },
            {
                "generator_id": "g6_union",
                "site_id": "s3",
                "engine_id": "e1",
                "outcome": "OVERCORRECTION",
                "generator_rank": 0,
                "gt_text": "fine",
                "candidate_text": "finer",
                "pool": "natural",
            },
        ]
    )


def _lexicons() -> pd.DataFrame:
    return pd.DataFrame([{"engine_id": "e1", "token": t} for t in ("word", "due", "fine")])


class TestOpportunity:
    def test_availability_at_each_k_is_hand_computed(self) -> None:
        table = opportunity_table(_census(), _proposals())
        rung = table[table["generator_id"] == "g6_union"]
        all_stratum = rung[rung["stratum"] == "all"].set_index("k")
        # Three error sites (s1, s2, s4). s1's best repair sits at rank 1, s2's at
        # rank 0, s4 has none.
        assert all_stratum.loc[1, "n_available"] == 1
        assert all_stratum.loc[1, "availability"] == 1 / 3
        assert all_stratum.loc[2, "n_available"] == 2
        assert all_stratum.loc[2, "availability"] == 2 / 3
        deletion = rung[rung["stratum"] == "deletion"].set_index("k")
        assert deletion.loc[1, "availability"] == 1.0
        assert all_stratum.loc[1, "exact_availability"] == 1 / 3
        assert all_stratum.loc[2, "mean_first_beneficial_rank"] == 0.5

    def test_clean_sites_are_not_in_the_denominator(self) -> None:
        table = opportunity_table(_census(), _proposals())
        all_stratum = table[(table["generator_id"] == "g6_union") & (table["stratum"] == "all")]
        assert (all_stratum["n_error_sites"] == 3).all()  # s1, s2, s4


class TestQuality:
    def test_rates_and_ratio_are_hand_computed(self) -> None:
        table = candidate_quality_table(_census(), _proposals())
        row = table[(table["generator_id"] == "g6_union") & (table["engine_id"] == "e1")].iloc[0]
        assert row["n_candidates"] == 4
        assert row["n_beneficial"] == 2
        assert row["n_harmful"] == 2
        assert row["harmful_beneficial_ratio"] == 1.0
        assert row["candidates_per_site"] == 1.0  # 4 candidates over 4 sites

    def test_a_rung_engine_cell_with_no_pool_candidates_is_an_explicit_zero(self) -> None:
        """The R-62 rule in the quality grid: an observed zero is a row, not a gap.

        g5_structural is observed on engine e2, so its empty cell on e1 (every candidate
        there removed by the audit overlay, or none generated) must be recorded with
        zero candidates -- a missing row would read as unmeasured evidence.
        """
        proposals = pd.concat(
            [
                _proposals(),
                pd.DataFrame(
                    [
                        {
                            "generator_id": "g5_structural",
                            "site_id": "s2",
                            "engine_id": "e2",
                            "outcome": "TRUE_CORRECTION",
                            "generator_rank": 0,
                            "gt_text": "due",
                            "candidate_text": "due",
                            "pool": "natural",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        census = pd.concat(
            [
                _census(),
                _census().assign(site_id="s1-e2", engine_id="e2"),
            ],
            ignore_index=True,
        )
        table = candidate_quality_table(census, proposals)
        cell = table[(table["generator_id"] == "g5_structural") & (table["engine_id"] == "e1")]
        assert len(cell) == 1
        assert cell["n_candidates"].iloc[0] == 0
        assert cell["n_beneficial"].iloc[0] == 0
        assert cell["candidates_per_site"].iloc[0] == 0.0

    def test_harm_policy_sensitivity_reclassifies_lateral_and_partial_changes(self) -> None:
        proposals = pd.concat(
            [
                _proposals(),
                pd.DataFrame(
                    [
                        {
                            "generator_id": "g6_union",
                            "site_id": "s1",
                            "engine_id": "e1",
                            "outcome": "LATERAL_CHANGE",
                            "generator_rank": 2,
                            "gt_text": "word",
                            "candidate_text": "wird",
                            "pool": "natural",
                        },
                        {
                            "generator_id": "g6_union",
                            "site_id": "s4",
                            "engine_id": "e1",
                            "outcome": "PARTIAL_IMPROVEMENT",
                            "generator_rank": 0,
                            "gt_text": "unknowableword",
                            "candidate_text": "unknowablew0rd",
                            "pool": "natural",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        strict = candidate_quality_table(_census(), proposals).iloc[0]
        non_improving = candidate_quality_table(
            _census(), proposals, policy=HarmPolicy.NON_IMPROVING
        ).iloc[0]
        exact = candidate_quality_table(_census(), proposals, policy=HarmPolicy.EXACT_ONLY).iloc[0]
        assert strict["n_beneficial"] == 3
        assert strict["n_harmful"] == 2
        assert non_improving["n_beneficial"] == 3
        assert non_improving["n_harmful"] == 3
        assert exact["n_beneficial"] == 2
        assert exact["n_harmful"] == 4
        assert exact["harm_policy"] == "exact_only"


class TestStructuralCoverage:
    def test_only_structural_strata_survive(self) -> None:
        table = structural_coverage_table(_census(), _proposals())
        assert set(table["stratum"]) <= {"deletion", "insertion", "segmentation"}

    def test_true_repair_operations_use_the_protocol_mapping(self) -> None:
        table = structural_operation_table(_census(), _proposals(), k=2)
        rung = table[table["generator_id"] == "g6_union"].set_index("operation")
        assert rung.loc["substitution", "n_error_sites"] == 2
        assert rung.loc["insertion", "n_error_sites"] == 1
        assert rung.loc["insertion", "availability"] == 1.0


def test_region_opportunity_requires_exact_repair_with_an_insertion_member() -> None:
    regions = pd.DataFrame(
        [
            {
                "generator_id": "g5_structural",
                "region_id": "r-insertion",
                "engine_id": "e1",
                "left_kind": "substitution",
                "right_kind": "insertion",
                "outcome": "PARTIAL_IMPROVEMENT",
                "pool": "natural",
            },
            {
                "generator_id": "g5_structural",
                "region_id": "r-ordinary",
                "engine_id": "e1",
                "left_kind": "substitution",
                "right_kind": "segmentation",
                "outcome": "PARTIAL_IMPROVEMENT",
                "pool": "natural",
            },
        ]
    )
    # The denominator is the audited eligible-pair population, not a pre-aggregated
    # census: ten eligible adjacent pairs on one document cluster.
    pairs = pd.DataFrame(
        [
            {
                "population": "study_evaluable_adjacency",
                "region_id": f"e1:region:{index:05d}",
                "document_id": "d1",
                "engine_id": "e1",
                "state": "eligible",
            }
            for index in range(10)
        ]
    )
    row = region_opportunity_table(regions, pairs).iloc[0]
    assert row["n_regions_beneficial"] == 1
    assert row["beneficial_region_share"] == 0.1


def test_dataset_opportunity_keeps_the_dataset_identifier() -> None:
    table = opportunity_by_dataset_table(_census(), _proposals())
    assert set(table["dataset_id"]) == {"funsd"}


def test_dataset_structural_and_failure_tables_keep_the_dataset_identifier() -> None:
    structural = structural_operation_by_dataset_table(_census(), _proposals())
    failures = failure_taxonomy_by_dataset_table(_census(), _proposals(), _lexicons())
    assert set(structural["dataset_id"]) == {"funsd"}
    assert set(failures["dataset_id"]) == {"funsd"}


class TestPreservation:
    def test_clean_site_proposals_are_counted(self) -> None:
        table = preservation_table(_census(), _proposals())
        row = table[(table["generator_id"] == "g6_union") & (table["engine_id"] == "e1")].iloc[0]
        assert row["n_clean_sites"] == 1
        assert row["n_clean_sites_proposed"] == 1
        assert row["clean_span_proposal_rate"] == 1.0

    def test_zero_clean_proposals_are_an_explicit_zero_not_a_missing_row(self) -> None:
        proposals = _proposals()[_proposals()["site_id"] != "s3"]
        table = preservation_table(_census(), proposals)
        row = table[(table["generator_id"] == "g6_union") & (table["engine_id"] == "e1")]
        assert len(row) == 1
        assert row["n_clean_sites_proposed"].iloc[0] == 0
        assert row["clean_span_proposal_rate"].iloc[0] == 0.0


class TestBudget:
    def test_recall_and_burden_at_each_k(self) -> None:
        table = budget_table(_census(), _proposals())
        row = table[(table["generator_id"] == "g6_union") & (table["k"] == 1)].iloc[0]
        # Exact repairs within top-1: s2 only, over three error sites.
        assert row["exact_recall"] == 1 / 3
        # Harmful proposals within top-1: s1 rank 0 and s3 rank 0, over four sites.
        assert row["harmful_burden"] == 0.5
        k2 = table[(table["generator_id"] == "g6_union") & (table["k"] == 2)].iloc[0]
        assert k2["exact_recall"] == 2 / 3


class TestFailureTaxonomy:
    def test_unrepaired_oov_site_is_vocabulary_absent(self) -> None:
        table = failure_taxonomy_table(_census(), _proposals(), _lexicons(), rung="g6_union", k=4)
        row = table[table["failure_class"] == "beneficial_absent_vocabulary"]
        assert row["n"].iloc[0] == 1  # s4: gt not in the fold lexicon

    def test_deletion_site_is_inexpressible_for_the_lexical_rung(self) -> None:
        table = failure_taxonomy_table(_census(), _proposals(), _lexicons(), rung="g0_lexical", k=4)
        inexpressible = table[table["failure_class"] == "repair_not_expressible_by_rung"]
        # For g0 every non-substitution error site is inexpressible: s2 (deletion) and s4
        # is substitution, so exactly one.
        assert inexpressible["n"].sum() == 1


def test_outcomes_are_matched_case_insensitively() -> None:
    """The enum serializes lowercase on CSV round-trip; uppercase matching alone returns
    an empty opportunity table -- a defect the synthetic fixtures masked."""
    proposals = _proposals()
    proposals["outcome"] = proposals["outcome"].str.lower()
    table = opportunity_table(_census(), proposals)
    rung = table[(table["generator_id"] == "g6_union") & (table["stratum"] == "all")]
    assert not rung.empty
    assert rung.set_index("k").loc[1, "n_available"] == 1


@pytest.mark.parametrize(
    "analyze",
    [opportunity_table, candidate_quality_table, preservation_table, budget_table],
)
def test_headline_tables_reject_a_challenge_pool_row(analyze: object) -> None:
    proposals = _proposals()
    proposals.loc[0, "pool"] = "challenge"
    with pytest.raises(ValueError, match="non-natural"):
        analyze(_census(), proposals)  # type: ignore[operator]


def test_headline_tables_fail_closed_when_pool_provenance_is_missing() -> None:
    with pytest.raises(ValueError, match="missing the required 'pool'"):
        opportunity_table(_census(), _proposals().drop(columns="pool"))

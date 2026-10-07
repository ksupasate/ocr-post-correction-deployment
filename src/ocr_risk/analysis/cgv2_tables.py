"""CGV2 §10 tables, computed from the study's canonical CSVs and nothing else.

Every table here reads the census, proposals, region proposals, fold lexicons, contrasts,
and oracle CSVs that ``ocr-risk cgv2 study`` wrote. Nothing upstream is recomputed, no
number is entered by hand, and the failure taxonomy's rung-capability map is the one
frozen in the protocol (§3).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

import pandas as pd

from ocr_risk.edits.outcome import is_beneficial, is_harmful
from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER
from ocr_risk.schemas.enums import HarmPolicy, OutcomeIfAccepted
from ocr_risk.stats import cluster_bootstrap

__all__ = [
    "audit_clean_pool",
    "budget_table",
    "candidate_quality_by_dataset_table",
    "candidate_quality_table",
    "failure_taxonomy_by_dataset_table",
    "failure_taxonomy_table",
    "opportunity_by_dataset_table",
    "opportunity_table",
    "preservation_table",
    "region_opportunity_table",
    "structural_coverage_table",
    "structural_operation_by_dataset_table",
    "structural_operation_table",
]

STRUCTURAL_KINDS = ("deletion", "insertion", "segmentation")
K_GRID = (1, 2, 4, 8)

# The protocol §3 expressibility map, per rung. A site kind absent from a rung's row is
# a repair that rung cannot express -- the first failure-taxonomy class.
_EXPRESSIBLE: dict[str, frozenset[str]] = {
    "g0_lexical": frozenset({"substitution"}),
    "g1_error_gated": frozenset({"substitution"}),
    "g2_byt5": frozenset({"substitution", "segmentation"}),
    "g2_byt5_ctx0": frozenset({"substitution", "segmentation"}),
    "g3_edit_aware": frozenset({"substitution", "insertion", "segmentation"}),
    "g4_union": frozenset({"substitution", "insertion", "segmentation"}),
    "g5_structural": frozenset({"deletion", "segmentation"}),
    "g6_union": frozenset({"substitution", "deletion", "insertion", "segmentation"}),
}


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().eq("true")


def audit_clean_pool(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    region_proposals: pd.DataFrame | None,
    site_audit: pd.DataFrame,
    site_projection: pd.DataFrame,
    region_projection: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, int]]:
    """Apply the outcome-independent alignment validity overlay to canonical rows.

    The overlay cleans the *pool*, not the benchmark: the census keeps every evaluable
    site so availability denominators stay the pre-registered ones (dropping anchor-less
    sites would silently shrink the insertion stratum and turn CG3 into missing evidence
    instead of a measured zero), while candidates enter headline tables only when their
    site projection and recomputed label are valid. Sites the audit cannot ground carry
    availability 0 by construction -- an honest measurement, not a hidden exclusion.
    """
    required_site = {"site_id", "state"}
    if not required_site <= set(site_audit.columns):
        raise ValueError("alignment_site_audit.csv lacks site_id/state")

    required_projection = {"candidate_id", "site_id", "state", "projection_correct", "label_valid"}
    if not required_projection <= set(site_projection.columns):
        raise ValueError("site_projection_audit.csv lacks the validity-overlay columns")
    if site_projection["candidate_id"].astype(str).duplicated().any():
        raise ValueError("site projection audit contains duplicate candidate ids")
    eligible_candidate_ids = set(
        site_projection[
            (site_projection["state"].astype(str) == "eligible")
            & _boolean_series(site_projection["projection_correct"])
            & _boolean_series(site_projection["label_valid"])
        ]["candidate_id"].astype(str)
    )

    audited_site_ids = set(site_audit["site_id"].astype(str))
    unknown_sites = set(proposals["site_id"].astype(str)) - audited_site_ids
    if unknown_sites:
        raise ValueError(
            f"{len(unknown_sites)} proposal site(s) are absent from the alignment site audit"
        )
    unaudited_candidates = set(proposals["candidate_id"].astype(str)) - set(
        site_projection["candidate_id"].astype(str)
    )
    if unaudited_candidates:
        raise ValueError(
            f"{len(unaudited_candidates)} site candidate(s) are absent from the projection "
            "audit; a stale audit must fail, not silently shrink the pool"
        )

    clean_census = census.copy()
    clean_proposals = proposals[
        proposals["candidate_id"].astype(str).isin(eligible_candidate_ids)
    ].copy()

    clean_regions = region_proposals.copy() if region_proposals is not None else None
    if clean_regions is not None and not clean_regions.empty:
        if region_projection is None or region_projection.empty:
            raise ValueError("region proposals exist without region_projection_audit.csv")
        required_region = {
            "candidate_id",
            "state",
            "projection_correct",
            "label_valid",
        }
        if not required_region <= set(region_projection.columns):
            raise ValueError("region_projection_audit.csv lacks the validity-overlay columns")
        if region_projection["candidate_id"].astype(str).duplicated().any():
            raise ValueError("region projection audit contains duplicate candidate ids")
        unaudited_regions = set(clean_regions["candidate_id"].astype(str)) - set(
            region_projection["candidate_id"].astype(str)
        )
        if unaudited_regions:
            raise ValueError(
                f"{len(unaudited_regions)} region candidate(s) are absent from the region "
                "projection audit; a stale audit must fail, not silently shrink the pool"
            )
        eligible_region_ids = set(
            region_projection[
                (region_projection["state"].astype(str) == "eligible")
                & _boolean_series(region_projection["projection_correct"])
                & _boolean_series(region_projection["label_valid"])
            ]["candidate_id"].astype(str)
        )
        clean_regions = clean_regions[
            clean_regions["candidate_id"].astype(str).isin(eligible_region_ids)
        ].copy()

    site_states = site_audit["state"].astype(str).value_counts().to_dict()
    counts = {
        "site_rows_before": len(census),
        "site_rows_after": len(clean_census),
        "site_candidate_rows_before": len(proposals),
        "site_candidate_rows_after": len(clean_proposals),
        "region_candidate_rows_before": len(region_proposals)
        if region_proposals is not None
        else 0,
        "region_candidate_rows_after": len(clean_regions) if clean_regions is not None else 0,
        "sites_audit_eligible": int(site_states.get("eligible", 0)),
        "sites_audit_ambiguous": int(site_states.get("ambiguous", 0)),
        "sites_audit_unresolved": int(site_states.get("unresolved", 0)),
        "sites_audit_excluded": int(site_states.get("excluded", 0)),
    }
    return clean_census, clean_proposals, clean_regions, counts


def _coerce_policy(policy: HarmPolicy | str) -> HarmPolicy:
    return policy if isinstance(policy, HarmPolicy) else HarmPolicy(str(policy))


def _outcome(value: str) -> OutcomeIfAccepted:
    return OutcomeIfAccepted(str(value).lower())


def _beneficial(outcome: str, policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING) -> bool:
    return is_beneficial(_outcome(outcome), _coerce_policy(policy))


def _harmful(outcome: str, policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING) -> bool:
    return is_harmful(_outcome(outcome), _coerce_policy(policy))


def _assert_natural_pool(frame: pd.DataFrame, artifact: str) -> None:
    """Fail closed when a challenge/diagnostic row reaches a headline denominator."""
    if frame.empty:
        return
    if "pool" not in frame.columns:
        raise ValueError(f"{artifact} is missing the required 'pool' provenance column")
    values = frame["pool"].fillna("").astype(str).str.lower()
    forbidden = frame[values != "natural"]
    if not forbidden.empty:
        examples = sorted(set(forbidden["pool"].astype(str)))[:5]
        msg = (
            f"{artifact} contains {len(forbidden)} non-natural candidate(s): {examples}; "
            "challenge/diagnostic pools cannot enter CGV2 headline analyses"
        )
        raise ValueError(msg)


def _normalized_outcomes(proposals: pd.DataFrame) -> pd.DataFrame:
    """Uppercase the outcome column: the enum serializes lowercase on CSV round-trip."""
    _assert_natural_pool(proposals, "candidate proposals")
    frame = proposals.copy()
    frame["outcome"] = frame["outcome"].astype(str).str.upper()
    return frame


def _native_site_pool(proposals: pd.DataFrame) -> pd.DataFrame:
    """Return configured-rung rows; cap-8 extras remain only in K-grid analyses."""
    if proposals.empty:
        return proposals.copy()
    expected = proposals["generator_id"].map(
        {rung: spec.max_candidates for rung, spec in GENERATOR_LADDER.items()}
    )
    if expected.isna().any():
        unknown = sorted(set(proposals.loc[expected.isna(), "generator_id"].astype(str)))
        raise ValueError(f"unknown generator ids have no frozen native cap: {unknown}")
    if "generator_native_cap" in proposals.columns:
        recorded = pd.to_numeric(proposals["generator_native_cap"], errors="coerce")
        mismatch = recorded.isna() | recorded.ne(expected.astype(int))
        if mismatch.any():
            raise ValueError(
                f"{int(mismatch.sum())} site proposal(s) disagree with the frozen native cap"
            )
    ranks = pd.to_numeric(proposals["generator_rank"], errors="coerce")
    return proposals[ranks < expected.astype(int)].copy()


def _native_region_pool(proposals: pd.DataFrame) -> pd.DataFrame:
    """Apply the structural region generator's independent frozen cap of two."""
    if proposals.empty:
        return proposals.copy()
    recorded = (
        pd.to_numeric(proposals["generator_native_cap"], errors="coerce")
        if "generator_native_cap" in proposals.columns
        else pd.Series(2, index=proposals.index, dtype=int)
    )
    if recorded.isna().any() or recorded.ne(2).any():
        raise ValueError("region proposals disagree with the frozen native cap of 2")
    ranks = pd.to_numeric(proposals["generator_rank"], errors="coerce")
    return proposals[ranks < recorded].copy()


def opportunity_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    k_grid: tuple[int, ...] = K_GRID,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """Availability by rung x engine x stratum x K: the CGV2-H1 endpoint table.

    Availability is the share of *error sites* (``d_before > 0``) of the stratum where at
    least one candidate at rank < K would count as a repair. The exact variant restricts
    to TRUE_CORRECTION; both are pre-registered (§11).
    """
    rows: list[dict[str, object]] = []
    columns = [
        "generator_id",
        "engine_id",
        "harm_policy",
        "stratum",
        "k",
        "n_error_sites",
        "n_available",
        "availability",
        "n_exact_available",
        "exact_availability",
        "mean_first_beneficial_rank",
        "median_first_beneficial_rank",
        "candidates_per_error_site",
    ]
    proposals = _normalized_outcomes(proposals)
    policy = _coerce_policy(policy)
    repairable = proposals[
        proposals["outcome"].map(lambda value: _beneficial(value, policy))
    ].copy()
    # At a deletion-shaped site (empty GT) only the exact edit repairs (A6).
    needs_exact = ~repairable["gt_text"].fillna("").str.strip().astype(bool)
    repairable["counts_as_repair"] = (repairable["outcome"] == "TRUE_CORRECTION") | (
        (repairable["outcome"] == "PARTIAL_IMPROVEMENT") & ~needs_exact
    )
    all_proposals = proposals.copy()

    for rung in sorted(proposals["generator_id"].unique()):
        rung_proposals = repairable[repairable["generator_id"] == rung]
        by_site: dict[tuple[str, str], int] = {}
        exact_by_site: dict[tuple[str, str], int] = {}
        for row in rung_proposals.itertuples(index=False):
            key = (str(row.engine_id), str(row.site_id))
            rank = int(str(row.generator_rank))
            if bool(row.counts_as_repair) and (key not in by_site or rank < by_site[key]):
                by_site[key] = rank
            if str(row.outcome) == "TRUE_CORRECTION" and (
                key not in exact_by_site or rank < exact_by_site[key]
            ):
                exact_by_site[key] = rank
        for engine_id, engine_sites in census[census["d_before"] > 0].groupby(
            "engine_id", sort=True
        ):
            strata = {"all": engine_sites}
            for kind in STRUCTURAL_KINDS:
                strata[kind] = engine_sites[engine_sites["site_kind"] == kind]
            for stratum, sites in strata.items():
                for k in k_grid:
                    engine = str(engine_id)
                    hits = sum(
                        1
                        for site_id in sites["site_id"]
                        if by_site.get((engine, str(site_id)), k) < k
                    )
                    exact_hits = sum(
                        1
                        for site_id in sites["site_id"]
                        if exact_by_site.get((engine, str(site_id)), k) < k
                    )
                    ranks = [
                        by_site[(engine, str(site_id))]
                        for site_id in sites["site_id"]
                        if by_site.get((engine, str(site_id)), k) < k
                    ]
                    site_ids = set(sites["site_id"].astype(str))
                    candidate_count = len(
                        all_proposals[
                            (all_proposals["generator_id"] == rung)
                            & (all_proposals["engine_id"].astype(str) == engine)
                            & (all_proposals["site_id"].astype(str).isin(site_ids))
                            & (all_proposals["generator_rank"] < k)
                        ]
                    )
                    rows.append(
                        {
                            "generator_id": rung,
                            "engine_id": engine_id,
                            "harm_policy": policy.value,
                            "stratum": stratum,
                            "k": k,
                            "n_error_sites": len(sites),
                            "n_available": hits,
                            "availability": hits / len(sites) if len(sites) else float("nan"),
                            "n_exact_available": exact_hits,
                            "exact_availability": exact_hits / len(sites)
                            if len(sites)
                            else float("nan"),
                            "mean_first_beneficial_rank": sum(ranks) / len(ranks)
                            if ranks
                            else float("nan"),
                            "median_first_beneficial_rank": float(pd.Series(ranks).median())
                            if ranks
                            else float("nan"),
                            "candidates_per_error_site": candidate_count / len(sites)
                            if len(sites)
                            else float("nan"),
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def opportunity_by_dataset_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """The opportunity table with datasets kept separate, including the stress track."""
    tables: list[pd.DataFrame] = []
    for dataset_id, dataset_census in census.groupby("dataset_id", sort=True):
        dataset_sites = set(dataset_census["site_id"].astype(str))
        dataset_proposals = proposals[proposals["site_id"].astype(str).isin(dataset_sites)]
        table = opportunity_table(dataset_census, dataset_proposals, policy=policy)
        if not table.empty:
            tables.append(table.assign(dataset_id=dataset_id))
    if not tables:
        return pd.DataFrame(
            columns=[*opportunity_table(census, proposals, policy=policy).columns, "dataset_id"]
        )
    return pd.concat(tables, ignore_index=True)


def candidate_quality_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    region_proposals: pd.DataFrame | None = None,
    pool_rung: str = "g6_union",
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """CGV2-H3: beneficial/harmful/neutral rates, ratios, and per-site burdens.

    Region candidates count toward the pool rung they belong to (protocol section 14:
    the CGV2 pool is g6's site candidates plus g5's regions), so the harmful burden the
    gate bounds is the burden of the pool that actually exists, not of its site half.
    """
    policy = _coerce_policy(policy)
    frame = _normalized_outcomes(_native_site_pool(proposals))
    if region_proposals is not None and not region_proposals.empty:
        regions = _normalized_outcomes(_native_region_pool(region_proposals))
        regions = regions.assign(
            generator_id=pool_rung,
            site_id=regions["region_id"],
            gt_text=regions["region_gt"],
        )
        columns = [
            "generator_id",
            "site_id",
            "engine_id",
            "outcome",
            "generator_rank",
            "gt_text",
            "candidate_text",
        ]
        frame = pd.concat([frame[columns], regions[columns]], ignore_index=True)
    frame["class"] = frame["outcome"].map(
        lambda outcome: (
            "beneficial"
            if _beneficial(str(outcome), policy)
            else "harmful"
            if _harmful(str(outcome), policy)
            else "neutral"
        )
    )
    n_sites = census.groupby("engine_id")["site_id"].count()
    rows: list[dict[str, object]] = []
    # The complete rung x engine grid: a rung whose pool candidates were all removed by
    # the audit overlay records an explicit zero, never a missing row (the R-62 rule).
    rungs_present = (
        sorted(set(proposals["generator_id"].astype(str))) if not proposals.empty else []
    )
    if region_proposals is not None and not region_proposals.empty:
        rungs_present = sorted(set(rungs_present) | {pool_rung})
    cells = {
        (str(rung), str(engine_id))
        for rung in rungs_present
        for engine_id in n_sites.index.astype(str)
    }
    cells.update(
        (str(rung), str(engine_id))
        for rung, engine_id in frame[["generator_id", "engine_id"]].itertuples(index=False)
    )
    for rung, engine_id in sorted(cells):
        group = frame[
            (frame["generator_id"].astype(str) == rung)
            & (frame["engine_id"].astype(str) == engine_id)
        ]
        n = len(group)
        n_beneficial = int((group["class"] == "beneficial").sum()) if n else 0
        n_harmful = int((group["class"] == "harmful").sum()) if n else 0
        rows.append(
            {
                "generator_id": rung,
                "engine_id": engine_id,
                "harm_policy": policy.value,
                "n_candidates": n,
                "n_beneficial": n_beneficial,
                "n_harmful": n_harmful,
                "n_neutral": n - n_beneficial - n_harmful,
                "n_sites": int(n_sites.get(engine_id, 0)),
                "beneficial_rate": n_beneficial / n if n else float("nan"),
                "harmful_rate": n_harmful / n if n else float("nan"),
                "harmful_beneficial_ratio": n_harmful / n_beneficial
                if n_beneficial
                else float("inf"),
                "candidates_per_site": n / int(n_sites.get(engine_id, 0))
                if n_sites.get(engine_id)
                else float("nan"),
                "useful_candidates_per_site": n_beneficial / int(n_sites.get(engine_id, 0))
                if n_sites.get(engine_id)
                else float("nan"),
                "harmful_candidates_per_site": n_harmful / int(n_sites.get(engine_id, 0))
                if n_sites.get(engine_id)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def candidate_quality_by_dataset_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    region_proposals: pd.DataFrame | None = None,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """Candidate composition by dataset, so OCR-D never disappears into a pooled mean."""
    tables: list[pd.DataFrame] = []
    for dataset_id, dataset_census in census.groupby("dataset_id", sort=True):
        site_ids = set(dataset_census["site_id"].astype(str))
        dataset_proposals = proposals[proposals["site_id"].astype(str).isin(site_ids)]
        dataset_regions = (
            region_proposals[region_proposals["dataset_id"] == dataset_id]
            if region_proposals is not None and not region_proposals.empty
            else None
        )
        table = candidate_quality_table(
            dataset_census, dataset_proposals, dataset_regions, policy=policy
        )
        if not table.empty:
            tables.append(table.assign(dataset_id=dataset_id))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def structural_coverage_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """CGV2-H2: availability on each structural stratum, by rung x engine x kind."""
    table = opportunity_table(census, proposals, k_grid=(4,), policy=policy)
    return table[table["stratum"] != "all"].copy()


def _site_repair_operation(site_kind: str, n_spans: int, gt_text: str) -> str:
    kind = site_kind
    if kind in {"clean", "substitution"}:
        return "substitution"
    if kind == "deletion":
        return "insertion"
    if kind == "insertion":
        return "deletion"
    if kind != "segmentation":
        return "mixed"
    n_ocr = n_spans or 1
    n_gt = len(gt_text.split())
    if n_ocr == 1 and n_gt > 1:
        return "split"
    if n_ocr > 1 and n_gt == 1:
        return "merge"
    return "mixed"


def _cluster_rate_interval(
    population: pd.DataFrame,
    hit_ids: set[str],
    *,
    id_column: str,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, object]:
    items = [
        (
            str(document_id),
            len(group),
            int(group[id_column].astype(str).isin(hit_ids).sum()),
        )
        for document_id, group in population.groupby("document_id", sort=True)
    ]

    def _rate(sample: Sequence[tuple[str, int, int]]) -> float:
        denominator = sum(row[1] for row in sample)
        return sum(row[2] for row in sample) / denominator if denominator else float("nan")

    result = cluster_bootstrap(
        items,
        lambda item: item[0],
        _rate,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        bounds=(0.0, 1.0),
    )
    return {
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "ci_level": result.ci_level,
        "n_resamples": result.n_resamples,
        "n_documents": result.n_clusters,
        "standard_error": result.standard_error,
        "degenerate_interval": result.degenerate_interval,
    }


def structural_operation_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    k: int = 4,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
    n_resamples: int = 10_000,
    seed: int = 7,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Availability by true repair operation: substitution/insertion/deletion/split/merge/mixed."""
    proposals = _normalized_outcomes(proposals)
    policy = _coerce_policy(policy)
    error_sites = census[census["d_before"] > 0].copy()
    if "n_spans" not in error_sites:
        error_sites["n_spans"] = 1
    error_sites["operation"] = [
        _site_repair_operation(str(kind), int(str(n_spans or 1)), str(gt_text))
        for kind, n_spans, gt_text in zip(
            error_sites["site_kind"], error_sites["n_spans"], error_sites["gt_text"], strict=True
        )
    ]
    repairable = proposals[
        (proposals["generator_rank"] < k)
        & proposals["outcome"].map(lambda value: _beneficial(value, policy))
        & (
            proposals["gt_text"].fillna("").str.strip().astype(bool)
            | (proposals["outcome"] == "TRUE_CORRECTION")
        )
    ]
    exact = proposals[
        (proposals["generator_rank"] < k) & (proposals["outcome"] == "TRUE_CORRECTION")
    ]
    rows: list[dict[str, object]] = []
    for rung in sorted(proposals["generator_id"].unique()):
        rung_repairable = set(repairable[repairable["generator_id"] == rung]["site_id"].astype(str))
        rung_exact = set(exact[exact["generator_id"] == rung]["site_id"].astype(str))
        for (engine_id, operation), group in error_sites.groupby(
            ["engine_id", "operation"], sort=True
        ):
            ids = set(group["site_id"].astype(str))
            n = len(ids)
            n_available = len(ids & rung_repairable)
            n_exact = len(ids & rung_exact)
            interval = _cluster_rate_interval(
                group,
                ids & rung_repairable,
                id_column="site_id",
                n_resamples=n_resamples,
                seed=seed,
                ci_level=ci_level,
            )
            rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "harm_policy": policy.value,
                    "operation": operation,
                    "k": k,
                    "n_error_sites": n,
                    "n_available": n_available,
                    "availability": n_available / n if n else float("nan"),
                    "n_exact_available": n_exact,
                    "exact_availability": n_exact / n if n else float("nan"),
                    **interval,
                }
            )
    return pd.DataFrame(rows)


def structural_operation_by_dataset_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    k: int = 4,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
    n_resamples: int = 10_000,
    seed: int = 7,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Structural-operation coverage with the historical stress track kept separate."""
    tables: list[pd.DataFrame] = []
    for dataset_id, dataset_census in census.groupby("dataset_id", sort=True):
        site_ids = set(dataset_census["site_id"].astype(str))
        dataset_proposals = proposals[proposals["site_id"].astype(str).isin(site_ids)]
        table = structural_operation_table(
            dataset_census,
            dataset_proposals,
            k,
            policy,
            n_resamples,
            seed,
            ci_level,
        )
        if not table.empty:
            tables.append(table.assign(dataset_id=dataset_id))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def region_opportunity_table(
    region_proposals: pd.DataFrame | None,
    region_pairs: pd.DataFrame | None,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
    n_resamples: int = 10_000,
    seed: int = 7,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Region n:m opportunity against the independently audited eligible-pair denominator."""
    columns = [
        "generator_id",
        "engine_id",
        "harm_policy",
        "operation",
        "n_eligible_regions",
        "n_regions_proposed",
        "n_regions_beneficial",
        "n_regions_exact",
        "n_candidates",
        "beneficial_region_share",
        "ci_lower",
        "ci_upper",
        "ci_level",
        "n_resamples",
        "n_documents",
        "standard_error",
        "degenerate_interval",
    ]
    if region_proposals is None or region_proposals.empty:
        return pd.DataFrame(columns=columns)
    regions = _normalized_outcomes(region_proposals)
    policy = _coerce_policy(policy)
    eligible_pairs = (
        region_pairs[region_pairs["state"].astype(str) == "eligible"].copy()
        if region_pairs is not None and not region_pairs.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for engine_id, group in regions.groupby("engine_id", sort=True):
        beneficial_mask = group["outcome"].map(lambda value: _beneficial(value, policy))
        needs_exact = (group["left_kind"] == "insertion") | (group["right_kind"] == "insertion")
        beneficial = group[
            beneficial_mask & (~needs_exact | (group["outcome"] == "TRUE_CORRECTION"))
        ]
        exact = group[group["outcome"] == "TRUE_CORRECTION"]
        engine_pairs = eligible_pairs[eligible_pairs["engine_id"] == engine_id]
        denominator = len(engine_pairs)
        beneficial_ids = set(beneficial["region_id"].astype(str))
        interval = _cluster_rate_interval(
            engine_pairs,
            beneficial_ids,
            id_column="region_id",
            n_resamples=n_resamples,
            seed=seed,
            ci_level=ci_level,
        )
        rows.append(
            {
                "generator_id": "g6_union",
                "engine_id": engine_id,
                "harm_policy": policy.value,
                "operation": "region_n_to_m",
                "n_eligible_regions": denominator,
                "n_regions_proposed": group["region_id"].nunique(),
                "n_regions_beneficial": beneficial["region_id"].nunique(),
                "n_regions_exact": exact["region_id"].nunique(),
                "n_candidates": len(group),
                "beneficial_region_share": beneficial["region_id"].nunique() / denominator
                if denominator
                else float("nan"),
                **interval,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def preservation_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    region_proposals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Preservation risk: how often each rung proposes at already-correct sites.

    Region proposals count at every clean site they touch (protocol section 14, D), so a
    region joining two correct words is overcorrection opportunity the gate can see.
    """
    _assert_natural_pool(proposals, "candidate proposals")
    if region_proposals is not None:
        _assert_natural_pool(region_proposals, "region proposals")
    proposals = _native_site_pool(proposals)
    if region_proposals is not None and not region_proposals.empty:
        region_proposals = _native_region_pool(region_proposals)
    clean = census[census["d_before"] == 0]
    clean_ids = set(clean["site_id"])
    clean_by_engine = {
        str(engine_id): set(group["site_id"]) for engine_id, group in clean.groupby("engine_id")
    }
    proposed: dict[tuple[str, str], set[str]] = {}
    proposal_counts: dict[tuple[str, str], int] = {}
    for row in proposals.itertuples(index=False):
        site_id = str(row.site_id)
        if site_id in clean_ids:
            key = (str(row.generator_id), str(row.engine_id))
            proposed.setdefault(key, set()).add(site_id)
            proposal_counts[key] = proposal_counts.get(key, 0) + 1
    if region_proposals is not None and not region_proposals.empty:
        for row in region_proposals.itertuples(index=False):
            for site_id in (str(row.left_site_id), str(row.right_site_id)):
                if site_id in clean_ids:
                    key = ("g6_union", str(row.engine_id))
                    proposed.setdefault(key, set()).add(site_id)
                    proposal_counts[key] = proposal_counts.get(key, 0) + 1

    rows: list[dict[str, object]] = []
    rungs = set(proposals["generator_id"].astype(str))
    if region_proposals is not None and not region_proposals.empty:
        rungs.add("g6_union")
    for rung in sorted(rungs):
        for engine_id, engine_clean in sorted(clean_by_engine.items()):
            site_ids = proposed.get((rung, engine_id), set())
            touched = len(site_ids & engine_clean)
            rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "n_clean_sites": len(engine_clean),
                    "n_clean_sites_proposed": touched,
                    "n_unnecessary_proposals": proposal_counts.get((rung, engine_id), 0),
                    "clean_span_proposal_rate": touched / len(engine_clean)
                    if engine_clean
                    else float("nan"),
                    "unnecessary_proposals_per_clean_site": proposal_counts.get(
                        (rung, engine_id), 0
                    )
                    / len(engine_clean)
                    if engine_clean
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def budget_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """The K-grid budget study: recall@K and harmful burden@K (protocol §12)."""
    proposals = _normalized_outcomes(proposals)
    policy = _coerce_policy(policy)
    rows: list[dict[str, object]] = []
    for rung, group in proposals.groupby("generator_id", sort=True):
        for engine_id, engine_group in group.groupby("engine_id", sort=True):
            for k in K_GRID:
                within = engine_group[engine_group["generator_rank"] < k]
                repairable = within[within["outcome"] == "TRUE_CORRECTION"]["site_id"].nunique()
                beneficial = within[
                    (
                        within["outcome"].map(lambda value: _beneficial(value, policy))
                        & within["gt_text"].fillna("").str.strip().astype(bool)
                    )
                    | (within["outcome"] == "TRUE_CORRECTION")
                ]["site_id"].nunique()
                harmful = int(within["outcome"].map(lambda value: _harmful(value, policy)).sum())
                engine_census = census[census["engine_id"] == engine_id]
                n_error = int((engine_census["d_before"] > 0).sum())
                rows.append(
                    {
                        "generator_id": rung,
                        "engine_id": engine_id,
                        "harm_policy": policy.value,
                        "k": k,
                        "exact_recall": repairable / n_error if n_error else float("nan"),
                        "beneficial_recall": beneficial / n_error if n_error else float("nan"),
                        "harmful_burden": harmful / len(engine_census)
                        if len(engine_census)
                        else float("nan"),
                        "candidates_per_site": len(within) / len(engine_census)
                        if len(engine_census)
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def failure_taxonomy_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    fold_lexicons: pd.DataFrame,
    rung: str = "g6_union",
    k: int = 4,
    twin_sites: pd.DataFrame | None = None,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """One class per unrepaired error site, in the protocol section-13 precedence order.

    ``twin_sites`` (from ``ocr-risk analyze alignment-twin``) supplies the per-site R-37
    signature; sites also count as alignment-ambiguous when their weakest component
    confidence sits at the floor. The historical-charset class is the OCR-D long-s: a
    ground-truth character the modern-German engines cannot emit.
    """
    proposals = _normalized_outcomes(proposals)
    policy = _coerce_policy(policy)
    lexicon_by_engine = {
        str(engine_id): set(group["token"])
        for engine_id, group in fold_lexicons.groupby("engine_id")
    }
    twin_ids = (
        set(twin_sites[twin_sites["twin_signature"].astype(bool)]["site_id"].astype(str))
        if twin_sites is not None and not twin_sites.empty
        else set()
    )
    expressible = _EXPRESSIBLE.get(rung, frozenset())
    rung_proposals = proposals[proposals["generator_id"] == rung]
    repairable_sites = set(
        rung_proposals[
            (rung_proposals["generator_rank"] < k)
            & (
                (
                    rung_proposals["outcome"].map(lambda value: _beneficial(value, policy))
                    & rung_proposals["gt_text"].fillna("").str.strip().astype(bool)
                )
                | (rung_proposals["outcome"] == "TRUE_CORRECTION")
            )
        ]["site_id"]
    )
    beyond_k = set(
        rung_proposals[
            (rung_proposals["generator_rank"] >= k)
            & (
                (
                    rung_proposals["outcome"].map(lambda value: _beneficial(value, policy))
                    & rung_proposals["gt_text"].fillna("").str.strip().astype(bool)
                )
                | (rung_proposals["outcome"] == "TRUE_CORRECTION")
            )
        ]["site_id"]
    )

    rows: list[dict[str, object]] = []
    expected_families = (
        "correct_candidate_not_expressible",
        "expressible_but_not_generated",
        "generated_beyond_K",
        "alignment_ambiguity",
        "segmentation_failure",
        "language_or_model_issue",
        "normalization_mismatch",
        "source_illegibility",
        "historical_character_issue",
        "other",
    )
    family_for_class = {
        "repair_not_expressible_by_rung": "correct_candidate_not_expressible",
        "expressible_not_generated": "expressible_but_not_generated",
        "generated_beyond_K": "generated_beyond_K",
        "alignment_ambiguous_site": "alignment_ambiguity",
        "segmentation_failure": "segmentation_failure",
        "beneficial_absent_vocabulary": "language_or_model_issue",
        "normalization_mismatch": "normalization_mismatch",
        "source_illegibility": "source_illegibility",
        "historical_charset": "historical_character_issue",
        "other": "other",
    }
    error_sites = census[census["d_before"] > 0]
    for engine_id, engine_sites in error_sites.groupby("engine_id", sort=True):
        counts: dict[str, int] = {}
        lexicon = lexicon_by_engine.get(str(engine_id), set())
        for site in engine_sites.itertuples(index=False):
            site_id = str(site.site_id)
            if site_id in repairable_sites:
                continue
            kind = str(site.site_kind)
            gt = str(site.gt_text)
            tokens = gt.split()
            if kind not in expressible:
                label = "repair_not_expressible_by_rung"
            elif site_id in beyond_k:
                label = "generated_beyond_K"
            elif site_id in twin_ids or _at_confidence_floor(site):
                label = "alignment_ambiguous_site"
            elif str(site.dataset_id) == "ocrd_sbb" and "\u017f" in gt:
                label = "historical_charset"
            elif _normalization_equivalent(str(site.ocr_text), gt):
                label = "normalization_mismatch"
            elif kind == "substitution" and tokens and all(t in lexicon for t in tokens):
                label = "expressible_not_generated"
            elif tokens and not any(t in lexicon for t in tokens):
                label = "beneficial_absent_vocabulary"
            elif kind == "segmentation":
                label = "segmentation_failure"
            else:
                label = "other"
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        seen_families: set[str] = set()
        for label, n in sorted(counts.items()):
            family = family_for_class[label]
            seen_families.add(family)
            rows.append(
                {
                    "generator_id": rung,
                    "engine_id": engine_id,
                    "harm_policy": policy.value,
                    "failure_class": label,
                    "failure_family": family,
                    "n": n,
                    "share": n / total if total else float("nan"),
                    "classification_basis": "machine_rule",
                }
            )
        for family in expected_families:
            if family not in seen_families:
                rows.append(
                    {
                        "generator_id": rung,
                        "engine_id": engine_id,
                        "harm_policy": policy.value,
                        "failure_class": family,
                        "failure_family": family,
                        "n": 0,
                        "share": 0.0,
                        "classification_basis": (
                            "manual_audit_required"
                            if family == "source_illegibility"
                            else "machine_rule"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def failure_taxonomy_by_dataset_table(
    census: pd.DataFrame,
    proposals: pd.DataFrame,
    fold_lexicons: pd.DataFrame,
    rung: str = "g6_union",
    k: int = 4,
    twin_sites: pd.DataFrame | None = None,
    policy: HarmPolicy | str = HarmPolicy.STRICT_WORSENING,
) -> pd.DataFrame:
    """Failure taxonomy by corpus, preserving OCR-D as a labelled stress track."""
    tables: list[pd.DataFrame] = []
    for dataset_id, dataset_census in census.groupby("dataset_id", sort=True):
        site_ids = set(dataset_census["site_id"].astype(str))
        dataset_proposals = proposals[proposals["site_id"].astype(str).isin(site_ids)]
        dataset_twins = (
            twin_sites[twin_sites["site_id"].astype(str).isin(site_ids)]
            if twin_sites is not None and not twin_sites.empty
            else twin_sites
        )
        table = failure_taxonomy_table(
            dataset_census,
            dataset_proposals,
            fold_lexicons,
            rung,
            k,
            dataset_twins,
            policy,
        )
        if not table.empty:
            tables.append(table.assign(dataset_id=dataset_id))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


_ALIGN_CONFIDENCE_FLOOR = 0.30


def _at_confidence_floor(site: object) -> bool:
    confidence = getattr(site, "min_align_confidence", None)
    try:
        return confidence is not None and float(str(confidence)) <= _ALIGN_CONFIDENCE_FLOOR
    except ValueError:
        return False


def _normalization_equivalent(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    return left != right and normalize(left) == normalize(right)

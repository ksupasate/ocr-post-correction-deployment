"""Resampling and multiplicity.

The load-bearing test here is ``test_cluster_bootstrap_is_wider_than_naive``: it
demonstrates on correlated data that resampling edits instead of documents produces
intervals that are too narrow. Without it, someone could "simplify" the resampling unit
back to the edit and every test would still pass while every reported interval became
dishonest.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr_risk.stats import (
    benjamini_hochberg,
    cliffs_delta,
    cluster_bootstrap,
    cohens_h,
    group_by_cluster,
    holm_bonferroni,
    paired_cluster_bootstrap,
)


class Obs:
    """An observation with a document id and a value."""

    __slots__ = ("document_id", "value")

    def __init__(self, document_id: str, value: float) -> None:
        self.document_id = document_id
        self.value = value


def _mean(items) -> float:  # type: ignore[no-untyped-def]
    values = [o.value for o in items]
    return float(np.mean(values)) if values else 0.0


def _cluster(o: Obs) -> str:
    return o.document_id


# --- cluster bootstrap ------------------------------------------------------------------
def test_group_by_cluster_partitions() -> None:
    items = [Obs("a", 1.0), Obs("b", 2.0), Obs("a", 3.0)]
    groups = group_by_cluster(items, _cluster)
    assert set(groups) == {"a", "b"}
    assert len(groups["a"]) == 2


def test_estimate_matches_the_statistic_on_the_full_sample() -> None:
    items = [Obs("d1", 1.0), Obs("d1", 2.0), Obs("d2", 3.0)]
    result = cluster_bootstrap(items, _cluster, _mean, n_resamples=200, seed=1)
    assert result.estimate == pytest.approx(2.0)


def test_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(2)
    items = [Obs(f"d{i}", float(rng.normal(5.0, 1.0))) for i in range(40)]
    result = cluster_bootstrap(items, _cluster, _mean, n_resamples=500, seed=3)
    assert result.lower <= result.estimate <= result.upper
    assert result.n_clusters == 40


def test_bootstrap_is_deterministic_under_a_fixed_seed() -> None:
    items = [Obs(f"d{i % 5}", float(i)) for i in range(50)]
    a = cluster_bootstrap(items, _cluster, _mean, n_resamples=300, seed=11)
    b = cluster_bootstrap(items, _cluster, _mean, n_resamples=300, seed=11)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_cluster_bootstrap_is_wider_than_naive_on_correlated_data() -> None:
    """The reason the resampling unit is the document.

    Here every edit within a document shares that document's offset, so edits are
    strongly correlated. Resampling edits treats them as independent and shrinks the
    interval by roughly sqrt(edits per document) — turning a null result into a finding.
    """
    rng = np.random.default_rng(5)
    items: list[Obs] = []
    for document in range(12):
        # A large per-document effect and small within-document noise: exactly the
        # structure a page of OCR from one scan has.
        offset = float(rng.normal(0.0, 3.0))
        items.extend(Obs(f"d{document}", offset + float(rng.normal(0.0, 0.1))) for _ in range(60))

    clustered = cluster_bootstrap(items, _cluster, _mean, n_resamples=800, seed=7)
    # The naive alternative: pretend every edit is its own cluster.
    naive = cluster_bootstrap(items, lambda o: str(id(o)), _mean, n_resamples=800, seed=7)

    clustered_width = clustered.upper - clustered.lower
    naive_width = naive.upper - naive.lower
    assert clustered_width > 3 * naive_width, (
        f"document-level CI ({clustered_width:.4f}) is not materially wider than the "
        f"edit-level one ({naive_width:.4f}); the correlation structure is being ignored"
    )


def test_single_cluster_reports_nan_rather_than_false_certainty() -> None:
    """One document carries no information about between-document variability. A
    zero-width interval would claim certainty we do not have."""
    items = [Obs("only", float(v)) for v in range(20)]
    result = cluster_bootstrap(items, _cluster, _mean, n_resamples=100)
    assert np.isnan(result.lower)
    assert np.isnan(result.upper)
    assert result.n_clusters == 1


# --- paired bootstrap ---------------------------------------------------------------------
def test_paired_bootstrap_detects_a_consistent_difference() -> None:
    """A small but consistent per-document advantage must be detectable, which is what
    the pairing buys."""
    rng = np.random.default_rng(9)
    a: list[Obs] = []
    b: list[Obs] = []
    for document in range(25):
        base = float(rng.normal(0.0, 5.0))  # large between-document variance
        a.append(Obs(f"d{document}", base + 0.5))
        b.append(Obs(f"d{document}", base))
    result = paired_cluster_bootstrap(a, b, _cluster, _mean, n_resamples=800, seed=4)
    assert result.estimate == pytest.approx(0.5, abs=1e-9)
    assert result.excludes_zero, "pairing failed to cancel the shared per-document term"


def test_unpaired_bootstrap_would_miss_that_difference() -> None:
    """Contrast: without pairing, the between-document variance swamps the effect."""
    rng = np.random.default_rng(9)
    a: list[Obs] = []
    b: list[Obs] = []
    for document in range(25):
        base = float(rng.normal(0.0, 5.0))
        a.append(Obs(f"d{document}", base + 0.5))
        b.append(Obs(f"d{document}", base))
    independent_a = cluster_bootstrap(a, _cluster, _mean, n_resamples=800, seed=4)
    independent_b = cluster_bootstrap(b, _cluster, _mean, n_resamples=800, seed=4)
    unpaired_width = (independent_a.upper - independent_a.lower) + (
        independent_b.upper - independent_b.lower
    )
    paired = paired_cluster_bootstrap(a, b, _cluster, _mean, n_resamples=800, seed=4)
    assert (paired.upper - paired.lower) < unpaired_width


def test_paired_bootstrap_uses_only_shared_clusters() -> None:
    a = [Obs("d1", 1.0), Obs("d2", 1.0), Obs("only_in_a", 100.0)]
    b = [Obs("d1", 0.0), Obs("d2", 0.0)]
    result = paired_cluster_bootstrap(a, b, _cluster, _mean, n_resamples=200, seed=1)
    assert result.n_clusters == 2
    assert result.estimate == pytest.approx(1.0)


# --- multiplicity ----------------------------------------------------------------------------
def test_holm_adjusts_upward_and_stays_monotone() -> None:
    adjusted = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    values = [t.adjusted_p_value for t in adjusted]
    assert all(t.adjusted_p_value >= t.p_value for t in adjusted), "adjustment must not shrink"
    assert any(t.adjusted_p_value > t.p_value for t in adjusted), "adjustment did nothing"
    assert values == sorted(values), "adjusted p-values must be non-decreasing"


def test_holm_rejects_a_marginal_result_that_survives_alone() -> None:
    """The point of the correction: p = 0.04 is significant on its own and is not once
    it is one of four tests in a family."""
    alone = holm_bonferroni({"only": 0.04})
    assert alone[0].significant

    family = holm_bonferroni({"a": 0.04, "b": 0.30, "c": 0.40, "d": 0.60})
    assert not next(t for t in family if t.label == "a").significant


def test_holm_is_more_conservative_than_bh() -> None:
    p_values = {"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.03, "e": 0.04}
    holm = {t.label: t.adjusted_p_value for t in holm_bonferroni(p_values)}
    bh = {t.label: t.adjusted_p_value for t in benjamini_hochberg(p_values)}
    assert all(holm[label] >= bh[label] for label in p_values)


def test_empty_family_is_handled() -> None:
    assert holm_bonferroni({}) == []
    assert benjamini_hochberg({}) == []


def test_adjusted_p_values_never_exceed_one() -> None:
    adjusted = holm_bonferroni({f"t{i}": 0.9 for i in range(10)})
    assert all(t.adjusted_p_value <= 1.0 for t in adjusted)


# --- effect sizes -------------------------------------------------------------------------------
def test_cohens_h_is_zero_for_equal_proportions() -> None:
    assert cohens_h(0.3, 0.3) == pytest.approx(0.0)


def test_cohens_h_is_signed() -> None:
    assert cohens_h(0.5, 0.2) > 0
    assert cohens_h(0.2, 0.5) < 0


def test_cliffs_delta_bounds() -> None:
    assert cliffs_delta([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert cliffs_delta([1.0, 2.0], [3.0, 4.0]) == pytest.approx(-1.0)
    assert cliffs_delta([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_cliffs_delta_of_empty_input() -> None:
    assert cliffs_delta([], [1.0]) == 0.0


def test_index_bootstrap_matches_the_generic_path_exactly() -> None:
    """The fast path must be the same procedure, not merely a similar one.

    cluster_bootstrap_indices exists because the generic path spends its time unpacking
    Python tuples at pilot scale. A performance change to a headline interval is only
    acceptable if it is bit-for-bit the same estimate, so this asserts equality rather
    than closeness -- both draw the same cluster positions from the same seed.
    """
    import numpy as np

    from ocr_risk.stats import cluster_bootstrap, cluster_bootstrap_indices

    rng = np.random.default_rng(3)
    documents = [f"doc-{i // 7}" for i in range(140)]
    values = rng.normal(size=140)

    generic = cluster_bootstrap(
        list(zip(documents, values, strict=True)),
        lambda item: str(item[0]),
        lambda items: float(np.mean([v for _, v in items])) if items else 0.0,
        n_resamples=500,
        seed=11,
    )
    fast = cluster_bootstrap_indices(
        documents,
        lambda idx: float(np.mean(values[idx])) if idx.size else 0.0,
        n_resamples=500,
        seed=11,
    )

    assert fast.estimate == pytest.approx(generic.estimate)
    assert fast.lower == pytest.approx(generic.lower)
    assert fast.upper == pytest.approx(generic.upper)
    assert fast.standard_error == pytest.approx(generic.standard_error)
    assert fast.p_value_two_sided == pytest.approx(generic.p_value_two_sided)
    assert fast.n_clusters == generic.n_clusters


def test_index_bootstrap_widens_a_zero_variance_interval_too() -> None:
    """The rule-of-three fallback must not be lost on the fast path."""

    from ocr_risk.stats import cluster_bootstrap_indices

    documents = [f"doc-{i // 5}" for i in range(40)]
    result = cluster_bootstrap_indices(documents, lambda idx: 0.0, n_resamples=200, seed=1)
    assert result.upper > result.lower, "a zero-event interval must not claim certainty"
    assert result.upper == pytest.approx(3.0 / result.n_clusters)


def test_minimum_detectable_effect_scales_with_the_standard_error() -> None:
    """A null is only informative if the design could have seen the effect.

    Reported beside every non-significant result, because "we did not detect an effect"
    and "there is no effect" are different claims and a null whose detectable effect
    exceeds the quantity being compared supports only the first.
    """
    from ocr_risk.stats import minimum_detectable_effect

    # At alpha=0.05 and 80% power the multiplier is z(0.975) + z(0.80) = 1.960 + 0.842.
    assert minimum_detectable_effect(1.0) == pytest.approx(2.8016, abs=1e-3)
    assert minimum_detectable_effect(0.5) == pytest.approx(1.4008, abs=1e-3)


def test_minimum_detectable_effect_accounts_for_the_multiplicity_family() -> None:
    """Holm's smallest p must clear alpha/m, so that is the level a study is powered at."""
    from ocr_risk.stats import minimum_detectable_effect

    single = minimum_detectable_effect(1.0, n_tests=1)
    family_of_four = minimum_detectable_effect(1.0, n_tests=4)
    assert family_of_four > single, "correcting for four tests raises the detectable effect"


def test_minimum_detectable_effect_is_nan_for_a_degenerate_standard_error() -> None:
    import math

    from ocr_risk.stats import minimum_detectable_effect

    assert math.isnan(minimum_detectable_effect(0.0))
    assert math.isnan(minimum_detectable_effect(float("nan")))


def test_paired_bootstrap_refuses_a_degenerate_resampling_distribution() -> None:
    """A constant-score arm makes the paired difference independent of the draw.

    Every resample then returns the same number: zero sampling variability, a ~1e-16
    interval, and — because the bootstrap p falls to its floor 2/(B+1) when every draw is
    identical — MAXIMAL significance. Three rows of the shipped H1 table were flagged
    degraded on exactly that basis. The unpaired paths had a guard; this one did not.

    NaN rather than a widened interval: the unpaired statistic is a proportion and the
    rule of three is a real bound, but a paired difference of two arbitrary statistics has
    no such bound, and inventing one would be worse than admitting the resampling carries
    no information.
    """
    import math

    from ocr_risk.stats import paired_cluster_bootstrap

    documents = [f"doc-{i // 4}" for i in range(40)]
    # Arm A is constant at 1.0 and arm B at 0.5, so the difference is 0.5 whatever is drawn.
    items_a = [(d, 1.0) for d in documents]
    items_b = [(d, 0.5) for d in documents]
    result = paired_cluster_bootstrap(
        items_a,
        items_b,
        lambda item: str(item[0]),
        lambda items: float(sum(v for _, v in items) / len(items)) if items else 0.0,
        n_resamples=500,
        seed=3,
    )
    assert result.estimate == pytest.approx(0.5)
    assert math.isnan(result.lower) and math.isnan(result.upper)
    assert math.isnan(result.p_value_two_sided), "a degenerate draw must not report a p"
    assert result.standard_error == 0.0


def test_paired_bootstrap_still_returns_an_interval_when_the_draw_varies() -> None:
    import numpy as np

    from ocr_risk.stats import paired_cluster_bootstrap

    rng = np.random.default_rng(5)
    documents = [f"doc-{i // 4}" for i in range(80)]
    items_a = [(d, float(v)) for d, v in zip(documents, rng.normal(1.0, 1.0, 80), strict=True)]
    items_b = [(d, float(v)) for d, v in zip(documents, rng.normal(0.0, 1.0, 80), strict=True)]
    result = paired_cluster_bootstrap(
        items_a,
        items_b,
        lambda item: str(item[0]),
        lambda items: float(np.mean([v for _, v in items])) if items else 0.0,
        n_resamples=500,
        seed=3,
    )
    assert result.upper > result.lower
    assert 0.0 <= result.p_value_two_sided <= 1.0


def test_rule_of_three_widens_toward_the_interior_and_stays_in_range() -> None:
    """A proportion pinned at its maximum must not get an upper bound above 1.

    The original fallback always widened upward, because a collapsed interval has
    `estimate == lower` so the "saturated at the top" branch could never fire. A coverage
    of 1.0 or a risk of 1.0 therefore produced an upper bound of 1.15.
    """
    from ocr_risk.stats import cluster_bootstrap_indices

    documents = [f"doc-{i // 5}" for i in range(40)]

    saturated = cluster_bootstrap_indices(documents, lambda idx: 1.0, n_resamples=100, seed=1)
    assert saturated.upper == pytest.approx(1.0)
    assert saturated.lower == pytest.approx(1.0 - 3.0 / saturated.n_clusters)
    assert saturated.degenerate_interval

    zero_event = cluster_bootstrap_indices(documents, lambda idx: 0.0, n_resamples=100, seed=1)
    assert zero_event.lower == pytest.approx(0.0)
    assert zero_event.upper == pytest.approx(3.0 / zero_event.n_clusters)
    assert zero_event.degenerate_interval


def test_no_bound_is_invented_for_a_statistic_with_no_declared_range() -> None:
    """The rule of three is a binomial result. Grafting 3/n onto a calibration error or a
    paired difference would be dimensionally meaningless, so those callers get NaN."""
    import math

    from ocr_risk.stats import cluster_bootstrap_indices

    result = cluster_bootstrap_indices(
        [f"doc-{i // 5}" for i in range(40)],
        lambda idx: 0.42,
        n_resamples=100,
        seed=1,
        bounds=None,
    )
    assert math.isnan(result.lower) and math.isnan(result.upper)
    assert result.degenerate_interval, "still flagged, so the caller knows why it is NaN"


def test_a_resampled_interval_is_not_flagged_degenerate() -> None:
    """The flag exists to tell grafted endpoints from measured ones; it must not overfire."""
    import numpy as np

    from ocr_risk.stats import cluster_bootstrap_indices

    rng = np.random.default_rng(2)
    values = rng.uniform(0, 1, 120)
    result = cluster_bootstrap_indices(
        [f"doc-{i // 6}" for i in range(120)],
        lambda idx: float(values[idx].mean()),
        n_resamples=300,
        seed=1,
    )
    assert not result.degenerate_interval
    assert result.upper > result.lower


def test_multi_paired_bootstrap_matches_the_scalar_one_exactly() -> None:
    """Five metrics from one resampling pass must be the same numbers as five passes.

    The H1 contrast evaluates five calibration quantities that all fall out of a single
    calibration_report. Bootstrapping them separately recomputed it five times per resample
    and discarded four fifths each time, which put the full sensitivity analysis at ~3
    hours. Sharing the pass is only acceptable if it is bit-for-bit the same result.
    """
    import numpy as np

    from ocr_risk.stats import paired_cluster_bootstrap, paired_cluster_bootstrap_multi

    rng = np.random.default_rng(9)
    documents = [f"doc-{i // 5}" for i in range(100)]
    values_a = rng.normal(1.0, 1.0, 100)
    values_b = rng.normal(0.0, 1.0, 100)
    items_a = [(d, float(v)) for d, v in zip(documents, values_a, strict=True)]
    items_b = [(d, float(v)) for d, v in zip(documents, values_b, strict=True)]

    def both(items: list[tuple[str, float]]) -> dict[str, float]:
        arr = np.array([v for _, v in items]) if items else np.array([0.0])
        return {"mean": float(arr.mean()), "spread": float(arr.std())}

    shared = paired_cluster_bootstrap_multi(
        items_a, items_b, lambda item: str(item[0]), both, n_resamples=400, seed=21
    )
    for key in ("mean", "spread"):
        separate = paired_cluster_bootstrap(
            items_a,
            items_b,
            lambda item: str(item[0]),
            lambda items, k=key: both(list(items))[k],
            n_resamples=400,
            seed=21,
        )
        assert shared[key].estimate == pytest.approx(separate.estimate)
        assert shared[key].lower == pytest.approx(separate.lower)
        assert shared[key].upper == pytest.approx(separate.upper)
        assert shared[key].standard_error == pytest.approx(separate.standard_error)
        assert shared[key].p_value_two_sided == pytest.approx(separate.p_value_two_sided)


def test_multi_paired_bootstrap_flags_a_degenerate_metric_independently() -> None:
    """One constant metric must not poison the interval of a varying one."""
    import math

    import numpy as np

    from ocr_risk.stats import paired_cluster_bootstrap_multi

    rng = np.random.default_rng(4)
    documents = [f"doc-{i // 5}" for i in range(60)]
    noise = rng.normal(0.0, 1.0, 60)
    items_a = [(d, float(v)) for d, v in zip(documents, noise, strict=True)]
    items_b = [(d, 0.0) for d in documents]

    def metrics(items: list[tuple[str, float]]) -> dict[str, float]:
        arr = np.array([v for _, v in items]) if items else np.array([0.0])
        return {"varying": float(arr.mean()), "constant": 1.0}

    results = paired_cluster_bootstrap_multi(
        items_a, items_b, lambda item: str(item[0]), metrics, n_resamples=300, seed=2
    )
    assert results["constant"].degenerate_interval
    assert math.isnan(results["constant"].lower)
    assert not results["varying"].degenerate_interval
    assert results["varying"].upper > results["varying"].lower

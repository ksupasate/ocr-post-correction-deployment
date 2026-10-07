"""Calibration and risk control.

Two tests here are statistical validity checks rather than unit tests, and they are the
reason this module exists:

- ``test_ltt_controls_risk_under_exchangeability`` runs a Monte-Carlo simulation and
  asserts the guarantee actually holds. Implementing a bound is not the same as the bound
  being correct.
- ``test_empirical_controller_is_optimistically_biased`` demonstrates the failure the
  guarantee exists to prevent, so the difference between the two controllers is measured
  rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr_risk.calibrate import (
    NotFittedCalibratorError,
    TemperatureCalibrator,
    build_calibrator,
)
from ocr_risk.metrics import brier_score, calibration_report
from ocr_risk.risk import (
    ScoredCandidate,
    bentkus_p_value,
    decide_sites,
    hoeffding_p_value,
    risk_upper_bound,
    select_threshold,
)

CALIBRATORS = ("identity", "platt", "isotonic", "temperature", "beta")


def _miscalibrated(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that rank well but are systematically overconfident."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, size=n)
    outcomes = (rng.uniform(size=n) < truth).astype(np.float64)
    # A monotone distortion: ranking is preserved, probabilities are not.
    scores = np.clip(truth**0.4, 0.0, 1.0)
    return scores, outcomes


# --- calibrators ------------------------------------------------------------------------
@pytest.mark.parametrize("method", CALIBRATORS)
def test_calibrator_must_be_fitted_first(method: str) -> None:
    """Fitting at transform time would use the data being evaluated (vector L2)."""
    calibrator = build_calibrator(method)
    with pytest.raises(NotFittedCalibratorError, match="before fit"):
        calibrator.transform(np.array([0.5]))


@pytest.mark.parametrize("method", CALIBRATORS)
def test_calibrator_outputs_are_probabilities(method: str) -> None:
    scores, outcomes = _miscalibrated(500)
    calibrator = build_calibrator(method)
    calibrator.fit(scores, outcomes)
    out = calibrator.transform(scores)
    assert out.shape == scores.shape
    assert np.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.parametrize("method", ["platt", "isotonic", "temperature", "beta"])
def test_calibration_improves_brier_on_miscalibrated_scores(method: str) -> None:
    """The whole point: a calibrator must make the probabilities better."""
    scores, outcomes = _miscalibrated()
    calibrator = build_calibrator(method)
    calibrator.fit(scores, outcomes)
    calibrated = calibrator.transform(scores)
    assert brier_score(calibrated, outcomes) < brier_score(scores, outcomes)


@pytest.mark.parametrize("method", ["platt", "isotonic", "temperature", "beta"])
def test_calibration_reduces_expected_calibration_error(method: str) -> None:
    scores, outcomes = _miscalibrated()
    calibrator = build_calibrator(method)
    calibrator.fit(scores, outcomes)
    before = calibration_report(scores, outcomes, n_bins=10)
    after = calibration_report(calibrator.transform(scores), outcomes, n_bins=10)
    assert after.ece_equal_mass < before.ece_equal_mass


@pytest.mark.parametrize("method", CALIBRATORS)
def test_calibration_preserves_ranking(method: str) -> None:
    """Every calibrator here is monotone, so it changes the values on a risk-coverage
    curve but not the ordering. That keeps 'the probabilities were wrong' separable from
    'the verifier was wrong', which H1 needs."""
    scores, outcomes = _miscalibrated(800)
    calibrator = build_calibrator(method)
    calibrator.fit(scores, outcomes)
    calibrated = calibrator.transform(scores)

    order_before = np.argsort(scores, kind="stable")
    calibrated_in_order = calibrated[order_before]
    assert np.all(np.diff(calibrated_in_order) >= -1e-9), f"{method} reordered the scores"


def test_temperature_recovers_a_known_distortion() -> None:
    """Inject a known temperature and check it is estimated back — a direct test that the
    fit is doing what it claims, not merely reducing a loss."""
    rng = np.random.default_rng(3)
    n = 20_000
    logits = rng.normal(0.0, 2.0, size=n)
    true_probabilities = 1.0 / (1.0 + np.exp(-logits))
    outcomes = (rng.uniform(size=n) < true_probabilities).astype(np.float64)

    known_temperature = 2.5
    distorted = 1.0 / (1.0 + np.exp(-logits / known_temperature))

    calibrator = TemperatureCalibrator()
    calibrator.fit(distorted, outcomes)
    # Recovering T means mapping the distorted scores back, i.e. dividing by 1/T.
    assert calibrator.temperature == pytest.approx(1.0 / known_temperature, rel=0.15)


def test_identity_calibrator_changes_nothing() -> None:
    scores, outcomes = _miscalibrated(200)
    calibrator = build_calibrator("identity")
    calibrator.fit(scores, outcomes)
    assert np.allclose(calibrator.transform(scores), scores)


@pytest.mark.parametrize("method", CALIBRATORS)
def test_single_class_calibration_falls_back_to_the_base_rate(method: str) -> None:
    calibrator = build_calibrator(method)
    scores = np.linspace(0.1, 0.9, 20)
    calibrator.fit(scores, np.ones(20))
    out = calibrator.transform(scores)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", CALIBRATORS)
def test_empty_calibration_set_is_survivable(method: str) -> None:
    calibrator = build_calibrator(method)
    calibrator.fit(np.array([]), np.array([]))
    assert np.all(np.isfinite(calibrator.transform(np.array([0.3, 0.7]))))


@pytest.mark.parametrize("method", CALIBRATORS)
def test_calibrator_identity_is_stable_and_specific(method: str) -> None:
    """The id is recorded on every prediction, so it must change when the fit changes."""
    a = build_calibrator(method)
    b = build_calibrator(method)
    scores, outcomes = _miscalibrated(400, seed=1)
    a.fit(scores, outcomes)
    b.fit(scores, outcomes)
    assert a.identity() == b.identity()
    assert a.identity().startswith(method)

    c = build_calibrator(method)
    c.fit(*_miscalibrated(400, seed=2))
    if method != "identity":
        assert c.identity() != a.identity()


def test_calibrator_rejects_shape_mismatch() -> None:
    calibrator = build_calibrator("platt")
    with pytest.raises(ValueError, match="shape mismatch"):
        calibrator.fit(np.array([0.1, 0.2]), np.array([1.0]))


def test_unknown_calibration_method_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown calibration method"):
        build_calibrator("magic")


# --- bounds ---------------------------------------------------------------------------------
def test_p_value_is_one_when_the_observation_exceeds_the_null() -> None:
    assert hoeffding_p_value(100, 0.2, 0.1) == 1.0
    assert bentkus_p_value(100, 0.2, 0.1) == 1.0


def test_p_values_shrink_with_more_evidence() -> None:
    small = hoeffding_p_value(50, 0.01, 0.1)
    large = hoeffding_p_value(5000, 0.01, 0.1)
    assert large < small


def test_bentkus_is_tighter_in_the_small_risk_regime() -> None:
    """The regime this project operates in: a 1% tolerance, where Hoeffding's
    variance-free bound needs far more data than a per-fold split has."""
    n, observed, null = 500, 0.002, 0.01
    assert bentkus_p_value(n, observed, null) < hoeffding_p_value(n, observed, null)


def test_upper_bound_exceeds_the_observed_risk() -> None:
    assert risk_upper_bound(5, 100, delta=0.1) > 0.05


def test_upper_bound_tightens_with_sample_size() -> None:
    assert risk_upper_bound(50, 1000, 0.1) < risk_upper_bound(5, 100, 0.1)


def test_upper_bound_with_no_data_rules_nothing_out() -> None:
    assert risk_upper_bound(0, 0) == 1.0


# --- controllers ------------------------------------------------------------------------------
def _risky(n: int, base_risk: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores that carry real signal: harmful edits score lower on average."""
    rng = np.random.default_rng(seed)
    harmful = rng.uniform(size=n) < base_risk
    scores = np.where(harmful, rng.beta(2, 6, size=n), rng.beta(6, 2, size=n))
    return scores, harmful


def test_empirical_controller_finds_a_threshold() -> None:
    scores, harmful = _risky(2000, 0.2, seed=1)
    decision = select_threshold(scores, harmful, epsilon=0.05, controller="empirical")
    assert decision.feasible
    assert decision.observed_risk <= 0.05
    assert decision.coverage > 0.0


def test_ltt_is_more_conservative_than_empirical() -> None:
    """The price of the guarantee, and it should be visible."""
    scores, harmful = _risky(1500, 0.25, seed=2)
    empirical = select_threshold(scores, harmful, epsilon=0.05, controller="empirical")
    ltt = select_threshold(scores, harmful, epsilon=0.05, controller="ltt_bentkus")
    if ltt.feasible and empirical.feasible:
        assert ltt.coverage <= empirical.coverage


def test_controller_abstains_rather_than_exceeding_the_tolerance() -> None:
    """Zero coverage at zero risk is a valid operating point. Returning the least-bad
    threshold instead would silently break the promise the experiment made."""
    rng = np.random.default_rng(4)
    scores = rng.uniform(size=300)
    harmful = np.ones(300, dtype=bool)  # everything is harmful
    decision = select_threshold(scores, harmful, epsilon=0.01, controller="ltt_bentkus")
    assert not decision.feasible
    assert decision.n_accepted == 0
    assert "abstaining" in decision.note


def test_empty_calibration_set_abstains() -> None:
    decision = select_threshold(np.array([]), np.array([], dtype=bool), epsilon=0.05)
    assert not decision.feasible
    assert decision.n_accepted == 0


def test_controller_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        select_threshold(np.array([0.1, 0.2]), np.array([True]), epsilon=0.1)


def test_unknown_controller_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown risk controller"):
        select_threshold(np.array([0.5]), np.array([False]), epsilon=0.1, controller="magic")


@pytest.mark.slow
def test_ltt_controls_risk_under_exchangeability() -> None:
    """Monte-Carlo validity check.

    Fit a threshold on one exchangeable sample, apply it to a fresh one, and count how
    often realized risk exceeds epsilon. The guarantee says at most delta of the time.
    Implementing a bound is not the same as the bound being correct, and this is the only
    test that distinguishes the two.
    """
    epsilon, delta, trials = 0.10, 0.10, 200
    violations = 0
    for trial in range(trials):
        calibration_scores, calibration_harmful = _risky(600, 0.3, seed=1000 + trial)
        decision = select_threshold(
            calibration_scores, calibration_harmful, epsilon=epsilon, delta=delta
        )
        if not decision.feasible:
            continue  # abstaining never violates the bound

        test_scores, test_harmful = _risky(600, 0.3, seed=9000 + trial)
        accepted = test_scores >= decision.tau
        if not np.any(accepted):
            continue
        realized = float(np.count_nonzero(test_harmful & accepted) / np.count_nonzero(accepted))
        if realized > epsilon:
            violations += 1

    violation_rate = violations / trials
    # Allowance above delta for Monte-Carlo error at 200 trials.
    assert violation_rate <= delta + 0.06, (
        f"LTT violated its own guarantee in {violation_rate:.1%} of trials "
        f"(nominal delta={delta}); the bound is not being applied correctly"
    )


@pytest.mark.slow
def test_empirical_controller_is_optimistically_biased() -> None:
    """The failure the guarantee exists to prevent.

    A threshold selected on the same sample it is measured on overfits that sample, so
    realized risk on fresh data exceeds epsilon far more often than delta would allow.
    Measuring the gap is what justifies paying for LTT's conservatism.
    """
    epsilon, trials = 0.10, 200
    empirical_violations = 0
    ltt_violations = 0

    for trial in range(trials):
        calibration_scores, calibration_harmful = _risky(400, 0.3, seed=2000 + trial)
        test_scores, test_harmful = _risky(400, 0.3, seed=7000 + trial)

        for controller, counter in (("empirical", "e"), ("ltt_bentkus", "l")):
            decision = select_threshold(
                calibration_scores,
                calibration_harmful,
                epsilon=epsilon,
                delta=0.1,
                controller=controller,
            )
            if not decision.feasible:
                continue
            accepted = test_scores >= decision.tau
            if not np.any(accepted):
                continue
            realized = float(np.count_nonzero(test_harmful & accepted) / np.count_nonzero(accepted))
            if realized > epsilon:
                if counter == "e":
                    empirical_violations += 1
                else:
                    ltt_violations += 1

    assert empirical_violations > ltt_violations, (
        f"the naive controller violated the tolerance {empirical_violations} times and LTT "
        f"{ltt_violations}; if these are equal the guarantee is buying nothing and the "
        "extra machinery is unjustified"
    )


# --- site policy ---------------------------------------------------------------------------------
def _candidates() -> list[ScoredCandidate]:
    return [
        ScoredCandidate("c1", "s1", "d1", 0.9, generator_rank=1),
        ScoredCandidate("c2", "s1", "d1", 0.4, generator_rank=0),
        ScoredCandidate("c3", "s2", "d1", 0.2, generator_rank=0),
    ]


def test_at_most_one_candidate_is_accepted_per_site() -> None:
    """Two accepted edits at one site would conflict textually and double-count a single
    repair in both coverage and harm."""
    decisions = decide_sites(_candidates(), tau=0.1)
    assert len(decisions) == 2
    assert all(d.accepted_candidate_id is not None for d in decisions)
    assert {d.site_id for d in decisions} == {"s1", "s2"}


def test_argmax_policy_takes_the_best_scoring_candidate() -> None:
    decisions = {d.site_id: d for d in decide_sites(_candidates(), tau=0.5)}
    assert decisions["s1"].accepted_candidate_id == "c1"
    assert decisions["s2"].accepted_candidate_id is None


def test_top1_policy_considers_only_the_generators_first_choice() -> None:
    decisions = {
        d.site_id: d for d in decide_sites(_candidates(), tau=0.5, policy="top1_above_tau")
    }
    # c2 is rank 0 but scores 0.4, below tau, so the site is preserved even though a
    # higher-scoring alternative exists.
    assert decisions["s1"].accepted_candidate_id is None


def test_none_policy_preserves_everything() -> None:
    decisions = decide_sites(_candidates(), tau=0.0, policy="none")
    assert all(not d.accepted for d in decisions)


def test_threshold_above_every_score_accepts_nothing() -> None:
    decisions = decide_sites(_candidates(), tau=1.5)
    assert all(not d.accepted for d in decisions)


def test_unknown_site_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown site policy"):
        decide_sites(_candidates(), tau=0.5, policy="magic")


def test_decisions_are_deterministic_under_ties() -> None:
    tied = [
        ScoredCandidate("b", "s", "d", 0.7),
        ScoredCandidate("a", "s", "d", 0.7),
    ]
    first = decide_sites(tied, tau=0.5)[0].accepted_candidate_id
    second = decide_sites(list(reversed(tied)), tau=0.5)[0].accepted_candidate_id
    assert first == second


# --------------------------------------------------- the exact Bernoulli upper bound


def test_clopper_pearson_with_no_data_rules_nothing_out() -> None:
    from ocr_risk.risk import clopper_pearson_upper

    assert clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert clopper_pearson_upper(3, 0, 0.05) == 1.0


def test_clopper_pearson_saturates_when_everything_is_harmful() -> None:
    from ocr_risk.risk import clopper_pearson_upper

    assert clopper_pearson_upper(5, 5, 0.05) == 1.0
    assert clopper_pearson_upper(6, 5, 0.05) == 1.0


def test_clopper_pearson_matches_the_closed_form_at_zero_successes() -> None:
    """With no observed harm the exact bound is ``1 - delta ** (1 / n)``, in closed form."""
    from ocr_risk.risk import clopper_pearson_upper

    for n in (5, 25, 100, 400):
        expected = 1.0 - 0.05 ** (1.0 / n)
        assert clopper_pearson_upper(0, n, 0.05) == pytest.approx(expected, rel=1e-9)


def test_clopper_pearson_is_tighter_than_bentkus_for_a_bernoulli() -> None:
    """Bentkus pays a factor of e for holding over any bounded loss; the exact bound does not."""
    from ocr_risk.risk import clopper_pearson_upper, risk_upper_bound

    for n, k in ((25, 0), (50, 1), (100, 2), (250, 5)):
        assert clopper_pearson_upper(k, n, 0.05) < risk_upper_bound(k, n, 0.05)


def test_clopper_pearson_covers_the_true_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monte-Carlo coverage: the bound must fail no more often than delta."""
    import numpy as np

    from ocr_risk.risk import clopper_pearson_upper

    generator = np.random.default_rng(20260907)
    truth, n, trials, delta = 0.08, 60, 4000, 0.05
    draws = generator.binomial(n, truth, size=trials)
    misses = sum(1 for k in draws.tolist() if clopper_pearson_upper(int(k), n, delta) < truth)
    assert misses / trials <= delta


def test_clopper_pearson_is_monotone_in_the_observed_count() -> None:
    from ocr_risk.risk import clopper_pearson_upper

    bounds = [clopper_pearson_upper(k, 100, 0.05) for k in range(0, 20)]
    assert bounds == sorted(bounds)


# --------------------------------------------------- the pre-registered cut grid


def test_a_supplied_grid_replaces_the_score_quantiles() -> None:
    """The controller must search the declared family, not the observed quantiles."""
    import numpy as np

    from ocr_risk.risk import select_threshold

    scores = np.linspace(0.0, 1.0, 200)
    harmful = scores < 0.5
    decision = select_threshold(
        scores, harmful, epsilon=0.10, controller="empirical", thresholds=[0.9, 0.95]
    )
    assert decision.tau in {0.9, 0.95}
    assert decision.feasible


def test_a_smaller_grid_buys_a_wider_per_test_level() -> None:
    """Multiplicity is over whatever grid is searched, so a small family certifies more."""
    import numpy as np

    generator = np.random.default_rng(7)
    scores = generator.normal(size=3000)
    harmful = generator.random(3000) < 1.0 / (1.0 + np.exp(3.0 * scores + 1.5))
    grid = np.quantile(scores, [1.0 - q for q in (0.05, 0.10, 0.20, 0.30)])
    coarse = select_threshold(
        scores,
        harmful,
        epsilon=0.10,
        delta=0.05,
        controller="ltt_clopper_pearson",
        thresholds=grid,
    )
    fine = select_threshold(
        scores, harmful, epsilon=0.10, delta=0.05, controller="ltt_clopper_pearson", n_grid=200
    )
    assert coarse.feasible
    assert coarse.risk_upper_bound <= fine.risk_upper_bound or fine.n_accepted <= coarse.n_accepted


def test_the_clopper_pearson_controller_is_registered_and_conservative() -> None:
    import numpy as np

    from ocr_risk.risk import CONTROLLERS, select_threshold

    assert "ltt_clopper_pearson" in CONTROLLERS
    generator = np.random.default_rng(11)
    scores = generator.normal(size=1500)
    harmful = generator.random(1500) < 1.0 / (1.0 + np.exp(3.0 * scores + 1.5))
    empirical = select_threshold(scores, harmful, epsilon=0.10, controller="empirical")
    certified = select_threshold(
        scores, harmful, epsilon=0.10, delta=0.05, controller="ltt_clopper_pearson"
    )
    assert certified.n_accepted <= empirical.n_accepted
    assert certified.risk_upper_bound >= certified.observed_risk


# --------------------------------------------------- the informative Bayesian bound


def test_beta_binomial_needs_a_prior_and_refuses_to_invent_one() -> None:
    import numpy as np

    from ocr_risk.risk import select_threshold

    scores = np.linspace(0.0, 1.0, 50)
    harmful = scores < 0.2
    with pytest.raises(ValueError, match="needs a prior"):
        select_threshold(scores, harmful, epsilon=0.10, controller="ltt_beta_binomial")


def test_beta_binomial_reduces_to_the_posterior_quantile() -> None:
    from scipy.stats import beta as beta_distribution

    from ocr_risk.risk import beta_binomial_upper

    bound = beta_binomial_upper(2, 40, prior_harmful=1.0, prior_total=20.0, delta=0.05)
    expected = beta_distribution.ppf(0.95, 1.0 + 2, (20.0 - 1.0) + (40 - 2))
    assert bound == pytest.approx(float(expected), rel=1e-9)


def test_beta_binomial_with_no_data_and_no_prior_rules_nothing_out() -> None:
    from ocr_risk.risk import beta_binomial_upper

    assert beta_binomial_upper(0, 0, 0.0, 0.0, 0.05) == 1.0
    assert beta_binomial_upper(0, 0, 0.0, -1.0, 0.05) == 1.0


def test_an_optimistic_prior_makes_the_bayesian_bound_anti_conservative() -> None:
    """The point of carrying it: a prior from a shifted domain understates the true risk."""
    from ocr_risk.risk import beta_binomial_upper, clopper_pearson_upper

    optimistic = beta_binomial_upper(3, 30, prior_harmful=0.2, prior_total=20.0, delta=0.05)
    exact = clopper_pearson_upper(3, 30, 0.05)
    assert optimistic < exact


def test_the_beta_binomial_controller_places_a_cut() -> None:
    import numpy as np

    from ocr_risk.risk import select_threshold

    generator = np.random.default_rng(3)
    scores = generator.normal(size=800)
    harmful = generator.random(800) < 1.0 / (1.0 + np.exp(3.0 * scores + 1.5))
    decision = select_threshold(
        scores,
        harmful,
        epsilon=0.10,
        delta=0.05,
        controller="ltt_beta_binomial",
        prior=(1.0, 20.0),
    )
    assert decision.controller == "ltt_beta_binomial"
    assert decision.risk_upper_bound >= decision.observed_risk


def test_beta_binomial_refuses_a_degenerate_posterior() -> None:
    """A posterior with no mass on one side is not a bound; it rules nothing out.

    Both degenerate shapes are reachable with real data: a prior worth zero observations makes
    ``alpha`` zero when nothing harmful was seen, and a prior that is entirely harmful makes the
    other shape zero when everything observed was harmful too.
    """
    from ocr_risk.risk import beta_binomial_upper

    assert beta_binomial_upper(0, 5, prior_harmful=0.0, prior_total=0.0, delta=0.05) == 1.0
    assert beta_binomial_upper(5, 5, prior_harmful=1.0, prior_total=1.0, delta=0.05) == 1.0

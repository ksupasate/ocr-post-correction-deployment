"""Probability calibrators.

Calibration is a **separate, separately-frozen artifact** from the model that produced the
scores. Keeping them apart is what makes the few-shot protocol expressible at all — weights
stay frozen while the calibrator is refit — and what lets an auditor check that a reported
probability was actually calibrated, and by which transform.

Every calibrator here is monotone in the score, so calibration changes the *values* on the
risk-coverage curve but not the *ranking*. That is a deliberate constraint: a
rank-changing calibrator would conflate "the probabilities were wrong" with "the verifier
was wrong", and H1 needs those separable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ocr_risk.io.hashing import canonical_hash

__all__ = [
    "BetaCalibrator",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "NotFittedCalibratorError",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "build_calibrator",
]

FloatArray = NDArray[np.float64]
_EPS = 1e-6


class NotFittedCalibratorError(RuntimeError):
    """Raised when a calibrator transforms before being fitted on the calibration split."""


@runtime_checkable
class Calibrator(Protocol):
    """Maps raw verifier scores onto calibrated probabilities."""

    method: str

    def fit(self, scores: FloatArray, positive: FloatArray) -> None:
        """Fit on the calibration split only.

        ``positive`` is the event being predicted — here, that accepting the edit is
        *safe*. Orientation matches the verifier's score, so higher means safer throughout.
        """
        ...

    def transform(self, scores: FloatArray) -> FloatArray: ...

    def identity(self) -> str:
        """A stable id recorded on every prediction, so the transform is traceable."""
        ...


@dataclass(slots=True)
class _BaseCalibrator:
    method: str = "identity"
    _fitted: bool = False
    _params: dict[str, Any] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def identity(self) -> str:
        return f"{self.method}:{canonical_hash(self._params)[:12]}"

    def _require_fitted(self) -> None:
        if not self._fitted:
            msg = (
                f"calibrator {self.method!r} used before fit(). Calibration must be fitted "
                "on the calibration split; fitting it here would use the data being "
                "evaluated (leakage vector L2)."
            )
            raise NotFittedCalibratorError(msg)

    @staticmethod
    def _check(scores: FloatArray, positive: FloatArray) -> tuple[FloatArray, FloatArray]:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        positive = np.asarray(positive, dtype=np.float64).ravel()
        if scores.shape != positive.shape:
            msg = f"shape mismatch: {scores.shape} scores vs {positive.shape} outcomes"
            raise ValueError(msg)
        return scores, positive


@dataclass(slots=True)
class IdentityCalibrator(_BaseCalibrator):
    """No transform. The control condition for "did calibration help at all?"."""

    method: str = "identity"

    def fit(self, scores: FloatArray, positive: FloatArray) -> None:
        self._check(scores, positive)
        self._params = {"method": "identity"}
        self._fitted = True

    def transform(self, scores: FloatArray) -> FloatArray:
        self._require_fitted()
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


@dataclass(slots=True)
class PlattCalibrator(_BaseCalibrator):
    """Logistic (Platt) scaling: a two-parameter sigmoid fitted to the scores.

    Parametric and low-variance, which is the usual argument for preferring it over
    isotonic on a few hundred calibration points. That argument does not automatically
    win here: with a heavily imbalanced pool the sigmoid smooths away the small
    high-scoring group that risk control needs to isolate, and on this project's synthetic
    corpus isotonic certified strictly more operating points. Which calibrator to use is
    therefore an empirical question per corpus, not a default to assume — hence all five
    are available and the choice is configured and recorded.
    """

    method: str = "platt"
    _model: LogisticRegression | None = None
    _constant: float | None = None

    def fit(self, scores: FloatArray, positive: FloatArray) -> None:
        scores, positive = self._check(scores, positive)
        labels = (positive > 0.5).astype(np.int32)
        if labels.size == 0 or len(np.unique(labels)) < 2:
            # One class only: the base rate is the best available estimate, and a fitted
            # sigmoid would encode an accident of the split.
            self._constant = float(labels.mean()) if labels.size else 0.5
            self._model = None
        else:
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(
                scores.reshape(-1, 1), labels
            )
            self._constant = None
        self._params = {
            "method": "platt",
            "n": int(scores.size),
            "coef": None if self._model is None else float(self._model.coef_[0][0]),
            "intercept": None if self._model is None else float(self._model.intercept_[0]),
            "constant": self._constant,
        }
        self._fitted = True

    def transform(self, scores: FloatArray) -> FloatArray:
        self._require_fitted()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if self._model is None:
            return np.full(scores.shape, self._constant if self._constant is not None else 0.5)
        probabilities: FloatArray = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.clip(probabilities, 0.0, 1.0)


@dataclass(slots=True)
class IsotonicCalibrator(_BaseCalibrator):
    """Isotonic regression: non-parametric and monotone.

    More flexible than Platt and correspondingly hungrier for data — with a small
    calibration split it will fit noise. In exchange it can represent a sharp step that a
    sigmoid cannot, which matters when the safe candidates are a small high-scoring
    minority: that is exactly the group a risk controller must isolate. Report it against
    Platt rather than choosing between them on principle.
    """

    method: str = "isotonic"
    _model: IsotonicRegression | None = None
    _constant: float | None = None

    def fit(self, scores: FloatArray, positive: FloatArray) -> None:
        scores, positive = self._check(scores, positive)
        if scores.size == 0 or len(np.unique(positive > 0.5)) < 2:
            self._constant = float((positive > 0.5).mean()) if positive.size else 0.5
            self._model = None
        else:
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                scores, (positive > 0.5).astype(np.float64)
            )
            self._constant = None
        self._params = {
            "method": "isotonic",
            "n": int(scores.size),
            "constant": self._constant,
            # The fitted step function is the parameter set; hashing its knots makes the
            # transform identifiable in the run record.
            "knots": None
            if self._model is None
            else canonical_hash(
                [self._model.X_thresholds_.tolist(), self._model.y_thresholds_.tolist()]
            ),
        }
        self._fitted = True

    def transform(self, scores: FloatArray) -> FloatArray:
        self._require_fitted()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if self._model is None:
            return np.full(scores.shape, self._constant if self._constant is not None else 0.5)
        return np.clip(np.asarray(self._model.predict(scores), dtype=np.float64), 0.0, 1.0)


@dataclass(slots=True)
class TemperatureCalibrator(_BaseCalibrator):
    """Single-parameter temperature scaling on the score's logit.

    The minimal correction: it can fix systematic over- or under-confidence and nothing
    else. That makes it the sharpest instrument for H1 — if a temperature fitted on one
    engine fails on another, the miscalibration is not a simple confidence offset.
    """

    method: str = "temperature"
    _temperature: float = 1.0

    def fit(self, scores: FloatArray, positive: FloatArray) -> None:
        scores, positive = self._check(scores, positive)
        labels = (positive > 0.5).astype(np.float64)
        if scores.size == 0 or len(np.unique(labels)) < 2:
            self._temperature = 1.0
        else:
            logits = _logit(np.clip(scores, _EPS, 1.0 - _EPS))

            def negative_log_likelihood(log_temperature: float) -> float:
                temperature = float(np.exp(log_temperature))
                p = 1.0 / (1.0 + np.exp(-logits / temperature))
                p = np.clip(p, _EPS, 1.0 - _EPS)
                return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

            # Optimize in log space so the temperature stays positive by construction.
            result = minimize_scalar(negative_log_likelihood, bounds=(-4.0, 4.0), method="bounded")
            self._temperature = float(np.exp(result.x))
        self._params = {"method": "temperature", "temperature": self._temperature}
        self._fitted = True

    @property
    def temperature(self) -> float:
        return self._temperature

    def transform(self, scores: FloatArray) -> FloatArray:
        self._require_fitted()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        logits = _logit(np.clip(scores, _EPS, 1.0 - _EPS))
        return np.clip(1.0 / (1.0 + np.exp(-logits / self._temperature)), 0.0, 1.0)


@dataclass(slots=True)
class BetaCalibrator(_BaseCalibrator):
    """Beta calibration: a three-parameter family that Platt cannot express.

    Fits ``sigmoid(a*log p - b*log(1-p) + c)``, which allows asymmetric corrections —
    useful when a verifier is well calibrated at one end of the range and not the other,
    a pattern a single sigmoid cannot represent.

    ``a`` and ``b`` are constrained to be **non-negative**, as Kull et al. require. An
    unconstrained fit can return a negative coefficient, which makes the map non-monotone
    and reorders the risk-coverage curve: the calibrator would then change *which* edits
    are accepted, not just what probability they carry, conflating "the probabilities
    were wrong" with "the verifier was wrong". H1 needs those separable.
    """

    method: str = "beta"
    _coefficients: tuple[float, float, float] | None = None
    _constant: float | None = None

    def fit(self, scores: FloatArray, positive: FloatArray) -> None:
        scores, positive = self._check(scores, positive)
        labels = (positive > 0.5).astype(np.float64)
        if scores.size == 0 or len(np.unique(labels)) < 2:
            self._constant = float(labels.mean()) if labels.size else 0.5
            self._coefficients = None
        else:
            design = _beta_design(scores)

            def negative_log_likelihood(theta: FloatArray) -> float:
                logits = design @ theta[:2] + theta[2]
                # log(1 + exp(-|z|)) + max(z, 0) - z*y, the overflow-safe form.
                loss = np.logaddexp(0.0, logits) - labels * logits
                return float(loss.mean())

            solution = minimize(
                negative_log_likelihood,
                x0=np.array([1.0, 1.0, 0.0]),
                method="L-BFGS-B",
                bounds=[(0.0, None), (0.0, None), (None, None)],
            )
            self._coefficients = (
                float(solution.x[0]),
                float(solution.x[1]),
                float(solution.x[2]),
            )
            self._constant = None
        self._params = {
            "method": "beta",
            "n": int(scores.size),
            "coef": None if self._coefficients is None else list(self._coefficients[:2]),
            "intercept": None if self._coefficients is None else self._coefficients[2],
            "constant": self._constant,
        }
        self._fitted = True

    def transform(self, scores: FloatArray) -> FloatArray:
        self._require_fitted()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if self._coefficients is None:
            return np.full(scores.shape, self._constant if self._constant is not None else 0.5)
        a, b, c = self._coefficients
        logits = _beta_design(scores) @ np.array([a, b]) + c
        probabilities: FloatArray = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(probabilities, 0.0, 1.0)


def _logit(p: FloatArray) -> FloatArray:
    return np.log(p / (1.0 - p))


def _beta_design(scores: FloatArray) -> FloatArray:
    clipped = np.clip(scores, _EPS, 1.0 - _EPS)
    return np.column_stack([np.log(clipped), -np.log(1.0 - clipped)])


_CALIBRATORS: dict[str, type[_BaseCalibrator]] = {
    "identity": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
    "beta": BetaCalibrator,
}


def build_calibrator(method: str) -> Calibrator:
    try:
        return _CALIBRATORS[method]()  # type: ignore[return-value]
    except KeyError:
        known = ", ".join(sorted(_CALIBRATORS))
        msg = f"unknown calibration method {method!r}; known: {known}"
        raise KeyError(msg) from None

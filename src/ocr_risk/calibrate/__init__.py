"""Probability calibrators. Distinct from model training, and separately frozen.

Layer 7. Calibration metrics live in :mod:`ocr_risk.metrics.calibration`, because they are
metrics computed from artifacts; this package only fits and applies the transforms.
"""

from __future__ import annotations

from ocr_risk.calibrate.calibrators import (
    BetaCalibrator,
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    NotFittedCalibratorError,
    PlattCalibrator,
    TemperatureCalibrator,
    build_calibrator,
)

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

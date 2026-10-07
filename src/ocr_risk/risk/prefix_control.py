"""Declared families of nested acceptance prefixes, and simultaneous control over them.

A deployment does not choose a threshold from a continuum. It chooses one of a family of
acceptance depths declared before any label is read, and the confidence level has to be spread
over that family or the guarantee is void by construction. ``risk.select_threshold`` already does
this for a single bound with a union correction; this module separates the two decisions --
*which prefixes are on the menu* and *how the level is spread across them* -- so that a stage can
vary one while holding the other fixed.

That separation is the point. A nine-point family under a union bound spends a factor of nine on
multiplicity and throws away the resolution between its points; a two-hundred-point family under
the same rule keeps the resolution and spends a factor of two hundred. Fixed-sequence gatekeeping
buys the resolution without the factor, at the cost of stopping at the first prefix it cannot
certify.

**Fixed-sequence validity.** Order the prefixes shallow to deep and let ``j*`` be the first index
whose true risk exceeds the tolerance. The procedure deploys ``k_m`` only after rejecting
``H_1 ... H_m`` in sequence, so deploying an unsafe prefix (``m >= j*``) requires ``H_j*`` -- a
true null -- to have been rejected, which happens with probability at most ``delta`` if each
marginal test is valid at ``delta``. No independence between the tests is required. What the
argument does *not* give is power: if risk is not monotone in depth, the procedure stops early
and never sees a safe prefix beyond the first unsafe one. The derivation, and what it does and
does not assume, is in ``docs/sgv15b/risk_bound_design.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PREFIX_CONTROLS",
    "PrefixDecision",
    "assert_nested",
    "prefix_thresholds",
    "select_prefix",
]

FloatArray = NDArray[np.float64]

PREFIX_CONTROLS = ("union", "fixed_sequence")
"""``union`` splits the level across the declared family and tests every member.
``fixed_sequence`` tests in declared order at the full level and stops at the first failure."""


def assert_nested(depths: Sequence[float]) -> tuple[float, ...]:
    """A declared prefix family: strictly increasing acceptance depths inside ``(0, 1]``.

    Strictness is not pedantry. Fixed sequence derives its validity from a fixed a-priori order,
    and a family that is unordered, empty, or contains a depth outside the unit interval is not
    the object the theorem is about.
    """
    family = tuple(float(d) for d in depths)
    if not family:
        msg = "a declared prefix family must contain at least one depth"
        raise ValueError(msg)
    if any(not 0.0 < d <= 1.0 for d in family):
        msg = f"acceptance depths must lie in (0, 1]; got {family}"
        raise ValueError(msg)
    if any(later <= earlier for earlier, later in pairwise(family)):
        msg = f"acceptance depths must be strictly increasing, shallow to deep; got {family}"
        raise ValueError(msg)
    return family


def prefix_thresholds(
    scores: NDArray[np.float64] | list[float], depths: Sequence[float]
) -> FloatArray:
    """Score thresholds at the declared acceptance depths of an **unlabelled** score vector.

    Quantiles of the certification pool's scores, so the family is fixed by the ranking and the
    declared depths alone. Handing this function an outcome label, an evaluation score, or an
    oracle would make the family data-dependent and void the multiplicity argument; the leakage
    suite asserts the call sites.
    """
    family = assert_nested(depths)
    observations = np.asarray(scores, dtype=np.float64).ravel()
    if observations.size == 0:
        return np.full(len(family), np.inf, dtype=np.float64)
    return np.asarray(np.quantile(observations, [1.0 - d for d in family]), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PrefixDecision:
    """Which prefix a control procedure deploys, and what it paid to get there."""

    control: str
    family_size: int
    per_test_delta: float
    certified: tuple[int, ...]
    """Indices of every prefix the procedure certified. Under fixed sequence this is a prefix of
    the family by construction; under the union bound it need not be contiguous."""
    selected: int
    """Index of the deployed prefix, or ``-1`` when the procedure certified nothing."""
    tested: int
    """How many members were actually examined. Fixed sequence stops early; the union bound does
    not, and the difference is the label-free part of what gatekeeping saves."""
    stopped_at: int
    """Index of the first member fixed sequence could not certify, or ``-1``."""
    feasible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "control": self.control,
            "family_size": self.family_size,
            "per_test_delta": self.per_test_delta,
            "certified": list(self.certified),
            "selected": self.selected,
            "tested": self.tested,
            "stopped_at": self.stopped_at,
            "feasible": self.feasible,
        }


def select_prefix(
    depths: Sequence[float],
    certifies: Callable[[int, float], bool],
    *,
    delta: float,
    control: str = "fixed_sequence",
) -> PrefixDecision:
    """Deploy the deepest prefix the declared control procedure can certify.

    ``certifies(index, per_test_delta)`` is the caller's test for one member of the family. It is
    a callback rather than a bound so that the same control procedure serves a candidate-level
    Clopper-Pearson certificate and a document-level cluster certificate without either of them
    knowing about multiplicity, which is what lets the two be crossed in a factorial.
    """
    family = assert_nested(depths)
    if control not in PREFIX_CONTROLS:
        msg = f"unknown prefix control {control!r}; known: {', '.join(PREFIX_CONTROLS)}"
        raise ValueError(msg)
    if not 0.0 < delta < 1.0:
        msg = f"delta must lie strictly inside (0, 1); got {delta}"
        raise ValueError(msg)

    size = len(family)
    if control == "union":
        per_test = delta / size
        certified = tuple(index for index in range(size) if certifies(index, per_test))
        return PrefixDecision(
            control=control,
            family_size=size,
            per_test_delta=per_test,
            certified=certified,
            selected=max(certified) if certified else -1,
            tested=size,
            stopped_at=-1,
            feasible=bool(certified),
        )

    certified_list: list[int] = []
    stopped = -1
    for index in range(size):
        if not certifies(index, delta):
            stopped = index
            break
        certified_list.append(index)
    return PrefixDecision(
        control=control,
        family_size=size,
        per_test_delta=delta,
        certified=tuple(certified_list),
        selected=certified_list[-1] if certified_list else -1,
        tested=len(certified_list) + (1 if stopped >= 0 else 0),
        stopped_at=stopped,
        feasible=bool(certified_list),
    )

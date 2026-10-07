"""Character-level alignment by dynamic programming.

A weighted edit distance (Needleman-Wunsch with free matches) that additionally supports
**block substitutions**: the classic OCR confusions are not one-for-one. "rn" read as "m"
is a two-characters-for-one error, and scoring it as a substitution plus a deletion at
full cost makes the aligner prefer a worse path. Giving those blocks a reduced cost is a
model of the *scanner*, not of language, which is why the pair list lives in configuration.

The result carries the alignment path, so callers can project character correspondences
back onto spans, and a second-best cost, which is what makes an "the aligner nearly tied"
signal available downstream instead of a confident-looking arbitrary choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

__all__ = ["CharAlignment", "EditCosts", "Op", "align_chars", "parse_confusable_pairs"]

_MAX_DEFAULT_CELLS = 250_000


class Op(IntEnum):
    """One step of an alignment path."""

    MATCH = 0
    SUBSTITUTE = 1
    INSERT = 2
    """A character present in the OCR with no ground-truth counterpart."""
    DELETE = 3
    """A ground-truth character the OCR did not produce."""
    BLOCK = 4
    """A confusable multi-character substitution, e.g. ``rn`` -> ``m``."""


@dataclass(frozen=True, slots=True)
class EditCosts:
    """Cost model for the DP. Every value is configuration, not a constant."""

    substitute: float = 1.0
    insert: float = 1.0
    delete: float = 1.0
    confusable: float = 0.6
    blocks: tuple[tuple[str, str], ...] = ()
    """Directed ``(ocr_text, gt_text)`` pairs charged ``confusable`` instead of full cost."""
    max_cells: int = _MAX_DEFAULT_CELLS
    """Above this the exact DP is refused; the caller falls back or marks UNRESOLVED
    rather than silently producing a low-quality alignment."""

    @property
    def block_map(self) -> dict[tuple[int, int], set[tuple[str, str]]]:
        """Blocks indexed by their (ocr_len, gt_len) shape, for a cheap inner loop."""
        grouped: dict[tuple[int, int], set[tuple[str, str]]] = {}
        for ocr_text, gt_text in self.blocks:
            grouped.setdefault((len(ocr_text), len(gt_text)), set()).add((ocr_text, gt_text))
        return grouped


def parse_confusable_pairs(specs: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Expand ``"rn|m"`` specifications into directed pairs, both ways.

    Confusion is symmetric at the level of *which* characters get mixed up: an engine that
    reads "rn" as "m" also reads "m" as "rn" on a different page. Single-character pairs
    where both sides are equal are dropped as meaningless.
    """
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        left, _, right = spec.partition("|")
        if not left or not right or left == right:
            continue
        pairs.append((left, right))
        pairs.append((right, left))
    return tuple(dict.fromkeys(pairs))


@dataclass(slots=True)
class CharAlignment:
    """Result of aligning two character strings."""

    ops: list[tuple[Op, int, int, int, int]] = field(default_factory=list)
    """``(op, ocr_start, ocr_end, gt_start, gt_end)`` half-open spans, in order."""
    cost: float = 0.0
    ocr_length: int = 0
    gt_length: int = 0

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """Character index pairs implied by MATCH, SUBSTITUTE, and BLOCK steps.

        MATCH and SUBSTITUTE runs advance both sides in lockstep, so their correspondence
        is positional: the k-th character of the run maps to the k-th, *not* to all of
        them. Taking a cross product over a merged run would link every span on a line to
        every token on it and collapse the whole line into one component.

        BLOCK steps genuinely are many-to-many: a ``rn`` -> ``m`` confusion has no finer
        correspondence to report, so every character links to every character.
        """
        out: list[tuple[int, int]] = []
        for op, a0, a1, b0, b1 in self.ops:
            if op in (Op.INSERT, Op.DELETE):
                continue
            if op is Op.BLOCK:
                out.extend((i, j) for i in range(a0, a1) for j in range(b0, b1))
            else:
                out.extend(zip(range(a0, a1), range(b0, b1), strict=False))
        return out

    @property
    def agreement(self) -> float:
        """Fraction of the longer string covered by exact matches."""
        matched = sum(a1 - a0 for op, a0, a1, _, _ in self.ops if op is Op.MATCH)
        denominator = max(self.ocr_length, self.gt_length)
        return matched / denominator if denominator else 1.0


def align_chars(ocr: str, gt: str, costs: EditCosts) -> CharAlignment:
    """Align two strings exactly under ``costs``.

    Raises :class:`ValueError` when the problem exceeds ``costs.max_cells``. Refusing is
    deliberate: an aligner that silently degrades on hard input produces confident-looking
    garbage, and the whole point of this stage is that low-quality matches stay visible.
    """
    n, m = len(ocr), len(gt)
    if n * m > costs.max_cells:
        msg = f"alignment problem too large: {n}x{m} exceeds max_cells={costs.max_cells}"
        raise ValueError(msg)
    if n == 0 and m == 0:
        return CharAlignment()
    if n == 0:
        return CharAlignment(ops=[(Op.DELETE, 0, 0, 0, m)], cost=costs.delete * m, gt_length=m)
    if m == 0:
        return CharAlignment(ops=[(Op.INSERT, 0, n, 0, 0)], cost=costs.insert * n, ocr_length=n)

    blocks = costs.block_map
    # dp[i][j] = cost of aligning ocr[:i] with gt[:j]; back[i][j] = (op, di, dj).
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[Op, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + costs.insert
        back[i][0] = (Op.INSERT, 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + costs.delete
        back[0][j] = (Op.DELETE, 0, 1)

    for i in range(1, n + 1):
        ocr_char = ocr[i - 1]
        row, previous_row = dp[i], dp[i - 1]
        back_row = back[i]
        for j in range(1, m + 1):
            gt_char = gt[j - 1]
            if ocr_char == gt_char:
                best, best_op = previous_row[j - 1], (Op.MATCH, 1, 1)
            else:
                best, best_op = previous_row[j - 1] + costs.substitute, (Op.SUBSTITUTE, 1, 1)

            candidate = previous_row[j] + costs.insert
            if candidate < best:
                best, best_op = candidate, (Op.INSERT, 1, 0)

            candidate = row[j - 1] + costs.delete
            if candidate < best:
                best, best_op = candidate, (Op.DELETE, 0, 1)

            for (di, dj), members in blocks.items():
                if di > i or dj > j:
                    continue
                if (ocr[i - di : i], gt[j - dj : j]) in members:
                    candidate = dp[i - di][j - dj] + costs.confusable
                    if candidate < best:
                        best, best_op = candidate, (Op.BLOCK, di, dj)

            row[j] = best
            back_row[j] = best_op

    ops: list[tuple[Op, int, int, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:  # pragma: no cover - the borders are always initialized
            break
        op, di, dj = step
        ops.append((op, i - di, i, j - dj, j))
        i, j = i - di, j - dj
    ops.reverse()

    return CharAlignment(ops=_merge_runs(ops), cost=dp[n][m], ocr_length=n, gt_length=m)


def _merge_runs(
    ops: list[tuple[Op, int, int, int, int]],
) -> list[tuple[Op, int, int, int, int]]:
    """Collapse consecutive identical operations into runs.

    Purely a compaction of the path; the character spans are unchanged. It makes the
    op list readable in a fixture and cheap to scan when inducing components.
    """
    merged: list[tuple[Op, int, int, int, int]] = []
    for op, a0, a1, b0, b1 in ops:
        if merged and merged[-1][0] is op and op in (Op.MATCH, Op.SUBSTITUTE, Op.INSERT, Op.DELETE):
            prev_op, pa0, pa1, pb0, pb1 = merged[-1]
            if pa1 == a0 and pb1 == b0:
                merged[-1] = (prev_op, pa0, a1, pb0, b1)
                continue
        merged.append((op, a0, a1, b0, b1))
    return merged

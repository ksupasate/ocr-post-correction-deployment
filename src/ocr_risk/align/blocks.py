"""Anchor-and-recurse decomposition for long unanchored text.

Used when the ground truth carries no geometry, so a whole page is one region and an exact
dynamic program over it would be quadratic in page length.

The method is the one behind patience diff: find *uniquely occurring* long common
substrings, treat them as fixed points, and recurse into the gaps. Uniqueness is the load-
bearing property — a substring occurring twice on a page ("the ") gives no information
about which occurrence corresponds to which, and using it as an anchor would lock in an
arbitrary and possibly wrong correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Block", "decompose"]


@dataclass(frozen=True, slots=True)
class Block:
    """A region of the two strings to align independently."""

    ocr_start: int
    ocr_end: int
    gt_start: int
    gt_end: int
    anchored: bool = False
    """True for a verbatim common substring: identical by construction, no DP needed."""

    @property
    def cells(self) -> int:
        return (self.ocr_end - self.ocr_start) * (self.gt_end - self.gt_start)


def _unique_common_substring(ocr: str, gt: str, min_length: int) -> tuple[int, int, int] | None:
    """Longest substring occurring exactly once in each string. ``(ocr_i, gt_j, length)``.

    Scans from the longest plausible length downward and stops at the first hit, so the
    strongest available anchor is chosen.
    """
    limit = min(len(ocr), len(gt))
    for length in range(limit, min_length - 1, -1):
        seen: dict[str, int] = {}
        for i in range(len(ocr) - length + 1):
            piece = ocr[i : i + length]
            seen[piece] = -1 if piece in seen else i
        for piece, ocr_index in seen.items():
            if ocr_index < 0:
                continue
            first = gt.find(piece)
            if first < 0 or gt.find(piece, first + 1) >= 0:
                continue  # absent, or ambiguous in the ground truth
            return (ocr_index, first, length)
    return None


def decompose(
    ocr: str,
    gt: str,
    *,
    min_anchor_length: int,
    max_block_cells: int,
    ocr_offset: int = 0,
    gt_offset: int = 0,
    depth: int = 0,
    max_depth: int = 24,
) -> list[Block]:
    """Split a pair of strings into blocks small enough to align exactly.

    Blocks that remain too large after recursion are returned as-is; the caller decides
    whether to mark them ``UNRESOLVED`` rather than aligning them badly.
    """
    block = Block(ocr_offset, ocr_offset + len(ocr), gt_offset, gt_offset + len(gt))
    if not ocr or not gt or block.cells <= max_block_cells or depth >= max_depth:
        return [block]

    found = _unique_common_substring(ocr, gt, min_anchor_length)
    if found is None:
        return [block]

    ocr_index, gt_index, length = found
    left = decompose(
        ocr[:ocr_index],
        gt[:gt_index],
        min_anchor_length=min_anchor_length,
        max_block_cells=max_block_cells,
        ocr_offset=ocr_offset,
        gt_offset=gt_offset,
        depth=depth + 1,
        max_depth=max_depth,
    )
    middle = Block(
        ocr_offset + ocr_index,
        ocr_offset + ocr_index + length,
        gt_offset + gt_index,
        gt_offset + gt_index + length,
        anchored=True,
    )
    right = decompose(
        ocr[ocr_index + length :],
        gt[gt_index + length :],
        min_anchor_length=min_anchor_length,
        max_block_cells=max_block_cells,
        ocr_offset=ocr_offset + ocr_index + length,
        gt_offset=gt_offset + gt_index + length,
        depth=depth + 1,
        max_depth=max_depth,
    )
    return [*left, middle, *right]

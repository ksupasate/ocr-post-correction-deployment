"""Document-identity primitives for freshness and duplicate audits.

Freshness is a claim about *document identity*, not about id strings: the same page
reappearing under another filename defeats an id-level check. These helpers give the
qualification audits one shared definition of content-level identity:

- ``average_hash`` — a normalized perceptual fingerprint of the page image (aHash on a
  fixed grayscale grid). Coarse by design: it exists to catch "the same photograph,
  lightly re-encoded", not to measure visual similarity for reporting.
- ``hamming_distance`` / ``connected_groups`` — near-duplicate grouping under a fixed
  bit threshold, as connected components (transitivity matters: A~B and B~C groups
  A,B,C even when A and C differ in more than the threshold).

Text identity deliberately reuses :func:`ocr_risk.datasets.synthetic.shingle_hash`
rather than defining a second text fingerprint here.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from PIL import Image

__all__ = [
    "AHASH_DEFAULT_SIZE",
    "AHASH_DEFAULT_THRESHOLD_BITS",
    "DHASH_DEFAULT_SIZE",
    "DHASH_DEFAULT_THRESHOLD_BITS",
    "TEXT_SIMHASH_THRESHOLD_BITS",
    "average_hash",
    "connected_groups",
    "difference_hash",
    "group_by_value",
    "hamming_distance",
    "near_document_identity",
    "normalized_pixel_sha256",
    "text_simhash",
]

AHASH_DEFAULT_SIZE = 8
AHASH_DEFAULT_THRESHOLD_BITS = 6
DHASH_DEFAULT_SIZE = 16
DHASH_DEFAULT_THRESHOLD_BITS = 24
TEXT_SIMHASH_THRESHOLD_BITS = 8


def average_hash(image: Image.Image, *, size: int = AHASH_DEFAULT_SIZE) -> int:
    """Return the ``size * size``-bit average hash of a page image as an int.

    Deterministic and dependency-free: resize to the fixed grid in grayscale, threshold
    each pixel against the grid's mean, pack bits row-major. Two renderings of the same
    page land within a few bits; two different receipts do not.
    """
    grid = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = cast(list[int], list(grid.get_flattened_data()))
    mean = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | (1 if value > mean else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def difference_hash(image: Image.Image, *, size: int = DHASH_DEFAULT_SIZE) -> int:
    """Return a ``size * size``-bit horizontal difference hash.

    dHash complements the deliberately coarse aHash: two mostly-white receipt pages can
    share a mean-threshold pattern while differing in local edge structure. Requiring
    both hashes sharply reduces that false-positive mode in the C1 near-duplicate audit.
    """
    grid = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = cast(list[int], list(grid.get_flattened_data()))
    bits = 0
    stride = size + 1
    for y in range(size):
        row = y * stride
        for x in range(size):
            bits = (bits << 1) | (1 if pixels[row + x] > pixels[row + x + 1] else 0)
    return bits


def normalized_pixel_sha256(image: Image.Image) -> str:
    """Hash decoded RGB pixels plus dimensions, independent of image encoding."""
    rgb = image.convert("RGB")
    prefix = f"RGB:{rgb.width}x{rgb.height}:".encode("ascii")
    return hashlib.sha256(prefix + rgb.tobytes()).hexdigest()


def text_simhash(text: str, *, bits: int = 64) -> int:
    """Return a deterministic character-shingle SimHash without retaining GT text."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    shingles = [normalized[i : i + 3] for i in range(max(1, len(normalized) - 2))]
    if not normalized:
        shingles = [""]
    weights = [0] * bits
    for shingle in shingles:
        value = int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[: bits // 8])
        for index in range(bits):
            weights[index] += 1 if value & (1 << index) else -1
    result = 0
    for index, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << index
    return result


def near_document_identity(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    ahash_threshold: int = AHASH_DEFAULT_THRESHOLD_BITS,
    dhash_threshold: int = DHASH_DEFAULT_THRESHOLD_BITS,
    text_threshold: int = TEXT_SIMHASH_THRESHOLD_BITS,
    aspect_ratio_relative_tolerance: float = 0.02,
) -> bool:
    """Conservative same-document predicate for lightly transformed page images.

    A pair must agree simultaneously on coarse tone, edge structure, normalized GT-text
    structure, and aspect ratio. No single weak signal can create a duplicate group.
    Exact pixel/annotation identity is handled separately and does not need this test.
    """
    left_width, left_height = left["width"], left["height"]
    right_width, right_height = right["width"], right["height"]
    if not all(
        isinstance(value, (int, float))
        for value in (left_width, left_height, right_width, right_height)
    ):
        raise TypeError("document identity dimensions must be numeric")
    assert isinstance(left_width, (int, float))
    assert isinstance(left_height, (int, float))
    assert isinstance(right_width, (int, float))
    assert isinstance(right_height, (int, float))
    left_ratio = left_width / left_height
    right_ratio = right_width / right_height
    ratio_delta = abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
    if ratio_delta > aspect_ratio_relative_tolerance:
        return False
    return (
        hamming_distance(int(str(left["ahash"]), 16), int(str(right["ahash"]), 16))
        <= ahash_threshold
        and hamming_distance(int(str(left["dhash"]), 16), int(str(right["dhash"]), 16))
        <= dhash_threshold
        and hamming_distance(
            int(str(left["gt_text_simhash"]), 16),
            int(str(right["gt_text_simhash"]), 16),
        )
        <= text_threshold
    )


def connected_groups(
    members: Sequence[str],
    neighbors: Callable[[str, str], bool],
) -> list[list[str]]:
    """Group ``members`` into connected components under the symmetric ``neighbors``.

    Order is deterministic: components are emitted sorted by first member in the input
    order, and each component is sorted. Union-find over indices keeps this O(n^2)
    neighbour calls in the worst case, which is the honest cost when neighbours are
    defined by a distance threshold rather than a hash bucket.
    """
    parent = list(range(len(members)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if neighbors(members[i], members[j]):
                union(i, j)

    buckets: dict[int, list[str]] = {}
    for i, member in enumerate(members):
        buckets.setdefault(find(i), []).append(member)
    groups = [sorted(group) for group in buckets.values()]
    groups.sort(key=lambda group: members.index(group[0]))
    return [group for group in groups if len(group) > 1]


def group_by_value(mapping: Mapping[str, str]) -> dict[str, list[str]]:
    """Invert ``id -> fingerprint`` into ``fingerprint -> [ids]``, keeping only dups."""
    buckets: dict[str, list[str]] = {}
    for key, value in mapping.items():
        buckets.setdefault(value, []).append(key)
    return {value: sorted(keys) for value, keys in buckets.items() if len(keys) > 1}

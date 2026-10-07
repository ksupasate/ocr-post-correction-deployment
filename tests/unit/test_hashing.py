"""Canonical hashing: stability, and refusal to hash things that are not reproducible."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ocr_risk.io.hashing import (
    canonical_hash,
    canonical_json,
    canonicalize,
    file_sha256,
    hash_str,
    sha256_of_bytes,
    short,
    stable_string_set_hash,
)
from ocr_risk.schemas import BBox


def test_mapping_order_does_not_change_the_hash() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_nested_mapping_order_does_not_change_the_hash() -> None:
    left = {"outer": {"z": [1, {"q": 1, "p": 2}], "a": 1}}
    right = {"outer": {"a": 1, "z": [1, {"p": 2, "q": 1}]}}
    assert canonical_hash(left) == canonical_hash(right)


def test_list_order_does_change_the_hash() -> None:
    """For a list, order is content; treating [a, b] and [b, a] alike would erase a real
    difference (e.g. verifier ranking)."""
    assert canonical_hash([1, 2]) != canonical_hash([2, 1])


def test_set_order_does_not_change_the_hash() -> None:
    assert canonical_hash({"b", "a", "c"}) == canonical_hash({"c", "b", "a"})


def test_negative_zero_hashes_as_zero() -> None:
    """Two numerically identical configs must not differ by a sign bit on zero."""
    assert canonical_hash({"x": -0.0}) == canonical_hash({"x": 0.0})


def test_non_finite_floats_are_representable() -> None:
    """NaN and infinity must hash to something stable rather than raising, so a config
    carrying them is still traceable."""
    digest = canonical_hash({"a": math.nan, "b": math.inf, "c": -math.inf})
    assert len(digest) == 64
    assert canonical_hash({"a": math.nan}) != canonical_hash({"a": math.inf})


def test_int_and_float_are_distinguished() -> None:
    assert canonical_hash({"n": 1}) != canonical_hash({"n": 1.0})


def test_bool_is_not_conflated_with_int() -> None:
    assert canonicalize(True) is True
    assert canonical_hash({"f": True}) != canonical_hash({"f": 1})


def test_pydantic_models_hash_by_content() -> None:
    a = BBox(x0=0.0, y0=0.0, x1=1.0, y1=2.0)
    b = BBox(x0=0.0, y0=0.0, x1=1.0, y1=2.0)
    assert canonical_hash(a) == canonical_hash(b)
    assert canonical_hash(a) != canonical_hash(BBox(x0=0.0, y0=0.0, x1=1.0, y1=3.0))


def test_generators_are_refused() -> None:
    """A generator's iteration order is not reproducible, and consuming it here would
    also silently exhaust the caller's iterator."""
    with pytest.raises(TypeError, match="iteration order"):
        canonical_hash(x for x in range(3))


def test_unhashable_type_is_refused() -> None:
    with pytest.raises(TypeError, match="cannot canonicalize"):
        canonical_hash(object())


def test_paths_hash_posix_style() -> None:
    """A digest must not depend on the OS separator, or Linux CI would disagree with
    macOS for the same logical path."""
    assert canonical_json(Path("a/b/c.txt")) == '"a/b/c.txt"'


def test_string_set_hash_ignores_order_and_duplicates() -> None:
    assert stable_string_set_hash(["b", "a", "b"]) == stable_string_set_hash(["a", "b"])


def test_string_set_hash_detects_membership_change() -> None:
    assert stable_string_set_hash(["a", "b"]) != stable_string_set_hash(["a", "b", "c"])


def test_file_sha256_matches_bytes_hash(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"ocr-risk" * 5000
    path.write_bytes(payload)
    assert file_sha256(path) == sha256_of_bytes(payload)


def test_file_sha256_handles_multichunk_files(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    payload = bytes(range(256)) * 20_000  # > 1 MiB read chunk
    path.write_bytes(payload)
    assert file_sha256(path) == sha256_of_bytes(payload)


def test_hash_str_is_plain_sha256_of_utf8() -> None:
    assert hash_str("é") == sha256_of_bytes("é".encode())


def test_short_truncates() -> None:
    assert short("0123456789abcdef") == "01234567"
    assert short("0123456789abcdef", 4) == "0123"


def test_numpy_scalars_canonicalize_to_the_same_value_as_python_numbers() -> None:
    """Since numpy 2, repr(np.float64(0.42)) is the string "np.float64(0.42)".

    canonicalize() called repr() on anything that is a float, and np.float64 IS a float
    subclass -- so every numpy scalar in an engine payload was written to the write-once
    raw layer as an unparseable string. docTR reports word geometry as numpy scalars, so
    every one of its stored bounding boxes was corrupt: evidence destroyed at the moment
    of recording, discovered only when a later stage tried to multiply it by a page width.
    It also made a numpy scalar and the identical Python float hash differently.
    """
    import numpy as np

    assert canonicalize(np.float64(0.42182373687664043)) == canonicalize(0.42182373687664043)
    assert canonical_hash(np.float64(0.42)) == canonical_hash(0.42)
    # np.int64 is NOT an int subclass, so a bare isinstance check dropped through to the
    # "cannot canonicalize" error rather than to the float branch.
    assert canonicalize(np.int64(7)) == 7
    assert canonical_hash(np.int32(7)) == canonical_hash(7)
    assert canonicalize(np.float32(1.5)) == canonicalize(1.5)


def test_a_numpy_payload_survives_the_json_round_trip_as_numbers() -> None:
    """The raw store writes canonical JSON, so an unparseable encoding is data loss."""
    import json

    import numpy as np

    payload = {"geometry": [[np.float64(0.1), np.float64(0.2)], [np.float64(0.3), 0.4]]}
    restored = json.loads(canonical_json(payload))
    assert [[float(v) for v in point] for point in restored["geometry"]] == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

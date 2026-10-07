"""Canonical, stable content hashing.

Every provenance claim in this repository reduces to "these bytes produced those bytes",
so hashing must be stable across dict insertion order, float formatting, Python runs, and
machines. A hash that changes for cosmetic reasons is worse than no hash: it teaches
people to ignore mismatches.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

__all__ = [
    "canonical_hash",
    "canonicalize",
    "file_sha256",
    "hash_str",
    "sha256_of_bytes",
    "short",
    "stable_string_set_hash",
]

_CHUNK = 1 << 20
_NAN = "__nan__"
_POS_INF = "__inf__"
_NEG_INF = "__-inf__"


def canonicalize(value: Any) -> Any:
    """Reduce ``value`` to a JSON-safe form with every ordering ambiguity removed.

    - mappings are key-sorted (insertion order must not change a hash);
    - sets and frozensets become sorted lists (iteration order is not stable);
    - floats are normalized so ``-0.0``/``0.0`` and the non-finite values are
      representable and consistent across platforms;
    - pydantic models and enums are reduced to their data.

    Sequences keep their order: for a list, order *is* content.
    """
    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        # Frozen dataclasses are the codebase's value objects (PartitionSpec, CropPolicy,
        # EditCosts). Field order is declaration order, which is stable, and the mapping
        # branch below sorts the keys anyway.
        return canonicalize({f.name: getattr(value, f.name) for f in fields(value)})
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, bool) or value is None or isinstance(value, int | str):
        return value
    # numbers.Integral / numbers.Real rather than int / float: numpy scalars register with
    # the ABCs, and np.int64 is NOT an int subclass, so a bare isinstance check would drop
    # through to the "cannot canonicalize" error on any payload carrying one.
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        # float() first, and this is not cosmetic. np.float64 IS a float subclass, so it
        # reached the branch below -- and since numpy 2, repr(np.float64(0.42)) is the
        # string "np.float64(0.42)", which json then stored verbatim. Every docTR word
        # geometry was written to the write-once raw layer as an unparseable string:
        # evidence destroyed at the moment of recording, and only discovered when a later
        # stage tried to multiply it. It also meant a numpy scalar and the identical
        # Python float hashed differently.
        number = float(value)
        if math.isnan(number):
            return _NAN
        if math.isinf(number):
            return _POS_INF if number > 0 else _NEG_INF
        # repr() is the shortest round-tripping form and is platform-stable; normalizing
        # negative zero keeps two numerically equal configs from hashing differently.
        return repr(number + 0.0)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, Mapping):
        return {
            str(k): canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, frozenset | set):
        return sorted(
            json.dumps(canonicalize(v), sort_keys=True, ensure_ascii=False) for v in value
        )
    if isinstance(value, Sequence):
        return [canonicalize(v) for v in value]
    if isinstance(value, Iterable):
        msg = (
            f"refusing to hash iterator of type {type(value)!r}: iteration order is not "
            "guaranteed reproducible. Materialize it into a list or set first."
        )
        raise TypeError(msg)
    msg = f"cannot canonicalize value of type {type(value)!r} for hashing"
    raise TypeError(msg)


def canonical_json(value: Any) -> str:
    """Deterministic JSON text for ``value``."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    """sha256 of the canonical JSON encoding of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_str(text: str) -> str:
    """sha256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path | str) -> str:
    """Streaming sha256 of a file, safe for images and parquet far larger than memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def stable_string_set_hash(values: Iterable[str]) -> str:
    """Order-independent hash of a set of identifiers.

    Used for the document-set digests in :class:`~ocr_risk.schemas.SplitDescriptor`, where
    the *membership* of a split is what must be auditable, not the order it was iterated.
    """
    return canonical_hash(sorted(set(values)))


def short(digest: str, length: int = 8) -> str:
    """First ``length`` characters of a hex digest, for run identifiers and log lines."""
    return digest[:length]


def faithful_json(value: Any) -> str:
    """Deterministic JSON that keeps numbers as JSON numbers.

    Distinct from :func:`canonical_json`, and the distinction matters. Canonical JSON
    encodes floats via ``repr`` -- as *strings* -- so that two numerically equal values
    always hash identically regardless of platform float formatting. That is right for
    hashing and wrong for storage: a raw engine response written that way comes back with
    every measurement as a string, which is the reformatting the provenance rules forbid,
    and it silently broke a parser that did arithmetic on a coordinate.

    Numpy scalars are converted to Python numbers (exactly -- float64 to float is
    lossless), because ``json`` cannot encode them and the alternative was ``repr``,
    which since numpy 2 produces ``"np.float64(0.42)"``.
    """
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    """Recursively convert to JSON-native types, preserving numbers as numbers."""
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable({f.name: getattr(value, f.name) for f in fields(value)})
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        # JSON has no NaN or Infinity. Recording the token preserves the distinction
        # between "the engine reported a non-finite value" and "the engine reported
        # nothing", which writing null would erase.
        if math.isnan(number):
            return _NAN
        if math.isinf(number):
            return _POS_INF if number > 0 else _NEG_INF
        return number
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, frozenset | set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, Sequence):
        return [_jsonable(v) for v in value]
    msg = f"cannot serialize value of type {type(value)!r} for storage"
    raise TypeError(msg)

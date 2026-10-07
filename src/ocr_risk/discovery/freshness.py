"""The confirmatory reserve: which documents Track-A code must refuse to touch.

The 99 untouched FUNSD documents (49 training_data remainder + the 50 testing_data
split, per ``manifests/cgv3/partition_provenance.json``) are the CGV3 confirmatory
reserve. Everything a development run can reach -- discovery, generation, lexicon
fitting, pilots -- must reject them loudly rather than quietly include them, because
the cost of an accidental read is not a wrong number but a spent partition.

The guard is deliberately dumb: it knows document *ids*, nothing about content. The
ids come from the tracked provenance manifest at call time, so the guard and the
inventory cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

__all__ = [
    "DevelopmentDocumentError",
    "assert_development_documents",
    "confirmatory_document_ids",
]

DEFAULT_PROVENANCE = Path("manifests/cgv3/partition_provenance.json")


class DevelopmentDocumentError(ValueError):
    """A confirmatory-reserve document reached a development code path."""


@lru_cache(maxsize=1)
def _reserve_stems(provenance_path: str) -> frozenset[tuple[str, str]]:
    payload = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    inventory = payload["untouched_inventory"]["funsd"]
    pairs: set[tuple[str, str]] = {
        ("training_data", stem) for stem in inventory["training_data_remaining_ids"]
    }
    pairs |= {("testing_data", stem) for stem in inventory["testing_data_ids"]}
    return frozenset(pairs)


def confirmatory_document_ids(
    provenance_path: Path | str = DEFAULT_PROVENANCE,
) -> frozenset[str]:
    """The full document ids of the confirmatory reserve, as the corpus would name them."""
    return frozenset(
        f"funsd-{split}-{stem}" for split, stem in _reserve_stems(str(provenance_path))
    )


def assert_development_documents(
    document_ids: object,
    *,
    context: str,
    provenance_path: Path | str = DEFAULT_PROVENANCE,
) -> None:
    """Refuse any confirmatory-reserve document entering a development code path.

    ``document_ids`` may be any iterable of ids; a single string is treated as one id.
    Raises :class:`DevelopmentDocumentError` naming the offender and the caller, so a
    contamination is loud, traceable, and impossible to mistake for an empty selection.
    """
    if isinstance(document_ids, str):
        ids: tuple[str, ...] = (document_ids,)
    elif isinstance(document_ids, Iterable):
        ids = tuple(str(item) for item in document_ids)
    else:
        raise TypeError("document_ids must be an iterable of ids or a single id")
    reserve = confirmatory_document_ids(provenance_path)
    offenders = sorted(set(ids) & reserve)
    if offenders:
        shown = ", ".join(offenders[:5])
        raise DevelopmentDocumentError(
            f"{context}: {len(offenders)} confirmatory-reserve document(s) reached a "
            f"development code path ({shown}). The fresh partition is locked until the "
            "pre-confirmatory freeze; record a freshness defect if this was not intended."
        )

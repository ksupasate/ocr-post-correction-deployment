"""The global document partition — the second split axis.

The benchmark is **matched-source**: every engine reads the same page. Holding out an
engine therefore does *not* hold out the page, so a verifier trained on engine B's reading
of document D and tested on engine A's reading of D has already seen D's ground truth.
That is leakage vector L4, and it is the single largest validity threat in this design.

The fix is a second axis: one document partition, drawn once, hashed, and **identical for
every engine and every fold**. A partition redrawn per fold would rotate documents between
roles and reintroduce exactly the contamination it exists to prevent, which is why the
partition hash is recorded on every run and checked by the audit.

Assignment is by content hash rather than by shuffling, so the partition is stable under
corpus growth: adding documents does not move the ones already assigned.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ocr_risk.io.hashing import canonical_hash, hash_str, stable_string_set_hash
from ocr_risk.schemas.documents import SourceDocument
from ocr_risk.schemas.enums import SplitRole

__all__ = ["DocumentPartition", "PartitionSpec", "build_partition", "duplicate_groups"]

_ROLE_ORDER = (SplitRole.FIT, SplitRole.CALIBRATE, SplitRole.EVALUATE)


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """How the partition is drawn. Part of the config, and hashed into it."""

    seed: int = 20260817
    fit_fraction: float = 0.6
    calibrate_fraction: float = 0.2
    stratify_by: tuple[str, ...] = ("dataset_id",)

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.fit_fraction - self.calibrate_fraction


@dataclass(frozen=True, slots=True)
class DocumentPartition:
    """A frozen assignment of documents to roles."""

    role_of: dict[str, SplitRole]
    spec_hash: str
    partition_sha256: str
    duplicate_groups: tuple[tuple[str, ...], ...] = ()
    strata: dict[str, str] | None = None

    def documents(self, role: SplitRole) -> frozenset[str]:
        return frozenset(doc for doc, assigned in self.role_of.items() if assigned is role)

    def digest(self, role: SplitRole) -> str:
        """Order-independent hash of a role's membership, for the run record."""
        return stable_string_set_hash(self.documents(role))

    def counts(self) -> dict[str, int]:
        return {role.value: len(self.documents(role)) for role in _ROLE_ORDER}

    def assert_disjoint(self) -> None:
        """Roles must not overlap, and a duplicated page must not straddle two of them.

        The role sets cannot overlap: ``role_of`` maps each document to exactly one role,
        so the pairwise check below is unsatisfiable through this class and is kept only
        as a tripwire for a future representation change. The check that can actually fire
        is the second one. :meth:`load` reads ``role_of`` and ``duplicate_groups`` from a
        manifest and verifies the recorded hash — but the hash covers only the assignment,
        so a partition written by an older build, or regenerated after the corpus changed,
        can place two scans of the *same page* on opposite sides of the split and load
        cleanly. That is leakage vector L8, and it was reachable until this check existed.
        """
        for first_index, first in enumerate(_ROLE_ORDER):
            for second in _ROLE_ORDER[first_index + 1 :]:
                overlap = self.documents(first) & self.documents(second)
                if overlap:  # pragma: no cover - unreachable while role_of is a mapping
                    msg = (
                        f"document partition is not disjoint: {len(overlap)} document(s) "
                        f"in both {first.value} and {second.value}, e.g. {sorted(overlap)[:3]}"
                    )
                    raise ValueError(msg)

        for group in self.duplicate_groups:
            roles = {self.role_of[doc] for doc in group if doc in self.role_of}
            if len(roles) > 1:
                msg = (
                    f"duplicate group {sorted(group)[:3]} spans roles "
                    f"{sorted(r.value for r in roles)}. Two scans of the same page on "
                    "opposite sides of the split is the same content appearing in "
                    "training and in test (leakage vector L8)."
                )
                raise ValueError(msg)

    def to_manifest(self) -> dict[str, object]:
        return {
            "partition_sha256": self.partition_sha256,
            "spec_hash": self.spec_hash,
            "counts": self.counts(),
            "digests": {role.value: self.digest(role) for role in _ROLE_ORDER},
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "role_of": {doc: role.value for doc, role in sorted(self.role_of.items())},
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n", "utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> DocumentPartition:
        """Read a frozen partition, verifying it is the one it claims to be.

        The recorded ``partition_sha256`` is recomputed from the assignment actually read.
        Without that check, editing a document's role in the manifest would move it
        between fit and evaluate while every run record still cited the original hash —
        the audit would compare stale hashes and report a stable partition.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        role_of = {doc: SplitRole(role) for doc, role in payload["role_of"].items()}
        recorded = payload["partition_sha256"]
        recomputed = partition_digest(payload["spec_hash"], role_of)
        if recorded != recomputed:
            msg = (
                f"{path} declares partition_sha256={recorded} but its assignment hashes "
                f"to {recomputed}. The file was edited after it was written; a run citing "
                "the stale hash would look reproducible and would not be."
            )
            raise ValueError(msg)
        return cls(
            role_of=role_of,
            spec_hash=payload["spec_hash"],
            partition_sha256=recorded,
            duplicate_groups=tuple(tuple(group) for group in payload.get("duplicate_groups", [])),
        )

    def covers(self, document_ids: Iterable[str]) -> tuple[str, ...]:
        """Documents absent from this partition. Empty means the corpus is covered."""
        return tuple(sorted(set(document_ids) - set(self.role_of)))


def partition_digest(spec_hash: str, role_of: Mapping[str, SplitRole]) -> str:
    """The partition's identity: its spec plus the assignment it produced.

    Defined once and used by both construction and loading. Two implementations of the
    same hash is how a verification check silently stops verifying.
    """
    return canonical_hash({"spec": spec_hash, "assignment": sorted(role_of.items())})


def duplicate_groups(documents: Sequence[SourceDocument]) -> list[list[str]]:
    """Documents that are the same page, or near-duplicates of one.

    Both must land in the same bucket. Two scans of one page split across train and test
    is the same content appearing on both sides (leakage vector L8), and near-duplicates
    are caught by a shingle hash because a few differing characters would defeat an exact
    image hash.
    """
    by_key: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        by_key[f"image:{document.image_sha256}"].add(document.document_id)
        if document.gt_text_shingle_hash:
            by_key[f"shingle:{document.gt_text_shingle_hash}"].add(document.document_id)
        if document.work_id:
            # Pages of one work are assigned as a unit. They are not duplicates -- the
            # hash and shingle checks see nothing, because they are genuinely different
            # pages -- but they share a typeface, a scan session and a binding, so a
            # verifier fitted on pages 3-7 of a volume and evaluated on page 8 has already
            # seen that printing.
            by_key[f"work:{document.dataset_id}:{document.work_id}"].add(document.document_id)

    # Union any keys that share a document, so a chain of near-duplicates forms one group.
    groups: list[set[str]] = []
    for members in by_key.values():
        if len(members) < 2:
            continue
        merged = set(members)
        remaining: list[set[str]] = []
        for group in groups:
            if group & merged:
                merged |= group
            else:
                remaining.append(group)
        remaining.append(merged)
        groups = remaining
    return sorted((sorted(group) for group in groups), key=lambda g: g[0])


def _stratum_of(document: SourceDocument, keys: Sequence[str], extra: dict[str, str]) -> str:
    """The stratum this document belongs to, from the requested keys.

    An unresolvable key is an error, not something to drop. Silently skipping it turns a
    request for two-way stratification into one-way, or into no stratification at all, and
    the partition still looks fine: it is disjoint, it hashes, every run record agrees.

    That is not hypothetical. This pilot's config asked for
    ``[dataset_id, noise_tertile]``, but noise tertiles are only computed for the synthetic
    corpus, so on the real corpora the key resolved to nothing and the partition was
    stratified by ``dataset_id`` alone -- a silent departure from the pre-registered
    protocol that a reviewer found by recomputing the spec hash.
    """
    parts: list[str] = []
    for key in keys:
        if key in extra:
            parts.append(f"{key}={extra[key]}")
        elif hasattr(document, key):
            parts.append(f"{key}={getattr(document, key)}")
        else:
            msg = (
                f"cannot stratify document {document.document_id!r} by {key!r}: it is "
                "neither a document field nor supplied in stratum_overrides. Dropping it "
                "would silently weaken the partition to the keys that did resolve."
            )
            raise KeyError(msg)
    return "|".join(parts) or "all"


def build_partition(
    documents: Sequence[SourceDocument],
    spec: PartitionSpec,
    stratum_overrides: dict[str, dict[str, str]] | None = None,
) -> DocumentPartition:
    """Draw the partition, once, deterministically.

    Assignment is by hash of ``(seed, document_id)`` rather than by shuffling, so the
    result is stable under corpus growth: adding documents leaves existing assignments
    untouched, and a rerun on a subset agrees with the full run.

    Within each stratum, documents are ordered by their hash and cut at the configured
    fractions. That keeps the role proportions right per stratum, so a fold cannot end up
    with all the noisy pages in its test set by chance.
    """
    if not documents:
        return DocumentPartition(
            role_of={}, spec_hash=canonical_hash(spec), partition_sha256=canonical_hash([])
        )

    overrides = stratum_overrides or {}
    duplicates = duplicate_groups(documents)

    # A duplicate group is assigned as a unit, keyed by its lexicographically first member.
    canonical_of: dict[str, str] = {}
    for group in duplicates:
        for member in group:
            canonical_of[member] = group[0]

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        anchor = canonical_of.get(document.document_id, document.document_id)
        if anchor != document.document_id:
            continue  # assigned with its group's anchor
        stratum = _stratum_of(document, spec.stratify_by, overrides.get(document.document_id, {}))
        by_stratum[stratum].append(document.document_id)

    role_of: dict[str, SplitRole] = {}
    for stratum in sorted(by_stratum):
        # Order by hash (a stable, arbitrary permutation), then cut at the exact
        # fractions. See _cut_points for why exact proportions win over stability.
        anchors = sorted(by_stratum[stratum], key=lambda doc: (hash_str(f"{spec.seed}:{doc}"), doc))
        n_fit, n_cal = _cut_points(len(anchors), spec)
        for index, anchor in enumerate(anchors):
            if index < n_fit:
                role = SplitRole.FIT
            elif index < n_fit + n_cal:
                role = SplitRole.CALIBRATE
            else:
                role = SplitRole.EVALUATE
            role_of[anchor] = role
            # Every member of a duplicate group came from `documents`, so no membership
            # test is needed here -- one used to guard this loop, and it was a branch that
            # could not be taken, which reads as a handled case and is not one.
            for member in _group_of(anchor, duplicates):
                role_of[member] = role

    partition = DocumentPartition(
        role_of=role_of,
        spec_hash=canonical_hash(spec),
        partition_sha256=partition_digest(canonical_hash(spec), role_of),
        duplicate_groups=tuple(tuple(group) for group in duplicates),
    )
    partition.assert_disjoint()
    return partition


def _cut_points(n: int, spec: PartitionSpec) -> tuple[int, int]:
    """Sizes of the fit and calibrate splits for ``n`` documents.

    **Exact proportions, not stability under corpus growth.** The two are mutually
    exclusive: holding the fractions exact while adding documents forces some existing
    document to change role. Per-document hash bucketing gives the opposite trade, but at
    the corpus sizes this project actually runs (tens of pages per pilot track) it leaves
    the evaluation split badly undersized — a three-document test set cannot support a
    document-level bootstrap, and that is a problem on *every* run.

    So the partition is an artifact: drawn once, saved to
    ``manifests/splits/document_partition.json``, and reused. Regenerating it after the
    corpus changes produces a different ``partition_sha256``, which the leakage audit
    surfaces rather than letting it pass unnoticed.
    """
    n_fit = round(n * spec.fit_fraction)
    n_cal = round(n * spec.calibrate_fraction)

    # Rounding can starve a role. An empty evaluation split measures nothing, and an empty
    # calibration split leaves the threshold unfittable, so borrow from the largest.
    if n >= 3:
        while n_fit + n_cal >= n:
            if n_fit >= n_cal:
                n_fit -= 1
            else:
                n_cal -= 1
        if n_cal == 0:
            n_cal, n_fit = 1, min(n_fit, n - 2)
    return n_fit, n_cal


def _group_of(anchor: str, duplicates: Iterable[Sequence[str]]) -> tuple[str, ...]:
    for group in duplicates:
        if group and group[0] == anchor:
            return tuple(group)
    return (anchor,)

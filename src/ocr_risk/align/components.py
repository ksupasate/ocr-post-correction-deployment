"""Stage C: project a character path onto span-to-token components, and type them.

The character alignment says which characters correspond. This turns that into the object
the rest of the pipeline needs: connected components of the OCR-span / ground-truth-token
bipartite graph, each labelled with the *kind* of disagreement it represents.

The relation is read off the component's shape, which is what makes the taxonomy
mechanical rather than heuristic:

======================  ==================================================
shape                   relation
======================  ==================================================
1 span, 1 token         ``ONE_TO_ONE``
N spans, 1 token        ``SPLIT``   (the engine broke a word apart)
1 span, N tokens        ``MERGE``   (the engine ran words together)
N spans, M tokens       ``MANY_TO_MANY``
1+ spans, 0 tokens      ``OCR_INSERTION`` (hallucinated text)
0 spans, 1+ tokens      ``OCR_DELETION``  (omitted text)
======================  ==================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ocr_risk.align.char_dp import CharAlignment, Op
from ocr_risk.align.streams import Stream
from ocr_risk.schemas.enums import AlignmentRelation

__all__ = ["Component", "classify_relation", "induce_components"]


@dataclass(frozen=True, slots=True)
class Component:
    """One connected group of OCR spans and ground-truth tokens."""

    span_indices: tuple[int, ...]
    token_indices: tuple[int, ...]
    relation: AlignmentRelation
    matched_chars: int = 0
    ocr_chars: int = 0
    gt_chars: int = 0
    origin: str = "in_region"
    """How this component came to exist, which decides whether it is a finding.

    ``in_region``
        the character DP inside an anchored region found it. An insertion here is
        positive evidence that the engine hallucinated text.
    ``orphan``
        the item never entered any anchored region. An insertion here says only that
        anchoring failed to place it — the engine may well have read it correctly and the
        aligner lost it. Reporting these as findings is how an aligner manufactures
        omissions: measured on 40 real FUNSD forms, 38% of "deletions" had an
        exact-text "insertion" twin on the same page.
    ``refused_block``
        the block exceeded the DP limit and alignment was refused. Not a finding at all.
    """

    @property
    def char_agreement(self) -> float:
        """Matched characters over the longer side."""
        denominator = max(self.ocr_chars, self.gt_chars)
        return self.matched_chars / denominator if denominator else 1.0


def classify_relation(n_spans: int, n_tokens: int) -> AlignmentRelation:
    """Type a component from its shape alone."""
    if n_spans and not n_tokens:
        return AlignmentRelation.OCR_INSERTION
    if n_tokens and not n_spans:
        return AlignmentRelation.OCR_DELETION
    if n_spans == 1 and n_tokens == 1:
        return AlignmentRelation.ONE_TO_ONE
    if n_spans > 1 and n_tokens == 1:
        return AlignmentRelation.SPLIT
    if n_spans == 1 and n_tokens > 1:
        return AlignmentRelation.MERGE
    return AlignmentRelation.MANY_TO_MANY


class _UnionFind:
    """Disjoint sets over span nodes ``(0, i)`` and token nodes ``(1, j)``."""

    __slots__ = ("parent",)

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def add(self, node: tuple[int, int]) -> None:
        self.parent.setdefault(node, node)

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:  # path compression
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        self.add(a)
        self.add(b)
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def induce_components(
    alignment: CharAlignment,
    ocr: Stream,
    gt: Stream,
    span_offset: int = 0,
    token_offset: int = 0,
) -> list[Component]:
    """Build components from a character alignment over two streams.

    ``span_offset`` and ``token_offset`` translate stream-local indices back to
    document-level ones, so a per-region alignment composes into a whole-page result.
    """
    union_find = _UnionFind()
    for index in range(len(ocr.owner)):
        if ocr.owner[index] >= 0:
            union_find.add((0, ocr.owner[index]))
    for index in range(len(gt.owner)):
        if gt.owner[index] >= 0:
            union_find.add((1, gt.owner[index]))

    matched: dict[tuple[int, int], int] = {}
    for op, a0, a1, b0, b1 in alignment.ops:
        if op is Op.INSERT or op is Op.DELETE:
            continue  # links nothing: no counterpart exists on the other side
        if op is Op.BLOCK:
            # A confusable block (rn -> m) is genuinely many-to-many; every character on
            # one side corresponds to the whole group on the other.
            character_pairs = [(i, j) for i in range(a0, a1) for j in range(b0, b1)]
        else:
            # MATCH and SUBSTITUTE runs advance in lockstep, so the correspondence is
            # positional. Linking every character of a run to every other would union all
            # spans on a line with all tokens on it and collapse the line into a single
            # MANY_TO_MANY component.
            character_pairs = list(zip(range(a0, a1), range(b0, b1), strict=False))

        for i, j in character_pairs:
            span_id, token_id = ocr.owner[i], gt.owner[j]
            if span_id < 0 or token_id < 0:
                continue  # a separator character belongs to no item
            union_find.union((0, span_id), (1, token_id))
            if op is Op.MATCH:
                key = (0, span_id)
                matched[key] = matched.get(key, 0) + 1

    groups: dict[tuple[int, int], tuple[set[int], set[int]]] = {}
    for node in union_find.parent:
        root = union_find.find(node)
        spans, tokens = groups.setdefault(root, (set(), set()))
        (spans if node[0] == 0 else tokens).add(node[1])

    components: list[Component] = []
    for spans, tokens in groups.values():
        span_tuple = tuple(sorted(spans))
        token_tuple = tuple(sorted(tokens))
        components.append(
            Component(
                span_indices=tuple(i + span_offset for i in span_tuple),
                token_indices=tuple(j + token_offset for j in token_tuple),
                relation=classify_relation(len(span_tuple), len(token_tuple)),
                matched_chars=sum(matched.get((0, i), 0) for i in span_tuple),
                ocr_chars=sum(ocr.owner.count(i) for i in span_tuple),
                gt_chars=sum(gt.owner.count(j) for j in token_tuple),
            )
        )

    components.sort(key=lambda c: (c.token_indices or (10**9,), c.span_indices or (10**9,)))
    return components


def orphan_components(
    span_indices: Sequence[int],
    token_indices: Sequence[int],
    ocr_lengths: Sequence[int],
    origin: str = "orphan",
) -> list[Component]:
    """Components for items that never entered an anchored region.

    These must appear in the artifact rather than being dropped for tidiness, but they
    are tagged ``orphan`` and are **not** findings: "the engine omitted this word" and
    "anchoring could not place this word" are different claims, and only the first is
    evidence about recognition.
    """
    components: list[Component] = []
    for index, length in zip(span_indices, ocr_lengths, strict=True):
        components.append(
            Component(
                span_indices=(index,),
                token_indices=(),
                relation=AlignmentRelation.OCR_INSERTION,
                ocr_chars=length,
                origin=origin,
            )
        )
    for index in token_indices:
        components.append(
            Component(
                span_indices=(),
                token_indices=(index,),
                relation=AlignmentRelation.OCR_DELETION,
                origin=origin,
            )
        )
    return components

"""Leave-one-engine-out folds, and the diagnostics that surround them.

Four protocols, each answering a different question:

``loeo_zero_shot``
    The headline. Fit and calibrate on the other engines, evaluate on the held-out one.
    Nothing from the held-out engine touches any fitted quantity.

``loeo_few_shot_recal``
    Weights stay frozen from the zero-shot fold; a small sample of held-out-engine
    documents — disjoint from the evaluation set — is permitted for recalibration only.
    Answers "how much target-engine data buys back the loss?"

``in_engine_oracle``
    Includes the held-out engine in fitting. Not a competing method: the gap between it
    and zero-shot **is the measured cost of engine shift**, which is what H1 is about.

``matched_in_engine``
    The recovery phase's corrected reference. Like ``in_engine_oracle`` it lets the target
    engine into fitting, but on a fit set of the **same size** — three engines, one of
    which is the target, formed by substituting the target for one donor engine. Read
    against ``loeo_zero_shot`` it isolates target-engine exposure from training volume,
    which ``in_engine_oracle`` conflated (see ``docs/h1_recovery/amendment4_confound.md``).

``diagnostic_doc_overlap``
    Deliberately violates the document split, to quantify how much the matched-source
    design would inflate results without it. Tagged ``leaky`` and barred from headline
    tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ocr_risk.io.hashing import hash_str
from ocr_risk.schemas.enums import SplitRole
from ocr_risk.splits.document_partition import DocumentPartition
from ocr_risk.splits.plan import SplitPlan

__all__ = [
    "PROTOCOLS",
    "MatchedPair",
    "few_shot_documents",
    "loeo_folds",
    "matched_pairs",
    "pairwise_folds",
]

PROTOCOLS = (
    "loeo_zero_shot",
    "loeo_few_shot_recal",
    "in_engine_oracle",
    "matched_in_engine",
    "diagnostic_doc_overlap",
)


def loeo_folds(
    engines: Sequence[str],
    partition: DocumentPartition,
    frame: pd.DataFrame,
    protocol: str = "loeo_zero_shot",
    few_shot_k: int = 0,
    few_shot_repeat: int = 0,
    selection_scope: str = "fit_only",
    allow_document_overlap: bool = False,
) -> list[SplitPlan]:
    """One fold per engine, under the named protocol.

    ``selection_scope`` and ``allow_document_overlap`` are threaded from the config
    rather than derived from the protocol name alone. A researcher who tunes against fold
    test metrics and honestly declares it must end up with a run record that says so —
    otherwise the audit check written to catch that can never fire, and the field is not
    "recorded but unenforced" but simply not recorded.
    """
    if protocol not in PROTOCOLS:
        msg = f"unknown protocol {protocol!r}; known: {', '.join(PROTOCOLS)}"
        raise ValueError(msg)
    if len(engines) < 2:
        msg = f"leave-one-engine-out needs at least 2 engines; got {list(engines)}"
        raise ValueError(msg)

    all_engines = frozenset(engines)
    plans: list[SplitPlan] = []

    for held_out in sorted(all_engines):
        few_shot: frozenset[str] = frozenset()
        if protocol == "loeo_few_shot_recal" and few_shot_k > 0:
            few_shot = few_shot_documents(partition, few_shot_k, seed=few_shot_repeat)

        plans.append(
            SplitPlan(
                fold_id=f"{protocol}:{held_out}"
                + (f":k{few_shot_k}r{few_shot_repeat}" if few_shot else ""),
                protocol=protocol,
                # The engine being evaluated is the same in every protocol; what differs
                # is whether it is also allowed into fitting.
                held_out_engines=frozenset({held_out}),
                all_engines=all_engines,
                partition=partition,
                frame=frame,
                allow_document_overlap=(
                    allow_document_overlap or protocol == "diagnostic_doc_overlap"
                ),
                include_held_out_in_fitting=protocol == "in_engine_oracle",
                selection_scope=selection_scope,
                few_shot_documents=few_shot,
            )
        )
    return plans


def few_shot_documents(partition: DocumentPartition, k: int, seed: int = 0) -> frozenset[str]:
    """Pick ``k`` calibration documents for target-engine recalibration.

    Drawn from ``D_cal``, never ``D_test``: recalibrating on pages that are then evaluated
    would make the few-shot result meaningless. Selection is by hash so a given ``(k,
    seed)`` is reproducible, and so the ``k`` documents for ``k=5`` are a subset of those
    for ``k=10`` — which makes the sweep a nested sequence rather than four unrelated
    samples.
    """
    candidates = sorted(partition.documents(SplitRole.CALIBRATE))
    ordered = sorted(candidates, key=lambda doc: (hash_str(f"fewshot:{seed}:{doc}"), doc))
    return frozenset(ordered[: max(0, k)])


def pairwise_folds(
    engines: Sequence[str], partition: DocumentPartition, frame: pd.DataFrame
) -> list[SplitPlan]:
    """Every ordered pair: fit on one engine, evaluate on another.

    Fills the cross-engine transfer matrix. With four engines the leave-one-out protocol
    gives only four points, which is thin ground for a claim about generalization; the
    twelve off-diagonal pairs say considerably more about which shifts are hard.
    """
    all_engines = frozenset(engines)
    plans: list[SplitPlan] = []
    for source in sorted(all_engines):
        for target in sorted(all_engines):
            if source == target:
                continue
            # Fit on `source` alone: every other engine is withheld, and only `target`
            # is evaluated. Restricting all_engines to the pair keeps the fit scope exact.
            plans.append(
                SplitPlan(
                    fold_id=f"pairwise:{source}->{target}",
                    protocol="pairwise_transfer",
                    held_out_engines=frozenset({target}),
                    all_engines=frozenset({source, target}),
                    partition=partition,
                    frame=frame,
                )
            )
    return plans


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """One held-out engine, read two ways on the same fit-set size.

    ``reference`` substitutes the target engine for ``donor``; ``zero_shot`` is the
    ordinary LOEO fold. Both fit on three engines, calibrate on the same documents, and
    evaluate on exactly the same rows — so the only systematic difference left between
    them is whether the target engine was among the three.
    """

    held_out: str
    donor: str
    zero_shot: SplitPlan
    reference: SplitPlan

    @property
    def pair_id(self) -> str:
        return f"{self.held_out}<-{self.donor}"


def matched_pairs(
    engines: Sequence[str],
    partition: DocumentPartition,
    frame: pd.DataFrame,
    selection_scope: str = "fit_only",
) -> list[MatchedPair]:
    """Every (held-out engine, donor engine) substitution, as paired plans.

    Averaging over the donor is deliberate. A single fixed donor would make the reference
    arm's fit set depend on one arbitrary choice, and with four engines that choice moves
    the fit population by up to 13% and its harmful fraction by up to 0.096 — enough to
    be mistaken for the effect being measured. Every donor is used once per target, so the
    donor's identity is marginalized rather than assumed away.
    """
    all_engines = frozenset(engines)
    if len(all_engines) < 3:
        msg = (
            f"a matched substitution needs at least 3 engines so the reference can drop "
            f"one and still hold two others; got {sorted(all_engines)}"
        )
        raise ValueError(msg)

    pairs: list[MatchedPair] = []
    for held_out in sorted(all_engines):
        others = sorted(all_engines - {held_out})
        zero_shot = SplitPlan(
            fold_id=f"loeo_zero_shot:{held_out}",
            protocol="loeo_zero_shot",
            held_out_engines=frozenset({held_out}),
            all_engines=all_engines,
            partition=partition,
            frame=frame,
            selection_scope=selection_scope,
        )
        for donor in others:
            reference = SplitPlan(
                fold_id=f"matched_in_engine:{held_out}<-{donor}",
                protocol="matched_in_engine",
                held_out_engines=frozenset({held_out}),
                all_engines=all_engines,
                partition=partition,
                frame=frame,
                include_held_out_in_fitting=True,
                fit_engines_override=frozenset({held_out, *(e for e in others if e != donor)}),
                selection_scope=selection_scope,
            )
            pairs.append(
                MatchedPair(
                    held_out=held_out,
                    donor=donor,
                    # A fresh zero-shot plan per pair would be identical; sharing one keeps
                    # the arm literally the same object, so "the same zero-shot fold" is a
                    # fact about the run rather than a claim about two constructions.
                    zero_shot=zero_shot,
                    reference=reference,
                )
            )
    return pairs

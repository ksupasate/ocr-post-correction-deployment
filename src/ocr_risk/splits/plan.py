"""``SplitPlan`` — the only gateway to labeled data.

Fitting code never reads a table. It asks the plan for a view, and the view is scoped to a
role: which engines it may see, which documents, and whether it may see labels at all.
Anything outside that scope raises :class:`LeakageError` rather than returning rows.

The protocol both axes must satisfy, always together:

======================  ==============================  ==================
role                    engines                         documents
======================  ==============================  ==================
``fit``                 all except the held-out engine  ``D_fit``
``dev``                 all except the held-out engine  ``D_fit`` subset
``calibrate``           all except the held-out engine  ``D_cal``
``evaluate``            **only** the held-out engine    ``D_test``
======================  ==============================  ==================

The evaluate view deliberately does **not** expose labels through the fitting API. Labels
for evaluation are read by :mod:`ocr_risk.metrics`, which the layering test forbids
``verify``, ``calibrate``, and ``risk`` from importing — so a threshold cannot be chosen
against test outcomes without the violation showing up as an import error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from ocr_risk.io.hashing import stable_string_set_hash
from ocr_risk.schemas.enums import SplitRole
from ocr_risk.schemas.run_record import SplitDescriptor
from ocr_risk.splits.document_partition import DocumentPartition

__all__ = [
    "CalibrationView",
    "EvaluationView",
    "FitView",
    "LeakageError",
    "SplitPlan",
    "SplitView",
]


class LeakageError(RuntimeError):
    """Raised when data outside a role's scope is requested.

    Always a bug, never a data condition: if this fires, some code reached past the split
    plan, and any number it produced is invalid.
    """


@dataclass(frozen=True, slots=True)
class SplitView:
    """A role-scoped window onto the experiment's rows.

    ``frame`` is already filtered. The scope is carried alongside so the view can prove
    what it was allowed to see, and so the audit can re-derive it independently.
    """

    role: SplitRole
    fold_id: str
    engines: frozenset[str]
    documents: frozenset[str]
    frame: pd.DataFrame
    exposes_labels: bool

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def is_empty(self) -> bool:
        return self.frame.empty

    def labels(self, column: str) -> pd.Series:
        """Read a label column, if this role is permitted to see one.

        ``evaluate`` is not. Selecting a threshold or fitting a calibrator against test
        outcomes is the leak this whole module exists to prevent, so it raises here rather
        than being caught later by review.
        """
        if not self.exposes_labels:
            msg = (
                f"role {self.role.value!r} may not read label column {column!r}. "
                "Fitting or threshold selection against evaluation labels is leakage "
                "(vectors L1 and L2); evaluation metrics are computed in ocr_risk.metrics "
                "from immutable artifacts instead."
            )
            raise LeakageError(msg)
        if column not in self.frame.columns:
            msg = f"label column {column!r} is not present in this view"
            raise KeyError(msg)
        return self.frame[column]

    def require_columns(self, columns: Sequence[str]) -> None:
        missing = [c for c in columns if c not in self.frame.columns]
        if missing:
            msg = f"view for role {self.role.value!r} is missing columns {missing}"
            raise KeyError(msg)

    def digest(self) -> str:
        return stable_string_set_hash(self.documents)


# Aliases that make a function signature state its own scope requirement. A calibrator
# typed to accept a CalibrationView cannot be handed an evaluation view by accident.
FitView = SplitView
CalibrationView = SplitView
EvaluationView = SplitView


@dataclass(slots=True)
class SplitPlan:
    """Binds the engine axis and the document axis for one fold."""

    fold_id: str
    protocol: str
    held_out_engines: frozenset[str]
    all_engines: frozenset[str]
    partition: DocumentPartition
    frame: pd.DataFrame
    allow_document_overlap: bool = False
    """Deliberate contamination, for the ``doc_overlap`` diagnostic only. Marks the run
    ``leaky`` and bars it from every headline table."""
    include_held_out_in_fitting: bool = False
    """The ``in_engine_oracle`` diagnostic: the evaluated engine is *also* used for
    fitting. Not a competing method — the gap between it and zero-shot is the measured
    cost of engine shift, which is what H1 is about. Recorded in the protocol so it can
    never be mistaken for a zero-shot result."""
    fit_engines_override: frozenset[str] | None = None
    """The exact engine set permitted in fitting, when the protocol needs a custom one.

    The matched-handicap reference needs a *three*-engine fit set containing the target
    engine, because ``in_engine_oracle`` fits on four while zero-shot fits on three — a
    23-39% training-volume advantage that the H1 contrast attributed to engine exposure.
    Setting this explicitly is what makes the two arms the same size."""
    selection_scope: str = "fit_only"
    few_shot_documents: frozenset[str] = field(default_factory=frozenset)
    """Held-out-engine documents permitted for few-shot recalibration. Empty in the
    zero-shot protocol."""

    def __post_init__(self) -> None:
        unknown = self.held_out_engines - self.all_engines
        if unknown:
            msg = f"held-out engines {sorted(unknown)} are not among the experiment's engines"
            raise ValueError(msg)
        if self.fit_engines_override is not None:
            outside = self.fit_engines_override - self.all_engines
            if outside:
                msg = f"fit engines {sorted(outside)} are not among the experiment's engines"
                raise ValueError(msg)
            target = self.fit_engines_override & self.held_out_engines
            if target and not self.include_held_out_in_fitting:
                msg = (
                    f"fit engines {sorted(target)} include the held-out engine while the "
                    "plan does not declare in-engine fitting. A reference arm that trains "
                    "on its own target must say so, or it reads as a zero-shot result."
                )
                raise LeakageError(msg)
        self.partition.assert_disjoint()
        overlap = self.few_shot_documents & self.partition.documents(SplitRole.EVALUATE)
        if overlap:
            msg = (
                f"few-shot documents overlap the evaluation split ({len(overlap)} documents). "
                "Recalibrating on pages that are then evaluated is leakage."
            )
            raise LeakageError(msg)

    @property
    def train_engines(self) -> frozenset[str]:
        if self.fit_engines_override is not None:
            return self.fit_engines_override
        if self.include_held_out_in_fitting:
            return self.all_engines
        return self.all_engines - self.held_out_engines

    def engines_for(self, role: SplitRole) -> frozenset[str]:
        if role is SplitRole.EVALUATE:
            return self.held_out_engines
        return self.train_engines

    def documents_for(self, role: SplitRole) -> frozenset[str]:
        if self.allow_document_overlap and role in (SplitRole.FIT, SplitRole.DEV):
            # The leaky diagnostic: fit on every document, to measure how much the
            # matched-source design would inflate results without a document split.
            return frozenset(self.partition.role_of)
        if role is SplitRole.DEV:
            return self.partition.documents(SplitRole.FIT)
        if role is SplitRole.CALIBRATE and self.few_shot_documents:
            # Few-shot recalibration: weights stay frozen, and a small held-out-engine
            # sample disjoint from D_test is permitted.
            return self.partition.documents(SplitRole.CALIBRATE) | self.few_shot_documents
        return self.partition.documents(role)

    def scope_for(self, role: SplitRole) -> tuple[frozenset[str], frozenset[str]]:
        """The ``(engines, documents)`` a role may actually reach.

        One definition, used by :meth:`view`, :meth:`descriptor` and
        :meth:`assert_no_leakage` alike. They previously each re-derived the scope, and
        they did not agree: the descriptor recorded the widened few-shot calibration
        engines while the guard re-read ``engines_for``, which cannot widen — so the guard
        was checking an expression that was unsatisfiable by construction and could never
        fire. A guard nobody can make fire is documentation, not enforcement.
        """
        engines = self.engines_for(role)
        if role is SplitRole.CALIBRATE and self.few_shot_documents:
            engines = engines | self.held_out_engines
        return frozenset(engines), self.documents_for(role)

    def view(self, role: SplitRole) -> SplitView:
        """Mint a role-scoped view. The only supported way to get rows."""
        engines, documents = self.scope_for(role)

        if role is SplitRole.CALIBRATE and self.few_shot_documents:
            # In few-shot mode the calibration set spans train engines on D_cal plus the
            # held-out engine on the few-shot documents. The cross terms -- a train engine
            # on a few-shot page, the held-out engine on D_cal -- are excluded, so the
            # mask is not the product of the two sets.
            mask = (
                self.frame["engine_id"].isin(self.train_engines)
                & self.frame["document_id"].isin(self.partition.documents(SplitRole.CALIBRATE))
            ) | (
                self.frame["engine_id"].isin(self.held_out_engines)
                & self.frame["document_id"].isin(self.few_shot_documents)
            )
        else:
            mask = self.frame["engine_id"].isin(engines) & self.frame["document_id"].isin(documents)

        return SplitView(
            role=role,
            fold_id=self.fold_id,
            engines=engines,
            documents=documents,
            frame=self.frame.loc[mask].reset_index(drop=True),
            exposes_labels=role is not SplitRole.EVALUATE,
        )

    def descriptor(self) -> SplitDescriptor:
        """The auditable statement of this fold, for the run record.

        Engine sets are read off :meth:`scope_for`, not off ``engines_for``. Under few-shot
        recalibration the calibration scope deliberately widens to include the held-out
        engine while ``engines_for`` still returns the train engines — so recording the
        latter would produce a run record that understates its own calibration scope, and
        the audit check written to catch exactly that could never fire.
        """
        fit_engines, fit_documents = self.scope_for(SplitRole.FIT)
        calibrate_engines, calibrate_documents = self.scope_for(SplitRole.CALIBRATE)
        evaluate_engines, evaluate_documents = self.scope_for(SplitRole.EVALUATE)
        return SplitDescriptor(
            fold_id=self.fold_id,
            protocol=self.protocol,
            held_out_engines=tuple(sorted(self.held_out_engines)),
            fit_engines=tuple(sorted(fit_engines)),
            calibrate_engines=tuple(sorted(calibrate_engines)),
            evaluate_engines=tuple(sorted(evaluate_engines)),
            n_fit_evaluate_shared_documents=len(fit_documents & evaluate_documents),
            n_calibrate_evaluate_shared_documents=len(calibrate_documents & evaluate_documents),
            fit_documents_sha256=stable_string_set_hash(fit_documents),
            calibrate_documents_sha256=stable_string_set_hash(calibrate_documents),
            evaluate_documents_sha256=stable_string_set_hash(evaluate_documents),
            n_fit_documents=len(fit_documents),
            n_calibrate_documents=len(calibrate_documents),
            n_evaluate_documents=len(evaluate_documents),
            document_partition_sha256=self.partition.partition_sha256,
            selection_scope=self.selection_scope,
            leaky=self.allow_document_overlap,
        )

    def assert_no_leakage(self) -> None:
        """Runtime check of both axes. The audit re-derives this independently."""
        # Few-shot recalibration is the ONE sanctioned way target-engine rows reach the
        # calibration view. A plan labelled zero-shot that carries them is either a
        # mislabelled few-shot run or a leak, and both are worse than an error: the whole
        # zero-shot invariant otherwise rests on a single expression in loeo_folds, so any
        # other construction path could produce one and nothing would object.
        if self.few_shot_documents and self.protocol != "loeo_few_shot_recal":
            msg = (
                f"protocol {self.protocol!r} carries {len(self.few_shot_documents)} "
                "few-shot document(s), which put the held-out engine into the calibration "
                "view. Only 'loeo_few_shot_recal' may do that."
            )
            raise LeakageError(msg)

        # Read the SCOPE, not the defining expression. Both loops below check what a view
        # of that role would actually expose, so a subclass or a future edit that widens
        # `scope_for` is caught here instead of sailing through a check that recomputes
        # the same narrow answer it is meant to be verifying.
        evaluate_engines, test_documents = self.scope_for(SplitRole.EVALUATE)
        if not self.include_held_out_in_fitting:
            for role in (SplitRole.FIT, SplitRole.CALIBRATE):
                shared_engines = self.scope_for(role)[0] & evaluate_engines
                if shared_engines and not self.few_shot_documents:
                    msg = (
                        f"{role.value} shares engine(s) {sorted(shared_engines)} with the "
                        "held-out engine; the zero-shot protocol forbids it (vector L1/L2)"
                    )
                    raise LeakageError(msg)

        if self.allow_document_overlap:
            return  # deliberate, and flagged leaky in the descriptor

        for role in (SplitRole.FIT, SplitRole.CALIBRATE):
            shared = self.scope_for(role)[1] & test_documents
            if shared:
                msg = (
                    f"{role.value} shares {len(shared)} document(s) with the evaluation "
                    f"split, e.g. {sorted(shared)[:3]}. The benchmark is matched-source, "
                    "so holding out an engine does not hold out the page (vector L4)."
                )
                raise LeakageError(msg)

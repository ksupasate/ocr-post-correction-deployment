"""The provenance envelope written by every pipeline stage.

The design test for this module is: *given only a ``run_record.json``, can someone
re-derive the same bytes months later?* Every field exists because its absence would make
the answer no. If reproduction depends on something a person remembers rather than
something recorded here, that is a defect in this schema.
"""

from __future__ import annotations

from pydantic import Field

from ocr_risk.schemas.base import RecordModel
from ocr_risk.schemas.enums import StageName

__all__ = [
    "ArtifactRef",
    "EnvironmentState",
    "GitState",
    "RunRecord",
    "SplitDescriptor",
]


class GitState(RecordModel):
    """Code identity at run time.

    ``dirty`` is recorded rather than refused: refusing to run on a dirty tree would push
    researchers toward not recording state at all. A dirty run is reproducible only
    best-effort, and saying so is more useful than pretending the commit identifies it.
    """

    commit: str
    branch: str
    dirty: bool
    diff_sha256: str | None = None
    """Hash of the uncommitted diff, so a dirty run is at least identifiable."""
    remote: str | None = None


class EnvironmentState(RecordModel):
    """Interpreter, platform, hardware, and dependency identity."""

    python_version: str
    platform: str
    machine: str
    cpu_count: int = Field(ge=1)
    total_memory_bytes: int | None = None
    uv_lock_sha256: str | None = None
    package_digest: str
    """Canonical hash over the installed distribution name/version pairs."""
    thread_limits: dict[str, str] = Field(default_factory=dict)
    """BLAS/OMP thread caps in force. Float reduction order depends on these, so they are
    part of what determines whether a rerun is bit-identical."""


class ArtifactRef(RecordModel):
    """A pointer to one file produced or consumed by a run.

    The pair of ``run_id`` and ``sha256`` is what turns the artifact tree into a
    verifiable DAG: a consumer records the exact bytes it read, so a silently regenerated
    upstream artifact surfaces as a broken link instead of a wrong number.
    """

    run_id: str
    stage: StageName
    path: str
    sha256: str
    n_rows: int | None = None


class SplitDescriptor(RecordModel):
    """Exactly which engines and documents were reachable in each role.

    This is the auditable statement of the leave-one-engine-out protocol. The post-hoc
    leakage audit re-derives disjointness from these hashes without trusting the runtime
    guard that was active when the run happened.
    """

    fold_id: str
    protocol: str
    """e.g. ``loeo_zero_shot``, ``loeo_few_shot_recal``, ``in_engine_oracle``."""
    held_out_engines: tuple[str, ...]
    fit_engines: tuple[str, ...]
    calibrate_engines: tuple[str, ...]
    evaluate_engines: tuple[str, ...]
    fit_documents_sha256: str
    calibrate_documents_sha256: str
    evaluate_documents_sha256: str
    n_fit_documents: int = Field(ge=0)
    n_calibrate_documents: int = Field(ge=0)
    n_evaluate_documents: int = Field(ge=0)
    n_fit_evaluate_shared_documents: int = 0
    """Documents reachable in BOTH the fit and evaluate roles. Recorded as a count rather
    than inferred from digests: two role sets overlapping *partially* have different
    digests, so a digest comparison sees nothing wrong — and partial overlap is the shape
    a real bug takes. Must be zero outside a declared leaky diagnostic."""
    n_calibrate_evaluate_shared_documents: int = 0
    """As above, for the calibration and evaluation roles."""
    document_partition_sha256: str
    """Identity of the global document partition. Must be *identical across every fold*;
    a differing value means the partition was redrawn and documents rotated across roles."""
    selection_scope: str = "fit_only"
    """Where model/hyperparameter choices were made. Anything other than ``fit_only`` or
    ``dev_only`` disqualifies the run from headline tables."""
    leaky: bool = False
    """True only for deliberately contaminated diagnostic runs, which exist to measure the
    inflation that contamination causes."""


class RunRecord(RecordModel):
    """Complete provenance for one stage execution."""

    run_id: str
    stage: StageName
    experiment: str | None = None
    created_at_utc: str
    duration_seconds: float = Field(ge=0.0)
    code_version: str
    code_sha256: str
    """Hash over the ``src/ocr_risk`` source tree; catches uncommitted logic changes."""
    config_sha256: str
    config_path: str
    seeds: dict[str, int] = Field(default_factory=dict)
    git: GitState
    environment: EnvironmentState
    dataset_ids: tuple[str, ...] = ()
    dataset_manifest_sha256: str | None = None
    engine_fingerprints: dict[str, str] = Field(default_factory=dict)
    model_ids: dict[str, str] = Field(default_factory=dict)
    """Identifier per role, e.g. ``{"verifier": ..., "candidate_generator": ...}``."""
    calibrator_id: str | None = None
    split: SplitDescriptor | None = None
    folds: tuple[SplitDescriptor, ...] = ()
    """Every fold this run executed. ``split`` is the first of them, kept for
    single-fold stages; the audit reads THIS, because auditing one descriptor out of
    twenty-eight makes the cross-fold partition-stability check vacuous — a partition
    redrawn per fold would be recorded faithfully and reported as stable."""
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    cache_key: str | None = None
    """Hash of (input digests, config, code version). A later run with the same key may
    reuse this run's outputs instead of recomputing them."""
    outputs_digest: str | None = None
    """Hash over the ``outputs.sha256`` manifest. The determinism check compares this
    between two runs; timestamps are deliberately excluded so a rerun of identical work
    produces an identical value."""
    synthetic: bool = False
    """True when any input is synthetic. Propagates through the DAG so a downstream figure
    cannot lose the label."""
    leaky: bool = False
    notes: str | None = None

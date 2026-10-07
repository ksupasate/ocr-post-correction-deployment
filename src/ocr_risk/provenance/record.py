"""Assembly of run identifiers and provenance envelopes."""

from __future__ import annotations

from datetime import UTC, datetime

from ocr_risk.io.hashing import short
from ocr_risk.provenance.envcapture import capture_environment, code_sha256
from ocr_risk.provenance.gitstate import capture_git_state
from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.run_record import ArtifactRef, RunRecord, SplitDescriptor

__all__ = ["build_run_record", "make_run_id", "utc_now_iso"]


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(stage: StageName, config_sha256: str, git_commit: str) -> str:
    """``<stage>-<utc>-<config8>-<git8>``.

    Config and commit are both in the identifier so two runs that differ only in
    configuration, or only in code, are distinguishable at a glance in a directory listing.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage.value}-{stamp}-{short(config_sha256)}-{short(git_commit)}"


def build_run_record(
    *,
    run_id: str,
    stage: StageName,
    config_sha256: str,
    config_path: str,
    duration_seconds: float,
    experiment: str | None = None,
    seeds: dict[str, int] | None = None,
    dataset_ids: tuple[str, ...] = (),
    dataset_manifest_sha256: str | None = None,
    engine_fingerprints: dict[str, str] | None = None,
    model_ids: dict[str, str] | None = None,
    calibrator_id: str | None = None,
    split: SplitDescriptor | None = None,
    folds: tuple[SplitDescriptor, ...] = (),
    inputs: tuple[ArtifactRef, ...] = (),
    outputs: tuple[ArtifactRef, ...] = (),
    synthetic: bool = False,
    leaky: bool = False,
    notes: str | None = None,
    code_version: str | None = None,
) -> RunRecord:
    """Collect git, environment, and stage facts into one immutable envelope."""
    from ocr_risk import __version__

    git = capture_git_state()
    return RunRecord(
        run_id=run_id,
        stage=stage,
        experiment=experiment,
        created_at_utc=utc_now_iso(),
        duration_seconds=duration_seconds,
        code_version=code_version or __version__,
        code_sha256=code_sha256(),
        config_sha256=config_sha256,
        config_path=config_path,
        seeds=seeds or {},
        git=git,
        environment=capture_environment(),
        dataset_ids=dataset_ids,
        dataset_manifest_sha256=dataset_manifest_sha256,
        engine_fingerprints=engine_fingerprints or {},
        model_ids=model_ids or {},
        calibrator_id=calibrator_id,
        split=split,
        folds=folds,
        inputs=inputs,
        outputs=outputs,
        # Synthetic provenance is monotone: if any input was synthetic, so is the output.
        # Without this a figure several stages downstream could quietly lose the label.
        synthetic=synthetic,
        leaky=leaky or (split.leaky if split else False),
        notes=notes,
    )

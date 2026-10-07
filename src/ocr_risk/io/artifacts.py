"""The artifact store: run-scoped, immutable, content-hashed stage outputs.

Every stage execution produces exactly one run directory. The directory is the unit of
provenance: it records what code and config produced it, which upstream artifacts it read
(by hash), and the hash of everything it wrote. Walking ``inputs.json`` upward turns the
whole tree into a verifiable DAG, which is what lets a figure be traced back to a source
manifest months later.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
import yaml

from ocr_risk.io.hashing import canonical_hash, canonical_json, faithful_json, file_sha256
from ocr_risk.io.parquet import read_records, read_table, write_records
from ocr_risk.io.paths import artifact_root
from ocr_risk.provenance.record import build_run_record, make_run_id
from ocr_risk.schemas.arrow import TableSpec, get_table
from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.run_record import ArtifactRef, RunRecord, SplitDescriptor

__all__ = ["ArtifactStore", "ImmutableRunError", "RunWriter"]

RUN_RECORD = "run_record.json"
OUTPUTS_MANIFEST = "outputs.sha256"
INPUTS_MANIFEST = "inputs.json"
RESOLVED_CONFIG = "config.resolved.yaml"
_MAX_RUN_ID_ATTEMPTS = 1000


class ImmutableRunError(RuntimeError):
    """Raised on an attempt to modify an already-finalized run directory."""


class RunWriter:
    """Open, writable handle to one stage execution.

    Used as a context manager; the provenance envelope is written on clean exit. An
    exception leaves the directory without a ``run_record.json``, which marks it as
    incomplete so a partial run can never be mistaken for a finished one.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        stage: StageName,
        run_id: str,
        directory: Path,
        config_sha256: str,
        resolved_config: dict[str, Any],
        experiment: str | None,
        cache_key: str | None,
    ) -> None:
        self.store = store
        self.stage = stage
        self.run_id = run_id
        self.directory = directory
        self.config_sha256 = config_sha256
        self.resolved_config = resolved_config
        self.experiment = experiment
        self.cache_key = cache_key

        self._started = time.monotonic()
        self._outputs: list[ArtifactRef] = []
        self._inputs: list[ArtifactRef] = []
        self._seeds: dict[str, int] = {}
        self._engine_fingerprints: dict[str, str] = {}
        self._model_ids: dict[str, str] = {}
        self._dataset_ids: tuple[str, ...] = ()
        self._dataset_manifest_sha256: str | None = None
        self._calibrator_id: str | None = None
        self._split: SplitDescriptor | None = None
        self._folds: tuple[SplitDescriptor, ...] = ()
        self._synthetic = False
        self._leaky = False
        self._notes: str | None = None
        self._finalized = False
        self.record: RunRecord | None = None

    # --- configuration -----------------------------------------------------------
    def add_input(self, ref: ArtifactRef) -> None:
        self._inputs.append(ref)

    def add_inputs(self, refs: list[ArtifactRef]) -> None:
        self._inputs.extend(refs)

    def inherit_from(self, record: RunRecord) -> None:
        """Consume an upstream run: record its outputs as inputs and inherit its taint.

        ``synthetic`` and ``leaky`` propagate forward monotonically. A downstream stage
        cannot launder a synthetic or deliberately contaminated input into a clean output.
        """
        self._inputs.extend(record.outputs)
        self._synthetic = self._synthetic or record.synthetic
        self._leaky = self._leaky or record.leaky

    def set_seeds(self, seeds: dict[str, int]) -> None:
        self._seeds.update(seeds)

    def set_split(self, split: SplitDescriptor) -> None:
        self._split = split
        self._leaky = self._leaky or split.leaky

    def set_folds(self, folds: Sequence[SplitDescriptor]) -> None:
        """Record every fold, not only the first.

        The post-hoc audit re-derives isolation from the run record. If the record names
        one fold out of many, the audit's cross-fold checks have nothing to compare.
        """
        self._folds = tuple(folds)
        if folds:
            self.set_split(folds[0])
        for fold in folds:
            self._leaky = self._leaky or fold.leaky

    def set_datasets(
        self, dataset_ids: tuple[str, ...], manifest_sha256: str | None = None
    ) -> None:
        self._dataset_ids = dataset_ids
        self._dataset_manifest_sha256 = manifest_sha256

    def set_engine_fingerprints(self, fingerprints: dict[str, str]) -> None:
        self._engine_fingerprints.update(fingerprints)

    def set_model_ids(self, model_ids: dict[str, str]) -> None:
        self._model_ids.update(model_ids)

    def set_calibrator(self, calibrator_id: str) -> None:
        self._calibrator_id = calibrator_id

    def mark_synthetic(self, synthetic: bool = True) -> None:
        self._synthetic = self._synthetic or synthetic

    def mark_leaky(self, leaky: bool = True, note: str | None = None) -> None:
        self._leaky = self._leaky or leaky
        if note:
            self._notes = f"{self._notes}\n{note}" if self._notes else note

    def set_notes(self, notes: str) -> None:
        self._notes = notes

    # --- writing -----------------------------------------------------------------
    def _guard(self) -> None:
        if self._finalized:
            msg = f"run {self.run_id} is finalized; artifacts are immutable"
            raise ImmutableRunError(msg)

    def write_records(self, table_name: str, records: list[Any]) -> ArtifactRef:
        """Write a canonical table as Parquet and register it as an output."""
        self._guard()
        spec: TableSpec = get_table(table_name)
        path = self.directory / f"{table_name}.parquet"
        write_records(path, spec, records)
        return self._register(path, n_rows=len(records))

    def write_arrow(self, table_name: str, table: pa.Table) -> ArtifactRef:
        self._guard()
        spec = get_table(table_name)
        path = self.directory / f"{table_name}.parquet"
        from ocr_risk.io.parquet import write_table

        write_table(path, spec, table)
        return self._register(path, n_rows=table.num_rows)

    def write_json(self, filename: str, payload: Any) -> ArtifactRef:
        """Write a JSON artifact, keeping numbers as numbers.

        faithful_json rather than canonical_json. Canonical encoding writes floats as
        *strings* so that numerically equal values always hash identically -- right for
        hashing, wrong for an artifact that will be read back and used arithmetically. It
        is still deterministic (sorted keys, fixed separators), so the content hash is
        stable and the determinism check still means what it says.
        """
        self._guard()
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(faithful_json(payload) + "\n", encoding="utf-8")
        return self._register(path)

    def write_text(self, filename: str, text: str) -> ArtifactRef:
        self._guard()
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return self._register(path)

    def write_bytes(self, filename: str, data: bytes) -> ArtifactRef:
        self._guard()
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self._register(path)

    def _register(self, path: Path, n_rows: int | None = None) -> ArtifactRef:
        ref = ArtifactRef(
            run_id=self.run_id,
            stage=self.stage,
            path=path.relative_to(self.directory).as_posix(),
            sha256=file_sha256(path),
            n_rows=n_rows,
        )
        self._outputs.append(ref)
        return ref

    # --- finalization ------------------------------------------------------------
    def __enter__(self) -> Self:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / RESOLVED_CONFIG).write_text(
            yaml.safe_dump(self.resolved_config, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # Leave the directory without a run record: incomplete work must not be
            # discoverable as a finished run.
            return
        self.finalize()

    def finalize(self) -> RunRecord:
        """Write the manifests and the provenance envelope."""
        if self._finalized:
            msg = f"run {self.run_id} already finalized"
            raise ImmutableRunError(msg)

        manifest_lines = sorted(f"{ref.sha256}  {ref.path}" for ref in self._outputs)
        manifest_text = "\n".join(manifest_lines) + ("\n" if manifest_lines else "")
        (self.directory / OUTPUTS_MANIFEST).write_text(manifest_text, encoding="utf-8")

        (self.directory / INPUTS_MANIFEST).write_text(
            canonical_json([ref.model_dump(mode="python") for ref in self._inputs]) + "\n",
            encoding="utf-8",
        )

        record = build_run_record(
            run_id=self.run_id,
            stage=self.stage,
            config_sha256=self.config_sha256,
            config_path=RESOLVED_CONFIG,
            duration_seconds=max(0.0, time.monotonic() - self._started),
            experiment=self.experiment,
            seeds=self._seeds,
            dataset_ids=self._dataset_ids,
            dataset_manifest_sha256=self._dataset_manifest_sha256,
            engine_fingerprints=self._engine_fingerprints,
            model_ids=self._model_ids,
            calibrator_id=self._calibrator_id,
            split=self._split,
            folds=self._folds,
            inputs=tuple(self._inputs),
            outputs=tuple(self._outputs),
            synthetic=self._synthetic,
            leaky=self._leaky,
            notes=self._notes,
        )
        record = record.model_copy(
            update={"cache_key": self.cache_key, "outputs_digest": canonical_hash(manifest_lines)}
        )
        (self.directory / RUN_RECORD).write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        self._finalized = True
        self.record = record
        return record


class ArtifactStore:
    """Discovery and read access over the artifact tree."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        # Resolved per access so tests can redirect OCR_RISK_ARTIFACT_ROOT.
        return self._root if self._root is not None else artifact_root()

    def begin(
        self,
        *,
        stage: StageName,
        config_sha256: str,
        resolved_config: dict[str, Any],
        experiment: str | None = None,
        git_commit: str = "0" * 40,
        cache_key: str | None = None,
    ) -> RunWriter:
        # Re-running the same stage with the same config and commit inside one second
        # (a --force rerun, or several folds dispatched together) would otherwise collide
        # on the timestamped id and be refused as an already-finalized directory.
        base_id = make_run_id(stage, config_sha256, git_commit)
        run_id, directory, attempt = base_id, self.root / stage.value / base_id, 1
        while directory.exists():
            attempt += 1
            run_id = f"{base_id}-{attempt}"
            directory = self.root / stage.value / run_id
            if attempt > _MAX_RUN_ID_ATTEMPTS:
                msg = f"could not allocate a unique run id for {base_id}"
                raise ImmutableRunError(msg)
        return RunWriter(
            self,
            stage=stage,
            run_id=run_id,
            directory=directory,
            config_sha256=config_sha256,
            resolved_config=resolved_config,
            experiment=experiment,
            cache_key=cache_key,
        )

    # --- discovery ---------------------------------------------------------------
    def run_dir(self, record: RunRecord) -> Path:
        return self.root / record.stage.value / record.run_id

    def iter_records(self, stage: StageName | None = None) -> Iterator[RunRecord]:
        """Yield finalized run records, oldest first by run_id (which embeds the time)."""
        stages = (
            [stage.value] if stage else [d.name for d in sorted(self.root.glob("*")) if d.is_dir()]
        )
        for stage_name in stages:
            stage_dir = self.root / stage_name
            if not stage_dir.is_dir():
                continue
            for run_dir in sorted(stage_dir.iterdir()):
                path = run_dir / RUN_RECORD
                if path.is_file():
                    yield RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def find_runs(
        self,
        stage: StageName | None = None,
        experiment: str | None = None,
        cache_key: str | None = None,
    ) -> list[RunRecord]:
        return [
            r
            for r in self.iter_records(stage)
            if (experiment is None or r.experiment == experiment)
            and (cache_key is None or r.cache_key == cache_key)
        ]

    def latest(self, stage: StageName, experiment: str | None = None) -> RunRecord | None:
        runs = self.find_runs(stage, experiment)
        return runs[-1] if runs else None

    def find_cached(self, stage: StageName, cache_key: str) -> RunRecord | None:
        """Return a completed run with a matching cache key, if one exists and is intact."""
        for record in self.find_runs(stage, cache_key=cache_key):
            if self.verify_outputs(record):
                return record
        return None

    def load_record(self, run_id: str, stage: StageName | None = None) -> RunRecord:
        for record in self.iter_records(stage):
            if record.run_id == run_id:
                return record
        msg = f"no finalized run with id {run_id!r} under {self.root}"
        raise FileNotFoundError(msg)

    # --- reading -----------------------------------------------------------------
    def path_of(self, record: RunRecord, ref: ArtifactRef) -> Path:
        return self.run_dir(record) / ref.path

    def output_ref(self, record: RunRecord, filename: str) -> ArtifactRef:
        for ref in record.outputs:
            if ref.path == filename:
                return ref
        available = ", ".join(r.path for r in record.outputs) or "(none)"
        msg = f"run {record.run_id} has no output {filename!r}; available: {available}"
        raise FileNotFoundError(msg)

    def read_table(self, record: RunRecord, table_name: str) -> pa.Table:
        ref = self.output_ref(record, f"{table_name}.parquet")
        return read_table(self.path_of(record, ref), get_table(table_name))

    def read_records(self, record: RunRecord, table_name: str) -> list[Any]:
        ref = self.output_ref(record, f"{table_name}.parquet")
        return read_records(self.path_of(record, ref), get_table(table_name))

    def read_json(self, record: RunRecord, filename: str) -> Any:
        ref = self.output_ref(record, filename)
        return json.loads(self.path_of(record, ref).read_text(encoding="utf-8"))

    def verify_outputs(self, record: RunRecord) -> bool:
        """Re-hash every declared output and confirm the bytes are unchanged."""
        run_dir = self.run_dir(record)
        for ref in record.outputs:
            path = run_dir / ref.path
            if not path.is_file() or file_sha256(path) != ref.sha256:
                return False
        return True

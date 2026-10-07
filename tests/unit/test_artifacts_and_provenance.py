"""Artifact immutability, provenance capture, and DAG traversal.

The property under test throughout: an artifact tree must be able to answer "where did
this come from, and are those bytes still what they claimed to be?" without trusting
anyone's memory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ocr_risk.io.artifacts import ArtifactStore, ImmutableRunError
from ocr_risk.io.raw_store import ImmutableWriteError, RawStore
from ocr_risk.provenance import (
    build_lineage,
    capture_environment,
    capture_git_state,
    render_lineage,
)
from ocr_risk.provenance.record import make_run_id
from ocr_risk.schemas import SourceDocument, StageName
from ocr_risk.schemas.run_record import SplitDescriptor


def _doc(n: int) -> SourceDocument:
    return SourceDocument(
        document_id=f"doc-{n:04d}",
        dataset_id="synthetic",
        image_path=f"raw/source/synthetic/doc-{n:04d}.png",
        image_sha256=f"{n:064d}",
        page_index=0,
        width=100,
        height=200,
        gt_text="hello world",
        gt_policy="synthetic_exact",
        has_gt_geometry=False,
        n_gt_tokens=2,
        license_id="synthetic-generated",
    )


@pytest.fixture
def store(isolated_env: Path) -> ArtifactStore:
    return ArtifactStore()


# --- run writing ----------------------------------------------------------------------
def test_run_writes_all_provenance_files(store: ArtifactStore) -> None:
    with store.begin(
        stage=StageName.MANIFEST, config_sha256="a" * 64, resolved_config={"name": "t"}
    ) as run:
        run.write_records("documents", [_doc(1), _doc(2)])
        run.set_datasets(("synthetic",))
        run.mark_synthetic()

    record = run.record
    assert record is not None
    directory = store.run_dir(record)
    for name in ("run_record.json", "outputs.sha256", "inputs.json", "config.resolved.yaml"):
        assert (directory / name).is_file(), f"missing provenance file {name}"

    assert record.synthetic is True
    assert record.outputs[0].n_rows == 2
    assert record.environment.package_digest
    assert record.code_sha256 != "unknown"


def test_failed_run_leaves_no_run_record(store: ArtifactStore) -> None:
    """A crashed stage must not be discoverable as a finished run."""
    with (
        pytest.raises(RuntimeError, match="stage blew up"),
        store.begin(stage=StageName.SITES, config_sha256="b" * 64, resolved_config={}) as run,
    ):
        run.write_records("documents", [_doc(1)])
        msg = "stage blew up"
        raise RuntimeError(msg)

    assert not (run.directory / "run_record.json").exists()
    assert store.find_runs(StageName.SITES) == []


def test_finalized_run_rejects_further_writes(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.SITES, config_sha256="c" * 64, resolved_config={}) as run:
        run.write_records("documents", [_doc(1)])
    with pytest.raises(ImmutableRunError, match="immutable"):
        run.write_records("documents", [_doc(2)])


def test_double_finalize_is_rejected(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.SITES, config_sha256="d" * 64, resolved_config={}) as run:
        run.write_records("documents", [_doc(1)])
    with pytest.raises(ImmutableRunError, match="already finalized"):
        run.finalize()


def test_outputs_digest_ignores_wall_clock(store: ArtifactStore) -> None:
    """Two runs of identical work must agree on outputs_digest even though their
    timestamps and durations differ; otherwise the determinism check is useless."""
    digests = []
    for _ in range(2):
        with store.begin(
            stage=StageName.MANIFEST, config_sha256="e" * 64, resolved_config={}
        ) as run:
            run.write_records("documents", [_doc(1), _doc(2)])
        assert run.record is not None
        digests.append(run.record.outputs_digest)
    assert digests[0] == digests[1]


def test_verify_outputs_detects_tampering(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.MANIFEST, config_sha256="f" * 64, resolved_config={}) as run:
        ref = run.write_records("documents", [_doc(1)])
    record = run.record
    assert record is not None
    assert store.verify_outputs(record) is True

    (store.run_dir(record) / ref.path).write_bytes(b"corrupted")
    assert store.verify_outputs(record) is False


def test_read_back_records(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.MANIFEST, config_sha256="1" * 64, resolved_config={}) as run:
        run.write_records("documents", [_doc(7)])
    record = run.record
    assert record is not None
    restored = store.read_records(record, "documents")
    assert restored == [_doc(7)]


def test_missing_output_names_what_is_available(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.MANIFEST, config_sha256="2" * 64, resolved_config={}) as run:
        run.write_records("documents", [_doc(1)])
    record = run.record
    assert record is not None
    with pytest.raises(FileNotFoundError, match=r"documents\.parquet"):
        store.read_table(record, "spans")


# --- taint propagation ------------------------------------------------------------------
def test_synthetic_and_leaky_flags_propagate_downstream(store: ArtifactStore) -> None:
    """A downstream stage must not be able to launder a synthetic or contaminated input
    into a clean-looking output."""
    with store.begin(
        stage=StageName.MANIFEST, config_sha256="3" * 64, resolved_config={}
    ) as upstream:
        upstream.write_records("documents", [_doc(1)])
        upstream.mark_synthetic()
        upstream.mark_leaky(note="deliberate diagnostic")
    assert upstream.record is not None

    with store.begin(stage=StageName.SITES, config_sha256="4" * 64, resolved_config={}) as down:
        down.inherit_from(upstream.record)
        down.write_records("documents", [_doc(1)])

    assert down.record is not None
    assert down.record.synthetic is True
    assert down.record.leaky is True


def test_leaky_split_marks_the_run(store: ArtifactStore) -> None:
    split = SplitDescriptor(
        fold_id="fold-a",
        protocol="diagnostic_doc_overlap",
        held_out_engines=("synth_a",),
        fit_engines=("synth_b",),
        calibrate_engines=("synth_b",),
        evaluate_engines=("synth_a",),
        fit_documents_sha256="x" * 64,
        calibrate_documents_sha256="y" * 64,
        evaluate_documents_sha256="z" * 64,
        n_fit_documents=10,
        n_calibrate_documents=5,
        n_evaluate_documents=5,
        document_partition_sha256="p" * 64,
        leaky=True,
    )
    with store.begin(stage=StageName.PREDICT, config_sha256="5" * 64, resolved_config={}) as run:
        run.set_split(split)
        run.write_records("documents", [_doc(1)])
    assert run.record is not None
    assert run.record.leaky is True


# --- caching ------------------------------------------------------------------------
def test_find_cached_returns_intact_run_only(store: ArtifactStore) -> None:
    with store.begin(
        stage=StageName.MANIFEST, config_sha256="6" * 64, resolved_config={}, cache_key="ck-1"
    ) as run:
        ref = run.write_records("documents", [_doc(1)])
    record = run.record
    assert record is not None

    assert store.find_cached(StageName.MANIFEST, "ck-1") is not None
    assert store.find_cached(StageName.MANIFEST, "ck-other") is None

    # A cache hit on corrupted outputs would silently reuse wrong bytes.
    (store.run_dir(record) / ref.path).write_bytes(b"corrupted")
    assert store.find_cached(StageName.MANIFEST, "ck-1") is None


# --- lineage ---------------------------------------------------------------------------
def test_lineage_walks_back_to_the_source(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.MANIFEST, config_sha256="7" * 64, resolved_config={}) as a:
        a.write_records("documents", [_doc(1)])
    with store.begin(stage=StageName.SITES, config_sha256="8" * 64, resolved_config={}) as b:
        assert a.record is not None
        b.inherit_from(a.record)
        b.write_records("documents", [_doc(1)])
    with store.begin(stage=StageName.EVALUATE, config_sha256="9" * 64, resolved_config={}) as c:
        assert b.record is not None
        c.inherit_from(b.record)
        c.write_json("metrics.json", {"coverage": 0.5})

    assert c.record is not None
    report = build_lineage(store, c.record.run_id)
    assert report.complete is True
    assert {n.record.stage.value for n in report.nodes} == {"evaluate", "sites", "manifest"}
    assert len(report.nodes) == 3
    assert "manifest" in render_lineage(report)


def test_lineage_flags_a_broken_chain(store: ArtifactStore) -> None:
    """Deleting an upstream run must surface as a broken link, not a silently shorter
    lineage."""
    with store.begin(stage=StageName.MANIFEST, config_sha256="a1" * 32, resolved_config={}) as a:
        a.write_records("documents", [_doc(1)])
    with store.begin(stage=StageName.SITES, config_sha256="b1" * 32, resolved_config={}) as b:
        assert a.record is not None
        b.inherit_from(a.record)
        b.write_records("documents", [_doc(1)])

    assert a.record is not None
    (store.run_dir(a.record) / "run_record.json").unlink()

    assert b.record is not None
    report = build_lineage(store, b.record.run_id)
    assert report.complete is False
    assert a.record.run_id in report.missing_runs


def test_lineage_flags_corrupted_outputs(store: ArtifactStore) -> None:
    with store.begin(stage=StageName.MANIFEST, config_sha256="c1" * 32, resolved_config={}) as a:
        ref = a.write_records("documents", [_doc(1)])
    assert a.record is not None
    (store.run_dir(a.record) / ref.path).write_bytes(b"nope")

    report = build_lineage(store, a.record.run_id)
    assert report.complete is False
    assert a.record.run_id in report.corrupted_runs


def test_lineage_of_unknown_run_reports_missing(store: ArtifactStore) -> None:
    report = build_lineage(store, "does-not-exist")
    assert report.complete is False
    assert report.missing_runs == ["does-not-exist"]


# --- raw store immutability -------------------------------------------------------------
def test_raw_store_refuses_to_change_bytes(tmp_path: Path) -> None:
    raw = RawStore()
    path = tmp_path / "engine" / "doc.json"
    raw.write_bytes(path, b'{"a": 1}')
    with pytest.raises(ImmutableWriteError, match="write-once"):
        raw.write_bytes(path, b'{"a": 2}')


def test_raw_store_rewrite_of_identical_bytes_is_a_noop(tmp_path: Path) -> None:
    """Pipelines re-run; an idempotent re-download must not be an error."""
    raw = RawStore()
    path = tmp_path / "doc.json"
    first = raw.write_bytes(path, b"same")
    second = raw.write_bytes(path, b"same")
    assert first == second


def test_raw_store_leaves_no_temp_file(tmp_path: Path) -> None:
    raw = RawStore()
    path = tmp_path / "doc.json"
    raw.write_bytes(path, b"payload")
    assert list(tmp_path.glob("*.tmp")) == []


# --- provenance capture ---------------------------------------------------------------
def test_git_state_is_captured_for_this_repo(tmp_path: Path) -> None:
    """Exercise real Git capture even when the software came from a ZIP."""
    subprocess.run(["git", "init", "-b", "fixture"], cwd=tmp_path, check=True, capture_output=True)
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("synthetic provenance fixture\n")
    subprocess.run(["git", "add", "fixture.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "Synthetic fixture",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    state = capture_git_state(tmp_path)
    assert state.commit != "unknown"
    assert len(state.commit) == 40
    assert state.branch == "fixture"
    assert not state.dirty
    fixture.write_text("changed synthetic fixture\n")
    state = capture_git_state(tmp_path)
    assert state.dirty
    assert state.diff_sha256 is not None, "a dirty tree must record a diff hash"


def test_git_state_without_repository_is_explicit(tmp_path: Path) -> None:
    state = capture_git_state(tmp_path)
    assert state.commit == "unknown"
    assert state.remote is None


def test_environment_capture_is_complete() -> None:
    env = capture_environment()
    assert env.python_version.startswith("3.")
    assert env.cpu_count >= 1
    assert len(env.package_digest) == 64


def test_run_id_encodes_stage_config_and_commit() -> None:
    run_id = make_run_id(StageName.ALIGN, "f" * 64, "abc1234def")
    assert run_id.startswith("align-")
    assert run_id.endswith("-ffffffff-abc1234d")


def test_run_record_json_is_readable_and_sorted(store: ArtifactStore) -> None:
    """The record must be diff-friendly: a human comparing two runs should see only real
    differences, not key reordering."""
    with store.begin(stage=StageName.MANIFEST, config_sha256="d1" * 32, resolved_config={}) as run:
        run.write_records("documents", [_doc(1)])
    assert run.record is not None
    text = (store.run_dir(run.record) / "run_record.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert list(payload) == sorted(payload)

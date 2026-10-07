"""Shared test configuration.

Determinism is a hard requirement (`.claude/rules/tests.md`): tests must not depend on
wall clock, locale, filesystem ordering, or hash randomization.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Pin interpreter-level sources of nondeterminism before collection."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    # Keep BLAS single-threaded so float reduction order is stable across machines.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    # No network, ever. The byt5 generator's availability check goes through
    # huggingface_hub, which reaches out to look for a safetensors conversion even with
    # `local_files_only=True` -- so a test suite that merely asks "is this generator
    # available?" was making an HTTP request, and would behave differently on a machine
    # with a warm cache. Forced off here rather than per test, because the next model
    # added would reintroduce it.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("OCR_RISK_ALLOW_MODEL_DOWNLOAD", "0")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every writable research layer into ``tmp_path``.

    Prevents tests from touching the developer's real data/artifact/cache trees.
    """
    for var, sub in (
        ("OCR_RISK_DATA_ROOT", "data"),
        ("OCR_RISK_ARTIFACT_ROOT", "artifacts"),
        ("OCR_RISK_CACHE_DIR", "cache"),
        # Manifests too: a frozen document partition is written here, and a test that
        # wrote one into the tracked tree committed a stale 16-document partition that
        # then refused to load against the 48-document corpus it was named for.
        ("OCR_RISK_MANIFEST_ROOT", "manifests"),
    ):
        target = tmp_path / sub
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(var, str(target))
    return tmp_path

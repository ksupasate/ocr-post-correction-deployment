"""Config resolution, override handling, and hash stability.

The config hash is what ties a figure to the settings that produced it, so it must depend
on the *effective* configuration and nothing else — not on key order, not on which
defaults happened to be written out explicitly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ocr_risk.config import (
    ConfigCycleError,
    ExperimentConfig,
    apply_overrides,
    config_hash,
    deep_merge,
    load_config,
    resolve_includes,
)
from ocr_risk.io.paths import project_root


def _write(path: Path, mapping: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


# --- merge semantics ------------------------------------------------------------------
def test_deep_merge_recurses_into_mappings() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    overlay = {"a": {"y": 99, "z": 3}}
    assert deep_merge(base, overlay) == {"a": {"x": 1, "y": 99, "z": 3}, "b": 1}


def test_deep_merge_replaces_lists_wholesale() -> None:
    """Element-wise list merging would make it impossible for an ablation config to
    *remove* an entry from an inherited list."""
    merged = deep_merge({"verifiers": [1, 2, 3]}, {"verifiers": [9]})
    assert merged["verifiers"] == [9]


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


# --- include DAG ----------------------------------------------------------------------
def test_child_overrides_parent(tmp_path: Path) -> None:
    _write(tmp_path / "parent.yaml", {"name": "parent", "seed": 1, "shared": "from-parent"})
    child = _write(
        tmp_path / "child.yaml", {"extends": ["parent.yaml"], "name": "child", "seed": 2}
    )
    resolved = resolve_includes(child)
    assert resolved == {"name": "child", "seed": 2, "shared": "from-parent"}


def test_multiple_parents_merge_left_to_right(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"v": "a", "only_a": 1})
    _write(tmp_path / "b.yaml", {"v": "b", "only_b": 2})
    child = _write(tmp_path / "c.yaml", {"extends": ["a.yaml", "b.yaml"]})
    resolved = resolve_includes(child)
    assert resolved == {"v": "b", "only_a": 1, "only_b": 2}


def test_diamond_inheritance_resolves(tmp_path: Path) -> None:
    _write(tmp_path / "root.yaml", {"v": 0})
    _write(tmp_path / "left.yaml", {"extends": ["root.yaml"], "left": 1})
    _write(tmp_path / "right.yaml", {"extends": ["root.yaml"], "right": 1})
    child = _write(tmp_path / "child.yaml", {"extends": ["left.yaml", "right.yaml"]})
    assert resolve_includes(child) == {"v": 0, "left": 1, "right": 1}


def test_cycle_is_detected(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"extends": ["b.yaml"]})
    _write(tmp_path / "b.yaml", {"extends": ["a.yaml"]})
    with pytest.raises(ConfigCycleError, match="circular"):
        resolve_includes(tmp_path / "a.yaml")


def test_missing_parent_is_an_error(tmp_path: Path) -> None:
    child = _write(tmp_path / "c.yaml", {"extends": ["nope.yaml"]})
    with pytest.raises(FileNotFoundError, match=r"nope\.yaml"):
        resolve_includes(child)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        resolve_includes(path)


# --- overrides ------------------------------------------------------------------------
def test_override_sets_nested_value() -> None:
    result = apply_overrides({"risk": {"delta": 0.1}}, ["risk.delta=0.25"])
    assert result["risk"]["delta"] == 0.25


def test_override_parses_yaml_types() -> None:
    result = apply_overrides(
        {},
        ["risk.epsilon_grid=[0.02, 0.01]", "splits.allow_document_overlap=true", "seed=5"],
    )
    assert result["risk"]["epsilon_grid"] == [0.02, 0.01]
    assert result["splits"]["allow_document_overlap"] is True
    assert result["seed"] == 5


def test_malformed_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="malformed override"):
        apply_overrides({}, ["no_equals_sign"])


def test_unknown_key_fails_validation(tmp_path: Path) -> None:
    """A typo must fail loudly rather than be silently ignored."""
    path = _write(tmp_path / "e.yaml", {"name": "x", "engines": [], "nonsense_key": 1})
    with pytest.raises(Exception, match="nonsense_key"):
        load_config(path)


# --- hashing --------------------------------------------------------------------------
def test_hash_ignores_key_order() -> None:
    a = {"alpha": 1, "beta": {"x": 1, "y": 2}}
    b = {"beta": {"y": 2, "x": 1}, "alpha": 1}
    assert config_hash(a) == config_hash(b)


def test_hash_changes_with_value() -> None:
    assert config_hash({"epsilon": 0.01}) != config_hash({"epsilon": 0.02})


def test_hash_is_stable_across_processes(tmp_path: Path) -> None:
    """PYTHONHASHSEED must not reach the digest; a hash that drifts between runs is
    worse than no hash because it trains people to ignore mismatches."""
    script = (
        "from ocr_risk.io.hashing import canonical_hash;"
        "print(canonical_hash({'b': [1, 2.5, None], 'a': {'z': True, 'y': 'text'}}))"
    )
    digests = set()
    for seed in ("0", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root(),
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_hash_reflects_defaults_not_what_was_typed(tmp_path: Path) -> None:
    """Two configs with the same *effective* settings must hash identically, even if one
    spells out a default the other inherits."""
    terse = _write(
        tmp_path / "terse.yaml",
        {
            "name": "x",
            "engines": [{"id": "a", "adapter": "synthetic"}, {"id": "b", "adapter": "synthetic"}],
        },
    )
    verbose = _write(
        tmp_path / "verbose.yaml",
        {
            "name": "x",
            "engines": [
                {"id": "a", "adapter": "synthetic", "enabled": True, "params": {}},
                {"id": "b", "adapter": "synthetic", "enabled": True, "params": {}},
            ],
            "stats": {"n_bootstrap": 10000},
        },
    )
    assert load_config(terse).sha256 == load_config(verbose).sha256


# --- model validation -----------------------------------------------------------------
def test_loeo_requires_at_least_two_engines() -> None:
    with pytest.raises(ValueError, match="at least 2 enabled engines"):
        ExperimentConfig(
            name="x",
            engines=({"id": "only", "adapter": "synthetic"},),  # type: ignore[arg-type]
        )


def test_duplicate_engine_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate engine ids"):
        ExperimentConfig(
            name="x",
            engines=(
                {"id": "dup", "adapter": "synthetic"},  # type: ignore[arg-type]
                {"id": "dup", "adapter": "synthetic"},  # type: ignore[arg-type]
            ),
        )


def test_epsilon_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        ExperimentConfig(name="x", risk={"epsilon_grid": [1.5]})  # type: ignore[arg-type]


def test_primary_harm_policy_must_be_in_sensitivity_set() -> None:
    """The primary policy has to appear in the sensitivity analysis; otherwise the
    headline number would have no reported alternative."""
    with pytest.raises(ValueError, match="sensitivity_harm_policies"):
        ExperimentConfig(
            name="x",
            risk={"harm_policy": "exact_only", "sensitivity_harm_policies": ["strict_worsening"]},  # type: ignore[arg-type]
        )


def test_split_fractions_must_leave_room_for_test() -> None:
    with pytest.raises(ValueError, match="no documents for the evaluation split"):
        ExperimentConfig(name="x", splits={"fit_fraction": 0.8, "calibrate_fraction": 0.3})  # type: ignore[arg-type]


def test_test_fraction_is_the_remainder() -> None:
    cfg = ExperimentConfig(
        name="x",
        engines=(
            {"id": "a", "adapter": "synthetic"},  # type: ignore[arg-type]
            {"id": "b", "adapter": "synthetic"},  # type: ignore[arg-type]
        ),
        splits={"fit_fraction": 0.6, "calibrate_fraction": 0.2},  # type: ignore[arg-type]
    )
    assert cfg.splits.test_fraction == pytest.approx(0.2)


# --- the real repository configs must load --------------------------------------------
def test_smoke_config_loads_and_is_marked_synthetic() -> None:
    resolved = load_config(project_root() / "configs/experiments/smoke_synthetic.yaml")
    cfg = resolved.config
    assert cfg.name == "smoke_synthetic"
    assert cfg.synthetic is True, "the smoke config must self-identify as synthetic"
    assert len(cfg.enabled_engine_ids) == 4
    assert cfg.splits.allow_document_overlap is False
    assert len(resolved.sha256) == 64


def test_smoke_config_has_the_full_ablation_ladder() -> None:
    cfg = load_config(project_root() / "configs/experiments/smoke_synthetic.yaml").config
    configs = {v.evidence_config for v in cfg.verifiers}
    assert configs == {"v0", "v1", "v2", "v3", "v4", "v5", "v6"}

"""RH1: does unseen-engine transfer cost the model its ability to rank edits?

Runs the matched-handicap contrast on the **candidate pool the H1 pilot already produced**,
reusing its ``candidates`` artifact rather than regenerating one. That is deliberate: the
question RH1 asks first is whether H1's pattern survives the correction, and changing the
generator at the same time as the design would leave the difference unattributable. The
frozen generator matters for the *next* study, not for this one; the readiness gate is what
reads it.

Both arms of a pair are handed the identical evaluation rows, the identical candidates, and
— after :func:`ocr_risk.experiments.matched.build_match` — the identical number of harmful
and safe fitting examples. A ``MatchCertificate`` per pair is written beside the
predictions, and the run refuses to proceed if any registered constraint is violated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ocr_risk.candidates.pipeline import confidences_for_site
from ocr_risk.config.models import ExperimentConfig
from ocr_risk.evidence.crops import CropPolicy, build_recipe, materialize
from ocr_risk.experiments.loeo_runner import ConfSample, FoldResult, methods_from_config, run_fold
from ocr_risk.experiments.matched import MatchCertificate, build_match
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.paths import data_root, partition_manifest_path
from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.evidence import EvidenceBundle
from ocr_risk.splits import MatchedPair, matched_pairs
from ocr_risk.splits.document_partition import DocumentPartition

__all__ = ["CandidatePoolBundle", "Rh1Run", "UnmatchedArmsError", "load_pool", "run_rh1"]


@dataclass(slots=True)
class CandidatePoolBundle:
    """A candidate pool reloaded from artifacts, ready to hand to the fold runner.

    Rebuilt from the recorded spans and sites rather than from the evidence bundle alone.
    The bundle stores a site's summarized confidence but not the *scale* it was measured
    on, and ``normalize_confidence`` returns ``None`` without one — so a pool reconstructed
    from bundles would silently hand every fold an empty confidence channel and the arms
    would agree because neither had any signal.
    """

    frame: pd.DataFrame
    bundles: dict[str, EvidenceBundle]
    conf_samples: dict[str, ConfSample]
    crops: dict[str, Path]
    partition: DocumentPartition
    source_run_id: str


def load_pool(
    store: ArtifactStore,
    source_experiment: str,
    *,
    materialize_crops: bool = True,
    expected_partition_id: str | None = None,
) -> CandidatePoolBundle:
    """Reload a finished experiment's candidate pool, with confidences and crops intact.

    ``expected_partition_id`` names the frozen partition this study *claims* to use. The
    partition actually comes from the pool, so without the check the RH1 config's whole
    ``splits:`` block is inert: it can say ``partition_id: h1_pilot`` while ``--source``
    supplies something else, and the run record would repeat the claim.
    """
    candidates_run = store.latest(StageName.CANDIDATES, source_experiment)
    canonical_run = store.latest(StageName.CANONICALIZE, source_experiment)
    sites_run = store.latest(StageName.SITES, source_experiment)
    manifest_run = store.latest(StageName.MANIFEST, source_experiment)
    missing = [
        name
        for name, record in (
            ("candidates", candidates_run),
            ("canonicalize", canonical_run),
            ("sites", sites_run),
            ("manifest", manifest_run),
        )
        if record is None
    ]
    if missing or candidates_run is None or canonical_run is None or sites_run is None:
        msg = f"experiment {source_experiment!r} is missing artifacts: {missing}"
        raise FileNotFoundError(msg)
    assert manifest_run is not None

    partition = DocumentPartition.load(store.run_dir(candidates_run) / "document_partition.json")
    if expected_partition_id:
        declared = partition_manifest_path(expected_partition_id)
        if not declared.is_file():
            msg = f"no frozen partition manifest at {declared} for {expected_partition_id!r}"
            raise FileNotFoundError(msg)
        expected = DocumentPartition.load(declared)
        if expected.partition_sha256 != partition.partition_sha256:
            msg = (
                f"the pool from {source_experiment!r} carries partition "
                f"{partition.partition_sha256[:16]}, but this study declares "
                f"{expected_partition_id!r} = {expected.partition_sha256[:16]}. Two arms "
                "evaluated on different documents is not a transfer contrast."
            )
            raise ValueError(msg)

    spans_by_id = {s.span_id: s for s in store.read_records(canonical_run, "spans")}
    sites_by_id = {s.site_id: s for s in store.read_records(sites_run, "sites")}
    documents = {d.document_id: d for d in store.read_records(manifest_run, "documents")}
    bundles = {b.candidate_id: b for b in store.read_records(candidates_run, "evidence")}
    labels = {label.candidate_id: label for label in store.read_records(candidates_run, "labels")}

    rows: list[dict[str, object]] = []
    conf_samples: dict[str, ConfSample] = {}
    crops: dict[str, Path] = {}
    crop_policy = CropPolicy()
    for candidate in store.read_records(candidates_run, "candidates"):
        label = labels.get(candidate.candidate_id)
        site = sites_by_id.get(candidate.site_id)
        if label is None or site is None:
            continue
        confidences, scale = confidences_for_site(site, spans_by_id)
        conf_samples[candidate.candidate_id] = ConfSample(
            engine_id=candidate.engine_id, native_confidences=confidences, conf_scale=scale
        )
        document = documents.get(site.document_id)
        if materialize_crops and site.bbox is not None and document is not None:
            image = data_root() / document.image_path
            if image.is_file():
                recipe = build_recipe(document.image_sha256, site.bbox, crop_policy)
                crops[candidate.candidate_id] = materialize(recipe, image)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "site_id": candidate.site_id,
                "document_id": candidate.document_id,
                "dataset_id": candidate.dataset_id,
                "engine_id": candidate.engine_id,
                "generator_id": candidate.generator_id,
                "generator_rank": candidate.generator_rank,
                "pool": candidate.pool.value,
                "is_synthetic_hard_negative": candidate.is_synthetic_hard_negative,
                "outcome_if_accepted": label.outcome_if_accepted.value,
                "outcome_if_rejected": label.outcome_if_rejected.value,
                "d_before": label.d_before,
                "d_after": label.d_after,
                "delta": label.delta,
            }
        )

    return CandidatePoolBundle(
        frame=pd.DataFrame(rows),
        bundles=bundles,
        conf_samples=conf_samples,
        crops=crops,
        partition=partition,
        source_run_id=candidates_run.run_id,
    )


class UnmatchedArmsError(RuntimeError):
    """Raised when a pair's arms fail a registered matching constraint.

    Not a warning. The whole point of the arm is that the two conditions differ in one
    dimension, and a run that proceeds with an unverified match produces exactly the kind
    of number this phase exists to retract.
    """


@dataclass(slots=True)
class Rh1Run:
    """Everything the matched study produced."""

    fold_results: list[FoldResult] = field(default_factory=list)
    certificates: list[MatchCertificate] = field(default_factory=list)
    pairs: list[MatchedPair] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return all(c.matched for c in self.certificates)


def run_rh1(
    config: ExperimentConfig,
    partition: DocumentPartition,
    frame: pd.DataFrame,
    bundles: dict[str, EvidenceBundle],
    conf_samples: dict[str, ConfSample],
    crops: dict[str, Path] | None = None,
    engines: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Rh1Run:
    """Run every (target engine, donor) pair, both arms, under every enabled method."""
    run = Rh1Run()
    engine_ids = list(engines or config.enabled_engine_ids)
    run.pairs = matched_pairs(
        engine_ids, partition, frame, selection_scope=config.splits.selection_scope
    )
    methods = methods_from_config(config)

    shared: dict[str, object] = {
        "calibrator": config.calibration.method,
        "controller": config.risk.controller,
        "harm_policy": config.risk.harm_policy.value,
        "site_policy": config.risk.site_policy,
        "epsilon_grid": list(config.risk.epsilon_grid),
        "n_threshold_grid": config.risk.n_threshold_grid,
        "verifiers": [m.method_id for m in methods],
        "candidate_generators": [g.id for g in config.candidates.generators if g.enabled],
        "hard_negatives_in_evaluation": config.candidates.hard_negatives_in_evaluation,
    }

    for pair in run.pairs:
        zero_shot_match, reference_match, certificate = build_match(
            pair, config.risk.harm_policy, seed=config.seed, shared=shared
        )
        run.certificates.append(certificate)
        if not certificate.matched:
            violations = "; ".join(str(v) for v in certificate.violations)
            msg = (
                f"pair {certificate.pair_id} is not matched: {violations}. "
                "A contrast between arms that differ in more than target-engine exposure "
                "measures something other than transfer."
            )
            raise UnmatchedArmsError(msg)

        for plan, match, arm in (
            (pair.zero_shot, zero_shot_match, "loeo_zero_shot"),
            (pair.reference, reference_match, "matched_in_engine"),
        ):
            plan.assert_no_leakage()
            for method in methods:
                if progress is not None:
                    progress(f"{certificate.pair_id} {arm} {method.method_id}")
                # The zero-shot plan is shared across a target engine's three pairs, so
                # its own fold id repeats; the donor has to be in the id the PREDICTIONS
                # carry, or the three runs are indistinguishable and the paired join
                # collapses them.
                result = run_fold(
                    plan,
                    method,
                    config,
                    bundles,
                    crops,
                    conf_samples,
                    match=match,
                    fold_id=f"{arm}:{certificate.pair_id}",
                )
                result.diagnostics["arm"] = arm
                result.diagnostics["pair_id"] = certificate.pair_id
                result.diagnostics["held_out_engine"] = certificate.held_out_engine
                result.diagnostics["donor_engine"] = certificate.donor_engine
                run.fold_results.append(result)
    return run

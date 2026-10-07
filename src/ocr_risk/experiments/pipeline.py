"""End-to-end experiment orchestration.

Composes the fifteen lifecycle stages into one command. Each stage writes an immutable,
provenance-linked run, and each consumes the previous run's outputs *by hash* — so the
chain from a figure back to the source manifest is verifiable rather than assumed.

The predict/calibrate/decide stages run per fold and per method through
:mod:`ocr_risk.experiments.loeo_runner`, which is where the split discipline lives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ocr_risk.candidates import build_generator
from ocr_risk.candidates.pipeline import (
    SiteContext,
    build_labels,
    build_observation_views,
    confidences_for_site,
    generate_candidates,
)
from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response, rebuild_stream
from ocr_risk.config import ResolvedConfig
from ocr_risk.datasets import build_dataset
from ocr_risk.datasets.synthetic import noise_level_of
from ocr_risk.edits import build_sites
from ocr_risk.engines import OCREngineAdapter, PageInput, build_engine
from ocr_risk.evidence import CropPolicy, EvidenceBuilder, materialize
from ocr_risk.experiments.loeo_runner import (
    ConfSample,
    FoldResult,
    methods_from_config,
    run_fold,
)
from ocr_risk.io.artifacts import ArtifactStore, RunWriter
from ocr_risk.io.hashing import canonical_hash
from ocr_risk.io.paths import data_root, partition_manifest_path
from ocr_risk.io.raw_store import RawStore
from ocr_risk.provenance import capture_git_state
from ocr_risk.schemas.alignment import AlignmentRecord
from ocr_risk.schemas.candidates import Candidate, CandidateLabel
from ocr_risk.schemas.documents import GtToken, SourceDocument
from ocr_risk.schemas.enums import SplitRole, StageName
from ocr_risk.schemas.evidence import EvidenceBundle
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.schemas.spans import CanonicalSpan
from ocr_risk.splits import DocumentPartition, PartitionSpec, build_partition, loeo_folds

__all__ = ["PipelineState", "run_experiment"]


@dataclass(slots=True)
class PipelineState:
    """Data carried between stages within one experiment run."""

    documents: list[SourceDocument] = field(default_factory=list)
    gt_tokens: list[GtToken] = field(default_factory=list)
    spans: list[CanonicalSpan] = field(default_factory=list)
    sites: list[CorrectionSite] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    labels: list[CandidateLabel] = field(default_factory=list)
    evidence: list[EvidenceBundle] = field(default_factory=list)
    noise_levels: dict[str, str] = field(default_factory=dict)
    crops: dict[str, Path] = field(default_factory=dict)
    conf_samples: dict[str, ConfSample] = field(default_factory=dict)
    """Raw confidence readings per candidate. Held here rather than in the evidence
    bundle because standardizing them requires a fold-scoped population."""
    fold_results: list[FoldResult] = field(default_factory=list)


def _recognize_dataset(
    *,
    engine: OCREngineAdapter,
    engine_id: str,
    fingerprint: str,
    documents: Sequence[SourceDocument],
    raw_store: RawStore,
    policy: CanonicalizationPolicy,
    state: PipelineState,
) -> None:
    """Recognize one dataset with one engine, appending canonical spans.

    The raw response is written once and re-read rather than canonicalized straight from
    memory, so the bytes that reach the rest of the pipeline are the bytes on disk. A
    reformatting bug in the store would otherwise stay invisible until months later.
    """
    for document in documents:
        path = raw_store.engine_response_path(
            document.dataset_id, engine_id, fingerprint, document.document_id
        )
        if not path.exists():
            page = PageInput(
                document_id=document.document_id,
                dataset_id=document.dataset_id,
                image_path=data_root() / document.image_path,
                image_sha256=document.image_sha256,
                width=document.width,
                height=document.height,
            )
            raw_store.write_engine_response(engine.recognize(page))
        raw = raw_store.read_engine_response(path)
        state.spans.extend(
            canonicalize_response(
                raw=raw,
                parsed=engine.parse(raw),
                policy=policy,
                conf_scale_name=engine.confidence_scale.name,
                raw_ref=path.relative_to(data_root()).as_posix(),
            )
        )


def _begin(store: ArtifactStore, stage: StageName, resolved: ResolvedConfig) -> RunWriter:
    return store.begin(
        stage=stage,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=resolved.config.name,
        git_commit=capture_git_state().commit,
    )


def run_experiment(
    resolved: ResolvedConfig,
    store: ArtifactStore | None = None,
    *,
    materialize_crops: bool = True,
    progress: list[str] | None = None,
) -> PipelineState:
    """Run every stage of one experiment, writing an artifact per stage.

    ``progress`` is appended to in place, so a caller can stream stage messages.
    """
    cfg = resolved.config
    store = store or ArtifactStore()
    state = PipelineState()
    log: list[str] = progress if progress is not None else []

    # --- acquire + manifest -----------------------------------------------------------
    for dataset_cfg in cfg.datasets:
        if not dataset_cfg.enabled:
            continue
        dataset = build_dataset(dataset_cfg.id, **dataset_cfg.params)
        if not dataset.preflight().available and hasattr(dataset, "materialize"):
            dataset.materialize()  # synthetic corpora render on demand
        dataset.preflight().raise_if_unavailable()

        for document_bundle in dataset.documents(limit=dataset_cfg.max_documents):
            state.documents.append(document_bundle.document)
            state.gt_tokens.extend(document_bundle.gt_tokens)
            if dataset_cfg.id == "synthetic":
                state.noise_levels[document_bundle.document.document_id] = noise_level_of(
                    document_bundle.document.document_id
                )

    with _begin(store, StageName.MANIFEST, resolved) as run:
        run.write_records("documents", state.documents)
        run.write_records("gt_tokens", state.gt_tokens)
        run.write_json(
            "manifest_stats.json",
            {
                "n_documents": len(state.documents),
                "n_gt_tokens": len(state.gt_tokens),
                "noise_levels": state.noise_levels,
                "synthetic": cfg.synthetic,
            },
        )
        run.set_datasets(cfg.enabled_dataset_ids)
        run.mark_synthetic(cfg.synthetic)
    manifest_run = run.record
    assert manifest_run is not None
    log.append(f"manifest: {len(state.documents)} documents")

    # --- ocr + canonicalize -------------------------------------------------------------
    raw_store = RawStore()
    policy = CanonicalizationPolicy(unicode_policy=cfg.alignment.unicode_policy)
    fingerprints: dict[str, str] = {}

    documents_by_dataset: dict[str, list[SourceDocument]] = {}
    for document in state.documents:
        documents_by_dataset.setdefault(document.dataset_id, []).append(document)

    # One engine instance per (engine, dataset), because recognition language is a
    # property of the corpus. Without per-dataset parameters a single `lang: eng` would
    # read German Fraktur with an English model and produce plausible-looking garbage.
    for engine_cfg in cfg.engines:
        if not engine_cfg.enabled:
            continue
        for dataset_id, dataset_documents in sorted(documents_by_dataset.items()):
            params = engine_cfg.params_for(dataset_id)
            engine = build_engine(engine_cfg.adapter, engine_id=engine_cfg.id, **params)

            # Prefer the recorded fingerprint of a completed recognition pass. Asking the
            # engine for it imports the backend, and PaddlePaddle and PyTorch cannot
            # coexist in one process here — so a run covering all four engines would
            # crash even though every response was already on disk. Recognition happens
            # one engine per process; everything after it reads bytes.
            recorded = raw_store.read_fingerprint(dataset_id, engine_cfg.id)
            cached = recorded is not None and all(
                raw_store.has_engine_response(
                    dataset_id, engine_cfg.id, str(recorded["fingerprint"]), d.document_id
                )
                for d in dataset_documents
            )
            if cached and recorded is not None:
                fingerprint_value = str(recorded["fingerprint"])
            else:
                engine.availability().require()
                fingerprint_value = engine.fingerprint().fingerprint
            # Keyed per dataset only where an override actually applies, so the common
            # single-language case keeps a plain engine_id key in the run record.
            fingerprint_key = (
                f"{engine_cfg.id}@{dataset_id}"
                if engine_cfg.dataset_params.get(dataset_id)
                else engine_cfg.id
            )
            fingerprints[fingerprint_key] = fingerprint_value

            _recognize_dataset(
                engine=engine,
                engine_id=engine_cfg.id,
                fingerprint=fingerprint_value,
                documents=dataset_documents,
                raw_store=raw_store,
                policy=policy,
                state=state,
            )
            raw_store.record_fingerprint(
                dataset_id, engine_cfg.id, {"fingerprint": fingerprint_value, "params": params}
            )
    with _begin(store, StageName.CANONICALIZE, resolved) as run:
        run.inherit_from(manifest_run)
        run.write_records("spans", state.spans)
        run.set_engine_fingerprints(fingerprints)
        run.mark_synthetic(cfg.synthetic)
    canonical_run = run.record
    assert canonical_run is not None
    # Distinct engine ids, not fingerprint records: an engine configured per dataset
    # contributes several fingerprints and is still one engine on the leave-one-out axis.
    n_engines = len({s.engine_id for s in state.spans})
    log.append(
        f"canonicalize: {len(state.spans)} spans across {n_engines} engines "
        f"({len(fingerprints)} engine/dataset configurations)"
    )

    # --- align + sites -------------------------------------------------------------------
    from ocr_risk.align import align_document, summarize

    documents_by_id = {d.document_id: d for d in state.documents}
    tokens_by_document: dict[str, list[GtToken]] = {}
    for token in state.gt_tokens:
        tokens_by_document.setdefault(token.document_id, []).append(token)

    spans_by_pair: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for span in state.spans:
        spans_by_pair.setdefault((span.document_id, span.engine_id), []).append(span)

    alignments: list[AlignmentRecord] = []
    for (document_id, _engine), spans in sorted(spans_by_pair.items()):
        alignments.extend(
            align_document(
                documents_by_id[document_id],
                spans,
                tokens_by_document.get(document_id, []),
                cfg.alignment,
            ).records
        )

    with _begin(store, StageName.ALIGN, resolved) as run:
        run.inherit_from(canonical_run)
        run.write_records("alignments", alignments)
        run.write_json(
            "alignment_stats.json",
            {"per_engine": {k: v.as_dict() for k, v in summarize(alignments).items()}},
        )
        run.mark_synthetic(cfg.synthetic)
    align_run = run.record
    assert align_run is not None
    log.append(f"align: {len(alignments)} components")

    alignments_by_pair: dict[tuple[str, str], list[AlignmentRecord]] = {}
    for record in alignments:
        alignments_by_pair.setdefault((record.document_id, record.engine_id), []).append(record)
    for key, records in sorted(alignments_by_pair.items()):
        state.sites.extend(build_sites(records, spans_by_pair.get(key, []), cfg.sites))

    with _begin(store, StageName.SITES, resolved) as run:
        run.inherit_from(align_run)
        run.write_records("sites", state.sites)
        run.mark_synthetic(cfg.synthetic)
    sites_run = run.record
    assert sites_run is not None
    log.append(f"sites: {len(state.sites)} correction sites")

    # --- the document partition, drawn once and reused by every fold ----------------------
    #
    # Loaded from the frozen manifest when one exists. Recomputing it every run is not
    # equivalent: the cuts are at exact fractions over whatever documents are present, so
    # changing max_documents, enabling a corpus, or a raw file going missing rotates
    # documents between roles between runs — and every run record would still report a
    # self-consistent partition. Freezing it makes that rotation visible as a changed
    # hash instead of invisible.
    partition_spec = PartitionSpec(
        seed=cfg.splits.partition_seed,
        fit_fraction=cfg.splits.fit_fraction,
        calibrate_fraction=cfg.splits.calibrate_fraction,
        stratify_by=cfg.splits.stratify_by,
    )
    partition_path = partition_manifest_path(cfg.splits.partition_id or cfg.name)
    if partition_path.is_file():
        partition = DocumentPartition.load(partition_path)
        # The frozen partition must be the one this config asks for. Without this check a
        # partition drawn under a different spec loads silently: the pilot's was stratified
        # by dataset_id alone while the config and the pre-registration both said
        # [dataset_id, noise_tertile], and nothing noticed until a reviewer recomputed the
        # spec hash. A frozen artifact that does not match its config is not reproducible,
        # it is merely stable.
        expected_spec = canonical_hash(partition_spec)
        if partition.spec_hash != expected_spec:
            msg = (
                f"{partition_path} was drawn under spec {partition.spec_hash[:16]} but this "
                f"config specifies {expected_spec[:16]} "
                f"(seed={partition_spec.seed}, stratify_by={list(partition_spec.stratify_by)}). "
                "Re-freeze it deliberately, or point splits.partition_id at the partition "
                "this config actually describes."
            )
            raise ValueError(msg)
        uncovered = partition.covers(d.document_id for d in state.documents)
        if uncovered:
            msg = (
                f"{partition_path} does not cover {len(uncovered)} document(s) in this "
                f"corpus, e.g. {list(uncovered[:3])}. A frozen partition and a changed "
                "corpus cannot both be honoured; re-freeze it deliberately rather than "
                "letting documents fall outside every role."
            )
            raise ValueError(msg)
        log.append(
            f"partition: loaded frozen {partition_path.name} ({partition.partition_sha256[:12]})"
        )
    else:
        partition = build_partition(
            state.documents,
            partition_spec,
            stratum_overrides={
                doc: {"noise_tertile": level} for doc, level in state.noise_levels.items()
            },
        )
        partition.save(partition_path)
        log.append(f"partition: froze {partition_path.name} ({partition.partition_sha256[:12]})")

    # --- candidates + labels + evidence ----------------------------------------------------
    spans_by_id = {s.span_id: s for s in state.spans}
    streams = {key: rebuild_stream(group) for key, group in spans_by_pair.items()}

    contexts: list[SiteContext] = []
    for site in state.sites:
        if not site.evaluable:
            continue
        document = documents_by_id[site.document_id]
        stream = streams.get((site.document_id, site.engine_id), "")
        confidences, scale = confidences_for_site(site, spans_by_id)
        width = cfg.evidence.context_window_chars
        contexts.append(
            SiteContext(
                site=site,
                context_before=stream[max(0, site.char_start - width) : site.char_start],
                context_after=stream[site.char_end : site.char_end + width],
                native_confidences=confidences,
                conf_scale=scale,
                image_sha256=document.image_sha256,
                image_width=document.width,
                image_height=document.height,
            )
        )

    # Generator resources are fitted on the FIT split only: a lexicon built over every
    # document would leak test-document vocabulary into the generator (vector L5).
    #
    # The engine axis matters here too. A lexicon is fitted once and shared by every
    # fold, so it may only contain text from engines no fold holds out — and under
    # leave-one-engine-out every engine is held out by some fold. The intersection is
    # empty, so the shared lexicon is built from the *ground-truth-blind* OCR text of the
    # fit documents under engines common to all folds; with LOEO that is no engine, so
    # the lexicon is instead built per fold. Building it once over all engines would let
    # a fold's held-out engine contribute vocabulary that suppresses candidate generation
    # exactly where that engine misreads (vector L5, engine axis).
    fit_documents = partition.documents(SplitRole.FIT)
    fit_corpus = [c.site.ocr_text for c in contexts if c.site.document_id in fit_documents]

    generators = []
    for spec in cfg.candidates.generators:
        if not spec.enabled:
            continue
        generator = build_generator(spec.kind, generator_id=spec.id, **spec.params)
        if not generator.available():
            continue
        generator.fit(fit_corpus)
        generators.append((generator, spec.max_candidates))

    state.candidates = generate_candidates(contexts, generators, cfg.candidates)
    state.labels = build_labels(state.candidates, [c.site for c in contexts])

    # No confidence featurizer here, deliberately. A z-score needs a population, and
    # under leave-one-engine-out the admissible population is fold-dependent: the fit
    # engines of one fold include the engine another fold holds out. A bundle built once
    # and shared across folds therefore cannot carry one without leaking, so the z-score
    # is computed inside run_fold from this fold's fit view (vector L3).
    builder = EvidenceBuilder(
        crop_policy=CropPolicy(
            padding_ratio=cfg.evidence.crop_padding_ratio,
            padding_min_px=cfg.evidence.crop_padding_min_px,
            target_height=cfg.evidence.crop_target_height,
            grayscale=cfg.evidence.crop_grayscale,
        ),
        context_chars=cfg.evidence.context_window_chars,
        confidence_featurizer=None,
    )

    from ocr_risk.evidence.crops import build_recipe

    for view, site_context in build_observation_views(state.candidates, contexts):
        site = site_context.site
        start = len(site_context.context_before)
        evidence_bundle = builder.build(
            view,
            ocr_stream=site_context.context_before + site.ocr_text + site_context.context_after,
            char_start=start,
            char_end=start + len(site.ocr_text),
            native_confidences=site_context.native_confidences,
            conf_scale=site_context.conf_scale,
        )
        state.evidence.append(evidence_bundle)
        state.conf_samples[view.candidate_id] = ConfSample(
            engine_id=view.engine_id,
            native_confidences=tuple(site_context.native_confidences),
            conf_scale=site_context.conf_scale,
        )

        if materialize_crops and site.bbox is not None:
            recipe = build_recipe(view.image_sha256, site.bbox, builder.crop_policy)
            image = data_root() / documents_by_id[site.document_id].image_path
            if image.is_file():
                state.crops[view.candidate_id] = materialize(recipe, image)

    with _begin(store, StageName.CANDIDATES, resolved) as run:
        run.inherit_from(sites_run)
        run.write_records("candidates", state.candidates)
        run.write_records("labels", state.labels)
        run.write_records("evidence", state.evidence)
        run.write_json("document_partition.json", partition.to_manifest())
        run.mark_synthetic(cfg.synthetic)
    candidates_run = run.record
    assert candidates_run is not None
    log.append(f"candidates: {len(state.candidates)} proposed edits")

    # --- predict / calibrate / decide, per fold and method ---------------------------------
    frame = _analysis_frame(state)
    bundles = {b.candidate_id: b for b in state.evidence}
    plans = loeo_folds(
        list(cfg.enabled_engine_ids),
        partition,
        frame,
        protocol=cfg.splits.protocol,
        few_shot_k=cfg.splits.few_shot_k[0] if cfg.splits.few_shot_k else 0,
        selection_scope=cfg.splits.selection_scope,
        allow_document_overlap=cfg.splits.allow_document_overlap,
    )
    methods = methods_from_config(cfg)

    for plan in plans:
        plan.assert_no_leakage()
        for method in methods:
            state.fold_results.append(
                run_fold(plan, method, cfg, bundles, state.crops, state.conf_samples)
            )

    predictions = [p for r in state.fold_results for p in r.predictions]
    decisions = [d for r in state.fold_results for d in r.decisions]

    with _begin(store, StageName.PREDICT, resolved) as run:
        run.inherit_from(candidates_run)
        run.write_records("predictions", predictions)
        run.write_json(
            "folds.json",
            [
                {
                    "fold_id": r.fold_id,
                    "method_id": r.method_id,
                    "calibrator_id": r.calibrator_id,
                    "diagnostics": r.diagnostics,
                    "split": r.descriptor.model_dump(mode="json"),
                }
                for r in state.fold_results
            ],
        )
        if state.fold_results:
            run.set_folds([r.descriptor for r in state.fold_results])
        run.mark_synthetic(cfg.synthetic)
    predict_run = run.record
    assert predict_run is not None
    log.append(
        f"predict: {len(predictions)} scores over {len(plans)} folds x {len(methods)} methods"
    )

    with _begin(store, StageName.DECIDE, resolved) as run:
        run.inherit_from(predict_run)
        run.write_records("decisions", decisions)
        run.write_json("thresholds.json", [t for r in state.fold_results for t in r.thresholds])
        if state.fold_results:
            run.set_folds([r.descriptor for r in state.fold_results])
        run.mark_synthetic(cfg.synthetic)
    log.append(f"decide: {len(decisions)} site decisions")

    return state


def _analysis_frame(state: PipelineState) -> pd.DataFrame:
    """One row per candidate, joining what the split plan and runner need.

    Built once here so the split plan filters a single frame rather than each stage
    re-deriving its own view of the data.
    """
    label_by_id = {label.candidate_id: label for label in state.labels}
    rows = []
    for candidate in state.candidates:
        label = label_by_id.get(candidate.candidate_id)
        if label is None:
            continue
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "site_id": candidate.site_id,
                "document_id": candidate.document_id,
                "dataset_id": candidate.dataset_id,
                "engine_id": candidate.engine_id,
                "generator_id": candidate.generator_id,
                "generator_rank": candidate.generator_rank,
                "is_synthetic_hard_negative": candidate.is_synthetic_hard_negative,
                "outcome_if_accepted": label.outcome_if_accepted.value,
                "outcome_if_rejected": label.outcome_if_rejected.value,
                "d_before": label.d_before,
                "d_after": label.d_after,
                "delta": label.delta,
            }
        )
    return pd.DataFrame(rows)

"""Corpus -> engines -> canonical spans, end to end on a small synthetic corpus.

The scientifically load-bearing check here is the last one: the simulated engines must
disagree with each other in *different ways*. Four engines that corrupt text identically
would let a leave-one-engine-out experiment pass while measuring nothing.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response
from ocr_risk.datasets.synthetic import SyntheticDataset
from ocr_risk.engines import build_engine
from ocr_risk.engines.base import PageInput
from ocr_risk.io.paths import data_root
from ocr_risk.io.raw_store import ImmutableWriteError, RawStore

PAGES = 6

ENGINE_PARAMS = {
    "synth_a": {"seed": 1001, "char_error_rate": 0.045, "confusion_profile": "visual"},
    "synth_b": {
        "seed": 1002,
        "char_error_rate": 0.022,
        "confusion_profile": "segmentation",
        "split_rate": 0.12,
        "merge_rate": 0.10,
        "confidence_scale": "synthetic_0_100",
    },
    "synth_c": {"seed": 1003, "char_error_rate": 0.035, "confusion_profile": "numeric"},
    "synth_d": {"seed": 1004, "char_error_rate": 0.008, "confusion_profile": "mixed"},
}


@pytest.fixture
def corpus(isolated_env: Path) -> SyntheticDataset:
    adapter = SyntheticDataset(n_documents=PAGES, seed=4242)
    adapter.materialize()
    return adapter


def _recognize_all(corpus: SyntheticDataset) -> dict[str, list]:  # type: ignore[type-arg]
    store = RawStore()
    spans_by_engine: dict[str, list] = {}
    bundles = list(corpus.documents())
    for engine_id, params in ENGINE_PARAMS.items():
        adapter = build_engine("synthetic", engine_id=engine_id, **params)
        collected = []
        for bundle in bundles:
            document = bundle.document
            page = PageInput(
                document_id=document.document_id,
                dataset_id=document.dataset_id,
                image_path=data_root() / document.image_path,
                image_sha256=document.image_sha256,
                width=document.width,
                height=document.height,
            )
            raw = adapter.recognize(page)
            path = store.write_engine_response(raw)
            collected.extend(
                canonicalize_response(
                    raw=raw,
                    parsed=adapter.parse(raw),
                    policy=CanonicalizationPolicy(),
                    conf_scale_name=adapter.confidence_scale.name,
                    raw_ref=path.relative_to(data_root()).as_posix(),
                )
            )
        spans_by_engine[engine_id] = collected
    return spans_by_engine


# --- corpus ---------------------------------------------------------------------------
def test_corpus_materializes_and_reads_back(corpus: SyntheticDataset) -> None:
    report = corpus.preflight()
    assert report.available
    assert report.n_documents == PAGES

    bundles = list(corpus.documents())
    assert len(bundles) == PAGES
    for bundle in bundles:
        assert bundle.document.gt_text
        assert bundle.gt_tokens
        assert bundle.document.n_gt_tokens == len(bundle.gt_tokens)
        assert bundle.document.has_gt_geometry


def test_gt_token_offsets_index_the_gt_text(corpus: SyntheticDataset) -> None:
    """Offsets must be exact even when the same word repeats on a page."""
    for bundle in corpus.documents():
        text = bundle.document.gt_text
        for token in bundle.gt_tokens:
            assert text[token.char_start : token.char_end] == token.text


def test_gt_boxes_lie_inside_the_page(corpus: SyntheticDataset) -> None:
    for bundle in corpus.documents():
        page = bundle.document
        for token in bundle.gt_tokens:
            assert token.bbox is not None
            assert 0 <= token.bbox.x0 <= page.width
            assert 0 <= token.bbox.y1 <= page.height


def test_corpus_generation_is_deterministic(isolated_env: Path) -> None:
    """Re-rendering must reproduce identical bytes. The write-once raw store enforces
    this: a nondeterministic generator would raise instead of silently drifting."""
    first = SyntheticDataset(n_documents=3, seed=99)
    first.materialize()
    digests = {b.document.document_id: b.document.image_sha256 for b in first.documents()}

    second = SyntheticDataset(n_documents=3, seed=99)
    assert second.materialize() == 0  # nothing to write; all pages already present
    assert {b.document.document_id: b.document.image_sha256 for b in second.documents()} == digests


def test_growing_the_corpus_does_not_perturb_existing_pages(isolated_env: Path) -> None:
    small = SyntheticDataset(n_documents=3, seed=7)
    small.materialize()
    before = {b.document.document_id: b.document.image_sha256 for b in small.documents()}

    larger = SyntheticDataset(n_documents=6, seed=7)
    larger.materialize()
    after = {b.document.document_id: b.document.image_sha256 for b in larger.documents()}

    assert all(after[doc_id] == digest for doc_id, digest in before.items())
    assert len(after) == 6


def test_preflight_reports_remediation_when_absent(isolated_env: Path) -> None:
    report = SyntheticDataset(n_documents=5).preflight()
    assert not report.available
    assert "ocr-risk data synth" in report.remediation
    with pytest.raises(FileNotFoundError, match="not usable"):
        report.raise_if_unavailable()


# --- raw layer ------------------------------------------------------------------------
def test_raw_responses_are_write_once(corpus: SyntheticDataset) -> None:
    store = RawStore()
    bundle = next(iter(corpus.documents()))
    document = bundle.document
    adapter = build_engine("synthetic", engine_id="synth_a", seed=1)
    page = PageInput(
        document_id=document.document_id,
        dataset_id=document.dataset_id,
        image_path=data_root() / document.image_path,
        image_sha256=document.image_sha256,
        width=document.width,
        height=document.height,
    )
    path = store.write_engine_response(adapter.recognize(page))
    with pytest.raises(ImmutableWriteError, match="write-once"):
        store.write_bytes(path, b"different bytes")


def test_different_engine_configs_write_to_different_paths(corpus: SyntheticDataset) -> None:
    """Re-running an engine after a config change must not overwrite earlier evidence."""
    store = RawStore()
    bundle = next(iter(corpus.documents()))
    document = bundle.document
    page = PageInput(
        document_id=document.document_id,
        dataset_id=document.dataset_id,
        image_path=data_root() / document.image_path,
        image_sha256=document.image_sha256,
        width=document.width,
        height=document.height,
    )
    paths = {
        store.write_engine_response(
            build_engine("synthetic", engine_id="e", seed=1, char_error_rate=rate).recognize(page)
        )
        for rate in (0.01, 0.20)
    }
    assert len(paths) == 2


# --- cross-engine behaviour ---------------------------------------------------------------
def test_every_engine_reads_every_page(corpus: SyntheticDataset) -> None:
    """Matched-source is the core benchmark principle: engine shift must not be
    confounded with document shift."""
    spans = _recognize_all(corpus)
    document_sets = {
        engine: {s.document_id for s in engine_spans} for engine, engine_spans in spans.items()
    }
    assert len({frozenset(v) for v in document_sets.values()}) == 1
    assert len(next(iter(document_sets.values()))) == PAGES


def test_engines_disagree_with_each_other(corpus: SyntheticDataset) -> None:
    spans = _recognize_all(corpus)
    readings = {
        engine: tuple(
            s.text for s in sorted(engine_spans, key=lambda s: (s.document_id, s.reading_order))
        )
        for engine, engine_spans in spans.items()
    }
    assert len(set(readings.values())) == len(readings), "engines produced identical readings"


def test_engines_differ_in_error_rate_and_segmentation(corpus: SyntheticDataset) -> None:
    """The profiles must differ in *kind*, not only in seed: a low-noise engine and a
    segmentation-prone one are the two ends the cross-engine study depends on."""
    spans = _recognize_all(corpus)
    gt_tokens = {b.document.document_id: {t.text for t in b.gt_tokens} for b in corpus.documents()}

    error_rate = {}
    for engine, engine_spans in spans.items():
        wrong = sum(1 for s in engine_spans if s.text not in gt_tokens[s.document_id])
        error_rate[engine] = wrong / max(len(engine_spans), 1)

    # synth_d is the deliberately clean engine; synth_a the noisiest.
    assert error_rate["synth_d"] < error_rate["synth_a"], error_rate
    # synth_b splits and merges aggressively, so its span count departs most from the rest.
    counts = {engine: len(s) for engine, s in spans.items()}
    assert counts["synth_b"] != counts["synth_d"], counts


def test_confidence_scales_stay_distinct_across_engines(corpus: SyntheticDataset) -> None:
    """synth_b reports 0-100 and the others 0-1. Pooling them without the scale would
    silently compare incomparable numbers."""
    spans = _recognize_all(corpus)
    scales = {
        engine: {s.conf_scale for s in engine_spans} for engine, engine_spans in spans.items()
    }
    assert scales["synth_b"] == {"synthetic_0_100"}
    assert scales["synth_a"] == {"synthetic_0_1"}

    b_confs = [s.native_conf_recognition for s in spans["synth_b"] if s.native_conf_recognition]
    assert max(b_confs) > 1.0, "0-100 confidences were normalized away"


def test_canonical_spans_have_consistent_offsets(corpus: SyntheticDataset) -> None:
    spans = _recognize_all(corpus)
    for engine_spans in spans.values():
        by_document: dict[str, list] = {}
        for span in engine_spans:
            by_document.setdefault(span.document_id, []).append(span)
        for document_spans in by_document.values():
            ordered = sorted(document_spans, key=lambda s: s.reading_order)
            assert [s.reading_order for s in ordered] == list(range(len(ordered)))
            for span in ordered:
                assert span.char_end - span.char_start == len(span.text)
            for previous, current in pairwise(ordered):
                assert current.char_start >= previous.char_end

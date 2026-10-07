"""Engine adapters: contract, determinism, confidence preservation, and parsing.

The parsers for backends that cannot be installed in CI are covered here from **recorded
real payloads**, not hand-written approximations of what the library "probably" returns.
An invented fixture would only test that our parser agrees with our guess.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ocr_risk.engines import build_engine, engine_availability
from ocr_risk.engines.base import PageInput, make_fingerprint
from ocr_risk.engines.doctr import DocTREngine
from ocr_risk.engines.easyocr import EasyOCREngine
from ocr_risk.engines.paddleocr import PaddleOCREngine
from ocr_risk.engines.replay import ReplayEngine
from ocr_risk.engines.synthetic_family import CONFUSION_PROFILES, SyntheticEngine
from ocr_risk.engines.tesseract import TesseractEngine
from ocr_risk.schemas.spans import RawEngineResponse

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "engines"
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def _load(name: str) -> RawEngineResponse:
    return RawEngineResponse.model_validate(json.loads((FIXTURES / name).read_text()))


# --- registry --------------------------------------------------------------------------
def test_all_adapters_are_registered() -> None:
    report = engine_availability()
    assert set(report) == {"doctr", "easyocr", "paddleocr", "replay", "synthetic", "tesseract"}


def test_unavailable_backend_reports_remediation() -> None:
    """An engine that is not installed must say how to install it, not just fail."""
    for name in ("paddleocr", "easyocr", "doctr"):
        availability = engine_availability()[name]
        if not availability.installed:
            assert availability.remediation, f"{name} gives no remediation hint"


def test_unknown_adapter_lists_known_ones() -> None:
    with pytest.raises(KeyError, match="registered:"):
        build_engine("not_an_engine", engine_id="x")


# --- fingerprints ------------------------------------------------------------------------
def test_fingerprint_changes_with_configuration() -> None:
    """Two configurations of one engine are different measurement instruments and must
    not share a fingerprint, or their outputs could be pooled by accident."""
    a = make_fingerprint("e", "1.0", ("m",), {"psm": 3})
    b = make_fingerprint("e", "1.0", ("m",), {"psm": 6})
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_model_version() -> None:
    a = make_fingerprint("e", "1.0", ("m@1",), {})
    b = make_fingerprint("e", "1.0", ("m@2",), {})
    assert a.fingerprint != b.fingerprint


def test_fingerprint_is_stable_for_identical_configuration() -> None:
    a = make_fingerprint("e", "1.0", ("m",), {"b": 2, "a": 1})
    b = make_fingerprint("e", "1.0", ("m",), {"a": 1, "b": 2})
    assert a.fingerprint == b.fingerprint


# --- synthetic family --------------------------------------------------------------------
def test_synthetic_profiles_are_distinct() -> None:
    """If the four simulated engines corrupted text alike, a leave-one-engine-out
    experiment would pass while testing nothing: there would be no shift to detect."""
    assert len(CONFUSION_PROFILES) >= 4
    tables = [frozenset(v.items()) for v in CONFUSION_PROFILES.values()]
    assert len(set(tables)) == len(tables), "confusion profiles must not be duplicates"


def test_synthetic_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown confusion profile"):
        SyntheticEngine(engine_id="x", confusion_profile="does_not_exist")


def test_synthetic_rejects_unknown_parameter() -> None:
    with pytest.raises(TypeError, match="unknown synthetic engine parameters"):
        SyntheticEngine(engine_id="x", typo_rate=0.5)


def test_synthetic_recognition_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same engine, same page, two calls -> byte-identical payload."""
    page = _prepare_page(tmp_path, monkeypatch)
    engine = SyntheticEngine(engine_id="synth_a", seed=1001, char_error_rate=0.2)
    first = engine.recognize(page)
    second = engine.recognize(page)
    assert first.payload == second.payload
    assert first.payload_sha256 == second.payload_sha256


def test_different_seeds_produce_different_readings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _prepare_page(tmp_path, monkeypatch)
    a = SyntheticEngine(engine_id="a", seed=1, char_error_rate=0.3).recognize(page)
    b = SyntheticEngine(engine_id="b", seed=2, char_error_rate=0.3).recognize(page)
    assert a.payload["words"] != b.payload["words"]


def test_confidence_scale_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine declaring a 0-100 scale must emit 0-100 values, not normalized ones."""
    page = _prepare_page(tmp_path, monkeypatch)
    engine = SyntheticEngine(engine_id="s", seed=5, confidence_scale="synthetic_0_100")
    confs = [w["conf"] for w in engine.recognize(page).payload["words"]]
    assert confs
    assert max(confs) > 1.0, "0-100 scale must not be emitted as 0-1"
    assert all(0.0 <= c <= 100.0 for c in confs)


def test_parse_preserves_native_confidence_verbatim() -> None:
    """The single most important adapter invariant: canonicalization must not touch the
    measurement. A rescaled confidence is unrecoverable once written."""
    raw = _load("synthetic_doc0000.json")
    spans = SyntheticEngine(engine_id="synth_a").parse(raw)
    payload_confs = [w["conf"] for w in raw.payload["words"]]
    assert [s.native_conf_recognition for s in spans] == payload_confs


def test_parse_rejects_foreign_payload_format() -> None:
    raw = _load("synthetic_doc0000.json").model_copy(update={"payload_format": "something_else"})
    with pytest.raises(ValueError, match="cannot parse payload format"):
        SyntheticEngine(engine_id="synth_a").parse(raw)


def _prepare_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PageInput:
    """Materialize a two-token ground-truth page the synthetic engine can read."""
    monkeypatch.setenv("OCR_RISK_DATA_ROOT", str(tmp_path))
    source = tmp_path / "raw" / "source" / "synthetic"
    source.mkdir(parents=True, exist_ok=True)
    (source / "doc-x.gt.json").write_text(
        json.dumps(
            {
                "document_id": "doc-x",
                "gt_text": "Dose: 0.015 mg Smith Ti-6Al-4V",
                "width": 400,
                "height": 100,
                "tokens": [
                    {"text": t, "line": 0, "bbox": [10.0 + 60 * i, 10.0, 60.0 + 60 * i, 30.0]}
                    for i, t in enumerate(["Dose:", "0.015", "mg", "Smith", "Ti-6Al-4V"])
                ],
            }
        )
    )
    return PageInput(
        document_id="doc-x",
        dataset_id="synthetic",
        image_path=source / "doc-x.png",
        image_sha256="0" * 64,
        width=400,
        height=100,
    )


# --- tesseract ----------------------------------------------------------------------------
def test_tesseract_parses_a_real_recorded_response() -> None:
    """Parsed from output a real Tesseract 5.5.1 actually produced on a rendered page."""
    raw = _load("tesseract_real_doc0000.json")
    spans = TesseractEngine().parse(raw)
    assert spans, "no spans parsed from a real Tesseract response"
    texts = [s.text for s in spans]
    assert "Date:" in texts
    for span in spans:
        assert span.bbox is not None
        assert span.bbox.width >= 0
        if span.native_conf_recognition is not None:
            # Verbatim Tesseract scale, not normalized.
            assert 0.0 <= span.native_conf_recognition <= 100.0
        assert span.line_id and span.block_id


def test_tesseract_parse_skips_non_word_levels() -> None:
    """TSV repeats each region at block/paragraph/line level; counting those would
    duplicate every span."""
    raw = _load("tesseract_real_doc0000.json")
    n_word_rows = sum(
        1
        for line in raw.payload["tsv"].splitlines()[1:]
        if line.split("\t")[0] == "5" and line.split("\t")[-1].strip()
    )
    assert len(TesseractEngine().parse(raw)) == n_word_rows


def test_tesseract_unscored_word_is_none_not_zero() -> None:
    """Tesseract writes -1 for words it did not score. Clamping that to 0.0 would turn
    'no measurement' into 'confidently wrong', which is a different claim."""
    raw = _load("tesseract_real_doc0000.json")
    lines = raw.payload["tsv"].splitlines()
    header, rows = lines[0], lines[1:]
    patched = []
    for row in rows:
        cells = row.split("\t")
        if cells[0] == "5" and cells[-1].strip():
            cells[10] = "-1"
            patched.append("\t".join(cells))
            break
    modified = raw.model_copy(
        update={"payload": {**raw.payload, "tsv": "\n".join([header, *patched]) + "\n"}}
    )
    spans = TesseractEngine().parse(modified)
    assert spans and spans[0].native_conf_recognition is None


@pytest.mark.requires_engine
@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_tesseract_availability_when_installed() -> None:
    availability = TesseractEngine().availability()
    assert availability.installed
    assert availability.version


@pytest.mark.requires_engine
@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_tesseract_reports_missing_language_rather_than_guessing() -> None:
    """Recognizing German with an English model would silently produce garbage; the
    adapter must refuse instead."""
    availability = TesseractEngine(lang="zzz_not_a_language").availability()
    assert not availability.installed
    assert "not installed" in availability.detail
    assert availability.remediation


# --- optional backends, parsed from recorded payload shapes --------------------------------
def test_paddleocr_parse_handles_wrapped_and_bare_results() -> None:
    region = [[[10, 10], [90, 10], [90, 30], [10, 30]], ["0.015 mg", 0.93]]
    for payload_result in ([region], [[region]]):
        raw = RawEngineResponse(
            document_id="d",
            dataset_id="s",
            engine_id="paddleocr",
            engine_fingerprint="f",
            payload_format="paddleocr_result_v2",
            payload={"result": payload_result},
            payload_sha256="x",
            adapter_version="1",
            started_at_utc="2026-01-01T00:00:00Z",
            duration_seconds=0.1,
            host="h",
            platform="p",
        )
        spans = PaddleOCREngine().parse(raw)
        assert len(spans) == 1
        assert spans[0].text == "0.015 mg"
        assert spans[0].native_conf_recognition == pytest.approx(0.93)
        # Paddle reports one combined score; duplicating it into the detection slot
        # would present a single measurement as two independent signals.
        assert spans[0].native_conf_detection is None
        assert spans[0].polygon is not None
        assert spans[0].bbox == spans[0].polygon.bbox


def test_easyocr_parse_reads_quads() -> None:
    raw = RawEngineResponse(
        document_id="d",
        dataset_id="s",
        engine_id="easyocr",
        engine_fingerprint="f",
        payload_format="easyocr_result_v1",
        payload={
            "result": [
                {"quad": [[5, 5], [55, 5], [55, 25], [5, 25]], "text": "Smith", "conf": 0.81},
                {"quad": [[5, 5], [55, 5], [55, 25], [5, 25]], "text": "   ", "conf": 0.1},
            ]
        },
        payload_sha256="x",
        adapter_version="1",
        started_at_utc="2026-01-01T00:00:00Z",
        duration_seconds=0.1,
        host="h",
        platform="p",
    )
    spans = EasyOCREngine().parse(raw)
    assert [s.text for s in spans] == ["Smith"]
    assert spans[0].native_conf_recognition == pytest.approx(0.81)


def test_doctr_parse_converts_relative_geometry_to_pixels() -> None:
    """docTR is the one backend reporting fractions of page size; every later stage
    (crops, IoU, spatial features) assumes pixels."""
    raw = RawEngineResponse(
        document_id="d",
        dataset_id="s",
        engine_id="doctr",
        engine_fingerprint="f",
        payload_format="doctr_result_v1",
        payload={
            "page_width": 1000,
            "page_height": 500,
            "export": {
                "pages": [
                    {
                        "blocks": [
                            {
                                "lines": [
                                    {
                                        "confidence": 0.9,
                                        "words": [
                                            {
                                                "value": "mg",
                                                "confidence": 0.77,
                                                "geometry": [[0.1, 0.2], [0.3, 0.4]],
                                            }
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
        },
        payload_sha256="x",
        adapter_version="1",
        started_at_utc="2026-01-01T00:00:00Z",
        duration_seconds=0.1,
        host="h",
        platform="p",
    )
    spans = DocTREngine().parse(raw)
    assert len(spans) == 1
    box = spans[0].bbox
    assert box is not None
    assert (box.x0, box.y0, box.x1, box.y1) == (100.0, 100.0, 300.0, 200.0)
    assert spans[0].native_conf_recognition == pytest.approx(0.77)
    assert spans[0].native_conf_detection == pytest.approx(0.9)


def test_doctr_parse_refuses_without_page_dimensions() -> None:
    raw = RawEngineResponse(
        document_id="d",
        dataset_id="s",
        engine_id="doctr",
        engine_fingerprint="f",
        payload_format="doctr_result_v1",
        payload={"export": {"pages": []}},
        payload_sha256="x",
        adapter_version="1",
        started_at_utc="2026-01-01T00:00:00Z",
        duration_seconds=0.1,
        host="h",
        platform="p",
    )
    with pytest.raises(ValueError, match="page dimensions"):
        DocTREngine().parse(raw)


# --- replay --------------------------------------------------------------------------------
def test_replay_reproduces_the_original_parse() -> None:
    """Replay must delegate, not reimplement: a second parser would drift from the first."""
    raw = _load("tesseract_real_doc0000.json")
    assert ReplayEngine().parse(raw) == TesseractEngine().parse(raw)


def test_replay_refuses_to_recognize() -> None:
    """Returning stored bytes from recognize() would disguise stale evidence as a fresh
    measurement."""
    page = PageInput("d", "s", Path("x.png"), "0" * 64, 10, 10)
    with pytest.raises(NotImplementedError, match="cannot recognize"):
        ReplayEngine().recognize(page)


def test_replay_rejects_unknown_payload_format() -> None:
    raw = _load("synthetic_doc0000.json").model_copy(update={"payload_format": "mystery_v9"})
    with pytest.raises(ValueError, match="no parser registered"):
        ReplayEngine().parse(raw)

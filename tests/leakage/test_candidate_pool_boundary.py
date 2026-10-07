"""The natural pool and the challenge pool must never merge.

Synthesized hard negatives are ~99% harmful by construction and can outnumber naturally
generated candidates several to one. Pooling them does not make the benchmark harder in a
useful way — it changes the question from "how much genuine repair can be safely
automated" to "can you reject adversarial examples", and the second has a much better
answer. The pilot kept them apart with a boolean named for its cause; these tests hold the
boundary at the schema and at every denominator that crosses it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from ocr_risk.candidates.base import CandidateProposal, GenerationContext
from ocr_risk.candidates.pipeline import SiteContext, generate_candidates
from ocr_risk.candidates.registry import build_generator
from ocr_risk.config.models import CandidateConfig
from ocr_risk.experiments.generator_study import GeneratorSpec, run_generator_study
from ocr_risk.schemas.candidates import Candidate
from ocr_risk.schemas.enums import CandidatePool, HarmPolicy, SiteKind, SplitRole
from ocr_risk.schemas.sites import CorrectionSite
from ocr_risk.splits.document_partition import DocumentPartition


def _candidate(**overrides: object) -> Candidate:
    fields: dict[str, object] = {
        "candidate_id": "c1",
        "site_id": "s1",
        "document_id": "d1",
        "dataset_id": "funsd",
        "engine_id": "tesseract",
        "candidate_text": "address",
        "generator_id": "lexical",
        "generator_version": "1",
        "generator_rank": 0,
    }
    fields.update(overrides)
    return Candidate(**fields)  # type: ignore[arg-type]


def test_a_candidate_is_natural_unless_it_says_otherwise() -> None:
    assert _candidate().pool is CandidatePool.NATURAL


def test_the_pool_cannot_disagree_with_the_fact_it_is_derived_from() -> None:
    """One stored fact, one typed accessor. Two columns is how the denominator drifts."""
    assert _candidate(is_synthetic_hard_negative=True).pool is CandidatePool.CHALLENGE
    assert "pool" not in Candidate.model_fields, (
        "a second stored copy of the same fact can be written out of step with the first"
    )
    with pytest.raises(ValidationError):
        _candidate(pool=CandidatePool.CHALLENGE)


def test_the_pool_survives_a_parquet_round_trip(tmp_path: object) -> None:
    """Derived or not, the distinction has to still be there after a write and a read."""
    from pathlib import Path

    from ocr_risk.io.parquet import read_records, write_records
    from ocr_risk.schemas.arrow import get_table

    spec = get_table("candidates")
    path = Path(str(tmp_path)) / "candidates.parquet"
    write_records(
        path,
        spec,
        [
            _candidate(candidate_id="c1"),
            _candidate(
                candidate_id="c2",
                is_synthetic_hard_negative=True,
                hard_negative_family="numeric_magnitude",
            ),
        ],
    )
    assert [c.pool for c in read_records(path, spec)] == [
        CandidatePool.NATURAL,
        CandidatePool.CHALLENGE,
    ]


class _StubHardNegative:
    """A generator that emits challenge candidates, standing in for the real one."""

    generator_id = "stub_hard_negative"
    version = "1"

    def available(self) -> bool:
        return True

    def fit(self, corpus: object) -> None:
        return None

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        return [
            CandidateProposal(
                text=context.original_ocr.replace("0", "8") + "X",
                is_synthetic_hard_negative=True,
                hard_negative_family="numeric_magnitude",
            )
        ]


def _site_context() -> object:

    site = CorrectionSite(
        site_id="s1",
        document_id="d1",
        dataset_id="funsd",
        engine_id="tesseract",
        alignment_ids=("a1",),
        ocr_span_ids=("sp1",),
        gt_token_ids=("g1",),
        ocr_text="addres",
        gt_text="address",
        d_before=1,
        site_kind=SiteKind.SUBSTITUTION,
        evaluable=True,
        char_start=0,
        char_end=6,
        reading_order_start=0,
        min_align_confidence=1.0,
    )
    return SiteContext(
        site=site,
        context_before="zip ",
        context_after=" NY",
        native_confidences=(0.9,),
        conf_scale="tesseract_0_100",
        image_sha256="0" * 64,
        image_width=100,
        image_height=100,
    )


def test_the_generation_pipeline_labels_each_pool_from_its_generator() -> None:
    lexical = build_generator("lexical", generator_id="lexical")
    lexical.fit(["address address invoice invoice"])
    candidates = generate_candidates(
        [_site_context()],  # type: ignore[list-item]
        [(lexical, 2), (_StubHardNegative(), 1)],  # type: ignore[list-item]
        CandidateConfig(max_candidates_per_site=8),
    )
    by_pool = {c.pool for c in candidates}
    assert CandidatePool.NATURAL in by_pool, "the lexical generator must have proposed"
    assert CandidatePool.CHALLENGE in by_pool, "the stub must have produced a challenge row"
    for candidate in candidates:
        assert (candidate.pool is CandidatePool.CHALLENGE) == candidate.is_synthetic_hard_negative


def test_the_headline_selection_mask_keeps_the_challenge_pool_out() -> None:
    """The runner's own filter, checked against the enum rather than the boolean."""
    from ocr_risk.config.models import ExperimentConfig
    from ocr_risk.experiments.loeo_runner import _selection_mask

    frame = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3"],
            "is_synthetic_hard_negative": [False, True, False],
            "pool": ["natural", "challenge", "natural"],
        }
    )
    from ocr_risk.config.models import EngineConfig

    engines = (
        EngineConfig(id="tesseract", adapter="tesseract"),
        EngineConfig(id="doctr", adapter="doctr"),
    )
    config = ExperimentConfig(name="x", engines=engines)
    assert config.candidates.hard_negatives_in_evaluation is False
    assert list(_selection_mask(frame, config)) == [True, False, True]

    opted_in = ExperimentConfig(
        name="x", engines=engines, candidates=CandidateConfig(hard_negatives_in_evaluation=True)
    )
    assert list(_selection_mask(frame, opted_in)) == [True, True, True]


def test_the_generator_study_never_emits_a_challenge_candidate() -> None:
    """The readiness study measures the natural pool. Nothing else may enter it."""
    import inspect

    from ocr_risk.experiments import generator_ladder, generator_study

    for spec in generator_ladder.GENERATOR_LADDER.values():
        assert spec.kind != "hard_negative", f"{spec.id} would put challenge rows in the study"
    source = inspect.getsource(generator_study.run_generator_study)
    assert "is_synthetic_hard_negative" not in source


def test_the_generator_study_refuses_a_hard_negative_spec() -> None:
    """Red team: a challenge generator handed to the study must fail, not be absorbed.

    The ladder assertion above is static; this exercises the runtime guard so a future
    ladder edit (or a caller passing custom specs) cannot silently pool challenge rows
    into the natural study's denominators.
    """
    site = CorrectionSite(
        site_id="d1:tesseract:0",
        document_id="d1",
        dataset_id="funsd",
        engine_id="tesseract",
        alignment_ids=("a0",),
        ocr_span_ids=("sp0",),
        gt_token_ids=("g0",),
        ocr_text="0.015",
        gt_text="0.015",
        d_before=0,
        site_kind=SiteKind.CLEAN,
        evaluable=True,
        char_start=0,
        char_end=5,
        reading_order_start=0,
        min_align_confidence=1.0,
    )
    context = SiteContext(
        site=site,
        context_before="zip ",
        context_after=" NY",
        native_confidences=(0.9,),
        conf_scale="tesseract_0_100",
        image_sha256="0" * 64,
        image_width=100,
        image_height=100,
    )
    partition = DocumentPartition(
        role_of={"d1": SplitRole.CALIBRATE}, spec_hash="s" * 64, partition_sha256="p" * 64
    )
    spec = GeneratorSpec(id="hard_negative", kind="hard_negative", params={}, max_candidates=2)
    with pytest.raises(ValueError, match="natural candidates only"):
        run_generator_study([context], partition, [spec], HarmPolicy.STRICT_WORSENING)

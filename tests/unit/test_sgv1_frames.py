"""SGV1 dual-frame construction: the freeze order, the guards, and the sampling.

Hand-computed expectations throughout -- the per-document cap and inverse-probability
weights are verified against arithmetic done in the test body, not against the
implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ocr_risk.experiments.sgv1_design import (
    FRAME_DEGRADATION,
    FRAME_NATURAL,
    VALID_FRAMES,
)
from ocr_risk.experiments.sgv1_frames import (
    FRAME_CLASS_UNRESOLVED,
    OUTCOME_UNRESOLVED,
    FrameError,
    attach_labels,
    build_frame_a,
    build_frame_b,
    build_matched_pairs,
    evaluation_class,
    freeze_candidates,
)


def _candidates() -> pd.DataFrame:
    """Three documents, sites mixing classes so pairs exist.

    Layout (candidate_id -> doc, site, class it will receive):
      doc-a: s1 c0000,c0001 ben | c0002,c0003 harm;  s2 c0004 ben | c0005 harm
      doc-b: s3 c0006,c0007 ben | c0008,c0009 harm;  s4 c0010 neutral
      doc-c: s5 c0011,c0012 ben | c0013,c0014 harm;  s7 c0015 harm; s8 c0016 neutral
    Populations: beneficial 7 (3+2+2), harmful 8 (3+2+3), neutral 2.
    """
    rows = []
    spec = [
        ("doc-a", "s1", 2, 2),
        ("doc-a", "s2", 1, 1),
        ("doc-b", "s3", 2, 2),
        ("doc-b", "s4", 0, 0),
        ("doc-c", "s5", 2, 2),
        ("doc-c", "s7", 0, 1),
        ("doc-c", "s8", 0, 0),
    ]
    kinds_for_site = {
        "s4": [("lateral_change", 1)],
        "s8": [("identity", 1)],
    }
    index = 0
    for document_id, site_id, n_beneficial, n_harmful in spec:
        kinds = kinds_for_site.get(site_id) or [
            ("true_correction", n_beneficial),
            ("miscorrection", n_harmful),
        ]
        for kind, count in kinds:
            for _ in range(count):
                rows.append(
                    {
                        "candidate_id": f"c-{index:04d}",
                        "site_id": site_id,
                        "document_id": document_id,
                        "engine_id": "synth_a",
                        "generator_id": "rule_pool",
                        "original_ocr": "ocr",
                        "candidate_text": "cand",
                        "_class_hint": kind,
                    }
                )
                index += 1
    return pd.DataFrame(rows)


def _labels(candidates: pd.DataFrame) -> pd.DataFrame:
    harmful = {"miscorrection", "overcorrection"}
    return pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"],
            "outcome": candidates["_class_hint"],
            "is_harmful": candidates["_class_hint"].isin(harmful),
            "d_before": [0] * len(candidates),
            "d_after": [0] * len(candidates),
        }
    )


@pytest.fixture()
def frozen_pool():
    candidates = _candidates().drop(columns=["_class_hint"])
    freeze = freeze_candidates(candidates, frame=FRAME_NATURAL)
    labeled = attach_labels(candidates, freeze, _labels(_candidates()))
    return candidates, freeze, labeled


class TestFreeze:
    def test_freeze_records_pool_shape(self, frozen_pool) -> None:
        _, freeze, _ = frozen_pool
        assert freeze.n_rows == 17
        assert freeze.n_documents == 3
        assert freeze.generator_ids == ("rule_pool",)
        assert freeze.frame == FRAME_NATURAL

    def test_freeze_hash_is_exact_not_set_semantics(self, frozen_pool) -> None:
        candidates, freeze, _ = frozen_pool
        reordered = candidates.iloc[::-1].reset_index(drop=True)
        # Same rows in a different order are a different frozen table...
        assert (
            freeze_candidates(reordered, frame=FRAME_NATURAL).candidates_sha256
            != freeze.candidates_sha256
        )
        # ...so it is refused wherever the freeze binding is re-checked.
        with pytest.raises(FrameError, match="does not match its freeze record"):
            attach_labels(reordered, freeze, _labels(_candidates()))

    def test_same_table_refreezes_identically(self, frozen_pool) -> None:
        candidates, freeze, _ = frozen_pool
        again = freeze_candidates(candidates, frame=FRAME_NATURAL)
        assert again.candidates_sha256 == freeze.candidates_sha256

    def test_unknown_frame_rejected(self) -> None:
        candidates = _candidates().drop(columns=["_class_hint"])
        with pytest.raises(FrameError, match="unknown frame"):
            freeze_candidates(candidates, frame="favourite")

    @pytest.mark.parametrize("column", ["gt_text", "outcome", "is_harmful", "d_after", "label"])
    def test_gt_or_label_columns_rejected_at_freeze(self, column: str) -> None:
        candidates = _candidates().drop(columns=["_class_hint"])
        contaminated = candidates.assign(**{column: ["x"] * len(candidates)})
        with pytest.raises(FrameError, match="ground-truth/label columns"):
            freeze_candidates(contaminated, frame=FRAME_NATURAL)

    def test_missing_key_columns_rejected(self) -> None:
        candidates = _candidates().drop(columns=["_class_hint", "generator_id"])
        with pytest.raises(FrameError, match="missing key columns"):
            freeze_candidates(candidates, frame=FRAME_NATURAL)


class TestAttachLabels:
    def test_labels_only_enter_through_the_whitelist(self, frozen_pool) -> None:
        candidates, freeze, _ = frozen_pool
        labels = _labels(_candidates()).assign(notes=["stray"] * len(candidates))
        labeled = attach_labels(candidates, freeze, labels)
        assert "notes" not in labeled.columns
        assert {"outcome", "is_harmful", "d_before", "d_after"} <= set(labeled.columns)

    def test_unlabeled_candidate_is_never_silently_dropped(self, frozen_pool) -> None:
        candidates, freeze, _ = frozen_pool
        short_labels = _labels(_candidates()).iloc[:-1]
        with pytest.raises(FrameError, match="no label"):
            attach_labels(candidates, freeze, short_labels)

    def test_duplicate_labels_rejected(self, frozen_pool) -> None:
        candidates, freeze, _ = frozen_pool
        duplicated = pd.concat([_labels(_candidates()), _labels(_candidates()).iloc[[0]]])
        with pytest.raises(FrameError, match="duplicate candidate_id"):
            attach_labels(candidates, freeze, duplicated)

    def test_labels_require_the_frozen_table(self, frozen_pool) -> None:
        candidates, freeze, _ = frozen_pool
        tampered = candidates.copy()
        tampered.loc[0, "candidate_text"] = "different"
        with pytest.raises(FrameError, match="does not match its freeze record"):
            attach_labels(tampered, freeze, _labels(_candidates()))


class TestFrameA:
    def test_cap_and_weights_hand_computed(self, frozen_pool) -> None:
        _, freeze, labeled = frozen_pool
        # Populations: beneficial 7 (doc-a 3, doc-b 2, doc-c 2), harmful 8
        # (doc-a 3, doc-b 2, doc-c 3), neutral 2 (excluded from the risk set).
        frame_a, populations = build_frame_a(
            labeled, freeze, per_document_cap=2, min_per_class=5, seed=7
        )
        # The unresolved population is stated even when it is zero: a record that omits
        # it cannot be distinguished later from one where nothing was checked.
        assert populations == {"beneficial": 7, "harmful": 8, "neutral": 2, "unresolved": 0}

        # Cap 2 per document per class: beneficial 2+2+2 = 6 of 7 -> weight 7/6;
        # harmful 2+2+2 = 6 of 8 -> weight 8/6.
        per_class_doc = frame_a.groupby(["evaluation_stratum", "document_id"]).size()
        assert (per_class_doc <= 2).all()
        weights = frame_a.groupby("evaluation_stratum")["sampling_weight"].first()
        assert weights["beneficial"] == pytest.approx(7 / 6)
        assert weights["harmful"] == pytest.approx(8 / 6)
        assert len(frame_a) == 12
        assert set(frame_a["frame"]) == {FRAME_NATURAL}

    def test_min_per_class_floor_fails_closed(self, frozen_pool) -> None:
        _, freeze, labeled = frozen_pool
        with pytest.raises(FrameError, match="below the required minimum"):
            build_frame_a(labeled, freeze, per_document_cap=1, min_per_class=99, seed=7)

    def test_sampling_is_seed_deterministic(self, frozen_pool) -> None:
        _, freeze, labeled = frozen_pool
        first, _ = build_frame_a(labeled, freeze, per_document_cap=1, min_per_class=2, seed=11)
        second, _ = build_frame_a(labeled, freeze, per_document_cap=1, min_per_class=2, seed=11)
        pd.testing.assert_frame_equal(first, second)
        other, _ = build_frame_a(labeled, freeze, per_document_cap=1, min_per_class=2, seed=12)
        assert not first["candidate_id"].equals(other["candidate_id"])

    def test_tampered_pool_cannot_be_sampled(self, frozen_pool) -> None:
        _, freeze, labeled = frozen_pool
        tampered = labeled.copy()
        tampered.loc[0, "original_ocr"] = "rewritten"
        with pytest.raises(FrameError, match="does not match its freeze record"):
            build_frame_a(tampered, freeze, per_document_cap=2, min_per_class=1, seed=7)


class TestMatchedPairs:
    def test_pairs_are_complete_sorted_and_deterministic(self) -> None:
        # A fixed Frame A table (doc-level sampling in build_frame_a is exercised in
        # TestFrameA; here the pair construction itself must be exact).
        frame_a = pd.DataFrame(
            {
                "site_id": ["s1", "s1", "s1", "s2", "s2"],
                "document_id": ["doc-a"] * 5,
                "engine_id": ["synth_a"] * 5,
                "candidate_id": ["c-b2", "c-b1", "c-h1", "c-b3", "c-n1"],
                "evaluation_class": [
                    "beneficial",
                    "beneficial",
                    "harmful",
                    "beneficial",
                    "neutral",
                ],
            }
        )
        pairs = build_matched_pairs(frame_a)
        # s1: {c-b1, c-b2} x {c-h1} in sorted order; s2 has no harmful member.
        expected = [
            ("s1", "c-b1", "c-h1"),
            ("s1", "c-b2", "c-h1"),
        ]
        assert (
            list(
                pairs[["site_id", "plus_candidate_id", "minus_candidate_id"]].itertuples(
                    index=False, name=None
                )
            )
            == expected
        )

    def test_pairs_never_cross_sites_documents_or_engines(self) -> None:
        # Same site, but every triple (site, document, engine) is distinct: pairing
        # on site membership alone would wrongly emit pairs here.
        frame_a = pd.DataFrame(
            {
                "site_id": ["s1", "s1", "s1"],
                "document_id": ["doc-a", "doc-a", "doc-b"],
                "engine_id": ["synth_a", "synth_b", "synth_a"],
                "candidate_id": ["c-b1", "c-h1", "c-h2"],
                "evaluation_class": ["beneficial", "harmful", "harmful"],
            }
        )
        assert build_matched_pairs(frame_a).empty

    def test_multi_harmful_site_gives_the_full_cross_product(self) -> None:
        frame_a = pd.DataFrame(
            {
                "site_id": ["s1"] * 5,
                "document_id": ["doc-a"] * 5,
                "engine_id": ["synth_a"] * 5,
                "candidate_id": ["c-b1", "c-b2", "c-h1", "c-h2", "c-h3"],
                "evaluation_class": [
                    "beneficial",
                    "beneficial",
                    "harmful",
                    "harmful",
                    "harmful",
                ],
            }
        )
        assert len(build_matched_pairs(frame_a)) == 6  # 2 x 3, complete cross product

    def test_single_class_site_yields_no_pairs(self) -> None:
        single = pd.DataFrame(
            {
                "site_id": ["s9"] * 3,
                "document_id": ["doc-c"] * 3,
                "engine_id": ["synth_a"] * 3,
                "candidate_id": ["c-1", "c-2", "c-3"],
                "evaluation_class": ["beneficial", "beneficial", "neutral"],
            }
        )
        assert build_matched_pairs(single).empty


class TestFrameB:
    def test_natural_stream_is_every_frozen_candidate(self, frozen_pool) -> None:
        _, freeze, labeled = frozen_pool
        frame_b = build_frame_b(labeled, freeze)
        assert len(frame_b) == 17  # nothing sampled away, nothing balanced
        assert set(frame_b["sampling_weight"]) == {1.0}
        assert set(frame_b["frame"]) == {FRAME_NATURAL}

    def test_frame_b_refuses_non_natural_freeze(self) -> None:
        candidates = _candidates().drop(columns=["_class_hint"])
        degradation_freeze = freeze_candidates(candidates, frame=FRAME_DEGRADATION)
        labeled = attach_labels(candidates, degradation_freeze, _labels(_candidates()))
        with pytest.raises(FrameError, match="natural stream"):
            build_frame_b(labeled, degradation_freeze)

    def test_degradation_is_a_valid_frame_for_frame_a(self) -> None:
        candidates = _candidates().drop(columns=["_class_hint"])
        degradation_freeze = freeze_candidates(candidates, frame=FRAME_DEGRADATION)
        labeled = attach_labels(candidates, degradation_freeze, _labels(_candidates()))
        frame_a, _ = build_frame_a(
            labeled, degradation_freeze, per_document_cap=2, min_per_class=3, seed=7
        )
        assert set(frame_a["frame"]) == {FRAME_DEGRADATION}


class TestEvaluationClass:
    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            ("true_correction", "beneficial"),
            ("partial_improvement", "beneficial"),
            ("miscorrection", "harmful"),
            ("overcorrection", "harmful"),
            ("lateral_change", "neutral"),
            ("identity", "neutral"),
        ],
    )
    def test_taxonomy_mapping(self, outcome: str, expected: str) -> None:
        assert evaluation_class(outcome) == expected


def test_valid_frames_vocabulary() -> None:
    assert frozenset({"natural", "controlled_degradation", "challenge"}) == VALID_FRAMES


def test_rng_helper_is_not_needed_module_level() -> None:
    # The module must not keep module-level RNG state: sampling seeds are arguments.
    import ocr_risk.experiments.sgv1_frames as module

    assert not any(
        isinstance(getattr(module, name, None), np.random.Generator) for name in dir(module)
    )


class TestUnresolved:
    """A candidate with no established region ground truth is not an example.

    It cannot be a negative one -- nothing says the edit is bad -- and it cannot be a
    neutral one, because neutral is a measurement. Before ``unresolved`` existed as a
    class, ``evaluation_class`` mapped it to neutral, which would have put 69,464
    unlabelable SGV1 candidates into the benchmark as if they had been scored.
    """

    def test_unresolved_is_its_own_class_not_neutral(self) -> None:
        assert evaluation_class(OUTCOME_UNRESOLVED) == FRAME_CLASS_UNRESOLVED
        assert evaluation_class("lateral_change") == "neutral"
        assert evaluation_class("identity") == "neutral"

    def _pool(self):
        candidates = _candidates().drop(columns=["_class_hint"])
        labels = _labels(_candidates())
        # Turn one harmful and one beneficial row into unresolved rows.
        labels.loc[labels.index[0], "outcome"] = OUTCOME_UNRESOLVED
        labels.loc[labels.index[0], "is_harmful"] = False
        labels.loc[labels.index[-1], "outcome"] = OUTCOME_UNRESOLVED
        labels.loc[labels.index[-1], "is_harmful"] = False
        freeze = freeze_candidates(candidates, frame=FRAME_NATURAL)
        return candidates, freeze, attach_labels(candidates, freeze, labels)

    def test_frame_a_never_samples_an_unresolved_candidate(self) -> None:
        _, freeze, labeled = self._pool()
        frame_a, populations = build_frame_a(
            labeled, freeze, per_document_cap=5, min_per_class=1, seed=3
        )
        assert populations[FRAME_CLASS_UNRESOLVED] == 2
        assert FRAME_CLASS_UNRESOLVED not in set(frame_a["evaluation_class"])
        assert OUTCOME_UNRESOLVED not in set(frame_a["outcome"])
        unresolved_ids = set(labeled.loc[labeled["outcome"] == OUTCOME_UNRESOLVED, "candidate_id"])
        assert not (set(frame_a["candidate_id"]) & unresolved_ids)

    def test_matched_pairs_never_reference_an_unresolved_candidate(self) -> None:
        _, freeze, labeled = self._pool()
        frame_a, _ = build_frame_a(labeled, freeze, per_document_cap=5, min_per_class=1, seed=3)
        pairs = build_matched_pairs(frame_a)
        unresolved_ids = set(labeled.loc[labeled["outcome"] == OUTCOME_UNRESOLVED, "candidate_id"])
        referenced = set(pairs["plus_candidate_id"]) | set(pairs["minus_candidate_id"])
        assert not (referenced & unresolved_ids)

    def test_frame_b_keeps_unresolved_rows_visible(self) -> None:
        _, freeze, labeled = self._pool()
        frame_b = build_frame_b(labeled, freeze)
        assert len(frame_b) == len(labeled)
        assert (frame_b["evaluation_class"] == FRAME_CLASS_UNRESOLVED).sum() == 2

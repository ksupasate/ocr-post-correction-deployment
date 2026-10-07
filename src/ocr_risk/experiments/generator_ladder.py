"""The generator ladder the readiness study compares.

Declared as data in one place so the study, the gate, and the documentation cannot
describe different ladders. Candidate generation is a **baseline subsystem**: every rung
here is established prior art (a lexicon-based corrector, detection before correction, a
byte-level seq2seq model), and none of it is claimed as a contribution. The question the
ladder answers is whether *any* of them produces a pool worth verifying.
"""

from __future__ import annotations

from ocr_risk.experiments.generator_study import GeneratorSpec

__all__ = ["GENERATOR_LADDER"]

GENERATOR_LADDER: dict[str, GeneratorSpec] = {
    "g0_lexical": GeneratorSpec(
        id="g0_lexical",
        kind="lexical",
        params={"max_edit_distance": 2, "min_lexicon_frequency": 2, "skip_in_vocabulary": True},
        max_candidates=4,
    ),
    "g1_error_gated": GeneratorSpec(
        id="g1_error_gated",
        kind="error_gated",
        params={
            "inner_kind": "lexical",
            "inner_params": {"max_edit_distance": 2, "min_lexicon_frequency": 2},
            "threshold": 0.5,
        },
        max_candidates=4,
    ),
    "g2_byt5": GeneratorSpec(
        id="g2_byt5",
        kind="byt5",
        params={"context_chars": 40, "num_beams": 4, "batch_size": 24},
        max_candidates=1,
        # English only. The checkpoint is fine-tuned on English; scoring it on German
        # Fraktur or Indonesian receipts would measure a language mismatch and report it
        # as generator quality.
        datasets=("funsd",),
    ),
    "g4_union": GeneratorSpec(
        id="g4_union",
        kind="composite",
        # The configuration `candidates/pipeline.py` actually builds when a config lists
        # more than one generator: every member's proposals, de-duplicated. The rungs were
        # scored individually, which is not what the pipeline runs, and the union's reach
        # is strictly the union of theirs while its clean-span exposure is the worse of
        # the two -- so it is a real trade, not a free improvement.
        params={
            "members": [
                {
                    "id": "lexical",
                    "kind": "lexical",
                    "params": {
                        "max_edit_distance": 2,
                        "min_lexicon_frequency": 2,
                        "skip_in_vocabulary": True,
                    },
                },
                {
                    "id": "edit_aware",
                    "kind": "edit_aware",
                    "params": {
                        "inner_kind": "error_gated",
                        "inner_params": {
                            "inner_kind": "lexical",
                            "inner_params": {
                                "max_edit_distance": 2,
                                "min_lexicon_frequency": 2,
                            },
                            "threshold": 0.5,
                        },
                    },
                },
            ]
        },
        max_candidates=4,
    ),
    "g3_edit_aware": GeneratorSpec(
        id="g3_edit_aware",
        kind="edit_aware",
        params={
            "inner_kind": "error_gated",
            "inner_params": {
                "inner_kind": "lexical",
                "inner_params": {"max_edit_distance": 2, "min_lexicon_frequency": 2},
                "threshold": 0.5,
            },
        },
        max_candidates=4,
    ),
    # --- CGV2 rungs (docs/cgv2/protocol.md §2). The frozen rungs above are unchanged. ---
    "g5_structural": GeneratorSpec(
        id="g5_structural",
        kind="structural",
        params={"min_bigram": 2},
        max_candidates=8,
    ),
    "g2_byt5_ctx0": GeneratorSpec(
        id="g2_byt5_ctx0",
        kind="byt5",
        params={"context_chars": 0, "num_beams": 4, "batch_size": 24},
        max_candidates=1,
        datasets=("funsd",),
    ),
    "g6_union": GeneratorSpec(
        id="g6_union",
        kind="composite",
        # Structural first: it fires only at structural sites, so at ordinary sites the
        # union is edit_aware's proposals in their own order, and at structural sites the
        # structural shapes lead instead of being pushed out by the inner corrector.
        params={
            "members": [
                {"id": "structural", "kind": "structural", "params": {"min_bigram": 2}},
                {
                    "id": "edit_aware",
                    "kind": "edit_aware",
                    "params": {
                        "inner_kind": "error_gated",
                        "inner_params": {
                            "inner_kind": "lexical",
                            "inner_params": {
                                "max_edit_distance": 2,
                                "min_lexicon_frequency": 2,
                            },
                            "threshold": 0.5,
                        },
                    },
                },
            ]
        },
        max_candidates=8,
    ),
}

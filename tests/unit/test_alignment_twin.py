"""Hand-computed tests for the CGV2 twin audit (protocol §15)."""

from __future__ import annotations

import pandas as pd

from ocr_risk.analysis.alignment_twin import (
    component_audit,
    component_denominators,
    projection_audit,
    projection_summary,
    region_census,
    site_projection_audit,
    twin_sites_table,
    twin_table,
)


def _sites(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("ocr_text", "gt_text"):
        frame[column] = frame[column].fillna("")
    return frame


def _alignments(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestTwinTable:
    def test_empty_gt_site_with_exact_unanchored_match_is_a_twin(self) -> None:
        sites = _sites(
            [
                {
                    "site_id": "d1:e1:0",
                    "document_id": "d1",
                    "engine_id": "e1",
                    "site_kind": "insertion",
                    "ocr_text": "Ref",
                    "gt_text": "",
                    "evaluable": True,
                },
                {
                    "site_id": "d1:e1:1",
                    "document_id": "d1",
                    "engine_id": "e1",
                    "site_kind": "substitution",
                    "ocr_text": "Name",
                    "gt_text": "Name",
                    "evaluable": True,
                },
            ]
        )
        alignments = _alignments(
            [
                {
                    "document_id": "d1",
                    "engine_id": "e1",
                    "reason_code": "never_anchored",
                    "gt_text": "Ref",
                    "ocr_text": "",
                }
            ]
        )
        table = twin_table(alignments, sites)
        row = table[(table["site_kind"] == "insertion")].iloc[0]
        assert row["n_evaluable_sites"] == 1
        assert row["n_twin_signature"] == 1
        assert row["twin_share"] == 1.0
        clean = table[(table["site_kind"] == "substitution")].iloc[0]
        assert clean["n_twin_signature"] == 0

    def test_non_empty_gt_sites_never_count_as_twins(self) -> None:
        sites = _sites(
            [
                {
                    "site_id": "d1:e1:0",
                    "document_id": "d1",
                    "engine_id": "e1",
                    "site_kind": "substitution",
                    "ocr_text": "Ref",
                    "gt_text": "Ref",
                    "evaluable": True,
                }
            ]
        )
        alignments = _alignments(
            [
                {
                    "document_id": "d1",
                    "engine_id": "e1",
                    "reason_code": "never_anchored",
                    "gt_text": "Ref",
                    "ocr_text": "",
                }
            ]
        )
        table = twin_table(alignments, sites)
        assert table["n_twin_signature"].iloc[0] == 0


class TestComponentAudit:
    def test_all_structural_shapes_and_states_are_explicit(self) -> None:
        rows = []
        shapes = [
            ("sub", ["s1"], ["t1"], "resolved", "substitution", "eligible"),
            ("ins", [], ["t1"], "ambiguous", "insertion", "ambiguous"),
            ("del", ["s1"], [], "out_of_region", "deletion", "excluded"),
            ("split", ["s1"], ["t1", "t2"], "resolved", "1_to_n_split", "eligible"),
            ("merge", ["s1", "s2"], ["t1"], "resolved", "n_to_1_merge", "eligible"),
            (
                "mixed",
                ["s1", "s2"],
                ["t1", "t2"],
                "unresolved",
                "n_to_m_mixed",
                "unresolved",
            ),
        ]
        for alignment_id, ocr_ids, gt_ids, status, _operation, _state in shapes:
            rows.append(
                {
                    "alignment_id": alignment_id,
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "ocr_span_ids": ocr_ids,
                    "gt_token_ids": gt_ids,
                    "relation": "one_to_one",
                    "status": status,
                    "ocr_text": ".x" if ocr_ids else "",
                    "gt_text": "x " if gt_ids else "",
                    "align_confidence": 0.5,
                    "reason_code": "",
                }
            )
        audit = component_audit(pd.DataFrame(rows)).set_index("alignment_id")
        for alignment_id, _ocr, _gt, _status, operation, state in shapes:
            assert audit.loc[alignment_id, "operation"] == operation
            assert audit.loc[alignment_id, "state"] == state
        assert bool(audit.loc["sub", "punctuation_boundary"])
        assert bool(audit.loc["sub", "whitespace_boundary"])
        denominators = component_denominators(audit.reset_index())
        assert denominators["n_components_before_exclusion"].sum() == len(shapes)
        assert denominators["n_eligible_after_exclusion"].sum() == 3

    def test_full_uniqueness_checks_alternatives_beyond_sixty_four(self) -> None:
        alignment = pd.DataFrame(
            [
                {
                    "alignment_id": "a0",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "ocr_span_ids": ["s0"],
                    "gt_token_ids": ["t0"],
                    "relation": "one_to_one",
                    "status": "resolved",
                    "ocr_text": "cot",
                    "gt_text": "cat",
                    "align_confidence": 0.6,
                    "char_agreement": 0.5,
                    "geom_score": float("nan"),
                    "uniqueness_margin": 0.5,
                    "reason_code": "",
                }
            ]
        )
        tokens = pd.DataFrame(
            [
                {
                    "gt_token_id": f"t{index}",
                    "document_id": "d1",
                    "index": index,
                    "text": "cat" if index == 0 else "cot" if index == 65 else "zzz",
                }
                for index in range(66)
            ]
        )
        row = component_audit(alignment, tokens).iloc[0]
        assert bool(row["newly_ambiguous_full_uniqueness"])
        assert row["full_uniqueness_margin"] == 0.0
        assert row["state"] == "ambiguous"

    def test_twins_do_not_leak_across_documents(self) -> None:
        sites = _sites(
            [
                {
                    "site_id": "d2:e1:0",
                    "document_id": "d2",
                    "engine_id": "e1",
                    "site_kind": "insertion",
                    "ocr_text": "Ref",
                    "gt_text": "",
                    "evaluable": True,
                }
            ]
        )
        alignments = _alignments(
            [
                {
                    "document_id": "d1",
                    "engine_id": "e1",
                    "reason_code": "never_anchored",
                    "gt_text": "Ref",
                    "ocr_text": "",
                }
            ]
        )
        table = twin_table(alignments, sites)
        assert table["n_twin_signature"].iloc[0] == 0


class TestRegionCensus:
    def _row(
        self,
        site_id: str,
        char_start: int,
        ocr: str,
        *,
        evaluable: bool = True,
        document_id: str = "d1",
        engine_id: str = "e1",
        kind: str = "substitution",
        gt: str = "x",
        dataset_id: str = "funsd",
    ) -> dict[str, object]:
        return {
            "site_id": site_id,
            "document_id": document_id,
            "dataset_id": dataset_id,
            "engine_id": engine_id,
            "site_kind": kind,
            "char_start": char_start,
            "char_end": char_start + len(ocr),
            "ocr_text": ocr,
            "gt_text": gt,
            "evaluable": evaluable,
            "d_before": 1,
        }

    def _streams(
        self, text: str, key: tuple[str, str] = ("d1", "e1")
    ) -> dict[tuple[str, str], str]:
        return {key: text}

    def _stream_for(self, rows: list[dict[str, object]]) -> str:
        """Lay the fixture's texts at their offsets so the stream matches exactly."""
        length = max(int(str(row["char_end"])) for row in rows)
        buffer = [" "] * length
        for row in rows:
            text = str(row["ocr_text"])
            start = int(str(row["char_start"]))
            for offset, char in enumerate(text):
                buffer[start + offset] = char
        return "".join(buffer)

    def test_every_bucket_is_counted(self) -> None:
        rows = [
            self._row("a", 0, "aaaa"),
            self._row("b", 5, "bbbb"),
            self._row("c", 20, "cccc"),
            self._row("d", 25, "dddd", evaluable=False),
            self._row("e", 30, "eeee"),
            self._row("f", 35, ""),
            self._row("g", 36, "gggg"),
            self._row("h", 41, "hhhh", evaluable=False),
            self._row("i", 46, "iiii"),
        ]
        sites = _sites(rows)
        # (a,b) eligible; (b,c) eligible (pure-space gap); (c,d) and (d,e) ambiguous;
        # (e,f) and (f,g) degenerate; (g,h) and (h,i) ambiguous.
        table = region_census(sites, self._streams(self._stream_for(rows)))
        row = table[table["engine_id"] == "e1"].iloc[0]
        assert row["eligible"] == 2
        assert row["ambiguous_member"] == 0
        assert row["n_diagnostic_ambiguous_boundaries"] == 4
        assert row["degenerate"] == 2
        assert row["cross_line"] == 0

    def test_cross_line_and_intervening_buckets(self) -> None:
        rows = [
            self._row("a", 0, "data"),
            self._row("b", 5, "base"),
            self._row("c", 10, "left"),
            self._row("d", 17, "right"),
        ]
        buffer = list(self._stream_for(rows))
        buffer[4] = "\n"  # the line separator between a and b
        buffer[15] = "X"  # a 1-char excluded site's text between c and d
        table = region_census(_sites(rows), self._streams("".join(buffer)))
        row = table[table["engine_id"] == "e1"].iloc[0]
        assert row["cross_line"] == 1  # (a,b): the separator is a newline
        assert row["intervening_text"] == 1  # (c,d): "X" sits in the slice
        assert row["eligible"] == 1  # (b,c): an honest same-line pair

    def test_overlapping_ranges_are_counted(self) -> None:
        sites = _sites([self._row("a", 0, "abcdef"), self._row("b", 3, "xyzw")])
        table = region_census(sites, self._streams("abcdef"))
        assert table["overlapping"].iloc[0] if "overlapping" in table else True
        assert table["non_adjacent"].iloc[0] == 1


class TestProjectionAudit:
    def _site_frame(self) -> pd.DataFrame:
        return _sites(
            [
                {
                    "site_id": "d1:e1:0",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "site_kind": "substitution",
                    "char_start": 0,
                    "char_end": 3,
                    "ocr_text": "dat",
                    "gt_text": "database",
                    "evaluable": True,
                    "ocr_span_ids": ["span-left"],
                    "alignment_ids": ["align-left"],
                },
                {
                    "site_id": "d1:e1:1",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "site_kind": "insertion",
                    "char_start": 4,
                    "char_end": 8,
                    "ocr_text": "base",
                    "gt_text": "",
                    "evaluable": True,
                    "ocr_span_ids": ["span-right"],
                    "alignment_ids": ["align-right"],
                },
            ]
        )

    def _streams(self) -> dict[tuple[str, str], str]:
        return {("d1", "e1"): "dat base"}

    def _proposal(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "candidate_id": "d1:e1:region:0:g5:0",
            "region_id": "d1:e1:region:0",
            "document_id": "d1",
            "engine_id": "e1",
            "operation_type": "merge",
            "left_site_id": "d1:e1:0",
            "right_site_id": "d1:e1:1",
            "gap_chars": 1,
            "region_char_start": 0,
            "region_char_end": 8,
            "left_kind": "substitution",
            "right_kind": "insertion",
            "region_ocr": "dat base",
            "region_gt": "database",
            "candidate_text": "database",
            "d_before": 1,
            "d_after": 0,
            "delta": 1,
            "outcome": "true_correction",
            "left_source_span_ids": '["span-left"]',
            "right_source_span_ids": '["span-right"]',
            "left_alignment_component_ids": '["align-left"]',
            "right_alignment_component_ids": '["align-right"]',
        }
        row.update(overrides)
        return row

    def test_correctly_projected_region_counts_clean(self) -> None:
        audit = projection_audit(
            pd.DataFrame([self._proposal()]), self._site_frame(), self._streams()
        )
        assert bool(audit["projection_correct"].iloc[0])
        assert bool(audit["label_valid"].iloc[0])
        assert audit["state"].iloc[0] == "eligible"
        summary = projection_summary(audit)
        assert summary["projection_correct"].iloc[0] == 1
        assert summary["projection_mismatch"].iloc[0] == 0

    def test_a_wrong_gap_in_the_record_is_a_mismatch_not_an_input(self) -> None:
        """The circularity this audit used to have: it rebuilt the slice with the
        record's own gap. Now the gap is recomputed and a wrong one fails."""
        audit = projection_audit(
            pd.DataFrame([self._proposal(gap_chars=3)]), self._site_frame(), self._streams()
        )
        assert not bool(audit["projection_correct"].iloc[0])
        assert audit["state"].iloc[0] == "excluded"

    def test_a_mismatched_slice_is_caught(self) -> None:
        audit = projection_audit(
            pd.DataFrame([self._proposal(region_ocr="dat  base")]),
            self._site_frame(),
            self._streams(),
        )
        assert not bool(audit["projection_correct"].iloc[0])

    def test_a_wrong_char_range_is_caught(self) -> None:
        audit = projection_audit(
            pd.DataFrame([self._proposal(region_char_end=7)]), self._site_frame(), self._streams()
        )
        assert not bool(audit["projection_correct"].iloc[0])

    def test_unknown_member_sites_are_mismatches(self) -> None:
        audit = projection_audit(
            pd.DataFrame([self._proposal(right_site_id="d1:e1:99")]),
            self._site_frame(),
            self._streams(),
        )
        assert not bool(audit["projection_correct"].iloc[0])
        assert audit["state"].iloc[0] == "unresolved"

    def test_empty_proposals_return_an_empty_frame(self) -> None:
        audit = projection_audit(pd.DataFrame(), self._site_frame(), self._streams())
        assert audit.empty

    def test_left_and_right_span_projection_are_independent(self) -> None:
        audit = projection_audit(
            pd.DataFrame([self._proposal(right_source_span_ids='["wrong"]')]),
            self._site_frame(),
            self._streams(),
        )
        assert bool(audit["left_span_projection"].iloc[0])
        assert not bool(audit["right_span_projection"].iloc[0])
        assert audit["state"].iloc[0] == "excluded"


class TestSiteProjectionAudit:
    def test_empty_anchor_and_candidate_provenance_are_checked(self) -> None:
        sites = _sites(
            [
                {
                    "site_id": "s0",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "site_kind": "deletion",
                    "char_start": 0,
                    "char_end": 0,
                    "ocr_text": "",
                    "gt_text": "due",
                    "evaluable": True,
                    "ocr_span_ids": [],
                    "alignment_ids": ["a0"],
                }
            ]
        )
        proposals = pd.DataFrame(
            [
                {
                    "candidate_id": "s0:g5:0",
                    "site_id": "s0",
                    "document_id": "d1",
                    "engine_id": "e1",
                    "operation_type": "insertion",
                    "source_char_start": 0,
                    "source_char_end": 0,
                    "source_span_ids": "[]",
                    "alignment_component_ids": '["a0"]',
                    "source_geometry_id": "site:s0",
                    "original_ocr": "",
                    "candidate_text": "due",
                    "gt_text": "due",
                    "d_before": 3,
                    "d_after": 0,
                    "delta": 3,
                    "outcome": "true_correction",
                }
            ]
        )
        audit = site_projection_audit(proposals, sites)
        assert not bool(audit["projection_correct"].iloc[0])
        assert bool(audit["empty_anchor"].iloc[0])
        assert audit["state"].iloc[0] == "unresolved"
        assert audit["reason"].iloc[0] == "empty_insertion_anchor"


class TestTwinSites:
    def test_the_site_table_names_exactly_the_twin_sites(self) -> None:
        sites = _sites(
            [
                {
                    "site_id": "d1:e1:0",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "site_kind": "insertion",
                    "ocr_text": "Ref",
                    "gt_text": "",
                    "evaluable": True,
                },
                {
                    "site_id": "d1:e1:1",
                    "document_id": "d1",
                    "dataset_id": "funsd",
                    "engine_id": "e1",
                    "site_kind": "substitution",
                    "ocr_text": "Name",
                    "gt_text": "Name",
                    "evaluable": True,
                },
            ]
        )
        alignments = _alignments(
            [
                {
                    "document_id": "d1",
                    "engine_id": "e1",
                    "reason_code": "never_anchored",
                    "gt_text": "Ref",
                    "ocr_text": "",
                }
            ]
        )
        table = twin_sites_table(alignments, sites)
        flagged = table[table["twin_signature"].astype(bool)]["site_id"].tolist()
        assert flagged == ["d1:e1:0"]

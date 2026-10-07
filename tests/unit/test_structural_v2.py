"""Hand-computed tests for the g7 structural generator at OCR-only anchors.

Every case names the fold corpus and the anchor and asserts the exact proposals --
text, shape, and score -- that must (and must not) appear. The competence set is the
one the substitution rungs cannot express: insertion at gaps, split at tokens,
merge and pair substitution at token pairs.
"""

from __future__ import annotations

from collections import Counter

from ocr_risk.candidates.structural_v2 import StructuralV2Generator
from ocr_risk.discovery.enumerator import DiscoveredSite
from ocr_risk.discovery.views import OcrPageView, OcrTokenView
from ocr_risk.schemas.enums import AnchorKind, DiscoveryProvenance

__all__ = ["TestContract", "TestInsertion", "TestMergeAndPairSubstitution", "TestSplit"]


def _generator(corpus: list[str]) -> StructuralV2Generator:
    generator = StructuralV2Generator()
    generator.fit(corpus)
    return generator


def _token(span_id: str, text: str, start: int) -> OcrTokenView:
    return OcrTokenView(
        span_id=span_id,
        text=text,
        line_id="L1",
        char_start=start,
        char_end=start + len(text),
        normalized_confidence=0.99,
        x0=float(start * 10),
        x1=float(start * 10 + 10 * len(text)),
    )


def _page(*tokens: OcrTokenView) -> OcrPageView:
    return OcrPageView(document_id="d1", dataset_id="funsd", engine_id="tesseract", tokens=tokens)


def _site(kind: AnchorKind, *spans: str, start: int = 0, end: int = 1) -> DiscoveredSite:
    return DiscoveredSite(
        site_id=f"d1:tesseract:dsite:{kind.value}:{'-'.join(spans)}",
        document_id="d1",
        dataset_id="funsd",
        engine_id="tesseract",
        anchor_kind=kind,
        anchor_ref="\0".join((kind.value, *spans)),
        char_start=start,
        char_end=end,
        site_type="insertion" if kind is AnchorKind.GAP else "substitution",
        suspicion_score=0.5,
        provenance_reason=DiscoveryProvenance.GAP_ANOMALY
        if kind is AnchorKind.GAP
        else DiscoveryProvenance.LEXICAL_ANOMALY,
        signals={},
    )


class TestInsertion:
    def test_a_gap_anchor_yields_the_fold_attested_token(self) -> None:
        # bigrams over three lines: (total, of):3 and (of, 12):3 -- the missing "of"
        # between "total" and "12" carries evidence 3 + 3 + 3 = 9 (count plus both
        # one-sided attestations).
        generator = _generator(["total of 12", "total of 12", "total of 12"])
        page = _page(_token("s0", "total", 0), _token("s1", "12", 6))
        proposals = generator.propose_at_site(
            _site(AnchorKind.GAP, "s0", "s1", start=5, end=6), page
        )
        assert [p.text for p in proposals] == ["of"]
        assert proposals[0].metadata["edit_shape"] == "insertion"
        assert proposals[0].score == 9.0

    def test_an_unattested_gap_yields_nothing(self) -> None:
        # No fold bigram touches "date" or "smith" on either side.
        generator = _generator(["total of", "total of"])
        page = _page(_token("s0", "date", 0), _token("s1", "smith", 5))
        assert generator.propose_at_site(_site(AnchorKind.GAP, "s0", "s1"), page) == []


class TestSplit:
    def test_a_merged_token_splits_into_its_lexicon_halves(self) -> None:
        generator = _generator(["smith john", "smith john"])
        page = _page(_token("s0", "smithjohn", 0))
        proposals = generator.propose_at_site(_site(AnchorKind.TOKEN, "s0", start=0, end=9), page)
        assert [p.text for p in proposals] == ["smith john"]
        assert proposals[0].metadata["edit_shape"] == "split"

    def test_the_most_balanced_cut_wins_and_ties_break_left(self) -> None:
        # vocab {abc, defghi, abcdef, ghi}: "abcdefghi" = abc|defghi and abcdef|ghi,
        # both balance 3; the left-string tiebreak picks "abc" < "abcdef". (Halves must
        # clear the shared lexicon's 3-character minimum -- 2-char parts never enter.)
        generator = _generator(["abc defghi", "abc defghi", "abcdef ghi", "abcdef ghi"])
        page = _page(_token("s0", "abcdefghi", 0))
        proposals = generator.propose_at_site(_site(AnchorKind.TOKEN, "s0"), page)
        assert [p.text for p in proposals] == ["abc defghi"]

    def test_a_known_word_or_short_token_never_splits(self) -> None:
        generator = _generator(["data base", "data base", "ab ab"])
        page = _page(_token("s0", "data", 0), _token("s1", "xy", 5))
        assert generator.propose_at_site(_site(AnchorKind.TOKEN, "s0"), page) == []
        assert generator.propose_at_site(_site(AnchorKind.TOKEN, "s1"), page) == []


class TestMergeAndPairSubstitution:
    def test_a_pair_joining_to_a_lexicon_word_merges(self) -> None:
        generator = _generator(["invoice total", "invoice 12"])
        page = _page(_token("s0", "invo", 0), _token("s1", "ice", 4))
        proposals = generator.propose_at_site(
            _site(AnchorKind.TOKEN_PAIR, "s0", "s1", start=0, end=8), page
        )
        assert proposals[0].text == "invoice"
        assert proposals[0].metadata["edit_shape"] == "merge"
        assert proposals[0].score == 1.0

    def test_a_near_miss_join_proposes_the_pair_substitution(self) -> None:
        # "invoicee" is not in the lexicon; its distance-1 neighbour "invoice" is.
        generator = _generator(["invoice total", "invoice 12"])
        page = _page(_token("s0", "invoice", 0), _token("s1", "e", 7))
        proposals = generator.propose_at_site(
            _site(AnchorKind.TOKEN_PAIR, "s0", "s1", start=0, end=8), page
        )
        shapes = [(p.text, p.metadata["edit_shape"], p.score) for p in proposals]
        assert ("invoice", "pair_substitution", -1.0) in shapes

    def test_proposals_are_capped_per_site(self) -> None:
        generator = StructuralV2Generator(max_per_site=1)
        generator.fit(["invoice total", "invoice 12"])
        page = _page(_token("s0", "invo", 0), _token("s1", "ice", 4))
        proposals = generator.propose_at_site(
            _site(AnchorKind.TOKEN_PAIR, "s0", "s1", start=0, end=8), page
        )
        assert len(proposals) == 1
        assert proposals[0].metadata["edit_shape"] == "merge"


class TestContract:
    def test_the_rung_never_proposes_without_an_anchor(self) -> None:
        # The discovery/generation boundary: g7 has no site-less fallback.
        from ocr_risk.candidates.base import GenerationContext

        assert (
            _generator(["total of 12"]).propose(
                GenerationContext(
                    site_id="x",
                    document_id="d1",
                    dataset_id="funsd",
                    engine_id="tesseract",
                    original_ocr="total",
                    context_before="",
                    context_after="",
                ),
                4,
            )
            == []
        )

    def test_fit_builds_the_vocabulary_and_bigrams_from_the_fold_corpus(self) -> None:
        generator = _generator(["total of 12", "total of 12"])
        # "of" and "12" are shorter than the shared lexicon's 3-character minimum:
        # known words are at least 3 characters, exactly as every other rung defines
        # them. Bigrams keep the raw tokens -- attestation is not length-filtered.
        assert set(generator.lexicon) == {"total"}
        assert generator._bigrams == Counter({("total", "of"): 2, ("of", "12"): 2})
        assert generator.available()

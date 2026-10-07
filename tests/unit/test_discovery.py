"""The CGV3 OCR-only site enumerator on hand-constructed pages.

Every case is hand-computed: the fixture names the OCR signals, the test asserts the
exact site (anchor kind, position, type, provenance) that must — and must not —
appear. The six structural classes are each exercised, and the GT-blindness invariant
is attacked directly: a page view cannot carry ground truth, and the package cannot
even import the layers that hold it.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ocr_risk.discovery import (
    DiscoveryResources,
    DiscoveryRules,
    OcrPageView,
    OcrTokenView,
    enumerate_sites,
)
from ocr_risk.schemas.enums import AnchorKind, DiscoveryProvenance

LEXICON = frozenset(
    {"invoice", "total", "amount", "date", "database", "smith", "john", "data", "base"}
)
# "total of 12" and "amount of" attest the token "of" between total->12 / amount->...
BIGRAMS = Counter({("total", "of"): 4, ("of", "12"): 3, ("amount", "of"): 2, ("of", "due"): 2})


def _token(
    span_id: str,
    text: str,
    *,
    char_start: int,
    conf: float | None = 0.99,
    line: str = "L1",
    x0: float | None = 0.0,
    width: float = 40.0,
) -> OcrTokenView:
    return OcrTokenView(
        span_id=span_id,
        text=text,
        line_id=line,
        char_start=char_start,
        char_end=char_start + len(text),
        normalized_confidence=conf,
        x0=x0,
        x1=None if x0 is None else x0 + width,
    )


def _page(*tokens: OcrTokenView, engine: str = "tesseract") -> OcrPageView:
    return OcrPageView(
        document_id="doc-1",
        dataset_id="funsd",
        engine_id=engine,
        tokens=tokens,
    )


def _resources() -> DiscoveryResources:
    return DiscoveryResources(lexicon=LEXICON, bigrams=BIGRAMS)


class TestTokenSites:
    def test_a_low_confidence_token_becomes_a_substitution_site(self) -> None:
        # "amount" is a lexicon word, so only the confidence signal fires -- the rule
        # is tested in isolation; the combined-signal case lives in the dedup tests.
        page = _page(
            _token("s0", "invoice", char_start=0),
            _token("s1", "amount", char_start=8, conf=0.31),
            _token("s2", "total", char_start=15),
        )
        sites = enumerate_sites(
            page, DiscoveryResources(lexicon=LEXICON), DiscoveryRules(conf_floor=0.55)
        )
        assert len(sites) == 1
        site = sites[0]
        assert site.anchor_kind is AnchorKind.TOKEN
        assert site.anchor_ref.endswith("s1")
        assert (site.char_start, site.char_end) == (8, 14)
        assert site.site_type == "substitution"
        assert site.provenance_reason is DiscoveryProvenance.LOW_CONFIDENCE_TOKEN
        assert site.signals == {"normalized_confidence": "0.3100"}
        assert site.suspicion_score == pytest.approx(0.69)

    def test_an_oov_token_between_lexicon_neighbours_is_a_lexical_site(self) -> None:
        page = _page(
            _token("s0", "invoice", char_start=0),
            _token("s1", "tttal", char_start=8),  # OOV, normal confidence
            _token("s2", "amount", char_start=14),
        )
        sites = enumerate_sites(page, _resources())
        assert [s.anchor_ref[-2:] for s in sites] == ["s1"]
        assert sites[0].provenance_reason is DiscoveryProvenance.LEXICAL_ANOMALY
        assert sites[0].signals["oov_token"] == "tttal"

    def test_an_oov_token_in_an_oov_neighbourhood_is_not_a_site(self) -> None:
        # Every neighbour is out of vocabulary too: nothing marks this token out.
        page = _page(
            _token("s0", "qqqxx", char_start=0),
            _token("s1", "zzzww", char_start=6),
            _token("s2", "aaavv", char_start=12),
        )
        assert enumerate_sites(page, _resources()) == []

    def test_confidence_floor_differs_by_engine(self) -> None:
        rules = DiscoveryRules(conf_floor=0.4, conf_floor_by_engine={"doctr": 0.9})
        tokens = (_token("s0", "invoice", char_start=0, conf=0.5),)
        assert (
            enumerate_sites(
                _page(_token("s0", "invoice", char_start=0, conf=0.5), engine="tesseract"),
                _resources(),
                rules,
            )
            == []
        )
        doctr = enumerate_sites(_page(*tokens, engine="doctr"), _resources(), rules)
        assert len(doctr) == 1


class TestSplitSites:
    def test_a_token_concatenating_two_lexicon_words_is_a_split_site(self) -> None:
        page = _page(
            _token("s0", "date", char_start=0),
            _token("s1", "smithjohn", char_start=5),
            _token("s2", "total", char_start=14),
        )
        sites = enumerate_sites(page, _resources())
        assert len(sites) == 1
        assert sites[0].site_type == "split"
        # The specific structural evidence outranks the generic OOV signal the same
        # token also fires; both remain recorded in `signals`.
        assert sites[0].provenance_reason is DiscoveryProvenance.POSSIBLE_SPLIT
        assert sites[0].signals["concatenated_words"] == "smithjohn"
        assert sites[0].signals["oov_token"] == "smithjohn"

    def test_a_single_lexicon_word_is_not_a_split(self) -> None:
        page = _page(_token("s0", "database", char_start=0), _token("s1", "date", char_start=9))
        assert enumerate_sites(page, _resources()) == []


class TestMergeSites:
    def test_an_adjacent_pair_joining_to_a_lexicon_word_is_a_merge_site(self) -> None:
        page = _page(
            _token("s0", "total", char_start=0),
            _token("s1", "data", char_start=6),
            _token("s2", "base", char_start=11),
            _token("s3", "date", char_start=16),
        )
        sites = enumerate_sites(page, _resources())
        merges = [s for s in sites if s.site_type == "merge"]
        assert len(merges) == 1
        site = merges[0]
        assert site.anchor_kind is AnchorKind.TOKEN_PAIR
        assert site.anchor_ref.endswith("s1\0s2")
        assert (site.char_start, site.char_end) == (6, 15)
        assert site.signals["joined_lexicon_word"] == "database"

    def test_a_line_boundary_pair_never_merges(self) -> None:
        # Same join, different lines: the separator-free join would be a false repair
        # across a line break.
        page = _page(
            _token("s0", "data", char_start=0, line="L1"),
            _token("s1", "base", char_start=5, line="L2"),
        )
        assert enumerate_sites(page, _resources()) == []


class TestGapSites:
    def test_an_abnormally_wide_gap_becomes_an_insertion_site(self) -> None:
        # Within-line gaps are ~4-5px with MAD 1, so the threshold is 5 + 6x1 = 11px;
        # only the 60px gap fires. No bigrams, so the geometry rule stands alone.
        geometry_only = DiscoveryResources(lexicon=LEXICON, bigrams=Counter())
        page = _page(
            _token("s0", "total", char_start=0, x0=0.0, width=40.0),
            _token("s1", "date", char_start=6, x0=44.0, width=40.0),
            _token("s2", "john", char_start=11, x0=89.0, width=40.0),
            _token("s3", "smith", char_start=16, x0=133.0, width=40.0),
            _token("s4", "invoice", char_start=22, x0=233.0, width=40.0),
            _token("s5", "amount", char_start=30, x0=278.0, width=40.0),
        )
        sites = enumerate_sites(page, geometry_only, DiscoveryRules(gap_mad_k=6.0))
        gaps = [s for s in sites if s.anchor_kind is AnchorKind.GAP]
        assert len(gaps) == 1
        site = gaps[0]
        assert site.anchor_ref.endswith("s3\0s4")
        assert site.site_type == "insertion"
        assert site.provenance_reason is DiscoveryProvenance.GAP_ANOMALY
        # The anchor is the observable boundary between the two spans -- never [0, 0].
        assert (site.char_start, site.char_end) == (21, 22)
        assert site.char_start > 0

    def test_a_fold_attested_missing_token_is_an_insertion_site_without_geometry(self) -> None:
        # No boxes at all: the sequence rule alone must anchor the missing "of".
        page = _page(
            _token("s0", "total", char_start=0, conf=None, x0=None),
            _token("s1", "12", char_start=6, conf=None, x0=None),
        )
        sites = enumerate_sites(page, _resources())
        assert len(sites) == 1
        site = sites[0]
        assert site.anchor_kind is AnchorKind.GAP
        assert site.provenance_reason is DiscoveryProvenance.SEQUENCE_ANOMALY
        assert site.signals["fold_attested_between"] == "of"
        assert (site.char_start, site.char_end) == (5, 6)

    def test_unattested_adjacent_pairs_produce_no_gap_site(self) -> None:
        page = _page(
            _token("s0", "date", char_start=0, conf=None, x0=None),
            _token("s1", "john", char_start=5, conf=None, x0=None),
        )
        assert enumerate_sites(page, _resources()) == []

    def test_mad_zero_pages_collapse_the_gap_threshold_to_the_pixel_floor(self) -> None:
        # Documented degeneracy (pre-confirmatory review, lead A): when more than half
        # of a page's within-line gaps are zero -- overlapping boxes, doctr/easyocr --
        # median and MAD are both 0 and the threshold degenerates to gap_min_pixels,
        # so the ordinary 6px word spacing a healthy page ignores (control page below)
        # fires as an insertion site. Pinned deliberately: this is frozen Track-A
        # behaviour, carried into the confirmatory run with its site-precision cost
        # reported, not silently patched after the freeze.
        geometry_only = DiscoveryResources(lexicon=LEXICON, bigrams=Counter())
        words = ["total", "date", "john", "smith", "amount", "invoice"]
        chars = [0, 6, 11, 16, 22, 29]
        rules = DiscoveryRules(gap_mad_k=6.0)

        def page_at(x0s: list[float]) -> OcrPageView:
            return _page(
                *(
                    _token(f"s{i}", word, char_start=chars[i], x0=x0, width=40.0)
                    for i, (word, x0) in enumerate(zip(words, x0s))
                )
            )

        degenerate = page_at([0.0, 40.0, 80.0, 120.0, 160.0, 206.0])  # gaps 0,0,0,0,6
        healthy = page_at([0.0, 44.0, 88.0, 133.0, 178.0, 224.0])  # gaps 4,4,5,5,6
        fired = [
            s
            for s in enumerate_sites(degenerate, geometry_only, rules)
            if s.anchor_kind is AnchorKind.GAP
        ]
        assert [s.anchor_ref.split("\0")[1:] for s in fired] == [["s4", "s5"]]
        assert fired[0].signals["box_gap_pixels"] == "6.0"
        assert not [
            s
            for s in enumerate_sites(healthy, geometry_only, rules)
            if s.anchor_kind is AnchorKind.GAP
        ]


class TestDeterminismAndDedup:
    def test_multiple_signals_at_one_anchor_collapse_into_one_site(self) -> None:
        # "smithjonn" is OOV (lexical) AND splits into smith+jonn? -- no, "jonn" is not
        # in the lexicon; "smithjohn" is both OOV and a concatenation of two entries.
        page = _page(
            _token("s0", "date", char_start=0),
            _token("s1", "smithjohn", char_start=5, conf=0.2),
            _token("s2", "total", char_start=14),
        )
        sites = enumerate_sites(page, _resources(), DiscoveryRules(conf_floor=0.55))
        assert len(sites) == 1
        site = sites[0]
        # Most specific evidence wins both the type and the reason; every weaker
        # signal survives in `signals` and the suspicion is the maximum fired.
        assert site.site_type == "split"
        assert site.provenance_reason is DiscoveryProvenance.POSSIBLE_SPLIT
        assert set(site.signals) == {"normalized_confidence", "oov_token", "concatenated_words"}
        assert site.suspicion_score == pytest.approx(0.8)

    def test_site_ids_are_deterministic_and_stream_ordered(self) -> None:
        page = _page(
            _token("s0", "recieve", char_start=0, conf=0.1),
            _token("s1", "smithjohn", char_start=8),
            _token("s2", "data", char_start=18),
            _token("s3", "base", char_start=23),
            _token("s4", "total", char_start=28),
        )
        first = enumerate_sites(page, _resources())
        second = enumerate_sites(page, _resources())
        assert [s.site_id for s in first] == [s.site_id for s in second]
        assert [s.site_id for s in first] == [
            "doc-1:tesseract:dsite:00000",
            "doc-1:tesseract:dsite:00001",
            "doc-1:tesseract:dsite:00002",
        ]
        assert [s.char_start for s in first] == sorted(s.char_start for s in first)


class TestGroundTruthBlindness:
    def test_the_page_view_has_no_ground_truth_fields(self) -> None:
        """Red team: there is nowhere for GT to enter, so it cannot be added silently."""
        fields = OcrPageView.__dataclass_fields__ | OcrTokenView.__dataclass_fields__
        forbidden = {"gt_text", "gt_tokens", "ground_truth", "alignment_ids", "d_before"}
        assert forbidden.isdisjoint(fields), sorted(forbidden & set(fields))

    def test_the_enumerator_package_cannot_import_ground_truth_layers(self) -> None:
        """Enforced by the architecture test; asserted here too so the unit suite
        fails on it directly rather than only via the architecture run."""
        import ocr_risk.discovery.enumerator as module
        import ocr_risk.discovery.views as views_module

        for source in (module, views_module):
            assert not any(
                source_module.startswith(("ocr_risk.align", "ocr_risk.edits", "ocr_risk.datasets"))
                for source_module in _imported_modules(source)
            ), f"{source.__name__} imports a ground-truth layer"

    def test_ground_truth_cannot_alter_discovery_through_the_view(self) -> None:
        """The page view is the only input; pages identical in OCR differ in nothing."""
        page = _page(
            _token("s0", "recieve", char_start=0, conf=0.2),
            _token("s1", "total", char_start=8),
        )
        baseline = enumerate_sites(page, _resources())
        # A hypothetical GT correction exists only in the evaluator's world; there is
        # no code path to feed it back, and re-running the identical view is identical.
        assert enumerate_sites(page, _resources()) == baseline


def _imported_modules(module: object) -> set[str]:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))  # type: ignore[arg-type]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ocr_risk"):
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ocr_risk"):
                    found.add(alias.name)
    return found


def test_the_view_projects_canonical_spans_and_drops_everything_else() -> None:
    from ocr_risk.schemas.base import BBox
    from ocr_risk.schemas.spans import CanonicalSpan

    spans = (
        CanonicalSpan(
            span_id="sp1",
            document_id="doc-1",
            dataset_id="funsd",
            engine_id="tesseract",
            engine_fingerprint="tess-1",
            text="total",
            reading_order=0,
            line_id="L1",
            bbox=BBox(x0=0.0, y0=0.0, x1=40.0, y1=12.0),
            native_conf_recognition=95.0,
            conf_scale="tesseract_word_conf_0_100",
            char_start=0,
            char_end=5,
            raw_ref="raw/ocr.json",
            raw_index=0,
        ),
        CanonicalSpan(
            span_id="sp2",
            document_id="doc-1",
            dataset_id="funsd",
            engine_id="tesseract",
            engine_fingerprint="tess-1",
            text="12",
            reading_order=1,
            line_id="L1",
            bbox=BBox(x0=44.0, y0=0.0, x1=56.0, y1=12.0),
            native_conf_recognition=95.0,
            conf_scale="tesseract_word_conf_0_100",
            char_start=6,
            char_end=8,
            raw_ref="raw/ocr.json",
            raw_index=1,
        ),
    )
    view = OcrPageView.from_spans(spans)
    assert view.tokens[0].normalized_confidence == pytest.approx(0.95)
    assert view.tokens[1].normalized_confidence == pytest.approx(0.95)
    # The fold bigram attests "of" between them: an insertion site appears from the
    # projection alone, with the native confidence rescaled on its declared scale.
    sites = enumerate_sites(view, _resources())
    assert len(sites) == 1
    assert sites[0].anchor_kind is AnchorKind.GAP
    assert sites[0].signals["fold_attested_between"] == "of"


def test_the_view_refuses_mixed_pages_and_empty_pages() -> None:
    from ocr_risk.schemas.base import BBox
    from ocr_risk.schemas.spans import CanonicalSpan

    def _span(span_id: str, document_id: str, engine_id: str) -> CanonicalSpan:
        return CanonicalSpan(
            span_id=span_id,
            document_id=document_id,
            dataset_id="funsd",
            engine_id=engine_id,
            engine_fingerprint="f",
            text="x",
            reading_order=0,
            bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
            char_start=0,
            char_end=1,
            raw_ref="raw/ocr.json",
            raw_index=0,
        )

    with pytest.raises(ValueError, match=r"one \(document, engine\) pair"):
        OcrPageView.from_spans((_span("a", "d1", "e1"), _span("b", "d2", "e1")))
    with pytest.raises(ValueError, match="empty page"):
        OcrPageView.from_spans(())


class TestConfirmatoryReserve:
    def test_the_reserve_contains_exactly_the_untouched_funsd_documents(self) -> None:
        from ocr_risk.discovery.freshness import confirmatory_document_ids

        reserve = confirmatory_document_ids()
        assert len(reserve) == 99
        assert sum(1 for item in reserve if item.startswith("funsd-testing_data-")) == 50
        assert sum(1 for item in reserve if item.startswith("funsd-training_data-")) == 49

    def test_a_reserve_document_cannot_enter_a_development_path(self) -> None:
        """Red team: the guard fires loudly, naming the offender and the caller."""
        from ocr_risk.discovery.freshness import (
            DevelopmentDocumentError,
            assert_development_documents,
            confirmatory_document_ids,
        )

        offender = sorted(confirmatory_document_ids())[0]
        with pytest.raises(DevelopmentDocumentError, match="cgv3 track-a pilot"):
            assert_development_documents(
                ["funsd-training_data-0000971160", offender], context="cgv3 track-a pilot"
            )
        # Seen corpus documents pass untouched (the guard returns None silently).
        assert_development_documents(
            ["funsd-training_data-0000971160", "cord-test-0000"], context="cgv3 track-a pilot"
        )

    def test_a_single_id_string_is_treated_as_one_document(self) -> None:
        from ocr_risk.discovery.freshness import (
            DevelopmentDocumentError,
            assert_development_documents,
            confirmatory_document_ids,
        )

        offender = sorted(confirmatory_document_ids())[0]
        with pytest.raises(DevelopmentDocumentError):
            assert_development_documents(offender, context="cgv3 unit test")

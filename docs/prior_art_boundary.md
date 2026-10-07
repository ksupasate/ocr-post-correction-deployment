# Prior-art boundary

This page exists to keep the project honest about what it is not claiming. It is the
reference the integrity audit (`ocr-risk audit docs`) and the
[research-integrity rule](../.claude/rules/research-integrity.md) enforce against.

## Claimed as established or substantial prior art

The project claims **no novelty** for any of the following. Each has substantial existing
literature, and treating any of them as a contribution would be indefensible:

| | |
|---|---|
| detecting OCR errors before correction | detector → corrector pipelines |
| OCR confidence as a signal | confidence-aware error detection |
| confidence-aware correction | ByT5 and byte-level models |
| character-level or byte-level OCR correction | transformer-based OCR correction |
| LLM OCR post-correction | multimodal OCR correction |
| using a source image | using bounding boxes |
| minimal or local editing as a concept | generic selective prediction |
| generic abstention | using multiple OCR engines |
| recognizing that overcorrection exists | CER/WER alternatives in general |
| correction provenance | risk-controlled prediction in general |

## The narrower space this project operates in

Five things, stated as **candidate contributions** and **research hypotheses**, not as
established results:

1. **Candidate-specific source-grounded edit verification** — verifying a *particular*
   proposed edit against the pixels that produced the span, rather than scoring a
   transcription or detecting an error.
2. **Harmful accepted-edit probability as the estimated quantity** — the object of
   estimation is `P(accepting this edit improves the transcription)`, not `P(this span is
   wrong)`.
3. **Risk-controlled post-correction under unseen OCR-engine shift** — with the
   distribution-free guarantee and, importantly, a measurement of how far that guarantee
   degrades when exchangeability fails.
4. **Matched same-source multi-engine evaluation** — every engine reads identical
   documents, so engine shift is not confounded with document shift.
5. **Correction-specific risk-coverage evaluation under engine shift** — Coverage@Risk on
   *edits*, with a harm taxonomy, rather than aggregate CER.

No directly equivalent formulation of (1)–(5) as a combined framework was identified in
the prior art reviewed for this project. That is a statement about a review, not a proof
of absence, and it should be re-examined before any submission.

## Language rules

**Never write** "the first", "first to", "novel", "state of the art", "SOTA",
"unprecedented", "we are the only", or "breakthrough". `ocr-risk audit docs` fails the
build on these.

**Write instead:**

- "candidate contribution"
- "research hypothesis"
- "no directly equivalent formulation identified in the reviewed prior art"

The rule is not stylistic. A novelty claim is a claim about the entire literature, and
this project has not surveyed the entire literature. Cautious phrasing is the accurate
phrasing.

## Four things documentation must keep separate

A recurring failure in research writing is letting these blur into one another. Every
page in `docs/` must keep them visibly distinct:

1. **Established prior art** — what is already known.
2. **Project hypothesis** — what we believe and are testing.
3. **Proposed method** — what we built.
4. **Actual experimental result** — what we measured, and from which artifact.

Anything in category 4 must trace to a saved artifact via `figure_manifest.json` or a run
record. A number that cannot be traced is either a placeholder to delete or a fabrication;
there is no third case.

## Citations

Every citation needs a DOI or URL that was actually retrieved. If it cannot be verified,
mark it `[UNVERIFIED]` rather than presenting it as checked. Never invent an author list,
venue, year, or DOI.

At the time of writing, **no citations have been verified in this repository**, which is
why none appear in the documentation. `manifests/datasets/funsd.json` records
`"citation": {"verified": false}` for the same reason.

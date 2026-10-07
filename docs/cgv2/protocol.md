# Candidate Generation Study v2 (CGV2) — protocol

**Status: FROZEN before any evaluate-role run.** This document is committed before the
CGV2 confirmatory study executes. The evaluate-role run record must reference the commit
that adds this file. Anything changed after that commit appears in §18 as a numbered,
dated amendment with its reason — never silently.

**Predecessors, untouched.** The H1 pre-registration and its verdict (`H1 NOT SUPPORTED`),
the RH1 verdict (`PARTIALLY SUPPORTED`), the recovery-phase generator gate
(`GENERATOR PARTIALLY READY`), and the H2-readiness gate (`H2 NOT READY`,
`results/generated/h2_readiness_gate.json`) are historical artifacts. CGV2 does not
modify them, does not re-open them, and writes its outputs to `results/generated/cgv2/`.
The frozen generator `g3_edit_aware` stays frozen at its ladder configuration; new rungs
are added beside it.

## 1. Question

> Can a leakage-free, structurally expressive candidate generator create enough correct
> repair opportunities across OCR engines to make candidate-specific source-grounded
> verification scientifically testable?

The central dependency is `OCR error → correct candidate exists → verifier can decide`.
If the correct candidate does not exist in the candidate set, no verifier recovers it.
CGV2 measures the middle link. It does **not** test H2 and does not build a verifier.

## 2. Naming and ladder mapping

Canonical study ids are retained; the prompt's G0–G4 map onto them as:

| CGV2 rung | canonical id | status |
|---|---|---|
| G0 frozen baseline | `g0_lexical` | existing, unchanged (K=4) |
| G1 leakage-free lexical / gated | `g0_lexical`, `g1_error_gated` as run by the study's leave-engine-out lexicon rule, plus the new fold certificates (§7) | existing machinery; CGV2 adds certificates, not generators |
| G2 neural | `g2_byt5` (unchanged, K=1, FUNSD only) and `g2_byt5_ctx0` — the same checkpoint at its own documented default `context_chars = 0`, a diagnostic rung closing the open item in `implementation_log.md` | checkpoint pinned at revision `19d5c2fd…` |
| G3 structural | **`g5_structural`** (new) | designed in §5 |
| G4 controlled union | **`g6_union`** (new) = `g3_edit_aware` ∪ `g5_structural` at site level, plus `g5_structural`'s region candidates | §5.4 |

`g3_edit_aware` (frozen) is the incumbent in every comparison.

## 3. Terminology — the frozen mapping between edit operations and site kinds

The repository names site kinds from the *alignment's* perspective; repairs are named from
the *edit's* perspective. This table is normative for every CGV2 table and figure:

| repair operation (this doc) | site kind (`edits/sites.py`) | meaning | expressible before CGV2? |
|---|---|---|---|
| substitution | `substitution`, `clean` | 1:1 token replacement | yes |
| deletion | `insertion` | OCR text with no GT counterpart; fix = propose `""` | yes (`g3` since recovery) |
| split | `segmentation` (1 span : n GT) | OCR merged what GT keeps apart; fix = re-insert separator | yes (`g3`, gated on both halves in lexicon) |
| merge | `segmentation` (n spans : 1 GT) | OCR split what GT keeps joined | nominally (whole-string replacement), practically ~never proposed |
| insertion | `deletion` | GT token with no OCR span; fix = propose the missing text | **no — zero proposals by construction** |
| region / n:m | adjacent site pairs | repair requires jointly rewriting ≥2 sites | **no — structurally inexpressible** |
| unresolved | non-evaluable sites | alignment AMBIGUOUS/UNRESOLVED/OUT_OF_REGION | excluded; denominators reported |

Measured on the real pilot sites table (`artifacts/sites/sites-20260818T154337Z…`,
98,925 evaluable sites): clean 51,409; substitution 15,774; segmentation 9,981
(7,397 one-span→multi-GT, 2,727 multi-span); insertion-kind 16,646; deletion-kind 5,115
(doctr 741, easyocr 2,154, paddleocr 1,146, tesseract 1,074).

## 4. Data, engines, folds — unchanged

302 matched documents (FUNSD 100, CORD 100, OCR-D-SBB 102) × four real engines
(Tesseract, PaddleOCR, EasyOCR, docTR), the frozen `pilot_loeo_zero_shot` artifacts.
No OCR re-inference; candidate generation consumes frozen spans/sites. The global
document partition and LOEO engine folds are reused as-is. OCR-D remains a historical
stress track: reported per dataset, excluded from headline aggregates, per
`benchmark_design.md`. Roles: `calibrate` = development (design and threshold sanity),
`evaluate` = confirmatory (touched once, after this protocol is committed).

## 5. Generator designs

All generators are ground-truth-blind: they receive only `GenerationContext`
(`original_ocr`, `context_before/after`, native confidences, `n_spans`) and fitted
resources built from allowed training material.

### 5.1 `g5_structural` (CGV2-G3)

Inner substitution competence is **not** reimplemented — `g5` contributes only the shapes
no existing rung produces. Fitted resources (both from the fold's allowed corpus only):
the `LexicalGenerator` lexicon (same construction, so "known word" means the same thing
across rungs) and bigram counts over the same token stream.

Site-level proposals (existing `Candidate` schema, `edit_shape` metadata retained):

- **insertion** (at `deletion`-kind sites, `original_ocr.strip() == ""`): propose the
  most frequent token `t` from the fitted corpus with
  `count(prev, t) ≥ min_bigram and count(t, next) ≥ min_bigram`, where `prev` is the last
  token of `context_before` and `next` the first of `context_after`; `min_bigram = 2`.
  At most one proposal per site. Sites with no attested bigram continuation receive none.
- **merge** (at multi-span sites, `n_spans > 1`): propose the separator-free join when it
  is a lexicon entry; also the hyphen-join when that is. At most two proposals.

Region proposals (§6): `region_join` and `region_substitution`.

### 5.2 `g2_byt5_ctx0`

Identical to `g2_byt5` except `context_chars = 0` (the checkpoint's own documented
default). Purpose: separate "the model copied its input" from "the projection could not
attribute the edit" — an explicitly documented open item, not a tuning sweep. One
configuration, frozen here, FUNSD only, K=1.

### 5.3 What CGV2 does not add

No LLM rung, no multimodal rewriter, no retrained neural model, no tuning of frozen
rungs' parameters. Trigger rules are few, lexicon-grounded, and frozen here before any
evaluate-role result exists.

### 5.4 `g6_union` (CGV2-G4)

Composite of `g3_edit_aware` and `g5_structural` under the existing
`CompositeGenerator` semantics (member order, first-occurrence-wins dedup by text,
bounded K), plus `g5_structural`'s region candidates as a separate region pool layer.
Provenance is retained per candidate (`generator_id`), so union tables decompose exactly
into their members.

## 6. Region-candidate representation (n:m across sites)

**Unit.** A region is an ordered pair of adjacent evaluable sites `(i, i+1)` in the same
`(document_id, engine_id)`, contiguous in the linearized OCR stream
(`rebuild_stream`: exact char offsets, single-space gaps). Region size is frozen at
exactly two sites. Region OCR text = the exact stream slice
`stream[i.char_start : i+1.char_end]`; region GT text = space-join of member `gt_text`s,
mirroring site GT construction.

**Storage.** New tables `region_candidates` and `region_labels` under
`results/generated/cgv2/` — the frozen `candidates`/`labels` tables are not touched, so
frozen evaluation labels remain valid and historical consumers are unaffected. Fields:
`region_id = {document_id}:{engine_id}:region:{index:05d}`, member `site_ids`,
`char_start/char_end`, `region_ocr_text`, `region_gt_text` (labels only), `candidate_text`,
`generator_id/version/rank/score`, `edit_shape`, `pool`, `fold_id`.

**Labeling.** The existing string-pure functions, imported, never reimplemented:
`d_before = lev(region_ocr, region_gt)`, `d_after = lev(candidate, region_gt)`,
`classify_accepted` with the same harm policies. The A6 exact-repair rule extends to
regions: at a region with empty GT only the exact edit counts as a repair.

**Triggers (both GT-blind, fold-lexicon-grounded, frozen):**
- `region_join` — the separator-free concatenation of the two sites' OCR texts is a
  fold-lexicon entry (the cross-site merge repair: `data` + `base` → `database`);
- `region_substitution` — a fold-lexicon entry within Levenshtein ≤ 2 of the
  separator-free concatenation (cross-boundary word-pair repairs).

**Eligibility denominators.** Regions are counted as: eligible (both sites evaluable and
contiguous), ambiguous (either member site non-evaluable), or excluded (non-adjacent in
reading order). Every structural table reports these denominators; ambiguous regions are
never forced into evaluation.

## 7. Leakage control (first-class)

For every LOEO fold `h` (held-out engine), the fold's allowed material is
`(E \ {h}) × D_fit` — train engines' fit documents, exactly as
`experiment-leakage.md` defines the fit role. Every fitted resource (`g0`/`g1`/`g3`
lexicons, `g5` lexicon and bigrams) is built from that material only.

- **Machine-readable certificate per fold**: `results/generated/cgv2/leakage_certificates.json`
  records, for each (fold, rung): lexicon size, the held-out engine, and the assertion
  `|lexicon ∩ forbidden| == 0`, where `forbidden` is the vocabulary the held-out engine
  uniquely contributed to the all-engines corpus. Verified by rebuilding through
  `LexicalGenerator.fit` with and without the forbidden source (the
  `measure_lexicon_leak.py` construction — one definition of "the lexicon", no reimplementation).
- **Red-team tests**: a sentinel token injected as the held-out engine's contribution must
  be absent from the fold lexicon and from every proposal at that engine's sites; a
  `GenerationContext` carrying a GT-derived string into a forbidden interface must be
  rejected (extending `tests/leakage/test_generator_is_gt_blind.py`).
- The study's existing leave-engine-out lexicon rule stays; the certificates make it
  auditable per fold rather than asserted.

## 8. Ground-truth blindness

GT is used only for: evaluation labels (site and region), offline error taxonomy, oracle
opportunity, and post-hoc analysis — never for generation, ranking, pruning, or rung
selection. `g5`'s triggers read confidence, vocabulary membership, and bigram counts from
the fold's fit corpus; nothing else. Red-team tests per §7. Rung selection uses
`calibrate`-role evidence only; the `evaluate` role is run once, after this protocol is
committed, and no design change follows from it (any such change is an amendment).

## 9. Pools

Natural pool only, everywhere. Challenge/hard-negative candidates never enter CGV2
tables; the existing `tests/leakage/test_candidate_pool_boundary.py` guards are extended
to the region layer (`pool == natural` asserted on every row; a challenge row in a
readiness calculation raises).

## 10. Metrics

Existing definitions are reused by import (`metrics/generator.py`; one definition, one
place — `test_metrics_centralized.py` enforces it): `error_repair_opportunity`,
`oracle_safe_coverage`, `oracle_repair_recall`, `clean_span_proposal_rate`,
`beneficial/harmful_candidate_rate`, `exact_correction_rate`. Regions enter these
functions as records with the same shape (d_before, outcomes, deltas, needs_deletion) —
no reimplementation.

New CGV2 tables (all computed from canonical artifacts, no manual numbers):

- **candidate opportunity** — availability (`P(∃ beneficial candidate)`, i.e.
  repairable-site share) and exact availability (`TRUE_CORRECTION` share), by rung ×
  engine × dataset × stratum (§3 kinds) × K.
- **candidate quality** — beneficial / harmful / neutral fractions, harmful:beneficial
  ratio, candidates per site, useful candidates per site, by rung × engine × stratum.
- **structural coverage** — §3 table restricted to segmentation/insertion/deletion-kind
  strata, per operation.
- **preservation risk** — `clean_span_proposal_rate` per rung, plus the region analogue
  (regions touching ≥1 clean site that received a region proposal).
- **budget study** — §12.
- **failure taxonomy** — §13.
- **oracle opportunity** — `oracle_safe_coverage` and oracle-accepted-edits per engine,
  per rung, with the §14 counting rule for regions.

## 11. Hypotheses and endpoints

**CGV2-H1 — correct-candidate availability.** *Primary comparison (frozen):*
`g6_union` vs `g3_edit_aware`, Δ `error_repair_opportunity`, per engine, `evaluate` role,
harm policy `strict_worsening`, paired document-cluster bootstrap
(`stats.bootstrap.cluster_bootstrap`, document = resampling unit), 95% CI and two-sided p.
*Multiplicity family:* the four per-engine tests of this comparison; Holm at α = 0.05.
Also reported: exact-availability variant; `g5` vs no-rung on its own strata (diagnostic).

**CGV2-H2 — structural error coverage.** Δ availability on the **structural stratum**
(site kinds `deletion`, `insertion`, `segmentation`), same machinery, family = four
per-engine tests, Holm. Per-operation breakdown (merge / split / insertion / region) is
reported as descriptive tables with CIs, not gated.

**CGV2-H3 — harmful-candidate control.** Pre-registered bounds, evaluated per engine on
`evaluate` role, `g6_union`: (i) harmful:beneficial ratio ≤ 2 × `g3`'s ratio on that
engine; (ii) `clean_span_proposal_rate` ≤ 0.30 (the retired `g0` baseline's dev-partition
level — a bound set from published dev data, not from any evaluate result). Candidates
per site and useful candidates per site reported for context.

**CGV2-H4 — oracle opportunity / H2 identifiability.** Oracle opportunity per rung per
engine under the §14 counting rule; the H2-readiness gate recomputed from the frozen
criteria with the §14 amendment. No new readiness threshold is invented; the amendment
changes only how region and insertion candidates are *counted*, not any threshold.

**Sensitivities (pre-registered, not gated):** harm policy `non_improving`; exact-only
variant; with/without OCR-D; with/without R-37 twin-signature sites (§15); per-dataset
tables.

## 12. Candidate budget study (K-grid, frozen)

K ∈ **{1, 2, 4, 8}**. Generation for the K-study runs at cap 8 (a diagnostic
configuration that does not modify the frozen rungs' K=4/1 study pools); recall@K =
availability using candidates with `generator_rank < K` only. Reported: recall@K by rung
× engine, and harmful-candidate burden@K (harmful proposals per site within top-K).
`g2` rungs emit K=1 by construction; their curves are flat and labelled as such.
Figures A and B (§17) render these tables.

## 13. Failure taxonomy (error-class analysis)

For every evaluable **error site without a beneficial candidate** in the `g6_union`
evaluate pool, exactly one class, in precedence order:

1. `repair_not_expressible_by_rung` — the site's correct edit shape is outside the rung's
   expressible set (§3 table);
2. `expressible_not_generated` — shape expressible, no proposal fired (trigger gates: vocabulary, confidence, bigram count);
3. `generated_beyond_K` — a beneficial candidate exists at rank ≥ K;
4. `beneficial_absent_vocabulary` — the GT string is not in the fold lexicon (OOV ground truth);
5. `alignment_ambiguous_site` — the site is evaluable but carries the R-37 twin signature or `min_align_confidence` at the floor (sensitivity stratum, not an excuse: counted, not dropped);
6. `historical_charset` — OCR-D long-s / Fraktur forms the engine cannot emit;
7. `normalization_mismatch` — GT differs only by transcription convention;
8. `other`.

Counts and shares by rung × engine × class, plus a manually audited sample of 30 cases
(documented with site ids and text) for interpretability. The pre-existing
`candidate_failure_classes.csv` taxonomy remains for the site-level view; this extends,
not replaces, it.

## 14. H2-readiness counting amendment (minimum, frozen)

The frozen criteria (`h2_readiness_protocol.md` §3: A ≥ 245 oracle-accepted edits on ≥3/4
engines; B oracle safe coverage ≥ 0.05 on ≥3/4; C ≥ 1 000 natural non-identity candidates
per engine and ≥ 20 test documents; D ≥ 30% clean sites and ≥ 100 clean-site proposals per
engine; E 100% natural; F matched + alignment sensitivity) are unchanged in every
threshold. The amendment defines how the CGV2 pool is counted:

- The CGV2 pool for gate recomputation is `g6_union`: site-level candidates **plus**
  region candidates, `evaluate` role, natural only.
- **A (accepted edits):** a site counts as oracle-accepted if a site-level acceptance
  repairs it, **or** a region-level acceptance whose members include it repairs the
  region. A region acceptance marks each member site accepted — so A counts repaired
  sites and stays directly comparable to the historical criterion; no double counting is
  possible because the unit is the site either way.
- **B (safe coverage):** oracle-accepted sites / evaluable sites — the augmented
  `oracle_safe_coverage`.
- **C:** natural non-identity candidates = site + region candidates.
- **D:** clean-site proposals include region proposals touching a clean site.
- **E, F:** unchanged; F additionally requires the §15 audit artifacts to exist.

The historical gate artifact is preserved; CGV2 writes
`results/generated/cgv2/h2_readiness_gate.json`.

## 15. Alignment twin audit (mandatory before confirmatory evaluation)

Extends ADR-17's twin logic to the in-region side (R-37) and to candidate-to-target
projection, as committed producers (no second alignment implementation):

- **View A — OCR-to-GT error structure.**
  `results/generated/cgv2/alignment_twin.csv`: per engine × site kind — n sites, sites
  whose OCR/GT text exactly matches an unanchored opposite-side token on the same page
  (the R-37 signature), share. `alignment_component_audit.csv` retains every component's
  structural operation and explicit eligible / ambiguous / excluded / unresolved state.
  `region_pair_audit.csv` and `region_census.csv` retain adjacent-pair ids, boundary
  diagnostics, reasons, and denominators before and after exclusion.
- **View B — candidate-to-target projection.**
  `site_projection_audit.csv` and `region_projection_audit.csv`: for every candidate —
  source span/component ids, left/right member projections, independently recomputed
  range/gap, punctuation/whitespace boundaries, overlap/empty-anchor cases, and label
  validity. `alignment_audit_gate.json` passes only when every candidate projects and all
  denominators reconcile.
- **R-37 is measured, not fixed, in CGV2.** Fixing it would change alignment statuses →
  sites → every frozen pool number, invalidating the frozen evaluation labels (a §20 stop
  condition). Instead: the twin signature is computed per site, and every headline CGV2
  table has a twin-signature sensitivity variant. The fix remains a recommendation for
  the next phase.

CLI: `ocr-risk analyze alignment-twin` (mirrors the R-36 precedent).

## 16. Generator gate (pre-registered)

Written before any `evaluate`-role result exists. Thresholds derive from the frozen H2
criteria, published dev-partition numbers, and arithmetic — never from an evaluate result.

- **CG1 (availability):** the CGV2-H1 primary contrast is positive with Holm-adjusted
  95% CI excluding 0 on ≥ 3/4 engines.
- **CG2 (harm bounded):** both CGV2-H3 bounds hold on 4/4 engines.
- **CG3 (structural coverage):** the CGV2-H2 structural-stratum contrast is positive with
  CI excluding 0 on ≥ 3/4 engines, **and** insertion-stratum availability is > 0 on
  ≥ 3/4 engines (from exactly 0 today).
- **CG4 (H2 floor):** the §14-recomputed H2 gate meets criteria A and B on ≥ 3/4 engines.

Verdict rule (mirrors the existing precedence): all four → **GENERATOR READY**; CG4 fails
but ≥ 1 of CG1–CG3 met → **GENERATOR PARTIALLY READY**; none of CG1–CG3 met →
**GENERATOR NOT READY**; any criterion whose evidence was not produced → **INCONCLUSIVE**.
Output: `results/generated/cgv2/generator_gate.json`. The H2 verdict is recomputed
independently afterward and may disagree with CG4 only by recording why.

## 17. Figures and report

Figures A–E (budget-recall; beneficial/harmful composition; structural coverage;
per-engine availability; oracle/H2 diagnostic), regenerated from canonical CSVs and
hashed in `figure_manifest.json`. Final report `docs/cgv2/report.md` with the eighteen
required sections, separating preregistered / amended / exploratory / confirmatory.

## 18. Amendment log

**A1 (before any evaluate-role run; precision, not a threshold change).** Region
triggers apply only where both member sites have non-empty OCR text. A region with an
empty-OCR member would fire `region_substitution` on the other member's text alone —
a site-level edit wearing a region id — so such pairs count in the §15 census as
`degenerate` and receive no region candidates. Second, the A6 exact-repair rule extends
to regions in its original spirit: a region with **any** empty-GT member (an
`insertion`-kind site) requires an exact repair — candidate == region GT — for its
acceptance to count as a repair, because edit distance otherwise scores a merged
hallucination (`data base` joined to `database` against a reference of `data`) as a
partial improvement. Implemented as the region's `needs_deletion` flag being true
whenever either member is insertion-kind.

**A2 (after the calibrate-role run, before any evaluate-role run; generator-design
change, no threshold change).** The v1 insertion trigger — both the previous-token and
next-token bigram attested ≥ 2, candidate in the min-length-3 fold lexicon — fired
**zero** times across all 16 780 calibration sites. The same census shows why: the
empty-OCR stratum's ground truth is **3 characters at the median** (top values `-`, `1`,
`of`, `(`, `to`, `☐`, `mm`), and only 14.6% of it is in the fold lexicon at all, so the
vocabulary gate excluded most of the population by construction. The v2 trigger: the
insertion vocabulary is the full fold token distribution (length ≥ 1, frequency ≥ 2,
punctuation and digits included), and attestation is **one-sided** — the forward bigram
(prev, t) or the backward bigram (t, next) attested ≥ 2 suffices, scored by the sum of
the two counts, at most one proposal, unattested sites still receive nothing. Rationale:
a missing form value follows its field label, and whatever follows it is the next
field's first token, which no fold bigram constrains. The calibration evidence for this
amendment is development-role only; no evaluate-role data existed when it was written.
The merge and region triggers are unchanged — their small positive dev deltas are the
honest structural finding and are not tuned.

**A3 (before any evaluate-role run; representation corrections from the independent
alignment review, no threshold change).** The review of the region layer found the
original reconstruction wrong in ways the protocol's own §15 audit could not see, and
the corrections change what the frozen words *mean*, not what they require:

1. **Region OCR is the exact stream slice.** `region_ocr = stream[left.char_start :
   right.char_end]`, never site text plus assumed separators. Measured on the frozen
   corpus: 17.7% of offset-adjacent pairs are separated by a **newline**, not a space,
   and 597 pairs contain a non-evaluable site's characters inside the "gap". The pair
   census now names these: `cross_line`, `intervening_text`, `text_mismatch`,
   `non_adjacent`, `degenerate`, `no_stream`, `eligible`. Only `eligible` (pure
   same-line whitespace middle) produces region candidates. The offset-gap rule and its
   1–3 window are withdrawn — subsumed by the exact-slice rule.
2. **Empty-OCR sites take their context from alignment neighbours.** A span-less site
   pins to char range [0, 0], so its stream window is empty *by construction* — the A2
   trigger still fired zero times because it never saw a context. The before/after text
   for those sites now comes from the OCR text of the adjacent alignment components
   (GT-blind: OCR text and ordering only), for every rung.
3. **View B is non-circular.** The projection audit recomputes the slice, the char
   range, and the gap from the sites table and the stream; the study's recorded `gap`
   and range are checked, never trusted. Region records carry `region_char_start` and
   `region_char_end` as §6 always listed.
4. **Region candidates count in the harm bounds.** `candidate_quality` and
   `preservation` include the region pool for `g6_union` (protocol §14's pool), so CG2
   and criterion D bound the pool that exists. The oracle table separates
   `sites_via_region` from site-level credit (the credit-both rule of §14 unchanged).
5. **The R-37 sensitivity is wired.** `alignment_twin_sites.csv` names every
   twin-signature site; `cgv2 tables --sensitivity no_twin` re-derives every table and
   the primary contrasts excluding them; the failure taxonomy gains
   `alignment_ambiguous_site` (twin signature or component confidence at the floor) and
   `historical_charset` (OCR-D long-s `ſ`).
6. **Every structural unit has an explicit disposition.** The audit now retains all
   OCR↔GT component shapes (1:1, 0:n, n:0, 1:n, n:1, n:m), every adjacent region pair,
   and every site/region candidate projection with an `eligible`, `ambiguous`, `excluded`,
   or `unresolved` state. Denominators before and after exclusion are machine-reconciled;
   candidate-level overlap, punctuation/whitespace boundary, and empty-anchor flags are
   descriptive audit fields, never new selection rules.
7. **Candidate provenance and pool membership are materialized.** Site and region rows
   retain deterministic candidate ids, fold/config/version/score/rank, member-generator
   provenance, source span and alignment ids, operation type, K, ambiguity status, and
   `pool`. A deduplicated source-geometry table retains boxes, polygons, normalized
   coordinates, and crop recipes by stable reference. Headline tables, oracle counts, and
   readiness criterion E now fail closed if `pool` is missing or not `natural`; this
   replaces an unfireable hard-coded pass without changing the criterion.

All corrections were made on calibration-role evidence and review findings only; no
evaluate-role data existed when this amendment was written.

**A4 (after the clean calibration preflight, before any evaluate-role run; enforcement of
already-frozen scope and lineage rules, no generator or threshold change).** The preflight
found four implementation gaps between the executable analysis and the text frozen above:

1. Section 4 already classifies OCR-D-SBB as a historical/model-availability stress track
   excluded from headline engine-transfer aggregates, but the first CGV2 contrast/oracle
   writers still pooled it. Headline contrasts, tables, figures, and gates now use CORD +
   FUNSD only; the immutable canonical candidate files retain all three corpora, and
   per-dataset plus explicit `with_ocrd` sensitivity artifacts retain the stress result.
2. H2 criterion C previously read `g6_union.n_candidates` from the quality table (which A3
   had already augmented with regions) and then added the region count a second time. It
   now counts site rows plus region rows exactly once from the canonical natural-pool CSVs.
3. The R-37 command re-derived tables and contrasts but omitted the augmented oracle. It
   now re-derives the oracle as well. Criterion F requires the complete audit files, matching
   proposal hashes, and the role-matched no-twin contrast/opportunity/quality/oracle family;
   an unattached boolean cannot pass it.
4. The pre-registered `non_improving`, `exact_only`, with-OCR-D, and no-twin variants now
   have explicit CLI paths and hashed analysis manifests. The separate generator and H2
   JSON gates are emitted by `cgv2 gates`, closing the protocol's machine-readable-output
   requirement.
5. The descriptive region-opportunity table counted partial improvements even when a
   member was insertion-kind, contrary to A1's already-frozen exact-repair rule. It now
   applies the same exact requirement as the augmented oracle; candidate quality retains
   the outcome taxonomy itself, as specified.
6. Preservation omitted a rung × engine row when that generator made zero proposals on
   clean sites, turning the safest possible observed value into missing evidence. The
   complete rung × engine grid now records an explicit zero with the real clean-site
   denominator.

These are fail-closed lineage and scope corrections. They do not alter a candidate,
candidate rank, K, hypothesis, endpoint direction, multiplicity family, uncertainty
method, or gate threshold. No evaluate-role artifact existed or was inspected when A4 was
written.

**A5 (during the final calibration release audit, before any evaluate-role run;
enforcement of the frozen statistical configuration).** The resolved experiment config
sets `stats.n_bootstrap: 10000`, `bootstrap_seed: 7`, `ci_level: 0.95`, and
`bootstrap_unit: document`; `engine_contrast` also defaults to 10,000. The CGV2 wrapper
nevertheless overrode the resample count with 2,000, an undocumented performance shortcut.
The wrapper now receives all bootstrap settings from the resolved config, the study JSON
records them, every contrast row records them, and sensitivity re-derivation refuses a
study without that provenance. All calibration contrasts are regenerated at 10,000 before
confirmatory release. This changes Monte Carlo precision only; it does not change the
paired statistic, unit, interval level, multiplicity family, effect direction, or any gate
threshold. No evaluate-role artifact existed or was inspected when A5 was written.

**A7 (after the evaluate-role run and the independent reviews; validity repairs forced by
review findings R-65/R-66/R-67, no threshold or hypothesis change).** The reviews found
two Critical defects: the natural pool's site topology is constructed from GT alignments
(R-65), and the evaluate partition was already analyzed before CGV2 was designed (R-66).
The repairs change what the frozen tables *measure*, and are recorded here because they
were written after the evaluate run existed — designed against the review findings and the
audit machinery, never tuned on an outcome value:

1. **The alignment-validity overlay.** The section-15 audit gains a per-site eligibility
   state (`eligible` / `ambiguous` / `excluded` / `unresolved`) decided from the frozen
   artifacts alone: the uniqueness margin is recomputed against *every* GT token on the
   page (the frozen aligner considered only the first 64, so a near-tie later in reading
   order could be silently resolved); a GT-only component with no OCR span is an
   `unresolved` empty-insertion-anchor, because its `[0, 0]` range is a serialization
   default, not a location; a multi-span site whose stream slice differs from its text is
   `ambiguous`, because its min/max envelope is not an editable source region. Candidate
   labels (`d_before`, `d_after`, `delta`, outcome) are recomputed from the strings,
   independently of the study. The audit never rewrites a frozen artifact.
2. **The overlay cleans the pool, not the benchmark.** Census denominators keep every
   evaluable site — dropping anchor-less sites would silently shrink the insertion
   stratum and turn CG3 into missing evidence instead of a measured zero. Candidates
   enter headline tables only with a valid projection and a recomputed-valid label; an
   audit that does not cover every candidate aborts rather than silently shrinking the
   pool. Consequence, measured not assumed: the 5,115 deletion-kind sites are exactly
   the unresolved-anchor population, so the insertion stratum's verifiable-pool
   availability is zero by construction — the quantification of R-65, not a generator
   failure.
3. **Native frozen caps in headline tables.** Site rungs count `generator_rank < ` the
   rung's frozen ladder cap (4 for `g0`/`g1`/`g3`/`g4`, 1 for the `g2` rungs, and 8 for
   `g5_structural`/`g6_union` — the caps the ladder froze at CGV2-2 for the §12 budget
   study); region candidates count `rank < 2`. The cap-8 K-grid remains the diagnostic
   budget study. Contrasts always carry their native caps, standard error, and minimum
   detectable effect.
4. **Region denominators from the audited eligible-pair population**, with a
   document-cluster bootstrap interval on region-opportunity and structural-operation
   cells (resamples/seed/level from the study's frozen bootstrap block).
5. **Role-scoped audits.** Each role's View B binds that role's proposals by hash under
   `audits/{role}/`; the tables and gates commands refuse a missing or incomplete
   role-scoped audit, and criterion F additionally verifies the audit outputs and the
   hash-bound no-twin manifest.
6. **The integrity gate (new, fail-closed).** A tracked review file
   (`docs/cgv2/integrity_review.json`) records reviewer findings with severity and
   dispositions; `ocr-risk cgv2 integrity` derives `cgv2_integrity_gate.json` from that
   file plus the immutable artifacts (pool provenance, role isolation, fold-scoped
   resources, selection scope, partition freshness re-derived mechanically;
   GT-blind site construction must be affirmatively certified or the gate stays closed).
   The verdict rules gain a floor: a produced-but-failed integrity review makes the
   generator verdict **NOT READY** regardless of numerical criteria, and the gates
   command refuses to run without the integrity gate. Under R-65/R-66 the evaluate pass
   is `exploratory_only` and no CGV2 table may be read as confirmatory.

The immutable study artifacts (census, proposals, region proposals, fold lexicons,
certificates, study record) are unchanged; every derived table, contrast, oracle, figure,
and gate is regenerated from them under these rules. No criterion threshold, hypothesis,
endpoint direction, multiplicity family, or uncertainty method changed.

## 19. Execution order and freeze discipline

1. Implement `g5_structural`, `g2_byt5_ctx0`, `g6_union`, the region layer, certificates,
   twin audit, analyses — with unit tests, `make check` green, `make smoke` green.
2. Tiny real-data pilot (one engine, one fold subset); runtime measured.
3. `calibrate`-role study over the full ladder; design sanity only. Threshold changes
   after this point are amendments.
4. **Commit this protocol** (already done at step 0 of the phase; any later edit is §18).
5. `evaluate`-role study, run once.
6. Analyses, figures, gates, reviews (A leakage, B alignment, C statistics,
   D research integrity — each asked to invalidate), fixes, report.

## 20. Stop conditions

Stop and report rather than silently repairing the scientific question if: the structural
representation invalidates frozen evaluation labels; target-engine leakage cannot be
eliminated (certificate assertion fails); alignment ambiguity prevents reliable structural
labels (twin/ambiguity share so large the structural stratum is uninterpretable); the
ByT5 checkpoint cannot be reproduced; any `evaluate`-role data is inspected before the
protocol commit; or the pool remains too sparse for the pre-registered analysis. A
scientifically justified NOT READY is preferable to an uninterpretable positive.

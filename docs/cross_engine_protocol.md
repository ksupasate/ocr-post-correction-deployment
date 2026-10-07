# Cross-engine protocol

The highest-stakes specification in the repository. A leak here silently invalidates every
number in the paper, and the failure is invisible in the results — they simply look good.

## Two axes, always applied together

### Why one axis is not enough

The benchmark is **matched-source**: every engine reads the same page, because otherwise
engine shift would be confounded with document shift. That design decision creates the
project's largest validity threat.

Holding out an engine does **not** hold out the page. A verifier fitted on engine B's
reading of document D and evaluated on engine A's reading of D has already seen D's ground
truth. This is leakage vector **L4**, and it is easy to reintroduce because each fold looks
internally correct.

### The protocol

| role | engines | documents |
|---|---|---|
| `fit` | all except the held-out engine | `D_fit` |
| `dev` | all except the held-out engine | subset of `D_fit` |
| `calibrate` | all except the held-out engine | `D_cal` |
| `evaluate` | **only** the held-out engine | `D_test` |

`D_fit / D_cal / D_test` come from **one** partition, drawn once, hashed, and identical
across every engine and every fold. A partition redrawn per fold would rotate documents
between roles and undo the split while every fold still validated individually — which is
why `document_partition_sha256` is recorded on every run and the audit rejects a protocol
whose folds disagree on it.

### Order of operations, per fold

1. fit the featurizer and verifier on `train engines × D_fit`
2. fit the calibrator on `train engines × D_cal`
3. select `tau(epsilon)` on the same calibration scope
4. **freeze all three**
5. score and decide on `held-out engine × D_test`

Steps 1–3 never see the held-out engine or the test documents.

## How that is enforced

Three independent mechanisms, because any one of them could have a bug:

**By type.** `SplitPlan.view(role=...)` is the only way to obtain rows. The `evaluate` view
refuses to expose labels at all — `view.labels("harmful")` raises `LeakageError` — so
selecting a threshold against test outcomes fails loudly rather than being caught in
review.

**By layering.** `verify/`, `calibrate/`, and `risk/` may not import `metrics/`
(`tests/architecture/test_layering.py`), so evaluation labels are unreachable by the other
route as well.

**By post-hoc audit.** `ocr-risk audit leakage` re-derives the engine and document sets
from finished run records, independently of the guard that was active when the run
happened. A bug in the guard would pass the guard's own tests and be caught here.

## The twelve vectors

| # | vector | mechanism |
|---|---|---|
| L1 | threshold selected on held-out labels | `evaluate` view exposes no labels |
| L2 | calibrator fitted on the held-out engine | `Calibrator.fit` takes a calibration view |
| L3 | feature statistics pooled across splits | fitted transformers; `transform` before `fit` raises |
| L4 | **same document under a train engine and the test engine** | global document partition |
| L5 | lexicon built from test text | generators take an explicit fit-split corpus |
| L6 | hard negatives from test-document GT used in fitting | tagged, and excluded from headline evaluation |
| L7 | config tuned against fold test metrics | `selection_scope` recorded; audit rejects |
| L8 | near-duplicate documents split across roles | shingle + image hash forces one bucket |
| L9 | ground truth reaching the verifier as a feature | `ObservationView` has no label field |
| L10 | subsampling applied to evaluation | fit-only option |
| L11 | calibration set reused for evaluation | disjointness asserted at construction |
| L12 | engine fingerprint drift | fingerprint on every span |

Each has a named test in `tests/leakage/`, several of them **red-team tests** that attempt
the violation and assert it raises.

## Protocols

| protocol | what it answers |
|---|---|
| `loeo_zero_shot` | the headline. Nothing from the held-out engine touches any fitted quantity. |
| `loeo_few_shot_recal` | how much target-engine data buys back the loss. Weights stay frozen; a `k`-document sample from `D_cal` — disjoint from `D_test` — may refit the calibrator. |
| `in_engine_oracle` | **not a competing method.** The held-out engine is included in fitting, and the gap to zero-shot is the measured cost of engine shift. This is what H1 reads. |
| `diagnostic_doc_overlap` | deliberately violates the document split, to *quantify* the inflation that violation causes. Flagged `leaky`, and `analysis/tables.py` raises if it reaches a headline table. |
| `pairwise_transfer` | fit on one engine, evaluate on another. 12 off-diagonal folds, because four leave-one-out points is thin ground for a generalization claim. |

## The document partition

Assignment orders documents by a seeded hash and cuts at the configured fractions, so the
proportions are exact.

**This is a deliberate trade.** Exact proportions and stability-under-corpus-growth are
mutually exclusive: holding the fractions exact while adding documents forces some existing
document to change role. Per-document hash bucketing gives the opposite trade, but at the
corpus sizes this project runs (tens of pages per pilot track) it left the evaluation split
at three documents — too few for a document-level bootstrap, and a problem on *every* run.

So the partition is an **artifact**: drawn once, saved to
`manifests/splits/<partition_id>.document_partition.json`, and reused. It is loaded on
every run and verified against its own recorded hash, because a partition that is
recomputed each time rotates documents between roles whenever the corpus changes while
every run record still looks self-consistent.

`partition_id` is a config field rather than the experiment name. The two arms of H1 —
zero-shot and the engine-aware reference — **must** draw on the same partition, or the
contrast between them is confounded by which documents each happened to evaluate.

Three kinds of document are assigned **as a unit**:

- **the same page**, by `image_sha256`;
- **near-duplicates**, by ground-truth shingle hash;
- **pages of the same work**, by `work_id`.

The third is the one the first two cannot see. Two pages of one book are genuinely
different pages with different content, so no hash relates them — but they share a
typeface, a scan session, a binding and often a running head, and a verifier fitted on
pages 3-7 of a volume and evaluated on page 8 has already seen that printing. The
historical track is 102 pages drawn from 31 volumes, so without work-level grouping most
volumes would straddle the partition.

## Known limitation

Four engines means four folds. That is a small number for a claim about generalizing to
unseen engines, and no amount of bootstrap resampling fixes it: the uncertainty is over
*engines*, and there are four. Per-fold results are always reported, the pairwise matrix
is implemented but was not run in this pilot, so it adds nothing here, and
the platform limitations record (excluded historical documentation) states the constraint
plainly.

# Statistical methodology

## The resampling unit is the document

Edits within a page share a scan, a font, a degradation, and often a systematic engine
failure. Resampling individual edits treats correlated observations as independent and
produces intervals far too narrow — not a conservative approximation, a wrong answer that
makes non-results look significant.

`tests/unit/test_stats.py::test_cluster_bootstrap_is_wider_than_naive_on_correlated_data`
demonstrates this on constructed data: the document-level interval is more than three
times wider than the edit-level one. The test exists so nobody can "simplify" the
resampling unit back and keep a green suite while every reported interval becomes
dishonest.

## Intervals

**Percentile bootstrap over clusters**, ≥ 10 000 resamples by default. Not BCa: its
acceleration term is jackknifed over clusters and is unstable when there are few documents
— exactly the regime per-fold analyses live in. A wider honest interval beats a narrower
one built on an unstable correction.

With fewer than two clusters the result is `NaN` rather than a zero-width interval. One
document carries no information about between-document variability, and reporting
certainty we do not have would be worse than reporting nothing.

## Paired comparisons

Method contrasts on the same documents use `paired_cluster_bootstrap`, which resamples the
**same** draw of documents for both methods. Bootstrapping independently discards the
pairing and inflates the interval, obscuring a real difference between two methods
evaluated on identical pages. Both behaviours are tested against each other.

Effect sizes accompany every difference: Cohen's *h* for proportions, Cliff's delta where
distribution-free is appropriate. A statistically distinguishable change of 0.2 percentage
points is not the same claim as a meaningful one.

## Multiplicity

Holm by default. Holm controls the family-wise error rate under **arbitrary dependence**,
which is the honest assumption here: folds share documents and the epsilon levels are
nested, so the tests are dependent in ways not worth modelling. Benjamini–Hochberg is
available for exploratory families where FDR is the appropriate weaker guarantee; the
choice is recorded.

**The family is declared before the run.** For H1 it is the four target engines, *within*
each metric — not the twelve engine × metric combinations. Pooling them is a stricter
procedure, and choosing between the two after seeing results is precisely what declaring
the family in advance prevents.

A demonstration test: `p = 0.04` is significant alone and is not once it is one of four
tests in a family.

### p-values come from the resample distribution

Holm needs p-values it can *order*. Encoding "the interval excluded zero" as the boundary
value `1 - ci_level` gives a family of identical p-values, and Holm's first comparison
(`p <= alpha/m`) then fails for every member — so a family of four could never contain a
single rejection, whatever the data said. `BootstrapResult.p_value_two_sided` is therefore
a real bootstrap p from the draws, with the customary `+1` correction in numerator and
denominator: the smallest attainable value is `2 / (B + 1)`, never zero, because a
bootstrap of `B` resamples cannot distinguish a smaller p from zero and should not claim
to.

Significance is necessary and not sufficient for a finding of degradation. A surviving p
says the difference is not zero; the *direction* is what makes it degradation rather than
improvement, and both are required.

## The risk bound, and its violated assumption

Learn-then-Test with Hoeffding and Bentkus p-values, taking the tighter pointwise. Bentkus
is substantially tighter in the small-risk regime this project operates in; at a 1%
tolerance Hoeffding needs far more calibration data than a per-fold split has. `delta` is
split across the threshold grid, because selecting over that grid is itself a multiple
comparison and ignoring it would void the guarantee.

`test_ltt_controls_risk_under_exchangeability` verifies by Monte-Carlo that realized risk
on fresh exchangeable data exceeds `epsilon` in at most `delta` of trials. Implementing a
bound is not the same as the bound holding.

**Exchangeability is violated by construction under cross-engine shift.** That is what H1
is about. The analysis therefore reports realized risk against the nominal bound rather
than assuming the guarantee transfers, and the size of the violation is a result rather
than a caveat.

This is what `deployed_operating_points` is for, and why it is reported beside
`coverage_at_risk` rather than instead of it. The two answer different questions:

| table | threshold | what `risk <= epsilon` means there |
|---|---|---|
| `deployed_operating_points` | the one the controller certified on the calibration split, applied unchanged | a measurement that can fail, and failing is the H1 result |
| `coverage_at_risk` | re-derived on the evaluation sample subject to empirical risk within tolerance | true by construction — a constrained in-sample optimum, an upper bound and not a claim |

Reporting only the second would make "the tolerance held" unfalsifiable.

`test_empirical_controller_is_optimistically_biased` measures what the guarantee buys: a
threshold selected on the same sample it is measured on violates the tolerance measurably
more often.

## The recovery phase: RH1's machinery

The recovery hypothesis (RH1) asks whether cross-engine transfer costs the model its
ability to *rank* beneficial edits above harmful ones. Its statistics differ from H1's in
four declared ways, all frozen in `docs/h1_recovery/h2_readiness_protocol.md` §4 before
the run:

- **Endpoint on the raw score.** ROC AUC and the other discrimination endpoints are
  computed on the verifier's raw output, not the calibrated one. Isotonic calibration is
  a step function and collapsed the raw scores to a different number of levels in each
  arm; ranking on the raw score is the invariant the endpoint was chosen for (amendment
  A1).
- **The verdict row pools the donors.** Each target engine has three donor substitutions,
  and they are three views of one comparison, not three independent replicates: the
  pooled row is the mean over donors of the paired delta under a **joint** document
  resample — documents are drawn once and all three deltas recomputed on the same draw —
  so the interval reflects the shared evaluation set (amendment A3).
- **The family is the four target engines.** Holm is applied within each
  `(metric, verifier)` across the four pooled per-engine rows. The per-donor rows are
  diagnostics and carry no adjusted p; correcting across them would make each family
  three engines wide instead of four, weaker than pre-registered in the anti-conservative
  direction. The minimum detectable effect is reported with `n_tests = 4` to match.
- **A pre-committed policy-agreement rule.** H1's verdict flipped across harm policies
  with no rule for what to conclude. RH1's rule was fixed before the run: if the primary
  and sensitivity policies disagree about how many engines degraded, the verdict is
  PARTIALLY SUPPORTED and the disagreement is the headline — including from SUPPORTED,
  where an earlier implementation of the downgrade was silently a no-op. The sensitivity
  tables carry `arms_matched_for_this_policy = false` and are sensitivity, not matched
  contrasts (amendment A4). A sensitivity table with no pooled rows, or an engine with a
  missing or degenerate pooled interval, is evidence that was not produced: the gate
  reads INCONCLUSIVE rather than counting either as a measured zero.

## Declared limitations

- **Four engines means four folds.** No resampling fixes this: the uncertainty is over
  *engines*, and there are four. Engine is treated as a fixed, enumerated factor, not a
  random sample. Per-fold results are always reported. The recovery phase's 12 pairwise
  transfer folds are three views of each engine, not 12 independent engines, which is
  why the verdict is read from the donor-marginalized pooled row.
- **Alignment ambiguity is not uniform across engines** and is reported beside the results
  it could confound.
- **Selective risk has a random denominator**, so the fixed-denominator joint harm rate is
  reported alongside it.

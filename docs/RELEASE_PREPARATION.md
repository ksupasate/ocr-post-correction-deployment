# Release-only software adaptations

The scientific files are unchanged relative to the analysis that produced the reported results; the adaptations below affect packaging and testing only.

- Plot export metadata records a null Git HEAD when extracted without .git; no ranking, policy, source data, captions or plot calculations changed. Two focused tests cover absent Git metadata/executable.
- VF2 group-integrity test fixture reads the final VF2 execution registry instead of the superseded VF1B precursor; group/mutation assertions are unchanged.
- Runtime manifest describes the curated implementation. Required corpus/group and reserve-access metadata are retained; reserve outcomes are not released.
- Two tests requiring historical real OCR/C1 outputs are excluded. The historical Git-tracked-results test is replaced by an explicit release privacy-boundary test. Acceptance and coverage floors are unchanged.
- Data-licensing documentation follows the existing dataset licence registry; a dangling excluded-document link is removed. New helpers stage, hash-validate and replay frozen Markdown tables without fitting or scientific re-evaluation.

- The recorded synthetic Tesseract fixture has redacted host/platform metadata and a relative image command path. Its TSV is unchanged, and the existing canonical hash function updates the payload hash. The TSV payload and its canonical hash are otherwise unchanged.

- Git provenance tests now create a temporary committed synthetic repository and check clean/dirty states explicitly; a separate test checks extraction without a repository. This preserves real Git capture assertions while making ZIP extraction testable.

- The synthetic CLI baseline test redirects that legacy command's output into its temporary fixture and asserts the file exists. This prevents test products from entering the software/evidence trees; the CLI implementation and scientific outputs are unchanged.

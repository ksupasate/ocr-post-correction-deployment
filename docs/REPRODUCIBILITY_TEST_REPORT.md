# Reproducibility test report

For "When ranking is not enough: selecting deployment cutoffs for OCR post-correction under generator and population shift". **Software/frozen-artifact checks PASS; publication reconciliation PENDING.**

The clean local approximation used separate fresh virtual environments for the software and
the staged replay, Python 3.11.15 on macOS/ARM64, and the unchanged uv.lock. It did not run a new
scientific experiment, corpus download, full-corpus OCR, API generation or real-data model training.
The numeric reproducibility basis is frozen corrected VF2 evidence. Tests use synthetic fixtures.

| Command / context | Recorded result |
|---|---|
| `uv sync --locked --extra dev` in software and staged replay | PASS; 59 installed distributions |
| `python scripts/release_artifact_checks.py --resource ../online_resource_3` from software | PASS; 17 numerical/classification comparisons |
| `python scripts/stage_release_replay.py --resource ../online_resource_3 --output ../replay` | PASS; fresh directory, source hashes verified; no experiments |
| `make -k check` in staged replay, final clean run | PASS; Ruff, strict mypy (168 files), 1,888 tests passed, 46 skipped |
| Coverage global/package gates | PASS; 90.21% total; original per-package floors unchanged |
| `.venv/bin/ruff check src tests scripts` and format check after existing environment helper restored | PASS; 299 Python files formatted |
| `.venv/bin/python scripts/validate_runtime_manifest.py` from software | PASS; 99 paths verified |
| `uv run --locked python scripts/plot_paper3_vf2_figures.py --output-dir rebuild/figures` in replay | PASS; Figures 2-4; all 135 audit records identical |
| `uv run --locked python scripts/replay_frozen_tables.py --output rebuild/tables/P3_VF2_RESULT_TABLES.md` in replay | PASS; frozen full-pool Markdown tables byte-identical |
| `tectonic --keep-logs --outdir <BUILD_DIR> supplement.tex` from OR1 source | PASS; Tectonic 0.16.9; 16-page draft PDF |
| Supplement projection check against authoritative registry | PASS; all 196 recorded value projections identical |
| Supplement layout/reference build check | PASS as draft; all pages visually reviewed; no overfull boxes, missing glyphs or undefined references/citations |
| CFF against official version 1.2.0 JSON schema | PASS schema; author/release metadata remains pending |
| Curated text secret/privacy scan and Parquet schema audit | PASS; 496 Parquet schemas; no prohibited text columns |
| Archive member, CRC, traversal and payload SHA-256 checks | PASS; all four ZIPs; see audit/ARCHIVE_VALIDATION.json |
| Component/root manifest and master checksum verification | Run after final assembly; recorded in audit/FINAL_VALIDATION.json |

Final full-check output is retained in `audit/FINAL_MAKE_CHECK.log`, with workspace paths sanitized.
Figure audit JSON, CSV and captions are byte-identical to the frozen originals. Plot PDF bytes
may depend on rendering libraries/fonts; original final Figure 2-4 PDFs remain unchanged.
The figure validator also checks no-action policies have no harmful-fraction coordinate,
stricter-harm scoring is unchanged, and matched-site restrictions use the registered population.
Existing CSV serialization differences at floating-point machine precision are documented by
the original validator; authoritative registry values take precedence.

Initial preparation checks exposed missing inherited software metadata, tests requiring
superseded real-output artifacts, and the need to stage OR3 for VF2 group tests. These were
resolved with the original metadata and explicit test scope; no acceptance floor was lowered.
The first full staged run passed 1,886 tests but failed a Git-history assumption in ZIP extraction;
coverage reached 90.18%. A subsequent run exposed a synthetic CLI baseline written into the
software tree and was interrupted after the issue was identified. Release-only tests now use
a temporary Git repository and temporary baseline output. The final complete run above passes.
Focused engine-fixture (26 tests), provenance (23 tests), restored metadata (51 tests),
CLI dependency (33 tests), and VF2/reporting/plot (20 tests) checks also passed.
The excluded historical tests and every adaptation are documented in the audit directory.

Known boundaries: the final manuscript/source bibliography, final Figure 1, and final Table 1/2
TeX mapping are unavailable. OR1 is a newly created draft, not a certified final Supplement.
Manuscript authors/contact details are placeholders by author instruction; supplementary rights
and licence approval are pending. The literature CSV is the genuine source format; no XLSX
is supplied. No fully portable corpus-to-results rebuild is claimed: withheld original OCR,
generator/context caches and fitted vocabularies are needed to satisfy frozen reconstruction
hashes. Optional real-data/backend checks may skip when their licensed inputs are unavailable.

After extracting the complete release tree, validate hashes with:

```sh
python github_repo/scripts/release_artifact_checks.py --resource online_resource_3
python github_repo/scripts/release_artifact_checks.py --manifest online_resource_1/MANIFEST.json
python github_repo/scripts/release_artifact_checks.py --manifest online_resource_2/MANIFEST.json
python github_repo/scripts/release_artifact_checks.py --manifest online_resource_3/MANIFEST.json
python github_repo/scripts/release_artifact_checks.py --manifest github_repo/MANIFEST.json
python github_repo/scripts/release_artifact_checks.py --manifest RELEASE_MANIFEST.json
python github_repo/scripts/release_artifact_checks.py --checksums checksums/SHA256SUMS.txt
```

The SHA-256 ledger excludes itself. Root manifests exclude their own JSON/CSV pair and the
master ledger to avoid self-reference; component manifests exclude themselves and are hashed
by the parent ledger. All scientific content files and archives have explicit hashes.

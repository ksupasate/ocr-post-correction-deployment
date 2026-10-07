# Reproducibility test report

For "When ranking is not enough: selecting deployment cutoffs for OCR post-correction under
generator and population shift". Software and frozen-artifact checks PASS.

Validation used separate fresh virtual environments for the software and the staged replay,
Python 3.11 on macOS/ARM64, and the unchanged `uv.lock`. No new scientific experiment, corpus
download, full-corpus OCR, API generation, or real-data model training was performed. The
numerical reproducibility basis is the frozen corrected result registry; tests use synthetic
fixtures.

| Command / context | Recorded result |
|---|---|
| `uv sync --locked --extra dev` in software and staged replay | PASS; 59 installed distributions |
| `python scripts/release_artifact_checks.py --resource ../online_resource_3` from software | PASS; 17 numerical/classification comparisons |
| `python scripts/stage_release_replay.py --resource ../online_resource_3 --output ../replay` | PASS; fresh directory, source hashes verified; no experiments |
| `make -k check` in staged replay | PASS; ruff, strict mypy (168 files), 1,889 tests passed, 46 skipped |
| Coverage global/package gates | PASS; 90.21% total; per-package floors unchanged |
| `uv run --locked python scripts/plot_paper3_vf2_figures.py --output-dir rebuild/figures` in replay | PASS; Figures 2-4; all 135 audit records identical |
| `uv run --locked python scripts/replay_frozen_tables.py --output rebuild/tables/P3_VF2_RESULT_TABLES.md` in replay | PASS; frozen full-pool Markdown tables byte-identical |
| Component/root manifest and master checksum verification | PASS (run after final assembly) |

Figure audit JSON, CSV, and captions are byte-identical to the frozen originals. Plot PDF bytes
may depend on rendering libraries/fonts; the frozen Figure 2-4 PDFs remain authoritative. The
figure validator also checks that no-action policies carry no harmful-fraction coordinate,
that stricter-harm scoring is unchanged, and that matched-site restrictions use the registered
population. CSV serialization differences at floating-point machine precision are documented
by the original validator; registry values take precedence.

Known boundaries: no fully portable corpus-to-results rebuild is claimed. The withheld
original OCR outputs, generator/context caches, and fitted vocabularies are needed to satisfy
the frozen reconstruction hashes recorded in Online Resource 3 provenance. Optional
real-data/backend checks skip when their licensed inputs are unavailable.

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

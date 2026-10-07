# Software for "When ranking is not enough: selecting deployment cutoffs for OCR post-correction under generator and population shift"

**Provisional release workspace. Final manuscript reconciliation and author metadata remain pending.**

This is the software component of a corrected VF2 research release. It includes the actual
`src/ocr_risk` implementation, required script dependencies, environment lock, configurations,
tests, prompts and parsers embedded in those implementations. It contains no legacy result packages.
The evidence is a corrective re-analysis on previously observed test outcomes, with FUNSD
form/document and OCR-D volume group isolation. Ranking quality, site-winner ranking, cutoff
selection and deployment are distinct; stricter-harm and matched-site population sensitivities
have separate meanings. No causal generator effect or formal safety certificate is claimed.

## Install and environment

Use Python >=3.11 and uv. The unchanged `pyproject.toml` and `uv.lock` are authoritative:

```sh
uv sync --locked --extra dev
```

Heavy OCR backends and external model/API extras are optional. They are unnecessary for frozen
artifact replay. `make smoke` exercises synthetic data only and produces no research result.
See [the test report](docs/REPRODUCIBILITY_TEST_REPORT.md) for tested commands and
[release preparation notes](docs/RELEASE_PREPARATION.md) for documented adaptations.

## Reproduce frozen figures and tables

Online Resource 3 is supplied separately to avoid large numerical files in the software archive.
From this directory, with OR3 extracted at `../online_resource_3`:

```sh
uv run --locked python scripts/release_artifact_checks.py --resource ../online_resource_3
uv run --locked python scripts/release_artifact_checks.py --manifest ../online_resource_3/MANIFEST.json
uv run --locked python scripts/stage_release_replay.py --resource ../online_resource_3 --output ../replay
cd ../replay
uv sync --locked --extra dev
make check
uv run --locked python scripts/plot_paper3_vf2_figures.py --output-dir rebuild/figures
uv run --locked python scripts/replay_frozen_tables.py --output rebuild/tables/P3_VF2_RESULT_TABLES.md
```

The plotter checks frozen implementation/input hashes and source values before rendering
Figures 2-4. It never fits a model or evaluates new performance. The table replay checks byte equality against the frozen corrected result tables.
Full tests run in the staged workspace because VF2 group-integrity tests use OR3 manifests;
authoritative CSVs are also readable directly.
Numeric source audits must agree; PDF bytes can depend on fonts and rendering libraries.
The final main manuscript, Table 1/2 source mapping and final Figure 1 remain unverified.

## Online Resources

1. Evidence-grounded Supplement draft with methods, results and literature review; author/main
   manuscript reconciliation is still required.
2. Existing corrected study-level literature coding CSV and bibliography/schema, kept separate
   from experimental evidence.
3. Corrected registries, source tables, figure data, allocation/group identifiers, numeric
   features, labels, scores and provenance. This is the numerical reproducibility basis.

## Original corpora and reconstruction boundaries

Source corpus text/images are not redistributed where doing so could conflict with source
dataset terms. Candidate text, raw OCR/annotations, fitted vocabulary resources and original
model weights are withheld. Obtain FUNSD from https://guillaumejaume.github.io/FUNSD/ and
OCR-D ground truth from https://github.com/OCR-D/gt_structure_text under their own terms.
Acquisition scripts are `scripts/download_funsd.py` and `scripts/download_ocrd_sbb.py`; inspect
the dataset manifests and licence registry first. Do not automatically download or run OCR.

Acquiring corpora alone cannot promise byte-identical reconstruction: original frozen OCR,
generator caches and contextual inputs must match the source hashes. OR3 retains reconstruction
input manifests. No single fully portable corpus-to-final-artifact rebuild has been established
by this preparation. Do not substitute regenerated text for a missing frozen artifact silently.
`paper3_vf2_*` scientific execution is preserved for provenance, but is not a normal replay step.
It can require private/licensed inputs and separate authorization for training or generation.

## Citation and licence

Software metadata is in `CITATION.cff`; the manuscript title is supplied by the author, with
manuscript-author order explicitly left as a placeholder at the author's request. No DOI is assigned.
The known existing repository URL is https://github.com/ksupasate/ocr-risk; no new public release has been pushed.

Author software remains MIT. `THIRD_PARTY_NOTICES.md` records scope and dependency boundaries;
third-party datasets/models keep their own terms. Supplementary-material licensing is pending
author review; CC BY 4.0 must not be applied to third-party source material by assumption.

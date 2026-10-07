# OCR Post-Correction Deployment Evaluation

This repository contains the code, configuration, tests, and frozen result artifacts for the
study "When ranking is not enough: selecting deployment cutoffs for OCR post-correction under
generator and population shift". The study evaluates the relationship between ranking
candidate corrections and selecting a deployment cutoff when the correction generator and the
candidate-site population change.

## Overview

Automatic OCR post-correction proposes replacement text for spans an OCR engine may have
misread. A correction can repair an error, but it can also damage text that was already
correct. The pipeline in this repository separates two decisions:

- **Ranking.** Each correction site may receive several candidate corrections. A
  LambdaMART-style boosted-tree ranker orders the candidates at a site using features computed
  without ground-truth labels.
- **Deployment.** Only the highest-ranked candidate at a site (the *site winner*) enters the
  automatic decision. A cutoff selected from labelled cutoff-selection pages determines
  whether the site winner is applied or the original OCR text is preserved. An applied edit is
  *harmful* when it increases character-level edit distance to the reference transcription.

The ranker is transferred in both directions between a text-only (TXT) correction generator
and a vision-language (VLM) generator, evaluated on scanned forms (FUNSD) and historical
German prints (OCR-D). The analysis reports ranking quality (AUROC), deployment metrics
(application coverage, harmful fraction among applied edits, exact-repair recall), and
sensitivity of the deployment conclusions to a stricter harm definition and to sites shared by
both generators. The full experimental design is described in the manuscript and its
Supplement (Online Resource 1).

## Repository structure

- `src/ocr_risk/` — the research platform: candidate construction, alignment, evidence
  features, ranking, cutoff-selection policies, risk control, metrics, statistics, and
  experiment runners.
- `scripts/` — analysis pipeline for the reported study, figure/table replay, dataset
  acquisition, and release validation scripts.
- `configs/` — YAML configuration for experiments, datasets, engines, and verifiers.
- `tests/` — unit, architecture, leakage, invariants, integration, and smoke tests.
- `docs/` — protocol and methodology documentation, dataset licensing notes, and the
  reproducibility test report.
- `manifests/` — dataset, licence, split, and group manifests used by the pipeline.
- `results/` — synthetic smoke-test outputs and retained generated support files.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The environment is pinned by
`uv.lock`:

```sh
git clone https://github.com/ksupasate/ocr-post-correction-deployment.git
cd ocr-post-correction-deployment
uv sync --locked --extra dev
```

Heavy OCR backends (PaddleOCR, EasyOCR, docTR) and external model APIs are optional extras.
They are not needed for replaying the reported results, and the test suite runs without them.

## Reproducing reported artifacts

The public release reproduces the reported tables and figures from the frozen result
registry. It does not regenerate OCR outputs, correction candidates, or model fits.

Online Resource 3 (numerical reproducibility package) is distributed separately. Extract it
next to this repository, then verify and replay:

```sh
# Verify the frozen registry and package integrity
uv run --locked python scripts/release_artifact_checks.py --resource ../online_resource_3
uv run --locked python scripts/release_artifact_checks.py --manifest ../online_resource_3/MANIFEST.json

# Stage an isolated replay workspace and check the software there
uv run --locked python scripts/stage_release_replay.py \
    --resource ../online_resource_3 --output ../replay
cd ../replay
uv sync --locked --extra dev
make check

# Regenerate Figures 2-4 and the frozen result tables
uv run --locked python scripts/plot_paper3_vf2_figures.py --output-dir rebuild/figures
uv run --locked python scripts/replay_frozen_tables.py \
    --output rebuild/tables/P3_VF2_RESULT_TABLES.md
```

The figure script checks frozen implementation and input hashes and re-renders from the
registered observations; it never fits a model. The table replay checks byte equality against
the frozen result tables. Regenerated PDF bytes can differ in fonts and rendering libraries;
the registered source data is authoritative. Full tests are run in the staged workspace
because the group-integrity fixtures read Online Resource 3 manifests.

A complete corpus-to-results rebuild is not part of the public package: it requires the
original corpora, frozen OCR outputs, and generator caches whose hashes are recorded in the
Online Resource 3 provenance records.

## Tests

```sh
make check   # ruff + ruff format check + strict mypy + pytest with coverage floors
```

The last validated run passed 1,889 tests (46 skipped without optional engines/licensed data)
at 90.2% total coverage, with the original per-package coverage floors enforced as a gate.
`make smoke` exercises the pipeline end to end on synthetic data only.

## Data

The evaluation corpora are public and are not redistributed here:

- **FUNSD** — scanned business forms with word-level annotations
  (https://guillaumejaume.github.io/FUNSD/).
- **OCR-D ground truth** — historical German prints with diplomatic transcriptions
  (https://github.com/OCR-D/gt_structure_text).

Acquisition helpers are provided (`scripts/download_funsd.py`,
`scripts/download_ocrd_sbb.py`). They verify sha256 checksums against `manifests/datasets/`
before use. Dataset licences are recorded in `manifests/licenses/registry.yaml` and
`docs/data_licensing.md`; follow the original dataset terms. The reproducibility package
contains page identifiers, numeric features, labels, and scores — not corpus text or images.

## Online Resources

- **Online Resource 1** — Supplement: methods, extended results, prompts, and the extended
  literature review.
- **Online Resource 2** — coded literature matrix (study-level coding of the reviewed work).
- **Online Resource 3** — numerical registry, figure/table source data, allocation and group
  manifests, and provenance records backing the replay commands above.

## Citation

See `CITATION.cff`. If you use this software, cite the study:

> Supasate Vorathammathorn, Wassana Sintarasirikulchai, Theerat Sakdejayont, Sooksan
> Panichpapiboon. *When ranking is not enough: selecting deployment cutoffs for OCR
> post-correction under generator and population shift.*

A DOI will be added once the manuscript and archival deposits are published.

## Licence

Source code in this repository is released under the MIT License. Datasets, model weights,
and third-party materials remain under their own terms; see `THIRD_PARTY_NOTICES.md` for the
dependency and scope boundaries.

## Related article

"When ranking is not enough: selecting deployment cutoffs for OCR post-correction under
generator and population shift" — manuscript prepared for submission to the International
Journal on Document Analysis and Recognition (IJDAR). The Supplement and reproducibility
package are supplied as Online Resources 1–3.

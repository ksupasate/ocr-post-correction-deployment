# Release notes

## v1.0.0

Initial public release of the software associated with "When ranking is not enough: selecting
deployment cutoffs for OCR post-correction under generator and population shift".

Includes the `ocr_risk` implementation (candidate construction, alignment, evidence features,
ranking, cutoff-selection policies, metrics, statistics), the analysis pipeline behind the
reported results, the locked environment, configuration files, prompts and parsers, the test
suite with coverage gates, and the frozen-artifact replay and validation scripts.

The reported figures and tables are reproduced from the frozen numerical registry in Online
Resource 3; the release does not regenerate OCR outputs, correction candidates, or model fits.
Source corpus images and text are not redistributed; FUNSD and OCR-D are obtained from their
original sources under their own terms.

The frozen Python package version remains 0.1.0; v1.0.0 is the artifact release version.

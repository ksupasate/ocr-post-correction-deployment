UV ?= uv
RUN := $(UV) run
SMOKE_CONFIG ?= configs/experiments/smoke_synthetic.yaml
RESULTS_ROOT ?= results/generated

.PHONY: help setup lint fmt type test test-fast cov smoke smoke-clean audit check clean env-mirrors

help:
	@echo "setup        Create the environment from uv.lock (no conda, no GPU, no downloads)"
	@echo "lint         ruff check"
	@echo "fmt          ruff format + ruff check --fix"
	@echo "type         mypy (strict) over src/ocr_risk"
	@echo "test         Full pytest suite, then the coverage floors as a hard gate"
	@echo "test-fast    Unit + architecture + leakage tests only"
	@echo "smoke        Synthetic end-to-end research pipeline -> $(RESULTS_ROOT)"
	@echo "determinism  Re-run the smoke experiment and compare output digests"
	@echo "audit        Leakage + licenses + runtime manifest + docs integrity"
	@echo "check        lint + type + test (the pre-commit gate)"
	@echo "env-mirrors  Regenerate environment.yml / requirements.txt from uv.lock"

setup:
	$(UV) sync --extra dev

lint:
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

fmt:
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

type:
	$(RUN) mypy

# The coverage floors are a GATE, not a report. coverage.py enforces only a global
# fail_under, so the per-package floors go through scripts/check_coverage.py, which reads
# them from [tool.ocr_risk.coverage] in pyproject.toml -- the same numbers the docs quote.
test:
	$(RUN) pytest --cov --cov-report=term-missing
	$(RUN) python scripts/check_coverage.py

test-fast:
	$(RUN) pytest tests/unit tests/architecture tests/leakage tests/invariants

cov:
	$(RUN) pytest --cov --cov-report=html
	@echo "open htmlcov/index.html"

smoke:
	$(RUN) ocr-risk experiment run --config $(SMOKE_CONFIG)
	$(RUN) ocr-risk experiment status --experiment smoke_synthetic
	$(RUN) ocr-risk analyze report --experiment smoke_synthetic --out $(RESULTS_ROOT)
	$(RUN) ocr-risk gate pilot --experiment smoke_synthetic
	$(RUN) ocr-risk provenance $(RESULTS_ROOT)/figure_manifest.json

smoke-clean:
	rm -rf artifacts/* cache/* $(RESULTS_ROOT)/*
	$(MAKE) smoke

# Determinism needs two runs of the same experiment to compare; a single run reports
# NOT COMPARED rather than passing vacuously.
determinism:
	$(RUN) ocr-risk experiment run --config $(SMOKE_CONFIG)
	$(RUN) ocr-risk audit determinism --experiment smoke_synthetic

audit:
	$(RUN) ocr-risk audit leakage --experiment smoke_synthetic
	$(RUN) ocr-risk audit licenses
	$(RUN) ocr-risk audit docs
	$(RUN) python scripts/validate_runtime_manifest.py

check: lint type test

env-mirrors:
	$(RUN) python scripts/export_env_mirrors.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Slim makefile for NGF-alpha

UV ?= uv
VENV ?= .venv
PYTHON := $(VENV)/bin/python

SAMPLES ?= 50
OUT_JSON ?= summary.json
OUT_CSV ?= results.csv
LOG ?= INFO

.PHONY: init venv sync bench nb kernel

init: venv sync ## Create venv and install pinned deps

venv: ## Create virtual environment with uv
	$(UV) venv $(VENV)

sync: ## Sync environment from pyproject.lock
	$(UV) sync

bench: ## Run Stage 11 benchmark with defaults (override via vars)
	$(PYTHON) arc-benchmark-latest.py \
	  --samples $(SAMPLES) --log $(LOG) \
	  --out_json $(OUT_JSON) --out_csv $(OUT_CSV)

nb: ## Launch Jupyter Notebook from venv
	$(VENV)/bin/jupyter notebook

kernel: ## Register Jupyter kernel named ngf-alpha
	$(PYTHON) -m ipykernel install --user --name ngf-alpha


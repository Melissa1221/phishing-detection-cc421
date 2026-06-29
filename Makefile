PYTHON = .venv/Scripts/python
PIP    = .venv/Scripts/pip

.PHONY: run run-sample run-no-cv evaluate cross-eval test

run:
	$(PYTHON) -m src.run_pipeline

run-sample:
	$(PYTHON) -m src.run_pipeline --sample 2000

run-no-cv:
	$(PYTHON) -m src.run_pipeline --no-cv

cross-eval:
	$(PYTHON) -m src.cross_eval

test:
	$(PYTHON) -m pytest tests/ -v
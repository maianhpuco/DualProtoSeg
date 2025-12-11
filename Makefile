
PYTHON ?= python3
CONFIG ?= config.yaml
GPU ?= 0

.PHONY: run help

run:
	$(PYTHON) train.py --config $(CONFIG) --gpu $(GPU)

help:
	@echo "Usage: make run [CONFIG=path] [GPU=id]"
	@echo "Defaults: CONFIG=config.yaml GPU=0"


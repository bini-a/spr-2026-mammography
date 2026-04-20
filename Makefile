UV := $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)

.PHONY: help install run train predict rerun compare clean

CONFIG ?=
EXP    ?=

help:
	@echo ""
	@echo "SPR 2026 — Experiment Commands"
	@echo "================================"
	@echo "  make install                              Install dependencies"
	@echo "  make run     CONFIG=configs/expNNN.yaml   Train + predict"
	@echo "  make train   CONFIG=configs/expNNN.yaml   Train only"
	@echo "  make predict CONFIG=configs/expNNN.yaml   Predict only"
	@echo "  make rerun   EXP=exp001_tfidf_logreg      Re-run from saved config"
	@echo "  make compare                              Show all results ranked by OOF F1"
	@echo "  make clean                                Remove __pycache__ files"
	@echo ""

install:
	$(UV) sync

run:
	@[ -n "$(CONFIG)" ] || (echo "Usage: make run CONFIG=configs/<name>.yaml" && exit 1)
	$(UV) run python run.py $(CONFIG)

train:
	@[ -n "$(CONFIG)" ] || (echo "Usage: make train CONFIG=configs/<name>.yaml" && exit 1)
	$(UV) run python run.py $(CONFIG) --train

predict:
	@[ -n "$(CONFIG)" ] || (echo "Usage: make predict CONFIG=configs/<name>.yaml" && exit 1)
	$(UV) run python run.py $(CONFIG) --predict

rerun:
	@[ -n "$(EXP)" ] || (echo "Usage: make rerun EXP=exp001_tfidf_logreg" && exit 1)
	$(UV) run python run.py --rerun $(EXP) --train

compare:
	$(UV) run python run.py --compare

compare-full:
	$(UV) run python run.py --compare --full

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

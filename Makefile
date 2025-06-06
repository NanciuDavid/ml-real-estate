# ML Real Estate Project Makefile
.PHONY: help install clean train evaluate test jupyter

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  clean       - Clean generated files"
	@echo "  train       - Train all models"
	@echo "  evaluate    - Evaluate and compare models"
	@echo "  test        - Run unit tests"
	@echo "  jupyter     - Start Jupyter notebook server"
	@echo "  setup-env   - Setup virtual environment"

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf results/models/*.pkl
	rm -rf results/plots/*.png
	rm -rf results/reports/*.html

train:
	python scripts/train_all_models.py

evaluate:
	python scripts/evaluate_models.py

test:
	python -m pytest tests/ -v

jupyter:
	jupyter notebook notebooks/

setup-env:
	python -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate" 
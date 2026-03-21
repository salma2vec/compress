.PHONY: test lint benchmark install clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check compress/ tests/

benchmark:
	python benchmarks/gpt2_compression.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build

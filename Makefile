.PHONY: lint typecheck test test-verbose

lint:
	ruff check .

typecheck:
	basedpyright

test:
	pytest -q

test-verbose:
	pytest -v

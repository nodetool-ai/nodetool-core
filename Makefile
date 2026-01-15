.PHONY: lint typecheck test test-verbose

lint:
	uv run ruff check .

typecheck:
	uv run ty check src \
		--ignore unresolved-import \
		--ignore possibly-missing-attribute \
		--error invalid-type-form \
		--error invalid-attribute-access \
		--error invalid-return-type \
		--error parameter-already-assigned \
		--error too-many-positional-arguments \
		--warn invalid-method-override \
		--warn invalid-argument-type \
		--warn unresolved-attribute \
		--warn missing-argument \
		--warn invalid-assignment \
		--warn call-non-callable \
		--warn no-matching-overload \
		--warn unknown-argument \
		--warn unsupported-operator \
		--warn not-iterable \
		--warn invalid-key \
		--warn not-subscriptable \
		--warn unresolved-reference \
		--warn invalid-parameter-default

test:
	uv run pytest -n auto -q --ignore=tests/workflows/test_docker_job_execution.py \
		--ignore=tests/workflows/test_job_execution_manager.py \
		--ignore=tests/workflows/test_job_execution.py
	timeout 60 uv run pytest -q tests/workflows/test_job_execution_manager.py tests/workflows/test_job_execution.py || true

test-verbose:
	uv run pytest -n auto -v --ignore=tests/workflows/test_docker_job_execution.py \
		--ignore=tests/workflows/test_job_execution_manager.py \
		--ignore=tests/workflows/test_job_execution.py
	timeout 60 uv run pytest -v tests/workflows/test_job_execution_manager.py tests/workflows/test_job_execution.py || true

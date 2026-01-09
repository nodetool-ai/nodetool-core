.PHONY: lint typecheck test test-verbose

lint:
	ruff check .

typecheck:
	pyright src \
	  --warn unresolved-import \
	  --warn invalid-method-override \
	  --warn invalid-argument-type \
	  --warn unresolved-attribute \
	  --warn invalid-attribute-access \
	  --warn missing-argument \
	  --warn invalid-type-form \
	  --warn invalid-return-type \
	  --warn invalid-assignment \
	  --warn parameter-already-assigned \
	  --warn call-non-callable \
	  --warn no-matching-overload \
	  --warn unknown-argument \
	  --warn unsupported-operator \
	  --warn not-iterable \
	  --warn invalid-key \
	  --warn non-subscriptable \
	  --warn unresolved-reference \
	  --warn too-many-positional-arguments \
	  --warn invalid-parameter-default \
	  --ignore possibly-missing-attribute

test:
	pytest -q

test-verbose:
	pytest -v

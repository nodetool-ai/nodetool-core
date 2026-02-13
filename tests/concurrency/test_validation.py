"""
Tests for async validation utilities.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from nodetool.concurrency.validation import (
    AnyValidatorComposer,
    AsyncValidator,
    ConditionalValidator,
    ValidationError,
    ValidationResult,
    ValidatorComposer,
    validate_async,
    validate_with_retries,
)


class SimpleValidator(AsyncValidator):
    """A simple validator that checks if a value is a string."""

    async def validate_async(self, value: Any) -> ValidationResult:
        if isinstance(value, str):
            return ValidationResult(is_valid=True)
        return ValidationResult(is_valid=False, errors=[f"Expected string, got {type(value).__name__}"])


class LengthValidator(AsyncValidator):
    """Validates string length."""

    def __init__(self, min_length: int = 0, max_length: int = 100):
        self.min_length = min_length
        self.max_length = max_length

    async def validate_async(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(is_valid=False, errors=["Value must be a string"])

        length = len(value)
        if length < self.min_length:
            return ValidationResult(
                is_valid=False,
                errors=[f"String too short: {length} < {self.min_length}"],
            )
        if length > self.max_length:
            return ValidationResult(
                is_valid=False,
                errors=[f"String too long: {length} > {self.max_length}"],
            )

        return ValidationResult(is_valid=True)


class AsyncURLValidator(AsyncValidator):
    """Simulates an async URL validator with network call."""

    def __init__(self, fail_attempts: int = 0):
        self.fail_attempts = fail_attempts
        self.attempts = 0

    async def validate_async(self, value: Any) -> ValidationResult:
        await asyncio.sleep(0.01)  # Simulate network delay

        if not isinstance(value, str):
            return ValidationResult(is_valid=False, errors=["Value must be a string"])

        self.attempts += 1
        if self.attempts <= self.fail_attempts:
            raise ConnectionError("Simulated network failure")

        if not value.startswith(("http://", "https://")):
            return ValidationResult(is_valid=False, errors=["Invalid URL format"])

        return ValidationResult(is_valid=True)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_add_error(self) -> None:
        """Test adding an error marks result as invalid."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid

        result.add_error("Test error")
        assert not result.is_valid
        assert "Test error" in result.errors

    def test_add_warning(self) -> None:
        """Test adding a warning doesn't affect validity."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid
        assert "Test warning" in result.warnings

    def test_merge(self) -> None:
        """Test merging two results."""
        result1 = ValidationResult(is_valid=True, warnings=["Warning 1"])
        result2 = ValidationResult(is_valid=False, errors=["Error 1"], warnings=["Warning 2"])

        result1.merge(result2)

        assert not result1.is_valid
        assert "Error 1" in result1.errors
        assert "Warning 1" in result1.warnings
        assert "Warning 2" in result1.warnings


class TestAsyncValidator:
    """Tests for AsyncValidator base class."""

    @pytest.mark.asyncio
    async def test_simple_validator_pass(self) -> None:
        """Test simple validator passes for valid input."""
        validator = SimpleValidator()
        result = await validator.validate_async("hello")
        assert result.is_valid
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_simple_validator_fail(self) -> None:
        """Test simple validator fails for invalid input."""
        validator = SimpleValidator()
        result = await validator.validate_async(123)
        assert not result.is_valid
        assert "Expected string" in result.errors[0]

    @pytest.mark.asyncio
    async def test_callable_interface(self) -> None:
        """Test validator can be called like a function."""
        validator = SimpleValidator()
        coro = validator("test")  # Returns a coroutine
        result = await coro
        assert result.is_valid


class TestValidateAsync:
    """Tests for validate_async function."""

    @pytest.mark.asyncio
    async def test_multiple_validators_all_pass(self) -> None:
        """Test all validators passing."""
        result = await validate_async(
            "hello",
            [SimpleValidator(), LengthValidator(min_length=1, max_length=10)],
        )
        assert result.is_valid
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_multiple_validators_one_fails(self) -> None:
        """Test one validator failing."""
        result = await validate_async(
            "x" * 150,
            [SimpleValidator(), LengthValidator(max_length=100)],
        )
        assert not result.is_valid
        assert "too long" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_stop_on_first_error(self) -> None:
        """Test stopping on first error."""
        result = await validate_async(
            123,
            [SimpleValidator(), LengthValidator(max_length=10)],
            stop_on_first_error=True,
        )
        assert not result.is_valid
        # Should only have error from first validator
        assert len(result.errors) == 1
        assert "Expected string" in result.errors[0]

    @pytest.mark.asyncio
    async def test_function_validators(self) -> None:
        """Test using callable functions as validators."""

        async def custom_validator(value: Any) -> ValidationResult:
            return ValidationResult(is_valid=isinstance(value, str) and len(value) > 0)

        result = await validate_async("test", [custom_validator])
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_aggregates_warnings(self) -> None:
        """Test warnings are aggregated from all validators."""

        class WarningValidator(AsyncValidator):
            async def validate_async(self, value: Any) -> ValidationResult:
                return ValidationResult(
                    is_valid=True, warnings=[f"Warning for {repr(value)}"]
                )

        result = await validate_async("test", [SimpleValidator(), WarningValidator()])
        assert result.is_valid
        assert len(result.warnings) == 1


class TestValidateWithRetries:
    """Tests for validate_with_retries function."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        """Test validator succeeding immediately."""
        validator = SimpleValidator()
        result = await validate_with_retries("hello", validator)
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self) -> None:
        """Test retries on transient failures."""
        validator = AsyncURLValidator(fail_attempts=2)
        result = await validate_with_retries("https://example.com", validator, max_retries=3)
        assert result.is_valid
        assert validator.attempts == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self) -> None:
        """Test failure after exhausting retries."""
        validator = AsyncURLValidator(fail_attempts=10)
        result = await validate_with_retries("https://example.com", validator, max_retries=3)
        assert not result.is_valid
        assert "failed after 3 attempts" in result.errors[0]

    @pytest.mark.asyncio
    async def test_custom_delay(self) -> None:
        """Test custom retry delay."""
        validator = AsyncURLValidator(fail_attempts=1)
        import time

        start = time.time()
        result = await validate_with_retries(
            "https://example.com", validator, max_retries=2, delay=0.1
        )
        elapsed = time.time() - start

        assert result.is_valid
        assert elapsed >= 0.1  # At least one delay occurred


class TestValidatorComposer:
    """Tests for ValidatorComposer class."""

    @pytest.mark.asyncio
    async def test_all_of_composition(self) -> None:
        """Test AND composition with all_of."""
        composer = ValidatorComposer.all_of(
            [SimpleValidator(), LengthValidator(min_length=3, max_length=10)]
        )
        result = await composer.validate_async("hello")
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_all_of_one_fails(self) -> None:
        """Test AND composition with one failure."""
        composer = ValidatorComposer.all_of(
            [SimpleValidator(), LengthValidator(min_length=10)]
        )
        result = await composer.validate_async("hi")
        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_and_operator(self) -> None:
        """Test using & operator for composition."""
        composer = ValidatorComposer.all_of([SimpleValidator()]) & LengthValidator(
            max_length=10
        )
        result = await composer.validate_async("test")
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_any_of_composition(self) -> None:
        """Test OR composition with any_of."""
        composer = ValidatorComposer.any_of(
            [
                SimpleValidator(),  # Would pass
                LengthValidator(min_length=100),  # Would fail
            ]
        )
        result = await composer.validate_async("test")
        assert result.is_valid


class TestAnyValidatorComposer:
    """Tests for AnyValidatorComposer (OR logic)."""

    @pytest.mark.asyncio
    async def test_one_passes_succeeds(self) -> None:
        """Test succeeding when one validator passes."""

        class AlwaysFailValidator(AsyncValidator):
            async def validate_async(self, value: Any) -> ValidationResult:
                return ValidationResult(is_valid=False, errors=["Always fails"])

        composer = AnyValidatorComposer([SimpleValidator(), AlwaysFailValidator()])
        result = await composer.validate_async("test")
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_all_fail(self) -> None:
        """Test failure when all validators fail."""

        class FailValidator(AsyncValidator):
            def __init__(self, name: str):
                self.name = name

            async def validate_async(self, value: Any) -> ValidationResult:
                return ValidationResult(is_valid=False, errors=[f"{self.name} failed"])

        composer = AnyValidatorComposer([FailValidator("A"), FailValidator("B")])
        result = await composer.validate_async(123)
        assert not result.is_valid
        assert len(result.errors) == 2
        assert "A failed" in result.errors
        assert "B failed" in result.errors


class TestConditionalValidator:
    """Tests for ConditionalValidator."""

    @pytest.mark.asyncio
    async def test_predicate_true_validates(self) -> None:
        """Test validation when predicate returns True."""
        validator = ConditionalValidator(
            predicate=lambda v: isinstance(v, str) and v.startswith("http"),
            validator=AsyncURLValidator(),
        )
        result = await validator.validate_async("https://example.com")
        assert result.is_valid
        assert result.context.get("conditional") is None  # Was validated, not skipped

    @pytest.mark.asyncio
    async def test_predicate_false_skips(self) -> None:
        """Test skipping validation when predicate returns False."""
        validator = ConditionalValidator(
            predicate=lambda v: isinstance(v, str) and v.startswith("http"),
            validator=AsyncURLValidator(),
        )
        result = await validator.validate_async("ftp://example.com")
        assert result.is_valid
        assert result.context.get("conditional") == "skipped"

    @pytest.mark.asyncio
    async def test_async_predicate(self) -> None:
        """Test with async predicate."""

        async def async_pred(value: Any) -> bool:
            await asyncio.sleep(0.01)
            return isinstance(value, str)

        validator = ConditionalValidator(
            predicate=async_pred, validator=SimpleValidator()
        )
        result = await validator.validate_async("test")
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_predicate_exception(self) -> None:
        """Test handling exceptions in predicate."""

        def failing_pred(value: Any) -> bool:
            raise ValueError("Predicate failed")

        validator = ConditionalValidator(
            predicate=failing_pred, validator=SimpleValidator()
        )
        result = await validator.validate_async("test")
        assert not result.is_valid
        assert "Conditional validation failed" in result.errors[0]


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_string_representation(self) -> None:
        """Test error message formatting."""
        error = ValidationError(
            value="test",
            validator_name="TestValidator",
            message="Validation failed",
            context={"field": "username"},
        )
        error_str = str(error)
        assert "TestValidator" in error_str
        assert "Validation failed" in error_str
        assert "context:" in error_str

    def test_string_representation_no_context(self) -> None:
        """Test error message without context."""
        error = ValidationError(
            value=123, validator_name="IntValidator", message="Not an integer"
        )
        error_str = str(error)
        assert "IntValidator" in error_str
        assert "Not an integer" in error_str
        assert "context:" not in error_str

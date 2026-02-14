"""
Async validation utilities for nodetool.

This module provides asynchronous validation patterns for use across nodes,
workflows, and providers. It fills the gap between synchronous type checking
and the async-first architecture of nodetool-core.
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


@dataclass
class ValidationError(Exception):
    """Raised when async validation fails."""

    value: Any
    validator_name: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        ctx_str = f" (context: {self.context})" if self.context else ""
        return f"Validation failed for '{self.validator_name}': {self.message}{ctx_str}"


@dataclass
class ValidationResult:
    """Result of an async validation operation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message to this result."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message to this result."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.context.update(other.context)
        return self


class AsyncValidator:
    """
    Base class for async validators.

    Subclasses should implement the validate_async method to provide
    custom async validation logic.

    Example:
        class URLValidator(AsyncValidator):
            async def validate_async(self, value: Any) -> ValidationResult:
                if not isinstance(value, str):
                    return ValidationResult(False, errors=["Value must be a string"])

                # Async URL check
                is_valid_url = await check_url_async(value)
                if not is_valid_url:
                    return ValidationResult(False, errors=["Invalid URL format"])

                return ValidationResult(is_valid=True)
    """

    async def validate_async(self, value: Any) -> ValidationResult:
        """
        Validate a value asynchronously.

        Args:
            value: The value to validate.

        Returns:
            A ValidationResult indicating success or failure with details.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate_async()"
        )

    def __call__(self, value: Any) -> Coroutine[Any, Any, ValidationResult]:
        """
        Allow validator instances to be called like functions.

        Returns a coroutine that must be awaited.
        """
        return self.validate_async(value)


async def validate_async(
    value: Any,
    validators: list[AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]]],
    stop_on_first_error: bool = False,
) -> ValidationResult:
    """
    Run multiple async validators on a value.

    This function executes all provided validators (or stops early on error
    if configured) and aggregates their results.

    Args:
        value: The value to validate.
        validators: List of async validators (instances or callables returning coroutines).
        stop_on_first_error: If True, stop validation on first error (default: False).

    Returns:
        A combined ValidationResult from all validators.

    Example:
        result = await validate_async(
            user_input,
            [
                StringValidator(),
                LengthValidator(min=1, max=100),
                URLValidator(),
            ],
            stop_on_first_error=True,
        )
        if not result.is_valid:
            print(f"Validation failed: {result.errors}")
    """
    result = ValidationResult(is_valid=True)

    for validator in validators:
        try:
            if isinstance(validator, AsyncValidator):
                validator_result = await validator.validate_async(value)
            else:
                validator_result = await validator(value)

            if not validator_result.is_valid:
                if stop_on_first_error:
                    return validator_result
                result.merge(validator_result)
            else:
                # Merge warnings and context even if valid
                result.warnings.extend(validator_result.warnings)
                result.context.update(validator_result.context)
        except Exception as e:
            error_msg = f"Validator {validator.__class__.__name__ if hasattr(validator, '__class__') else 'function'} raised exception: {e}"
            log.error(error_msg, exc_info=True)
            result.add_error(error_msg)
            if stop_on_first_error:
                break

    return result


async def validate_with_retries(
    value: Any,
    validator: AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]],
    max_retries: int = 3,
    delay: float = 0.5,
) -> ValidationResult:
    """
    Validate a value with retries on transient failures.

    Useful for validators that make external calls (e.g., URL checks, API validations)
    that may fail transiently.

    Args:
        value: The value to validate.
        validator: The async validator to run.
        max_retries: Maximum number of retry attempts (default: 3).
        delay: Delay between retries in seconds (default: 0.5).

    Returns:
        ValidationResult from the validator, or failed result if all retries exhausted.

    Example:
        result = await validate_with_retries(
            url,
            URLValidator(),
            max_retries=5,
            delay=1.0,
        )
    """
    import asyncio

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            if isinstance(validator, AsyncValidator):
                return await validator.validate_async(value)
            else:
                return await validator(value)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                log.warning(
                    f"Validation attempt {attempt + 1} failed, retrying in {delay}s: {e}",
                    extra={"attempt": attempt + 1, "max_retries": max_retries},
                )
                await asyncio.sleep(delay)

    # All retries exhausted
    error_msg = f"Validation failed after {max_retries} attempts: {last_error}"
    log.error(error_msg)
    return ValidationResult(
        is_valid=False,
        errors=[error_msg],
    )


class ValidatorComposer:
    """
    Compose multiple validators with logical operators.

    Provides fluent API for combining validators with AND/OR logic.

    Example:
        validator = (
            ValidatorComposer.all_of([
                StringValidator(),
                LengthValidator(min=1, max=100),
            ])
            & URLValidator()
        )

        result = await validator.validate_async(value)
    """

    def __init__(
        self,
        validators: list[
            AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]]
        ],
    ) -> None:
        self.validators = validators

    async def validate_async(self, value: Any) -> ValidationResult:
        """Validate using all composed validators."""
        return await validate_async(value, self.validators, stop_on_first_error=False)

    @classmethod
    def all_of(
        cls,
        validators: list[
            AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]]
        ],
    ) -> "ValidatorComposer":
        """Create a composer that requires all validators to pass (AND logic)."""
        return cls(validators)

    @classmethod
    def any_of(
        cls,
        validators: list[
            AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]]
        ],
    ) -> "AnyValidatorComposer":
        """Create a composer that requires at least one validator to pass (OR logic)."""
        return AnyValidatorComposer(validators)

    def __and__(
        self, other: AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]]
    ) -> "ValidatorComposer":
        """Combine with another validator using AND logic (& operator)."""
        if isinstance(other, ValidatorComposer):
            return ValidatorComposer(self.validators + other.validators)
        else:
            return ValidatorComposer([*self.validators, other])


class AnyValidatorComposer(AsyncValidator):
    """
    Composer that requires at least one validator to pass (OR logic).

    Example:
        validator = ValidatorComposer.any_of([
            URLValidator(),
            EmailValidator(),
            FilePathValidator(),
        ])

        result = await validator.validate_async(value)
    """

    def __init__(
        self,
        validators: list[
            AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]]
        ],
    ) -> None:
        self.validators = validators

    async def validate_async(self, value: Any) -> ValidationResult:
        """Validate using OR logic - at least one validator must pass."""
        results: list[ValidationResult] = []

        for validator in self.validators:
            try:
                if isinstance(validator, AsyncValidator):
                    result = await validator.validate_async(value)
                else:
                    result = await validator(value)
                results.append(result)

                if result.is_valid:
                    # At least one passed
                    all_warnings = [r for res in results for r in res.warnings]
                    return ValidationResult(
                        is_valid=True,
                        warnings=all_warnings,
                        context={"validator": "any_of", "attempted": len(results)},
                    )
            except Exception as e:
                log.warning(
                    f"Validator in any_of raised exception: {e}",
                    extra={"validator": validator.__class__.__name__ if hasattr(validator, '__class__') else 'function'},
                )
                results.append(
                    ValidationResult(is_valid=False, errors=[f"Exception: {e}"])
                )

        # All validators failed
        all_errors = [err for res in results for err in res.errors]
        return ValidationResult(
            is_valid=False,
            errors=all_errors,
            context={"validator": "any_of", "attempted": len(results)},
        )


class ConditionalValidator(AsyncValidator):
    """
    Validator that conditionally applies validation based on a predicate.

    Useful for conditional validation logic based on value properties or external state.

    Example:
        validator = ConditionalValidator(
            predicate=lambda v: isinstance(v, str) and v.startswith("http"),
            validator=URLValidator(),
        )

        # Only validates if value starts with "http"
        result = await validator.validate_async(value)
    """

    def __init__(
        self,
        predicate: Callable[[Any], bool | Coroutine[Any, Any, bool]],
        validator: AsyncValidator | Callable[[Any], Coroutine[Any, Any, ValidationResult]],
    ) -> None:
        """
        Initialize a conditional validator.

        Args:
            predicate: Function or async function that returns True if validation should be applied.
            validator: The validator to apply if predicate returns True.
        """
        self.predicate = predicate
        self.validator = validator

    async def validate_async(self, value: Any) -> ValidationResult:
        """Validate only if predicate returns True."""
        try:
            # Check predicate (sync or async)
            pred_result = self.predicate(value)
            should_validate = (
                await pred_result if isinstance(pred_result, Coroutine) else pred_result
            )  # type: ignore

            if not should_validate:
                return ValidationResult(
                    is_valid=True,
                    context={"conditional": "skipped", "reason": "predicate returned False"},
                )

            # Apply validator
            if isinstance(self.validator, AsyncValidator):
                return await self.validator.validate_async(value)
            else:
                return await self.validator(value)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Conditional validation failed: {e}"],
                context={"predicate": str(self.predicate)},
            )


__all__ = [
    "AnyValidatorComposer",
    "AsyncValidator",
    "ConditionalValidator",
    "ValidationError",
    "ValidationResult",
    "ValidatorComposer",
    "validate_async",
    "validate_with_retries",
]

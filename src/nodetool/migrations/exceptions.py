"""
Exception classes for the migration system.

Provides specific exception types for different migration failure scenarios.
"""


class MigrationError(Exception):
    """Base exception for migration-related errors."""

    def __init__(self, message: str, migration_version: str | None = None):
        self.migration_version = migration_version
        super().__init__(message)


class LockError(MigrationError):
    """Raised when migration lock cannot be acquired or released."""

    pass


class ChecksumError(MigrationError):
    """Raised when migration checksum validation fails."""

    def __init__(
        self,
        message: str,
        migration_version: str,
        expected_checksum: str,
        actual_checksum: str,
    ):
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        super().__init__(message, migration_version)


class BaselineError(MigrationError):
    """Raised when baselining fails."""

    pass


class MigrationDiscoveryError(MigrationError):
    """Raised when migration discovery fails."""

    pass


class RollbackError(MigrationError):
    """Raised when migration rollback fails."""

    pass

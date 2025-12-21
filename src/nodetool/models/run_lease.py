"""
RunLease model for workflow execution concurrency control.

This model implements lease-based locking to ensure only one worker
processes a workflow run at a time in a distributed system.
"""

from datetime import datetime, timedelta
from typing import Optional

from nodetool.models.base_model import DBField, DBIndex, DBModel
from nodetool.models.condition_builder import Field


@DBIndex(columns=["expires_at"], name="idx_run_leases_expires")
class RunLease(DBModel):
    """
    Represents a lease on a workflow run for exclusive execution.
    
    Leases prevent multiple workers from processing the same run concurrently.
    They automatically expire after a TTL, allowing recovery if a worker crashes.
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "run_leases",
            "primary_key": "run_id",
        }

    run_id: str = DBField(hash_key=True)
    worker_id: str = DBField()
    acquired_at: datetime = DBField(default_factory=datetime.now)
    expires_at: datetime = DBField()

    @classmethod
    async def acquire(
        cls, run_id: str, worker_id: str, ttl_seconds: int = 60
    ) -> Optional["RunLease"]:
        """
        Acquire a lease on a run.
        
        Args:
            run_id: The workflow run identifier
            worker_id: Identifier for this worker (e.g., hostname, PID)
            ttl_seconds: Time-to-live for the lease in seconds
            
        Returns:
            RunLease if acquired, None if run is already leased to another worker
        """
        now = datetime.now()
        expires = now + timedelta(seconds=ttl_seconds)
        
        # Check for existing lease
        existing = await cls.get(run_id)
        
        if existing:
            # Check if expired
            if existing.expires_at < now:
                # Expired - take over
                existing.worker_id = worker_id
                existing.acquired_at = now
                existing.expires_at = expires
                await existing.save()
                return existing
            else:
                # Still held by another worker
                return None
        
        # No existing lease - create new one
        lease = cls(
            run_id=run_id,
            worker_id=worker_id,
            acquired_at=now,
            expires_at=expires,
        )
        await lease.save()
        return lease

    async def renew(self, ttl_seconds: int = 60) -> None:
        """
        Renew this lease to extend its expiration time.
        
        Args:
            ttl_seconds: Time-to-live for the renewed lease in seconds
        """
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        await self.save()

    async def release(self) -> None:
        """
        Release this lease, allowing other workers to acquire it.
        """
        await self.delete()

    def is_expired(self) -> bool:
        """
        Check if this lease has expired.
        
        Returns:
            True if the lease has expired
        """
        return self.expires_at < datetime.now()

    def is_held_by(self, worker_id: str) -> bool:
        """
        Check if this lease is held by a specific worker.
        
        Args:
            worker_id: Worker identifier to check
            
        Returns:
            True if this lease is held by the specified worker
        """
        return self.worker_id == worker_id and not self.is_expired()

    @classmethod
    async def cleanup_expired(cls) -> int:
        """
        Remove all expired leases from the database.
        
        Returns:
            Number of leases removed
        """
        adapter = await cls.adapter()
        now = datetime.now()
        
        # Query for expired leases
        results, _ = await adapter.query(
            condition=Field("expires_at").less_than(now),
            limit=1000,
        )
        
        # Delete each expired lease
        count = 0
        for row in results:
            await cls.delete(row["run_id"])
            count += 1
        
        return count

    @classmethod
    def from_dict(cls, data: dict) -> "RunLease":
        """
        Create RunLease instance from dictionary.
        
        Args:
            data: Dictionary containing lease data
            
        Returns:
            RunLease instance
        """
        return cls(
            run_id=data["run_id"],
            worker_id=data["worker_id"],
            acquired_at=data.get("acquired_at", datetime.now()),
            expires_at=data["expires_at"],
        )

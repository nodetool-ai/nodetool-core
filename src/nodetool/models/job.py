from datetime import datetime
from typing import Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.base_model import DBField, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field

log = get_logger(__name__)


class Job(DBModel):
    @classmethod
    def get_table_schema(cls):
        return {"table_name": "nodetool_jobs"}

    id: str = DBField()
    user_id: str = DBField(default="")
    job_type: str = DBField(default="")
    workflow_id: str = DBField(default="")
    started_at: datetime = DBField(default_factory=datetime.now)
    finished_at: datetime | None = DBField(default=None)
    graph: dict = DBField(default_factory=dict)
    params: dict = DBField(default_factory=dict)
    error: str | None = DBField(default=None)
    cost: float | None = DBField(default=None)
    logs: list[dict] | None = DBField(default=None)

    @classmethod
    async def find(cls, user_id: str, job_id: str):
        job = await cls.get(job_id)
        return job if job and job.user_id == user_id else None

    @classmethod
    async def create(cls, workflow_id: str, user_id: str, **kwargs):  # type: ignore[override]
        return await super().create(
            id=create_time_ordered_uuid(),
            workflow_id=workflow_id,
            user_id=user_id,
            **kwargs,
        )

    @classmethod
    async def paginate(
        cls,
        user_id: str,
        workflow_id: str | None = None,
        limit: int = 10,
        start_key: str | None = None,
    ):
        if workflow_id:
            items, key = await cls.query(
                Field("workflow_id").equals(workflow_id).and_(Field("id").greater_than(start_key or "")),
                limit=limit,
            )
            return items, key
        elif user_id:
            items, key = await cls.query(
                Field("user_id").equals(user_id).and_(Field("id").greater_than(start_key or "")),
                limit=limit,
            )
            return items, key
        else:
            raise ValueError("Must provide either user_id or workflow_id")

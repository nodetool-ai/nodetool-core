import pytest

from nodetool.models.job import Job
from tests.conftest import make_job


@pytest.mark.asyncio
async def test_find_job(user_id: str):
    job = await make_job(user_id)

    found_job = await Job.find(user_id, job.id)

    assert found_job is not None
    assert job.id == found_job.id

    # Test finding a job that does not exist in the database
    not_found_job = await Job.find(user_id, "invalid_id")
    assert not_found_job is None


@pytest.mark.asyncio
async def test_create_job(user_id: str):
    job = await Job.create(
        workflow_id="workflow_id",
        user_id=user_id,
    )

    reloaded_job = await Job.find(user_id, job.id)

    assert reloaded_job is not None
    assert reloaded_job.id is not None
    assert reloaded_job.user_id == user_id
    assert reloaded_job.started_at is not None


@pytest.mark.asyncio
async def test_paginate_jobs(user_id: str):
    for _i in range(12):
        await Job.create(
            workflow_id="workflow_id",
            user_id=user_id,
        )

    jobs, last_evaluated_key = await Job.paginate(user_id=user_id, limit=10)
    assert len(jobs) > 0

    jobs, last_evaluated_key = await Job.paginate(
        user_id=user_id, start_key=last_evaluated_key
    )
    assert len(jobs) > 0


@pytest.mark.asyncio
async def test_paginate_jobs_by_workflow(user_id: str):
    for _i in range(10):
        await Job.create(
            workflow_id="workflow_id",
            user_id=user_id,
        )

    for _i in range(10):
        await Job.create(
            workflow_id="another",
            user_id=user_id,
        )

    jobs, _last_evaluated_key = await Job.paginate(
        user_id=user_id, workflow_id="workflow_id"
    )
    assert len(jobs) == 10

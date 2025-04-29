-- Migration for Job model

CREATE TABLE IF NOT EXISTS nodetool_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    job_type TEXT,
    status TEXT,
    workflow_id TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    graph JSONB,
    error TEXT,
    cost REAL
); 
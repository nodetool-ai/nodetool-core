CREATE TABLE IF NOT EXISTS nodetool_jobs (
id TEXT PRIMARY KEY,
user_id TEXT,
job_type TEXT,
status TEXT,
workflow_id TEXT,
started_at TEXT,
finished_at TEXT,
graph TEXT,
error TEXT,
cost REAL
)
CREATE TABLE IF NOT EXISTS run_node_state (
id TEXT PRIMARY KEY,
run_id TEXT NOT NULL,
node_id TEXT NOT NULL,
status TEXT NOT NULL,
attempt INTEGER NOT NULL DEFAULT 1,
scheduled_at TEXT,
started_at TEXT,
completed_at TEXT,
failed_at TEXT,
suspended_at TEXT,
updated_at TEXT NOT NULL,
last_error TEXT,
retryable INTEGER NOT NULL DEFAULT 0,
suspension_reason TEXT,
resume_state_json TEXT,
outputs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_node_state_run_status
ON run_node_state(run_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_node_state_run_node
ON run_node_state(run_id, node_id);
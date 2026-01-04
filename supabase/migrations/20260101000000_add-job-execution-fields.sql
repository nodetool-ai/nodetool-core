CREATE INDEX IF NOT EXISTS idx_run_state_worker ON run_state(worker_id)
CREATE INDEX IF NOT EXISTS idx_run_state_heartbeat ON run_state(heartbeat_at)
CREATE INDEX IF NOT EXISTS idx_run_state_recovery ON run_state(status, heartbeat_at)
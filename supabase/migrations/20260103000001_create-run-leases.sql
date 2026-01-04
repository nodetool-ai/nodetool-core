CREATE TABLE IF NOT EXISTS run_leases (
run_id TEXT PRIMARY KEY,
worker_id TEXT NOT NULL,
acquired_at TEXT NOT NULL,
expires_at TEXT NOT NULL
)
CREATE INDEX IF NOT EXISTS idx_run_leases_expires
ON run_leases(expires_at)
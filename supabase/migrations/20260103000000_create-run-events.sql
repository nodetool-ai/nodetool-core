CREATE TABLE IF NOT EXISTS run_events (
id TEXT PRIMARY KEY,
run_id TEXT NOT NULL,
seq INTEGER NOT NULL,
event_type TEXT NOT NULL,
event_time TEXT NOT NULL,
node_id TEXT,
payload TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_events_run_seq
ON run_events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_run_events_run_node
ON run_events(run_id, node_id);
CREATE INDEX IF NOT EXISTS idx_run_events_run_type
ON run_events(run_id, event_type);
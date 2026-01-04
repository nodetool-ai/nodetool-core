CREATE TABLE IF NOT EXISTS trigger_inputs (
id TEXT PRIMARY KEY,
input_id TEXT NOT NULL UNIQUE,
run_id TEXT NOT NULL,
node_id TEXT NOT NULL,
payload_json TEXT,
processed INTEGER NOT NULL DEFAULT 0,
processed_at TEXT,
cursor TEXT,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trigger_input_run_node_processed
ON trigger_inputs(run_id, node_id, processed);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trigger_input_id
ON trigger_inputs(input_id);
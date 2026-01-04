CREATE TABLE IF NOT EXISTS run_inbox_messages (
id TEXT PRIMARY KEY,
message_id TEXT NOT NULL UNIQUE,
run_id TEXT NOT NULL,
node_id TEXT NOT NULL,
handle TEXT NOT NULL,
msg_seq INTEGER NOT NULL,
payload_json TEXT,
payload_ref TEXT,
status TEXT NOT NULL,
claim_worker_id TEXT,
claim_expires_at TEXT,
consumed_at TEXT,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL
)
CREATE INDEX IF NOT EXISTS idx_inbox_run_node_handle_seq
ON run_inbox_messages(run_id, node_id, handle, msg_seq)
CREATE INDEX IF NOT EXISTS idx_inbox_run_node_handle_status
ON run_inbox_messages(run_id, node_id, handle, status)
CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_message_id
ON run_inbox_messages(message_id)
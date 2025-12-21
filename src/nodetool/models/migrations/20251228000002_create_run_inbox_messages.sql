-- Migration: Create run_inbox_messages table
-- Durable inbox for idempotent node message delivery
-- Supports at-least-once and exactly-once semantics

CREATE TABLE IF NOT EXISTS run_inbox_messages (
    id TEXT PRIMARY KEY,
    
    -- Message identification
    message_id TEXT NOT NULL UNIQUE,  -- Idempotency key
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    handle TEXT NOT NULL,  -- Input handle name
    
    -- Sequencing
    msg_seq INTEGER NOT NULL,  -- Monotonic per (run_id, node_id, handle)
    
    -- Payload
    payload_json TEXT,  -- JSON (inline for small messages)
    payload_ref TEXT,  -- External storage reference for large payloads
    
    -- Status and consumption
    status TEXT NOT NULL,  -- pending | claimed | consumed
    claim_worker_id TEXT,
    claim_expires_at TIMESTAMP,
    consumed_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inbox_run_node_handle_seq ON run_inbox_messages(run_id, node_id, handle, msg_seq);
CREATE INDEX IF NOT EXISTS idx_inbox_run_node_handle_status ON run_inbox_messages(run_id, node_id, handle, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_message_id ON run_inbox_messages(message_id);

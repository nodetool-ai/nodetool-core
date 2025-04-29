-- Migration for Message model

CREATE TABLE IF NOT EXISTS nodetool_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT,
    user_id TEXT,
    tool_call_id TEXT,
    role TEXT,
    name TEXT,
    content JSONB,
    tool_calls JSONB,
    created_at TIMESTAMP
); 
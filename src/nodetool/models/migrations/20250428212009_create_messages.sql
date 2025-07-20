-- Migration for Message model

CREATE TABLE IF NOT EXISTS nodetool_messages (
    id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT '',
    workflow_id TEXT,
    graph JSONB,
    thread_id TEXT,
    tools JSONB,
    tool_call_id TEXT,
    role TEXT,
    name TEXT,
    content JSONB,
    tool_calls JSONB,
    collections JSONB,
    input_files JSONB,
    output_files JSONB,
    created_at TIMESTAMP,
    provider TEXT,
    model TEXT,
    agent_mode BOOLEAN,
    workflow_assistant BOOLEAN,
    help_mode BOOLEAN
); 
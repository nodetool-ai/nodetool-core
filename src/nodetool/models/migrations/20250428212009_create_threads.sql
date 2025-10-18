-- Migration for Thread model

CREATE TABLE IF NOT EXISTS nodetool_threads (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

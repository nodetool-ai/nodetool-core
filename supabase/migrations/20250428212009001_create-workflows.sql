CREATE TABLE IF NOT EXISTS nodetool_workflows (
id TEXT PRIMARY KEY,
user_id TEXT,
access TEXT,
created_at TEXT,
updated_at TEXT,
name TEXT,
tags TEXT,
description TEXT,
thumbnail TEXT,
graph TEXT,
settings TEXT,
receive_clipboard INTEGER
)
CREATE INDEX IF NOT EXISTS idx_nodetool_workflows_user_id
ON nodetool_workflows (user_id)
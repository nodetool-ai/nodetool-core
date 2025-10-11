-- Migration to add agent execution tracking fields to messages table

ALTER TABLE nodetool_messages
ADD COLUMN IF NOT EXISTS agent_execution_id TEXT;

ALTER TABLE nodetool_messages
ADD COLUMN IF NOT EXISTS execution_event_type TEXT;

# Workflow Versions API

This document describes the REST API for managing workflow versions, including the autosave feature for automatic version tracking.

## Overview

Workflow versions allow you to save snapshots of your workflow at any point in time. The backend-based autosave feature automatically creates versions based on configured settings, reducing data loss from browser crashes or accidental closures.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workflows/{id}/versions` | Create a manual version |
| `GET` | `/api/workflows/{id}/versions` | List all versions |
| `GET` | `/api/workflows/{id}/versions/{version}` | Get specific version |
| `POST` | `/api/workflows/{id}/versions/{version}/restore` | Restore to version |
| `POST` | `/api/workflows/{id}/autosave` | Create autosave version |

---

## Version Management

### Create Manual Version

Create a new version snapshot of a workflow.

**Endpoint:** `POST /api/workflows/{id}/versions`

**Request Body:**
```json
{
  "name": "Version name (optional)",
  "description": "Version description (optional)"
}
```

**Response:**
```json
{
  "id": "uuid",
  "workflow_id": "workflow-uuid",
  "version": 1,
  "created_at": "2026-01-04T10:00:00Z",
  "name": "Version 1",
  "description": "",
  "graph": { "nodes": [], "edges": [] },
  "save_type": "manual",
  "autosave_metadata": {}
}
```

### List Versions

Get all versions of a workflow with pagination.

**Endpoint:** `GET /api/workflows/{id}/versions`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cursor` | int | - | Version number to start pagination after |
| `limit` | int | 100 | Maximum versions to return |

**Response:**
```json
{
  "next": 10,
  "versions": [
    {
      "id": "uuid",
      "workflow_id": "workflow-uuid",
      "version": 15,
      "created_at": "2026-01-04T10:00:00Z",
      "name": "Version 15",
      "description": "",
      "graph": { "nodes": [], "edges": [] },
      "save_type": "autosave",
      "autosave_metadata": { "client_id": "abc", "trigger_reason": "autosave" }
    }
  ]
}
```

### Get Specific Version

Retrieve a particular version by version number.

**Endpoint:** `GET /api/workflows/{id}/versions/{version}`

**Response:** Same as Create Version response.

### Restore Version

Restore a workflow to a previous version state.

**Endpoint:** `POST /api/workflows/{id}/versions/{version}/restore`

**Response:** Returns the restored `Workflow` object.

---

## Autosave Feature

The autosave feature provides automatic version creation based on configurable settings. The backend controls all save timing and rate limiting.

### Autosave Endpoint

**Endpoint:** `POST /api/workflows/{id}/autosave`

**Request Body:**
```json
{
  "save_type": "autosave" | "checkpoint" | "manual",
  "description": "Version description (optional)",
  "force": false,  // Optional: bypass debouncing and max versions check
  "client_id": "unique-client-id"  // Optional: for deduplication
}
```

**Response:**
```json
{
  "version": {
    "id": "uuid",
    "workflow_id": "workflow-uuid",
    "version": 16,
    "created_at": "2026-01-04T10:00:00Z",
    "name": "Autosave 16",
    "description": "",
    "graph": { "nodes": [], "edges": [] },
    "save_type": "autosave",
    "autosave_metadata": {
      "client_id": "unique-client-id",
      "trigger_reason": "autosave"
    }
  },
  "message": "autosaved",
  "skipped": false
}
```

**Skipped Response:**
```json
{
  "version": null,
  "message": "skipped (too soon)",
  "skipped": true
}
```

### Save Types

| Type | Description |
|------|-------------|
| `autosave` | Automatically created by the autosave system |
| `manual` | User-created version via `/versions` endpoint |
| `checkpoint` | Version created before significant operations (e.g., workflow run) |
| `restore` | Version created when restoring from a previous version |

### Backend Logic

The autosave endpoint implements the following validation:

1. **Workflow Access**: User must own the workflow
2. **Rate Limiting**: Skips autosaves within 30 seconds of the last autosave (configurable)
3. **Max Versions**: Skips when exceeding 20 autosaves per workflow (configurable)
4. **Force Flag**: When `force: true`, bypasses rate limiting and max versions checks

### Cleanup

After creating an autosave version, the backend asynchronously:
- Deletes autosaves exceeding the max versions limit
- Deletes autosaves older than 7 days (configurable)

---

## Configuration

Autosave behavior can be configured via environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AUTOSAVE_ENABLED` | `true` | Enable/disable autosave |
| `AUTOSAVE_INTERVAL_MINUTES` | `5` | Interval between autosaves |
| `AUTOSAVE_MIN_INTERVAL_SECONDS` | `30` | Minimum interval between saves |
| `AUTOSAVE_MAX_VERSIONS_PER_WORKFLOW` | `20` | Max autosaves to keep per workflow |
| `AUTOSAVE_KEEP_DAYS` | `7` | Days to keep autosave versions |

---

## Database Schema

The `workflow_versions` table stores all versions:

```sql
CREATE TABLE nodetool_workflow_versions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    name TEXT DEFAULT '',
    description TEXT DEFAULT '',
    graph TEXT DEFAULT '{}',
    save_type TEXT DEFAULT 'manual' CHECK(save_type IN ('autosave', 'manual', 'checkpoint', 'restore')),
    autosave_metadata TEXT DEFAULT '{}'
);

CREATE INDEX idx_nodetool_workflow_versions_save_type
ON nodetool_workflow_versions (workflow_id, save_type, created_at);
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | TEXT | Unique version UUID |
| `workflow_id` | TEXT | Reference to parent workflow |
| `user_id` | TEXT | User who created the version |
| `version` | INTEGER | Sequential version number |
| `created_at` | TEXT | ISO timestamp |
| `name` | TEXT | Version name (e.g., "Autosave 15") |
| `description` | TEXT | Optional description |
| `graph` | TEXT | JSON-encoded workflow graph |
| `save_type` | TEXT | Type of save (autosave/manual/checkpoint/restore) |
| `autosave_metadata` | TEXT | JSON metadata (client_id, trigger_reason) |

---

## Frontend Integration

### Triggering Autosaves

Frontend should call the autosave endpoint when:
- A significant edit occurs (e.g., node added/removed)
- Before workflow execution
- On window/tab close (may use sendBeacon)

```javascript
async function triggerAutosave(workflowId, clientId) {
  const response = await fetch(`/api/workflows/${workflowId}/autosave`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      save_type: 'autosave',
      client_id: clientId
    })
  });
  
  const result = await response.json();
  
  if (!result.skipped) {
    showNotification('Autosaved');
  }
  // Silently ignore skipped saves
}
```

### Save Before Run

Create a checkpoint version before workflow execution:

```javascript
async function saveBeforeRun(workflowId) {
  await fetch(`/api/workflows/${workflowId}/autosave`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      save_type: 'checkpoint',
      description: 'Pre-run checkpoint',
      force: true  // Ensure checkpoint is created
    })
  });
}
```

---

## Error Handling

| Error Code | Description |
|------------|-------------|
| 400 | Invalid request body |
| 401 | Unauthorized |
| 404 | Workflow or version not found |
| 422 | Validation error (missing required fields) |
| 500 | Internal server error |

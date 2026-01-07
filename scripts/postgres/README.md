# PostgreSQL Setup for nodetool

This directory contains configuration and scripts for running PostgreSQL locally.

## Files

- `postgresql.conf` - PostgreSQL server configuration
- `start.sh` - Script to start/stop PostgreSQL
- `.env.postgres.example` - Environment variables template

## Quick Start

1. Copy the environment template:
   ```bash
   cp .env.postgres.example .env.postgres.local
   ```

2. Edit `.env.postgres.local` with your settings (optional - defaults work for local dev)

3. Start PostgreSQL:
   ```bash
   ./start.sh start
   ```

4. Verify it's running:
   ```bash
   ./start.sh status
   ```

## Commands

```bash
./start.sh start    # Start PostgreSQL
./start.sh stop     # Stop PostgreSQL
./start.sh restart  # Restart PostgreSQL
./start.sh status   # Check status
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PORT` | Port to listen on | 5432 |
| `POSTGRES_DB` | Database name | nodetool |
| `POSTGRES_USER` | Database user | nodetool |
| `POSTGRES_PASSWORD` | Database password | nodetool_password |
| `POSTGRES_DATA_DIR` | Data directory | `~/.local/share/nodetool/postgres` |
| `POSTGRES_CONF_DIR` | Config directory | `~/.config/nodetool/postgres` |

## Integration with nodetool

To use PostgreSQL with nodetool, set these environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=nodetool
export POSTGRES_USER=nodetool
export POSTGRES_PASSWORD=your_password
```

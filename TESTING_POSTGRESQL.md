# PostgreSQL Testing Guide

This document describes how to run the test suite with PostgreSQL instead of the default SQLite.

## Quick Start

### Using Docker Compose (Recommended)

1. Start the PostgreSQL test database:
```bash
docker compose -f docker-compose.test.yml up -d
```

2. Run tests with PostgreSQL:
```bash
export USE_POSTGRES_FOR_TESTS=1
export POSTGRES_TEST_DB=nodetool_test
export POSTGRES_TEST_USER=nodetool_test
export POSTGRES_TEST_PASSWORD=nodetool_test_password
export POSTGRES_TEST_HOST=localhost
export POSTGRES_TEST_PORT=5433

pytest -v
```

Or use the provided environment file:
```bash
export $(cat .env.test.postgres | xargs)
pytest -v
```

3. Stop the test database when done:
```bash
docker compose -f docker-compose.test.yml down
```

### Using Your Own PostgreSQL Instance

If you have your own PostgreSQL instance, configure the connection parameters:

```bash
export USE_POSTGRES_FOR_TESTS=1
export POSTGRES_TEST_DB=your_test_db
export POSTGRES_TEST_USER=your_user
export POSTGRES_TEST_PASSWORD=your_password
export POSTGRES_TEST_HOST=your_host
export POSTGRES_TEST_PORT=5432

pytest -v
```

**Note:** The test suite will create and drop test schemas automatically, but it will NOT create the database itself. Ensure the database exists before running tests.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_POSTGRES_FOR_TESTS` | Enable PostgreSQL for tests (set to `1`) | `0` (uses SQLite) |
| `POSTGRES_TEST_DB` | Database name | `nodetool_test` |
| `POSTGRES_TEST_USER` | Database user | `nodetool_test` |
| `POSTGRES_TEST_PASSWORD` | Database password | `nodetool_test_password` |
| `POSTGRES_TEST_HOST` | Database host | `localhost` |
| `POSTGRES_TEST_PORT` | Database port | `5433` |

## How It Works

### Test Isolation

- Each pytest worker (when using `-n auto` for parallel execution) gets its own schema
- Schemas are named `test_schema` or `test_schema_<worker_id>` for parallel runs
- All tables are truncated between tests for clean state
- Schemas are automatically dropped after the test session

### Performance

- PostgreSQL tests use a connection pool shared across the test session
- Schemas are created in memory where possible (via tmpfs in docker-compose)
- Table truncation is fast using PostgreSQL's `TRUNCATE CASCADE`

### Compatibility

All existing tests should work with both SQLite and PostgreSQL. The test fixtures automatically detect which backend to use based on the `USE_POSTGRES_FOR_TESTS` environment variable.

## CI/CD Integration

The GitHub Actions CI workflow automatically runs the test suite twice:
1. First with SQLite (default behavior)
2. Then with PostgreSQL (using GitHub Actions services)

This ensures compatibility with both database backends.

## Troubleshooting

### Connection Refused

If you get a connection refused error:
- Ensure PostgreSQL is running: `docker compose -f docker-compose.test.yml ps`
- Wait for health check: `docker compose -f docker-compose.test.yml logs postgres-test`
- Check port availability: `lsof -i :5433`

### Schema Already Exists

If you get schema already exists errors:
- Clean up manually: `docker compose -f docker-compose.test.yml down -v`
- Or connect and drop: `psql -h localhost -p 5433 -U nodetool_test -d nodetool_test -c "DROP SCHEMA IF EXISTS test_schema CASCADE;"`

### Migration Errors

If migrations fail:
- Check PostgreSQL logs: `docker compose -f docker-compose.test.yml logs`
- Verify database exists: `psql -h localhost -p 5433 -U nodetool_test -l`
- Reset database: `docker compose -f docker-compose.test.yml down -v && docker compose -f docker-compose.test.yml up -d`

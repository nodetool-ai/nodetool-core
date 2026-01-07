#!/bin/bash
# Create test database for nodetool tests
# Usage: ./scripts/postgres/create_test_db.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POSTGRES_HOST="${POSTGRES_TEST_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_TEST_PORT:-5433}"
POSTGRES_USER="${POSTGRES_USER:-nodetool}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-nodetool_password}"
POSTGRES_SUPERUSER="${POSTGRES_SUPERUSER:-postgres}"

TEST_DB_NAME="${POSTGRES_TEST_DB:-nodetool_test}"
TEST_DB_USER="${POSTGRES_TEST_USER:-nodetool_test}"
TEST_DB_PASSWORD="${POSTGRES_TEST_PASSWORD:-nodetool_test_password}"

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1"
}

wait_for_postgres() {
    local max_attempts=30
    local attempt=0
    log_info "Waiting for PostgreSQL on $POSTGRES_HOST:$POSTGRES_PORT..."
    while ! psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_SUPERUSER" -c "SELECT 1" &>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -gt $max_attempts ]; then
            log_error "PostgreSQL not available after $max_attempts attempts"
            return 1
        fi
        sleep 1
    done
    log_info "PostgreSQL is ready!"
}

create_test_database() {
    log_info "Creating test database '$TEST_DB_NAME' and user '$TEST_DB_USER' on port $POSTGRES_PORT..."

    wait_for_postgres || exit 1

    export PGPASSWORD="$POSTGRES_PASSWORD"

    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_SUPERUSER" <<EOF
-- Create test user if it doesn't exist
DO
\$do\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = '$TEST_DB_USER') THEN
      CREATE USER $TEST_DB_USER WITH PASSWORD '$TEST_DB_PASSWORD';
   END IF;
END
\$do\$;

-- Create test database if it doesn't exist
SELECT 'CREATE DATABASE $TEST_DB_NAME OWNER $TEST_DB_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$TEST_DB_NAME')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $TEST_DB_NAME TO $TEST_DB_USER;

-- Connect to the database and set up schema permissions
\c $TEST_DB_NAME
GRANT CREATE ON SCHEMA public TO $TEST_DB_USER;
GRANT USAGE ON SCHEMA public TO $TEST_DB_USER;
EOF

    log_info "Test database '$TEST_DB_NAME' created successfully!"
    log_info "Connection info:"
    echo "  Host: $POSTGRES_HOST"
    echo "  Port: $POSTGRES_PORT"
    echo "  Database: $TEST_DB_NAME"
    echo "  User: $TEST_DB_USER"
}

create_test_database

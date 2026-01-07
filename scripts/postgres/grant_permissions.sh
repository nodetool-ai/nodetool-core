#!/bin/bash
# Grant PostgreSQL permissions for nodetool
# Usage: ./scripts/postgres/grant_permissions.sh [password]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env.postgres.example"

POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"

grant_permissions_as_postgres() {
    export PGPASSWORD="$POSTGRES_PASSWORD"
    echo "Granting permissions as postgres superuser..."
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "GRANT CREATE ON DATABASE nodetool TO nodetool;"
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "GRANT USAGE ON SCHEMA public TO nodetool;"
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "GRANT CREATE ON SCHEMA public TO nodetool;"

    echo "Granting permissions for nodetool_test database..."
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "GRANT CREATE ON DATABASE nodetool_test TO nodetool_test;"
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "GRANT USAGE ON SCHEMA public TO nodetool_test;"
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "GRANT CREATE ON SCHEMA public TO nodetool_test;"
}

grant_permissions_as_user() {
    CURRENT_USER=$(whoami)
    echo "Postgres superuser not available, granting permissions as current user ($CURRENT_USER)..."
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$CURRENT_USER" -d nodetool -c "GRANT CREATE ON SCHEMA public TO nodetool;"
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$CURRENT_USER" -d nodetool_test -c "GRANT CREATE ON SCHEMA public TO nodetool_test;"
}

if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "SELECT 1" &>/dev/null; then
    grant_permissions_as_postgres
else
    grant_permissions_as_user
fi

echo "Done!"

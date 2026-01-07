#!/bin/bash
# PostgreSQL Start Script for nodetool
# Usage: ./start.sh [start|stop|restart|status]

set -e

# Configuration
POSTGRES_PORT=${POSTGRES_PORT:-5432}
POSTGRES_DB=${POSTGRES_DB:-nodetool}
POSTGRES_USER=${POSTGRES_USER:-nodetool}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-nodetool_password}
POSTGRES_DATA_DIR=${POSTGRES_DATA_DIR:-"$HOME/.local/share/nodetool/postgres"}
POSTGRES_CONF_DIR="$HOME/.config/nodetool/postgres"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create directories
mkdir -p "$POSTGRES_DATA_DIR"
mkdir -p "$POSTGRES_CONF_DIR"

# Copy default config if none exists
if [ ! -f "$POSTGRES_CONF_DIR/postgresql.conf" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/postgresql.conf" ]; then
        cp "$SCRIPT_DIR/postgresql.conf" "$POSTGRES_CONF_DIR/"
        log_info "Copied default postgresql.conf to $POSTGRES_CONF_DIR"
    fi
fi

start_postgres() {
    if pg_isready -p "$POSTGRES_PORT" >/dev/null 2>&1; then
        log_warn "PostgreSQL is already running on port $POSTGRES_PORT"
        return 0
    fi

    log_info "Starting PostgreSQL..."

    # Initialize database if needed
    if [ ! -d "$POSTGRES_DATA_DIR/base" ]; then
        log_info "Initializing database directory..."
        initdb -D "$POSTGRES_DATA_DIR" --no-locale --encoding=UTF8
    fi

    # Start PostgreSQL
    pg_ctl -D "$POSTGRES_DATA_DIR" -o "-p $POSTGRES_PORT -c config_file=$POSTGRES_CONF_DIR/postgresql.conf" -l "$POSTGRES_DATA_DIR/postgres.log" start -w

    # Wait for PostgreSQL to be ready
    local count=0
    while ! pg_isready -p "$POSTGRES_PORT" >/dev/null 2>&1; do
        sleep 1
        count=$((count + 1))
        if [ $count -gt 30 ]; then
            log_error "PostgreSQL failed to start within 30 seconds"
            cat "$POSTGRES_DATA_DIR/postgres.log" 2>/dev/null || true
            exit 1
        fi
    done

    # Create database and user if they don't exist
    log_info "Setting up database and user..."
    psql -h localhost -p "$POSTGRES_PORT" -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';" 2>/dev/null || true
    psql -h localhost -p "$POSTGRES_PORT" -U postgres -c "CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;" 2>/dev/null || true
    psql -h localhost -p "$POSTGRES_PORT" -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;" 2>/dev/null || true

    log_info "PostgreSQL started successfully on port $POSTGRES_PORT"
    log_info "Database: $POSTGRES_DB, User: $POSTGRES_USER"
}

stop_postgres() {
    if ! pg_isready -p "$POSTGRES_PORT" >/dev/null 2>&1; then
        log_warn "PostgreSQL is not running"
        return 0
    fi

    log_info "Stopping PostgreSQL..."
    pg_ctl -D "$POSTGRES_DATA_DIR" -o "-p $POSTGRES_PORT" stop -w
    log_info "PostgreSQL stopped"
}

restart_postgres() {
    stop_postgres
    sleep 2
    start_postgres
}

status_postgres() {
    if pg_isready -p "$POSTGRES_PORT" >/dev/null 2>&1; then
        log_info "PostgreSQL is running on port $POSTGRES_PORT"
        echo ""
        echo "Connection info:"
        echo "  Host: localhost"
        echo "  Port: $POSTGRES_PORT"
        echo "  Database: $POSTGRES_DB"
        echo "  User: $POSTGRES_USER"
        return 0
    else
        log_error "PostgreSQL is not running on port $POSTGRES_PORT"
        return 1
    fi
}

# Main command handler
case "${1:-start}" in
    start)
        start_postgres
        ;;
    stop)
        stop_postgres
        ;;
    restart)
        restart_postgres
        ;;
    status)
        status_postgres
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status]"
        exit 1
        ;;
esac

#!/bin/bash
# Start NodeTool server with OpenTelemetry auto-instrumentation
# Usage: ./scripts/server-otel.sh [--production] [--otel-endpoint <url>]
#
# Prerequisites:
#   - Install with extras: pip install -e ".[all-extras]" or uv sync --all-extras
#   - opentelemetry-distro includes auto-instrumentation for FastAPI, Uvicorn, HTTP
#
# Environment Variables (can also be set in .env):
#   OTEL_SERVICE_NAME          - Name of the service (default: nodetool-api)
#   OTEL_TRACES_EXPORTER       - Comma-separated exporters: console,otlp (default: console,otlp)
#   OTEL_EXPORTER_OTLP_ENDPOINT - OTLP collector endpoint (default: http://localhost:4317)
#   OTEL_EXPORTER_OTLP_PROTOCOL - OTLP protocol: grpc/http/protobuf (default: grpc)
#   OTEL_RESOURCE_ATTRIBUTES   - Additional resource attributes (e.g., "deployment.environment=production")
#   OTEL_SAMPLING_RATIO        - Sampling ratio 0.0-1.0 (default: 1.0)
#   OTEL_METRICS_EXPORTER      - Metrics exporter: console,otlp,none (default: otlp)
#   OTEL_LOGS_EXPORTER         - Logs exporter: console,otlp,none (default: none)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
PRODUCTION=false
OTEL_ENDPOINT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            PRODUCTION=true
            shift
            ;;
        --otel-endpoint)
            OTEL_ENDPOINT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--production] [--otel-endpoint <url>]"
            echo ""
            echo "Options:"
            echo "  --production          Enable production mode"
            echo "  --otel-endpoint       OTLP collector endpoint"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  OTEL_SERVICE_NAME          Service name (default: nodetool-api)"
            echo "  OTEL_TRACES_EXPORTER       Exporters: console,otlp (default: console,otlp)"
            echo "  OTEL_EXPORTER_OTLP_ENDPOINT OTLP endpoint (default: http://localhost:4317)"
            echo "  OTEL_METRICS_EXPORTER      Metrics exporter (default: otlp)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load environment variables from .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Set defaults for OpenTelemetry
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-nodetool-api}"
export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-console,otlp}"
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4317}"
export OTEL_EXPORTER_OTLP_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-grpc}"
export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-otlp}"
export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"

# Override endpoint if provided as argument
if [ -n "$OTEL_ENDPOINT" ]; then
    export OTEL_EXPORTER_OTLP_ENDPOINT="$OTEL_ENDPOINT"
fi

# Add deployment environment if not already set
if [ -z "$OTEL_RESOURCE_ATTRIBUTES" ] && [ -n "$ENV" ]; then
    export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=$ENV"
fi

# Ensure service name is set
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-nodetool-api}"

echo "=============================================="
echo "NodeTool Server with OpenTelemetry"
echo "=============================================="
echo "Service Name: $OTEL_SERVICE_NAME"
echo "Traces Exporter: $OTEL_TRACES_EXPORTER"
echo "OTLP Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"
echo "Metrics Exporter: $OTEL_METRICS_EXPORTER"
echo "=============================================="

# Use opentelemetry-instrument to auto-instrument the uvicorn server
exec opentelemetry-instrument \
    uvicorn \
    nodetool.api.app:app \
    --host 0.0.0.0 \
    --port 7777 \
    --app-dir "$PROJECT_ROOT/src" \
    $([ "$PRODUCTION" = true ] && echo "--workers 4" || echo "--reload")

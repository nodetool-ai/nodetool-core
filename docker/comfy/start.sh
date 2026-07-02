#!/usr/bin/env bash
# Entrypoint for the nodetool-worker-comfy image: launch headless ComfyUI on
# loopback, wait until its HTTP API answers, then launch the NodeTool worker
# that proxies it. If either process dies the container exits so the
# orchestrator (RunPod) restarts it.
set -uo pipefail

COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
WORKER_HOST="${NODETOOL_WORKER_HOST:-0.0.0.0}"
WORKER_PORT="${NODETOOL_WORKER_PORT:-7777}"

export COMFYUI_URL="${COMFYUI_URL:-http://${COMFY_HOST}:${COMFY_PORT}}"
export COMFY_MODELS_DIR="${COMFY_MODELS_DIR:-${WORKSPACE_DIR}/models}"
export NODETOOL_COMFY_ENABLED="${NODETOOL_COMFY_ENABLED:-1}"

# Persist models, inputs, outputs and temp files on the network volume.
mkdir -p \
  "${COMFY_MODELS_DIR}" \
  "${WORKSPACE_DIR}/input" \
  "${WORKSPACE_DIR}/output" \
  "${WORKSPACE_DIR}/temp"

echo "[nodetool-worker-comfy] starting ComfyUI on ${COMFY_HOST}:${COMFY_PORT}" >&2
# ${COMFY_ARGS} is intentionally unquoted: it carries extra CLI flags
# (e.g. "--lowvram --disable-smart-memory").
# shellcheck disable=SC2086
python /opt/ComfyUI/main.py \
  --listen "${COMFY_HOST}" \
  --port "${COMFY_PORT}" \
  --extra-model-paths-config /opt/comfy/extra_model_paths.yaml \
  --input-directory "${WORKSPACE_DIR}/input" \
  --output-directory "${WORKSPACE_DIR}/output" \
  --temp-directory "${WORKSPACE_DIR}/temp" \
  ${COMFY_ARGS:-} &
COMFY_PID=$!

# Wait for the ComfyUI HTTP API before starting the worker so the first
# comfy.* message never races the server boot (model scans can take a while).
STARTUP_TIMEOUT="${COMFY_STARTUP_TIMEOUT:-300}"
for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "[nodetool-worker-comfy] ComfyUI exited during startup" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
    echo "[nodetool-worker-comfy] ComfyUI is up" >&2
    break
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
  echo "[nodetool-worker-comfy] ComfyUI did not come up within ${STARTUP_TIMEOUT}s" >&2
  kill -TERM "${COMFY_PID}" 2>/dev/null
  exit 1
fi

echo "[nodetool-worker-comfy] starting NodeTool worker on ${WORKER_HOST}:${WORKER_PORT}" >&2
python -m nodetool.worker --host "${WORKER_HOST}" --port "${WORKER_PORT}" &
WORKER_PID=$!

shutdown() {
  kill -TERM "${COMFY_PID}" "${WORKER_PID}" 2>/dev/null
}
trap shutdown TERM INT

# Exit as soon as either process dies.
wait -n "${COMFY_PID}" "${WORKER_PID}"
EXIT_CODE=$?
shutdown
wait
exit "${EXIT_CODE}"

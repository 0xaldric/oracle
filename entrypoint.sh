#!/bin/bash
# ─── Rush Oracle Entrypoint ─────────────────────────────────────────────────
# Starts stream_server.py in background, then round_manager_rush.py
# All configuration via environment variables (no .env file)
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo "  Rush Oracle (Docker + GPU)"
echo "  Camera:    ${CAMERA_ID:-peace-bridge}"
echo "  Model:     ${YOLO_MODEL:-yolov8x.pt}"
echo "  FPS:       ${TARGET_FPS:-8}"
echo "  WS Port:   ${WS_PORT:-8765}"
echo "  Duration:  ${ROUND_DURATION:-300}s"
echo "  Betting:   ${BETTING_WINDOW:-150}s"
echo "  Factory:   ${FACTORY_ADDRESS:-not set}"
echo "  RPC:       ${RPC_URL:-not set}"
echo "============================================================"

# Validate required env vars
for var in PRIVATE_KEY FACTORY_ADDRESS RPC_URL; do
  if [ -z "${!var}" ]; then
    echo "[ERROR] Required env var $var is not set"
    exit 1
  fi
done

# ── Start stream_server.py in background ────────────────────────────────────
echo "[Entrypoint] Starting stream_server.py..."
python3 stream_server.py \
  --camera "${CAMERA_ID:-peace-bridge}" \
  --port "${WS_PORT:-8765}" \
  --model "${YOLO_MODEL:-yolov8x.pt}" \
  --fps "${TARGET_FPS:-8}" &
STREAM_PID=$!

# Wait for stream server to be ready
echo "[Entrypoint] Waiting for stream_server on port ${WS_PORT:-8765}..."
for i in $(seq 1 30); do
  if python3 -c "
import asyncio, websockets
async def check():
    ws = await websockets.connect('ws://localhost:${WS_PORT:-8765}', open_timeout=2)
    await ws.close()
asyncio.run(check())
" 2>/dev/null; then
    echo "[Entrypoint] stream_server ready"
    break
  fi
  if ! kill -0 $STREAM_PID 2>/dev/null; then
    echo "[ERROR] stream_server exited unexpectedly"
    exit 1
  fi
  sleep 2
done

# ── Start round_manager_rush.py in foreground ───────────────────────────────
echo "[Entrypoint] Starting round_manager_rush.py..."
exec python3 round_manager_rush.py "$@"

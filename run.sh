#!/bin/bash
# ─── Rush Oracle Runner (no Docker, no ngrok) ──────────────────────────────
# Usage:
#   ./run.sh                    # full mode (stream + rounds)
#   ./run.sh --stream-only      # stream server only
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/python3"

# ── Preflight checks ──────────────────────────────────────────────────────
if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found. Run: ./setup.sh"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "ERROR: .env not found. Copy .env.example and fill in values."
    exit 1
fi

# ── Load .env ──────────────────────────────────────────────────────────────
set -a
source "$SCRIPT_DIR/.env"
set +a

# ── Validate required vars ────────────────────────────────────────────────
for var in PRIVATE_KEY RPC_URL FACTORY_ADDRESS; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set in .env"
        exit 1
    fi
done

WS_PORT="${WS_PORT:-8765}"
CAMERA="${CAMERA_ID:-peace-bridge}"
YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"

echo "═══════════════════════════════════════════════════"
echo "  Rush Oracle"
echo "  Camera:  $CAMERA"
echo "  Model:   $YOLO_MODEL"
echo "  WS port: $WS_PORT"
echo "═══════════════════════════════════════════════════"

# ── Clean previous instances ──────────────────────────────────────────────
pkill -f "round_manager_rush.py" 2>/dev/null || true
pkill -f "stream_server.py" 2>/dev/null || true
pkill -f "watchdog.py" 2>/dev/null || true
kill $(lsof -ti :"$WS_PORT") 2>/dev/null || true
rm -f /tmp/rush_oracle.lock 2>/dev/null || true
sleep 1

# ── Graceful shutdown ─────────────────────────────────────────────────────
WATCHDOG_PID=""
STREAM_PID=""

_cleanup() {
    echo ""
    echo "[Oracle] Shutting down..."
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
    [ -n "$STREAM_PID" ] && kill "$STREAM_PID" 2>/dev/null
    sleep 2
    [ -n "$WATCHDOG_PID" ] && kill -9 "$WATCHDOG_PID" 2>/dev/null
    [ -n "$STREAM_PID" ] && kill -9 "$STREAM_PID" 2>/dev/null
    exit 0
}
trap '_cleanup' TERM INT

# ── Start stream server ──────────────────────────────────────────────────
cd "$SCRIPT_DIR"

echo "[Oracle] Starting stream server..."
"$VENV" -u stream_server.py \
    --camera "$CAMERA" \
    --port "$WS_PORT" \
    --model "$YOLO_MODEL" &
STREAM_PID=$!

# Wait for stream server ready
echo "[Oracle] Waiting for stream server..."
for i in $(seq 1 30); do
    if "$VENV" -c "
import asyncio, websockets
async def t():
    async with websockets.connect('ws://localhost:$WS_PORT', open_timeout=2) as ws:
        await asyncio.wait_for(ws.recv(), timeout=2)
asyncio.run(t())
" 2>/dev/null; then
        echo "[Oracle] Stream server ready!"
        break
    fi
    sleep 2
done

# ── Stream-only or full mode ─────────────────────────────────────────────
if [ "$1" = "--stream-only" ]; then
    echo "[Oracle] Stream-only mode — no rounds"
    wait "$STREAM_PID"
else
    echo "[Oracle] Starting watchdog + round manager..."
    "$VENV" -u watchdog.py "$@" &
    WATCHDOG_PID=$!

    # Monitor: restart stream if it dies, exit if watchdog dies
    while true; do
        if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
            echo "[Oracle] Watchdog died — shutting down"
            kill "$STREAM_PID" 2>/dev/null
            exit 1
        fi
        if ! kill -0 "$STREAM_PID" 2>/dev/null; then
            echo "[Oracle] Stream server died — restarting..."
            sleep 3
            "$VENV" -u stream_server.py \
                --camera "$CAMERA" \
                --port "$WS_PORT" \
                --model "$YOLO_MODEL" &
            STREAM_PID=$!
        fi
        sleep 5
    done
fi

# ─── Rush Oracle — GPU Docker Image ──────────────────────────────────────────
# Runs: stream_server.py (YOLO + BoT-SORT) + round_manager_rush.py
#
# Build:
#   docker build -t rush-oracle .
#
# Run (all config via env vars, no .env file):
#   docker run --gpus all \
#     -e PRIVATE_KEY=0x... \
#     -e FACTORY_ADDRESS=0x... \
#     -e RPC_URL=https://... \
#     -e FEE_RECIPIENT=0x... \
#     -e ROUND_DURATION=300 \
#     -e BETTING_WINDOW=150 \
#     -e WS_PORT=8765 \
#     -e CAMERA_ID=peace-bridge \
#     -e YOLO_MODEL=yolov8x.pt \
#     -e TARGET_FPS=8 \
#     -p 8765:8765 \
#     rush-oracle
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl ca-certificates ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt supervision lap

# ── Node.js dependencies ────────────────────────────────────────────────────
COPY package.json .
RUN npm install --omit=dev

# ── Application code ────────────────────────────────────────────────────────
COPY round_manager_rush.py .
COPY stream_server.py .
COPY counter.py .
COPY signer.py .
COPY watchdog.py .
COPY orphan_recovery.py .
COPY publisher_v2.js .
COPY cameras.json .
COPY botsort_custom.yaml .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# ── Pre-download YOLO model (optional, can also mount at runtime) ───────────
ARG YOLO_MODEL=yolov8x.pt
RUN python3 -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')"

# ── Defaults ────────────────────────────────────────────────────────────────
ENV WS_PORT=8765
ENV ROUND_DURATION=300
ENV BETTING_WINDOW=150
ENV CAMERA_ID=peace-bridge
ENV YOLO_MODEL=yolov8x.pt
ENV TARGET_FPS=8

EXPOSE 8765

ENTRYPOINT ["./entrypoint.sh"]

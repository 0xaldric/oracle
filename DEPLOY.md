# Oracle Server — Deploy & Operations Guide

## Server Info

```
SSH:     ssh -p 30008 user@205.200.239.11 -i ~/.ssh/id_tensor
GPU:     NVIDIA RTX 4090 (24GB VRAM)
OS:      Ubuntu 24.04, Python 3.12, Node 18
WS:      port 8765 (forwarded to public port 30010)
```

---

## First-Time Setup

### 1. SSH vào server

```bash
ssh -p 30008 user@205.200.239.11 -i ~/.ssh/id_tensor
```

### 2. Clone repo

```bash
cd ~
git clone https://github.com/0xaldric/oracle.git
cd oracle
```

### 3. Setup Python venv + deps

```bash
chmod +x setup.sh run.sh
./setup.sh
```

### 4. Fix PyTorch cho GPU (QUAN TRỌNG)

Default pip install sẽ cài PyTorch cu130, nhưng driver chỉ support CUDA 12.7.
Phải force reinstall với cu126:

```bash
source venv/bin/activate
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU:

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected: `CUDA: True, GPU: NVIDIA GeForce RTX 4090`

### 5. Install npm deps

```bash
npm install
```

### 6. Tạo .env

```bash
cp .env.example .env   # hoặc tạo mới
nano .env
```

Nội dung:

```env
PRIVATE_KEY=0x...                    # Oracle wallet private key
FACTORY_ADDRESS=0x...                # MarketFactory contract address
RPC_URL=https://...                  # BSC Testnet RPC
FEE_RECIPIENT=0x...                  # Fee recipient address
ROUND_DURATION=300                   # Counting window (seconds)
BETTING_WINDOW=150                   # Betting window (seconds)
WS_PORT=8765                         # WebSocket port
CAMERA_ID=peace-bridge               # Camera ID from cameras.json
YOLO_MODEL=yolov8x.pt               # YOLO model (x=best, n=fastest)
TARGET_FPS=8                         # Target frames per second
LEDGER_URL=http://localhost:4000/api # Backend API URL
LEDGER_API_KEY=...                   # Backend API key (x-api-key header)
ABLY_API_KEY=...                     # Ably real-time key (keyName:keySecret)
```

### 7. Install systemd service

```bash
sudo cp rush-oracle-full.service /etc/systemd/system/rush-oracle.service
sudo systemctl daemon-reload
sudo systemctl enable rush-oracle
sudo systemctl start rush-oracle
```

---

## Daily Operations

### Service management

```bash
# Xem trạng thái
sudo systemctl status rush-oracle

# Restart
sudo systemctl restart rush-oracle

# Dừng
sudo systemctl stop rush-oracle

# Logs real-time
sudo journalctl -u rush-oracle -f

# 50 dòng log gần nhất
sudo journalctl -u rush-oracle -n 50 --no-pager
```

### GPU monitoring

```bash
nvidia-smi
# Expected: python3 process using ~1.2GB VRAM, GPU ~50-60%
```

### Run modes

**Full mode** (default — stream + rounds):
```bash
./run.sh
```

**Stream only** (no on-chain rounds):
```bash
./run.sh --stream-only
```

Đổi mode trong systemd: sửa `ExecStart` trong service file rồi:
```bash
sudo systemctl daemon-reload
sudo systemctl restart rush-oracle
```

---

## Troubleshooting

### Oracle crash loop

```bash
sudo journalctl -u rush-oracle -n 100 --no-pager
# Tìm ERROR messages
```

### Port 8765 bị chiếm

```bash
ss -tlnp | grep 8765
# Kill process cũ:
kill $(lsof -ti :8765)
sudo systemctl restart rush-oracle
```

### YouTube stream URL hết hạn

```bash
rm ~/oracle/.url_cache.json
sudo systemctl restart rush-oracle
```

### GPU not detected (CUDA available: False)

PyTorch compiled CUDA version phải <= driver CUDA version:

```bash
nvidia-smi | grep "CUDA Version"  # driver supports up to this version
python3 -c "import torch; print(torch.version.cuda)"  # torch compiled with this

# Nếu torch CUDA > driver CUDA, reinstall:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### yt-dlp warning "No supported JavaScript runtime"

Stream vẫn hoạt động, warning này không ảnh hưởng. Nếu muốn fix:

```bash
pip install deno-vm
```

### WebSocket timeout khi connect

Nếu dùng yolov8x trên CPU → bị block event loop. Giải pháp:
- Đảm bảo GPU hoạt động (xem mục GPU above)
- Hoặc đổi sang model nhẹ hơn: `YOLO_MODEL=yolov8n.pt` trong `.env`

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  systemd: rush-oracle                           │
│  └─ run.sh                                      │
│     ├─ stream_server.py (WebSocket :8765)       │
│     │  ├─ yt-dlp → YouTube live stream URL      │
│     │  ├─ OpenCV → read video frames            │
│     │  ├─ YOLOv8x → detect vehicles (GPU)      │
│     │  └─ WebSocket → broadcast frames + count  │
│     │                                           │
│     └─ watchdog.py                              │
│        └─ round_manager_rush.py                 │
│           ├─ Create market on-chain (BSC)       │
│           ├─ Start counting (WS → stream_server)│
│           ├─ Wait ROUND_DURATION seconds        │
│           ├─ Resolve market on-chain            │
│           ├─ POST result to LEDGER_URL          │
│           ├─ Publish to Ably (real-time)        │
│           └─ Next round (alternate cameras)     │
└─────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
   ws://0.0.0.0:8765        BSC Testnet
   (port 30010 public)      (on-chain markets)
```

## WebSocket Protocol

### Client receives:

```jsonc
// 1. On connect — init message
{"type": "init", "state": "counting", "count": 5, "cameraId": "peace-bridge",
 "marketAddress": "0x...", "roundId": 1, "duration": 300, "remaining": 262.4}

// 2. During round — count updates (JSON)
{"type": "count", "state": "counting", "count": 7, "marketAddress": "0x...", "remaining": 200.1}

// 3. Video frames — binary (JPEG, ~60KB each)
<binary data>

// 4. Idle between rounds
{"type": "idle", "state": "waiting"}

// 5. Round complete
{"type": "round_complete", "count": 42, "marketAddress": "0x...", "result": "under"}
```

### Client can send:

```jsonc
{"type": "ping"}                    // → receives {"type": "pong"}
{"type": "start_round", ...}       // (round_manager only)
{"type": "stop_round"}             // (round_manager only)
```

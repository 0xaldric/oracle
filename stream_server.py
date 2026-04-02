"""
SinalBet Oracle — Live YOLO Streaming Server (v3 — Supervision SOTA)

State-of-the-art vehicle counting using:
  - YOLO12x for detection (attention-centric architecture)
  - BoT-SORT for multi-object tracking (persistent IDs)
  - Supervision LineZone with minimum_crossing_threshold (no jitter double-counts)
  - DetectionsSmoother for temporal stability
  - Per-class counting (car, motorcycle, bus, truck)

Usage:
    python stream_server.py --stream "https://...m3u8" --duration 300
    python stream_server.py --camera caltrans-100 --duration 300

Frontend connects to ws://localhost:8765
"""

import argparse
import asyncio
import hashlib
import json
import os
import queue
import sys
import time
import subprocess
import threading

import faulthandler
faulthandler.enable()

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import supervision as sv
    import websockets
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Run: pip install ultralytics supervision websockets opencv-python")
    sys.exit(1)


# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
VEHICLE_NAMES = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}

# Colors
NEON_GREEN = (136, 255, 0)
NEON_YELLOW = (0, 204, 255)

# Output frame width — higher = better detection but larger JPEG
OUTPUT_WIDTH = int(os.environ.get('OUTPUT_WIDTH', '1280'))


class VehicleCounter:
    """Vehicle counter: ID tracking + line crossing confirmation.

    - Red box = detected, tracking, hasn't crossed line yet
    - Green box = crossed the line, counted
    - Green line on the road = counting trigger
    - Counts both directions (in + out)
    - Never recounts same ID
    """

    def __init__(self, model_name='yolo12x.pt', confidence=0.20,
                 line_position=0.45, line_angle=10, line_points=None,
                 count_mode='line', min_frames=5, lanes=None, **_kwargs):
        self.count_mode = count_mode
        self.min_frames = min_frames
        self.seen_frames = {}
        self.lanes_config = lanes  # list of lane dicts from cameras.json
        self.lanes = []            # processed lanes (set on first frame)
        self.lanes_ready = False
        self.model = YOLO(model_name)

        # Round/source context — set by StreamServer._start_round()
        self._round_id = 0
        self._source_id = ""
        # Callback fired on each individual crossing — StreamServer broadcasts it
        self._on_vehicle_counted = None
        self.confidence = confidence
        self.line_position = line_position  # 0-1, where on x-axis the line center is
        self.line_angle = line_angle
        self.custom_line_points = line_points    # "x1,y1,x2,y2" for line 1
        self.custom_line_points2 = _kwargs.get('line_points2')  # optional line 2
        self.line2_start = None
        self.line2_end = None

        # Tracking state
        self.prev_pos = {}        # tid -> last (cx, cy)
        self.counted_ids = set()  # IDs that crossed the line
        self.counted_number = {}  # tid -> sequential count number (#1, #2, #3...)
        self._last_seen = {}      # tid -> frame_index when last observed
        self._frame_count = 0
        self._PRUNE_INTERVAL = 100  # prune every ~8s at 13fps
        self._PRUNE_AGE = 90        # remove IDs not seen for ~7s
        self.total_count = 0
        self.count_in = 0         # crossed left-to-right
        self.count_out = 0        # crossed right-to-left

        # Anti-double-count: recent crossing positions with timestamps
        # If a new ID crosses within DEDUP_RADIUS pixels of a recent crossing, skip it
        self._recent_crossings: list[tuple[int, int, float]] = []  # (cx, cy, time)
        self.DEDUP_RADIUS = 35     # pixels — must be far enough from recent crossing
        self.DEDUP_WINDOW = 1.5    # seconds — how long to remember a crossing

        # Per-class counts
        self.class_counts = {2: 0, 3: 0, 5: 0, 7: 0}

        # Line endpoints (set on first frame)
        self.line_start = None    # (x, y) top point
        self.line_end = None      # (x, y) bottom point

        # Annotators
        self.trace_annotator = sv.TraceAnnotator(
            thickness=1, trace_length=20,
            color=sv.Color.from_hex("#00ff88")
        )
        self.smoother = sv.DetectionsSmoother(length=3)

    def _is_duplicate_crossing(self, cx: int, cy: int) -> bool:
        """Check if a crossing at (cx, cy) is too close to a recent crossing."""
        import time as _time
        now = _time.time()
        # Clean old entries
        self._recent_crossings = [
            (x, y, t) for x, y, t in self._recent_crossings
            if now - t < self.DEDUP_WINDOW
        ]
        # Check distance to all recent crossings
        for rx, ry, _ in self._recent_crossings:
            dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
            if dist < self.DEDUP_RADIUS:
                return True  # Too close — likely same vehicle with new ID
        # Not a duplicate — record this crossing
        self._recent_crossings.append((cx, cy, now))
        return False

    def _prune_stale_tracks(self):
        """Remove tracker IDs not seen recently from tracking dicts.
        Prevents unbounded memory growth that leads to SIGSEGV in C++ tracker."""
        cutoff = self._frame_count - self._PRUNE_AGE
        stale = [tid for tid, last in self._last_seen.items() if last < cutoff]
        for tid in stale:
            self.seen_frames.pop(tid, None)
            self.prev_pos.pop(tid, None)
            self.counted_number.pop(tid, None)
            del self._last_seen[tid]
        # NEVER prune counted_ids — must persist for recount prevention
        if stale:
            print(f"[Prune] Removed {len(stale)} stale IDs (frame {self._frame_count})")

    def _merge_overlapping(self, detections):
        """Remove smaller detections contained inside larger ones.
        Handles: truck cab+trailer overlap, AND cars on car carriers (cegonhas).
        If a small box is mostly inside a big box, it's cargo, not a vehicle."""
        if len(detections) < 2 or detections.tracker_id is None:
            return detections

        keep = [True] * len(detections)
        areas = []
        for i in range(len(detections)):
            a = (detections.xyxy[i][2] - detections.xyxy[i][0]) * (detections.xyxy[i][3] - detections.xyxy[i][1])
            areas.append(a)

        for i in range(len(detections)):
            if not keep[i]:
                continue
            for j in range(len(detections)):
                if i == j or not keep[j]:
                    continue

                # Check if j is inside i (smaller inside larger)
                if areas[j] >= areas[i]:
                    continue

                # Calculate how much of j is inside i
                xi1 = max(detections.xyxy[i][0], detections.xyxy[j][0])
                yi1 = max(detections.xyxy[i][1], detections.xyxy[j][1])
                xi2 = min(detections.xyxy[i][2], detections.xyxy[j][2])
                yi2 = min(detections.xyxy[i][3], detections.xyxy[j][3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

                # If 50%+ of the smaller box is inside the larger = cargo, not vehicle
                containment = inter / areas[j] if areas[j] > 0 else 0
                if containment > 0.5:
                    keep[j] = False

        mask = np.array(keep)
        return detections[mask]

    def _setup_lanes(self, w, h):
        """Initialize lane zones and lines from config."""
        if not self.lanes_config:
            return
        for lc in self.lanes_config:
            # Parse zone polygon (x1,y1,x2,y2,...) as fractions
            zone_pts = lc["zone"]
            polygon = []
            for i in range(0, len(zone_pts), 2):
                polygon.append([int(zone_pts[i] * w), int(zone_pts[i+1] * h)])
            polygon = np.array(polygon)

            # Parse line
            lp = [float(p) for p in lc["line"].split(",")]
            line_start = (int(lp[0] * w), int(lp[1] * h))
            line_end = (int(lp[2] * w), int(lp[3] * h))

            self.lanes.append({
                "name": lc["name"],
                "direction": lc["direction"],
                "polygon": polygon,
                "line_start": line_start,
                "line_end": line_end,
                "count": 0,
            })
            print(f"[Lane] {lc['name']}: zone={polygon.tolist()}, line={line_start}->{line_end}")
        self.lanes_ready = True

    def _point_in_polygon(self, px, py, polygon):
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _cross_product_sign(self, px, py, ax, ay, bx, by):
        """Which side of line AB is point P? Positive = left, Negative = right."""
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    def process_frame(self, frame):
        """Detect, track, count on angled line crossing, annotate."""
        h, w = frame.shape[:2]
        self._frame_count += 1
        import math

        # Setup lanes on first frame
        if not self.lanes_ready and self.lanes_config:
            self._setup_lanes(w, h)

        # Set line endpoints on first frame
        if self.line_start is None:
            if self.custom_line_points:
                # Custom line: "x1,y1,x2,y2" as fractions of w,h
                parts = [float(p) for p in self.custom_line_points.split(",")]
                self.line_start = (int(parts[0] * w), int(parts[1] * h))
                self.line_end = (int(parts[2] * w), int(parts[3] * h))
            else:
                cx = int(w * self.line_position)
                angle_rad = math.radians(self.line_angle)
                top_y = 10
                bot_y = h - 10
                span = bot_y - top_y
                dx = int(span * math.tan(angle_rad))
                self.line_start = (cx - dx // 2, top_y)
                self.line_end = (cx + dx // 2, bot_y)
            # Second line (optional)
            if self.custom_line_points2:
                p2 = [float(p) for p in self.custom_line_points2.split(",")]
                self.line2_start = (int(p2[0] * w), int(p2[1] * h))
                self.line2_end = (int(p2[2] * w), int(p2[3] * h))
                print(f"[Counter] Line 1: {self.line_start} -> {self.line_end}")
                print(f"[Counter] Line 2: {self.line2_start} -> {self.line2_end}")
            else:
                print(f"[Counter] Line: {self.line_start} -> {self.line_end}")

        lx1, ly1 = self.line_start
        lx2, ly2 = self.line_end

        # Detect + track
        _t_det = time.monotonic()
        results = self.model.track(
            frame, verbose=False, conf=self.confidence,
            classes=VEHICLE_CLASSES, persist=True,
            tracker="bytetrack.yaml", imgsz=640, device=0
        )[0]
        _t_det_done = time.monotonic()

        detections = sv.Detections.from_ultralytics(results)

        if detections.class_id is not None and len(detections) > 0:
            mask = np.isin(detections.class_id, VEHICLE_CLASSES)
            detections = detections[mask]

        has_tracker = (detections.tracker_id is not None and
                       len(detections) > 0 and
                       any(t is not None for t in detections.tracker_id))

        if has_tracker:
            detections = self.smoother.update_with_detections(detections)
            detections = self._merge_overlapping(detections)

            if self.lanes_ready and self.lanes:
                # ─── Lane-based counting with bottom-center anchor ───
                for i in range(len(detections)):
                    tid = detections.tracker_id[i]
                    if tid is None or tid in self.counted_ids:
                        continue
                    x1, y1, x2, y2 = detections.xyxy[i]
                    # Bottom-center anchor = better ground-plane position
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)
                    cls = detections.class_id[i] if detections.class_id is not None else 2

                    # Track must persist min_frames before counting
                    self.seen_frames[tid] = self.seen_frames.get(tid, 0) + 1
                    if self.seen_frames[tid] < self.min_frames:
                        # Still store position for line-crossing detection
                        for lane in self.lanes:
                            ls, le = lane["line_start"], lane["line_end"]
                            side = self._cross_product_sign(cx, cy, ls[0], ls[1], le[0], le[1])
                            self.prev_pos[tid] = (cx, cy, side)
                        continue

                    for lane in self.lanes:
                        # Check if vehicle anchor is in this lane's zone
                        if not self._point_in_polygon(cx, cy, lane["polygon"]):
                            continue

                        # Check if crosses this lane's line
                        ls = lane["line_start"]
                        le = lane["line_end"]
                        side = self._cross_product_sign(cx, cy, ls[0], ls[1], le[0], le[1])

                        if tid in self.prev_pos and len(self.prev_pos[tid]) > 2:
                            prev_side = self.prev_pos[tid][2]
                            if prev_side is not None and prev_side * side < 0:
                                if not self._is_duplicate_crossing(cx, cy):
                                    self.counted_ids.add(tid)
                                    self.total_count += 1
                                    self.counted_number[tid] = self.total_count
                                    lane["count"] += 1
                                    direction = lane["direction"]
                                    if direction == "toward":
                                        self.count_in += 1
                                    else:
                                        self.count_out += 1
                                    self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
                                    # Emit discrete vehicle_counted event
                                    if self._on_vehicle_counted:
                                        self._on_vehicle_counted({
                                            "type": "vehicle_counted",
                                            "roundId": self._round_id,
                                            "sourceId": self._source_id,
                                            "trackId": int(tid),
                                            "lineId": lane["name"],
                                            "count": self.total_count,
                                            "seq": self._frame_count,
                                            "timestamp": time.time(),
                                            "classId": int(cls),
                                            "direction": direction,
                                        })
                                else:
                                    self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
                                break  # counted, don't check other lanes

                    # Store per-lane side for the first lane the vehicle is in
                    for lane in self.lanes:
                        ls, le = lane["line_start"], lane["line_end"]
                        if self._point_in_polygon(cx, cy, lane["polygon"]):
                            side = self._cross_product_sign(cx, cy, ls[0], ls[1], le[0], le[1])
                            self.prev_pos[tid] = (cx, cy, side)
                            break
                    else:
                        # Not in any lane — still track position with first lane's line
                        ls, le = self.lanes[0]["line_start"], self.lanes[0]["line_end"]
                        side = self._cross_product_sign(cx, cy, ls[0], ls[1], le[0], le[1])
                        self.prev_pos[tid] = (cx, cy, side)

            elif self.count_mode == 'uid':
                # ─── Unique ID mode: count every vehicle seen min_frames+ ───
                # Uses bottom-center anchor and tracks movement direction
                # to filter out stationary/phantom detections.
                for i in range(len(detections)):
                    tid = detections.tracker_id[i]
                    if tid is None:
                        continue
                    x1, y1, x2, y2 = detections.xyxy[i]
                    # Bottom-center = better ground-plane anchor than centroid
                    anchor_x = int((x1 + x2) / 2)
                    anchor_y = int(y2)
                    cls = detections.class_id[i] if detections.class_id is not None else 2

                    self.seen_frames[tid] = self.seen_frames.get(tid, 0) + 1

                    # Track position history for direction validation
                    if tid not in self.prev_pos:
                        self.prev_pos[tid] = (anchor_x, anchor_y)

                    if tid not in self.counted_ids and self.seen_frames[tid] >= self.min_frames:
                        # Check that the vehicle actually moved (not a parked/phantom detection)
                        first_pos = self.prev_pos[tid]
                        dx = abs(anchor_x - first_pos[0])
                        dy = abs(anchor_y - first_pos[1])
                        if dx + dy >= 15 and not self._is_duplicate_crossing(anchor_x, anchor_y):
                            self.counted_ids.add(tid)
                            self.total_count += 1
                            self.counted_number[tid] = self.total_count
                            self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
                            if self._on_vehicle_counted:
                                self._on_vehicle_counted({
                                    "type": "vehicle_counted",
                                    "roundId": self._round_id,
                                    "sourceId": self._source_id,
                                    "trackId": int(tid),
                                    "lineId": "uid",
                                    "count": self.total_count,
                                    "seq": self._frame_count,
                                    "timestamp": time.time(),
                                    "classId": int(cls),
                                    "direction": "in",
                                })
            else:
                # ─── Line crossing mode ───
                for i in range(len(detections)):
                    tid = detections.tracker_id[i]
                    if tid is None:
                        continue
                    x1, y1, x2, y2 = detections.xyxy[i]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cls = detections.class_id[i] if detections.class_id is not None else 2

                    side1 = self._cross_product_sign(cx, cy, lx1, ly1, lx2, ly2)
                    side2 = None
                    if self.line2_start:
                        side2 = self._cross_product_sign(cx, cy,
                            self.line2_start[0], self.line2_start[1],
                            self.line2_end[0], self.line2_end[1])

                    if tid not in self.counted_ids and tid in self.prev_pos:
                        ps1 = self.prev_pos[tid][2]
                        ps2 = self.prev_pos[tid][3] if len(self.prev_pos[tid]) > 3 else None
                        crossed = False
                        # Check line 1
                        if ps1 is not None and ps1 * side1 < 0:
                            crossed = True
                        # Check line 2
                        if not crossed and side2 is not None and ps2 is not None and ps2 * side2 < 0:
                            crossed = True

                        if crossed and not self._is_duplicate_crossing(cx, cy):
                            self.counted_ids.add(tid)
                            self.total_count += 1
                            self.counted_number[tid] = self.total_count
                            direction = "in" if side1 > 0 else "out"
                            if side1 > 0:
                                self.count_in += 1
                            else:
                                self.count_out += 1
                            self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
                            crossed_line = "line1" if (ps1 is not None and ps1 * side1 < 0) else "line2"
                            if self._on_vehicle_counted:
                                self._on_vehicle_counted({
                                    "type": "vehicle_counted",
                                    "roundId": self._round_id,
                                    "sourceId": self._source_id,
                                    "trackId": int(tid),
                                    "lineId": crossed_line,
                                    "count": self.total_count,
                                    "seq": self._frame_count,
                                    "timestamp": time.time(),
                                    "classId": int(cls),
                                    "direction": direction,
                                })

                    self.prev_pos[tid] = (cx, cy, side1, side2)

            # Update last-seen timestamps for active tracker IDs
            for i in range(len(detections)):
                tid = detections.tracker_id[i]
                if tid is not None:
                    self._last_seen[tid] = self._frame_count

        # Periodic pruning of stale tracker IDs to prevent memory bloat
        if self._frame_count % self._PRUNE_INTERVAL == 0:
            self._prune_stale_tracks()

        if has_tracker:
            # Draw traces
            frame = self.trace_annotator.annotate(frame, detections)

            # Draw boxes: RED = tracking, GREEN = counted (with number)
            for i in range(len(detections)):
                tid = detections.tracker_id[i]
                if tid is None:
                    continue
                bx1, by1, bx2, by2 = detections.xyxy[i].astype(int)
                counted = tid in self.counted_ids
                color = (0, 255, 0) if counted else (0, 0, 255)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                # Count numbers removed — frontend handles display

        # ─── Draw counting line (only in line mode) ──────────

        # Draw lanes
        if self.lanes_ready and self.lanes:
            for lane in self.lanes:
                # Draw zone polygon (subtle)
                pts = lane["polygon"].reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (100, 100, 100), 1)
                # Draw counting line (green)
                cv2.line(frame, lane["line_start"], lane["line_end"], NEON_GREEN, 2)
                # Lane label
                lx, ly = lane["line_start"]
                cv2.putText(frame, f'{lane["name"]}: {lane["count"]}', (lx + 5, ly - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, NEON_GREEN, 1, cv2.LINE_AA)
        elif self.count_mode == 'line' and self.line_start:
            cv2.line(frame, self.line_start, self.line_end, NEON_GREEN, 2)
            if self.line2_start:
                cv2.line(frame, self.line2_start, self.line2_end, NEON_GREEN, 2)

        # HUD removed — frontend handles count/timer display

        # Debug timing (detect vs draw)
        if self._frame_count % 100 == 0:
            det_ms = (_t_det_done - _t_det) * 1000
            draw_ms = (time.monotonic() - _t_det_done) * 1000
            print(f"[Counter] detect={det_ms:.0f}ms draw={draw_ms:.0f}ms total={det_ms+draw_ms:.0f}ms")

        return frame, self.total_count


_URL_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".url_cache.json")
_URL_CACHE_TTL = 3600  # 1 hour — HLS URLs expire, force refresh sooner

def _load_url_cache():
    try:
        with open(_URL_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_url_cache(cache):
    try:
        with open(_URL_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass

def get_stream_url(youtube_url, force_refresh=False):
    """Extract direct stream URL using yt-dlp. Cached on disk to avoid rate limits."""
    import os

    if 'youtube.com' not in youtube_url and 'youtu.be' not in youtube_url:
        return youtube_url

    # Return cached URL if fresh (unless force refresh)
    cache = _load_url_cache()
    if not force_refresh and youtube_url in cache:
        cached_url, cached_ts = cache[youtube_url]
        age = time.time() - cached_ts
        if age < _URL_CACHE_TTL:
            print(f"[yt-dlp] Using cached URL (age {int(age)}s)")
            return cached_url

    if force_refresh:
        print(f"[yt-dlp] Force refresh for camera switch")

    deno_path = os.path.expanduser("~/.deno/bin")
    env = os.environ.copy()
    if deno_path not in env.get("PATH", ""):
        env["PATH"] = f"{deno_path}:{env.get('PATH', '')}"

    commands = [
        ['yt-dlp', '--remote-components', 'ejs:github', '-f', 'best[height<=720]', '-g', youtube_url],
        ['yt-dlp', '-f', 'best[height<=720]', '-g', youtube_url],
        ['yt-dlp', '-f', 'best', '-g', youtube_url],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            if result.returncode == 0:
                url = result.stdout.strip()
                print(f"[yt-dlp] OK: {url[:80]}...")
                cache[youtube_url] = [url, time.time()]
                _save_url_cache(cache)
                return url
        except FileNotFoundError:
            return youtube_url
        except subprocess.TimeoutExpired:
            continue

    # yt-dlp failed — return cached even if stale, better than nothing
    if youtube_url in cache:
        print("[yt-dlp] FAILED — using stale cached URL")
        return cache[youtube_url][0]

    return youtube_url


def _is_youtube_url(url):
    return 'youtube.com' in url or 'youtu.be' in url


class StreamlinkPipe:
    """Reads frames from a live stream using streamlink → ffmpeg pipe.

    Instead of yt-dlp extracting a URL and cv2.VideoCapture opening it,
    we pipe: streamlink (HLS fetch + buffer) → ffmpeg (decode to raw BGR) → numpy frames.

    Benefits:
    - streamlink handles HLS segment fetching, buffering, and reconnects internally
    - ffmpeg decodes reliably via pipe (no cv2 HLS quirks)
    - ~2min buffer gives smooth playback even with network hiccups
    - No stale URL issues — streamlink resolves URLs on its own
    """

    def __init__(self, stream_url, width=1280, quality='best'):
        self._stream_url = stream_url
        self._width = width
        self._quality = quality
        self._proc = None      # streamlink | ffmpeg pipeline
        self._sl_proc = None
        self._frame_size = None
        self._frame_h = 0
        self._frame_w = 0
        self._reader_running = False
        self._latest_frame = None
        self._frame_lock = None

    def _find_streamlink(self):
        """Find streamlink binary."""
        import shutil
        sl = shutil.which('streamlink')
        if sl:
            return sl
        # Check common local paths
        for p in [os.path.expanduser('~/.local/bin/streamlink'), '/usr/local/bin/streamlink']:
            if os.path.isfile(p):
                return p
        return None

    def start(self):
        """Start the streamlink → ffmpeg pipeline. Returns (width, height, fps)."""
        self.stop()

        sl_bin = self._find_streamlink()
        if sl_bin is None:
            raise RuntimeError("streamlink not found — install with: pip install streamlink")

        is_yt = _is_youtube_url(self._stream_url)

        # Step 1: Probe stream resolution with a quick streamlink → ffprobe
        print(f"[Streamlink] Probing stream: {self._stream_url[:60]}...")
        probe_w, probe_h, probe_fps = self._probe_stream(sl_bin, is_yt)
        if probe_w == 0:
            # Fallback defaults
            probe_w, probe_h, probe_fps = 1280, 720, 30
            print(f"[Streamlink] Probe failed, using defaults: {probe_w}x{probe_h} @ {probe_fps}fps")

        # Scale to output width
        if probe_w > self._width:
            scale_factor = self._width / probe_w
            out_w = self._width
            out_h = int(probe_h * scale_factor)
            # Ensure even dimensions for ffmpeg
            out_h = out_h if out_h % 2 == 0 else out_h + 1
        else:
            out_w = probe_w
            out_h = probe_h

        self._frame_w = out_w
        self._frame_h = out_h
        self._frame_size = out_w * out_h * 3  # BGR24

        # Step 2: Build streamlink → ffmpeg pipe
        # streamlink outputs raw MPEG-TS to stdout
        sl_cmd = [sl_bin, '--stdout']
        if is_yt:
            sl_cmd += ['--plugin-dirs', '/dev/null']  # use built-in YouTube plugin
        sl_cmd += [
            '--stream-segment-threads', '3',
            '--hls-live-edge', '6',         # buffer ~6 segments (~12-18s)
            '--retry-streams', '5',          # retry every 5s if stream dies
            '--retry-max', '0',              # retry forever
            '--retry-open', '5',             # retry opening 5 times
            self._stream_url,
            self._quality,
        ]

        # ffmpeg reads from stdin, outputs raw BGR24 frames to stdout
        ffmpeg_cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            '-i', 'pipe:0',                 # read from stdin (streamlink output)
            '-vf', f'scale={out_w}:{out_h}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            '-an',                           # drop audio
            'pipe:1',                        # output raw frames to stdout
        ]

        print(f"[Streamlink] Starting pipe: {out_w}x{out_h} @ {probe_fps}fps")
        print(f"[Streamlink] CMD: {' '.join(sl_cmd[:6])}... | ffmpeg ...")

        # Launch: streamlink stdout → ffmpeg stdin → our stdout
        sl_proc = subprocess.Popen(
            sl_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024*1024,  # 1MB buffer
        )

        ff_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=sl_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 2,  # buffer 2 frames
        )

        # Allow streamlink to receive SIGPIPE if ffmpeg dies
        sl_proc.stdout.close()

        self._sl_proc = sl_proc
        self._proc = ff_proc

        # Start internal drain thread to prevent pipe backpressure
        self._start_reader_thread()

        print(f"[Streamlink] Pipeline started (sl pid={sl_proc.pid}, ff pid={ff_proc.pid})")
        return out_w, out_h, probe_fps

    def _probe_stream(self, sl_bin, is_yt):
        """Quick probe to get stream dimensions and FPS."""
        try:
            sl_cmd = [sl_bin, '--stdout']
            if is_yt:
                sl_cmd += ['--plugin-dirs', '/dev/null']
            sl_cmd += [self._stream_url, self._quality]

            probe_cmd = [
                'ffprobe',
                '-loglevel', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'csv=p=0',
                '-i', 'pipe:0',
            ]

            sl = subprocess.Popen(sl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ff = subprocess.Popen(probe_cmd, stdin=sl.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sl.stdout.close()

            out, _ = ff.communicate(timeout=30)
            sl.terminate()
            sl.wait(timeout=5)

            if out:
                # ffprobe may output multiple lines — take the last valid one
                import re
                text = out.decode().strip()
                # Match pattern like "1920,1080,30/1" or "1280,720,25"
                match = re.search(r'(\d+),(\d+),(\d+(?:/\d+)?)', text)
                if match:
                    w = int(match.group(1))
                    h = int(match.group(2))
                    fps_str = match.group(3)
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = int(num) / max(int(den), 1)
                    else:
                        fps = float(fps_str)
                    print(f"[Streamlink] Probed: {w}x{h} @ {fps:.1f}fps")
                    return w, h, fps
        except Exception as e:
            print(f"[Streamlink] Probe error: {e}")

        return 0, 0, 30

    def _start_reader_thread(self):
        """Start internal reader thread that continuously drains the pipe.

        This prevents pipe buffer backpressure — the reader always consumes
        frames as fast as ffmpeg produces them, keeping only the latest frame.
        Without this, the pipe buffer fills up, ffmpeg blocks, streamlink blocks,
        and the entire chain deadlocks.
        """
        import threading

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._reader_running = True

        def _drain():
            while self._reader_running:
                if self._proc is None or self._proc.poll() is not None:
                    time.sleep(0.1)
                    continue
                try:
                    raw = self._proc.stdout.read(self._frame_size)
                    if len(raw) == self._frame_size:
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                            (self._frame_h, self._frame_w, 3)
                        ).copy()
                        with self._frame_lock:
                            self._latest_frame = frame
                    else:
                        # Pipe broken or partial read
                        time.sleep(0.1)
                except Exception:
                    time.sleep(0.1)

        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        self._drain_thread = t

    def read(self):
        """Read the latest BGR frame. Returns (success, frame) like cv2.VideoCapture.read().

        Non-blocking — returns the most recent frame from the internal drain thread.
        The drain thread continuously reads from the pipe to prevent backpressure.
        """
        if self._proc is None or self._proc.poll() is not None:
            return False, None

        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None  # consume it

        if frame is not None:
            return True, frame
        return False, None

    def stop(self):
        """Kill the pipeline."""
        self._reader_running = False
        for proc_attr in ('_proc', '_sl_proc'):
            proc = getattr(self, proc_attr, None)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                setattr(self, proc_attr, None)

    def isOpened(self):
        return self._proc is not None and self._proc.poll() is None

    @property
    def width(self):
        return self._frame_w

    @property
    def height(self):
        return self._frame_h


class CloudflareBroadcaster:
    """Re-streams YOLO-annotated frames to Cloudflare Stream via ffmpeg RTMPS.

    Runs ffmpeg as a subprocess, piping raw BGR frames to stdin.
    ffmpeg encodes to H.264 and pushes to Cloudflare Live Input.

    Usage:
        Set these env vars:
          CF_STREAM_ENABLED=true
          CF_RTMPS_URL=rtmps://live.cloudflare.com:443/live/
          CF_STREAM_KEY=<your-stream-key>
          CF_VIDEO_UID=<your-video-uid>
    """

    def __init__(self, width=1920, height=1080, fps=8):
        self.enabled = os.environ.get('CF_STREAM_ENABLED', '').lower() in ('true', '1', 'yes')
        self.rtmps_url = os.environ.get('CF_RTMPS_URL', 'rtmps://live.cloudflare.com:443/live/')
        self.stream_key = os.environ.get('CF_STREAM_KEY', '')
        self.video_uid = os.environ.get('CF_VIDEO_UID', '')
        self.width = width
        self.height = height
        self.fps = fps
        # CF pipe uses smaller resolution for faster writes (1.55MB vs 2.7MB per frame)
        self._cf_w = 960
        self._cf_h = 540
        self._cf_fps = 45
        self._proc = None
        self._frame_count = 0
        self._writer_thread = None
        self._alive = False
        # Large queue: YOLO pushes ~6fps, CF writer reads smoothly.
        # CF can be up to 5 min behind real-time but always smooth.
        self._frame_q = queue.Queue(maxsize=1800)  # ~5 min at 6fps YOLO

        if self.enabled and not self.stream_key:
            print("[CF] WARNING: CF_STREAM_ENABLED=true but CF_STREAM_KEY is empty — disabling")
            self.enabled = False

        if self.enabled:
            print(f"[CF] Cloudflare Stream broadcast enabled")
            print(f"[CF]   RTMPS: {self.rtmps_url}")
            print(f"[CF]   Video UID: {self.video_uid}")

    def _check_nvenc(self):
        """Check if NVENC hardware encoder is available."""
        try:
            r = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
            return 'h264_nvenc' in r.stdout
        except Exception:
            return False

    def start(self):
        """Start ffmpeg subprocess for RTMPS streaming."""
        if not self.enabled:
            return
        self._stop_proc()

        cw, ch, cfps = self._cf_w, self._cf_h, self._cf_fps
        rtmps_dest = f"{self.rtmps_url}{self.stream_key}"
        # Try NVENC (GPU encoding) first, fall back to libx264
        use_nvenc = self._check_nvenc()
        if use_nvenc:
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{cw}x{ch}',
                '-r', str(cfps),
                '-i', '-',
                '-c:v', 'h264_nvenc',       # GPU encoding
                '-preset', 'p1',            # fastest NVENC preset
                '-tune', 'll',              # low latency
                '-rc', 'cbr',               # constant bitrate
                '-pix_fmt', 'yuv420p',
                '-g', str(cfps * 2),
                '-b:v', '500k',
                '-maxrate', '600k',
                '-bufsize', '300k',         # smaller buffer = lower latency
                '-f', 'flv',
                rtmps_dest,
            ]
            print(f"[CF] Using NVENC (GPU) encoding @ {cw}x{ch} {cfps}fps")
        else:
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{cw}x{ch}',
                '-r', str(cfps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-g', str(cfps * 2),
                '-b:v', '500k',
                '-maxrate', '600k',
                '-bufsize', '300k',
                '-f', 'flv',
                rtmps_dest,
            ]
            print(f"[CF] Using libx264 (CPU) encoding @ {cw}x{ch} {cfps}fps")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._frame_count = 0
            print(f"[CF] ffmpeg started (pid={self._proc.pid})")

            # Increase pipe buffer to reduce write blocking
            try:
                import fcntl
                F_SETPIPE_SZ = 1031
                pipe_sz = fcntl.fcntl(self._proc.stdin.fileno(), F_SETPIPE_SZ, 1048576)
                print(f"[CF] pipe buffer set to {pipe_sz} bytes")
            except Exception as e:
                print(f"[CF] pipe buffer resize failed (ok): {e}")

            # Writer thread: pops YOLO frames from queue, repeats each to fill 30fps.
            # Uses monotonic clock to maintain exact 30fps pace.
            # write() may block (pipe full) — that's fine, clock catches up after.
            # Queue buffers ~30s of YOLO frames for resilience.
            # Queue-based writer with fractional rate matching.
            # Main loop pushes ~12fps. Writer outputs 45fps.
            # Uses a float accumulator to pop frames at exact input rate.
            # pop_rate adapts to keep queue at TARGET_Q depth.
            # Result: perfectly smooth 45fps, content updates ~12fps evenly.
            self._alive = True
            TARGET_Q = 1080      # target queue depth (~3 min at 6fps YOLO)
            MIN_Q_START = 360    # don't start playing until queue has 1 min buffer
            def _cf_writer():
                proc = self._proc
                fq = self._frame_q
                interval = 1.0 / cfps
                current = None
                frame_n = 0
                new_count = 0
                hold_count = 0
                # pop_rate: fraction of frames popped per write.
                # 6fps YOLO input / 45fps output = 0.133 (pop 1 every ~7.5 writes)
                pop_rate = 6.0 / cfps
                pop_accum = 0.0
                next_time = time.monotonic()
                last_log = time.monotonic()
                last_adjust = time.monotonic()
                buffering = True

                try:
                    # Phase 1: Buffer — wait until queue has MIN_Q_START frames
                    print(f"[CF Writer] buffering... waiting for {MIN_Q_START} frames")
                    while proc and proc.poll() is None and self._alive and buffering:
                        qs = fq.qsize()
                        if qs >= MIN_Q_START:
                            buffering = False
                            print(f"[CF Writer] buffer ready ({qs} frames), starting playback")
                            break
                        # Feed a black/hold frame to keep ffmpeg alive
                        if current is None:
                            try:
                                current = fq.get(timeout=1)
                                new_count += 1
                            except queue.Empty:
                                continue
                        now = time.monotonic()
                        wait = next_time - now
                        if wait > 0:
                            time.sleep(wait)
                        next_time += interval
                        if time.monotonic() - next_time > 1.0:
                            next_time = time.monotonic()
                        try:
                            proc.stdin.write(current)
                            frame_n += 1
                        except (BrokenPipeError, IOError) as e:
                            print(f"[CF Writer] pipe error during buffer: {e}")
                            break
                        # Log during buffering
                        now_m = time.monotonic()
                        if now_m - last_log >= 5.0:
                            last_log = now_m
                            print(f"[CF Writer] buffering... q={qs}/{MIN_Q_START}")

                    # Phase 2: Playback — smooth adaptive consumption
                    next_time = time.monotonic()
                    while proc and proc.poll() is None and self._alive:
                        # Fractional pop: accumulate and pop when >= 1.0
                        pop_accum += pop_rate
                        if pop_accum >= 1.0:
                            pop_accum -= 1.0
                            try:
                                current = fq.get_nowait()
                                new_count += 1
                            except queue.Empty:
                                pop_accum = 0.0
                                hold_count += 1
                                if current is None:
                                    time.sleep(0.03)
                                    next_time = time.monotonic()
                                    continue
                        elif current is None:
                            try:
                                current = fq.get(timeout=0.1)
                                new_count += 1
                            except queue.Empty:
                                continue
                        # Wait until next frame time
                        now = time.monotonic()
                        wait = next_time - now
                        if wait > 0:
                            time.sleep(wait)
                        next_time += interval
                        if time.monotonic() - next_time > 1.0:
                            next_time = time.monotonic()
                        try:
                            proc.stdin.write(current)
                            frame_n += 1
                        except (BrokenPipeError, IOError) as e:
                            print(f"[CF Writer] pipe error: {e}")
                            break
                        # Adapt pop_rate every 2s based on queue depth
                        now_m = time.monotonic()
                        if now_m - last_adjust >= 2.0:
                            last_adjust = now_m
                            qs = fq.qsize()
                            if qs > TARGET_Q + 30:
                                pop_rate = min(0.5, pop_rate + 0.005)   # consume faster
                            elif qs < TARGET_Q - 30:
                                pop_rate = max(0.05, pop_rate - 0.005)  # consume slower
                        # Debug every 10s
                        if now_m - last_log >= 10.0:
                            last_log = now_m
                            qs = fq.qsize()
                            ufps = new_count / 10.0
                            print(f"[CF Writer] frames={frame_n} new={new_count}({ufps:.1f}fps) hold={hold_count} q={qs} rate={pop_rate:.3f}")
                            new_count = 0
                            hold_count = 0
                except Exception as e:
                    print(f"[CF Writer] FATAL: {e}")
                print(f"[CF Writer] exiting (wrote {frame_n} frames, q={fq.qsize()})")
            self._writer_thread = threading.Thread(target=_cf_writer, daemon=True, name="cf-writer")
            self._writer_thread.start()
        except FileNotFoundError:
            print("[CF] ERROR: ffmpeg not found! Install with: apt install ffmpeg")
            self.enabled = False
        except Exception as e:
            print(f"[CF] ERROR starting ffmpeg: {e}")
            self.enabled = False

    def send_frame(self, frame):
        """Push a YOLO-annotated frame into the queue. Never blocks main loop.

        Writer thread pops frames and repeats each ~10x to fill 30fps.
        Queue buffers ~30s so stdin blocking never causes stream stutter.
        """
        if not self.enabled or self._proc is None:
            return
        # Auto-restart if ffmpeg or writer thread died
        if self._proc.poll() is not None:
            print(f"[CF] ffmpeg exited (code={self._proc.returncode}), restarting...")
            self.start()
            if not self.enabled or self._proc is None:
                return
        if self._writer_thread is not None and not self._writer_thread.is_alive():
            print("[CF] writer thread died, restarting...")
            self.start()
            if not self.enabled or self._proc is None:
                return

        h, w = frame.shape[:2]
        if w != self._cf_w or h != self._cf_h:
            frame = cv2.resize(frame, (self._cf_w, self._cf_h),
                               interpolation=cv2.INTER_LINEAR)
        data = frame.tobytes()
        try:
            self._frame_q.put_nowait(data)
        except queue.Full:
            pass  # queue full — oldest frames stay, newest dropped
        self._frame_count += 1

    def stop(self):
        """Stop ffmpeg subprocess."""
        self._stop_proc()
        if self._frame_count > 0:
            print(f"[CF] Stopped after {self._frame_count} frames")

    def _stop_proc(self):
        self._alive = False
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


class StreamServer:
    """Persistent WebSocket server that streams processed video frames.

    Runs continuously — never exits on round end.
    Round lifecycle controlled via WS messages (start_round / stop_round).

    States:
      IDLE      → streaming video, no counting
      COUNTING  → streaming video + counting vehicles
    """

    # Maximum number of evidence frame sets to keep on disk
    MAX_EVIDENCE_DIRS = 100
    # Interval (seconds) between evidence frame captures
    EVIDENCE_INTERVAL = 30
    # JPEG quality for evidence frames
    EVIDENCE_JPEG_QUALITY = 80

    # States
    STATE_IDLE = "idle"
    STATE_COUNTING = "counting"

    def __init__(self, stream_url, host='0.0.0.0', port=8765,
                 model='yolo12x.pt', confidence=0.10, line_pos=0.5,
                 line_angle=10, line_points=None, line_points2=None,
                 count_mode='uid', lanes=None, target_fps=8, camera_id='',
                 roi=None, **kwargs):
        self.stream_url = stream_url
        self.host = host
        self.port = port
        self.target_fps = target_fps
        self.camera_id = camera_id
        self._roi_polygons = roi  # list of polygons (fractions 0-1), applied on first frame
        self._roi_mask = None     # numpy mask, created once per resolution

        # YOLO config — stored for creating fresh counters per round
        self._model_name = model
        self._confidence = confidence
        self._line_pos = line_pos
        self._line_angle = line_angle
        self._line_points = line_points
        self._line_points2 = line_points2
        self._count_mode = count_mode
        self._lanes = lanes

        # Create initial counter (for YOLO model loading + visual annotations in IDLE)
        self.counter = VehicleCounter(model_name=model, confidence=confidence,
                                       line_position=line_pos, line_angle=line_angle,
                                       line_points=line_points,
                                       line_points2=line_points2,
                                       count_mode=count_mode, lanes=lanes,
                                       min_frames=3)
        self.clients = set()
        self.running = False

        # Cloudflare Stream broadcast (re-stream YOLO frames via ffmpeg → RTMPS)
        self._cf = CloudflareBroadcaster(
            width=OUTPUT_WIDTH,
            height=0,    # will be set on first frame
            fps=target_fps,
        )

        # Round state — controlled by WS messages
        self._state = self.STATE_IDLE
        self._round_active = False
        self._round_start_time = 0.0
        self._round_duration = 300
        self._round_market = ''
        self._round_id = 0

        # Evidence
        self._init_evidence()

    def _build_roi_mask(self, h, w):
        """Create binary mask from ROI polygons. Called once on first frame."""
        if not self._roi_polygons:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        # Detect format: [[x,y],...] (single polygon) vs [[[x,y],...]] (multiple)
        roi = self._roi_polygons
        if roi and isinstance(roi[0], (int, float)):
            # Flat list — shouldn't happen but handle it
            roi = [roi]
        elif roi and isinstance(roi[0], list) and isinstance(roi[0][0], (int, float)):
            # Single polygon: [[x,y],[x,y],...] — wrap in list
            roi = [roi]
        # Now roi is [[[x,y],...], ...] — list of polygons
        for poly in roi:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        # Convert to 3-channel mask for bitwise_and
        self._roi_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        print(f"[ROI] Mask created: {w}x{h}, {len(self._roi_polygons)} polygon(s)")
        return self._roi_mask

    def apply_roi(self, frame):
        """Mask frame to ROI — pixels outside become black (YOLO ignores them)."""
        if self._roi_polygons is None:
            return frame
        if self._roi_mask is None or self._roi_mask.shape[:2] != frame.shape[:2]:
            self._build_roi_mask(frame.shape[0], frame.shape[1])
        if self._roi_mask is not None:
            return cv2.bitwise_and(frame, self._roi_mask)
        return frame

    def _init_evidence(self):
        """Called from __init__ — separated to avoid dead code after return."""
        self._evidence_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
        self._evidence_frames: list[str] = []
        self._evidence_hashes: list[str] = []
        self._evidence_final: str | None = None
        self._last_evidence_time: float = 0.0

    def _new_counter(self):
        """Create a fresh VehicleCounter — reuses the already-loaded YOLO model."""
        # min_frames=1 for crossing modes (cross-product sign flip is sufficient gate)
        # min_frames=3 only for uid mode (needs movement validation, no crossing line)
        mf = 3 if self._count_mode == 'uid' else 1
        c = VehicleCounter(model_name=self._model_name, confidence=self._confidence,
                           line_position=self._line_pos, line_angle=self._line_angle,
                           line_points=self._line_points, line_points2=self._line_points2,
                           count_mode=self._count_mode, lanes=self._lanes, min_frames=mf)
        c.model = self.counter.model  # reuse loaded model — no GPU reload
        return c

    def _switch_camera(self, new_camera_id):
        """Switch to a different camera. Updates stream URL, line points, ROI, and signals reader to reconnect."""
        from pathlib import Path
        cameras_path = Path(__file__).parent / "cameras.json"
        with open(cameras_path) as f:
            data = json.load(f)
        cam = None
        for c in data["cameras"]:
            if c["id"] == new_camera_id:
                cam = c
                break
        if not cam:
            print(f"[SWITCH] Camera '{new_camera_id}' not found — staying on {self.camera_id}")
            return
        old_id = self.camera_id
        self.camera_id = cam["id"]
        self.stream_url = cam.get("streamUrl", cam.get("imageUrl"))

        # Update line points
        self._line_points = cam.get("linePoints")
        self._line_points2 = cam.get("linePoints2")
        self._count_mode = "line" if self._line_points else "uid"
        self._lanes = cam.get("lanes")

        # Update ROI
        self._roi_polygons = cam.get("roi")
        self._roi_mask = None  # force rebuild on next frame

        # Signal reader thread to reconnect to new stream
        if hasattr(self, '_switch_event'):
            self._switch_event.set()

        print(f"[SWITCH] Camera: {old_id} -> {cam['id']} ({cam['name']})")
        if self._line_points:
            print(f"[SWITCH] Line: {self._line_points}")
        if self._line_points2:
            print(f"[SWITCH] Line2: {self._line_points2}")
        if self._roi_polygons:
            print(f"[SWITCH] ROI: {len(self._roi_polygons)} points")

    def _start_round(self, data):
        """Begin a new counting round. All state is clean."""
        # Check if we need to switch cameras
        requested_camera = data.get("cameraId", "")
        if requested_camera and requested_camera != self.camera_id:
            self._switch_camera(requested_camera)

        self._round_market = data.get("marketAddress", "")
        self._round_duration = data.get("duration", 300)
        self._round_id = data.get("roundId", 0)
        self._round_start_time = time.time()
        self._round_active = True
        self._state = self.STATE_COUNTING
        # Fresh counter — zero carryover
        self.counter = self._new_counter()
        # Wire round context and vehicle_counted callback into counter
        self.counter._round_id = self._round_id
        self.counter._source_id = self.camera_id
        self._pending_vehicle_events = []  # collect events from sync callback
        self.counter._on_vehicle_counted = lambda evt: self._pending_vehicle_events.append(evt)
        # Fresh evidence
        self._evidence_frames = []
        self._evidence_hashes = []
        self._evidence_final = None
        self._last_evidence_time = 0.0
        self._ensure_evidence_dir()
        print(f"[STATE] counting — market={self._round_market[:10]}... duration={self._round_duration}s round={self._round_id}")

    def _stop_round(self):
        """End current round. Write result, broadcast final, go IDLE."""
        if not self._round_active:
            return None
        count = self.counter.total_count
        result = {
            "count": count,
            "in_count": self.counter.class_counts.get(2, 0),
            "out_count": self.counter.class_counts.get(7, 0),
            "duration": self._round_duration,
            "stream": self.stream_url,
            "timestamp": int(time.time()),
            "marketAddress": self._round_market,
            "roundId": self._round_id,
            "evidence": {
                "frames": list(self._evidence_frames),
                "final_frame": self._evidence_final,
                "frame_hashes": list(self._evidence_hashes),
            },
        }
        # Write result file (unique per round to avoid stale reads)
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        self._round_active = False
        self._state = self.STATE_IDLE
        print(f"[STATE] idle — round {self._round_id} complete: {count} vehicles")
        return result

    MAX_CLIENTS = 20  # hard limit — prevents resource exhaustion

    async def register(self, ws):
        # Reject if too many clients
        if len(self.clients) >= self.MAX_CLIENTS:
            print(f"[WS] Rejected — max clients ({self.MAX_CLIENTS}) reached")
            await ws.close(1013, "max clients reached")
            return False
        self.clients.add(ws)
        print(f"[WS] Client connected ({len(self.clients)} total)")
        # Send current state
        init_msg = {
            "type": "init",
            "stream": self.stream_url,
            "state": self._state,
            "count": self.counter.total_count,
            "cameraId": self.camera_id,
        }
        # Include Cloudflare videoUid for broadcast fallback mode
        if self._cf.video_uid:
            init_msg["videoUid"] = self._cf.video_uid
        if self._round_active:
            init_msg["marketAddress"] = self._round_market
            init_msg["roundId"] = self._round_id
            init_msg["duration"] = self._round_duration
            elapsed = time.time() - self._round_start_time
            init_msg["elapsed"] = round(elapsed, 1)
            init_msg["remaining"] = max(0, round(self._round_duration - elapsed, 1))
        await ws.send(json.dumps(init_msg))
        return True

    async def unregister(self, ws):
        self.clients.discard(ws)
        print(f"[WS] Client disconnected ({len(self.clients)} total)")

    async def _broadcast_json(self, msg):
        """Send JSON-only message to all clients. Cleans up dead connections."""
        if not self.clients:
            return
        raw = json.dumps(msg) if isinstance(msg, dict) else msg
        dead = set()
        async def send_to(ws):
            try:
                await asyncio.wait_for(ws.send(raw), timeout=2)
            except Exception:
                dead.add(ws)
        await asyncio.gather(*(send_to(ws) for ws in list(self.clients)))
        if dead:
            self.clients -= dead
            for ws in dead:
                try:
                    await ws.close()
                except Exception:
                    pass

    def _ensure_evidence_dir(self) -> None:
        """Create evidence/ directory and clean up old evidence if needed."""
        os.makedirs(self._evidence_dir, exist_ok=True)
        # Clean up oldest evidence files if we exceed MAX_EVIDENCE_DIRS sets.
        # Evidence files are named {timestamp}_{elapsed}s.jpg or {timestamp}_final.jpg
        # Group by timestamp prefix and remove oldest groups.
        try:
            files = sorted(os.listdir(self._evidence_dir))
            if not files:
                return
            # Extract unique timestamp prefixes
            prefixes: list[str] = []
            seen: set[str] = set()
            for f in files:
                prefix = f.split("_")[0]
                if prefix not in seen:
                    seen.add(prefix)
                    prefixes.append(prefix)
            # Remove oldest groups if over limit
            while len(prefixes) > self.MAX_EVIDENCE_DIRS:
                old_prefix = prefixes.pop(0)
                for f in files:
                    if f.startswith(old_prefix):
                        try:
                            os.remove(os.path.join(self._evidence_dir, f))
                        except OSError:
                            pass
        except OSError:
            pass

    def _save_evidence_frame(self, annotated_frame, timestamp: int, elapsed: float, is_final: bool = False) -> None:
        """Save an annotated frame as JPEG evidence and record its SHA-256 hash."""
        if is_final:
            filename = f"{timestamp}_final.jpg"
        else:
            filename = f"{timestamp}_{int(elapsed)}s.jpg"

        filepath = os.path.join(self._evidence_dir, filename)
        rel_path = f"evidence/{filename}"

        success, jpeg_buf = cv2.imencode(
            '.jpg', annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.EVIDENCE_JPEG_QUALITY]
        )
        if not success:
            print(f"[Evidence] Failed to encode frame: {rel_path}")
            return

        jpeg_bytes = jpeg_buf.tobytes()

        # Write file
        with open(filepath, 'wb') as f:
            f.write(jpeg_bytes)

        # Compute SHA-256
        sha = hashlib.sha256(jpeg_bytes).hexdigest()
        frame_hash = f"sha256:{sha}"

        if is_final:
            self._evidence_final = rel_path
        else:
            self._evidence_frames.append(rel_path)

        self._evidence_hashes.append(frame_hash)
        print(f"[Evidence] Saved {rel_path} ({len(jpeg_bytes)} bytes, {frame_hash[:20]}...)")

    def _video_pipeline(self, loop):
        """Video processing pipeline — runs in a SEPARATE THREAD.

        All CV2/YOLO/ffmpeg work happens here so the asyncio event loop
        stays free to manage WebSocket connections without CLOSE-WAIT buildup.
        """
        import queue as _queue

        while True:
            try:
                self._run_video_pipeline(loop, _queue)
            except Exception as e:
                print(f"\n[RESTART] Video pipeline crashed: {e}")
                import traceback
                traceback.print_exc()
                print("[RESTART] Restarting pipeline in 5s...")
                time.sleep(5)

    def _run_video_pipeline(self, loop, _queue):
        """Single run of the video pipeline. Raises on error for auto-restart."""
        _sl_pipe = [None]  # unused, kept for cleanup compatibility

        direct_url = get_stream_url(self.stream_url)
        print(f"\n[Stream] Opening video...")
        cap = cv2.VideoCapture(direct_url)

        if not cap.isOpened():
            print("[ERROR] Could not open video stream!")
            raise RuntimeError("Could not open video stream")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"[Stream] Opened. FPS: {fps:.0f}")
        print(f"[Stream] Output width: {OUTPUT_WIDTH}px")
        print(f"[STATE] idle — waiting for round")

        # Start Cloudflare broadcast if enabled
        if self._cf.enabled:
            ret_test, frame_test = cap.read()
            if ret_test and frame_test is not None:
                h_t, w_t = frame_test.shape[:2]
                if w_t > OUTPUT_WIDTH:
                    scale = OUTPUT_WIDTH / w_t
                    out_h = int(h_t * scale)
                else:
                    out_h = h_t
                if out_h % 2 != 0:
                    out_h += 1
                self._cf.width = OUTPUT_WIDTH
                self._cf.height = out_h
                self._cf.start()

        self.running = True
        frame_idx = 0
        frame_interval = 1.0 / self.target_fps
        server_start = time.time()

        # Reader thread
        import threading
        _frame_q = _queue.Queue(maxsize=2)
        _reader_alive = [True]
        _cap_holder = [cap]         # cv2.VideoCapture (HLS direct) or None
        _last_frame_time = [time.time()]
        self._switch_event = threading.Event()

        _force_refresh = [False]  # Force yt-dlp refresh after camera switch
        _ff_proc_holder = [None]  # ffmpeg decoder subprocess

        def _kill_ff():
            """Kill ffmpeg decoder subprocess."""
            p = _ff_proc_holder[0]
            if p is not None:
                try:
                    p.kill()
                    p.wait(timeout=3)
                except Exception:
                    pass
                _ff_proc_holder[0] = None

        def _start_ff(url):
            """Start ffmpeg decoder: HLS URL → raw BGR frames on stdout."""
            _kill_ff()
            cmd = [
                'ffmpeg',
                '-loglevel', 'warning',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-rw_timeout', '5000000',  # 5s network timeout (microseconds)
                '-i', url,
                '-vf', f'scale={OUTPUT_WIDTH}:-2',
                '-pix_fmt', 'bgr24',
                '-f', 'rawvideo',
                '-an',
                'pipe:1',
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=OUTPUT_WIDTH * 1080 * 3 * 2,
            )
            _ff_proc_holder[0] = proc
            print(f"[Reader] ffmpeg decoder started (pid={proc.pid})")
            return proc

        # We need frame dimensions — probe first frame
        def _read_first_frame(proc, max_w=OUTPUT_WIDTH):
            """Read first frame to determine actual dimensions."""
            # ffmpeg scale with -2 ensures even height, but we need to know it
            # Read enough bytes for max possible frame, then derive
            # Actually, we know the scale: OUTPUT_WIDTH x (proportional height)
            # For 1920x1080 → 1280x720, for 1280x720 → 1280x720
            # Use probe result or default 720
            return 720  # will be corrected from actual cap if needed

        def _reader():
            frame_h = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 720]
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1920
            if orig_w > OUTPUT_WIDTH:
                frame_h[0] = int(frame_h[0] * OUTPUT_WIDTH / orig_w)
            if frame_h[0] % 2 != 0:
                frame_h[0] += 1
            frame_size = [OUTPUT_WIDTH * frame_h[0] * 3]

            # Release cv2 cap — we'll use ffmpeg pipe instead
            try:
                cap.release()
            except Exception:
                pass
            _cap_holder[0] = None

            while _reader_alive[0]:
                # ── Camera switch ──
                if self._switch_event.is_set():
                    self._switch_event.clear()
                    _force_refresh[0] = True
                    print("[Reader] Camera switch — reconnecting...")
                    _kill_ff()
                    while not _frame_q.empty():
                        try:
                            _frame_q.get_nowait()
                        except _queue.Empty:
                            break

                # ── Start ffmpeg if needed ──
                proc = _ff_proc_holder[0]
                if proc is None or proc.poll() is not None:
                    if proc is not None and proc.poll() is not None:
                        print(f"[Reader] ffmpeg exited (code={proc.returncode}), restarting...")
                    url = get_stream_url(self.stream_url, force_refresh=_force_refresh[0])
                    _force_refresh[0] = False
                    try:
                        proc = _start_ff(url)
                        _last_frame_time[0] = time.time()
                        print(f"[Reader] ffmpeg started, frame_size={frame_size[0]} ({OUTPUT_WIDTH}x{frame_h[0]})")

                        # Start a watchdog thread that kills ffmpeg if stuck
                        def _watchdog(p, holder):
                            while p.poll() is None and _reader_alive[0]:
                                if time.time() - _last_frame_time[0] > 20:
                                    print("[Watchdog] No frames 20s — killing ffmpeg")
                                    try:
                                        p.kill()
                                    except Exception:
                                        pass
                                    holder[0] = None
                                    _force_refresh[0] = True
                                    return
                                time.sleep(1)
                        threading.Thread(target=_watchdog, args=(proc, _ff_proc_holder), daemon=True).start()
                    except Exception as e:
                        print(f"[Reader] ffmpeg start error: {e}")
                        _kill_ff()
                        time.sleep(3)
                    continue

                # ── Blocking read of exactly one frame ──
                # This blocks until ffmpeg outputs a full frame.
                # The watchdog thread kills ffmpeg if it blocks too long.
                try:
                    raw = proc.stdout.read(frame_size[0])
                    if len(raw) != frame_size[0]:
                        print(f"[Reader] Incomplete frame ({len(raw)}/{frame_size[0]}), restarting...")
                        _kill_ff()
                        _force_refresh[0] = True
                        continue

                    _last_frame_time[0] = time.time()
                    f = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (frame_h[0], OUTPUT_WIDTH, 3)).copy()

                    # CF is now fed from YOLO worker (not reader)

                    # Replace whatever is in the main loop queue with latest frame
                    while not _frame_q.empty():
                        try:
                            _frame_q.get_nowait()
                        except _queue.Empty:
                            break
                    try:
                        _frame_q.put(f, timeout=0.1)
                    except _queue.Full:
                        pass

                except Exception as e:
                    print(f"[Reader] Read error: {e}")
                    _kill_ff()
                    time.sleep(1)

        threading.Thread(target=_reader, daemon=True).start()

        # YOLO inference thread
        _yolo_q = _queue.Queue(maxsize=1)
        _yolo_result = [None, 0]
        _yolo_lock = threading.Lock()

        _yolo_frame_id = [0]
        _yolo_last_log = [time.monotonic()]
        def _yolo_worker():
            while _reader_alive[0]:
                try:
                    yf = _yolo_q.get(timeout=1)
                except _queue.Empty:
                    continue
                t0 = time.monotonic()
                yolo_input = self.apply_roi(yf)
                annotated, cnt = self.counter.process_frame(yolo_input)
                yolo_ms = (time.monotonic() - t0) * 1000
                with _yolo_lock:
                    _yolo_result[0] = annotated
                    _yolo_result[1] = cnt

                # Push YOLO-annotated frame to CF queue.
                # Composite with ROI if needed, then send.
                if self._cf.enabled and self._cf._proc is not None:
                    cf_frame = annotated
                    if annotated.shape == yf.shape and self._roi_mask is not None:
                        roi_inv = cv2.bitwise_not(self._roi_mask)
                        bg = cv2.bitwise_and(yf, roi_inv)
                        fg = cv2.bitwise_and(annotated, self._roi_mask)
                        cf_frame = cv2.add(bg, fg)
                    self._cf.send_frame(cf_frame)

                _yolo_frame_id[0] += 1
                now = time.monotonic()
                if now - _yolo_last_log[0] >= 5.0:
                    _yolo_last_log[0] = now
                    cfq = self._cf._frame_q.qsize() if self._cf.enabled else 0
                    print(f"[YOLO] frame={_yolo_frame_id[0]} infer={yolo_ms:.0f}ms count={cnt} cfq={cfq}")

        threading.Thread(target=_yolo_worker, daemon=True).start()

        # Helper: thread-safe broadcast via asyncio event loop
        def _safe_broadcast(msg_dict):
            """Schedule a WS broadcast on the asyncio event loop (non-blocking)."""
            asyncio.run_coroutine_threadsafe(self._broadcast_json(msg_dict), loop)

        last_ws_time = 0

        try:
            while self.running:
                frame_start = time.time()

                # ── Check round duration expiry ──────────────────────
                if self._round_active:
                    round_elapsed = frame_start - self._round_start_time
                    if round_elapsed >= self._round_duration:
                        result = self._stop_round()
                        if result:
                            _safe_broadcast({
                                "type": "final",
                                "count": result["count"],
                                "in_count": result.get("in_count", 0),
                                "out_count": result.get("out_count", 0),
                                "duration": result["duration"],
                                "marketAddress": result.get("marketAddress", ""),
                            })
                            _safe_broadcast({
                                "type": "round_complete",
                                **result,
                            })

                # ── Get frame ────────────────────────────────────────
                _t0 = time.monotonic()
                try:
                    frame = _frame_q.get(timeout=0.1)
                except _queue.Empty:
                    continue
                _t_get = time.monotonic()

                frame_idx += 1

                # Resize
                h, w = frame.shape[:2]
                if w > OUTPUT_WIDTH:
                    scale = OUTPUT_WIDTH / w
                    frame = cv2.resize(frame, (OUTPUT_WIDTH, int(h * scale)),
                                      interpolation=cv2.INTER_LINEAR)

                # Feed frame to YOLO thread
                try:
                    _yolo_q.put_nowait(frame.copy())
                except _queue.Full:
                    pass

                # Use latest YOLO result
                with _yolo_lock:
                    display = _yolo_result[0] if _yolo_result[0] is not None else frame
                    count = _yolo_result[1]

                # Composite with ROI if needed
                if display is not frame and display.shape == frame.shape:
                    if self._roi_mask is not None:
                        roi_inv = cv2.bitwise_not(self._roi_mask)
                        bg = cv2.bitwise_and(frame, roi_inv)
                        fg = cv2.bitwise_and(display, self._roi_mask)
                        display = cv2.add(bg, fg)
                _t_proc = time.monotonic()

                # Broadcast vehicle_counted events
                if self._round_active and hasattr(self, '_pending_vehicle_events'):
                    for evt in self._pending_vehicle_events:
                        _safe_broadcast(evt)
                    self._pending_vehicle_events.clear()

                # Debug overlay
                uptime = frame_start - server_start
                fps_actual = frame_idx / max(uptime, 0.1)
                state_tag = self._state.upper()
                if self._round_active:
                    re = round(time.time() - self._round_start_time, 1)
                    dbg = f"{state_tag} seq:{frame_idx} fps:{fps_actual:.1f} round:{self._round_id} {re}s"
                else:
                    dbg = f"{state_tag} seq:{frame_idx} fps:{fps_actual:.1f}"
                h_d, w_d = display.shape[:2]
                cv2.putText(display, dbg, (w_d - 420, h_d - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 136), 1)

                # CF is now fed directly from reader thread at 30fps
                _t_cf = time.monotonic()

                # Debug timing every 5s
                if frame_idx % 150 == 0:
                    print(f"[MainLoop] get={(_t_get-_t0)*1000:.0f}ms proc={(_t_proc-_t_get)*1000:.0f}ms cf={(_t_cf-_t_proc)*1000:.0f}ms total={(_t_cf-_t0)*1000:.0f}ms")

                # WS: JSON-only state updates at 1Hz
                now = time.time()
                if now - last_ws_time >= 1.0:
                    last_ws_time = now
                    _video_uid_field = {"videoUid": self._cf.video_uid} if self._cf.video_uid else {}
                    if self._round_active:
                        round_elapsed = now - self._round_start_time
                        msg = {
                            "type": "count",
                            "state": "counting",
                            "count": count,
                            "count_in": self.counter.count_in,
                            "count_out": self.counter.count_out,
                            "elapsed": round(round_elapsed, 1),
                            "remaining": max(0, round(self._round_duration - round_elapsed, 1)),
                            "marketAddress": self._round_market,
                            "cameraId": self.camera_id,
                            "roundId": self._round_id,
                            "seq": frame_idx,
                            **_video_uid_field,
                        }
                    else:
                        msg = {
                            "type": "idle",
                            "state": "waiting",
                            "cameraId": self.camera_id,
                            **_video_uid_field,
                        }
                    _safe_broadcast(msg)

                # Evidence capture
                if self._round_active:
                    round_elapsed = time.time() - self._round_start_time
                    round_ts = int(self._round_start_time)
                    if round_elapsed - self._last_evidence_time >= self.EVIDENCE_INTERVAL:
                        self._save_evidence_frame(display, round_ts, round_elapsed)
                        self._last_evidence_time = round_elapsed
                    if self._round_duration - round_elapsed < frame_interval * 2:
                        self._save_evidence_frame(display, round_ts, round_elapsed, is_final=True)

                # FPS throttle
                elapsed = time.time() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            _reader_alive[0] = False
            if _cap_holder[0] is not None:
                try:
                    _cap_holder[0].release()
                except Exception:
                    pass
            try:
                _kill_ff()
            except Exception:
                pass
            self._cf.stop()

    async def handler(self, ws):
        accepted = await self.register(ws)
        if accepted is False:
            return
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    elif msg_type == "start_round":
                        self._start_round(data)
                        await ws.send(json.dumps({
                            "type": "round_started",
                            "roundId": self._round_id,
                            "cameraId": self.camera_id,
                        }))


                    elif msg_type == "stop_round":
                        result = self._stop_round()
                        if result:
                            await ws.send(json.dumps({"type": "round_complete", **result}))
                            await self._broadcast_json({
                                "type": "final",
                                "count": result["count"],
                                "in_count": result.get("in_count", 0),
                                "out_count": result.get("out_count", 0),
                                "duration": result["duration"],
                                "marketAddress": result.get("marketAddress", ""),
                            })
                        else:
                            await ws.send(json.dumps({"type": "error", "message": "no active round"}))

                    elif msg_type == "get_state":
                        await ws.send(json.dumps({
                            "type": "state",
                            "state": self._state,
                            "roundActive": self._round_active,
                            "roundId": self._round_id,
                            "marketAddress": self._round_market,
                            "count": self.counter.total_count,
                        }))

                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosedError:
            pass  # normal disconnect — no traceback needed
        finally:
            await self.unregister(ws)

    async def start(self):
        print(f"\n{'='*55}")
        print(f"  SinalBet Live Oracle Server (v4 — Persistent)")
        print(f"  WebSocket: ws://{self.host}:{self.port}")
        print(f"  Stream: {self.stream_url}")
        print(f"  Camera: {self.camera_id}")
        print(f"  Model: {self.counter.model.model_name}")
        print(f"  Mode: persistent (rounds via WS control)")
        print(f"  Target FPS: {self.target_fps}")
        if self._cf.enabled:
            print(f"  CF Broadcast: ON (videoUid={self._cf.video_uid})")
        else:
            print(f"  CF Broadcast: OFF")
        print(f"{'='*55}\n")

        import threading
        loop = asyncio.get_event_loop()

        # Start video pipeline in a separate thread — keeps event loop free for WS
        video_thread = threading.Thread(
            target=self._video_pipeline, args=(loop,), daemon=True
        )
        video_thread.start()
        print("[Server] Video pipeline started in background thread")

        # WS server runs on the asyncio event loop — never blocked by video
        async with websockets.serve(
            self.handler, self.host, self.port,
            ping_interval=30,
            ping_timeout=60,
        ):
            # Keep event loop alive forever
            while True:
                await asyncio.sleep(1)


def load_camera(camera_id):
    from pathlib import Path
    cameras_path = Path(__file__).parent / "cameras.json"
    with open(cameras_path) as f:
        data = json.load(f)
    for cam in data["cameras"]:
        if cam["id"] == camera_id:
            return cam
    print(f"[ERROR] Camera '{camera_id}' not found")
    sys.exit(1)


def _kill_port(port: int):
    """Kill any process holding the port so we never get 'address already in use'."""
    import signal as _sig
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
        for pid in result.stdout.strip().split('\n'):
            if pid.strip():
                try:
                    os.kill(int(pid.strip()), _sig.SIGKILL)
                    print(f"[Startup] Killed stale process {pid.strip()} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
        if result.stdout.strip():
            time.sleep(1)
    except FileNotFoundError:
        pass  # lsof not available


def main():
    parser = argparse.ArgumentParser(description='SinalBet Live Oracle Server (v4 — Persistent)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stream', '-s', help='Direct stream URL (HLS, RTSP, YouTube)')
    group.add_argument('--camera', help='Camera ID from cameras.json')
    parser.add_argument('--port', '-p', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--model', '-m', default=os.environ.get('YOLO_MODEL', 'yolo12x.pt'),
                       help='YOLO model (yolo12n=fast, yolo12s=balanced, yolo12x=best)')
    parser.add_argument('--confidence', '-c', type=float, default=0.15, help='Detection confidence')
    parser.add_argument('--line', '-l', type=float, default=0.5, help='Counting line position (0-1)')
    parser.add_argument('--angle', '-a', type=float, default=10, help='Line tilt in degrees')
    parser.add_argument('--line-points', type=str, default=None,
                       help='Line 1 as "x1,y1,x2,y2" fractions')
    parser.add_argument('--line-points2', type=str, default=None,
                       help='Line 2 (optional) as "x1,y1,x2,y2" fractions')
    parser.add_argument('--mode', choices=['line', 'uid'], default='uid',
                       help='Counting mode: line=crossing, uid=unique IDs (default: uid)')
    parser.add_argument('--fps', type=int, default=int(os.environ.get('TARGET_FPS', '15')), help='Target output FPS')

    args = parser.parse_args()

    stream_url = args.stream
    cam_lanes = None
    cam_id = ''
    if args.camera:
        cam = load_camera(args.camera)
        stream_url = cam.get("streamUrl", cam.get("imageUrl"))
        cam_id = cam["id"]
        cam_lanes = cam.get("lanes")
        if not args.line_points and cam.get("linePoints"):
            args.line_points = cam["linePoints"]
            args.mode = "line"
            print(f"[Camera] Line from config: {args.line_points}")
        if not args.line_points2 and cam.get("linePoints2"):
            args.line_points2 = cam["linePoints2"]
            print(f"[Camera] Line 2 from config: {args.line_points2}")
        cam_roi = cam.get("roi")
        if cam_roi:
            print(f"[Camera] ROI mask: {len(cam_roi)} polygon(s)")
        print(f"[Camera] {cam['name']} ({cam.get('source','')}) — {cam['type'].upper()}")
        if cam_lanes:
            print(f"[Camera] {len(cam_lanes)} lanes configured")

    _kill_port(args.port)

    server = StreamServer(
        stream_url=stream_url,
        port=args.port,
        model=args.model,
        confidence=args.confidence,
        line_pos=args.line,
        line_angle=args.angle,
        line_points=args.line_points,
        line_points2=args.line_points2,
        count_mode=args.mode,
        lanes=cam_lanes,
        target_fps=args.fps,
        camera_id=cam_id,
        roi=cam_roi if args.camera else None,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")


if __name__ == '__main__':
    main()

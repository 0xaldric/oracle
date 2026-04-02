#!/usr/bin/env python3
"""
Camera Line Calibrator — Click to set counting lines for cameras.json

Usage:
    python calibrate.py --camera peace-bridge
    python calibrate.py --stream "https://www.youtube.com/watch?v=DnUFAShZKus"
    python calibrate.py --image snapshot.jpg

Controls:
    Left Click   — Place point (2 points = 1 line)
    R            — Reset all points
    S            — Save & print JSON config
    1            — Switch to Line 1 (green)
    2            — Switch to Line 2 (cyan)
    P            — Toggle ROI polygon mode (click to add polygon vertices)
    Q / ESC      — Quit

Output:
    Prints linePoints, linePoints2, and roi as JSON-ready values.
"""

import argparse
import json
import sys
import os
import subprocess
import time

import cv2
import numpy as np


class Calibrator:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.original = frame.copy()
        self.h, self.w = frame.shape[:2]

        self.line1_points = []  # [(x,y), (x,y)]
        self.line2_points = []
        self.roi_points = []
        self.active_line = 1  # 1 or 2
        self.roi_mode = False

        self.window = "Camera Calibrator — Click to set counting lines"

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.roi_mode:
            self.roi_points.append((x, y))
            print(f"  ROI point #{len(self.roi_points)}: ({x/self.w:.4f}, {y/self.h:.4f})")
        elif self.active_line == 1:
            if len(self.line1_points) < 2:
                self.line1_points.append((x, y))
                frac = f"({x/self.w:.3f}, {y/self.h:.3f})"
                print(f"  Line 1 point {len(self.line1_points)}: pixel=({x},{y}) frac={frac}")
            else:
                print("  Line 1 already has 2 points. Press R to reset or 2 to switch to Line 2")
        elif self.active_line == 2:
            if len(self.line2_points) < 2:
                self.line2_points.append((x, y))
                frac = f"({x/self.w:.3f}, {y/self.h:.3f})"
                print(f"  Line 2 point {len(self.line2_points)}: pixel=({x},{y}) frac={frac}")
            else:
                print("  Line 2 already has 2 points. Press R to reset or 1 to switch to Line 1")

        self.redraw()

    def redraw(self):
        self.frame = self.original.copy()

        # Draw ROI polygon
        if len(self.roi_points) > 1:
            pts = np.array(self.roi_points, dtype=np.int32)
            cv2.polylines(self.frame, [pts], isClosed=len(self.roi_points) > 2,
                         color=(255, 0, 255), thickness=2)
        for p in self.roi_points:
            cv2.circle(self.frame, p, 4, (255, 0, 255), -1)

        # Draw Line 1 (green)
        for p in self.line1_points:
            cv2.circle(self.frame, p, 6, (0, 255, 0), -1)
        if len(self.line1_points) == 2:
            cv2.line(self.frame, self.line1_points[0], self.line1_points[1],
                    (0, 255, 0), 2)

        # Draw Line 2 (cyan)
        for p in self.line2_points:
            cv2.circle(self.frame, p, 6, (255, 255, 0), -1)
        if len(self.line2_points) == 2:
            cv2.line(self.frame, self.line2_points[0], self.line2_points[1],
                    (255, 255, 0), 2)

        # Status text
        mode = "ROI POLYGON" if self.roi_mode else f"LINE {self.active_line}"
        color = (255, 0, 255) if self.roi_mode else ((0, 255, 0) if self.active_line == 1 else (255, 255, 0))
        cv2.putText(self.frame, f"Mode: {mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(self.frame, "Click=point  R=reset  S=save  1/2=line  P=roi  Q=quit", (10, self.h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window, self.frame)

    def get_config(self):
        result = {}
        if len(self.line1_points) == 2:
            p1, p2 = self.line1_points
            result["linePoints"] = f"{p1[0]/self.w:.3f},{p1[1]/self.h:.3f},{p2[0]/self.w:.3f},{p2[1]/self.h:.3f}"
        if len(self.line2_points) == 2:
            p1, p2 = self.line2_points
            result["linePoints2"] = f"{p1[0]/self.w:.3f},{p1[1]/self.h:.3f},{p2[0]/self.w:.3f},{p2[1]/self.h:.3f}"
        if len(self.roi_points) >= 3:
            result["roi"] = [[round(p[0]/self.w, 4), round(p[1]/self.h, 4)] for p in self.roi_points]
        return result

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, min(self.w, 1280), min(self.h, 720))
        cv2.setMouseCallback(self.window, self.mouse_callback)
        self.redraw()

        print("\n=== Camera Calibrator ===")
        print("Click to place points. 2 clicks = 1 counting line.")
        print("Keys: 1=Line1(green) 2=Line2(cyan) P=ROI R=Reset S=Save Q=Quit\n")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):
                self.line1_points.clear()
                self.line2_points.clear()
                self.roi_points.clear()
                self.active_line = 1
                self.roi_mode = False
                print("  [Reset all points]")
                self.redraw()
            elif key == ord('1'):
                self.active_line = 1
                self.roi_mode = False
                print("  [Switched to Line 1]")
                self.redraw()
            elif key == ord('2'):
                self.active_line = 2
                self.roi_mode = False
                print("  [Switched to Line 2]")
                self.redraw()
            elif key == ord('p'):
                self.roi_mode = not self.roi_mode
                print(f"  [ROI mode: {'ON' if self.roi_mode else 'OFF'}]")
                self.redraw()
            elif key == ord('s'):
                config = self.get_config()
                if config:
                    print("\n" + "=" * 50)
                    print("CONFIG (copy to cameras.json):")
                    print("=" * 50)
                    print(json.dumps(config, indent=2))
                    print("=" * 50 + "\n")
                else:
                    print("  [No lines set yet]")

        cv2.destroyAllWindows()
        return self.get_config()


def grab_frame_youtube(url):
    """Grab a single frame from a YouTube live stream."""
    print(f"Fetching stream URL from YouTube...")
    try:
        result = subprocess.run(
            ['yt-dlp', '-f', 'best[height<=720]', '-g', url],
            capture_output=True, text=True, timeout=30
        )
        stream_url = result.stdout.strip()
    except Exception:
        # Try alternative
        result = subprocess.run(
            ['yt-dlp', '-f', 'best', '-g', url],
            capture_output=True, text=True, timeout=30
        )
        stream_url = result.stdout.strip()

    if not stream_url:
        print(f"ERROR: Could not get stream URL from {url}")
        sys.exit(1)

    print(f"Capturing frame...")
    cap = cv2.VideoCapture(stream_url)
    for _ in range(10):  # skip first frames
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not capture frame")
        sys.exit(1)

    return frame


def grab_frame_hls(url):
    """Grab a single frame from an HLS stream."""
    print(f"Capturing frame from HLS...")
    cap = cv2.VideoCapture(url)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not capture frame")
        sys.exit(1)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Camera Line Calibrator')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--camera', help='Camera ID from cameras.json')
    group.add_argument('--stream', '-s', help='Stream URL (YouTube, HLS)')
    group.add_argument('--image', '-i', help='Path to a snapshot image')
    parser.add_argument('--save-frame', help='Save captured frame to this path')

    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Could not read image {args.image}")
            sys.exit(1)
    elif args.camera:
        cameras_path = os.path.join(os.path.dirname(__file__), 'cameras.json')
        with open(cameras_path) as f:
            data = json.load(f)
        cam = None
        for c in data["cameras"]:
            if c["id"] == args.camera:
                cam = c
                break
        if not cam:
            print(f"ERROR: Camera '{args.camera}' not found")
            sys.exit(1)

        url = cam.get("streamUrl", cam.get("imageUrl"))
        if cam.get("type") == "youtube":
            frame = grab_frame_youtube(url)
        elif cam.get("type") == "hls":
            frame = grab_frame_hls(url)
        elif cam.get("type") == "jpeg":
            import urllib.request
            resp = urllib.request.urlopen(url)
            arr = np.frombuffer(resp.read(), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            frame = grab_frame_hls(url)

        print(f"Camera: {cam['name']} ({cam['id']})")
        if cam.get("linePoints"):
            print(f"Existing linePoints: {cam['linePoints']}")
        if cam.get("linePoints2"):
            print(f"Existing linePoints2: {cam['linePoints2']}")
    else:
        url = args.stream
        if 'youtube.com' in url or 'youtu.be' in url:
            frame = grab_frame_youtube(url)
        else:
            frame = grab_frame_hls(url)

    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)
        print(f"Frame saved to {args.save_frame}")

    # Resize if too large
    h, w = frame.shape[:2]
    if w > 1920:
        scale = 1920 / w
        frame = cv2.resize(frame, (1920, int(h * scale)))

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibrator(frame)
    config = cal.run()

    if config:
        print("\n=== FINAL CONFIG ===")
        print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()

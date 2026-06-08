#!/usr/bin/env python3
"""
Live depth-estimation harness for the overslam SLAM pipeline.

Watches the input-logger JSONL for Insert keydown events. While active,
pulls frames from a live ffmpeg.exe screen-capture pipe and runs one or
both depth estimators borrowed from ~/turntable:

  - flow : OpenCV Farneback optical flow (real-time) + depth-from-flow
           heuristic (1 / |flow| with an eps floor). The turntable
           repo's brute-force dense_motion_field + orbit-based
           depth_from_flow are available via --flow-impl turntable
           / --orbit-period; both are research-grade and won't run
           in real time at typical desktop resolutions.

  - da   : DepthAnythingEstimator from ~/turntable/depth_anything.py
           (HuggingFace Depth-Anything-V2-Small by default).

Reports per-method processing FPS and mean / p50 / p95 latency. The
goal is to inform a single decision: which mechanism is viable for
real-time SLAM depth.

Run inside WSL2; requires ffmpeg.exe on the Windows PATH plus (for the
DA path) a CUDA-capable PyTorch.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Make the turntable code importable.
TURNTABLE_DIR = Path(os.path.expanduser("~/turntable"))
sys.path.insert(0, str(TURNTABLE_DIR))


# ---------- ffmpeg helpers ----------
FFMPEG = os.environ.get("FFMPEG", "ffmpeg.exe")

def probe_desktop_size() -> tuple[int, int]:
    """Ask ffmpeg.exe what the desktop resolution is."""
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-f", "gdigrab", "-i", "desktop",
         "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=15,
    )
    blob = out.stderr or ""
    m = re.search(r"(\d{3,5})x(\d{3,5})", blob)
    if not m:
        raise RuntimeError(
            f"could not parse desktop size from ffmpeg output:\n{blob[-500:]}"
        )
    return int(m.group(1)), int(m.group(2))

def start_capture(width: int, height: int, fps: int) -> subprocess.Popen:
    """Spawn ffmpeg.exe streaming raw BGR24 frames to stdout."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-f", "gdigrab", "-framerate", str(fps),
        "-draw_mouse", "0",  # cursor adds artificial flow
        "-i", "desktop",
        "-vf", f"scale={width}:{height}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )


# ---------- Frame reader thread ----------
class LatestFrameSource:
    """Drop-old frame source: always returns the most recent frame.

    A reader thread pulls raw bytes from ffmpeg.exe's stdout, decodes
    them as BGR images, and stores only the latest. The main loop polls
    `get()`; if the depth pipeline is slower than the capture rate the
    older frames are simply discarded, so the FPS we measure reflects
    the pipeline's actual throughput.
    """
    def __init__(self, proc: subprocess.Popen, width: int, height: int):
        import numpy as np
        self._np = np
        self.proc = proc
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._lock = threading.Lock()
        self._latest = None
        self._latest_idx = -1
        self._consumed_idx = -1
        self._stop = threading.Event()
        self._frames_in = 0
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        np = self._np
        stdout = self.proc.stdout
        n = self.frame_bytes
        while not self._stop.is_set():
            buf = stdout.read(n)
            if not buf or len(buf) < n:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self._lock:
                self._latest = frame
                self._latest_idx += 1
                self._frames_in += 1

    def get(self):
        """Return (frame, idx) if a new frame is available, else (None, -1)."""
        with self._lock:
            if self._latest_idx == self._consumed_idx:
                return None, -1
            self._consumed_idx = self._latest_idx
            return self._latest, self._latest_idx

    def stats(self):
        with self._lock:
            return self._frames_in, time.perf_counter() - self._t0

    def stop(self):
        self._stop.set()
        try:
            self.proc.terminate()
        except Exception:
            pass


# ---------- Input-log watcher ----------
class InsertWatcher:
    """Tail the JSONL log and toggle `active` on Insert keydown."""
    def __init__(self, path: Path):
        self.path = path
        self.active = False
        self._stop = threading.Event()
        self._on_toggle = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, on_toggle=None):
        self._on_toggle = on_toggle
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        # Wait until the file exists; the user may launch the logger after us.
        while not self._stop.is_set() and not self.path.exists():
            time.sleep(0.2)
        if self._stop.is_set():
            return
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if (evt.get("type") == "key"
                        and evt.get("event") == "down"
                        and evt.get("name") == "Insert"
                        and not evt.get("repeat", False)):
                    self.active = not self.active
                    if self._on_toggle:
                        self._on_toggle(self.active)


# ---------- Depth methods ----------
class FlowDepth:
    """Optical-flow-based depth.

    Default: cv2.calcOpticalFlowFarneback between consecutive grayscale
    frames, depth = 1 / (|flow| + eps). This is a real-time-friendly
    proxy; a production pipeline would resolve ego-motion from the
    input log and triangulate, but the goal here is wall-clock
    feasibility, not photometric accuracy.

    --flow-impl turntable swaps the flow estimator for the brute-force
    dense_motion_field from ~/turntable/affine_vector_field.py. With
    --orbit-period N it also uses the orbit-based depth_from_flow from
    orbit_depth_utils.py instead of the inverse-magnitude proxy.
    """
    name = "flow"

    def __init__(self, impl: str = "farneback",
                 orbit_period: int | None = None,
                 capture_fps: int = 30):
        import cv2
        import numpy as np
        self.cv2 = cv2
        self.np = np
        self.impl = impl
        self.orbit_period = orbit_period
        self.capture_fps = capture_fps
        self.prev_gray = None
        if impl == "turntable":
            from affine_vector_field import dense_motion_field
            self._dense = dense_motion_field
        if orbit_period:
            from orbit_depth_utils import depth_from_flow, infer_rotation_direction
            self._depth_from_flow = depth_from_flow
            self._infer_rot = infer_rotation_direction

    def estimate(self, frame_bgr):
        cv2 = self.cv2; np = self.np
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None  # need two frames
        if self.impl == "turntable":
            flow, _, _ = self._dense(self.prev_gray, gray)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
        self.prev_gray = gray
        if self.orbit_period:
            rot = self._infer_rot(flow)
            depth = self._depth_from_flow(
                flow, frame_width=gray.shape[1],
                period_frames=self.orbit_period, fps=self.capture_fps,
                rot_dir_sign=rot,
            )
        else:
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            depth = 1.0 / (mag + 0.5)
        return depth


class DADepth:
    name = "da"

    def __init__(self, model_id: str | None = None):
        from depth_anything import DepthAnythingEstimator
        kwargs = {"model_id": model_id} if model_id else {}
        self.est = DepthAnythingEstimator(**kwargs)

    def estimate(self, frame_bgr):
        return self.est.estimate(frame_bgr)


# ---------- Stats tracker ----------
class MethodStats:
    def __init__(self, name: str):
        self.name = name
        self.lat = collections.deque(maxlen=200)
        self.processed = 0
        self.t_window_start = time.perf_counter()
        self.frames_in_window = 0

    def record(self, dt_s: float):
        self.lat.append(dt_s)
        self.processed += 1
        self.frames_in_window += 1

    def maybe_report(self, every_s: float = 1.0):
        now = time.perf_counter()
        elapsed = now - self.t_window_start
        if elapsed < every_s:
            return None
        if self.frames_in_window == 0:
            self.t_window_start = now
            return None
        fps = self.frames_in_window / elapsed
        lats = sorted(self.lat)
        mean_ms = (sum(lats) / len(lats)) * 1000
        p50 = lats[len(lats) // 2] * 1000
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))] * 1000
        msg = (f"[{self.name:>4}] {fps:6.2f} fps "
               f"| mean {mean_ms:6.1f} ms | p50 {p50:6.1f} | p95 {p95:6.1f} "
               f"| total {self.processed}")
        self.t_window_start = now
        self.frames_in_window = 0
        return msg


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("flow", "da", "both"), default="both",
                    help="Which depth method to run (default: both)")
    ap.add_argument("--logfile", required=True,
                    help="Path to the input-logger JSONL (the file log-input.sh writes)")
    ap.add_argument("--downscale", type=float, default=2.0,
                    help="Capture downscale factor; output frame is desktop / N (default: 2.0)")
    ap.add_argument("--fps", type=int, default=30,
                    help="Capture framerate hint passed to ffmpeg gdigrab (default: 30)")
    ap.add_argument("--flow-impl", choices=("farneback", "turntable"), default="farneback",
                    help="Optical-flow implementation (default: farneback). "
                         "'turntable' is the brute-force dense_motion_field from "
                         "~/turntable; not real-time at typical resolutions.")
    ap.add_argument("--orbit-period", type=int, default=None,
                    help="If set, use ~/turntable's depth_from_flow with this orbit "
                         "period (in frames) instead of the inverse-magnitude proxy.")
    ap.add_argument("--da-model", default=None,
                    help="DepthAnything model id (default: Depth-Anything-V2-Small-hf)")
    args = ap.parse_args()

    logfile = Path(args.logfile)

    # Probe desktop size, compute downscaled capture size.
    print("Probing desktop size via ffmpeg.exe ...", flush=True)
    dw, dh = probe_desktop_size()
    w = max(2, int(dw / args.downscale) // 2 * 2)
    h = max(2, int(dh / args.downscale) // 2 * 2)
    print(f"Desktop: {dw}x{dh}  ->  capture {w}x{h} "
          f"({args.fps} fps, downscale {args.downscale}x)")

    # Build methods.
    methods = []
    if args.method in ("flow", "both"):
        print("Initialising flow-field depth...", flush=True)
        methods.append((FlowDepth(impl=args.flow_impl,
                                  orbit_period=args.orbit_period,
                                  capture_fps=args.fps),
                        MethodStats("flow")))
    if args.method in ("da", "both"):
        print("Initialising DepthAnything (this loads weights on first run)...",
              flush=True)
        methods.append((DADepth(model_id=args.da_model),
                        MethodStats("da")))

    # Start capture.
    print("Starting ffmpeg.exe capture...", flush=True)
    proc = start_capture(w, h, args.fps)
    src = LatestFrameSource(proc, w, h)

    # Start log watcher.
    def on_toggle(state):
        print(f"\n[insert] depth estimation {'ENABLED' if state else 'paused'}",
              flush=True)
    watcher = InsertWatcher(logfile)
    watcher.start(on_toggle=on_toggle)

    print(f"\nWatching {logfile} for Insert keydown to toggle.")
    print("Press Insert (with the logger running) to start/stop depth processing.")
    print("Ctrl+C here to quit.\n", flush=True)

    stop = {"flag": False}
    def handle_sigint(sig, frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    last_status = time.perf_counter()
    try:
        while not stop["flag"]:
            if not watcher.active:
                time.sleep(0.05)
                # Print idle status occasionally.
                now = time.perf_counter()
                if now - last_status > 5.0:
                    fin, dt = src.stats()
                    cap_fps = fin / dt if dt > 0 else 0
                    print(f"[idle] capture {cap_fps:5.1f} fps "
                          f"({fin} frames). press Insert to start.",
                          flush=True)
                    last_status = now
                continue
            frame, idx = src.get()
            if frame is None:
                time.sleep(0.002)
                continue
            for method, stats in methods:
                t0 = time.perf_counter()
                try:
                    method.estimate(frame)
                except Exception as e:
                    print(f"[{method.name}] error: {e}", flush=True)
                    continue
                dt = time.perf_counter() - t0
                stats.record(dt)
                msg = stats.maybe_report()
                if msg:
                    print(msg, flush=True)
    finally:
        watcher.stop()
        src.stop()
        # Final summary.
        print("\n--- summary ---")
        fin, dt = src.stats()
        cap_fps = fin / dt if dt > 0 else 0
        print(f"capture: {fin} frames in {dt:.1f}s ({cap_fps:.2f} fps)")
        for _, stats in methods:
            if not stats.lat:
                print(f"[{stats.name}] no frames processed")
                continue
            lats = sorted(stats.lat)
            mean_ms = (sum(lats) / len(lats)) * 1000
            p50 = lats[len(lats) // 2] * 1000
            p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))] * 1000
            print(f"[{stats.name}] processed {stats.processed} frames "
                  f"| mean {mean_ms:.1f} ms | p50 {p50:.1f} | p95 {p95:.1f} ms")


if __name__ == "__main__":
    main()

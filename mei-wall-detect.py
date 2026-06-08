#!/usr/bin/env python3
"""
Mei Ice Wall placement detector.

State machine driven by the input log + on-demand OCR on a screen-capture
pipe. Each detected placement emits a JSONL event line plus a directory
with pre/post frames and a heuristic wall mask, suitable for training a
later YOLO/U-Net on.

States
------
  IDLE         -- waiting for Mouse5 (x2) down
  PENDING_OCR  -- x2 was pressed; running Tesseract on the center strip
                  for up to ocr_window_s to confirm BUILD/CANCEL UI
  PLACING      -- UI confirmed; tracking orientation via x2 press parity
  CAPTURING    -- left-click placed; waiting for the wall to fully appear
                  before diffing pre/post frames into a mask

Orientation parity
------------------
  press 1 (entering placement) -> Mei face (length side toward camera)
  press 2                      -> Ling face (depth side toward camera)
  press 3                      -> Mei face again, ...

Outputs (under --outdir)
------------------------
  events.jsonl                                    -- one line per placement / cancel
  events/<utc-iso>__<orient>/
    event.json                                    -- full metadata
    pre.png, post.png                             -- raw frames
    mask.png, diff.png                            -- heuristic mask + diff viz

Run in WSL2. Requires ffmpeg.exe on Windows PATH and tesseract installed
in WSL (apt install tesseract-ocr; pip install pytesseract).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

FFMPEG = os.environ.get("FFMPEG", "ffmpeg.exe")


# ---------- ffmpeg capture helpers (parallel to depth-eval.py) ----------
def probe_desktop_size() -> tuple[int, int]:
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-f", "gdigrab", "-i", "desktop",
         "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=15,
    )
    m = re.search(r"(\d{3,5})x(\d{3,5})", out.stderr or "")
    if not m:
        raise RuntimeError(f"could not parse desktop size:\n{(out.stderr or '')[-500:]}")
    return int(m.group(1)), int(m.group(2))

def start_capture(width: int, height: int, fps: int) -> subprocess.Popen:
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-f", "gdigrab", "-framerate", str(fps),
        "-draw_mouse", "0", "-i", "desktop",
        "-vf", f"scale={width}:{height}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


# ---------- Frame ring buffer with wall-clock timestamps ----------
class FrameRing:
    """Holds the last `seconds` worth of (t_wall_ns, frame_bgr) tuples."""
    def __init__(self, proc: subprocess.Popen, width: int, height: int,
                 fps: int, seconds: float = 1.5):
        import numpy as np
        self.np = np
        self.proc = proc
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._buf = collections.deque(maxlen=max(8, int(fps * seconds)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.frames_in = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        stdout = self.proc.stdout
        n = self.frame_bytes
        while not self._stop.is_set():
            # Reads on an unbuffered pipe may return fewer than n bytes;
            # accumulate until we have a full frame.
            buf = bytearray()
            while len(buf) < n and not self._stop.is_set():
                chunk = stdout.read(n - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
            if len(buf) < n:
                break
            t_wall_ns = time.time_ns()
            frame = self.np.frombuffer(bytes(buf), dtype=self.np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
            with self._lock:
                self._buf.append((t_wall_ns, frame))
                self.frames_in += 1

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else (None, None)

    def snapshot(self):
        """Return a list copy of (ts, frame) tuples."""
        with self._lock:
            return list(self._buf)

    def closest_before(self, t_wall_ns: int):
        """Most recent frame with timestamp <= t_wall_ns."""
        with self._lock:
            best = None
            for ts, fr in self._buf:
                if ts <= t_wall_ns:
                    best = (ts, fr)
                else:
                    break
            return best if best else (None, None)

    def closest_at_or_after(self, t_wall_ns: int):
        """Oldest frame with timestamp >= t_wall_ns."""
        with self._lock:
            for ts, fr in self._buf:
                if ts >= t_wall_ns:
                    return (ts, fr)
        return (None, None)

    def stop(self):
        self._stop.set()
        try:
            self.proc.terminate()
        except Exception:
            pass


# ---------- Async OCR worker ----------
class OcrWorker:
    """Run OCR on background thread; always processes the latest submission.

    submit(frame, ts, kwargs) overwrites any pending request. The worker
    pulls one at a time, runs the OCR fn, and pushes (ts, txt, dt_s) onto
    the result queue. Main loop drains results between event processing
    so the state machine never blocks on OCR.
    """
    def __init__(self, ocr_fn):
        import queue as _q
        self.ocr_fn = ocr_fn
        self._lock = threading.Lock()
        self._pending = None
        self._cv = threading.Condition(self._lock)
        self._results = _q.Queue()
        self._stop = threading.Event()
        self.busy = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame, ts, kwargs):
        with self._cv:
            self._pending = (frame, ts, kwargs)
            self._cv.notify()

    def get_result(self):
        import queue as _q
        try:
            return self._results.get_nowait()
        except _q.Empty:
            return None

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def _run(self):
        while not self._stop.is_set():
            with self._cv:
                while self._pending is None and not self._stop.is_set():
                    self._cv.wait(timeout=0.1)
                if self._stop.is_set():
                    return
                req = self._pending
                self._pending = None
                self.busy = True
            frame, ts, kwargs = req
            t0 = time.perf_counter()
            try:
                txt = self.ocr_fn(frame, **kwargs)
            except Exception as e:
                txt = f"<ocr error: {e}>"
            self._results.put((ts, txt, time.perf_counter() - t0))
            self.busy = False


# ---------- Per-event clip saver ----------
class ClipSaver:
    """Background thread that writes frames from the FrameRing to an
    .mp4 file for a fixed duration starting at start_ts_ns.

    Polls ring.snapshot() periodically and writes any frames newer than
    last_written_ts that fall in [start_ts_ns, start_ts_ns + duration].
    Multiple instances can run concurrently for overlapping events.
    """
    def __init__(self, out_path, start_ts_ns: int, duration_s: float,
                 ring, fps: int = 30, fourcc_str: str = "mp4v"):
        import cv2
        self.cv2 = cv2
        self.out_path = str(out_path)
        self.start_ts_ns = start_ts_ns
        self.end_ts_ns = start_ts_ns + int(duration_s * 1_000_000_000)
        self.duration_s = duration_s
        self.ring = ring
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.writer = None
        self.last_written_ts = start_ts_ns - 1
        self.frames_written = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    def _run(self):
        try:
            while not self._stop.is_set():
                snap = self.ring.snapshot()
                for ts, fr in snap:
                    if ts <= self.last_written_ts:
                        continue
                    if ts >= self.end_ts_ns:
                        break
                    if self.writer is None:
                        h, w = fr.shape[:2]
                        self.writer = self.cv2.VideoWriter(
                            self.out_path, self.fourcc, self.fps, (w, h),
                        )
                        if not self.writer.isOpened():
                            print(f"[clip] failed to open {self.out_path}",
                                  flush=True)
                            return
                    self.writer.write(fr)
                    self.last_written_ts = ts
                    self.frames_written += 1
                # Done once we're past the end + a small grace so the ring
                # gets a chance to capture the last frame.
                if time.time_ns() >= self.end_ts_ns + 200_000_000:
                    break
                time.sleep(0.05)
        finally:
            if self.writer is not None:
                self.writer.release()
            print(f"[clip] wrote {self.frames_written} frames "
                  f"({self.duration_s:.1f}s) to {self.out_path}", flush=True)


# ---------- Input log tailer ----------
class LogTailer:
    """Pushes parsed JSONL events from the input log into a queue."""
    def __init__(self, path: Path, q):
        self.path = path
        self.q = q
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set() and not self.path.exists():
            time.sleep(0.2)
        if self._stop.is_set():
            return
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.02)
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                self.q.put(evt)


# ---------- OCR ----------
def _ocr_one_box(frame_bgr, x_frac, y_frac, psm, upscale, text_max, outline_min,
                 debug_save_prefix=None, label=""):
    """Crop -> grayscale -> upscale -> invert -> drop mid-tones -> OCR.
    Returns uppercase Tesseract output for the given fractional bbox."""
    import cv2
    import numpy as np
    import pytesseract
    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * x_frac[0]), int(w * x_frac[1])
    y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    inv = cv2.bitwise_not(gray)
    keep = (inv < text_max) | (inv > outline_min)
    cleaned = np.where(keep, inv, 255).astype(np.uint8)

    if debug_save_prefix:
        suffix = f"_{label}" if label else ""
        cv2.imwrite(str(debug_save_prefix) + f"{suffix}_roi.png",     roi)
        cv2.imwrite(str(debug_save_prefix) + f"{suffix}_cleaned.png", cleaned)

    cfg = (f"--psm {psm} --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    return pytesseract.image_to_string(cleaned, config=cfg).strip().upper()


# Default tight boxes derived from connected-component analysis of the
# selftest cleaned image at 2560x1440 desktop -> 1280x720 capture.
# Resolution-invariant because they're fractional.
DEFAULT_BUILD_BOX  = ((0.31, 0.37), (0.47, 0.53))
DEFAULT_CANCEL_BOX = ((0.625, 0.71), (0.47, 0.53))


def ocr_center_strip(frame_bgr,
                     build_box=DEFAULT_BUILD_BOX,
                     cancel_box=DEFAULT_CANCEL_BOX,
                     psm=7, upscale=3.0,
                     text_max=30, outline_min=180,
                     debug_save_prefix=None,
                     # Back-compat: older callers may pass these but they
                     # are ignored once we're using per-word boxes.
                     roi_h_frac=None, roi_w_frac=None) -> str:
    """Run OCR on two tight per-word boxes (BUILD and CANCEL) and
    return their combined output. PSM 7 (single-line) is correct now
    that each box contains exactly one word."""
    txt_b = _ocr_one_box(frame_bgr, *build_box, psm=psm, upscale=upscale,
                         text_max=text_max, outline_min=outline_min,
                         debug_save_prefix=debug_save_prefix, label="build")
    txt_c = _ocr_one_box(frame_bgr, *cancel_box, psm=psm, upscale=upscale,
                         text_max=text_max, outline_min=outline_min,
                         debug_save_prefix=debug_save_prefix, label="cancel")
    return f"{txt_b} | {txt_c}"


def selftest_ocr(args, w, h):
    """Grab one frame from the live capture and OCR it now, print result.

    Use this with the placement UI visible on screen to verify the ROI
    covers the text and Tesseract can read it.
    """
    delay = max(0, args.selftest_delay)
    print(f"\n[selftest] you have {delay}s to alt-tab and open Mei's placement UI...",
          flush=True)
    for remaining in range(delay, 0, -1):
        print(f"  {remaining}...", flush=True)
        time.sleep(1)
    print("[selftest] capturing one frame for OCR...", flush=True)
    proc = start_capture(w, h, args.fps)
    nbytes = w * h * 3
    # Try up to ~3s of frames before giving up.
    deadline = time.time() + 3.0
    buf = b""
    while time.time() < deadline:
        chunk = proc.stdout.read(nbytes - len(buf))
        if not chunk:
            break
        buf += chunk
        if len(buf) >= nbytes:
            break
    proc.terminate()
    try:
        err = proc.stderr.read().decode("utf-8", errors="replace")
    except Exception:
        err = ""
    if len(buf) < nbytes:
        print(f"[selftest] failed to read a frame (got {len(buf)}/{nbytes} bytes)",
              flush=True)
        if err.strip():
            print("[selftest] ffmpeg.exe stderr:", flush=True)
            print(err, flush=True)
        else:
            print("[selftest] ffmpeg.exe produced no stderr output.\n"
                  "Most common cause: Overwatch in exclusive-fullscreen mode.\n"
                  "Switch the game to Borderless Windowed and try again.",
                  flush=True)
        return
    import numpy as np
    frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3)).copy()
    out_prefix = Path(args.outdir) / "_selftest"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = ocr_center_strip(frame, psm=args.ocr_psm,
                           build_box=(tuple(args.ocr_build_x), tuple(args.ocr_build_y)),
                           cancel_box=(tuple(args.ocr_cancel_x), tuple(args.ocr_cancel_y)),
                           text_max=args.ocr_text_max,
                           outline_min=args.ocr_outline_min,
                           debug_save_prefix=out_prefix)
    import re as _re
    hit = bool(_re.search(r"BUILD|CANCEL", txt))
    print(f"[selftest] OCR text: {txt!r}")
    print(f"[selftest] regex match (BUILD|CANCEL): {hit}  "
          f"-> live mode {'WOULD' if hit else 'would NOT'} transition to PLACING")
    print(f"[selftest] artifacts: {out_prefix}_roi.png, _inv.png, _cleaned.png")
    print(f"[selftest] inspect _cleaned.png — that's what Tesseract sees")


# ---------- Mask heuristic ----------
def compute_wall_mask(pre_bgr, post_bgr):
    """Diff heuristic: blur, absdiff, threshold, keep largest centered blob."""
    import cv2
    import numpy as np
    a = cv2.GaussianBlur(pre_bgr,  (5, 5), 0)
    b = cv2.GaussianBlur(post_bgr, (5, 5), 0)
    diff = cv2.absdiff(a, b)
    gdiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gdiff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    num, labels, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
    h, w = th.shape
    cx, cy = w / 2, h / 2
    best_idx = -1
    best_score = -1.0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 200:
            continue
        # Score = area weighted by inverse distance from frame center.
        dx = cents[i][0] - cx
        dy = cents[i][1] - cy
        dist = (dx * dx + dy * dy) ** 0.5
        score = area / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_idx = i
    mask = np.zeros_like(th)
    if best_idx > 0:
        mask[labels == best_idx] = 255
    return mask, gdiff


# ---------- Fast text-presence pre-filter ----------
def quick_text_check(frame_bgr, build_box, cancel_box,
                     text_max=30, outline_min=180,
                     min_dark_pixels=150, min_horizontal_span=0.35):
    """Cheap (1-2 ms) heuristic: in each box, run the same invert + drop
    mid-tones cleanup we use for OCR, then check whether the surviving
    dark pixels (a) are numerous enough and (b) span a meaningful chunk
    of the box width horizontally. A single concentrated blob (e.g. a
    scene shadow) fails the horizontal-span check even if it passes the
    pixel count. Text-shaped content passes both.
    """
    import cv2
    import numpy as np
    h, w = frame_bgr.shape[:2]
    for (x_frac, y_frac) in (build_box, cancel_box):
        x0, x1 = int(w * x_frac[0]), int(w * x_frac[1])
        y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        keep = (inv < text_max) | (inv > outline_min)
        cleaned = np.where(keep, inv, 255)
        dark = cleaned < 50
        if int(dark.sum()) < min_dark_pixels:
            continue
        cols_with_dark = int(dark.any(axis=0).sum())
        if cols_with_dark < dark.shape[1] * min_horizontal_span:
            continue
        return True
    return False


def sweep_ring_for_ui(ring, args, ocr_hit_re, outdir, debug_dir, press_ns):
    """Iterate every frame in the ring. Pre-filter with quick_text_check;
    on candidates run full OCR. Returns (ts, txt) of the first OCR match
    found, else None. Sequential and synchronous; call when the async
    OCR pipeline has already given up (window timeout) so the few-second
    sweep cost is acceptable as a fallback.
    """
    build_box  = (tuple(args.ocr_build_x),  tuple(args.ocr_build_y))
    cancel_box = (tuple(args.ocr_cancel_x), tuple(args.ocr_cancel_y))
    frames = ring.snapshot()
    if not frames:
        return None
    n_total = len(frames)
    n_cand = 0
    n_match = 0
    t_start = time.perf_counter()
    for ts, frame in frames:
        if not quick_text_check(frame, build_box, cancel_box,
                                text_max=args.ocr_text_max,
                                outline_min=args.ocr_outline_min):
            continue
        n_cand += 1
        rel_ms = (ts - press_ns) / 1e6
        dbg_prefix = (debug_dir / f"sweep_+{int(rel_ms)}ms_{ts}") \
            if debug_dir else None
        txt = ocr_center_strip(
            frame, psm=args.ocr_psm,
            build_box=build_box, cancel_box=cancel_box,
            text_max=args.ocr_text_max, outline_min=args.ocr_outline_min,
            debug_save_prefix=dbg_prefix,
        )
        if ocr_hit_re.search(txt):
            n_match += 1
            elapsed = time.perf_counter() - t_start
            print(f"  [sweep] match @ +{rel_ms:.0f}ms after {n_cand}/{n_total} "
                  f"frame(s) OCR'd in {elapsed:.1f}s: {txt!r}", flush=True)
            return ts, txt
    elapsed = time.perf_counter() - t_start
    print(f"  [sweep] no match in {n_total} frames "
          f"({n_cand} candidate(s), {elapsed:.1f}s)", flush=True)
    return None


# ---------- Timeline dump (debug) ----------
def dump_timeline_frames(ring, press_ns, outdir, tag):
    """Save every frame in the ring buffer named by ms-from-press."""
    import cv2
    sub = Path(outdir) / "_timeline" / tag
    sub.mkdir(parents=True, exist_ok=True)
    frames = ring.snapshot()
    for ts, fr in frames:
        rel_ms = (ts - press_ns) / 1e6
        sign = "+" if rel_ms >= 0 else "-"
        name = f"{sign}{abs(rel_ms):06.0f}ms.png"
        cv2.imwrite(str(sub / name), fr)
    print(f"  [timeline] wrote {len(frames)} frames to {sub}", flush=True)


# ---------- Output writer ----------
def write_event(outdir: Path, events_path: Path, evt: dict,
                pre=None, post=None, mask=None, diff=None):
    import cv2
    iso = dt.datetime.fromtimestamp(evt["t_wall_ns"] / 1e9, dt.timezone.utc) \
        .strftime("%Y%m%dT%H%M%S_%f")[:-3]
    tag = f"{iso}__{evt.get('orientation', 'cancel')}"
    sub = outdir / "events" / tag
    sub.mkdir(parents=True, exist_ok=True)
    if pre is not None:  cv2.imwrite(str(sub / "pre.png"),  pre)
    if post is not None: cv2.imwrite(str(sub / "post.png"), post)
    if mask is not None: cv2.imwrite(str(sub / "mask.png"), mask)
    if diff is not None: cv2.imwrite(str(sub / "diff.png"), diff)
    (sub / "event.json").write_text(json.dumps(evt, indent=2))
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt) + "\n")
    return sub


# ---------- Main state machine ----------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logfile", required=True,
                    help="input-logger JSONL path (live-tailed)")
    ap.add_argument("--outdir", default="./mei_walls",
                    help="output root (default ./mei_walls)")
    ap.add_argument("--downscale", type=float, default=2.0,
                    help="capture downscale factor (default 2.0)")
    ap.add_argument("--fps", type=int, default=30,
                    help="capture framerate (default 30)")
    ap.add_argument("--ability-button", default="x2",
                    choices=("x1", "x2", "middle"),
                    help="mouse button bound to Ice Wall (default x2 = Mouse5)")
    ap.add_argument("--ocr-window-s", type=float, default=2.0,
                    help="seconds after ability press to wait for BUILD/CANCEL OCR")
    ap.add_argument("--ocr-submit-ms", type=int, default=30,
                    help="how often the main loop submits a fresh frame to the "
                         "background OCR worker (default 30ms). The worker drops "
                         "any pending request when a newer one arrives, so this "
                         "is just a refresh rate, not a per-OCR cost.")
    ap.add_argument("--ocr-initial-delay-ms", type=int, default=150,
                    help="ms after ability press before submitting the first "
                         "OCR (default 150). OW + gdigrab+pipe latency.")
    ap.add_argument("--dump-timeline", action="store_true",
                    help="when PENDING_OCR exits (match or timeout), dump every "
                         "frame in the ring buffer named by ms-from-press, so "
                         "you can see when the UI actually appeared.")
    ap.add_argument("--no-sweep", action="store_true",
                    help="disable the ring-buffer sweep fallback that runs on "
                         "window timeout. The sweep does a fast pixel-density "
                         "pre-filter then full OCR on candidates -- catches "
                         "single-frame UI flashes the live polling missed.")
    ap.add_argument("--clip-duration-s", type=float, default=20.0,
                    help="seconds of video to save after each successful "
                         "placement event (starting at the post-frame). "
                         "The first ~5s (wall lifetime) anchor metric pose; "
                         "the rest is used by continue-pose.py to keep "
                         "tracking via the semantic feature map after the "
                         "wall drops. Saved as clip.mp4 in the event dir. "
                         "Set 0 to disable clip storage.")
    ap.add_argument("--pre-offset-ms", type=int, default=80,
                    help="pre-click frame offset (default -80ms)")
    ap.add_argument("--post-offset-ms", type=int, default=900,
                    help="post-click frame offset in ms (default 900). Has to "
                         "be long enough for Mei's wall-rise animation to "
                         "finish — too short and the post frame still shows "
                         "the pre-placement scene.")
    ap.add_argument("--debug-ocr", action="store_true",
                    help="save raw + thresholded OCR ROI on every check, "
                         "and print every OCR result to stdout")
    ap.add_argument("--ocr-psm", type=int, default=7,
                    help="Tesseract page-segmentation mode (default 7 = "
                         "single line, which suits the per-word tight boxes)")
    ap.add_argument("--ocr-build-x", type=float, nargs=2,
                    default=DEFAULT_BUILD_BOX[0], metavar=("X0", "X1"),
                    help=f"BUILD box x fractions (default "
                         f"{DEFAULT_BUILD_BOX[0][0]} {DEFAULT_BUILD_BOX[0][1]})")
    ap.add_argument("--ocr-build-y", type=float, nargs=2,
                    default=DEFAULT_BUILD_BOX[1], metavar=("Y0", "Y1"),
                    help=f"BUILD box y fractions (default "
                         f"{DEFAULT_BUILD_BOX[1][0]} {DEFAULT_BUILD_BOX[1][1]})")
    ap.add_argument("--ocr-cancel-x", type=float, nargs=2,
                    default=DEFAULT_CANCEL_BOX[0], metavar=("X0", "X1"),
                    help=f"CANCEL box x fractions (default "
                         f"{DEFAULT_CANCEL_BOX[0][0]} {DEFAULT_CANCEL_BOX[0][1]})")
    ap.add_argument("--ocr-cancel-y", type=float, nargs=2,
                    default=DEFAULT_CANCEL_BOX[1], metavar=("Y0", "Y1"),
                    help=f"CANCEL box y fractions (default "
                         f"{DEFAULT_CANCEL_BOX[1][0]} {DEFAULT_CANCEL_BOX[1][1]})")
    ap.add_argument("--ocr-text-max", type=int, default=30,
                    help="post-invert: keep pixels with value < this (text "
                         "body). Default 30.")
    ap.add_argument("--ocr-outline-min", type=int, default=180,
                    help="post-invert: keep pixels with value > this (outline "
                         "ring). Default 180. Anything between these two is "
                         "scene noise -> set to white.")
    ap.add_argument("--selftest-ocr", action="store_true",
                    help="grab one frame now, OCR it, dump artifacts and exit. "
                         "Open Mei's wall placement UI in-game first.")
    ap.add_argument("--selftest-delay", type=int, default=5,
                    help="seconds to wait before the selftest grab (default 5) "
                         "so you can alt-tab and open the placement UI.")
    args = ap.parse_args()

    # Verify deps early
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError as e:
        sys.stderr.write(
            f"missing dependency: {e}\n"
            "install with:\n"
            "  sudo apt install tesseract-ocr\n"
            "  pip install pytesseract opencv-python numpy\n"
        )
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    events_path = outdir / "events.jsonl"
    debug_dir = outdir / "_ocr_debug" if args.debug_ocr else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print("Probing desktop size...", flush=True)
    dw, dh = probe_desktop_size()
    w = max(2, int(dw / args.downscale) // 2 * 2)
    h = max(2, int(dh / args.downscale) // 2 * 2)
    print(f"Desktop {dw}x{dh} -> capture {w}x{h} @ {args.fps} fps")

    if args.selftest_ocr:
        selftest_ocr(args, w, h)
        return

    print("Starting capture...", flush=True)
    proc = start_capture(w, h, args.fps)
    # Buffer must span pre_offset + post_offset + the worst-case sweep
    # latency + safety. A ring sweep on a full buffer takes ~1s, during
    # which old frames keep getting evicted -- if the user click was
    # early in the window, the pre-frame can fall off the back before
    # CAPTURING gets to it.
    ring_seconds = max(4.0,
        (args.pre_offset_ms + args.post_offset_ms) / 1000.0 + 2.5)
    ring = FrameRing(proc, w, h, args.fps, seconds=ring_seconds)

    import queue as _q
    evt_q = _q.Queue()
    tail = LogTailer(Path(args.logfile), evt_q)
    ocr_worker = OcrWorker(ocr_center_strip)
    clip_savers = []  # list of in-flight ClipSaver threads

    stop = {"flag": False}
    def handle_sigint(sig, frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    # State
    state = "IDLE"
    press_count = 0
    pending_since_ns = 0
    last_submit_ns = 0
    orientation = None
    click_event = None
    pending_click = None  # set when user clicks before OCR confirms

    OCR_HIT_RE = re.compile(r"BUILD|CANCEL")
    btn = args.ability_button

    def orient_for(count: int) -> str:
        return "mei" if count % 2 == 1 else "ling"

    print(f"\nWatching {args.logfile}; ability button = {btn}")
    print(f"Outputs -> {outdir}")
    print("Press the ability button in-game to begin a placement. Ctrl+C to quit.\n",
          flush=True)

    while not stop["flag"]:
        # Drain events
        try:
            evt = evt_q.get(timeout=0.05)
        except _q.Empty:
            evt = None

        now_ns = time.time_ns()

        if evt is not None and evt.get("type") == "mouse_button":
            b = evt.get("button"); e = evt.get("event")
            t = evt.get("t_wall_ns")

            if state == "IDLE" and b == btn and e == "down":
                state = "PENDING_OCR"
                press_count = 1
                pending_since_ns = t
                last_submit_ns = 0
                orientation = orient_for(press_count)
                print(f"[{state}] ability press -> waiting on OCR "
                      f"(orient={orientation})", flush=True)

            elif state == "PENDING_OCR" and b == btn and e == "down":
                press_count += 1
                orientation = orient_for(press_count)

            elif state == "PENDING_OCR" and b == "left" and e == "down":
                # Fast placement: user clicked before OCR could confirm.
                # Buffer the click + snapshot the pre-frame NOW so the
                # sweep delay can't evict it.
                pre_target = t - args.pre_offset_ms * 1_000_000
                _, pre_snapshot = ring.closest_before(pre_target)
                pending_click = {"click_t_wall_ns": t,
                                 "orientation": orientation,
                                 "press_count": press_count,
                                 "pre_frame": pre_snapshot}
                pre_t = t - 50_000_000  # 50ms before click
                _, pre_fr = ring.closest_before(pre_t)
                if pre_fr is not None:
                    dbg_prefix = (debug_dir / f"retro_{now_ns}") if debug_dir else None
                    ocr_worker.submit(pre_fr, t, dict(
                        psm=args.ocr_psm,
                        build_box=(tuple(args.ocr_build_x), tuple(args.ocr_build_y)),
                        cancel_box=(tuple(args.ocr_cancel_x), tuple(args.ocr_cancel_y)),
                        text_max=args.ocr_text_max,
                        outline_min=args.ocr_outline_min,
                        debug_save_prefix=dbg_prefix,
                    ))
                    print(f"[PENDING_OCR] left-click buffered; retroactive OCR "
                          f"on pre-click frame submitted", flush=True)
                else:
                    print("[PENDING_OCR] left-click but no pre-click frame in "
                          "ring; will rely on remaining live OCR results",
                          flush=True)

            elif state == "PENDING_OCR" and b == "right" and e == "down":
                print(f"[{state}] right-click before OCR; reset", flush=True)
                state = "IDLE"
                pending_click = None

            elif state == "PLACING" and b == btn and e == "down":
                press_count += 1
                orientation = orient_for(press_count)
                print(f"[PLACING] orient -> {orientation} (presses={press_count})",
                      flush=True)

            elif state == "PLACING" and b == "left" and e == "down":
                state = "CAPTURING"
                pre_target = t - args.pre_offset_ms * 1_000_000
                _, pre_snapshot = ring.closest_before(pre_target)
                click_event = {"click_t_wall_ns": t,
                               "orientation": orientation,
                               "press_count": press_count,
                               "pre_frame": pre_snapshot}
                print(f"[CAPTURING] placed (orient={orientation}); "
                      f"will diff in {args.post_offset_ms} ms", flush=True)

            elif state == "PLACING" and b == "right" and e == "down":
                placed = {
                    "type": "mei_wall_cancel",
                    "t_wall_ns": t,
                    "orientation": orientation,
                    "press_count": press_count,
                }
                write_event(outdir, events_path, placed)
                print(f"[IDLE] canceled (presses={press_count})", flush=True)
                state = "IDLE"

        # ---- Drain async OCR results ----
        while True:
            res = ocr_worker.get_result()
            if res is None:
                break
            ocr_ts, txt, ocr_dt = res
            if state == "PENDING_OCR":
                rel_ms = (ocr_ts - pending_since_ns) / 1e6
                if args.debug_ocr:
                    print(f"  [ocr +{rel_ms:6.0f}ms run={ocr_dt*1000:.0f}ms] "
                          f"{txt[:120]!r}", flush=True)
                if OCR_HIT_RE.search(txt):
                    if pending_click is not None:
                        # User already clicked. Skip PLACING, go straight to
                        # CAPTURING with the buffered click info.
                        print(f"[CAPTURING] retroactive OCR confirmed "
                              f"@ +{rel_ms:.0f}ms; using buffered click "
                              f"(orient={pending_click['orientation']})",
                              flush=True)
                        if args.dump_timeline:
                            dump_timeline_frames(ring, pending_since_ns, outdir,
                                                 tag=f"retro_+{int(rel_ms)}ms")
                        click_event = pending_click
                        pending_click = None
                        state = "CAPTURING"
                    else:
                        print(f"[PLACING] OCR match @ +{rel_ms:.0f}ms: {txt!r} "
                              f"(orient={orientation})", flush=True)
                        if args.dump_timeline:
                            dump_timeline_frames(ring, pending_since_ns, outdir,
                                                 tag=f"match_+{int(rel_ms)}ms")
                        state = "PLACING"

        # ---- Capture post-frame as soon as it's due ----
        # Done here (in PENDING_OCR) so it runs *before* the sweep and the
        # frame doesn't get evicted by the sweep delay.
        if (state == "PENDING_OCR" and pending_click is not None
                and "post_frame" not in pending_click):
            post_t = pending_click["click_t_wall_ns"] + args.post_offset_ms * 1_000_000
            if now_ns >= post_t + 30_000_000:
                _, post_snap = ring.closest_at_or_after(post_t)
                pending_click["post_frame"] = post_snap
                if args.debug_ocr:
                    rel_ms = (post_t - pending_since_ns) / 1e6
                    print(f"  [post-capture] @ +{rel_ms:.0f}ms "
                          f"({'frame found' if post_snap is not None else 'MISS'})",
                          flush=True)

        # ---- Submit fresh OCR work / handle window timeout ----
        if state == "PENDING_OCR":
            elapsed_ms = (now_ns - pending_since_ns) / 1e6
            # If a click is buffered, extend the effective window until
            # post-frame is captured -- the sweep needs to run *after*
            # the post-frame is safely snapshotted.
            effective_window_ms = args.ocr_window_s * 1000
            if (pending_click is not None
                    and "post_frame" not in pending_click):
                click_rel = (pending_click["click_t_wall_ns"]
                             - pending_since_ns) / 1e6
                effective_window_ms = max(effective_window_ms,
                                          click_rel + args.post_offset_ms + 100)
            if elapsed_ms > effective_window_ms:
                # Fallback: live OCR didn't match. Sweep every frame in the
                # ring buffer with the cheap pre-filter + full OCR on
                # candidates. Catches single-frame UI flashes the polling
                # missed.
                if not args.no_sweep:
                    print(f"[sweep] window expired; sweeping {ring.frames_in} "
                          "captured frames in ring...", flush=True)
                    swept = sweep_ring_for_ui(
                        ring, args, OCR_HIT_RE, outdir,
                        debug_dir, pending_since_ns,
                    )
                else:
                    swept = None
                if swept is not None and pending_click is not None:
                    print(f"[CAPTURING] sweep confirmed UI; using buffered "
                          f"click (orient={pending_click['orientation']}, "
                          f"pre={'yes' if pending_click.get('pre_frame') is not None else 'NO'}, "
                          f"post={'yes' if pending_click.get('post_frame') is not None else 'NO'})",
                          flush=True)
                    click_event = pending_click
                    pending_click = None
                    state = "CAPTURING"
                else:
                    if pending_click is not None:
                        print(f"[IDLE] window expired with buffered click; "
                              "sweep found no UI either; dropping", flush=True)
                        pending_click = None
                    elif swept is not None:
                        print(f"[IDLE] sweep found UI but no click was "
                              "buffered; nothing to record", flush=True)
                    else:
                        print(f"[IDLE] OCR window expired ({elapsed_ms:.0f} ms); "
                              "no BUILD/CANCEL matched", flush=True)
                    if args.dump_timeline:
                        dump_timeline_frames(ring, pending_since_ns, outdir,
                                             tag="timeout")
                    state = "IDLE"
            elif elapsed_ms < args.ocr_initial_delay_ms:
                pass  # too early — UI hasn't rendered yet
            elif (now_ns - last_submit_ns) / 1e6 >= args.ocr_submit_ms:
                last_submit_ns = now_ns
                ts, fr = ring.latest()
                if fr is None:
                    if args.debug_ocr:
                        print(f"  [submit] no frame in ring "
                              f"(frames_in={ring.frames_in})", flush=True)
                else:
                    dbg_prefix = (debug_dir / f"ocr_+{int(elapsed_ms)}ms_{now_ns}") \
                        if debug_dir else None
                    ocr_worker.submit(fr, now_ns, dict(
                        psm=args.ocr_psm,
                        build_box=(tuple(args.ocr_build_x), tuple(args.ocr_build_y)),
                        cancel_box=(tuple(args.ocr_cancel_x), tuple(args.ocr_cancel_y)),
                        text_max=args.ocr_text_max,
                        outline_min=args.ocr_outline_min,
                        debug_save_prefix=dbg_prefix,
                    ))

        # CAPTURING: wait for the post-click frame to be available
        if state == "CAPTURING" and click_event is not None:
            click_t = click_event["click_t_wall_ns"]
            post_target = click_t + args.post_offset_ms * 1_000_000
            if now_ns >= post_target + 30_000_000:  # 30 ms safety margin
                # Use the pre/post snapshots taken at the right moments;
                # fall back to the ring only if no snapshot was saved.
                pre = click_event.get("pre_frame")
                if pre is None:
                    pre_target = click_t - args.pre_offset_ms * 1_000_000
                    _, pre = ring.closest_before(pre_target)
                post = click_event.get("post_frame")
                if post is None:
                    _, post = ring.closest_at_or_after(post_target)
                if pre is None or post is None:
                    print(f"[CAPTURING] missing {'pre' if pre is None else ''}"
                          f"{' and ' if (pre is None and post is None) else ''}"
                          f"{'post' if post is None else ''} frame; "
                          "skipping mask", flush=True)
                    mask = diff = None
                    area = 0
                else:
                    mask, diff = compute_wall_mask(pre, post)
                    area = int((mask > 0).sum())
                placed = {
                    "type": "mei_wall_placed",
                    "t_wall_ns": click_t,
                    "orientation": click_event["orientation"],
                    "press_count": click_event["press_count"],
                    "capture_resolution": [w, h],
                    "mask_area_px": area,
                    "pre_offset_ms":  args.pre_offset_ms,
                    "post_offset_ms": args.post_offset_ms,
                    "clip_duration_s": args.clip_duration_s,
                    "clip_start_wall_ns": post_target,
                    "fps": args.fps,
                }
                event_dir = write_event(outdir, events_path, placed,
                                        pre=pre, post=post, mask=mask, diff=diff)
                if args.clip_duration_s > 0:
                    saver = ClipSaver(
                        event_dir / "clip.mp4",
                        start_ts_ns=post_target,
                        duration_s=args.clip_duration_s,
                        ring=ring, fps=args.fps,
                    )
                    clip_savers.append(saver)
                print(f"[IDLE] wrote event ({placed['orientation']}, "
                      f"mask area={area}px)"
                      f"{'; clip saving started' if args.clip_duration_s > 0 else ''}\n",
                      flush=True)
                state = "IDLE"
                click_event = None

    tail.stop()
    # Wait for any in-flight clip savers to finish writing before stopping
    # the ring -- otherwise they'd lose the last second of footage.
    pending = [s for s in clip_savers if s._thread.is_alive()]
    if pending:
        print(f"Waiting for {len(pending)} clip saver(s) to finish...",
              flush=True)
        for s in pending:
            s.join(timeout=args.clip_duration_s + 5)
    ring.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()

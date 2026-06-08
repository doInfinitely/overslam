#!/usr/bin/env python3.10
"""Mei Cartographer -- autonomous map-exploration bot for OW2.

Single-rate lockstep loop (decided v1): grab -> map -> decide -> act at
the SLAM rate (~3-5 fps). Heuristic frontier/novelty explorer (v1).
Localizer is both an offline PNG->pose tool and the live respawn
relocalizer (see localize_shot()).

Pipeline per tick:
  1. grab frame
  2. DeathDetector: if death-cam UI -> enter DEAD; freeze mapping, drop a
     hazard marker at last good pose; wait for respawn (spawn room, marked
     by the ice wall placed at recording start) -> relocalize -> resume.
  3. Mapper.integrate(frame, pose): accumulate geometry + 3D features +
     coverage.  [geometry/feature placement = NEXT integration, wires in
     reconstruct-scene.py / extract-features-3d.py / continue-pose-vo.py]
  4. novelty = how new this shot is (visual, vs keyframe memory).
  5. Explorer.propose(novelty, motion_buffer, coverage) -> input plan.
  6. ScreenIO.apply(plan).
  Every alive frame is a character-free HARD NEGATIVE -> saved for the
  character-detector training set.

Map persists per --map name so other characters (flyers, wall-climbers)
extend a saved base map.

Windows-only at runtime (mss + pydirectinput). On Linux it imports for
syntax/analysis; heavy bits are guarded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import carto_geom as G


# ---------------------------------------------------------------------
# Screen capture + input  (Windows runtime)
# ---------------------------------------------------------------------

# The bot may ONLY navigate (move + jump + crouch + look). Abilities and
# fire are BANNED because they break the character-free hard negatives or
# pollute the map:
#   - shift  = Cryofreeze -> THIRD-PERSON camera, puts Mei in frame.
#   - e      = Ice Wall    -> spawns geometry into the scene.
#   - q      = Blizzard ult-> drone/effects in frame.
#   - mouse / v = fire/melee -> projectiles + muzzle effects in frame.
ALLOWED_KEYS = {"w", "a", "s", "d", "space", "ctrl"}
BANNED_KEYS = {"shift", "e", "q", "v", "f", "1", "2", "3"}

# Tesseract binary -- set path explicitly so it works without PATH changes.
import pytesseract as _tess
for _tc in [
    os.environ.get("PYTESSERACT_CMD"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]:
    if _tc and Path(_tc).exists():
        _tess.pytesseract.tesseract_cmd = _tc
        break


class ScreenIO:
    def __init__(self):
        import mss  # noqa
        factory = getattr(mss, "MSS", None) or mss.mss
        self._sct = factory()
        import pydirectinput
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0
        self.pdi = pydirectinput
        mon = self._sct.monitors[1]
        self.W, self.H = mon["width"], mon["height"]

    def grab(self) -> np.ndarray:
        import cv2
        shot = self._sct.grab(self._sct.monitors[1])
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

    # --- movement primitives (banned keys are hard-blocked here so no
    # explorer change can ever trigger Cryofreeze/abilities/fire) ---
    def _check(self, key):
        if key in BANNED_KEYS:
            raise ValueError(f"BANNED key '{key}' (would break clean "
                             f"first-person hard negatives)")
        return key

    def hold(self, key, down=True):
        self._check(key)
        (self.pdi.keyDown if down else self.pdi.keyUp)(key)

    def look(self, dx, dy):
        # relative mouse move = camera look (no mouse buttons -> no fire)
        self.pdi.moveRel(int(dx), int(dy), relative=True)

    def tap(self, key):
        self._check(key)
        self.pdi.press(key)

    def release_all(self, keys):
        for k in keys:
            try:
                self.pdi.keyUp(k)
            except Exception:
                pass


# ---------------------------------------------------------------------
# 30fps background capture thread — real-time Farneback wall detection
# during walk bursts. The main SLAM loop stays at its normal 3-5fps;
# this thread runs independently so wall contact is flagged within ~167ms
# (5 frames × 33ms) rather than waiting for the next SLAM tick.
# ---------------------------------------------------------------------

class CaptureThread(threading.Thread):
    """Continuous 30fps screen capture with real-time radial optical flow.

    Call set_walking(True) just before holding W so the thread starts
    accumulating flow frames. is_wall_contact() returns True once
    STUCK_FRAMES consecutive frames all show radial flow < FLOW_THRESH.
    The main loop polls this flag every ~20ms and cuts the burst early."""

    STUCK_FRAMES = 6      # 6 × 33ms = 200ms to confirm contact
    FLOW_THRESH  = 1.2    # mean radial px/frame; below = not moving forward
    THUMB_W      = 160
    THUMB_H      = 90

    def __init__(self, fps: float = 30.0):
        super().__init__(daemon=True)
        self._period      = 1.0 / fps
        self._frame_q     = _queue.Queue(maxsize=1)   # always-latest frame
        self._walking     = False
        self._contact     = threading.Event()
        self._stop        = threading.Event()
        self._lock        = threading.Lock()
        # Burst start: discard stale prev_g so first flow pair is both
        # captured during the burst (avoids false positives from 8s-stale frames).
        self._flush_prev  = threading.Event()
        # Full-res frame captured at burst start, used for flow triangulation.
        self._flow_baseline      = None
        self._flow_baseline_lock = threading.Lock()
        # All frames captured during the burst (full-res), for multi-view fusion.
        self._burst_buf      : list = []
        self._burst_buf_lock = threading.Lock()
        import mss as _mss
        factory    = getattr(_mss, "MSS", None) or _mss.mss
        self._sct  = factory()
        mon = self._sct.monitors[1]
        self._monitor = {"top": mon["top"], "left": mon["left"],
                         "width": mon["width"], "height": mon["height"]}

    def run(self):
        import cv2
        prev_g  = None
        stuck_n = 0
        fh, fw  = self.THUMB_H, self.THUMB_W
        # Pre-build normalised radial grid once.
        yg = (np.arange(fh, dtype=np.float32) - fh / 2)[:, None]
        xg = (np.arange(fw, dtype=np.float32) - fw / 2)[None, :]
        rg = np.sqrt(xg ** 2 + yg ** 2) + 1e-3

        while not self._stop.is_set():
            t0  = time.time()
            raw = self._sct.grab(self._monitor)
            frame = np.asarray(raw)[:, :, :3]   # BGR, full res

            # Always keep the queue at the latest frame (drop stale one).
            try:
                self._frame_q.get_nowait()
            except _queue.Empty:
                pass
            self._frame_q.put(frame)

            small = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Burst-start flush: prev_g may be 8s stale (captured during SLAM).
            # Discard it so the first flow pair is both from within the burst.
            # Also save the full-res frame as the flow triangulation baseline.
            if self._flush_prev.is_set():
                self._flush_prev.clear()
                with self._flow_baseline_lock:
                    self._flow_baseline = frame
                with self._burst_buf_lock:
                    self._burst_buf = [frame]
                prev_g  = gray   # fresh baseline; skip flow this frame
                stuck_n = 0
            else:
                with self._lock:
                    walking = self._walking
                if walking:
                    with self._burst_buf_lock:
                        self._burst_buf.append(frame)
                if walking and prev_g is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_g, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    # Radial component: flow · (x̂,ŷ) / r
                    fwd = float(
                        (flow[..., 0] * xg / rg + flow[..., 1] * yg / rg).mean())
                    if fwd < self.FLOW_THRESH:
                        stuck_n += 1
                        if stuck_n >= self.STUCK_FRAMES:
                            self._contact.set()
                    else:
                        stuck_n = 0
                        self._contact.clear()
                else:
                    stuck_n = 0
                prev_g = gray

            dt = time.time() - t0
            if dt < self._period:
                time.sleep(self._period - dt)

    def latest_frame(self, timeout: float = 0.5) -> "np.ndarray | None":
        """Block until a frame arrives (up to timeout seconds)."""
        try:
            return self._frame_q.get(timeout=timeout)
        except _queue.Empty:
            return None

    def flow_baseline(self) -> "np.ndarray | None":
        """Full-res frame captured at the last burst start (for triangulation)."""
        with self._flow_baseline_lock:
            return self._flow_baseline

    def burst_frames(self) -> list:
        """All full-res frames captured during the last burst, in order."""
        with self._burst_buf_lock:
            return list(self._burst_buf)

    def set_walking(self, on: bool):
        with self._lock:
            self._walking = on
        if on:
            self._flush_prev.set()   # discard stale pre-burst prev_g
        else:
            self._contact.clear()

    def is_wall_contact(self) -> bool:
        return self._contact.is_set()

    def clear_wall_contact(self):
        self._contact.clear()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------
# Death / respawn detection  (validated against the fall screenshot)
# ---------------------------------------------------------------------

class DeathDetector:
    """OCR the top strip for 'SPECTATING'. The death banner is red text,
    so we isolate it via R-channel dominance before OCR.
    Rate-limited to once per second to avoid blocking the main loop."""
    OCR_CFG = ("--psm 6 --oem 1 "
               "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self):
        self._last_check = 0.0
        self._last_result = False

    def is_dead(self, img_bgr) -> bool:
        now = time.time()
        if now - self._last_check < 1.0:
            return self._last_result
        import cv2, pytesseract
        H = img_bgr.shape[0]
        roi = img_bgr[0: int(H * 0.08), :]
        b, g, r = cv2.split(roi)
        red = ((r.astype(np.int16) - g.astype(np.int16)) > 40) & (r > 120)
        text_img = np.where(red, 0, 255).astype(np.uint8)
        # Downscale to 640-wide before OCR — text is big enough to read small.
        scale = 640.0 / text_img.shape[1]
        text_img = cv2.resize(text_img, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        txt = pytesseract.image_to_string(text_img,
                                          config=self.OCR_CFG).upper()
        self._last_result = "SPECTATING" in txt or "DEATH" in txt
        self._last_check = now
        return self._last_result


# ---------------------------------------------------------------------
# Visual novelty (v1 coverage proxy until geometry mapper is wired in)
# ---------------------------------------------------------------------

class NoveltyNet:
    """ResNet50 layer3 mean-pooled descriptor; novelty = 1 - max cosine
    to stored keyframes. A view is novel if it doesn't resemble anything
    seen. Keyframes accumulate as the 'looked-at' memory."""
    def __init__(self):
        import torch
        from torchvision import models, transforms
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(self.dev)
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.keyframes: list[np.ndarray] = []

    def descriptor(self, img_bgr) -> np.ndarray:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        nh = 384
        nw = int(round(w / h * nh / 32)) * 32
        pil = pil.resize((max(32, nw), nh), Image.BICUBIC)
        out = {}
        hk = self.model.layer3.register_forward_hook(
            lambda m, i, o: out.__setitem__("x", o))
        with self.torch.no_grad():
            self.model(self.tfm(pil).unsqueeze(0).to(self.dev))
        hk.remove()
        d = out["x"][0].mean(dim=(1, 2)).cpu().float().numpy()
        return d / (np.linalg.norm(d) + 1e-9)

    def novelty(self, desc) -> float:
        if not self.keyframes:
            return 1.0
        K = np.stack(self.keyframes)
        return float(1.0 - (K @ desc).max())

    def add_keyframe(self, desc):
        self.keyframes.append(desc)


# ---------------------------------------------------------------------
# Depth source: DepthAnything (turntable repo, Windows runtime) with a
# graceful fall-back to optical-flow triangulation (no model needed).
# ---------------------------------------------------------------------

class DepthSource:
    """Self-contained depth estimator using Depth-Anything-V2 via
    HuggingFace transformers (pip install transformers).
    No external repo required. Downloads weights on first use (~100 MB).
    Falls back to None (flow-triangulation only) if unavailable."""

    MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, prefer_da=True):
        self._model     = None
        self._proc      = None
        self._dev       = None
        self._transform = None   # MiDaS transform if using torch.hub fallback
        self._midas     = False
        if not prefer_da:
            return

        import torch
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Try HuggingFace transformers (DA-V2) first -------------------
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._proc  = AutoImageProcessor.from_pretrained(self.MODEL)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                self.MODEL).to(self._dev).eval()
            print(f"depth: DA-V2 loaded (device={self._dev})")
            return
        except Exception as e:
            print(f"depth: DA-V2 unavailable ({e}); trying MiDaS fallback")

        # --- Fallback: MiDaS via torch.hub (no HuggingFace required) ------
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                   trust_repo=True)
            self._model = midas.to(self._dev).eval()
            transforms  = torch.hub.load("intel-isl/MiDaS", "transforms",
                                         trust_repo=True)
            self._transform = transforms.small_transform
            self._midas = True
            print(f"depth: MiDaS-small loaded (device={self._dev})")
        except Exception as e2:
            print(f"depth: MiDaS unavailable ({e2}); depth disabled")

    def relative(self, frame_bgr) -> "np.ndarray | None":
        """Return relative (unscaled) HxW depth map, or None."""
        if self._model is None:
            return None
        import cv2, torch
        H, W = frame_bgr.shape[:2]

        if self._midas:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self._dev)
            with torch.no_grad():
                d = self._model(inp).squeeze().cpu().numpy().astype(np.float32)
        else:
            from PIL import Image
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = self._proc(images=Image.fromarray(rgb), return_tensors="pt")
            inputs = {k: v.to(self._dev) for k, v in inputs.items()}
            with torch.no_grad():
                d = self._model(**inputs).predicted_depth.squeeze().cpu().numpy().astype(np.float32)

        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        return d


# ---------------------------------------------------------------------
# DepthThread: DA-V2 inference in background.
# Main loop stays unblocked; DA results arrive asynchronously and are
# used to enrich the map when ready (roughly every 8s on a typical GPU).
# ---------------------------------------------------------------------

class DepthThread(threading.Thread):
    """DA-V2 inference in background.

    Two submission modes:
      request()              — normal per-tick frame; result via poll().
      request_batch_tagged() — panoramic scan: list of (frame, pose_tag);
                               each result arrives via poll_scan() as
                               (depth_map, pose_tag) so the caller can
                               reproject with the correct capture pose.
    """
    def __init__(self, depth_src: DepthSource):
        super().__init__(daemon=True)
        self._src        = depth_src
        self._inbox      = _queue.Queue(maxsize=1)   # untagged: always-latest frame
        self._scan_inbox = _queue.Queue()            # tagged: unlimited, never dropped
        self._outbox     = _queue.Queue(maxsize=1)   # untagged results
        self._scan_outbox = _queue.Queue()           # tagged results, unlimited
        self._stop       = threading.Event()

    def run(self):
        while not self._stop.is_set():
            # Drain scan frames first (priority); fall back to regular frame.
            try:
                item = self._scan_inbox.get_nowait()
            except _queue.Empty:
                try:
                    item = self._inbox.get(timeout=0.1)
                except _queue.Empty:
                    continue
            if isinstance(item, tuple):
                frame, tag = item
            else:
                frame, tag = item, None
            d = self._src.relative(frame)
            if d is None:
                continue
            if tag is None:
                try: self._outbox.get_nowait()
                except _queue.Empty: pass
                self._outbox.put(d)
            else:
                self._scan_outbox.put((d, tag))

    def request(self, frame: np.ndarray):
        """Add frame for inference if queue has space."""
        try:
            self._inbox.put_nowait(frame)
        except _queue.Full:
            pass

    def request_batch_tagged(self, tagged_frames: list):
        """Panoramic scan: push all frames onto the dedicated scan inbox (never dropped)."""
        for item in tagged_frames:
            self._scan_inbox.put(item)

    def poll(self) -> "np.ndarray | None":
        """Non-blocking poll for regular (untagged) result."""
        try: return self._outbox.get_nowait()
        except _queue.Empty: return None

    def poll_scan(self) -> "tuple | None":
        """Non-blocking poll for scan result; returns (depth_map, pose_tag) or None."""
        try: return self._scan_outbox.get_nowait()
        except _queue.Empty: return None

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------
# Flow-based metric depth triangulation.
# Two frames separated by a known forward translation → depth via
# d = focal * baseline / radial_disparity (inverse disparity).
# Reliable range: ~0.5–10m at Mei's 5.5m/s with a 0.5s burst baseline.
# Returns NaN where flow is too small (far walls → use DA instead).
# ---------------------------------------------------------------------

def flow_depth_map(frame_a: np.ndarray, frame_b: np.ndarray,
                   baseline_m: float, focal_px: float,
                   out_h: int, out_w: int,
                   thumb_w: int = 320, thumb_h: int = 180,
                   min_disp: float = 0.5):
    """Triangulate metric depth from two forward-motion frames.

    frame_a / frame_b : BGR, any resolution (will be downscaled to thumb).
    baseline_m        : camera translation between the two frames (metres).
    focal_px          : full-resolution focal length (pixels).
    out_h, out_w      : output depth map shape (should match game frame).
    min_disp          : minimum radial flow (px, thumb-space) below which
                        depth is unreliable (set to NaN).

    Returns (depth_map HxW float32 metres, mean_radial_px float, focal_t float).
    mean_radial_px is the mean outward radial flow in thumbnail pixels.
    Forward displacement (geometric, no calibration): disp_m = mean_radial_px * Z / THUMB_MEAN_R
    flow_vel_cal corrects for da_scale error: disp_m /= flow_vel_cal"""
    import cv2
    scale_x = thumb_w / frame_a.shape[1]
    focal_t = focal_px * scale_x   # focal in thumbnail pixels

    ga = cv2.cvtColor(cv2.resize(frame_a, (thumb_w, thumb_h),
                                  interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(frame_b, (thumb_w, thumb_h),
                                  interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        ga, gb, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    cx = thumb_w / 2.0
    cy = thumb_h / 2.0
    xg = (np.arange(thumb_w, dtype=np.float32) - cx)[None, :]
    yg = (np.arange(thumb_h, dtype=np.float32) - cy)[:, None]
    rg = np.sqrt(xg ** 2 + yg ** 2) + 1e-3

    # Radial (outward) component of flow = forward-motion disparity.
    radial = flow[..., 0] * xg / rg + flow[..., 1] * yg / rg
    mean_radial_px = float(np.clip(radial, 0, None).mean())

    with np.errstate(divide='ignore', invalid='ignore'):
        depth_t = np.where(radial >= min_disp,
                           focal_t * baseline_m / radial,
                           np.nan).astype(np.float32)

    return (cv2.resize(depth_t, (out_w, out_h),
                       interpolation=cv2.INTER_NEAREST),
            mean_radial_px, focal_t)


def wiggle_depth_map(frames: list, strafe_speed: float, frame_dt: float,
                     focal_px: float, out_h: int, out_w: int,
                     thumb_w: int = 320, thumb_h: int = 180) -> np.ndarray:
    """Dense depth map from a left-right strafe wiggle.

    Each consecutive pair has a small lateral displacement so the FOE
    oscillates — every pixel gets flow.  Per-pair depth:
        D = focal_t × (strafe_speed × frame_dt) / |flow_magnitude|
    Median across all pairs gives a robust dense estimate.
    """
    import cv2
    scale_x = thumb_w / frames[0].shape[1]
    focal_t = focal_px * scale_x
    lateral_baseline = strafe_speed * frame_dt

    grays = [cv2.cvtColor(cv2.resize(f, (thumb_w, thumb_h),
                                     interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY)
             for f in frames]

    pair_depths = []
    for i in range(len(grays) - 1):
        fl = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.where(mag >= 0.3,
                         focal_t * lateral_baseline / mag,
                         np.nan).astype(np.float32)
        pair_depths.append(d)

    if not pair_depths:
        return np.full((out_h, out_w), np.nan, dtype=np.float32)

    depth_t = np.nanmedian(np.stack(pair_depths, axis=0), axis=0).astype(np.float32)
    return cv2.resize(depth_t, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def accumulate_flow_depth_map(frames: list, full_baseline_m: float,
                              focal_px: float, out_h: int, out_w: int,
                              thumb_w: int = 320, thumb_h: int = 180,
                              min_disp: float = 0.5):
    """Depth from accumulated per-pair flow at full-burst baseline accuracy.

    Each consecutive frame pair has a small displacement (~0.1 m) so
    Farneback tracks reliably.  Summing the per-pair flow vectors gives
    the same total displacement as the first→last pair but without the
    large-displacement tracking failures that occur when features move
    more than ~30 px between frames.

    Returns (depth_map HxW float32 metres, mean_radial_px float, focal_t float).
    mean_radial_px is the accumulated outward radial flow over the full burst,
    consistent with the calibration walk which also used a full-burst baseline.
    """
    import cv2
    scale_x = thumb_w / frames[0].shape[1]
    focal_t = focal_px * scale_x

    grays = [cv2.cvtColor(cv2.resize(f, (thumb_w, thumb_h),
                                     interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY)
             for f in frames]

    accum_x = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    accum_y = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        accum_x += flow[..., 0]
        accum_y += flow[..., 1]

    cx = thumb_w / 2.0
    cy = thumb_h / 2.0
    xg = (np.arange(thumb_w, dtype=np.float32) - cx)[None, :]
    yg = (np.arange(thumb_h, dtype=np.float32) - cy)[:, None]
    rg = np.sqrt(xg ** 2 + yg ** 2) + 1e-3

    radial = accum_x * xg / rg + accum_y * yg / rg
    mean_radial_px = float(np.clip(radial, 0, None).mean())

    with np.errstate(divide='ignore', invalid='ignore'):
        depth_t = np.where(radial >= min_disp,
                           focal_t * full_baseline_m / radial,
                           np.nan).astype(np.float32)

    return (cv2.resize(depth_t, (out_w, out_h),
                       interpolation=cv2.INTER_NEAREST),
            mean_radial_px, focal_t)


# ---------------------------------------------------------------------
# DepthFusion: multi-view consistency filter.
#
# A 3D point is real if it survives free-space carving from other views.
# Starburst rays that land at 50 m will be in the "free space" region
# of any frame that saw the actual wall at 3 m in that direction.
#
# For each new depth map we:
#   1. Project every pixel to a 3D candidate point P.
#   2. Project P into each buffered (pose, depth) view.
#   3. If a buffered view saw something CLOSER than P in that direction,
#      it has carved P as free space.  Count free-space votes.
#   4. Keep P only if fewer than CARVE_FRAC of visible views carved it.
# ---------------------------------------------------------------------

class DepthFusion:
    """Rolling buffer of depth maps with multi-view carving filter."""

    CARVE_FRAC = 0.40   # reject if >40% of visible views carve this point

    def __init__(self, K: np.ndarray, max_frames: int = 8):
        self.K   = K
        self.buf: list = []          # [(pose, depth_HxW), ...]
        self.max = max_frames

    def reset(self):
        self.buf.clear()

    def push(self, pose, depth_m: np.ndarray):
        """Add a new (pose, depth) observation to the buffer."""
        self.buf.append((pose, depth_m))
        if len(self.buf) > self.max:
            self.buf.pop(0)

    def filter(self, pose, depth_m: np.ndarray) -> np.ndarray:
        """Return a copy of depth_m with multi-view-inconsistent pixels NaN'd.

        Points that are carved as free space by CARVE_FRAC of buffered views
        are set to NaN so they don't contribute to the voxel cloud."""
        if len(self.buf) == 0:
            return depth_m

        H, W  = depth_m.shape
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        yaw, pitch, t = pose
        R_cur = G.R_no_roll(yaw, pitch)

        # Unproject current depth to camera-space rays (unit dirs) + depths.
        ug = (np.arange(W, dtype=np.float32) - cx) / fx   # (W,)
        vg = (np.arange(H, dtype=np.float32) - cy) / fy   # (H,)
        ug2, vg2 = np.meshgrid(ug, vg)                     # (H,W)
        d_cur = depth_m.copy()
        valid = np.isfinite(d_cur) & (d_cur > 0.1)

        # World coords of each valid pixel.
        cam_dirs = np.stack([ug2, vg2, np.ones_like(ug2)], axis=-1)  # (H,W,3)
        world_pts = (cam_dirs * d_cur[..., None]) @ R_cur.T + t       # (H,W,3)

        free_votes = np.zeros((H, W), dtype=np.int16)
        vis_counts = np.zeros((H, W), dtype=np.int16)

        for b_pose, b_depth in self.buf:
            b_yaw, b_pitch, b_t = b_pose
            R_buf = G.R_no_roll(b_yaw, b_pitch)

            # Project world_pts into buffered camera.
            rel   = (world_pts - b_t) @ R_buf          # (H,W,3) cam coords
            Zb    = rel[..., 2]                        # depth in buf cam
            in_front = (Zb > 0.1)
            u_b = (rel[..., 0] / Zb * fx + cx).astype(np.int32)
            v_b = (rel[..., 1] / Zb * fy + cy).astype(np.int32)
            in_frame = in_front & (u_b >= 0) & (u_b < W) & (v_b >= 0) & (v_b < H)

            pixels_ok = valid & in_frame
            if not pixels_ok.any():
                continue

            ub_ok = u_b[pixels_ok]
            vb_ok = v_b[pixels_ok]
            Zb_ok = Zb[pixels_ok]

            b_meas = b_depth[vb_ok, ub_ok]
            # Carved if buffered view saw something closer (allowing 15% margin)
            carved = np.isfinite(b_meas) & (b_meas < Zb_ok * 0.85)

            vis_counts[pixels_ok] += 1
            free_votes[pixels_ok] += carved.astype(np.int16)

        # Reject pixels carved by >= CARVE_FRAC of visible views.
        seen = vis_counts > 0
        carve_ratio = np.where(seen, free_votes / np.maximum(vis_counts, 1), 0.0)
        bad = seen & (carve_ratio >= self.CARVE_FRAC)
        out = d_cur.copy()
        out[bad] = np.nan
        return out


# ---------------------------------------------------------------------
# GeometryMapper: FEATURE-localized SLAM.
#   pose  = relocalize(frame's semantic peaks -> 3D feature DB) via PnP
#   seed  = input dead-reckon (ONLY a PnP gate / fallback, not the pose)
#   map   = depth -> voxel cloud + coverage; peaks -> 3D feature DB grow
# The feature DB (3D semantic features) is what lets us localize any shot
# in a scanned area; it is shared across heroes.
# ---------------------------------------------------------------------

import carto_features as FT


def _semantic_radial_vel(prev_peaks, curr_peaks, img_w, img_h,
                          max_dist=350):
    """Estimate forward camera translation from semantic feature displacement.

    Match ResNet peaks between consecutive frames by channel identity (same
    channel = same semantic concept → same real-world region).  Subtract the
    median shift (dominated by look-rotation) to isolate the translation
    residual, then return its mean radial component.

    Positive  →  camera moved forward (features expand outward from centre).
    Near zero →  no translation — stuck against a wall or stationary.

    max_dist is large (350 px) so matches survive the 6-8 s inter-frame gap
    and the ±18° look oscillation (~335 px at 103° FOV / 1920 width).
    """
    if prev_peaks is None or curr_peaks is None:
        return 0.0
    if len(prev_peaks) == 0 or len(curr_peaks) == 0:
        return 0.0

    cx, cy = img_w / 2.0, img_h / 2.0
    pc = prev_peaks[:, 0].astype(np.int32)
    cc = curr_peaks[:, 0].astype(np.int32)
    common = np.intersect1d(pc, cc)
    if len(common) == 0:
        return 0.0

    pu_l, pv_l, du_l, dv_l = [], [], [], []
    md2 = max_dist ** 2

    for ch in common:
        pm = prev_peaks[pc == ch, 1:3]   # (N, 2) u, v
        cm = curr_peaks[cc == ch, 1:3]   # (M, 2) u, v
        for pu, pv in pm:
            d2 = (cm[:, 0] - pu) ** 2 + (cm[:, 1] - pv) ** 2
            idx = int(d2.argmin())
            if d2[idx] > md2:
                continue
            pu_l.append(pu); pv_l.append(pv)
            du_l.append(float(cm[idx, 0] - pu))
            dv_l.append(float(cm[idx, 1] - pv))

    if len(du_l) < 4:
        return 0.0

    du = np.array(du_l, np.float32)
    dv = np.array(dv_l, np.float32)
    pu = np.array(pu_l, np.float32)
    pv = np.array(pv_l, np.float32)

    # Remove bulk rotation: median shift is the dominant rotational component.
    # Residual is the differential (translation-induced) displacement.
    du -= np.median(du)
    dv -= np.median(dv)

    rx = pu - cx; ry = pv - cy
    r  = np.sqrt(rx ** 2 + ry ** 2) + 1e-3
    return float(np.mean((du * rx + dv * ry) / r))


class GeometryMapper:
    def __init__(self, map_dir: Path, K, motion_cfg, depth: DepthSource,
                 voxel=0.15, cover_cell=0.5, depth_step=8,
                 extractor: "FT.FeatureExtractor | None" = None):
        self.map_dir = map_dir
        self.K = K
        self.cfg = motion_cfg
        self.depth = depth
        self.cloud = G.VoxelCloud(voxel)
        self.cover = G.CoverageGrid(cover_cell)
        self.fmap = FT.FeatureMap3D()           # 3D semantic feature DB
        self.extractor = extractor              # ResNet multi-scale peaks
        self.depth_step = depth_step
        self.pose = (0.0, 0.0, np.zeros(3))      # yaw,pitch,t (spawn origin)
        self.scale = motion_cfg.get("da_scale", 1.0)
        self.localized = False                   # last tick's pose came from features?
        self.hazards = []
        self.n = 0
        self._prev_peaks = None                  # peaks from previous tick for motion est.
        self.fusion = DepthFusion(K, max_frames=8)  # multi-view carving filter
        self.flow_radar  = np.full(N_RADAR_BINS, np.nan, dtype=np.float32)
        self.scan_count  = 0

    def start_scan(self, flow_radar: np.ndarray, base_yaw: float) -> None:
        """Store flow radar from the just-completed spin."""
        self.scan_count += 1
        self.flow_radar  = flow_radar.copy()

    def set_pose(self, yaw, pitch, t):
        self.pose = (yaw, pitch, np.asarray(t, float))

    def step(self, frame, prev_frame, mdx, mdy,
             depth_override: "np.ndarray | None" = None,
             flow_disp_m: "float | None" = None):
        """Localize via features, then map geometry + grow feature DB.
        Returns (added_voxels, cov, info).

        depth_override : pre-computed HxW metric depth map (metres).
        flow_disp_m    : actual forward displacement this tick estimated from
                         optical flow (metres).  Used for dead reckoning when
                         PnP fails — zero when Mei hit a wall, nonzero when
                         she genuinely moved."""
        self.n += 1
        prev_pose = self.pose
        rad_per_px = self.cfg["mouse_rad_per_px"]

        # Update yaw/pitch from the mouse moves we just issued (exact).
        prev_yaw, prev_pitch, prev_t = prev_pose
        new_yaw   = prev_yaw + mdx * rad_per_px
        new_pitch = float(np.clip(prev_pitch + mdy * rad_per_px, -1.4, 1.4))

        # Seed = known rotation + last known position.
        seed = (new_yaw, new_pitch, prev_t)

        # Extract peaks once; reuse for both relocalization and feature DB growth.
        peaks_cache = None
        if self.extractor is not None:
            peaks_cache = self.extractor.peaks(frame)

        # --- POSE = feature relocalization; seed gates the 2D<->3D match ---
        pose_feat, inl = self._relocalize_from_peaks(peaks_cache, seed)
        if pose_feat is not None:
            jump = float(np.linalg.norm(
                np.array(pose_feat[2]) - np.array(prev_pose[2])))
            if jump > 3.0 and not self.localized:
                # Large first-relocalization jump: old geometry was projected
                # from the wrong seed position — discard it.
                self.cloud.clear()
                self.cover.reset()
                self.fusion.reset()
            self.pose = pose_feat
            self.localized = True
        else:
            # Dead reckon from flow: advance position by actual displacement.
            # This correctly gives zero when Mei walks into a wall (no parallax).
            if flow_disp_m is not None and flow_disp_m > 0.05:
                fwd = np.array([math.sin(new_yaw), 0.0, math.cos(new_yaw)])
                new_t = prev_t + fwd * flow_disp_m
            else:
                new_t = prev_t
            self.pose = (new_yaw, new_pitch, new_t)
            self.localized = False
        yaw, pitch, t = self.pose

        # --- depth: caller-supplied > DA inference ---
        if depth_override is not None:
            depth_m = depth_override
        else:
            da = self.depth.relative(frame)
            depth_m = da * self.scale if da is not None else None

        added = 0
        n_newfeat = 0
        if depth_m is not None and np.isfinite(depth_m).any():
            # Multi-view carving: filter depth_m against the fusion buffer
            # to remove pixels that other views carved as free space.
            depth_fused = self.fusion.filter((yaw, pitch, t), depth_m)
            # Push this depth into the buffer AFTER filtering so it becomes
            # a reference for future frames.
            self.fusion.push((yaw, pitch, t), depth_m)
            # Clip depth outliers per-frame before projecting — DA scale drifts
            # at long range so the top 5% of pixels (far corridors, sky) produce
            # rays that diverge wildly in world space.
            d95 = float(np.nanpercentile(depth_fused, 95))
            depth_fused = np.where(depth_fused > d95, np.nan, depth_fused)
            # geometry
            pts, _ = G.unproject(depth_fused, yaw, pitch, t, self.K,
                                 step=self.depth_step)
            added = self.cloud.add(pts)
            cp = G.cam_pos(yaw, pitch, t)
            self.cover.observe(pts, (cp[0], cp[2]), yaw)
            if peaks_cache is not None:
                n_newfeat = self._place_features_from_peaks(
                    peaks_cache, depth_m, yaw, pitch, t)
        # Forward depth: median of a small annular ring around center.
        # The exact center pixel has near-zero radial flow so is always NaN
        # in flow depth — sample a ring at ~10% of frame width instead.
        center_depth_m = None
        if depth_m is not None:
            dH, dW = depth_m.shape
            r0, r1 = max(1, dH // 10), max(2, dH // 5)
            cy2, cx2 = dH // 2, dW // 2
            yg2 = (np.arange(dH) - cy2)[:, None]
            xg2 = (np.arange(dW) - cx2)[None, :]
            rg2 = np.sqrt(xg2**2 + yg2**2)
            ring = (rg2 >= r0) & (rg2 <= r1)
            vals = depth_m[ring]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                center_depth_m = float(np.median(vals))

        # Semantic feature radial velocity: did the camera actually translate?
        # Uses consecutive SLAM-frame peak sets (already computed above).
        feat_radial = _semantic_radial_vel(
            self._prev_peaks, peaks_cache, self.K[0, 2] * 2, self.K[1, 2] * 2)
        self._prev_peaks = peaks_cache

        return added, self.cover.stats(), {"inliers": inl, "newfeat": n_newfeat,
                                            "localized": self.localized,
                                            "center_depth_m": center_depth_m,
                                            "feat_radial": feat_radial}

    def _relocalize_from_peaks(self, peaks, seed):
        """Pose from pre-computed peaks. Returns (pose, n_inliers) or (None, 0)."""
        if peaks is None or len(self.fmap) == 0 or peaks.shape[0] == 0:
            return None, 0
        pose, inl, _ = FT.relocalize(peaks, self.fmap, self.K, seed=seed)
        return (pose, inl) if pose is not None else (None, 0)

    def _place_features_from_peaks(self, peaks, depth_m, yaw, pitch, t):
        """Place pre-computed peaks into 3D feature DB."""
        if peaks.shape[0] == 0:
            return 0
        H, W = depth_m.shape
        R = G.R_no_roll(yaw, pitch)
        cp = G.cam_pos(yaw, pitch, t)
        xyz, ch, resp = [], [], []
        for c, u, v, val in peaks:
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            d = depth_m[vi, ui]
            if not np.isfinite(d) or d < 0.2 or d > 50.0:
                continue
            x = (u - self.K[0, 2]) * d / self.K[0, 0]
            y = (v - self.K[1, 2]) * d / self.K[1, 1]
            Pw = np.array([x, y, d]) @ R + cp
            xyz.append(Pw); ch.append(int(c)); resp.append(float(val))
        if not xyz:
            return 0
        return self.fmap.add(np.array(xyz, np.float32),
                             np.array(ch), np.array(resp))

    def mark_hazard(self, note):
        cp = G.cam_pos(*self.pose)
        self.hazards.append({"n": self.n, "note": note,
                             "cam_xyz": [float(c) for c in cp]})

    def save(self):
        self.map_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.map_dir / "cloud.npy", self.cloud.points())
        self.fmap.save(self.map_dir / "features.npz")
        (self.map_dir / "coverage.json").write_text(json.dumps({
            "voxels": len(self.cloud), "features3d": len(self.fmap),
            **self.cover.stats(), "hazards": self.hazards,
            "pose": {"yaw": self.pose[0], "pitch": self.pose[1],
                     "t": [float(x) for x in self.pose[2]]},
        }, indent=2))

    def load(self):
        ok = False
        f = self.map_dir / "cloud.npy"
        if f.exists():
            self.cloud.add(np.load(f)); ok = True
        ff = self.map_dir / "features.npz"
        if ff.exists():
            self.fmap.load(ff); ok = True
        return ok


# ---------------------------------------------------------------------
from carto_explorer import FrontierExplorer


# ---------------------------------------------------------------------
# Mei wall appearance detector -- the START SIGNAL for cartography.
# The user places a Mei ice wall somewhere in the spawn room; the bot
# polls until it detects the blue wall appearing (frame diff + blue HSV
# mask), then immediately calibrates scale + begins exploring.
# ---------------------------------------------------------------------

def _mei_wall_metric_dist(frame_bgr):
    """Return metric distance to Mei wall using its known 2.0m height.

    d = fy * 2.0 / h_px  (pinhole, OW2 74° vFOV)
    Returns float metres or None if wall not clearly detected."""
    import cv2 as _cv2
    hsv = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv,
                        np.array([90, 50, 60],  np.uint8),
                        np.array([130, 255, 255], np.uint8))
    nl, lbl, stats, _ = _cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nl < 2:
        return None
    best = 1 + int(stats[1:, _cv2.CC_STAT_AREA].argmax())
    if stats[best, _cv2.CC_STAT_AREA] < 2000:
        return None
    h_px = stats[best, _cv2.CC_STAT_HEIGHT]
    if h_px < 30:
        return None
    H = frame_bgr.shape[0]
    fy = (H / 2.0) / np.tan(np.radians(74) / 2)
    return float(fy * 2.0 / h_px)


def _calibrate_flow_velocity(io, depth_src, K, da_post, da_scale, frame_a):
    """Walk toward the Mei wall to fit flow_vel_cal.

    true_disp is measured from the wall's pixel-height change (exact metric,
    no da_scale dependency).  avg_depth from DA absorbs any remaining
    da_scale error into flow_vel_cal.

    flow_vel_cal satisfies:  disp_m = mean_radial * da_depth / THUMB_MEAN_R / flow_vel_cal"""
    import cv2 as _cv2
    thumb_w, thumb_h = 320, 180

    # Measure exact metric distance from wall pixel height in frame_a.
    d_a = _mei_wall_metric_dist(frame_a)
    if d_a is None:
        print("  flow vel cal skipped (wall not detected in start frame) — using 1.0")
        return 1.0
    print(f"  wall dist start: {d_a:.2f}m")

    # Walk straight at the wall for 0.6s.
    io.hold("w", True)
    time.sleep(0.6)
    io.hold("w", False)
    time.sleep(0.1)

    frame_b = io.grab()
    d_b = _mei_wall_metric_dist(frame_b)
    if d_b is None:
        print("  flow vel cal skipped (wall not detected in end frame) — using 1.0")
        return 1.0, 5.5, 5.5
    print(f"  wall dist end:   {d_b:.2f}m")

    true_disp = d_a - d_b   # positive = moved closer
    if true_disp < 0.05:
        print(f"  flow vel cal skipped (true_disp={true_disp:.3f}m too small) — using 1.0")
        return 1.0, 5.5, 5.5

    def _gray(f):
        return _cv2.cvtColor(
            _cv2.resize(f, (thumb_w, thumb_h), interpolation=_cv2.INTER_AREA),
            _cv2.COLOR_BGR2GRAY)
    fl = _cv2.calcOpticalFlowFarneback(
        _gray(frame_a), _gray(frame_b), None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    cx, cy = thumb_w / 2.0, thumb_h / 2.0
    xg = (np.arange(thumb_w, dtype=np.float32) - cx)[None, :]
    yg = (np.arange(thumb_h, dtype=np.float32) - cy)[:, None]
    rg = np.sqrt(xg**2 + yg**2) + 1e-3
    mean_rad = float(np.clip(fl[..., 0]*xg/rg + fl[..., 1]*yg/rg, 0, None).mean())

    if mean_rad < 0.3:
        print(f"  flow vel cal skipped (mean_rad={mean_rad:.2f}px too small) — using 1.0")
        return 1.0, 5.5, 5.5

    # avg_depth from DA (scaled) — matches what da_depth_last is at runtime.
    # flow_vel_cal absorbs any da_scale error.
    if da_post is not None:
        da_b_raw = depth_src.relative(frame_b)
        if da_b_raw is not None:
            avg_depth = float(np.nanpercentile(
                np.stack([da_post, da_b_raw]).ravel() * da_scale, 75))
        else:
            avg_depth = float(np.nanpercentile(da_post * da_scale, 75))
    else:
        # Fall back: use wall distances as proxy for scene depth.
        avg_depth = (d_a + d_b) / 2.0

    flow_vel_cal = float(mean_rad * avg_depth / THUMB_MEAN_R / true_disp)
    move_speed_m  = true_disp / 0.6   # actual m/s from calibration walk
    print(f"  flow vel: Δd={true_disp:.3f}m  rad={mean_rad:.2f}px  "
          f"avg_depth={avg_depth:.2f}m  flow_vel_cal={flow_vel_cal:.4f}  "
          f"move_speed={move_speed_m:.2f}m/s")

    # Strafe calibration — wall still visible at d_b.
    # Strafe right for 0.5s collecting frames, then restore.
    focal_thumb = float(K[0, 0]) * thumb_w / io.W
    strafe_frames = [frame_b]
    strafe_t0 = time.time()
    io.hold("d", True)
    while time.time() - strafe_t0 < 0.5:
        strafe_frames.append(io.grab())
        time.sleep(0.033)
    io.hold("d", False)
    strafe_actual = time.time() - strafe_t0
    io.hold("a", True)
    time.sleep(strafe_actual)
    io.hold("a", False)

    total_horiz_px = 0.0
    for i in range(len(strafe_frames) - 1):
        fl_s = _cv2.calcOpticalFlowFarneback(
            _gray(strafe_frames[i]), _gray(strafe_frames[i + 1]), None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
        total_horiz_px += float(np.abs(fl_s[..., 0]).mean())

    if total_horiz_px > 0.3:
        # total_horiz_px = focal_thumb × V_strafe × strafe_actual / d_b
        strafe_speed = total_horiz_px * d_b / (focal_thumb * strafe_actual)
    else:
        strafe_speed = move_speed_m
    print(f"  strafe: horiz={total_horiz_px:.1f}px  D={d_b:.2f}m  "
          f"focal_t={focal_thumb:.1f}  strafe_speed={strafe_speed:.2f}m/s")
    return flow_vel_cal, move_speed_m, strafe_speed


def wait_for_mei_wall(io: "ScreenIO", depth_src: "DepthSource", K,
                      timeout: float = 300.0):
    """Block until the user completes a Mei ice wall placement.

    Uses the same OCR pipeline as mei-wall-detect.py: tight per-word
    boxes, grayscale→invert→drop-midtones→upscale 3x, PSM 7 single-line.
    Returns (post_wall_frame, da_scale, flow_vel_cal)."""
    import cv2
    import pytesseract
    print("waiting for Mei ice wall placement (press E -> click to place)...")

    # Tight fractional boxes from mei-wall-detect.py calibration at 1280x720.
    BUILD_X,  BUILD_Y  = (0.31, 0.37), (0.47, 0.53)
    CANCEL_X, CANCEL_Y = (0.625, 0.71), (0.47, 0.53)
    OCR_CFG = ("--psm 7 --oem 1 "
               "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def _ocr_box(frame, x_frac, y_frac):
        h, w = frame.shape[:2]
        x0, x1 = int(w * x_frac[0]), int(w * x_frac[1])
        y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0,
                          interpolation=cv2.INTER_CUBIC)
        inv = cv2.bitwise_not(gray)
        # Keep text body (very dark after invert) and outline ring (very bright).
        # Everything in between is scene noise -> white.
        keep = (inv < 30) | (inv > 180)
        cleaned = np.where(keep, inv, 255).astype(np.uint8)
        return pytesseract.image_to_string(cleaned, config=OCR_CFG).strip().upper()

    def _has_ui(frame):
        tb = _ocr_box(frame, BUILD_X,  BUILD_Y)
        tc = _ocr_box(frame, CANCEL_X, CANCEL_Y)
        return "BUILD" in tb or "CANCEL" in tb or "BUILD" in tc or "CANCEL" in tc

    POLL = 0.05
    t_end = time.time() + timeout
    state = "IDLE"

    while time.time() < t_end:
        time.sleep(POLL)
        frame = io.grab()

        if state == "IDLE":
            if _has_ui(frame):
                print("  placement UI detected, waiting for left-click...")
                state = "PENDING"

        elif state == "PENDING":
            if not _has_ui(frame):
                print("  wall placed, waiting for spawn...")
                time.sleep(0.8)
                post = io.grab()
                da = depth_src.relative(post)
                da_scale = 1.0
                if da is not None:
                    s = G.detect_mei_wall_da_scale(post, da)
                    if s is not None:
                        da_scale = s
                        print(f"  da_scale = {da_scale:.3f} (wall height fit)")
                    else:
                        print("  da_scale fit failed, using 1.0")

                # Walk toward the wall immediately — it vanishes in ~2s.
                # Use DA depth change as ground truth to fit flow→velocity.
                result = _calibrate_flow_velocity(
                    io, depth_src, K, da, da_scale, post)
                flow_vel_cal, move_speed, strafe_speed = result

                print("  ready to calibrate and explore")
                return post, da_scale, flow_vel_cal, move_speed, strafe_speed

    print("  timed out -- starting without wall anchor")
    return io.grab(), 1.0, 1.0, 5.5, 5.5


# ---------------------------------------------------------------------
# Auto-calibration: measure mouse_rad_per_px and move_speed from live
# gameplay immediately after the wall trigger. No hard-coded values.
# ---------------------------------------------------------------------

CALIB_SPIN_STEP    = 1000   # mouse pixels per spin step
CALIB_SPIN_SAMPLES = 150    # samples to collect (~5+ rotations at typical sensitivity)

# Mean pixel distance from image center for 320×180 thumbnails.
# For forward motion: mean_radial_px = THUMB_MEAN_R * Δz / Z
# so Δz = mean_radial_px * Z / THUMB_MEAN_R (no focal length needed).
THUMB_MEAN_R = 97.925   # px — precomputed for thumb_w=320 thumb_h=180


def auto_calibrate(io: "ScreenIO", depth_src: "DepthSource", K,
                   calib_path: Path, da_scale: float = 1.0,
                   flow_vel_cal: float = 1.0, move_speed: float = 5.5,
                   strafe_speed: float = 5.5):
    """Spin right a full 360° to measure mouse_rad_per_px.

    flow_vel_cal should be pre-measured by _calibrate_flow_velocity (called
    from wait_for_mei_wall) before the bot starts its 360° spin.

    Returns cfg dict: {mouse_rad_per_px, pitch_rad_per_px, da_scale,
                       flow_vel_cal, move_speed, strafe_speed}"""
    if calib_path.exists():
        cfg = json.loads(calib_path.read_text())
        if "flow_vel_cal" not in cfg:
            cfg["flow_vel_cal"] = 1.0
        if "move_speed" not in cfg:
            cfg["move_speed"] = 5.5
        if "strafe_speed" not in cfg:
            cfg["strafe_speed"] = cfg["move_speed"]
        print(f"calibration loaded: mouse_rad/px={cfg['mouse_rad_per_px']:.5f} "
              f"da_scale={cfg['da_scale']:.3f}  "
              f"flow_vel_cal={cfg['flow_vel_cal']:.3f}  "
              f"move_speed={cfg['move_speed']:.2f}m/s  "
              f"strafe_speed={cfg['strafe_speed']:.2f}m/s")
        return cfg

    import cv2
    print("calibrating mouse sensitivity (multi-rotation period detection)...")
    time.sleep(0.5)

    def _thumb(bgr):
        return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                          (160, 90)).ravel().astype(np.float32)

    ref   = _thumb(io.grab())
    corrs = []   # correlation at each sample
    total_px = 0

    # Spin fast, sampling every CALIB_SPIN_STEP pixels.
    # Collect CALIB_SPIN_SAMPLES samples; the period of the signal = 360°.
    while len(corrs) < CALIB_SPIN_SAMPLES:
        io.look(CALIB_SPIN_STEP, 0)
        total_px += CALIB_SPIN_STEP
        time.sleep(0.02)
        corrs.append(float(np.corrcoef(ref, _thumb(io.grab()))[0, 1]))

    print(f"  collected {len(corrs)} samples over {total_px} px", flush=True)

    # Autocorrelation of the (zero-mean) correlation signal.
    v  = np.array(corrs) - np.mean(corrs)
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    ac = ac / ac[0]

    # Search for the dominant period between 5 and N/2 samples.
    lo, hi = 5, len(ac) // 2
    peak_lag  = int(np.argmax(ac[lo:hi])) + lo
    period_px = peak_lag * CALIB_SPIN_STEP
    mouse_rad = (2 * math.pi) / period_px
    print(f"  period = {period_px} px (lag {peak_lag})  "
          f"mouse_rad/px = {mouse_rad:.5f} rad")

    # Spin back quickly.
    remaining = total_px
    while remaining > 0:
        chunk = min(remaining, CALIB_SPIN_STEP)
        io.look(-chunk, 0)
        remaining -= chunk
        time.sleep(0.02)
    time.sleep(0.3)

    cfg = {"mouse_rad_per_px": mouse_rad, "pitch_rad_per_px": mouse_rad,
           "da_scale": da_scale, "flow_vel_cal": flow_vel_cal,
           "move_speed": move_speed, "strafe_speed": strafe_speed}
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(json.dumps(cfg, indent=2))
    print(f"  saved -> {calib_path}")
    return cfg


# ---------------------------------------------------------------------
# Localizer  (offline PNG->pose tool + live respawn reloc)
# ---------------------------------------------------------------------

def localize_shot(map_dir: Path, image_path: Path, K):
    """Offline: estimate the camera pose of a screenshot within a saved
    map by matching its semantic peaks to the 3D feature DB (cold, no
    seed). Returns a pose dict or None."""
    import cv2
    import carto_features as FT
    fmap = FT.FeatureMap3D()
    fmap.load(map_dir / "features.npz")
    ex = FT.FeatureExtractor()
    img = cv2.imread(str(image_path))
    peaks = ex.peaks(img)
    pose, inl, ncorr = FT.relocalize(peaks, fmap, K, seed=None)
    if pose is None:
        return {"localized": False, "correspondences": ncorr}
    yaw, pitch, t = pose
    cp = G.cam_pos(yaw, pitch, t)
    return {"localized": True, "inliers": inl, "correspondences": ncorr,
            "yaw_deg": math.degrees(yaw), "pitch_deg": math.degrees(pitch),
            "cam_xyz": [float(c) for c in cp]}


# ---------------------------------------------------------------------
# Hard-negative saver
# ---------------------------------------------------------------------

class NegativeSaver:
    """Every alive cartographer frame has no players in it -> a clean
    hard negative for the character detector. Saves a sampled subset."""
    def __init__(self, out_dir: Path | None, every: int = 4):
        self.out = out_dir
        self.every = every
        self.i = 0
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, img_bgr, map_name: str):
        if not self.out:
            return
        self.i += 1
        if self.i % self.every:
            return
        import cv2
        ts = int(time.time() * 1000)
        cv2.imwrite(str(self.out / f"{map_name}_{ts}.png"), img_bgr)


# ---------------------------------------------------------------------
# Panoramic wall scan
# ---------------------------------------------------------------------

N_RADAR_BINS   = 36     # 10° per bin, full 360° polar map
SCAN_FPS       = 30     # target capture rate during spin
SPIN_SECS      = 1.5    # time budget for full 360° sweep → ~45 frames at 8° each

# Minimum mean Farneback flow (px, thumbnail space) that counts as textured
# scene.  Below this = blank wall with nothing for the tracker to follow.

# Metric distances displayed on radar PNGs


def _save_radar_png(flow_radar: np.ndarray,
                    map_dir: "Path",
                    spin_n: int,
                    fwd_deg: float) -> None:
    """Polar radar PNG — pure flow signal, no metric overlay.

    flow_radar : N_RADAR_BINS mean flow magnitudes (nan = not scanned).
    bin-0 = forward direction at scan start.
    Bar height = how open the direction is (tall = open, short = wall).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [radar #{spin_n:03d}: py -3.12 -m pip install matplotlib]",
              flush=True)
        return

    n = N_RADAR_BINS
    step_rad = 2 * math.pi / n
    thetas = np.array([b * step_rad for b in range(n)], dtype=np.float64)
    bar_w  = step_rad * 0.85

    # Use scan-relative thresholding so the radar works regardless of step size.
    # With ~8° steps the rotational flow alone is ~21px, swamping a fixed
    # absolute threshold.  Instead: bins in the bottom quartile of the scan
    # are "open" (low texture), everything above is "wall" (tracked texture).
    valid_mags = flow_radar[np.isfinite(flow_radar) & ~np.isnan(flow_radar)]
    if len(valid_mags) >= 4:
        mag_lo  = float(np.percentile(valid_mags, 25))   # open/wall split
        mag_hi  = float(np.percentile(valid_mags, 90))   # normalise wall height
    else:
        mag_lo  = float(valid_mags.min()) if len(valid_mags) else 0.0
        mag_hi  = float(valid_mags.max()) if len(valid_mags) else 1.0
    mag_hi = max(mag_hi, mag_lo + 1e-3)

    heights = np.zeros(n, dtype=np.float64)
    colors  = ["#222233"] * n

    for b in range(n):
        mag = float(flow_radar[b])
        if np.isnan(mag):
            heights[b] = 0.04           # unseen — tiny stub
            colors[b]  = "#222233"
        elif mag <= mag_lo:
            heights[b] = 1.0            # open / low texture — full bar
            colors[b]  = "#00ee88"      # green
        else:
            # Wall: bar height inversely proportional to flow strength
            t = min(1.0, (mag - mag_lo) / (mag_hi - mag_lo))
            heights[b] = max(0.12, 1.0 - t)
            colors[b]  = "#ff4433"      # red

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(8, 8), dpi=120)
    fig.patch.set_facecolor("#181820")
    ax.set_facecolor("#181820")
    ax.set_theta_zero_location("N")     # 0° at top = forward
    ax.set_theta_direction(-1)          # clockwise = right turn

    ax.bar(thetas, heights, width=bar_w, color=colors,
           alpha=0.88, edgecolor="#1a1a28", linewidth=0.4)

    # Angle labels every 30°
    ax.set_xticks([math.radians(d) for d in range(0, 360, 30)])
    ax.set_xticklabels([f"{d}°" for d in range(0, 360, 30)],
                       color="#aaaacc", fontsize=8)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], color="white")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#aaaacc")
    ax.spines["polar"].set_color("#444455")
    ax.grid(color="#333344", linewidth=0.5, alpha=0.6)

    ax.set_title(
        f"radar #{spin_n:03d}  ·  fwd={fwd_deg:.0f}°\n"
        f"green=open  red=wall (taller=farther)  dark=unseen",
        color="white", fontsize=9, pad=12)

    out = map_dir / f"radar_{spin_n:04d}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [radar → {out.name}]", flush=True)


def _save_scan_heightmap(strips: list, map_dir: "Path",
                         scan_n: int, base_yaw: float,
                         scan_t: np.ndarray,
                         depth_max: float = 15.0) -> None:
    """2-D radial heightmap: angle (0-360°) × world-height, coloured by depth.

    strips : list of (pts_Nx3_world, yaw_rad) — the voxel strips projected
             from each scan DA frame.
    scan_t : 3-vector camera position at scan time (world coords).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    N_A = N_RADAR_BINS          # angle bins — same as radar
    N_H = 40                    # height bins
    H_LO, H_HI = -2.0, 4.0    # metres relative to camera Y

    depth_grid = np.full((N_H, N_A), np.nan, dtype=np.float32)
    cam_y = float(scan_t[1])

    for pts, yaw_rad in strips:
        if len(pts) == 0:
            continue
        yaw_rel = float((yaw_rad - base_yaw) % (2 * math.pi))
        bin_a = int(yaw_rel / (2 * math.pi) * N_A) % N_A

        heights = pts[:, 1] - cam_y
        dx = pts[:, 0] - float(scan_t[0])
        dz = pts[:, 2] - float(scan_t[2])
        depths = np.sqrt(dx ** 2 + dz ** 2)

        for h, d in zip(heights.tolist(), depths.tolist()):
            if not (H_LO <= h <= H_HI and 0.2 < d < depth_max):
                continue
            bin_h = int((h - H_LO) / (H_HI - H_LO) * N_H)
            bin_h = max(0, min(N_H - 1, bin_h))
            if np.isnan(depth_grid[bin_h, bin_a]) or d < depth_grid[bin_h, bin_a]:
                depth_grid[bin_h, bin_a] = d

    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
    fig.patch.set_facecolor("#0d0d12")
    ax.set_facecolor("#0d0d12")

    im = ax.imshow(depth_grid, aspect="auto", origin="lower",
                   extent=[0, 360, H_LO, H_HI],
                   cmap="plasma_r", vmin=0.5, vmax=depth_max,
                   interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("depth (m)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.axhline(0, color="#00ff88", linewidth=0.8, alpha=0.6, linestyle="--",
               label="camera level")
    ax.set_xlabel("angle relative to scan start (°)", color="white", fontsize=9)
    ax.set_ylabel("height rel. camera (m)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_xticks(range(0, 361, 30))
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.set_title(
        f"scan #{scan_n:04d}  ·  radial heightmap  "
        f"(yellow=near  purple=far  black=unseen)",
        color="white", fontsize=9, pad=6)

    plt.tight_layout()
    out = map_dir / f"heightmap_{scan_n:04d}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [heightmap → {out.name}]", flush=True)


def do_wall_scan(io: "ScreenIO", capture: "CaptureThread",
                 da_thread: "DepthThread", base_pose: tuple,
                 motion_cfg: dict, map_dir: "Path", spin_n: int
                 ) -> "tuple[int, float | None, np.ndarray]":
    """360° horizontal sweep triggered on wall contact.

    Rotates continuously at ~30 fps, collecting every frame.  Between each
    consecutive frame pair the Farneback flow is computed and mapped to a
    radar bin.  All frames are tagged with their capture yaw and sent to DA
    for metric depth (arrives asynchronously via poll_scan()).

    Returns (n_frames, best_deg_relative, flow_radar[N_RADAR_BINS]).
    flow_radar is bot-relative: bin 0 = forward at scan start.
    """
    import cv2 as _cv2

    base_yaw, base_pitch, base_t = base_pose
    rad_per_px = motion_cfg["mouse_rad_per_px"]

    # Pixels for a full 360° turn, split into SCAN_FPS*SPIN_SECS equal steps.
    total_px  = int(2.0 * math.pi / rad_per_px)
    n_steps   = max(8, int(SCAN_FPS * SPIN_SECS))   # ~45
    step_px   = max(1, total_px // n_steps)
    sleep_s   = SPIN_SECS / n_steps                 # ~33 ms per step

    def _gray(f):
        s = _cv2.resize(f, (160, 90), interpolation=_cv2.INTER_AREA)
        return _cv2.cvtColor(s, _cv2.COLOR_BGR2GRAY)

    # Capture baseline frame before any rotation.
    scan_frames   = []    # list of (gray_160x90, world_yaw_rad)
    tagged_frames = []    # list of (full_frame, (yaw, pitch, t))

    f0 = capture.latest_frame(timeout=0.10)
    if f0 is not None:
        scan_frames.append((_gray(f0), base_yaw))

    cumulative_px = 0
    for _ in range(n_steps):
        io.look(step_px, 0)
        cumulative_px += step_px
        scan_yaw = base_yaw + cumulative_px * rad_per_px
        time.sleep(sleep_s)
        f = capture.latest_frame(timeout=sleep_s * 1.5)
        if f is not None:
            scan_frames.append((_gray(f), scan_yaw))
            tagged_frames.append((f, (scan_yaw, base_pitch, base_t, spin_n)))

    # Return camera to starting yaw.
    io.look(-cumulative_px, 0)

    da_thread.request_batch_tagged(tagged_frames)

    # ── Build flow_radar from ALL consecutive frame pairs ──────────────
    # Each pair (frame_i → frame_i+1) covers the angular slice swept during
    # that step.  We spread each measurement across the N_RADAR_BINS bins
    # that the pair's arc overlaps so the radar has no gaps.
    bins_per_step = max(1, N_RADAR_BINS // n_steps)
    flow_accum    = [[] for _ in range(N_RADAR_BINS)]

    for i in range(len(scan_frames) - 1):
        ga, yaw_a = scan_frames[i]
        gb, yaw_b = scan_frames[i + 1]

        flow = _cv2.calcOpticalFlowFarneback(
            ga, gb, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
        mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())

        # Map this pair's arc to radar bins (bot-relative: 0 = forward).
        arc_start_rel = (yaw_a - base_yaw) % (2 * math.pi)
        arc_end_rel   = (yaw_b - base_yaw) % (2 * math.pi)
        b0 = int(arc_start_rel / (2 * math.pi) * N_RADAR_BINS) % N_RADAR_BINS
        b1 = int(arc_end_rel   / (2 * math.pi) * N_RADAR_BINS) % N_RADAR_BINS
        # Fill all bins this arc covers (inclusive, wrapping).
        b = b0
        while True:
            flow_accum[b].append(mag)
            if b == b1:
                break
            b = (b + 1) % N_RADAR_BINS

    flow_radar = np.array(
        [float(np.mean(v)) if v else float("nan") for v in flow_accum],
        dtype=np.float32)

    # Save radar PNG immediately (DA metric overlay added later).
    _save_radar_png(flow_radar, map_dir, spin_n, math.degrees(base_yaw))

    # ── Best reorientation direction: lowest-flow arc ──────────────────
    valid = [(b, flow_radar[b]) for b in range(N_RADAR_BINS)
             if np.isfinite(flow_radar[b])]
    best_deg = None
    if valid and any(m < float("inf") for _, m in valid):
        best_b   = min(valid, key=lambda x: x[1])[0]
        raw_deg  = best_b * 360.0 / N_RADAR_BINS
        if raw_deg > 180.0:
            raw_deg -= 360.0
        best_deg = raw_deg

    return len(tagged_frames), best_deg, flow_radar


def _unproject_depth_fwd(depth_map: np.ndarray, K: np.ndarray,
                          yaw: float, max_d: float = 8.0) -> np.ndarray:
    """Unproject a forward-facing full-res depth map to local-frame 3D points.
    Camera at origin, pointing in direction yaw (world north = yaw=0).
    """
    H, W = depth_map.shape
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    R  = G.R_no_roll(yaw, 0.0)
    row_step = max(1, H // 50)
    col_step = max(1, W // 80)
    pts = []
    for row in range(0, H, row_step):
        for col in range(0, W, col_step):
            d = float(depth_map[row, col])
            if not (np.isfinite(d) and 0.3 < d <= max_d):
                continue
            ray = np.array([(col - cx) / fx, (row - cy) / fy, 1.0])
            ray /= np.linalg.norm(ray)
            pts.append(R @ (ray * d))
    return (np.array(pts, dtype=np.float32)
            if pts else np.empty((0, 3), dtype=np.float32))


def _project_scan_strips(scan_results: list, flow_radar: np.ndarray,
                          motion_cfg: dict, K: np.ndarray,
                          io_W: int, io_H: int) -> np.ndarray:
    """Project DA scan-strip depths to Nx3 local-frame points (scan pos = origin)."""
    da_scale_v   = motion_cfg.get("da_scale", 1.0)
    flow_vel_cal = motion_cfg.get("flow_vel_cal", 1.0)
    focal_px     = float(K[0, 0])
    n_steps_s    = max(8, int(SCAN_FPS * SPIN_SECS))
    step_angle_s = 2.0 * math.pi / n_steps_s
    focal_t      = focal_px * 160.0 / io_W

    pts_list = []
    for scan_depth, pose_tag in scan_results:
        s_yaw, s_pitch, s_t, _ = pose_tag
        scan_depth_m = scan_depth * da_scale_v

        bin_rel  = int((s_yaw % (2 * math.pi)) / (2 * math.pi) * N_RADAR_BINS) % N_RADAR_BINS
        flow_mag = float(flow_radar[bin_rel]) if np.isfinite(flow_radar[bin_rel]) else 0.0
        max_d    = (min(12.0, focal_t * step_angle_s / (flow_mag * flow_vel_cal))
                    if flow_mag > 0.3 else 12.0)

        sH, sW = scan_depth_m.shape
        sy   = sH / io_H
        fy_s = float(K[1, 1]) * sy
        cy_s = float(K[1, 2]) * sy
        R_s  = G.R_no_roll(s_yaw, s_pitch)
        cx_px   = sW // 2
        strip_d = np.nanmedian(scan_depth_m[:, max(0, cx_px - 4): cx_px + 5], axis=1)

        for row in range(0, sH, max(1, sH // 40)):
            d = float(strip_d[row])
            if not (np.isfinite(d) and 0.2 < d <= max_d):
                continue
            v_n = (row - cy_s) / fy_s
            ray = np.array([0.0, v_n, 1.0], dtype=np.float32)
            ray /= float(np.linalg.norm(ray))
            pt  = s_t + (R_s @ ray) * d
            pts_list.append(pt)

    return (np.array(pts_list, dtype=np.float32)
            if pts_list else np.empty((0, 3), dtype=np.float32))


def _render_topdown(pts: np.ndarray, out_path, radius: float = 15.0,
                    cell: float = 0.25) -> None:
    """Top-down heightmap centred on Mei, coloured by elevation.

    pts : Nx3 world-space points (X right/west, Y up, Z north).
    Negates X so east is right in the image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = int(2 * radius / cell) + 1
    grid = np.full((n, n), np.nan, dtype=np.float32)
    origin = n // 2

    for pt in pts:
        # negate X: world +X = west → display +X = east
        di = int((-pt[0] + radius) / cell)
        dj = int(( pt[2] + radius) / cell)
        if 0 <= di < n and 0 <= dj < n:
            y = float(pt[1])
            if np.isnan(grid[dj, di]) or y < grid[dj, di]:
                grid[dj, di] = y

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor("#181820")
    ax.set_facecolor("#181820")

    y_vals = grid[np.isfinite(grid)]
    vmin = float(np.percentile(y_vals, 2))  if len(y_vals) else -2.0
    vmax = float(np.percentile(y_vals, 98)) if len(y_vals) else  4.0
    if vmax - vmin < 0.5:
        vmax = vmin + 1.0

    im = ax.imshow(grid, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax,
                   extent=[-radius, radius, -radius, radius],
                   interpolation="nearest")

    # Mei + forward arrow (north = +Z = up)
    ax.plot(0, 0, "o", color="#00ff88", markersize=10,
            markeredgecolor="white", markeredgewidth=1.0, zorder=10)
    ax.annotate("", xy=(0, 1.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#00ff88", lw=2), zorder=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("elevation (m)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("X  (east, m)",  color="white", fontsize=10)
    ax.set_ylabel("Z  (north, m)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color="#333344", linewidth=0.5, alpha=0.4)
    for label, xy in [("N", (0, radius*.9)), ("E", (radius*.9, 0)),
                      ("S", (0, -radius*.9)), ("W", (-radius*.9, 0))]:
        ax.text(*xy, label, ha="center", va="center",
                color="#aaaacc", fontsize=12, fontweight="bold")
    ax.set_title("Top-down scan heightmap  ·  Mei at centre",
                 color="white", fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [topdown → {out_path}]", flush=True)


# ---------------------------------------------------------------------
# Main lockstep loop + state machine
# ---------------------------------------------------------------------

MOVE_KEYS = ("w", "a", "s", "d")


def run(args):
    io = ScreenIO()
    death = DeathDetector()
    K = G.intrinsics(G.focal_from_fov(args.fov, io.W), io.W, io.H)
    depth = DepthSource(prefer_da=not args.no_da)

    # calibrate mouse sensitivity + move speed (skips if cache exists)
    map_dir = Path(args.maps_dir) / args.map
    calib_path = map_dir / "calibration.json"

    # Wait for the user to place a Mei ice wall -- that IS the start
    # signal. It also gives us the DA depth scale from the wall's known
    # 2.0m height. Skipped on subsequent runs where calib is cached.
    da_scale       = args.da_scale
    flow_vel_cal   = 1.0
    move_speed_cal = 5.5
    strafe_speed_cal = 5.5
    if not calib_path.exists():
        _, da_scale, flow_vel_cal, move_speed_cal, strafe_speed_cal = wait_for_mei_wall(io, depth, K)

    motion_cfg = auto_calibrate(io, depth, K, calib_path, da_scale, flow_vel_cal, move_speed_cal, strafe_speed_cal)
    motion_cfg["da_scale"] = da_scale

    # ── Walk-then-scan mode ──────────────────────────────────────────────
    # Walk forward in bursts, render a flow-depth heightmap per burst.
    # On wall contact: backup 0.4 s → 360° scan → final topdown → exit.

    focal_px   = float(K[0, 0])
    move_speed  = float(motion_cfg.get("move_speed", 5.5))
    map_dir.mkdir(parents=True, exist_ok=True)

    print("warming up depth model...", flush=True)
    _wf = io.grab(); depth.relative(_wf); del _wf
    print("ready.", flush=True)

    da_thread = DepthThread(depth)
    da_thread.start()
    capture = CaptureThread(fps=30.0)
    capture.start()

    yaw          = 0.0
    pos          = np.zeros(3, dtype=np.float32)   # world XYZ; Y assumed flat
    burst_n      = 0
    world_pts: list = []   # all walk pts in world frame for final render
    da_depth_last = None   # most recent DA result (scaled), fills FOE blind spot
    da_scale_v    = motion_cfg.get("da_scale", 1.0)

    try:
        while True:
            da_thread.request(io.grab())

            # burst forward
            io.hold("w", True)
            capture.set_walking(True)
            t0 = time.time()
            burst_wall = False
            while time.time() - t0 < args.walk_burst:
                if capture.is_wall_contact():
                    burst_wall = True
                    break
                time.sleep(0.02)
            burst_actual = time.time() - t0
            io.release_all(MOVE_KEYS)
            capture.set_walking(False)

            # strafe wiggle: A→D→A fills the FOE dead zone with lateral flow.
            # Skip if we just hit a wall (about to back up and scan anyway).
            strafe_spd   = float(motion_cfg.get("strafe_speed", move_speed))
            wiggle_frames = []
            wiggle_dt     = 1.0 / 30.0   # ~30 fps grab interval
            if not burst_wall:
                for _key, _dur in (("a", 0.15), ("d", 0.30), ("a", 0.15)):
                    io.hold(_key, True)
                    _t0 = time.time()
                    while time.time() - _t0 < _dur:
                        wiggle_frames.append(io.grab())
                        time.sleep(wiggle_dt)
                    io.hold(_key, False)

            # pick up any DA result that arrived during the burst + wiggle
            da_result = da_thread.poll()
            if da_result is not None:
                da_depth_last = da_result * da_scale_v

            burst_frames = capture.burst_frames()
            burst_n += 1

            # flow depth for this burst (accurate near-range, synchronous).
            # Fill FOE dead zone: wiggle depth first, DA as final fallback.
            if len(burst_frames) >= 2 and burst_actual > 0.02:
                import cv2 as _cv2
                baseline = burst_actual * move_speed
                depth_flow, _, _ = accumulate_flow_depth_map(
                    burst_frames, baseline, focal_px, io.H, io.W)
                if len(wiggle_frames) >= 2:
                    depth_wiggle = wiggle_depth_map(
                        wiggle_frames, strafe_spd, wiggle_dt,
                        focal_px, io.H, io.W)
                    depth_combined = np.where(
                        np.isfinite(depth_flow), depth_flow, depth_wiggle)
                elif da_depth_last is not None:
                    da_resized = _cv2.resize(
                        da_depth_last, (io.W, io.H),
                        interpolation=_cv2.INTER_LINEAR)
                    depth_combined = np.where(
                        np.isfinite(depth_flow), depth_flow, da_resized)
                else:
                    depth_combined = depth_flow
                pts_local = _unproject_depth_fwd(depth_combined, K, yaw, max_d=8.0)
                if len(pts_local):
                    # save burst heightmap (bot at centre in local frame)
                    _render_topdown(pts_local,
                                    map_dir / f"burst_{burst_n:04d}.png",
                                    radius=10.0)
                    print(f"  burst {burst_n}: {len(pts_local)} pts  "
                          f"disp≈{baseline:.2f}m", flush=True)
                    # accumulate in world frame
                    wp = pts_local.copy()
                    wp[:, 0] += pos[0]; wp[:, 2] += pos[2]
                    world_pts.extend(wp.tolist())

            # advance position estimate
            disp = burst_actual * move_speed
            pos  = pos + np.array([math.sin(yaw), 0.0, math.cos(yaw)]) * disp

            if burst_wall:
                print(f"Wall at pos≈({pos[0]:.1f}, {pos[2]:.1f}) — "
                      f"backup + scan", flush=True)
                capture.clear_wall_contact()
                io.hold("s", True); time.sleep(0.4); io.hold("s", False)

                n_frames, _, flow_radar = do_wall_scan(
                    io, capture, da_thread,
                    (yaw, 0.0, pos), motion_cfg, map_dir, 1)

                print(f"Scan: {n_frames} frames queued. Waiting for DA...")
                scan_results = []
                deadline = time.time() + max(60.0, n_frames * 0.8)
                while len(scan_results) < n_frames and time.time() < deadline:
                    r = da_thread.poll_scan()
                    if r is not None:
                        scan_results.append(r)
                        print(f"  DA {len(scan_results)}/{n_frames}  ",
                              end="\r", flush=True)
                    else:
                        time.sleep(0.02)
                print(f"\nGot {len(scan_results)}/{n_frames} results")

                scan_pts = _project_scan_strips(
                    scan_results, flow_radar, motion_cfg, K, io.W, io.H)
                print(f"Scan projected {len(scan_pts)} pts", flush=True)

                # final topdown: walk pts + scan pts, centred on final pos
                walk_arr = (np.array(world_pts, dtype=np.float32)
                            if world_pts else np.empty((0, 3), np.float32))
                if len(walk_arr):
                    walk_arr[:, 0] -= pos[0]
                    walk_arr[:, 2] -= pos[2]
                all_pts = (np.concatenate([walk_arr, scan_pts], axis=0)
                           if len(walk_arr) else scan_pts)
                traveled = float(np.linalg.norm(pos[[0, 2]]))
                _render_topdown(all_pts, map_dir / "topdown_final.png",
                                radius=max(15.0, traveled + 8.0))
                break

    finally:
        da_thread.stop()
        capture.stop()
        io.release_all(MOVE_KEYS)
        return

    # ── Exploration loop (commented out) ────────────────────────────────
    if False:
        import carto_features as FT
        extractor = FT.FeatureExtractor(peak_thresh=args.peak_thresh)
        mapper = GeometryMapper(map_dir, K, motion_cfg, depth,
                                voxel=args.voxel, depth_step=args.depth_step,
                                extractor=extractor)
        if mapper.load():
            print(f"loaded base map '{args.map}': {len(mapper.cloud)} voxels, "
                  f"{len(mapper.fmap)} 3D features (extending it)")
    explorer = FrontierExplorer(motion_cfg["mouse_rad_per_px"],
                                cov_cell=mapper.cover.cell)
    negs = NegativeSaver(Path(args.negatives) if args.negatives else None,
                         every=args.neg_every)

    period = 1.0 / args.fps
    state = "EXPLORING"
    prev_frame = None
    last_look = (0, 0)
    held = ["w"]
    # Warm up both models before the loop so the first tick isn't secretly
    # slow (PyTorch first-inference JIT + HuggingFace preprocessor init).
    print("warming up models (first inference on CPU can take 30-60s)...",
          flush=True)
    _wf = io.grab()
    depth.relative(_wf)
    if extractor is not None:
        extractor.peaks(_wf)
    del _wf
    print("ready.", flush=True)

    # DA runs asynchronously; main loop submits a frame each tick and
    # picks up results whenever they arrive (~every 8s on a typical GPU).
    da_thread = DepthThread(depth)
    da_thread.start()

    # 30fps capture: wall contact detection + flow depth triangulation.
    capture = CaptureThread(fps=30.0)
    capture.start()

    focal_px    = float(K[0, 0])
    move_speed  = float(motion_cfg.get("move_speed", 5.5))
    depth_last    = None   # most recent usable depth map (flow or DA)
    da_depth_last = None   # DA-only depth — breaks circularity in flow_disp
    flow_disp_m   = None   # flow-based forward displacement from last burst
    flow_vel_cal  = motion_cfg.get("flow_vel_cal", 1.0)
    traj_xz: list = []    # (world_x, world_z) per tick for render_map.py

    # Per-scan strip accumulation for the 2-D radial heightmap.
    # Keys are spin_n; each entry is a list of (pts_Nx3, yaw_rad) tuples.
    _scan_strips: dict = {}
    _scan_meta:   dict = {}   # spin_n → (base_yaw, scan_t, n_expected_frames)

    t_end = time.time() + args.duration if args.duration else float("inf")
    last_held_w    = False   # did the previous tick hold W?
    stuck_walls    = 0       # consecutive WALL ticks with near-zero new voxels
    STUCK_THRESH   = 4       # back up after this many stuck ticks
    STUCK_VOX      = 500     # "near-zero" new voxels threshold
    print(f"cartographer running  map={args.map}  fps={args.fps}  "
          f"fov={args.fov}  (Ctrl-C to stop)")
    try:
        while time.time() < t_end:
            t0 = time.time()
            frame = io.grab()
            if death.is_dead(frame):
                if state != "DEAD":
                    print("[fall] death-cam -> freeze mapping, drop hazard")
                    io.release_all(MOVE_KEYS)
                    mapper.mark_hazard("fell off map")
                    state = "DEAD"
                    prev_frame = None
                time.sleep(period)
                continue
            elif state == "DEAD":
                print("[respawn] spawn room -> reset pose to spawn anchor")
                mapper.set_pose(0.0, 0.0, np.zeros(3))
                state = "EXPLORING"

            # ── Depth: compute from burst frames then optionally enrich with DA
            #
            # We no longer block the tick on DA inference.  Instead:
            #   flow_depth  — triangulated from burst-start vs burst-end frames;
            #                 metric, 0–10m reliable, computed in <50ms.
            #   da_depth    — async DA-V2 result (arrives ~every 8s); covers
            #                 far range where flow disparity is sub-pixel.
            #   Blend       — flow fills near range; DA fills far-range NaNs.
            #   depth_last  — previous tick's merged depth used as fallback when
            #                 neither is available (no burst, DA still running).
            #
            # DA is requested every tick; the thread ignores requests while busy
            # so naturally runs at one inference per ~8s.

            da_thread.request(frame)

            # Plan & act.
            added, cov, info = mapper.step(frame, prev_frame,
                                           last_look[0], last_look[1],
                                           depth_override=depth_last,
                                           flow_disp_m=flow_disp_m)
            flow_disp_m = None   # consumed; will be set again after next burst
            negs.maybe_save(frame, args.map)

            plan = explorer.propose(mapper.cloud, mapper.cover,
                                    mapper.pose, added)
            held = [k for k in plan["hold"] if k in ALLOWED_KEYS]

            # Stop, turn, then walk for a controlled burst.
            io.release_all(MOVE_KEYS)
            if plan["look"] != (0, 0):
                io.look(*plan["look"])
            last_look = plan["look"]
            for k in plan["tap"]:
                io.tap(k)

            burst_wall    = False
            depth_flow    = None
            burst_actual  = 0.0
            if held:
                for k in held:
                    io.hold(k, True)
                capture.set_walking(True)
                t_burst_start = time.time()
                t_burst_end   = t_burst_start + args.walk_burst
                while time.time() < t_burst_end:
                    if capture.is_wall_contact():
                        burst_wall = True
                        break
                    time.sleep(0.02)
                burst_actual = time.time() - t_burst_start
                io.release_all(MOVE_KEYS)
                capture.set_walking(False)

                # Accumulate per-pair flow over the full burst, then compute
                # depth from the summed flow + full baseline.  Each pair uses
                # a small displacement (~0.1 m) so Farneback is reliable;
                # summing recovers full-baseline accuracy without the
                # large-displacement tracking failures of a single first→last pair.
                burst_frames_list = capture.burst_frames()
                flow_disp_m    = None
                depth_flow     = None
                mean_radial_px = 0.0
                focal_t = focal_px * 320.0 / io.W
                if len(burst_frames_list) >= 2 and burst_actual > 0.02:
                    full_baseline = burst_actual * move_speed
                    depth_flow, mean_radial_px, focal_t = accumulate_flow_depth_map(
                        burst_frames_list, full_baseline, focal_px, io.H, io.W)
                    if mean_radial_px > 0.3 and da_depth_last is not None:
                        depth_75 = float(np.nanpercentile(da_depth_last, 75))
                        if np.isfinite(depth_75) and depth_75 > 0.3:
                            flow_disp_m = (mean_radial_px * depth_75
                                          / THUMB_MEAN_R / flow_vel_cal)

                if burst_wall:
                    yaw_now = mapper.pose[0]
                    cx_now  = float(mapper.pose[2][0])
                    cz_now  = float(mapper.pose[2][2])
                    explorer.mark_obstacle_ahead(cx_now, cz_now, yaw_now)
                    # Back away from the wall before scanning so the bot has
                    # clearance to actually walk in whichever direction the
                    # radar identifies as open.
                    io.hold("s", True)
                    time.sleep(0.4)
                    io.hold("s", False)
                    n_scan, flow_deg, flow_radar = do_wall_scan(
                        io, capture, da_thread, mapper.pose, motion_cfg,
                        map_dir, mapper.scan_count + 1)
                    mapper.start_scan(flow_radar, mapper.pose[0])
                    _scan_meta[mapper.scan_count] = (
                        mapper.pose[0],
                        np.array(mapper.pose[2], dtype=np.float32),
                        n_scan,
                        flow_radar.copy())

                    # Block all high-flow radar directions in the nav grid.
                    explorer.apply_radar_obstacles(
                        flow_radar, mapper.pose[0], cx_now, cz_now)

                    # Reorientation: longest clear radar arc → nav fallback.
                    radar_rel = explorer.radar_best_yaw(flow_radar)
                    if radar_rel is not None:
                        reorient_deg = math.degrees(radar_rel)
                        source = "radar"
                    else:
                        best_yaw, best_run = explorer.best_open_yaw(
                            cx_now, cz_now)
                        if best_run > 0:
                            dyaw = best_yaw - yaw_now
                            dyaw = (dyaw + math.pi) % (2*math.pi) - math.pi
                            reorient_deg = math.degrees(dyaw)
                            source = f"nav(run={best_run})"
                        else:
                            reorient_deg = 0.0
                            source = "none"

                    reorient_px = int(
                        math.radians(reorient_deg) / motion_cfg["mouse_rad_per_px"])
                    if abs(reorient_px) > 30:
                        io.look(reorient_px, 0)
                        last_look = (last_look[0] + reorient_px, last_look[1])
                    else:
                        reorient_deg = 0.0

                    explorer.report_wall_contact()
                    capture.clear_wall_contact()
                    print(f"  [wall scan: {n_scan} frames → DA, "
                          f"reoriented {reorient_deg:+.0f}° via {source}]",
                          flush=True)

                    # Stuck counter: consecutive walls with tiny new geometry.
                    if added < STUCK_VOX:
                        stuck_walls += 1
                    else:
                        stuck_walls = 0

                    if stuck_walls >= STUCK_THRESH:
                        stuck_walls = 0
                        print("  [stuck — backing up]", flush=True)
                        io.hold("s", True)
                        time.sleep(0.6)
                        io.hold("s", False)
                        # Clear nav obstacles ahead so A* can route freshly.
                        explorer.clear_obstacles_near(
                            cx_now, cz_now, radius=2)
                else:
                    stuck_walls = 0

            # Reproject scan frames as radial depth strips (one vertical
            # profile per angular step).  Full-frame unprojection fills the
            # cloud with millions of doorway voxels and causes the starburst;
            # the centre-column strip gives correct wall + doorway geometry
            # at ~45 strips × H/step voxels each instead of full H×W frames.
            while True:
                scan_result = da_thread.poll_scan()
                if scan_result is None:
                    break
                scan_depth, pose_tag = scan_result
                s_yaw, s_pitch, s_t, s_scan_n = pose_tag   # 4-tuple from do_wall_scan
                scan_depth_m = scan_depth * mapper.scale

                sH, sW = scan_depth_m.shape
                sy = sH / io.H
                fy_s = float(mapper.K[1, 1]) * sy
                cy_s = float(mapper.K[1, 2]) * sy
                R_s  = G.R_no_roll(s_yaw, s_pitch)

                cx_px   = sW // 2
                strip_d = np.nanmedian(
                    scan_depth_m[:, max(0, cx_px - 4): cx_px + 5], axis=1)

                floor_y_min = float(s_t[1]) - 3.0   # reject >3m below camera

                project_strip = False
                if s_scan_n in _scan_meta:
                    _, _, _, fr_s = _scan_meta[s_scan_n]
                    fr_valid = fr_s[np.isfinite(fr_s)]
                    if len(fr_valid):
                        flow_thresh = float(np.percentile(fr_valid, 75))
                        bin_rel = int(
                            ((s_yaw - _scan_meta[s_scan_n][0]) % (2 * math.pi))
                            / (2 * math.pi) * N_RADAR_BINS) % N_RADAR_BINS
                        project_strip = (np.isfinite(fr_s[bin_rel])
                                         and fr_s[bin_rel] >= flow_thresh)
                else:
                    project_strip = True   # meta not yet stored — project anyway

                pts_list = []
                step_r = max(1, mapper.depth_step)
                for row in range(0, sH, step_r):
                    d = float(strip_d[row])
                    if not (np.isfinite(d) and d > 0.2):
                        continue
                    v_n = (row - cy_s) / fy_s
                    ray = np.array([0.0, v_n, 1.0], dtype=np.float32)
                    ray /= float(np.linalg.norm(ray))
                    pt = s_t + (R_s @ ray) * d
                    if float(pt[1]) >= floor_y_min:
                        pts_list.append(pt)

                pts = (np.array(pts_list, dtype=np.float32)
                       if pts_list else np.empty((0, 3), np.float32))
                if project_strip and len(pts):
                    mapper.cloud.add(pts)
                    mapper.cover.observe(pts, (float(s_t[0]), float(s_t[2])), s_yaw)

                # Flush completed/superseded scans, accumulate current.
                for old_n in [k for k in list(_scan_strips) if k != s_scan_n]:
                    if old_n in _scan_meta and _scan_strips[old_n]:
                        ob, ot, _, ofr = _scan_meta.pop(old_n)
                        _save_scan_heightmap(
                            _scan_strips.pop(old_n), map_dir, old_n, ob, ot)

                _scan_strips.setdefault(s_scan_n, []).append((pts, s_yaw))
                if s_scan_n in _scan_meta:
                    base_yaw_s, scan_t_s, n_exp, _ = _scan_meta[s_scan_n]
                    if len(_scan_strips[s_scan_n]) >= n_exp:
                        _save_scan_heightmap(
                            _scan_strips.pop(s_scan_n),
                            map_dir, s_scan_n, base_yaw_s, scan_t_s)
                        del _scan_meta[s_scan_n]

            # Merge depth sources: flow (near) + DA (far) + fallback.
            da_result = da_thread.poll()
            if da_result is not None:
                da_scaled     = da_result * mapper.scale
                da_depth_last = da_scaled   # DA-only; used for flow displacement
            else:
                da_scaled = None

            t = mapper.pose[2]
            traj_xz.append((float(t[0]), float(t[2])))

            # DA is the calibrated depth source for geometry.
            # Flow depth is used only for odometry (flow_disp_m) and is NOT
            # written into depth_last — burst-frame flow produces 20-250m
            # junk values that create the starburst artifact in the cloud.
            if da_scaled is not None:
                depth_last = da_scaled
            # else: depth_last keeps previous DA depth

            # ── Inter-tick wall scoring (multi-signal backup) ─────────────
            feat_radial = info.get("feat_radial", 0.0)
            center_d    = info.get("center_depth_m")

            if not burst_wall and last_held_w:
                wall_hits = 0
                if feat_radial < 0.3:                        wall_hits += 2
                if added < 200:                              wall_hits += 1
                if center_d is not None and center_d < 0.4: wall_hits += 1
                if wall_hits >= 2:
                    yaw_now = mapper.pose[0]
                    cx_now  = float(mapper.pose[2][0])
                    cz_now  = float(mapper.pose[2][2])
                    explorer.mark_obstacle_ahead(cx_now, cz_now, yaw_now)
                    explorer.report_wall_contact()

            last_held_w = bool(held)

            prev_frame = frame
            el = time.time() - t0
            loc = "feat" if info["localized"] else "seed"
            cd_str = f"{center_d:.2f}" if center_d is not None else "n/a"
            da_tag   = " DA" if da_result is not None else ""
            wall_tag = " WALL" if burst_wall else ""
            depth_src = ("da"    if da_result is not None else
                         "stale" if depth_last is not None else "—")
            print(f"  tick {mapper.n:4d}  +{added:4d}vox  "
                  f"held={held}  depth={depth_src}  cdepth={cd_str}  "
                  f"[{loc} inl={info['inliers']}]{da_tag}{wall_tag}  "
                  f"{el*1000:.0f}ms",
                  flush=True)
            if mapper.n % 20 == 0:
                print(f"  -- voxels={len(mapper.cloud)} 3Dfeat={len(mapper.fmap)} "
                      f"seen={cov['seen_cells']} frontier={cov['frontier_cells']}")
            if el < period:
                time.sleep(period - el)
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        import signal as _signal
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)   # ignore further Ctrl-C
        da_thread.stop()
        capture.stop()
        io.release_all(MOVE_KEYS)
        # Flush any accumulated-but-unsaved heightmaps.
        for sn, strips in list(_scan_strips.items()):
            if sn in _scan_meta and strips:
                ob, ot, _, _fr = _scan_meta.pop(sn)
                _save_scan_heightmap(strips, map_dir, sn, ob, ot)
        print("saving map (do not interrupt)...", flush=True)
        mapper.save()
        if traj_xz:
            np.save(map_dir / "trajectory.npy", np.array(traj_xz, np.float32))
        print(f"saved map -> {map_dir}  ({len(mapper.cloud)} voxels, "
              f"{len(mapper.fmap)} 3D features, {len(mapper.hazards)} hazards)",
              flush=True)
        _signal.signal(_signal.SIGINT, _signal.SIG_DFL)   # restore


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="map name (load/save)")
    ap.add_argument("--maps-dir", default="maps")
    ap.add_argument("--negatives", default=None,
                    help="dir to save character-free hard-negative frames")
    ap.add_argument("--neg-every", type=int, default=4)
    ap.add_argument("--fps", type=float, default=4.0, help="lockstep rate")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run (0 = until Ctrl-C)")
    # --- geometry calibration ---
    ap.add_argument("--fov", type=float, default=103.0,
                    help="horizontal FOV in degrees (OW2 hipfire default)")
    ap.add_argument("--da-scale", type=float, default=1.0,
                    help="DepthAnything->metre scale override (auto-calibrated "
                         "via wall height on first run if not set)")
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--depth-step", type=int, default=8,
                    help="pixel stride when unprojecting depth")
    ap.add_argument("--no-da", action="store_true",
                    help="skip DepthAnything; use flow-triangulation depth only")
    ap.add_argument("--recalibrate", action="store_true",
                    help="delete saved calibration and re-run it")
    ap.add_argument("--peak-thresh", type=float, default=2.5,
                    help="ResNet feature peak threshold")
    ap.add_argument("--walk-burst", type=float, default=0.5,
                    help="seconds to hold W per tick (default 0.5)")
    # offline localizer query
    ap.add_argument("--localize", default=None,
                    help="PNG to localize within --map, then exit")
    args = ap.parse_args()

    if args.localize:
        K = G.intrinsics(G.focal_from_fov(args.fov, 2560), 2560, 1440)
        print(json.dumps(localize_shot(
            Path(args.maps_dir) / args.map, Path(args.localize), K), indent=2))
        return
    if args.recalibrate:
        calib = Path(args.maps_dir) / args.map / "calibration.json"
        if calib.exists():
            calib.unlink()
            print(f"deleted {calib} -- will re-calibrate on next run")
        return
    run(args)


if __name__ == "__main__":
    main()

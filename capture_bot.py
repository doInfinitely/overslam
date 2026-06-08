"""End-to-end capture bot for the OW2 hero gallery.

Run this on the Windows machine where Overwatch is open. Bring the hero
select to the foreground and HIGHLIGHT WUYANG -- that's the start
signal. The bot will then:

  for each hero in the gallery (bboxes 0..N-1):
    click center -> OCR the yellow-box name (skip click if already lit)
    click again to enter the hero-detail screen
    click SKINS:
      for each skin in the list (down-arrow scroll):
        OCR selected name from the cyan-outlined row
        if the orange EQUIP button is present, click it
        rotate 360 capturing frames (drag-based; calibrate
          steps-per-360 on the first skin via frame-diff)
        save frames as <root>/<hero>/<skin>/standing/####.png
      ESC up
    click EMOTES:
      for each emote in the list:
        OCR name; skip if 'RANDOM FROM FAVORITES'
        if EQUIP visible, click
        sample a couple of seconds of frames, find period via
          autocorrelation, lock the start frame
        for each rotation step, let the emote play one period,
          capture frames; rotate; repeat until 360 covered
        save as <root>/<hero>/<skin>/<emote>/rot<NN>/frame<MM>.png
      ESC up
    ESC up to hero select
  done

Dependencies (Windows): pyautogui, pydirectinput, pytesseract, opencv,
numpy, PIL. tesseract must be installed and on PATH or set via
PYTESSERACT_CMD env var.

Configuration knobs at the top of the file -- tune as needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Vision helpers we built earlier.
import gallery_ui

import pyautogui
import pydirectinput
import pytesseract

# PrintScreen key = panic kill switch. Press it at any time to
# immediately exit the bot (no cleanup).
try:
    import keyboard  # type: ignore
    keyboard.add_hotkey("print screen", lambda: os._exit(130))
    print("(press PRINTSCREEN to kill the bot)")
except Exception as _e:
    print(f"(warning: 'keyboard' module not available, PRINTSCREEN kill disabled: {_e})")

_TESS_CANDIDATES = [
    os.environ.get("PYTESSERACT_CMD"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for _c in _TESS_CANDIDATES:
    if _c and Path(_c).exists():
        pytesseract.pytesseract.tesseract_cmd = _c
        break

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0          # no implicit delay; we sleep explicitly
pydirectinput.FAILSAFE = False     # rely on PRINTSCREEN kill switch instead
pydirectinput.PAUSE = 0

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_BBOXES = "screenshots/gallery_bboxes.json"
DEFAULT_UI     = "screenshots/gallery_ui_bboxes.json"

# Start-signal poll: which hero name we wait to see highlighted before
# taking over. Normalize OCR comparisons by stripping spaces / dots.
START_HERO_NAME = "WUYANG"

# HEROES nav tab (top-left). Clicking it always returns to the hero
# gallery from anywhere in the hero menus -- our recovery anchor.
# Verified against the screenshot: "HEROES" text center is ~x=175,
# y=58 on a 2560x1440 screen (the blue tab spans ~x=100..260).
HEROES_TAB = (175, 58)

# Pause after each click before the next screenshot/action.
CLICK_PAUSE = 0.35

# Drag parameters for one rotation step. The drag MUST be across the
# middle of the character render -- dragging too close to the top
# rotates the camera instead of the model. (1280, 720) is dead-center
# on a 2560x1440 screen.
#
# A fast flick across the center of the render. The drag must contain
# at least a few intermediate motion events or the game treats it as a
# click -- pyautogui.move(..., duration) with a small duration emits
# those events. ~0.08s for an 800px flick matches a human drag speed.
DRAG_FROM   = (1280, 720)
DRAG_DELTA  = (800, 0)
DRAG_TIME   = 0.08
# A drag is built from this many small cursor saccades while the button
# is held -- the stream of move events is what the game reads as a drag.
DRAG_SACCADES = 24
DRAG_SACCADE_DELAY = 0.004   # seconds between saccades (~0.1s total)

# Timebox: spend this many seconds spinning+capturing per phase.
SKIN_SECONDS  = 5.0
EMOTE_SECONDS = 5.0
CAPTURE_FPS   = 30
# Number of small drag flicks to spread across one timebox window so the
# model does roughly a full turn over the 5s.
DRAGS_PER_WINDOW = 24

# Retained for the (now unused) calibration-based rotate_and_capture /
# detect_emote_period helpers.
MAX_ROTATION_STEPS = 80
ROTATION_DIFF_THRESHOLD = 0.04

# Emote period detection: sample this many seconds at ~30fps. OW2
# emotes typically loop in 3-7 s, so we need a longer sample window.
# We also skip a warm-up prefix at the start so the diff baseline is
# taken after the animation has actually begun (the first half-second
# is usually the same idle pose).
EMOTE_SAMPLE_SEC = 8.0
EMOTE_SAMPLE_FPS = 30
EMOTE_WARMUP_FRAMES = int(0.5 * 30)   # skip first 0.5s before measuring
EMOTE_PERIOD_MIN_FRAMES = 30          # >= 1 second
EMOTE_PERIOD_MAX_FRAMES = int(EMOTE_SAMPLE_SEC * EMOTE_SAMPLE_FPS) - 2

# Output root for captured frames. Defaults to ./captures relative to
# wherever the script is launched from -- override with the
# CAPTURE_ROOT env var (or --capture-root CLI flag).
CAPTURE_ROOT = Path(os.environ.get("CAPTURE_ROOT", "captures")).resolve()


# ---------------------------------------------------------------------
# Screenshot helper -- uses mss (fast, ~3ms) when available, with a
# per-thread instance (mss handles are not shareable across threads).
# Falls back to pyautogui.
# ---------------------------------------------------------------------

import threading

_thread_local = threading.local()
_SCREEN_W, _SCREEN_H = pyautogui.size()


def _get_sct():
    if getattr(_thread_local, "sct_tried", False):
        return getattr(_thread_local, "sct", None)
    _thread_local.sct_tried = True
    try:
        import mss
        factory = getattr(mss, "MSS", None) or mss.mss
        _thread_local.sct = factory()
    except Exception:
        _thread_local.sct = None
    return _thread_local.sct


def grab_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    sct = _get_sct()
    if sct is not None:
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
    pil = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def grab() -> np.ndarray:
    """Full-screen screenshot as BGR ndarray."""
    return grab_region(0, 0, _SCREEN_W, _SCREEN_H)


# ---------------------------------------------------------------------
# Background full-screen video recorder
# ---------------------------------------------------------------------

class Recorder(threading.Thread):
    """Continuously records the full screen to an mp4 at a target FPS,
    writing a sidecar CSV (video_frame_idx, epoch_ms) so frames can be
    aligned to event timestamps later -- the source of truth for
    re-segmenting state if the live OCR tagging is wrong."""

    def __init__(self, out_mp4: Path, fps: int = 30):
        super().__init__(daemon=True)
        self.fps = fps
        self.out_mp4 = out_mp4
        self._stop = threading.Event()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(out_mp4), fourcc, fps,
                                      (_SCREEN_W, _SCREEN_H))
        self.csv_f = out_mp4.with_suffix(".frames.csv").open("w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["video_frame_idx", "epoch_ms"])
        self.n_frames = 0

    def run(self):
        period = 1.0 / self.fps
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period
            frame = grab_region(0, 0, _SCREEN_W, _SCREEN_H)
            self.writer.write(frame)
            self.csv.writerow([self.n_frames, int(time.time() * 1000)])
            self.n_frames += 1

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        self.writer.release()
        self.csv_f.close()
        print(f"recorder: wrote {self.n_frames} frames to {self.out_mp4.name}")


# ---------------------------------------------------------------------
# Background PNG writer -- decouples disk I/O from the capture loop so
# the timeboxed capture stays at 30fps and respects its wall-clock
# budget (PNG encoding of a 1500x1300 frame is ~40ms, far too slow to
# do inline).
# ---------------------------------------------------------------------

class FrameWriter(threading.Thread):
    def __init__(self, maxsize: int = 512):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self.dropped = 0

    def submit(self, path: Path, frame: np.ndarray):
        try:
            self.q.put_nowait((path, frame))
        except queue.Full:
            self.dropped += 1

    def run(self):
        while not (self._stop.is_set() and self.q.empty()):
            try:
                path, frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            cv2.imwrite(str(path), frame)
            self.q.task_done()

    def stop(self):
        self._stop.set()
        self.join(timeout=30)
        if self.dropped:
            print(f"framewriter: dropped {self.dropped} frames (queue full)")


# Module-level writer, created in main().
FRAME_WRITER: "FrameWriter | None" = None


# ---------------------------------------------------------------------
# Yellow-box name OCR (the highlighted hero in the gallery)
# ---------------------------------------------------------------------

def find_highlighted_hero(img: np.ndarray):
    """Find the yellow-outlined hero tile + OCR its name plate.
    Returns (bbox_yellow, bbox_name_strip, name) or (None, None, '')."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = (((hsv[..., 0] >= 15) & (hsv[..., 0] <= 35))
              & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)
              ).astype(np.uint8) * 255
    nl, _, stats, _ = cv2.connectedComponentsWithStats(yellow, connectivity=8)
    best = None
    best_a = 0
    H, W = img.shape[:2]
    for i in range(1, nl):
        x, y, w, h, a = (int(stats[i, k]) for k in range(5))
        if a < 800:
            continue
        # Only consider tiles in the hero-strip vertical band.
        if y < H * 0.55 or y > H * 0.95:
            continue
        # Roughly square outline (the yellow ring around a tile)
        if not (0.5 < w / max(h, 1) < 1.6):
            continue
        if a > best_a:
            best_a = a
            best = (x, y, w, h)
    if best is None:
        return None, None, ""
    x, y, w, h = best
    # White name strip inside the lower portion of the yellow box.
    region = img[y + int(h * 0.60):y + h, x:x + w]
    hsv2 = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    white = ((hsv2[..., 2] > 180) & (hsv2[..., 1] < 60)
             ).astype(np.uint8) * 255
    nl2, _, st2, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    sb = None; sb_a = 0
    for i in range(1, nl2):
        ww = int(st2[i, 2])
        if int(st2[i, 4]) > sb_a and ww > w * 0.4:
            sb_a = int(st2[i, 4])
            sb = (int(st2[i, 0]) + x,
                  int(st2[i, 1]) + y + int(h * 0.60),
                  ww, int(st2[i, 3]))
    if sb is None:
        return best, None, ""
    sx, sy, sw, sh = sb
    crop = img[sy + 2:sy + sh - 6, sx + 4:sx + sw - 4]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return best, sb, ""
    big = cv2.resize(crop, (crop.shape[1] * 6, crop.shape[0] * 6),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.copyMakeBorder(bw, 40, 40, 40, 40,
                            cv2.BORDER_CONSTANT, value=255)
    cfg = ("--psm 7 --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ.")
    text = pytesseract.image_to_string(bw, config=cfg).strip()
    return best, sb, text


def _normalize_name(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalpha())


import difflib

# Current OW2 roster as of June 2026 (51 heroes). Names normalized to
# alnum-only uppercase to match _normalize_name(OCR).
HERO_ROSTER = [
    # Tank (14)
    "DVA", "DOMINA", "DOOMFIST", "HAZARD", "JUNKERQUEEN", "MAUGA",
    "ORISA", "RAMATTRA", "REINHARDT", "ROADHOG", "SIGMA", "WINSTON",
    "WRECKINGBALL", "ZARYA",
    # Damage (23)
    "ANRAN", "ASHE", "BASTION", "CASSIDY", "ECHO", "EMRE", "FREJA",
    "GENJI", "HANZO", "JUNKRAT", "MEI", "PHARAH", "REAPER", "SIERRA",
    "SOJOURN", "SOLDIER76", "SOMBRA", "SYMMETRA", "TORBJORN", "TRACER",
    "VENDETTA", "VENTURE", "WIDOWMAKER",
    # Support (14)
    "ANA", "BAPTISTE", "BRIGITTE", "ILLARI", "JETPACKCAT", "JUNO",
    "KIRIKO", "LIFEWEAVER", "LUCIO", "MERCY", "MIZUKI", "MOIRA",
    "WUYANG", "ZENYATTA",
]


def match_hero(ocr_name: str) -> str:
    """Fuzzy-match a garbled/truncated OCR hero name to the roster
    (e.g. RORDHOG -> ROADHOG, DOOMEIST -> DOOMFIST). Falls back to the
    cleaned OCR string if nothing scores well enough."""
    q = _normalize_name(ocr_name)
    if not q:
        return ""
    best, best_score = None, 0.0
    for h in HERO_ROSTER:
        if q == h:
            return h
        if q in h or h in q:
            score = min(len(q), len(h)) / max(len(q), len(h))
        else:
            score = difflib.SequenceMatcher(None, q, h).ratio()
        if score > best_score:
            best_score, best = score, h
    return best if (best and best_score >= 0.6) else q


def _safe_filename(s: str, fallback: str = "unknown") -> str:
    """Coerce an OCR'd name into a safe filesystem component."""
    out = []
    for ch in s.upper():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch in " .":
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or fallback


# Sub-screen header ("SKINS 20/26" / "EMOTES 8/14" / "HIGHLIGHT
# INTROS 4/10") sits just under the big hero name in the left panel.
SUBSCREEN_HDR_REGION = (70, 238, 360, 47)   # x, y, w, h on 2560x1440
SUBSCREENS = ["SKINS", "EMOTES", "HIGHLIGHTINTROS", "VICTORYPOSES",
              "WEAPONCHARMS", "SOUVENIRS", "VOICELINES", "SPRAYS",
              "WEAPONVARIANTS", "NAMECARDS"]


def detect_subscreen(img: np.ndarray) -> str:
    """OCR the header under the hero name and classify which sub-screen
    we're on. Returns one of SUBSCREENS or '' if unrecognised."""
    x, y, w, h = SUBSCREEN_HDR_REGION
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).sum() > bw.size * 0.5:
        bw = 255 - bw
    bw = cv2.copyMakeBorder(bw, 30, 30, 30, 30,
                            cv2.BORDER_CONSTANT, value=255)
    cfg = ("--psm 7 --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ ")
    raw = pytesseract.image_to_string(bw, config=cfg).strip()
    alpha = "".join(ch for ch in raw.upper() if ch.isalpha())
    for s in SUBSCREENS:
        if alpha.startswith(s[:5]) or s in alpha or alpha in s:
            return s
    return ""


# Region of the left-panel menu tree (SKINS / HIGHLIGHT INTROS / EMOTES
# / ...). The item set & vertical positions vary per hero, so we locate
# rows by OCR rather than hardcoded y.
MENU_TREE_REGION = (120, 400, 420, 760)   # x, y, w, h on 2560x1440


def find_menu_item_xy(img: np.ndarray, target: str):
    """Locate the menu-tree row whose label contains `target` (e.g.
    'EMOTES', 'SKINS') and return a click point (x, y) centered on that
    row, or None if not found."""
    rx, ry, rw, rh = MENU_TREE_REGION
    crop = img[ry:ry + rh, rx:rx + rw]
    big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).sum() > bw.size * 0.5:
        bw = 255 - bw
    data = pytesseract.image_to_data(bw, config="--psm 6",
                                     output_type=pytesseract.Output.DICT)
    target = target.upper()
    for i, txt in enumerate(data["text"]):
        word = "".join(ch for ch in txt.upper() if ch.isalpha())
        if not word:
            continue
        if word == target or (len(word) >= 4 and word in target):
            # back to original coords (we upscaled 2x)
            cy = ry + (data["top"][i] + data["height"][i] // 2) // 2
            return (rx + rw // 2, cy)
    return None


def open_menu_item(sess: "CaptureSession", key: str, target: str,
                   expect: str, tries: int = 4) -> bool:
    """Find the menu row labelled `target` by OCR, click it, and verify
    via the sub-screen header that we landed on `expect`. Retries by
    ESCing back to the menu tree. Returns True on success."""
    for attempt in range(tries):
        img = grab()
        pt = find_menu_item_xy(img, target)
        if pt is None:
            # Fall back to the (approximate) hardcoded position.
            pt = (sess.menu[key]["cx"], sess.menu[key]["cy"])
            print(f"  open_menu_item({key}): OCR couldn't find {target!r}, "
                  f"using fallback {pt}")
        click(*pt)
        time.sleep(0.6)
        # Clicking the SKINS/EMOTES row can only open THAT sub-list, so
        # the presence of the cyan selected-row box is sufficient proof
        # we got there -- the header OCR is too flaky to gate on. If no
        # cyan row, the click didn't register; just retry it (no ESC,
        # which would dump us out of the hero detail entirely).
        if gallery_ui.find_selected_row(grab()) is not None:
            return True
        print(f"  open_menu_item({key}): not in a sub-list yet "
              f"(try {attempt + 1}/{tries}), re-clicking {target}")
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------

@dataclass
class CalibState:
    steps_per_360: int | None = None       # learned from first rotation
    first_frame_at_step_0: np.ndarray | None = None


@dataclass
class CaptureSession:
    bboxes: list[dict]
    ui: dict
    calib: CalibState = field(default_factory=CalibState)

    @property
    def menu(self):
        return self.ui["menu"]

    @property
    def char_area(self):
        ca = self.ui["char_area"]
        return ca["x"], ca["y"], ca["w"], ca["h"]


# ---------------------------------------------------------------------
# Input primitives
# ---------------------------------------------------------------------

def click(x: int, y: int):
    pydirectinput.moveTo(x, y)
    time.sleep(0.05)
    pydirectinput.click()
    time.sleep(CLICK_PAUSE)


def press(key: str, times: int = 1, delay: float = 0.1):
    for _ in range(times):
        pydirectinput.press(key)
        time.sleep(delay)


def drag_step(dx: int | None = None):
    """Rotate the character with a held-button drag composed of many
    small cursor saccades. The OW model viewer reads rotation from the
    STREAM of mouse-move events during the hold -- a single big move
    barely registers, so we emit ~DRAG_SACCADES incremental moves while
    the button is down, like a real human drag."""
    fx, fy = DRAG_FROM
    if dx is None:
        dx = DRAG_DELTA[0]
    dy = DRAG_DELTA[1]
    n = max(1, DRAG_SACCADES)
    pyautogui.moveTo(fx, fy)
    pyautogui.mouseDown()
    try:
        acc_x = acc_y = 0
        for i in range(1, n + 1):
            tx = round(dx * i / n)
            ty = round(dy * i / n)
            step_x = tx - acc_x
            step_y = ty - acc_y
            acc_x, acc_y = tx, ty
            pyautogui.move(step_x, step_y)
            time.sleep(DRAG_SACCADE_DELAY)
    finally:
        pyautogui.mouseUp()


def rotate_capture_timeboxed(out_dir: Path, seconds: float,
                             sess: "CaptureSession",
                             drags: int = DRAGS_PER_WINDOW):
    """For exactly `seconds` of WALL-CLOCK time, spin the model with a
    series of small drags spread evenly across the window while grabbing
    char-area frames at ~CAPTURE_FPS. Disk writes go to the background
    FrameWriter so they don't blow the time budget. Frame filenames are
    the millisecond timestamp, so they align to the full-screen video's
    frames.csv sidecar.

    The continuous Recorder already saves the full screen; these are the
    tight per-phase image frames (the primary dataset)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = sess.char_area
    t0 = time.time()
    t_end = t0 + seconds
    frame_period = 1.0 / CAPTURE_FPS
    drag_interval = seconds / max(1, drags)   # seconds between flicks
    next_frame = t0
    next_drag = t0          # drag immediately at the start too
    while True:
        now = time.time()
        if now >= t_end:
            break
        # Time-driven drag: fire whenever we cross a drag boundary.
        if now >= next_drag:
            drag_step()
            next_drag += drag_interval
            now = time.time()
        # Pace frame grabs to CAPTURE_FPS.
        if now >= next_frame:
            f = grab_region(x, y, w, h)
            ts = int(time.time() * 1000)
            if FRAME_WRITER is not None:
                FRAME_WRITER.submit(out_dir / f"{ts}.png", f)
            else:
                cv2.imwrite(str(out_dir / f"{ts}.png"), f)
            next_frame += frame_period
            # If we've fallen behind, don't spiral -- resync.
            if next_frame < now:
                next_frame = now + frame_period
        else:
            time.sleep(min(0.002, max(0, next_frame - now)))


# ---------------------------------------------------------------------
# Rotation / capture
# ---------------------------------------------------------------------

def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel diff, normalised to 0..1."""
    if a.shape != b.shape:
        return 1.0
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() / 255.0)


def rotate_and_capture(sess: CaptureSession, out_dir: Path,
                       save_every: int = 1) -> int:
    """Drag-rotate until the frame matches the starting frame again
    (one full 360). Captures a frame after each drag.

    Returns the number of steps actually used; updates sess.calib on
    the first call so subsequent rotations can validate against it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = sess.char_area
    frame0 = grab_region(x, y, w, h)
    cv2.imwrite(str(out_dir / "0000.png"), frame0)

    learned = sess.calib.steps_per_360 is not None
    cap_steps = (sess.calib.steps_per_360 + 4) if learned else MAX_ROTATION_STEPS

    frames = [frame0]
    for step in range(1, cap_steps + 1):
        drag_step()
        f = grab_region(x, y, w, h)
        if step % save_every == 0:
            cv2.imwrite(str(out_dir / f"{step:04d}.png"), f)
        frames.append(f)
        d = frame_diff(f, frame0)
        if not learned:
            print(f"    step {step:2d}  diff={d:.4f}")
        # If we know the expected step count, only check there.
        if learned:
            if step >= sess.calib.steps_per_360 - 2 and d < ROTATION_DIFF_THRESHOLD:
                if step < sess.calib.steps_per_360 - 1 or step > sess.calib.steps_per_360 + 1:
                    print(f"  ! rotation anomaly: closed at step {step}, "
                          f"expected {sess.calib.steps_per_360}")
                return step
        else:
            # First time: need at least a few steps before matching.
            if step >= MAX_ROTATION_STEPS // 4 and d < ROTATION_DIFF_THRESHOLD:
                sess.calib.steps_per_360 = step
                print(f"  calibrated steps/360 = {step}  (diff={d:.4f})")
                return step

    if not learned:
        print(f"  rotation did NOT close within {cap_steps} steps; "
              f"defaulting calibration to {cap_steps}")
        sess.calib.steps_per_360 = cap_steps
    return cap_steps


# ---------------------------------------------------------------------
# Emote period detection
# ---------------------------------------------------------------------

def detect_emote_period(sess: CaptureSession,
                        sample_sec: float = EMOTE_SAMPLE_SEC):
    """Capture sample_sec of frames and find the emote loop length.

    Skips an initial warm-up so the diff baseline is taken AFTER the
    emote has visibly started. The period is the smallest lag >=
    EMOTE_PERIOD_MIN_FRAMES at which `frames[base + lag]` re-matches
    `frames[base]` AND `frames[base + 2*lag]` also matches -- requiring
    two repeats rules out false matches from a frame that just happens
    to look similar."""
    x, y, w, h = sess.char_area
    n_frames = int(sample_sec * EMOTE_SAMPLE_FPS)
    frames = []
    t0 = time.time()
    for i in range(n_frames):
        target_t = t0 + i / EMOTE_SAMPLE_FPS
        sleep = target_t - time.time()
        if sleep > 0:
            time.sleep(sleep)
        frames.append(grab_region(x, y, w, h))
    small = [cv2.resize(f, (160, 90)) for f in frames]
    base_idx = min(EMOTE_WARMUP_FRAMES, len(small) // 4)
    base = small[base_idx]
    diffs = np.array([frame_diff(base, f) for f in small])
    period = None
    best_score = 1.0
    # Look for the lag with the lowest summed diff at base+k AND
    # base+2k (cycle confirms on a second repeat).
    for k in range(EMOTE_PERIOD_MIN_FRAMES, EMOTE_PERIOD_MAX_FRAMES):
        if base_idx + 2 * k >= len(diffs):
            break
        s = diffs[base_idx + k] + diffs[base_idx + 2 * k]
        if s < best_score:
            best_score = s
            period = k
    if period is None:
        period = int(np.argmin(diffs[base_idx + EMOTE_PERIOD_MIN_FRAMES:])) \
                 + EMOTE_PERIOD_MIN_FRAMES
    return period, frames


# ---------------------------------------------------------------------
# Skin / emote loops
# ---------------------------------------------------------------------

def equip_if_visible(img: np.ndarray | None = None) -> bool:
    """If the orange EQUIP button is on screen, click it (so the
    selection becomes the equipped skin and the orange overlay clears).
    If no EQUIP button is visible -- either because the item is already
    equipped or only an UNLOCK button is shown -- return False without
    clicking anything."""
    if img is None:
        img = grab()
    btn, btn_text = gallery_ui.find_action_button(img)
    if btn is None:
        return False
    if "EQUIP" not in btn_text.upper():
        # Could be UNLOCK; skip.
        return False
    bx, by, bw, bh = btn
    click(bx + bw // 2, by + bh // 2)
    time.sleep(0.6)
    return True


def scroll_to_top_skin(max_presses: int = 40):
    """Press UP until the selected row's OCR == RANDOM FROM FAVORITES
    (the topmost item in the list), then press DOWN once to land on
    the first real skin."""
    last_name = None
    same_count = 0
    for _ in range(max_presses):
        img = grab()
        _, name = gallery_ui.ocr_selected_row(img)
        n = name.upper().replace(" ", "")
        # OCR sometimes drops the leading "RA" because of the icon CC
        # filter (e.g. "INDOMFROMFAVORITES"). Match the unique suffix
        # FAVORITES instead.
        if "FAVORITES" in n:
            print(f"  reached top (RANDOM FROM FAVORITES), stepping down once")
            press("down")
            time.sleep(0.2)
            img = grab()
            _, first = gallery_ui.ocr_selected_row(img)
            print(f"  first skin: {first!r}")
            return True
        if n == last_name:
            same_count += 1
            if same_count >= 2:
                print(f"  could not scroll above {name!r}")
                return False
        else:
            same_count = 0
            last_name = n
        press("up")
        time.sleep(0.15)
    return False


def navigate_skins(sess: CaptureSession, hero_name: str):
    """Click SKINS, land on the default OVERWATCH skin, equip it, spin
    for SKIN_SECONDS capturing the standing pose, then hand off to the
    emote sweep. Only the default skin is collected."""
    if not open_menu_item(sess, "skins", "SKINS", "SKINS"):
        print("  could not open SKINS, skipping hero")
        return
    scroll_to_top_skin()   # lands on the first real skin (OVERWATCH default)

    img = grab()
    _, name = gallery_ui.ocr_selected_row(img)
    if not name:
        print("  skin: no selected row detected, skipping hero")
        return
    print(f"  default skin: {name!r}")
    equip_if_visible(img)

    out_dir = (CAPTURE_ROOT / _safe_filename(hero_name)
               / _safe_filename(name) / "standing")
    rotate_capture_timeboxed(out_dir, SKIN_SECONDS, sess)

    navigate_emotes(sess, hero_name, name)


def navigate_emotes(sess: CaptureSession, hero_name: str, skin_name: str):
    """ESC up to the menu tree, click EMOTES, then for every emote spin
    for EMOTE_SECONDS while it plays, capturing frames."""
    press("escape")
    time.sleep(0.4)
    if not open_menu_item(sess, "emotes", "EMOTES", "EMOTES"):
        print("  could not open EMOTES, skipping emote sweep")
        press("escape")
        time.sleep(0.4)
        return
    seen = set()
    empty_streak = 0
    for i in range(40):
        img = grab()
        bbox, name = gallery_ui.ocr_selected_row(img)
        if not name:
            # Some emotes have non-alphabetic names (e.g. ^O^) that OCR
            # returns as empty. Skip the row instead of breaking the
            # loop. After two consecutive empties assume we've left the
            # emote sublist.
            empty_streak += 1
            if empty_streak >= 2:
                break
            press("down")
            continue
        empty_streak = 0
        if "FAVORITES" in name.upper() or name.upper().startswith("RANDOM"):
            press("down"); continue
        if name in seen:
            break
        seen.add(name)

        # Don't equip emotes -- the EQUIP button opens a slot picker we'd
        # get stuck in. The gallery preview plays the emote in-place.
        print(f"  emote: {name!r}")
        emote_dir = (CAPTURE_ROOT / _safe_filename(hero_name)
                     / _safe_filename(skin_name)
                     / f"emote__{_safe_filename(name)}")
        rotate_capture_timeboxed(emote_dir, EMOTE_SECONDS, sess)

        press("down")
        time.sleep(0.25)
    # Back to the menu tree.
    press("escape")
    time.sleep(0.4)


# ---------------------------------------------------------------------
# Hero loop + start signal
# ---------------------------------------------------------------------

def wait_for_start_signal(target_name: str = START_HERO_NAME,
                          poll_interval: float = 0.5):
    print(f"waiting for hero highlight = {target_name!r}...")
    target_n = _normalize_name(target_name)
    while True:
        img = grab()
        _, _, name = find_highlighted_hero(img)
        if name:
            if _normalize_name(name) == target_n:
                print(f"start signal: {name!r}")
                return
        time.sleep(poll_interval)


def count_portrait_tiles(img: np.ndarray) -> int:
    """Count hero-portrait tiles in the bottom strip via their white
    bracket borders. This is the reliable 'are we at the gallery'
    signal -- the strip of ~40+ tiles only exists on the gallery
    screen, independent of which hero is highlighted or mid-animation
    yellow boxes."""
    H, W = img.shape[:2]
    strip = img[int(H * 0.66):int(H * 0.92)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 200) & (hsv[..., 1] < 40)).astype(np.uint8) * 255
    nl, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    hs = strip.shape[0]
    tile_w = int(round(0.20 * hs))
    w_lo, w_hi = int(tile_w * 0.6), int(tile_w * 1.15)
    n = 0
    for i in range(1, nl):
        w = int(stats[i, 2]); h = int(stats[i, 3])
        # A tile border arc: tile-width wide, not too tall.
        if w_lo <= w <= w_hi and h <= int(tile_w * 0.95):
            n += 1
    return n


def is_at_gallery(img: np.ndarray, min_tiles: int = 12) -> bool:
    return count_portrait_tiles(img) >= min_tiles


def wait_for_stable_screen(timeout: float = 3.0, diff_thresh: float = 0.015):
    """Grab frames until two consecutive ones barely differ (the screen
    has finished loading/animating). Returns the last frame."""
    t_end = time.time() + timeout
    prev = grab()
    while time.time() < t_end:
        time.sleep(0.2)
        cur = grab()
        if frame_diff(cv2.resize(prev, (320, 180)),
                      cv2.resize(cur, (320, 180))) < diff_thresh:
            return cur
        prev = cur
    return prev


def ensure_at_hero_gallery(max_escapes: int = 6) -> bool:
    """Get back to the hero gallery. Each attempt: wait for the screen
    to settle, then confirm we're at the gallery by detecting the
    portrait-tile strip (NOT the transient yellow highlight). Escalate
    ESC -> HEROES nav tab until the strip appears."""
    for attempt in range(max_escapes + 1):
        img = wait_for_stable_screen()
        if is_at_gallery(img):
            return True
        # First couple of tries use ESC; after that, click HEROES tab.
        if attempt < 2:
            press("escape")
        else:
            click(*HEROES_TAB)
        time.sleep(0.4)
    # Final settle + check.
    return is_at_gallery(wait_for_stable_screen())


def wait_until(predicate, timeout: float = 4.0, interval: float = 0.15,
               settle: int = 2):
    """Poll `predicate(img)` on fresh screenshots until it returns a
    truthy value on `settle` consecutive grabs (so we don't act on a
    mid-transition frame), or until timeout. Returns the last truthy
    value, or None on timeout."""
    t_end = time.time() + timeout
    hits = 0
    last = None
    while time.time() < t_end:
        val = predicate(grab())
        if val:
            hits += 1
            last = val
            if hits >= settle:
                return last
        else:
            hits = 0
        time.sleep(interval)
    return None


def _enter_hero_and_capture(sess: CaptureSession, enter_xy, hero_name_n: str):
    """Click `enter_xy` to open the hero detail, confirm we actually
    left the gallery (the portrait strip is gone -- NOT the hero name,
    which OCRs empty for some heroes), then run the skin+emote sweep."""
    entered = False
    for _ in range(3):
        click(*enter_xy)
        if wait_until(lambda im: not is_at_gallery(im),
                      timeout=3.0, settle=2):
            entered = True
            break
    if not entered:
        print(f"  could not enter hero detail for {hero_name_n!r}, skipping")
        return
    time.sleep(0.4)   # let the loadout menu finish animating in
    navigate_skins(sess, hero_name_n)
    press("escape", times=2, delay=0.4)


def process_hero(sess: CaptureSession, idx: int, skip_existing: bool = False,
                 name_override: str | None = None):
    if not ensure_at_hero_gallery():
        print(f"[{idx:02d}] not at hero gallery, skipping")
        return
    b = sess.bboxes[idx]
    cx, cy = b["x"] + b["w"] // 2, b["y"] + b["h"] // 2

    # If desired hero is NOT already highlighted, click to highlight.
    img = grab()
    yb, _, current = find_highlighted_hero(img)
    same_position = False
    if yb is not None:
        yx, yy, yw, yh = yb
        same_position = (abs((yx + yw // 2) - cx) < b["w"] * 0.6 and
                         abs((yy + yh // 2) - cy) < b["h"] * 0.6)
    if not same_position:
        click(cx, cy)

    # Wait for the highlighted-hero name plate to settle before reading
    # it (the gallery animates the yellow box / name in).
    hero_name = wait_until(lambda im: find_highlighted_hero(im)[2] or None,
                           timeout=4.0)
    hero_name = hero_name or ""
    hero_name_n = name_override.upper() if name_override else (
        match_hero(hero_name) or f"hero{idx}")

    # In --missing mode, skip heroes that already have a directory.
    if skip_existing and (CAPTURE_ROOT / hero_name_n).is_dir():
        print(f"[{idx:02d}] {hero_name_n!r} already captured, skipping")
        return

    print(f"[{idx:02d}] hero = {hero_name_n!r}  (ocr={hero_name!r})")
    _enter_hero_and_capture(sess, (cx, cy), hero_name_n)


def process_current_hero(sess: CaptureSession, name_override: str | None):
    """Re-record whichever hero is CURRENTLY highlighted in the gallery
    (the yellow-boxed tile). Use --name to force the directory name when
    the hero's name plate OCRs poorly."""
    if not ensure_at_hero_gallery():
        print("not at hero gallery, aborting")
        return
    yb = wait_until(lambda im: find_highlighted_hero(im)[0], timeout=4.0)
    if yb is None:
        print("no highlighted hero found")
        return
    yx, yy, yw, yh = yb
    ycx, ycy = yx + yw // 2, yy + yh // 2
    # The yellow selection box includes the name plate, so its center
    # sits low (on the plate, not the tile). Snap to the nearest known
    # tile bbox center -- the verified click target that actually enters.
    best = min(sess.bboxes,
               key=lambda b: (b["x"] + b["w"] // 2 - ycx) ** 2
                             + (b["y"] + b["h"] // 2 - ycy) ** 2)
    enter_xy = (best["x"] + best["w"] // 2, best["y"] + best["h"] // 2)
    _, _, ocr = find_highlighted_hero(grab())
    hero_name_n = (name_override.upper() if name_override
                   else (match_hero(ocr) or "hero_current"))
    print(f"current hero = {hero_name_n!r}  (ocr={ocr!r})  enter={enter_xy}")
    _enter_hero_and_capture(sess, enter_xy, hero_name_n)


def main():
    global CAPTURE_ROOT, FRAME_WRITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--bboxes", default=DEFAULT_BBOXES)
    ap.add_argument("--ui", default=DEFAULT_UI)
    ap.add_argument("--start-from", type=int, default=0,
                    help="hero bbox index to start with (default 0)")
    ap.add_argument("--no-wait", action="store_true",
                    help="skip the WUYANG start signal poll")
    ap.add_argument("--current", action="store_true",
                    help="re-record ONLY the currently-highlighted hero, "
                         "then exit (for fixing skipped/undersampled heroes)")
    ap.add_argument("--missing", action="store_true",
                    help="walk every tile but only record heroes that "
                         "don't yet have a directory under --capture-root")
    ap.add_argument("--box", type=int, default=None,
                    help="record EXACTLY this tile index (see "
                         "gallery_bboxes_numbered.png), then exit")
    ap.add_argument("--name", default=None,
                    help="force the hero directory name (use with "
                         "--current when the name plate OCRs poorly)")
    ap.add_argument("--capture-root", default=None,
                    help=f"root directory for captured frames "
                         f"(default: {CAPTURE_ROOT})")
    args = ap.parse_args()
    if args.capture_root:
        CAPTURE_ROOT = Path(args.capture_root).resolve()
    CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"capture root: {CAPTURE_ROOT}")

    bboxes = json.loads(Path(args.bboxes).read_text())["boxes"]
    ui = json.loads(Path(args.ui).read_text())
    sess = CaptureSession(bboxes=bboxes, ui=ui)

    # --current/--box always start immediately (no WUYANG wait).
    if not args.no_wait and not args.current and args.box is None:
        wait_for_start_signal()

    # Start the continuous full-screen 30fps recording + async PNG writer.
    if args.box is not None:
        tag = args.name or f"box{args.box:02d}"
    elif args.current:
        tag = args.name or "current"
    elif args.missing:
        tag = "missing"
    else:
        tag = f"from{args.start_from:02d}"
    vid_path = CAPTURE_ROOT / f"session_{tag}.mp4"
    recorder = Recorder(vid_path, fps=CAPTURE_FPS)
    recorder.start()
    print(f"recording full screen -> {vid_path}")

    FRAME_WRITER = FrameWriter()
    FRAME_WRITER.start()

    try:
        if args.box is not None:
            process_hero(sess, args.box, name_override=args.name)
        elif args.current:
            process_current_hero(sess, args.name)
        else:
            for idx in range(args.start_from, len(bboxes)):
                try:
                    process_hero(sess, idx, skip_existing=args.missing)
                except KeyboardInterrupt:
                    print("interrupted")
                    return
                except Exception as e:
                    print(f"[{idx:02d}] FAILED: {e}")
                    # Try to escape back to hero select.
                    press("escape", times=3, delay=0.5)
    finally:
        recorder.stop()
        FRAME_WRITER.stop()


if __name__ == "__main__":
    main()

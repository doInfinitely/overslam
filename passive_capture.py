"""Passive capture bot.

Watches the screen while YOU drive the OW2 hero gallery. Continuously
takes screenshots, detects the current context (hero, equipped skin,
selected row), and saves tagged frames to:

    <root>/<HERO>/<SKIN>/<context>/<timestamp>.png

  - context == 'standing'  when the cyan-outlined row is the same as
                            the green-checked row (you're sitting on
                            the equipped skin)
  - context == '<emote-name>' when the panel shows an emote row
  - context == '<skin-name>'  when you've highlighted a different skin

PRINTSCREEN kills the bot.

Run on the Windows machine. Manually navigate to the hero you want,
into the skin/emote sublists, etc. The bot just observes and saves.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import gallery_ui

import pyautogui
import pytesseract

_TESS_CANDIDATES = [
    os.environ.get("PYTESSERACT_CMD"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for _c in _TESS_CANDIDATES:
    if _c and Path(_c).exists():
        pytesseract.pytesseract.tesseract_cmd = _c
        break

try:
    import keyboard  # type: ignore
    keyboard.add_hotkey("print screen", lambda: os._exit(130))
    print("(press PRINTSCREEN to kill the bot)")
except Exception as _e:
    print(f"(warning: 'keyboard' module not available: {_e})")

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

POLL_HZ        = 8       # screenshots/second for state detection + frame save
SAVE_EVERY     = 1       # save every Nth screenshot (1 = every grab)
CHAR_AREA      = (800, 100, 1500, 1300)   # render region we save
CAPTURE_ROOT   = Path(os.environ.get("CAPTURE_ROOT", "captures")).resolve()


# ---------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------

def grab() -> np.ndarray:
    pil = pyautogui.screenshot()
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def grab_region(x, y, w, h) -> np.ndarray:
    pil = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# Highlighted-hero (yellow box) + name OCR -- borrowed from capture_bot.
# ---------------------------------------------------------------------

def find_highlighted_hero(img: np.ndarray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = (((hsv[..., 0] >= 15) & (hsv[..., 0] <= 35))
              & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)
              ).astype(np.uint8) * 255
    nl, _, stats, _ = cv2.connectedComponentsWithStats(yellow, connectivity=8)
    best = None; best_a = 0
    H, W = img.shape[:2]
    for i in range(1, nl):
        x, y, w, h, a = (int(stats[i, k]) for k in range(5))
        if a < 800: continue
        if y < H * 0.55 or y > H * 0.95: continue
        if not (0.5 < w / max(h, 1) < 1.6): continue
        if a > best_a: best_a = a; best = (x, y, w, h)
    if best is None:
        return None, ""
    x, y, w, h = best
    region = img[y + int(h * 0.60):y + h, x:x + w]
    hsv2 = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    white = ((hsv2[..., 2] > 180) & (hsv2[..., 1] < 60)).astype(np.uint8) * 255
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
        return best, ""
    sx, sy, sw, sh = sb
    crop = img[sy + 2:sy + sh - 6, sx + 4:sx + sw - 4]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return best, ""
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
    return best, text


# ---------------------------------------------------------------------
# Equipped skin detection: green check icon in the left panel
# ---------------------------------------------------------------------

def find_equipped_row(img: np.ndarray, panel_frac: float = 0.30):
    """Find the row that has a GREEN CHECK icon (currently equipped item)
    in the left panel. Returns (bbox, OCR-name) of the row containing
    the check, or (None, '')."""
    H, W = img.shape[:2]
    panel_w = int(W * panel_frac)
    panel = img[:, :panel_w]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] >= 40) & (hsv[..., 0] <= 80)
             & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)
             ).astype(np.uint8) * 255
    nl, _, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    # Use the cyan-outlined row's X-range (if there is one on screen) so
    # the icon and text positions line up with what ocr_row_image was
    # tuned for. Fall back to a percentage-based default.
    sel = gallery_ui.find_selected_row(img, panel_frac)
    if sel is not None:
        sx, _, sw, _ = sel
        row_x0, row_x1 = sx + 4, sx + sw - 4
    else:
        row_x0 = int(panel_w * 0.227)
        row_x1 = panel_w - 4
    for i in range(1, nl):
        x, y, w, h, a = (int(stats[i, k]) for k in range(5))
        if not (15 <= w <= 50 and 15 <= h <= 50): continue
        if x < panel_w * 0.7: continue
        row_half = 28
        ry0 = max(0, y + h // 2 - row_half)
        ry1 = min(panel.shape[0], y + h // 2 + row_half)
        row = panel[ry0:ry1, row_x0:row_x1]
        name = gallery_ui.ocr_row_image(row)
        return (x, y, w, h), name
    return None, ""


# ---------------------------------------------------------------------
# Sub-screen header detection ("SKINS 20/26" / "EMOTES 8/14" / ...)
# ---------------------------------------------------------------------

# The header text sits just under the big hero name in the left panel.
SUBSCREEN_HDR_REGION = (70, 238, 360, 47)   # x, y, w, h on 2560x1440

# Canonical sub-screens (left-panel menu items). Map a noisy OCR token
# to one of these by substring.
SUBSCREENS = ["SKINS", "EMOTES", "HIGHLIGHTINTROS", "VICTORYPOSES",
              "WEAPONCHARMS", "SOUVENIRS", "VOICELINES", "SPRAYS",
              "WEAPONVARIANTS", "NAMECARDS"]


def detect_subscreen(img: np.ndarray) -> str:
    """OCR the header under the hero name and classify the sub-screen.
    Returns one of SUBSCREENS or '' if not recognised (e.g. the menu
    tree / hero gallery)."""
    x, y, w, h = SUBSCREEN_HDR_REGION
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).sum() > bw.size * 0.5:
        bw = 255 - bw
    bw = cv2.copyMakeBorder(bw, 30, 30, 30, 30,
                            cv2.BORDER_CONSTANT, value=255)
    cfg = ("--psm 7 --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ ")
    raw = pytesseract.image_to_string(bw, config=cfg).strip()
    tok = _norm(raw)   # drop digits-less? keep alnum, strip the count
    # Strip the trailing "20/26" count -- keep leading alphabetic run.
    alpha = "".join(ch for ch in tok if ch.isalpha())
    for s in SUBSCREENS:
        if alpha.startswith(s[:5]) or s in alpha or alpha in s:
            return s
    return ""


# ---------------------------------------------------------------------
# Hero roster + fuzzy matching
# ---------------------------------------------------------------------

import difflib

# Current OW2 roster (display names, uppercased, alnum-only for match).
# Current OW2 roster as of June 2026 (51 heroes). Names normalized to
# alnum-only uppercase to match _norm(OCR).
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
    """Fuzzy-match a (possibly truncated/garbled) OCR hero name to the
    canonical roster. Returns the canonical name, or the cleaned OCR
    string if no good match is found."""
    q = _norm(ocr_name)
    if not q:
        return ""
    # Exact / substring first (handles dropped leading letters like
    # 'ERWATCH' or 'DOOMEIST').
    best = None
    best_score = 0.0
    for h in HERO_ROSTER:
        if q == h:
            return h
        # Substring either way scores high.
        if q in h or h in q:
            score = min(len(q), len(h)) / max(len(q), len(h))
        else:
            score = difflib.SequenceMatcher(None, q, h).ratio()
        if score > best_score:
            best_score = score
            best = h
    if best is not None and best_score >= 0.6:
        return best
    return q


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe(s: str, fallback: str = "unknown") -> str:
    out = []
    for ch in s.upper():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch in " .":
            out.append("_")
    return "".join(out).strip("_") or fallback


def _norm(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

@dataclass
class State:
    hero: str = ""             # canonical hero (sticky)
    equipped_skin: str = ""    # last skin we saw an EQUIP-fade event for
    selected_row: str = ""     # currently cyan-outlined row
    subscreen: str = ""        # SKINS / EMOTES / ... from the header
    equip_visible: bool = False  # orange EQUIP button on screen now
    in_gallery: bool = False   # currently looking at hero gallery

    def capture_path(self) -> tuple[str, str] | None:
        """Return (skin_subdir, pose_subdir) for the current frame, or
        None if we shouldn't be saving right now.

        SKINS screen:
          - selected == equipped  -> ('<equipped>', 'standing')
          - selected != equipped  -> ('<selected>', 'preview')
        EMOTES screen:
          - ('<equipped>', 'emote__<selected>')
        Other sub-screens: not captured (return None)."""
        if self.in_gallery or not self.hero:
            return None
        sel = _safe(self.selected_row, "")
        eq = _safe(self.equipped_skin, "")
        sel_n = _norm(self.selected_row)
        eq_n = _norm(self.equipped_skin)
        if self.subscreen == "SKINS":
            if not eq:
                return None
            if sel_n and eq_n and (sel_n in eq_n or eq_n in sel_n):
                return (eq, "standing")
            if sel:
                return (sel, "preview")
            return None
        if self.subscreen == "EMOTES":
            if not eq or not sel:
                return None
            return (eq, f"emote__{sel}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", default=None)
    ap.add_argument("--quiet", action="store_true",
                    help="don't print every state change")
    args = ap.parse_args()
    root = Path(args.capture_root).resolve() if args.capture_root else CAPTURE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    print(f"capture root: {root}")

    log_path = root / "events.csv"
    log_f = log_path.open("a", newline="")
    log = csv.writer(log_f)
    if log_path.stat().st_size == 0:
        log.writerow(["time", "hero", "equipped_skin", "selected_row",
                      "in_gallery", "context"])

    state = State()
    prev_snapshot = None
    prev_selected_row = ""
    prev_equip_visible = False
    frame_idx = 0
    next_t = time.time()
    period = 1.0 / POLL_HZ

    while True:
        now = time.time()
        if now < next_t:
            time.sleep(next_t - now)
        next_t = now + period

        img = grab()

        # ----- state detection -----
        yb, hero_name = find_highlighted_hero(img)
        if hero_name:
            state.in_gallery = True
            new_hero = match_hero(hero_name)
            if new_hero != state.hero and new_hero:
                t = time.strftime("%H:%M:%S")
                print(f"[{t}] HERO {state.hero!r} -> {new_hero!r} "
                      f"(ocr={hero_name!r})")
                log.writerow([t, "HERO_CHANGE", new_hero, "", "", ""])
                log_f.flush()
                # New hero -- forget previous skin lock + subscreen.
                state.equipped_skin = ""
                state.subscreen = ""
            state.hero = new_hero
            state.selected_row = ""
            state.equip_visible = False
        else:
            state.in_gallery = False
            new_sub = detect_subscreen(img)
            if new_sub:
                if new_sub != state.subscreen:
                    t = time.strftime("%H:%M:%S")
                    print(f"[{t}] SUBSCREEN -> {new_sub}")
                    log.writerow([t, "SUBSCREEN", state.hero, "", new_sub, ""])
                    log_f.flush()
                state.subscreen = new_sub
            _, row_name = gallery_ui.ocr_selected_row(img)
            state.selected_row = row_name
            btn, btn_text = gallery_ui.find_action_button(img)
            equip_now = (btn is not None and "EQUIP" in btn_text.upper())

            # ----- click event: selected row changed -----
            if (_norm(row_name) and
                    _norm(row_name) != _norm(prev_selected_row)):
                t = time.strftime("%H:%M:%S")
                print(f"[{t}] [{state.subscreen or '?'}] CLICKED {row_name!r}")
                log.writerow([t, "CLICK", state.hero, state.subscreen,
                              row_name, ""])
                log_f.flush()

            # ----- equip event: orange EQUIP just faded AND we're
            # still on the same row -- means user actually equipped it
            # (not just scrolled to an already-equipped row). Only valid
            # on the SKINS screen. -----
            if (state.subscreen == "SKINS"
                    and prev_equip_visible and not equip_now
                    and _norm(row_name) == _norm(prev_selected_row)
                    and row_name):
                t = time.strftime("%H:%M:%S")
                print(f"[{t}] EQUIPPED {row_name!r}")
                log.writerow([t, "EQUIP", state.hero, row_name, "", ""])
                log_f.flush()
                state.equipped_skin = row_name

            state.equip_visible = equip_now
            prev_selected_row = row_name
            prev_equip_visible = equip_now

        # ----- frame capture -----
        cap = state.capture_path()
        if cap is not None:
            cx, cy, cw, ch = CHAR_AREA
            frame = grab_region(cx, cy, cw, ch)
            skin_sub, pose_sub = cap
            out_dir = root / _safe(state.hero) / skin_sub / pose_sub
            out_dir.mkdir(parents=True, exist_ok=True)
            if frame_idx % SAVE_EVERY == 0:
                ts = f"{int(time.time() * 1000)}"
                cv2.imwrite(str(out_dir / f"{ts}.png"), frame)
        frame_idx += 1

        # ----- coarse state log on any change -----
        ctx = "/".join(cap) if cap else "-"
        snapshot = (state.hero, state.equipped_skin, state.selected_row,
                    state.subscreen, state.in_gallery, ctx,
                    state.equip_visible)
        if snapshot != prev_snapshot:
            if not args.quiet:
                t = time.strftime("%H:%M:%S")
                print(f"[{t}] hero={state.hero!r:12s} "
                      f"sub={state.subscreen!r:8s} "
                      f"eq={state.equipped_skin!r:16s} "
                      f"sel={state.selected_row!r:16s} "
                      f"equip_btn={state.equip_visible}  -> {ctx}")
            prev_snapshot = snapshot


if __name__ == "__main__":
    main()

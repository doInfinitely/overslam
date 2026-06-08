"""Vision helpers for the OW2 hero-gallery flow.

Exposes:
  * STATIC_BBOXES         -- known fixed UI element positions (menu items,
                             panel layout) for 2560x1440 res
  * find_selected_row()   -- locate the cyan-outlined "currently
                             selected" row in the left panel
  * ocr_selected_row()    -- OCR the text of that row, filtering out
                             rarity icons and lock/check/star glyphs
  * ocr_strip_text()      -- generic helper used by selected-name.py too
                             (italic OW font with text-CC trimming)
"""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract


# Positions are absolute pixel coords in 2560x1440 screenshots.
STATIC_BBOXES = {
    "panel_width_frac": 0.30,
    # Hero-detail menu tree (screenshot 121329). Row centers, click here
    # to enter the corresponding sub-screen.
    "menu": {
        "skins":           {"cx": 380, "cy": 450},
        "highlight_intros":{"cx": 380, "cy": 525},
        "emotes":          {"cx": 380, "cy": 605},
        "victory_poses":   {"cx": 380, "cy": 685},
        "weapon_charms":   {"cx": 380, "cy": 765},
        "souvenirs":       {"cx": 380, "cy": 845},
    },
    # Approximate area used by the character render -- needed when
    # rotating (drag origin) and for image-diff during period detection.
    "char_area": {"x": 800, "y": 100, "w": 1500, "h": 1300},
}


def find_selected_row(img_bgr: np.ndarray, panel_frac: float = 0.30):
    """Return bbox (x,y,w,h) of the cyan-outlined selected row in the
    left panel, or None."""
    H, W = img_bgr.shape[:2]
    panel = img_bgr[:, :int(W * panel_frac)]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= 85) & (hsv[..., 0] <= 115)
            & (hsv[..., 1] > 200) & (hsv[..., 2] > 200)
            ).astype(np.uint8) * 255
    nl, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_a = 0
    for i in range(1, nl):
        a = int(stats[i, 4])
        w = int(stats[i, 2]); h = int(stats[i, 3])
        if w < 300 or h < 30 or w / max(h, 1) > 20:
            continue
        if a > best_a:
            best_a = a
            best = (int(stats[i, 0]), int(stats[i, 1]), w, h)
    return best


def ocr_row_image(row_bgr: np.ndarray) -> str:
    """OCR the text in one panel-row image, filtering out icon CCs.
    Used by both ocr_selected_row (cyan-outlined row) and any external
    caller that has already cropped a single row from the panel."""
    if row_bgr.size == 0:
        return ""
    big = cv2.resize(row_bgr, (row_bgr.shape[1] * 3, row_bgr.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).sum() > bw.size * 0.5:
        bw = 255 - bw
    Hb, Wb = bw.shape
    nl2, _, st2, _ = cv2.connectedComponentsWithStats(255 - bw, connectivity=8)
    # Collect letter-shaped CCs first, then use a GAP test to identify
    # and drop the leftmost CC if it's separated from the next CC by a
    # large gap (that's the icon; letters are closely packed).
    candidates = []
    for i in range(1, nl2):
        cx, cy, cw, ch, ca = st2[i]
        if not (35 <= ch <= 80):
            continue
        if cw > ch * 1.4:
            continue
        if cx + cw > Wb * 0.85:
            continue
        if ca < 25:
            continue
        candidates.append((int(cx), int(cy), int(cw), int(ch)))
    candidates.sort(key=lambda c: c[0])
    text_ccs = candidates
    # If the gap between the first two candidates is larger than the
    # median gap among the rest, the first is the category icon.
    if len(candidates) >= 3:
        gaps = []
        for i in range(len(candidates) - 1):
            gaps.append(candidates[i + 1][0] - (candidates[i][0] + candidates[i][2]))
        first_gap = gaps[0]
        rest_median = float(np.median(gaps[1:])) if len(gaps) > 1 else first_gap
        if first_gap > max(20, rest_median * 2.5):
            text_ccs = candidates[1:]
    if not text_ccs:
        return ""
    x0 = min(t[0] for t in text_ccs) - 8
    x1 = max(t[0] + t[2] for t in text_ccs) + 8
    y0 = min(t[1] for t in text_ccs) - 4
    y1 = max(t[1] + t[3] for t in text_ccs) + 4
    tight = bw[max(0, y0):min(Hb, y1), max(0, x0):min(Wb, x1)]
    tight = cv2.copyMakeBorder(tight, 30, 30, 30, 30,
                               cv2.BORDER_CONSTANT, value=255)
    cfg = ("--psm 7 --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ.- ")
    return pytesseract.image_to_string(tight, config=cfg).strip()


def ocr_selected_row(img_bgr: np.ndarray, panel_frac: float = 0.30):
    """Locate the cyan-outlined selected row, OCR its text."""
    bbox = find_selected_row(img_bgr, panel_frac)
    if bbox is None:
        return None, ""
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    panel = img_bgr[:, :int(W * panel_frac)]
    inner = panel[y + 4:y + h - 4, x + 4:x + w - 4]
    return bbox, ocr_row_image(inner)


def find_action_button(img_bgr: np.ndarray):
    """Return (bbox, text) of the orange center-bottom button -- 'EQUIP'
    when the highlighted item can be equipped, 'UNLOCK' when it must be
    purchased, None if no button is visible (already equipped)."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 22)
            & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)
            ).astype(np.uint8) * 255
    nl, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None; best_a = 0
    for i in range(1, nl):
        x, y, w, h, a = (int(stats[i, k]) for k in range(5))
        # Center-bottom button: wide rectangle near vertical bottom 30%
        # and roughly horizontally centered.
        if y < H * 0.7:
            continue
        if w < 150 or h < 30 or w / max(h, 1) < 2.5:
            continue
        cx = x + w // 2
        if abs(cx - W // 2) > W * 0.15:
            continue
        if a > best_a:
            best_a = a; best = (x, y, w, h)
    if best is None:
        return None, ""
    x, y, w, h = best
    crop = img_bgr[y:y + h, x:x + w]
    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).sum() > bw.size * 0.5:
        bw = 255 - bw
    bw = cv2.copyMakeBorder(bw, 40, 40, 40, 40,
                            cv2.BORDER_CONSTANT, value=255)
    cfg = ("--psm 7 --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = pytesseract.image_to_string(bw, config=cfg).strip()
    return best, text


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()
    img = cv2.imread(args.image)
    bbox, text = ocr_selected_row(img)
    print(f"selected row: bbox={bbox}  text={text!r}")
    btn, btn_t = find_action_button(img)
    print(f"action button: bbox={btn}  text={btn_t!r}")

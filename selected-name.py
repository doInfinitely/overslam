#!/usr/bin/env python
"""From a stored portrait bbox (loaded from gallery_bboxes.json), find
the yellow selection box that surrounds the bbox center, locate the
white name strip inside it at the bottom, and OCR the strip with
tesseract.

Usage:
    ./selected-name.py <screenshot.png> --idx 0
    ./selected-name.py <screenshot.png> --idx 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytesseract


def yellow_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask the yellow selection-highlight outline (warm yellow/orange,
    high saturation and value)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]; s = hsv[..., 1]; v = hsv[..., 2]
    return (((h >= 15) & (h <= 35)) & (s > 120) & (v > 150)
            ).astype(np.uint8) * 255


def find_yellow_box(img: np.ndarray, cx: int, cy: int,
                    search_radius: int = 80) -> tuple[int, int, int, int]:
    """Find the bounding box of the yellow CC that contains or is
    nearest to (cx, cy). search_radius gates which CCs we consider."""
    mask = yellow_mask(img)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_d = None
    for i in range(1, nl):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        # CC must enclose the reference center, or be very close to it.
        inside = x <= cx <= x + w and y <= cy <= y + h
        # distance from CC center to (cx, cy)
        dx = (x + w // 2) - cx
        dy = (y + h // 2) - cy
        d = (dx * dx + dy * dy) ** 0.5
        if not inside and d > search_radius:
            continue
        if best is None or d < best_d:
            best = (int(x), int(y), int(w), int(h))
            best_d = d
    if best is None:
        raise RuntimeError(f"no yellow CC near ({cx},{cy})")
    return best


def find_name_strip(img: np.ndarray, ybox: tuple[int, int, int, int]
                    ) -> tuple[int, int, int, int]:
    """Inside the yellow box, find the white rectangle at the bottom
    (the name label)."""
    x, y, w, h = ybox
    # Take the lower ~40% of the yellow box and look for white pixels.
    y0 = y + int(h * 0.60)
    y1 = y + h
    region = img[y0:y1, x:x + w]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 180) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
    # Largest white CC = the name strip.
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    best = None
    best_a = 0
    for i in range(1, nl):
        wx, wy, ww, wh, area = stats[i]
        if area > best_a and ww > w * 0.4:
            best_a = area
            best = (int(wx + x), int(wy + y0), int(ww), int(wh))
    if best is None:
        raise RuntimeError("no white name strip found inside yellow box")
    return best


def ocr_name(img: np.ndarray, strip: tuple[int, int, int, int]) -> str:
    sx, sy, sw, sh = strip
    # Crop INWARD a few pixels to drop the orange selection border that
    # frames the white name plate -- otherwise the stripe contaminates
    # thresholding and tesseract.
    crop = img[sy + 2:sy + sh - 6, sx + 4:sx + sw - 4]
    big = cv2.resize(crop, (crop.shape[1] * 6, crop.shape[0] * 6),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # White margin around the text so tesseract treats it as a clean line.
    bw = cv2.copyMakeBorder(bw, 40, 40, 40, 40,
                            cv2.BORDER_CONSTANT, value=255)
    cfg = ("--psm 7 --oem 1 "
           "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ.")
    text = pytesseract.image_to_string(bw, config=cfg).strip()
    return text, big, bw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--bboxes", default="screenshots/gallery_bboxes.json")
    ap.add_argument("--idx", type=int, required=True,
                    help="index of the reference bbox in the JSON")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    bboxes = json.loads(Path(args.bboxes).read_text())
    b = bboxes["boxes"][args.idx]
    cx, cy = b["x"] + b["w"] // 2, b["y"] + b["h"] // 2
    print(f"reference bbox idx={args.idx}: ({b['x']}, {b['y']}, "
          f"{b['w']}, {b['h']})  center=({cx},{cy})")

    ybox = find_yellow_box(img, cx, cy)
    print(f"yellow box: x={ybox[0]} y={ybox[1]} w={ybox[2]} h={ybox[3]}")

    strip = find_name_strip(img, ybox)
    print(f"name strip: x={strip[0]} y={strip[1]} w={strip[2]} h={strip[3]}")

    text, big, bw = ocr_name(img, strip)
    print(f"OCR: '{text}'")

    if args.debug:
        out = img.copy()
        cv2.rectangle(out, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 0), 2)
        cv2.rectangle(out, (ybox[0], ybox[1]),
                      (ybox[0] + ybox[2], ybox[1] + ybox[3]), (0, 255, 255), 3)
        cv2.rectangle(out, (strip[0], strip[1]),
                      (strip[0] + strip[2], strip[1] + strip[3]), (0, 0, 255), 2)
        p = Path(args.image)
        dbg_p = p.with_name(p.stem + f"_idx{args.idx}_dbg.png")
        cv2.imwrite(str(dbg_p), out)
        print(f"wrote {dbg_p}")
        cv2.imwrite("/tmp/name_strip_bw.png", bw)


if __name__ == "__main__":
    main()

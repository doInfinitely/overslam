#!/usr/bin/env python
"""Locate the character-portrait boxes in the OW2 hero-gallery UI by
detecting the curved bracket ("sideways parenthesis") shapes that sit
along the top and bottom edge of every unselected portrait tile.

Algorithm:
  1. HSV-threshold for near-white pixels.
  2. Connected components on the mask -- each arc/bracket is one CC.
  3. Filter CCs by bounding-box width (tile-width range) and height
     (arc thickness, short). The arcs of unselected tiles all share the
     same width; the selected tile is wider so it falls outside the
     accepted width range automatically.
  4. Cluster the survivors into top-row and bottom-row Y-bands. Pair an
     arc in the top band with one in the bottom band when their X
     bounding boxes overlap by IoU > 0.6 -- that pair is one tile.

Pass two images via --image (one each) and we report all tile boxes
from each; the selected tile differs between them so the union covers
every cell.

Usage:
    ./gallery-portraits.py <screenshot.png> [<screenshot2.png> ...]
                          [--out <name>] [--debug]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def locate_strip(img: np.ndarray) -> tuple[int, int]:
    H = img.shape[0]
    return int(H * 0.66), int(H * 0.92)


def white_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Near-white pixels (high value, low saturation)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    return ((v > 200) & (s < 40)).astype(np.uint8) * 255


def arc_components(white: np.ndarray, w_lo: int, w_hi: int,
                   h_lo: int, h_hi: int,
                   ) -> list[tuple[int, int, int, int]]:
    """Return CC bounding boxes (x, y, w, h) whose shape matches an arc
    (wide, short) in the expected tile-width range."""
    nl, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    out: list[tuple[int, int, int, int]] = []
    for i in range(1, nl):
        x, y, w, h, _ = stats[i]
        if w_lo <= w <= w_hi and h_lo <= h <= h_hi:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def cluster_y_bands(arcs, gap_tol: int):
    """Group arcs by Y-centre using 1-D agglomeration. Returns a list of
    (y_centre, [arc_indices]) sorted by y."""
    ys = sorted(range(len(arcs)), key=lambda i: arcs[i][1])
    bands: list[list[int]] = []
    for i in ys:
        y = arcs[i][1]
        if bands:
            last_y = arcs[bands[-1][-1]][1]
            if y - last_y <= gap_tol:
                bands[-1].append(i)
                continue
        bands.append([i])
    return [(float(np.mean([arcs[k][1] for k in b])), b) for b in bands]


def find_portrait_boxes(img_bgr: np.ndarray,
                        debug_path: Path | None = None
                        ) -> tuple[list[tuple[int, int, int, int]],
                                   tuple[int, int]]:
    y0, y1 = locate_strip(img_bgr)
    strip = img_bgr[y0:y1]
    H_strip, W_strip = strip.shape[:2]

    # Tile geometry priors. Tiles are roughly square; we observed widths
    # ~70px and heights ~75px on 2560x1440 -- about 0.20 * H_strip.
    tile_w = int(round(0.20 * H_strip))
    w_lo, w_hi = int(tile_w * 0.70), int(tile_w * 1.05)  # exclude wider SELECTED tile
    # An arc CC may be merged with the text label sitting just below the
    # arc (INITIATOR / BRUISER / STALWART labels). The bbox TOP still
    # marks the arc location, so we let h go up to ~tile_w; only reject
    # huge blobs that obviously aren't tile borders.
    h_lo, h_hi = 1, int(tile_w * 0.95)

    white = white_mask(strip)
    arcs = arc_components(white, w_lo, w_hi, h_lo, h_hi)

    if debug_path is not None:
        dbg = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in arcs:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 200, 255), 1)
        cv2.imwrite(str(debug_path), dbg)

    # Cluster into Y-bands. Bracket arcs of one tile lie within a few
    # pixels of each other vertically.
    band_tol = max(3, int(tile_w * 0.05))
    bands_all = cluster_y_bands(arcs, gap_tol=band_tol)
    # Keep only bands large enough to be a real tile row -- a single
    # tile row has 15-30 arcs; stray text or highlight CCs cluster in
    # smaller bands.
    min_band = 10
    bands = [(y, idxs) for (y, idxs) in bands_all if len(idxs) >= min_band]

    # Pair successive bands as (top, bottom). The expected gap between
    # the top arc and bottom arc of one tile is roughly H_t ~= tile_w.
    gap_min = int(tile_w * 0.80)
    gap_max = int(tile_w * 1.40)

    boxes: list[tuple[int, int, int, int]] = []
    used_band = [False] * len(bands)
    for i in range(len(bands)):
        if used_band[i]:
            continue
        y_top, idx_top = bands[i]
        # find a partner band below
        best_j = -1
        for j in range(i + 1, len(bands)):
            if used_band[j]:
                continue
            y_bot, _ = bands[j]
            gap = y_bot - y_top
            if gap_min <= gap <= gap_max:
                best_j = j
                break
        if best_j < 0:
            continue
        used_band[i] = used_band[best_j] = True
        y_bot, idx_bot = bands[best_j]
        # Median row height (top of top-arc to bottom of bot-arc) lets
        # us synthesize a box from a lone top-arc when no bot-arc pairs.
        row_h = int(round(y_bot - y_top + 6))
        matched_top: set[int] = set()
        for ti in idx_top:
            tx, ty, tw, th = arcs[ti]
            for bi in idx_bot:
                bx, by, bw, bh = arcs[bi]
                ix0 = max(tx, bx)
                ix1 = min(tx + tw, bx + bw)
                if ix1 <= ix0:
                    continue
                inter = ix1 - ix0
                union = min(tw, bw)
                if inter / float(union) < 0.6:
                    continue
                x = min(tx, bx)
                x_end = max(tx + tw, bx + bw)
                width = x_end - x
                height = (by + bh) - ty
                boxes.append((x, ty + y0, width, height))
                matched_top.add(ti)
                break
        # Lone top arcs (no bot pair) -- synthesize a box using the row
        # height inferred from the band gap.
        for ti in idx_top:
            if ti in matched_top:
                continue
            tx, ty, tw, th = arcs[ti]
            boxes.append((tx, ty + y0, tw, row_h))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, (y0, y1)


def union_boxes(box_lists, x_tol: int = 12, y_tol: int = 8):
    """Merge box lists from multiple images, deduplicating tiles that
    appear in multiple inputs (same X within tolerance, same Y row)."""
    merged: list[tuple[int, int, int, int]] = []
    for bxs in box_lists:
        for x, y, w, h in bxs:
            dup = False
            for mx, my, mw, mh in merged:
                if abs(mx - x) <= x_tol and abs(my - y) <= y_tol:
                    dup = True; break
            if not dup:
                merged.append((x, y, w, h))
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def interpolate_row_gaps(boxes, pitch_tol: float = 0.12):
    """Fill missing tiles within a row when the X-gap between two
    detected tiles is a clean integer multiple of the row's modal
    tile pitch."""
    # Cluster boxes into rows by Y.
    by_row: dict[int, list[tuple[int, int, int, int]]] = {}
    for b in boxes:
        key = None
        for k in by_row:
            if abs(k - b[1]) <= 8:
                key = k; break
        if key is None:
            by_row[b[1]] = [b]
        else:
            by_row[key].append(b)

    filled: list[tuple[int, int, int, int, bool]] = []
    for y_row, row in by_row.items():
        row.sort(key=lambda b: b[0])
        # Compute modal pitch from adjacent gaps (consider only "tight"
        # neighbor gaps -- between 0.90 and 1.10 times the typical tile
        # width; this excludes section-divider gaps).
        ws = [b[2] for b in row]
        tile_w = int(np.median(ws))
        gaps = [row[i + 1][0] - row[i][0] for i in range(len(row) - 1)]
        tight = [g for g in gaps if abs(g - tile_w) <= tile_w * 0.10]
        pitch = float(np.median(tight)) if tight else float(tile_w)

        for i, b in enumerate(row):
            filled.append((*b, False))
            if i + 1 >= len(row):
                continue
            gap = row[i + 1][0] - b[0]
            # Try k = 2, 3, 4 -- gap should be ~ k * pitch.
            for k in (2, 3, 4):
                if abs(gap - k * pitch) <= pitch * pitch_tol:
                    step = gap / k
                    for j in range(1, k):
                        nx = int(round(b[0] + j * step))
                        ny = b[1]
                        nw = tile_w
                        nh = b[3]
                        filled.append((nx, ny, nw, nh, True))
                    break
    filled.sort(key=lambda b: (b[1], b[0]))
    boxes_out = [(x, y, w, h) for (x, y, w, h, _) in filled]
    inferred = {i for i, t in enumerate(filled) if t[4]}
    return boxes_out, inferred


def annotate(img: np.ndarray, boxes, strip, inferred=None):
    out = img.copy()
    y0, y1 = strip
    cv2.line(out, (0, y0), (out.shape[1], y0), (60, 60, 60), 1)
    cv2.line(out, (0, y1), (out.shape[1], y1), (60, 60, 60), 1)
    inferred = set(inferred or [])
    for i, (x, y, w, h) in enumerate(boxes):
        col = (0, 150, 255) if i in inferred else (0, 255, 80)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        # Center label horizontally over the bbox so it visually
        # corresponds to the tile it actually encloses.
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.55, 2)
        lx = x + (w - tw) // 2
        ly = y - 4
        cv2.rectangle(out, (lx - 3, ly - th - 2),
                      (lx + tw + 3, ly + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+",
                    help="one or more screenshots to process")
    ap.add_argument("--debug", action="store_true",
                    help="dump white-mask + accepted arc bboxes")
    args = ap.parse_args()

    per_image = []
    for img_arg in args.images:
        p_in = Path(img_arg)
        img = cv2.imread(str(p_in))
        if img is None:
            print(f"could not read {p_in}"); continue
        out_path = p_in.with_name(p_in.stem + "_boxes.png")
        dbg = out_path.with_name(out_path.stem + "_white.png") \
            if args.debug else None
        boxes, strip = find_portrait_boxes(img, debug_path=dbg)
        per_image.append((p_in, img, boxes, strip))
        print(f"{p_in.name}: {len(boxes)} boxes")
        if boxes:
            ws = [b[2] for b in boxes]; hs = [b[3] for b in boxes]
            print(f"  width  {min(ws)}..{max(ws)}  median {int(np.median(ws))}")
            print(f"  height {min(hs)}..{max(hs)}  median {int(np.median(hs))}")
        cv2.imwrite(str(out_path), annotate(img, boxes, strip))
        print(f"  wrote {out_path}")
        if dbg is not None:
            print(f"  wrote {dbg}")

    if len(per_image) >= 2:
        merged_boxes = union_boxes([b for _, _, b, _ in per_image])
        filled, inferred = interpolate_row_gaps(merged_boxes)
        print(f"\nunion across {len(per_image)} images: {len(merged_boxes)} boxes")
        print(f"after gap-fill: {len(filled)} boxes ({len(inferred)} inferred)")
        # Annotate the FIRST image with the union+filled boxes for review.
        p_first, img_first, _, strip_first = per_image[0]
        out_path = p_first.with_name("union_filled_boxes.png")
        cv2.imwrite(str(out_path), annotate(img_first, filled, strip_first,
                                            inferred=inferred))
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()

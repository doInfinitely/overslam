#!/usr/bin/env python3
"""Offline wall tracker.

Per event directory, opens clip.mp4 + event.json. For each frame:
  1. Build a blue mask from the frame (HSV in the ice-wall range).
  2. Run the analyze-walls rectangle search, biased toward the previous
     frame's bbox via a Gaussian prior. Search is restricted to a window
     around the predicted position (prev bbox + median flow vector).
  3. Compute Farneback dense flow vs the previous frame.

Outputs per event:
  - track.json   : per-frame {frame_idx, t_ms, bbox, score, iou,
                   corners (front face approx), flow_summary}
  - track.mp4    : clip with bbox overlay + frame index

Standalone offline script; reads only files inside the event dir
(except --events-dir to enumerate). No live capture.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# Default tuning — same blue + gun-zone parameters as analyze-walls.
DEFAULTS = dict(
    blue_h_min=90, blue_h_max=130, blue_s_min=50, blue_v_min=40,
    gun_x_frac=0.50, gun_y_frac=0.55,
)


def blue_mask_of(frame_bgr, p):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(
        hsv,
        np.array([p["blue_h_min"], p["blue_s_min"], p["blue_v_min"]], np.uint8),
        np.array([p["blue_h_max"], 255,             255             ], np.uint8),
    )
    H, W = m.shape
    gx0 = int(W * p["gun_x_frac"])
    gy0 = int(H * p["gun_y_frac"])
    m[gy0:, gx0:] = 0
    return m


def search_rect(blue, prior=None, sigma_px=120.0,
                widths=(30, 50, 70, 100, 140, 180, 220, 280, 350, 450,
                        550, 700, 850, 1000, 1150, 1280),
                heights=(30, 50, 70, 100, 140, 180, 240, 320, 420,
                         520, 620, 720),
                step_px=12):
    """Rectangle search on a blue mask. Score = IoU(blue, R) * prior_score.

    prior: (cx_pred, cy_pred, w_pred, h_pred) or None. If given, prefers
    rectangles whose center is near (cx_pred, cy_pred) and whose size is
    near (w_pred, h_pred). For the first frame, pass None to get a flat
    search.
    """
    H, W = blue.shape
    target = (blue > 0).astype(np.float64)
    total = float(target.sum())
    if total < 1:
        return None
    t_int = cv2.integral(target)

    if prior is not None:
        cxp, cyp, wp, hp = prior
        # Search around the prior; size variation +/-50%
        # but clamp to candidate grid
        widths = [w for w in widths if 0.4 * wp <= w <= 2.5 * wp] or list(widths)
        heights = [h for h in heights if 0.4 * hp <= h <= 2.5 * hp] or list(heights)

    best = None
    best_score = -1.0
    for w in widths:
        if w > W:
            continue
        for h in heights:
            if h > H:
                continue
            xs = np.arange(0, W - w + 1, step_px)
            ys = np.arange(0, H - h + 1, step_px)
            if xs.size == 0 or ys.size == 0:
                continue
            X, Y = np.meshgrid(xs, ys)
            X1 = X + w
            Y1 = Y + h
            area = float(w * h)
            t_sum = (t_int[Y1, X1] - t_int[Y, X1]
                     - t_int[Y1, X] + t_int[Y, X])
            iou = t_sum / (area + total - t_sum + 1e-6)
            if prior is not None:
                cx = X + w / 2
                cy = Y + h / 2
                d2 = (cx - cxp) ** 2 + (cy - cyp) ** 2
                prior_score = np.exp(-d2 / (2 * sigma_px ** 2))
            else:
                prior_score = 1.0
            score = iou * prior_score
            idx = np.unravel_index(np.argmax(score), score.shape)
            if score[idx] > best_score:
                best_score = float(score[idx])
                best = (int(X[idx]), int(Y[idx]), int(w), int(h),
                        float(iou[idx]))
    return best  # (x, y, w, h, iou) or None


def flow_summary(prev_gray, curr_gray, bbox):
    """Farneback dense flow; return median (dx, dy) inside the bbox."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    x, y, w, h = bbox
    sub = flow[y:y + h, x:x + w]
    if sub.size == 0:
        return 0.0, 0.0, 0.0
    dx = float(np.median(sub[..., 0]))
    dy = float(np.median(sub[..., 1]))
    mag = float(np.median(np.sqrt(sub[..., 0] ** 2 + sub[..., 1] ** 2)))
    return dx, dy, mag


def front_face_quad(blue_mask, min_area=2000):
    """Detect the wall front face as a perspective quadrilateral via the
    four corner-extrema of the largest blue component.

    For a roughly-rectangular blob viewed obliquely, the extrema along the
    (x+y) and (x-y) diagonals ARE the four corners, and -- unlike an
    axis-aligned bbox or a minAreaRect -- they preserve the perspective
    foreshortening that encodes camera yaw/pitch.

    Returns (corners (4,2) float32 ordered TL,TR,BR,BL, area, iou_with_mask)
    or (None, 0, 0.0).
    """
    bin_ = (blue_mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, 8)
    if n <= 1:
        return None, 0, 0.0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area:
        return None, area, 0.0
    ys, xs = np.where(labels == i)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    s = pts[:, 0] + pts[:, 1]   # x + y
    d = pts[:, 0] - pts[:, 1]   # x - y
    TL = pts[np.argmin(s)]      # top-left  (min x+y)
    BR = pts[np.argmax(s)]      # bottom-right (max x+y)
    TR = pts[np.argmax(d)]      # top-right (max x-y)
    BL = pts[np.argmin(d)]      # bottom-left (min x-y)
    corners = np.stack([TL, TR, BR, BL], axis=0).astype(np.float32)
    # IoU of the filled quad with the blue mask (quality signal)
    H, W = blue_mask.shape
    quad = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(quad, corners.astype(np.int32), 255)
    inter = int(((quad > 0) & (blue_mask > 0)).sum())
    union = int(((quad > 0) | (blue_mask > 0)).sum())
    return corners, area, (inter / max(union, 1))


def bbox_to_corners(bbox):
    """(x, y, w, h) -> 4x2 float32 array of (TL, TR, BR, BL)."""
    x, y, w, h = bbox
    return np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32)


def corners_iou_with_blue(corners, blue_mask):
    """Polygon-fill the 4 corners and compute IoU with the blue mask.
    Returns 0 if corners are degenerate / out of frame."""
    H, W = blue_mask.shape
    poly = corners.astype(np.int32)
    if (poly[:, 0].max() < 0 or poly[:, 0].min() >= W
            or poly[:, 1].max() < 0 or poly[:, 1].min() >= H):
        return 0.0
    quad_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, poly, 255)
    inter = int(((quad_mask > 0) & (blue_mask > 0)).sum())
    union = int(((quad_mask > 0) | (blue_mask > 0)).sum())
    return inter / max(union, 1)


def track_corners_lk(prev_gray, curr_gray, corners):
    """Lucas-Kanade with pyramid. corners: (4, 2) float32.
    Returns (new_corners, status[4], err[4])."""
    new_corners, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray,
        corners.reshape(-1, 1, 2),
        None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    return (new_corners.reshape(-1, 2),
            status.reshape(-1).astype(bool),
            err.reshape(-1))


def track_one_event(event_dir: Path, args):
    clip = event_dir / "clip.mp4"
    evt_json = event_dir / "event.json"
    if not clip.exists():
        return None, "no clip.mp4"
    if not evt_json.exists():
        return None, "no event.json"
    evt = json.loads(evt_json.read_text())
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return None, f"cannot open {clip}"
    fps = cap.get(cv2.CAP_PROP_FPS) or float(evt.get("fps", 30))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = event_dir / "track.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))

    p = DEFAULTS
    records = []
    prev_gray = None
    prev_corners = None     # (4, 2) float32 -- LK-tracked corners
    prev_bbox = None        # (x, y, w, h) bbox enclosing the corners
    median_flow = (0.0, 0.0)
    redetect_count = 0
    # Wall-loss detector: once iou_blue stays below `lost_iou` for
    # `lost_window` consecutive frames the wall is considered gone and
    # we truncate the trajectory there.
    consecutive_lost = 0
    wall_lost = False

    t0 = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blue = blue_mask_of(frame, p)

        corners = None
        iou_blue = 0.0
        redetected = False

        # ----- Primary: perspective-quad from blue-mask corner extrema -----
        quad, area, iou_q = front_face_quad(blue, min_area=args.min_wall_area)
        if quad is not None and iou_q >= args.min_corner_iou:
            corners = quad
            iou_blue = iou_q
        else:
            # Couldn't detect a confident front-face quad this frame.
            redetected = True
            redetect_count += 1
            if quad is not None:
                corners = quad          # low-confidence detection, still use it
                iou_blue = iou_q
            elif prev_corners is not None:
                corners = prev_corners.copy()
                iou_blue = 0.0
            else:
                corners = bbox_to_corners((W // 4, H // 4, W // 2, H // 2))
                iou_blue = 0.0

        # Bbox enclosing the (possibly non-axis-aligned) corners
        x0 = int(max(0, corners[:, 0].min()))
        y0 = int(max(0, corners[:, 1].min()))
        x1 = int(min(W, corners[:, 0].max()))
        y1 = int(min(H, corners[:, 1].max()))
        bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

        # Dense Farneback summary inside the bbox (still useful for the
        # state vector / debug)
        if prev_gray is not None:
            dx, dy, mag = flow_summary(prev_gray, gray, bbox)
            median_flow = (dx, dy)
        else:
            dx, dy, mag = 0.0, 0.0, 0.0

        records.append({
            "frame_idx": frame_idx,
            "t_ms": frame_idx * 1000.0 / fps,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "iou_blue": float(iou_blue),
            "corners": corners.astype(int).tolist(),
            "flow_median_dx_dy": [float(dx), float(dy)],
            "flow_median_mag":   float(mag),
            "redetected": bool(redetected),
        })

        # Overlay: draw tracked quadrilateral + bbox
        viz = frame.copy()
        poly = corners.astype(np.int32).reshape(-1, 1, 2)
        col = (0, 0, 255) if redetected else (0, 255, 0)
        cv2.polylines(viz, [poly], isClosed=True, color=col, thickness=2)
        for cx, cy in corners.astype(int):
            cv2.circle(viz, (cx, cy), 3, (255, 255, 0), -1)
        cv2.putText(viz, f"f={frame_idx} iou={iou_blue:.2f} "
                         f"{'lowconf' if redetected else 'quad'} "
                         f"flow=({dx:+.1f},{dy:+.1f})",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(viz)

        prev_gray = gray
        prev_corners = corners
        prev_bbox = bbox
        frame_idx += 1

        # Hard time cap: Mei walls last ~5s by game spec. The iou-based
        # loss detector gets fooled by the rect-search fallback locking
        # onto incidental blue (sky/UI/distant ice), so enforce a hard
        # cutoff regardless.
        if args.max_track_s > 0 and (frame_idx / fps) >= args.max_track_s:
            wall_lost = True
            print(f"    hit hard wall-lifetime cap at frame {frame_idx} "
                  f"({args.max_track_s}s); truncating wall-phase trajectory",
                  flush=True)
            break

        # Wall-loss detection (iou-based; usually fires before the hard cap
        # if the player turns away from the wall)
        if iou_blue < args.lost_iou:
            consecutive_lost += 1
        else:
            consecutive_lost = 0
        if consecutive_lost >= args.lost_window:
            wall_lost = True
            print(f"    wall lost after frame {frame_idx} "
                  f"({consecutive_lost} consecutive frames iou<{args.lost_iou}); "
                  "truncating trajectory", flush=True)
            records = records[:-consecutive_lost]
            break

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0

    # Write trajectory
    trajectory = {
        "event_name": event_dir.name,
        "clip_path": str(clip),
        "fps": fps,
        "n_frames": len(records),
        "n_clip_frames": frame_idx,
        "wall_lost_at_frame": (frame_idx - consecutive_lost) if wall_lost else None,
        "frame_size": [W, H],
        "orientation": evt.get("orientation"),
        "n_redetections": redetect_count,
        "wall_world_origin_note": "front-face bottom-center; +Y up, +X along width, +Z into wall",
        "corners_layout": "TL, TR, BR, BL (initialized axis-aligned at post-frame; "
                          "tracked per-corner via Lucas-Kanade after)",
        "frames": records,
    }
    (event_dir / "track.json").write_text(json.dumps(trajectory, indent=2))
    return {
        "name": event_dir.name,
        "n_frames": frame_idx,
        "n_redetect": redetect_count,
        "elapsed_s": elapsed,
        "track_mp4": str(out_video),
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events",
                    help="root containing per-event directories")
    ap.add_argument("--event", default=None,
                    help="single event dir name to process (default: all)")
    ap.add_argument("--prior-sigma-px", type=float, default=120.0,
                    help="Gaussian prior std-dev for bbox-center continuity "
                         "(default 120 px). Smaller = tighter tracking, "
                         "larger = more freedom for fast camera moves.")
    ap.add_argument("--search-step-px", type=int, default=12,
                    help="per-frame rectangle search step (default 12)")
    ap.add_argument("--min-corner-iou", type=float, default=0.55,
                    help="min IoU between the detected front-face quad and the "
                         "blue mask to treat the detection as confident "
                         "(default 0.55). A good front-face fit fills most of "
                         "its convex hull.")
    ap.add_argument("--min-wall-area", type=int, default=2000,
                    help="min blue-component area (px) to attempt quad "
                         "detection (default 2000)")
    ap.add_argument("--lost-iou", type=float, default=0.15,
                    help="if iou_blue stays below this for --lost-window "
                         "frames the wall is considered gone (melted, "
                         "occluded, etc.) and the trajectory is truncated "
                         "there (default 0.15)")
    ap.add_argument("--lost-window", type=int, default=6,
                    help="consecutive low-iou frames before declaring wall "
                         "lost (default 6 = ~200ms at 30fps)")
    ap.add_argument("--max-track-s", type=float, default=5.5,
                    help="hard cap on wall-phase tracking duration in seconds "
                         "(default 5.5; Mei walls last ~5s). Frames past this "
                         "are left for continue-pose.py's feature-map "
                         "relocalization. Set 0 to disable the hard cap.")
    args = ap.parse_args()

    root = Path(args.events_dir)
    if args.event:
        subs = [root / args.event]
    else:
        subs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Tracking {len(subs)} event(s)...")
    for d in subs:
        rec, err = track_one_event(d, args)
        if err:
            print(f"  [skip ] {d.name}: {err}")
        else:
            print(f"  [done ] {rec['name']}: {rec['n_frames']} frames "
                  f"in {rec['elapsed_s']:.1f}s ({rec['n_redetect']} re-detects) "
                  f"-> {rec['track_mp4']}")


if __name__ == "__main__":
    main()

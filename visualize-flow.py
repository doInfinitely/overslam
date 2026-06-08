#!/usr/bin/env python3
"""Visualize Farneback optical flow per event clip.

For each event with clip.mp4 produces:
  <event>/flow.mp4         : side-by-side video, [orig | flow-HSV]
  <event>/flow_samples/    : 6 PNG stills spaced across the clip

Flow color convention (standard):
  hue = flow direction (0 = right, going through cyan/blue/magenta etc.)
  brightness = flow magnitude (clipped at --mag-max px)
  saturation = full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def flow_to_hsv(flow, mag_max=20.0):
    """flow: (H, W, 2) -> BGR uint8 visualization."""
    H, W = flow.shape[:2]
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)        # 0-180
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(255, mag * (255.0 / mag_max)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def color_wheel_legend(size=120, mag_max=20.0):
    """Small color-wheel image: hue = angle, brightness = mag-to-center."""
    r = size // 2
    y, x = np.ogrid[-r:r, -r:r]
    mag = np.sqrt(x * x + y * y).astype(np.float32)
    ang = np.arctan2(y, x).astype(np.float32)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) * 180 / (2 * np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(255, mag / r * 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mask = (mag > r).astype(np.uint8)
    bgr[mask > 0] = (40, 40, 40)
    cv2.putText(bgr, "->", (r + 6, r + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1)
    return bgr


def annotate(viz_bgr, label):
    cv2.putText(viz_bgr, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(viz_bgr, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return viz_bgr


def flow_arrows(bgr, flow, grid=24, scale=2.0, mag_min=0.5, mag_max=30.0):
    """Overlay flow vectors as arrows on a copy of bgr.
    grid    : pixel spacing of the sample lattice (smaller = denser).
    scale   : arrow length multiplier; arrow length = flow * scale.
    mag_min : skip arrows whose flow magnitude is below this (keeps the
              static parts of the frame uncluttered).
    mag_max : flow magnitude that maps to the brightest arrow color.
    """
    viz = bgr.copy()
    H, W = flow.shape[:2]
    ys = np.arange(grid // 2, H, grid)
    xs = np.arange(grid // 2, W, grid)
    for y in ys:
        for x in xs:
            dx, dy = float(flow[y, x, 0]), float(flow[y, x, 1])
            mag = (dx * dx + dy * dy) ** 0.5
            if mag < mag_min:
                continue
            # Color arrow by magnitude: yellow (small) -> green -> cyan (big)
            t = min(1.0, mag / mag_max)
            color = (int(255 * t),                 # B grows
                     int(255),                     # G stays max
                     int(255 * (1.0 - t)))         # R fades
            x2 = int(round(x + dx * scale))
            y2 = int(round(y + dy * scale))
            cv2.arrowedLine(viz, (x, y), (x2, y2), color, 1,
                            cv2.LINE_AA, tipLength=0.35)
    return viz


def process_event(event_dir: Path, args):
    clip = event_dir / "clip.mp4"
    if not clip.exists():
        return None, "no clip.mp4"
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return None, "cannot open clip"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 2:
        cap.release()
        return None, "clip too short"

    # Output video: side-by-side, so width = 2*W
    suffix = "_arrows" if args.mode == "arrows" else ""
    out_video = event_dir / f"flow{suffix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (2 * W, H))

    # Sample frame indices for stills
    n_samples = args.n_samples
    sample_set = set(int(round(i * (n - 2) / max(1, n_samples - 1)))
                     for i in range(n_samples))
    samples_dir = event_dir / "flow_samples"
    samples_dir.mkdir(exist_ok=True)

    legend = color_wheel_legend(size=140, mag_max=args.mag_max)

    ok, prev_bgr = cap.read()
    if not ok:
        cap.release()
        return None, "cannot read first frame"
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    fi = 0
    saved_samples = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
        )
        if args.mode == "arrows":
            right = flow_arrows(bgr, flow,
                                grid=args.arrow_grid,
                                scale=args.arrow_scale,
                                mag_min=args.arrow_mag_min,
                                mag_max=args.mag_max)
            right_label = (f"flow {fi}->{fi+1}  arrows "
                           f"(grid={args.arrow_grid}px, "
                           f"x{args.arrow_scale} scale)")
        else:
            right = flow_to_hsv(flow, mag_max=args.mag_max)
            lh, lw = legend.shape[:2]
            right[H - lh - 8:H - 8, W - lw - 8:W - 8] = legend
            right_label = (f"flow {fi}->{fi+1}  "
                           f"(mag clip={args.mag_max}px)")

        side = np.concatenate([
            annotate(bgr.copy(),    f"frame {fi}"),
            annotate(right.copy(),  right_label),
        ], axis=1)
        writer.write(side)

        if fi in sample_set:
            out_png = samples_dir / f"flow_f{fi:04d}.png"
            cv2.imwrite(str(out_png), side)
            saved_samples += 1

        prev_gray = gray
        prev_bgr = bgr
        fi += 1

    cap.release()
    writer.release()
    return {"event": event_dir.name, "pairs": fi,
            "video": str(out_video), "samples": saved_samples}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None,
                    help="single event name (default: all with clip.mp4)")
    ap.add_argument("--n-samples", type=int, default=6,
                    help="number of PNG stills to dump per event (default 6)")
    ap.add_argument("--mode", choices=("color", "arrows"), default="color",
                    help="color = HSV-encoded direction (hue) + magnitude "
                         "(brightness); arrows = sparse vectors overlaid on "
                         "the original frame. (default: color)")
    ap.add_argument("--mag-max", type=float, default=20.0,
                    help="flow magnitude (px) that saturates the brightness "
                         "channel (color mode) or the arrow color ramp "
                         "(arrows mode). Default 20.")
    ap.add_argument("--arrow-grid", type=int, default=24,
                    help="(arrows mode) pixel spacing of the sample lattice. "
                         "Smaller = denser. Default 24.")
    ap.add_argument("--arrow-scale", type=float, default=2.0,
                    help="(arrows mode) arrow length multiplier. Default 2.0.")
    ap.add_argument("--arrow-mag-min", type=float, default=0.5,
                    help="(arrows mode) skip arrows whose flow is below this "
                         "magnitude in px. Keeps static regions uncluttered. "
                         "Default 0.5.")
    args = ap.parse_args()

    root = Path(args.events_dir)
    if args.event:
        subs = [root / args.event]
    else:
        subs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Visualizing flow for {len(subs)} event(s)...")
    for d in subs:
        rec, err = process_event(d, args)
        if err:
            print(f"  [skip] {d.name}: {err}")
            continue
        print(f"  [done] {rec['event']}: {rec['pairs']} pairs, "
              f"{rec['samples']} samples -> {rec['video']}")


if __name__ == "__main__":
    main()

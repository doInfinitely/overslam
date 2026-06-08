#!/usr/bin/env python3
"""Visualize per-frame flow-triangulated depth.

For each event with clip.mp4 + pose.json + track.json:
  1. For each consecutive frame pair (i, i+1) with valid poses,
     compute Farneback flow and triangulate per-pixel depth in
     frame_i's camera frame (same code path as reconstruct-scene.py
     --depth flow).
  2. Colorize the depth map: close = warm, far = cool. NaN (no flow,
     low parallax, behind camera) shows the original frame through.
  3. Save:
       <event>/depth_overlay.mp4 : depth colormap blended onto frame
       <event>/depth_samples/    : 6 PNG stills (side-by-side
                                    [orig | overlay])

Depth output here is metric (OW-m, or placeholder-units if you
haven't filled calibration.json's wall dims yet).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ---------- copies of the geometry helpers from reconstruct-scene.py ----------
def rotation_world_to_cam(yaw, pitch):
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw = np.array([[ cy, 0, sy],
                      [  0, 1,  0],
                      [-sy, 0, cy]])
    R_pitch = np.array([[1,  0,   0  ],
                        [0,  cp, -sp],
                        [0,  sp,  cp]])
    return R_pitch @ R_yaw


def relative_pose(pose_a, pose_b):
    Ra = rotation_world_to_cam(pose_a["yaw_rad"], pose_a["pitch_rad"])
    Rb = rotation_world_to_cam(pose_b["yaw_rad"], pose_b["pitch_rad"])
    ta = np.array(pose_a["t"]); tb = np.array(pose_b["t"])
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


def flow_depth_map(frame_a, frame_b, pose_a, pose_b, K,
                   min_flow_px=1.5, max_depth=80.0):
    H, W = frame_a.shape[:2]
    R_rel, t_rel = relative_pose(pose_a, pose_b)
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if float(np.linalg.norm(t_rel)) < 1e-3:
        return depth
    g1 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
    )
    ys = np.arange(H); xs = np.arange(W)
    U, V = np.meshgrid(xs, ys)
    U2 = U + flow[..., 0]; V2 = V + flow[..., 1]
    valid = ((U2 >= 0) & (U2 < W - 1) & (V2 >= 0) & (V2 < H - 1)
             & (np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) > min_flow_px))
    if valid.sum() < 100:
        return depth
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R_rel, t_rel.reshape(3, 1)])
    pts1 = np.vstack([U[valid].astype(np.float32),
                      V[valid].astype(np.float32)])
    pts2 = np.vstack([U2[valid].astype(np.float32),
                      V2[valid].astype(np.float32)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    z = (pts4d[:3] / pts4d[3])[2]
    good = (z > 0.05) & (z < max_depth) & np.isfinite(z)
    z = np.where(good, z, np.nan)
    depth.reshape(-1)[(V[valid] * W + U[valid]).astype(np.int64)] = z.astype(np.float32)
    return depth


# ---------- depth -> color overlay ----------
def inpaint_depth(depth, max_island_radius_px=None, radius=4):
    """Fill NaN regions of a (H, W) float32 depth map by propagating
    from the boundary of valid depths inward (cv2.inpaint with Telea's
    fast marching method).

    max_island_radius_px : if set, only fill island regions whose
        smallest enclosing distance to a valid pixel is <= this. Stops
        absurd extrapolation across large no-flow voids (e.g. sky).
        Implemented by zeroing the mask on pixels whose distance-to-
        nearest-valid is greater than the cutoff.
    """
    H, W = depth.shape
    valid = np.isfinite(depth)
    if valid.sum() == 0:
        return depth.copy(), np.zeros_like(depth, dtype=bool)
    mask = (~valid).astype(np.uint8) * 255
    if max_island_radius_px is not None:
        dist = cv2.distanceTransform((~valid).astype(np.uint8),
                                     cv2.DIST_L2, 5)
        mask[dist > max_island_radius_px] = 0
    d_in = np.where(valid, depth, 0.0).astype(np.float32)
    filled = cv2.inpaint(d_in, mask, inpaintRadius=radius,
                         flags=cv2.INPAINT_TELEA)
    # Wherever we didn't ask cv2 to fill, restore NaN
    out = np.where(mask > 0, filled, np.where(valid, depth, np.nan))
    inpainted_mask = (mask > 0)
    return out.astype(np.float32), inpainted_mask


def colorize_depth(depth, d_near, d_far, colormap=cv2.COLORMAP_TURBO):
    """depth: (H, W) float, NaN where invalid. Returns (color, mask).
    color is BGR uint8 (NaN -> black), mask is bool of valid pixels."""
    valid = np.isfinite(depth)
    # Normalize valid pixels into [0, 255]
    d = np.zeros_like(depth, dtype=np.float32)
    d[valid] = np.clip(
        (depth[valid] - d_near) / max(1e-6, d_far - d_near),
        0.0, 1.0,
    )
    # Invert so CLOSE = warm (high value into a "hot" colormap end)
    inv = 1.0 - d
    inv8 = (inv * 255).astype(np.uint8)
    color = cv2.applyColorMap(inv8, colormap)
    color[~valid] = 0
    return color, valid


def overlay_depth_on_frame(bgr, depth, d_near, d_far, alpha=0.6):
    color, valid = colorize_depth(depth, d_near, d_far)
    out = bgr.copy()
    blend = cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0.0)
    out[valid] = blend[valid]
    return out, valid


def annotate(viz, label):
    cv2.putText(viz, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(viz, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return viz


def depth_legend(H, W=40, d_near=0.5, d_far=10.0):
    """Vertical colorbar showing the depth-to-color mapping."""
    bar = np.linspace(0, 1, H).astype(np.float32)
    bar = 1.0 - bar  # so near (top) is warm
    bar8 = (bar * 255).astype(np.uint8)
    bar8 = np.tile(bar8[:, None], (1, W))
    img = cv2.applyColorMap(bar8, cv2.COLORMAP_TURBO)
    # Labels
    cv2.putText(img, f"{d_near:.1f}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{d_far:.1f}", (2, H - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def process_event(event_dir: Path, args):
    pose_p  = event_dir / "pose.json"
    track_p = event_dir / "track.json"
    clip_p  = event_dir / "clip.mp4"
    if not (pose_p.exists() and track_p.exists() and clip_p.exists()):
        return None, "missing pose/track/clip"
    pose = json.loads(pose_p.read_text())
    pose_frames = pose["frames"]
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]])

    cap = cv2.VideoCapture(str(clip_p))
    if not cap.isOpened():
        return None, "cannot open clip"
    fps = cap.get(cv2.CAP_PROP_FPS) or float(pose.get("fps", 30))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = min(len(frames), len(pose_frames))
    if n < 2:
        return None, "clip too short"

    out_video = event_dir / "depth_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (2 * W, H))

    sample_set = set(int(round(i * (n - 2) / max(1, args.n_samples - 1)))
                     for i in range(args.n_samples))
    samples_dir = event_dir / "depth_samples"
    samples_dir.mkdir(exist_ok=True)

    # Helper: which frame pairs have valid pose?
    def pair_ok(i):
        pf  = pose_frames[i]
        npf = pose_frames[i + 1]
        return (pf.get("converged") and npf.get("converged")
                and pf.get("reproj_rms_px", 1e9) <= args.max_rms_px
                and npf.get("reproj_rms_px", 1e9) <= args.max_rms_px)

    # ----- Pass 1: compute all depth maps and collect valid samples
    # for auto-range fit. Store the depth maps so pass 2 doesn't redo
    # Farneback.
    depths = [None] * (n - 1)
    inpaint_masks = [None] * (n - 1)
    all_valid = []
    for i in range(n - 1):
        if not pair_ok(i):
            continue
        d = flow_depth_map(
            frames[i], frames[i + 1], pose_frames[i], pose_frames[i + 1], K,
            min_flow_px=args.min_flow_px, max_depth=args.depth_cutoff,
        )
        if args.fill:
            d, mask_inp = inpaint_depth(
                d,
                max_island_radius_px=args.fill_max_radius,
                radius=args.fill_inpaint_radius,
            )
            inpaint_masks[i] = mask_inp
        depths[i] = d
        vals = d[np.isfinite(d)]
        if vals.size:
            # subsample if very large
            if vals.size > 5000:
                idx = np.random.choice(vals.size, 5000, replace=False)
                vals = vals[idx]
            all_valid.append(vals)

    if args.d_near is not None and args.d_far is not None:
        d_lo, d_hi = float(args.d_near), float(args.d_far)
        src = "fixed"
    elif all_valid:
        cat = np.concatenate(all_valid)
        d_lo = float(np.percentile(cat, args.lo_pct))
        d_hi = float(np.percentile(cat, args.hi_pct))
        if d_hi - d_lo < 0.05:
            d_hi = d_lo + 0.05
        src = (f"auto p{args.lo_pct:g}-p{args.hi_pct:g} from "
               f"{cat.size} samples")
    else:
        d_lo, d_hi = 0.3, 8.0
        src = "default (no valid depth found)"
    print(f"  depth range: {d_lo:.2f} - {d_hi:.2f} ow-m  ({src})")

    legend = depth_legend(H=200, W=44, d_near=d_lo, d_far=d_hi)

    # ----- Pass 2: render frames with the fitted range
    n_done = 0
    for i in range(n - 1):
        if not pair_ok(i) or depths[i] is None:
            right = frames[i].copy()
            cv2.rectangle(right, (0, 0), (W, H), (40, 40, 40), -1)
            right = cv2.addWeighted(frames[i], 0.4, right, 0.6, 0.0)
            annotate(right, f"flow {i}->{i+1}  (no depth)")
        else:
            right, valid = overlay_depth_on_frame(
                frames[i], depths[i], d_lo, d_hi, alpha=args.alpha,
            )
            lh, lw = legend.shape[:2]
            right[H - lh - 8:H - 8, W - lw - 8:W - 8] = legend
            valid_pct = 100 * float(valid.sum()) / (H * W)
            annotate(right, f"flow {i}->{i+1}  "
                            f"depth {d_lo:.2f}-{d_hi:.2f} ow-m  "
                            f"valid={valid_pct:.0f}%")
        annotate(frames[i], f"frame {i}")
        side = np.concatenate([frames[i], right], axis=1)
        writer.write(side)
        if i in sample_set:
            cv2.imwrite(str(samples_dir / f"depth_f{i:04d}.png"), side)
        n_done += 1

    writer.release()
    return {"event": event_dir.name, "pairs": n_done,
            "video": str(out_video),
            "d_lo": d_lo, "d_hi": d_hi}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--d-near", type=float, default=None,
                    help="depth (OW-m) mapped to the warmest color. "
                         "If unset, derived from --lo-pct of valid depths.")
    ap.add_argument("--d-far",  type=float, default=None,
                    help="depth (OW-m) mapped to the coolest color. "
                         "If unset, derived from --hi-pct of valid depths.")
    ap.add_argument("--lo-pct", type=float, default=5.0,
                    help="percentile of valid depths used as d-near when "
                         "auto-fitting (default 5)")
    ap.add_argument("--hi-pct", type=float, default=95.0,
                    help="percentile of valid depths used as d-far when "
                         "auto-fitting (default 95)")
    ap.add_argument("--depth-cutoff", type=float, default=20.0,
                    help="absolute triangulation cutoff in OW-m (default 20). "
                         "Different from d-far -- this clamps insane "
                         "triangulations BEFORE the auto-fit sees them.")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="depth-colormap opacity over the frame (default 0.6)")
    ap.add_argument("--fill", action="store_true",
                    help="inpaint NaN (low-flow) islands by propagating "
                         "depth from their perimeter inward (cv2.inpaint, "
                         "Telea fast marching).")
    ap.add_argument("--fill-max-radius", type=float, default=None,
                    help="when --fill, only fill pixels within this distance "
                         "(in px) of a valid pixel. Prevents extrapolation "
                         "across huge no-flow voids like sky. Default: no "
                         "limit (fill everything).")
    ap.add_argument("--fill-inpaint-radius", type=int, default=4,
                    help="cv2.inpaint propagation radius (default 4 px)")
    ap.add_argument("--min-flow-px", type=float, default=1.5,
                    help="min flow magnitude to trust triangulation "
                         "(default 1.5)")
    ap.add_argument("--max-rms-px", type=float, default=100.0,
                    help="skip frames whose pose RMS exceeds this (default 100)")
    args = ap.parse_args()

    root = Path(args.events_dir)
    subs = [root / args.event] if args.event else \
        sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Visualizing depth for {len(subs)} event(s)...")
    for d in subs:
        rec, err = process_event(d, args)
        if err:
            print(f"  [skip] {d.name}: {err}")
            continue
        print(f"  [done] {rec['event']}: {rec['pairs']} pairs -> {rec['video']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-event point cloud reconstruction.

For each event directory containing clip.mp4 + pose.json + track.json:
  1. Run DepthAnything on each clip frame -> relative depth map.
  2. Calibrate depth to OW-m per frame using the wall front-face corners
     as a known-3D fiducial (their expected camera-frame z is computed
     from the pose; DA depth at those pixels is fit against it).
  3. Unproject subsampled pixels to world via the per-frame pose.
  4. Voxelize and aggregate observations per voxel.
  5. Write pointcloud.ply -- consistent voxels keep their median color,
     "disputed" voxels (high color variance across observations) get
     red-tinted so dynamic objects are visually distinguishable.

Assumes pose.json was produced by pose-from-track.py (no-roll
parameterization, OpenCV convention: world Y down, +Z into wall).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# DepthAnything wrapper lives in the turntable repo
sys.path.insert(0, os.path.expanduser("~/turntable"))


def rotation_world_to_cam(yaw, pitch):
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ])
    R_pitch = np.array([
        [1,  0,   0  ],
        [0,  cp, -sp],
        [0,  sp,  cp],
    ])
    return R_pitch @ R_yaw


def wall_world_corners(orientation: str, mei: float, ling: float, zhou: float):
    """Same convention as pose-from-track.py: origin at wall front-face
    bottom-center, +X width, +Y down, +Z into wall. Returns TL, TR, BR, BL."""
    w = mei if orientation == "mei" else ling
    return np.array([
        [-w / 2, -zhou, 0.0],
        [+w / 2, -zhou, 0.0],
        [+w / 2,  0,    0.0],
        [-w / 2,  0,    0.0],
    ])


def calibrate_scale(da_depth, corners_image_px, corners_world, K, R, t):
    """Fit s such that s * DA_depth(pixel) ~ expected_z_cam at the wall corners.
    Returns (s, residual_px) or (None, None) if data is degenerate."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Expected camera-frame z for each world corner
    cam_pts = corners_world @ R.T + t  # (N, 3)
    z_expected = cam_pts[:, 2]
    if (z_expected <= 0).any():
        return None, None

    # Sample DA depth at each image-corner pixel
    H, W = da_depth.shape
    da_vals = []
    for (u, v) in corners_image_px:
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < W and 0 <= vi < H:
            da_vals.append(da_depth[vi, ui])
        else:
            da_vals.append(np.nan)
    da_vals = np.array(da_vals)
    ok = np.isfinite(da_vals) & (da_vals > 1e-6)
    if ok.sum() < 2:
        return None, None
    # Least-squares: minimize (s*da - z_expected)^2 -> s = (da.z) / (da.da)
    da = da_vals[ok]
    ze = z_expected[ok]
    s = float((da * ze).sum() / (da * da).sum())
    resid = float(np.sqrt(np.mean((s * da - ze) ** 2)))
    return s, resid


def unproject(depth_m, K, R, cam_pos_world, subsample=8,
              z_min=0.2, z_max=50.0):
    """Convert OW-m depth map to (N, 3) world points + (N,) image y, x indices."""
    H, W = depth_m.shape
    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    V, U = np.meshgrid(ys, xs, indexing="ij")  # (Hs, Ws)
    D = depth_m[V, U]
    mask = np.isfinite(D) & (D > z_min) & (D < z_max)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (U - cx) * D / fx
    y_cam = (V - cy) * D / fy
    z_cam = D
    P_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (Hs, Ws, 3)
    # World coords (row-vec form): P_world = P_cam @ R + cam_pos_world
    P_world = P_cam @ R + cam_pos_world

    return P_world[mask], V[mask], U[mask]


def relative_pose(pose_a, pose_b):
    """Relative pose s.t. P_camB = R_rel @ P_camA + t_rel.
    Each pose is {yaw_rad, pitch_rad, t} where P_cam = R(yaw,pitch) @ P_world + t.
    """
    Ra = rotation_world_to_cam(pose_a["yaw_rad"], pose_a["pitch_rad"])
    Rb = rotation_world_to_cam(pose_b["yaw_rad"], pose_b["pitch_rad"])
    ta = np.array(pose_a["t"])
    tb = np.array(pose_b["t"])
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


def flow_depth_map(frame_a, frame_b, pose_a, pose_b, K,
                   min_flow_px=1.5, max_depth=80.0):
    """Triangulate per-pixel depth in frame_a's camera frame using
    Farneback flow + the relative pose. Returns (depth_a, valid_mask)
    both shape (H, W); depth in OW-m, NaN where invalid."""
    H, W = frame_a.shape[:2]
    R_rel, t_rel = relative_pose(pose_a, pose_b)
    # If translation baseline is tiny, triangulation is degenerate; bail early.
    baseline = float(np.linalg.norm(t_rel))
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if baseline < 1e-3:
        return depth, np.zeros((H, W), dtype=bool)

    g1 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
    )

    ys = np.arange(H)
    xs = np.arange(W)
    U, V = np.meshgrid(xs, ys)
    U2 = U + flow[..., 0]
    V2 = V + flow[..., 1]
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    valid = ((U2 >= 0) & (U2 < W - 1) & (V2 >= 0) & (V2 < H - 1)
             & (flow_mag > min_flow_px))
    if valid.sum() < 100:
        return depth, valid

    # Projection matrices (OpenCV convention)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R_rel, t_rel.reshape(3, 1)])
    u1 = U[valid].astype(np.float32)
    v1 = V[valid].astype(np.float32)
    u2 = U2[valid].astype(np.float32)
    v2 = V2[valid].astype(np.float32)
    pts1 = np.vstack([u1, v1])
    pts2 = np.vstack([u2, v2])
    # cv2.triangulatePoints expects 2xN; returns 4xN homogeneous
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = pts4d[:3] / pts4d[3]
    z_cam1 = pts3d[2]

    # Discard impossible triangulations (behind camera or absurdly far)
    good = (z_cam1 > 0.05) & (z_cam1 < max_depth) & np.isfinite(z_cam1)
    z_cam1 = np.where(good, z_cam1, np.nan)

    depth_flat = depth.reshape(-1)
    flat_idx = (V[valid] * W + U[valid]).astype(np.int64)
    depth_flat[flat_idx] = z_cam1.astype(np.float32)
    valid_out = ~np.isnan(depth)
    return depth, valid_out


def write_ply(path: Path, points, colors, dispute_flags=None):
    """Write an ASCII PLY with optional dispute marker rendered as red tint."""
    n = len(points)
    if dispute_flags is None:
        dispute_flags = np.zeros(n, dtype=bool)
    cols = colors.copy()
    # Red-tint disputed voxels: increase R, dampen G/B
    cols[dispute_flags, 0] = np.minimum(255, cols[dispute_flags, 0] // 2)  # B
    cols[dispute_flags, 1] = np.minimum(255, cols[dispute_flags, 1] // 2)  # G
    cols[dispute_flags, 2] = 255                                            # R
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (b, g, r) in zip(points, cols):
            # PLY is RGB; we stored BGR -> swap on write
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def carve_free_space(centers, carve_frames, K, voxel_size,
                     free_frac=0.30, min_visible=2, margin_voxels=2.0):
    """Return a boolean keep-mask: drop voxels observed in free space.

    For each frame (R world->cam, cam_pos, z_cam depth map), project every
    voxel center, sample the observed depth at that pixel, and mark the
    voxel 'free' for that view if its camera-frame z is closer than the
    observed surface by more than `margin` (voxel sits between camera and
    surface -> the ray passed through empty space there). Voxels seen as
    free in >= free_frac of the views that observed them (>= min_visible)
    are carved out.
    """
    n = len(centers)
    if n == 0 or not carve_frames:
        return np.ones(n, dtype=bool)
    free = np.zeros(n, dtype=np.float32)
    vis = np.zeros(n, dtype=np.float32)
    margin = margin_voxels * voxel_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for (R, cam_pos, depth) in carve_frames:
        H, W = depth.shape
        P_cam = (centers - cam_pos) @ R.T
        z = P_cam[:, 2]
        in_front = z > 0.05
        u = fx * P_cam[:, 0] / np.maximum(z, 1e-3) + cx
        v = fy * P_cam[:, 1] / np.maximum(z, 1e-3) + cy
        in_img = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not in_img.any():
            continue
        ui = np.clip(u.astype(np.int32), 0, W - 1)
        vi = np.clip(v.astype(np.int32), 0, H - 1)
        d_obs = depth[vi, ui]
        valid = in_img & np.isfinite(d_obs) & (d_obs > 0)
        is_free = valid & (z < (d_obs - margin))
        free += is_free.astype(np.float32)
        vis += valid.astype(np.float32)
    frac = free / np.maximum(vis, 1.0)
    carve = (vis >= float(min_visible)) & (frac >= free_frac)
    return ~carve


def per_frame_depth(frame_a, frame_b, pose_a, pose_b, corners_a_image_px,
                    corners_world, K, mode, da, args):
    """Return (depth_m, source_mask) where source_mask values are
    0=invalid, 1=flow, 2=da, 3=da_rescaled_by_flow."""
    H, W = frame_a.shape[:2]
    depth = np.full((H, W), np.nan, dtype=np.float32)
    source = np.zeros((H, W), dtype=np.uint8)

    R_a = rotation_world_to_cam(pose_a["yaw_rad"], pose_a["pitch_rad"])
    t_a = np.array(pose_a["t"])

    if mode in ("flow", "fused") and frame_b is not None and pose_b is not None:
        flow_d, flow_valid = flow_depth_map(
            frame_a, frame_b, pose_a, pose_b, K,
            min_flow_px=args.min_flow_px,
            max_depth=args.z_max,
        )
        depth[flow_valid] = flow_d[flow_valid]
        source[flow_valid] = 1

    if mode in ("da", "fused"):
        da_rel = da.estimate(frame_a)
        # Prefer scaling DA to match flow depth where both exist (gets us
        # a consistent metric scale across the frame).
        scale = None
        scale_source = ""
        if mode == "fused" and (source == 1).any():
            ok = (source == 1) & np.isfinite(da_rel) & (da_rel > 1e-6)
            if ok.sum() > 200:
                # Robust scale: median ratio
                scale = float(np.median(depth[ok] / da_rel[ok]))
                scale_source = f"fit_to_flow({int(ok.sum())}px)"
        if scale is None:
            scale, _ = calibrate_scale(da_rel, corners_a_image_px,
                                       corners_world, K, R_a, t_a)
            scale_source = "wall_corners"
        if scale is None:
            # No flow to fit and no wall corners → no way to set metric scale.
            # Returning DA at relative units would land its points at random
            # distances; skip DA fill entirely for this frame.
            return depth, source, {"da_scale_source": "skipped_no_anchor"}
        da_depth = da_rel * scale

        if mode == "da":
            ok = np.isfinite(da_depth) & (da_depth > 0)
            depth[ok] = da_depth[ok]
            source[ok] = 2
        else:  # fused: fill flow-invalid pixels with DA
            needs_da = (source == 0) & np.isfinite(da_depth) & (da_depth > 0)
            depth[needs_da] = da_depth[needs_da]
            source[needs_da] = 3
        return depth, source, {"da_scale_source": scale_source,
                                "da_scale": float(scale)}
    return depth, source, {}


def reconstruct_event(event_dir: Path, cal, args, da):
    pose_p  = event_dir / "pose.json"
    track_p = event_dir / "track.json"
    clip_p  = event_dir / "clip.mp4"
    if not (pose_p.exists() and track_p.exists() and clip_p.exists()):
        return None, "missing pose/track/clip"
    pose = json.loads(pose_p.read_text())
    track = json.loads(track_p.read_text())
    pose_frames = pose["frames"]
    track_frames = track["frames"]
    if len(pose_frames) != len(track_frames):
        return None, ("frame count mismatch: pose vs track. "
                      "Rerun pose-from-track and track-wall together.")
    # Optional VO continuation: extends wall-phase poses past the 5 s
    # wall lifetime using frame-to-frame visual odometry.
    n_wall_frames = len(pose_frames)
    n_vo_frames = 0
    vo_p = event_dir / "continued_pose_vo.json"
    if vo_p.exists() and not args.no_vo:
        vo = json.loads(vo_p.read_text())
        vo_frames = vo.get("frames", [])
        # Sanity: VO frame_idx should pick up right after the wall phase.
        if vo_frames and vo_frames[0].get("frame_idx") == n_wall_frames:
            pose_frames = pose_frames + vo_frames
            n_vo_frames = len(vo_frames)
    orientation = pose.get("orientation", "mei")
    corners_world = wall_world_corners(orientation,
                                       cal["mei"], cal["ling"], cal["zhou"])

    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]])

    cap = cv2.VideoCapture(str(clip_p))
    if not cap.isOpened():
        return None, "cannot open clip.mp4"

    # Pre-read all frames (~15s clip at 30fps with downscaled capture ~ 800MB max).
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = min(len(frames), len(pose_frames))

    voxel = {}
    carve_frames = []   # (R_world2cam, cam_pos, depth_zcam) per used frame
    vsize = args.voxel_size
    stats = {"frames_used": 0, "frames_skipped": 0, "points_added": 0,
             "px_from_flow": 0, "px_from_da": 0,
             "rejected_rms": 0, "rejected_coverage": 0, "carved": 0}
    t0 = time.perf_counter()

    # Gun-zone mask (matches analyze-walls.py / train-voxel.py convention).
    H_img, W_img = frames[0].shape[:2] if frames else (0, 0)
    gx0 = int(W_img * args.gun_x_frac)
    gy0 = int(H_img * args.gun_y_frac)

    for i in range(n):
        pf = pose_frames[i]
        # Wall-phase frames have a track entry with wall corners; VO frames
        # (i >= n_wall_frames) don't — corners are only needed for DA-scale
        # anchoring, and the flow/fused paths can fit DA scale to flow instead.
        is_vo = i >= n_wall_frames
        tf = track_frames[i] if not is_vo else None
        # Wall RMS measures noisy wall-corner detection residual (floor ~33 px);
        # VO RMS measures PnP inlier residual (median ~10 px). Use separate
        # thresholds so a unified 20 px cutoff doesn't kill every wall frame.
        rms_cutoff = args.max_vo_rms_px if is_vo else args.max_rms_px
        if not pf.get("converged") or pf.get("reproj_rms_px", 1e9) > rms_cutoff:
            stats["frames_skipped"] += 1
            stats["rejected_rms"] += 1
            continue
        # For flow/fused, we triangulate i -> i+1; for the last valid frame
        # there's no "next", so fall back to DA-only for it.
        nxt_pose = None
        nxt_frame = None
        if i + 1 < n:
            npf = pose_frames[i + 1]
            nxt_cutoff = args.max_vo_rms_px if (i + 1) >= n_wall_frames \
                else args.max_rms_px
            if npf.get("converged") and npf.get("reproj_rms_px", 1e9) <= nxt_cutoff:
                nxt_pose = npf
                nxt_frame = frames[i + 1]
        eff_mode = args.depth
        if eff_mode == "flow" and nxt_frame is None:
            # No flow possible -> drop this frame for pure flow mode
            stats["frames_skipped"] += 1
            continue
        if eff_mode == "fused" and nxt_frame is None:
            eff_mode = "da"

        cam_pos = np.array(pf["camera_pos_world"])
        R = rotation_world_to_cam(pf["yaw_rad"], pf["pitch_rad"])
        # For VO frames there's no wall; pass empty corners (flow path ignores
        # them; fused path falls back to fit_to_flow for DA scale).
        corners_image_px = (np.array(tf["corners"], dtype=float)
                            if tf is not None else np.zeros((0, 2)))

        depth, source, extra = per_frame_depth(
            frames[i], nxt_frame, pf, nxt_pose,
            corners_image_px, corners_world, K, eff_mode, da, args,
        )

        # Gun-zone exclusion: NaN out depth + zero source in the gun rect
        if not args.no_gun_mask:
            depth[gy0:, gx0:] = np.nan
            source[gy0:, gx0:] = 0

        # Frame-quality filter: require enough flow-derived pixels.
        # (Source=1 means flow-triangulated; ignore DA-only frames here.)
        if eff_mode in ("flow", "fused") and args.min_flow_coverage > 0:
            flow_coverage = float((source == 1).sum()) / max(1, source.size)
            if flow_coverage < args.min_flow_coverage:
                stats["frames_skipped"] += 1
                stats["rejected_coverage"] += 1
                continue

        stats["px_from_flow"] += int(((source == 1)).sum())
        stats["px_from_da"]   += int(((source == 2) | (source == 3)).sum())

        pts, vs, us = unproject(depth, K, R, cam_pos,
                                subsample=args.subsample,
                                z_min=args.z_min, z_max=args.z_max)
        if len(pts) == 0:
            stats["frames_skipped"] += 1
            continue
        bgr = frames[i][vs, us]

        # Voxelize (vectorized aggregation)
        keys = np.floor(pts / vsize).astype(np.int64)
        for j in range(len(pts)):
            k = (int(keys[j, 0]), int(keys[j, 1]), int(keys[j, 2]))
            entry = voxel.get(k)
            b, g, r = int(bgr[j, 0]), int(bgr[j, 1]), int(bgr[j, 2])
            if entry is None:
                voxel[k] = [b, g, r, b * b, g * g, r * r, 1]
            else:
                entry[0] += b; entry[1] += g; entry[2] += r
                entry[3] += b * b; entry[4] += g * g; entry[5] += r * r
                entry[6] += 1

        # Keep this frame's depth + pose for free-space carving.
        if not args.no_carve:
            carve_frames.append((R, cam_pos.astype(np.float32), depth))

        stats["frames_used"] += 1
        stats["points_added"] += len(pts)

    if not voxel:
        return None, "no voxels accumulated"

    # Per-voxel aggregates
    n_v = len(voxel)
    centers = np.zeros((n_v, 3), dtype=np.float32)
    colors = np.zeros((n_v, 3), dtype=np.uint8)
    counts = np.zeros(n_v, dtype=np.int32)
    color_std = np.zeros(n_v, dtype=np.float32)
    for i, (k, v) in enumerate(voxel.items()):
        c = v[6]
        # Center of voxel (use grid center for snapping)
        centers[i] = (np.array(k, dtype=np.float64) + 0.5) * vsize
        mean_b = v[0] / c; mean_g = v[1] / c; mean_r = v[2] / c
        colors[i, 0] = int(np.clip(mean_b, 0, 255))
        colors[i, 1] = int(np.clip(mean_g, 0, 255))
        colors[i, 2] = int(np.clip(mean_r, 0, 255))
        # Color variance (sum of channel variances)
        var_b = max(0.0, v[3] / c - mean_b ** 2)
        var_g = max(0.0, v[4] / c - mean_g ** 2)
        var_r = max(0.0, v[5] / c - mean_r ** 2)
        color_std[i] = float(np.sqrt(var_b + var_g + var_r))
        counts[i] = c

    # Filter: drop voxels seen too few times (likely DA noise)
    keep = counts >= args.min_observations
    centers = centers[keep]
    colors  = colors[keep]
    counts  = counts[keep]
    color_std = color_std[keep]

    # Free-space carving: delete voxels that other views see *through*
    # (the voxel sits in front of the observed surface along those rays).
    # Kills floaters from bad flow correspondences / pose bias.
    if not args.no_carve and carve_frames:
        keep_c = carve_free_space(
            centers, carve_frames, K, vsize,
            free_frac=args.carve_free_frac,
            min_visible=args.carve_min_visible,
            margin_voxels=args.carve_margin_voxels)
        stats["carved"] = int((~keep_c).sum())
        centers = centers[keep_c]; colors = colors[keep_c]
        counts = counts[keep_c]; color_std = color_std[keep_c]

    # Dispute flag: high color variance AND seen multiple times
    dispute = (counts >= 3) & (color_std > args.dispute_std)

    if args.drop_disputed:
        keep2 = ~dispute
        centers = centers[keep2]
        colors  = colors[keep2]
        counts  = counts[keep2]
        color_std = color_std[keep2]
        dispute = np.zeros(len(centers), dtype=bool)

    elapsed = time.perf_counter() - t0
    ply_path = event_dir / "pointcloud.ply"
    write_ply(ply_path, centers, colors, dispute_flags=dispute)

    meta = {
        "event": event_dir.name,
        "depth_mode": args.depth,
        "voxel_size_ow_m": vsize,
        "n_voxels_total": int(n_v),
        "n_voxels_kept": int(len(centers)),
        "n_disputed": int(dispute.sum()),
        "drop_disputed": bool(args.drop_disputed),
        "n_frames_used": stats["frames_used"],
        "n_frames_skipped": stats["frames_skipped"],
        "rejected_rms": stats.get("rejected_rms", 0),
        "rejected_coverage": stats.get("rejected_coverage", 0),
        "carved": stats.get("carved", 0),
        "points_added": stats["points_added"],
        "px_from_flow": stats["px_from_flow"],
        "px_from_da":   stats["px_from_da"],
        "elapsed_s": elapsed,
        "min_observations": args.min_observations,
        "dispute_std_threshold": args.dispute_std,
        "ply_path": str(ply_path),
    }
    (event_dir / "pointcloud_meta.json").write_text(json.dumps(meta, indent=2))
    return meta, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None,
                    help="single event (default: all)")
    ap.add_argument("--calibration", default="./calibration.json")
    ap.add_argument("--voxel-size", type=float, default=0.05,
                    help="voxel size in OW-m (default 0.05)")
    ap.add_argument("--subsample", type=int, default=8,
                    help="image pixel stride when unprojecting (default 8 -> "
                         "1/64 of pixels per frame). Lower = denser cloud, "
                         "slower + more memory.")
    ap.add_argument("--z-min", type=float, default=0.2,
                    help="discard depths below this (OW-m, default 0.2)")
    ap.add_argument("--z-max", type=float, default=50.0,
                    help="discard depths above this (OW-m, default 50.0)")
    ap.add_argument("--min-observations", type=int, default=2,
                    help="drop voxels seen fewer times than this (default 2)")
    ap.add_argument("--dispute-std", type=float, default=45.0,
                    help="per-voxel color std (sum-of-channels stdev) above "
                         "which the voxel is flagged as disputed and rendered "
                         "red in the PLY (default 45)")
    ap.add_argument("--max-vo-rms-px", type=float, default=20.0,
                    help="max acceptable reprojection RMS in pixels for "
                         "VO-continuation frames (PnP inlier residual, "
                         "default 20; rejects fast-pan tracking glitches)")
    ap.add_argument("--max-rms-px", type=float, default=100.0,
                    help="max acceptable reprojection RMS for wall-phase "
                         "frames (noisy wall-corner residual, floor ~30 px; "
                         "default 100). See also --max-vo-rms-px.")
    ap.add_argument("--depth", choices=("flow", "da", "fused"), default="fused",
                    help="depth source. flow=triangulate from Farneback flow + "
                         "known relative pose (metric, temporally consistent, "
                         "but blank where there's no parallax). da=DA monocular "
                         "scaled by wall-corner fiducial (works everywhere but "
                         "noisier). fused=flow where confident, DA elsewhere, "
                         "with DA scale fit to flow depth in overlap regions.")
    ap.add_argument("--min-flow-px", type=float, default=1.5,
                    help="min flow magnitude (px) for triangulation to be "
                         "trusted; smaller flow = noisier depth (default 1.5)")
    ap.add_argument("--drop-disputed", action="store_true",
                    help="drop disputed voxels entirely instead of red-tinting "
                         "them. Useful when dispute is dominated by pose drift "
                         "(scattered red everywhere) and you want to see only "
                         "the consensus geometry.")
    ap.add_argument("--min-flow-coverage", type=float, default=0.05,
                    help="reject frames whose flow-triangulated pixels cover "
                         "less than this fraction of the (non-gun-zone) frame "
                         "(default 0.05). Frames with no parallax have no "
                         "metric anchor and add unreliable depth.")
    ap.add_argument("--gun-x-frac", type=float, default=0.50,
                    help="x fraction where Mei's gun-zone starts; pixels in "
                         "[gun_x, W] x [gun_y, H] are NaN'd before unprojection "
                         "(default 0.50)")
    ap.add_argument("--gun-y-frac", type=float, default=0.55,
                    help="y fraction where gun-zone starts (default 0.55)")
    ap.add_argument("--no-gun-mask", action="store_true",
                    help="disable gun-zone exclusion")
    ap.add_argument("--no-vo", action="store_true",
                    help="don't extend the wall-phase reconstruction with "
                         "continued_pose_vo.json (post-wall VO poses)")
    ap.add_argument("--no-carve", action="store_true",
                    help="disable free-space carving (the floater removal)")
    ap.add_argument("--carve-free-frac", type=float, default=0.30,
                    help="carve a voxel if >= this fraction of views that "
                         "observed it saw it as free space (default 0.30)")
    ap.add_argument("--carve-min-visible", type=int, default=2,
                    help="min views that must observe a voxel before it can "
                         "be carved (default 2)")
    ap.add_argument("--carve-margin-voxels", type=float, default=2.0,
                    help="free-space margin in voxel widths: voxel is 'free' "
                         "for a view if its z is closer than observed depth "
                         "by more than this (default 2.0; larger = more "
                         "conservative, carves less)")
    args = ap.parse_args()

    # Load calibration (same fields as pose-from-track)
    cal_path = Path(args.calibration)
    cal_raw = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    def g(keys, default=None):
        cur = cal_raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur
    cal = {
        "mei":  g(["mei_wall", "length_ow_m"], 1.0),
        "ling": g(["mei_wall", "depth_ow_m"],  1.0),
        "zhou": g(["mei_wall", "height_ow_m"], 1.0),
    }
    print(f"Wall dims (mei, ling, zhou) = ({cal['mei']}, {cal['ling']}, {cal['zhou']}) ow-m")
    if cal["mei"] == 1.0 and cal["ling"] == 1.0 and cal["zhou"] == 1.0:
        print("  (placeholder dims; point cloud will be in placeholder units, not OW-m)")

    if args.depth in ("da", "fused"):
        print("Loading DepthAnything (downloads weights on first run)...")
        from depth_anything import DepthAnythingEstimator
        da = DepthAnythingEstimator()
        print("DA ready.")
    else:
        da = None
        print("Depth mode = flow; skipping DepthAnything load.")

    root = Path(args.events_dir)
    subs = [root / args.event] if args.event else \
        sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"\nReconstructing {len(subs)} event(s)...")
    for d in subs:
        meta, err = reconstruct_event(d, cal, args, da)
        if err:
            print(f"  [skip] {d.name}: {err}")
            continue
        flow_frac = meta["px_from_flow"] / max(1, meta["px_from_flow"] + meta["px_from_da"])
        reject_str = ""
        if meta.get("rejected_rms", 0) or meta.get("rejected_coverage", 0):
            reject_str = (f" rejected: {meta.get('rejected_rms', 0)} rms + "
                          f"{meta.get('rejected_coverage', 0)} coverage;")
        carved = meta.get("carved", 0)
        carve_str = f" carved={carved}" if carved else ""
        print(f"  [done] {meta['event']}: {meta['n_voxels_kept']} voxels "
              f"({meta['n_disputed']} disputed), "
              f"{meta['n_frames_used']} frames "
              f"(flow={flow_frac*100:.0f}% of pixels),{reject_str}{carve_str} "
              f"{meta['elapsed_s']:.1f}s  -> {meta['ply_path']}")


if __name__ == "__main__":
    main()

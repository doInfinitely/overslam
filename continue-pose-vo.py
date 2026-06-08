#!/usr/bin/env python3
"""Frame-to-frame visual odometry past the wall phase.

The wall gives metric pose for the first ~5s. After it drops we track
the camera by chaining frame-to-frame, NOT by relocalizing against the
distant wall-phase map (CNN features aren't viewpoint-invariant enough
for that). Standard monocular VO with the wall providing initial scale:

  Seed (at last wall frame L):
    - Compute fused flow+DA depth at L (flow vs L-1, both wall poses known).
    - Backproject a sparse grid of valid-depth pixels -> 3D world points,
      remember each point's pixel location in L. These are the map points.

  Per subsequent frame i:
    1. Lucas-Kanade track each map point's pixel from i-1 to i.
    2. Drop points that left the frame or failed LK.
    3. solvePnPRansac(map 3D, tracked 2D) -> pose; refine no-roll.
    4. Prune points with high reprojection residual.
    5. If surviving points < --min-track, RE-SEED: scale DA depth at
       frame i to the surviving points' camera-frame z, backproject a
       fresh grid, add as new map points (keeps the map alive as the
       camera moves into new territory).

Output <event>/continued_pose_vo.json -- per-frame pose in the same
world frame as pose.json, continuing past last_wall_tracked_frame.
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

try:
    from scipy.optimize import least_squares
except ImportError:
    raise SystemExit("pip install scipy")

sys.path.insert(0, os.path.expanduser("~/turntable"))


# ---------- pose math ----------
def R_no_roll(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rp @ Ry


def project(world, yaw, pitch, t, K):
    R = R_no_roll(yaw, pitch)
    pc = world @ R.T + t
    z = np.where(np.abs(pc[:, 2]) < 1e-3, 1e-3, pc[:, 2])
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def pnp_no_roll(world, image, K, init):
    def resid(p):
        yaw, pitch, tx, ty, tz = p
        if tz < 0.02:
            return np.full(image.size, 1e3)
        proj, _ = project(world, yaw, pitch, np.array([tx, ty, tz]), K)
        return (proj - image).ravel()
    try:
        r = least_squares(resid, init, method="lm", max_nfev=80)
        return r.x, float(np.sqrt(np.mean(r.fun ** 2))), True
    except Exception:
        return init, float("inf"), False


def rotation_from_pose(pf):
    return R_no_roll(pf["yaw_rad"], pf["pitch_rad"])


# ---------- depth (flow + DA fused) ----------
def flow_depth(fa, fb, R_rel, t_rel, K, min_flow_px=1.5, max_depth=40.0):
    H, W = fa.shape[:2]
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if float(np.linalg.norm(t_rel)) < 1e-3:
        return depth
    g1 = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    ys = np.arange(H); xs = np.arange(W)
    U, V = np.meshgrid(xs, ys)
    U2 = U + flow[..., 0]; V2 = V + flow[..., 1]
    valid = ((U2 >= 0) & (U2 < W - 1) & (V2 >= 0) & (V2 < H - 1)
             & (np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) > min_flow_px))
    if valid.sum() < 100:
        return depth
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R_rel, t_rel.reshape(3, 1)])
    pts1 = np.vstack([U[valid].astype(np.float32), V[valid].astype(np.float32)])
    pts2 = np.vstack([U2[valid].astype(np.float32), V2[valid].astype(np.float32)])
    p4 = cv2.triangulatePoints(P1, P2, pts1, pts2)
    z = (p4[:3] / p4[3])[2]
    good = (z > 0.05) & (z < max_depth) & np.isfinite(z)
    z = np.where(good, z, np.nan)
    depth.reshape(-1)[(V[valid] * W + U[valid]).astype(np.int64)] = z.astype(np.float32)
    return depth


def fuse_da(flow_d, da_rel):
    ok = np.isfinite(flow_d) & np.isfinite(da_rel) & (da_rel > 1e-6)
    if int(ok.sum()) < 200:
        return flow_d.copy()
    s = float(np.median(flow_d[ok] / da_rel[ok]))
    out = flow_d.copy()
    fill = ~np.isfinite(out) & np.isfinite(da_rel)
    out[fill] = (da_rel * s)[fill]
    return out


def read_ply_points(path: Path):
    with open(path, "r") as f:
        if not f.readline().startswith("ply"):
            raise SystemExit(f"{path} not a ply")
        n = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("end_header"):
                break
        pts = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            pts[i] = [float(x) for x in f.readline().split()[:3]]
        return pts


def project_world_to_pixels(world_pts, R, t, K, W, H):
    """Project world points into a camera (R world->cam, t). Returns
    (kept_world, kept_pix) for points in front + in frame."""
    P_cam = world_pts @ R.T + t
    z = P_cam[:, 2]
    u = K[0, 0] * P_cam[:, 0] / np.maximum(z, 1e-3) + K[0, 2]
    v = K[1, 1] * P_cam[:, 1] / np.maximum(z, 1e-3) + K[1, 2]
    m = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix = np.stack([u[m], v[m]], axis=-1).astype(np.float32)
    return world_pts[m].astype(np.float32), pix


def backproject(depth, K, R, cam_pos, grid_step):
    """Sample valid-depth pixels on a grid; return world points + pixel uv."""
    H, W = depth.shape
    ys = np.arange(grid_step // 2, H, grid_step)
    xs = np.arange(grid_step // 2, W, grid_step)
    V, U = np.meshgrid(ys, xs, indexing="ij")
    D = depth[V, U]
    m = np.isfinite(D) & (D > 0.05) & (D < 40.0)
    u = U[m].astype(np.float32); v = V[m].astype(np.float32); d = D[m]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    P_cam = np.stack([x, y, d], axis=-1)
    P_world = P_cam @ R + cam_pos  # R is world->cam; world = cam @ R + pos
    pix = np.stack([u, v], axis=-1).astype(np.float32)
    return P_world.astype(np.float32), pix


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", required=True)
    ap.add_argument("--grid-step", type=int, default=12,
                    help="pixel stride for map-point seeding (default 12)")
    ap.add_argument("--min-track", type=int, default=300,
                    help="re-seed when surviving tracked points drop below this")
    ap.add_argument("--pnp-reproj-px", type=float, default=4.0,
                    help="solvePnPRansac inlier reprojection threshold (px)")
    ap.add_argument("--prune-px", type=float, default=6.0,
                    help="drop map points whose reprojection residual exceeds "
                         "this after solving (px)")
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--lk-win", type=int, default=31,
                    help="Lucas-Kanade window size (default 31; bigger = "
                         "survives larger inter-frame motion)")
    ap.add_argument("--lk-levels", type=int, default=5,
                    help="LK pyramid levels (default 5)")
    ap.add_argument("--max-blind-frames", type=int, default=45,
                    help="give up after this many consecutive blind "
                         "(prediction-only) frames (default 45 = 1.5s @ 30fps)")
    ap.add_argument("--no-da", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    event_dir = Path(args.events_dir) / args.event
    pose_p = event_dir / "pose.json"
    clip_p = event_dir / "clip.mp4"
    if not (pose_p.exists() and clip_p.exists()):
        raise SystemExit("need pose.json and clip.mp4")
    pose = json.loads(pose_p.read_text())
    pose_frames = pose["frames"]
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]],
                 dtype=np.float32)

    tracked = [f["frame_idx"] for f in pose_frames if f.get("converged")]
    if len(tracked) < 2:
        raise SystemExit("need >=2 converged wall poses")
    L = max(tracked)
    pose_by_idx = {f["frame_idx"]: f for f in pose_frames if f.get("converged")}
    print(f"wall phase ends at frame {L}; continuing with VO")

    cap = cv2.VideoCapture(str(clip_p))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if L + 1 >= n:
        raise SystemExit("no post-wall frames; record a longer clip")

    da = None
    if not args.no_da:
        try:
            from depth_anything import DepthAnythingEstimator
            print("loading DepthAnything...")
            da = DepthAnythingEstimator()
        except Exception as e:
            print(f"DA unavailable ({e}); flow-only seeding")

    # ---- Seed map points at frame L ----
    pa = pose_by_idx[L]
    Ra = rotation_from_pose(pa)
    ta = np.array(pa["t"], dtype=np.float32)
    cam_pos_L = np.array(pa["camera_pos_world"], dtype=np.float32)

    # Prefer seeding from the existing metric reconstruction (pointcloud.ply)
    # projected into frame L -- robust even when frame L has no parallax.
    pc_path = event_dir / "pointcloud.ply"
    map_world = map_pix = None
    if pc_path.exists():
        world_pts = read_ply_points(pc_path)
        map_world, map_pix = project_world_to_pixels(world_pts, Ra, ta, K, W, H)
        print(f"seeded {len(map_world):,} map points from pointcloud.ply "
              f"projected into frame {L}")
    # Fallback: flow+DA depth at frame L (needs parallax)
    if map_world is None or len(map_world) < 50:
        Lm1 = max(i for i in tracked if i < L)
        pb = pose_by_idx[Lm1]
        Rb = rotation_from_pose(pb); tb = np.array(pb["t"], dtype=np.float32)
        R_rel = Rb @ Ra.T; t_rel = tb - R_rel @ ta
        d_seed = flow_depth(frames[L], frames[Lm1], R_rel, t_rel, K)
        if da is not None:
            d_seed = fuse_da(d_seed, da.estimate(frames[L]))
        map_world, map_pix = backproject(d_seed, K, Ra, cam_pos_L, args.grid_step)
        print(f"fallback seed: {len(map_world):,} map points from depth at frame {L}")
    if len(map_world) < 6:
        raise SystemExit("could not seed map points; run reconstruct-scene "
                         "first so pointcloud.ply exists")

    prev_gray = cv2.cvtColor(frames[L], cv2.COLOR_BGR2GRAY)
    seed_params = np.array([pa["yaw_rad"], pa["pitch_rad"],
                            pa["t"][0], pa["t"][1], pa["t"][2]])
    pose_history = [seed_params]      # for constant-velocity prediction
    last_da_scale = None              # carried-forward DA scale factor

    # Larger pyramid + window so LK survives bigger inter-frame motion.
    lk_params = dict(winSize=(args.lk_win, args.lk_win), maxLevel=args.lk_levels,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def predict_pose():
        if len(pose_history) >= 2:
            return pose_history[-1] + (pose_history[-1] - pose_history[-2])
        return pose_history[-1].copy()

    def reseed_from_da(frame, params, prefer_survivors=True):
        """Backproject DA depth at the given pose. Scale from surviving
        map points if possible, else from last_da_scale. Returns
        (world, pix) or (None, None)."""
        nonlocal last_da_scale
        if da is None:
            return None, None
        R = R_no_roll(params[0], params[1])
        t_vec = params[2:5]
        cam_pos = (-R.T @ t_vec).astype(np.float32)
        da_rel = da.estimate(frame)
        scale = None
        if prefer_survivors and len(map_world) >= 10:
            P_cam = map_world @ R.T + t_vec
            z_known = P_cam[:, 2]
            uu = np.clip(map_pix[:, 0].astype(int), 0, W - 1)
            vv = np.clip(map_pix[:, 1].astype(int), 0, H - 1)
            da_at = da_rel[vv, uu]
            ok_s = np.isfinite(da_at) & (da_at > 1e-6) & (z_known > 0)
            if ok_s.sum() >= 10:
                scale = float(np.median(z_known[ok_s] / da_at[ok_s]))
                last_da_scale = scale
        if scale is None:
            scale = last_da_scale
        if scale is None:
            return None, None
        d_new = da_rel * scale
        return backproject(d_new, K, R, cam_pos, args.grid_step)

    out_frames = []
    n_ok = n_fail = n_reseed = n_blind = 0
    consec_blind = 0
    t0 = time.perf_counter()
    end = n if args.max_frames is None else min(n, L + 1 + args.max_frames)
    for i in range(L + 1, end):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        pred = predict_pose()
        measured = None
        n_inliers = 0
        rms = float("nan")

        # --- Try LK + PnP for a measured pose ---
        if len(map_pix) > 0:
            new_pix, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray,
                np.ascontiguousarray(map_pix.reshape(-1, 1, 2)), None, **lk_params)
            if new_pix is not None:
                new_pix = new_pix.reshape(-1, 2)
                st = status.reshape(-1).astype(bool)
                inb = (new_pix[:, 0] >= 0) & (new_pix[:, 0] < W) & \
                      (new_pix[:, 1] >= 0) & (new_pix[:, 1] < H)
                keep = st & inb
                map_world = map_world[keep]
                map_pix = new_pix[keep]
                if len(map_world) >= 6:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        map_world.reshape(-1, 1, 3), map_pix.reshape(-1, 1, 2),
                        K, None, iterationsCount=200,
                        reprojectionError=args.pnp_reproj_px, confidence=0.999,
                        flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False)
                    if ok and inliers is not None and len(inliers) >= args.min_inliers:
                        R_cv, _ = cv2.Rodrigues(rvec)
                        t_cv = tvec.reshape(3)
                        # Decompose cv2's 6-DOF pose to no-roll (yaw, pitch);
                        # OW has no roll, so we just drop the roll component.
                        yaw0 = math.atan2(R_cv[0, 2], R_cv[2, 2])
                        pitch0 = math.asin(max(-1.0, min(1.0, -R_cv[1, 2])))
                        cv_params = np.array([yaw0, pitch0, t_cv[0], t_cv[1], t_cv[2]])
                        # Optional scipy refit; keep it only if it actually
                        # lowers the reprojection error (the LM can diverge to
                        # a degenerate tz~0 solution otherwise).
                        proj_cv, _ = project(map_world, yaw0, pitch0, cv_params[2:5], K)
                        rms_cv = float(np.sqrt(np.mean(
                            np.sum((proj_cv - map_pix) ** 2, axis=1))))
                        params, rms_re, ok_re = pnp_no_roll(
                            map_world, map_pix, K, cv_params)
                        if ok_re and rms_re < rms_cv:
                            rms = rms_re
                        else:
                            params, rms = cv_params, rms_cv
                        # prune high-residual points
                        proj, _ = project(map_world, params[0], params[1],
                                          params[2:5], K)
                        good = np.linalg.norm(proj - map_pix, axis=1) < args.prune_px
                        map_world = map_world[good]; map_pix = map_pix[good]
                        measured = params
                        n_inliers = int(len(inliers))

        if measured is not None:
            params = measured
            predicted_flag = False
            consec_blind = 0
            n_ok += 1
        else:
            # --- Blind recovery: use predicted pose, re-seed from DA ---
            params = pred
            predicted_flag = True
            consec_blind += 1
            n_blind += 1
            nw, npix = reseed_from_da(frames[i], params, prefer_survivors=False)
            if nw is not None:
                map_world, map_pix = nw, npix
                n_reseed += 1

        R = R_no_roll(params[0], params[1])
        cam_pos = (-R.T @ params[2:5])
        out_frames.append({
            "frame_idx": i,
            "converged": not predicted_flag,
            "predicted": bool(predicted_flag),
            "yaw_rad": float(params[0]), "pitch_rad": float(params[1]),
            "yaw_deg": float(math.degrees(params[0])),
            "pitch_deg": float(math.degrees(params[1])),
            "t": [float(params[2]), float(params[3]), float(params[4])],
            "camera_pos_world": cam_pos.tolist(),
            "n_tracked": int(len(map_world)),
            "n_inliers": n_inliers,
            "reproj_rms_px": float(rms) if not math.isnan(rms) else None,
        })
        pose_history.append(params)

        # --- Top up the map when running low (measured frames) ---
        if not predicted_flag and len(map_world) < args.min_track and da is not None:
            nw, npix = reseed_from_da(frames[i], params, prefer_survivors=True)
            if nw is not None:
                map_world = np.concatenate([map_world, nw], axis=0)
                map_pix = np.concatenate([map_pix, npix], axis=0)
                n_reseed += 1

        prev_gray = gray
        if consec_blind > args.max_blind_frames:
            print(f"  giving up at frame {i}: {consec_blind} consecutive "
                  f"blind frames (trajectory diverged)")
            break
        if (i - L) % 30 == 0:
            tag = "BLIND" if predicted_flag else f"inl={n_inliers}"
            print(f"  frame {i}: tracked={len(map_world)} {tag} "
                  f"rms={rms:.1f}px  ({time.perf_counter()-t0:.0f}s)")

    out = {
        "event": args.event,
        "last_wall_tracked_frame": L,
        "n_reseed": n_reseed,
        "frames": out_frames,
    }
    out["n_ok"] = n_ok
    out["n_blind"] = n_blind
    (event_dir / "continued_pose_vo.json").write_text(json.dumps(out, indent=2))
    print(f"\nDone. {n_ok} measured, {n_blind} blind(predicted), "
          f"{n_reseed} reseeds -> {event_dir / 'continued_pose_vo.json'}")
    good = [f for f in out_frames if f["converged"]]
    if good:
        rms = [f["reproj_rms_px"] for f in good if f["reproj_rms_px"] is not None]
        yaws = [f["yaw_deg"] for f in good]
        # full-trajectory path (measured + blind, in order)
        allf = [f for f in out_frames]
        cps = np.array([f["camera_pos_world"] for f in allf])
        path = float(np.linalg.norm(np.diff(cps, axis=0), axis=1).sum()) if len(cps) > 1 else 0
        print(f"median rms={np.median(rms):.1f}px (measured)  "
              f"yaw=({min(yaws):+.0f}..{max(yaws):+.0f})deg  "
              f"path={path:.1f} ow-m over {len(allf)} frames "
              f"({len(good)} measured)")


if __name__ == "__main__":
    main()

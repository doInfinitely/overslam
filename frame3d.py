#!/usr/bin/env python3
"""Single-frame 3D unprojection: flow depth vs DA vs fused.

For one frame of an event clip, compute three depth maps:
  - flow : Farneback flow + triangulation against the next frame
           (metric, sparse where parallax is low)
  - da   : DepthAnything relative depth, scaled to match flow in overlap
           (DA's median ratio to flow becomes the scale factor)
  - fused: flow where valid, scaled DA elsewhere

Each map is unprojected to world via the frame's pose. Writes three PLYs
to the event dir and opens a pygame viewer with F/A/U toggling between
them in-place (same orbit camera, so you can directly compare).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    raise SystemExit("pip install pygame")


# ---------- Geometry helpers ----------
def rotation_world_to_cam_np(yaw, pitch):
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw   = np.array([[ cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return R_pitch @ R_yaw


def relative_pose(pa, pb):
    Ra = rotation_world_to_cam_np(pa["yaw_rad"], pa["pitch_rad"])
    Rb = rotation_world_to_cam_np(pb["yaw_rad"], pb["pitch_rad"])
    ta = np.array(pa["t"], dtype=np.float32); tb = np.array(pb["t"], dtype=np.float32)
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


def flow_depth_map(fa, fb, R_rel, t_rel, K, min_flow_px=1.5, max_depth=30.0):
    """Triangulate per-pixel depth at frame_a from flow to frame_b."""
    H, W = fa.shape[:2]
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if float(np.linalg.norm(t_rel)) < 1e-3:
        return depth
    g1 = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None, pyr_scale=0.5, levels=3, winsize=21,
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
    pts1 = np.vstack([U[valid].astype(np.float32), V[valid].astype(np.float32)])
    pts2 = np.vstack([U2[valid].astype(np.float32), V2[valid].astype(np.float32)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    z = (pts4d[:3] / pts4d[3])[2]
    good = (z > 0.05) & (z < max_depth) & np.isfinite(z)
    z = np.where(good, z, np.nan)
    depth.reshape(-1)[(V[valid] * W + U[valid]).astype(np.int64)] = z.astype(np.float32)
    return depth


def da_scaled(da_rel, flow_d):
    """DA depth scaled to match flow in overlap regions. Returns DA-only
    map (NaN where DA invalid), in OW-m."""
    ok = np.isfinite(da_rel) & (da_rel > 1e-6) & np.isfinite(flow_d)
    if int(ok.sum()) < 200:
        return None  # not enough overlap
    s = float(np.median(flow_d[ok] / da_rel[ok]))
    out = da_rel * s
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out, s


def fuse(flow_d, da_d):
    """Flow where valid, DA elsewhere."""
    out = flow_d.copy()
    fill = ~np.isfinite(out) & np.isfinite(da_d) & (da_d > 0)
    out[fill] = da_d[fill]
    return out


def unproject_to_world(depth, bgr, K, R, cam_pos, subsample=4):
    """Per-pixel unproject to world coords. Returns (pts (N,3), colors (N,3) BGR uint8)."""
    H, W = depth.shape
    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    V, U = np.meshgrid(ys, xs, indexing="ij")
    D = depth[V, U]
    mask = np.isfinite(D) & (D > 0.05) & (D < 60.0)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (U - cx) * D / fx
    y_cam = (V - cy) * D / fy
    z_cam = D
    P_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    P_world = P_cam @ R + cam_pos
    return P_world[mask], bgr[V[mask], U[mask]]


def write_ply(path: Path, points, colors_bgr):
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (b, g, r) in zip(points, colors_bgr):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


# ---------- Orbit-camera pygame viewer ----------
class OrbitCamera:
    def __init__(self, target, distance):
        self.target = np.asarray(target, dtype=np.float32)
        self.distance = float(distance)
        self.yaw = 0.4
        self.pitch = 0.3
        self.fov_y_deg = 55.0

    def basis(self):
        cp = math.cos(self.pitch); sp = math.sin(self.pitch)
        cy = math.cos(self.yaw);   sy = math.sin(self.yaw)
        offset = np.array([cp * sy, -sp, -cp * cy], dtype=np.float32) * self.distance
        pos = self.target + offset
        fwd = self.target - pos
        fwd /= np.linalg.norm(fwd) + 1e-9
        up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right /= np.linalg.norm(right)
        cam_down = np.cross(fwd, right)
        R = np.stack([right, cam_down, fwd], axis=0)
        return pos, R


def project_points(P_world, pos, R, fx, fy, cx, cy):
    pc = (P_world - pos) @ R.T
    z = pc[:, 2]
    u = fx * pc[:, 0] / np.maximum(z, 1e-3) + cx
    v = fy * pc[:, 1] / np.maximum(z, 1e-3) + cy
    return np.stack([u, v], axis=-1), z


def blit_points(surface, uv, z, colors_bgr, W, H):
    mask = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not mask.any():
        return 0
    px = uv[mask, 0].astype(np.int32)
    py = uv[mask, 1].astype(np.int32)
    cols = colors_bgr[mask]
    order = np.argsort(-z[mask])
    px = px[order]; py = py[order]; cols = cols[order]
    pix = pygame.surfarray.pixels3d(surface)
    pix[px, py, 0] = cols[:, 2]  # R from BGR
    pix[px, py, 1] = cols[:, 1]
    pix[px, py, 2] = cols[:, 0]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = np.clip(px + dx, 0, W - 1)
        ny = np.clip(py + dy, 0, H - 1)
        pix[nx, ny, 0] = cols[:, 2]
        pix[nx, ny, 1] = cols[:, 1]
        pix[nx, ny, 2] = cols[:, 0]
    del pix
    return int(mask.sum())


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", required=True)
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index in the clip (default: middle of the clip)")
    ap.add_argument("--subsample", type=int, default=2,
                    help="image pixel stride when unprojecting (default 2)")
    ap.add_argument("--min-flow-px", type=float, default=1.5)
    ap.add_argument("--max-depth",  type=float, default=30.0)
    ap.add_argument("--save-only", action="store_true",
                    help="write PLYs and exit, no viewer")
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
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)

    cap = cv2.VideoCapture(str(clip_p))
    raw = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        raw.append(fr)
    cap.release()
    n = min(len(raw), len(pose_frames))
    if n < 2:
        raise SystemExit("clip too short")

    # Pick frame
    i = args.frame if args.frame is not None else n // 2
    if not (0 <= i < n - 1):
        raise SystemExit(f"frame {i} out of range [0, {n - 2}]")
    pa = pose_frames[i]
    pb = pose_frames[i + 1]
    if not pa.get("converged"):
        raise SystemExit(f"frame {i} pose not converged")
    R_rel, t_rel = relative_pose(pa, pb)
    print(f"frame {i}: rel translation = {np.linalg.norm(t_rel):.3f} ow-m  "
          f"-- yaw={math.degrees(pa['yaw_rad']):.1f}deg, "
          f"pitch={math.degrees(pa['pitch_rad']):.1f}deg")

    # Flow depth (full res)
    print("computing flow depth...")
    flow_d = flow_depth_map(raw[i], raw[i + 1], R_rel, t_rel, K,
                            min_flow_px=args.min_flow_px,
                            max_depth=args.max_depth)
    n_flow = int(np.isfinite(flow_d).sum())
    print(f"  flow valid: {n_flow:,} px ({n_flow * 100.0 / (H * W):.1f}%)")

    # DA depth
    print("loading DepthAnything...")
    sys.path.insert(0, os.path.expanduser("~/turntable"))
    try:
        from depth_anything import DepthAnythingEstimator
        da_est = DepthAnythingEstimator()
        print("estimating DA...")
        da_rel = da_est.estimate(raw[i])
    except Exception as e:
        print(f"DA failed: {e}")
        da_rel = None

    da_d, da_scale = None, None
    if da_rel is not None:
        res = da_scaled(da_rel, flow_d)
        if res is not None:
            da_d, da_scale = res
            print(f"  DA scale (flow / da median): {da_scale:.3f}")

    fused_d = None
    if da_d is not None:
        fused_d = fuse(flow_d, da_d)
        n_fused = int(np.isfinite(fused_d).sum())
        print(f"  fused valid: {n_fused:,} px ({n_fused * 100.0 / (H * W):.1f}%)")

    # Unproject each mode
    R = rotation_world_to_cam_np(pa["yaw_rad"], pa["pitch_rad"])
    cam_pos = np.array(pa["camera_pos_world"], dtype=np.float32)

    sets = {}  # mode -> (points, colors)
    for mode, d in (("flow", flow_d), ("da", da_d), ("fused", fused_d)):
        if d is None:
            continue
        pts, cols = unproject_to_world(d, raw[i], K, R, cam_pos,
                                        subsample=args.subsample)
        sets[mode] = (pts, cols)
        write_ply(event_dir / f"frame{i:04d}_{mode}.ply", pts, cols)
        print(f"  wrote {mode}: {len(pts):,} pts -> frame{i:04d}_{mode}.ply")

    if args.save_only or not sets:
        return

    # Viewer
    pygame.init()
    pygame.display.set_caption(
        f"frame3d frame {i} -- F=flow A=da U=fused B=bg Esc=quit"
    )
    SCR_W, SCR_H = 1200, 720
    screen = pygame.display.set_mode((SCR_W, SCR_H))
    font = pygame.font.SysFont("monospace", 14)
    fy_proj = (SCR_H / 2) / math.tan(math.radians(55.0) / 2)
    fx_proj = fy_proj
    pcx, pcy = SCR_W / 2, SCR_H / 2

    # Initial camera = centroid of first available set
    first_mode = "fused" if "fused" in sets else ("flow" if "flow" in sets else "da")
    centroid = sets[first_mode][0].mean(axis=0)
    spread = float(np.percentile(np.linalg.norm(sets[first_mode][0] - centroid, axis=1), 75)) * 2.5
    cam = OrbitCamera(target=centroid, distance=max(spread, 1.0))

    mode = first_mode
    backgrounds = [(20, 20, 24), (130, 130, 130), (240, 240, 240), (0, 0, 0)]
    bg_idx = 0
    dragging = False
    panning = False
    last_pos = (0, 0)
    clock = pygame.time.Clock()

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_f and "flow" in sets:
                    mode = "flow"
                elif ev.key == pygame.K_a and "da" in sets:
                    mode = "da"
                elif ev.key == pygame.K_u and "fused" in sets:
                    mode = "fused"
                elif ev.key == pygame.K_b:
                    bg_idx = (bg_idx + 1) % len(backgrounds)
                elif ev.key == pygame.K_r:
                    cam.target = sets[mode][0].mean(axis=0)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        panning = True
                    else:
                        dragging = True
                    last_pos = ev.pos
                elif ev.button == 4:
                    cam.distance *= 0.85
                elif ev.button == 5:
                    cam.distance *= 1.18
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False; panning = False
            elif ev.type == pygame.MOUSEMOTION:
                dx = ev.pos[0] - last_pos[0]
                dy = ev.pos[1] - last_pos[1]
                if dragging:
                    cam.yaw   += dx * 0.008
                    cam.pitch = float(np.clip(cam.pitch + dy * 0.008, -1.4, 1.4))
                if panning:
                    pos, Rcam = cam.basis()
                    speed = cam.distance * 0.0015
                    cam.target -= Rcam[0] * dx * speed
                    cam.target -= Rcam[1] * dy * speed
                last_pos = ev.pos

        screen.fill(backgrounds[bg_idx])
        pos, R_cam = cam.basis()
        pts, cols = sets[mode]
        uv, z = project_points(pts, pos, R_cam, fx_proj, fy_proj, pcx, pcy)
        drawn = blit_points(screen, uv, z, cols, SCR_W, SCR_H)

        hud = [
            f"frame {i}   mode: {mode}   points: {len(pts):,}   drawn: {drawn:,}",
            f"cam dist={cam.distance:.2f} yaw={math.degrees(cam.yaw):.0f} "
            f"pitch={math.degrees(cam.pitch):.0f}",
            f"F=flow ({len(sets.get('flow', ([],))[0]):,})   "
            f"A=da ({len(sets.get('da', ([],))[0]):,})   "
            f"U=fused ({len(sets.get('fused', ([],))[0]):,})",
            "drag=orbit shift+drag=pan scroll=zoom R=recenter B=bg Esc=quit",
        ]
        for j, t in enumerate(hud):
            screen.blit(font.render(t, True, (235, 235, 235)), (8, 6 + j * 16))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

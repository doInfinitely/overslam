#!/usr/bin/env python3
"""Progressive TSDF voxel mesh reconstructor (live pygame).

Same TSDF + marching-cubes pipeline as before, but integrates one frame
at a time and re-renders the current surface in a pygame window so you
can watch the scene materialize. Orbit camera works while integration
runs.

Controls
--------
  drag                : orbit
  shift+drag          : pan
  scroll              : zoom
  Space               : pause/resume integration
  N                   : step one frame (when paused)
  R                   : recenter on observed surface centroid
  X                   : toggle world-axes overlay
  M                   : run marching cubes now (use after integration)
  S                   : save current mesh.ply
  Esc                 : quit

The voxel grid bbox is the 5-95 percentile of the existing pointcloud
(robust to outliers) plus padding; if the resulting grid exceeds the
budget, voxel-size is auto-bumped.
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
    import pygame
except ImportError:
    raise SystemExit("pip install pygame")

try:
    from skimage.measure import marching_cubes
except ImportError:
    raise SystemExit("pip install scikit-image")


# ---------- Geometry / depth helpers ----------
def rotation_world_to_cam(yaw, pitch):
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw   = np.array([[ cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_pitch = np.array([[1, 0, 0],    [0, cp, -sp], [0, sp, cp]])
    return R_pitch @ R_yaw


def relative_pose(pa, pb):
    Ra = rotation_world_to_cam(pa["yaw_rad"], pa["pitch_rad"])
    Rb = rotation_world_to_cam(pb["yaw_rad"], pb["pitch_rad"])
    ta = np.array(pa["t"]); tb = np.array(pb["t"])
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


def flow_depth_map(fa, fb, pa, pb, K, min_flow_px=1.5, max_depth=30.0):
    H, W = fa.shape[:2]
    R_rel, t_rel = relative_pose(pa, pb)
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


def read_ply_points(path: Path):
    with open(path, "r") as f:
        line = f.readline()
        if not line.startswith("ply"):
            raise SystemExit(f"{path} is not ply")
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


def integrate_frame(sdf, weight, color, voxels_world, frame_bgr, depth_m,
                    yaw, pitch, cam_pos, K, mu, max_depth):
    H, W = depth_m.shape
    R = rotation_world_to_cam(yaw, pitch)
    P_cam = (voxels_world - cam_pos) @ R.T
    z = P_cam[:, 2]
    in_front = z > 0.05
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * P_cam[:, 0] / np.maximum(z, 1e-3) + cx)
    v = (fy * P_cam[:, 1] / np.maximum(z, 1e-3) + cy)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front
    if not in_img.any():
        return
    indices_in = np.where(in_img)[0]
    ui = u[in_img].astype(np.int32)
    vi = v[in_img].astype(np.int32)
    d_obs = depth_m[vi, ui]
    finite = np.isfinite(d_obs) & (d_obs > 0) & (d_obs < max_depth)
    if not finite.any():
        return
    valid_idx = indices_in[finite]
    d_obs = d_obs[finite]
    z_v = z[valid_idx]
    sd = d_obs - z_v
    in_band = np.abs(sd) <= mu
    if not in_band.any():
        return
    valid_idx = valid_idx[in_band]
    sd = sd[in_band]
    ui = ui[finite][in_band]
    vi = vi[finite][in_band]
    tsdf = np.clip(sd / mu, -1.0, 1.0).astype(np.float32)
    rgb = frame_bgr[vi, ui].astype(np.float32)

    w_old = weight[valid_idx]
    w_new = w_old + 1.0
    sdf[valid_idx] = (sdf[valid_idx] * w_old + tsdf) / w_new
    color[valid_idx] = (color[valid_idx] * w_old[:, None] + rgb) / w_new[:, None]
    weight[valid_idx] = w_new


# ---------- pygame orbit camera + projection ----------
class OrbitCamera:
    def __init__(self, target, distance):
        self.target = np.array(target, dtype=float)
        self.distance = float(distance)
        self.yaw = 0.5
        self.pitch = 0.3

    def basis(self):
        cp = math.cos(self.pitch); sp = math.sin(self.pitch)
        cy = math.cos(self.yaw);   sy = math.sin(self.yaw)
        # World Y is DOWN (OpenCV) -> our "up" in world is -Y.
        # Negative Z so we start on the camera-at-placement side: +X
        # in world (right at placement) maps to screen right.
        offset = np.array([cp * sy, -sp, -cp * cy]) * self.distance
        pos = self.target + offset
        fwd = self.target - pos
        fwd /= np.linalg.norm(fwd) + 1e-9
        up_world = np.array([0.0, -1.0, 0.0])
        right = np.cross(fwd, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        cam_down = np.cross(fwd, right)
        R = np.stack([right, cam_down, fwd], axis=0)
        return pos, R


def project_points(P_world, pos, R, fx, fy, cx, cy):
    """Vectorized projection of (N, 3) world points to screen + z.
    Returns (uv (N, 2) float32, z (N,) float32). Caller filters."""
    pc = (P_world - pos) @ R.T
    z = pc[:, 2]
    u = fx * pc[:, 0] / np.maximum(z, 1e-3) + cx
    v = fy * pc[:, 1] / np.maximum(z, 1e-3) + cy
    return np.stack([u, v], axis=-1), z


def blit_points(surface, uv, z, colors_u8, W, H):
    """Splat a (N, 2) uv array as 1-pixel points onto surface."""
    if len(uv) == 0:
        return 0
    mask = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not mask.any():
        return 0
    px = uv[mask, 0].astype(np.int32)
    py = uv[mask, 1].astype(np.int32)
    cols = colors_u8[mask]
    # Z-buffer-like back-to-front draw: sort by descending z
    order = np.argsort(-z[mask])
    px = px[order]; py = py[order]; cols = cols[order]
    pix = pygame.surfarray.pixels3d(surface)
    # Pygame's surfarray is (W, H, 3) RGB; our cols are BGR -> swap
    pix[px, py, 0] = cols[:, 2]  # R
    pix[px, py, 1] = cols[:, 1]  # G
    pix[px, py, 2] = cols[:, 0]  # B
    # Also splat 2-pixel neighborhood for density on small clouds
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = np.clip(px + dx, 0, W - 1)
        ny = np.clip(py + dy, 0, H - 1)
        pix[nx, ny, 0] = cols[:, 2]
        pix[nx, ny, 1] = cols[:, 1]
        pix[nx, ny, 2] = cols[:, 0]
    del pix
    return int(mask.sum())


def draw_axes(surface, target, pos, R, fx, fy, cx, cy, length=0.5):
    for axis, color in [
        (np.array([length, 0, 0]), (220, 60, 60)),
        (np.array([0, length, 0]), (60, 220, 60)),
        (np.array([0, 0, length]), (60, 90, 220)),
    ]:
        pts = np.stack([target, target + axis], axis=0)
        uv, z = project_points(pts, pos, R, fx, fy, cx, cy)
        if z[0] <= 0.1 or z[1] <= 0.1:
            continue
        pygame.draw.line(surface, color,
                         (int(uv[0, 0]), int(uv[0, 1])),
                         (int(uv[1, 0]), int(uv[1, 1])), 2)


# ---------- Run a single event progressively ----------
def voxelize_event_live(event_dir, args, da):
    pose_p  = event_dir / "pose.json"
    clip_p  = event_dir / "clip.mp4"
    ply_p   = event_dir / "pointcloud.ply"
    if not pose_p.exists() or not clip_p.exists():
        return None, "need pose.json and clip.mp4"
    if not ply_p.exists():
        return None, "need pointcloud.ply (run reconstruct-scene first)"
    pose = json.loads(pose_p.read_text())
    pose_frames = pose["frames"]
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]])

    # Robust bbox: percentile of pointcloud points (outliers blow up min/max).
    pts = read_ply_points(ply_p)
    lo = np.percentile(pts, args.bbox_pct_lo, axis=0) - args.bbox_pad
    hi = np.percentile(pts, args.bbox_pct_hi, axis=0) + args.bbox_pad
    print(f"  bbox p{args.bbox_pct_lo:g}-p{args.bbox_pct_hi:g}: "
          f"{lo} .. {hi}")

    # Auto-bump voxel size until grid fits the budget
    vsz = args.voxel_size
    extent = hi - lo
    while True:
        nx, ny, nz = (int(np.ceil(e / vsz)) for e in extent)
        n_total = nx * ny * nz
        if n_total <= args.max_voxels:
            break
        vsz *= 1.25
    if vsz != args.voxel_size:
        print(f"  voxel size auto-bumped {args.voxel_size:.3f} -> {vsz:.3f} ow-m "
              f"to fit budget")
    print(f"  voxel grid: ({nx}, {ny}, {nz})  ({n_total:,} voxels)")

    xs = lo[0] + (np.arange(nx) + 0.5) * vsz
    ys = lo[1] + (np.arange(ny) + 0.5) * vsz
    zs = lo[2] + (np.arange(nz) + 0.5) * vsz
    Y, X, Z = np.meshgrid(ys, xs, zs, indexing="ij")  # broadcasting order
    voxels_world = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    sdf    = np.ones(n_total, dtype=np.float32)
    weight = np.zeros(n_total, dtype=np.float32)
    color  = np.zeros((n_total, 3), dtype=np.float32)
    mu = vsz * args.trunc_voxels

    # Pre-read frames
    cap = cv2.VideoCapture(str(clip_p))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    n = min(len(frames), len(pose_frames))

    # pygame
    pygame.init()
    pygame.display.set_caption(
        f"voxelize-live -- {event_dir.name}  (space pause, M mc, S save, Esc quit)"
    )
    SCR_W, SCR_H = 1280, 720
    screen = pygame.display.set_mode((SCR_W, SCR_H))
    font = pygame.font.SysFont("monospace", 14)
    fy_proj = (SCR_H / 2) / math.tan(math.radians(60.0) / 2)
    fx_proj = fy_proj
    pcx, pcy = SCR_W / 2, SCR_H / 2
    centroid = (lo + hi) / 2
    cam = OrbitCamera(target=centroid, distance=max(extent) * 1.4)

    frame_idx = 0
    paused = False
    show_axes = True
    dragging = False
    panning = False
    last_mouse = (0, 0)
    mesh_verts = None
    mesh_faces = None
    integ_time_total = 0.0
    last_msg = "starting..."
    surf_band_cycle = (0.04, 0.08, 0.15, 0.30, 0.50)
    surf_band = args.surface_band
    color_by_sdf = False

    clock = pygame.time.Clock()

    def do_mc():
        sdf3 = sdf.reshape((nx, ny, nz))
        weight3 = weight.reshape((nx, ny, nz))
        sdf3 = np.where(weight3 >= args.min_weight, sdf3, 1.0).astype(np.float32)
        try:
            verts, faces, _, _ = marching_cubes(
                sdf3, level=0.0, spacing=(vsz,) * 3, allow_degenerate=False,
            )
            verts_w = verts + lo.astype(np.float32)
            return verts_w, faces.astype(np.int32)
        except Exception as e:
            return None, str(e)

    def save_mesh():
        if mesh_verts is None:
            return "no mesh yet (press M)"
        # per-vertex color via trilinear
        color3 = color.reshape((nx, ny, nz, 3))
        gp = (mesh_verts - lo) / vsz - 0.5
        gp_c = np.clip(gp, 0, np.array([nx, ny, nz]) - 1.0001)
        g0 = np.floor(gp_c).astype(np.int32)
        gf = gp_c - g0
        vc = np.zeros((len(mesh_verts), 3), dtype=np.float32)
        for dxd in (0, 1):
            for dyd in (0, 1):
                for dzd in (0, 1):
                    wgt = ((dxd * gf[:, 0] + (1 - dxd) * (1 - gf[:, 0]))
                           * (dyd * gf[:, 1] + (1 - dyd) * (1 - gf[:, 1]))
                           * (dzd * gf[:, 2] + (1 - dzd) * (1 - gf[:, 2])))
                    ix = np.clip(g0[:, 0] + dxd, 0, nx - 1)
                    iy = np.clip(g0[:, 1] + dyd, 0, ny - 1)
                    iz = np.clip(g0[:, 2] + dzd, 0, nz - 1)
                    vc += wgt[:, None] * color3[ix, iy, iz]
        vc = np.clip(vc, 0, 255).astype(np.uint8)
        out = event_dir / "mesh.ply"
        with open(out, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(mesh_verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element face {len(mesh_faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for (x, y, z), (b, g, r) in zip(mesh_verts, vc):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            for fc in mesh_faces:
                f.write(f"3 {int(fc[0])} {int(fc[1])} {int(fc[2])}\n")
        return f"saved {len(mesh_verts)} verts -> {out}"

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_n:
                    if paused and frame_idx < n - 1:
                        paused = False
                        # advance 1 then re-pause -- handled below by clamping
                        single_step = True
                elif ev.key == pygame.K_r:
                    surf_mask = (weight > 0) & (np.abs(sdf) < 0.3)
                    if surf_mask.any():
                        cam.target = voxels_world[surf_mask].mean(axis=0)
                elif ev.key == pygame.K_x:
                    show_axes = not show_axes
                elif ev.key == pygame.K_m:
                    last_msg = "running marching cubes..."
                    pygame.display.set_caption(last_msg)
                    mesh_verts, mesh_faces = do_mc()
                    if isinstance(mesh_faces, str):
                        last_msg = f"mc failed: {mesh_faces}"
                        mesh_verts = mesh_faces = None
                    else:
                        last_msg = f"mesh: {len(mesh_verts)} verts, {len(mesh_faces)} faces"
                elif ev.key == pygame.K_s:
                    last_msg = save_mesh()
                elif ev.key == pygame.K_t:
                    # Cycle to the next looser threshold
                    cur_i = min(range(len(surf_band_cycle)),
                                key=lambda j: abs(surf_band_cycle[j] - surf_band))
                    surf_band = surf_band_cycle[(cur_i + 1) % len(surf_band_cycle)]
                    last_msg = f"surface_band -> {surf_band}"
                elif ev.key == pygame.K_c:
                    color_by_sdf = not color_by_sdf
                    last_msg = f"color by sdf: {color_by_sdf}"
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        panning = True
                    else:
                        dragging = True
                    last_mouse = ev.pos
                elif ev.button == 4:
                    cam.distance *= 0.85
                elif ev.button == 5:
                    cam.distance *= 1.18
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False; panning = False
            elif ev.type == pygame.MOUSEMOTION:
                dx = ev.pos[0] - last_mouse[0]
                dy = ev.pos[1] - last_mouse[1]
                if dragging:
                    cam.yaw   += dx * 0.008
                    cam.pitch = float(np.clip(cam.pitch + dy * 0.008, -1.4, 1.4))
                if panning:
                    pos, R = cam.basis()
                    speed = cam.distance * 0.0015
                    cam.target -= R[0] * dx * speed
                    cam.target -= R[1] * dy * speed
                last_mouse = ev.pos

        # Integrate next frame if not paused
        if not paused and frame_idx < n - 1:
            pa = pose_frames[frame_idx]; pb = pose_frames[frame_idx + 1]
            if (pa.get("converged") and pb.get("converged")
                    and pa.get("reproj_rms_px", 1e9) <= args.max_rms_px
                    and pb.get("reproj_rms_px", 1e9) <= args.max_rms_px):
                t0 = time.perf_counter()
                depth = flow_depth_map(
                    frames[frame_idx], frames[frame_idx + 1], pa, pb, K,
                    min_flow_px=args.min_flow_px, max_depth=args.max_depth,
                )
                if da is not None:
                    da_rel = da.estimate(frames[frame_idx])
                    ok = np.isfinite(depth) & np.isfinite(da_rel) & (da_rel > 1e-6)
                    if ok.sum() > 200:
                        s = float(np.median(depth[ok] / da_rel[ok]))
                        da_depth = da_rel * s
                        fill = ~np.isfinite(depth) & np.isfinite(da_depth) & (da_depth > 0)
                        depth[fill] = da_depth[fill]
                cam_pos = np.array(pa["camera_pos_world"], dtype=np.float32)
                integrate_frame(
                    sdf, weight, color, voxels_world,
                    frames[frame_idx], depth,
                    pa["yaw_rad"], pa["pitch_rad"], cam_pos, K,
                    mu=mu, max_depth=args.max_depth,
                )
                integ_time_total += time.perf_counter() - t0
            frame_idx += 1

        # Render: extract surface voxels and project
        screen.fill((18, 18, 22))
        pos, R = cam.basis()
        if show_axes:
            draw_axes(screen, cam.target, pos, R, fx_proj, fy_proj, pcx, pcy,
                      length=0.5 * vsz * 10)
        # Surface voxels = small |sdf| AND multi-observation
        surf_mask = ((weight >= args.render_min_weight)
                     & (np.abs(sdf) < surf_band))
        n_surf = int(surf_mask.sum())
        drawn = 0
        if n_surf > 0:
            pts_w = voxels_world[surf_mask]
            if color_by_sdf:
                # red = positive sdf (in front of surface), blue = negative (behind)
                s = sdf[surf_mask]
                t = np.clip((s + 1) * 127.5, 0, 255).astype(np.uint8)  # 0..255
                # Build BGR: more positive sdf -> warmer (red), negative -> cooler (blue)
                cols = np.zeros((n_surf, 3), dtype=np.uint8)
                cols[:, 0] = 255 - t       # B
                cols[:, 1] = 80            # G
                cols[:, 2] = t             # R
            else:
                cols = np.clip(color[surf_mask], 0, 255).astype(np.uint8)
            # Subsample if too many for the framebuffer to handle smoothly
            if n_surf > args.max_render_points:
                idx = np.random.choice(n_surf, args.max_render_points, replace=False)
                pts_w = pts_w[idx]; cols = cols[idx]
            uv, zc = project_points(pts_w, pos, R, fx_proj, fy_proj, pcx, pcy)
            drawn = blit_points(screen, uv, zc, cols, SCR_W, SCR_H)

        # Mesh overlay (if computed)
        if mesh_verts is not None:
            uv_m, zm = project_points(mesh_verts, pos, R, fx_proj, fy_proj, pcx, pcy)
            # Draw faces as wireframe (lines) -- can be slow for big meshes
            if len(mesh_faces) < args.mesh_max_faces:
                vis = (zm > 0.1) & (uv_m[:, 0] >= 0) & (uv_m[:, 0] < SCR_W) \
                      & (uv_m[:, 1] >= 0) & (uv_m[:, 1] < SCR_H)
                for tri in mesh_faces:
                    a, b, c = tri
                    if vis[a] and vis[b] and vis[c]:
                        pygame.draw.polygon(
                            screen, (200, 220, 200),
                            [(int(uv_m[a, 0]), int(uv_m[a, 1])),
                             (int(uv_m[b, 0]), int(uv_m[b, 1])),
                             (int(uv_m[c, 0]), int(uv_m[c, 1]))],
                            1,
                        )

        # HUD
        lines = [
            f"frame {frame_idx} / {n - 1}   "
            f"{'PAUSED' if paused else 'running'}   "
            f"surface voxels: {n_surf:,}   drawn: {drawn:,}",
            f"voxel size: {vsz:.3f} ow-m   grid: {nx}x{ny}x{nz}   "
            f"integ total: {integ_time_total:.1f}s",
            f"cam dist={cam.distance:.2f}  yaw={math.degrees(cam.yaw):.0f}  "
            f"pitch={math.degrees(cam.pitch):.0f}",
            f"surf_band={surf_band:.2f} (T cycle) "
            f"min_w={args.render_min_weight}  color_by_sdf={color_by_sdf} (C)",
            last_msg,
            "Space=pause N=step R=recenter X=axes M=mc S=save Esc=quit",
        ]
        for i, t in enumerate(lines):
            screen.blit(font.render(t, True, (230, 230, 230)), (8, 6 + i * 16))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return {
        "event": event_dir.name,
        "n_voxels": int(n_total),
        "frames_integrated": int(frame_idx),
        "integ_total_s": integ_time_total,
        "mesh_saved": mesh_verts is not None,
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", required=True)
    ap.add_argument("--voxel-size", type=float, default=0.05,
                    help="initial voxel edge (ow-m). Auto-bumped if budget "
                         "exceeded. Default 0.05.")
    ap.add_argument("--max-voxels", type=int, default=12_000_000)
    ap.add_argument("--trunc-voxels", type=float, default=4.0,
                    help="TSDF truncation in voxel widths (default 4)")
    ap.add_argument("--surface-band", type=float, default=0.10,
                    help="|sdf| threshold for live-rendered surface points "
                         "(default 0.10). Press T at runtime to cycle "
                         "through (0.04, 0.08, 0.15, 0.30, 0.50). Smaller "
                         "= thinner surface but sparser display.")
    ap.add_argument("--render-min-weight", type=float, default=3.0,
                    help="don't render surface voxels with fewer than this "
                         "many observations (default 3). Filters single-"
                         "frame noise from the live view.")
    ap.add_argument("--min-weight", type=float, default=2.0,
                    help="ignore voxels with fewer than this obs in marching "
                         "cubes (default 2)")
    ap.add_argument("--bbox-pct-lo", type=float, default=2.0)
    ap.add_argument("--bbox-pct-hi", type=float, default=98.0)
    ap.add_argument("--bbox-pad", type=float, default=0.3)
    ap.add_argument("--max-rms-px", type=float, default=100.0)
    ap.add_argument("--min-flow-px", type=float, default=1.5)
    ap.add_argument("--max-depth",   type=float, default=30.0)
    ap.add_argument("--max-render-points", type=int, default=120_000)
    ap.add_argument("--mesh-max-faces", type=int, default=80_000,
                    help="don't render the mesh-wireframe overlay if it has "
                         "more faces than this (default 80k)")
    ap.add_argument("--no-da", action="store_true",
                    help="skip DepthAnything (flow-only depth)")
    args = ap.parse_args()

    da = None
    if not args.no_da:
        sys.path.insert(0, os.path.expanduser("~/turntable"))
        print("Loading DepthAnything...")
        try:
            from depth_anything import DepthAnythingEstimator
            da = DepthAnythingEstimator()
            print("DA ready.")
        except Exception as e:
            print(f"DA failed ({e}); flow-only depth.")

    event_dir = Path(args.events_dir) / args.event
    print(f"Voxelize-live: {event_dir.name}")
    rec, err = voxelize_event_live(event_dir, args, da)
    if err:
        print(f"  [skip] {err}")
        return
    print(f"  done. {rec['frames_integrated']} frames integrated, "
          f"{rec['integ_total_s']:.1f}s total integration.")


if __name__ == "__main__":
    main()

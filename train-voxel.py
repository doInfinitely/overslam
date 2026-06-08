#!/usr/bin/env python3
"""Differentiable voxel-volume training (turntable-style) for an event.

Pipeline (mirrors turntable/video_orbit_voxel_recon.py):
  1. Load clip frames + per-frame poses for an event.
  2. Resize frames to a small training resolution (rays are expensive).
  3. Compute scene center + radius from pointcloud.ply (so world coords
     map to grid_sample's [-1, 1]^3 cleanly).
  4. VoxelVolume(grid_size^3): learnable per-voxel (sigma, rgb).
  5. For each iteration:
        - Pick a subset of views.
        - Generate rays for each view, sample volume along them.
        - Alpha-composite -> rendered RGB.
        - MSE vs downsampled GT + TV smoothness reg.
        - Adam step.
  6. Periodic snapshot: render a few held-out views into <out>/snapshots/.
  7. Final: save volume.npz, sigma>thresh voxels as colored PLY.

Differences from turntable:
  - Per-frame poses (no orbit assumption).
  - Scene center/radius derived from our pointcloud, not fixed.
  - Image resize keeps the OpenCV (Y-down) intrinsics consistent.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("pip install torch")


# ---------- Geometry helpers (our project conventions) ----------
def rotation_world_to_cam(yaw, pitch, device="cpu"):
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw   = torch.tensor([[ cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
                            dtype=torch.float32, device=device)
    R_pitch = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]],
                            dtype=torch.float32, device=device)
    return R_pitch @ R_yaw


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


# ---------- VoxelVolume (adapted from turntable) ----------
class VoxelVolume(nn.Module):
    def __init__(self, grid_size=64, init_density_logit=-5.0,
                 sigma_scale=20.0):
        super().__init__()
        self.grid_size = grid_size
        self.sigma_scale = sigma_scale
        self.density = nn.Parameter(
            torch.full((1, 1, grid_size, grid_size, grid_size),
                       init_density_logit, dtype=torch.float32)
        )
        # color logits via sigmoid; init around 0 -> mid-gray
        self.color = nn.Parameter(
            torch.full((1, 3, grid_size, grid_size, grid_size),
                       0.0, dtype=torch.float32)
        )
        # Free-space suppression: 1.0 where a voxel was observed in free
        # space by enough views (cull step sets this). 0.0 elsewhere.
        self.register_buffer(
            "suppress_mask",
            torch.zeros((1, 1, grid_size, grid_size, grid_size),
                        dtype=torch.float32),
        )

    def forward(self):
        sigma = F.softplus(self.density) * self.sigma_scale
        sigma = sigma * (1.0 - self.suppress_mask)
        rgb   = torch.sigmoid(self.color)
        return sigma, rgb


def compute_suppress_mask(voxel_centers_world, poses_shifted, depth_zcam_maps,
                          K_train, image_size, min_free_frac=0.30,
                          min_visible=3, margin=0.1, device="cuda"):
    """For each voxel, count how often it sits in observed free space
    (z_cam < observed_depth - margin) vs how often it's visible at all.
    Voxels with free_frac >= min_free_frac across >= min_visible views
    get marked for suppression.

    voxel_centers_world : (N, 3) in the shifted world frame used during training
    poses_shifted        : list of (R, t) with t already shifted to scene center
    depth_zcam_maps     : list of (Ht, Wt) tensors with z_cam (NaN where invalid)
    K_train             : (3, 3) intrinsics at training resolution
    image_size          : (Ht, Wt)
    """
    Ht, Wt = image_size
    n_v = voxel_centers_world.shape[0]
    free_count = torch.zeros(n_v, device=device, dtype=torch.float32)
    visible_count = torch.zeros(n_v, device=device, dtype=torch.float32)
    fx, fy = K_train[0, 0], K_train[1, 1]
    cx, cy = K_train[0, 2], K_train[1, 2]

    for (R, t), d_map in zip(poses_shifted, depth_zcam_maps):
        P_cam = voxel_centers_world @ R.T + t
        z = P_cam[:, 2]
        in_front = z > 0.05
        u = fx * P_cam[:, 0] / torch.clamp(z, min=1e-3) + cx
        v = fy * P_cam[:, 1] / torch.clamp(z, min=1e-3) + cy
        in_img = (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht) & in_front
        if not in_img.any():
            continue
        ui = u.long().clamp(0, Wt - 1)
        vi = v.long().clamp(0, Ht - 1)
        d_obs = d_map[vi, ui]
        valid_depth = torch.isfinite(d_obs) & in_img & (d_obs > 0)
        is_free = valid_depth & (z < (d_obs - margin))
        free_count = free_count + is_free.float()
        visible_count = visible_count + valid_depth.float()

    free_frac = free_count / torch.clamp(visible_count, min=1.0)
    suppress = (visible_count >= float(min_visible)) & (free_frac >= min_free_frac)
    return suppress, free_count, visible_count


# ---------- Ray generation / sampling (adapted from turntable) ----------
def world_to_grid(pts_world, center, radius):
    # Map [center - radius, center + radius] -> [-1, 1]
    return (pts_world - center) / radius


def generate_rays(H, W, K_inv, R, t, n_samples, near, far,
                  scene_center, device):
    """Generate (1, S, H, W, 3) ray samples in world coords.
    K_inv: (3, 3); R world->cam (3, 3); t world->cam (3,).
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # (H, W, 3)
    dirs_cam = (K_inv @ pix.reshape(-1, 3).T).T               # (H*W, 3)
    dirs_cam = dirs_cam / (dirs_cam.norm(dim=-1, keepdim=True) + 1e-9)
    dirs_world = (R.T @ dirs_cam.T).T                          # (H*W, 3)
    cam_pos = (-R.T @ t).reshape(1, 3)
    ts = torch.linspace(near, far, n_samples, device=device).view(-1, 1, 1)
    dirs_world = dirs_world.reshape(1, H, W, 3)
    pts = cam_pos.view(1, 1, 1, 3) + ts.unsqueeze(-1) * dirs_world
    return pts.unsqueeze(0)  # (1, S, H, W, 3)


def sample_volume(sigma, rgb, pts_world, center, radius):
    pts_grid = world_to_grid(pts_world, center, radius)  # (1, S, H, W, 3)
    _, S, H, W, _ = pts_grid.shape
    grid = pts_grid.view(1, S, H, W, 3)
    sigma_s = F.grid_sample(
        sigma, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (1, 1, S, H, W)
    rgb_s = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (1, 3, S, H, W)
    return sigma_s.squeeze(1), rgb_s.permute(0, 2, 3, 4, 1)


def volume_render(sigma_s, rgb_s, n_samples, near=None, far=None,
                  return_depth=False):
    """NeRF-style front-to-back alpha compositing.
    sigma_s: (1, S, H, W); rgb_s: (1, S, H, W, 3).
    If return_depth=True, also returns expected ray-distance (1, H, W) in
    the same world units as the ray parameterization (linspace(near, far))."""
    delta = 1.0 / n_samples
    alpha = 1.0 - torch.exp(-sigma_s * delta)
    a_sh = torch.cat([torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1)
    T = torch.cumprod(1.0 - a_sh + 1e-10, dim=1)
    w = T * alpha
    rgb_out = (w.unsqueeze(-1) * rgb_s).sum(dim=1)              # (1, H, W, 3)
    rgb_out = rgb_out.permute(0, 3, 1, 2)
    if not return_depth:
        return rgb_out
    S = sigma_s.shape[1]
    ts = torch.linspace(near, far, S, device=sigma_s.device).view(1, S, 1, 1)
    depth_out = (w * ts).sum(dim=1)                              # (1, H, W)
    return rgb_out, depth_out


def fuse_flow_da(flow_d, da_rel):
    """Fuse flow-triangulated depth with DA's relative depth, scaled so
    DA matches flow in overlap. Returns a depth map in OW-m with
    coverage = flow union DA (where flow gives the metric scale)."""
    ok = np.isfinite(flow_d) & np.isfinite(da_rel) & (da_rel > 1e-6)
    if int(ok.sum()) < 200:
        return flow_d.copy()  # not enough overlap to fit scale
    s = float(np.median(flow_d[ok] / da_rel[ok]))
    da_depth = da_rel * s
    fused = flow_d.copy()
    fill = ~np.isfinite(fused) & np.isfinite(da_depth) & (da_depth > 0)
    fused[fill] = da_depth[fill]
    return fused


def flow_depth_map_full(fa, fb, R_rel, t_rel, K_full,
                        min_flow_px=1.5, max_depth=80.0):
    """Triangulate depth at full resolution. K_full is the full-res
    intrinsic matrix as a numpy array."""
    H, W = fa.shape[:2]
    if float(np.linalg.norm(t_rel)) < 1e-3:
        return np.full((H, W), np.nan, dtype=np.float32)
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
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if valid.sum() < 100:
        return depth
    P1 = K_full @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_full @ np.hstack([R_rel, t_rel.reshape(3, 1)])
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


def tv3d(x):
    """Total-variation reg on a 5D (1, C, D, H, W) tensor."""
    dx = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().mean()
    dy = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()
    dz = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
    return dx + dy + dz


def export_voxels_ply(sigma, rgb, center, radius, out_path: Path,
                      sigma_thresh=0.5):
    """Save occupied voxels (sigma > thresh) as a colored point cloud."""
    s = sigma.detach().cpu().numpy()[0, 0]
    c = rgb.detach().cpu().numpy()[0]
    occ = s > sigma_thresh
    D, H, W = s.shape
    idx = np.argwhere(occ)
    if len(idx) == 0:
        out_path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        return 0
    # Voxel idx rows are (d, h, w). grid_sample's (x, y, z) maps to
    # (W, H, D) -- so world x comes from w, y from h, z from d.
    xg = -1.0 + 2.0 * idx[:, 2] / (W - 1)
    yg = -1.0 + 2.0 * idx[:, 1] / (H - 1)
    zg = -1.0 + 2.0 * idx[:, 0] / (D - 1)
    pts_world = np.stack([xg, yg, zg], axis=-1) * radius + center
    colors = c[:, idx[:, 0], idx[:, 1], idx[:, 2]].T  # (N, 3) in [0, 1]
    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_world)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts_world, colors_u8):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
    return len(pts_world)


# ---------- Main training ----------
def train_event(event_dir: Path, args):
    pose_p = event_dir / "pose.json"
    clip_p = event_dir / "clip.mp4"
    ply_p  = event_dir / "pointcloud.ply"
    if not (pose_p.exists() and clip_p.exists() and ply_p.exists()):
        return None, "need pose.json, clip.mp4, pointcloud.ply"

    pose = json.loads(pose_p.read_text())
    pose_frames = pose["frames"]
    W0, H0 = pose["intrinsics"]["image_size"]
    focal0 = pose["intrinsics"]["focal_px"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Optional DepthAnything for filling in low-flow regions
    da = None
    if not args.no_da:
        import os, sys as _sys
        _sys.path.insert(0, os.path.expanduser("~/turntable"))
        try:
            from depth_anything import DepthAnythingEstimator
            print("Loading DepthAnything...")
            da = DepthAnythingEstimator()
            print("DA ready.")
        except Exception as e:
            print(f"DA failed ({e}); using flow-only depth.")

    # Scene center + radius from pointcloud (percentile, robust)
    pts = read_ply_points(ply_p)
    lo = np.percentile(pts, args.bbox_pct_lo, axis=0)
    hi = np.percentile(pts, args.bbox_pct_hi, axis=0)
    center_np = (lo + hi) / 2
    extent = hi - lo
    radius = float(extent.max() / 2 + args.bbox_pad)
    print(f"scene center: {center_np}  radius: {radius:.2f} ow-m")
    center = torch.tensor(center_np, dtype=torch.float32, device=device)

    # Training image size: downscale H0xW0 to ~--train-h pixels tall
    s = args.train_h / H0
    Ht = int(round(H0 * s))
    Wt = int(round(W0 * s))
    focal = focal0 * s
    K = torch.tensor([[focal, 0, Wt / 2.0],
                      [0, focal, Ht / 2.0],
                      [0, 0, 1.0]], dtype=torch.float32, device=device)
    K_inv = torch.inverse(K)
    print(f"train resolution: {Wt}x{Ht}  focal: {focal:.1f} px")

    # Load + resize frames
    cap = cv2.VideoCapture(str(clip_p))
    raw = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        raw.append(fr)
    cap.release()
    n = min(len(raw), len(pose_frames))

    # Full-resolution intrinsic (for accurate flow triangulation)
    K_full_np = np.array([[focal0, 0, W0 / 2.0],
                          [0, focal0, H0 / 2.0],
                          [0, 0, 1.0]], dtype=np.float32)

    # Gun-zone mask at training resolution: 0 inside the gun zone, 1 elsewhere.
    # Same convention as analyze-walls.py (bottom-right rectangle).
    gx0 = int(Wt * args.gun_x_frac)
    gy0 = int(Ht * args.gun_y_frac)
    use_mask = np.ones((Ht, Wt), dtype=np.float32)
    if not args.no_gun_mask:
        use_mask[gy0:, gx0:] = 0.0
    use_mask_t = torch.from_numpy(use_mask).to(device)
    # Full-res version for masking depth before downsample
    gx0_full = int(W0 * args.gun_x_frac)
    gy0_full = int(H0 * args.gun_y_frac)

    # Pre-scan poses to identify candidates (converged + low RMS) and their
    # partners (the next valid pose, needed for flow triangulation).
    candidate_idx = []
    for i in range(n):
        pf = pose_frames[i]
        if pf.get("converged") and pf.get("reproj_rms_px", 1e9) <= args.max_rms_px:
            candidate_idx.append(i)
    if len(candidate_idx) < 2:
        return None, (f"only {len(candidate_idx)} converged poses with rms<={args.max_rms_px}; "
                      f"loosen --max-rms-px")

    # First pass: compute flow depth per candidate, check coverage,
    # accept/reject. We need at least one partner ahead of each candidate.
    gt_tensors = []
    poses_Rt = []
    accepted_idx = []   # indices into `raw` / `pose_frames` we accepted
    partners_for = []   # parallel: partner index used for triangulation
    flow_depth_cache = []  # full-res flow depth per accepted frame (to fuse w/ DA)

    n_rejected_coverage = 0
    n_rejected_no_partner = 0
    for c_pos, i in enumerate(candidate_idx[:-1]):
        partner = candidate_idx[c_pos + 1]
        pa = pose_frames[i]; pb = pose_frames[partner]
        Ra_np = np.array(rotation_world_to_cam(pa["yaw_rad"], pa["pitch_rad"]).cpu())
        Rb_np = np.array(rotation_world_to_cam(pb["yaw_rad"], pb["pitch_rad"]).cpu())
        ta_np = np.array(pa["t"], dtype=np.float32)
        tb_np = np.array(pb["t"], dtype=np.float32)
        R_rel = Rb_np @ Ra_np.T
        t_rel = tb_np - R_rel @ ta_np
        flow_full = flow_depth_map_full(
            raw[i], raw[partner], R_rel, t_rel, K_full_np,
            min_flow_px=args.depth_min_flow_px,
            max_depth=args.depth_max,
        )
        # Mask gun zone in full-res flow before measuring coverage
        if not args.no_gun_mask:
            flow_full[gy0_full:, gx0_full:] = np.nan
        coverage = float(np.isfinite(flow_full).mean())
        if coverage < args.min_flow_coverage:
            n_rejected_coverage += 1
            continue
        # Accept: build the training tensors
        fr = cv2.resize(raw[i], (Wt, Ht), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
        gt_tensors.append(gt)
        R = rotation_world_to_cam(pa["yaw_rad"], pa["pitch_rad"], device=device)
        t = torch.tensor(pa["t"], dtype=torch.float32, device=device)
        poses_Rt.append((R, t))
        accepted_idx.append(i)
        partners_for.append(partner)
        flow_depth_cache.append(flow_full)
    V = len(gt_tensors)
    if V < 2:
        return None, (f"only {V} accepted after coverage filter "
                      f"(rejected {n_rejected_coverage} for low flow coverage)")
    print(f"loaded {V} usable views "
          f"(rejected {n_rejected_coverage} for flow_coverage<{args.min_flow_coverage})")
    if not args.no_gun_mask:
        print(f"gun-mask: bottom-right region from ({gx0_full}, {gy0_full}) "
              f"in full-res / ({gx0}, {gy0}) in train-res")
    gt_stack = torch.stack(gt_tensors, dim=0)
    valid_indices = accepted_idx  # downstream alias

    # Pre-compute flow-triangulated depth maps. We need them for two
    # purposes: (a) optional photometric+depth supervision, where we use
    # ray-distance; (b) periodic free-space culling, where we use the
    # camera-frame z (z_cam) directly.
    depth_dist_tensors = []     # ray-distance, for depth supervision
    depth_zcam_tensors = []     # z_cam, for culling
    depth_valid_tensors = []
    need_depth = args.depth_weight > 0 or args.cull_every > 0
    if need_depth:
        print(f"pre-computing fused depth for {V} views (flow already done)...")
        # Precompute per-pixel cos(theta) for ray-dist conversion.
        K_train_np = np.array([[focal, 0, Wt / 2.0],
                               [0, focal, Ht / 2.0],
                               [0, 0, 1.0]], dtype=np.float32)
        K_train_inv = np.linalg.inv(K_train_np)
        ys = np.arange(Ht); xs = np.arange(Wt)
        U, V_grid = np.meshgrid(xs, ys)
        pix = np.stack([U, V_grid, np.ones_like(U)], axis=-1).astype(np.float32)
        rays_cam = pix.reshape(-1, 3) @ K_train_inv.T
        rays_norm = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
        cos_theta = rays_norm[:, 2].reshape(Ht, Wt)
        cos_theta_t = torch.from_numpy(cos_theta).to(device)
        gun_mask_bool = (use_mask < 0.5)  # True inside gun zone
        for k, i in enumerate(valid_indices):
            d_full = flow_depth_cache[k]
            if da is not None:
                da_rel = da.estimate(raw[i])
                if not args.no_gun_mask:
                    da_rel = da_rel.copy()
                    da_rel[gy0_full:, gx0_full:] = np.nan
                d_full = fuse_flow_da(d_full, da_rel)
            d = cv2.resize(d_full, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
            d_t = torch.from_numpy(d).to(device)
            # Re-apply gun mask at training res in case nearest interp leaked
            if not args.no_gun_mask:
                d_t = torch.where(gun_mask_bool,
                                   torch.full_like(d_t, float('nan')), d_t)
            valid = torch.isfinite(d_t) & (d_t > 0.1)
            ray_dist = d_t / (cos_theta_t + 1e-6)
            ray_dist = torch.where(valid, ray_dist,
                                    torch.full_like(ray_dist, float('nan')))
            depth_dist_tensors.append(ray_dist)
            zcam = torch.where(valid, d_t, torch.full_like(d_t, float('nan')))
            depth_zcam_tensors.append(zcam)
            depth_valid_tensors.append(valid)
        n_total_valid = sum(int(v.sum()) for v in depth_valid_tensors)
        coverage_pct = 100.0 * n_total_valid / (V * Ht * Wt)
        print(f"depth supervision: {n_total_valid:,} valid pixels "
              f"({coverage_pct:.1f}% across {V} views)")

    # Voxel grid: world coords need to be in [-radius, radius] when normalized.
    # Center the world frame on the scene so volume sampling works.
    # Adjust each pose's t so that camera position in shifted frame is
    # cam_pos_world - center. Since t = -R @ cam_pos, the shifted
    # t_shifted = R @ (cam_pos_shifted - 0) = R @ (cam_pos - center) = t + R @ center.
    poses_shifted = []
    for (R, t) in poses_Rt:
        t_new = t + R @ center
        poses_shifted.append((R, t_new))
    # And the scene center in the shifted frame is now (0, 0, 0).
    # So generate_rays is called with scene_center = 0 (just the radius matters
    # in world_to_grid below).
    zero_center = torch.zeros(3, device=device)

    vol = VoxelVolume(
        grid_size=args.grid_size,
        init_density_logit=args.init_density_logit,
        sigma_scale=args.sigma_scale,
    ).to(device)

    # Pre-build (N, 3) voxel centers in the SHIFTED world frame for culling.
    # grid_sample's last-axis convention is (x, y, z) where x indexes W,
    # y indexes H, z indexes D. So for density[0,0,d,h,w] the world coord
    # is (lin[w], lin[h], lin[d]) * radius (in shifted frame).
    K_train_t = torch.tensor(K_train_np, dtype=torch.float32, device=device)
    gs = args.grid_size
    lin = -1.0 + 2.0 * torch.arange(gs, device=device).float() / (gs - 1)
    Dg, Hg, Wg = torch.meshgrid(lin, lin, lin, indexing="ij")
    voxel_centers_shifted = torch.stack(
        [Wg.reshape(-1), Hg.reshape(-1), Dg.reshape(-1)], dim=-1,
    ) * radius  # (N, 3) (x, y, z) shifted-world centers

    opt = optim.Adam([
        {"params": [vol.density], "lr": args.lr_density},
        {"params": [vol.color],   "lr": args.lr_color},
    ])

    out = event_dir / "voxel_train"
    out.mkdir(exist_ok=True)
    snap_dir = out / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap_idx = np.linspace(0, V - 1, num=min(args.snapshot_views, V),
                           dtype=int).tolist()

    n_samples = args.n_samples
    # Ray near/far in radius-units (since we centered the world on the scene)
    # Camera could be inside or outside [-radius, radius]. Pick generously.
    near, far = args.ray_near, args.ray_far

    ema = None
    t0_all = time.perf_counter()
    for it in range(args.iters):
        if args.views_per_iter and args.views_per_iter < V:
            batch = torch.randperm(V)[:args.views_per_iter].tolist()
        else:
            batch = list(range(V))

        opt.zero_grad()
        sigma, rgb = vol()
        mse_acc = 0.0
        depth_acc = 0.0
        for vi in batch:
            R, t = poses_shifted[vi]
            pts_world = generate_rays(
                Ht, Wt, K_inv, R, t, n_samples,
                near=near, far=far,
                scene_center=zero_center, device=device,
            )
            sigma_s, rgb_s = sample_volume(
                sigma, rgb, pts_world, zero_center, radius,
            )
            if args.depth_weight > 0:
                img_full, ray_dist_pred = volume_render(
                    sigma_s, rgb_s, n_samples,
                    near=near, far=far, return_depth=True,
                )
            else:
                img_full = volume_render(sigma_s, rgb_s, n_samples)
                ray_dist_pred = None
            img = img_full[0]
            # Photometric loss masked by gun-zone (zero weight inside it).
            diff2 = (img - gt_stack[vi]) ** 2  # (3, H, W)
            m = use_mask_t.unsqueeze(0)  # (1, H, W)
            l_photo = (diff2 * m).sum() / (m.sum() * 3 + 1e-6) / len(batch)
            # Depth loss (only on flow-valid pixels, which already exclude gun zone)
            if args.depth_weight > 0 and depth_valid_tensors[vi].any():
                vmask = depth_valid_tensors[vi]
                pred_d = ray_dist_pred[0][vmask]
                obs_d = depth_dist_tensors[vi][vmask]
                l_depth = ((pred_d - obs_d) ** 2).mean() / len(batch)
                l = l_photo + args.depth_weight * l_depth
                depth_acc += float(l_depth.item())
            else:
                l = l_photo
            l.backward(retain_graph=True)
            mse_acc += float(l_photo.item())

        # Regularizers
        if args.tv_weight > 0 or args.alpha_weight > 0:
            reg = 0.0
            if args.tv_weight > 0:
                reg = reg + args.tv_weight * (tv3d(vol.density) + tv3d(vol.color))
            if args.alpha_weight > 0:
                reg = reg + args.alpha_weight * sigma.mean()
            reg.backward()
            reg_val = float(reg.item())
        else:
            reg_val = 0.0

        opt.step()
        total = mse_acc + reg_val
        ema = total if ema is None else (0.9 * ema + 0.1 * total)

        # ---- Free-space cull ----
        if (args.cull_every > 0
                and it + 1 >= args.cull_start
                and (it + 1) % args.cull_every == 0):
            with torch.no_grad():
                sup_flat, free_count, vis_count = compute_suppress_mask(
                    voxel_centers_shifted, poses_shifted, depth_zcam_tensors,
                    K_train_t, (Ht, Wt),
                    min_free_frac=args.cull_free_frac,
                    min_visible=args.cull_min_visible,
                    margin=args.cull_margin,
                    device=device,
                )
                # The flat order is (w fastest, h, d outermost). Reshape
                # into (D, H, W) with the inverse of meshgrid + stack:
                new_mask = sup_flat.float().view(gs, gs, gs)  # (D, H, W)
                # But our flat ordering is (d, h, w) iter with w fastest
                # -- view as (D, H, W) directly matches.
                vol.suppress_mask.copy_(new_mask.unsqueeze(0).unsqueeze(0))
                n_sup = int(sup_flat.sum())
                n_vis = int((vis_count > 0).sum())
                print(f"  [cull@{it+1}] suppressed {n_sup:,} voxels "
                      f"({n_sup * 100.0 / (gs**3):.1f}% of grid); "
                      f"{n_vis:,} ever visible")
        if it % args.log_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                s_max = float(sigma.max().item())
                s_mean = float(sigma.mean().item())
            depth_str = f" depth={depth_acc:.4e}" if args.depth_weight > 0 else ""
            print(f"[{it:5d}/{args.iters}] loss={total:.4e} ema={ema:.4e} "
                  f"mse={mse_acc:.4e}{depth_str} reg={reg_val:.4e}  "
                  f"sigma mean={s_mean:.3f} max={s_max:.2f}  "
                  f"({time.perf_counter() - t0_all:.0f}s)")

        if args.snapshot_every and (it + 1) % args.snapshot_every == 0:
            with torch.no_grad():
                sigma, rgb = vol()
                for vi in snap_idx:
                    R, t = poses_shifted[vi]
                    pts_world = generate_rays(
                        Ht, Wt, K_inv, R, t, n_samples,
                        near=near, far=far,
                        scene_center=zero_center, device=device,
                    )
                    sigma_s, rgb_s = sample_volume(
                        sigma, rgb, pts_world, zero_center, radius,
                    )
                    img = volume_render(sigma_s, rgb_s, n_samples)[0]
                    np_img = (img.clamp(0, 1).permute(1, 2, 0)
                              .cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(
                        str(snap_dir / f"iter{it+1:05d}_v{vi:03d}.png"),
                        cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR),
                    )
            print(f"  [snapshot] iter {it+1} -> {snap_dir}")

    # Save final volume + ply
    with torch.no_grad():
        sigma, rgb = vol()
    np.savez_compressed(
        out / "volume.npz",
        density_logit=vol.density.detach().cpu().numpy(),
        color_logit=vol.color.detach().cpu().numpy(),
        scene_center=center_np,
        scene_radius=np.float32(radius),
        grid_size=np.int32(args.grid_size),
        sigma_scale=np.float32(args.sigma_scale),
    )
    # Export sigma > thresh voxels as point cloud (in ORIGINAL world frame,
    # i.e. un-shifted, so it lines up with pointcloud.ply / mesh.ply).
    n_pts = export_voxels_ply(
        sigma, rgb,
        center=center_np, radius=radius,
        out_path=out / "voxels.ply",
        sigma_thresh=args.export_sigma_thresh,
    )
    print(f"\nDone. {n_pts} occupied voxels exported to {out}/voxels.ply")
    return {"event": event_dir.name, "n_views": V, "n_voxels_exported": n_pts}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", required=True)

    ap.add_argument("--grid-size", type=int, default=64,
                    help="cubic voxel-grid side length (default 64). "
                         "Memory scales O(N^3); 128 is a stretch.")
    ap.add_argument("--init-density-logit", type=float, default=-5.0)
    ap.add_argument("--sigma-scale", type=float, default=20.0)

    ap.add_argument("--train-h", type=int, default=120,
                    help="training image height in px (width scales to "
                         "keep aspect). Default 120; capture is 720 so "
                         "default downscales 6x.")
    ap.add_argument("--n-samples", type=int, default=64,
                    help="ray samples per pixel (default 64)")
    ap.add_argument("--ray-near", type=float, default=0.1,
                    help="ray near plane in scene-radius units (default 0.1)")
    ap.add_argument("--ray-far", type=float, default=4.0,
                    help="ray far plane in scene-radius units (default 4.0)")

    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr-density", type=float, default=2e-2)
    ap.add_argument("--lr-color",   type=float, default=2e-2)
    ap.add_argument("--views-per-iter", type=int, default=4,
                    help="number of views to render per iteration "
                         "(default 4). All views per iter is more memory.")
    ap.add_argument("--tv-weight",    type=float, default=1e-3)
    ap.add_argument("--alpha-weight", type=float, default=1e-4)
    ap.add_argument("--depth-weight", type=float, default=0.1,
                    help="weight on the depth-supervision loss "
                         "(default 0.1). Set 0 to disable. The flow-"
                         "triangulated depth is pre-computed at start.")
    ap.add_argument("--depth-min-flow-px", type=float, default=1.5,
                    help="min flow magnitude for triangulation (default 1.5)")
    ap.add_argument("--depth-max", type=float, default=30.0,
                    help="max trusted depth in ow-m (default 30)")
    ap.add_argument("--cull-every", type=int, default=200,
                    help="run free-space voxel cull every N iters "
                         "(default 200). Set 0 to disable. Uses the "
                         "flow-triangulated depth maps; suppresses "
                         "voxels seen in observed free space.")
    ap.add_argument("--cull-start", type=int, default=200,
                    help="iter at which to begin culling (default 200; "
                         "give the volume a warm-up before slamming "
                         "suppressions in)")
    ap.add_argument("--cull-free-frac", type=float, default=0.30,
                    help="fraction of valid views in which a voxel must "
                         "appear in free space to be suppressed (default 0.30)")
    ap.add_argument("--cull-min-visible", type=int, default=3,
                    help="minimum number of views in which a voxel must be "
                         "observed (any state) before it can be suppressed "
                         "(default 3)")
    ap.add_argument("--cull-margin", type=float, default=0.10,
                    help="margin in ow-m: a voxel is considered free if "
                         "z_cam < observed_depth - margin (default 0.10)")
    ap.add_argument("--no-da", action="store_true",
                    help="skip DepthAnything; use flow-only depth (sparse) "
                         "for supervision and culling. Default uses fused "
                         "flow+DA depth -- denser, more aggressive cull.")

    ap.add_argument("--bbox-pct-lo", type=float, default=2.0)
    ap.add_argument("--bbox-pct-hi", type=float, default=98.0)
    ap.add_argument("--bbox-pad",    type=float, default=0.3)

    ap.add_argument("--max-rms-px", type=float, default=30.0,
                    help="reject poses with reproj RMS above this (default 30 px)")
    ap.add_argument("--min-flow-coverage", type=float, default=0.05,
                    help="reject frames whose flow-triangulated depth covers "
                         "less than this fraction of the (un-masked) frame "
                         "(default 0.05). Flow needs parallax; frames with "
                         "no usable flow have no metric anchor.")
    ap.add_argument("--gun-x-frac", type=float, default=0.50,
                    help="x fraction where Mei's gun-zone exclusion starts "
                         "(default 0.50). Pixels in [gun_x, W] x [gun_y, H] "
                         "are zero-weighted in photometric loss and NaN'd "
                         "in depth maps.")
    ap.add_argument("--gun-y-frac", type=float, default=0.55,
                    help="y fraction where gun-zone exclusion starts (default 0.55)")
    ap.add_argument("--no-gun-mask", action="store_true",
                    help="disable the gun-zone exclusion mask")

    ap.add_argument("--log-every",      type=int, default=10)
    ap.add_argument("--snapshot-every", type=int, default=200)
    ap.add_argument("--snapshot-views", type=int, default=3)
    ap.add_argument("--export-sigma-thresh", type=float, default=0.5)
    args = ap.parse_args()

    event_dir = Path(args.events_dir) / args.event
    print(f"Training voxel volume on {event_dir.name}")
    rec, err = train_event(event_dir, args)
    if err:
        print(f"  [skip] {err}")


if __name__ == "__main__":
    main()

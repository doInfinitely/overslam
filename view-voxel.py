#!/usr/bin/env python3
"""Volumetric viewer for a trained voxel volume.

Loads volume.npz produced by train-voxel.py and opens an interactive
pygame window. Two render modes:
  V : NeRF-style volume render (slow, faithful; matches training)
  C : voxel-cubes / points (fast, depth-sorted point splat)

Controls
--------
  drag                 : orbit
  scroll               : zoom
  Arrow keys           : orbit step
  + / -                : zoom step
  V                    : volume render mode
  C                    : cubes / points mode
  B                    : cycle background (black -> gray -> white -> dark)
  R                    : recenter on scene_center from the npz
  T                    : cycle sigma threshold for cubes mode
  Esc                  : quit
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pygame
except ImportError:
    raise SystemExit("pip install pygame")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("pip install torch")


# ---------- Rendering primitives (matches train-voxel) ----------
def world_to_grid(pts_world, center, radius):
    return (pts_world - center) / radius


def generate_rays(H, W, K_inv, R, t, n_samples, near, far, device):
    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # (H, W, 3)
    dirs_cam = (K_inv @ pix.reshape(-1, 3).T).T
    dirs_cam = dirs_cam / (dirs_cam.norm(dim=-1, keepdim=True) + 1e-9)
    dirs_world = (R.T @ dirs_cam.T).T
    cam_pos = (-R.T @ t).reshape(1, 3)
    ts = torch.linspace(near, far, n_samples, device=device).view(-1, 1, 1)
    dirs_world = dirs_world.reshape(1, H, W, 3)
    pts = cam_pos.view(1, 1, 1, 3) + ts.unsqueeze(-1) * dirs_world
    return pts.unsqueeze(0)  # (1, S, H, W, 3)


def sample_volume(sigma, rgb, pts_world, center, radius):
    pts_grid = world_to_grid(pts_world, center, radius)
    _, S, H, W, _ = pts_grid.shape
    grid = pts_grid.view(1, S, H, W, 3)
    sigma_s = F.grid_sample(
        sigma, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    rgb_s = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return sigma_s.squeeze(1), rgb_s.permute(0, 2, 3, 4, 1)


def volume_render(sigma_s, rgb_s, n_samples, bg=(0, 0, 0)):
    delta = 1.0 / n_samples
    alpha = 1.0 - torch.exp(-sigma_s * delta)
    a_sh = torch.cat([torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1)
    T = torch.cumprod(1.0 - a_sh + 1e-10, dim=1)
    w = T * alpha
    rgb_out = (w.unsqueeze(-1) * rgb_s).sum(dim=1)  # (1, H, W, 3)
    # Background where total alpha is low
    alpha_total = w.sum(dim=1)  # (1, H, W)
    bg_t = torch.tensor(bg, dtype=torch.float32,
                        device=rgb_out.device).view(1, 1, 1, 3)
    rgb_out = rgb_out + (1 - alpha_total.unsqueeze(-1)) * bg_t
    return rgb_out.permute(0, 3, 1, 2)  # (1, 3, H, W)


# ---------- Orbit camera ----------
class OrbitCamera:
    def __init__(self, center, radius_init):
        self.target = np.asarray(center, dtype=np.float32)
        self.distance = float(radius_init)
        self.yaw = 0.4
        self.pitch = 0.3
        self.fov_y_deg = 55.0

    def eye_and_basis(self):
        cp = math.cos(self.pitch); sp = math.sin(self.pitch)
        cy = math.cos(self.yaw);   sy = math.sin(self.yaw)
        # World Y is DOWN (OpenCV). Camera-at-placement side: -Z.
        offset = np.array([cp * sy, -sp, -cp * cy], dtype=np.float32) * self.distance
        eye = self.target + offset
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd) + 1e-9
        up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right /= np.linalg.norm(right)
        cam_down = np.cross(fwd, right)
        R = np.stack([right, cam_down, fwd], axis=0)
        t = -R @ eye
        return eye, R, t


# ---------- Cubes / points mode (CPU, fast) ----------
def render_points(sigma_np, rgb_np, cam_pos, R, t,
                  W, H, fy_proj, fx_proj, cx, cy, scene_center, scene_radius,
                  thresh, bg):
    """Splat sigma>thresh voxels as colored points with painter's algorithm."""
    D, Hg, Wg = sigma_np.shape
    if not hasattr(render_points, "cache") or render_points.cache.get("shape") != (D, Hg, Wg):
        idx = np.argwhere(sigma_np >= 0)  # all voxel indices
        # World positions
        gn = np.array([D, Hg, Wg])
        gnorm = -1.0 + 2.0 * idx / (gn - 1)
        pts = gnorm * scene_radius + scene_center
        render_points.cache = {
            "shape": (D, Hg, Wg),
            "idx": idx,
            "pts": pts.astype(np.float32),
        }
    cache = render_points.cache
    occ = sigma_np[cache["idx"][:, 0], cache["idx"][:, 1], cache["idx"][:, 2]] > thresh
    if not occ.any():
        img = np.full((H, W, 3), bg, dtype=np.uint8)
        return img
    pts = cache["pts"][occ]
    cols = rgb_np[cache["idx"][occ, 0], cache["idx"][occ, 1], cache["idx"][occ, 2]]
    cols_u8 = np.clip(cols * 255, 0, 255).astype(np.uint8)
    # Camera-frame
    P_cam = (pts - cam_pos) @ R.T
    z = P_cam[:, 2]
    in_front = z > 0.01
    u = fx_proj * P_cam[:, 0] / np.maximum(z, 1e-3) + cx
    v = fy_proj * P_cam[:, 1] / np.maximum(z, 1e-3) + cy
    in_img = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not in_img.any():
        return np.full((H, W, 3), bg, dtype=np.uint8)
    u = u[in_img].astype(np.int32)
    v = v[in_img].astype(np.int32)
    cols_u8 = cols_u8[in_img]
    z = z[in_img]
    # Painter's: sort back-to-front (descending z)
    order = np.argsort(-z)
    u = u[order]; v = v[order]; cols_u8 = cols_u8[order]
    img = np.full((H, W, 3), bg, dtype=np.uint8)
    img[v, u] = cols_u8
    # Mild dilation (3x3) to fill point gaps
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        un = np.clip(u + dx, 0, W - 1); vn = np.clip(v + dy, 0, H - 1)
        img[vn, un] = cols_u8
    return img


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume_npz", help="path to volume.npz from train-voxel.py")
    ap.add_argument("--render-res", type=int, default=200,
                    help="internal render height (default 200); width auto")
    ap.add_argument("--window-size", type=int, default=900)
    ap.add_argument("--n-samples", type=int, default=64,
                    help="ray samples for volume mode (default 64)")
    ap.add_argument("--ray-near", type=float, default=0.05,
                    help="near plane (default 0.05 of distance)")
    ap.add_argument("--ray-far", type=float, default=4.0,
                    help="far plane in scene-radius units (default 4.0)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = np.load(args.volume_npz)
    density_logit = data["density_logit"]
    color_logit   = data["color_logit"]
    scene_center  = data["scene_center"].astype(np.float32)
    scene_radius  = float(data["scene_radius"])
    sigma_scale   = float(data["sigma_scale"]) if "sigma_scale" in data else 20.0
    grid_size     = int(data["grid_size"]) if "grid_size" in data else density_logit.shape[-1]
    print(f"grid {density_logit.shape}  center {scene_center}  radius {scene_radius:.2f}  "
          f"sigma_scale {sigma_scale}")

    # Reconstruct sigma + rgb as numpy (cubes mode) and torch (volume mode)
    sigma_np = (np.log1p(np.exp(density_logit[0, 0])) * sigma_scale).astype(np.float32)
    rgb_np   = 1.0 / (1.0 + np.exp(-color_logit[0]))  # sigmoid
    rgb_np   = rgb_np.transpose(1, 2, 3, 0).astype(np.float32)  # (D, H, W, 3)
    sigma_t = torch.from_numpy(sigma_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    rgb_t   = torch.from_numpy(rgb_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # Pygame
    pygame.init()
    pygame.display.set_caption("voxel viewer  V=volume C=cubes B=bg R=recenter T=thresh Esc=quit")
    WIN = args.window_size
    screen = pygame.display.set_mode((WIN, WIN))
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    # Internal render resolution -- volume mode renders here, then upscale to window
    RH = args.render_res
    RW = RH  # square render

    cam = OrbitCamera(center=scene_center, radius_init=scene_radius * 2.0)
    K_int_np = np.array([
        [(RH / 2) / math.tan(math.radians(cam.fov_y_deg) / 2), 0, RW / 2.0],
        [0, (RH / 2) / math.tan(math.radians(cam.fov_y_deg) / 2), RH / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float32)
    K_int = torch.from_numpy(K_int_np).to(device)
    K_inv = torch.inverse(K_int)
    fy_proj = K_int_np[1, 1]; fx_proj = K_int_np[0, 0]
    cx_proj = K_int_np[0, 2]; cy_proj = K_int_np[1, 2]

    backgrounds = [(8, 8, 12), (128, 128, 128), (240, 240, 240), (28, 30, 38)]
    bg_idx = 0
    thresh_cycle = [0.5, 1.0, 2.0, 5.0, 0.25]
    thresh_idx = 0
    mode = "volume"
    dragging = False
    last_mouse = (0, 0)

    running = True
    while running:
        clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_v:
                    mode = "volume"
                elif ev.key == pygame.K_c:
                    mode = "cubes"
                elif ev.key == pygame.K_b:
                    bg_idx = (bg_idx + 1) % len(backgrounds)
                elif ev.key == pygame.K_t:
                    thresh_idx = (thresh_idx + 1) % len(thresh_cycle)
                elif ev.key == pygame.K_r:
                    cam.target = scene_center.copy()
                elif ev.key == pygame.K_LEFT:
                    cam.yaw -= 0.05
                elif ev.key == pygame.K_RIGHT:
                    cam.yaw += 0.05
                elif ev.key == pygame.K_UP:
                    cam.pitch = float(np.clip(cam.pitch - 0.05, -1.4, 1.4))
                elif ev.key == pygame.K_DOWN:
                    cam.pitch = float(np.clip(cam.pitch + 0.05, -1.4, 1.4))
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    cam.distance *= 0.9
                elif ev.key == pygame.K_MINUS:
                    cam.distance *= 1.111
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    dragging = True
                    last_mouse = ev.pos
                elif ev.button == 4:
                    cam.distance *= 0.9
                elif ev.button == 5:
                    cam.distance *= 1.111
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                dx = ev.pos[0] - last_mouse[0]
                dy = ev.pos[1] - last_mouse[1]
                cam.yaw   += dx * 0.008
                cam.pitch  = float(np.clip(cam.pitch + dy * 0.008, -1.4, 1.4))
                last_mouse = ev.pos

        bg = backgrounds[bg_idx]
        eye, R_np, t_np = cam.eye_and_basis()

        if mode == "volume":
            R = torch.from_numpy(R_np).to(device)
            t = torch.from_numpy(t_np).to(device)
            with torch.no_grad():
                # Center is at scene_center -> generate rays in world, sample
                # the volume normalized against (scene_center, scene_radius).
                pts = generate_rays(
                    RH, RW, K_inv, R, t, args.n_samples,
                    near=args.ray_near, far=args.ray_far * scene_radius,
                    device=device,
                )
                center_t = torch.from_numpy(scene_center).to(device)
                sigma_s, rgb_s = sample_volume(sigma_t, rgb_t, pts,
                                                center_t, scene_radius)
                img_t = volume_render(sigma_s, rgb_s, args.n_samples,
                                       bg=tuple(c / 255.0 for c in bg))
            img_np = img_t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            img_u8 = (img_np * 255).astype(np.uint8)
        else:
            img_u8 = render_points(
                sigma_np, rgb_np, eye, R_np, t_np,
                RW, RH, fy_proj, fx_proj, cx_proj, cy_proj,
                scene_center, scene_radius,
                thresh=thresh_cycle[thresh_idx], bg=bg,
            )

        # Display: convert (H, W, 3) RGB -> pygame surface (W, H, 3)
        surf = pygame.surfarray.make_surface(np.swapaxes(img_u8, 0, 1))
        surf = pygame.transform.smoothscale(surf, (WIN, WIN))
        screen.blit(surf, (0, 0))

        hud = [
            f"mode: {mode}   render: {RW}x{RH} -> {WIN}x{WIN}",
            f"yaw={math.degrees(cam.yaw):.0f}  pitch={math.degrees(cam.pitch):.0f}  "
            f"dist={cam.distance:.2f}  target=({cam.target[0]:+.2f},{cam.target[1]:+.2f},{cam.target[2]:+.2f})",
            f"bg: rgb{bg}   {'thresh: ' + str(thresh_cycle[thresh_idx]) if mode == 'cubes' else ''}",
            "V=volume C=cubes B=bg R=recenter T=thresh Esc=quit",
        ]
        for i, h in enumerate(hud):
            surf_text = font.render(h, True, (255, 255, 255), bg)
            screen.blit(surf_text, (8, 6 + i * 16))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

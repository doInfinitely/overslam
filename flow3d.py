#!/usr/bin/env python3
"""3D flow-arrow viewer (pygame).

For each consecutive frame pair (i, i+1) of an event clip:
  - Farneback dense flow between i and i+1.
  - Triangulated depth at frame i.
  - For each sampled pixel (u, v) with valid depth d_a:
      tail_world = unproject(u, v, d_a) -> world frame via pose i
      head_world = unproject(u+du, v+dv, d_b) -> world frame via pose i
                   where d_b = depth_a[u+du, v+dv]  (depth at destination
                   pixel in the SAME frame's depth map)
  - The arrow head - tail in world coords picks up a Z component whenever
    the flow crosses depth boundaries (you, the viewer, moving forward
    relative to the geometry).

Pygame viewer:
  - Orbit camera (drag = orbit, shift+drag = pan, scroll = zoom)
  - A / D : prev / next frame
  - Q / E : -5 / +5 frames
  - R    : recenter on scene centroid
  - Esc  : quit
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    raise SystemExit("pip install pygame")


# ---------- Geometry (shared with reconstruct/visualize-depth) ----------
def rotation_world_to_cam(yaw, pitch):
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw = np.array([[ cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return R_pitch @ R_yaw


def relative_pose(pa, pb):
    Ra = rotation_world_to_cam(pa["yaw_rad"], pa["pitch_rad"])
    Rb = rotation_world_to_cam(pb["yaw_rad"], pb["pitch_rad"])
    ta = np.array(pa["t"]); tb = np.array(pb["t"])
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


def flow_depth_map(fa, fb, pa, pb, K, min_flow_px=1.5, max_depth=20.0):
    H, W = fa.shape[:2]
    R_rel, t_rel = relative_pose(pa, pb)
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if float(np.linalg.norm(t_rel)) < 1e-3:
        return depth, None
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
        return depth, flow
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
    return depth, flow


# ---------- Compute arrows per frame ----------
def compute_arrows(event_dir: Path, args):
    pose_p  = event_dir / "pose.json"
    clip_p  = event_dir / "clip.mp4"
    if not (pose_p.exists() and clip_p.exists()):
        raise SystemExit("need pose.json and clip.mp4 in the event dir")
    pose = json.loads(pose_p.read_text())
    pose_frames = pose["frames"]
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]])

    cap = cv2.VideoCapture(str(clip_p))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = min(len(frames), len(pose_frames))

    grid = args.grid
    ys = np.arange(grid // 2, H, grid)
    xs = np.arange(grid // 2, W, grid)

    arrows_per_frame = []  # list of (tails, heads, colors) per frame pair
    for i in range(n - 1):
        pa = pose_frames[i]
        pb = pose_frames[i + 1]
        if (not pa.get("converged") or not pb.get("converged")
                or pa.get("reproj_rms_px", 1e9) > args.max_rms_px
                or pb.get("reproj_rms_px", 1e9) > args.max_rms_px):
            arrows_per_frame.append((np.zeros((0, 3)), np.zeros((0, 3)),
                                     np.zeros((0, 3))))
            continue
        depth, flow = flow_depth_map(
            frames[i], frames[i + 1], pa, pb, K,
            min_flow_px=args.min_flow_px, max_depth=args.max_depth,
        )
        if flow is None:
            arrows_per_frame.append((np.zeros((0, 3)), np.zeros((0, 3)),
                                     np.zeros((0, 3))))
            continue

        Ra = rotation_world_to_cam(pa["yaw_rad"], pa["pitch_rad"])
        cam_pos_a = np.array(pa["camera_pos_world"])
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        tails = []
        heads = []
        colors = []
        for v0 in ys:
            for u0 in xs:
                d = float(depth[v0, u0])
                if not np.isfinite(d):
                    continue
                du = float(flow[v0, u0, 0])
                dv = float(flow[v0, u0, 1])
                u1 = int(np.clip(u0 + du, 0, W - 1))
                v1 = int(np.clip(v0 + dv, 0, H - 1))
                d_dest = float(depth[v1, u1])
                if not np.isfinite(d_dest):
                    continue
                # Source point in cam_a frame -> world
                P_a_cam = np.array([(u0 - cx) * d / fx,
                                     (v0 - cy) * d / fy,
                                     d])
                tail_w = P_a_cam @ Ra + cam_pos_a
                # Destination expressed in cam_a frame too (so arrows
                # show the apparent 3D motion as the source-camera sees
                # it). We keep this in cam_a's frame intentionally.
                P_b_cam = np.array([(u0 + du - cx) * d_dest / fx,
                                     (v0 + dv - cy) * d_dest / fy,
                                     d_dest])
                head_w = P_b_cam @ Ra + cam_pos_a
                tails.append(tail_w)
                heads.append(head_w)

                # Color the arrow by its 3D direction:
                # X red, Y green, Z blue, with magnitude as brightness.
                vec = head_w - tail_w
                mag = float(np.linalg.norm(vec))
                if mag < 1e-9:
                    colors.append((0.5, 0.5, 0.5))
                else:
                    nx, ny, nz = vec / mag
                    colors.append((abs(nx), abs(ny), abs(nz)))

        if tails:
            arrows_per_frame.append((
                np.array(tails, dtype=np.float32),
                np.array(heads, dtype=np.float32),
                np.array(colors, dtype=np.float32),
            ))
        else:
            arrows_per_frame.append((np.zeros((0, 3)), np.zeros((0, 3)),
                                     np.zeros((0, 3))))

    return arrows_per_frame, n - 1


# ---------- Pygame orbit-camera viewer ----------
class OrbitCamera:
    def __init__(self, target, distance):
        self.target = np.array(target, dtype=float)
        self.distance = float(distance)
        self.yaw   = 0.0
        self.pitch = 0.3
        self.fov_y_deg = 60.0

    def position(self):
        cp = math.cos(self.pitch); sp = math.sin(self.pitch)
        cy = math.cos(self.yaw);   sy = math.sin(self.yaw)
        # World Y is DOWN -> camera "up" maps to -Y. Negative Z so the
        # default view is from the camera-at-placement side, making +X
        # in world (right at placement) appear screen-right.
        offset = np.array([cp * sy, -sp, -cp * cy]) * self.distance
        return self.target + offset

    def basis(self):
        """Return (pos, R) where R rotates world -> view (OpenCV-ish camera).
        Camera x = right, y = down (Y-down world too), z = forward."""
        pos = self.position()
        fwd = self.target - pos
        fwd /= np.linalg.norm(fwd) + 1e-9
        # World "up" is -Y (because world Y is down). Camera "down" is +Y.
        up_world = np.array([0.0, -1.0, 0.0])
        right = np.cross(fwd, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        cam_down = np.cross(fwd, right)  # right-handed: fwd x right = -up
        R = np.stack([right, cam_down, fwd], axis=0)
        return pos, R


def project(P_world, pos, R, fx, fy, cx, cy):
    """World point -> screen pixel + camera-frame z (for clipping/sort)."""
    p_cam = R @ (P_world - pos)
    z = p_cam[2]
    if z <= 0.05:
        return None, z
    u = fx * p_cam[0] / z + cx
    v = fy * p_cam[1] / z + cy
    return (u, v), z


def run_viewer(arrows_per_frame, n_pairs):
    pygame.init()
    pygame.display.set_caption("flow3d -- A/D scrub, drag orbit, scroll zoom, R recenter, Esc quit")
    W, H = 1280, 720
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Initial camera target = centroid of first non-empty frame's arrows
    centroid = np.zeros(3)
    for tails, heads, _ in arrows_per_frame:
        if len(tails):
            centroid = tails.mean(axis=0)
            break
    cam = OrbitCamera(target=centroid, distance=4.0)

    fov_y = math.radians(cam.fov_y_deg)
    fy = (H / 2) / math.tan(fov_y / 2)
    fx = fy
    cx, cy_ = W / 2, H / 2

    current = 0
    dragging = False
    panning = False
    last_pos = (0, 0)
    arrow_scale = 1.0
    show_axes = True

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if ev.key in (pygame.K_d, pygame.K_RIGHT):
                    current = min(n_pairs - 1, current + 1)
                if ev.key in (pygame.K_a, pygame.K_LEFT):
                    current = max(0, current - 1)
                if ev.key == pygame.K_e:
                    current = min(n_pairs - 1, current + 5)
                if ev.key == pygame.K_q:
                    current = max(0, current - 5)
                if ev.key == pygame.K_r:
                    tails, _, _ = arrows_per_frame[current]
                    if len(tails):
                        cam.target = tails.mean(axis=0)
                if ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    arrow_scale *= 1.2
                if ev.key == pygame.K_MINUS:
                    arrow_scale /= 1.2
                if ev.key == pygame.K_x:
                    show_axes = not show_axes
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        panning = True
                    else:
                        dragging = True
                    last_pos = ev.pos
                if ev.button == 4:
                    cam.distance *= 0.85
                if ev.button == 5:
                    cam.distance *= 1.18
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False; panning = False
            if ev.type == pygame.MOUSEMOTION:
                dx = ev.pos[0] - last_pos[0]
                dy = ev.pos[1] - last_pos[1]
                if dragging:
                    cam.yaw   += dx * 0.008
                    cam.pitch  = float(np.clip(cam.pitch + dy * 0.008, -1.4, 1.4))
                if panning:
                    _, R = cam.basis()
                    right = R[0]; down = R[1]
                    speed = cam.distance * 0.0015
                    cam.target -= right * dx * speed
                    cam.target -= down  * dy * speed
                last_pos = ev.pos

        screen.fill((20, 20, 25))
        pos, R = cam.basis()

        # World axes through target (small)
        if show_axes:
            base = cam.target
            axis_len = 0.5
            for axis, color in [(np.array([axis_len, 0, 0]), (220, 60, 60)),
                                (np.array([0, axis_len, 0]), (60, 220, 60)),
                                (np.array([0, 0, axis_len]), (60, 90, 220))]:
                a = project(base,           pos, R, fx, fy, cx, cy_)[0]
                b = project(base + axis,    pos, R, fx, fy, cx, cy_)[0]
                if a is not None and b is not None:
                    pygame.draw.line(screen, color, a, b, 2)

        # Arrows
        tails, heads, colors = arrows_per_frame[current]
        drawn = 0
        for i in range(len(tails)):
            tail = tails[i]
            head = tails[i] + (heads[i] - tails[i]) * arrow_scale
            a, za = project(tail, pos, R, fx, fy, cx, cy_)
            b, zb = project(head, pos, R, fx, fy, cx, cy_)
            if a is None or b is None:
                continue
            col = colors[i]
            c = (int(255 * col[0]), int(255 * col[1]), int(255 * col[2]))
            pygame.draw.line(screen, c, a, b, 1)
            # Small dot at the tail to mark the scene point
            pygame.draw.circle(screen, (180, 180, 180), (int(a[0]), int(a[1])), 1)
            drawn += 1

        # HUD
        lines = [
            f"frame pair {current} / {n_pairs - 1}   arrows={drawn}/{len(tails)}",
            f"cam: dist={cam.distance:.2f}  yaw={math.degrees(cam.yaw):.0f}  pitch={math.degrees(cam.pitch):.0f}",
            f"target=({cam.target[0]:+.2f}, {cam.target[1]:+.2f}, {cam.target[2]:+.2f})",
            f"arrow_scale={arrow_scale:.2f}   (+/- to change, x toggle axes)",
            "A/D step, Q/E jump 5, R recenter, Esc quit  |  axis colors: X red, Y green, Z blue",
        ]
        for i, t in enumerate(lines):
            surf = font.render(t, True, (220, 220, 220))
            screen.blit(surf, (8, 6 + i * 16))
        pygame.display.flip()
        clock.tick(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", required=True,
                    help="event name (subdir of --events-dir)")
    ap.add_argument("--grid", type=int, default=32,
                    help="pixel-grid stride for arrow samples (default 32)")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--min-flow-px", type=float, default=0.8)
    ap.add_argument("--max-rms-px", type=float, default=100.0)
    args = ap.parse_args()

    event_dir = Path(args.events_dir) / args.event
    print(f"Computing 3D arrows for {args.event} ...")
    arrows, n_pairs = compute_arrows(event_dir, args)
    nonempty = sum(1 for t, _, _ in arrows if len(t) > 0)
    print(f"  {n_pairs} pairs total, {nonempty} with arrows.")
    print("Starting pygame viewer...")
    run_viewer(arrows, n_pairs)


if __name__ == "__main__":
    main()

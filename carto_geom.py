"""Geometry core for Mei Cartographer -- ported from reconstruct-scene.py
and continue-pose-vo.py into one importable module (the originals are
hyphenated scripts with main()).

Camera convention (same as the SLAM pipeline):
    P_cam = R(yaw,pitch) @ P_world + t          # R = world->cam, no roll
    cam_pos_world = -R.T @ t
    +X world = right, +Y = DOWN, +Z = forward/into-scene
Pixel:  u = fx * Xc/Zc + cx,   v = fy * Yc/Zc + cy

Pure numpy except flow_depth (cv2, lazy import) so the camera math is
testable anywhere; cv2 is only needed at Windows runtime.
"""
from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------

def R_no_roll(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rp @ Ry


def intrinsics(focal_px: float, W: int, H: int) -> np.ndarray:
    return np.array([[focal_px, 0, W / 2.0],
                     [0, focal_px, H / 2.0],
                     [0, 0, 1.0]])


def focal_from_fov(fov_deg: float, W: int) -> float:
    """Horizontal-FOV -> focal in px. OW2 default FOV is ~103 (hipfire)."""
    return (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def cam_pos(yaw, pitch, t) -> np.ndarray:
    return -R_no_roll(yaw, pitch).T @ np.asarray(t, float)


def project(world: np.ndarray, yaw, pitch, t, K):
    R = R_no_roll(yaw, pitch)
    pc = world @ R.T + t
    z = np.where(np.abs(pc[:, 2]) < 1e-3, 1e-3, pc[:, 2])
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def relative_pose(yaw_a, pitch_a, t_a, yaw_b, pitch_b, t_b):
    Ra, Rb = R_no_roll(yaw_a, pitch_a), R_no_roll(yaw_b, pitch_b)
    R_rel = Rb @ Ra.T
    t_rel = np.asarray(t_b, float) - R_rel @ np.asarray(t_a, float)
    return R_rel, t_rel


# ---------------------------------------------------------------------
# Pose refinement (PnP, no roll) -- needs scipy
# ---------------------------------------------------------------------

def pnp_no_roll(world, image, K, init, max_nfev=80):
    from scipy.optimize import least_squares

    def resid(p):
        yaw, pitch, tx, ty, tz = p
        if tz < 0.02:
            return np.full(image.size, 1e3)
        proj, _ = project(world, yaw, pitch, np.array([tx, ty, tz]), K)
        return (proj - image).ravel()
    try:
        r = least_squares(resid, init, method="lm", max_nfev=max_nfev)
        return r.x, float(np.sqrt(np.mean(r.fun ** 2))), True
    except Exception:
        return np.asarray(init, float), float("inf"), False


# ---------------------------------------------------------------------
# Auto-calibration: measure mouse_rad_per_px and move_speed from
# live gameplay -- no hard-coded values required.
# ---------------------------------------------------------------------

def calibrate_mouse_rad(frames, mouse_dxs, K):
    """Estimate yaw_rad per mouse_px from consecutive frame pairs.

    Method: dominant horizontal optical-flow shift dx_flow on each pair
    relates to yaw change as:
        dx_flow ~ -focal_x * tan(dyaw) ~ -focal_x * dyaw   (small angle)
    so dyaw = -mean(dx_flow) / focal_x.
    We compare to the commanded mouse_dx to get the scale.

    frames    : list of BGR ndarray (grabbed while spinning)
    mouse_dxs : list of mouse delta-x values (one per inter-frame gap)
    K         : 3x3 camera matrix
    Returns mouse_rad_per_px or None if too noisy."""
    import cv2
    fx = K[0, 0]
    ratios = []
    for i in range(len(frames) - 1):
        if abs(mouse_dxs[i]) < 5:
            continue                             # skip nearly-still frames
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, 0.5, 3, 21, 3, 7, 1.5, 0)
        # Median horizontal flow (centre strip, avoids border effects)
        H, W = g1.shape
        strip = flow[H // 4:3 * H // 4, W // 8:7 * W // 8, 0]
        dx_flow = float(np.median(strip))
        # dyaw from flow; positive mouse_dx = look right = negative flow
        dyaw_flow = -dx_flow / fx
        dyaw_cmd  = mouse_dxs[i]                # in px (sign convention: right = +)
        ratios.append(dyaw_flow / dyaw_cmd)
    if len(ratios) < 3:
        return None
    med = float(np.median(ratios))
    std = float(np.std(ratios))
    if std / max(abs(med), 1e-9) > 0.3:         # too noisy
        return None
    return med


def calibrate_move_speed(frames_walk, depth_maps, dt):
    """Estimate move_speed (m/s) from a forward-walk segment.

    Method: successive depth maps give the scene translation via the
    change in the median near-depth (ground plane / walls in front).
    For a pure forward walk, near-depth increases by ~speed*dt per tick.

    Returns speed_m_s or None."""
    if len(depth_maps) < 2:
        return None
    deltas = []
    for i in range(len(depth_maps) - 1):
        d0 = depth_maps[i];   d1 = depth_maps[i + 1]
        ok0 = np.isfinite(d0) & (d0 > 0.3) & (d0 < 20)
        ok1 = np.isfinite(d1) & (d1 > 0.3) & (d1 < 20)
        if ok0.sum() < 100 or ok1.sum() < 100:
            continue
        m0 = float(np.percentile(d0[ok0], 20))   # near-depth (close objects)
        m1 = float(np.percentile(d1[ok1], 20))
        dd = m1 - m0                              # negative = moving closer
        if -5 < dd < 0:                           # sanity: moved <5m this tick
            deltas.append(-dd / dt)
    if len(deltas) < 3:
        return None
    return float(np.median(deltas))


# ---------------------------------------------------------------------
# Motion prior from COMMANDED inputs (the live advantage over the
# offline pipeline -- we know what we pressed)
# ---------------------------------------------------------------------

def predict_pose(prev, held_keys, mouse_dx, mouse_dy, dt, cfg):
    """Dead-reckon the next pose from the inputs we just issued.
    `prev` = (yaw, pitch, t). cfg has mouse_rad_per_px, move_speed (m/s),
    pitch_rad_per_px. Returns (yaw, pitch, t) -- a SEED for VO refine."""
    yaw, pitch, t = prev
    yaw = yaw + mouse_dx * cfg["mouse_rad_per_px"]
    pitch = float(np.clip(pitch + mouse_dy * cfg["pitch_rad_per_px"],
                          -1.4, 1.4))
    # Ground-plane move from WASD, in world frame, relative to yaw.
    fwd = (("w" in held_keys) - ("s" in held_keys))
    strafe = (("d" in held_keys) - ("a" in held_keys))
    if fwd or strafe:
        speed = cfg["move_speed"] * dt
        # world forward (yaw) on the X-Z ground plane (+Y is down)
        wx = math.sin(yaw) * fwd + math.cos(yaw) * strafe
        wz = math.cos(yaw) * fwd - math.sin(yaw) * strafe
        dpos = np.array([wx, 0.0, wz]) * speed
        cp = cam_pos(yaw, pitch, t) + dpos
        t = -R_no_roll(yaw, pitch) @ cp
    return yaw, pitch, np.asarray(t, float)


# ---------------------------------------------------------------------
# Depth: optical-flow triangulation (no depth model needed) + DA fuse
# ---------------------------------------------------------------------

def flow_depth(fa, fb, R_rel, t_rel, K, min_flow_px=1.5, max_depth=60.0):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"cv2 unavailable (Windows runtime only): {e}") from e
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
             & (np.hypot(flow[..., 0], flow[..., 1]) > min_flow_px))
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
    """Scale DA relative-depth to the metric flow depth, fill flow holes."""
    ok = np.isfinite(flow_d) & np.isfinite(da_rel) & (da_rel > 1e-6)
    if int(ok.sum()) < 200:
        return flow_d.copy()
    s = float(np.median(flow_d[ok] / da_rel[ok]))
    out = flow_d.copy()
    fill = ~np.isfinite(out) & np.isfinite(da_rel)
    out[fill] = (da_rel * s)[fill]
    return out


# ---------------------------------------------------------------------
# Unproject depth -> world points (+ pixel uv for feature/colour lookup)
# ---------------------------------------------------------------------

def unproject(depth_m, yaw, pitch, t, K, step=8, z_min=0.2, z_max=50.0):
    H, W = depth_m.shape
    R = R_no_roll(yaw, pitch)
    cp = cam_pos(yaw, pitch, t)
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    V, U = np.meshgrid(ys, xs, indexing="ij")
    D = depth_m[V, U]
    m = np.isfinite(D) & (D > z_min) & (D < z_max)
    u = U[m].astype(np.float32); v = V[m].astype(np.float32); d = D[m]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    P_cam = np.stack([x, y, d], axis=-1)
    P_world = P_cam @ R + cp                 # world = R.T@P_cam + cam_pos
    return P_world.astype(np.float32), np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------
# Wall bootstrap (spawn ice wall -> initial pose + metric scale)
# ---------------------------------------------------------------------

def wall_world_corners(orientation, mei, ling, zhou):
    """Origin at wall front-face bottom-center, +X width, +Y down,
    +Z into wall. Returns TL, TR, BR, BL."""
    w = mei if orientation == "mei" else ling
    return np.array([[-w / 2, -zhou, 0.0], [+w / 2, -zhou, 0.0],
                     [+w / 2, 0, 0.0], [-w / 2, 0, 0.0]])


def detect_mei_wall_da_scale(frame_bgr, da_relative,
                              wall_height_m: float = 2.0):
    """Detect the Mei ice wall in `frame_bgr` by its blue HSV colour,
    find its pixel bounding box, then fit the DA scale so that the
    wall's pixel height maps to `wall_height_m` via the pinhole model.

    Returns da_scale (float) or None if the wall isn't clearly visible.
    The wall colour: hue 90-130 (cyan-blue), moderate-high S and V."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([90, 50, 60], np.uint8),
                       np.array([130, 255, 255], np.uint8))
    # keep only the largest connected blue region
    nl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nl < 2:
        return None
    best = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    area = stats[best, cv2.CC_STAT_AREA]
    if area < 2000:                        # too small, probably noise
        return None
    x0 = stats[best, cv2.CC_STAT_LEFT]
    y0 = stats[best, cv2.CC_STAT_TOP]
    h_px = stats[best, cv2.CC_STAT_HEIGHT]
    x1 = x0 + stats[best, cv2.CC_STAT_WIDTH]
    if h_px < 30:
        return None
    # median DA depth across the wall blob
    blob_mask = (lbl == best)
    da_vals = da_relative[blob_mask]
    da_vals = da_vals[np.isfinite(da_vals) & (da_vals > 1e-6)]
    if len(da_vals) < 100:
        return None
    da_med = float(np.median(da_vals))
    # Metric depth d = da_scale * da_med; from pinhole:
    #   h_px = fy * wall_height_m / d   =>   d = fy * wall_height_m / h_px
    H, W = frame_bgr.shape[:2]
    fy = (H / 2.0) / np.tan(np.radians(74) / 2)  # OW2 ~74 deg vertical FOV
    d_expected = fy * wall_height_m / h_px
    scale = d_expected / da_med
    return float(scale)


def calibrate_scale(da_depth, corners_px, corners_world, yaw, pitch, t, K):
    R = R_no_roll(yaw, pitch)
    cam_pts = corners_world @ R.T + t
    ze = cam_pts[:, 2]
    if (ze <= 0).any():
        return None, None
    H, W = da_depth.shape
    vals = []
    for (u, v) in corners_px:
        ui, vi = int(round(u)), int(round(v))
        vals.append(da_depth[vi, ui] if (0 <= ui < W and 0 <= vi < H) else np.nan)
    vals = np.array(vals); ok = np.isfinite(vals) & (vals > 1e-6)
    if ok.sum() < 2:
        return None, None
    da, ze = vals[ok], ze[ok]
    s = float((da * ze).sum() / (da * da).sum())
    return s, float(np.sqrt(np.mean((s * da - ze) ** 2)))


# ---------------------------------------------------------------------
# Map structures: voxel-hashed cloud + 2D floor coverage grid
# ---------------------------------------------------------------------

class VoxelCloud:
    """Sparse occupied-voxel set with running mean colour/normal accum."""
    def __init__(self, voxel: float = 0.15):
        self.voxel = voxel
        self.cells: dict[tuple, list] = {}      # ijk -> [sumX,sumY,sumZ,n]

    def add(self, pts: np.ndarray):
        if len(pts) == 0:
            return 0
        keys = np.floor(pts / self.voxel).astype(np.int64)
        added = 0
        for p, k in zip(pts, keys):
            kk = (int(k[0]), int(k[1]), int(k[2]))
            c = self.cells.get(kk)
            if c is None:
                self.cells[kk] = [p[0], p[1], p[2], 1]; added += 1
            else:
                c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; c[3] += 1
        return added

    def points(self) -> np.ndarray:
        if not self.cells:
            return np.zeros((0, 3), np.float32)
        return np.array([[c[0] / c[3], c[1] / c[3], c[2] / c[3]]
                         for c in self.cells.values()], np.float32)

    def clear(self):
        self.cells.clear()

    def __len__(self):
        return len(self.cells)


class CoverageGrid:
    """2D floor-plan occupancy of where the camera has LOOKED (footprints
    of unprojected points) + visited camera cells + per-cell view-yaw
    bins, so the explorer can find frontiers and unobserved directions."""
    def __init__(self, cell: float = 0.5, yaw_bins: int = 8):
        self.cell = cell
        self.yaw_bins = yaw_bins
        self.seen: set[tuple] = set()          # floor cells with geometry
        self.visited: set[tuple] = set()       # cells the camera stood in
        self.view: dict[tuple, set] = {}       # cell -> set of yaw bins looked

    def _c(self, x, z):
        return (int(math.floor(x / self.cell)), int(math.floor(z / self.cell)))

    def observe(self, world_pts, cam_xz, yaw):
        for p in world_pts[::4]:
            self.seen.add(self._c(p[0], p[2]))
        vc = self._c(cam_xz[0], cam_xz[1])
        self.visited.add(vc)
        b = int(((yaw % (2 * math.pi)) / (2 * math.pi)) * self.yaw_bins) % self.yaw_bins
        self.view.setdefault(vc, set()).add(b)

    def frontier_cells(self):
        """Seen floor cells that have an unseen 4-neighbour = exploration
        frontier."""
        fr = []
        for (i, j) in self.seen:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (i + di, j + dj) not in self.seen:
                    fr.append((i, j)); break
        return fr

    def unobserved_yaws(self, cam_xz):
        vc = self._c(cam_xz[0], cam_xz[1])
        done = self.view.get(vc, set())
        return [b for b in range(self.yaw_bins) if b not in done]

    def stats(self):
        return {"seen_cells": len(self.seen), "visited_cells": len(self.visited),
                "frontier_cells": len(self.frontier_cells())}

    def reset(self):
        self.seen.clear()
        self.visited.clear()
        self.view.clear()

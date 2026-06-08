#!/usr/bin/env python3
"""Pygame PLY point-cloud viewer (works under WSLg where Open3D's EGL
renderer fails), with optional translucent camera-image planes.

Usage:
  ./view-ply.py mei_walls/events/<event>/pointcloud.ply
  ./view-ply.py --event <event>          # loads that event's pointcloud.ply
                                           # + poses + clip for image planes

Controls:
  drag         orbit
  shift+drag   pan
  scroll       zoom
  R            recenter on cloud centroid
  B            cycle background
  +/-          point size
  I            toggle translucent camera-image planes (needs --event)
  F            toggle camera frustums
  J            cameras: nearest-to-view only / evenly-subsampled
  [ / ]        fewer / more cameras shown
  O            toggle boundary-shell-only (cull interior voxels)
  H            toggle floor/wall/ceiling recolor (by per-cell outward normal)
  Esc          quit
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pygame
except ImportError:
    raise SystemExit("pip install pygame")

try:
    import cv2
except ImportError:
    cv2 = None  # image planes need cv2 for the perspective warp


def rotation_world_to_cam(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rp @ Ry


def read_ply(path: Path):
    with open(path, "r") as f:
        if not f.readline().startswith("ply"):
            raise SystemExit(f"{path} not a ply")
        n = 0; props = []; in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise SystemExit("unexpected EOF in header")
            tok = line.split()
            if tok and tok[0] == "element":
                in_vertex = (tok[1] == "vertex")
                if in_vertex:
                    n = int(tok[2])
            elif tok and tok[0] == "property" and in_vertex:
                props.append(tok[-1])
            if line.startswith("end_header"):
                break
        idx = {name: i for i, name in enumerate(props)}
        has_rgb = all(c in idx for c in ("red", "green", "blue"))
        pts = np.empty((n, 3), dtype=np.float32)
        cols = np.empty((n, 3), dtype=np.uint8) if has_rgb else None
        for i in range(n):
            parts = f.readline().split()
            pts[i] = (float(parts[idx["x"]]), float(parts[idx["y"]]),
                      float(parts[idx["z"]]))
            if has_rgb:
                cols[i] = (int(parts[idx["red"]]), int(parts[idx["green"]]),
                           int(parts[idx["blue"]]))
        if cols is None:
            cols = np.full((n, 3), 200, dtype=np.uint8)
        return pts, cols


def load_cameras(event_dir: Path, max_load=400):
    """Load per-frame (cam_pos, R_world2cam, frame_bgr) for frames that
    have a converged pose. Returns (list_of_cams, K_orig, (W,H)) or None."""
    pose_p = event_dir / "pose.json"
    clip_p = event_dir / "clip.mp4"
    if not (pose_p.exists() and clip_p.exists()) or cv2 is None:
        return None
    pose = json.loads(pose_p.read_text())
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])
    frames_meta = [f for f in pose["frames"] if f.get("converged")]
    # also fold in continued VO poses if present
    cont = event_dir / "continued_pose_vo.json"
    if cont.exists():
        cj = json.loads(cont.read_text())
        frames_meta += [f for f in cj["frames"] if f.get("converged")]
    cap = cv2.VideoCapture(str(clip_p))
    raw = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        raw.append(fr)
    cap.release()
    cams = []
    for f in frames_meta:
        i = f["frame_idx"]
        if i >= len(raw):
            continue
        R = rotation_world_to_cam(f["yaw_rad"], f["pitch_rad"])
        cam_pos = np.array(f["camera_pos_world"], dtype=np.float32)
        # downsample frame for fast warping
        small = cv2.resize(raw[i], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        cams.append({"pos": cam_pos, "R": R, "img": small})
        if len(cams) >= max_load:
            break
    return cams, K, (W, H)


class OrbitCamera:
    def __init__(self, target, distance):
        self.target = np.asarray(target, dtype=np.float32)
        self.distance = float(distance)
        self.yaw = 0.4; self.pitch = 0.3

    def basis(self):
        cp = math.cos(self.pitch); sp = math.sin(self.pitch)
        cy = math.cos(self.yaw);   sy = math.sin(self.yaw)
        offset = np.array([cp * sy, -sp, -cp * cy], dtype=np.float32) * self.distance
        pos = self.target + offset
        fwd = self.target - pos
        fwd /= np.linalg.norm(fwd) + 1e-9
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return pos, np.stack([right, down, fwd], axis=0)


def project(P_world, pos, R, fproj, cx, cy):
    pc = (P_world - pos) @ R.T
    z = pc[:, 2]
    u = fproj * pc[:, 0] / np.maximum(z, 1e-3) + cx
    v = fproj * pc[:, 1] / np.maximum(z, 1e-3) + cy
    return np.stack([u, v], axis=-1), z


def image_plane_corners(cam, K_orig, W_img, H_img, dist):
    """4 world-space corners (TL,TR,BR,BL) of the image plane at `dist`."""
    fx, fy = K_orig[0, 0], K_orig[1, 1]
    cx, cy = K_orig[0, 2], K_orig[1, 2]
    R = cam["R"]; pos = cam["pos"]
    out = []
    for (u, v) in [(0, 0), (W_img, 0), (W_img, H_img), (0, H_img)]:
        d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0]) * dist
        out.append(d @ R + pos)  # cam->world
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--event", default=None,
                    help="event name under mei_walls/events; loads its "
                         "pointcloud.ply + poses + clip for camera planes")
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--plane-dist", type=float, default=1.0,
                    help="distance (ow-m) to place camera image planes")
    ap.add_argument("--plane-alpha", type=float, default=0.45)
    ap.add_argument("--n-cameras", type=int, default=8,
                    help="number of camera planes to show (subsampled)")
    ap.add_argument("--voxel-size", type=float, default=0.05,
                    help="voxel grid spacing used to build the cloud; needed "
                         "for boundary-shell toggle (default 0.05 — matches "
                         "reconstruct-scene)")
    ap.add_argument("--shell-bin", type=int, default=3,
                    help="coarse-bin factor used by the boundary-shell "
                         "toggle (O key). Default 3 -> 3*voxel-size cells. "
                         "Larger = bigger cells -> more interior points "
                         "culled, leaving only the outermost shell.")
    args = ap.parse_args()

    event_dir = None
    if args.event:
        event_dir = Path(args.events_dir) / args.event
        if args.path is None:
            args.path = str(event_dir / "pointcloud.ply")
    if args.path is None:
        sys.exit("give a .ply path or --event")

    p = Path(args.path)
    pts, cols = read_ply(p)
    print(f"loaded {len(pts):,} points from {p}")
    if not args.rgb:
        cols = cols[:, ::-1].copy()

    # Boundary-shell mask: bin points into a coarser grid (shell_voxel) so
    # dense regions group into solid cells, then mark a point as "boundary"
    # iff its coarse cell has at least one unoccupied 6-neighbor. Lets us
    # cull interior points of dense surfaces (wall face, ground, etc.) so
    # the scene's outline is easier to see.
    vsize = args.voxel_size
    shell_vsize = vsize * args.shell_bin
    keys = np.floor(pts / shell_vsize).astype(np.int64)
    kmin = keys.min(axis=0)
    ks = keys - kmin
    P1 = int(ks[:, 1].max()) + 4
    P2 = int(ks[:, 2].max()) + 4
    hashes = ks[:, 0] * (P1 * P2) + ks[:, 1] * P2 + ks[:, 2]
    occ_sorted = np.unique(hashes)
    # Per-coarse-cell boundary flag AND outward normal (sum of unit vectors
    # pointing toward each empty neighbor). Normal lets us classify each
    # cell as floor / wall / ceiling.
    cell_is_boundary = np.zeros(len(occ_sorted), dtype=bool)
    ck = np.stack([occ_sorted // (P1 * P2),
                   (occ_sorted // P2) % P1,
                   occ_sorted % P2], axis=1)
    n_acc = np.zeros((len(occ_sorted), 3), dtype=np.float32)
    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0),
                       (0, -1, 0), (0, 1, 0),
                       (0, 0, -1), (0, 0, 1)):
        nh = ((ck[:, 0] + dx) * (P1 * P2)
              + (ck[:, 1] + dy) * P2 + (ck[:, 2] + dz))
        idx = np.searchsorted(occ_sorted, nh)
        idx_c = np.clip(idx, 0, len(occ_sorted) - 1)
        present = (idx < len(occ_sorted)) & (occ_sorted[idx_c] == nh)
        empty = ~present
        cell_is_boundary |= empty
        n_acc[empty, 0] += dx
        n_acc[empty, 1] += dy
        n_acc[empty, 2] += dz
    norms = np.linalg.norm(n_acc, axis=1) + 1e-6
    nrm = n_acc / norms[:, None]   # (cells, 3) outward unit normal
    # World convention: +Y is DOWN. So an upward-pointing outward normal
    # has ny < 0 -> floor; ny > 0 (points down) -> ceiling; else wall.
    floor_cell = cell_is_boundary & (nrm[:, 1] < -0.55)

    pt_cell_idx = np.searchsorted(occ_sorted, hashes)
    boundary_mask = cell_is_boundary[pt_cell_idx]
    # Two-class scheme: "floor" = outward normal points up; everything else
    # boundary (vertical normals or downward-pointing floater overhangs) is
    # "wall". Per-point colors flipped to BGR unless --rgb is set so they
    # render correctly through the existing pipeline.
    FLOOR_RGB = np.array([170, 145, 105], dtype=np.uint8)  # warm tan
    WALL_RGB  = np.array([115, 145, 180], dtype=np.uint8)  # cool blue-gray
    rgb_to_bgr = slice(None, None, -1) if not args.rgb else slice(None)
    fclass = floor_cell[pt_cell_idx]
    wclass = boundary_mask & ~fclass
    normal_cols = np.full_like(cols, 90)
    normal_cols[fclass] = FLOOR_RGB[rgb_to_bgr]
    normal_cols[wclass] = WALL_RGB[rgb_to_bgr]

    n_bnd  = int(boundary_mask.sum())
    n_flr  = int(fclass.sum())
    n_wall = int(wclass.sum())
    print(f"boundary shell ({shell_vsize:.2f} m bin): "
          f"{n_bnd:,} / {len(pts):,} points ({100.0 * n_bnd / max(1, len(pts)):.0f}% — O toggles)")
    print(f"normal classes: floor={n_flr:,} wall={n_wall:,}  (H toggles)")

    # If no --event given, infer it from the ply's folder (pose.json sibling).
    if event_dir is None and (p.parent / "pose.json").exists():
        event_dir = p.parent
        print(f"auto-detected event dir: {event_dir}")

    cams = K_orig = img_wh = None
    if event_dir is not None:
        loaded = load_cameras(event_dir)
        if loaded:
            cams, K_orig, img_wh = loaded
            print(f"loaded {len(cams)} camera poses for image planes "
                  "(press F=frustums, I=images)")
        else:
            print("no poses/clip (or cv2 missing) -- image planes disabled")
    else:
        print("no event context -- image planes disabled "
              "(use --event or point at a ply inside an event dir)")

    pygame.init()
    pygame.display.set_caption(f"{p.name}  I/F=cams O=shell H=normal drag=orbit Esc=quit")
    WIN = args.window
    screen = pygame.display.set_mode((WIN, WIN))
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    fproj = (WIN / 2) / math.tan(math.radians(55.0) / 2)
    cx = cy = WIN / 2

    centroid = pts.mean(axis=0)
    spread = float(np.percentile(np.linalg.norm(pts - centroid, axis=1), 75)) * 2.5
    cam = OrbitCamera(centroid, max(spread, 1.0))
    backgrounds = [(18, 18, 22), (130, 130, 130), (245, 245, 245), (0, 0, 0)]
    bg_idx = 0; splat = 2
    show_images = False; show_frustums = bool(cams)
    n_cams = args.n_cameras
    nearest_mode = False   # show only camera(s) aligned with current view
    boundary_only = False  # show only boundary-shell voxels (toggle O)
    normal_mode = False    # H: recolor by floor/wall normal class

    # precompute each camera's world-space forward for nearest-mode
    if cams:
        cam_fwds = np.array([c["R"][2, :] for c in cams])  # R.T@[0,0,1] = R[2]
    dragging = panning = False; last = (0, 0)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: running = False
                elif ev.key == pygame.K_r: cam.target = pts.mean(axis=0)
                elif ev.key == pygame.K_b: bg_idx = (bg_idx + 1) % len(backgrounds)
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS): splat = min(3, splat + 1)
                elif ev.key == pygame.K_MINUS: splat = max(1, splat - 1)
                elif ev.key == pygame.K_i and cams: show_images = not show_images
                elif ev.key == pygame.K_f and cams: show_frustums = not show_frustums
                elif ev.key == pygame.K_j and cams: nearest_mode = not nearest_mode
                elif ev.key == pygame.K_LEFTBRACKET: n_cams = max(1, n_cams - 2)
                elif ev.key == pygame.K_RIGHTBRACKET: n_cams = n_cams + 2
                elif ev.key == pygame.K_o: boundary_only = not boundary_only
                elif ev.key == pygame.K_h: normal_mode = not normal_mode
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT: panning = True
                    else: dragging = True
                    last = ev.pos
                elif ev.button == 4: cam.distance *= 0.85
                elif ev.button == 5: cam.distance *= 1.18
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = panning = False
            elif ev.type == pygame.MOUSEMOTION:
                dx = ev.pos[0] - last[0]; dy = ev.pos[1] - last[1]
                if dragging:
                    cam.yaw += dx * 0.008
                    cam.pitch = float(np.clip(cam.pitch + dy * 0.008, -1.4, 1.4))
                if panning:
                    _, R = cam.basis(); sp = cam.distance * 0.0015
                    cam.target -= R[0] * dx * sp; cam.target -= R[1] * dy * sp
                last = ev.pos

        bg = backgrounds[bg_idx]
        screen.fill(bg)
        pos, R = cam.basis()

        # --- point selection: O hollows out the interior, H recolors by
        # floor/wall/ceiling normal class. They compose freely.
        active_cols = normal_cols if normal_mode else cols
        if boundary_only:
            draw_pts = pts[boundary_mask]
            draw_cols = active_cols[boundary_mask]
        else:
            draw_pts = pts
            draw_cols = active_cols
        if draw_pts is not None:
            uv, z = project(draw_pts, pos, R, fproj, cx, cy)
            vis = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < WIN) & (uv[:, 1] >= 0) & (uv[:, 1] < WIN)
        else:
            vis = np.zeros(0, dtype=bool)
        if vis.any():
            order = np.argsort(-z[vis])
            px = uv[vis, 0][order].astype(np.int32)
            py = uv[vis, 1][order].astype(np.int32)
            cc = draw_cols[vis][order]  # type: ignore
            arr = pygame.surfarray.pixels3d(screen)
            arr[px, py] = cc
            if splat >= 2:
                for ddx, ddy in ((1, 0), (0, 1), (1, 1)):
                    arr[np.clip(px + ddx, 0, WIN - 1), np.clip(py + ddy, 0, WIN - 1)] = cc
            del arr

        # --- camera image planes / frustums ---
        if cams and (show_images or show_frustums):
            if nearest_mode:
                # cameras whose forward best aligns with the orbit view dir
                align = cam_fwds @ R[2]   # R[2] = orbit forward
                sel = np.argsort(-align)[:max(1, n_cams)]
            else:
                sel = np.linspace(0, len(cams) - 1, min(n_cams, len(cams))).astype(int)
            W_img, H_img = img_wh
            for ci in sel:
                c = cams[ci]
                quad_w = image_plane_corners(c, K_orig, W_img, H_img, args.plane_dist)
                quad_uv, quad_z = project(quad_w, pos, R, fproj, cx, cy)
                pos_uv, pos_z = project(c["pos"][None, :], pos, R, fproj, cx, cy)
                if (quad_z <= 0.05).any():
                    continue
                qpix = quad_uv.astype(np.int32)
                if show_frustums:
                    # lines: cam center -> 4 corners, + quad outline
                    if pos_z[0] > 0.05:
                        cpt = (int(pos_uv[0, 0]), int(pos_uv[0, 1]))
                        for k in range(4):
                            pygame.draw.line(screen, (90, 200, 255), cpt,
                                             (qpix[k, 0], qpix[k, 1]), 1)
                    pygame.draw.lines(screen, (90, 200, 255), True,
                                      [tuple(qpix[k]) for k in range(4)], 1)
                if show_images:
                    # warp the (downsampled) frame onto the projected quad
                    src = np.float32([[0, 0], [c["img"].shape[1], 0],
                                      [c["img"].shape[1], c["img"].shape[0]],
                                      [0, c["img"].shape[0]]])
                    dst = quad_uv.astype(np.float32)
                    M = cv2.getPerspectiveTransform(src, dst)
                    warp = cv2.warpPerspective(c["img"], M, (WIN, WIN))
                    mask = cv2.warpPerspective(
                        np.full(c["img"].shape[:2], 255, np.uint8), M, (WIN, WIN))
                    # alpha-blend warped (BGR) onto screen (RGB)
                    arr = pygame.surfarray.pixels3d(screen)  # (W,H,3) RGB
                    warp_rgb = warp[:, :, ::-1]               # BGR->RGB
                    warp_t = np.transpose(warp_rgb, (1, 0, 2))  # (W,H,3)
                    m_t = (np.transpose(mask, (1, 0)) > 0)
                    a = args.plane_alpha
                    arr[m_t] = (arr[m_t] * (1 - a) + warp_t[m_t] * a).astype(np.uint8)
                    del arr

        hud = [
            f"{p.name}  {len(pts):,} pts" + (f"  | {len(cams)} cams" if cams else ""),
            f"dist={cam.distance:.2f} yaw={math.degrees(cam.yaw):.0f} pitch={math.degrees(cam.pitch):.0f}",
        ]
        if cams:
            hud.append(f"I=images:{'on' if show_images else 'off'}  "
                       f"F=frustums:{'on' if show_frustums else 'off'}  "
                       f"J=nearest:{'on' if nearest_mode else 'off'}  "
                       f"[/]=cams:{n_cams}")
        hud.append(f"O=shell:{'on' if boundary_only else 'off'}  "
                   f"H=normal:{'on' if normal_mode else 'off'}  "
                   "drag=orbit shift+drag=pan scroll=zoom R=recenter B=bg Esc=quit")
        for i, t in enumerate(hud):
            screen.blit(font.render(t, True, (235, 235, 235)), (8, 6 + i * 16))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

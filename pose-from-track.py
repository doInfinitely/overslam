#!/usr/bin/env python3
"""Camera pose estimation from a tracked wall.

Reads <event-dir>/track.json (output of track-wall.py) and
calibration.json. Per frame, solves a 5-DOF nonlinear least-squares
problem for the camera's (yaw, pitch, tx, ty, tz) given the bbox
corners as the wall front-face corners and the wall dimensions as
known world coordinates.

Conventions
-----------
World frame: origin at wall front-face bottom-center.
  +X = along wall width (rightward when facing front of wall)
  +Y = down (camera convention; world up corresponds to -Y)
  +Z = into the wall (away from camera at placement time)

Camera frame: OpenCV. +X right, +Y down, +Z forward.

Rotation has no roll: R = R_pitch(p) @ R_yaw(y). Yaw rotates around
world Y (so positive yaw = camera turns right). Pitch rotates around
the yawed X (positive pitch = camera looks down).

Outputs <event-dir>/pose.json: per-frame
  {frame_idx, t_ms, yaw_rad, pitch_rad, tx, ty, tz, reproj_rms_px,
   converged, camera_pos_world}

Camera position in world = -R.T @ t.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    raise SystemExit("scipy required: pip install scipy")


def load_calibration(path: Path):
    """Read calibration.json. Returns dict of resolved values (with
    fallbacks for any field not yet filled in)."""
    cal = {}
    if path.exists():
        cal = json.loads(path.read_text())

    def get(keys, default=None):
        cur = cal
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur

    fov_deg = get(["game_settings", "fov_deg"], default=103.0)  # OW default
    focal_px = get(["camera", "focal_length_px"])  # None if not measured

    # Wall dimensions in OW-units; fall back to 1.0 placeholder per axis
    # if the user hasn't measured them yet (results are then in
    # placeholder units, not OW-m).
    mei  = get(["mei_wall", "length_ow_m"], 1.0)  # 1 Mei
    ling = get(["mei_wall", "depth_ow_m"],  1.0)  # 1 Ling
    zhou = get(["mei_wall", "height_ow_m"], 1.0)  # 1 Zhou

    return {
        "fov_deg": fov_deg,
        "focal_px": focal_px,
        "mei":  mei,
        "ling": ling,
        "zhou": zhou,
        "raw":  cal,
    }


def derive_focal_from_fov(W: int, fov_deg: float) -> float:
    """Horizontal FOV -> focal length in pixels (pinhole)."""
    return (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def build_K(W: int, H: int, focal_px: float) -> np.ndarray:
    return np.array([
        [focal_px, 0,        W / 2.0],
        [0,        focal_px, H / 2.0],
        [0,        0,        1.0    ],
    ])


def front_face_world_pts(orientation: str, cal) -> np.ndarray:
    """4 corners of the front face in world coords (TL, TR, BR, BL).
    The visible side at placement: width = mei if orientation=='mei',
    width = ling if orientation=='ling'. Height is always zhou.
    """
    if orientation == "mei":
        w = cal["mei"]
    elif orientation == "ling":
        w = cal["ling"]
    else:
        w = cal["mei"]  # default
    h = cal["zhou"]
    # World Y is DOWN; the wall extends from y=0 (base) to y=-h (top).
    return np.array([
        [-w / 2, -h, 0.0],  # TL
        [+w / 2, -h, 0.0],  # TR
        [+w / 2,  0, 0.0],  # BR
        [-w / 2,  0, 0.0],  # BL
    ])


def rotation_world_to_cam(yaw: float, pitch: float) -> np.ndarray:
    """No-roll: R = R_pitch @ R_yaw. Both rotate around camera-frame axes."""
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


def project(world_pts: np.ndarray, yaw, pitch, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project N world points to image plane. Returns (N, 2)."""
    R = rotation_world_to_cam(yaw, pitch)
    pts_cam = world_pts @ R.T + t          # (N, 3)
    z = pts_cam[:, 2]
    z = np.where(np.abs(z) < 1e-3, 1e-3, z)
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def residuals(params, world_pts, image_pts, K):
    yaw, pitch, tx, ty, tz = params
    # Penalize cameras that put the wall behind: tz must keep the
    # wall in front of the camera (we set wall at world z=0 and
    # camera at world z = -tz_world; in OpenCV camera frame, t is
    # the translation s.t. P_cam = R @ P_world + t, and tz here is
    # roughly the camera-to-wall distance along view axis -- so tz > 0).
    if tz < 0.1:
        return np.full(image_pts.size, 1e3)
    t = np.array([tx, ty, tz])
    return (project(world_pts, yaw, pitch, t, K) - image_pts).ravel()


def initial_guess(world_pts, image_pts, K, prev=None):
    """Pose initialization. If prev is given, use it. Otherwise estimate
    distance from apparent wall width."""
    if prev is not None:
        return np.array(prev, dtype=float)
    # Estimate tz from wall pixel width vs world width.
    wall_world_w = abs(world_pts[1, 0] - world_pts[0, 0])  # TR.x - TL.x
    wall_img_w   = abs(image_pts[1, 0] - image_pts[0, 0])  # TR.u - TL.u
    fx = K[0, 0]
    if wall_img_w < 1:
        tz0 = 5.0
    else:
        tz0 = wall_world_w * fx / wall_img_w
    # Center the camera on the wall horizontally; eye-height above base.
    return np.array([0.0, 0.0, 0.0, -wall_world_w * 0.3, tz0])


def solve_one_frame(world_pts, image_pts, K, prev=None):
    x0 = initial_guess(world_pts, image_pts, K, prev=prev)
    try:
        res = least_squares(
            residuals, x0,
            args=(world_pts, image_pts, K),
            method="lm", max_nfev=200,
        )
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        return res.x.tolist(), rms, bool(res.success)
    except Exception as e:
        return None, float("inf"), False


def process_event(event_dir: Path, cal, args):
    track_p = event_dir / "track.json"
    if not track_p.exists():
        return None, "no track.json (run track-wall.py first)"
    tj = json.loads(track_p.read_text())
    W, H = tj["frame_size"]
    orientation = tj.get("orientation", "mei")
    world_pts = front_face_world_pts(orientation, cal)

    focal_px = cal["focal_px"]
    if focal_px is None:
        focal_px = derive_focal_from_fov(W, cal["fov_deg"])
        focal_source = f"derived from FOV={cal['fov_deg']}deg"
    else:
        focal_source = "from calibration.json"
    K = build_K(W, H, focal_px)

    pose_frames = []
    prev = None
    failed = 0
    for fr in tj["frames"]:
        corners = np.array(fr["corners"], dtype=float)  # TL, TR, BR, BL
        if corners.shape != (4, 2):
            continue
        params, rms, ok = solve_one_frame(world_pts, corners, K, prev=prev)
        if params is None:
            failed += 1
            pose_frames.append({
                "frame_idx": fr["frame_idx"],
                "t_ms": fr["t_ms"],
                "converged": False,
                "reproj_rms_px": None,
            })
            continue
        yaw, pitch, tx, ty, tz = params
        # Camera position in world: -R.T @ t
        R = rotation_world_to_cam(yaw, pitch)
        cam_pos_world = (-R.T @ np.array([tx, ty, tz])).tolist()
        pose_frames.append({
            "frame_idx": fr["frame_idx"],
            "t_ms": fr["t_ms"],
            "yaw_rad":   float(yaw),
            "pitch_rad": float(pitch),
            "yaw_deg":   float(math.degrees(yaw)),
            "pitch_deg": float(math.degrees(pitch)),
            "t":         [float(tx), float(ty), float(tz)],
            "camera_pos_world": cam_pos_world,
            "reproj_rms_px": rms,
            "converged": ok,
        })
        # Warm-start next frame from this one if convergence was good
        if ok and rms < args.max_warmstart_rms_px:
            prev = params

    out = {
        "event_name": event_dir.name,
        "orientation": orientation,
        "intrinsics": {
            "focal_px": float(focal_px),
            "focal_source": focal_source,
            "image_size": [W, H],
        },
        "wall_world": {
            "mei":  cal["mei"],
            "ling": cal["ling"],
            "zhou": cal["zhou"],
            "note": "ow-units (placeholder 1.0 if not measured in calibration.json)",
        },
        "n_frames": len(pose_frames),
        "n_failed": failed,
        "frames": pose_frames,
    }
    (event_dir / "pose.json").write_text(json.dumps(out, indent=2))
    return out, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None,
                    help="single event name (default: all)")
    ap.add_argument("--calibration", default="./calibration.json")
    ap.add_argument("--max-warmstart-rms-px", type=float, default=20.0,
                    help="if a frame's reprojection RMS exceeds this we don't "
                         "warm-start the next frame from it (default 20 px)")
    args = ap.parse_args()

    cal = load_calibration(Path(args.calibration))
    print(f"Calibration: focal_px={cal['focal_px']}  "
          f"FOV={cal['fov_deg']}  wall=({cal['mei']}, {cal['ling']}, {cal['zhou']}) "
          f"(mei,ling,zhou)")
    if cal["focal_px"] is None:
        print("  (focal_px not measured; using FOV-derived default. "
              "Fill calibration.json for metric output.)")
    if cal["mei"] == 1.0 and cal["ling"] == 1.0 and cal["zhou"] == 1.0:
        print("  (wall dims not measured; pose output is in placeholder units, "
              "not OW-m. Fill calibration.json for true scale.)")

    root = Path(args.events_dir)
    subs = [root / args.event] if args.event else \
        sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"\nProcessing {len(subs)} event(s)...")
    for d in subs:
        out, err = process_event(d, cal, args)
        if err:
            print(f"  [skip] {d.name}: {err}")
            continue
        good = [f for f in out["frames"] if f["converged"]]
        if not good:
            print(f"  [done] {d.name}: 0 frames converged")
            continue
        rms_med = float(np.median([f["reproj_rms_px"] for f in good]))
        yaws  = [f["yaw_deg"]   for f in good]
        pitches = [f["pitch_deg"] for f in good]
        tzs   = [f["t"][2]      for f in good]
        print(f"  [done] {d.name}: {len(good)}/{out['n_frames']} frames "
              f"converged, median rms={rms_med:.1f}px  "
              f"yaw=({min(yaws):+.1f}..{max(yaws):+.1f})deg  "
              f"pitch=({min(pitches):+.1f}..{max(pitches):+.1f})deg  "
              f"tz=({min(tzs):.1f}..{max(tzs):.1f})ow-m")


if __name__ == "__main__":
    main()

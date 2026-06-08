#!/usr/bin/env python3
"""Align state vectors with per-frame camera motion.

For each event:
  - Load <event>/pose.json (output of pose-from-track.py).
  - Look up the event clip's wall-clock start (from event.json's
    clip_start_wall_ns) so we can map frame indices to wall-clock ns.
  - Slice the global state-vector npz at the matching bin range.
  - Convert the per-frame (yaw, pitch, camera_pos_world) trajectory
    into per-frame camera-local motion: (yaw_rate, pitch_rate,
    v_forward, v_right, v_up) all in [units]/sec.
  - Mask out frames where pose-from-track failed to converge or had
    high reprojection RMS.
  - Save per-event aligned pairs and a concatenated training file.

Camera-local motion convention:
  yaw_rate, pitch_rate : rad/sec (signed)
  v_forward            : along camera optical axis (positive = into scene)
  v_right              : along camera right axis
  v_up                 : along world up (= -camera Y, since cam Y is down)
The motion at frame t is computed as a forward difference (t -> t+1)
divided by the frame dt. Last frame's motion is filled with zeros and
masked.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


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


def camera_axes(yaw, pitch):
    """Return (forward, right, up) unit vectors in world coords for a
    camera with the given no-roll orientation. Camera frame in our
    convention: +X right, +Y down, +Z forward.
    World axes are: cam_x = R.T @ [1,0,0], etc.
    """
    R = rotation_world_to_cam(yaw, pitch)
    Rt = R.T
    right   = Rt @ np.array([1.0, 0.0, 0.0])
    cam_y   = Rt @ np.array([0.0, 1.0, 0.0])  # cam down in world
    forward = Rt @ np.array([0.0, 0.0, 1.0])
    up_world = -cam_y
    return forward, right, up_world


def compute_motion(pose_frames):
    """From per-frame pose dicts, build arrays of camera-local motion.
    Returns dict of (N,)-arrays plus a valid-mask."""
    N = len(pose_frames)
    yaw      = np.full(N, np.nan)
    pitch    = np.full(N, np.nan)
    cam_pos  = np.full((N, 3), np.nan)
    rms      = np.full(N, np.nan)
    converged = np.zeros(N, dtype=bool)
    t_ms     = np.zeros(N, dtype=np.float64)

    for i, fr in enumerate(pose_frames):
        t_ms[i] = fr["t_ms"]
        if not fr.get("converged"):
            continue
        yaw[i]     = fr["yaw_rad"]
        pitch[i]   = fr["pitch_rad"]
        cam_pos[i] = fr["camera_pos_world"]
        rms[i]     = fr["reproj_rms_px"]
        converged[i] = True

    # Forward differences for rates and velocities; last frame zeroed.
    dt_s = np.diff(t_ms) / 1000.0
    dt_s = np.where(dt_s <= 0, np.nan, dt_s)
    yaw_rate   = np.zeros(N)
    pitch_rate = np.zeros(N)
    v_forward  = np.zeros(N)
    v_right    = np.zeros(N)
    v_up       = np.zeros(N)
    valid_motion = np.zeros(N, dtype=bool)

    for i in range(N - 1):
        if not (converged[i] and converged[i + 1]):
            continue
        dt = dt_s[i]
        if not np.isfinite(dt) or dt <= 0:
            continue
        dyaw   = (yaw[i + 1] - yaw[i])
        dpitch = (pitch[i + 1] - pitch[i])
        dpos   = cam_pos[i + 1] - cam_pos[i]
        fwd, right, up = camera_axes(yaw[i], pitch[i])
        yaw_rate[i]   = dyaw   / dt
        pitch_rate[i] = dpitch / dt
        v_forward[i]  = float(dpos @ fwd)   / dt
        v_right[i]    = float(dpos @ right) / dt
        v_up[i]       = float(dpos @ up)    / dt
        valid_motion[i] = True

    return {
        "t_ms":         t_ms,
        "yaw":          yaw,
        "pitch":        pitch,
        "yaw_rate":     yaw_rate,
        "pitch_rate":   pitch_rate,
        "v_forward":    v_forward,
        "v_right":      v_right,
        "v_up":         v_up,
        "rms":          rms,
        "converged":    converged,
        "valid_motion": valid_motion,
    }


def align_one(event_dir: Path, state_npz, args):
    pose_p  = event_dir / "pose.json"
    event_p = event_dir / "event.json"
    if not pose_p.exists():
        return None, "no pose.json"
    if not event_p.exists():
        return None, "no event.json"
    pose = json.loads(pose_p.read_text())
    evt  = json.loads(event_p.read_text())
    clip_start_ns = evt.get("clip_start_wall_ns")
    if clip_start_ns is None:
        return None, "event.json missing clip_start_wall_ns (recapture with newer mei-wall-detect)"

    motion = compute_motion(pose["frames"])
    N = len(pose["frames"])
    if N == 0:
        return None, "empty pose"

    # State vector slicing: find the bin whose end-time matches each
    # pose frame's wall-clock time.
    bin_ends_ns = state_npz["t_wall_ns"]
    pose_t_ns = clip_start_ns + (motion["t_ms"] * 1e6).astype(np.int64)
    bin_idx = np.searchsorted(bin_ends_ns, pose_t_ns)
    # Clamp to valid range
    in_range = (bin_idx >= 0) & (bin_idx < len(bin_ends_ns))
    sliced_inputs   = np.zeros((N, state_npz["inputs"].shape[1]), dtype=np.uint8)
    sliced_mouse_dx = np.zeros(N, dtype=np.float64)
    sliced_mouse_dy = np.zeros(N, dtype=np.float64)
    sliced_inputs[in_range]   = state_npz["inputs"][bin_idx[in_range]]
    sliced_mouse_dx[in_range] = state_npz["mouse_dx"][bin_idx[in_range]]
    sliced_mouse_dy[in_range] = state_npz["mouse_dy"][bin_idx[in_range]]

    # Final usable mask: in_range AND valid_motion AND rms reasonable
    usable = (in_range
              & motion["valid_motion"]
              & (~np.isnan(motion["rms"]))
              & (motion["rms"] <= args.max_rms_px))
    n_usable = int(usable.sum())

    out_npz = event_dir / "aligned.npz"
    np.savez_compressed(
        out_npz,
        t_ms=motion["t_ms"],
        inputs=sliced_inputs,
        input_names=state_npz["input_names"],
        mouse_dx=sliced_mouse_dx,
        mouse_dy=sliced_mouse_dy,
        yaw=motion["yaw"],
        pitch=motion["pitch"],
        yaw_rate=motion["yaw_rate"],
        pitch_rate=motion["pitch_rate"],
        v_forward=motion["v_forward"],
        v_right=motion["v_right"],
        v_up=motion["v_up"],
        reproj_rms=motion["rms"],
        usable=usable,
    )
    return {
        "event": event_dir.name,
        "n_frames": N,
        "n_usable": n_usable,
        "out": str(out_npz),
        "median_rms": float(np.nanmedian(motion["rms"])),
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None,
                    help="single event name (default: all)")
    ap.add_argument("--state-npz", required=True,
                    help="state-vector.npz produced by state-vector.py")
    ap.add_argument("--max-rms-px", type=float, default=20.0,
                    help="drop frames whose pose RMS exceeds this (default 20 px)")
    ap.add_argument("--concat-output", default=None,
                    help="if set, also write a concatenated training npz "
                         "containing all usable frames across events")
    args = ap.parse_args()

    state_npz = np.load(args.state_npz)
    print(f"State vector: {state_npz['inputs'].shape}  "
          f"({len(state_npz['t_wall_ns'])} bins)")

    root = Path(args.events_dir)
    subs = [root / args.event] if args.event else \
        sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Aligning {len(subs)} event(s)...\n")

    all_rows = []  # for concat output
    for d in subs:
        rec, err = align_one(d, state_npz, args)
        if err:
            print(f"  [skip ] {d.name}: {err}")
            continue
        print(f"  [done ] {rec['event']}: {rec['n_usable']}/{rec['n_frames']} "
              f"usable, median_rms={rec['median_rms']:.1f}px")
        if args.concat_output:
            ev = np.load(rec["out"])
            usable = ev["usable"]
            if not usable.any():
                continue
            all_rows.append({
                "inputs":     ev["inputs"][usable],
                "mouse_dx":   ev["mouse_dx"][usable],
                "mouse_dy":   ev["mouse_dy"][usable],
                "yaw_rate":   ev["yaw_rate"][usable],
                "pitch_rate": ev["pitch_rate"][usable],
                "v_forward":  ev["v_forward"][usable],
                "v_right":    ev["v_right"][usable],
                "v_up":       ev["v_up"][usable],
                "event":      np.repeat(rec["event"], int(usable.sum())),
            })

    if args.concat_output and all_rows:
        cat = {}
        for k in ("inputs", "mouse_dx", "mouse_dy", "yaw_rate", "pitch_rate",
                  "v_forward", "v_right", "v_up", "event"):
            cat[k] = np.concatenate([r[k] for r in all_rows], axis=0)
        cat["input_names"] = state_npz["input_names"]
        np.savez_compressed(args.concat_output, **cat)
        print(f"\nWrote concatenated training set: {args.concat_output} "
              f"({len(cat['mouse_dx'])} frames)")


if __name__ == "__main__":
    main()

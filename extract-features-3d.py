#!/usr/bin/env python3
"""Multi-scale CNN feature extraction with 3D placement.

Per event with clip.mp4 + pose.json + track.json:
  1. Load a pretrained ResNet50.
  2. For each frame, build an image pyramid (scales centered on 1.0).
  3. Forward each pyramid level through the network up to the chosen
     layer (default layer3 -> mid-level features, 1024 channels).
  4. Per (channel, scale) find spatial peaks: local maxima > threshold.
  5. Trace each peak's receptive field back to the original image,
     look up depth at the RF center (fused flow + DA depth, same as
     reconstruct-scene.py), project to world coords via the per-frame
     pose.
  6. NMS in 3D per (layer, channel) so features detected on adjacent
     frames at the same world point don't double-count.

Output:
  <event>/features.npz : per-feature world_xyz, layer_id, channel,
                         scale_idx, response, frame_idx
  <event>/features.ply : visualization (channel-hashed color)

The (layer, channel) identity is the matching key for downstream
multi-event alignment. Scale is recorded but not used for identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# DepthAnything wrapper lives in the turntable repo
sys.path.insert(0, os.path.expanduser("~/turntable"))


# ---------- Geometry / pose helpers (shared with reconstruct-scene.py) ----------
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


def wall_world_corners(orientation, mei, ling, zhou):
    w = mei if orientation == "mei" else ling
    return np.array([
        [-w / 2, -zhou, 0.0],
        [+w / 2, -zhou, 0.0],
        [+w / 2,  0,    0.0],
        [-w / 2,  0,    0.0],
    ])


def calibrate_scale(da_depth, corners_image_px, corners_world, K, R, t):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cam_pts = corners_world @ R.T + t
    z_expected = cam_pts[:, 2]
    if (z_expected <= 0).any():
        return None
    H, W = da_depth.shape
    da_vals = []
    for (u, v) in corners_image_px:
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < W and 0 <= vi < H:
            da_vals.append(da_depth[vi, ui])
        else:
            da_vals.append(np.nan)
    da_vals = np.array(da_vals)
    ok = np.isfinite(da_vals) & (da_vals > 1e-6)
    if ok.sum() < 2:
        return None
    return float((da_vals[ok] * z_expected[ok]).sum()
                 / (da_vals[ok] * da_vals[ok]).sum())


def relative_pose(pose_a, pose_b):
    Ra = rotation_world_to_cam(pose_a["yaw_rad"], pose_a["pitch_rad"])
    Rb = rotation_world_to_cam(pose_b["yaw_rad"], pose_b["pitch_rad"])
    ta = np.array(pose_a["t"]); tb = np.array(pose_b["t"])
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


def flow_depth_map(frame_a, frame_b, pose_a, pose_b, K,
                   min_flow_px=1.5, max_depth=80.0):
    H, W = frame_a.shape[:2]
    R_rel, t_rel = relative_pose(pose_a, pose_b)
    depth = np.full((H, W), np.nan, dtype=np.float32)
    if float(np.linalg.norm(t_rel)) < 1e-3:
        return depth
    g1 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=21,
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


def fuse_depth(da_depth_scaled, flow_depth):
    """Flow where finite, DA elsewhere."""
    out = flow_depth.copy()
    needs = ~np.isfinite(out)
    out[needs] = da_depth_scaled[needs]
    return out


# ---------- Multi-scale CNN feature extraction ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr):
    """BGR uint8 -> float32 (3, H, W) ImageNet-normalized."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


class FeatureNet:
    """ResNet50 truncated to a chosen layer. Returns feature map (1, C, H', W')
    plus the layer's total stride relative to the input image."""
    LAYER_STRIDES = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}

    def __init__(self, layer_name="layer3", device=None):
        import torch
        import torchvision.models as M
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch = torch
        net = M.resnet50(weights="DEFAULT").eval().to(device)
        # Truncate by walking children up to the chosen layer.
        layers = []
        for name, module in net.named_children():
            layers.append((name, module))
            if name == layer_name:
                break
        self.body = torch.nn.Sequential(*[m for _, m in layers])
        self.body.eval()
        self.layer_name = layer_name
        self.stride = self.LAYER_STRIDES[layer_name]

    def forward(self, bgr):
        x = preprocess(bgr)
        t = self.torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.body(t)
        return feat.detach().cpu().numpy()[0]  # (C, H', W')


def find_peaks_2d(feat_map, thresh, max_per_channel=8):
    """feat_map: (C, H, W). Returns list of (channel, y, x, value)."""
    out = []
    C, H, W = feat_map.shape
    # Per-channel local max via 3x3 dilation comparison
    pad = 1
    for c in range(C):
        f = feat_map[c]
        if f.max() < thresh:
            continue
        # Compare to 3x3 neighborhood max
        m = cv2.dilate(f, np.ones((3, 3), np.uint8))
        peak_mask = (f == m) & (f >= thresh)
        ys, xs = np.where(peak_mask)
        if ys.size == 0:
            continue
        vals = f[ys, xs]
        if vals.size > max_per_channel:
            top = np.argpartition(-vals, max_per_channel - 1)[:max_per_channel]
            ys, xs, vals = ys[top], xs[top], vals[top]
        for y, x, v in zip(ys, xs, vals):
            out.append((c, int(y), int(x), float(v)))
    return out


def rf_center_in_original(peak_x, peak_y, stride, scale):
    """Convert a peak in the feature map to a pixel coord in the
    ORIGINAL (un-pyramided) image."""
    # Feature pixel -> input-to-network image pixel: center of receptive field
    in_x = (peak_x + 0.5) * stride
    in_y = (peak_y + 0.5) * stride
    # Network input was the SCALED image, so divide by scale to get back to
    # the original frame's pixel coords.
    return in_x / scale, in_y / scale


# ---------- PLY writer ----------
def channel_color(layer_id, channel):
    """Stable BGR color from (layer, channel) via hash."""
    h = hashlib.md5(f"{layer_id}:{channel}".encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


def write_ply(path: Path, points, colors):
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (b, g, r) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


# ---------- Main per-event ----------
def extract_event(event_dir: Path, cal, args, net, da):
    pose_p  = event_dir / "pose.json"
    track_p = event_dir / "track.json"
    clip_p  = event_dir / "clip.mp4"
    if not (pose_p.exists() and track_p.exists() and clip_p.exists()):
        return None, "missing pose/track/clip"
    pose = json.loads(pose_p.read_text())
    track = json.loads(track_p.read_text())
    pose_frames, track_frames = pose["frames"], track["frames"]
    if len(pose_frames) != len(track_frames):
        return None, "pose vs track frame count mismatch"
    orientation = pose.get("orientation", "mei")
    corners_world = wall_world_corners(orientation,
                                       cal["mei"], cal["ling"], cal["zhou"])
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]])

    cap = cv2.VideoCapture(str(clip_p))
    if not cap.isOpened():
        return None, "cannot open clip.mp4"
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = min(len(frames), len(pose_frames))

    # All extracted features (deduplicated below at the very end).
    feats = []  # list of dicts

    layer_id = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}[args.layer]
    scales = [float(s) for s in args.scales]

    t0 = time.perf_counter()
    n_fwd = 0
    for i in range(n):
        pf = pose_frames[i]
        tf = track_frames[i]
        if not pf.get("converged") or pf.get("reproj_rms_px", 1e9) > args.max_rms_px:
            continue
        # Depth: flow if next frame ok, else DA only.
        nxt_pose, nxt_frame = None, None
        if i + 1 < n:
            npf = pose_frames[i + 1]
            if npf.get("converged") and npf.get("reproj_rms_px", 1e9) <= args.max_rms_px:
                nxt_pose, nxt_frame = npf, frames[i + 1]
        depth = np.full((H, W), np.nan, dtype=np.float32)
        if nxt_frame is not None:
            depth = flow_depth_map(frames[i], nxt_frame, pf, nxt_pose, K,
                                   min_flow_px=args.min_flow_px,
                                   max_depth=args.z_max)
        if da is not None:
            R = rotation_world_to_cam(pf["yaw_rad"], pf["pitch_rad"])
            t_vec = np.array(pf["t"])
            corners_image_px = np.array(tf["corners"], dtype=float)
            da_rel = da.estimate(frames[i])
            # Prefer flow-derived scale where possible
            ok = np.isfinite(depth) & np.isfinite(da_rel) & (da_rel > 1e-6)
            if ok.sum() > 200:
                s = float(np.median(depth[ok] / da_rel[ok]))
            else:
                s = calibrate_scale(da_rel, corners_image_px, corners_world,
                                    K, R, t_vec) or 1.0
            depth = fuse_depth(da_rel * s, depth)

        # Per-frame world transform
        R = rotation_world_to_cam(pf["yaw_rad"], pf["pitch_rad"])
        cam_pos = np.array(pf["camera_pos_world"])

        # Multi-scale forward + peaks
        for si, scale in enumerate(scales):
            Hs, Ws = int(round(H * scale)), int(round(W * scale))
            if Hs < 64 or Ws < 64:
                continue
            scaled = cv2.resize(frames[i], (Ws, Hs),
                                interpolation=cv2.INTER_LINEAR)
            fmap = net.forward(scaled)   # (C, H', W')
            n_fwd += 1
            peaks = find_peaks_2d(fmap, thresh=args.peak_thresh,
                                  max_per_channel=args.max_per_channel)
            for ch, py, px, val in peaks:
                u, v = rf_center_in_original(px, py, net.stride, scale)
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < W and 0 <= vi < H):
                    continue
                d = depth[vi, ui]
                if not np.isfinite(d) or d < args.z_min or d > args.z_max:
                    continue
                # Project pixel + depth -> world (row-vec form)
                x_cam = (u - K[0, 2]) * d / K[0, 0]
                y_cam = (v - K[1, 2]) * d / K[1, 1]
                z_cam = d
                P_cam = np.array([x_cam, y_cam, z_cam])
                P_world = P_cam @ R + cam_pos
                feats.append({
                    "x": float(P_world[0]), "y": float(P_world[1]), "z": float(P_world[2]),
                    "layer_id": layer_id,
                    "channel": int(ch),
                    "scale_idx": int(si),
                    "scale": float(scale),
                    "response": float(val),
                    "frame_idx": int(i),
                })

    # 3D NMS per (layer, channel) so adjacent frames don't double-count.
    voxel_keys = {}  # (lay, ch, vx, vy, vz) -> idx into feats with best response
    vs = args.dedupe_voxel
    for j, f in enumerate(feats):
        k = (f["layer_id"], f["channel"],
             int(f["x"] / vs), int(f["y"] / vs), int(f["z"] / vs))
        cur = voxel_keys.get(k)
        if cur is None or feats[cur]["response"] < f["response"]:
            voxel_keys[k] = j
    keep_idx = sorted(voxel_keys.values())
    feats = [feats[j] for j in keep_idx]

    # Save
    out_npz = event_dir / "features.npz"
    arrs = {
        "xyz":        np.array([[f["x"], f["y"], f["z"]] for f in feats], dtype=np.float32),
        "layer_id":   np.array([f["layer_id"] for f in feats], dtype=np.int16),
        "channel":    np.array([f["channel"]  for f in feats], dtype=np.int32),
        "scale_idx":  np.array([f["scale_idx"] for f in feats], dtype=np.int16),
        "scale":      np.array([f["scale"]    for f in feats], dtype=np.float32),
        "response":   np.array([f["response"] for f in feats], dtype=np.float32),
        "frame_idx":  np.array([f["frame_idx"] for f in feats], dtype=np.int32),
    }
    np.savez_compressed(out_npz, **arrs)

    # PLY visualization (one point per feature, colored by channel hash)
    pts = arrs["xyz"]
    colors = np.array(
        [channel_color(layer_id, int(c)) for c in arrs["channel"]],
        dtype=np.uint8,
    )
    out_ply = event_dir / "features.ply"
    write_ply(out_ply, pts, colors)

    elapsed = time.perf_counter() - t0
    return {
        "event": event_dir.name,
        "n_features": len(feats),
        "n_forward_passes": n_fwd,
        "unique_channels": int(len(set(arrs["channel"].tolist()))),
        "elapsed_s": elapsed,
        "npz": str(out_npz),
        "ply": str(out_ply),
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None)
    ap.add_argument("--calibration", default="./calibration.json")
    ap.add_argument("--layer", choices=("layer1", "layer2", "layer3", "layer4"),
                    default="layer3")
    ap.add_argument("--scales", nargs="+", default=("0.5", "0.707", "1.0", "1.414", "2.0"),
                    help="image-pyramid scales centered at 1.0")
    ap.add_argument("--peak-thresh", type=float, default=4.0,
                    help="min activation for a peak (after ReLU; default 4.0)")
    ap.add_argument("--max-per-channel", type=int, default=4,
                    help="cap peaks per channel per scale per frame (default 4)")
    ap.add_argument("--dedupe-voxel", type=float, default=0.15,
                    help="voxel size (OW-m) for 3D NMS per (layer, channel)")
    ap.add_argument("--z-min", type=float, default=0.2)
    ap.add_argument("--z-max", type=float, default=80.0)
    ap.add_argument("--min-flow-px", type=float, default=1.5)
    ap.add_argument("--max-rms-px", type=float, default=20.0)
    ap.add_argument("--no-da", action="store_true",
                    help="skip DepthAnything fallback (flow-only depth)")
    args = ap.parse_args()

    cal_path = Path(args.calibration)
    cal_raw = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    def g(keys, default=None):
        cur = cal_raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur
    cal = {"mei":  g(["mei_wall", "length_ow_m"], 1.0),
           "ling": g(["mei_wall", "depth_ow_m"],  1.0),
           "zhou": g(["mei_wall", "height_ow_m"], 1.0)}

    print(f"Loading ResNet50 / {args.layer}...")
    net = FeatureNet(layer_name=args.layer)
    print(f"  stride={net.stride}  device={net.device}")

    da = None
    if not args.no_da:
        print("Loading DepthAnything...")
        from depth_anything import DepthAnythingEstimator
        da = DepthAnythingEstimator()

    root = Path(args.events_dir)
    subs = [root / args.event] if args.event else \
        sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"\nExtracting features from {len(subs)} event(s)...")
    for d in subs:
        meta, err = extract_event(d, cal, args, net, da)
        if err:
            print(f"  [skip] {d.name}: {err}")
            continue
        print(f"  [done] {meta['event']}: {meta['n_features']} features "
              f"across {meta['unique_channels']} channels "
              f"({meta['n_forward_passes']} fwd, "
              f"{meta['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()

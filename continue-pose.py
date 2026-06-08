#!/usr/bin/env python3
"""Continue camera tracking past the wall lifetime via the feature map.

The Mei wall gives metric pose only while it's up (~5s). During that
window extract-features-3d.py places CNN features in the world frame
(features.npz). After the wall drops we relocalize each subsequent
frame against that 3D feature map:

  per post-wall frame:
    1. Extract multi-scale CNN feature peaks (same net/layer as the map).
    2. For each 2D peak, candidate 3D matches = map features with the
       same (layer, channel).
    3. RANSAC no-roll PnP (5-DOF: yaw, pitch, tx, ty, tz) on the
       candidate set; score by reprojection inliers.
    4. Keep the pose if enough inliers; warm-start the next frame from it.

Outputs <event>/continued_pose.json: per-frame pose for frames beyond
the wall-tracked range, in the same world frame as pose.json.

Requires clips longer than the wall lifetime (record with
mei-wall-detect.py --clip-duration-s 20 or so).
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

try:
    from scipy.optimize import least_squares
except ImportError:
    raise SystemExit("pip install scipy")

sys.path.insert(0, os.path.expanduser("~/turntable"))


# ---------- no-roll pose math (shared convention with pose-from-track) ----------
def R_no_roll(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return R_pitch @ R_yaw


def project(world_pts, yaw, pitch, t, K):
    R = R_no_roll(yaw, pitch)
    pc = world_pts @ R.T + t
    z = pc[:, 2]
    z = np.where(np.abs(z) < 1e-3, 1e-3, z)
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def pnp_no_roll(world_pts, image_pts, K, init):
    """5-DOF PnP. Returns (params, rms, ok)."""
    def resid(p):
        yaw, pitch, tx, ty, tz = p
        if tz < 0.05:
            return np.full(image_pts.size, 1e3)
        proj, _ = project(world_pts, yaw, pitch, np.array([tx, ty, tz]), K)
        return (proj - image_pts).ravel()
    try:
        r = least_squares(resid, init, method="lm", max_nfev=60)
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        return r.x, rms, True
    except Exception:
        return init, float("inf"), False


# ---------- CNN feature extractor (copied from extract-features-3d) ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


class FeatureNet:
    LAYER_STRIDES = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}

    def __init__(self, layer_name="layer3", device=None):
        import torch
        import torchvision.models as M
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch = torch
        net = M.resnet50(weights="DEFAULT").eval().to(device)
        layers = []
        for name, module in net.named_children():
            layers.append((name, module))
            if name == layer_name:
                break
        self.body = torch.nn.Sequential(*[m for _, m in layers]).eval()
        self.layer_name = layer_name
        self.stride = self.LAYER_STRIDES[layer_name]

    def forward(self, bgr):
        x = preprocess(bgr)
        t = self.torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.body(t)
        return feat.detach().cpu().numpy()[0]


def find_peaks_2d(feat_map, thresh, max_per_channel=4):
    out = []
    C, H, W = feat_map.shape
    for c in range(C):
        f = feat_map[c]
        if f.max() < thresh:
            continue
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


def rf_center(px, py, stride, scale):
    return (px + 0.5) * stride / scale, (py + 0.5) * stride / scale


def extract_frame_features(net, frame, scales, peak_thresh, max_per_channel,
                           layer_id):
    """Returns list of (u, v, layer_id, channel, response) in ORIGINAL
    image pixel coords."""
    H, W = frame.shape[:2]
    out = []
    for scale in scales:
        Hs, Ws = int(round(H * scale)), int(round(W * scale))
        if Hs < 64 or Ws < 64:
            continue
        scaled = cv2.resize(frame, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
        fmap = net.forward(scaled)
        for ch, py, px, val in find_peaks_2d(fmap, peak_thresh, max_per_channel):
            u, v = rf_center(px, py, net.stride, scale)
            if 0 <= u < W and 0 <= v < H:
                out.append((u, v, layer_id, int(ch), val))
    return out


# ---------- candidate building + RANSAC PnP ----------
def build_pnp_candidates(frame_feats, map_xyz, map_layer, map_channel,
                         top_k_per_class=4):
    """Match 2D frame features to 3D map features by (layer, channel).
    Returns image_pts (M, 2), world_pts (M, 3)."""
    # Index map by (layer, channel)
    by_key = {}
    for j in range(len(map_xyz)):
        by_key.setdefault((int(map_layer[j]), int(map_channel[j])), []).append(j)

    img_pts, wld_pts = [], []
    # Per-frame-feature, cap candidates per class
    for (u, v, lid, ch, val) in frame_feats:
        cands = by_key.get((lid, ch))
        if not cands:
            continue
        c = cands
        if len(c) > top_k_per_class:
            # keep the highest-response map features for this class
            order = sorted(c, key=lambda j: -float(map_xyz[j][0]))  # arbitrary but stable
            c = order[:top_k_per_class]
        for j in c:
            img_pts.append([u, v])
            wld_pts.append(map_xyz[j])
    return np.array(img_pts, dtype=np.float32), np.array(wld_pts, dtype=np.float32)


def ransac_pnp(image_pts, world_pts, K, init, n_iter=2000, inlier_tol_px=8.0,
               min_inliers=12, seed=0):
    """Fast PnP RANSAC via OpenCV (C++), then one no-roll scipy refit on
    the inliers. Candidate pairs are fed as-is; bogus (layer, channel)
    matches are rejected by RANSAC as outliers.

    Returns (params5, n_inliers, rms) or (None, n_inliers, None).
    """
    N = len(image_pts)
    if N < 4:
        return None, 0, None
    obj = world_pts.reshape(-1, 1, 3).astype(np.float32)
    img = image_pts.reshape(-1, 1, 2).astype(np.float32)
    K32 = K.astype(np.float32)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K32, None,
            iterationsCount=n_iter,
            reprojectionError=float(inlier_tol_px),
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None, 0, None
    if not ok or inliers is None:
        return None, 0, None
    inl = inliers.reshape(-1)
    if len(inl) < min_inliers:
        return None, int(len(inl)), None

    # Seed the no-roll refit from the OpenCV pose.
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.reshape(3)
    yaw0 = math.atan2(R_cv[0, 2], R_cv[2, 2])
    pitch0 = math.asin(max(-1.0, min(1.0, -R_cv[1, 2])))
    init_refit = np.array([yaw0, pitch0, t_cv[0], t_cv[1], t_cv[2]])
    params, rms, _ = pnp_no_roll(world_pts[inl], image_pts[inl], K, init_refit)
    return params, int(len(inl)), rms


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", required=True)
    ap.add_argument("--layer", choices=("layer1", "layer2", "layer3", "layer4"),
                    default="layer3", help="must match the layer used for "
                                           "the feature map (default layer3)")
    ap.add_argument("--scales", nargs="+", default=("0.5", "0.707", "1.0", "1.414", "2.0"))
    ap.add_argument("--peak-thresh", type=float, default=4.0)
    ap.add_argument("--max-per-channel", type=int, default=4)
    ap.add_argument("--top-k-per-class", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--inlier-tol-px", type=float, default=8.0)
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap how many post-wall frames to process")
    args = ap.parse_args()

    event_dir = Path(args.events_dir) / args.event
    feat_p = event_dir / "features.npz"
    pose_p = event_dir / "pose.json"
    clip_p = event_dir / "clip.mp4"
    if not feat_p.exists():
        raise SystemExit("need features.npz (run extract-features-3d.py)")
    if not (pose_p.exists() and clip_p.exists()):
        raise SystemExit("need pose.json and clip.mp4")

    fmap = np.load(feat_p)
    map_xyz = fmap["xyz"].astype(np.float32)
    map_layer = fmap["layer_id"]
    map_channel = fmap["channel"]
    print(f"feature map: {len(map_xyz):,} features, "
          f"{len(set(zip(map_layer.tolist(), map_channel.tolist())))} (layer,channel) classes")

    pose = json.loads(pose_p.read_text())
    pose_frames = pose["frames"]
    W, H = pose["intrinsics"]["image_size"]
    focal = pose["intrinsics"]["focal_px"]
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)

    # Last wall-tracked frame index (where pose.json stops being reliable)
    tracked = [f["frame_idx"] for f in pose_frames if f.get("converged")]
    last_tracked = max(tracked) if tracked else -1
    print(f"wall-tracked frames: 0..{last_tracked} "
          f"({len(tracked)} converged)")

    # Load full clip
    cap = cv2.VideoCapture(str(clip_p))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    n = len(frames)
    print(f"clip has {n} frames; continuing from frame {last_tracked + 1}")
    if last_tracked + 1 >= n:
        raise SystemExit("no post-wall frames in clip; record with a longer "
                         "--clip-duration-s (e.g. 20)")

    print(f"loading ResNet50 / {args.layer}...")
    net = FeatureNet(layer_name=args.layer)
    layer_id_map = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
    layer_id = layer_id_map[args.layer]
    scales = [float(s) for s in args.scales]

    # Warm-start from the last good wall pose
    last_pose = None
    for f in reversed(pose_frames):
        if f.get("converged"):
            last_pose = np.array([f["yaw_rad"], f["pitch_rad"],
                                  f["t"][0], f["t"][1], f["t"][2]])
            break
    if last_pose is None:
        raise SystemExit("no converged wall pose to warm-start from")

    out_frames = []
    n_ok = 0
    n_fail = 0
    t0 = time.perf_counter()
    end = n if args.max_frames is None else min(n, last_tracked + 1 + args.max_frames)
    for i in range(last_tracked + 1, end):
        feats = extract_frame_features(
            net, frames[i], scales, args.peak_thresh,
            args.max_per_channel, layer_id)
        img_pts, wld_pts = build_pnp_candidates(
            feats, map_xyz, map_layer, map_channel,
            top_k_per_class=args.top_k_per_class)
        if len(img_pts) < 4:
            n_fail += 1
            out_frames.append({"frame_idx": i, "converged": False,
                               "reason": "too few candidates"})
            continue
        params, n_in, rms = ransac_pnp(
            img_pts, wld_pts, K, init=last_pose,
            n_iter=args.n_iter, inlier_tol_px=args.inlier_tol_px,
            min_inliers=args.min_inliers)
        if params is None:
            n_fail += 1
            out_frames.append({"frame_idx": i, "converged": False,
                               "reason": f"pnp inliers<{args.min_inliers} ({n_in})"})
            continue
        yaw, pitch, tx, ty, tz = params
        R = R_no_roll(yaw, pitch)
        cam_pos = (-R.T @ np.array([tx, ty, tz])).tolist()
        out_frames.append({
            "frame_idx": i,
            "converged": True,
            "yaw_rad": float(yaw), "pitch_rad": float(pitch),
            "yaw_deg": float(math.degrees(yaw)),
            "pitch_deg": float(math.degrees(pitch)),
            "t": [float(tx), float(ty), float(tz)],
            "camera_pos_world": cam_pos,
            "n_candidates": int(len(img_pts)),
            "n_inliers": int(n_in),
            "reproj_rms_px": float(rms),
        })
        last_pose = np.array(params)  # warm-start next frame
        n_ok += 1
        if (i - last_tracked) % 10 == 0:
            print(f"  frame {i}: inliers={n_in}/{len(img_pts)} "
                  f"rms={rms:.1f}px  ({time.perf_counter()-t0:.0f}s)")

    out = {
        "event": args.event,
        "layer": args.layer,
        "last_wall_tracked_frame": last_tracked,
        "n_continued_ok": n_ok,
        "n_continued_fail": n_fail,
        "frames": out_frames,
    }
    (event_dir / "continued_pose.json").write_text(json.dumps(out, indent=2))
    print(f"\nDone. {n_ok} frames relocalized, {n_fail} failed "
          f"-> {event_dir / 'continued_pose.json'}")
    if n_ok:
        good = [f for f in out_frames if f["converged"]]
        rms_med = float(np.median([f["reproj_rms_px"] for f in good]))
        yaws = [f["yaw_deg"] for f in good]
        print(f"median reproj rms = {rms_med:.1f}px   "
              f"yaw range = ({min(yaws):+.1f}..{max(yaws):+.1f})deg")


if __name__ == "__main__":
    main()

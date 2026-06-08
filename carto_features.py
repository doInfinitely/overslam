"""Semantic-feature localization for Mei Cartographer.

Ported from extract-features-3d.py: multi-scale ResNet layer3 peaks,
(layer,channel) identity as the match key, placed in 3D. The map is a
database of (world_xyz, channel, scale, response). Localization =
match a new frame's 2D channel-peaks to the 3D DB and solvePnP.

Pose comes from FEATURES, not dead reckoning. A seed pose (e.g. the
input dead-reckon prior, or the last pose) is used ONLY to gate the
2D<->3D matching so PnP is tractable; cold relocalization (no seed,
after respawn) matches against all same-channel points.

torch/cv2 only needed at Windows runtime; FeatureMap3D + match gating
are pure numpy and unit-testable.
"""
from __future__ import annotations

import math
import numpy as np

import carto_geom as G


# ---------------------------------------------------------------------
# Multi-scale ResNet peak extractor  (runtime: torch + cv2)
# ---------------------------------------------------------------------

LAYER_STRIDE = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}


class FeatureExtractor:
    def __init__(self, layer="layer3", scales=(0.707, 1.0, 1.414),
                 peak_thresh=2.5, max_per_channel=6):
        import torch
        import torchvision.models as M
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = M.resnet50(weights="DEFAULT").eval().to(self.dev)
        layers = []
        for name, mod in net.named_children():
            layers.append(mod)
            if name == layer:
                break
        self.body = torch.nn.Sequential(*layers).eval()
        self.stride = LAYER_STRIDE[layer]
        self.scales = scales
        self.peak_thresh = peak_thresh
        self.max_per_channel = max_per_channel
        self._mean = np.array([0.485, 0.456, 0.406], np.float32)
        self._std = np.array([0.229, 0.224, 0.225], np.float32)

    def _fwd(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        x = self.torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            return self.body(x).detach().cpu().numpy()[0]   # (C,H',W')

    def peaks(self, bgr):
        """Return Nx4 array [channel, u, v, response] in ORIGINAL-image
        pixel coords, pooled across scales (channel identity is the
        scale-invariant key)."""
        import cv2
        import torch.nn.functional as _F
        H, W = bgr.shape[:2]
        out = []
        for scale in self.scales:
            Hs, Ws = int(round(H * scale)), int(round(W * scale))
            if Hs < 64 or Ws < 64:
                continue
            fmap = self._fwd(cv2.resize(bgr, (Ws, Hs), interpolation=cv2.INTER_LINEAR))
            C, fh, fw = fmap.shape

            # GPU max-pool NMS: replaces C individual cv2.dilate calls.
            fmap_t = self.torch.from_numpy(fmap).to(self.dev).unsqueeze(0)  # (1,C,H,W)
            m_t    = _F.max_pool2d(fmap_t, kernel_size=3, stride=1, padding=1)
            fmap_np = fmap_t.squeeze(0).cpu().numpy()
            m_np    = m_t.squeeze(0).cpu().numpy()
            hit = (fmap_np == m_np) & (fmap_np >= self.peak_thresh)
            if not hit.any():
                continue
            c_idx, y_idx, x_idx = np.where(hit)
            vals = fmap_np[c_idx, y_idx, x_idx]

            # Per-channel top-K
            for c in np.unique(c_idx):
                sel = (c_idx == c)
                ys, xs, vs = y_idx[sel], x_idx[sel], vals[sel]
                if vs.size > self.max_per_channel:
                    top = np.argpartition(-vs, self.max_per_channel - 1)[:self.max_per_channel]
                    ys, xs, vs = ys[top], xs[top], vs[top]
                us = (xs + 0.5) * self.stride / scale
                vv = (ys + 0.5) * self.stride / scale
                valid = (us >= 0) & (us < W) & (vv >= 0) & (vv < H)
                for u_, v_, val_ in zip(us[valid], vv[valid], vs[valid]):
                    out.append((int(c), float(u_), float(v_), float(val_)))

        if not out:
            return np.zeros((0, 4), np.float32)
        return np.array(out, np.float32)


# ---------------------------------------------------------------------
# 3D feature database  (pure numpy)
# ---------------------------------------------------------------------

class FeatureMap3D:
    def __init__(self, dedupe_voxel=0.2):
        self.dedupe_voxel = dedupe_voxel
        self.xyz = np.zeros((0, 3), np.float32)
        self.channel = np.zeros((0,), np.int32)
        self.response = np.zeros((0,), np.float32)
        self._by_channel: dict[int, list[int]] = {}
        self._vox: dict[tuple, int] = {}     # (ch,vx,vy,vz)->row (NMS)

    def add(self, xyz, channel, response):
        """Add features with per-(channel,voxel) NMS keeping best response."""
        vs = self.dedupe_voxel
        new_x, new_c, new_r = [], [], []
        for p, c, r in zip(xyz, channel, response):
            key = (int(c), int(p[0] / vs), int(p[1] / vs), int(p[2] / vs))
            row = self._vox.get(key)
            if row is None:
                idx = self.xyz.shape[0] + len(new_x)
                self._vox[key] = idx
                new_x.append(p); new_c.append(int(c)); new_r.append(float(r))
            elif row >= self.xyz.shape[0]:
                # collision within this same batch: keep larger response
                j = row - self.xyz.shape[0]
                if new_r[j] < r:
                    new_x[j] = p; new_r[j] = float(r)
            else:
                if self.response[row] < r:
                    self.xyz[row] = p; self.response[row] = float(r)
        if new_x:
            base = self.xyz.shape[0]
            self.xyz = np.vstack([self.xyz, np.array(new_x, np.float32)])
            self.channel = np.concatenate([self.channel, np.array(new_c, np.int32)])
            self.response = np.concatenate([self.response, np.array(new_r, np.float32)])
            for off, c in enumerate(new_c):
                self._by_channel.setdefault(int(c), []).append(base + off)
        return len(new_x)

    def rows_for_channel(self, c):
        return self._by_channel.get(int(c), [])

    def __len__(self):
        return self.xyz.shape[0]

    def save(self, path):
        np.savez_compressed(path, xyz=self.xyz, channel=self.channel,
                            response=self.response)

    def load(self, path):
        z = np.load(path)
        self.add(z["xyz"], z["channel"], z["response"])


# ---------------------------------------------------------------------
# Localization: 2D channel-peaks -> 3D DB -> PnP
# ---------------------------------------------------------------------

def build_correspondences(peaks, fmap: FeatureMap3D, K, seed=None,
                          gate_px=60.0, max_per_peak=4):
    """Return (obj_pts Nx3, img_pts Nx2) candidate matches by channel
    identity. If `seed`=(yaw,pitch,t) given, gate each 2D peak to map
    points of the same channel that PROJECT within gate_px (drastically
    cuts ambiguity); else take the strongest-response same-channel
    points (cold relocalization)."""
    if len(fmap) == 0 or peaks.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)

    proj_uv = None
    if seed is not None:
        yaw, pitch, t = seed
        proj_uv, z = G.project(fmap.xyz, yaw, pitch, t, K)
        proj_uv = np.where((z > 0.05)[:, None], proj_uv, np.nan)

    obj, img = [], []
    for ch, u, v, _val in peaks:
        rows = fmap.rows_for_channel(int(ch))
        if not rows:
            continue
        rows = np.array(rows)
        if proj_uv is not None:
            d = np.hypot(proj_uv[rows, 0] - u, proj_uv[rows, 1] - v)
            sel = rows[np.isfinite(d) & (d < gate_px)]
            if sel.size == 0:
                continue
            # nearest few
            order = np.argsort(np.hypot(proj_uv[sel, 0] - u, proj_uv[sel, 1] - v))
            sel = sel[order[:max_per_peak]]
        else:
            # cold: strongest-response candidates for this channel
            r = fmap.response[rows]
            sel = rows[np.argsort(-r)[:max_per_peak]]
        for s in sel:
            obj.append(fmap.xyz[s]); img.append((u, v))
    if not obj:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)
    return np.array(obj, np.float32), np.array(img, np.float32)


def relocalize(peaks, fmap, K, seed=None, gate_px=60.0,
               reproj_px=4.0, min_inliers=12):
    """Feature-based pose. Returns (pose|None, n_inliers, n_corr).
    pose = (yaw, pitch, t)."""
    import cv2
    obj, img = build_correspondences(peaks, fmap, K, seed, gate_px)
    if obj.shape[0] < min_inliers:
        return None, 0, obj.shape[0]
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, None,
        reprojectionError=reproj_px, confidence=0.999, iterationsCount=200,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, 0 if inliers is None else len(inliers), obj.shape[0]
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.ravel()
    # decompose to no-roll (yaw,pitch) and refine
    yaw0 = math.atan2(R_cv[0, 2], R_cv[2, 2])
    pitch0 = math.asin(max(-1.0, min(1.0, -R_cv[1, 2])))
    init = np.array([yaw0, pitch0, t_cv[0], t_cv[1], t_cv[2]])
    inl = inliers.ravel()
    params, rms, ok2 = G.pnp_no_roll(obj[inl], img[inl], K, init)
    if not ok2:
        params = init
    return (float(params[0]), float(params[1]), np.array(params[2:5])), \
        int(len(inl)), int(obj.shape[0])

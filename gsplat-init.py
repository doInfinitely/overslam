#!/usr/bin/env python3
"""Initialize a 3D Gaussian Splat from our voxel mesh or point cloud.

Each surface vertex (from <event>/mesh.ply if it exists, else points
from <event>/pointcloud.ply) becomes a single 3D Gaussian. No
differentiable rendering / no training -- this is just a format
conversion so the geometry can be viewed in standard 3DGS viewers
(SuperSplat https://superspl.at/editor, Polycam, the official Inria
viewer, etc.).

Output: <event>/splat.ply in the de-facto-standard 3DGS .ply format:
  x, y, z                   position
  nx, ny, nz                normals (zeros; not used by 3DGS renderers)
  f_dc_0, f_dc_1, f_dc_2    DC spherical-harmonic color coefficients
  opacity                   logit-space opacity
  scale_0, scale_1, scale_2 log-space isotropic scale
  rot_0, rot_1, rot_2, rot_3 quaternion (identity = 1, 0, 0, 0)

Renderable color = (f_dc * C0_basis) + 0.5, where C0_basis = 1 / (2*sqrt(pi)).
We invert that to encode the input vertex colors faithfully.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np


C0_BASIS = 1.0 / (2.0 * math.sqrt(math.pi))   # ~0.282094


def read_ply(path: Path):
    """Read an ASCII PLY with x/y/z + (optional) r/g/b vertex properties.
    Returns (positions Nx3 float32, colors Nx3 uint8 or None)."""
    with open(path, "r") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path}: unexpected EOF in header")
            header.append(line)
            if line.startswith("end_header"):
                break
        # Parse element vertex N + property list
        n_vert = 0
        props = []
        in_vertex = False
        for line in header:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "element":
                in_vertex = (tokens[1] == "vertex")
                if in_vertex:
                    n_vert = int(tokens[2])
            elif tokens[0] == "property" and in_vertex:
                props.append(tokens[-1])  # name is last token
        idx = {name: i for i, name in enumerate(props)}
        has_rgb = all(c in idx for c in ("red", "green", "blue"))
        pos = np.empty((n_vert, 3), dtype=np.float32)
        col = np.empty((n_vert, 3), dtype=np.uint8) if has_rgb else None
        for i in range(n_vert):
            parts = f.readline().split()
            pos[i] = float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])
            if has_rgb:
                col[i] = (int(parts[idx["red"]]), int(parts[idx["green"]]), int(parts[idx["blue"]]))
        return pos, col


def write_3dgs_ply(path: Path, positions, colors_bgr_or_rgb, *,
                   colors_are_bgr=True, opacity=0.95, scale=0.02):
    """Write Gaussian splats as a binary 3DGS .ply.

    positions       : (N, 3) float32 world coords
    colors_..._     : (N, 3) uint8 BGR (or RGB if colors_are_bgr=False)
    opacity         : scalar or (N,) float in (0, 1). Stored as logit.
    scale           : scalar or (N,) float (linear meters/units). Stored as log.
    """
    n = len(positions)
    if colors_are_bgr:
        rgb_u8 = colors_bgr_or_rgb[:, [2, 1, 0]]
    else:
        rgb_u8 = colors_bgr_or_rgb
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    # Inverse of: rendered = f_dc * C0 + 0.5
    f_dc = ((rgb01 - 0.5) / C0_BASIS).astype(np.float32)

    opacity_arr = np.broadcast_to(np.asarray(opacity, dtype=np.float32),
                                  (n,)).copy()
    opacity_arr = np.clip(opacity_arr, 1e-4, 1 - 1e-4)
    op_logit = np.log(opacity_arr / (1 - opacity_arr)).astype(np.float32)

    scale_arr = np.broadcast_to(np.asarray(scale, dtype=np.float32),
                                (n,)).copy()
    log_scale = np.log(np.maximum(scale_arr, 1e-6)).astype(np.float32)
    log_scale3 = np.broadcast_to(log_scale[:, None], (n, 3)).copy()  # isotropic

    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0  # identity quaternion

    normals = np.zeros((n, 3), dtype=np.float32)

    # Assemble per-vertex record matching the property list below.
    # Order: x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity s0 s1 s2 r0 r1 r2 r3
    rows = np.concatenate([
        positions.astype(np.float32),
        normals,
        f_dc,
        op_logit[:, None],
        log_scale3,
        rot,
    ], axis=1).astype(np.float32)

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def process_event(event_dir: Path, args):
    mesh_p = event_dir / "mesh.ply"
    cloud_p = event_dir / "pointcloud.ply"
    if args.source == "mesh" or (args.source == "auto" and mesh_p.exists()):
        src = mesh_p
    else:
        src = cloud_p
    if not src.exists():
        return None, f"no source ply (looked for {mesh_p.name} / {cloud_p.name})"
    positions, colors = read_ply(src)
    if colors is None:
        # Default to white if source has no colors
        colors = np.full_like(positions, 200, dtype=np.uint8)

    n0 = len(positions)
    if args.subsample > 1:
        idx = np.arange(0, n0, args.subsample)
        positions = positions[idx]
        colors = colors[idx]

    out = event_dir / "splat.ply"
    write_3dgs_ply(out, positions, colors,
                   colors_are_bgr=(src.name == "pointcloud.ply"),
                   opacity=args.opacity, scale=args.scale)
    return {
        "event": event_dir.name,
        "source": src.name,
        "n_in": int(n0),
        "n_out": int(len(positions)),
        "splat_ply": str(out),
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--event", default=None)
    ap.add_argument("--source", choices=("auto", "mesh", "cloud"),
                    default="auto",
                    help="prefer mesh.ply or pointcloud.ply (default: auto = "
                         "mesh if present)")
    ap.add_argument("--scale", type=float, default=0.025,
                    help="isotropic Gaussian scale in ow-m (default 0.025). "
                         "Should be similar to the source's voxel size.")
    ap.add_argument("--opacity", type=float, default=0.92,
                    help="initial opacity in [0, 1] (default 0.92)")
    ap.add_argument("--subsample", type=int, default=1,
                    help="keep every Nth source point (default 1 = all)")
    args = ap.parse_args()

    root = Path(args.events_dir)
    subs = [root / args.event] if args.event else \
        sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Initializing splats for {len(subs)} event(s)...")
    for d in subs:
        rec, err = process_event(d, args)
        if err:
            print(f"  [skip] {d.name}: {err}")
            continue
        print(f"  [done] {rec['event']}: {rec['n_out']} splats "
              f"from {rec['source']} -> {rec['splat_ply']}")


if __name__ == "__main__":
    main()

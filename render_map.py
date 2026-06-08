#!/usr/bin/env python3
"""Top-down floorplan from Mei Cartographer voxel map.

Usage:
    python render_map.py --map kingsrow [--maps-dir maps] [--out floorplan.png]
    python render_map.py --map kingsrow --xlim -40 40 --zlim -40 40  # zoom to spawn

Colours:
    dark background  — unmapped / void
    plasma gradient  — navigable floor, coloured by elevation (low=dark, high=bright)
    dim mid-tone     — 1-cell clearance buffer around walls (where A* won't go)
    white            — wall / impassable geometry
    lime circle      — last known bot pose
    red ×            — hazard markers (fell off map)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

CELL      = 0.5
BG_COLOUR = np.array([0.10, 0.10, 0.12, 1.0])


def load_cloud(map_dir: Path):
    cloud = np.load(map_dir / "cloud.npy")
    meta  = json.loads((map_dir / "coverage.json").read_text())
    traj_path = map_dir / "trajectory.npy"
    traj = np.load(traj_path) if traj_path.exists() else None
    return cloud, meta, traj


def build_grid(cloud: np.ndarray):
    """Build a height map: every XZ cell -> minimum Y of its voxels."""
    # Crop Y outliers first: keep only voxels near the median floor plane.
    # Bad-pose artifacts project depth rays to ±30m in Y; real geometry
    # clusters within a few metres of the spawn floor.
    y_all = cloud[:, 1]
    y_med = float(np.median(y_all))
    y_lo  = float(np.percentile(y_all, 5))
    y_hi  = float(np.percentile(y_all, 95))
    y_span = max(y_hi - y_lo, 1.0)
    # Keep voxels within 1.5× the central 90th-pct span around median.
    y_keep = (y_all >= y_med - y_span * 1.5) & (y_all <= y_med + y_span * 1.5)
    cloud = cloud[y_keep]

    ci = np.floor(-cloud[:, 0] / CELL).astype(np.int32)   # negate: world +X = west; -X = east
    cj = np.floor(cloud[:, 2] / CELL).astype(np.int32)
    y  = cloud[:, 1]

    by_cell: dict = defaultdict(list)
    for i, j, yv in zip(ci, cj, y):
        by_cell[(i, j)].append(yv)

    hmap = {(i, j): float(np.asarray(ys).min()) for (i, j), ys in by_cell.items()}

    # Density filter — drop isolated cells (outlier depth rays).
    MIN_NEIGHBORS = 3
    all_cells = set(hmap.keys())
    hmap = {k: v for k, v in hmap.items()
            if sum(1 for di in (-1,0,1) for dj in (-1,0,1)
                   if (di or dj) and (k[0]+di, k[1]+dj) in all_cells) >= MIN_NEIGHBORS}

    # Spatial crop: keep only cells within the IQR of XZ extent.
    # Depth-ray artifacts shoot far outside the actual play area; the real
    # geometry is the dense central cluster.
    if hmap:
        xs = np.array([k[0] for k in hmap])
        zs = np.array([k[1] for k in hmap])
        x_lo, x_hi = np.percentile(xs, 10), np.percentile(xs, 90)
        z_lo, z_hi = np.percentile(zs, 10), np.percentile(zs, 90)
        # Expand by 1.5× the IQR to keep real edges, not just median.
        xc = (x_lo + x_hi) / 2; xr = (x_hi - x_lo) * 0.75
        zc = (z_lo + z_hi) / 2; zr = (z_hi - z_lo) * 0.75
        hmap = {k: v for k, v in hmap.items()
                if abs(k[0] - xc) <= xr and abs(k[1] - zc) <= zr}

    return hmap


def render(hmap, meta, traj, out_path, xlim=None, zlim=None):
    def in_bounds(i, j):
        x, z = i * CELL, j * CELL
        if xlim and not (xlim[0] <= x <= xlim[1]): return False
        if zlim and not (zlim[0] <= z <= zlim[1]): return False
        return True

    cells = {k: v for k, v in hmap.items() if in_bounds(*k)}
    if not cells:
        print("No cells in the requested region.")
        return

    min_i = min(k[0] for k in cells); max_i = max(k[0] for k in cells)
    min_j = min(k[1] for k in cells); max_j = max(k[1] for k in cells)
    W = max_i - min_i + 1
    H = max_j - min_j + 1
    print(f"grid {W}×{H} cells  ({W*CELL:.0f}×{H*CELL:.0f} m)  {len(cells):,} filled")

    ys = list(cells.values())
    y_lo = float(np.percentile(ys, 2))
    y_hi = float(np.percentile(ys, 98))
    if y_hi - y_lo < 0.5: y_hi = y_lo + 1.0

    try:
        cmap = plt.colormaps["plasma"]
    except (AttributeError, TypeError):
        cmap = cm.get_cmap("plasma")
    norm = Normalize(vmin=y_lo, vmax=y_hi)

    img = np.tile(BG_COLOUR, (H, W, 1))
    for (i, j), fy in cells.items():
        row, col = j - min_j, i - min_i
        img[row, col] = np.array(cmap(norm(fy)))

    aspect = W / H
    fig_h  = 14
    fig_w  = max(10, fig_h * aspect + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("#181820")
    ax.set_facecolor("#181820")

    # ci was built from -cloud[:,0] so min_i maps to the most-positive original X.
    # The display X axis now runs east (right) as expected.
    extent = [min_i * CELL, (max_i + 1) * CELL,
              min_j * CELL, (max_j + 1) * CELL]
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")

    # Trajectory path (negate X to match display convention)
    if traj is not None and len(traj) > 1:
        ax.plot(-traj[:, 0], traj[:, 1], "-", color="#00aaff",
                linewidth=1.2, alpha=0.7, zorder=19, label="path")
        step = max(1, len(traj) // 80)
        ax.plot(-traj[::step, 0], traj[::step, 1], ".",
                color="#00aaff", markersize=2, alpha=0.5, zorder=19)

    # Bot pose (final position)
    pose = meta.get("pose", {})
    t    = pose.get("t", [0, 0, 0])
    yaw  = pose.get("yaw", 0.0)
    if t:
        bx, bz = -float(t[0]), float(t[2])   # negate X
        ax.plot(bx, bz, "o", color="#00ff88", markersize=9,
                markeredgecolor="white", markeredgewidth=0.8, zorder=20)
        arrow_len = max(W, H) * CELL * 0.025
        ax.annotate("",
                    xy=(bx - np.sin(yaw)*arrow_len, bz + np.cos(yaw)*arrow_len),
                    xytext=(bx, bz),
                    arrowprops=dict(arrowstyle="->", color="#00ff88", lw=1.5),
                    zorder=21)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("elevation (m)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Scale bar
    sb_len = 10.0
    sb_x0  = extent[0] + (extent[1]-extent[0])*0.04
    sb_z0  = extent[2] + (extent[3]-extent[2])*0.03
    ax.plot([sb_x0, sb_x0+sb_len], [sb_z0, sb_z0], "-", color="white", lw=2, zorder=15)
    ax.text(sb_x0+sb_len/2, sb_z0+(extent[3]-extent[2])*0.01,
            "10 m", color="white", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("X  (world east, m)", color="white", fontsize=10)
    ax.set_ylabel("Z  (world north, m)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    n_vox  = meta.get("voxels", "?")
    n_seen = meta.get("seen_cells", "?")
    ax.set_title(
        f"Kings Row  ·  {n_vox:,} voxels  ·  {len(cells):,} cells  ·  "
        f"{n_seen:,} seen  ·  elev {y_lo:.1f}–{y_hi:.1f} m",
        color="white", fontsize=10, pad=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"saved → {out_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map",      default="kingsrow")
    ap.add_argument("--maps-dir", default="maps")
    ap.add_argument("--out",      default=None,
                    help="output PNG path (default: maps/<map>/floorplan.png)")
    ap.add_argument("--xlim",     nargs=2, type=float, default=None,
                    metavar=("XMIN", "XMAX"),
                    help="world-X crop window in metres")
    ap.add_argument("--zlim",     nargs=2, type=float, default=None,
                    metavar=("ZMIN", "ZMAX"),
                    help="world-Z crop window in metres")
    args = ap.parse_args()

    map_dir  = Path(args.maps_dir) / args.map
    out_path = args.out or str(map_dir / "floorplan.png")

    cloud, meta, traj = load_cloud(map_dir)
    print(f"loaded {len(cloud):,} voxels from {map_dir}")
    if traj is not None:
        print(f"trajectory: {len(traj)} poses")
    hmap = build_grid(cloud)
    print(f"cells after density filter: {len(hmap):,}")
    render(hmap, meta, traj, out_path, xlim=args.xlim, zlim=args.zlim)


if __name__ == "__main__":
    main()

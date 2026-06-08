#!/usr/bin/env python3
"""RANSAC alignment of multiple events into one common world frame.

For each pair (ref, target):
  1. Load features.npz from each event (output of extract-features-3d.py).
  2. Build candidate matches: any (target_feat, ref_feat) sharing the
     same (layer, channel). Cap per (layer, channel) by response.
  3. RANSAC: sample 3 non-collinear correspondences, fit a rigid R, t
     via Kabsch, score by inlier count under --inlier-tol.
  4. Best transform aligns target -> ref world frame. Save to
     <target>/align.json and apply to <target>/pointcloud.ply, writing
     a merged.ply in the ref event dir.

Since both events are in OW-units (or both in placeholder-units) the
transform is rigid (rotation + translation, no scale).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    raise SystemExit("pip install scipy")


# ---------- Geometry ----------
def R_no_roll(yaw, pitch):
    """R = R_yaw(yaw) @ R_pitch(pitch). Rotation around world Y then
    around the yawed X. No roll."""
    cy = np.cos(yaw); sy = np.sin(yaw)
    cp = np.cos(pitch); sp = np.sin(pitch)
    return np.array([
        [ cy,  sy * sp,  sy * cp],
        [ 0.0,      cp,      -sp],
        [-sy,  cy * sp,  cy * cp],
    ])


def fit_no_roll(A, B, init=None):
    """Find (R(yaw, pitch), t) minimizing ||B - (A @ R.T + t)||^2.
    A, B : (N, 3). Uses LM via scipy.

    Returns (R, t, yaw, pitch)."""
    def residuals(params, A, B):
        yaw, pitch, tx, ty, tz = params
        R = R_no_roll(yaw, pitch)
        pred = A @ R.T + np.array([tx, ty, tz])
        return (pred - B).ravel()

    if init is None:
        # Warm-start from full Kabsch, ignoring its roll component.
        Rk, tk = kabsch(A, B)
        yaw0 = np.arctan2(Rk[0, 2], Rk[2, 2])
        pitch0 = np.arcsin(np.clip(-Rk[1, 2], -1.0, 1.0))
        init = np.array([yaw0, pitch0, tk[0], tk[1], tk[2]])

    res = least_squares(residuals, init, args=(A, B),
                        method="lm", max_nfev=50)
    yaw, pitch, tx, ty, tz = res.x
    return R_no_roll(yaw, pitch), np.array([tx, ty, tz]), yaw, pitch


def kabsch(A, B):
    """Rigid R, t such that B ≈ R @ A.T + t (A and B are (N, 3) row arrays)."""
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    Ac = A - cA
    Bc = B - cB
    H = Ac.T @ Bc                          # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T                     # (3, 3) proper rotation
    t = cB - R @ cA                         # (3,)
    return R, t


def apply_rt(R, t, P):
    """Apply (R, t) to row-vector points P (N, 3) -> (N, 3)."""
    return P @ R.T + t


# ---------- PLY I/O for the existing point-cloud format ----------
def read_pointcloud_ply(path: Path):
    pts = []
    cols = []
    with open(path, "r") as f:
        # Header
        line = f.readline()
        if not line.startswith("ply"):
            raise SystemExit(f"{path} is not a ply file")
        n = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("end_header"):
                break
        for _ in range(n):
            parts = f.readline().split()
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(parts) >= 6:
                cols.append([int(parts[5]), int(parts[4]), int(parts[3])])  # BGR
            else:
                cols.append([200, 200, 200])
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)


def write_pointcloud_ply(path: Path, pts, cols_bgr):
    n = len(pts)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (b, g, r) in zip(pts, cols_bgr):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


# ---------- Candidate pair generation ----------
def build_candidates(feat_A, feat_B, top_k_per_class=4):
    """For each (layer, channel) common to both feature sets, take the
    top-K features by response in each event and form their Cartesian
    product as candidate (A_idx, B_idx) pairs.
    """
    keys_A = list(zip(feat_A["layer_id"].tolist(), feat_A["channel"].tolist()))
    keys_B = list(zip(feat_B["layer_id"].tolist(), feat_B["channel"].tolist()))
    by_A: dict[tuple, list[int]] = {}
    by_B: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys_A):
        by_A.setdefault(k, []).append(i)
    for j, k in enumerate(keys_B):
        by_B.setdefault(k, []).append(j)

    pairs_a, pairs_b = [], []
    for k in by_A:
        if k not in by_B:
            continue
        # Take top-K by response in each event
        a_idx = np.array(by_A[k], dtype=np.int64)
        b_idx = np.array(by_B[k], dtype=np.int64)
        if a_idx.size > top_k_per_class:
            order = np.argsort(-feat_A["response"][a_idx])[:top_k_per_class]
            a_idx = a_idx[order]
        if b_idx.size > top_k_per_class:
            order = np.argsort(-feat_B["response"][b_idx])[:top_k_per_class]
            b_idx = b_idx[order]
        for a in a_idx:
            for b in b_idx:
                pairs_a.append(a)
                pairs_b.append(b)
    return np.array(pairs_a, dtype=np.int64), np.array(pairs_b, dtype=np.int64)


# ---------- RANSAC ----------
def ransac_align(A_xyz, B_xyz, n_iter=2000, inlier_tol=0.3, min_inliers=20,
                 seed=0, no_roll=True):
    """A_xyz, B_xyz: (N, 3) candidate-paired points (same length).
    no_roll=True constrains R to R_yaw @ R_pitch (no roll about camera
    forward), matching the OW camera's actual DOF.

    Returns (R, t, n_inliers, inlier_mask, best_resid) or (None, ..., 0, ...).
    """
    N = len(A_xyz)
    if N < 3:
        return None, None, 0, np.zeros(N, dtype=bool), float("inf")

    rng = np.random.default_rng(seed)
    best_n_in = 0
    best = (None, None, np.zeros(N, dtype=bool), float("inf"))

    for it in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        Pa = A_xyz[idx]; Pb = B_xyz[idx]
        # Non-colinearity check
        e1 = Pa[1] - Pa[0]; e2 = Pa[2] - Pa[0]
        if np.linalg.norm(np.cross(e1, e2)) < 1e-3:
            continue
        e1b = Pb[1] - Pb[0]; e2b = Pb[2] - Pb[0]
        if np.linalg.norm(np.cross(e1b, e2b)) < 1e-3:
            continue

        if no_roll:
            try:
                R, t, _, _ = fit_no_roll(Pa, Pb)
            except Exception:
                continue
        else:
            R, t = kabsch(Pa, Pb)
        residuals = np.linalg.norm(apply_rt(R, t, A_xyz) - B_xyz, axis=1)
        inliers = residuals < inlier_tol
        n_in = int(inliers.sum())
        if n_in > best_n_in:
            best_n_in = n_in
            best = (R, t, inliers, float(residuals[inliers].mean()
                                          if n_in else float("inf")))

    if best_n_in < min_inliers:
        return None, None, best_n_in, best[2], best[3]

    # Refit on all inliers for a better estimate
    R0, t0, inliers, _ = best
    if no_roll:
        R_ref, t_ref, _, _ = fit_no_roll(A_xyz[inliers], B_xyz[inliers])
    else:
        R_ref, t_ref = kabsch(A_xyz[inliers], B_xyz[inliers])
    resid = np.linalg.norm(apply_rt(R_ref, t_ref, A_xyz[inliers])
                            - B_xyz[inliers], axis=1)
    return R_ref, t_ref, int(inliers.sum()), inliers, float(resid.mean())


# ---------- Per-event align ----------
def load_features(event_dir: Path):
    f = event_dir / "features.npz"
    if not f.exists():
        return None
    data = np.load(f)
    return {
        "xyz":       data["xyz"],
        "layer_id":  data["layer_id"],
        "channel":   data["channel"],
        "response":  data["response"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events")
    ap.add_argument("--ref", required=True,
                    help="reference event name (others align to this one)")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="event names to align to ref (default: all others "
                         "with features.npz)")
    ap.add_argument("--top-k-per-class", type=int, default=4,
                    help="cap features per (layer, channel) per event before "
                         "enumerating candidates (default 4 -> up to "
                         "4x4=16 candidates per shared channel)")
    ap.add_argument("--n-iter", type=int, default=4000,
                    help="RANSAC iterations (default 4000)")
    ap.add_argument("--inlier-tol", type=float, default=0.3,
                    help="inlier distance threshold in ow-m (default 0.3)")
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-roll", action="store_true",
                    help="use unconstrained Kabsch instead of no-roll fit. "
                         "By default we constrain to OW's no-roll DOF; "
                         "this flag re-enables roll as a free parameter "
                         "for comparison.")
    ap.add_argument("--merge-output", default=None,
                    help="if set, write merged-into-ref-frame point cloud to "
                         "<ref-dir>/<merge-output>.ply (e.g. merged.ply). "
                         "Includes the ref's own pointcloud and each "
                         "successfully-aligned target's transformed pointcloud.")
    args = ap.parse_args()

    root = Path(args.events_dir)
    ref_dir = root / args.ref
    ref_feat = load_features(ref_dir)
    if ref_feat is None:
        raise SystemExit(f"ref event has no features.npz: {ref_dir}")
    print(f"ref event: {args.ref}  ({len(ref_feat['xyz']):,} features)")

    if args.targets is None:
        targets = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name != args.ref and (d / "features.npz").exists():
                targets.append(d.name)
    else:
        targets = args.targets
    if not targets:
        print("no targets to align")
        return
    print(f"aligning {len(targets)} target(s) to {args.ref}")

    # Optional merge: start with the ref's own pointcloud.
    merged_pts, merged_cols = None, None
    if args.merge_output:
        rp = ref_dir / "pointcloud.ply"
        if rp.exists():
            merged_pts, merged_cols = read_pointcloud_ply(rp)
            print(f"  merge: ref pointcloud has {len(merged_pts):,} points")
        else:
            print(f"  merge: ref has no pointcloud.ply, starting empty")
            merged_pts = np.zeros((0, 3), dtype=np.float32)
            merged_cols = np.zeros((0, 3), dtype=np.uint8)

    for tgt in targets:
        tgt_dir = root / tgt
        tgt_feat = load_features(tgt_dir)
        if tgt_feat is None:
            print(f"  [skip] {tgt}: no features.npz")
            continue
        print(f"\n-- {tgt}: {len(tgt_feat['xyz']):,} features --")

        pa_idx, pb_idx = build_candidates(
            tgt_feat, ref_feat,  # align target -> ref
            top_k_per_class=args.top_k_per_class,
        )
        if len(pa_idx) < 3:
            print(f"  [fail] only {len(pa_idx)} candidates "
                  f"(need shared (layer, channel) features)")
            continue
        A = tgt_feat["xyz"][pa_idx]
        B = ref_feat["xyz"][pb_idx]
        print(f"  candidates: {len(pa_idx):,} pairs across "
              f"{len(set(zip(tgt_feat['layer_id'][pa_idx].tolist(), tgt_feat['channel'][pa_idx].tolist())))} "
              f"shared (layer, channel) classes")

        t0 = time.perf_counter()
        R, t, n_in, inliers, resid = ransac_align(
            A, B, n_iter=args.n_iter,
            inlier_tol=args.inlier_tol,
            min_inliers=args.min_inliers,
            seed=args.seed,
            no_roll=not args.allow_roll,
        )
        elapsed = time.perf_counter() - t0
        if R is None:
            print(f"  [fail] best inliers={n_in} < {args.min_inliers} "
                  f"({elapsed:.1f}s)")
            continue
        # Yaw / pitch / roll from R for sanity
        # In our world frame (Y down): yaw is rotation about Y; pitch about X.
        # For diagnostics, just report euler angles ZYX-ish.
        pitch = np.arcsin(np.clip(-R[1, 2], -1, 1))
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[1, 0], R[1, 1])
        print(f"  [ok ] inliers={n_in}/{len(pa_idx)} "
              f"({100*n_in/len(pa_idx):.1f}%)  mean_resid={resid:.3f}m  "
              f"yaw={np.degrees(yaw):+.1f}deg  pitch={np.degrees(pitch):+.1f}deg  "
              f"roll={np.degrees(roll):+.1f}deg  "
              f"|t|={np.linalg.norm(t):.2f}m  ({elapsed:.1f}s)")

        # Save per-target align.json
        align_out = {
            "ref_event": args.ref,
            "R": R.tolist(),
            "t": t.tolist(),
            "n_inliers": int(n_in),
            "n_candidates": int(len(pa_idx)),
            "mean_residual_m": float(resid),
            "inlier_tol_m": float(args.inlier_tol),
            "yaw_deg": float(np.degrees(yaw)),
            "pitch_deg": float(np.degrees(pitch)),
            "roll_deg": float(np.degrees(roll)),
            "translation_m": float(np.linalg.norm(t)),
        }
        (tgt_dir / "align.json").write_text(json.dumps(align_out, indent=2))
        print(f"  saved {tgt_dir / 'align.json'}")

        # Merge this target's pointcloud into the ref frame
        if args.merge_output:
            tp = tgt_dir / "pointcloud.ply"
            if tp.exists():
                pts_tgt, cols_tgt = read_pointcloud_ply(tp)
                pts_in_ref = apply_rt(R, t, pts_tgt)
                merged_pts = np.concatenate([merged_pts, pts_in_ref], axis=0)
                merged_cols = np.concatenate([merged_cols, cols_tgt], axis=0)
                print(f"  merged: +{len(pts_tgt):,} pts (target's pointcloud)")
            else:
                print(f"  no pointcloud.ply in {tgt_dir}, nothing to merge")

    if args.merge_output and merged_pts is not None and len(merged_pts) > 0:
        out = ref_dir / args.merge_output
        if not out.suffix:
            out = out.with_suffix(".ply")
        write_pointcloud_ply(out, merged_pts, merged_cols)
        print(f"\nWrote merged pointcloud: {out}  ({len(merged_pts):,} pts)")


if __name__ == "__main__":
    main()

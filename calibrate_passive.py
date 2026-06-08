"""Calibration + passive geometry capture for Mei Cartographer.

Run this script, complete the Mei wall placement to calibrate, then explore
the map yourself.  The bot sits silent (no movement commands) and accumulates
voxels + 3D features from every frame you show it.  Press Ctrl-C when done;
the map is saved to maps/<map>/ for the autonomous explorer to use later.

Usage:
    py -3.12 calibrate_passive.py --map kingsrow
"""
import argparse, json, time
from pathlib import Path

import numpy as np

# reuse all the heavy machinery from mei_cartographer
from mei_cartographer import (
    ScreenIO, DepthSource, DeathDetector,
    GeometryMapper, wait_for_mei_wall, auto_calibrate,
    ALLOWED_KEYS, MOVE_KEYS,
)
import carto_geom as G
import carto_features as FT


def run(args):
    io    = ScreenIO()
    death = DeathDetector()
    K     = G.intrinsics(G.focal_from_fov(args.fov, io.W), io.W, io.H)
    depth = DepthSource(prefer_da=not args.no_da)

    map_dir    = Path(args.maps_dir) / args.map
    calib_path = map_dir / "calibration.json"

    # --- calibration (skipped if cached) ----------------------------------
    da_scale = args.da_scale
    if not calib_path.exists():
        _, da_scale = wait_for_mei_wall(io, depth)

    motion_cfg          = auto_calibrate(io, depth, K, calib_path, da_scale)
    motion_cfg["da_scale"] = da_scale

    # --- feature extractor + mapper ---------------------------------------
    extractor = FT.FeatureExtractor(peak_thresh=args.peak_thresh)
    mapper    = GeometryMapper(map_dir, K, motion_cfg, depth,
                               voxel=args.voxel, depth_step=args.depth_step,
                               extractor=extractor)
    if mapper.load():
        print(f"loaded existing map: {len(mapper.cloud)} voxels, "
              f"{len(mapper.fmap)} 3D features")

    # --- warm up inference models -----------------------------------------
    print("warming up models...", flush=True)
    _wf = io.grab()
    depth.relative(_wf)
    extractor.peaks(_wf)
    del _wf
    print("ready.", flush=True)

    # --- passive loop ------------------------------------------------------
    print()
    print("=== PASSIVE MODE — you drive, bot maps ===")
    print("Walk around the area you want to map.")
    print("Press Ctrl-C when done to save.")
    print()

    period     = 1.0 / args.fps
    prev_frame = None
    last_look  = (0, 0)

    try:
        while True:
            t0    = io.grab()   # reuse variable name to avoid confusion
            frame = t0
            t0    = time.time()

            if death.is_dead(frame):
                time.sleep(period)
                prev_frame = None
                continue

            added, cov, info = mapper.step(frame, prev_frame, 0, 0)
            prev_frame = frame

            el  = time.time() - t0
            loc = "feat" if info["localized"] else "seed"
            print(f"  +{added:4d}vox  [{loc} inl={info['inliers']:3d}]  "
                  f"voxels={len(mapper.cloud):6d}  "
                  f"seen={cov['seen_cells']:4d}  "
                  f"frontier={cov['frontier_cells']:4d}  "
                  f"{el*1000:.0f}ms",
                  flush=True)

            if el < period:
                time.sleep(period - el)

    except KeyboardInterrupt:
        print("\ndone.")

    finally:
        mapper.save()
        print(f"saved -> {map_dir}")
        print(f"  {len(mapper.cloud)} voxels")
        print(f"  {len(mapper.fmap)} 3D features")
        print(f"  {len(mapper.cover.visited)} visited cells")
        print(f"  {len(mapper.cover.seen)} seen cells")
        print(f"  {mapper.cover.stats()['frontier_cells']} frontier cells")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map",        required=True)
    ap.add_argument("--maps-dir",   default="maps")
    ap.add_argument("--fps",        type=float, default=2.0,
                    help="capture rate in Hz (lower = less CPU, 2 is fine)")
    ap.add_argument("--fov",        type=float, default=103.0)
    ap.add_argument("--da-scale",   type=float, default=1.0)
    ap.add_argument("--voxel",      type=float, default=0.15)
    ap.add_argument("--depth-step", type=int,   default=8)
    ap.add_argument("--no-da",      action="store_true")
    ap.add_argument("--peak-thresh",type=float, default=2.5)
    ap.add_argument("--recalibrate",action="store_true",
                    help="delete cached calibration and redo it")
    args = ap.parse_args()

    if args.recalibrate:
        calib = Path(args.maps_dir) / args.map / "calibration.json"
        if calib.exists():
            calib.unlink()
            print("calibration cleared.")

    run(args)


if __name__ == "__main__":
    main()

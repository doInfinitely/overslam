#!/usr/bin/env python3
"""Input log -> per-frame temporal state vector.

Reads the JSONL produced by input-logger.py and resamples to a fixed
rate (default 30 Hz, matching the screen capture). Per timestep emits:
  - binary held-state of each tracked input (keys + mouse buttons)
  - summed mouse dx/dy within the bin
  - timestamp in wall-clock ns and ms-since-start

Output is a single .npz so a TCN trainer can mmap it directly:
  t_wall_ns  : (N,) int64
  t_ms       : (N,) float64
  inputs     : (N, K) uint8  (0/1 held state at bin end)
  input_names: (K,) string list
  mouse_dx   : (N,) float64
  mouse_dy   : (N,) float64
  meta       : dict (fps, start_ns, etc.)

The bin definition is "state at the bin end-time"; mouse deltas are
sum over the bin interval. That matches what an RL/TCN model would
see as "what happened in this frame".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Default input set: keys that actually move Mei's character. Other
# heroes may need additions (e.g. dash abilities on E/Shift for Tracer,
# wall-climb on Space for Genji) -- pass --keys to override per session.
# Keyboard names match input-logger.py's VK_NAMES; mouse buttons are
# appended with a "Mouse_" prefix to avoid collisions with single-letter
# key names.
DEFAULT_KEYS = [
    "W", "A", "S", "D",   # planar movement
    "Space",              # jump
    "LControl",           # crouch
    "LShift",             # Mei Cryo-Freeze (locks motion ~4s)
]
DEFAULT_MOUSE = ["left", "right", "middle", "x1", "x2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logfile", required=True,
                    help="input-logger JSONL produced by input-logger.py")
    ap.add_argument("--output", required=True,
                    help="output .npz path")
    ap.add_argument("--fps", type=int, default=30,
                    help="resampling rate (Hz). Match this to your "
                         "screen-capture fps for frame alignment.")
    ap.add_argument("--start-wall-ns", type=int, default=None,
                    help="explicit t=0 wall-clock anchor (ns). Default: "
                         "the header's started_wall_ns, or first event ts.")
    ap.add_argument("--end-wall-ns", type=int, default=None,
                    help="explicit end wall-clock (ns). Default: footer's "
                         "ended_wall_ns, or last event ts.")
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS,
                    help=f"key names to track (default: {' '.join(DEFAULT_KEYS)})")
    ap.add_argument("--mouse-buttons", nargs="*", default=DEFAULT_MOUSE,
                    help=f"mouse buttons to track (default: {' '.join(DEFAULT_MOUSE)})")
    args = ap.parse_args()

    # Parse JSONL
    events = []
    header = None
    footer = None
    with open(args.logfile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            if t == "header":
                header = e
            elif t == "footer":
                footer = e
            else:
                events.append(e)

    if not events:
        raise SystemExit("no events found in log")

    # Determine time range
    start_ns = (args.start_wall_ns
                or (header and header.get("started_wall_ns"))
                or events[0]["t_wall_ns"])
    end_ns = (args.end_wall_ns
              or (footer and footer.get("ended_wall_ns"))
              or events[-1]["t_wall_ns"])
    if end_ns <= start_ns:
        raise SystemExit(f"bad time range: start={start_ns} end={end_ns}")

    bin_ns = int(1_000_000_000 / args.fps)
    n_bins = int((end_ns - start_ns) // bin_ns) + 1
    bin_ends_ns = start_ns + np.arange(1, n_bins + 1, dtype=np.int64) * bin_ns

    # Track all inputs in one binary vector for output convenience
    input_names = list(args.keys) + [f"Mouse_{b}" for b in args.mouse_buttons]
    K = len(input_names)
    name_to_idx = {name: i for i, name in enumerate(input_names)}

    inputs = np.zeros((n_bins, K), dtype=np.uint8)
    mouse_dx = np.zeros(n_bins, dtype=np.float64)
    mouse_dy = np.zeros(n_bins, dtype=np.float64)
    # Live state we update as events arrive
    held = np.zeros(K, dtype=np.uint8)

    # Statistics for reporting
    n_keys = 0
    n_btns = 0
    n_moves = 0
    n_dropped_keys = 0  # events for keys we don't track

    bin_idx = 0
    for e in events:
        t = e["t_wall_ns"]
        if t < start_ns:
            continue
        if t > end_ns:
            break
        # Advance bin_idx until t fits in the bin ending at bin_ends_ns[bin_idx]
        while bin_idx < n_bins and t >= bin_ends_ns[bin_idx]:
            inputs[bin_idx, :] = held  # snapshot at bin end
            bin_idx += 1
        if bin_idx >= n_bins:
            break
        et = e.get("type")
        if et == "key":
            n_keys += 1
            name = e.get("name")
            idx = name_to_idx.get(name)
            if idx is None:
                n_dropped_keys += 1
                continue
            held[idx] = 1 if e.get("event") == "down" else 0
        elif et == "mouse_button":
            n_btns += 1
            btn = e.get("button")
            idx = name_to_idx.get(f"Mouse_{btn}")
            if idx is None:
                continue
            held[idx] = 1 if e.get("event") == "down" else 0
        elif et == "mouse_move":
            n_moves += 1
            mouse_dx[bin_idx] += e.get("dx", 0)
            mouse_dy[bin_idx] += e.get("dy", 0)

    # Snapshot any remaining bins
    while bin_idx < n_bins:
        inputs[bin_idx, :] = held
        bin_idx += 1

    t_ms = (bin_ends_ns - start_ns).astype(np.float64) / 1_000_000.0

    meta = {
        "fps": args.fps,
        "start_wall_ns": int(start_ns),
        "end_wall_ns":   int(end_ns),
        "duration_s":    float((end_ns - start_ns) / 1e9),
        "n_bins":        int(n_bins),
        "n_inputs":      int(K),
        "logfile":       str(args.logfile),
        "n_events": {
            "key":          n_keys,
            "mouse_button": n_btns,
            "mouse_move":   n_moves,
            "untracked_keys": n_dropped_keys,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t_wall_ns=bin_ends_ns,
        t_ms=t_ms,
        inputs=inputs,
        input_names=np.array(input_names),
        mouse_dx=mouse_dx,
        mouse_dy=mouse_dy,
        meta=json.dumps(meta),
    )
    print(f"Wrote {out}")
    print(f"  duration: {meta['duration_s']:.1f}s "
          f"({n_bins} bins @ {args.fps}Hz)")
    print(f"  events: {n_keys} key, {n_btns} mouse_button, "
          f"{n_moves} mouse_move  ({n_dropped_keys} untracked keys)")
    # Top 5 most-active inputs by frames held
    held_frames = inputs.sum(axis=0)
    top = sorted(zip(held_frames, input_names), reverse=True)[:8]
    print("  most-held inputs:")
    for cnt, name in top:
        if cnt == 0:
            continue
        print(f"    {name:>14}: {cnt:>6} frames "
              f"({100*cnt/n_bins:.1f}% of session)")
    print(f"  mouse total: dx_sum={mouse_dx.sum():.0f}  "
          f"dy_sum={mouse_dy.sum():.0f}  "
          f"|dx|_mean={np.abs(mouse_dx).mean():.2f}/bin  "
          f"|dy|_mean={np.abs(mouse_dy).mean():.2f}/bin")


if __name__ == "__main__":
    main()

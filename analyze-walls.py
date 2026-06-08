#!/usr/bin/env python3
"""Post-process Mei wall events.

Iterates mei_walls/events/<event>/, filters which masks plausibly contain
a wall using two shape properties:
  1. mask base (bottom of largest connected component bbox) sits near the
     vertical midline of the frame -- a real wall is anchored at ground
     level, while masks that touch the bottom of the frame are typically
     dominated by Mei's body / weapon / ground reflection in the diff.
  2. the largest component is approximately rectangular -- fill ratio of
     the bounding box above a threshold.

For accepted events, samples the wall pixels in post.png (mask intersected
with an eroded version of itself to drop the ragged outline), converts to
HSV, and reports per-event + aggregate color stats. Writes:
  - <events-parent>/wall_color_analysis.json
  - <event-dir>/overlay.png  for each accepted event (debug viz)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np


def search_best_rect(mask, blue, base_y_target=0.55, base_y_tol=0.30,
                     widths=(30, 50, 70, 100, 140, 180, 220, 280, 350, 450,
                             550, 700, 850, 1000, 1150, 1280),
                     heights=(30, 50, 70, 100, 140, 180, 240, 320, 420,
                              520, 620, 720),
                     step_px=12):
    """Vectorized search over candidate rectangles.

    Score = IoU(R, M) * base_score, where M is the intersection of the
    diff mask with the blue-color mask (i.e. "pixels that both changed
    AND look like ice"), R is the candidate rectangle, and base_score
    is a triangular kernel peaked at base_y_target (fraction of H).

    IoU peaks when R equals M's extent -- coverage-style scores peaked
    at the smallest box that's fully inside M, which is why earlier
    runs all hit the 200 px floor. Inner loop over positions is
    vectorized via integral images.
    """
    H, W = mask.shape
    target = ((mask > 0) & (blue > 0)).astype(np.float64)
    total_target = float(target.sum())
    if total_target < 1:
        return None, 0.0
    t_int = cv2.integral(target)
    # Also keep mask/blue precisions for reporting (not for scoring).
    m_int = cv2.integral((mask > 0).astype(np.float64))
    b_int = cv2.integral((blue > 0).astype(np.float64))

    best_score = -1.0
    best = None
    for w in widths:
        if w > W:
            continue
        for h in heights:
            if h > H:
                continue
            xs = np.arange(0, W - w + 1, step_px)
            ys = np.arange(0, H - h + 1, step_px)
            if xs.size == 0 or ys.size == 0:
                continue
            X, Y = np.meshgrid(xs, ys)
            X1 = X + w
            Y1 = Y + h
            area = float(w * h)
            t_sum = (t_int[Y1, X1] - t_int[Y, X1]
                     - t_int[Y1, X] + t_int[Y, X])
            iou = t_sum / (area + total_target - t_sum + 1e-6)
            base_frac = Y1 / H
            base_score = np.clip(
                1.0 - np.abs(base_frac - base_y_target) / base_y_tol,
                0.0, 1.0,
            )
            score = iou * base_score
            idx = np.unravel_index(np.argmax(score), score.shape)
            if score[idx] > best_score:
                best_score = float(score[idx])
                m_sum = (m_int[Y1[idx], X1[idx]] - m_int[Y[idx], X1[idx]]
                         - m_int[Y1[idx], X[idx]] + m_int[Y[idx], X[idx]])
                b_sum = (b_int[Y1[idx], X1[idx]] - b_int[Y[idx], X1[idx]]
                         - b_int[Y1[idx], X[idx]] + b_int[Y[idx], X[idx]])
                best = (int(X[idx]), int(Y[idx]), int(w), int(h),
                        float(iou[idx]),
                        float(m_sum / area),
                        float(b_sum / area),
                        float(base_frac[idx]),
                        float(base_score[idx]))
    return best, best_score


def analyze_one(event_dir, args):
    mask_p = event_dir / "mask.png"
    post_p = event_dir / "post.png"
    if not mask_p.exists() or not post_p.exists():
        return None
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    post = cv2.imread(str(post_p))
    if mask is None or post is None:
        return None
    H, W = mask.shape

    # Blue mask of the post image: pixels in the ice-wall hue range.
    hsv = cv2.cvtColor(post, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.array([args.blue_h_min, args.blue_s_min, args.blue_v_min], np.uint8),
        np.array([args.blue_h_max, 255,             255            ], np.uint8),
    )

    # Gun zone: zero out the gun region in BOTH masks so the rectangle
    # search can't pick the scope reflections.
    gx0 = int(W * args.gun_x_frac)
    gy0 = int(H * args.gun_y_frac)
    blue_search = blue.copy()
    blue_search[gy0:, gx0:] = 0
    mask_search = mask.copy()
    mask_search[gy0:, gx0:] = 0

    rec = {"name": event_dir.name}

    # Search the rectangle that maximizes mask_cov * blue_cov * base_score.
    best, best_score = search_best_rect(
        mask_search, blue_search,
        base_y_target=args.base_y_target,
        base_y_tol=args.base_y_tol,
        step_px=args.search_step_px,
    )
    if best is None or best_score < args.min_score:
        rec["verdict"] = "skip"
        rec["reasons"] = [f"best rect score={best_score:.3f}<{args.min_score}"]
        return rec
    x, y, w, h, iou, mask_cov, blue_cov, base_frac, base_score = best
    rec["bbox"]        = (x, y, w, h)
    rec["score"]       = best_score
    rec["iou"]         = iou
    rec["mask_cov"]    = mask_cov
    rec["blue_cov"]    = blue_cov
    rec["base_y_frac"] = base_frac
    rec["base_score"]  = base_score

    # Color sampling: blue pixels of the post image inside the winning rect.
    # That gives us the wall pixels without sky / ceiling contamination
    # because we're already AND-ing with the blue mask.
    sample_mask = np.zeros_like(blue)
    sample_mask[y:y+h, x:x+w] = blue[y:y+h, x:x+w]
    sample_pixels = sample_mask > 0
    n_sample = int(sample_pixels.sum())
    if n_sample < 500:
        rec["verdict"] = "skip"
        rec["reasons"] = [f"too few blue pixels in winning rect ({n_sample})"]
        return rec

    bgr_pix = post[sample_pixels]
    hsv_pix = hsv[sample_pixels]
    rec["verdict"] = "wall"
    rec["sample_px"]   = n_sample
    rec["bgr_mean"]    = [float(v) for v in bgr_pix.mean(axis=0)]
    rec["bgr_median"]  = [int(np.median(bgr_pix[:, c])) for c in range(3)]
    rec["hsv_mean"]    = [float(v) for v in hsv_pix.mean(axis=0)]
    rec["hsv_median"]  = [int(np.median(hsv_pix[:, c])) for c in range(3)]
    rec["h_percentiles"] = [int(np.percentile(hsv_pix[:, 0], p))
                            for p in (10, 25, 50, 75, 90)]
    rec["s_percentiles"] = [int(np.percentile(hsv_pix[:, 1], p))
                            for p in (10, 25, 50, 75, 90)]

    # Overlay viz.
    overlay = post.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    swatch_bgr = tuple(int(v) for v in rec["bgr_median"])
    cv2.rectangle(overlay, (10, 10), (90, 90), swatch_bgr, -1)
    cv2.rectangle(overlay, (10, 10), (90, 90), (255, 255, 255), 1)
    cv2.putText(overlay, f"H={rec['hsv_median'][0]}", (95, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"S={rec['hsv_median'][1]}", (95, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"V={rec['hsv_median'][2]}", (95, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"score={best_score:.3f}", (95, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(event_dir / "overlay.png"), overlay)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="./mei_walls/events",
                    help="root containing per-event directories")
    ap.add_argument("--base-y-target", type=float, default=0.55,
                    help="ideal fraction of H where the wall base sits (default 0.55)")
    ap.add_argument("--base-y-tol", type=float, default=0.30,
                    help="triangular tolerance around base_y_target; score "
                         "falls to 0 at +/- this (default 0.30)")
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="reject events whose best rect score is below this "
                         "(default 0.05). Score = mask_cov * blue_cov * base_score.")
    ap.add_argument("--search-step-px", type=int, default=24,
                    help="step in px between candidate rectangle positions "
                         "(default 24). Smaller = finer search, slower.")
    ap.add_argument("--blue-h-min", type=int, default=90,
                    help="HSV hue lower bound for ice-wall blue (default 90)")
    ap.add_argument("--blue-h-max", type=int, default=130,
                    help="HSV hue upper bound for ice-wall blue (default 130)")
    ap.add_argument("--blue-s-min", type=int, default=50,
                    help="HSV saturation min (default 50; rejects desaturated grays)")
    ap.add_argument("--blue-v-min", type=int, default=40,
                    help="HSV value min (default 40; rejects near-black pixels)")
    ap.add_argument("--gun-x-frac", type=float, default=0.50,
                    help="x fraction where the gun-exclusion zone starts "
                         "(default 0.50). Everything to the right of this AND "
                         "below --gun-y-frac is zeroed before CC analysis.")
    ap.add_argument("--gun-y-frac", type=float, default=0.55,
                    help="y fraction where the gun-exclusion zone starts "
                         "(default 0.55). See --gun-x-frac.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    events_dir = Path(args.events_dir)
    if not events_dir.exists():
        print(f"events dir not found: {events_dir}")
        return
    subs = sorted([d for d in events_dir.iterdir() if d.is_dir()])
    print(f"Analyzing {len(subs)} events in {events_dir}\n")

    accepted = []
    skipped = []
    for d in subs:
        rec = analyze_one(d, args)
        if rec is None:
            print(f"[--] {d.name}: missing mask or post")
            continue
        if rec["verdict"] == "wall":
            accepted.append(rec)
            x, y, w, h = rec["bbox"]
            print(f"[WALL ] {rec['name']}: rect=({x},{y},{w}x{h}) "
                  f"score={rec['score']:.3f} iou={rec['iou']:.2f} "
                  f"mask_cov={rec['mask_cov']:.2f} blue_cov={rec['blue_cov']:.2f} "
                  f"base_y={rec['base_y_frac']:.2f}  "
                  f"BGR_med={rec['bgr_median']} HSV_med={rec['hsv_median']}")
        else:
            skipped.append(rec)
            print(f"[skip ] {rec['name']}: "
                  f"reasons: {', '.join(rec['reasons'])}")

    if not accepted:
        print("\nNo walls accepted. Adjust thresholds and re-run; "
              "raw shape stats are above.")
        return

    # Aggregate
    def col(key, ch):
        return [r[key][ch] for r in accepted]

    def stat(name, vals):
        s = (statistics.stdev(vals) if len(vals) > 1 else 0.0)
        return (f"{name}: mean={statistics.mean(vals):6.1f} "
                f"median={statistics.median(vals):6.1f} "
                f"std={s:5.1f}  range=({min(vals):.0f}, {max(vals):.0f})")

    print(f"\nAggregate over {len(accepted)} wall events:")
    print("  " + stat("B (hsv_median[0]->Hue)", col("hsv_median", 0)))
    print("  " + stat("S (hsv_median[1])     ", col("hsv_median", 1)))
    print("  " + stat("V (hsv_median[2])     ", col("hsv_median", 2)))
    print("  " + stat("BGR.B                  ", col("bgr_median", 0)))
    print("  " + stat("BGR.G                  ", col("bgr_median", 1)))
    print("  " + stat("BGR.R                  ", col("bgr_median", 2)))

    summary = {
        "n_events_total": len(subs),
        "n_walls_accepted": len(accepted),
        "n_skipped": len(skipped),
        "params": {
            "base_y_target": args.base_y_target,
            "base_y_tol": args.base_y_tol,
            "min_score": args.min_score,
            "search_step_px": args.search_step_px,
            "blue_h_range": [args.blue_h_min, args.blue_h_max],
            "blue_s_min": args.blue_s_min,
            "blue_v_min": args.blue_v_min,
            "gun_x_frac": args.gun_x_frac,
            "gun_y_frac": args.gun_y_frac,
        },
        "ice_wall_color": {
            "hsv_median_of_medians": [
                int(statistics.median(col("hsv_median", c))) for c in range(3)
            ],
            "bgr_median_of_medians": [
                int(statistics.median(col("bgr_median", c))) for c in range(3)
            ],
            "h_range_p10_p90": [
                int(min(r["h_percentiles"][0] for r in accepted)),
                int(max(r["h_percentiles"][-1] for r in accepted)),
            ],
            "s_range_p10_p90": [
                int(min(r["s_percentiles"][0] for r in accepted)),
                int(max(r["s_percentiles"][-1] for r in accepted)),
            ],
        },
        "per_event": [
            {k: v for k, v in r.items() if k != "bbox" or True}
            for r in accepted
        ],
    }
    out = events_dir.parent / "wall_color_analysis.json"
    out.write_text(json.dumps(summary, indent=2, default=list))
    print(f"\nWrote {out}")
    print(f"Per-event overlays written to <event>/overlay.png")


if __name__ == "__main__":
    main()

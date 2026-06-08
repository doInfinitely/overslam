#!/usr/bin/env python3.10
"""Classify an arbitrary screenshot against the 51 hero TF-IDF
prototypes, and render where the top hero's signature channels fire.

Run: python3.10 classify_image.py <image.png> [--root captures] [--topn 5]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import saliency_poses as sp   # reuse Featurizer, render_pose, turbo, build_prototypes


def l2(v, eps=1e-9): return v / (np.linalg.norm(v) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--root", default="captures")
    ap.add_argument("--topn", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="channel active for a hero if value >= alpha*max "
                         "across heroes (higher = sparser signature)")
    ap.add_argument("--df-max", type=int, default=6,
                    help="keep only channels active in <= this many heroes")
    ap.add_argument("--out", default="classify_out")
    args = ap.parse_args()

    root = Path(args.root)
    fz = sp.Featurizer()
    protos, names = sp.build_prototypes(root, fz, root / "_protos.npz")
    N = protos.shape[0]
    p = np.clip(protos, 0, None)
    active = p >= args.alpha * (p.max(0, keepdims=True) + 1e-9)   # (N,C)
    df = active.sum(0)                                            # (C,)
    idf = np.log(N / np.maximum(df, 1)).astype(np.float32)
    # Discriminative = strongly active for this hero AND rare overall.
    rare = df <= args.df_max                                      # (C,)
    sig = active & rare[None, :]                                  # (N,C) per-hero signature
    print(f"signature channels/hero: min {sig.sum(1).min()}  "
          f"med {int(np.median(sig.sum(1)))}  max {sig.sum(1).max()}  "
          f"(of {p.shape[1]})")

    feat, rsz = fz.spatial(Path(args.image))     # (C, Hf, Wf)
    if feat is None:
        raise SystemExit("could not read image")
    C, Hf, Wf = feat.shape
    fmap = np.clip(feat.reshape(C, -1), 0, None)             # (C, P) cells

    # TF-IDF-only LOCALIZED match: for each hero, the best single grid
    # cell whose feature vector (restricted to the hero's signature
    # channels) is most cosine-similar to the hero's TF-IDF signature.
    # This finds the hero WHEREVER it appears instead of averaging the
    # whole frame (background + your own weapon).
    best_cell = np.zeros(N, dtype=int)
    peak = np.zeros(N, dtype=np.float32)
    for h in range(N):
        m = sig[h]
        if m.sum() == 0:
            continue
        w = l2(p[h][m] * idf[m])                             # (k,)
        Q = fmap[m].T * idf[m]                               # (P, k)
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        cos = Qn @ w                                         # (P,)
        best_cell[h] = int(cos.argmax()); peak[h] = float(cos.max())

    order = np.argsort(-peak)[:args.topn]
    print(f"\n[tf-idf localized match] top-{args.topn} (peak cell cosine):")
    for r, h in enumerate(order, 1):
        cy, cx = divmod(best_cell[h], Wf)
        print(f"  {r}. {names[h]:14s} {peak[h]:.3f}  @cell({cx},{cy})  "
              f"({int(sig[h].sum())} sig ch)")

    # Render saliency for the top hero using ONLY its signature channels.
    top = order[0]
    char_w = (p[top] * idf * sig[top]).astype(np.float32)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem.replace(" ", "_")
    op = outdir / f"{stem}__{names[top]}.png"
    sp.render_pose(rsz, feat, char_w, 6,
                   f"BEST: {names[top]}  (peak {peak[top]:.2f})", op, font)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()

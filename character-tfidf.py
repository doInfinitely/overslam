#!/usr/bin/env python
"""One-shot character recognition prototype via channel TF-IDF.

Given two reference screenshots of distinct characters (wisp_tracer and
baihu_genji), find the ResNet50 feature channels that fire strongly for
one and not the other -- "Tracer channels" vs "Genji channels". Then
test whether those Tracer channels also fire on a held-out default
Tracer screenshot (one-shot generalization across skins).

Outputs (in <screenshots>/saliency/):
    wisp_crop.png, genji_crop.png, default_crop.png    -- middle thirds
    wisp_TRACER.png      -- wisp under Tracer-discriminative weights
    genji_GENJI.png      -- genji under Genji-discriminative weights
    default_TRACER.png   -- default tracer under the SAME Tracer weights
                            learned from wisp (the one-shot test)
    default_GENJI.png    -- default tracer under Genji weights
                            (negative control -- should be dim)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


INPUTS = {
    # TF-IDF weights are learned from these two only:
    "wisp":      "Screenshot 2026-06-01 095625.png",  # will-o-wisp Tracer, standing
    "genji":     "Screenshot 2026-06-01 095653.png",  # baihu Genji, standing
    # Test image — weights are *applied* but it does NOT influence learning:
    "wispdance": "Screenshot 2026-06-01 102008.png",  # will-o-wisp Tracer, dancing
}
LEARN_PAIR = ("wisp", "genji")


def crop_middle_third(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((w // 3, 0, 2 * w // 3, h))


def make_input_tensor(img: Image.Image, short_side: int = 384):
    w, h = img.size
    if w < h:
        new_w = short_side
        new_h = int(round(h / w * short_side / 32)) * 32
    else:
        new_h = short_side
        new_w = int(round(w / h * short_side / 32)) * 32
    img_r = img.resize((new_w, new_h), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img_r), (new_w, new_h)


@torch.no_grad()
def extract_layer3(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    out = {}
    h = model.layer3.register_forward_hook(
        lambda m, i, o: out.__setitem__("x", o))
    try:
        model(x.unsqueeze(0))
    finally:
        h.remove()
    return out["x"][0].cpu().float().numpy()


def pairwise_tfidf(desc_a: np.ndarray, desc_b: np.ndarray,
                   alpha: float = 0.6):
    """For two mean-pooled channel-descriptor vectors (one per image),
    return per-channel weights (w_a, w_b) that emphasise channels
    uniquely active in each image and zero out channels that fire in
    both.

    A channel c is "active" in image i iff desc_i[c] >= alpha * max(a,b).
    With N=2 docs, df=2 -> IDF=0 (shared, suppressed),
                  df=1 -> IDF=log 2 (unique to one image)."""
    desc = np.stack([desc_a, desc_b], axis=0)               # (2, C)
    max_c = desc.max(axis=0, keepdims=True) + 1e-9
    active = desc >= alpha * max_c                          # (2, C) bool
    df = active.sum(axis=0).astype(np.int32)                # (C,) ∈ {0,1,2}
    df_safe = np.maximum(df, 1)
    idf = np.log(2.0 / df_safe).astype(np.float32)          # 0 if both, log2 if one
    tf = np.clip(desc, 0, None)                             # (2, C)
    w = tf * idf[None, :] * active.astype(np.float32)       # (2, C)
    return w[0], w[1], df


def saliency_overlay(rgb: np.ndarray, sal: np.ndarray,
                     color_bgr: tuple, threshold: float = 0.35,
                     gamma: float = 0.6):
    """Threshold the normalized saliency at `threshold` (sub-threshold
    pixels show the grayscale base only), then tint the suprathreshold
    pixels with a single monochrome `color_bgr` so each map has a
    distinct identity colour."""
    H, W = rgb.shape[:2]
    sal = cv2.resize(sal.astype(np.float32), (W, H),
                     interpolation=cv2.INTER_CUBIC)
    sal -= sal.min()
    sal /= (sal.max() + 1e-9)
    sal = np.where(sal > threshold,
                   (sal - threshold) / (1.0 - threshold),
                   0.0)
    sal = sal ** gamma
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    tint = np.array(color_bgr, dtype=np.float32)[None, None, :]
    a = sal[:, :, None]
    out = gray_bgr * (1.0 - a) + tint * a
    return out.clip(0, 255).astype(np.uint8)


def apply_weights(feat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """feat: (C, H, W); weights: (C,) -> saliency map (H, W)."""
    return (feat * weights[:, None, None]).sum(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/home/remy/overslam/screenshots")
    ap.add_argument("--out", default=None)
    ap.add_argument("--short-side", type=int, default=384)
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="active threshold: channel is active in image i "
                         "if its activation >= alpha * max across images")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="per-map saliency cutoff (0..1) below which the "
                         "pixel stays grayscale; above which it gets "
                         "tinted (default 0.35)")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out) if args.out else src / "saliency"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ResNet50 (device={device})...")
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(device)

    crops = {}
    feats = {}
    for name, fname in INPUTS.items():
        img = Image.open(src / fname).convert("RGB")
        crop = crop_middle_third(img)
        crops[name] = crop
        crop.save(out_dir / f"{name}_crop.png")
        x, _ = make_input_tensor(crop, args.short_side)
        f = extract_layer3(model, x.to(device))
        feats[name] = f
        print(f"  {name}: cropped {img.size}->{crop.size}, feat {f.shape}")

    # Mean-pool the spatial feature maps to get per-image descriptors.
    desc = {k: f.mean(axis=(1, 2)) for k, f in feats.items()}

    # Learn discriminative weights from the LEARN_PAIR only.
    a_name, b_name = LEARN_PAIR
    tracer_w, genji_w, df = pairwise_tfidf(desc[a_name], desc[b_name],
                                            alpha=args.alpha)
    print(f"\nLearning Tracer/Genji weights from: {a_name}  vs  {b_name}")
    n_both    = int((df == 2).sum())
    n_tracer  = int((df == 1) & (tracer_w > 0)).sum() if False else int((tracer_w > 1e-9).sum())
    n_genji   = int((genji_w > 1e-9).sum())
    print(f"\nchannel doc-frequency (alpha={args.alpha}):")
    print(f"  df=2 (active in BOTH wisp & genji, IDF=0, suppressed): {n_both}")
    print(f"  Tracer-discriminative channels (unique to wisp): {n_tracer}")
    print(f"  Genji-discriminative channels  (unique to genji): {n_genji}")

    # Per-concept colour: Tracer-channels -> hot magenta, Genji-channels
    # -> lime green. Both are distinct from any character's actual skin.
    COLORS = {
        "TRACER": (200,  60, 235),   # BGR magenta/pink
        "GENJI":  ( 60, 235, 100),   # BGR lime green
    }

    cases = [
        # Sanity checks on the two learning images:
        ("wisp",      "TRACER", tracer_w),
        ("genji",     "GENJI",  genji_w),
        # One-shot test: same skin (Tracer), DIFFERENT pose (dancing).
        # Tracer-weights should light up the same Tracer body parts;
        # Genji-weights ideally should NOT.
        ("wispdance", "TRACER", tracer_w),
        ("wispdance", "GENJI",  genji_w),
    ]
    for name, label, w in cases:
        sal = apply_weights(feats[name], w)
        overlay = saliency_overlay(
            np.array(crops[name]), sal,
            color_bgr=COLORS[label], threshold=args.threshold)
        out_p = out_dir / f"{name}_{label}.png"
        cv2.imwrite(str(out_p), overlay)
        s_total = float((feats[name] * w[:, None, None]).clip(0, None).sum())
        print(f"  {out_p.name:30s}  sal range [{sal.min():.1f}, {sal.max():.1f}]  "
              f"total positive response = {s_total:.0f}")


if __name__ == "__main__":
    main()

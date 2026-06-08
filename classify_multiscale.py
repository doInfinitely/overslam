#!/usr/bin/env python3.10
"""Scale-invariant (pyramid) hero classification on an arbitrary image.

Uses the multi-scale feature design from extract-features-3d.py: run an
octave pyramid through ResNet50 layer3, per-channel MAX over scale x
space -> scale-invariant descriptor. Match (plain cosine) against
per-hero prototypes built the same way from standing frames.

Tests the gameplay screenshot two ways:
  - GLOBAL: whole frame.
  - REGION: a crop around the in-world character (the pyramid upsamples
    the small figure to a prototype-like scale).

Run: python3.10 classify_multiscale.py <image> [--region x0 y0 x1 y1]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

PYRAMID = (0.5, 0.707, 1.0, 1.414, 2.0)
SHORT_SIDE = 384
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class MSFeaturizer:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(self.dev)

    @torch.no_grad()
    def _layer3(self, pil):
        w, h = pil.size
        if w < h:
            nw = SHORT_SIDE; nh = int(round(h / w * SHORT_SIDE / 32)) * 32
        else:
            nh = SHORT_SIDE; nw = int(round(w / h * SHORT_SIDE / 32)) * 32
        r = pil.resize((max(32, nw), max(32, nh)), Image.BICUBIC)
        x = _NORM(transforms.functional.to_tensor(r)).unsqueeze(0).to(self.dev)
        out = {}
        hk = self.model.layer3.register_forward_hook(
            lambda m, i, o: out.__setitem__("x", o))
        try:
            self.model(x)
        finally:
            hk.remove()
        return out["x"][0]

    @torch.no_grad()
    def descriptor(self, pil: Image.Image) -> np.ndarray:
        w0, h0 = pil.size
        per = []
        for s in PYRAMID:
            rs = pil.resize((max(32, int(w0 * s)), max(32, int(h0 * s))),
                            Image.BICUBIC)
            per.append(self._layer3(rs).amax(dim=(1, 2)))
        return torch.stack(per).amax(0).cpu().float().numpy()


def list_standing(d): return sorted(d.glob("*/standing/*.png"))
def l2(v): return v / (np.linalg.norm(v) + 1e-9)


def build_protos(root, fz, cache, per_hero=15):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["protos"], list(z["names"])
    heroes = sorted(p for p in root.iterdir() if p.is_dir() and list_standing(p))
    protos, names = [], []
    for hd in heroes:
        st = list_standing(hd)
        idx = np.linspace(0, len(st) - 1, min(per_hero, len(st))).round().astype(int)
        ds = [fz.descriptor(Image.open(st[i]).convert("RGB")) for i in idx]
        protos.append(np.mean(ds, 0)); names.append(hd.name)
        print(f"  proto {hd.name:14s} ({len(ds)} shots)")
    protos = np.stack(protos); np.savez(cache, protos=protos, names=np.array(names))
    return protos, names


def rank(desc, protos, names, tag, topn=6):
    s = np.array([np.dot(l2(desc), l2(protos[h])) for h in range(len(names))])
    order = np.argsort(-s)[:topn]
    print(f"\n[{tag}] top-{topn}:")
    for r, h in enumerate(order, 1):
        print(f"  {r}. {names[h]:14s} {s[h]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--root", default="captures")
    ap.add_argument("--region", nargs=4, type=int, default=None,
                    help="x0 y0 x1 y1 crop around the in-world character")
    args = ap.parse_args()
    root = Path(args.root)
    fz = MSFeaturizer()
    protos, names = build_protos(root, fz, root / "_protos_ms.npz")
    print(f"device={fz.dev}  prototypes={len(names)}  scale-invariant pyramid={PYRAMID}")

    img = Image.open(args.image).convert("RGB")
    rank(fz.descriptor(img), protos, names, "GLOBAL (whole frame)")
    if args.region:
        x0, y0, x1, y1 = args.region
        crop = img.crop((x0, y0, x1, y1))
        crop.save("/tmp/region_crop.png")
        rank(fz.descriptor(crop), protos, names,
             f"REGION crop {args.region} (saved /tmp/region_crop.png)")


if __name__ == "__main__":
    main()

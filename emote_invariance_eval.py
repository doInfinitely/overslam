#!/usr/bin/env python3.10
"""Emote-invariance test for one-shot hero features.

Question: if we build ONE descriptor per hero from a STANDING frame,
does it still recognise that hero in EMOTE frames (different pose)?

Method:
  1. For each hero, take one standing frame -> ResNet50 layer3
     mean-pooled descriptor (1024-d). These 51 vectors are the
     one-shot "gallery" prototypes.
  2. Sample emote frames per hero, extract the same descriptor.
  3. Classify each emote frame by nearest prototype (cosine). Top-1 /
     top-5 accuracy = how pose/emote-invariant the standing features
     are.
  4. Repeat with TF-IDF channel weighting (down-weight channels active
     across many heroes, up-weight hero-discriminative ones) to see if
     the discriminative weighting improves generalization.

Run:  python3.10 emote_invariance_eval.py --root captures [--per-hero 12]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


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
    img_r = img.resize((max(32, new_w), max(32, new_h)), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img_r)


# Octave-spaced pyramid centered on 1.0 -- same design as
# extract-features-3d.py's multi-scale CNN feature extraction.
PYRAMID = (0.5, 0.707, 1.0, 1.414, 2.0)


class Featurizer:
    def __init__(self, multiscale: bool = False, pool: str = "mean"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(self.device)
        self.multiscale = multiscale
        self.pool = pool

    @torch.no_grad()
    def _layer3(self, pil_img):
        x = make_input_tensor(pil_img).unsqueeze(0).to(self.device)
        out = {}
        h = self.model.layer3.register_forward_hook(
            lambda m, i, o: out.__setitem__("x", o))
        try:
            self.model(x)
        finally:
            h.remove()
        return out["x"][0]                       # (C, H, W) tensor

    @torch.no_grad()
    def descriptor(self, path: Path) -> np.ndarray | None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None
        crop = crop_middle_third(img)
        if not self.multiscale:
            feat = self._layer3(crop)
            if self.pool == "max":
                return feat.amax(dim=(1, 2)).cpu().float().numpy()
            return feat.mean(dim=(1, 2)).cpu().float().numpy()
        # Scale-invariant: per-channel MAX over space at each pyramid
        # level, then MAX across levels (the strongest response of each
        # channel at any scale). Channel identity is the invariant.
        w0, h0 = crop.size
        per_scale = []
        for s in PYRAMID:
            rs = crop.resize((max(32, int(w0 * s)), max(32, int(h0 * s))),
                             Image.BICUBIC)
            feat = self._layer3(rs)
            per_scale.append(feat.amax(dim=(1, 2)))      # (C,)
        return torch.stack(per_scale).amax(dim=0).cpu().float().numpy()


def list_standing(hero_dir: Path):
    return sorted(hero_dir.glob("*/standing/*.png"))


def list_emotes(hero_dir: Path):
    return sorted(hero_dir.glob("*/emote__*/*.png"))


def l2norm(v, axis=-1, eps=1e-9):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def tfidf_weights(protos: np.ndarray, alpha: float = 0.5):
    """protos: (N, C) hero prototypes. A channel is 'active' for a hero
    if its value >= alpha * max over heroes. df = #heroes active; idf =
    log(N/df). Channels active in every hero (common texture) -> ~0."""
    N = protos.shape[0]
    p = np.clip(protos, 0, None)
    mx = p.max(axis=0, keepdims=True) + 1e-9
    active = p >= alpha * mx
    df = np.maximum(active.sum(axis=0), 1)
    idf = np.log(N / df).astype(np.float32)
    return idf


def evaluate(protos, hero_names, tests, test_labels, weights=None,
             tag=""):
    P = protos.copy()
    T = tests.copy()
    if weights is not None:
        P = P * weights[None, :]
        T = T * weights[None, :]
    P = l2norm(P)
    T = l2norm(T)
    sims = T @ P.T                              # (n_test, N)
    order = np.argsort(-sims, axis=1)
    top1 = order[:, 0]
    correct1 = (top1 == test_labels).mean()
    top5 = order[:, :5]
    correct5 = np.array([test_labels[i] in top5[i]
                         for i in range(len(test_labels))]).mean()
    print(f"\n[{tag}] top-1 = {correct1*100:.1f}%   top-5 = {correct5*100:.1f}%   "
          f"(n={len(test_labels)} emote frames, {len(hero_names)} heroes)")
    # per-hero top-1
    per = {}
    for i, lab in enumerate(test_labels):
        per.setdefault(lab, []).append(top1[i] == lab)
    worst = sorted(((np.mean(v), hero_names[k]) for k, v in per.items()))[:8]
    print(f"  worst heroes (top-1): " +
          ", ".join(f"{name} {acc*100:.0f}%" for acc, name in worst))
    return correct1, correct5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="captures")
    ap.add_argument("--per-hero", type=int, default=12,
                    help="emote frames sampled per hero")
    ap.add_argument("--proto-shots", type=int, default=0,
                    help="standing frames averaged into each hero "
                         "prototype (0 = use ALL standing frames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--multiscale", action="store_true",
                    help="use the scale-invariant pyramid (max over "
                         "scale x space) instead of single-scale pooling")
    ap.add_argument("--pool", default="mean", choices=["mean", "max"],
                    help="single-scale spatial pooling (ignored if "
                         "--multiscale)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    root = Path(args.root)
    heroes = sorted(d for d in root.iterdir()
                    if d.is_dir() and list_standing(d))
    fz = Featurizer(multiscale=args.multiscale, pool=args.pool)
    print(f"device={fz.device}  heroes={len(heroes)}  "
          f"multiscale={args.multiscale}  pool={args.pool}")

    proto_list, names, tests, labels = [], [], [], []
    for hi, hd in enumerate(heroes):
        st = list_standing(hd)
        # Prototype = mean descriptor over the standing turntable.
        if args.proto_shots and args.proto_shots < len(st):
            idx = np.linspace(0, len(st) - 1, args.proto_shots).round().astype(int)
            shots = [st[i] for i in idx]
        else:
            shots = st
        ds = [d for d in (fz.descriptor(p) for p in shots) if d is not None]
        if not ds:
            continue
        proto_list.append(np.mean(ds, axis=0))
        names.append(hd.name)
        em = list_emotes(hd)
        rng.shuffle(em)
        got = 0
        for p in em:
            if got >= args.per_hero:
                break
            dv = fz.descriptor(p)
            if dv is None:
                continue
            tests.append(dv); labels.append(len(names) - 1); got += 1
        print(f"  [{hi+1}/{len(heroes)}] {hd.name:14s} proto<{len(ds)} shots> "
              f"+ {got} emote frames")

    protos = np.stack(proto_list)               # (N, C)
    tests = np.stack(tests)                      # (n, C)
    labels = np.array(labels)

    # Plain cosine NN on raw mean-pooled descriptors.
    evaluate(protos, names, tests, labels, weights=None, tag="plain cosine")
    # TF-IDF channel-weighted.
    w = tfidf_weights(protos)
    evaluate(protos, names, tests, labels, weights=w, tag="tf-idf weighted")


if __name__ == "__main__":
    main()

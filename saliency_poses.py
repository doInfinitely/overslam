#!/usr/bin/env python3.10
"""Per-pose TF-IDF saliency maps for a hero.

For a given hero we learn its TF-IDF channel signature (which ResNet50
channels are both strongly active for this hero AND rare across the
other 50 heroes), then for each POSE (standing + every emote) we render:

  - the character (grayscale) with a heatmap of the hero's TF-IDF-
    weighted activation overlaid, and
  - the top-K salient grid regions marked + labelled with the TF-IDF
    contribution coming from that region.

One PNG per pose -> saliency_poses/<HERO>/<pose>.png

Prototypes for all heroes are cached to captures/_protos.npz so per-hero
runs are fast after the first.

Run:  python3.10 saliency_poses.py --hero GENJI [--root captures] [--topk 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

SHORT_SIDE = 384


def crop_middle_third(img: Image.Image) -> Image.Image:
    # The char_area captures already isolate the character, so we keep
    # the FULL frame (a middle-third crop would slice the character in
    # half, especially in spread-out emote poses).
    return img


def make_input(img: Image.Image):
    w, h = img.size
    if w < h:
        nw = SHORT_SIDE; nh = int(round(h / w * SHORT_SIDE / 32)) * 32
    else:
        nh = SHORT_SIDE; nw = int(round(w / h * SHORT_SIDE / 32)) * 32
    r = img.resize((max(32, nw), max(32, nh)), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(r), r


class Featurizer:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(self.dev)

    @torch.no_grad()
    def spatial(self, path: Path):
        """Return (feat CxHxW float32 ndarray, the resized RGB crop)."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None, None
        crop = crop_middle_third(img)
        x, rsz = make_input(crop)
        out = {}
        h = self.model.layer3.register_forward_hook(
            lambda m, i, o: out.__setitem__("x", o))
        try:
            self.model(x.unsqueeze(0).to(self.dev))
        finally:
            h.remove()
        return out["x"][0].cpu().float().numpy(), rsz


def list_standing(d): return sorted(d.glob("*/standing/*.png"))
def list_emote_dirs(d): return sorted(p for p in d.glob("*/emote__*") if p.is_dir())


def build_prototypes(root: Path, fz: Featurizer, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["protos"], list(z["names"])
    heroes = sorted(p for p in root.iterdir() if p.is_dir() and list_standing(p))
    protos, names = [], []
    for hd in heroes:
        st = list_standing(hd)
        ds = []
        for p in st:
            f, _ = fz.spatial(p)
            if f is not None:
                ds.append(f.mean(axis=(1, 2)))
        if ds:
            protos.append(np.mean(ds, axis=0)); names.append(hd.name)
            print(f"  proto {hd.name:14s} ({len(ds)} shots)")
    protos = np.stack(protos)
    np.savez(cache, protos=protos, names=np.array(names))
    return protos, names


def turbo(x):
    """Cheap turbo-ish colormap, x in [0,1] -> (R,G,B) uint8 arrays."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render_pose(rsz: Image.Image, feat: np.ndarray, char_w: np.ndarray,
                topk: int, title: str, out_path: Path, font):
    C, Hf, Wf = feat.shape
    sal = np.tensordot(char_w, feat, axes=([0], [0]))        # (Hf, Wf)
    sal_pos = np.clip(sal, 0, None)
    W, H = rsz.size
    sal_norm = sal_pos / (sal_pos.max() + 1e-9)
    heat = Image.fromarray(turbo(sal_norm), "RGB").resize((W, H), Image.BILINEAR)
    gray = rsz.convert("L").convert("RGB")
    # Threshold the overlay: leave low-saliency (background) grayscale,
    # ramp colour only over the salient regions.
    thr = 0.35
    a = np.asarray(Image.fromarray((sal_norm * 255).astype(np.uint8))
                   .resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    a = np.clip((a - thr) / (1 - thr), 0, 1) ** 0.7 * 0.85
    a = a[..., None]
    blend = (np.asarray(gray) * (1 - a) + np.asarray(heat) * a).astype(np.uint8)
    canvas = Image.fromarray(blend)
    draw = ImageDraw.Draw(canvas)

    # top-K salient grid cells, labelled with the hero's TF-IDF
    # activation at that region (the raw weighted value).
    order = np.argsort(-sal.flatten())[:topk]
    cw, ch = W / Wf, H / Hf
    r = int(min(cw, ch) * 0.6)
    for rank, idx in enumerate(order, 1):
        gy, gx = divmod(int(idx), Wf)
        cx, cy = int((gx + 0.5) * cw), int((gy + 0.5) * ch)
        val = float(sal[gy, gx])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=(255, 255, 255), width=2)
        label = f"#{rank} tfidf={val:.1f}"
        tw = 10 + len(label) * 7
        lx = min(cx + r + 2, W - tw)
        draw.rectangle([lx, cy - 10, lx + tw, cy + 10], fill=(0, 0, 0))
        draw.text((lx + 4, cy - 8), label, fill=(255, 255, 0), font=font)

    draw.rectangle([0, 0, W, 22], fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", required=True)
    ap.add_argument("--root", default="captures")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="TF-IDF active threshold (channel active for a hero "
                         "if value >= alpha * max over heroes)")
    ap.add_argument("--out", default="saliency_poses")
    args = ap.parse_args()

    root = Path(args.root)
    fz = Featurizer()
    protos, names = build_prototypes(root, fz, root / "_protos.npz")

    if args.hero.upper() == "ALL":
        targets = list(names)
    elif args.hero.upper() in names:
        targets = [args.hero.upper()]
    else:
        raise SystemExit(f"{args.hero} not found. have: {', '.join(names)}")

    # TF-IDF idf (shared): down-weights channels active across many heroes.
    N = protos.shape[0]
    p = np.clip(protos, 0, None)
    active = p >= args.alpha * (p.max(0, keepdims=True) + 1e-9)
    idf = np.log(N / np.maximum(active.sum(0), 1)).astype(np.float32)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for hero in targets:
        hi = names.index(hero)
        char_w = (p[hi] * idf).astype(np.float32)    # hero's TF-IDF signature
        hd = root / hero
        poses = []
        st = list_standing(hd)
        if st:
            poses.append(("standing", st[len(st) // 2]))
        for ed in list_emote_dirs(hd):
            frames = sorted(ed.glob("*.png"))
            if frames:
                poses.append((ed.name.replace("emote__", ""),
                              frames[len(frames) // 2]))
        outdir = Path(args.out) / hero
        n = 0
        for pose, frame in poses:
            feat, rsz = fz.spatial(frame)
            if feat is None:
                continue
            render_pose(rsz, feat, char_w, args.topk,
                        f"{hero}  /  {pose}", outdir / f"{pose}.png", font)
            n += 1
        print(f"{hero:14s} {int(active[hi].sum()):4d} active ch  "
              f"-> {n} poses  ({outdir})")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

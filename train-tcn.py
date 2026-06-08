#!/usr/bin/env python3
"""Train a TCN that predicts camera motion from input state.

Input  per timestep: K-dim binary inputs ++ [mouse_dx, mouse_dy]
                    -> (K+2)-dim feature vector
Output per timestep: [yaw_rate, pitch_rate, v_forward, v_right, v_up]
                    (5-dim, in camera-local frame, units/sec)

Loads aligned.npz files (output of align-pairs.py). Trains a small TCN
(causal 1D dilated convolutions, residual blocks) with MSE loss. Saves
the best checkpoint by validation MSE.

The dataset is built from sliding windows over each aligned.npz so the
TCN sees enough temporal context for velocity ramps. A held-out split
of full events (not random frames) is used for validation to avoid
temporal leakage.

Usage:
  ./train-tcn.py --aligned-glob 'mei_walls/events/*/aligned.npz' \
                 --out checkpoints/tcn.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np


try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit("pytorch required: pip install torch")


# ---------- Model ----------
class CausalConv1d(nn.Module):
    """Conv1d that pads only on the left, so output[t] depends on input[<=t]."""
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.c2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = (nn.Conv1d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = self.drop(self.relu(self.c1(x)))
        h = self.drop(self.relu(self.c2(h)))
        return self.relu(h + self.skip(x))


class TCN(nn.Module):
    def __init__(self, in_dim, out_dim,
                 channels=(32, 32, 64, 64),
                 kernel=3, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(prev, ch, kernel, dilation=2 ** i,
                                   dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev, out_dim, 1)

    def forward(self, x):
        # x: (B, T, in_dim) -> (B, T, out_dim)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.head(x)
        return x.transpose(1, 2)


# ---------- Dataset ----------
TARGET_KEYS = ("yaw_rate", "pitch_rate", "v_forward", "v_right", "v_up")


class WindowDataset(Dataset):
    """Sliding windows from a list of (X, Y, M) per-event arrays.
    X: (T, in_dim) features, Y: (T, 5) targets, M: (T,) usable mask.
    Only emits windows whose last frame is usable; loss is masked per-frame
    on the rest of the window.
    """
    def __init__(self, events, window=64):
        self.window = window
        self.index = []  # (event_i, start_t)
        self.events = events
        for ei, (X, Y, M) in enumerate(events):
            T = X.shape[0]
            for s in range(0, T - window + 1):
                if M[s + window - 1]:
                    self.index.append((ei, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, s = self.index[i]
        X, Y, M = self.events[ei]
        e = s + self.window
        return (X[s:e].astype(np.float32),
                Y[s:e].astype(np.float32),
                M[s:e].astype(np.float32))


def load_event(npz_path: Path):
    z = np.load(npz_path)
    inputs = z["inputs"].astype(np.float32)         # (T, K)
    dx = z["mouse_dx"].astype(np.float32)[:, None]  # (T, 1)
    dy = z["mouse_dy"].astype(np.float32)[:, None]  # (T, 1)
    X = np.concatenate([inputs, dx, dy], axis=1)    # (T, K+2)
    Y = np.stack([z[k] for k in TARGET_KEYS], axis=1).astype(np.float32)  # (T, 5)
    M = z["usable"].astype(bool)
    return X, Y, M


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned-glob", required=True,
                    help="glob for aligned.npz files (per event)")
    ap.add_argument("--out", required=True,
                    help="output checkpoint path (e.g. checkpoints/tcn.pt)")
    ap.add_argument("--window", type=int, default=64,
                    help="sliding window length (default 64 frames = ~2s @ 30fps)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of EVENTS held out for validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    paths = sorted(glob.glob(args.aligned_glob))
    if not paths:
        raise SystemExit(f"no files matched {args.aligned_glob}")
    events = []
    usable_total = 0
    for p in paths:
        X, Y, M = load_event(Path(p))
        if M.sum() < args.window:
            continue
        events.append((X, Y, M))
        usable_total += int(M.sum())
    if not events:
        raise SystemExit("no events with enough usable frames")
    in_dim = events[0][0].shape[1]
    print(f"Loaded {len(events)} event(s), {usable_total} usable frames total, "
          f"in_dim={in_dim}")

    # Per-event train/val split (avoid temporal leakage within an event)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(events))
    n_val = max(1, int(round(len(events) * args.val_frac)))
    val_idx = set(idx[:n_val].tolist())
    train_events = [events[i] for i in range(len(events)) if i not in val_idx]
    val_events   = [events[i] for i in range(len(events)) if i in val_idx]
    if not train_events:
        raise SystemExit("not enough events for a train split")
    print(f"Split: {len(train_events)} train, {len(val_events)} val event(s)")

    # Normalize targets per-dimension using training stats.
    # (Inputs are already roughly in unit range; mouse deltas may have larger
    # scale -- normalize those too via a simple std rescale.)
    all_y_train = np.concatenate([Y[M] for (_, Y, M) in train_events], axis=0)
    y_mean = all_y_train.mean(axis=0)
    y_std  = all_y_train.std(axis=0) + 1e-6
    all_x_train = np.concatenate([X for (X, _, _) in train_events], axis=0)
    # Only the last 2 columns (mouse_dx, dy) are continuous; keep binary cols.
    x_mean = np.zeros(in_dim, dtype=np.float32)
    x_std  = np.ones(in_dim, dtype=np.float32)
    x_mean[-2:] = all_x_train[:, -2:].mean(axis=0)
    x_std[-2:]  = all_x_train[:, -2:].std(axis=0) + 1e-6
    print(f"y_mean={y_mean}  y_std={y_std}")
    print(f"mouse dx,dy mean={x_mean[-2:]}  std={x_std[-2:]}")

    def normalize(events):
        out = []
        for X, Y, M in events:
            Xn = (X - x_mean) / x_std
            Yn = (Y - y_mean) / y_std
            out.append((Xn, Yn, M))
        return out

    train_norm = normalize(train_events)
    val_norm   = normalize(val_events)

    train_ds = WindowDataset(train_norm, window=args.window)
    val_ds   = WindowDataset(val_norm,   window=args.window)
    print(f"Windows: train={len(train_ds)}, val={len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("no training windows; need more data or smaller --window")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0)

    model = TCN(in_dim=in_dim, out_dim=5).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        n = 0
        for X, Y, M in train_dl:
            X = X.to(args.device); Y = Y.to(args.device); M = M.to(args.device)
            pred = model(X)
            sq = (pred - Y) ** 2
            mask = M.unsqueeze(-1)
            loss = (sq * mask).sum() / (mask.sum() * sq.shape[-1] + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss) * X.size(0)
            n += X.size(0)
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        per_dim = np.zeros(5)
        per_dim_n = 0.0
        with torch.no_grad():
            for X, Y, M in val_dl:
                X = X.to(args.device); Y = Y.to(args.device); M = M.to(args.device)
                pred = model(X)
                sq = (pred - Y) ** 2
                mask = M.unsqueeze(-1)
                loss = (sq * mask).sum() / (mask.sum() * sq.shape[-1] + 1e-6)
                val_loss += float(loss) * X.size(0)
                vn += X.size(0)
                pd = (sq * mask).sum(dim=(0, 1)).cpu().numpy()
                per_dim += pd
                per_dim_n += float(mask.sum())
        val_loss = val_loss / max(vn, 1)
        per_dim = per_dim / max(per_dim_n, 1)

        dt = time.perf_counter() - t0
        print(f"epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"per_dim={np.round(per_dim, 4)}  ({dt:.1f}s)")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "in_dim": in_dim,
                "out_dim": 5,
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "target_keys": TARGET_KEYS,
                "window": args.window,
                "epoch": epoch,
                "val_loss": val_loss,
            }, str(out))
    print(f"\nBest val loss = {best_val:.5f}   checkpoint: {out}")


if __name__ == "__main__":
    main()

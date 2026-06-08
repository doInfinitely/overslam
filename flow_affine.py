#!/usr/bin/env python3
"""
Pyramid affine optical flow with spatial smoothness regularisation.

Each swatch fits a 2x2 affine + translation minimising:
  L = MSE(pre_swatch, warp(post, A, t))  +  λ * ||∇(A,t)||²

The smoothness penalty forces the field to transition gracefully through
texture-free regions (e.g. the white fill around a black box outline)
rather than picking up spurious flow there.

Coarse-to-fine: optimise at 1/4 → 1/2 → full resolution, each level
initialised by the upscaled field from the level below.

Usage (test mode):
    python flow_affine.py
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_t(img: np.ndarray, device) -> torch.Tensor:
    """HxW uint8 grayscale → 1x1xHxW float32 [0,1] tensor."""
    return torch.tensor(img.astype(np.float32) / 255.0,
                        device=device).unsqueeze(0).unsqueeze(0)


def _gauss_blur(t: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    """1x1xHxW — Gaussian smooth so pixel centres are soft Dirac deltas."""
    r  = int(math.ceil(3 * sigma))
    ks = 2 * r + 1
    xs = torch.arange(ks, dtype=t.dtype, device=t.device) - r
    g  = torch.exp(-0.5 * (xs / sigma) ** 2)
    g  = g / g.sum()
    k  = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)   # 1x1xKxK
    return F.conv2d(t, k, padding=r)


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return img
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor),
                      interpolation=cv2.INTER_AREA)


# ── single-level solver ───────────────────────────────────────────────────────

def _fit_level(pre_raw: torch.Tensor, post_raw: torch.Tensor,
               swatch: int, lam: float, n_iters: int, lr: float,
               sigma_start: float = 4.0, sigma_end: float = 1.0,
               lam_warmup: float = 0.5,
               init_params=None, frame_cb=None, frame_every=5) -> torch.Tensor:
    """
    Fit an (n_y × n_x × 6) affine field at one resolution level.

    params layout per cell: [a00, a01, a10, a11, tx, ty]
    Warp (centre-relative): q = A·(p - c_swatch) + c_swatch + t
      t directly gives the flow vector at the swatch centre.

    sigma_start → sigma_end: Gaussian blur is annealed so the optimizer
    finds a smooth gradient basin at high sigma, then sharpens for accuracy.

    lam_warmup: fraction of iterations before smoothness turns on.
      Phase 1 [0, lam_warmup): data loss only → edge swatches lock to correct flow.
      Phase 2 [lam_warmup, 1]: smoothness linearly ramps to lam → fills interior.

    Returns params tensor (n_y, n_x, 6).
    """
    device = pre_raw.device
    _, _, H, W = pre_raw.shape
    n_y = H // swatch
    n_x = W // swatch
    H_c = n_y * swatch
    W_c = n_x * swatch

    N = n_y * n_x
    S = swatch

    # ── pixel centres within a swatch (subpixel: centre of pixel = +0.5) ─
    ys = torch.arange(S, dtype=torch.float32, device=device) + 0.5
    xs = torch.arange(S, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    local_xy = torch.stack([xx.ravel(), yy.ravel()], dim=1)   # (S², 2)  relative to swatch origin

    # ── swatch origins and centres ───────────────────────────────────────
    oy_v = torch.arange(n_y, dtype=torch.float32, device=device) * S
    ox_v = torch.arange(n_x, dtype=torch.float32, device=device) * S
    oy_g, ox_g = torch.meshgrid(oy_v, ox_v, indexing='ij')
    origins = torch.stack([ox_g.ravel(), oy_g.ravel()], dim=1)   # (N, 2)
    centres = origins + S / 2.0                                    # (N, 2)  swatch centres

    # ── positions relative to each swatch centre ─────────────────────────
    # Shape: (N, S², 2) — used for centre-relative affine
    abs_xy     = local_xy.unsqueeze(0) + origins.unsqueeze(1)     # (N, S², 2) absolute
    local_from_c = abs_xy - centres.unsqueeze(1)                  # (N, S², 2) relative to centre

    # ── initialise params ────────────────────────────────────────────────
    if init_params is not None:
        params = init_params.detach().clone()
    else:
        params = torch.zeros(N, 6, device=device)
        params[:, 0] = 1.0   # a00 = 1 (identity scale)
        params[:, 3] = 1.0   # a11 = 1
    params = params.reshape(N, 6).requires_grad_(True)

    opt = torch.optim.Adam([params], lr=lr)

    # ── pre-compute blurs at discrete sigma levels (log-spaced) ─────────
    # Fewer discrete levels → more iterations per level → better convergence.
    import math
    N_SIGMA = min(8, n_iters)
    sigma_vals = [math.exp(math.log(sigma_start) * (1 - i / max(N_SIGMA - 1, 1))
                           + math.log(sigma_end)  * (i / max(N_SIGMA - 1, 1)))
                  for i in range(N_SIGMA)]
    blurred_pairs = []
    with torch.no_grad():
        for s in sigma_vals:
            pt = _gauss_blur(pre_raw,  sigma=s)[:, :, :H_c, :W_c]
            qt = _gauss_blur(post_raw, sigma=s)[:, :, :H_c, :W_c]
            blurred_pairs.append((pt, qt))

    for it in range(n_iters):
        sigma_idx = int(it / n_iters * (N_SIGMA - 1) + 0.5)
        pre_b, post_b = blurred_pairs[sigma_idx]

        patches = (pre_b.unfold(2, S, S).unfold(3, S, S))[0, 0].reshape(N, 1, S, S).detach()
        pre  = pre_b
        post = post_b

        opt.zero_grad()

        A = params[:, :4].reshape(-1, 2, 2)   # (N, 2, 2)
        t = params[:, 4:]                      # (N, 2)

        # Centre-relative warp: q = A·(p - c) + c + t
        warped = (torch.bmm(local_from_c, A.transpose(1, 2))
                  + centres.unsqueeze(1) + t.unsqueeze(1))        # (N, S², 2)

        # Normalise to [-1,1] for grid_sample
        wx = warped[:, :, 0] / W_c * 2 - 1
        wy = warped[:, :, 1] / H_c * 2 - 1
        grid = torch.stack([wx, wy], dim=2).reshape(N, S, S, 2)

        sampled = F.grid_sample(
            post.expand(N, -1, -1, -1),
            grid, mode='bilinear', padding_mode='border',
            align_corners=False)                                   # (N, 1, S, S)

        data_loss = F.mse_loss(patches, sampled)

        # Smoothness: ramps in after lam_warmup fraction of iterations.
        # Phase 1 (data only): edge swatches converge freely to correct flow.
        # Phase 2 (data + smooth): smoothness fills interior by diffusion.
        frac = it / max(n_iters - 1, 1)
        if frac < lam_warmup:
            effective_lam = 0.0
        else:
            effective_lam = lam * (frac - lam_warmup) / (1.0 - lam_warmup)

        p2d = params.reshape(n_y, n_x, 6)
        dy  = p2d[1:, :, :] - p2d[:-1, :, :]
        dx  = p2d[:, 1:, :] - p2d[:, :-1, :]
        smooth_loss = (dy ** 2).mean() + (dx ** 2).mean()

        loss = data_loss + effective_lam * smooth_loss
        loss.backward()
        opt.step()

        if frame_cb is not None and it % frame_every == 0:
            frame_cb(params.detach().reshape(n_y, n_x, 6), it, float(loss))

    return params.detach().reshape(n_y, n_x, 6)


# ── coarse-to-fine pyramid ────────────────────────────────────────────────────

def pyramid_affine_flow(pre: np.ndarray, post: np.ndarray,
                        levels: int = 3,
                        swatch: int = 16,
                        lam: float = 0.05,
                        n_iters: int = 150,
                        lr: float = 0.02,
                        sigma_start: float = 8.0,
                        sigma_end: float = 0.8,
                        device: str = 'cpu',
                        frame_cb=None,
                        frame_every: int = 5) -> np.ndarray:
    """
    Pyramid affine optical flow.

    pre, post      — HxW uint8 grayscale
    sigma_start    — initial Gaussian blur (high = smooth landscape, good for large displacements)
    sigma_end      — final Gaussian blur (low = sharp, precise)
    Returns dense flow (H, W, 2) in pixels [dx, dy].
    """
    H, W = pre.shape[:2]
    params = None

    for level in range(levels - 1, -1, -1):   # coarse → fine
        factor = 2 ** level
        pre_l  = _downsample(pre,  factor)
        post_l = _downsample(post, factor)

        pre_raw  = _to_t(pre_l,  device)
        post_raw = _to_t(post_l, device)

        Hl, Wl = pre_raw.shape[2:]
        n_y = Hl // swatch
        n_x = Wl // swatch

        # Upsample from coarser level; translations scale ×2 (pixel coords double)
        if params is not None:
            p4d = params.permute(2, 0, 1).unsqueeze(0)
            p_up = F.interpolate(p4d, size=(n_y, n_x),
                                 mode='bilinear', align_corners=False)
            p_up = p_up[0].permute(1, 2, 0).clone()
            p_up[:, :, 4:] *= 2.0
            init = p_up.reshape(n_y * n_x, 6)
        else:
            init = None

        # At coarser levels sigma_start already covers the displacement;
        # sigma_end scales with factor so fine levels still end up sharp.
        level_sigma_end = max(sigma_end, sigma_end * factor * 0.5)

        print(f"  level {level}  ({Hl}×{Wl})  "
              f"grid {n_y}×{n_x}  swatch {swatch}px  "
              f"σ {sigma_start:.1f}→{level_sigma_end:.1f}", flush=True)
        params = _fit_level(pre_raw, post_raw, swatch, lam, n_iters, lr,
                            sigma_start=sigma_start,
                            sigma_end=level_sigma_end,
                            init_params=init,
                            frame_cb=frame_cb, frame_every=frame_every)

    return _dense_flow(params, swatch, H, W, device)


def _dense_flow(params: torch.Tensor, swatch: int,
                H: int, W: int, device: str) -> np.ndarray:
    """Upsample the swatch-level displacement field to (H, W, 2)."""
    n_y, n_x, _ = params.shape
    with torch.no_grad():
        A = params[:, :, :4].reshape(n_y * n_x, 2, 2)
        t = params[:, :, 4:].reshape(n_y * n_x, 2)

        # Centre-relative warp: flow at swatch centre = A·0 + t = t
        # (local_from_c = 0 at swatch centre, so flow = t directly)
        disp = t.reshape(n_y, n_x, 2)                      # (n_y, n_x, 2)

        # Bilinear upsample to full resolution
        d4 = disp.permute(2, 0, 1).unsqueeze(0)            # (1, 2, n_y, n_x)
        d_full = F.interpolate(d4, size=(H, W),
                               mode='bilinear', align_corners=False)
        return d_full[0].permute(1, 2, 0).cpu().numpy()    # (H, W, 2)


# ── test ─────────────────────────────────────────────────────────────────────

def _make_box_zoom(H: int = 256, W: int = 256, zoom: float = 1.08):
    """White background, black box outline; post = zoomed pre."""
    import cv2
    pre = np.full((H, W), 255, dtype=np.uint8)
    x0, y0, x1, y1 = W // 4, H // 4, 3 * W // 4, 3 * H // 4
    th = 4
    pre[y0:y0+th, x0:x1] = 0
    pre[y1-th:y1, x0:x1] = 0
    pre[y0:y1, x0:x0+th] = 0
    pre[y0:y1, x1-th:x1] = 0

    cx, cy = W / 2.0, H / 2.0
    M = np.float32([[zoom, 0, cx * (1 - zoom)],
                    [0, zoom, cy * (1 - zoom)]])
    post = cv2.warpAffine(pre, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return pre, post


def _flow_vis(flow: np.ndarray) -> np.ndarray:
    """Convert (H,W,2) flow to HSV-coloured RGB image."""
    import cv2
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang / (2 * np.pi) + 1) % 1 * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import cv2
    import imageio

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    ZOOM   = 1.08
    N_ITERS = 600   # per level — more to let sigma annealing converge
    LEVELS  = 3
    H, W    = 256, 256

    pre, post = _make_box_zoom(H=H, W=W, zoom=ZOOM)

    cy, cx = H / 2.0, W / 2.0
    ys_g, xs_g = np.mgrid[0:H, 0:W].astype(np.float32)
    # Forward flow: pre pixel at (x,y) lands at zoom*(x-cx)+cx in post.
    # Displacement = (zoom-1)*(x-cx), negative for x < cx (inward zoom).
    gt_flow = np.stack([(ZOOM - 1) * (xs_g - cx),
                        (ZOOM - 1) * (ys_g - cy)], axis=-1)

    step = 16
    ys_q = np.arange(step // 2, H, step)
    xs_q = np.arange(step // 2, W, step)
    yy_q, xx_q = np.meshgrid(ys_q, xs_q, indexing='ij')

    # ── frame builder ────────────────────────────────────────────────────
    gif_frames = []
    level_counter = [LEVELS - 1]   # mutable cell
    global_iter   = [0]

    def _make_frame(params, it, loss, level, full_H=H, full_W=W):
        flow = _dense_flow(params, 16, full_H, full_W, device)
        fig = plt.figure(figsize=(14, 7), facecolor='#0e0e14')
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.35, wspace=0.25)

        def _ax(pos, title, img, cmap=None, alpha=1.0):
            ax = fig.add_subplot(pos)
            ax.set_facecolor('#0e0e14')
            ax.imshow(img, cmap=cmap, origin='upper', alpha=alpha)
            ax.set_title(title, color='white', fontsize=9)
            ax.axis('off')
            return ax

        _ax(gs[0, 0], 'pre', pre, cmap='gray')
        _ax(gs[0, 1], 'post (zoom 1.08×)', post, cmap='gray')
        _ax(gs[0, 2], f'estimated flow  lv{level} it{it:04d}',
            _flow_vis(flow))

        _ax(gs[1, 0], 'ground-truth flow', _flow_vis(gt_flow))

        ax_e = fig.add_subplot(gs[1, 1])
        ax_e.set_facecolor('#0e0e14')
        ax_e.imshow(pre, cmap='gray', origin='upper', alpha=0.35)
        ax_e.quiver(xx_q, yy_q,
                    flow[yy_q, xx_q, 0], flow[yy_q, xx_q, 1],
                    color='cyan', scale=80, width=0.004)
        ax_e.set_title('estimated vectors', color='white', fontsize=9)
        ax_e.axis('off')

        ax_g = fig.add_subplot(gs[1, 2])
        ax_g.set_facecolor('#0e0e14')
        ax_g.imshow(pre, cmap='gray', origin='upper', alpha=0.35)
        ax_g.quiver(xx_q, yy_q,
                    gt_flow[yy_q, xx_q, 0], gt_flow[yy_q, xx_q, 1],
                    color='lime', scale=80, width=0.004)
        ax_g.set_title(f'ground truth   loss={loss:.5f}',
                       color='white', fontsize=9)
        ax_g.axis('off')

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return buf

    def frame_cb(params, it, loss):
        lv = level_counter[0]
        gif_frames.append(_make_frame(params, it, loss, lv))
        global_iter[0] += 1
        if global_iter[0] % 20 == 0:
            print(f"  lv{lv} it{it:4d}  loss={loss:.6f}  "
                  f"frames={len(gif_frames)}", flush=True)

    # ── run pyramid, collecting frames ───────────────────────────────────
    SIGMA_START = 8.0   # high sigma → overlapping blurred edges, smooth landscape
    SIGMA_END   = 0.8   # fine sigma for final precision

    print("running pyramid affine flow (collecting animation frames)...")
    params_out = None

    for level in range(LEVELS - 1, -1, -1):
        level_counter[0] = level
        factor = 2 ** level
        pre_l   = _downsample(pre,  factor)
        post_l  = _downsample(post, factor)
        pre_raw  = _to_t(pre_l,  device)
        post_raw = _to_t(post_l, device)
        Hl, Wl = pre_raw.shape[2:]
        n_y = Hl // 16
        n_x = Wl // 16

        if params_out is not None:
            p4d  = params_out.permute(2, 0, 1).unsqueeze(0)
            p_up = F.interpolate(p4d, size=(n_y, n_x),
                                 mode='bilinear', align_corners=False)
            p_up = p_up[0].permute(1, 2, 0).clone()
            p_up[:, :, 4:] *= 2.0
            init = p_up.reshape(n_y * n_x, 6)
        else:
            init = None

        level_sigma_end = max(SIGMA_END, SIGMA_END * factor * 0.5)
        print(f"\n  level {level}  ({Hl}×{Wl})  grid {n_y}×{n_x}"
              f"  σ {SIGMA_START:.1f}→{level_sigma_end:.1f}", flush=True)
        params_out = _fit_level(pre_raw, post_raw, 16, 0.001, N_ITERS, 0.02,
                                sigma_start=SIGMA_START,
                                sigma_end=level_sigma_end,
                                init_params=init,
                                frame_cb=frame_cb, frame_every=20)

    # ── save GIF ─────────────────────────────────────────────────────────
    gif_path = 'flow_affine_anim.gif'
    print(f"\nsaving {len(gif_frames)} frames → {gif_path} ...")
    imageio.mimsave(gif_path, gif_frames, fps=20, loop=0)
    print(f"done → {gif_path}")

    # ── also save final static frame ──────────────────────────────────────
    flow_final = _dense_flow(params_out, 16, H, W, device)
    imageio.imwrite('flow_affine_final.png',
                    _make_frame(params_out, N_ITERS, 0.0, 0))

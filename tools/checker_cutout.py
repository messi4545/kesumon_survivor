# /// script
# dependencies = ["pillow", "numpy", "scipy"]
# ///
"""4体のチェッカーボード背景を完全除去して evo1..4.png を生成"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

root = Path(__file__).resolve().parent.parent
img_dir = root / "img"

def build_checker_template(lum: np.ndarray):
    h, w = lum.shape
    corners = np.concatenate([
        lum[:70, :70].ravel(), lum[:70, -70:].ravel(),
        lum[-70:, :70].ravel(), lum[-70:, -70:].ravel()
    ])
    c_low = float(np.percentile(corners, 15))
    c_high = float(np.percentile(corners, 85))
    c_mid = 0.5 * (c_low + c_high)
    amp = 0.5 * (c_high - c_low)

    s_top = np.sign(lum[:40, :240] - c_mid).mean(axis=0)
    Ls = np.linspace(24.0, 36.0, 601)[:, None, None]
    phis = np.linspace(0, np.pi * 2, 120, endpoint=False)[None, :, None]
    xs = np.arange(240)[None, None, :]
    pred_x = np.sign(np.sin(np.pi * xs / Ls - phis))
    err_x = np.mean((pred_x - s_top[None, None, :]) ** 2, axis=2)
    ix, ipx = np.unravel_index(np.argmin(err_x), err_x.shape)
    Lx = float(Ls[ix, 0, 0])
    phix = float(phis[0, ipx, 0])

    s_left = np.sign(lum[:240, :40] - c_mid).mean(axis=1)
    Ls_y = np.linspace(Lx - 0.5, Lx + 0.5, 101)[:, None, None]
    pred_y = np.sign(np.sin(np.pi * xs / Ls_y - phis))
    err_y = np.mean((pred_y - s_left[None, None, :]) ** 2, axis=2)
    iy, ipy = np.unravel_index(np.argmin(err_y), err_y.shape)
    Ly = float(Ls_y[iy, 0, 0])
    phiy = float(phis[0, ipy, 0])

    X = np.arange(w)[None, :]
    Y = np.arange(h)[:, None]
    wx = np.clip(np.sin(np.pi * X / Lx - phix) * 10.0, -1.0, 1.0)
    wy = np.clip(np.sin(np.pi * Y / Ly - phiy) * 10.0, -1.0, 1.0)
    T = c_mid + amp * (wx * wy)
    if np.abs(lum[:30, :30] - T[:30, :30]).mean() > np.abs(lum[:30, :30] - (c_mid - amp * (wx * wy))[:30, :30]).mean():
        T = c_mid - amp * (wx * wy)
    is_edge = np.abs(wx * wy) < 0.85
    return T, c_low, c_high, is_edge

def remove_checkerboard_3(im: Image.Image) -> Image.Image:
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = (r + g + b) / 3.0

    T, c_low, c_high, is_edge = build_checker_template(lum)
    diff = np.abs(lum - T)
    diff_min = ndimage.minimum_filter(diff, size=3)

    match_strict = (sat < 14) & ((diff < 9.0) | (is_edge & (diff_min < 9.0) & (lum > c_low - 12)))
    match_loose = (sat < 20) & ((diff < 14.0) | (is_edge & (diff_min < 12.0) & (lum > c_low - 16)))

    t_ac = T - ndimage.uniform_filter(T, size=25)
    l_ac = lum - ndimage.uniform_filter(lum, size=25)
    corr = ndimage.uniform_filter(t_ac * l_ac, size=25) / (ndimage.uniform_filter(t_ac * t_ac, size=25) + 1e-5)

    tinted_checker = (corr > 0.45) & (lum > c_low - 30) & (sat < 65)
    match_strict = match_strict | (tinted_checker & (corr > 0.6))
    match_loose = match_loose | tinted_checker

    mean_diff_box = ndimage.uniform_filter(diff, size=15)
    checker_seed = (match_strict & (mean_diff_box < 8.5) & (sat < 12)) | ((corr > 0.65) & (lum > c_low - 25) & (sat < 55))

    border = np.zeros((h, w), dtype=bool)
    border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = True

    local_ok = (ndimage.uniform_filter(diff, size=13) < 11.5) | (corr > 0.35)
    bg_cand = match_loose & local_ok
    torso_protect = np.zeros((h, w), dtype=bool)
    torso_protect[int(h * 0.29):int(h * 0.76), int(w * 0.38):int(w * 0.64)] = True
    bg_cand = bg_cand & (~torso_protect)

    bg_mask = ndimage.binary_propagation((checker_seed | border) & bg_cand, mask=bg_cand)
    fg_mask = ~bg_mask
    fg_mask = ndimage.binary_opening(fg_mask, structure=np.ones((3, 3)))
    fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((3, 3)))

    lab, n = ndimage.label(fg_mask)
    if n > 0:
        sizes = ndimage.sum(fg_mask, lab, range(1, n + 1))
        max_s = sizes.max()
        fg_mask = np.isin(lab, [k + 1 for k, s in enumerate(sizes) if s >= max(200, max_s * 0.002)])

    bg_lab, bg_n = ndimage.label(~fg_mask)
    if bg_n > 0:
        bg_sizes = ndimage.sum(~fg_mask, bg_lab, range(1, bg_n + 1))
        tiny_holes = np.isin(bg_lab, [k + 1 for k, s in enumerate(bg_sizes) if s < 150])
        fg_mask = fg_mask | tiny_holes

    dist_in = ndimage.distance_transform_edt(fg_mask)
    alpha = np.clip((dist_in - 0.4) * 150.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    out = Image.fromarray(rgba, "RGBA")
    return out.crop(out.getbbox())

def remove_checkerboard_124(im: Image.Image, idx: int) -> Image.Image:
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = (r + g + b) / 3.0

    corners = np.concatenate([
        lum[:80, :80].ravel(), lum[:80, -80:].ravel(),
        lum[-80:, :80].ravel(), lum[-80:, -80:].ravel()
    ])
    c_low = np.percentile(corners, 12)
    c_high = np.percentile(corners, 88)

    mean_lum = ndimage.uniform_filter(lum, size=5)
    mean_sq = ndimage.uniform_filter(lum * lum, size=5)
    std_lum = np.sqrt(np.maximum(0, mean_sq - mean_lum * mean_lum))

    flat_low = (sat < 10) & (np.abs(lum - c_low) < 8.5) & (std_lum < 4.2)
    flat_high = (sat < 10) & (np.abs(lum - c_high) < 6.5) & (std_lum < 4.2)
    flat_low = ndimage.binary_opening(flat_low, structure=np.ones((4, 4)))
    flat_high = ndimage.binary_opening(flat_high, structure=np.ones((4, 4)))

    dil_low = ndimage.binary_dilation(flat_low, iterations=7)
    dil_high = ndimage.binary_dilation(flat_high, iterations=7)
    checker_seed = (flat_low & dil_high) | (flat_high & dil_low)

    is_bg_cand = (sat < 16) & (lum >= c_low - 18)

    if idx == 2:
        # 2枚目の右腕と胴体の間の小さな隙間もシードに追加
        underarm = np.zeros((h, w), dtype=bool)
        underarm[int(h * 0.36):int(h * 0.56), int(w * 0.58):int(w * 0.72)] = True
        checker_seed = checker_seed | (underarm & (sat < 14) & (lum > c_low - 10))

    if idx == 4:
        # 4枚目の腕の下〜腰横のケーブル隙間ゾーンのみ、緑発光が乗った明るい背景も抜く
        cable_gaps = np.zeros((h, w), dtype=bool)
        cable_gaps[int(h * 0.44):int(h * 0.96), int(w * 0.18):int(w * 0.36)] = True
        cable_gaps[int(h * 0.44):int(h * 0.96), int(w * 0.64):int(w * 0.82)] = True
        # ただし銀色の籠手（リストアーマー）部分は除外
        gauntlet = np.zeros((h, w), dtype=bool)
        gauntlet[int(h * 0.44):int(h * 0.68), int(w * 0.08):int(w * 0.28)] = True
        gauntlet[int(h * 0.44):int(h * 0.68), int(w * 0.72):int(w * 0.92)] = True
        cable_gaps = cable_gaps & (~gauntlet)
        gap_bg = cable_gaps & (lum > c_low - 6) & (sat < 26) & (std_lum < 7.0)
        checker_seed = checker_seed | gap_bg
        is_bg_cand = is_bg_cand | gap_bg

    border = np.zeros((h, w), dtype=bool)
    border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = True

    seeds = (checker_seed | border) & is_bg_cand
    bg_mask = ndimage.binary_propagation(seeds, mask=is_bg_cand)

    fg_mask = ~bg_mask
    fg_mask = ndimage.binary_opening(fg_mask, structure=np.ones((3, 3)))
    fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((3, 3)))

    lab, n = ndimage.label(fg_mask)
    if n > 0:
        sizes = ndimage.sum(fg_mask, lab, range(1, n + 1))
        max_s = sizes.max()
        fg_mask = np.isin(lab, [k + 1 for k, s in enumerate(sizes) if s >= max(250, max_s * 0.002)])

    dist_in = ndimage.distance_transform_edt(fg_mask)
    alpha = np.clip((dist_in - 0.5) * 140.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    out = Image.fromarray(rgba, "RGBA")
    return out.crop(out.getbbox())

prev = Image.new("RGBA", (260 * 4, 300), (200, 40, 160, 255))
for i in range(1, 5):
    im = Image.open(img_dir / f"src{i}.jpg")
    out = remove_checkerboard_3(im) if i == 3 else remove_checkerboard_124(im, i)
    out.save(img_dir / f"evo{i}.png")
    print("saved", i, out.size)
    th = out.copy()
    th.thumbnail((250, 280))
    prev.alpha_composite(th, ((i - 1) * 260 + 5, 10))
prev.save(root / "tools" / "preview.png")

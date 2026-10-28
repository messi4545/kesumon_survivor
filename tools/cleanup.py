# /// script
# dependencies = ["pillow", "numpy", "scipy"]
# ///
"""isnet の切り抜き結果を後処理（半透明の霧を除去・小さな島を除去・余白トリム）して evo*.png を出力"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

root = Path(__file__).resolve().parent.parent
for i in range(1, 5):
    im = Image.open(root / "tools" / "variants" / f"isnet-general-use_{i}.png").convert("RGBA")
    a = np.asarray(im).astype(np.float32)
    alpha = a[..., 3]
    # 半透明部分をカット（コントラスト強調）
    alpha = np.clip((alpha - 150) * 3.0, 0, 255)
    # 最大連結成分＋大きめの成分だけ残す
    mask = alpha > 20
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        keep = np.isin(lab, [k + 1 for k, s in enumerate(sizes) if s >= sizes.max() * 0.03])
        alpha[~keep] = 0
    a[..., 3] = alpha
    out = Image.fromarray(a.astype(np.uint8), "RGBA")
    # エッジを少し滑らかに
    al = out.getchannel("A").filter(ImageFilter.GaussianBlur(0.8))
    out.putalpha(al)
    out = out.crop(out.getbbox())
    out.save(root / "img" / f"evo{i}.png")
    print("saved", i, out.size)

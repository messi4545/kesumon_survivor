# /// script
# dependencies = ["rembg[cpu]", "pillow"]
# ///
"""複数モデルで背景除去を試し、比較用プレビューを作る"""
import sys
from pathlib import Path
from PIL import Image
from rembg import remove, new_session

root = Path(__file__).resolve().parent.parent
img_dir = root / "img"
out_dir = root / "tools" / "variants"
out_dir.mkdir(exist_ok=True)
models = sys.argv[1:] or ["isnet-general-use", "u2net", "birefnet-general-lite"]
rows = []
for m in models:
    s = new_session(m)
    row = []
    for i in range(1, 5):
        src = Image.open(img_dir / f"raw{i}.jpg").convert("RGB")
        src = src.resize((src.width * 2, src.height * 2), Image.LANCZOS)
        out = remove(src, session=s)
        out.save(out_dir / f"{m}_{i}.png")
        row.append(out)
    rows.append(row)
    print("done", m, flush=True)
cell = 240
prev = Image.new("RGBA", (cell * 4, cell * len(rows)), (200, 40, 160, 255))
for r, row in enumerate(rows):
    for c, im in enumerate(row):
        im = im.copy(); im.thumbnail((cell, cell))
        prev.alpha_composite(im, (c * cell, r * cell))
prev.save(root / "tools" / "compare.png")

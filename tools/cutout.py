# /// script
# dependencies = ["rembg[cpu]", "pillow"]
# ///
"""raw*.jpg の背景を除去して evo*.png を出力する"""
from pathlib import Path
from PIL import Image
from rembg import remove, new_session

img_dir = Path(__file__).resolve().parent.parent / "img"
session = new_session("isnet-general-use")
for i in range(1, 5):
    src = Image.open(img_dir / f"raw{i}.jpg").convert("RGB")
    src = src.resize((src.width * 2, src.height * 2), Image.LANCZOS)
    out = remove(src, session=session, alpha_matting=True)
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)
    out.save(img_dir / f"evo{i}.png")
    print("saved", i, out.size)

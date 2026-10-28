# /// script
# dependencies = ["pillow"]
# ///
from pathlib import Path
from PIL import Image
d = Path(__file__).resolve().parent.parent / "img"
ims = [Image.open(d / f"evo{i}.png") for i in range(1, 5)]
H = 300
out = Image.new("RGBA", (4 * 260, H + 20), (200, 40, 160, 255))
for i, im in enumerate(ims):
    im = im.copy(); im.thumbnail((250, H))
    out.alpha_composite(im, (i * 260 + 5, 10))
out.save(Path(__file__).resolve().parent / "preview.png")

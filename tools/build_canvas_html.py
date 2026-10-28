# /// script
# dependencies = ["pillow"]
# ///
"""HTML + CSS + JS + 4体の透過画像を1ファイルにまとめた Gemini Canvas 用単一HTMLを生成"""
import base64
import io
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent.parent
img_dir = root / "img"

# 透過WebP（高さ360px・透過維持）に圧縮してBase64化
data_urls = []
for i in range(1, 5):
    im = Image.open(img_dir / f"evo{i}.png").convert("RGBA")
    im.thumbnail((250, 250), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=70, method=6)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_urls.append(f"data:image/webp;base64,{b64}")
    print(f"evo{i} webp base64 len: {len(b64):,} chars")

html = (root / "index.html").read_text(encoding="utf-8")
css = (root / "style.css").read_text(encoding="utf-8")
js = (root / "game.js").read_text(encoding="utf-8")

# formSrc(i) と loadForm(i) が埋め込みData URLを使うように差し替え
embedded_js_array = "const EMBEDDED_IMGS = [\n  " + ",\n  ".join(f"'{u}'" for u in data_urls) + "\n];\n"
js = js.replace(
    "function formSrc(i) { return `img/evo${i + 1}.png`; }",
    embedded_js_array + "function formSrc(i) { return EMBEDDED_IMGS[i]; }"
)
js = js.replace("im.src = `img/evo${i + 1}.png`;", "im.src = EMBEDDED_IMGS[i];")
# 単独HTMLでは外部 bg.png を探しに行って404を出さないよう loadBg を無効化
js = js.replace("loadBg();", "// loadBg();")

single_html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
single_html = single_html.replace('<script src="game.js"></script>', f"<script>\n{js}\n</script>")

out_path = root / "kesmon_survivor_canvas.html"
out_path.write_text(single_html, encoding="utf-8")
print(f"Total single HTML size: {out_path.stat().st_size:,} bytes ({len(single_html):,} chars)")

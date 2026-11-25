#!/usr/bin/env python
"""Render the figure slides of the README and the project page.

    python tools/figures/render.py                 # all slides
    python tools/figures/render.py overview map    # some slides

Each slide is an HTML page tools/figures/slides/<name>.html whose <body> gives its CSS size
(data-size="1200x620", or "1200xauto" for the height of its content), its file format (data-format="png" or "jpg", with data-quality for JPEG) and its
themes (data-themes, default "light dark"). Headless Chromium renders it at twice the CSS size, once per theme (the
page reads ?theme=light|dark). Output: docs/assets/figures/<name>.<ext> for the light theme and
<name>-dark.<ext> for the dark theme. The pages are served over a local HTTP server rooted at the
repository, so they can load the fonts in docs/assets/fonts.

Before saving, the renderer reports text that leaves the slide or its image card and labels that
overlap each other. Needs playwright with Chromium (pip install playwright; playwright install chromium)
and Pillow.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import io
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLIDES = ROOT / "tools" / "figures" / "slides"
OUT = ROOT / "docs" / "assets" / "figures"
SCALE = 2
JPEG_QUALITY = 90

CHECK_JS = r"""
() => {
  const W = innerWidth, H = innerHeight, issues = [];
  const boxes = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (!n.textContent.trim()) continue;
    const el = n.parentElement;
    if (el.closest('[data-allow-overflow]')) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || +cs.opacity === 0) continue;
    const range = document.createRange(); range.selectNodeContents(n);
    const r = range.getBoundingClientRect();
    if (!r.width) continue;
    const text = n.textContent.trim().slice(0, 48);
    if (r.left < 1 || r.top < 1 || r.right > W - 1 || r.bottom > H - 1) issues.push(`outside the slide: "${text}"`);
    const card = el.closest('.card, [data-clip]');
    if (card) {
      const c = card.getBoundingClientRect();
      if (r.left < c.left - 0.5 || r.right > c.right + 0.5 || r.top < c.top - 0.5 || r.bottom > c.bottom + 0.5)
        issues.push(`outside its card: "${text}"`);
    }
    if (el.scrollWidth > el.clientWidth + 1 && ['hidden', 'clip'].includes(cs.overflowX)) issues.push(`clipped: "${text}"`);
    for (const q of range.getClientRects()) boxes.push([q.left, q.top, q.right, q.bottom, text, el]);
  }
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i], b = boxes[j];
    if (a[5] === b[5] || a[5].contains(b[5]) || b[5].contains(a[5])) continue;
    const ox = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), oy = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (ox > 1 && oy > 1) issues.push(`overlap: "${a[4]}" and "${b[4]}"`);
  }
  return issues;
}
"""


def slide_meta(path: Path) -> dict:
    head = path.read_text()
    m = re.search(r"<body([^>]*)>", head)
    attrs = dict(re.findall(r'data-([a-z]+)="([^"]*)"', m.group(1) if m else ""))
    w, h = attrs.get("size", "1200x600").split("x")
    w, h = int(w), (0 if h == "auto" else int(h))         # auto: the height of the slide's content
    return {"w": w, "h": h, "format": attrs.get("format", "png"), "themes": attrs.get("themes", "light dark").split(),
            "quality": int(attrs.get("quality", JPEG_QUALITY))}


def serve(root: Path):
    handler = functools.partial(QuietHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def save(png: bytes, out: Path, fmt: str, quality: int = JPEG_QUALITY):
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpg":
        im.save(out, quality=quality, subsampling=0, optimize=True, progressive=True)
    else:
        im.save(out, optimize=True)
    return out.stat().st_size


async def render(names: list[str], check_only: bool = False) -> int:
    from playwright.async_api import async_playwright
    httpd = serve(ROOT)
    port = httpd.server_address[1]
    problems = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for name in names:
            meta = slide_meta(SLIDES / f"{name}.html")
            for theme in meta["themes"]:
                page = await browser.new_page(viewport={"width": meta["w"], "height": meta["h"] or 2400},
                                              device_scale_factor=SCALE)
                await page.goto(f"http://127.0.0.1:{port}/tools/figures/slides/{name}.html?theme={theme}")
                await page.evaluate("document.fonts.ready")
                await page.wait_for_function("[...document.images].every(i => i.complete && i.naturalWidth > 0)")
                await page.evaluate("window.slideReady || true")
                if not meta["h"]:
                    h = await page.evaluate("""() => { const s = document.querySelector('.slide'); s.style.height = 'auto';
                        const h = Math.ceil(s.getBoundingClientRect().height) + 2; s.style.height = ''; return h; }""")
                    await page.set_viewport_size({"width": meta["w"], "height": h})
                await page.wait_for_timeout(150)
                issues = await page.evaluate(CHECK_JS)
                for s in issues:
                    print(f"  [{name}, {theme}] {s}")
                problems += len(issues)
                if not check_only:
                    ext = meta["format"]
                    out = OUT / (f"{name}.{ext}" if theme == meta["themes"][0] else f"{name}-{theme}.{ext}")
                    size = save(await page.screenshot(type="png"), out, ext, meta["quality"])
                    vp = page.viewport_size
                    print(f"[render] {out.relative_to(ROOT)}  {vp['width'] * SCALE}x{vp['height'] * SCALE}  {size / 1024:.0f} KB")
                await page.close()
        await browser.close()
    httpd.shutdown()
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="slide names (default: all slides with a data-size)")
    ap.add_argument("--check", action="store_true", help="only report layout problems")
    a = ap.parse_args()
    names = a.names or sorted(p.stem for p in SLIDES.glob("*.html") if 'data-size="' in p.read_text())
    n = asyncio.run(render(names, a.check))
    if n:
        print(f"[render] {n} layout problems")
    sys.exit(1 if n and a.check else 0)


if __name__ == "__main__":
    main()

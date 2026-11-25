#!/usr/bin/env python
"""Build the documentation pages of the project page from the Markdown documents.

    python tools/pages.py

docs/method.md, data.md, evaluation.md and performance.md stay the reference text; this tool converts each to
docs/<name>.html in the design of docs/index.html (header, contents sidebar, light and dark themes). It handles the
Markdown that the documents use: headings, paragraphs, pipe tables with alignment, fenced code, lists, inline code,
links (links to the other documents point to their pages) and emphasis. Run it after editing a document;
tools/readme_tables.py runs it after regenerating the tables.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/mkturkcan/streetnet"
PAGES = [("method", "Method"), ("data", "Data formats"), ("evaluation", "Evaluation"), ("performance", "Performance")]


def slug(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text).lower()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    return re.sub(r"\s+", "-", s.strip())


def inline(text: str) -> str:
    """Inline Markdown to HTML: code spans first, then links, bold and italics on the remaining text."""
    out, last = [], 0
    for m in re.finditer(r"`([^`]+)`", text):
        out.append(_inline_text(text[last:m.start()]))
        out.append(f"<code>{html.escape(m.group(1))}</code>")
        last = m.end()
    out.append(_inline_text(text[last:]))
    return "".join(out)


def _inline_text(t: str) -> str:
    t = html.escape(t, quote=False)

    def link(m):
        label, url = m.group(1), m.group(2)
        md = re.match(r"^([a-z_]+)\.md(#.*)?$", url)
        if md and any(md.group(1) == p for p, _ in PAGES):
            url = f"{md.group(1)}.html{md.group(2) or ''}"
        elif not re.match(r"^(https?:|#|mailto:)", url):
            url = f"{REPO}/blob/main/docs/{url}"
        return f'<a href="{html.escape(url)}">{label}</a>'
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, align, body = cells[0], cells[1], cells[2:]
    right = [a.endswith(":") and not a.startswith(":") for a in align]
    th = "".join(f'<th{" class=n" if right[i] else ""}>{inline(c)}</th>' for i, c in enumerate(head))
    trs = []
    for r in body:
        cls = ' class="total"' if r and r[0] in ("All classes",) else ""
        trs.append(f"<tr{cls}>" + "".join(f'<td{" class=n" if i < len(right) and right[i] else ""}>{inline(c)}</td>'
                                         for i, c in enumerate(r)) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def markdown(md: str) -> tuple[str, str, list[tuple[int, str, str]]]:
    """Returns (title, body html, headings [(level, id, text)])."""
    lines = md.splitlines()
    out, heads, title = [], [], ""
    i, para = 0, []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(s.strip() for s in para))}</p>")
            para.clear()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1:j]))
            out.append(f"<pre><code>{code}</code></pre>")
            i = j + 1
            continue
        if ln.strip().startswith("<!--"):
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = text
                out.append(f"<h1>{inline(text)}</h1>")
            else:
                hid = slug(text)
                heads.append((level, hid, text))
                out.append(f'<h{level} id="{hid}">{inline(text)}<a class="anchor" href="#{hid}" aria-label="Link to this section">#</a></h{level}>')
            i += 1
            continue
        if ln.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
            continue
        if re.match(r"^\s*[-*] ", ln):
            flush()
            items, j = [], i
            while j < len(lines) and (re.match(r"^\s*[-*] ", lines[j]) or (lines[j].startswith("  ") and items)):
                if re.match(r"^\s*[-*] ", lines[j]):
                    items.append(re.sub(r"^\s*[-*] ", "", lines[j]))
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            out.append("<ul>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ul>")
            i = j
            continue
        if not ln.strip():
            flush()
        else:
            para.append(ln)
        i += 1
    flush()
    return title, "\n".join(out), heads


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | StreetNet</title>
<meta name="description" content="StreetNet documentation: {title}.">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0d1117" media="(prefers-color-scheme: dark)">
<link rel="icon" href="assets/img/favicon.svg" type="image/svg+xml">
<link rel="preload" href="assets/fonts/Inter-Variable.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="assets/css/site.css">
<script src="assets/js/site.js"></script>
</head>
<body>
<!-- Generated by tools/pages.py from docs/{name}.md. Edit the Markdown document and run the tool. -->
<header class="nav">
  <div class="wrap">
    <a class="brand" href="index.html" aria-label="StreetNet, project page">
      <span class="mark" aria-hidden="true"><i style="background:var(--g0)"></i><i style="background:var(--g1)"></i><i style="background:var(--g2)"></i><i style="background:var(--g5)"></i></span>StreetNet</a>
    <nav class="links" aria-label="Documentation">{nav}</nav>
    <div class="end">
      {theme_button}
      <a class="icon-btn" href="{repo}/blob/main/docs/{name}.md" aria-label="This document on GitHub">Markdown</a>
      <button class="icon-btn menu-btn" id="menu" type="button" aria-expanded="false" aria-controls="sheet">Menu</button>
    </div>
  </div>
  <div class="sheet" id="sheet">{sheet}</div>
</header>
<main class="wrap doc">
  <aside class="toc" aria-label="Contents">
    <p>Contents</p>
{toc}
    <div class="pages"><p>Documentation</p>{pages}</div>
  </aside>
  <article class="md">
{body}
    <p class="doc-foot">Source: <a href="{repo}/blob/main/docs/{name}.md">docs/{name}.md</a>. <a href="index.html">Project page</a>.</p>
  </article>
</main>
</body>
</html>
"""

THEME_BUTTON = ('<button class="icon-btn" id="theme" type="button" aria-label="Switch the theme">'
                '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="2" width="12" height="12" rx="3.2" fill="none" '
                'stroke="currentColor" stroke-width="1.5"/><path d="M5.2 2.75h5.6a2.45 2.45 0 0 1 2.45 2.45v5.6z" '
                'fill="currentColor"/></svg></button>')


def build() -> list[Path]:
    written = []
    for name, label in PAGES:
        src = DOCS / f"{name}.md"
        title, body, heads = markdown(src.read_text())
        nav = "".join(f'<a href="{p}.html"{" class=on" if p == name else ""}>{t}</a>' for p, t in PAGES)
        toc = "\n".join(f'    <a class="l{lv}" href="#{hid}">{html.escape(re.sub("`", "", t))}</a>' for lv, hid, t in heads if lv <= 3)
        pages = "".join(f'<a href="{p}.html"{" class=on" if p == name else ""}>{t}</a>' for p, t in PAGES)
        sheet = '<a href="index.html">Project page</a>' + "".join(f'<a href="{p}.html">{t}</a>' for p, t in PAGES)
        page = TEMPLATE.format(title=title or label, name=name, nav=nav, toc=toc, pages=pages, sheet=sheet, body=body,
                               repo=REPO, theme_button=THEME_BUTTON)
        out = DOCS / f"{name}.html"
        out.write_text(page)
        written.append(out)
    return written


if __name__ == "__main__":
    for p in build():
        print(f"[pages] {p.relative_to(ROOT)}")

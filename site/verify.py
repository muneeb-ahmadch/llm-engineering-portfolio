#!/usr/bin/env python3
"""Grader for the portfolio site. Exit 0 = pass, 1 = fail.
Prints one 'FAIL: ...' line per problem. Locked after the baseline commit.

Modes:
  python3 site/verify.py                 full check (needs screenshots + review)
  python3 site/verify.py --quick         everything except screenshots and review
  python3 site/verify.py --no-review     everything except the review file
  python3 site/verify.py --snapshot      Step 0 only: snapshot the ORIGINAL site
  python3 site/verify.py --lock          Step 0 only: write LOCK.sha256 (once)
  python3 site/verify.py --contrast      print contrast table and exit
"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS, SITE = ROOT / "docs", ROOT / "site"
SNAP = SITE / "snapshots"
ARGS = set(sys.argv[1:])
QUICK = "--quick" in ARGS
NO_REVIEW = QUICK or "--no-review" in ARGS
FAILS = []

LOCKED = ["site/verify.py", "site/shoot.py", "site/review.py", "site/review/RUBRIC.md",
          ".claude/settings.json", ".claude/hooks/guard-write.py", ".claude/hooks/guard-bash.py",
          ".claude/hooks/after-write.sh", ".claude/hooks/on-stop.sh"]
SNAP_GLOB = ["site/snapshots/claims.json", "site/snapshots/reference/*.txt"]
SHOTS = ["index-desktop-light", "index-desktop-dark", "index-mobile-light", "index-mobile-dark",
         "mcp-desktop-light", "mcp-desktop-dark", "chunking-mobile-light", "chunking-mobile-dark"]
REPO_URL = "https://github.com/muneeb-ahmadch/llm-engineering-portfolio/"
NUM_RE = re.compile(r"\d+/\d+|\$?\d[\d,.]*(?:%|x|\u00d7)?")


def fail(msg):
    FAILS.append(msg)
    print("FAIL: " + msg)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def visible_text(html):
    p = _Text()
    p.feed(html)
    return re.sub(r"\s+", " ", "".join(p.parts)).strip()


def numeric_tokens(text):
    return Counter(t.rstrip(".,") for t in NUM_RE.findall(text))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def read(p):
    return Path(p).read_text(encoding="utf-8")


def original_fragment_body(html):
    """Body of an ORIGINAL reference fragment: after the back-link eyebrow line,
    before the footer div. Used for snapshots and for nothing else."""
    m = re.search(r'<p class="eyebrow"[^>]*>.*?\.\./index\.html.*?</p>\s*', html, re.S)
    start = m.end() if m else 0
    end = html.rfind('<div class="footer">')
    return html[start:end if end > 0 else len(html)]


def mdash_count(s):
    return s.count("\u2014") + s.count("&mdash;") + s.count("&#8212;")


# ---------- contrast ----------
def _lum(hexcol):
    h = hexcol.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    la, lb = sorted([_lum(a), _lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def palettes(css):
    """Return {'light': {token: hex}, 'dark': {token: hex}} from :root blocks."""
    out = {"light": {}, "dark": {}}
    for sel, body in re.findall(r"(:root[^{]*)\{([^}]*)\}", css):
        scheme = "dark" if "dark" in sel else "light"
        for k, v in re.findall(r"--([a-z-]+)\s*:\s*(#[0-9a-fA-F]{3,6})", body):
            out[scheme].setdefault(k, v)
    return out


def check_contrast(css):
    pal = palettes(css)
    pairs = [("ink", "bg", 4.5), ("ink-soft", "bg", 4.5), ("ink-faint", "bg", 4.5),
             ("accent", "bg", 4.5), ("good", "bg", 4.5), ("bad", "bg", 4.5),
             ("ink", "bg-panel", 4.5), ("ink-soft", "bg-panel", 4.5),
             ("ink-faint", "bg-panel", 4.5), ("accent", "bg-panel", 4.5),
             ("rule", "bg", 1.3)]
    rows = []
    for scheme in ("light", "dark"):
        p = pal[scheme]
        for fg, bg, need in pairs:
            if fg not in p or bg not in p:
                fail(f"contrast: token --{fg} or --{bg} missing in {scheme} palette")
                continue
            c = contrast(p[fg], p[bg])
            rows.append((scheme, fg, bg, c, need))
            if c < need:
                fail(f"contrast {scheme}: --{fg} on --{bg} is {c:.2f}:1, need {need}:1")
    return rows


# ---------- snapshot / lock (Step 0 only) ----------
def do_snapshot():
    if (SNAP / "claims.json").exists():
        sys.exit("refusing: snapshots already exist")
    (SNAP / "reference").mkdir(parents=True)
    html = read(DOCS / "index.html")
    vis = visible_text(html)
    required = []
    for m in re.finditer(r"<p(?:\s+class=\"(f|standfirst|note)\")?>(.*?)</p>", html, re.S):
        t = visible_text(m.group(2))
        if len(t) > 40:
            required.append(t)
    for m in re.finditer(r'<span class="l">(.*?)</span>', html):
        required.append(visible_text(m.group(1)))
    pre = re.search(r"<pre>(.*?)</pre>", html, re.S).group(1)
    configs = []
    for line in visible_text_lines(pre):
        parts = re.split(r"\s{2,}", line.strip().lstrip("\u2192 ").strip(), maxsplit=1)
        if len(parts) == 2:
            configs.append(re.sub(r"\s+", " ", parts[1]).strip())
    hrefs = sorted(set(h for h in re.findall(r'href="([^"]+)"', html)
                       if h.startswith(REPO_URL) or h.startswith("reference/")))
    refs = {}
    for f in sorted((DOCS / "reference").glob("*.html")):
        src = read(f)
        body = original_fragment_body(src)
        (SNAP / "reference" / (f.stem + ".txt")).write_text(visible_text(body), encoding="utf-8")
        refs[f.stem] = {"mdash": mdash_count(body), "title": visible_text(
            re.search(r"<title>(.*?)</title>", src, re.S).group(1))}
    (SNAP / "claims.json").write_text(json.dumps({
        "numeric_tokens": dict(numeric_tokens(vis)),
        "required_texts": required, "pipeline_configs": configs,
        "required_hrefs": hrefs, "reference": refs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("snapshot written:", len(required), "texts,", len(configs), "configs,",
          len(hrefs), "hrefs,", len(refs), "reference pages")


def visible_text_lines(pre_html):
    p = _Text()
    p.feed(pre_html)
    return "".join(p.parts).splitlines()


def do_lock():
    lock = SNAP / "LOCK.sha256"
    if lock.exists():
        sys.exit("refusing: LOCK.sha256 already exists")
    files = [ROOT / p for p in LOCKED] + [q for g in SNAP_GLOB for q in sorted(ROOT.glob(g))]
    lines = [f"{sha256(f)}  {rel(f)}" for f in files if f.exists()]
    missing = [rel(f) for f in files if not f.exists()]
    if missing:
        sys.exit("refusing: locked files missing: " + ", ".join(missing))
    lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("locked", len(lines), "files")


# ---------- checks ----------
def check_lock():
    lock = SNAP / "LOCK.sha256"
    if not lock.exists():
        return fail("site/snapshots/LOCK.sha256 missing (Step 0 not done)")
    for line in read(lock).splitlines():
        digest, path = line.split("  ", 1)
        p = ROOT / path
        if not p.exists():
            fail(f"locked file deleted: {path}")
        elif sha256(p) != digest:
            fail(f"locked file modified: {path}")


def check_projects_untouched():
    for cmd in (["git", "status", "--porcelain", "--", "projects/"],
                ["git", "ls-files", "--others", "--exclude-standard", "--", "projects/"]):
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if out:
            fail("projects/ changed or has new files:\n" + out)
    log = subprocess.run(["git", "log", "--format=%B"], cwd=ROOT, capture_output=True, text=True).stdout
    for bad in ("Co-Authored-By", "Generated with", "generated by"):
        if bad.lower() in log.lower():
            fail(f"git log contains '{bad}'")


def check_drift():
    tmp = Path(tempfile.mkdtemp(prefix="siteverify-"))
    try:
        r = subprocess.run([sys.executable, str(SITE / "build.py"), "--out", str(tmp)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return fail("build.py --out failed:\n" + r.stderr[-2000:])
        want = {str(p.relative_to(tmp)): p for p in tmp.rglob("*") if p.is_file()}
        have = {str(p.relative_to(DOCS)): p for p in DOCS.rglob("*") if p.is_file()}
        have.pop(".nojekyll", None)
        want.pop(".nojekyll", None)
        for k in sorted(set(want) - set(have)):
            fail(f"docs/ missing generated file {k}")
        for k in sorted(set(have) - set(want)):
            fail(f"docs/ has a file build.py did not generate: {k}")
        for k in sorted(set(have) & set(want)):
            if have[k].read_bytes() != want[k].read_bytes():
                fail(f"docs/{k} differs from a fresh build (hand edit or stale build)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_deleted():
    for p in ("docs/site.css", "docs/assets/style.css", "docs/assets/quiz.js", "docs/assets/quiz.css",
              "docs/assets/cosine.js", "docs/assets/cosine.css"):
        if (ROOT / p).exists():
            fail(f"{p} should have been removed")


def html_pages():
    return sorted(DOCS.rglob("*.html"))


def check_page_shell(path, html, titles):
    name = rel(path)
    low = html[:400].lower()
    if not low.startswith("<!doctype html>"):
        fail(f"{name}: must start with <!doctype html>")
    if '<html lang="en"' not in html:
        fail(f"{name}: missing <html lang=\"en\">")
    for needle, label in [("<main", "<main>"), ('<meta name="viewport"', "viewport meta"),
                          ('<meta name="description"', "description meta"),
                          ('<link rel="canonical"', "canonical link"),
                          ('property="og:title"', "og:title"), ('property="og:description"', "og:description"),
                          ('property="og:image"', "og:image"), ('name="twitter:card"', "twitter:card"),
                          ('rel="icon"', "favicon"), ('class="skip"', "skip link"),
                          ("<nav", "nav landmark"), ('class="theme-toggle"', "theme toggle"),
                          ("aria-label=", "aria-label"), ("<footer", "footer"),
                          ('name="color-scheme"', "color-scheme meta"), ("localStorage", "theme init script"),
                          ("location.search", "?theme= query handling")]:
        if needle not in html:
            fail(f"{name}: missing {label}")
    if html.count("<h1") != 1:
        fail(f"{name}: expected exactly one <h1>, found {html.count('<h1')}")
    if html.count("<main") != 1:
        fail(f"{name}: expected exactly one <main>")
    t = re.search(r"<title>(.*?)</title>", html, re.S)
    title = visible_text(t.group(1)) if t else ""
    if not title:
        fail(f"{name}: empty <title>")
    elif title in titles:
        fail(f"{name}: duplicate <title> '{title}'")
    titles.add(title)
    if "\u2014" in html:
        fail(f"{name}: contains a literal em dash")
    for bad in ("Co-Authored-By", "Generated with", "generated by"):
        if bad.lower() in html.lower():
            fail(f"{name}: contains '{bad}'")
    for pat, what in [(r'<script[^>]+src="https?://', "external script"),
                      (r'<link[^>]+rel="(?:stylesheet|preload|preconnect|dns-prefetch|modulepreload)"[^>]*href="https?://',
                       "external stylesheet/resource"),
                      (r'<link[^>]+href="https?://[^"]*"[^>]*rel="(?:stylesheet|preload|preconnect|dns-prefetch|modulepreload)"',
                       "external stylesheet/resource"),
                      (r'<img[^>]+src="https?://', "external image"),
                      (r"<iframe", "iframe"), (r"@import", "@import")]:
        if re.search(pat, html):
            fail(f"{name}: {what} not allowed")
    r = subprocess.run(["tidy", "-q", "-e", "--drop-empty-elements", "no", str(path)],
                       capture_output=True, text=True)
    if r.returncode == 2:
        errs = [l for l in r.stderr.splitlines() if "Error:" in l][:5]
        fail(f"{name}: tidy errors: " + " | ".join(errs))


def check_links(path, html):
    name = rel(path)
    ids = set(re.findall(r'\sid="([^"]+)"', html))
    for attr, val in re.findall(r'\s(href|src)="([^"]+)"', html):
        if val.startswith(("mailto:", "data:", "#", "http://", "https://")):
            if val.startswith("#") and val[1:] and val[1:] not in ids:
                fail(f"{name}: fragment {val} has no matching id")
            elif val.startswith(REPO_URL):
                m = re.match(re.escape(REPO_URL) + r"(?:blob|tree)/main/([^#?]*)", val)
                if m and not (ROOT / m.group(1)).exists():
                    fail(f"{name}: GitHub link to missing repo path {m.group(1)}")
            continue
        target, _, frag = val.partition("#")
        tpath = (path.parent / target).resolve() if target else path
        if not tpath.exists():
            fail(f"{name}: broken {attr} {val}")
        elif frag and tpath.suffix == ".html":
            if f'id="{frag}"' not in read(tpath):
                fail(f"{name}: {val} points to a missing id")


def check_sizes():
    idx = DOCS / "index.html"
    css = DOCS / "assets" / "site.css"
    js = DOCS / "assets" / "site.js"
    total = sum(p.stat().st_size for p in (idx, css, js) if p.exists())
    if total > 90 * 1024:
        fail(f"landing page weight {total} bytes exceeds 90 KB (html+css+js)")
    for p in (DOCS / "reference").glob("*.html"):
        if p.stat().st_size > 120 * 1024:
            fail(f"{rel(p)} exceeds 120 KB")


def check_css_js():
    css_path = DOCS / "assets" / "site.css"
    if not css_path.exists():
        return fail("docs/assets/site.css missing")
    css = read(css_path)
    if css.count("!important") > 4:
        fail(f"site.css uses !important {css.count('!important')} times (max 4)")
    for needle in ("prefers-reduced-motion", ":focus-visible", "prefers-color-scheme",
                   '[data-theme="dark"]', '[data-theme="light"]', "@media print", "tabular-nums"):
        if needle not in css:
            fail(f"site.css missing {needle}")
    if re.search(r"url\(\s*['\"]?https?://", css):
        fail("site.css references an external URL")
    if "@font-face" in css or "fonts.googleapis" in css:
        fail("site.css must not load web fonts")
    check_contrast(css)
    js = DOCS / "assets" / "site.js"
    if not js.exists():
        fail("docs/assets/site.js missing")
    elif js.stat().st_size > 6 * 1024:
        fail("site.js larger than 6 KB")
    for t in (SITE / "templates").glob("*.html"):
        if 'style="' in read(t):
            fail(f"{rel(t)}: inline style attribute in a template")
    for p in list((SITE / "templates").glob("*")) + list((SITE / "static").rglob("*")) + \
            list((SITE / "data").glob("*")) + [SITE / "build.py"]:
        if p.is_file() and p.suffix in (".html", ".css", ".js", ".json", ".py", ".svg") \
                and "\u2014" in p.read_text(encoding="utf-8", errors="ignore"):
            fail(f"{rel(p)}: literal em dash")


def check_claims():
    claims = json.loads(read(SNAP / "claims.json"))
    html = read(DOCS / "index.html")
    vis = visible_text(html)
    have = numeric_tokens(vis)
    for tok, n in claims["numeric_tokens"].items():
        if have[tok] < n:
            fail(f"index.html: number '{tok}' appeared {n}x on the old page, now {have[tok]}x")
    for t in claims["required_texts"]:
        if t not in vis:
            fail(f"index.html: original text missing verbatim: '{t[:70]}...'")
    for c in claims["pipeline_configs"]:
        if c not in vis:
            fail(f"index.html: pipeline config missing verbatim: '{c}'")
    hrefs = set(re.findall(r'href="([^"]+)"', html))
    for h in claims["required_hrefs"]:
        if h not in hrefs:
            fail(f"index.html: evidence link dropped: {h}")
    if mdash_count(html):
        fail("index.html: em dash (entity or literal) present")
    for nn in ("01", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"):
        if f'id="exp-{nn}"' not in html:
            fail(f"index.html: missing id exp-{nn}")
    for sec in ("experiments", "pipeline", "reference", "about", "reading-order"):
        if f'id="{sec}"' not in html:
            fail(f"index.html: missing section id {sec}")
    if html.count('class="exp"') + html.count('class="exp ') != 13:
        fail("index.html: expected 13 elements with class exp")
    refs = claims["reference"]
    out_dir = DOCS / "reference"
    got = {p.stem for p in out_dir.glob("*.html")}
    if got != set(refs):
        fail(f"docs/reference set mismatch: extra {sorted(got - set(refs))}, missing {sorted(set(refs) - got)}")
    for slug, meta in refs.items():
        p = out_dir / f"{slug}.html"
        if not p.exists():
            continue
        page = read(p)
        orig = read(SNAP / "reference" / f"{slug}.txt")
        if orig not in visible_text(page):
            fail(f"reference/{slug}.html: original body text is not preserved verbatim")
        if mdash_count(page) != meta["mdash"]:
            fail(f"reference/{slug}.html: em dash count {mdash_count(page)} != original {meta['mdash']}")
        for needle in ('class="toc"', 'class="doc-nav"', "On this page"):
            if needle not in page:
                fail(f"reference/{slug}.html: missing {needle}")
        n_h2 = len(re.findall(r"<h2[^>]*>", page))
        n_h2_id = len(re.findall(r'<h2[^>]*\sid="', page))
        if n_h2 and n_h2 != n_h2_id:
            fail(f"reference/{slug}.html: {n_h2 - n_h2_id} h2 elements without id")


def check_outputs_exist():
    for p in ("docs/404.html", "docs/sitemap.xml", "docs/robots.txt", "docs/assets/og.png",
              "docs/assets/favicon.svg", "docs/.nojekyll"):
        if not (ROOT / p).exists():
            fail(f"{p} missing")
    og = ROOT / "docs/assets/og.png"
    if og.exists() and og.stat().st_size < 15 * 1024:
        fail("og.png is suspiciously small (blank render?)")
    sm = ROOT / "docs/sitemap.xml"
    if sm.exists() and read(sm).count("<loc>") != 14:
        fail("sitemap.xml should list 14 URLs")


def newest_input_mtime():
    paths = [p for d in (DOCS, SITE / "templates", SITE / "static", SITE / "data", SITE / "content")
             for p in d.rglob("*") if p.is_file()]
    paths.append(SITE / "build.py")
    return max(p.stat().st_mtime for p in paths if p.exists())


def check_screenshots():
    d = SITE / "screenshots"
    latest = newest_input_mtime()
    hashes = {}
    for s in SHOTS:
        p = d / f"{s}.png"
        if not p.exists():
            fail(f"screenshot missing: {rel(p)}")
            continue
        if p.stat().st_size < 20 * 1024:
            fail(f"screenshot too small (blank?): {rel(p)}")
        if p.stat().st_mtime < latest:
            fail(f"screenshot stale (older than site inputs): {rel(p)}")
        hashes[s] = sha256(p)
    for base in ("index-desktop", "index-mobile", "mcp-desktop", "chunking-mobile"):
        a, b = hashes.get(base + "-light"), hashes.get(base + "-dark")
        if a and b and a == b:
            fail(f"{base}: light and dark screenshots are identical, theme did not render")
    return hashes


def check_review(hashes):
    p = SITE / "review" / "last.json"
    if not p.exists():
        return fail("site/review/last.json missing: run python3 site/review.py")
    try:
        r = json.loads(read(p))
    except Exception as e:
        return fail(f"review/last.json unreadable: {e}")
    if r.get("verdict") != "PASS":
        fail("independent review verdict is not PASS: " +
             "; ".join(f.get("issue", str(f))[:120] for f in r.get("findings", [])[:6]))
    if r.get("screenshot_hashes") != hashes:
        fail("review/last.json was produced for different screenshots; re-run review.py")


def main():
    if "--snapshot" in ARGS:
        return do_snapshot()
    if "--lock" in ARGS:
        return do_lock()
    if "--contrast" in ARGS:
        for row in check_contrast(read(DOCS / "assets" / "site.css")):
            print("%-5s %-9s on %-8s %5.2f (need %.1f)" % row)
        return
    check_lock()
    check_projects_untouched()
    if not (SITE / "build.py").exists():
        fail("site/build.py missing")
    else:
        check_drift()
    check_deleted()
    check_outputs_exist()
    titles = set()
    for p in html_pages():
        h = read(p)
        check_page_shell(p, h, titles)
        check_links(p, h)
    check_sizes()
    check_css_js()
    if (SNAP / "claims.json").exists():
        check_claims()
    else:
        fail("snapshots/claims.json missing")
    hashes = {}
    if not QUICK:
        hashes = check_screenshots()
    if not NO_REVIEW:
        check_review(hashes)
    print(f"{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s)"
          f"{' (quick mode)' if QUICK else ''}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

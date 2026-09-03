#!/usr/bin/env python3
"""Screenshots via headless Chrome. `python3 site/shoot.py` writes the 8 PNGs the grader
expects into site/screenshots/. `python3 site/shoot.py --og` renders site/static/og.html
to site/static/og.png (1200x630). Locked after the baseline commit.

Mobile shots are 500 px wide, not 390: Chrome on macOS clamps its layout viewport to a
500 px minimum in both headless modes, so asking for 390 renders at 500 and crops the
screenshot to 390, which looks like a broken page. 500 px exercises the same single-column
layout branch as 390 (the stylesheet's first breakpoint is 600 px)."""
import os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "site" / "screenshots"
# Chrome on macOS clamps a window to 500 px wide, so a 390 px viewport cannot be asked
# for directly: the mobile shots render the page inside a 390 px iframe (a real layout
# viewport of that width) and the capture is then cropped to those 390 columns.
# Desktop heights cover each page in full, footer included, measured from the output.
SHOTS = [  # name, page, width, height, theme, mode
    ("index-desktop-light", "docs/index.html", 1440, 6900, "light", "direct"),
    ("index-desktop-dark", "docs/index.html", 1440, 6900, "dark", "direct"),
    ("index-mobile-light", "docs/index.html", 390, 2600, "light", "narrow"),
    ("index-mobile-dark", "docs/index.html", 390, 2600, "dark", "narrow"),
    ("mcp-desktop-light", "docs/reference/mcp.html", 1440, 7350, "light", "direct"),
    ("mcp-desktop-dark", "docs/reference/mcp.html", 1440, 7350, "dark", "direct"),
    ("chunking-mobile-light", "docs/reference/chunking.html", 390, 3000, "light", "narrow"),
    ("chunking-mobile-dark", "docs/reference/chunking.html", 390, 3000, "dark", "narrow"),
]
CHROME_FLOOR = 500  # narrowest window Chrome will open


def page_url(page, theme):
    return "file://" + quote(str((ROOT / page).resolve())) + (f"?theme={theme}" if theme else "")


def run_chrome(url, w, h, out, extra=()):
    """Chrome 152 writes the PNG within seconds and then never exits: poll for the file,
    wait until its size is stable, then kill the whole process group."""
    out = Path(out)
    if out.exists():
        out.unlink()
    profile = tempfile.mkdtemp(prefix="chrome-shoot-")
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-crashpad",
           f"--user-data-dir={profile}", "--timeout=25000", "--virtual-time-budget=7000",
           f"--window-size={w},{h}", f"--screenshot={out}", *extra, url]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    t0, last = time.time(), -1
    try:
        while time.time() - t0 < 120:
            if out.exists():
                size = out.stat().st_size
                if size > 0 and size == last:
                    break
                last = size
            elif proc.poll() is not None:
                break
            time.sleep(0.5)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        shutil.rmtree(profile, ignore_errors=True)
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"screenshot failed for {url} after {time.time() - t0:.0f}s")
    return time.time() - t0


def crop(path, w, h):
    """Crop to w x h with sips, which ships with macOS. sips crops from the CENTRE and
    ignores --cropOffset, so the wrapper centres the iframe to make the two coincide."""
    r = subprocess.run(["sips", "--cropToHeightWidth", str(h), str(w),
                        str(path), "--out", str(path)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        sys.exit(f"crop failed for {path}: {r.stderr[-400:]}")


def shoot(page, w, h, out, theme=None, mode="direct"):
    out = Path(out)
    if mode == "direct":
        took = run_chrome(page_url(page, theme), w, h, out)
    else:
        wrapper = Path(tempfile.mkdtemp(prefix="chrome-narrow-")) / "wrapper.html"
        wrapper.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            "html,body{margin:0;padding:0;background:#888}"
            "body{display:flex;justify-content:center}"
            f"iframe{{width:{w}px;height:{h}px;border:0;display:block}}"
            "</style></head><body>"
            f"<iframe src=\"{page_url(page, theme)}\"></iframe></body></html>",
            encoding="utf-8")
        took = run_chrome("file://" + quote(str(wrapper)), max(w, CHROME_FLOOR), h, out,
                          extra=("--allow-file-access-from-files",))
        crop(out, w, h)
        shutil.rmtree(wrapper.parent, ignore_errors=True)
    print(f"wrote {out} {out.stat().st_size} bytes in {took:.1f}s ({w}x{h})")


def main():
    if "--og" in sys.argv:
        shoot("site/static/og.html", 1200, 630, ROOT / "site/static/og.png")
        return
    OUT.mkdir(exist_ok=True)
    for name, page, w, h, theme, mode in SHOTS:
        shoot(page, w, h, OUT / f"{name}.png", theme, mode)


if __name__ == "__main__":
    main()

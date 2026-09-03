#!/usr/bin/env python3
"""Screenshots via headless Chrome. `python3 site/shoot.py` writes the 8 PNGs the grader
expects into site/screenshots/. `python3 site/shoot.py --og` renders site/static/og.html
to site/static/og.png (1200x630). Locked after the baseline commit."""
import os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "site" / "screenshots"
SHOTS = [  # name, relative page, width, height, theme
    ("index-desktop-light", "docs/index.html", 1440, 2600, "light"),
    ("index-desktop-dark", "docs/index.html", 1440, 2600, "dark"),
    ("index-mobile-light", "docs/index.html", 390, 3200, "light"),
    ("index-mobile-dark", "docs/index.html", 390, 3200, "dark"),
    ("mcp-desktop-light", "docs/reference/mcp.html", 1440, 2000, "light"),
    ("mcp-desktop-dark", "docs/reference/mcp.html", 1440, 2000, "dark"),
    ("chunking-mobile-light", "docs/reference/chunking.html", 390, 2400, "light"),
    ("chunking-mobile-dark", "docs/reference/chunking.html", 390, 2400, "dark"),
]


def shoot(page, w, h, out, theme=None):
    """Chrome 152 writes the PNG within seconds and then never exits: poll for the file,
    wait until its size is stable, then kill the whole process group."""
    out = Path(out)
    if out.exists():
        out.unlink()
    profile = tempfile.mkdtemp(prefix="chrome-shoot-")
    url = "file://" + quote(str((ROOT / page).resolve())) + (f"?theme={theme}" if theme else "")
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-crashpad",
           f"--user-data-dir={profile}", "--timeout=15000", "--virtual-time-budget=5000",
           f"--window-size={w},{h}", f"--screenshot={out}", url]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    t0, last = time.time(), -1
    try:
        while time.time() - t0 < 90:
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
        sys.exit(f"screenshot failed for {page} after {time.time() - t0:.0f}s")
    print(f"wrote {out} {out.stat().st_size} bytes in {time.time() - t0:.1f}s")


def main():
    if "--og" in sys.argv:
        shoot("site/static/og.html", 1200, 630, ROOT / "site/static/og.png")
        return
    OUT.mkdir(exist_ok=True)
    for name, page, w, h, theme in SHOTS:
        shoot(page, w, h, OUT / f"{name}.png", theme)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent visual review. Spawns a separate Claude process that has never seen the
executor's context, gives it the locked rubric and the screenshots, and records its verdict.
Exit 0 on PASS, 1 otherwise. Locked after the baseline commit."""
import glob, hashlib, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "site" / "screenshots"
RUBRIC = ROOT / "site" / "review" / "RUBRIC.md"
LAST = ROOT / "site" / "review" / "last.json"
CANDIDATES = sorted(glob.glob(os.path.expanduser(
    "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
CLAUDE = os.environ.get("CLAUDE_BIN") or (CANDIDATES[-1] if CANDIDATES else "claude")
NAMES = ["index-desktop-light", "index-desktop-dark", "index-mobile-light", "index-mobile-dark",
         "mcp-desktop-light", "mcp-desktop-dark", "chunking-mobile-light", "chunking-mobile-dark"]


def main():
    files = [SHOTS / f"{n}.png" for n in NAMES]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit("missing screenshots: " + ", ".join(missing))
    hashes = {n: hashlib.sha256(f.read_bytes()).hexdigest() for n, f in zip(NAMES, files)}
    listing = "\n".join(f"- {n}: {f}" for n, f in zip(NAMES, files))
    prompt = (RUBRIC.read_text(encoding="utf-8") + "\n\n## Screenshots to review\n" + listing +
              "\n\nOpen every file with the Read tool before judging. Then output ONLY a JSON object "
              "on the last line, no prose after it, of the form "
              '{"verdict":"PASS"|"FAIL","findings":[{"screenshot":"name","severity":"major"|"minor","issue":"..."}]}')
    help_text = subprocess.run([CLAUDE, "--help"], capture_output=True, text=True).stdout
    tools_flag = "--allowedTools" if "--allowedTools" in help_text else "--allowed-tools"
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    r = subprocess.run([CLAUDE, "-p", prompt, "--model", "opus", "--max-turns", "14",
                        tools_flag, "Read", "--output-format", "json"],
                       cwd=SHOTS, env=env, capture_output=True, text=True, timeout=900)
    raw = r.stdout
    text = raw
    try:
        text = json.loads(raw).get("result", raw)
    except Exception:
        pass
    m = re.findall(r"\{.*\}", text, re.S)
    verdict = {"verdict": "FAIL", "findings": [{"severity": "major", "issue": "reviewer output unparseable"}]}
    for cand in reversed(m):
        try:
            v = json.loads(cand)
            if v.get("verdict") in ("PASS", "FAIL"):
                verdict = v
                break
        except Exception:
            continue
    majors = [f for f in verdict.get("findings", []) if f.get("severity") == "major"]
    if majors:
        verdict["verdict"] = "FAIL"
    LAST.parent.mkdir(exist_ok=True)
    LAST.write_text(json.dumps({**verdict, "screenshot_hashes": hashes, "raw": text[-6000:]},
                               indent=2), encoding="utf-8")
    print("REVIEW:", verdict["verdict"])
    for f in verdict.get("findings", []):
        print(f"  [{f.get('severity')}] {f.get('screenshot', '?')}: {f.get('issue')}")
    sys.exit(0 if verdict["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()

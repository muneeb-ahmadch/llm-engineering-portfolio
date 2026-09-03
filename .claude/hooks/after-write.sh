#!/bin/bash
# PostToolUse: after any edit under site/, rebuild and run the quick grader so the model
# sees failures immediately. Cannot block (the edit already happened); exit 2 surfaces stderr.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$(cat)"
FILE="$(printf '%s' "$INPUT" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("tool_input") or {}).get("file_path",""))')"
case "$FILE" in
  *"/site/"*|site/*) ;;
  *) exit 0 ;;
esac
cd "$ROOT" || exit 0
[ -f site/build.py ] || exit 0
OUT="$(python3 site/build.py 2>&1 && python3 site/verify.py --quick 2>&1)"
STATUS=$?
if [ $STATUS -ne 0 ]; then
  printf 'Quick grader after editing %s:\n%s\n' "$FILE" "$(printf '%s' "$OUT" | grep -E '^(FAIL|PASS|Traceback|  File|\w+Error)' | head -40)" >&2
  exit 2
fi
exit 0

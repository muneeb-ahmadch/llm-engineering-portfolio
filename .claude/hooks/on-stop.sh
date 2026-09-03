#!/bin/bash
# Stop gate. The session may not end until build, grader, screenshots and the independent
# review all pass. Blocks up to 4 times, then lets the session end with a loud banner so
# the user knows verification never passed.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COUNTER="$ROOT/.claude/stop-blocks"
cd "$ROOT" || exit 0
n=$(cat "$COUNTER" 2>/dev/null || echo 0)

block() {
  n=$((n + 1)); echo "$n" > "$COUNTER"
  if [ "$n" -gt 4 ]; then
    printf '\n==== VERIFICATION NEVER PASSED (%s blocks). The site is NOT done. ====\n%s\n' "$n" "$1" >&2
    exit 0
  fi
  printf 'STOP BLOCKED (%s/4). Fix these, then try to stop again:\n%s\n' "$n" "$1" >&2
  exit 2
}

[ -f site/build.py ] || block "site/build.py does not exist yet"
OUT="$(python3 site/build.py 2>&1)" || block "build.py failed: $OUT"
OUT="$(python3 site/verify.py --quick 2>&1)" || block "$(printf '%s' "$OUT" | grep -E '^(FAIL|Traceback|\w+Error)' | head -40)"
OUT="$(python3 site/shoot.py 2>&1)" || block "shoot.py failed: $(printf '%s' "$OUT" | tail -5)"
OUT="$(python3 site/verify.py --no-review 2>&1)" || block "$(printf '%s' "$OUT" | grep -E '^FAIL' | head -40)"
OUT="$(python3 site/review.py 2>&1)" || block "independent review failed: $(printf '%s' "$OUT" | tail -12)"
OUT="$(python3 site/verify.py 2>&1)" || block "$(printf '%s' "$OUT" | grep -E '^FAIL' | head -40)"
echo 0 > "$COUNTER"
printf 'STOP GATE PASSED: build, grader, screenshots and independent review are green.\n' >&2
exit 0

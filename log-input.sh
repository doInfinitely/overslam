#!/usr/bin/env bash
# WSL2 wrapper: launches input-logger.py on the Windows host via python.exe.
# The Python script captures Win32 raw mouse + keyboard events into JSONL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/input-logger.py"

WIN_USER="$(/mnt/c/Windows/System32/cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r\n')"
DEFAULT_OUT_DIR="/mnt/c/Users/${WIN_USER}/Videos"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -o, --output PATH   Output JSONL path (WSL or Windows). Default:
                      ${DEFAULT_OUT_DIR}/input-<timestamp>.jsonl
      --no-keyboard   Skip keyboard logging
      --no-mouse      Skip mouse logging
      --python EXE    Windows Python launcher (default: auto-detect py.exe or python.exe)
  -h, --help          Show this help

Run this *before* you start the screen recorder and the game. Stop with Ctrl+C
(both the recorder and the logger should be stopped at roughly the same time).
EOF
}

OUT=""
PY=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)   OUT="$2"; shift 2;;
    --no-keyboard) PASS_ARGS+=(--no-keyboard); shift;;
    --no-mouse)    PASS_ARGS+=(--no-mouse); shift;;
    --python)      PY="$2"; shift 2;;
    -h|--help)     usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

# Find a Windows Python: prefer py.exe (the Python launcher), fall back to python.exe.
if [[ -z "$PY" ]]; then
  if command -v py.exe >/dev/null 2>&1; then
    PY="py.exe -3"
  elif command -v python.exe >/dev/null 2>&1; then
    PY="python.exe"
  else
    cat >&2 <<EOF
No Windows Python found. Install Python on the Windows host:
  winget.exe install --id=Python.Python.3.12 -e
Then open a fresh WSL shell and re-run.
EOF
    exit 1
  fi
fi

if [[ -z "$OUT" ]]; then
  mkdir -p "$DEFAULT_OUT_DIR"
  OUT="${DEFAULT_OUT_DIR}/input-$(date +%Y%m%d-%H%M%S).jsonl"
fi

# Translate paths for python.exe (which runs in Windows context).
PY_SCRIPT_WIN="$(wslpath -w "$PY_SCRIPT")"
# If OUT is a /mnt/<drive>/ path or a WSL path, wslpath -w handles both.
# Make sure the parent dir exists.
mkdir -p "$(dirname "$OUT")"
OUT_WIN="$(wslpath -w "$OUT")"

echo "Logger script: $PY_SCRIPT_WIN"
echo "Output:        $OUT_WIN"
echo "Press Ctrl+C to stop."

# shellcheck disable=SC2086
exec $PY "$PY_SCRIPT_WIN" --output "$OUT_WIN" "${PASS_ARGS[@]}"

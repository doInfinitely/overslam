#!/usr/bin/env bash
# Record the Windows host screen + audio from inside WSL2 by invoking
# ffmpeg.exe on the host. The Linux ffmpeg cannot see the host display
# or audio endpoints, so we shell out to the Windows binary.
set -euo pipefail

WIN_USER="$(/mnt/c/Windows/System32/cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r\n')"
DEFAULT_OUT_DIR="/mnt/c/Users/${WIN_USER}/Videos"
FFMPEG="${FFMPEG:-ffmpeg.exe}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -a, --audio NAME   dshow audio device name (use --list to discover)
  -n, --no-audio     record video only
  -o, --output PATH  output file (Windows or /mnt/c path). Default: ${DEFAULT_OUT_DIR}/rec-<timestamp>.mkv
  -f, --fps N        framerate (default 30)
  -r, --region WxH+X+Y  capture a region instead of full desktop
      --list         list available dshow audio devices and exit
      --no-cursor    do not draw the mouse cursor in the recording
  -h, --help         show this help

Stop recording with Ctrl+C (ffmpeg writes the file trailer cleanly).

A sidecar <output>.meta.json is written alongside the video with the
start wall-clock time in UNIX ns. Combine that with input-logger.py's
t_wall_ns column to align input events to video frames.

For SLAM data capture, start everything in this order:
  1. ./log-input.sh -o input.jsonl     (input logger; keep running)
  2. ./record-windows.sh ...           (this script; keep running)
  3. launch the game and play
  4. Ctrl+C both, in any order

To also show keypresses on the recorded video (for visual debugging),
run a Windows overlay like Carnac (https://github.com/Code52/carnac)
or NohBoard — its overlay window gets captured by gdigrab.
EOF
}

AUDIO=""
NO_AUDIO=0
OUT=""
FPS=30
REGION=""
LIST=0
DRAW_MOUSE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--audio)     AUDIO="$2"; shift 2;;
    -n|--no-audio)  NO_AUDIO=1; shift;;
    -o|--output)    OUT="$2"; shift 2;;
    -f|--fps)       FPS="$2"; shift 2;;
    -r|--region)    REGION="$2"; shift 2;;
    --list)         LIST=1; shift;;
    --no-cursor)    DRAW_MOUSE=0; shift;;
    -h|--help)      usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if ! command -v "$FFMPEG" >/dev/null 2>&1; then
  cat >&2 <<EOF
ffmpeg.exe not found on the Windows PATH.

Install it on Windows (not in WSL):
  winget install --id=Gyan.FFmpeg -e
or download a static build from https://www.gyan.dev/ffmpeg/builds/ and
add its bin/ directory to the Windows PATH, then reopen this shell.
EOF
  exit 1
fi

if [[ "$LIST" -eq 1 ]]; then
  # ffmpeg writes the device list to stderr and exits non-zero; that is normal.
  "$FFMPEG" -hide_banner -list_devices true -f dshow -i dummy 2>&1 || true
  exit 0
fi

if [[ -z "$OUT" ]]; then
  mkdir -p "$DEFAULT_OUT_DIR"
  OUT="${DEFAULT_OUT_DIR}/rec-$(date +%Y%m%d-%H%M%S).mkv"
fi

# Convert /mnt/c/... -> C:\... so ffmpeg.exe writes via the native path
# (faster than going through the 9p translation layer).
to_win_path() {
  local p="$1"
  if [[ "$p" == /mnt/?/* ]]; then
    local drive="${p:5:1}"
    local rest="${p:7}"
    printf '%s:\\%s' "${drive^^}" "${rest//\//\\}"
  else
    printf '%s' "$p"
  fi
}
OUT_WIN="$(to_win_path "$OUT")"

# -use_wallclock_as_timestamps stamps each input's frames with the system
# wall clock so video PTS share a clock with input-logger.py's t_wall_ns.
VIDEO_ARGS=(-f gdigrab -framerate "$FPS" -draw_mouse "$DRAW_MOUSE" -use_wallclock_as_timestamps 1)
if [[ -n "$REGION" ]]; then
  if [[ "$REGION" =~ ^([0-9]+)x([0-9]+)\+([0-9]+)\+([0-9]+)$ ]]; then
    VIDEO_ARGS+=(-video_size "${BASH_REMATCH[1]}x${BASH_REMATCH[2]}" \
                 -offset_x "${BASH_REMATCH[3]}" -offset_y "${BASH_REMATCH[4]}" \
                 -i desktop)
  else
    echo "bad --region format, expected WxH+X+Y (e.g. 1280x720+100+50)" >&2
    exit 2
  fi
else
  VIDEO_ARGS+=(-i desktop)
fi

AUDIO_ARGS=()
if [[ "$NO_AUDIO" -eq 0 ]]; then
  if [[ -z "$AUDIO" ]]; then
    cat >&2 <<EOF
No audio device specified. Run with --list to see available devices, then
pass one with -a "Device Name". To capture system (desktop) audio you
generally need either:
  - "Stereo Mix" enabled in Windows Sound > Recording, or
  - a virtual loopback like VB-Cable / VoiceMeeter installed.
Re-run with -n to record video only.
EOF
    exit 2
  fi
  AUDIO_ARGS=(-f dshow -use_wallclock_as_timestamps 1 -i "audio=${AUDIO}")
fi

# Sidecar metadata: capture the wall-clock instant right before ffmpeg starts.
META="${OUT}.meta.json"
START_NS="$(date +%s%N)"
cat > "$META" <<EOF
{
  "video_path_win": "$OUT_WIN",
  "video_path_wsl": "$OUT",
  "started_wall_ns": $START_NS,
  "fps_requested": $FPS,
  "draw_mouse": $DRAW_MOUSE,
  "region": "${REGION:-fullscreen}",
  "audio_device": $( [[ -n "$AUDIO" ]] && printf '"%s"' "$AUDIO" || printf 'null' )
}
EOF
echo "Sidecar:      $META"
echo "Recording to: $OUT_WIN"
echo "Press Ctrl+C to stop."
exec "$FFMPEG" -hide_banner \
  "${VIDEO_ARGS[@]}" \
  "${AUDIO_ARGS[@]}" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  -c:a aac -b:a 160k \
  "$OUT_WIN"

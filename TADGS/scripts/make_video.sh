#!/usr/bin/env bash
# Create a video from rendered frames (uses ffmpeg).
#
# Usage:
#   scripts/make_video.sh [RENDER_DIR] [OUT.mp4] [FPS] [MODE]
#
#   RENDER_DIR  dir containing gt/ and renders/ (default: results/renders_k8_cam00_all50)
#   OUT.mp4     output file (default: <RENDER_DIR>/<name>.mp4)
#   FPS         frame rate (default: 10)
#   MODE        render | gt | side   (default: side = GT | render, side by side)
#
# Frames are read in sorted filename order, so gaps in numbering are fine.
set -euo pipefail

cd "$(dirname "$0")/.."
DIR="${1:-results/renders_k8_cam00_all50}"
NAME="$(basename "$DIR")"
OUT="${2:-$DIR/$NAME.mp4}"
FPS="${3:-10}"
MODE="${4:-side}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
[ -d "$DIR/renders" ] || { echo "missing $DIR/renders" >&2; exit 1; }

# yuv420p needs even dimensions -> pad/crop via scale filter
EVEN="scale=trunc(iw/2)*2:trunc(ih/2)*2"
COMMON=(-c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart)

case "$MODE" in
  render)
    ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$DIR/renders/*.png" \
      -vf "$EVEN" "${COMMON[@]}" "$OUT" ;;
  gt)
    ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$DIR/gt/*.png" \
      -vf "$EVEN" "${COMMON[@]}" "$OUT" ;;
  side)
    [ -d "$DIR/gt" ] || { echo "missing $DIR/gt" >&2; exit 1; }
    ffmpeg -y \
      -framerate "$FPS" -pattern_type glob -i "$DIR/gt/*.png" \
      -framerate "$FPS" -pattern_type glob -i "$DIR/renders/*.png" \
      -filter_complex "[0:v][1:v]hstack=inputs=2,$EVEN" \
      "${COMMON[@]}" "$OUT" ;;
  *) echo "MODE must be render|gt|side" >&2; exit 1 ;;
esac

echo "Wrote $OUT"

#!/usr/bin/env bash
# Open Blender with proctune loaded from this checkout: scripts/gui.sh [PROJECT.json [blender args]]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
"$root/scripts/blender-site.sh" >/dev/null
project="${1:-$root/examples/limb/limb.json}"
shift || true
exec blender "$@" --python "$root/scripts/blender_dev.py" -- "$project"

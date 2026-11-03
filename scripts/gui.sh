#!/usr/bin/env bash
# Open Blender with proctune loaded from this checkout: scripts/gui.sh [PROJECT.json [blender args]]
#
# Blender's Python (3.14 on Arch) has numpy but not what the example models use, so those go into
# .blender-site once, installed for that Python and put on Blender's path by scripts/blender_dev.py.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
site="$root/.blender-site"
py="$(blender -b --python-expr 'import sys; print("PYVER", f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null | awk '/^PYVER/ {print $2}')"
if [ ! -f "$site/.installed-$py" ]; then
    # an absolute path: uv resolves "python3.14" to its own free-threaded build (.upstream-issues/uv-python3.14-*)
    uv pip install --python "/usr/bin/python$py" --target "$site" scipy trimesh optuna
    touch "$site/.installed-$py"
fi
project="${1:-$root/examples/limb/limb.json}"
shift || true
exec blender "$@" --python "$root/scripts/blender_dev.py" -- "$project"

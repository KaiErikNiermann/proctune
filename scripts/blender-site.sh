#!/usr/bin/env bash
# Install what the example models import into .blender-site, for the Python that Blender embeds (once per
# Python version). Prints the directory.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
site="$root/.blender-site"
py="$(env -u WAYLAND_DISPLAY -u DISPLAY blender -b --factory-startup --python-expr 'import sys; print("PYVER", f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null | awk '/^PYVER/ {print $2}')"
if [ ! -f "$site/.installed-$py" ]; then
    # an absolute path: uv resolves "python3.14" to its own free-threaded build (.upstream-issues/uv-python3.14-*)
    uv pip install --quiet --python "/usr/bin/python$py" --target "$site" scipy trimesh optuna >&2
    touch "$site/.installed-$py"
fi
echo "$site"

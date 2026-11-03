#!/usr/bin/env bash
# Pre-push gate (run by the global pre-push dispatcher): lint, format, types, tests.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
poetry run ruff check .
poetry run ruff format --check .
poetry run pyright
poetry run pytest -q

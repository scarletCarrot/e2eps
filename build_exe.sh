#!/usr/bin/env bash
# Build the standalone E2EPS executable (Linux / macOS).
# Output: dist/e2eps/e2eps  (+ scenarios/ copied next to it)
set -euo pipefail
pip install -e ".[dev]"
pyinstaller --noconfirm --clean --name e2eps --onedir \
    --collect-submodules e2eps \
    --exclude-module tkinter \
    run_e2eps.py
cp -r scenarios dist/e2eps/scenarios
dist/e2eps/e2eps --version
echo "Built dist/e2eps/e2eps"

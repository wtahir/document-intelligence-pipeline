#!/usr/bin/env bash
# render_build.sh — Build script executed by Render during deployment.
# 1. Install Node + build the React frontend into frontend/dist/
# 2. Install Python dependencies into the Render environment

set -e

echo "=== [1/2] Building React frontend ==="
cd frontend
npm ci
npm run build
cd ..
echo "    React build complete → frontend/dist/"

echo "=== [2/2] Installing Python dependencies ==="
pip install -r requirements.txt
echo "    Python deps installed."

echo "=== Build complete ==="

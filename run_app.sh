#!/usr/bin/env bash
# run_app.sh — Start the FastAPI + React dashboard
#
# Usage:
#   ./run_app.sh          # production mode (serves built React from /api + /)
#   ./run_app.sh --dev    # development mode (FastAPI on :8000, Vite dev server on :5173)

set -e
cd "$(dirname "$0")"

# Activate venv if present
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

DEV_MODE=false
for arg in "$@"; do
  [[ "$arg" == "--dev" ]] && DEV_MODE=true
done

if $DEV_MODE; then
  echo "Starting FastAPI backend on http://localhost:8000 ..."
  uvicorn backend.main:app --reload --port 8000 &
  BACKEND_PID=$!

  echo "Starting Vite dev server on http://localhost:5173 ..."
  cd frontend && npm run dev &
  VITE_PID=$!

  echo ""
  echo "Open http://localhost:5173 in your browser."
  echo "Press Ctrl+C to stop both servers."

  trap "kill $BACKEND_PID $VITE_PID 2>/dev/null; exit" INT TERM
  wait
else
  # Production: build React then serve everything from FastAPI
  echo "Building React frontend..."
  cd frontend && npm run build && cd ..

  echo "Starting FastAPI (production) on http://localhost:8000 ..."
  echo "Open http://localhost:8000 in your browser."
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
fi

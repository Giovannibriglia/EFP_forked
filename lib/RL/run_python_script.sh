#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <dot_file> <depth> <agents_number> <goal_file> <state_repr>"
  exit 1
fi

DOT_FILE="$1"
DEPTH="$2"
AGENTNUMBER="$3"
GOALFILE="$4"
STATEREPRESENTATION="$5"

# ── Activate the virtual environment in that directory ──────────────────────────
VENV_PATH=".venv/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "❌ No .venv found at: $VENV_PATH"
  exit 1
fi

source "$VENV_PATH"

# ── Run the Python script ───────────────────────────────────────────────────────
python3 -u "./lib/RL/main.py" "$DOT_FILE" "$DEPTH" "$AGENTNUMBER" "$GOALFILE" "$STATEREPRESENTATION"

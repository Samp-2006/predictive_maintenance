#!/usr/bin/env bash
# ============================================================
# scripts/setup_and_run.sh
# One-shot script: creates venv, installs deps, generates data,
# trains the model, and launches the API.
#
# Usage:
#   chmod +x scripts/setup_and_run.sh
#   ./scripts/setup_and_run.sh
# ============================================================

set -e   # exit on first error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        PREDICTIVE MAINTENANCE SYSTEM SETUP           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────
echo "▸ Checking Python version..."
python3 --version

# ── 2. Virtual environment ────────────────────────────────
if [ ! -d "venv" ]; then
    echo "▸ Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "▸ Virtual environment active: $(which python)"

# ── 3. Install dependencies ───────────────────────────────
echo "▸ Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "▸ Dependencies installed."

# ── 4. Generate synthetic dataset ────────────────────────
echo ""
echo "▸ Generating synthetic audio dataset..."
python data/generate_data.py

# ── 5. Train model ────────────────────────────────────────
echo ""
echo "▸ Training CNN model..."
python training/train.py

# ── 6. Launch API ─────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  API ready!  Open http://localhost:8000/docs         ║"
echo "║  Frontend:   http://localhost:8000/ui                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
python run_server.py

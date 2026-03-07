#!/usr/bin/env bash
# run.sh — Launch the full pipeline and cleanly shut everything down on Ctrl-C.
#
# Process startup order:
#   1. SFML simulation (C++) — opens port 12345 & connects to port 12347
#   2. Path Planner (Python)  — opens port 12347, connects to port 12346
#   3. SLAM node (Python)     — connects to port 12345, opens port 12346
#
# All three are started together; they each retry/wait for their peers.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
PYTHON_DIR="${PROJECT_DIR}/python"
BINARY="${BUILD_DIR}/bin/main"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[run.sh]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${NC} $*"; }
error()   { echo -e "${RED}[run.sh]${NC} $*" >&2; }

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [[ ! -f "${BINARY}" ]]; then
    warn "Binary not found at ${BINARY}. Building now..."
    cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
fi

# ── PID tracking ──────────────────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    info "Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    # Give them a moment, then force-kill stragglers
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            warn "Force-killing PID ${pid}"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    info "Done."
}
trap cleanup INT TERM EXIT

# ── Launch ─────────────────────────────────────────────────────────────────────
info "Starting SFML simulation..."
"${BINARY}" &
PIDS+=($!)
info "  PID: ${PIDS[-1]}"

sleep 0.5   # give the binary a moment to open its sockets

info "Starting Path Planner..."
python3 "${PYTHON_DIR}/path_planner.py" &
PIDS+=($!)
info "  PID: ${PIDS[-1]}"

sleep 0.3

info "Starting SuperGlue SLAM node..."
(cd "${PYTHON_DIR}" && python3 superglue_2d_slam.py) &
PIDS+=($!)
info "  PID: ${PIDS[-1]}"

echo ""
info "All processes started. Press Ctrl-C to stop."
echo ""

# Wait for any child to exit (crash → we shut everything down)
wait -n "${PIDS[@]}" 2>/dev/null || true
warn "A process exited unexpectedly — shutting down."

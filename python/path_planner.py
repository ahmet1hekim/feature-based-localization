"""
path_planner.py
---------------
Reads drone pose (x, y, theta) from SLAM on port 12346.
Runs a fixed-wing P-controller toward goal (300, 700).
Sends command packet to C++ sim on port 12347.

Wire format (all little-endian float32):
  SLAM  → Planner : x(f), y(f), theta_deg(f)
  Planner → C++   : speed(f), turn(f), est_x(f), est_y(f), goal_x(f), goal_y(f),
                    path[0].x(f), path[0].y(f), ..., path[N-1].x(f), path[N-1].y(f)
                    Total: (6 + N_PATH*2) floats
"""

import math
import socket
import struct
import threading
import time

# ── Goal ──────────────────────────────────────────────────────────────────────
GOAL_X = 1680.0
GOAL_Y = 800.0

# ── Controller parameters ─────────────────────────────────────────────────────
# Fixed-wing: always moving, turn and move simultaneously → curved arcs
MIN_SPEED   = 0.6    # px/frame — floor so drone always creeps forward
MAX_SPEED   = 1.6    # px/frame — reduced to avoid overshooting
MAX_TURN    = 0.5    # °/frame
HEADING_KP  = 0.045  # °/frame per ° of heading error
GOAL_RADIUS = 35.0   # px — loosened to account for SLAM/actual offset

# ── Path preview ──────────────────────────────────────────────────────────────
N_PATH = 50          # number of preview waypoints to simulate and send

# ── EMA smoothing ─────────────────────────────────────────────────────────────
EMA_XY    = 0.25
EMA_THETA = 0.10

# ── Networking ────────────────────────────────────────────────────────────────
SLAM_HOST       = "127.0.0.1"
SLAM_PORT       = 12346
CMD_LISTEN_HOST = "0.0.0.0"
CMD_LISTEN_PORT = 12347

# ── Shared raw pose ───────────────────────────────────────────────────────────
_raw_lock  = threading.Lock()
_raw_state = {"x": None, "y": None, "theta": None}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def angle_ema(prev: float, new: float, alpha: float) -> float:
    """EMA on angle (°) with wraparound; result always normalised to [0, 360)."""
    diff = normalize_angle(new - prev)
    return (prev + alpha * diff) % 360.0


def recvall(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# SLAM receiver thread
# ──────────────────────────────────────────────────────────────────────────────

def slam_receiver_thread():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"[planner] Connecting to SLAM at {SLAM_HOST}:{SLAM_PORT}...")
            sock.connect((SLAM_HOST, SLAM_PORT))
            print("[planner] Connected to SLAM.")
            while True:
                data = recvall(sock, 12)
                x, y, theta = struct.unpack("fff", data)
                with _raw_lock:
                    _raw_state["x"]     = x
                    _raw_state["y"]     = y
                    _raw_state["theta"] = theta
        except (ConnectionError, OSError) as e:
            print(f"[planner] SLAM connection lost: {e} — retrying in 2s")
            time.sleep(2)


# ──────────────────────────────────────────────────────────────────────────────
# Control law
# ──────────────────────────────────────────────────────────────────────────────

def compute_commands(x: float, y: float, theta: float):
    """
    Fixed-wing controller: always moving, simultaneous turn → curved arcs.
    theta must be in [0, 360).
    Returns (speed px/frame, turn °/frame).
    """
    dx = GOAL_X - x
    dy = GOAL_Y - y
    dist = math.hypot(dx, dy)

    if dist < GOAL_RADIUS:
        return 0.0, 0.0

    # Desired SFML heading: moveVec = (sin θ, −cos θ), so θ = atan2(dx, −dy)
    desired = math.degrees(math.atan2(dx, -dy))
    err     = normalize_angle(desired - theta)

    # Turn: P-controller, clamped
    turn = max(-MAX_TURN, min(MAX_TURN, HEADING_KP * err))

    # Speed: reduce for very large errors and very near goal; floor at MIN_SPEED
    err_factor  = max(0.35, 1.0 - abs(err) / 150.0)
    dist_factor = min(1.0, dist / 120.0)          # start slowing at 120 px out
    speed = max(MIN_SPEED, MAX_SPEED * err_factor * dist_factor)

    return speed, turn


# ──────────────────────────────────────────────────────────────────────────────
# Path preview: simulate controller forward N_PATH steps
# ──────────────────────────────────────────────────────────────────────────────

def predict_path(x: float, y: float, theta: float) -> list[tuple[float, float]]:
    """
    Forward-simulate the fixed-wing controller to produce a predicted arc.
    Each step advances by the same (speed, turn) the real drone would use.
    """
    pts = []
    cx, cy, cth = x, y, theta
    for _ in range(N_PATH):
        speed, turn = compute_commands(cx, cy, cth)
        if speed == 0.0:
            break
        rad = math.radians(cth)
        cx  += math.sin(rad) * speed
        cy  -= math.cos(rad) * speed        # SFML Y-down
        cth  = (cth + turn) % 360.0
        pts.append((cx, cy))
    # Pad with last point so the packet always has N_PATH entries
    while len(pts) < N_PATH:
        pts.append(pts[-1] if pts else (x, y))
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t = threading.Thread(target=slam_receiver_thread, daemon=True)
    t.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((CMD_LISTEN_HOST, CMD_LISTEN_PORT))
    server.listen(1)
    print(f"[planner] Waiting for C++ sim on port {CMD_LISTEN_PORT}...")

    while True:
        conn, addr = server.accept()
        print(f"[planner] C++ sim connected from {addr}")

        smooth_x:     float | None = None
        smooth_y:     float | None = None
        smooth_theta: float | None = None

        try:
            while True:
                with _raw_lock:
                    rx, ry, rtheta = _raw_state["x"], _raw_state["y"], _raw_state["theta"]

                if rx is None:
                    zeros = [0.0] * (6 + N_PATH * 2)
                    conn.sendall(struct.pack("f" * len(zeros), *zeros))
                    time.sleep(1 / 30)
                    continue

                # EMA update — theta always stays in [0, 360)
                if smooth_x is None:
                    smooth_x, smooth_y  = rx, ry
                    smooth_theta        = rtheta % 360.0
                else:
                    smooth_x     += EMA_XY    * (rx    - smooth_x)
                    smooth_y     += EMA_XY    * (ry    - smooth_y)
                    smooth_theta  = angle_ema(smooth_theta, rtheta, EMA_THETA)

                speed, turn = compute_commands(smooth_x, smooth_y, smooth_theta)
                path        = predict_path(smooth_x, smooth_y, smooth_theta)
                dist        = math.hypot(GOAL_X - smooth_x, GOAL_Y - smooth_y)

                print(f"[planner] "
                      f"smooth=({smooth_x:.0f},{smooth_y:.0f},{smooth_theta:.0f}°)  "
                      f"dist={dist:.0f}  speed={speed:.2f}  turn={turn:.2f}°")

                # Pack: speed, turn, est_x, est_y, path[0].x, path[0].y, ...
                flat_path = [v for pt in path for v in pt]
                payload   = struct.pack("f" * (6 + N_PATH * 2), speed, turn,
                                        smooth_x, smooth_y,
                                        GOAL_X, GOAL_Y,
                                        *flat_path)
                conn.sendall(payload)
                time.sleep(1 / 30)

        except (ConnectionError, BrokenPipeError, OSError) as e:
            print(f"[planner] C++ connection lost: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()

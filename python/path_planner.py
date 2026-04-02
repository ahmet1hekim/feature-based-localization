"""
path_planner.py  (refactored — no TCP sockets)
-----------------------------------------------
Reads drone pose from pose_state (shared dict, written by slam thread).
Computes autopilot commands and predicted path arc.
Writes results to autopilot_state (shared dict, read by sim_engine / UI).

Public API (used by main.py):
    run_planner_thread(pose_state, pose_lock,
                       sim_engine, stop_event=None)

All control-law constants are UNCHANGED from the original.
"""

import math
import threading
import time
from typing import Optional

from sim_engine import AutopilotCmd, SimEngine, N_PATH

# ── Controller parameters (unchanged) ────────────────────────────────────────
GOAL_X      = 1680.0
GOAL_Y      = 800.0

MIN_SPEED   = 0.6
MAX_SPEED   = 1.6
MAX_TURN    = 0.5
HEADING_KP  = 0.045
GOAL_RADIUS = 80.0

EMA_XY      = 0.25
EMA_THETA   = 0.10

TICK_HZ     = 30   # command rate


# ── Helpers (unchanged logic) ─────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def angle_ema(prev: float, new: float, alpha: float) -> float:
    diff = normalize_angle(new - prev)
    return (prev + alpha * diff) % 360.0


def compute_commands(x: float, y: float, theta: float,
                     goal_x: float, goal_y: float):
    dx    = goal_x - x
    dy    = goal_y - y
    dist  = math.hypot(dx, dy)
    if dist < GOAL_RADIUS:
        return 0.0, 0.0
    desired    = math.degrees(math.atan2(dx, -dy))
    err        = normalize_angle(desired - theta)
    turn       = max(-MAX_TURN, min(MAX_TURN, HEADING_KP * err))
    err_factor = max(0.0, 1.0 - abs(err) / 130.0)   # 0 at ±130° heading error
    dist_factor = min(1.0, dist / 200.0)             # brake starts at 200px

    if dist < 250:
        # Near goal: remove MIN_SPEED floor so drone can stop and realign
        speed = MAX_SPEED * err_factor * dist_factor
    else:
        speed = max(MIN_SPEED, MAX_SPEED * err_factor * dist_factor)
    return speed, turn


def predict_path(x: float, y: float, theta: float,
                 goal_x: float, goal_y: float) -> list[tuple[float, float]]:
    pts = []
    cx, cy, cth = x, y, theta
    for _ in range(N_PATH):
        speed, turn = compute_commands(cx, cy, cth, goal_x, goal_y)
        if speed == 0.0:
            break
        rad  = math.radians(cth)
        cx  += math.sin(rad) * speed
        cy  -= math.cos(rad) * speed
        cth  = (cth + turn) % 360.0
        pts.append((cx, cy))
    pad = pts[-1] if pts else (x, y)
    while len(pts) < N_PATH:
        pts.append(pad)
    return pts


# ── Main thread function ──────────────────────────────────────────────────────

def run_planner_thread(
    pose_state:  dict,
    pose_lock:   threading.Lock,
    sim_engine:  SimEngine,
    stop_event:  Optional[threading.Event] = None,
) -> None:
    """
    Blocking function — call inside a daemon thread from main.py.
    Reads pose from shared dict, pushes AutopilotCmd to sim_engine.
    """
    smooth_x: Optional[float] = None
    smooth_y: Optional[float] = None
    smooth_theta: Optional[float] = None

    print("[planner] Started.")

    while stop_event is None or not stop_event.is_set():
        with pose_lock:
            rx     = pose_state.get("x")
            ry     = pose_state.get("y")
            rtheta = pose_state.get("theta")

        # Read current goal from sim_engine (may have been changed by UI click)
        with sim_engine.cmd_lock:
            goal_x = sim_engine.autopilot_cmd.goal_x
            goal_y = sim_engine.autopilot_cmd.goal_y

        if rx is None:
            # No pose yet — send zero command
            cmd = AutopilotCmd(goal_x=goal_x, goal_y=goal_y)
            sim_engine.apply_autopilot_cmd(cmd, active=False)
            time.sleep(1.0 / TICK_HZ)
            continue

        # EMA smoothing
        if smooth_x is None:
            smooth_x     = rx
            smooth_y     = ry
            smooth_theta = rtheta % 360.0
        else:
            smooth_x    += EMA_XY    * (rx    - smooth_x)
            smooth_y    += EMA_XY    * (ry    - smooth_y)
            smooth_theta = angle_ema(smooth_theta, rtheta, EMA_THETA)

        speed, turn = compute_commands(smooth_x, smooth_y, smooth_theta, goal_x, goal_y)
        path        = predict_path(smooth_x, smooth_y, smooth_theta, goal_x, goal_y)
        dist        = math.hypot(goal_x - smooth_x, goal_y - smooth_y)

        print(f"[planner] pos=({smooth_x:.0f},{smooth_y:.0f},{smooth_theta:.0f}°) "
              f"dist={dist:.0f}  speed={speed:.2f}  turn={turn:.2f}°")

        cmd = AutopilotCmd(
            speed      = speed,
            turn_angle = turn,
            est_x      = smooth_x,
            est_y      = smooth_y,
            goal_x     = goal_x,
            goal_y     = goal_y,
            path_x     = [p[0] for p in path],
            path_y     = [p[1] for p in path],
            path_len   = len(path),
        )
        sim_engine.apply_autopilot_cmd(cmd, active=True)
        time.sleep(1.0 / TICK_HZ)


if __name__ == "__main__":
    print("[planner] Run via python/main.py — this module is no longer standalone.")

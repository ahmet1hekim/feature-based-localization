"""
path_planner.py  (refactored — no TCP sockets)
-----------------------------------------------
Reads drone pose from pose_state (shared dict, written by slam thread).
Reads waypoints from sim_engine.
Generates a dense global path, uses Pure Pursuit for lookahead local goal.
Outputs AutopilotCmd to sim_engine.
"""

import math
import threading
import time
from typing import Optional

from sim_engine import AutopilotCmd, SimEngine, N_PATH

# ── Controller parameters ─────────────────────────────────────────────────────
MIN_SPEED   = 0.6
MAX_SPEED   = 1.6
MAX_TURN    = 0.5
HEADING_KP  = 0.045
GOAL_RADIUS = 30.0

EMA_XY      = 1.0
EMA_THETA   = 0.10

TICK_HZ     = 30   # command rate
LOOKAHEAD_D = 40.0 # Pure pursuit lookahead distance (px)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def angle_ema(prev: float, new: float, alpha: float) -> float:
    diff = normalize_angle(new - prev)
    return (prev + alpha * diff) % 360.0

def compute_commands(x: float, y: float, theta: float,
                     goal_x: float, goal_y: float, dist_to_final: float):
    dx    = goal_x - x
    dy    = goal_y - y
    dist  = math.hypot(dx, dy)
    
    if dist < 5.0 and dist_to_final < GOAL_RADIUS:
        return 0.0, 0.0
        
    desired    = math.degrees(math.atan2(dx, -dy))
    err        = normalize_angle(desired - theta)
    turn       = max(-MAX_TURN, min(MAX_TURN, HEADING_KP * err))
    
    err_factor = max(0.0, 1.0 - abs(err) / 130.0)
    dist_factor = min(1.0, dist_to_final / 100.0)

    if dist_to_final < 150:
        speed = MAX_SPEED * err_factor * dist_factor
    else:
        speed = max(MIN_SPEED, MAX_SPEED * err_factor * dist_factor)
        
    return speed, turn

def predict_path(x: float, y: float, theta: float,
                 goal_x: float, goal_y: float, dist_to_final: float) -> list[tuple[float, float]]:
    pts = []
    cx, cy, cth = x, y, theta
    for _ in range(N_PATH):
        speed, turn = compute_commands(cx, cy, cth, goal_x, goal_y, dist_to_final)
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

def generate_global_path(start_x: float, start_y: float, waypoints: list, step: float = 10.0) -> list[tuple[float, float]]:
    """Generate a dense path by discretizing straight lines between all waypoints."""
    if not waypoints:
        return []
        
    pts = []
    pts.append((start_x, start_y))
    
    current_x, current_y = start_x, start_y
    for wp in waypoints:
        wx, wy = wp
        dx = wx - current_x
        dy = wy - current_y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            ux, uy = dx / dist, dy / dist
            moved = 0.0
            while moved < dist:
                moved += step
                if moved >= dist:
                    pts.append((wx, wy))
                else:
                    pts.append((current_x + ux * moved, current_y + uy * moved))
                    
        current_x, current_y = wx, wy
        
    return pts

def get_pure_pursuit_lookahead(x: float, y: float, path: list[tuple[float, float]], lookahead: float):
    """Find the point on the path that is at least `lookahead` distance away from the drone."""
    if not path:
        return x, y
        
    # Find closest point on path
    closest_idx = 0
    min_dist = float('inf')
    for i, p in enumerate(path):
        d = math.hypot(p[0] - x, p[1] - y)
        if d < min_dist:
            min_dist = d
            closest_idx = i
            
    # Search forward from closest point to find a lookahead point
    for i in range(closest_idx, len(path)):
        p = path[i]
        d = math.hypot(p[0] - x, p[1] - y)
        if d >= lookahead:
            return p[0], p[1]
            
    return path[-1][0], path[-1][1]

# ── Main thread function ──────────────────────────────────────────────────────

def run_planner_thread(
    pose_state:  dict,
    pose_lock:   threading.Lock,
    sim_engine:  SimEngine,
    stop_event:  Optional[threading.Event] = None,
) -> None:
    """
    Blocking function — call inside a daemon thread from main.py.
    Reads pose from shared dict, generates dense path, computes pure pursuit lookahead,
    and pushes AutopilotCmd to sim_engine.
    """
    smooth_x: Optional[float] = None
    smooth_y: Optional[float] = None
    smooth_theta: Optional[float] = None

    print("[planner] Started.")

    while stop_event is None or not stop_event.is_set():
        if not sim_engine.is_running:
            time.sleep(1.0 / TICK_HZ)
            continue
            
        with pose_lock:
            rx     = pose_state.get("x")
            ry     = pose_state.get("y")
            rtheta = pose_state.get("theta")

        if rx is None:
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

        # 1. Check if we reached the active waypoint (the first one)
        waypoints = list(sim_engine.waypoints)  # snapshot
        if waypoints:
            active_wp = waypoints[0]
            dist_to_wp = math.hypot(active_wp[0] - smooth_x, active_wp[1] - smooth_y)
            if dist_to_wp < GOAL_RADIUS:
                sim_engine.remove_waypoint(0)
                waypoints.pop(0)
                print(f"[planner] Waypoint reached! {len(waypoints)} remaining.")

        if not waypoints:
            # No waypoints, stop
            cmd = AutopilotCmd(speed=0.0, turn_angle=0.0, est_x=smooth_x, est_y=smooth_y)
            sim_engine.apply_autopilot_cmd(cmd, active=True)
            time.sleep(1.0 / TICK_HZ)
            continue

        # 2. Generate Global Path
        global_path = generate_global_path(smooth_x, smooth_y, waypoints, step=10.0)
        
        # 3. Pure Pursuit Lookahead
        la_x, la_y = get_pure_pursuit_lookahead(smooth_x, smooth_y, global_path, LOOKAHEAD_D)
        
        # 4. Compute Commands using the lookahead as target
        final_wp = waypoints[-1]
        dist_to_final = math.hypot(final_wp[0] - smooth_x, final_wp[1] - smooth_y)
        
        speed, turn = compute_commands(smooth_x, smooth_y, smooth_theta, la_x, la_y, dist_to_final)
        
        # Predicted path arc
        path_arc = predict_path(smooth_x, smooth_y, smooth_theta, la_x, la_y, dist_to_final)

        cmd = AutopilotCmd(
            speed      = speed,
            turn_angle = turn,
            est_x      = smooth_x,
            est_y      = smooth_y,
            goal_x     = la_x,
            goal_y     = la_y,
            path_x     = [p[0] for p in path_arc],
            path_y     = [p[1] for p in path_arc],
            path_len   = len(path_arc),
            global_path_x=[p[0] for p in global_path],
            global_path_y=[p[1] for p in global_path],
        )
        sim_engine.apply_autopilot_cmd(cmd, active=True)
        time.sleep(1.0 / TICK_HZ)

if __name__ == "__main__":
    print("[planner] Run via python/main.py — this module is no longer standalone.")

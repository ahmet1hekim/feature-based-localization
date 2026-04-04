"""
path_planner.py
-----------------------------------------------
Reads drone pose from pose_state (shared dict, written by slam thread).
Reads waypoints from sim_engine.
Generates a static Centripetal Catmull-Rom global path, uses Pure Pursuit for lookahead local goal.
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
MAX_TURN    = 0.5    # Restored to 0.5 (15 deg/sec real hardware limit)
HEADING_KP  = 0.045  # Restored to original sluggish response
GOAL_RADIUS = 30.0

EMA_XY      = 1.0
EMA_THETA   = 0.10

TICK_HZ     = 30   # command rate
LOOKAHEAD_D = 90.0 # Pure pursuit lookahead distance (px)


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

def generate_hermite_path(start_x: float, start_y: float, start_theta: float, waypoints: list, step: float = 10.0) -> list[tuple[float, float]]:
    """Generate a dense curved path using a Kinematic Hermite Spline.
       We FORCE the tangents at each waypoint to be massive (>300px).
       This guarantees the global path exactly strikes the waypoints, but if the
       waypoints are too close or sharp, the massive tangents will mathematically
       force the path to sweep into giant teardrop loops to hit them — which is
       EXACTLY what a real drone with a 180px turning radius physically bounded expects!"""
    if not waypoints:
        return []
        
    pts = [(start_x, start_y)] + waypoints
    
    # Calculate tangents T for each point
    T = []
    
    # T[0] matches the drone's current exact heading but with a massive forward push
    rad_theta = math.radians(start_theta)
    # The drone needs ~180px radius turning, so give it ~300px of forward momentum buffer
    mag0 = max(300.0, math.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]))
    T.append((math.sin(rad_theta) * mag0, -math.cos(rad_theta) * mag0))
    
    for i in range(1, len(pts)-1):
        # Tangent direction based on neighboring points (Catmull-Rom style angle bisector)
        dx = pts[i+1][0] - pts[i-1][0]
        dy = pts[i+1][1] - pts[i-1][1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            ux, uy = dx/dist, dy/dist
        else:
            ux, uy = 1.0, 0.0
            
        # FORCE a massive tangent magnitude to guarantee wide sweeping curves capable of R=180!
        mag = max(300.0, dist * 0.8)
        T.append((ux * mag, uy * mag))
        
    # T[-1] straight into the last waypoint
    if len(pts) > 1:
        dx = pts[-1][0] - pts[-2][0]
        dy = pts[-1][1] - pts[-2][1]
        dist = math.hypot(dx, dy)
        mag_end = max(300.0, dist * 0.8)
        if dist > 0:
            T.append((dx/dist * mag_end, dy/dist * mag_end))
        else:
             T.append((0.0, 0.0))
    else:
        T.append((0.0, 0.0))
        
    # Interpolate using Cubic Hermite equations
    path = []
    for i in range(len(pts)-1):
        P0, P1 = pts[i], pts[i+1]
        T0, T1 = T[i], T[i+1]
        
        dist = math.hypot(P1[0]-P0[0], P1[1]-P0[1])
        # We need lots of samples because the curve might intentionally perform a large teardrop loop!
        arc_estimate = dist + math.hypot(T0[0], T0[1]) + math.hypot(T1[0], T1[1])
        num_samples = max(2, int(arc_estimate / (step * 0.75)))
        
        for j in range(num_samples):
            t = j / num_samples
            t2 = t * t
            t3 = t2 * t
            
            # Cubic Hermite basis functions
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            
            x = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
            y = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
            path.append((x, y))
            
    path.append(pts[-1])
    return path

def get_pure_pursuit_lookahead(x: float, y: float, path: list[tuple[float, float]], lookahead: float):
    """Find the point on the path that is at least `lookahead` distance away from the drone."""
    if not path:
        return x, y
        
    # Find closest point on path but bounding the search window to prevent jumping across overlaps!
    closest_idx = 0
    min_dist = float('inf')
    search_limit = min(len(path), 40)  # ~400px parametric window max
    for i in range(search_limit):
        p = path[i]
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
    """
    smooth_x: Optional[float] = None
    smooth_y: Optional[float] = None
    smooth_theta: Optional[float] = None
    
    # Path cache state to prevent reference track from moving or regenerating unexpectedly
    last_ui_waypoints = ()
    cached_global_path = []

    print("[planner] Started.")

    while stop_event is None or not stop_event.is_set():
        with pose_lock:
            rx     = pose_state.get("x")
            ry     = pose_state.get("y")
            rtheta = pose_state.get("theta")
            do_reset = pose_state.get("reset_planner", False)
            if do_reset:
                pose_state["reset_planner"] = False

        if do_reset:
            smooth_x = None
            smooth_y = None
            smooth_theta = None
            last_ui_waypoints = ()
            cached_global_path = []

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

        waypoints = list(sim_engine.waypoints)
        
        # 1. Stop processing if no waypoints
        if not waypoints:
            cmd = AutopilotCmd(speed=0.0, turn_angle=0.0, est_x=smooth_x, est_y=smooth_y)
            sim_engine.apply_autopilot_cmd(cmd, active=True)
            last_ui_waypoints = () 
            cached_global_path = []
            time.sleep(1.0 / TICK_HZ)
            continue

        # 2. Re-plan Global Path ONLY if the UI explicitly added/removed/reordered waypoints.
        waypoints_tuple = tuple(waypoints)
        if waypoints_tuple != last_ui_waypoints:
            cached_global_path = generate_hermite_path(smooth_x, smooth_y, smooth_theta, waypoints, step=10.0)
            last_ui_waypoints = waypoints_tuple
            
        # 3. Prune the static path behind the drone so it visually "eats" the trail
        if cached_global_path:
            min_dist = float('inf')
            closest_idx = 0
            search_limit = min(len(cached_global_path), 40) # Prevent skipping overlaps
            for i in range(search_limit):
                p = cached_global_path[i]
                d = math.hypot(p[0] - smooth_x, p[1] - smooth_y)
                if d < min_dist:
                    min_dist = d
                    closest_idx = i
            
            # Keep trail slightly behind the drone for smooth rendering
            keep_idx = max(0, closest_idx - 2)
            cached_global_path = cached_global_path[keep_idx:]

        # 4. Check if we reached the active UI waypoint
        active_wp = waypoints[0]
        dist_to_wp = math.hypot(active_wp[0] - smooth_x, active_wp[1] - smooth_y)
        if dist_to_wp < GOAL_RADIUS:
            sim_engine.remove_waypoint(0)
            waypoints.pop(0)
            # CRITICAL: Update the tracker so this automatic pop DOES NOT trigger a replan!
            last_ui_waypoints = tuple(waypoints)
            print(f"[planner] Waypoint reached! {len(waypoints)} remaining.")
            if not waypoints:
                continue

        # 5. Pure Pursuit Lookahead
        la_x, la_y = get_pure_pursuit_lookahead(smooth_x, smooth_y, cached_global_path, LOOKAHEAD_D)
        
        # 6. Compute Control Commands
        final_wp = waypoints[-1]
        dist_to_final = math.hypot(final_wp[0] - smooth_x, final_wp[1] - smooth_y)
        
        speed, turn = compute_commands(smooth_x, smooth_y, smooth_theta, la_x, la_y, dist_to_final)
        
        # If physics are paused by the user, zero out execution speeds but still update visuals!
        if not sim_engine.is_running:
            speed = 0.0
            turn = 0.0
            
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
            global_path_x=[p[0] for p in cached_global_path],
            global_path_y=[p[1] for p in cached_global_path],
        )
        sim_engine.apply_autopilot_cmd(cmd, active=True)
        time.sleep(1.0 / TICK_HZ)

if __name__ == "__main__":
    print("[planner] Run via python/main.py — this module is no longer standalone.")

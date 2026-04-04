import math
import threading
import time
from typing import Optional

from fbl.core.state import AutopilotCmd
from fbl.navigation.controller import get_pure_pursuit_lookahead, compute_commands, GOAL_RADIUS
from fbl.navigation.path_generator import generate_hermite_path, predict_path

EMA_XY      = 1.0
EMA_THETA   = 0.10
TICK_HZ     = 30
LOOKAHEAD_D = 90.0

def angle_ema(prev: float, new: float, alpha: float) -> float:
    diff = (new - prev + 180.0) % 360.0 - 180.0
    return (prev + alpha * diff) % 360.0

class NavigationNode(threading.Thread):
    """
    Planner thread separated from SimEngine.
    Retrieves waypoints from UI directly, and PoseState from SLAM directly.
    Injects commands into the physics.
    """
    def __init__(
        self,
        pose_state:  dict,
        pose_lock:   threading.Lock,
        get_waypoints_callback, # Returns list[tuple]
        remove_waypoint_callback, # Removes index
        is_running_callback,
        apply_cmd_callback, # Sends AutopilotCmd
        stop_event:  Optional[threading.Event] = None
    ):
        super().__init__(daemon=True, name="Planner")
        self.pose_state = pose_state
        self.pose_lock = pose_lock
        self.get_waypoints = get_waypoints_callback
        self.remove_waypoint = remove_waypoint_callback
        self.is_running = is_running_callback
        self.apply_cmd = apply_cmd_callback
        self.stop_event = stop_event

    def run(self) -> None:
        smooth_x: Optional[float] = None
        smooth_y: Optional[float] = None
        smooth_theta: Optional[float] = None
        
        last_ui_waypoints = ()
        cached_global_path = []

        print("[planner] Started.")

        while self.stop_event is None or not self.stop_event.is_set():
            with self.pose_lock:
                rx     = self.pose_state.get("x")
                ry     = self.pose_state.get("y")
                rtheta = self.pose_state.get("theta")
                do_reset = self.pose_state.get("reset_planner", False)
                if do_reset:
                    self.pose_state["reset_planner"] = False

            if do_reset:
                smooth_x = None
                smooth_y = None
                smooth_theta = None
                last_ui_waypoints = ()
                cached_global_path = []

            if rx is None:
                time.sleep(1.0 / TICK_HZ)
                continue

            if smooth_x is None:
                smooth_x     = rx
                smooth_y     = ry
                smooth_theta = rtheta % 360.0
            else:
                smooth_x    += EMA_XY    * (rx    - smooth_x)
                smooth_y    += EMA_XY    * (ry    - smooth_y)
                smooth_theta = angle_ema(smooth_theta, rtheta, EMA_THETA)

            waypoints = list(self.get_waypoints())
            
            if not waypoints:
                cmd = AutopilotCmd(speed=0.0, turn_angle=0.0, est_x=smooth_x, est_y=smooth_y)
                self.apply_cmd(cmd, active=True)
                last_ui_waypoints = () 
                cached_global_path = []
                time.sleep(1.0 / TICK_HZ)
                continue

            waypoints_tuple = tuple(waypoints)
            needs_replan = (waypoints_tuple != last_ui_waypoints)
            
            if not needs_replan and cached_global_path:
                dist_jump = math.hypot(cached_global_path[0][0] - smooth_x, cached_global_path[0][1] - smooth_y)
                if dist_jump > 200.0:
                    needs_replan = True

            if needs_replan:
                cached_global_path = generate_hermite_path(smooth_x, smooth_y, smooth_theta, waypoints, step=10.0)
                last_ui_waypoints = waypoints_tuple
                
            if cached_global_path:
                min_dist = float('inf')
                closest_idx = 0
                search_limit = min(len(cached_global_path), 40)
                for i in range(search_limit):
                    p = cached_global_path[i]
                    d = math.hypot(p[0] - smooth_x, p[1] - smooth_y)
                    if d < min_dist:
                        min_dist = d
                        closest_idx = i
                
                keep_idx = max(0, closest_idx - 2)
                cached_global_path = cached_global_path[keep_idx:]

            active_wp = waypoints[0]
            dist_to_wp = math.hypot(active_wp[0] - smooth_x, active_wp[1] - smooth_y)
            if dist_to_wp < GOAL_RADIUS:
                self.remove_waypoint(0)
                waypoints.pop(0)
                last_ui_waypoints = tuple(waypoints)
                print(f"[planner] Waypoint reached! {len(waypoints)} remaining.")
                if not waypoints:
                    continue

            la_x, la_y = get_pure_pursuit_lookahead(smooth_x, smooth_y, cached_global_path, LOOKAHEAD_D)
            
            final_wp = waypoints[-1]
            dist_to_final = math.hypot(final_wp[0] - smooth_x, final_wp[1] - smooth_y)
            
            speed, turn = compute_commands(smooth_x, smooth_y, smooth_theta, la_x, la_y, dist_to_final)
            
            if not self.is_running():
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
            self.apply_cmd(cmd, active=True)
            time.sleep(1.0 / TICK_HZ)

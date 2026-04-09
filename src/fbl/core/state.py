from dataclasses import dataclass, field

@dataclass
class AutopilotCmd:
    speed:      float = 0.0
    turn_angle: float = 0.0
    est_x:      float = -1.0
    est_y:      float = -1.0
    est_angle:  float = -1.0
    # Next local sub-goal
    goal_x:     float = 0.0
    goal_y:     float = 0.0
    # Short local path arc prediction (from path planner)
    path_x:     list  = field(default_factory=lambda: [0.0] * 50) # N_PATH=50
    path_y:     list  = field(default_factory=lambda: [0.0] * 50)
    path_len:   int   = 0
    # Global path points (for visualization from GlobalPlanner)
    global_path_x: list = field(default_factory=list)
    global_path_y: list = field(default_factory=list)

@dataclass
class DroneState:
    """Snapshot of the drone state for the UI to render each frame."""
    pos_x:    float = 640.0
    pos_y:    float = 360.0
    angle:    float = 0.0       # degrees, SFML convention
    cmd:      AutopilotCmd = field(default_factory=AutopilotCmd)
    bg_w:     int   = 0
    bg_h:     int   = 0
    is_running: bool = False
    waypoints: list = field(default_factory=list)

import math

MIN_SPEED   = 0.6
MAX_SPEED   = 1.6
MAX_TURN    = 0.5
HEADING_KP  = 0.045
GOAL_RADIUS = 30.0

def normalize_angle(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

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

def get_pure_pursuit_lookahead(x: float, y: float, path: list[tuple[float, float]], lookahead: float):
    if not path:
        return x, y
        
    closest_idx = 0
    min_dist = float('inf')
    search_limit = min(len(path), 40)
    for i in range(search_limit):
        p = path[i]
        d = math.hypot(p[0] - x, p[1] - y)
        if d < min_dist:
            min_dist = d
            closest_idx = i
            
    for i in range(closest_idx, len(path)):
        p = path[i]
        d = math.hypot(p[0] - x, p[1] - y)
        if d >= lookahead:
            return p[0], p[1]
            
    return path[-1][0], path[-1][1]

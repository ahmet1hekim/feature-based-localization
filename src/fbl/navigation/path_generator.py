import math
from fbl.navigation.controller import compute_commands

N_PATH = 50

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
    if not waypoints:
        return []
        
    pts = [(start_x, start_y)] + waypoints
    
    T = []
    rad_theta = math.radians(start_theta)
    mag0 = max(300.0, math.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]))
    T.append((math.sin(rad_theta) * mag0, -math.cos(rad_theta) * mag0))
    
    for i in range(1, len(pts)-1):
        dx = pts[i+1][0] - pts[i-1][0]
        dy = pts[i+1][1] - pts[i-1][1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            ux, uy = dx/dist, dy/dist
        else:
            ux, uy = 1.0, 0.0
            
        mag = max(300.0, dist * 0.8)
        T.append((ux * mag, uy * mag))
        
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
        
    path = []
    for i in range(len(pts)-1):
        P0, P1 = pts[i], pts[i+1]
        T0, T1 = T[i], T[i+1]
        
        dist = math.hypot(P1[0]-P0[0], P1[1]-P0[1])
        arc_estimate = dist + math.hypot(T0[0], T0[1]) + math.hypot(T1[0], T1[1])
        num_samples = max(2, int(arc_estimate / (step * 0.75)))
        
        for j in range(num_samples):
            t = j / num_samples
            t2 = t * t
            t3 = t2 * t
            
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            
            x = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
            y = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
            path.append((x, y))
            
    path.append(pts[-1])
    return path

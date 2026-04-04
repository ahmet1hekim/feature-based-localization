import math
import cv2
import numpy as np
from fbl.core.state import DroneState

MAP_W = 1050
MAP_H = 900 # Inherited from APP_H

class MapView:
    def __init__(self, bg_w: int, bg_h: int):
        self.bg_w, self.bg_h = bg_w, bg_h
        self.zoom   = max(MAP_W / bg_w, MAP_H / bg_h)
        
        # Center the view initially
        self.offset = np.array([
            self.bg_w / 2.0 - (MAP_W / self.zoom) / 2.0,
            self.bg_h / 2.0 - (MAP_H / self.zoom) / 2.0
        ])
        self.clamp()

    def w2s(self, wx, wy):
        return ((wx - self.offset[0]) * self.zoom,
                (wy - self.offset[1]) * self.zoom)

    def s2w(self, sx, sy):
        return (sx / self.zoom + self.offset[0],
                sy / self.zoom + self.offset[1])

    def clamp(self):
        max_ox = max(0.0, self.bg_w - MAP_W / self.zoom)
        max_oy = max(0.0, self.bg_h - MAP_H / self.zoom)
        self.offset[0] = max(0.0, min(self.offset[0], max_ox))
        self.offset[1] = max(0.0, min(self.offset[1], max_oy))

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.s2w(sx, sy)
        zoom_min = max(MAP_W / self.bg_w, MAP_H / self.bg_h)
        self.zoom = max(zoom_min, min(self.zoom * factor, 8.0))
        self.offset[0] = wx - sx / self.zoom
        self.offset[1] = wy - sy / self.zoom
        self.clamp()

def _dashed_line(img, p0, p1, color, dash=14, gap=8, thickness=1):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx/dist, dy/dist
    pos, drawing = 0.0, True
    while pos < dist:
        seg = dash if drawing else gap
        if drawing:
            end = min(pos+seg, dist)
            a = (int(p0[0]+ux*pos), int(p0[1]+uy*pos))
            b = (int(p0[0]+ux*end), int(p0[1]+uy*end))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        pos += seg
        drawing = not drawing

def render_map_frame(bg: np.ndarray, mv: MapView, state: DroneState,
                     show_slam: bool, show_arc: bool, show_dash: bool) -> np.ndarray:
    bg_h, bg_w = bg.shape[:2]
    ox, oy = mv.offset
    zoom   = mv.zoom

    ix0 = int(max(0, ox))
    iy0 = int(max(0, oy))
    ix1 = int(min(bg_w, ox + MAP_W / zoom)) + 1
    iy1 = int(min(bg_h, oy + MAP_H / zoom)) + 1
    ix1 = min(ix1, bg_w)
    iy1 = min(iy1, bg_h)

    canvas = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

    if ix1 > ix0 and iy1 > iy0:
        crop    = bg[iy0:iy1, ix0:ix1]
        scr_w   = max(1, int(round((ix1 - ix0) * zoom)))
        scr_h   = max(1, int(round((iy1 - iy0) * zoom)))
        resized = cv2.resize(crop, (scr_w, scr_h), interpolation=cv2.INTER_LINEAR)

        scr_x0 = int(round((ix0 - ox) * zoom))
        scr_y0 = int(round((iy0 - oy) * zoom))

        dst_x0 = max(0, scr_x0);  dst_y0 = max(0, scr_y0)
        dst_x1 = min(MAP_W, scr_x0 + scr_w)
        dst_y1 = min(MAP_H, scr_y0 + scr_h)
        src_x0 = dst_x0 - scr_x0;  src_y0 = dst_y0 - scr_y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            canvas[dst_y0:dst_y1, dst_x0:dst_x1] = \
                resized[src_y0:src_y1, src_x0:src_x1]

    def s(wx, wy):
        return (int((wx - ox) * zoom), int((wy - oy) * zoom))
    def dot_r():
        return max(2, int(2.5 * zoom))

    cmd = state.cmd

    if cmd.global_path_x and len(cmd.global_path_x) > 1:
        global_col = (180, 100, 255)
        for i in range(1, len(cmd.global_path_x)):
            cv2.line(canvas, 
                     s(cmd.global_path_x[i-1], cmd.global_path_y[i-1]), 
                     s(cmd.global_path_x[i],   cmd.global_path_y[i]), 
                     global_col, 2, cv2.LINE_AA)

    wp_r = max(4, int(8 * zoom))
    for i, wp in enumerate(state.waypoints):
        wp_color = (0, 100, 255) if i == 0 else (0, 200, 255)
        cv2.circle(canvas, s(wp[0], wp[1]), wp_r, wp_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, s(wp[0], wp[1]), wp_r, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), s(wp[0] + 15, wp[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if show_arc and cmd.path_len > 1:
        arc_col = (50, 220, 80)
        for i in range(1, cmd.path_len):
            cv2.line(canvas, s(cmd.path_x[i-1], cmd.path_y[i-1]),
                     s(cmd.path_x[i],   cmd.path_y[i]),
                     arc_col, 1, cv2.LINE_AA)
        for i in range(0, cmd.path_len, 5):
            cv2.circle(canvas, s(cmd.path_x[i], cmd.path_y[i]),
                       dot_r(), arc_col, -1, cv2.LINE_AA)

    has_active_mission = len(state.waypoints) > 0

    if show_dash and has_active_mission:
        _dashed_line(canvas, s(state.pos_x, state.pos_y),
                     s(cmd.goal_x, cmd.goal_y), (0, 220, 255), thickness=1)

    if has_active_mission:
        gr = max(6, int(10 * zoom))
        cv2.circle(canvas, s(cmd.goal_x, cmd.goal_y), gr, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, s(cmd.goal_x, cmd.goal_y), gr, (255, 255, 255),  1, cv2.LINE_AA)

    if show_slam and (cmd.est_x > 0 or cmd.est_y > 0):
        sr = max(5, int(10 * zoom))
        cv2.circle(canvas, s(cmd.est_x, cmd.est_y), sr, (255, 180, 50), -1, cv2.LINE_AA)
        cv2.circle(canvas, s(cmd.est_x, cmd.est_y), sr, (255, 255, 255),  1, cv2.LINE_AA)
        if has_active_mission:
            _dashed_line(canvas, s(cmd.est_x, cmd.est_y), s(cmd.goal_x, cmd.goal_y),
                         (255, 180, 50), dash=8, gap=6)

    rw = max(8, int(20 * zoom))
    rh = max(12, int(30 * zoom))
    rad = math.radians(state.angle)
    ca, sa_ = math.cos(rad), math.sin(rad)
    dx, dy = s(state.pos_x, state.pos_y)
    corners_local = [(-rw/2, -rh/2), (rw/2, -rh/2), (rw/2, rh/2), (-rw/2, rh/2)]
    corners = np.array(
        [(int(ca*lx - sa_*ly + dx), int(sa_*lx + ca*ly + dy)) for lx, ly in corners_local],
        dtype=np.int32)
    cv2.fillPoly(canvas,  [corners], (30, 30, 220))
    cv2.polylines(canvas, [corners], True, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

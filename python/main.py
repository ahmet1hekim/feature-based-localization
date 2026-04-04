"""
main.py — Unified entry point.
Renders everything via OpenCV, uploads to DPG dynamic textures.
No drawlists — just dpg.add_image widgets for reliable display.

Layout:
  ┌──────────────────────────────────┬────────────────┐
  │         Map View                 │  Drone Cam     │
  │  (zoom+pan=scroll/mid-drag,      ├────────────────┤
  │   left-click = set goal)         │  SG Matches    │
  │                                  ├────────────────┤
  │                                  │  Info / Ctrl   │
  └──────────────────────────────────┴────────────────┘
"""

import math
import os
import queue
import sys
import threading

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sim_engine import SimEngine, DroneState, AutopilotCmd, N_PATH
from superglue_2d_slam import run_slam_thread
from path_planner import run_planner_thread

# ── Layout ────────────────────────────────────────────────────────────────────
APP_W = 1600
APP_H = 900

MAP_W = 1050
MAP_H = APP_H

RIGHT_W     = APP_W - MAP_W   # 550
CAM_IMG_H   = 284
CAM_H       = CAM_IMG_H + 22
MATCH_IMG_H = 274
MATCH_H     = MATCH_IMG_H + 22
INFO_H      = APP_H - CAM_H - MATCH_H

BG_IMAGE = "dag.jpg"

# ── Shared comms ──────────────────────────────────────────────────────────────
pose_state: dict = {"x": None, "y": None, "theta": None}
pose_lock         = threading.Lock()
frame_queue       = queue.Queue(maxsize=1)   # keep only the LATEST frame for SLAM
match_queue       = queue.Queue(maxsize=4)
stop_event        = threading.Event()


# ── Zoom / pan state ──────────────────────────────────────────────────────────
class MapView:
    def __init__(self, bg_w: int, bg_h: int):
        self.zoom   = min(MAP_W / bg_w, MAP_H / bg_h)
        self.offset = np.array([0.0, 0.0])
        self.bg_w, self.bg_h = bg_w, bg_h

    def w2s(self, wx, wy):   # world → screen pixel on the MAP_W×MAP_H canvas
        return ((wx - self.offset[0]) * self.zoom,
                (wy - self.offset[1]) * self.zoom)

    def s2w(self, sx, sy):   # screen pixel → world
        return (sx / self.zoom + self.offset[0],
                sy / self.zoom + self.offset[1])

    def clamp(self):
        # Allow black borders: image can partially go off-screen
        # Just keep at least 100px of image visible in each direction
        margin = 100.0 / self.zoom      # world units of required overlap
        self.offset[0] = max(-MAP_W/self.zoom + margin,
                             min(self.offset[0], self.bg_w - margin))
        self.offset[1] = max(-MAP_H/self.zoom + margin,
                             min(self.offset[1], self.bg_h - margin))

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.s2w(sx, sy)
        self.zoom = max(0.05, min(self.zoom * factor, 8.0))
        self.offset[0] = wx - sx / self.zoom
        self.offset[1] = wy - sy / self.zoom
        self.clamp()


# ── OpenCV overlay drawing ────────────────────────────────────────────────────

# _s is now a closure built inside render_map_frame with the correct scale.


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
    """
    Render the map at NATURAL resolution: 1 world pixel = zoom screen pixels.
    No stretching.  Areas outside the bg image are black.
    Overlays use the same world→screen transform so they never drift.
    """
    bg_h, bg_w = bg.shape[:2]
    ox, oy = mv.offset
    zoom   = mv.zoom

    # Integer world-pixel boundaries of what the bg can supply
    ix0 = int(max(0, ox))
    iy0 = int(max(0, oy))
    ix1 = int(min(bg_w, ox + MAP_W / zoom)) + 1   # +1 to avoid 1-px gaps
    iy1 = int(min(bg_h, oy + MAP_H / zoom)) + 1
    ix1 = min(ix1, bg_w)
    iy1 = min(iy1, bg_h)

    canvas = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

    if ix1 > ix0 and iy1 > iy0:
        crop    = bg[iy0:iy1, ix0:ix1]
        scr_w   = max(1, int(round((ix1 - ix0) * zoom)))
        scr_h   = max(1, int(round((iy1 - iy0) * zoom)))
        resized = cv2.resize(crop, (scr_w, scr_h), interpolation=cv2.INTER_LINEAR)

        # Where the resized block lands on the canvas
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

    # All overlays use the exact same transform — pixel perfect.
    def s(wx, wy):
        return (int((wx - ox) * zoom), int((wy - oy) * zoom))

    def dot_r():
        return max(2, int(2.5 * zoom))

    cmd = state.cmd

    # Global path (curved spline / dense path provided by path_planner)
    if cmd.global_path_x and len(cmd.global_path_x) > 1:
        global_col = (180, 100, 255)
        for i in range(1, len(cmd.global_path_x)):
            cv2.line(canvas, 
                     s(cmd.global_path_x[i-1], cmd.global_path_y[i-1]), 
                     s(cmd.global_path_x[i],   cmd.global_path_y[i]), 
                     global_col, 2, cv2.LINE_AA)

    # Draw waypoints as circles
    wp_r = max(4, int(8 * zoom))
    for i, wp in enumerate(state.waypoints):
        wp_color = (0, 100, 255) if i == 0 else (0, 200, 255)
        cv2.circle(canvas, s(wp[0], wp[1]), wp_r, wp_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, s(wp[0], wp[1]), wp_r, (255, 255, 255), 1, cv2.LINE_AA)
        # Draw index
        cv2.putText(canvas, str(i), s(wp[0] + 15, wp[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Path arc
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

    # Dashed line: drone → current active local goal (which may be a lookahead point)
    if show_dash and has_active_mission:
        _dashed_line(canvas, s(state.pos_x, state.pos_y),
                     s(cmd.goal_x, cmd.goal_y), (0, 220, 255), thickness=1)

    # Local goal marker (pure pursuit lookahead point or actual waypoint)
    if has_active_mission:
        gr = max(6, int(10 * zoom))
        cv2.circle(canvas, s(cmd.goal_x, cmd.goal_y), gr, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, s(cmd.goal_x, cmd.goal_y), gr, (255, 255, 255),  1, cv2.LINE_AA)

    # SLAM estimated pos
    if show_slam and (cmd.est_x > 0 or cmd.est_y > 0):
        sr = max(5, int(10 * zoom))
        cv2.circle(canvas, s(cmd.est_x, cmd.est_y), sr, (255, 180, 50), -1, cv2.LINE_AA)
        cv2.circle(canvas, s(cmd.est_x, cmd.est_y), sr, (255, 255, 255),  1, cv2.LINE_AA)
        if has_active_mission:
            _dashed_line(canvas, s(cmd.est_x, cmd.est_y), s(cmd.goal_x, cmd.goal_y),
                         (255, 180, 50), dash=8, gap=6)

    # Drone (filled rotated rectangle)
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


# ── DPG texture helpers ───────────────────────────────────────────────────────
# Key: dpg.set_value accepts a contiguous float32 numpy array — NO .tolist()!
# Calling .tolist() on a 3.78M-element array costs ~100-200 ms / frame.

def _to_float32_flat(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → RGBA float32 flat contiguous array. img_bgr must be correct size already."""
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    return np.ascontiguousarray(rgba, dtype=np.float32) / 255.0


def _to_float32_flat_resized(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """BGR uint8 → resize to w×h → RGBA float32 flat."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return _to_float32_flat(resized)


def upload_map(img_bgr: np.ndarray):
    """Map image is already MAP_W×MAP_H — just convert and upload."""
    dpg.set_value("map_tex", _to_float32_flat(img_bgr).ravel())


def upload_resized(tag: str, img_bgr: np.ndarray, w: int, h: int):
    """Resize to w×h then upload."""
    dpg.set_value(tag, _to_float32_flat_resized(img_bgr, w, h).ravel())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    engine = SimEngine(BG_IMAGE, frame_queue)
    mv     = MapView(engine.bg_w, engine.bg_h)

    # Initial drone world position (center of bg image)
    init_x = engine.bg_w / 2.0
    init_y = engine.bg_h / 2.0

    dpg.create_context()
    dpg.create_viewport(title="Feature-Based Localization",
                        width=APP_W, height=APP_H, resizable=False)
    dpg.setup_dearpygui()

    # ── Textures (dynamic = designed for set_value updates) ───────────────────
    blank_map   = np.zeros((MAP_H,   MAP_W,   4), dtype=np.float32).ravel()
    blank_cam   = np.zeros((CAM_IMG_H,   RIGHT_W, 4), dtype=np.float32).ravel()
    blank_match = np.zeros((MATCH_IMG_H, RIGHT_W, 4), dtype=np.float32).ravel()

    with dpg.texture_registry():
        dpg.add_dynamic_texture(MAP_W,   MAP_H,     blank_map,   tag="map_tex")
        dpg.add_dynamic_texture(RIGHT_W, CAM_IMG_H, blank_cam,   tag="cam_tex")
        dpg.add_dynamic_texture(RIGHT_W, MATCH_IMG_H, blank_match, tag="match_tex")

    # Seed map with initial viewport immediately
    init_cmd = AutopilotCmd()
    init_cmd.goal_x = init_x
    init_cmd.goal_y = init_y
    init_frame = DroneState(pos_x=init_x, pos_y=init_y, cmd=init_cmd)
    upload_map(render_map_frame(engine.background, mv, init_frame, True, True, True))

    # ── Theme ─────────────────────────────────────────────────────────────────
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,   (18, 18, 25, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,    (18, 18, 25, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 80, 160, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,    (35, 35, 50,  255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,     (50, 90, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70,120,210,255))
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)
    dpg.bind_theme(theme)

    # ── Layout ────────────────────────────────────────────────────────────────
    with dpg.window(tag="main_win", no_title_bar=True, no_resize=True,
                    no_move=True, no_scrollbar=True,
                    width=APP_W, height=APP_H, pos=(0, 0)):

        with dpg.group(horizontal=True):

            # Map panel — just an image, overlays baked in via OpenCV
            with dpg.child_window(tag="map_panel", width=MAP_W, height=MAP_H,
                                  no_scrollbar=True, border=False):
                dpg.add_image("map_tex", width=MAP_W, height=MAP_H, tag="map_img")

            # Right column
            with dpg.child_window(width=RIGHT_W, height=APP_H,
                                  no_scrollbar=True, border=False):

                with dpg.child_window(height=CAM_H, no_scrollbar=True, border=False):
                    dpg.add_text("▶  Drone Camera", color=(160, 180, 255, 255))
                    dpg.add_image("cam_tex", width=RIGHT_W, height=CAM_IMG_H)

                dpg.add_separator()

                with dpg.child_window(height=MATCH_H, no_scrollbar=True, border=False):
                    dpg.add_text("▶  SuperGlue Matches", color=(160, 255, 180, 255))
                    dpg.add_image("match_tex", width=RIGHT_W, height=MATCH_IMG_H)

                dpg.add_separator()

                with dpg.child_window(height=INFO_H, no_scrollbar=True, border=False):
                    dpg.add_text("Controls & Info", color=(255, 220, 120, 255))
                    dpg.add_separator()
                    
                    # SIMULATION CONTROLS
                    with dpg.group(horizontal=True):
                        def on_start(): engine.is_running = True
                        def on_stop():  engine.is_running = False
                        def on_reset():
                            engine.reset_state()
                            with pose_lock:
                                pose_state["reset_slam"] = True
                                pose_state["reset_planner"] = True
                            
                            # Give visual feedback of match wipe
                            empty_match = np.zeros((MATCH_IMG_H, RIGHT_W, 3), dtype=np.uint8)
                            upload_resized("match_tex", empty_match, RIGHT_W, MATCH_IMG_H)
                        
                        dpg.add_button(label=" ▶ Start ", callback=on_start)
                        dpg.add_button(label=" ⏸ Stop ", callback=on_stop)
                        dpg.add_button(label=" ↺ Reset ", callback=on_reset)
                        
                    dpg.add_text(tag="txt_runtime_status", default_value="Status: STOPPED", color=(255, 100, 100, 255))
                    dpg.add_separator()
                    
                    # WAYPOINT CONTROLS
                    dpg.add_text("Waypoints (Global Path):")
                    dpg.add_listbox(items=[], tag="list_waypoints", width=-1, num_items=5)
                    
                    with dpg.group(horizontal=True):
                        def on_wp_up():
                            idx = dpg.get_value("list_waypoints")
                            if not idx: return
                            wps = engine.waypoints
                            try:
                                i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(wps)].index(idx)
                                if i > 0:
                                    wps[i-1], wps[i] = wps[i], wps[i-1]
                                    engine.reorder_waypoints(wps)
                            except ValueError: pass
                            
                        def on_wp_down():
                            idx = dpg.get_value("list_waypoints")
                            if not idx: return
                            wps = engine.waypoints
                            try:
                                i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(wps)].index(idx)
                                if i < len(wps) - 1:
                                    wps[i+1], wps[i] = wps[i], wps[i+1]
                                    engine.reorder_waypoints(wps)
                            except ValueError: pass

                        def on_wp_del():
                            idx = dpg.get_value("list_waypoints")
                            if not idx: return
                            wps = engine.waypoints
                            try:
                                i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(wps)].index(idx)
                                engine.remove_waypoint(i)
                            except ValueError: pass

                        def on_wp_clear(): engine.clear_waypoints()

                        dpg.add_button(label="↑", callback=on_wp_up)
                        dpg.add_button(label="↓", callback=on_wp_down)
                        dpg.add_button(label="Del", callback=on_wp_del)
                        dpg.add_button(label="Clear", callback=on_wp_clear)
                        
                    dpg.add_separator()

                    dpg.add_checkbox(label="SLAM estimate",  tag="chk_slam", default_value=True)
                    dpg.add_checkbox(label="Path arc",       tag="chk_arc",  default_value=True)
                    dpg.add_checkbox(label="Dashed line",    tag="chk_dash", default_value=True)
                    dpg.add_separator()
                    dpg.add_text("Scroll=zoom  Drag=pan  RClick=Ad Waypoint",
                                 color=(140, 140, 140, 180))
                    dpg.add_separator()
                    dpg.add_text("Drone", color=(160, 160, 160, 200))
                    dpg.add_text("x=---  y=---  θ=---", tag="txt_pos")
                    dpg.add_text("SLAM", color=(160, 160, 160, 200))
                    dpg.add_text("x=---  y=---",        tag="txt_slam")
                    dpg.add_text("Local Goal", color=(160, 160, 160, 200))
                    dpg.add_text("x=---  y=---",        tag="txt_goal")
                    dpg.add_separator()
                    dpg.add_text("←→ rotate  ↑↓ throttle  Space boost",
                                 color=(120, 120, 120, 160))

    # ── Mouse handlers ────────────────────────────────────────────────────────
    with dpg.handler_registry():
        def on_scroll(s, delta):
            mx, my = dpg.get_mouse_pos(local=False)
            if 0 <= mx <= MAP_W and 0 <= my <= MAP_H:
                mv.zoom_at(mx, my, 1.12 if delta > 0 else 1.0/1.12)

        # Track left-drag vs left-click (short click = nothing, drag = pan)
        _drag_accum = [0.0]

        def on_drag(s, data):
            if not dpg.is_item_hovered("map_panel"):
                return
            if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                _drag_accum[0] += abs(data[1]) + abs(data[2])
                mv.offset[0] -= data[1] / mv.zoom
                mv.offset[1] -= data[2] / mv.zoom
                mv.clamp()

        def on_mouse_down(s, btn):
            if btn == dpg.mvMouseButton_Left:
                _drag_accum[0] = 0.0  # reset on press

        def on_click(s, btn):
            if btn == dpg.mvMouseButton_Right:
                # Right-click appends waypoint
                mx, my = dpg.get_mouse_pos(local=False)
                if 0 <= mx <= MAP_W and 0 <= my <= MAP_H:
                    wx, wy = mv.s2w(mx, my)
                    engine.add_waypoint(wx, wy)
                    print(f"[main] Added Waypoint → ({wx:.0f}, {wy:.0f})")

        dpg.add_mouse_wheel_handler(callback=on_scroll)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=on_drag)
        dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=on_mouse_down)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=on_click)

    # ── Threads ───────────────────────────────────────────────────────────────
    slam_t = threading.Thread(
        target=run_slam_thread,
        kwargs=dict(frame_queue=frame_queue, pose_state=pose_state,
                    pose_lock=pose_lock, match_queue=match_queue,
                    stop_event=stop_event,
                    start_x=init_x, start_y=init_y),
        daemon=True, name="SLAM")
    plan_t = threading.Thread(
        target=run_planner_thread,
        kwargs=dict(pose_state=pose_state, pose_lock=pose_lock,
                    sim_engine=engine, stop_event=stop_event),
        daemon=True, name="Planner")
    slam_t.start()
    plan_t.start()
    print("[main] Threads started.")

    # ── Key map ───────────────────────────────────────────────────────────────
    key_map = {
        dpg.mvKey_Left:     "left",
        dpg.mvKey_Right:    "right",
        dpg.mvKey_Up:       "up",
        dpg.mvKey_Down:     "down",
        dpg.mvKey_Spacebar: "space",
    }

    latest_match: list = [None]

    dpg.set_primary_window("main_win", True)
    dpg.show_viewport()

    # ── Render loop ───────────────────────────────────────────────────────────
    try:
        while dpg.is_dearpygui_running():
            keys  = {v: dpg.is_key_down(k) for k, v in key_map.items()}
            state = engine.tick(keys)

            # Map: bake overlays into image then upload
            show_slam = dpg.get_value("chk_slam")
            show_arc  = dpg.get_value("chk_arc")
            show_dash = dpg.get_value("chk_dash")
            map_img   = render_map_frame(engine.background, mv, state,
                                         show_slam, show_arc, show_dash)
            upload_map(map_img)

            # Drone cam: only upload when a new frame actually arrived
            with engine._cam_lock:
                cam_pair    = engine._latest_cam
                cam_changed = cam_pair is not engine._prev_cam_uploaded
                if cam_changed:
                    engine._prev_cam_uploaded = cam_pair
            if cam_changed and cam_pair is not None:
                upload_resized("cam_tex", cam_pair[0], RIGHT_W, CAM_IMG_H)

            # Matches: drain queue, upload only if new frame arrived
            new_match = False
            try:
                while True:
                    _, vis = match_queue.get_nowait()
                    latest_match[0] = vis
                    new_match = True
            except queue.Empty:
                pass
            if new_match and latest_match[0] is not None:
                upload_resized("match_tex", latest_match[0], RIGHT_W, MATCH_IMG_H)

            # Info text
            cmd = state.cmd
            dpg.set_value("txt_pos",
                f"x={state.pos_x:.0f}  y={state.pos_y:.0f}  θ={state.angle:.1f}°")
            dpg.set_value("txt_slam",
                f"x={cmd.est_x:.0f}  y={cmd.est_y:.0f}"
                if (cmd.est_x > 0 or cmd.est_y > 0) else "x=---  y=---")
            dpg.set_value("txt_goal",
                f"x={cmd.goal_x:.0f}  y={cmd.goal_y:.0f}")

            status_str = "Status: RUNNING" if state.is_running else "Status: STOPPED"
            status_col = (100, 255, 100, 255) if state.is_running else (255, 100, 100, 255)
            dpg.set_value("txt_runtime_status", status_str)
            dpg.configure_item("txt_runtime_status", color=status_col)

            # Update Waypoints listbox items
            wp_strs = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(state.waypoints)]
            dpg.configure_item("list_waypoints", items=wp_strs)

            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        print("\n[main] Ctrl-C.")
    finally:
        stop_event.set()
        slam_t.join(timeout=3)
        plan_t.join(timeout=3)
        dpg.destroy_context()
        print("[main] Done.")


if __name__ == "__main__":
    main()

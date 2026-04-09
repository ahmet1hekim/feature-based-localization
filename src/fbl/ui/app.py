import queue
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from fbl.ui.map_view import MapView, render_map_frame, MAP_W, MAP_H
from fbl.core.engine import SimEngine
from fbl.core.state import DroneState

APP_W = 1600
APP_H = 900
RIGHT_W     = APP_W - MAP_W
CAM_IMG_H   = 284
CAM_H       = CAM_IMG_H + 22
MATCH_IMG_H = 274
MATCH_H     = MATCH_IMG_H + 22
INFO_H      = APP_H - CAM_H - MATCH_H

def _to_float32_flat(img_bgr: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    return np.ascontiguousarray(rgba, dtype=np.float32) / 255.0

def _to_float32_flat_resized(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return _to_float32_flat(resized)

def upload_map(img_bgr: np.ndarray):
    dpg.set_value("map_tex", _to_float32_flat(img_bgr).ravel())

def upload_resized(tag: str, img_bgr: np.ndarray, w: int, h: int):
    dpg.set_value(tag, _to_float32_flat_resized(img_bgr, w, h).ravel())

class Application:
    def __init__(self, engine: SimEngine, pose_state: dict, pose_lock, match_queue: queue.Queue):
        self.engine = engine
        self.pose_state = pose_state
        self.pose_lock = pose_lock
        self.match_queue = match_queue
        self.mv = MapView(engine.bg_w, engine.bg_h)
        self.latest_match = [None]

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Feature-Based Localization", width=APP_W, height=APP_H, resizable=False)
        dpg.setup_dearpygui()

        blank_map   = np.zeros((MAP_H,   MAP_W,   4), dtype=np.float32).ravel()
        blank_cam   = np.zeros((CAM_IMG_H,   RIGHT_W, 4), dtype=np.float32).ravel()
        blank_match = np.zeros((MATCH_IMG_H, RIGHT_W, 4), dtype=np.float32).ravel()

        with dpg.texture_registry():
            dpg.add_dynamic_texture(MAP_W,   MAP_H,     blank_map,   tag="map_tex")
            dpg.add_dynamic_texture(RIGHT_W, CAM_IMG_H, blank_cam,   tag="cam_tex")
            dpg.add_dynamic_texture(RIGHT_W, MATCH_IMG_H, blank_match, tag="match_tex")

        init_frame = DroneState(pos_x=self.engine.bg_w/2, pos_y=self.engine.bg_h/2)
        upload_map(render_map_frame(self.engine.background, self.mv, init_frame, True, True, True))

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

        with dpg.window(tag="main_win", no_title_bar=True, no_resize=True, no_move=True, no_scrollbar=True, width=APP_W, height=APP_H, pos=(0, 0)):
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="map_panel", width=MAP_W, height=MAP_H, no_scrollbar=True, border=False):
                    dpg.add_image("map_tex", width=MAP_W, height=MAP_H, tag="map_img")

                with dpg.child_window(width=RIGHT_W, height=APP_H, no_scrollbar=True, border=False):
                    with dpg.child_window(height=CAM_H, no_scrollbar=True, border=False):
                        dpg.add_text("  Drone Camera", color=(160, 180, 255, 255))
                        dpg.add_image("cam_tex", width=RIGHT_W, height=CAM_IMG_H)
                    dpg.add_separator()
                    with dpg.child_window(height=MATCH_H, no_scrollbar=True, border=False):
                        dpg.add_text("  Visual Odometry Matches", color=(160, 255, 180, 255))
                        dpg.add_image("match_tex", width=RIGHT_W, height=MATCH_IMG_H)
                    dpg.add_separator()
                    with dpg.child_window(height=INFO_H, no_scrollbar=True, border=False):
                        dpg.add_text("Controls & Info", color=(255, 220, 120, 255))
                        dpg.add_separator()
                        
                        def on_matcher_change(s, data):
                            with self.pose_lock:
                                self.pose_state["switch_matcher"] = data

                        dpg.add_combo(
                            items=["SuperGlue", "LightGlue", "MatchAnything"],
                            default_value="SuperGlue",
                            callback=on_matcher_change,
                            width=-1
                        )
                        dpg.add_separator()
                        
                        with dpg.group(horizontal=True):
                            def on_start(): self.engine.is_running = True
                            def on_stop():  self.engine.is_running = False
                            def on_reset():
                                self.engine.reset_state()
                                with self.pose_lock:
                                    self.pose_state["reset_vo"] = True
                                    self.pose_state["reset_planner"] = True
                                upload_resized("match_tex", np.zeros((MATCH_IMG_H, RIGHT_W, 3), dtype=np.uint8), RIGHT_W, MATCH_IMG_H)
                            
                            dpg.add_button(label="  Start ", callback=on_start)
                            dpg.add_button(label="  Stop ", callback=on_stop)
                            dpg.add_button(label="  Reset ", callback=on_reset)
                            
                        dpg.add_text(tag="txt_runtime_status", default_value="Status: STOPPED", color=(255, 100, 100, 255))
                        dpg.add_separator()
                        
                        dpg.add_text("Waypoints (Global Path):")
                        dpg.add_listbox(items=[], tag="list_waypoints", width=-1, num_items=5)
                        
                        with dpg.group(horizontal=True):
                            def on_wp_up():
                                idx = dpg.get_value("list_waypoints")
                                if not idx: return
                                wps = self.engine.waypoints
                                try:
                                    i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(wps)].index(idx)
                                    if i > 0:
                                        wps[i-1], wps[i] = wps[i], wps[i-1]
                                        self.engine.reorder_waypoints(wps)
                                except ValueError: pass
                                
                            def on_wp_down():
                                idx = dpg.get_value("list_waypoints")
                                if not idx: return
                                wps = self.engine.waypoints
                                try:
                                    i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(wps)].index(idx)
                                    if i < len(wps) - 1:
                                        wps[i+1], wps[i] = wps[i], wps[i+1]
                                        self.engine.reorder_waypoints(wps)
                                except ValueError: pass

                            def on_wp_del():
                                idx = dpg.get_value("list_waypoints")
                                if not idx: return
                                wps = self.engine.waypoints
                                try:
                                    i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(wps)].index(idx)
                                    self.engine.remove_waypoint(i)
                                except ValueError: pass

                            def on_wp_clear(): self.engine.clear_waypoints()

                            dpg.add_button(label="Move Up", callback=on_wp_up)
                            dpg.add_button(label="Move Down", callback=on_wp_down)
                            dpg.add_button(label="Del", callback=on_wp_del)
                            dpg.add_button(label="Clear", callback=on_wp_clear)
                            
                        dpg.add_separator()
                        dpg.add_checkbox(label="SLAM estimate",  tag="chk_slam", default_value=True)
                        dpg.add_checkbox(label="Path arc",       tag="chk_arc",  default_value=True)
                        dpg.add_checkbox(label="Dashed line",    tag="chk_dash", default_value=True)
                        dpg.add_separator()
                        dpg.add_text("Scroll=zoom  Drag=pan  RClick=Ad Waypoint", color=(140, 140, 140, 180))
                        dpg.add_separator()
                        dpg.add_text("Drone", color=(160, 160, 160, 200))
                        dpg.add_text("x=---  y=---  θ=---", tag="txt_pos")
                        dpg.add_text("SLAM", color=(160, 160, 160, 200))
                        dpg.add_text("x=---  y=---  θ=---", tag="txt_slam")
                        dpg.add_text("Local Goal", color=(160, 160, 160, 200))
                        dpg.add_text("x=---  y=---",        tag="txt_goal")

        with dpg.handler_registry():
            def on_scroll(s, delta):
                mx, my = dpg.get_mouse_pos(local=False)
                if 0 <= mx <= MAP_W and 0 <= my <= MAP_H:
                    self.mv.zoom_at(mx, my, 1.12 if delta > 0 else 1.0/1.12)

            _is_dragging = [False]
            _last_mouse_pos = [0.0, 0.0]

            def on_drag(s, data):
                if not dpg.is_item_hovered("map_panel"):
                    return
                if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                    mx, my = dpg.get_mouse_pos(local=False)
                    if not _is_dragging[0]:
                        _is_dragging[0] = True
                        _last_mouse_pos[0] = mx
                        _last_mouse_pos[1] = my
                        return
                    
                    dx = mx - _last_mouse_pos[0]
                    dy = my - _last_mouse_pos[1]
                    _last_mouse_pos[0] = mx
                    _last_mouse_pos[1] = my
                    
                    self.mv.offset[0] -= dx / self.mv.zoom
                    self.mv.offset[1] -= dy / self.mv.zoom
                    self.mv.clamp()

            def on_mouse_down(s, btn):
                pass

            def on_mouse_release(s, btn):
                if btn == dpg.mvMouseButton_Left:
                    _is_dragging[0] = False

            def on_click(s, btn):
                if btn == dpg.mvMouseButton_Right:
                    mx, my = dpg.get_mouse_pos(local=False)
                    if 0 <= mx <= MAP_W and 0 <= my <= MAP_H:
                        wx, wy = self.mv.s2w(mx, my)
                        self.engine.add_waypoint(wx, wy)
                        print(f"Added Waypoint → ({wx:.0f}, {wy:.0f})")

            dpg.add_mouse_wheel_handler(callback=on_scroll)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=on_drag)
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=on_mouse_down)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=on_mouse_release)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=on_click)

        self.key_map = {
            dpg.mvKey_Left:     "left",
            dpg.mvKey_Right:    "right",
            dpg.mvKey_Up:       "up",
            dpg.mvKey_Down:     "down",
            dpg.mvKey_Spacebar: "space",
        }
        dpg.set_primary_window("main_win", True)
        dpg.show_viewport()

    def run_frame(self):
        keys  = {v: dpg.is_key_down(k) for k, v in self.key_map.items()}
        state = self.engine.tick(keys)

        show_slam = dpg.get_value("chk_slam")
        show_arc  = dpg.get_value("chk_arc")
        show_dash = dpg.get_value("chk_dash")
        map_img   = render_map_frame(self.engine.background, self.mv, state, show_slam, show_arc, show_dash)
        upload_map(map_img)

        with self.engine._cam_lock:
            cam_pair    = self.engine._latest_cam
            cam_changed = cam_pair is not self.engine._prev_cam_uploaded
            if cam_changed:
                self.engine._prev_cam_uploaded = cam_pair
        if cam_changed and cam_pair is not None:
            upload_resized("cam_tex", cam_pair[0], RIGHT_W, CAM_IMG_H)

        new_match = False
        try:
            while True:
                _, vis = self.match_queue.get_nowait()
                self.latest_match[0] = vis
                new_match = True
        except queue.Empty:
            pass
        if new_match and self.latest_match[0] is not None:
            upload_resized("match_tex", self.latest_match[0], RIGHT_W, MATCH_IMG_H)

        cmd = state.cmd
        dpg.set_value("txt_pos", f"x={state.pos_x:.0f}  y={state.pos_y:.0f}  θ={state.angle:.1f}°")
        dpg.set_value("txt_slam", f"x={cmd.est_x:.0f}  y={cmd.est_y:.0f}  θ={cmd.est_angle:.1f}°" if (cmd.est_x > 0 or cmd.est_y > 0) else "x=---  y=---  θ=---°")
        dpg.set_value("txt_goal", f"x={cmd.goal_x:.0f}  y={cmd.goal_y:.0f}")

        status_str = "Status: RUNNING" if state.is_running else "Status: STOPPED"
        status_col = (100, 255, 100, 255) if state.is_running else (255, 100, 100, 255)
        dpg.set_value("txt_runtime_status", status_str)
        dpg.configure_item("txt_runtime_status", color=status_col)

        wp_strs = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(state.waypoints)]
        dpg.configure_item("list_waypoints", items=wp_strs)
        dpg.render_dearpygui_frame()

    def teardown(self):
        dpg.destroy_context()

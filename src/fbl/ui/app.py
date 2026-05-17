import os
import queue
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from fbl.ui.map_view import MapView, render_map_frame
from fbl.ui.benchmark_runner import BenchmarkRunner
from fbl.core.engine import SimEngine, ASSETS_DIR
from fbl.core.state import DroneState

# --------------------------------------------------------------------------
# Default / initial layout constants
# --------------------------------------------------------------------------
APP_W        = 1600
APP_H        = 900
MAP_FRACTION = 1050 / 1600   # map panel takes ~65.6 % of the width

# Height of a label row + separator (roughly constant, DPG internal)
_LABEL_SEP_H = 22


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
        self.latest_match = [None]

        # Layout dimensions — computed from viewport size
        self._vp_w = APP_W
        self._vp_h = APP_H
        self._compute_layout()

        self.mv = MapView(engine.bg_w, engine.bg_h, self._map_w, self._map_h)

        self.bench_event_queue: queue.Queue = queue.Queue(maxsize=32)
        self.bench_runner = BenchmarkRunner(
            engine, pose_state, pose_lock, self.bench_event_queue
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _compute_layout(self):
        w, h = self._vp_w, self._vp_h
        self._map_w   = int(w * MAP_FRACTION)
        self._map_h   = h  # used for map texture size
        # DPG inserts ItemSpacing.x (4px) between the two horizontal-group
        # children; subtract it from right_w so total = client_w exactly.
        self._right_w = w - self._map_w - 4

        # Cam / match heights track right_w so images keep their aspect ratio
        # (drone cam is 960×540 → 9:16;  match vis uses the same 16:9 crop)
        self._cam_img_h   = max(80, int(self._right_w * 9 / 16))
        self._cam_h       = self._cam_img_h + _LABEL_SEP_H
        self._match_img_h = max(60, int(self._right_w * 8 / 16))
        self._match_h     = self._match_img_h + _LABEL_SEP_H
        # info_panel uses height=-1 → fills whatever is left in right_col

    def _recreate_textures(self):
        """Delete + recreate dynamic textures at the new resolution."""
        for tag in ("map_tex", "cam_tex", "match_tex"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            if dpg.does_alias_exist(tag):
                dpg.remove_alias(tag)

        blank_map   = np.zeros((self._map_h,    self._map_w,   4), dtype=np.float32).ravel()
        blank_cam   = np.zeros((self._cam_img_h,   self._right_w, 4), dtype=np.float32).ravel()
        blank_match = np.zeros((self._match_img_h, self._right_w, 4), dtype=np.float32).ravel()

        with dpg.texture_registry():
            dpg.add_dynamic_texture(self._map_w,   self._map_h,      blank_map,   tag="map_tex")
            dpg.add_dynamic_texture(self._right_w, self._cam_img_h,  blank_cam,   tag="cam_tex")
            dpg.add_dynamic_texture(self._right_w, self._match_img_h, blank_match, tag="match_tex")

        # Re-bind textures on their image items
        dpg.configure_item("map_img",   texture_tag="map_tex",   width=self._map_w, height=self._map_h)
        dpg.configure_item("cam_img",   texture_tag="cam_tex",   width=self._right_w, height=self._cam_img_h)
        dpg.configure_item("match_img", texture_tag="match_tex", width=self._right_w, height=self._match_img_h)

    def _resize_layout(self):
        """Reconfigure child windows and images — outer panels use height=-1."""
        dpg.configure_item("map_panel",   width=self._map_w)
        dpg.configure_item("map_img",     width=self._map_w,  height=self._map_h)
        dpg.configure_item("right_col",   width=self._right_w)
        dpg.configure_item("cam_panel",   height=self._cam_h)
        dpg.configure_item("cam_img",     width=self._right_w, height=self._cam_img_h)
        dpg.configure_item("match_panel", height=self._match_h)
        dpg.configure_item("match_img",   width=self._right_w, height=self._match_img_h)
        # info_panel: height=-1, expands automatically — no configure needed

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="Feature-Based Localization",
            width=APP_W, height=APP_H,
            resizable=True,
            min_width=800, min_height=500,
        )
        dpg.setup_dearpygui()

        # Initial textures
        blank_map   = np.zeros((self._map_h,    self._map_w,   4), dtype=np.float32).ravel()
        blank_cam   = np.zeros((self._cam_img_h,   self._right_w, 4), dtype=np.float32).ravel()
        blank_match = np.zeros((self._match_img_h, self._right_w, 4), dtype=np.float32).ravel()

        with dpg.texture_registry():
            dpg.add_dynamic_texture(self._map_w,   self._map_h,       blank_map,   tag="map_tex")
            dpg.add_dynamic_texture(self._right_w, self._cam_img_h,   blank_cam,   tag="cam_tex")
            dpg.add_dynamic_texture(self._right_w, self._match_img_h, blank_match, tag="match_tex")

        init_frame = DroneState(pos_x=self.engine.bg_w / 2, pos_y=self.engine.bg_h / 2)
        upload_map(render_map_frame(self.engine.background, self.mv, init_frame, True, True, True))

        # ---- Theme -------------------------------------------------------
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,     (18, 18, 25, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,      (18, 18, 25, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,(40, 80, 160, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (35, 35, 50,  255))
                dpg.add_theme_color(dpg.mvThemeCol_Button,       (50, 90, 180, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,(70,120,210, 255))
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   0, 0)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)
        dpg.bind_theme(theme)

        # ---- Main window -------------------------------------------------
        with dpg.window(tag="main_win", no_title_bar=True, no_resize=True,
                        no_move=True, no_scrollbar=True, no_scroll_with_mouse=True,
                        width=self._vp_w, height=self._vp_h, pos=(0, 0)):
            with dpg.group(horizontal=True):
                # Left: map panel
                with dpg.child_window(tag="map_panel",
                                      width=self._map_w, height=-1,
                                      no_scrollbar=True, border=False):
                    dpg.add_image("map_tex", width=self._map_w, height=self._map_h,
                                  tag="map_img")

                # Right column
                with dpg.child_window(tag="right_col",
                                      width=self._right_w, height=-1,
                                      no_scrollbar=True, border=False):
                    # Camera feed
                    with dpg.child_window(tag="cam_panel",
                                          height=self._cam_h,
                                          no_scrollbar=True, border=False):
                        dpg.add_text("  Drone Camera", color=(160, 180, 255, 255))
                        dpg.add_image("cam_tex", width=self._right_w,
                                      height=self._cam_img_h, tag="cam_img")

                    dpg.add_separator()

                    # Match visualisation
                    with dpg.child_window(tag="match_panel",
                                          height=self._match_h,
                                          no_scrollbar=True, border=False):
                        dpg.add_text("  Visual Odometry Matches", color=(160, 255, 180, 255))
                        dpg.add_image("match_tex", width=self._right_w,
                                      height=self._match_img_h, tag="match_img")

                    dpg.add_separator()

                    # Info / controls — height=-1 fills remaining space
                    with dpg.child_window(tag="info_panel",
                                          height=-1,
                                          no_scrollbar=False, border=False):
                        with dpg.tab_bar():

                            # ── Simulation tab ────────────────────────────
                            with dpg.tab(label="Simulation"):
                                dpg.add_text("Controls & Info", color=(255, 220, 120, 255))
                                dpg.add_separator()

                                def on_matcher_change(s, data):
                                    with self.pose_lock:
                                        self.pose_state["switch_matcher"] = data
                                        self.pose_state["matcher_loading"] = True

                                dpg.add_combo(
                                    items=["SuperGlue", "LightGlue", "MatchAnything"],
                                    default_value="SuperGlue",
                                    callback=on_matcher_change,
                                    width=-1,
                                    tag="combo_matcher",
                                )
                                dpg.add_separator()

                                with dpg.group(horizontal=True):
                                    def on_start(): self.engine.is_running = True
                                    def on_stop():  self.engine.is_running = False
                                    def on_reset():
                                        self.engine.reset_state()
                                        with self.pose_lock:
                                            self.pose_state["reset_vo"]      = True
                                            self.pose_state["reset_planner"] = True
                                        upload_resized(
                                            "match_tex",
                                            np.zeros((self._match_img_h, self._right_w, 3), dtype=np.uint8),
                                            self._right_w, self._match_img_h,
                                        )

                                    dpg.add_button(label="  Start ", callback=on_start)
                                    dpg.add_button(label="  Stop ",  callback=on_stop)
                                    dpg.add_button(label="  Reset ", callback=on_reset)

                                dpg.add_text(tag="txt_runtime_status",
                                             default_value="Status: STOPPED",
                                             color=(255, 100, 100, 255))
                                dpg.add_separator()

                                dpg.add_text("Waypoints (Global Path):")
                                dpg.add_listbox(items=[], tag="list_waypoints", width=-1, num_items=5)

                                with dpg.group(horizontal=True):
                                    def on_wp_up():
                                        idx = dpg.get_value("list_waypoints")
                                        if not idx: return
                                        wps = self.engine.waypoints
                                        try:
                                            i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})"
                                                 for j, w in enumerate(wps)].index(idx)
                                            if i > 0:
                                                wps[i - 1], wps[i] = wps[i], wps[i - 1]
                                                self.engine.reorder_waypoints(wps)
                                        except ValueError:
                                            pass

                                    def on_wp_down():
                                        idx = dpg.get_value("list_waypoints")
                                        if not idx: return
                                        wps = self.engine.waypoints
                                        try:
                                            i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})"
                                                 for j, w in enumerate(wps)].index(idx)
                                            if i < len(wps) - 1:
                                                wps[i + 1], wps[i] = wps[i], wps[i + 1]
                                                self.engine.reorder_waypoints(wps)
                                        except ValueError:
                                            pass

                                    def on_wp_del():
                                        idx = dpg.get_value("list_waypoints")
                                        if not idx: return
                                        wps = self.engine.waypoints
                                        try:
                                            i = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})"
                                                 for j, w in enumerate(wps)].index(idx)
                                            self.engine.remove_waypoint(i)
                                        except ValueError:
                                            pass

                                    def on_wp_clear():
                                        self.engine.clear_waypoints()

                                    dpg.add_button(label="Move Up",   callback=on_wp_up)
                                    dpg.add_button(label="Move Down", callback=on_wp_down)
                                    dpg.add_button(label="Del",       callback=on_wp_del)
                                    dpg.add_button(label="Clear",     callback=on_wp_clear)

                                dpg.add_separator()
                                dpg.add_checkbox(label="SLAM estimate", tag="chk_slam", default_value=True)
                                dpg.add_checkbox(label="Path arc",      tag="chk_arc",  default_value=True)
                                dpg.add_checkbox(label="Dashed line",   tag="chk_dash", default_value=True)
                                dpg.add_separator()
                                dpg.add_text("Scroll=zoom  Drag=pan  RClick=Add Waypoint",
                                             color=(140, 140, 140, 180))
                                dpg.add_separator()
                                dpg.add_text("Drone",      color=(160, 160, 160, 200))
                                dpg.add_text("x=---  y=---  θ=---", tag="txt_pos")
                                dpg.add_text("SLAM",       color=(160, 160, 160, 200))
                                dpg.add_text("x=---  y=---  θ=---", tag="txt_slam")
                                dpg.add_text("Local Goal", color=(160, 160, 160, 200))
                                dpg.add_text("x=---  y=---",        tag="txt_goal")

                            # ── Benchmarks tab ────────────────────────────
                            with dpg.tab(label="Benchmarks"):
                                dpg.add_text("Matcher", color=(255, 220, 120, 255))
                                dpg.add_combo(
                                    items=["SuperGlue", "LightGlue", "MatchAnything"],
                                    default_value="SuperGlue",
                                    tag="bench_combo_matcher",
                                    width=-1,
                                )
                                dpg.add_separator()
                                dpg.add_text("Scenarios", color=(255, 220, 120, 255))

                                def _bench_start(key):
                                    matcher = dpg.get_value("bench_combo_matcher")
                                    dpg.set_value("bench_results", "")
                                    dpg.set_value("bench_saved",   "")
                                    dpg.set_value("bench_status",  "Status: Starting…")
                                    dpg.configure_item("bench_status",
                                                       color=(255, 200, 50, 255))
                                    self.bench_runner.start(key, matcher)

                                def _bench_all():
                                    matcher = dpg.get_value("bench_combo_matcher")
                                    dpg.set_value("bench_results", "")
                                    dpg.set_value("bench_saved",   "")
                                    dpg.set_value("bench_status",  "Status: Starting all…")
                                    dpg.configure_item("bench_status",
                                                       color=(255, 200, 50, 255))
                                    self.bench_runner.start_all(matcher)

                                with dpg.group(horizontal=True):
                                    dpg.add_button(label=" Straight ",
                                                   callback=lambda: _bench_start("straight"))
                                    dpg.add_button(label=" S-Curve  ",
                                                   callback=lambda: _bench_start("curved"))
                                    dpg.add_button(label=" Zigzag   ",
                                                   callback=lambda: _bench_start("zigzag"))

                                dpg.add_spacer(height=4)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label=" Run All ", callback=_bench_all)
                                    dpg.add_button(label="  Stop   ",
                                                   callback=lambda: self.bench_runner.stop())

                                dpg.add_separator()
                                dpg.add_text("Status: IDLE", tag="bench_status",
                                             color=(180, 180, 180, 255))
                                dpg.add_separator()
                                dpg.add_text("Results:", color=(160, 160, 160, 200))
                                dpg.add_input_text(
                                    tag="bench_results",
                                    multiline=True,
                                    readonly=True,
                                    width=-1,
                                    height=160,
                                    default_value="",
                                )
                                dpg.add_text("", tag="bench_saved",
                                             color=(100, 220, 100, 255))

        # ---- Input handlers ----------------------------------------------
        with dpg.handler_registry():
            def on_scroll(s, delta):
                mx, my = dpg.get_mouse_pos(local=False)
                if 0 <= mx <= self._map_w and 0 <= my <= self._map_h:
                    self.mv.zoom_at(mx, my, 1.12 if delta > 0 else 1.0 / 1.12)

            _is_dragging    = [False]
            _last_mouse_pos = [0.0, 0.0]

            def on_drag(s, data):
                if not dpg.is_item_hovered("map_panel"):
                    return
                if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                    mx, my = dpg.get_mouse_pos(local=False)
                    if not _is_dragging[0]:
                        _is_dragging[0]    = True
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

            def on_mouse_release(s, btn):
                if btn == dpg.mvMouseButton_Left:
                    _is_dragging[0] = False

            def on_click(s, btn):
                if btn == dpg.mvMouseButton_Right:
                    mx, my = dpg.get_mouse_pos(local=False)
                    if 0 <= mx <= self._map_w and 0 <= my <= self._map_h:
                        wx, wy = self.mv.s2w(mx, my)
                        self.engine.add_waypoint(wx, wy)
                        print(f"Added Waypoint → ({wx:.0f}, {wy:.0f})")

            dpg.add_mouse_wheel_handler(callback=on_scroll)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=on_drag)
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

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def run_frame(self):
        # --- Detect viewport resize (use client dims — excludes OS title bar) ---
        new_w = dpg.get_viewport_client_width()
        new_h = dpg.get_viewport_client_height()
        if new_w != self._vp_w or new_h != self._vp_h:
            self._vp_w = new_w
            self._vp_h = new_h
            self._compute_layout()
            self.mv.resize(self._map_w, self._map_h)
            self._recreate_textures()
            self._resize_layout()

        keys  = {v: dpg.is_key_down(k) for k, v in self.key_map.items()}
        state = self.engine.tick(keys)

        # --- Drain benchmark event queue ---
        try:
            while True:
                kind, payload = self.bench_event_queue.get_nowait()
                if kind == "status":
                    dpg.set_value("bench_status", f"Status: {payload}")
                    done_words = ("complete", "stopped", "error", "idle")
                    done = any(w in payload.lower() for w in done_words)
                    dpg.configure_item("bench_status",
                                       color=(100, 220, 100, 255) if done else (255, 200, 50, 255))
                elif kind == "results_txt":
                    dpg.set_value("bench_results", payload)
                elif kind == "saved":
                    dpg.set_value("bench_saved", f"Saved: {os.path.basename(payload)}")
        except queue.Empty:
            pass

        # --- Map update ---
        show_slam = dpg.get_value("chk_slam")
        show_arc  = dpg.get_value("chk_arc")
        show_dash = dpg.get_value("chk_dash")
        map_img   = render_map_frame(self.engine.background, self.mv, state,
                                     show_slam, show_arc, show_dash)
        upload_map(map_img)

        # Camera feed
        with self.engine._cam_lock:
            cam_pair    = self.engine._latest_cam
            cam_changed = cam_pair is not self.engine._prev_cam_uploaded
            if cam_changed:
                self.engine._prev_cam_uploaded = cam_pair
        if cam_changed and cam_pair is not None:
            upload_resized("cam_tex", cam_pair[0], self._right_w, self._cam_img_h)

        # Match visualisation
        new_match = False
        try:
            while True:
                _, vis = self.match_queue.get_nowait()
                self.latest_match[0] = vis
                new_match = True
        except queue.Empty:
            pass
        if new_match and self.latest_match[0] is not None:
            upload_resized("match_tex", self.latest_match[0], self._right_w, self._match_img_h)

        # --- Status / telemetry text ---
        cmd = state.cmd
        dpg.set_value("txt_pos",  f"x={state.pos_x:.0f}  y={state.pos_y:.0f}  θ={state.angle:.1f}°")
        dpg.set_value("txt_slam",
                      f"x={cmd.est_x:.0f}  y={cmd.est_y:.0f}  θ={cmd.est_angle:.1f}°"
                      if (cmd.est_x > 0 or cmd.est_y > 0) else "x=---  y=---  θ=---°")
        dpg.set_value("txt_goal", f"x={cmd.goal_x:.0f}  y={cmd.goal_y:.0f}")

        # Matcher loading takes priority in the status line
        with self.pose_lock:
            matcher_loading = self.pose_state.get("matcher_loading", False)

        if matcher_loading:
            dpg.set_value("txt_runtime_status", "Status: LOADING MODEL…")
            dpg.configure_item("txt_runtime_status", color=(255, 200, 50, 255))
        else:
            status_str = "Status: RUNNING" if state.is_running else "Status: STOPPED"
            status_col = (100, 255, 100, 255) if state.is_running else (255, 100, 100, 255)
            dpg.set_value("txt_runtime_status", status_str)
            dpg.configure_item("txt_runtime_status", color=status_col)

        wp_strs = [f"WP {j}: ({w[0]:.0f}, {w[1]:.0f})" for j, w in enumerate(state.waypoints)]
        dpg.configure_item("list_waypoints", items=wp_strs)
        dpg.render_dearpygui_frame()

    def teardown(self):
        dpg.destroy_context()

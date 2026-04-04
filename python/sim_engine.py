"""
sim_engine.py
-------------
Drone physics + frame generation. No TCP sockets, no rendering.

Responsibilities:
  - Maintain drone position / rotation.
  - Apply keyboard inputs and autopilot commands.
  - Crop + rotate the background map image to produce the drone-cam frame.
  - Push cam frames (numpy BGR) into `frame_queue` for the SLAM thread.
  - Expose DroneState for the UI to read each tick.

Nothing in here touches the display or the network.
"""

import math
import os
import queue
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── Constants (kept identical to main.cpp) ────────────────────────────────────
DRONE_CAM_W = 960
DRONE_CAM_H = 540
VIEW_W      = 1920   # logical view size for camera clamping
VIEW_H      = 1080

N_PATH      = 50     # must match path_planner.py

SPEED_DEFAULT = 3.5  # px / tick when keyboard held
SPEED_BOOST   = 2.5  # added per Space press

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR   = os.path.join(PROJECT_ROOT, "assets")


# ── Shared state ──────────────────────────────────────────────────────────────

@dataclass
class AutopilotCmd:
    speed:      float = 0.0
    turn_angle: float = 0.0
    est_x:      float = -1.0
    est_y:      float = -1.0
    # Next local sub-goal
    goal_x:     float = 300.0
    goal_y:     float = 700.0
    # Short local path arc prediction (from path planner)
    path_x:     list  = field(default_factory=lambda: [0.0] * N_PATH)
    path_y:     list  = field(default_factory=lambda: [0.0] * N_PATH)
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


# ── Sim Engine ────────────────────────────────────────────────────────────────

class SimEngine:
    """
    Call `tick(keys, dt_override)` every frame from the render thread.
    SLAM thread reads from `frame_queue`.
    Path planner writes to `autopilot_cmd` (AutopilotCmd protected by cmd_lock).
    """

    def __init__(self, bg_image_name: str = "dag.jpg",
                 frame_queue: queue.Queue | None = None):
        # Background image (colour, full res)
        bg_path = os.path.join(ASSETS_DIR, bg_image_name)
        self._bg = cv2.imread(bg_path)
        if self._bg is None:
            raise FileNotFoundError(f"Background image not found: {bg_path}")

        self.bg_h, self.bg_w = self._bg.shape[:2]

        self._init_x = self.bg_w / 2.0
        self._init_y = self.bg_h / 2.0

        # Drone state
        self._pos_x:  float = self._init_x
        self._pos_y:  float = self._init_y
        self._angle:  float = 0.0          # degrees
        self._speed_y: float = SPEED_DEFAULT

        # UI State
        self.is_running: bool = False
        self.waypoints: list[tuple[float, float]] = []

        # Keyboard-driven movement accumulator (set by tick callers)
        self._move_y: float = 0.0

        # Autopilot
        self.cmd_lock      = threading.Lock()
        self.autopilot_cmd = AutopilotCmd()
        self.autopilot_active = False

        # Outgoing cam frames for SLAM
        self.frame_queue: queue.Queue = frame_queue or queue.Queue(maxsize=4)

        # Latest cam frame for UI display (not consumed by SLAM)
        self._cam_lock = threading.Lock()
        self._latest_cam: tuple | None = None  # (bgr, angle)
        self._prev_cam_uploaded = None         # sentinel for change detection

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def background(self) -> np.ndarray:
        return self._bg

    def reset_state(self) -> None:
        """Reset drone to initial position, stop simulation."""
        self.is_running = False
        self._pos_x = self._init_x
        self._pos_y = self._init_y
        self._angle = 0.0
        self._move_y = 0.0
        with self.cmd_lock:
            self.autopilot_active = False

    def add_waypoint(self, world_x: float, world_y: float) -> None:
        """Called by UI to append a waypoint."""
        self.waypoints.append((float(world_x), float(world_y)))

    def clear_waypoints(self) -> None:
        self.waypoints.clear()
        
    def reorder_waypoints(self, new_order: list[tuple[float, float]]) -> None:
        """Override waypoints from the UI."""
        self.waypoints = new_order

    def remove_waypoint(self, index: int) -> None:
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)

    def apply_autopilot_cmd(self, cmd: AutopilotCmd, active: bool) -> None:
        """Called by path planner thread to push a new command."""
        with self.cmd_lock:
            self.autopilot_cmd = cmd
            self.autopilot_active = active

    def tick(self, keys: dict) -> DroneState:
        """
        Advance one simulation step.

        `keys` is a dict of booleans with keys:
            'left', 'right', 'up', 'down', 'space'

        Returns a DroneState snapshot for the UI.
        Also pushes a drone-cam frame onto self.frame_queue (non-blocking drop if full).
        """
        manual = False

        # Keyboard
        if keys.get("left"):
            self._angle -= 0.2
            manual = True
        if keys.get("right"):
            self._angle += 0.2
            manual = True
        if keys.get("down"):
            self._move_y = self._speed_y
            manual = True
        if keys.get("up"):
            self._move_y = -self._speed_y
            manual = True
        if keys.get("space"):
            self._speed_y += SPEED_BOOST

        # Autopilot (overridden by keyboard)
        with self.cmd_lock:
            cmd = AutopilotCmd(**self.autopilot_cmd.__dict__)
            active = self.autopilot_active

        if self.is_running:
            if not manual and active:
                # Planner already zero-ed speed when SLAM estimate is within GOAL_RADIUS
                self._angle  += cmd.turn_angle
                self._move_y  = cmd.speed

            # Movement
            rad = math.radians(self._angle)
            dx  =  math.sin(rad) * self._move_y
            dy  = -math.cos(rad) * self._move_y

            # Boundary clamping (half-drone-size = 15 px horiz, 10 px vert)
            nx = self._pos_x + dx
            ny = self._pos_y + dy
            hw, hh = 10.0, 15.0
            if hw <= nx <= self.bg_w - hw:
                self._pos_x = nx
            if hh <= ny <= self.bg_h - hh:
                self._pos_y = ny
        else:
            # When paused, no movement overrides
            pass

        self._move_y = 0.0

        # Generate drone-cam frame
        cam_frame = self._render_cam_frame()

        # Push to SLAM queue — always replace with the latest frame.
        # Drain the stale frame first so SLAM wakes up to the freshest image,
        # not a frame that was put right after the previous get() ~500ms ago.
        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(cam_frame)
        except queue.Full:
            pass

        # Store latest for UI display (UI reads this directly, no queue contention)
        with self._cam_lock:
            self._latest_cam = (cam_frame.copy(), self._angle)

        return DroneState(
            pos_x=self._pos_x,
            pos_y=self._pos_y,
            angle=self._angle,
            cmd=cmd,
            bg_w=self.bg_w,
            bg_h=self.bg_h,
            is_running=self.is_running,
            waypoints=list(self.waypoints),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _render_cam_frame(self) -> np.ndarray:
        """
        Produce a DRONE_CAM_W × DRONE_CAM_H BGR frame centred on the drone,
        rotated by -angle (same convention as the SFML drone-cam view).
        Equivalent to what the C++ window2 was capturing.
        """
        cx, cy = self._pos_x, self._pos_y
        angle  = self._angle

        # Build affine: rotate bg around drone position, then crop.
        # Step 1: translate drone to origin
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0,  1 ]], dtype=np.float64)

        # Step 2: rotate by -angle (counter-clockwise in image coords)
        rad = math.radians(-angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        R = np.array([[ cos_a, -sin_a, 0],
                      [ sin_a,  cos_a, 0],
                      [     0,      0, 1]], dtype=np.float64)

        # Step 3: translate origin to cam centre
        half_w = DRONE_CAM_W / 2.0
        half_h = DRONE_CAM_H / 2.0
        T2 = np.array([[1, 0, half_w],
                       [0, 1, half_h],
                       [0, 0,      1]], dtype=np.float64)

        M_full = (T2 @ R @ T1)[:2, :]  # 2×3 affine

        cam = cv2.warpAffine(
            self._bg, M_full,
            (DRONE_CAM_W, DRONE_CAM_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cam

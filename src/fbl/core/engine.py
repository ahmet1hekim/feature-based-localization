"""
Drone physics + camera frame generation. No rendering to screen directly here.
"""

import math
import os
import queue
import threading
import cv2
import numpy as np

from .state import DroneState, AutopilotCmd

DRONE_CAM_W = 960
DRONE_CAM_H = 540
VIEW_W      = 1920
VIEW_H      = 1080

N_PATH      = 50

SPEED_DEFAULT = 3.5
SPEED_BOOST   = 2.5

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ASSETS_DIR   = os.path.join(PROJECT_ROOT, "assets")

class SimEngine:
    def __init__(self, bg_image_name: str = "dag.jpg", frame_queue: queue.Queue | None = None):
        bg_path = os.path.join(ASSETS_DIR, bg_image_name)
        self._bg = cv2.imread(bg_path)
        if self._bg is None:
            raise FileNotFoundError(f"Background image not found: {bg_path}")

        self.bg_h, self.bg_w = self._bg.shape[:2]

        self._init_x = self.bg_w / 2.0
        self._init_y = self.bg_h / 2.0

        self._pos_x:  float = self._init_x
        self._pos_y:  float = self._init_y
        self._angle:  float = 0.0
        self._speed_y: float = SPEED_DEFAULT

        self.is_running: bool = False
        self.waypoints: list[tuple[float, float]] = []

        self._move_y: float = 0.0

        self.cmd_lock      = threading.Lock()
        self.autopilot_cmd = AutopilotCmd()
        self.autopilot_active = False

        self.frame_queue: queue.Queue = frame_queue or queue.Queue(maxsize=4)

        self._cam_lock = threading.Lock()
        self._latest_cam: tuple | None = None
        self._prev_cam_uploaded = None

    @property
    def background(self) -> np.ndarray:
        return self._bg

    def reset_state(self) -> None:
        self.is_running = False
        self._pos_x = self._init_x
        self._pos_y = self._init_y
        self._angle = 0.0
        self._move_y = 0.0
        with self.cmd_lock:
            self.autopilot_cmd = AutopilotCmd()
            self.autopilot_active = False

    def add_waypoint(self, world_x: float, world_y: float) -> None:
        self.waypoints.append((float(world_x), float(world_y)))

    def clear_waypoints(self) -> None:
        self.waypoints.clear()
        
    def reorder_waypoints(self, new_order: list[tuple[float, float]]) -> None:
        self.waypoints = new_order

    def remove_waypoint(self, index: int) -> None:
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)

    def apply_autopilot_cmd(self, cmd: AutopilotCmd, active: bool) -> None:
        with self.cmd_lock:
            self.autopilot_cmd = cmd
            self.autopilot_active = active

    def tick(self, keys: dict) -> DroneState:
        manual = False

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

        with self.cmd_lock:
            cmd = AutopilotCmd(**self.autopilot_cmd.__dict__)
            active = self.autopilot_active

        if self.is_running:
            if not manual and active:
                self._angle  += cmd.turn_angle
                self._move_y  = cmd.speed

            rad = math.radians(self._angle)
            dx  =  math.sin(rad) * self._move_y
            dy  = -math.cos(rad) * self._move_y

            nx = self._pos_x + dx
            ny = self._pos_y + dy
            hw, hh = 10.0, 15.0
            if hw <= nx <= self.bg_w - hw:
                self._pos_x = nx
            if hh <= ny <= self.bg_h - hh:
                self._pos_y = ny

        self._move_y = 0.0

        cam_frame = self._render_cam_frame()

        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(cam_frame)
        except queue.Full:
            pass

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

    def _render_cam_frame(self) -> np.ndarray:
        cx, cy = self._pos_x, self._pos_y
        angle  = self._angle

        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        rad = math.radians(-angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)
        T2 = np.array([[1, 0, DRONE_CAM_W/2.0], [0, 1, DRONE_CAM_H/2.0], [0, 0, 1]], dtype=np.float64)

        M_full = (T2 @ R @ T1)[:2, :]

        cam = cv2.warpAffine(
            self._bg, M_full, (DRONE_CAM_W, DRONE_CAM_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cam

import math
import queue
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .matchers.base import BaseMatcher

def normalize_deg(a: float) -> float:
    return (a + 360.0) % 360.0

class VoNode(threading.Thread):
    def __init__(
        self,
        matcher: BaseMatcher,
        frame_queue: queue.Queue,
        pose_state: dict,
        pose_lock: threading.Lock,
        match_queue: Optional[queue.Queue] = None,
        stop_event: Optional[threading.Event] = None,
        start_x: float = 640.0,
        start_y: float = 360.0
    ):
        super().__init__(daemon=True, name="VO")
        self.matcher = matcher
        self.frame_queue = frame_queue
        self.pose_state = pose_state
        self.pose_lock = pose_lock
        self.match_queue = match_queue
        self.stop_event = stop_event
        self.start_x = start_x
        self.start_y = start_y

    def run(self) -> None:
        print("[VO] Starting Visual Odometry thread...")

        locked_theta = 0.0
        locked_x     = self.start_x
        locked_y     = self.start_y

        past_frame_gray  = None
        _frame_ts: deque = deque(maxlen=60)  # rolling window for FPS

        while self.stop_event is None or not self.stop_event.is_set():
            switch_matcher = None
            with self.pose_lock:
                do_reset = self.pose_state.get("reset_vo", False)
                if do_reset:
                    self.pose_state["reset_vo"] = False
                    
                switch_matcher = self.pose_state.get("switch_matcher", None)
                if switch_matcher is not None:
                    self.pose_state["switch_matcher"] = None

            if switch_matcher is not None:
                print(f"[VO] Hot-swapping Matcher Strategy to: {switch_matcher}...")
                try:
                    if switch_matcher == "SuperGlue":
                        from fbl.vo.matchers.superglue import SuperGlueMatcher
                        self.matcher = SuperGlueMatcher()
                    elif switch_matcher == "LightGlue":
                        from fbl.vo.matchers.lightglue import LightGlueMatcher
                        self.matcher = LightGlueMatcher()
                    elif switch_matcher == "MatchAnything":
                        from fbl.vo.matchers.matchanything import MatchAnythingMatcher
                        self.matcher = MatchAnythingMatcher()
                    print(f"[VO] Matcher swapped! Resuming tracking from ({locked_x:.1f}, {locked_y:.1f}) seamlessly.")
                    with self.pose_lock:
                        self.pose_state["current_matcher"] = switch_matcher
                except Exception as e:
                    print(f"[VO] Failed to load {switch_matcher} Matcher: {e}")
                    print("[VO] Rolling back to previous matcher to prevent crash!")
                finally:
                    with self.pose_lock:
                        self.pose_state["matcher_loading"] = False

            if do_reset:
                locked_x = self.start_x
                locked_y = self.start_y
                locked_theta = 0.0
                past_frame_gray = None
                with self.pose_lock:
                    self.pose_state["x"] = self.start_x
                    self.pose_state["y"] = self.start_y
                    self.pose_state["theta"] = 0.0
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: break
                continue

            try:
                frame_bgr = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            h, w = frame_bgr.shape[:2]
            frame_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if past_frame_gray is None:
                past_frame_gray = frame_gray.copy()
                continue

            # 1) ROTATION ESTIMATION
            pts0, pts1, rot_vis = self.matcher.match(past_frame_gray, frame_gray)

            rot_deg = 0.0
            dx_img  = 0.0
            dy_img  = 0.0

            if len(pts0) >= 8:
                pts0_a = np.float32(pts0)
                pts1_a = np.float32(pts1)
                M, inliers = cv2.estimateAffinePartial2D(
                    pts0_a, pts1_a,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=2000,
                    confidence=0.99,
                )
                if M is not None and inliers is not None and int(inliers.sum()) >= 6:
                    cx, cy = w * 0.5, h * 0.5
                    M3    = np.vstack([M, [0, 0, 1]])
                    T_neg = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
                    T_pos = np.array([[1,0, cx],[0,1, cy],[0,0,1]])
                    M_c   = (T_pos @ M3 @ T_neg)[:2, :]
                    rot_deg = math.degrees(math.atan2(M_c[1,0], M_c[0,0]))
                    locked_theta -= rot_deg
                    locked_theta  = normalize_deg(locked_theta)

                    # 2) TRANSLATION — de-rotate RANSAC inlier matches analytically.
                    # This is equivalent to de-rotating the entire image and re-matching,
                    # but avoids calling the matcher a second time on a modified image.
                    inlier_mask = inliers.ravel() == 1
                    p0_in = pts0_a[inlier_mask]
                    p1_in = pts1_a[inlier_mask]

                    rad_r = math.radians(-rot_deg)
                    cos_r, sin_r = math.cos(rad_r), math.sin(rad_r)
                    p1_c = p1_in - np.array([cx, cy], dtype=np.float32)
                    p1_derot = np.column_stack([
                        cos_r * p1_c[:, 0] - sin_r * p1_c[:, 1] + cx,
                        sin_r * p1_c[:, 0] + cos_r * p1_c[:, 1] + cy,
                    ])
                    disp   = p1_derot - p0_in
                    dx_img = float(np.median(disp[:, 0]))
                    dy_img = float(np.median(disp[:, 1]))

            th = math.radians(locked_theta)
            locked_x -= math.cos(th) * dx_img - math.sin(th) * dy_img
            locked_y -= math.sin(th) * dx_img + math.cos(th) * dy_img

            if self.match_queue is not None and rot_vis is not None:
                try:
                    self.match_queue.put_nowait(("rot", rot_vis))
                except queue.Full:
                    pass


            past_frame_gray = frame_gray.copy()

            print(f"[VO] θ={locked_theta:.2f}°  x={locked_x:.2f}  y={locked_y:.2f}")

            _frame_ts.append(time.perf_counter())
            if len(_frame_ts) >= 2:
                vo_fps = (len(_frame_ts) - 1) / (_frame_ts[-1] - _frame_ts[0])
            else:
                vo_fps = 0.0

            with self.pose_lock:
                self.pose_state["x"]      = locked_x
                self.pose_state["y"]      = locked_y
                self.pose_state["theta"]  = locked_theta
                self.pose_state["vo_fps"] = vo_fps

import math
import queue
import threading
from typing import Optional

import cv2
import numpy as np

from .matchers.base import BaseMatcher

def normalize_deg(a: float) -> float:
    return (a + 360.0) % 360.0

def angle_error_deg(pred: float, real: float) -> float:
    return (pred - real + 180.0) % 360.0 - 180.0

def preprocess_image(img, center_x, center_y, target_w, target_h, angle=0):
    h, w = img.shape[:2]

    if angle != 0:
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
        cos = abs(rot_mat[0, 0])
        sin = abs(rot_mat[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rot_mat[0, 2] += (new_w / 2) - center_x
        rot_mat[1, 2] += (new_h / 2) - center_y
        rotated = cv2.warpAffine(
            img, rot_mat, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        center_x = new_w // 2
        center_y = new_h // 2
        h, w = rotated.shape[:2]
    else:
        rotated = img

    half_w, half_h = target_w // 2, target_h // 2
    x1, y1, x2, y2 = center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h
    ix1, iy1 = max(x1, 0), max(y1, 0)
    ix2, iy2 = min(x2, w),  min(y2, h)
    cropped = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    sx, sy   = ix1 - x1, iy1 - y1
    vw, vh   = ix2 - ix1, iy2 - iy1
    if vw > 0 and vh > 0:
        cropped[sy:sy+vh, sx:sx+vw] = rotated[iy1:iy2, ix1:ix2]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)


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
        past_frame_color = None

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
                except Exception as e:
                    print(f"[VO] Failed to load {switch_matcher} Matcher: {e}")
                    print("[VO] Rolling back to previous matcher to prevent crash!")

            if do_reset:
                locked_x = self.start_x
                locked_y = self.start_y
                locked_theta = 0.0
                past_frame_gray = None
                past_frame_color = None
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
            frame_color = frame_bgr
            frame_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if past_frame_gray is None:
                past_frame_gray  = frame_gray.copy()
                past_frame_color = frame_color.copy()
                continue

            # 1) ROTATION ESTIMATION (Visual Odometry)
            pts0, pts1, rot_vis = self.matcher.match(past_frame_gray, frame_gray)

            rot_deg = 0.0
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
                    M3 = np.vstack([M, [0, 0, 1]])
                    T_neg = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
                    T_pos = np.array([[1,0, cx],[0,1, cy],[0,0,1]])
                    M_c   = (T_pos @ M3 @ T_neg)[:2, :]
                    rot_deg = math.degrees(math.atan2(M_c[1,0], M_c[0,0]))
                    locked_theta -= rot_deg
                    locked_theta  = normalize_deg(locked_theta)

            if self.match_queue is not None and rot_vis is not None:
                try:
                    self.match_queue.put_nowait(("rot", rot_vis))
                except queue.Full:
                    pass

            # 2) TRANSLATION ESTIMATION (Visual Odometry)
            cx_i, cy_i = w // 2, h // 2
            curr_aligned = preprocess_image(frame_color, cx_i, cy_i, w, h, angle=-rot_deg)
            prev_aligned = past_frame_gray

            pts0_t, pts1_t, trans_vis = self.matcher.match(prev_aligned, curr_aligned)

            if len(pts0_t) >= 8:
                pts0_t = np.float32(pts0_t)
                pts1_t = np.float32(pts1_t)
                v       = pts1_t - pts0_t
                dx_img  = np.median(v[:, 0])
                dy_img  = np.median(v[:, 1])
                th      = math.radians(locked_theta)
                locked_x -= math.cos(th) * dx_img - math.sin(th) * dy_img
                locked_y -= math.sin(th) * dx_img + math.cos(th) * dy_img

            if self.match_queue is not None and trans_vis is not None:
                try:
                    self.match_queue.put_nowait(("trans", trans_vis))
                except queue.Full:
                    pass

            past_frame_gray  = frame_gray.copy()
            past_frame_color = frame_color.copy()

            print(f"[VO] θ={locked_theta:.2f}°  x={locked_x:.2f}  y={locked_y:.2f}")

            with self.pose_lock:
                self.pose_state["x"]     = locked_x
                self.pose_state["y"]     = locked_y
                self.pose_state["theta"] = locked_theta

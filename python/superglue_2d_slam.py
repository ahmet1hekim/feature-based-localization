"""
superglue_2d_slam.py  (refactored — no TCP sockets)
-----------------------------------------------------------
Reads drone-cam frames from an in-process queue (put by SimEngine).
Publishes pose to a shared dict (pose_state) watched by path_planner.

Public API (used by main.py):
    run_slam_thread(frame_queue, pose_state, pose_lock,
                    match_queue=None, stop_event=None)

`frame_queue`  : queue.Queue of cam_frame_bgr: np.ndarray
`pose_state`   : dict  {"x": float, "y": float, "theta": float}
`pose_lock`    : threading.Lock protecting pose_state
`match_queue`  : optional queue.Queue for (rot_vis, trans_vis) debug images
`stop_event`   : optional threading.Event; set it to stop the thread
"""

import math
import os
import queue
import threading
from typing import Optional

import cv2
import numpy as np
import torch
from externals.SuperGluePretrainedNetwork.models.matching import Matching

# ── Config ────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 1024},
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    },
}

# ── Helpers (unchanged logic) ─────────────────────────────────────────────────

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


def to_superpoint_tensor(gray_img, device):
    t = torch.from_numpy(gray_img).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(device)


def draw_superglue_matches(img0, img1, kpts0, kpts1, matches0, conf, conf_thresh=0.2):
    img0_vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
    h = max(img0_vis.shape[0], img1_vis.shape[0])
    w0 = img0_vis.shape[1]
    vis = np.zeros((h, img0_vis.shape[1] + img1_vis.shape[1], 3), dtype=np.uint8)
    vis[:img0_vis.shape[0], :img0_vis.shape[1]] = img0_vis
    vis[:img1_vis.shape[0], img0_vis.shape[1]:]  = img1_vis
    for i, j in enumerate(matches0):
        if j < 0 or conf[i] < conf_thresh:
            continue
        pt0 = tuple(map(int, kpts0[i]))
        pt1 = (int(kpts1[j][0]) + w0, int(kpts1[j][1]))
        cv2.circle(vis, pt0, 3, (0, 255, 0), -1)
        cv2.circle(vis, pt1,  3, (0, 255, 0), -1)
        cv2.line(vis, pt0, pt1, (0, 255, 0), 1)
    return vis


# ── Main thread function ──────────────────────────────────────────────────────

def run_slam_thread(
    frame_queue:  queue.Queue,
    pose_state:   dict,
    pose_lock:    threading.Lock,
    match_queue:  Optional[queue.Queue] = None,
    stop_event:   Optional[threading.Event] = None,
    start_x:      float = 640.0,
    start_y:      float = 360.0,
) -> None:
    """
    Blocking function — call inside a daemon thread from main.py.
    Loads the SuperGlue model, then loops reading frames from frame_queue.
    start_x/start_y should match the drone's initial world position.
    """
    print("[slam] Loading SuperGlue model...")
    matching = Matching(config).eval().to(device)
    print(f"[slam] Model ready on {device}")

    locked_theta = 0.0
    locked_x     = start_x
    locked_y     = start_y

    past_frame_gray  = None
    past_frame_color = None
    rot_deg = 0.0   # keep in scope for translation step

    while stop_event is None or not stop_event.is_set():
        with pose_lock:
            do_reset = pose_state.get("reset_slam", False)
            if do_reset:
                pose_state["reset_slam"] = False

        if do_reset:
            locked_x = start_x
            locked_y = start_y
            locked_theta = 0.0
            past_frame_gray = None
            past_frame_color = None
            with pose_lock:
                pose_state["x"] = start_x
                pose_state["y"] = start_y
                pose_state["theta"] = 0.0
            # Flush stale frames
            while not frame_queue.empty():
                try: frame_queue.get_nowait()
                except queue.Empty: break
            continue

        # Drain the queue — always process only the LATEST frame.
        # SuperGlue takes ~200-500 ms; the sim generates frames at ~60 fps.
        # Without draining, SLAM would lag hundreds of frames behind reality.
        try:
            frame_bgr = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        h, w = frame_bgr.shape[:2]
        frame_color = frame_bgr
        frame_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if past_frame_gray is None:
            past_frame_gray  = frame_gray.copy()
            past_frame_color = frame_color.copy()
            continue

        # ── 1) ROTATION ESTIMATION ────────────────────────────────────────────
        with torch.no_grad():
            pred = matching({
                "image0": to_superpoint_tensor(past_frame_gray, device),
                "image1": to_superpoint_tensor(frame_gray, device),
            })
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        matches = pred["matches0"]
        kpts0   = pred["keypoints0"]
        kpts1   = pred["keypoints1"]
        conf    = pred["matching_scores0"]

        pts0, pts1 = [], []
        for i, m in enumerate(matches):
            if m >= 0 and conf[i] > 0.5:
                pts0.append(kpts0[i])
                pts1.append(kpts1[m])

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

        # Optional: push rotation match vis
        if match_queue is not None:
            rot_vis = draw_superglue_matches(
                past_frame_gray, frame_gray, kpts0, kpts1, matches, conf, conf_thresh=0.5)
            try:
                match_queue.put_nowait(("rot", rot_vis))
            except queue.Full:
                pass

        # ── 2) TRANSLATION ────────────────────────────────────────────────────
        cx_i, cy_i = w // 2, h // 2
        curr_aligned = preprocess_image(frame_color, cx_i, cy_i, w, h, angle=-rot_deg)
        prev_aligned = past_frame_gray

        with torch.no_grad():
            pred_t = matching({
                "image0": to_superpoint_tensor(prev_aligned, device),
                "image1": to_superpoint_tensor(curr_aligned, device),
            })
            pred_t = {k: v[0].cpu().numpy() for k, v in pred_t.items()}

        matches_t = pred_t["matches0"]
        kpts0_t   = pred_t["keypoints0"]
        kpts1_t   = pred_t["keypoints1"]
        conf_t    = pred_t["matching_scores0"]

        pts0_t, pts1_t = [], []
        for i, m in enumerate(matches_t):
            if m >= 0 and conf_t[i] > 0.8:
                pts0_t.append(kpts0_t[i])
                pts1_t.append(kpts1_t[m])

        if len(pts0_t) >= 8:
            pts0_t = np.float32(pts0_t)
            pts1_t = np.float32(pts1_t)
            v       = pts1_t - pts0_t
            dx_img  = np.median(v[:, 0])
            dy_img  = np.median(v[:, 1])
            th      = math.radians(locked_theta)
            locked_x -= math.cos(th) * dx_img - math.sin(th) * dy_img
            locked_y -= math.sin(th) * dx_img + math.cos(th) * dy_img

        # Optional: push translation match vis
        if match_queue is not None:
            trans_vis = draw_superglue_matches(
                prev_aligned, curr_aligned,
                kpts0_t, kpts1_t, matches_t, conf_t, conf_thresh=0.8)
            try:
                match_queue.put_nowait(("trans", trans_vis))
            except queue.Full:
                pass

        past_frame_gray  = frame_gray.copy()
        past_frame_color = frame_color.copy()

        print(f"[slam] θ={locked_theta:.2f}°  x={locked_x:.2f}  y={locked_y:.2f}")

        with pose_lock:
            pose_state["x"]     = locked_x
            pose_state["y"]     = locked_y
            pose_state["theta"] = locked_theta


# ── Keep the old __main__ entry for standalone use (now no-ops without SimEngine) ──

if __name__ == "__main__":
    print("[slam] Run via python/main.py — this module is no longer standalone.")

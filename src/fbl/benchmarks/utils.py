"""Shared utilities for all benchmark tests."""
import math
import queue

import cv2
import numpy as np

DRONE_CAM_W = 960
DRONE_CAM_H = 540


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def make_engine(bg_image_name: str = "dag.jpg"):
    from fbl.core.engine import SimEngine
    engine = SimEngine(bg_image_name=bg_image_name)
    engine.is_running = True
    return engine


def reset_engine(engine):
    engine.reset_state()
    engine.is_running = True


def render_gray(engine) -> np.ndarray:
    frame = engine._render_cam_frame()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Waypoint computation — derived from actual background dimensions
# ---------------------------------------------------------------------------

def compute_safe_waypoints(engine) -> dict:
    """
    Derive all scenario waypoints from the loaded background dimensions.

    Routes go horizontally (eastward) to exploit landscape aspect ratios.
    The drone starts at image centre; sx steps move it right, sy offsets
    move it up/down for curves and zigzags.
    """
    w, h   = engine.bg_w, engine.bg_h
    cx, cy = engine._init_x, engine._init_y

    sx = w * 0.13    # horizontal step  (~13 % of width  per waypoint)
    sy = h * 0.15    # vertical offset  (~15 % of height for curves)
    mx = w * 0.10    # x safety margin
    my = h * 0.08    # y safety margin

    def safe(x, y):
        return (
            float(max(mx, min(w - mx, x))),
            float(max(my, min(h - my, y))),
        )

    return {
        # ---- VO accuracy path: straight then slight dip ------------------
        "vo_path": [
            safe(cx + sx,       cy),
            safe(cx + 2 * sx,   cy - sy),
            safe(cx + 3 * sx,   cy),
        ],
        # ---- Straight: three collinear horizontal waypoints --------------
        "straight": [
            safe(cx + sx,       cy),
            safe(cx + 2 * sx,   cy),
            safe(cx + 3 * sx,   cy),
        ],
        # ---- S-curve: alternating vertical offsets -----------------------
        "curved": [
            safe(cx + sx,       cy - sy),
            safe(cx + 2 * sx,   cy + sy),
            safe(cx + 3 * sx,   cy),
        ],
        # ---- Zigzag: large vertical swings (stress test) -----------------
        "zigzag": [
            safe(cx + sx,       cy - 1.8 * sy),
            safe(cx + 2 * sx,   cy + 1.8 * sy),
            safe(cx + 3 * sx,   cy),
        ],
    }


# ---------------------------------------------------------------------------
# Frame-pair generation
# ---------------------------------------------------------------------------

def generate_frame_pairs(engine, n_pairs: int = 100, speed: float = 3.5) -> list:
    """
    Drive the drone along a gently varying path and collect n_pairs consecutive
    grayscale frame pairs from the simulation output.
    """
    from fbl.core.state import AutopilotCmd

    warm_up = 15
    for _ in range(warm_up):
        engine.apply_autopilot_cmd(AutopilotCmd(speed=speed, turn_angle=0.0), True)
        engine.tick({})

    turn_schedule = [0.0] * 30 + [0.08] * 20 + [-0.08] * 20 + [0.0] * 30

    pairs = []
    prev_gray = render_gray(engine)

    for i in range(n_pairs):
        turn = turn_schedule[i % len(turn_schedule)]
        engine.apply_autopilot_cmd(AutopilotCmd(speed=speed, turn_angle=turn), True)
        engine.tick({})
        curr_gray = render_gray(engine)
        pairs.append((prev_gray.copy(), curr_gray.copy()))
        prev_gray = curr_gray.copy()

    return pairs


# ---------------------------------------------------------------------------
# RANSAC helper
# ---------------------------------------------------------------------------

def run_ransac(pts0: np.ndarray, pts1: np.ndarray):
    if len(pts0) < 8:
        return 0, None
    M, inliers = cv2.estimateAffinePartial2D(
        np.float32(pts0), np.float32(pts1),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if inliers is None:
        return 0, None
    return int(inliers.sum()), M


# ---------------------------------------------------------------------------
# Single VO step — mirrors VoNode.run() logic without threading
# ---------------------------------------------------------------------------

def vo_step(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    locked_x: float,
    locked_y: float,
    locked_theta: float,
    matcher,
) -> tuple:
    """
    Run one VO step on a consecutive grayscale frame pair.
    Returns (new_x, new_y, new_theta, n_inliers, n_matches, vis_bgr).
    """
    w, h = DRONE_CAM_W, DRONE_CAM_H

    pts0, pts1, vis = matcher.match(prev_gray, curr_gray)
    n_matches = len(pts0)
    n_inliers = 0
    rot_deg = dx_img = dy_img = 0.0

    if n_matches >= 8:
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
            n_inliers = int(inliers.sum())
            cx, cy = w * 0.5, h * 0.5
            M3    = np.vstack([M, [0, 0, 1]])
            T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            T_pos = np.array([[1, 0,  cx], [0, 1,  cy], [0, 0, 1]])
            M_c   = (T_pos @ M3 @ T_neg)[:2, :]
            rot_deg = math.degrees(math.atan2(M_c[1, 0], M_c[0, 0]))
            locked_theta = (locked_theta - rot_deg + 360.0) % 360.0

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

    return locked_x, locked_y, locked_theta, n_inliers, n_matches, vis


# ---------------------------------------------------------------------------
# Cross-track error + nearest-point helper
# ---------------------------------------------------------------------------

def nearest_path_point(pos_x: float, pos_y: float, path: list) -> tuple:
    """Return the closest point on the path polyline to (pos_x, pos_y)."""
    if not path:
        return pos_x, pos_y
    if len(path) == 1:
        return path[0]
    best_pt = path[0]
    best_d  = float("inf")
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-9:
            px, py = ax, ay
        else:
            t = max(0.0, min(1.0, ((pos_x - ax) * abx + (pos_y - ay) * aby) / ab2))
            px, py = ax + t * abx, ay + t * aby
        d = math.hypot(px - pos_x, py - pos_y)
        if d < best_d:
            best_d, best_pt = d, (px, py)
    return best_pt


def cross_track_error(pos_x: float, pos_y: float, path: list) -> float:
    pt = nearest_path_point(pos_x, pos_y, path)
    return math.hypot(pt[0] - pos_x, pt[1] - pos_y)


def trim_path_to_position(path: list, pos_x: float, pos_y: float, search_limit: int = 40) -> list:
    if not path:
        return path
    best_i, best_d = 0, float("inf")
    for i, (px, py) in enumerate(path[:search_limit]):
        d = math.hypot(px - pos_x, py - pos_y)
        if d < best_d:
            best_d, best_i = d, i
    return path[max(0, best_i - 2):]


# ---------------------------------------------------------------------------
# Matcher loader
# ---------------------------------------------------------------------------

def load_matcher(name: str):
    import importlib
    _registry = {
        "SuperGlue": ("fbl.vo.matchers.superglue", "SuperGlueMatcher"),
        "LightGlue": ("fbl.vo.matchers.lightglue", "LightGlueMatcher"),
    }
    if name not in _registry:
        raise ValueError(f"Unknown matcher '{name}'. Choose from {list(_registry)}")
    mod_name, cls_name = _registry[name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


# ---------------------------------------------------------------------------
# GUI visualizer — pushes BGR frames into a DearPyGui event queue
# ---------------------------------------------------------------------------

class DPGVisualizer:
    """
    Same public interface as Visualizer but pushes BGR frames into an event
    queue instead of opening cv2 windows.  Used by the GUI benchmark tab.

    Events pushed: ("map_frame", bgr), ("match_frame", bgr)
    """
    RENDER_W = 1280

    def __init__(self, bg: np.ndarray, event_queue: queue.Queue):
        self._q    = event_queue
        h, w       = bg.shape[:2]
        self.scale = self.RENDER_W / w
        self._bg   = cv2.resize(bg, (self.RENDER_W, int(h * self.scale)))

    def _d(self, x, y):
        return int(x * self.scale), int(y * self.scale)

    def _poly(self, coords):
        return np.array(
            [[int(x * self.scale), int(y * self.scale)] for x, y in coords],
            dtype=np.int32,
        )

    def _push(self, kind: str, img: np.ndarray):
        try:
            self._q.put_nowait((kind, img))
        except queue.Full:
            pass

    def show_matches(self, vis_bgr: np.ndarray, title: str = "", info: str = "") -> None:
        h, w = vis_bgr.shape[:2]
        disp  = cv2.resize(vis_bgr, (self.RENDER_W, max(1, int(h * self.RENDER_W / w))))
        if info:
            cv2.putText(disp, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2, cv2.LINE_AA)
        self._push("match_frame", disp)

    def show_trajectory(
        self, title, waypoints, path, gt_history, vo_history,
        gt_now=None, vo_now=None, cte_foot=None, info_lines=None,
    ) -> None:
        canvas = self._bg.copy()
        if len(path) >= 2:
            cv2.polylines(canvas, [self._poly(path)], False, (200, 180, 0), 1, cv2.LINE_AA)
        for i, (wx, wy) in enumerate(waypoints):
            p = self._d(wx, wy)
            cv2.circle(canvas, p, 9, (255, 255, 255), 2)
            cv2.putText(canvas, f"WP{i+1}", (p[0]+11, p[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if len(gt_history) >= 2:
            cv2.polylines(canvas, [self._poly(gt_history)], False, (0, 230, 0), 2, cv2.LINE_AA)
        if len(vo_history) >= 2:
            cv2.polylines(canvas, [self._poly(vo_history)], False, (0, 0, 230), 2, cv2.LINE_AA)
        if gt_now and cte_foot:
            cv2.line(canvas, self._d(*gt_now), self._d(*cte_foot), (0, 220, 220), 1, cv2.LINE_AA)
        if gt_now:
            cv2.circle(canvas, self._d(*gt_now), 7, (0, 255, 0), -1)
        if vo_now:
            cv2.circle(canvas, self._d(*vo_now), 7, (0, 0, 255), -1)
        cv2.circle(canvas, (12, 12), 5, (0, 255, 0), -1)
        cv2.putText(canvas, "Ground Truth", (22, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (12, 30), 5, (0, 0, 255), -1)
        cv2.putText(canvas, "VO Estimate", (22, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (7, 44), (17, 44), (200, 180, 0), 1)
        cv2.putText(canvas, "Hermite path", (22, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 180, 0), 1, cv2.LINE_AA)
        if info_lines:
            for i, line in enumerate(info_lines):
                cv2.putText(canvas, line, (10, 72 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        self._push("map_frame", canvas)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Standalone visualizer — OpenCV windows (used when running from CLI)
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Draws live benchmark progress into OpenCV windows.

    Two modes:
      show_matches()    – Test 1: side-by-side match overlay from the matcher
      show_trajectory() – Test 2/3: top-down map with GT/VO paths + CTE line
    """

    DISPLAY_W = 1280

    def __init__(self, bg: np.ndarray):
        h, w     = bg.shape[:2]
        self.scale = self.DISPLAY_W / w
        dh         = int(h * self.scale)
        self._bg   = cv2.resize(bg, (self.DISPLAY_W, dh))

    def _d(self, x: float, y: float) -> tuple:
        return int(x * self.scale), int(y * self.scale)

    def _poly(self, coords: list) -> np.ndarray:
        return np.array(
            [[int(x * self.scale), int(y * self.scale)] for x, y in coords],
            dtype=np.int32,
        )

    def show_matches(self, vis_bgr: np.ndarray, title: str, info: str = "") -> None:
        h, w = vis_bgr.shape[:2]
        disp  = cv2.resize(vis_bgr, (self.DISPLAY_W, int(h * self.DISPLAY_W / w)))
        if info:
            cv2.putText(disp, info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(title, disp)
        cv2.waitKey(1)

    def show_trajectory(
        self,
        title: str,
        waypoints: list,
        path: list,
        gt_history: list,
        vo_history: list,
        gt_now: tuple = None,
        vo_now: tuple = None,
        cte_foot: tuple = None,
        info_lines: list = None,
    ) -> None:
        canvas = self._bg.copy()
        if len(path) >= 2:
            cv2.polylines(canvas, [self._poly(path)], False, (200, 180, 0), 1, cv2.LINE_AA)
        for i, (wx, wy) in enumerate(waypoints):
            p = self._d(wx, wy)
            cv2.circle(canvas, p, 9, (255, 255, 255), 2)
            cv2.putText(canvas, f"WP{i+1}", (p[0] + 11, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if len(gt_history) >= 2:
            cv2.polylines(canvas, [self._poly(gt_history)], False, (0, 230, 0), 2, cv2.LINE_AA)
        if len(vo_history) >= 2:
            cv2.polylines(canvas, [self._poly(vo_history)], False, (0, 0, 230), 2, cv2.LINE_AA)
        if gt_now and cte_foot:
            cv2.line(canvas, self._d(*gt_now), self._d(*cte_foot), (0, 220, 220), 1, cv2.LINE_AA)
        if gt_now:
            cv2.circle(canvas, self._d(*gt_now), 7, (0, 255, 0), -1)
        if vo_now:
            cv2.circle(canvas, self._d(*vo_now), 7, (0, 0, 255), -1)
        cv2.circle(canvas, (12, 12), 5, (0, 255, 0), -1)
        cv2.putText(canvas, "Ground Truth", (22, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (12, 30), 5, (0, 0, 255), -1)
        cv2.putText(canvas, "VO Estimate",  (22, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (7, 44), (17, 44), (200, 180, 0), 1)
        cv2.putText(canvas, "Hermite path", (22, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 0), 1, cv2.LINE_AA)
        if info_lines:
            for i, line in enumerate(info_lines):
                cv2.putText(canvas, line, (10, 72 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Console table printer
# ---------------------------------------------------------------------------

def print_table(title: str, headers: list, rows: list) -> None:
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    bar = "=" * len(sep)

    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(sep)
    print("|" + "|".join(f" {h:<{col_w[i]}} " for i, h in enumerate(headers)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {str(c):<{col_w[i]}} " for i, c in enumerate(row)) + "|")
    print(sep)

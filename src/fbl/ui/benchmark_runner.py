import json
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np

from fbl.core.engine import ASSETS_DIR

_PROJ_ROOT            = os.path.abspath(os.path.join(ASSETS_DIR, ".."))
_POLL_HZ              = 30
_TIMEOUT_S            = 90.0
_MATCHER_LOAD_TIMEOUT = 45.0

SCENARIO_LABELS = {
    "straight": "Straight",
    "curved":   "S-Curve",
    "zigzag":   "Zigzag",
}


def _make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 200 and all(isinstance(x, float) for x in obj):
            sampled = obj[::10]
            if obj[-1] not in sampled:
                sampled.append(obj[-1])
            return sampled
        return [_make_serialisable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _compute_metrics(gt_history, vo_history, cte_list,
                     elapsed, total_wps, wps_reached, vo_hz=0.0):
    if not gt_history:
        return {}
    gt_arr  = np.array(gt_history, dtype=np.float64)
    vo_arr  = np.array(vo_history, dtype=np.float64)
    errors  = np.sqrt(np.sum((gt_arr - vo_arr) ** 2, axis=1))
    cte_arr = np.array(cte_list, dtype=np.float64) if cte_list else np.array([0.0])
    return {
        "vo_hz":             vo_hz,
        "mae":               float(np.mean(np.abs(errors))),
        "rmse":              float(np.sqrt(np.mean(errors ** 2))),
        "max_error":         float(np.max(errors)),
        "final_drift":       float(errors[-1]),
        "mean_cte":          float(np.mean(cte_arr)),
        "max_cte":           float(np.max(cte_arr)),
        "wp_success_rate":   wps_reached / total_wps * 100.0 if total_wps else 0.0,
        "completion_time_s": elapsed,
        "wps_reached":       wps_reached,
        "total_wps":         total_wps,
    }


def _format_result_block(scenario_key: str, matcher: str, m: dict) -> str:
    label = SCENARIO_LABELS.get(scenario_key, scenario_key).upper()
    sep   = "─" * 40
    lines = [
        f"{label}  —  {matcher}",
        sep,
        "LOCALIZATION  (VO vs Ground Truth)",
        f"  Mean Error   {m['mae']:>7.1f} px",
        f"  RMSE         {m['rmse']:>7.1f} px",
        f"  Max Error    {m['max_error']:>7.1f} px",
        f"  Final Drift  {m['final_drift']:>7.1f} px",
        f"  VO Speed     {m.get('vo_hz', 0.0):>7.1f} Hz",
        "",
        "PATH FOLLOWING  (Cross-Track Error)",
        f"  Mean CTE     {m['mean_cte']:>7.1f} px",
        f"  Max CTE      {m['max_cte']:>7.1f} px",
        f"  Waypoints    {m['wps_reached']}/{m['total_wps']}  "
        f"({m['wp_success_rate']:>3.0f}%)",
        f"  Time         {m['completion_time_s']:>7.1f} s",
        f"  VO frames    {round(m['completion_time_s'] * m.get('vo_hz', 0)):>7d}",
    ]
    return "\n".join(lines)


def _format_results(all_results: dict, matcher: str) -> str:
    return "\n\n".join(
        _format_result_block(key, matcher, m)
        for key, m in all_results.items()
    )


class BenchmarkRunner:
    """
    Runs benchmark scenarios using the live simulation engine.

    Sets waypoints on the real engine (just like the user does via right-click),
    then monitors GT position vs VO estimate and computes:
      - Localization accuracy: MAE, RMSE, max error, final drift, VO speed (Hz)
      - Path following: mean/max cross-track error, waypoint success rate, time

    Events pushed to event_queue:
      ("status",      str)  – one-line status update
      ("results_txt", str)  – structured results text
      ("saved",       str)  – full path of saved JSON
      ("done",        None) – finished or stopped
    """

    def __init__(self, engine, pose_state: dict, pose_lock,
                 event_queue: queue.Queue):
        self._engine     = engine
        self._pose_state = pose_state
        self._pose_lock  = pose_lock
        self._q          = event_queue
        self._stop       = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, scenario_key: str, matcher_name: str = "SuperGlue"):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_one, args=(scenario_key, matcher_name), daemon=True
        )
        self._thread.start()

    def start_all(self, matcher_name: str = "SuperGlue"):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_all, args=(matcher_name,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def _push(self, kind: str, payload):
        try:
            self._q.put_nowait((kind, payload))
        except queue.Full:
            pass

    def _switch_and_wait(self, matcher_name: str) -> bool:
        """Switch the live VoNode to matcher_name (skips if already loaded).
        Returns False if stopped or timed out."""
        with self._pose_lock:
            if self._pose_state.get("current_matcher") == matcher_name:
                return True
            self._pose_state["switch_matcher"]  = matcher_name
            self._pose_state["matcher_loading"] = True

        self._push("status", f"Loading {matcher_name}…")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < _MATCHER_LOAD_TIMEOUT:
            if self._stop.is_set():
                return False
            with self._pose_lock:
                if not self._pose_state.get("matcher_loading", False):
                    return True
            time.sleep(0.1)
        return False

    def _reset_simulation(self):
        self._engine.is_running = False
        self._engine.clear_waypoints()
        self._engine.reset_state()
        with self._pose_lock:
            self._pose_state["reset_vo"]      = True
            self._pose_state["reset_planner"] = True
        time.sleep(0.8)

    def _run_scenario(self, key: str, waypoints: list) -> dict | None:
        import math as _math
        from fbl.benchmarks.utils import cross_track_error
        from fbl.navigation.path_generator import generate_hermite_path

        label = SCENARIO_LABELS.get(key, key)
        self._push("status", f"Resetting for {label}…")
        self._reset_simulation()
        if self._stop.is_set():
            return None

        start_x = self._engine._init_x
        start_y = self._engine._init_y

        # Align the reference path's initial tangent with the direction to the
        # first waypoint instead of assuming the drone is pointing north.
        # This keeps CTE meaningful regardless of route orientation.
        dx = waypoints[0][0] - start_x
        dy = waypoints[0][1] - start_y
        start_theta = _math.degrees(_math.atan2(dx, -dy)) % 360.0
        hermite_path = generate_hermite_path(
            start_x, start_y, start_theta, waypoints, step=10.0
        )

        for wx, wy in waypoints:
            self._engine.add_waypoint(wx, wy)
        total_wps = len(waypoints)

        self._engine.is_running = True
        self._push("status", f"{label} running…")

        gt_history: list  = []
        vo_history: list  = []
        cte_list:   list  = []
        fps_samples: list = []
        t_start           = time.perf_counter()

        while not self._stop.is_set():
            elapsed = time.perf_counter() - t_start
            if elapsed > _TIMEOUT_S:
                break

            gt_x = self._engine._pos_x
            gt_y = self._engine._pos_y
            with self._pose_lock:
                vo_x = self._pose_state.get("x", start_x)
                vo_y = self._pose_state.get("y", start_y)
                fps  = self._pose_state.get("vo_fps", 0.0)

            if fps > 0.0:
                fps_samples.append(fps)

            gt_history.append((gt_x, gt_y))
            vo_history.append((vo_x, vo_y))
            cte = cross_track_error(gt_x, gt_y, hermite_path)
            cte_list.append(cte)

            wps_remaining = len(self._engine.waypoints)
            wps_reached   = total_wps - wps_remaining
            self._push("status",
                f"{label}: {elapsed:.0f}s  "
                f"CTE={cte:.1f}px  WP {wps_reached}/{total_wps}")

            if wps_remaining == 0:
                break

            time.sleep(1.0 / _POLL_HZ)

        elapsed     = time.perf_counter() - t_start
        wps_reached = total_wps - len(self._engine.waypoints)
        self._engine.is_running = False

        # Median of rolling-window samples — robust against startup ramp-up
        # and the occasional stale sample.
        vo_hz = float(np.median(fps_samples)) if fps_samples else 0.0

        return _compute_metrics(
            gt_history, vo_history, cte_list,
            elapsed, total_wps, wps_reached, vo_hz,
        )

    def _save(self, results: dict, matcher: str):
        fname = f"benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(_PROJ_ROOT, fname)
        with open(fpath, "w") as f:
            json.dump(
                _make_serialisable({"matcher": matcher, "results": results}),
                f, indent=2,
            )
        self._push("saved", fpath)

    def _run_one(self, scenario_key: str, matcher_name: str):
        try:
            if not self._switch_and_wait(matcher_name):
                self._push("status", "Stopped.")
                return

            from fbl.benchmarks.utils import compute_safe_waypoints
            waypoints = compute_safe_waypoints(self._engine)[scenario_key]

            result = self._run_scenario(scenario_key, waypoints)
            if result and not self._stop.is_set():
                all_res = {scenario_key: result}
                self._push("results_txt", _format_results(all_res, matcher_name))
                self._save(all_res, matcher_name)
                self._push("status",
                    f"{SCENARIO_LABELS.get(scenario_key, scenario_key)} complete")
            elif self._stop.is_set():
                self._push("status", "Stopped.")
        except Exception as exc:
            self._push("status", f"Error: {exc}")
        finally:
            self._push("done", None)

    def _run_all(self, matcher_name: str):
        try:
            if not self._switch_and_wait(matcher_name):
                self._push("status", "Stopped.")
                return

            from fbl.benchmarks.utils import compute_safe_waypoints
            scenarios   = compute_safe_waypoints(self._engine)
            all_results = {}

            for key in ("straight", "curved", "zigzag"):
                if self._stop.is_set():
                    break
                result = self._run_scenario(key, scenarios[key])
                if result:
                    all_results[key] = result

            if all_results and not self._stop.is_set():
                self._push("results_txt", _format_results(all_results, matcher_name))
                self._save(all_results, matcher_name)
                self._push("status", "All scenarios complete")
            elif self._stop.is_set():
                self._push("status", "Stopped.")
        except Exception as exc:
            self._push("status", f"Error: {exc}")
        finally:
            self._push("done", None)

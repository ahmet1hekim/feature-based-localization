"""
Test 2 – Visual Odometry Accuracy

The VO module is run frame-by-frame on a 3-waypoint simulated path that
contains both a straight segment and a curved segment.  Its estimated
position is compared against the ground-truth position provided by
SimEngine.

Pass --visualize to open a live top-down map window showing:
  green  – ground-truth drone trail
  red    – VO-estimated trail
  cyan   – planned Hermite path
  white  – waypoints

Metrics per matcher: MAE (px), RMSE (px), drift per frame.

Run standalone:  python -m fbl.benchmarks.test2 [--bg IMG] [--steps N] [-v]
"""
import os
import math
import time
import numpy as np

from fbl.benchmarks.utils import (
    make_engine, reset_engine, render_gray,
    vo_step, load_matcher, compute_safe_waypoints,
    Visualizer, print_table,
)


# ---------------------------------------------------------------------------
# Closed-loop VO trajectory simulation
# ---------------------------------------------------------------------------

def _simulate_vo_trajectory(engine, matcher, waypoints, max_steps=600,
                             vis=None, vis_title="", stop_event=None) -> tuple:
    from fbl.core.state import AutopilotCmd
    from fbl.navigation.controller import compute_commands, GOAL_RADIUS
    from fbl.navigation.path_generator import generate_hermite_path

    reset_engine(engine)
    hermite_path = generate_hermite_path(
        engine._pos_x, engine._pos_y, engine._angle, waypoints, step=10.0
    )

    prev_gray = render_gray(engine)
    vo_x, vo_y, vo_theta = engine._pos_x, engine._pos_y, engine._angle

    gt_history: list = []
    vo_history: list = []
    remaining_wps    = list(waypoints)

    for step in range(max_steps):
        if not remaining_wps:
            break
        if stop_event is not None and stop_event.is_set():
            break

        goal_x, goal_y = remaining_wps[0]
        dist_to_final  = math.hypot(
            remaining_wps[-1][0] - vo_x,
            remaining_wps[-1][1] - vo_y,
        )
        speed_cmd, turn_cmd = compute_commands(
            vo_x, vo_y, vo_theta, goal_x, goal_y, dist_to_final
        )
        engine.apply_autopilot_cmd(AutopilotCmd(speed=speed_cmd, turn_angle=turn_cmd), True)
        state = engine.tick({})

        curr_gray = render_gray(engine)
        vo_x, vo_y, vo_theta, n_inl, n_m, vis_img = vo_step(
            prev_gray, curr_gray, vo_x, vo_y, vo_theta, matcher
        )
        prev_gray = curr_gray.copy()

        gt_history.append((state.pos_x, state.pos_y))
        vo_history.append((vo_x, vo_y))

        err = math.hypot(state.pos_x - vo_x, state.pos_y - vo_y)

        if math.hypot(goal_x - state.pos_x, goal_y - state.pos_y) < GOAL_RADIUS:
            remaining_wps.pop(0)
            label = "DONE" if not remaining_wps else f"{len(remaining_wps)} left"
            print(f"\n    Waypoint reached at step {step+1}. {label}")

        print(
            f"  step={step+1:4d}  GT=({state.pos_x:.1f},{state.pos_y:.1f})"
            f"  VO=({vo_x:.1f},{vo_y:.1f})  err={err:.1f}px"
            f"  m={n_m} i={n_inl}",
            end="\r",
        )

        if vis is not None:
            vis.show_trajectory(
                title=vis_title,
                waypoints=waypoints,
                path=hermite_path,
                gt_history=gt_history,
                vo_history=vo_history,
                gt_now=(state.pos_x, state.pos_y),
                vo_now=(vo_x, vo_y),
                info_lines=[
                    f"Step {step+1}  |  matches={n_m}  inliers={n_inl}",
                    f"GT  ({state.pos_x:.1f}, {state.pos_y:.1f})",
                    f"VO  ({vo_x:.1f}, {vo_y:.1f})",
                    f"Error  {err:.2f} px",
                ],
            )

    print()
    return gt_history, vo_history, hermite_path


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(gt: list, vo: list) -> dict:
    errors = np.array([math.hypot(g[0]-v[0], g[1]-v[1]) for g, v in zip(gt, vo)])
    return {
        "n_frames":        len(errors),
        "mae":             float(np.mean(np.abs(errors))),
        "rmse":            float(np.sqrt(np.mean(errors**2))),
        "max_error":       float(np.max(errors)),
        "final_drift":     float(errors[-1]) if len(errors) else 0.0,
        "drift_per_frame": errors.tolist(),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_test2(bg_image: str = "dag.jpg", max_steps: int = 600, visualize: bool = False,
              vis=None, stop_event=None) -> dict:
    print("\n" + "=" * 64)
    print("  TEST 2 – Visual Odometry Accuracy")
    print(f"  Background : {bg_image}  |  Visualize : {visualize}")
    print("=" * 64)

    engine    = make_engine(bg_image)
    wps       = compute_safe_waypoints(engine)
    waypoints = wps["vo_path"]

    print(f"\n  Start : ({engine._init_x:.0f}, {engine._init_y:.0f})")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i+1}  : ({wp[0]:.0f}, {wp[1]:.0f})")
    print(f"  (Background {engine.bg_w}×{engine.bg_h}, "
          f"margins ~{engine.bg_w*0.10:.0f}px × {engine.bg_h*0.08:.0f}px)")

    _own_vis = vis is None
    if vis is None and visualize:
        vis = Visualizer(engine.background)
    results = {}

    for name in ("SuperGlue", "LightGlue"):
        if stop_event is not None and stop_event.is_set():
            break
        print(f"\n  ── {name} ──")
        matcher = load_matcher(name)

        t0 = time.perf_counter()
        gt_pos, vo_pos, _ = _simulate_vo_trajectory(
            engine, matcher, waypoints,
            max_steps=max_steps,
            vis=vis,
            vis_title=f"Test 2 – VO Accuracy ({name})",
            stop_event=stop_event,
        )
        elapsed = time.perf_counter() - t0

        metrics = _compute_metrics(gt_pos, vo_pos)
        metrics["wall_time_s"] = elapsed
        results[name] = metrics

        print(
            f"    Frames={metrics['n_frames']}  MAE={metrics['mae']:.2f}px  "
            f"RMSE={metrics['rmse']:.2f}px  Max={metrics['max_error']:.2f}px  "
            f"FinalDrift={metrics['final_drift']:.2f}px"
        )
        del matcher

    if _own_vis and vis:
        vis.close()

    headers = ["Matcher", "Frames", "MAE (px)", "RMSE (px)", "Max Error (px)", "Final Drift (px)"]
    rows = [
        [name, str(m["n_frames"]), f"{m['mae']:.2f}", f"{m['rmse']:.2f}",
         f"{m['max_error']:.2f}", f"{m['final_drift']:.2f}"]
        for name, m in results.items()
    ]
    print_table("Table – VO Accuracy (3-Waypoint Path)", headers, rows)

    print("\n  Per-frame drift sample (every 50 frames):")
    header2 = ["Frame"] + list(results.keys())
    n_max   = max(m["n_frames"] for m in results.values())
    rows2   = []
    for f in range(0, n_max, 50):
        row = [str(f + 1)]
        for m in results.values():
            d = m["drift_per_frame"]
            row.append(f"{d[f]:.2f}" if f < len(d) else f"{d[-1]:.2f}")
        rows2.append(row)
    print_table("Table – Drift Accumulation (px)", header2, rows2)

    return results


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import argparse
    p = argparse.ArgumentParser(description="Test 2 – VO Accuracy")
    p.add_argument("--bg",        default="dag.jpg")
    p.add_argument("--steps",     type=int, default=600)
    p.add_argument("--visualize", "-v", action="store_true")
    args = p.parse_args()
    run_test2(bg_image=args.bg, max_steps=args.steps, visualize=args.visualize)

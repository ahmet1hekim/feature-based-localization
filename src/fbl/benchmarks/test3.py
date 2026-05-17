"""
Test 3 – Path Following Performance (Table 4.4)

The UAV follows a Hermite-interpolated path using VO-estimated pose +
pure-pursuit controller.  Cross-track error (CTE) is measured against the
ground-truth position at every simulation frame.

Scenarios (waypoints derived from background dimensions):
  (a) straight  – three collinear waypoints
  (b) curved    – S-curve (alternating lateral offsets)
  (c) zigzag    – sudden direction changes (large lateral swings)

Each scenario is run with SuperGlue and LightGlue independently.

Run standalone:  python -m fbl.benchmarks.test3 [--bg IMG] [--steps N] [-v]
"""
import os
import math
import time
import numpy as np

from fbl.benchmarks.utils import (
    make_engine, reset_engine, render_gray,
    vo_step, cross_track_error, nearest_path_point,
    trim_path_to_position, load_matcher,
    compute_safe_waypoints, Visualizer, print_table,
)

LOOKAHEAD = 90.0
MAX_STEPS = 1200


# ---------------------------------------------------------------------------
# Single scenario run
# ---------------------------------------------------------------------------

def _run_scenario(
    engine, matcher, matcher_name, scenario_key,
    hermite_path, waypoints_world,
    max_steps=MAX_STEPS, vis=None, stop_event=None,
) -> dict:
    from fbl.core.state import AutopilotCmd
    from fbl.navigation.controller import (
        compute_commands, get_pure_pursuit_lookahead, GOAL_RADIUS,
    )

    reset_engine(engine)
    prev_gray = render_gray(engine)

    vo_x, vo_y, vo_theta = engine._pos_x, engine._pos_y, engine._angle
    local_path    = list(hermite_path)
    remaining_wps = list(waypoints_world)
    total_wps     = len(waypoints_world)
    wps_reached   = 0

    gt_history: list = []
    vo_history: list = []
    cte_list:   list = []
    start_time  = time.perf_counter()

    for step in range(max_steps):
        if not remaining_wps:
            break
        if stop_event is not None and stop_event.is_set():
            break

        local_path = trim_path_to_position(local_path, vo_x, vo_y)
        la_x, la_y = get_pure_pursuit_lookahead(vo_x, vo_y, local_path, LOOKAHEAD)

        final_wp      = remaining_wps[-1]
        dist_to_final = math.hypot(final_wp[0] - vo_x, final_wp[1] - vo_y)
        speed_cmd, turn_cmd = compute_commands(
            vo_x, vo_y, vo_theta, la_x, la_y, dist_to_final
        )

        engine.apply_autopilot_cmd(AutopilotCmd(speed=speed_cmd, turn_angle=turn_cmd), True)
        state = engine.tick({})

        curr_gray = render_gray(engine)
        vo_x, vo_y, vo_theta, n_inl, n_m, _ = vo_step(
            prev_gray, curr_gray, vo_x, vo_y, vo_theta, matcher
        )
        prev_gray = curr_gray.copy()

        cte  = cross_track_error(state.pos_x, state.pos_y, hermite_path)
        foot = nearest_path_point(state.pos_x, state.pos_y, hermite_path)
        cte_list.append(cte)

        gt_history.append((state.pos_x, state.pos_y))
        vo_history.append((vo_x, vo_y))

        goal_x, goal_y = remaining_wps[0]
        if math.hypot(goal_x - state.pos_x, goal_y - state.pos_y) < GOAL_RADIUS:
            wps_reached += 1
            remaining_wps.pop(0)

        print(
            f"  [{matcher_name}/{scenario_key}] "
            f"step={step+1:4d}  CTE={cte:6.1f}px  "
            f"GT=({state.pos_x:.0f},{state.pos_y:.0f})  "
            f"VO=({vo_x:.0f},{vo_y:.0f})  "
            f"wps={wps_reached}/{total_wps}",
            end="\r",
        )

        if vis is not None:
            vis.show_trajectory(
                title=f"Test 3 – {matcher_name} / {scenario_key}",
                waypoints=waypoints_world,
                path=hermite_path,
                gt_history=gt_history,
                vo_history=vo_history,
                gt_now=(state.pos_x, state.pos_y),
                vo_now=(vo_x, vo_y),
                cte_foot=foot,
                info_lines=[
                    f"Scenario: {scenario_key}  |  Matcher: {matcher_name}",
                    f"Step {step+1}  |  WP {wps_reached}/{total_wps}",
                    f"CTE  {cte:.1f} px",
                    f"matches={n_m}  inliers={n_inl}",
                ],
            )

    print()
    elapsed = time.perf_counter() - start_time
    cte_arr = np.array(cte_list) if cte_list else np.array([0.0])

    return {
        "scenario":          scenario_key,
        "matcher":           matcher_name,
        "mean_cte":          float(np.mean(cte_arr)),
        "max_cte":           float(np.max(cte_arr)),
        "wp_success_rate":   wps_reached / total_wps * 100.0 if total_wps else 0.0,
        "completion_time_s": elapsed,
        "n_frames":          len(cte_list),
        "wps_reached":       wps_reached,
        "total_wps":         total_wps,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_test3(bg_image: str = "dag.jpg", max_steps: int = MAX_STEPS, visualize: bool = False,
              vis=None, stop_event=None) -> list:
    print("\n" + "=" * 64)
    print("  TEST 3 – Path Following Performance")
    print(f"  Background : {bg_image}  |  Visualize : {visualize}")
    print("=" * 64)

    from fbl.navigation.path_generator import generate_hermite_path

    engine   = make_engine(bg_image)
    wps      = compute_safe_waypoints(engine)
    _own_vis = vis is None
    if vis is None and visualize:
        vis = Visualizer(engine.background)

    print(f"\n  Background {engine.bg_w}×{engine.bg_h}  "
          f"Start ({engine._init_x:.0f},{engine._init_y:.0f})")

    scenario_labels = {
        "straight": "(a) Straight path",
        "curved":   "(b) Narrow S-curve",
        "zigzag":   "(c) Sudden direction changes",
    }

    all_results: list = []

    for scenario_key, label in scenario_labels.items():
        if stop_event is not None and stop_event.is_set():
            break
        waypoints_world = wps[scenario_key]
        hermite_path    = generate_hermite_path(
            engine._init_x, engine._init_y, 0.0, waypoints_world, step=10.0
        )

        print(f"\n{'─'*64}")
        print(f"  {label}")
        print(f"  Waypoints : {[(round(x), round(y)) for x,y in waypoints_world]}")
        print(f"  Path pts  : {len(hermite_path)}")

        for matcher_name in ("SuperGlue", "LightGlue"):
            if stop_event is not None and stop_event.is_set():
                break
            print(f"\n  ── {matcher_name} ──")
            matcher = load_matcher(matcher_name)

            result = _run_scenario(
                engine, matcher, matcher_name,
                scenario_key, hermite_path, waypoints_world,
                max_steps=max_steps, vis=vis, stop_event=stop_event,
            )
            all_results.append(result)

            print(
                f"  Done | Mean CTE={result['mean_cte']:.1f}px  "
                f"Max CTE={result['max_cte']:.1f}px  "
                f"WP={result['wps_reached']}/{result['total_wps']}  "
                f"Time={result['completion_time_s']:.1f}s"
            )
            del matcher

    if _own_vis and vis:
        vis.close()

    headers = [
        "Scenario", "Matcher",
        "Mean CTE (px)", "Max CTE (px)",
        "WP Success (%)", "Time (s)", "Frames",
    ]
    rows = [
        [
            r["scenario"], r["matcher"],
            f"{r['mean_cte']:.1f}", f"{r['max_cte']:.1f}",
            f"{r['wp_success_rate']:.0f}%",
            f"{r['completion_time_s']:.1f}", str(r["n_frames"]),
        ]
        for r in all_results
    ]
    print_table("Table 4.4 – Path Following Performance", headers, rows)

    cmp_headers = ["Scenario", "Metric", "SuperGlue", "LightGlue"]
    cmp_rows    = []
    for sk in scenario_labels:
        sg = next((r for r in all_results if r["scenario"]==sk and r["matcher"]=="SuperGlue"), None)
        lg = next((r for r in all_results if r["scenario"]==sk and r["matcher"]=="LightGlue"), None)
        if sg and lg:
            cmp_rows += [
                [sk, "Mean CTE (px)",   f"{sg['mean_cte']:.1f}",          f"{lg['mean_cte']:.1f}"],
                ["",  "Max CTE (px)",   f"{sg['max_cte']:.1f}",           f"{lg['max_cte']:.1f}"],
                ["",  "WP Success (%)", f"{sg['wp_success_rate']:.0f}",   f"{lg['wp_success_rate']:.0f}"],
                ["",  "Time (s)",       f"{sg['completion_time_s']:.1f}", f"{lg['completion_time_s']:.1f}"],
            ]
    print_table("Table 4.4b – Per-Scenario Comparison", cmp_headers, cmp_rows)

    return all_results


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import argparse
    p = argparse.ArgumentParser(description="Test 3 – Path Following")
    p.add_argument("--bg",        default="dag.jpg")
    p.add_argument("--steps",     type=int, default=MAX_STEPS)
    p.add_argument("--visualize", "-v", action="store_true")
    args = p.parse_args()
    run_test3(bg_image=args.bg, max_steps=args.steps, visualize=args.visualize)

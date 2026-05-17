"""
Test 1 – Matcher Comparison (Table 4.1 / 4.2 / 4.3)

For 100 consecutive frame pairs from the simulation:
  - Avg match count (post-confidence threshold)
  - Avg RANSAC inlier count
  - Avg inlier ratio (%)
  - Avg processing time per frame (ms)

Run standalone:  python -m fbl.benchmarks.test1 [--pairs N] [--bg IMG] [-v]
"""
import os
import time
import numpy as np

from fbl.benchmarks.utils import (
    make_engine, generate_frame_pairs, run_ransac, print_table, Visualizer,
)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def _benchmark_matcher(matcher_name, matcher, frame_pairs, vis=None, stop_event=None) -> dict:
    n_pairs        = len(frame_pairs)
    n_matches_all  = []
    n_inliers_all  = []
    inlier_pct_all = []
    time_ms_all    = []

    for idx, (gray0, gray1) in enumerate(frame_pairs):
        if stop_event is not None and stop_event.is_set():
            break
        t0 = time.perf_counter()
        pts0, pts1, vis_img = matcher.match(gray0, gray1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        n_raw        = len(pts0)
        n_inliers, _ = run_ransac(pts0, pts1)
        ratio        = (n_inliers / n_raw * 100.0) if n_raw > 0 else 0.0

        n_matches_all.append(n_raw)
        n_inliers_all.append(n_inliers)
        inlier_pct_all.append(ratio)
        time_ms_all.append(elapsed_ms)

        print(
            f"  [{matcher_name}] {idx+1:3d}/{n_pairs} | "
            f"matches={n_raw:4d}  inliers={n_inliers:4d}  "
            f"ratio={ratio:5.1f}%  time={elapsed_ms:7.1f}ms"
        )

        if vis is not None and vis_img is not None:
            vis.show_matches(
                vis_img,
                title=f"Test 1 – {matcher_name}",
                info=(
                    f"Pair {idx+1}/{n_pairs}  |  "
                    f"matches={n_raw}  inliers={n_inliers}  "
                    f"ratio={ratio:.1f}%  {elapsed_ms:.0f}ms"
                ),
            )

    return {
        "matcher":        matcher_name,
        "avg_matches":    float(np.mean(n_matches_all)),
        "std_matches":    float(np.std(n_matches_all)),
        "avg_inliers":    float(np.mean(n_inliers_all)),
        "std_inliers":    float(np.std(n_inliers_all)),
        "avg_inlier_pct": float(np.mean(inlier_pct_all)),
        "avg_time_ms":    float(np.mean(time_ms_all)),
        "std_time_ms":    float(np.std(time_ms_all)),
        "n_pairs":        n_pairs,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_test1(n_pairs: int = 100, bg_image: str = "dag.jpg", visualize: bool = False,
              vis=None, stop_event=None) -> list:
    print("\n" + "=" * 64)
    print("  TEST 1 – Matcher Comparison")
    print(f"  Frame pairs : {n_pairs}  |  Background : {bg_image}")
    print(f"  Frame size  : 960 × 540")
    print("=" * 64)

    print("\n[Step 1/3]  Generating frame pairs from simulation output...")
    engine = make_engine(bg_image)
    pairs  = generate_frame_pairs(engine, n_pairs=n_pairs)
    print(f"  Generated {len(pairs)} pairs.")

    _own_vis = vis is None
    if vis is None and visualize:
        vis = Visualizer(engine.background)
    results = []

    print("\n[Step 2/3]  Benchmarking SuperGlue...")
    from fbl.vo.matchers.superglue import SuperGlueMatcher
    sg = SuperGlueMatcher(weights="outdoor")
    results.append(_benchmark_matcher("SuperGlue", sg, pairs, vis=vis, stop_event=stop_event))
    del sg

    print("\n[Step 3/3]  Benchmarking LightGlue...")
    from fbl.vo.matchers.lightglue import LightGlueMatcher
    lg = LightGlueMatcher()
    results.append(_benchmark_matcher("LightGlue", lg, pairs, vis=vis, stop_event=stop_event))
    del lg

    if _own_vis and vis:
        vis.close()

    headers = [
        "Matcher",
        "Avg Matches",
        "Avg RANSAC Inliers",
        "Inlier Ratio (%)",
        "Avg Time / Frame (ms)",
    ]
    rows = [
        [
            r["matcher"],
            f"{r['avg_matches']:.1f} ± {r['std_matches']:.1f}",
            f"{r['avg_inliers']:.1f} ± {r['std_inliers']:.1f}",
            f"{r['avg_inlier_pct']:.1f}",
            f"{r['avg_time_ms']:.1f} ± {r['std_time_ms']:.1f}",
        ]
        for r in results
    ]
    print_table("Table 4.1/4.2/4.3 – Matcher Comparison", headers, rows)
    return results


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import argparse
    p = argparse.ArgumentParser(description="Test 1 – Matcher Comparison")
    p.add_argument("--pairs",     type=int, default=100)
    p.add_argument("--bg",                  default="dag.jpg")
    p.add_argument("--visualize", "-v",     action="store_true")
    args = p.parse_args()
    run_test1(n_pairs=args.pairs, bg_image=args.bg, visualize=args.visualize)

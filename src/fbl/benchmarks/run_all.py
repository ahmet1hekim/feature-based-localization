"""
FBL Benchmark Suite – entry point

Usage:
    python -m fbl.benchmarks.run_all            # run all tests
    python -m fbl.benchmarks.run_all --test 1   # single test

Options:
    --test  {1,2,3,all}   Which test(s) to run  (default: all)
    --pairs INT            Frame pairs for Test 1 (default: 100)
    --steps INT            Max simulation steps for Tests 2 & 3
    --bg    FILENAME       Background image from assets/ (default: dag.jpg)
    --output FILE.json     Save results to JSON
    --visualize / -v       Show live OpenCV windows during each test
"""
import os
import argparse
import json
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


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


def main():
    p = argparse.ArgumentParser(description="FBL Benchmark Suite")
    p.add_argument("--test",    choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--pairs",   type=int, default=100)
    p.add_argument("--steps",   type=int, default=None)
    p.add_argument("--bg",      default="dag.jpg")
    p.add_argument("--output",  default=None)
    p.add_argument("--visualize", "-v", action="store_true")
    args = p.parse_args()

    all_results = {}
    t_total = time.perf_counter()

    if args.test in ("1", "all"):
        from fbl.benchmarks.test1 import run_test1
        all_results["test1"] = run_test1(
            n_pairs=args.pairs, bg_image=args.bg, visualize=args.visualize
        )

    if args.test in ("2", "all"):
        from fbl.benchmarks.test2 import run_test2
        kw = {"max_steps": args.steps} if args.steps else {}
        all_results["test2"] = run_test2(bg_image=args.bg, visualize=args.visualize, **kw)

    if args.test in ("3", "all"):
        from fbl.benchmarks.test3 import run_test3
        kw = {"max_steps": args.steps} if args.steps else {}
        all_results["test3"] = run_test3(bg_image=args.bg, visualize=args.visualize, **kw)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*64}")
    print(f"  All benchmarks completed in {elapsed:.1f}s")
    print(f"{'='*64}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(_make_serialisable(all_results), f, indent=2)
        print(f"  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()

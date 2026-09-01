#!/usr/bin/env python3
"""Measure the PCK normalizer's stability and pick a defensible degeneracy guard.

Background
----------
`metrics_core.rs::canonical_torso_size` normalizes PCK by the hip-to-hip
*width* (COCO joints 11<->12), falling back to the visible-keypoint bounding-box
diagonal only when that width is <= MIN_REFERENCE_EXTENT (1e-6).

Hip *width* foreshortens to ~0 whenever the subject turns side-on, so the 1e-6
absolute floor almost never fires: instead the PCK radius silently collapses and
the sample scores ~0 regardless of prediction quality. The metric ends up partly
measuring body orientation.

This script does not assume a fix. It measures:
  1. the distribution of hip width, bbox diagonal, and their ratio;
  2. how much of the metric's per-capture variance is explained by the
     denominator rather than by prediction quality;
  3. the effect of candidate *relative* guards (fall back to the bbox diagonal
     when hip < ratio * diag) on the spread of a fixed mean-pose baseline.

A good guard shrinks the spread of the mean-pose baseline across captures
without altering well-conditioned frontal frames.

Usage
-----
    python scripts/pck_denominator_study.py
    python scripts/pck_denominator_study.py --threshold 0.20 --data <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LEFT_HIP = 11
RIGHT_HIP = 12
VIS_T = 0.5
MIN_REFERENCE_EXTENT = 1e-6

# Candidate relative guards to evaluate (hip < r * diag  =>  use diag instead).
CANDIDATE_RATIOS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]


def load_frames(path: Path) -> list[list[list[float]]]:
    """Return per-frame 17x3 keypoint arrays from a .vis.jsonl file."""
    frames = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kps = rec.get("keypoints")
            if not kps or len(kps) < 17:
                continue
            frames.append(kps)
    return frames


def bbox_diagonal(kps) -> float:
    xs, ys = [], []
    for x, y, v in kps:
        if v >= VIS_T:
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0.0
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return math.hypot(max(w, 0.0), max(h, 0.0))


def hip_width(kps) -> float | None:
    """Hip<->hip distance, or None when either hip is not visible."""
    if kps[LEFT_HIP][2] < VIS_T or kps[RIGHT_HIP][2] < VIS_T:
        return None
    dx = kps[LEFT_HIP][0] - kps[RIGHT_HIP][0]
    dy = kps[LEFT_HIP][1] - kps[RIGHT_HIP][1]
    return math.hypot(dx, dy)


def torso_size(kps, guard_ratio: float) -> float | None:
    """canonical_torso_size with a *relative* degeneracy guard.

    guard_ratio == 0.0 reproduces today's Rust behaviour exactly.
    """
    diag = bbox_diagonal(kps)
    hip = hip_width(kps)
    if hip is not None and hip > MIN_REFERENCE_EXTENT:
        # Reject a foreshortened hip line: it is not a credible body scale.
        if guard_ratio <= 0.0 or diag <= 0.0 or hip >= guard_ratio * diag:
            return hip
    return diag if diag > MIN_REFERENCE_EXTENT else None


def pck(pred, gt, threshold: float, guard_ratio: float) -> tuple[int, int]:
    """Return (correct, total) for one frame; (0, 0) when unscoreable."""
    torso = torso_size(gt, guard_ratio)
    if torso is None:
        return 0, 0
    radius = threshold * torso
    correct = total = 0
    for j in range(17):
        if gt[j][2] < VIS_T:
            continue
        total += 1
        if math.hypot(pred[j][0] - gt[j][0], pred[j][1] - gt[j][1]) <= radius:
            correct += 1
    return correct, total


def mean_pose(frames) -> list[list[float]]:
    """Visibility-weighted mean pose - the constant-predictor baseline."""
    acc = [[0.0, 0.0, 0.0] for _ in range(17)]
    for kps in frames:
        for j in range(17):
            if kps[j][2] >= VIS_T:
                acc[j][0] += kps[j][0]
                acc[j][1] += kps[j][1]
                acc[j][2] += 1.0
    out = []
    for j in range(17):
        n = acc[j][2]
        out.append([acc[j][0] / n, acc[j][1] / n, 1.0] if n > 0 else [0.5, 0.5, 1.0])
    return out


def pct(values, q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\ahmed\RuView\data")
    ap.add_argument("--threshold", type=float, default=0.20, help="PCK threshold")
    args = ap.parse_args()

    files = sorted(Path(args.data).rglob("*.vis.jsonl"))
    if not files:
        print("No .vis.jsonl files found under", args.data)
        return 1

    captures = []
    for f in files:
        fr = load_frames(f)
        if fr:
            captures.append((f.name, fr))

    print("=" * 78)
    print("PCK DENOMINATOR STUDY".center(78))
    print("=" * 78)
    print(f"\ncaptures: {len(captures)}   frames: {sum(len(c[1]) for c in captures)}")

    # ---- 1. denominator distribution -------------------------------------
    hips, diags, ratios = [], [], []
    no_hip = 0
    for _, frames in captures:
        for kps in frames:
            d = bbox_diagonal(kps)
            h = hip_width(kps)
            if d > 0:
                diags.append(d)
            if h is None:
                no_hip += 1
                continue
            hips.append(h)
            if d > 0:
                ratios.append(h / d)

    def line(name, vals):
        if not vals:
            print(f"  {name:<22} (none)")
            return
        print(
            f"  {name:<22} min={min(vals):8.5f}  p10={pct(vals,.10):8.5f}  "
            f"p50={pct(vals,.50):8.5f}  p90={pct(vals,.90):8.5f}  max={max(vals):8.5f}"
        )

    print("\n[1] Denominator distribution")
    line("hip width", hips)
    line("bbox diagonal", diags)
    line("hip / diag ratio", ratios)
    if hips:
        print(f"\n  hip-width spread (max/min): {max(hips)/max(min(hips),1e-12):,.0f}x")
    print(f"  frames with no visible hip pair: {no_hip}")
    if ratios:
        for r in CANDIDATE_RATIOS[1:]:
            n = sum(1 for x in ratios if x < r)
            print(f"  ratio < {r:<5}: {n:6d} frames ({100*n/len(ratios):5.1f}%) would fall back to bbox")

    # ---- 2. baseline stability under each candidate guard -----------------
    print(f"\n[2] Mean-pose baseline PCK@{int(args.threshold*100)} per capture")
    print("    (a *constant* predictor - spread here is pure metric instability)\n")

    header = "    capture                       " + "".join(f"{r:>8}" for r in CANDIDATE_RATIOS)
    print(header)
    print("    " + "-" * (len(header) - 4))

    per_ratio_scores: dict[float, list[float]] = {r: [] for r in CANDIDATE_RATIOS}
    for name, frames in captures:
        mp = mean_pose(frames)
        row = f"    {name[:28]:<30}"
        for r in CANDIDATE_RATIOS:
            c = t = 0
            for kps in frames:
                a, b = pck(mp, kps, args.threshold, r)
                c += a
                t += b
            s = 100.0 * c / t if t else 0.0
            per_ratio_scores[r].append(s)
            row += f"{s:7.2f} "
        print(row)

    print("\n[3] Guard comparison (lower spread = more trustworthy metric)\n")
    print("    guard   min      max      spread   stdev    verdict")
    print("    " + "-" * 62)
    best = None
    for r in CANDIDATE_RATIOS:
        v = per_ratio_scores[r]
        lo, hi = min(v), max(v)
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
        spread = hi / max(lo, 1e-9)
        tag = "  <-- current (1e-6 absolute)" if r == 0.0 else ""
        print(f"    {r:<7} {lo:6.2f}  {hi:6.2f}  {spread:8.1f}x {sd:7.2f}{tag}")
        if r > 0.0 and (best is None or sd < best[1]):
            best = (r, sd)

    if best:
        print(f"\n    Lowest baseline variance at guard ratio = {best[0]} (stdev {best[1]:.2f})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

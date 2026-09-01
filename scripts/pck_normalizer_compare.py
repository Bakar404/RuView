#!/usr/bin/env python3
"""Compare candidate PCK normalizers for rotation robustness.

Why
---
`pck_denominator_study.py` showed that the current hip<->hip *width* normalizer
produces a 1582x spread in a constant-predictor baseline (0.05% .. 79.01%), and
that simply guarding it against degeneracy does not help: the measured
hip/bbox-diagonal ratio never exceeds 0.251, so any guard strict enough to catch
foreshortening (>=0.15) redirects ~96% of frames to the bounding-box fallback,
which is so loose that a constant pose scores 61-88%.

The defect is structural. Hip *width* is a horizontal segment, so it foreshortens
to ~0 whenever the subject turns side-on. A normalizer that spans the body's
*long* axis does not.

This script evaluates several normalizers on two axes:

  STABILITY  - spread / stdev of a fixed mean-pose baseline across captures.
               A constant predictor has constant skill, so any spread is pure
               metric noise.
  DISCRIMINATION - the absolute level of that baseline. A normalizer that makes
               a constant pose score ~80% has destroyed the metric's ability to
               distinguish a real model from a trivial one.

The winner minimises spread while keeping the baseline low enough to leave
headroom for a real model to demonstrate skill.

Usage
-----
    python scripts/pck_normalizer_compare.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# COCO indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

VIS_T = 0.5
EPS = 1e-6


def load_frames(path: Path):
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
            if kps and len(kps) >= 17:
                frames.append(kps)
    return frames


def vis(kps, i):
    return kps[i][2] >= VIS_T


def dist(kps, a, b):
    return math.hypot(kps[a][0] - kps[b][0], kps[a][1] - kps[b][1])


def midpoint(kps, a, b):
    return ((kps[a][0] + kps[b][0]) / 2.0, (kps[a][1] + kps[b][1]) / 2.0)


def bbox_diag(kps):
    xs = [kps[j][0] for j in range(17) if vis(kps, j)]
    ys = [kps[j][1] for j in range(17) if vis(kps, j)]
    if not xs:
        return 0.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


# --- normalizer definitions -------------------------------------------------
# Each returns a positive scale, or None when it cannot be computed.


def n_hip_width(kps):
    """CURRENT. Hip<->hip width, bbox fallback only on exact degeneracy."""
    if vis(kps, L_HIP) and vis(kps, R_HIP):
        d = dist(kps, L_HIP, R_HIP)
        if d > EPS:
            return d
    d = bbox_diag(kps)
    return d if d > EPS else None


def n_bbox(kps):
    """Bounding-box diagonal of visible keypoints."""
    d = bbox_diag(kps)
    return d if d > EPS else None


def n_torso_height(kps):
    """Shoulder-centre <-> hip-centre. Spans the body's long axis, so it is
    invariant to yaw (turning on the spot) - the exact failure mode of hip width."""
    if not (vis(kps, L_SHOULDER) and vis(kps, R_SHOULDER) and vis(kps, L_HIP) and vis(kps, R_HIP)):
        return None
    sx, sy = midpoint(kps, L_SHOULDER, R_SHOULDER)
    hx, hy = midpoint(kps, L_HIP, R_HIP)
    d = math.hypot(sx - hx, sy - hy)
    return d if d > EPS else None


def n_torso_diag(kps):
    """Classic Yang & Ramanan PCK torso diameter: shoulder->opposite hip.
    Mean of both diagonals when available."""
    ds = []
    if vis(kps, L_SHOULDER) and vis(kps, R_HIP):
        ds.append(dist(kps, L_SHOULDER, R_HIP))
    if vis(kps, R_SHOULDER) and vis(kps, L_HIP):
        ds.append(dist(kps, R_SHOULDER, L_HIP))
    if not ds:
        return None
    d = sum(ds) / len(ds)
    return d if d > EPS else None


def n_torso_height_bbox_fb(kps):
    """Torso height, falling back to bbox diagonal when the torso is not visible."""
    d = n_torso_height(kps)
    if d is not None:
        return d
    return n_bbox(kps)


def n_torso_diag_bbox_fb(kps):
    d = n_torso_diag(kps)
    if d is not None:
        return d
    return n_bbox(kps)


NORMALIZERS = [
    ("hip_width (CURRENT)", n_hip_width),
    ("bbox_diagonal", n_bbox),
    ("torso_height", n_torso_height),
    ("torso_height+bbox_fb", n_torso_height_bbox_fb),
    ("torso_diag (Y&R)", n_torso_diag),
    ("torso_diag+bbox_fb", n_torso_diag_bbox_fb),
]


def mean_pose(frames):
    acc = [[0.0, 0.0, 0.0] for _ in range(17)]
    for kps in frames:
        for j in range(17):
            if vis(kps, j):
                acc[j][0] += kps[j][0]
                acc[j][1] += kps[j][1]
                acc[j][2] += 1.0
    return [
        [acc[j][0] / acc[j][2], acc[j][1] / acc[j][2], 1.0] if acc[j][2] else [0.5, 0.5, 1.0]
        for j in range(17)
    ]


def pck_for(pred, frames, norm, threshold):
    c = t = 0
    unscoreable = 0
    for kps in frames:
        s = norm(kps)
        if s is None:
            unscoreable += 1
            continue
        r = threshold * s
        for j in range(17):
            if not vis(kps, j):
                continue
            t += 1
            if math.hypot(pred[j][0] - kps[j][0], pred[j][1] - kps[j][1]) <= r:
                c += 1
    return (100.0 * c / t if t else 0.0), unscoreable


def pctile(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\ahmed\RuView\data")
    ap.add_argument("--threshold", type=float, default=0.20)
    args = ap.parse_args()

    files = sorted(Path(args.data).rglob("*.vis.jsonl"))
    captures = [(f.name, load_frames(f)) for f in files]
    captures = [(n, fr) for n, fr in captures if fr]

    print("=" * 92)
    print("PCK NORMALIZER COMPARISON".center(92))
    print("=" * 92)
    print(f"\ncaptures: {len(captures)}   frames: {sum(len(c[1]) for c in captures)}"
          f"   threshold: PCK@{int(args.threshold*100)}\n")

    # --- scale distribution + coefficient of variation --------------------
    print("[1] Normalizer scale distribution (a stable normalizer has low CV)\n")
    print(f"    {'normalizer':<24}{'min':>9}{'p50':>9}{'max':>9}{'max/min':>10}{'CV':>8}{'n/a':>7}")
    print("    " + "-" * 76)
    for name, fn in NORMALIZERS:
        vals, na = [], 0
        for _, frames in captures:
            for kps in frames:
                v = fn(kps)
                if v is None:
                    na += 1
                else:
                    vals.append(v)
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        cv = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)) / mean
        print(f"    {name:<24}{min(vals):9.5f}{pctile(vals,.5):9.5f}{max(vals):9.5f}"
              f"{max(vals)/max(min(vals),1e-12):9.0f}x{cv:8.3f}{na:7d}")

    # --- baseline stability ------------------------------------------------
    print(f"\n[2] Mean-pose (constant predictor) PCK@{int(args.threshold*100)} per capture\n")
    hdr = f"    {'capture':<30}" + "".join(f"{n.split()[0][:11]:>13}" for n, _ in NORMALIZERS)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    scores = {n: [] for n, _ in NORMALIZERS}
    for name, frames in captures:
        mp = mean_pose(frames)
        row = f"    {name[:28]:<30}"
        for nname, fn in NORMALIZERS:
            s, _ = pck_for(mp, frames, fn, args.threshold)
            scores[nname].append(s)
            row += f"{s:12.2f} "
        print(row)

    print("\n[3] Verdict\n")
    print(f"    {'normalizer':<24}{'min':>8}{'max':>8}{'spread':>10}{'stdev':>8}   assessment")
    print("    " + "-" * 82)
    for nname, _ in NORMALIZERS:
        v = scores[nname]
        lo, hi = min(v), max(v)
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
        spread = hi / max(lo, 1e-9)
        if mean > 50:
            note = "TOO LOOSE - constant pose already wins"
        elif sd > 15:
            note = "UNSTABLE - measures orientation, not skill"
        elif sd > 8:
            note = "marginal"
        else:
            note = "STABLE and discriminative"
        print(f"    {nname:<24}{lo:8.2f}{hi:8.2f}{spread:9.1f}x{sd:8.2f}   {note}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())

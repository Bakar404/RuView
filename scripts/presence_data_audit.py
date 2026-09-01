#!/usr/bin/env python3
"""Audit existing ground truth for a PRESENCE / IN-FRAME / RANGE task.

The pose task (17 keypoints) needs ankles in frame and thousands of paired
samples. Presence detection does not: it needs class balance. This script
answers the three questions that decide whether the existing captures are
usable for the retargeted problem:

  1. PRESENCE  - are there any *absent* (empty-room) frames to learn against?
                 Without negatives a presence classifier cannot be trained or
                 honestly evaluated (it would score ~100% by always saying yes).
  2. IN-FRAME  - how often is the subject fully inside the camera FOV vs
                 clipped at an edge? This is the label for "in frame or not".
  3. RANGE     - the bounding-box diagonal is a monotone proxy for distance
                 (bigger box = closer). Is there enough spread to define
                 close/mid/far bins, or was every capture at one distance?

Usage
-----
    python scripts/presence_data_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

VIS_T = 0.5
EDGE = 0.02  # within 2% of a frame border counts as clipped


def load(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def visible_pts(kps):
    return [(x, y) for x, y, v in kps[:17] if v >= VIS_T]


def bbox_diag(pts):
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def clipped(pts):
    """True if the visible skeleton touches/exceeds a frame border."""
    if not pts:
        return True
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs) <= EDGE or max(xs) >= 1.0 - EDGE
            or min(ys) <= EDGE or max(ys) >= 1.0 - EDGE)


def pctl(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\ahmed\RuView\data")
    args = ap.parse_args()

    files = sorted(Path(args.data).rglob("*.vis.jsonl"))
    if not files:
        print("no ground-truth files found")
        return 1

    print("=" * 88)
    print("PRESENCE / IN-FRAME / RANGE DATA AUDIT".center(88))
    print("=" * 88)

    tot_frames = tot_absent = tot_clipped = 0
    all_diags = []
    span_rows = []

    print(f"\n{'capture':<34}{'frames':>8}{'absent':>9}{'clipped':>9}"
          f"{'diag p10':>10}{'diag p50':>10}{'diag p90':>10}")
    print("-" * 88)

    for f in files:
        recs = load(f)
        if not recs:
            continue
        n = len(recs)
        absent = sum(1 for r in recs if r.get("n_persons", 0) == 0)
        diags, clip = [], 0
        for r in recs:
            kps = r.get("keypoints") or []
            if len(kps) < 17:
                absent += 0
                continue
            pts = visible_pts(kps)
            if not pts:
                continue
            diags.append(bbox_diag(pts))
            if clipped(pts):
                clip += 1
        all_diags += diags
        tot_frames += n
        tot_absent += absent
        tot_clipped += clip
        ts = [r["ts_ns"] for r in recs if "ts_ns" in r]
        if len(ts) > 1:
            span_rows.append((max(ts) - min(ts)) / 1e9)
        print(f"{f.name[:32]:<34}{n:>8}{absent:>9}{clip:>9}"
              f"{pctl(diags,.10):>10.3f}{pctl(diags,.50):>10.3f}{pctl(diags,.90):>10.3f}")

    print("-" * 88)
    print(f"{'TOTAL':<34}{tot_frames:>8}{tot_absent:>9}{tot_clipped:>9}"
          f"{pctl(all_diags,.10):>10.3f}{pctl(all_diags,.50):>10.3f}{pctl(all_diags,.90):>10.3f}")

    print("\n[1] PRESENCE class balance")
    pres = tot_frames - tot_absent
    print(f"    present frames : {pres:6d}  ({100*pres/max(tot_frames,1):5.1f}%)")
    print(f"    absent  frames : {tot_absent:6d}  ({100*tot_absent/max(tot_frames,1):5.1f}%)")
    if tot_absent == 0:
        print("\n    >>> NO NEGATIVE CLASS. Every logged frame contains a person.")
        print("        A presence classifier trained on this scores 100% by always")
        print("        answering 'present' and has learned nothing. You need")
        print("        empty-room CSI recorded with the SAME node placement.")
    elif tot_absent < 0.15 * tot_frames:
        print(f"\n    >>> SEVERE IMBALANCE ({100*tot_absent/tot_frames:.1f}% negatives).")

    print("\n[2] IN-FRAME balance")
    print(f"    clipped at a border : {tot_clipped:6d} ({100*tot_clipped/max(tot_frames,1):5.1f}%)")
    print(f"    fully inside FOV    : {tot_frames-tot_clipped:6d} "
          f"({100*(tot_frames-tot_clipped)/max(tot_frames,1):5.1f}%)")

    print("\n[3] RANGE spread (bbox diagonal as a distance proxy)")
    if all_diags:
        lo, hi = min(all_diags), max(all_diags)
        print(f"    min={lo:.3f}  p50={pctl(all_diags,.5):.3f}  max={hi:.3f}  "
              f"({hi/max(lo,1e-9):.1f}x spread)")
        b = Counter()
        for d in all_diags:
            b["close (>0.85)" if d > 0.85 else "mid (0.55-0.85)" if d > 0.55 else "far (<=0.55)"] += 1
        for k in ("close (>0.85)", "mid (0.55-0.85)", "far (<=0.55)"):
            c = b.get(k, 0)
            print(f"      {k:<18}{c:6d} ({100*c/len(all_diags):5.1f}%)")

    if span_rows:
        print(f"\n[4] Total wall-clock captured: {sum(span_rows)/60:.1f} min "
              f"across {len(span_rows)} captures")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
